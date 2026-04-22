//! Heap-based Monagan-Pearce reducer.
//!
//! See `~/rustgb/docs/design-decisions.md` ADR-008 for the
//! design rationale and the Singular / FLINT / mathicgb
//! comparison that motivated this architecture.
//!
//! # Status — Phase 1 (scaffold only)
//!
//! This module currently contains only the data-structure
//! definitions for the reducer:
//!
//! * [`Reducer`] — one in-flight reducer (a polynomial `g_i`,
//!   the multiplier monomial `m_i`, the pre-negated
//!   coefficient `c_i`, the current term index `j_i`, and the
//!   sugar contribution).
//! * [`HeapNode`] — a max-heap entry by degrevlex, carrying
//!   the cached comparison key and a back-reference to the
//!   reducer slab index. The cached key is the packed
//!   monomial `m_i * g_i.terms[j_i]` already XOR'd against
//!   the ring's `cmp_flip_mask`, so plain lex compare on
//!   `[u64; 4]` is the correct max-heap ordering and the
//!   comparator needs no `&Ring` indirection.
//! * [`ReducerHeap`] — the full reducer state for one in-progress
//!   reduction: a slab of [`Reducer`]s plus a max-heap of
//!   [`HeapNode`]s. Public API is just construction at this
//!   phase.
//!
//! Phases 2-7 (per ADR-008's migration plan) will land the
//! actual reduction algorithm: heap operations, pop-with-
//! cancellation, lazy divisor addition, survivor materialisation,
//! integration into [`crate::bba::reduce_lobject`] behind a
//! feature flag, staging-validation, and (if successful)
//! retirement of the geobucket reducer.

use crate::field::Coeff;
use crate::monomial::Monomial;
use crate::poly::Poly;
use crate::ring::Ring;
use std::collections::BinaryHeap;
use std::sync::Arc;

/// One in-flight reducer in a [`ReducerHeap`].
///
/// Represents the pending product `coeff * multiplier * poly`,
/// of which the heap currently has term `index` queued. A new
/// reducer is added with `index = 0` (its leading term, which
/// by construction matches and cancels the partial reduction's
/// current leader); subsequent pops advance `index` past the
/// emitted term so the next term is queued for ordering.
///
/// `coeff` is **pre-negated** at insertion time so that when
/// the heap pops the chain `(old_leader, new_reducer.term_0)`
/// and sums their coefficients, the cancellation drops out
/// naturally without sign tracking inside the heap.
///
/// The `Reducer` borrows its source `poly` from the [`SBasis`];
/// lifetime `'a` ties the heap to the borrow of that basis for
/// the duration of one reduction.
#[derive(Debug)]
pub struct Reducer<'a> {
    /// Source polynomial `g_i`. Borrowed from the basis for
    /// the reduction's lifetime.
    pub poly: &'a Poly,
    /// Multiplier monomial `m_i = lm(LObject) / lm(g_i)` at the
    /// time this reducer was added.
    pub multiplier: Monomial,
    /// Pre-negated multiplier coefficient
    /// `c_i = -leader_coeff(LObject) / lc(g_i)`. With monic
    /// basis elements `lc(g_i) == 1`, this simplifies to
    /// `-leader_coeff`.
    pub coeff: Coeff,
    /// Index of the next term in `poly.terms` not yet queued
    /// in the heap. Starts at 0; advances by one for every
    /// term of this reducer popped off the heap.
    pub index: usize,
    /// Sugar contribution: `g_i.lm_deg() + multiplier.total_deg()`.
    /// Used to compute the LObject's running sugar as the
    /// max over all in-flight reducers (plus the initial sugar).
    pub sugar: u32,
}

/// A node in the max-heap by degrevlex.
///
/// Two fields:
/// * `cmp_key` is the packed monomial of the currently-queued
///   term `multiplier * g_i.terms[index]`, **already XOR'd
///   against the ring's `cmp_flip_mask`**. Lex compare of the
///   four `u64` words (MSB first) is the correct degrevlex max
///   ordering, so `Ord` on `HeapNode` reduces to a plain `cmp`
///   on the `[u64; 4]` cmp_key. This eliminates the need to
///   pass `&Ring` into the heap's internal comparator.
/// * `reducer_idx` is an index into the [`ReducerHeap`]'s
///   slab of [`Reducer`]s, identifying which in-flight reducer
///   this term belongs to.
///
/// At most one `HeapNode` per `Reducer` lives in the heap at
/// any time (FLINT's invariant): when we pop a node, we either
/// advance the source's `index` and push the next term back,
/// or the source has been exhausted and no new node is pushed.
#[derive(Debug, Clone)]
pub struct HeapNode {
    /// Packed monomial XOR'd against `ring.cmp_flip_mask`.
    /// See module docs for the `[u64; 4]` lex-compare convention.
    pub cmp_key: [u64; 4],
    /// Index into the slab of [`Reducer`]s.
    pub reducer_idx: usize,
}

impl PartialEq for HeapNode {
    fn eq(&self, other: &Self) -> bool {
        self.cmp_key == other.cmp_key
    }
}

impl Eq for HeapNode {}

impl PartialOrd for HeapNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapNode {
    /// Lex compare on the cached `cmp_key`, MSB-word first.
    /// Result is the degrevlex order on the underlying monomials
    /// (because `cmp_key` was constructed with the ring's
    /// `cmp_flip_mask` applied — see `Monomial::cmp_degrevlex`).
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        for i in (0..4).rev() {
            match self.cmp_key[i].cmp(&other.cmp_key[i]) {
                std::cmp::Ordering::Equal => {}
                ord => return ord,
            }
        }
        std::cmp::Ordering::Equal
    }
}

/// The state of one in-progress polynomial reduction.
///
/// Holds the slab of in-flight reducers and the max-heap of
/// pending product terms. One `ReducerHeap` is created per
/// LObject being reduced; it lives for the duration of that
/// reduction and is dropped (or consumed into a survivor `Poly`)
/// when the reduction terminates.
///
/// Lifetime `'a` ties this state to the borrow of the
/// [`SBasis`] whose polynomials the [`Reducer`]s reference.
#[derive(Debug)]
pub struct ReducerHeap<'a> {
    /// Owning ring reference. All monomials in the heap belong
    /// to this ring.
    ring: Arc<Ring>,
    /// Slab of in-flight reducers. Indexed by [`HeapNode::reducer_idx`].
    /// Grows monotonically — a reducer is never removed from the
    /// slab once added (its tail may be exhausted, in which case
    /// no `HeapNode` references it any more, but its slot stays).
    reducers: Vec<Reducer<'a>>,
    /// Max-heap of pending product terms, ordered by degrevlex
    /// via [`HeapNode::cmp_key`]. The std-library `BinaryHeap` is
    /// a max-heap and our [`HeapNode::cmp`] implements degrevlex
    /// via lex compare on the cached `cmp_key`, so push/pop/peek
    /// give the right semantics directly.
    heap: BinaryHeap<HeapNode>,
    /// Running sugar. Initialised at construction; updated to
    /// `max(self.sugar, reducer.sugar)` on each `push_reducer`.
    sugar: u32,
}

impl<'a> ReducerHeap<'a> {
    /// Construct an empty reducer state for a reduction starting
    /// at `initial_sugar`. Adding the LObject's polynomial as the
    /// first reducer (with `multiplier = 1`, `coeff = 1`) is the
    /// caller's responsibility (deferred to phase 4).
    pub fn new(ring: Arc<Ring>, initial_sugar: u32) -> Self {
        Self {
            ring,
            reducers: Vec::new(),
            heap: BinaryHeap::new(),
            sugar: initial_sugar,
        }
    }

    /// Borrow the ring this heap operates over.
    #[inline]
    pub fn ring(&self) -> &Arc<Ring> {
        &self.ring
    }

    /// Current sugar of the in-progress reduction. Equal to
    /// `max(initial_sugar, max over reducers of reducer.sugar)`.
    #[inline]
    pub fn sugar(&self) -> u32 {
        self.sugar
    }

    /// Number of in-flight reducers currently in the slab. Equal
    /// to the number of `push_reducer` calls so far (no removal).
    #[inline]
    pub fn reducer_count(&self) -> usize {
        self.reducers.len()
    }

    /// Number of heap nodes currently in flight. Each in-flight
    /// reducer contributes at most one node (FLINT's invariant);
    /// a node may be missing if the reducer's tail has been
    /// exhausted by repeated pop+advance.
    #[inline]
    pub fn heap_len(&self) -> usize {
        self.heap.len()
    }

    /// Whether the heap is empty. Equivalent to `self.heap_len() == 0`.
    /// When this returns true the reduction has terminated; either
    /// the LObject reduced to zero (no survivor), or the survivor
    /// has already been fully drained into a `Poly`.
    #[inline]
    pub fn heap_is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    // ----- Heap operations (phase 2) -----
    //
    // The heap is a max-heap by degrevlex via HeapNode::cmp on the
    // cached cmp_key. The underlying BinaryHeap is std-library;
    // these methods are thin wrappers that document the role of
    // each operation in the Monagan-Pearce reducer's lifecycle.

    /// Push a heap node. The node carries the pre-XOR'd cmp_key
    /// for the term `multiplier * g.terms[index]` for some reducer
    /// in the slab; the caller is responsible for constructing it
    /// (typically `Self::push_term`, deferred to phase 4 where the
    /// reducer-construction surface lands).
    #[inline]
    pub fn push_node(&mut self, node: HeapNode) {
        self.heap.push(node);
    }

    /// Look at the maximum heap node without removing it. Returns
    /// `None` if the heap is empty.
    ///
    /// Used by [`pop_with_cancellation`](Self::pop_with_cancellation)
    /// (phase 3) to peek at successive max entries and detect
    /// whether they share the leading `cmp_key` — the signal that
    /// terms cancel and need to be summed.
    #[inline]
    pub fn peek_max(&self) -> Option<&HeapNode> {
        self.heap.peek()
    }

    /// Remove and return the maximum heap node. Returns `None`
    /// if the heap is empty.
    ///
    /// In Monagan-Pearce, this corresponds to taking the next
    /// pending product to consider for emission. The caller must
    /// then either advance the source reducer's index (and push
    /// the next term back onto the heap) or, if the source's
    /// tail is exhausted, leave the slot vacant.
    #[inline]
    pub fn pop_max(&mut self) -> Option<HeapNode> {
        self.heap.pop()
    }

    // ----- Reducer management + cancellation pop (phase 3) -----

    /// Add a reducer to the slab and queue its leading term onto
    /// the heap. Returns the slab index of the new reducer.
    ///
    /// If `reducer.poly` is the zero polynomial (no terms past
    /// `index`), the reducer is added to the slab but no heap node
    /// is pushed — the reducer is effectively dead on arrival.
    /// This is consistent with FLINT's "at most one heap node per
    /// reducer" invariant: an exhausted reducer has zero in flight.
    ///
    /// Updates the running sugar to `max(self.sugar, reducer.sugar)`.
    pub fn push_reducer(&mut self, reducer: Reducer<'a>) -> usize {
        let idx = self.reducers.len();
        self.sugar = self.sugar.max(reducer.sugar);
        self.reducers.push(reducer);
        // Queue the leading term if non-empty.
        self.push_current_term(idx);
        idx
    }

    /// Compute the cmp_key of the term currently at the front of
    /// `reducers[reducer_idx]` (i.e. `multiplier * poly.terms[index]`
    /// XOR'd against the ring's cmp_flip_mask) and push a HeapNode
    /// for it. No-op if the reducer is exhausted (`index >= poly.len`).
    ///
    /// In the unusual case where the multiplier × term product
    /// would overflow the 7-bit-per-variable budget (ADR-005),
    /// the term is dropped and the reducer is effectively
    /// truncated. This matches the existing `KBucket::minus_m_mult_p`
    /// silent-noop-on-overflow semantics; see ADR-005's deferred
    /// "panic on overflow" enhancement.
    fn push_current_term(&mut self, reducer_idx: usize) {
        let r = &self.reducers[reducer_idx];
        if r.index >= r.poly.len() {
            return;
        }
        let Some(term_mono) = r.multiplier.mul(&r.poly.terms()[r.index], &self.ring) else {
            // Overflow: drop this term (and effectively truncate
            // the reducer at this index). Caller is responsible
            // for ensuring this doesn't happen for correctness;
            // see ADR-005.
            debug_assert!(
                false,
                "monomial product overflowed during heap reduction"
            );
            return;
        };
        let mask = self.ring.cmp_flip_mask();
        let cmp_key = std::array::from_fn(|i| term_mono.packed()[i] ^ mask[i]);
        self.heap.push(HeapNode {
            cmp_key,
            reducer_idx,
        });
    }

    /// Advance the named reducer past its current term and queue
    /// the next term onto the heap (if any).
    fn advance_reducer(&mut self, reducer_idx: usize) {
        self.reducers[reducer_idx].index += 1;
        self.push_current_term(reducer_idx);
    }

    /// Pop the next non-cancelled (coefficient, monomial) leader.
    /// Returns `None` when the heap is empty (the reduction has
    /// terminated; if the LObject's poly was the only reducer,
    /// then we've drained it; if reducers were added along the
    /// way and the chain cancelled to zero everywhere, the LObject
    /// reduced to zero).
    ///
    /// Algorithm: pop the max; if the next pops share the same
    /// `cmp_key`, they're contributing to the same monomial and
    /// their coefficients sum. Each contributing reducer's index
    /// is advanced past its emitted term. If the resulting sum is
    /// zero, the chain cancelled and we recurse on the next leader;
    /// if non-zero, that's the leader the caller wants.
    pub fn pop_with_cancellation(&mut self) -> Option<(Coeff, Monomial)> {
        loop {
            let max_node = self.heap.pop()?;
            // Collect the contributing reducer indices and compute
            // the monomial value (the same for all contributors,
            // by the same-cmp_key invariant).
            let chain_key = max_node.cmp_key;
            let max_idx = max_node.reducer_idx;
            let r = &self.reducers[max_idx];
            // Recover the actual monomial. Could rebuild from the
            // multiplier and source term, but the cmp_key is the
            // XOR'd packed bytes, so unpacking gets us back. Easier
            // to just recompute from the reducer state.
            let mono = r
                .multiplier
                .mul(&r.poly.terms()[r.index], &self.ring)
                .expect("just pushed; overflow already checked");
            let f = self.ring.field();
            let mut total_coeff = f.mul(r.coeff, r.poly.coeffs()[r.index]);

            // Advance the max contributor.
            let mut to_advance: Vec<usize> = Vec::with_capacity(2);
            to_advance.push(max_idx);

            // Drain any equal-cmp_key entries.
            while let Some(next) = self.heap.peek() {
                if next.cmp_key != chain_key {
                    break;
                }
                let next_node = self.heap.pop().unwrap();
                let nr = &self.reducers[next_node.reducer_idx];
                let next_coeff = f.mul(nr.coeff, nr.poly.coeffs()[nr.index]);
                total_coeff = f.add(total_coeff, next_coeff);
                to_advance.push(next_node.reducer_idx);
            }

            // Advance every contributor past its emitted term.
            for &idx in &to_advance {
                self.advance_reducer(idx);
            }

            if total_coeff != 0 {
                return Some((total_coeff, mono));
            }
            // total_coeff == 0: complete cancellation; loop and
            // try the next leader.
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Field;
    use crate::ordering::MonoOrder;

    fn mk_ring(nvars: u32) -> Arc<Ring> {
        Arc::new(Ring::new(nvars, MonoOrder::DegRevLex, Field::new(32003).unwrap()).unwrap())
    }

    #[test]
    fn empty_reducer_heap_constructs() {
        let r = mk_ring(3);
        let h = ReducerHeap::new(Arc::clone(&r), 0);
        assert_eq!(h.reducer_count(), 0);
        assert_eq!(h.heap_len(), 0);
        assert_eq!(h.sugar(), 0);
    }

    #[test]
    fn initial_sugar_is_preserved() {
        let r = mk_ring(3);
        let h = ReducerHeap::new(Arc::clone(&r), 17);
        assert_eq!(h.sugar(), 17);
    }

    #[test]
    fn heap_node_ord_matches_lex_on_cmp_key() {
        // cmp_key is the XOR-flipped packed monomial; lex compare
        // on the four u64 words MSB-first IS the degrevlex order.
        // Verify our Ord impl on HeapNode produces the right result
        // on hand-crafted keys.
        let a = HeapNode {
            cmp_key: [0, 0, 0, 0xFF_00_00_00_00_00_00_00],
            reducer_idx: 0,
        };
        let b = HeapNode {
            cmp_key: [0, 0, 0, 0x80_00_00_00_00_00_00_00],
            reducer_idx: 1,
        };
        // a's top byte is 0xFF, b's is 0x80 → a > b.
        assert!(a > b);
        assert!(b < a);
        assert_ne!(a, b);
    }

    #[test]
    fn heap_node_ord_walks_words_msb_first() {
        // Two nodes where word 3 differs: that's the only word
        // that should matter.
        let a = HeapNode {
            cmp_key: [0xFFFF, 0, 0, 0x10_00_00_00_00_00_00_00],
            reducer_idx: 0,
        };
        let b = HeapNode {
            cmp_key: [0, 0, 0, 0x20_00_00_00_00_00_00_00],
            reducer_idx: 1,
        };
        // word 3: a=0x10... < b=0x20...; lower words don't matter
        assert!(a < b);
    }

    #[test]
    fn heap_node_ord_falls_through_to_lower_words() {
        // word 3 equal; difference in word 0.
        let a = HeapNode {
            cmp_key: [5, 0, 0, 0x42_00_00_00_00_00_00_00],
            reducer_idx: 0,
        };
        let b = HeapNode {
            cmp_key: [3, 0, 0, 0x42_00_00_00_00_00_00_00],
            reducer_idx: 1,
        };
        assert!(a > b);
    }

    #[test]
    fn heap_node_eq_ignores_reducer_idx() {
        // Two nodes with the same cmp_key but different
        // reducer_idx are PartialEq-equal (same monomial, two
        // reducers contributing). This is the condition that
        // pop-with-cancellation will look for to chain entries.
        let a = HeapNode {
            cmp_key: [1, 2, 3, 4],
            reducer_idx: 0,
        };
        let b = HeapNode {
            cmp_key: [1, 2, 3, 4],
            reducer_idx: 7,
        };
        assert_eq!(a, b);
    }

    // ----- Phase 2: heap operations -----

    /// Helper: build a deterministic pseudo-random sequence of
    /// HeapNodes with diverse cmp_keys for property testing.
    /// Keys are spread across all four words to exercise the
    /// MSB-first lex-compare path.
    fn pseudo_random_nodes(n: usize, seed: u64) -> Vec<HeapNode> {
        let mut state = seed;
        let mut step = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        (0..n)
            .map(|i| HeapNode {
                cmp_key: [step(), step(), step(), step()],
                reducer_idx: i,
            })
            .collect()
    }

    /// Slow reference: pop the max repeatedly via sort.
    fn slow_drain_descending(mut nodes: Vec<HeapNode>) -> Vec<HeapNode> {
        nodes.sort();
        nodes.reverse();
        nodes
    }

    #[test]
    fn empty_heap_pop_and_peek_return_none() {
        let r = mk_ring(3);
        let mut h = ReducerHeap::new(Arc::clone(&r), 0);
        assert!(h.heap_is_empty());
        assert!(h.peek_max().is_none());
        assert!(h.pop_max().is_none());
        assert!(h.heap_is_empty());
    }

    #[test]
    fn push_then_pop_single_node() {
        let r = mk_ring(3);
        let mut h = ReducerHeap::new(Arc::clone(&r), 0);
        let n = HeapNode {
            cmp_key: [1, 2, 3, 4],
            reducer_idx: 42,
        };
        h.push_node(n.clone());
        assert_eq!(h.heap_len(), 1);
        assert!(!h.heap_is_empty());
        assert_eq!(h.peek_max(), Some(&n));
        assert_eq!(h.pop_max(), Some(n));
        assert!(h.heap_is_empty());
    }

    #[test]
    fn push_pop_drains_in_descending_order() {
        // Property test against the slow sort-based reference.
        // For each seed, push N nodes onto the heap and verify
        // that draining pops them in descending degrevlex order
        // (the same order the slow reference produces).
        let r = mk_ring(3);
        for &seed in &[0x1234_5678_9abc_def0u64, 0xdead_beef_cafe_babe, 1, 2, 0xff_ff] {
            for &n in &[1usize, 2, 5, 16, 64, 200] {
                let nodes = pseudo_random_nodes(n, seed);
                let mut h = ReducerHeap::new(Arc::clone(&r), 0);
                for node in &nodes {
                    h.push_node(node.clone());
                }
                assert_eq!(h.heap_len(), n);
                let mut got = Vec::with_capacity(n);
                while let Some(top) = h.pop_max() {
                    got.push(top);
                }
                assert!(h.heap_is_empty());
                let expected = slow_drain_descending(nodes);
                assert_eq!(
                    got.iter().map(|n| n.cmp_key).collect::<Vec<_>>(),
                    expected.iter().map(|n| n.cmp_key).collect::<Vec<_>>(),
                    "drain order mismatch for seed {seed:#x}, n = {n}"
                );
            }
        }
    }

    #[test]
    fn peek_matches_pop() {
        // After every push, peek_max should report the same cmp_key
        // that the next pop_max returns.
        let r = mk_ring(3);
        let nodes = pseudo_random_nodes(50, 0xfeed_face);
        let mut h = ReducerHeap::new(Arc::clone(&r), 0);
        for node in &nodes {
            h.push_node(node.clone());
            let peek_key = h.peek_max().unwrap().cmp_key;
            // Don't actually pop — instead, verify that on the
            // *next* pop later we'd see this key. Take a snapshot
            // by cloning the heap state for the check.
            let mut h2 = ReducerHeap::new(Arc::clone(&r), 0);
            // Re-build h2 from the underlying BinaryHeap's iterator
            // to avoid moving h.
            for n2 in h.heap.iter() {
                h2.push_node(n2.clone());
            }
            let popped = h2.pop_max().unwrap();
            assert_eq!(popped.cmp_key, peek_key);
        }
    }

    #[test]
    fn interleaved_push_and_pop_drains_correctly() {
        // Push some, pop some, push more, drain — verifies the
        // heap can absorb pops in the middle of building (the
        // lifecycle that pop-with-cancellation will exercise).
        let r = mk_ring(3);
        let nodes = pseudo_random_nodes(30, 0xa1b2_c3d4);
        let mut h = ReducerHeap::new(Arc::clone(&r), 0);

        // Push first 20.
        for n in &nodes[..20] {
            h.push_node(n.clone());
        }
        // Pop 5 (saving them).
        let mut popped: Vec<HeapNode> = (0..5).map(|_| h.pop_max().unwrap()).collect();
        // Push remaining 10.
        for n in &nodes[20..30] {
            h.push_node(n.clone());
        }
        // Drain.
        while let Some(top) = h.pop_max() {
            popped.push(top);
        }

        // Total nodes seen = 30; their cmp_keys, when sorted
        // descending, should match the input sorted descending.
        assert_eq!(popped.len(), 30);
        let mut got_keys: Vec<[u64; 4]> = popped.iter().map(|n| n.cmp_key).collect();
        got_keys.sort();
        let mut want_keys: Vec<[u64; 4]> = nodes.iter().map(|n| n.cmp_key).collect();
        want_keys.sort();
        assert_eq!(got_keys, want_keys);
    }

    // ----- Phase 3: push_reducer + pop_with_cancellation -----

    use crate::monomial::Monomial;
    use crate::poly::Poly;

    fn mono(r: &Ring, e: &[u32]) -> Monomial {
        Monomial::from_exponents(r, e).unwrap()
    }

    /// Helper: collect the full output of pop_with_cancellation
    /// into a Vec until the heap drains. Used as a "drive the
    /// reducer to completion" pattern in tests.
    fn drain_with_cancellation<'a>(
        h: &mut ReducerHeap<'a>,
    ) -> Vec<(Coeff, Monomial)> {
        let mut out = Vec::new();
        while let Some(pair) = h.pop_with_cancellation() {
            out.push(pair);
        }
        out
    }

    #[test]
    fn single_reducer_no_cancellation_emits_all_terms() {
        // Push one reducer (multiplier=1, coeff=1) over a 3-term
        // poly. Drain pop_with_cancellation; should emit each term
        // in descending order with coeff matching the source.
        let r = mk_ring(3);
        let one = Monomial::one(&r);
        let p = Poly::from_terms(
            &r,
            vec![
                (5, mono(&r, &[3, 0, 0])),  // x_0^3
                (2, mono(&r, &[1, 1, 0])),  // x_0 x_1
                (1, mono(&r, &[0, 0, 2])),  // x_2^2
            ],
        );
        let mut h = ReducerHeap::new(Arc::clone(&r), 0);
        h.push_reducer(Reducer {
            poly: &p,
            multiplier: one.clone(),
            coeff: 1,
            index: 0,
            sugar: 3,
        });
        assert_eq!(h.sugar(), 3);

        let out = drain_with_cancellation(&mut h);
        assert_eq!(out.len(), 3);
        // Terms emerge in descending degrevlex order — same as the
        // source polynomial's canonical order.
        assert_eq!(out[0].0, 5);
        assert_eq!(out[0].1, mono(&r, &[3, 0, 0]));
        assert_eq!(out[1].0, 2);
        assert_eq!(out[1].1, mono(&r, &[1, 1, 0]));
        assert_eq!(out[2].0, 1);
        assert_eq!(out[2].1, mono(&r, &[0, 0, 2]));
    }

    #[test]
    fn two_reducers_complete_cancellation_drains_to_zero() {
        // Two copies of the same poly with opposite-sign coeffs:
        // 1 * p + (-1) * p should cancel everywhere.
        let r = mk_ring(3);
        let p = Poly::from_terms(
            &r,
            vec![
                (5, mono(&r, &[2, 0, 0])),
                (3, mono(&r, &[0, 1, 0])),
            ],
        );
        let one = Monomial::one(&r);
        let f = r.field();
        let neg_one = f.neg(1); // = p - 1 in Z/p

        let mut h = ReducerHeap::new(Arc::clone(&r), 0);
        h.push_reducer(Reducer {
            poly: &p,
            multiplier: one.clone(),
            coeff: 1,
            index: 0,
            sugar: 2,
        });
        h.push_reducer(Reducer {
            poly: &p,
            multiplier: one,
            coeff: neg_one,
            index: 0,
            sugar: 2,
        });

        // Every leader cancels; pop_with_cancellation should
        // return None right away after eating all the cancelled
        // chains.
        let out = drain_with_cancellation(&mut h);
        assert!(out.is_empty(), "everything should cancel; got {out:?}");
        assert!(h.heap_is_empty());
    }

    #[test]
    fn partial_cancellation_yields_remainder() {
        // p = 5*x^2 + 3*y, q = 2*x^2 + 7*z.
        // 1 * p + 1 * q  →  (5+2)*x^2 + 3*y + 7*z
        // (assumed degrevlex: x^2 > y > z by degree, then by var index)
        let r = mk_ring(3);
        let p = Poly::from_terms(
            &r,
            vec![(5, mono(&r, &[2, 0, 0])), (3, mono(&r, &[0, 1, 0]))],
        );
        let q = Poly::from_terms(
            &r,
            vec![(2, mono(&r, &[2, 0, 0])), (7, mono(&r, &[0, 0, 1]))],
        );
        let one = Monomial::one(&r);
        let mut h = ReducerHeap::new(Arc::clone(&r), 0);
        h.push_reducer(Reducer {
            poly: &p,
            multiplier: one.clone(),
            coeff: 1,
            index: 0,
            sugar: 2,
        });
        h.push_reducer(Reducer {
            poly: &q,
            multiplier: one,
            coeff: 1,
            index: 0,
            sugar: 2,
        });
        let out = drain_with_cancellation(&mut h);
        // Three distinct monomials in the result: x^2 with combined
        // coeff 7, then either y or z next (degrevlex compares by
        // total degree first → both y and z are degree 1; then
        // tie-break by largest-index differing variable: z wins
        // tie-break, so y < z is wrong in degrevlex... let me verify).
        //
        // Degrevlex tie-break for two degree-1 monomials y vs z:
        // largest index where exps differ: z's index (2) > y's (1).
        // At that index: y has 0, z has 1. Smaller wins → y > z.
        // So order is: x^2 > y > z.
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].0, 7);
        assert_eq!(out[0].1, mono(&r, &[2, 0, 0]));
        assert_eq!(out[1].0, 3);
        assert_eq!(out[1].1, mono(&r, &[0, 1, 0]));
        assert_eq!(out[2].0, 7);
        assert_eq!(out[2].1, mono(&r, &[0, 0, 1]));
    }

    #[test]
    fn matches_poly_add_on_random_pairs() {
        // Property: pushing two reducers (1*p, 1*q) into the heap
        // and draining via pop_with_cancellation must yield the
        // same term sequence as p.add(q) does via poly::merge.
        // Verifies the heap reducer's output agrees with the
        // existing geobucket-equivalent (Poly::add) on simple
        // sum cases.
        let r = mk_ring(4);
        let pairs = vec![
            (
                vec![(3u32, vec![2, 1, 0, 0]), (5, vec![1, 0, 1, 0])],
                vec![(2, vec![2, 1, 0, 0]), (4, vec![0, 1, 1, 0])],
            ),
            (
                vec![(7, vec![3, 0, 0, 0]), (1, vec![0, 0, 0, 1])],
                vec![(2, vec![1, 2, 0, 0])],
            ),
            (
                vec![],
                vec![(11, vec![1, 1, 1, 0])],
            ),
        ];
        for (p_terms, q_terms) in pairs {
            let p = Poly::from_terms(
                &r,
                p_terms
                    .into_iter()
                    .map(|(c, e)| (c, mono(&r, &e)))
                    .collect(),
            );
            let q = Poly::from_terms(
                &r,
                q_terms
                    .into_iter()
                    .map(|(c, e)| (c, mono(&r, &e)))
                    .collect(),
            );

            // Reference: Poly::add via the geobucket-friendly merge.
            let want = p.add(&q, &r);
            let want_terms: Vec<(Coeff, Monomial)> =
                want.iter().map(|(c, m)| (c, m.clone())).collect();

            // Heap reducer: 1*p + 1*q.
            let one = Monomial::one(&r);
            let mut h = ReducerHeap::new(Arc::clone(&r), 0);
            h.push_reducer(Reducer {
                poly: &p,
                multiplier: one.clone(),
                coeff: 1,
                index: 0,
                sugar: 0,
            });
            h.push_reducer(Reducer {
                poly: &q,
                multiplier: one,
                coeff: 1,
                index: 0,
                sugar: 0,
            });
            let got = drain_with_cancellation(&mut h);

            assert_eq!(got, want_terms);
        }
    }

    #[test]
    fn sugar_is_max_over_pushed_reducers() {
        let r = mk_ring(2);
        let p = Poly::from_terms(&r, vec![(1, mono(&r, &[1, 0]))]);
        let one = Monomial::one(&r);
        let mut h = ReducerHeap::new(Arc::clone(&r), 7);
        assert_eq!(h.sugar(), 7);
        h.push_reducer(Reducer {
            poly: &p,
            multiplier: one.clone(),
            coeff: 1,
            index: 0,
            sugar: 3,
        });
        // 7 still wins over 3.
        assert_eq!(h.sugar(), 7);
        h.push_reducer(Reducer {
            poly: &p,
            multiplier: one,
            coeff: 1,
            index: 0,
            sugar: 12,
        });
        // 12 takes over.
        assert_eq!(h.sugar(), 12);
    }

    #[test]
    fn duplicate_cmp_keys_both_pop() {
        // Two nodes with the same cmp_key but different
        // reducer_idx should both be poppable. Order between
        // them is unspecified, but neither should be lost.
        let r = mk_ring(3);
        let mut h = ReducerHeap::new(Arc::clone(&r), 0);
        let a = HeapNode {
            cmp_key: [9, 0, 0, 0],
            reducer_idx: 1,
        };
        let b = HeapNode {
            cmp_key: [9, 0, 0, 0],
            reducer_idx: 2,
        };
        let c = HeapNode {
            cmp_key: [5, 0, 0, 0],
            reducer_idx: 3,
        };
        h.push_node(a.clone());
        h.push_node(c.clone());
        h.push_node(b.clone());
        // Top two should both have cmp_key [9, 0, 0, 0].
        let p1 = h.pop_max().unwrap();
        assert_eq!(p1.cmp_key, [9, 0, 0, 0]);
        let p2 = h.pop_max().unwrap();
        assert_eq!(p2.cmp_key, [9, 0, 0, 0]);
        // Reducer indices: should be {1, 2} between p1 and p2.
        let idxs = [p1.reducer_idx, p2.reducer_idx];
        assert!(idxs.contains(&1) && idxs.contains(&2));
        // Last pop is c.
        let p3 = h.pop_max().unwrap();
        assert_eq!(p3.cmp_key, [5, 0, 0, 0]);
        assert_eq!(p3.reducer_idx, 3);
        assert!(h.heap_is_empty());
    }
}
