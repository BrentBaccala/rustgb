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
