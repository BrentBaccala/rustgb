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
    /// via [`HeapNode::cmp_key`]. Phase 2 will add the operations
    /// that maintain this heap; phase 1 keeps it as a plain
    /// [`Vec`] so the type compiles without committing to a
    /// specific heap implementation yet.
    heap: Vec<HeapNode>,
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
            heap: Vec::new(),
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
}
