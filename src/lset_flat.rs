//! `LSet` — flat-Vec backend with parallel `lcm_divmasks` and a
//! sorted-pop heap of indices. Selectable via `cargo --features
//! flat_lset`.
//!
//! Design reference: ADR-026 in `~/rustgb/docs/design-decisions.md`.
//!
//! ## Storage
//!
//! ```text
//! pairs:        Vec<Pair>     // flat backing; append on insert
//! tombstones:   Vec<bool>     // parallel; true ⇔ slot logically deleted
//! lcm_divmasks: Vec<u64>      // parallel; zeroed on tombstone (so SIMD
//!                                scans never match a dead slot)
//! by_indices:   HashMap<(u32, u32), usize>
//!                              // O(1) lookup of the live slot for (i, j)
//! sorted:       BinaryHeap<Reverse<SortedKey>>
//!                              // (sugar, arrival, idx) for ordered pop
//! live:         usize          // count of !tombstoned slots
//! ```
//!
//! Compared to `lset.rs`'s `BinaryHeap<Reverse<HeapEntry>>` (where
//! each `HeapEntry` carries a full `Pair` clone), this layout
//! separates the *ordering index* (small `(sugar, arrival, idx)`
//! triples) from the *pair storage* (one entry per inserted pair,
//! never moved). That separation is what enables the SIMD-batched
//! filtered scan in `iter_filtered_subset`: `lcm_divmasks` is a
//! contiguous `&[u64]` we can hand straight to
//! `crate::simd::find_divmask_match`.
//!
//! ## Tombstoning
//!
//! `pop()` and `delete()` flip `tombstones[idx]` to `true`, zero
//! `lcm_divmasks[idx]`, drop the `by_indices` mapping (when it
//! still points at this slot), and decrement `live`. The pair
//! payload `pairs[idx]` is left in place — `pop` clones it before
//! tombstoning so the caller gets ownership. This costs one
//! `Pair::clone` per pop versus the heap backend's `mem::take`-
//! style move, which is dominated by the Phase-2 scan savings the
//! SIMD path provides.
//!
//! `insert()` reuses neither slots nor keys: every fresh pair goes
//! to `pairs.len()` and bumps `next_arrival`. This keeps `idx`
//! stable for the lifetime of the LSet and means the
//! `BinaryHeap<Reverse<(sugar, arrival, idx)>>` ordering is
//! deterministic against the same input stream as the heap
//! backend.
//!
//! ## Sorted-pop index
//!
//! A `BinaryHeap<Reverse<SortedKey>>` of `(sugar, arrival, idx)`.
//! `pop()` repeatedly pops the top of the sorted heap, skipping
//! entries whose `tombstones[idx]` is set, until it finds a live
//! slot or the heap empties. This is the same tombstone-on-pop
//! pattern used by `lset.rs`, but the heap entries are tiny
//! `(u32, u64, usize)` triples instead of full `Pair` clones —
//! pushing on insert is cheaper and the cache footprint of an
//! ordered scan is much smaller.
//!
//! ## Invariants (`assert_canonical`)
//!
//! 1. `pairs.len() == tombstones.len() == lcm_divmasks.len()`.
//! 2. Every live slot satisfies `lcm_divmasks[idx] ==
//!    pairs[idx].lcm_divmask` and `pairs[idx].assert_canonical`.
//! 3. Every tombstoned slot satisfies `lcm_divmasks[idx] == 0`
//!    (so `iter_filtered_subset` never yields a dead pair —
//!    `(any_mask & !0) == 0` is true only when `any_mask == 0`,
//!    and Phase-2 callers always pass a non-zero `h_lm_divmask`
//!    derived from a non-trivial leading monomial).
//! 4. `by_indices[(i, j)]` always references a non-tombstoned
//!    slot, and `pairs[that_slot].i == i && pairs[that_slot].j
//!    == j`.
//! 5. `live == pairs.len() - count(tombstones)`.
//!
//! ## Threading
//!
//! `Send + Sync` by construction (no interior mutability — all
//! mutation goes through `&mut self`). `SharedLSet` wraps this
//! backend in a `Mutex` exactly the same way as the heap backend.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use crate::pair::{Pair, PairKey};

/// Heap entry for the sorted-pop index. `Ord` matches `Pair::cmp`'s
/// ascending order on `(sugar, arrival)`, with `idx` as a final
/// tie-break so two entries are never equal in the heap (the heap
/// stores indices into a single `pairs` Vec, so `idx` is unique by
/// construction).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SortedKey {
    sugar: u32,
    arrival: u64,
    idx: usize,
}

impl Ord for SortedKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.sugar
            .cmp(&other.sugar)
            .then_with(|| self.arrival.cmp(&other.arrival))
            .then_with(|| self.idx.cmp(&other.idx))
    }
}
impl PartialOrd for SortedKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// S-pair queue ordered by sugar, ties broken by arrival.
///
/// Public API matches `lset::LSet` exactly. See module-level
/// docs for the storage shape and invariants.
#[derive(Debug, Default)]
pub struct LSet {
    /// Append-only flat storage. `pairs[idx]` may be a stale (i.e.
    /// tombstoned) pair: read it only when `!tombstones[idx]`.
    pairs: Vec<Pair>,
    /// `tombstones[idx] == true` means the slot is logically deleted.
    /// The Pair stays in `pairs` for memory reuse but is invisible
    /// to every public iterator and to `pop`.
    tombstones: Vec<bool>,
    /// Parallel array of `pair.lcm_divmask` values. **Zeroed when
    /// the corresponding slot is tombstoned**, so the SIMD scan
    /// `find_divmask_match(lcm_divmasks, h_lm_divmask, 0)` never
    /// reports a dead slot for a non-zero query mask.
    lcm_divmasks: Vec<u64>,
    /// O(1) lookup: index pair → the slot in `pairs` that currently
    /// holds the live pair for those indices. Always points at a
    /// non-tombstoned slot; updated by `insert` (which tombstones
    /// the previous live slot for the same `(i, j)`) and `delete`.
    by_indices: HashMap<(u32, u32), usize>,
    /// `BinaryHeap<Reverse<(sugar, arrival, idx)>>` for ordered pop.
    /// Entries for tombstoned slots are skipped on pop.
    sorted: BinaryHeap<Reverse<SortedKey>>,
    /// Monotonic identity stamped into `pair.key` at insert time.
    /// Matches `lset::LSet`'s field of the same name so test
    /// fixtures behave identically.
    next_key: u64,
    /// Live (non-tombstoned) count. Avoids scanning `tombstones`
    /// for every `len()` call.
    live: usize,
}

impl LSet {
    /// Empty queue.
    pub fn new() -> Self {
        Self {
            pairs: Vec::new(),
            tombstones: Vec::new(),
            lcm_divmasks: Vec::new(),
            by_indices: HashMap::new(),
            sorted: BinaryHeap::new(),
            next_key: 1,
            live: 0,
        }
    }

    /// Number of live (non-tombstoned, non-popped) pairs.
    #[inline]
    pub fn len(&self) -> usize {
        self.live
    }

    /// Whether there are no live pairs.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Insert a pair. If a live pair for `(pair.i, pair.j)` already
    /// exists, it is tombstoned first: there is at most one live
    /// pair per index pair. The freshly-inserted pair's key is
    /// stamped into the returned [`PairKey`] and into the pair's
    /// own `key` field (the `pair` argument is consumed).
    pub fn insert(&mut self, mut pair: Pair) -> PairKey {
        let key = PairKey(self.next_key);
        self.next_key += 1;
        pair.key = key;

        // If an existing live pair shares the same (i, j), tombstone
        // its slot first.
        let idx_key = (pair.i, pair.j);
        if let Some(&old_idx) = self.by_indices.get(&idx_key) {
            // by_indices invariant: it only points at live slots.
            debug_assert!(
                !self.tombstones[old_idx],
                "by_indices held a tombstoned slot"
            );
            self.tombstones[old_idx] = true;
            self.lcm_divmasks[old_idx] = 0;
            self.live -= 1;
        }

        // Append the new pair.
        let idx = self.pairs.len();
        let sorted_key = SortedKey {
            sugar: pair.sugar,
            arrival: pair.arrival,
            idx,
        };
        self.lcm_divmasks.push(pair.lcm_divmask);
        self.tombstones.push(false);
        self.pairs.push(pair);
        self.by_indices.insert(idx_key, idx);
        self.sorted.push(Reverse(sorted_key));
        self.live += 1;
        key
    }

    /// Pop the smallest-sugar / oldest-arrival live pair.
    ///
    /// Tombstoned entries in the sorted index are skipped on the way
    /// through. Returns a *clone* of the live pair (the storage
    /// slot is then tombstoned in place — see module docs).
    pub fn pop(&mut self) -> Option<Pair> {
        while let Some(Reverse(top)) = self.sorted.pop() {
            if self.tombstones[top.idx] {
                continue;
            }
            // Live slot. Clone the pair before tombstoning so the
            // caller gets ownership; the storage is then marked dead
            // and lcm_divmasks zeroed.
            let pair = self.pairs[top.idx].clone();
            // Drop the by_indices mapping (it should still point at
            // this slot).
            let key = (pair.i, pair.j);
            if self.by_indices.get(&key) == Some(&top.idx) {
                self.by_indices.remove(&key);
            }
            self.tombstones[top.idx] = true;
            self.lcm_divmasks[top.idx] = 0;
            self.live -= 1;
            return Some(pair);
        }
        debug_assert_eq!(
            self.live, 0,
            "LSet live count {} but sorted heap empty",
            self.live
        );
        None
    }

    /// Delete the live pair for `(i, j)` if any. Returns `true` if
    /// a live pair was actually deleted.
    pub fn delete(&mut self, i: u32, j: u32) -> bool {
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        let Some(idx) = self.by_indices.remove(&(i, j)) else {
            return false;
        };
        debug_assert!(
            !self.tombstones[idx],
            "by_indices held a tombstoned slot in delete"
        );
        self.tombstones[idx] = true;
        self.lcm_divmasks[idx] = 0;
        self.live -= 1;
        true
    }

    /// Whether `(i, j)` currently has a live pair.
    pub fn contains(&self, i: u32, j: u32) -> bool {
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        match self.by_indices.get(&(i, j)) {
            // by_indices invariant guarantees the slot is live.
            Some(&idx) => {
                debug_assert!(!self.tombstones[idx]);
                true
            }
            None => false,
        }
    }

    /// Iterate live pairs in undefined order. Useful for diagnostics
    /// and testing only; not a hot path.
    pub fn iter_live(&self) -> impl Iterator<Item = &Pair> + '_ {
        self.pairs
            .iter()
            .zip(self.tombstones.iter())
            .filter_map(|(pair, dead)| if *dead { None } else { Some(pair) })
    }

    /// Iterate live pairs whose `lcm_divmask` is a *superset* of
    /// `subset_mask`. SIMD-batched via [`crate::simd::find_divmask_match`]
    /// over the parallel `lcm_divmasks` array.
    ///
    /// **Tombstone safety:** dead slots have `lcm_divmasks[idx] ==
    /// 0`. The predicate `(subset_mask & !lcm_divmasks[idx]) == 0`
    /// is true at a dead slot iff `subset_mask == 0`. The Phase-2
    /// chain-criterion call sites always pass a non-zero
    /// `h_lm_divmask` (it's the divmask of a nonzero leading
    /// monomial — at least one bit must be set), so the iterator
    /// never yields a dead slot in practice. `assert_canonical`
    /// nails this down explicitly.
    pub fn iter_filtered_subset(
        &self,
        subset_mask: u64,
    ) -> FilteredSubsetIter<'_> {
        FilteredSubsetIter {
            pairs: &self.pairs,
            divmasks: &self.lcm_divmasks,
            tombstones: &self.tombstones,
            subset_mask,
            cursor: 0,
        }
    }

    /// Debug-only invariant check.
    pub fn assert_canonical(&self, ring: &crate::ring::Ring) {
        // 1. Lockstep lengths.
        assert_eq!(
            self.pairs.len(),
            self.tombstones.len(),
            "pairs / tombstones length mismatch"
        );
        assert_eq!(
            self.pairs.len(),
            self.lcm_divmasks.len(),
            "pairs / lcm_divmasks length mismatch"
        );

        // 2. Every live slot has matching divmask cache; every dead
        //    slot has lcm_divmasks[idx] == 0.
        let mut live_count = 0usize;
        for (idx, (pair, dead)) in self.pairs.iter().zip(self.tombstones.iter()).enumerate() {
            if *dead {
                assert_eq!(
                    self.lcm_divmasks[idx], 0,
                    "tombstoned slot {idx} has non-zero lcm_divmask {}",
                    self.lcm_divmasks[idx]
                );
            } else {
                live_count += 1;
                pair.assert_canonical(ring);
                assert_eq!(
                    self.lcm_divmasks[idx], pair.lcm_divmask,
                    "live slot {idx} lcm_divmasks out of sync"
                );
                let got = self.by_indices.get(&(pair.i, pair.j));
                assert_eq!(
                    got,
                    Some(&idx),
                    "by_indices disagreement for live pair at slot {idx}: {:?}",
                    (pair.i, pair.j)
                );
            }
        }

        // 3. Live count matches.
        assert_eq!(
            live_count, self.live,
            "live count {} disagrees with non-tombstoned slot count {}",
            self.live, live_count
        );

        // 4. by_indices only references live slots, and the slot's
        //    pair has the matching (i, j).
        for ((i, j), &idx) in &self.by_indices {
            assert!(
                idx < self.pairs.len(),
                "by_indices references oob slot {idx}"
            );
            assert!(
                !self.tombstones[idx],
                "by_indices references tombstoned slot {idx}"
            );
            assert_eq!(
                (self.pairs[idx].i, self.pairs[idx].j),
                (*i, *j),
                "by_indices key vs slot pair mismatch"
            );
        }
    }
}

/// Iterator yielded by [`LSet::iter_filtered_subset`]. Walks
/// `lcm_divmasks` via [`crate::simd::find_divmask_match`] and
/// returns `&pairs[idx]` for each hit.
pub struct FilteredSubsetIter<'a> {
    pairs: &'a [Pair],
    divmasks: &'a [u64],
    tombstones: &'a [bool],
    subset_mask: u64,
    cursor: usize,
}

impl<'a> Iterator for FilteredSubsetIter<'a> {
    type Item = &'a Pair;

    fn next(&mut self) -> Option<Self::Item> {
        // Divisibility predicate for "h_lm divides pair.lcm" is
        // `(h_lm_divmask & !pair.lcm_divmask) == 0` — every bit
        // set in `h_lm_divmask` must also be set in
        // `pair.lcm_divmask`. That's the *superset* test on the
        // candidate's divmask, dispatched through
        // `find_divmask_superset_match` (which looks for the first
        // `i >= cursor` with `(subset_mask & !divmasks[i]) == 0`).
        //
        // Tombstone safety: dead slots have `divmasks[idx] == 0`,
        // and `(subset_mask & !0) == subset_mask`, which is zero
        // only when `subset_mask == 0` — a degenerate input
        // forbidden at the Phase-2 call sites (every leading
        // monomial has at least one set bit). The defensive
        // tombstone check below catches the
        // `subset_mask == 0` corner.
        let hit = crate::simd::find_divmask_superset_match(
            self.divmasks,
            self.subset_mask,
            self.cursor,
        );
        if hit >= self.divmasks.len() {
            self.cursor = self.divmasks.len();
            return None;
        }
        // Defensive: even though the divmask = 0 invariant means
        // we should never land on a tombstoned slot for a non-zero
        // mask, guard against the subset_mask == 0 corner just in
        // case a future caller passes it.
        debug_assert!(
            !self.tombstones[hit] || self.subset_mask == 0,
            "filtered scan landed on tombstoned slot {hit} with non-zero subset_mask {:#x}",
            self.subset_mask
        );
        if self.tombstones[hit] {
            self.cursor = hit + 1;
            return self.next();
        }
        self.cursor = hit + 1;
        Some(&self.pairs[hit])
    }
}

#[cfg(test)]
mod tests {
    //! Backend-internal tests. The cross-backend contract tests
    //! live in `tests/lset_contract.rs`; what's here is specific
    //! to the flat backend's tombstone bookkeeping (the heap
    //! backend has nothing to test for these — it has no
    //! tombstone vector or zeroed-divmask invariant).

    use super::*;
    use crate::field::Field;
    use crate::monomial::Monomial;
    use crate::ordering::MonoOrder;
    use crate::ring::Ring;

    fn mk_ring(nvars: u32) -> Ring {
        Ring::new(nvars, MonoOrder::DegRevLex, Field::new(32003).unwrap()).unwrap()
    }

    fn mk_pair(r: &Ring, i: u32, j: u32, sugar: u32, arrival: u64) -> Pair {
        let lcm = Monomial::from_exponents(r, &vec![1u32; r.nvars() as usize]).unwrap();
        Pair::new(i, j, lcm, r, sugar, arrival)
    }

    #[test]
    fn tombstone_zeros_divmask() {
        let r = mk_ring(3);
        let mut l = LSet::new();
        l.insert(mk_pair(&r, 0, 1, 7, 0));
        l.insert(mk_pair(&r, 0, 2, 3, 1));
        // Trigger a tombstone via re-insert on (0, 1).
        l.insert(mk_pair(&r, 0, 1, 5, 2));
        l.assert_canonical(&r);
        // Slot 0 must be tombstoned with divmask = 0.
        assert!(l.tombstones[0]);
        assert_eq!(l.lcm_divmasks[0], 0);
        // Slot 1 (the live (0,2) pair) and slot 2 (the new (0,1)
        // pair) are live with the correct divmasks.
        assert!(!l.tombstones[1]);
        assert!(!l.tombstones[2]);
        assert_eq!(l.lcm_divmasks[1], l.pairs[1].lcm_divmask);
        assert_eq!(l.lcm_divmasks[2], l.pairs[2].lcm_divmask);
    }

    #[test]
    fn pop_zeros_divmask_and_tombstones() {
        let r = mk_ring(3);
        let mut l = LSet::new();
        l.insert(mk_pair(&r, 0, 1, 5, 0));
        let popped = l.pop().unwrap();
        assert_eq!((popped.i, popped.j), (0, 1));
        l.assert_canonical(&r);
        assert!(l.tombstones[0]);
        assert_eq!(l.lcm_divmasks[0], 0);
        assert_eq!(l.live, 0);
    }

    #[test]
    fn delete_zeros_divmask_and_tombstones() {
        let r = mk_ring(3);
        let mut l = LSet::new();
        l.insert(mk_pair(&r, 2, 5, 4, 0));
        assert!(l.delete(5, 2));
        l.assert_canonical(&r);
        assert!(l.tombstones[0]);
        assert_eq!(l.lcm_divmasks[0], 0);
    }

    #[test]
    fn iter_filtered_subset_skips_tombstones() {
        let r = mk_ring(3);
        let mut l = LSet::new();
        l.insert(mk_pair(&r, 0, 1, 1, 0));
        l.insert(mk_pair(&r, 0, 2, 2, 1));
        l.insert(mk_pair(&r, 1, 2, 3, 2));
        // Tombstone (0, 2) by re-insert.
        l.insert(mk_pair(&r, 0, 2, 9, 3));
        l.assert_canonical(&r);
        // All live pairs share the same LCM (xyz) because mk_pair
        // hands out [1,1,1] across the board, so any non-zero
        // subset of that lcm_divmask should match every live slot.
        let any_pair_mask = l.pairs[0].lcm_divmask;
        assert!(any_pair_mask != 0, "test setup: lcm_divmask must be non-zero");
        let yielded: Vec<(u32, u32)> = l
            .iter_filtered_subset(any_pair_mask)
            .map(|p| (p.i, p.j))
            .collect();
        // The originally-inserted (0, 2) at slot 1 was tombstoned;
        // the iterator must yield exactly the three live pairs.
        assert_eq!(yielded.len(), 3);
        assert!(yielded.contains(&(0, 1)));
        assert!(yielded.contains(&(0, 2)));
        assert!(yielded.contains(&(1, 2)));
    }
}
