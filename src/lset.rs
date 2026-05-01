//! `LSet` — priority queue of S-pairs with O(1) lookup.
//!
//! Design reference: `~/project/docs/rust-bba-port-plan.md` §7.3.
//!
//! The primary store is a `BinaryHeap<Reverse<HeapEntry>>` where each
//! `HeapEntry` carries a `Pair` plus its [`PairKey`]. A separate
//! `HashSet` tracks keys of pairs that have been logically deleted
//! but still sit in the heap (tombstone-on-pop). A `HashMap<(i, j),
//! PairKey>` gives O(1) access from index pair to the key of the
//! currently-live pair for those indices, which is how the
//! Gebauer–Möller chain criterion asks for deletions.
//!
//! ## Why tombstones (for now)
//!
//! The alternatives are `keyed_priority_queue` (external dep;
//! supports real deletion) or a `BTreeSet` keyed by
//! `(sugar, arrival, key)` (works, but slower inserts in practice).
//! Tombstones work without a new dependency and keep the hot path —
//! `pop` and `insert` — O(log n). If tombstone churn becomes a
//! measurable problem during the bba driver task, the layout
//! switches to `keyed_priority_queue`; see the port plan §7.3.
//!
//! ## Threading
//!
//! `LSet` is currently `Send + Sync` under `&mut self` mutation —
//! nothing interior. The parallel driver will wrap it in a mutex or
//! switch to a lock-free design; that's not this task's problem.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::pair::{Pair, PairKey};

/// Heap entry: the pair plus its identity. `Ord` delegates to `Pair`
/// but the key is compared last so two heap entries with the same
/// `(sugar, arrival, i, j)` but different keys are still considered
/// distinct (the caller has `(sugar, arrival)` collisions under
/// control in practice — arrival is monotonic — but the tie-break
/// keeps `HashSet::remove` semantics unambiguous).
#[derive(Clone, Debug)]
struct HeapEntry {
    pair: Pair,
    key: PairKey,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.pair == other.pair
    }
}
impl Eq for HeapEntry {}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.pair.cmp(&other.pair).then(self.key.cmp(&other.key))
    }
}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// S-pair queue ordered by sugar, ties broken by arrival.
///
/// The externally-visible contract is "pop the smallest-sugar /
/// oldest-arrival pair, or skip it if it was deleted". The deleted-
/// pair book-keeping is hidden; `len` returns the live count.
#[derive(Debug, Default)]
pub struct LSet {
    heap: BinaryHeap<Reverse<HeapEntry>>,
    deleted: HashSet<PairKey>,
    by_indices: HashMap<(u32, u32), PairKey>,
    next_key: u64,
    live: usize,
}

impl LSet {
    /// Empty queue.
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            deleted: HashSet::new(),
            by_indices: HashMap::new(),
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
    /// stamped into the returned [`PairKey`], and into the pair's
    /// own `key` field (the `pair` argument is consumed).
    pub fn insert(&mut self, mut pair: Pair) -> PairKey {
        let key = PairKey(self.next_key);
        self.next_key += 1;
        pair.key = key;

        // If an existing live pair shares the same (i, j), tombstone
        // it. The live count for that old pair is subtracted; the
        // new pair replaces it.
        let idx_key = (pair.i, pair.j);
        if let Some(old_key) = self.by_indices.insert(idx_key, key)
            && !self.deleted.contains(&old_key)
        {
            self.deleted.insert(old_key);
            self.live -= 1;
        }

        self.heap.push(Reverse(HeapEntry {
            pair: pair.clone(),
            key,
        }));
        self.live += 1;
        key
    }

    /// Pop the smallest-sugar / oldest-arrival live pair.
    ///
    /// Tombstoned entries are skipped (and their tombstones consumed
    /// on the way through).
    pub fn pop(&mut self) -> Option<Pair> {
        while let Some(Reverse(entry)) = self.heap.pop() {
            if self.deleted.remove(&entry.key) {
                continue;
            }
            // Live pair: remove the index mapping (if it still
            // points at this key — it may have been overwritten by a
            // later insert on the same (i, j) in which case the old
            // entry is already tombstoned and we wouldn't be here).
            if self.by_indices.get(&(entry.pair.i, entry.pair.j)) == Some(&entry.key) {
                self.by_indices.remove(&(entry.pair.i, entry.pair.j));
            }
            self.live -= 1;
            return Some(entry.pair);
        }
        debug_assert_eq!(self.live, 0, "LSet live count {} but heap empty", self.live);
        None
    }

    /// Delete the live pair for `(i, j)` if any. Returns `true` if
    /// a live pair was actually deleted.
    pub fn delete(&mut self, i: u32, j: u32) -> bool {
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        let Some(key) = self.by_indices.remove(&(i, j)) else {
            return false;
        };
        if self.deleted.insert(key) {
            self.live -= 1;
            true
        } else {
            // Shouldn't happen: by_indices only stores live keys.
            debug_assert!(false, "by_indices held a pre-tombstoned key");
            false
        }
    }

    /// Whether `(i, j)` currently has a live pair.
    pub fn contains(&self, i: u32, j: u32) -> bool {
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        match self.by_indices.get(&(i, j)) {
            Some(key) => !self.deleted.contains(key),
            None => false,
        }
    }

    /// Iterate live pairs in undefined order. Useful for diagnostics
    /// and testing only; not a hot path.
    pub fn iter_live(&self) -> impl Iterator<Item = &Pair> + '_ {
        self.heap.iter().filter_map(|Reverse(entry)| {
            if self.deleted.contains(&entry.key) {
                None
            } else {
                Some(&entry.pair)
            }
        })
    }

    /// Iterate live pairs whose `lcm_divmask` is a *superset* of
    /// `subset_mask` — i.e. `(subset_mask & !pair.lcm_divmask) == 0`.
    ///
    /// This is the chain-criterion Phase-2 fast-reject: a candidate
    /// pair `p` survives the divmask filter iff every bit set in the
    /// fresh element's `h_lm_divmask` is also set in `p.lcm_divmask`
    /// (a necessary condition for `h_lm | p.lcm`). This iterator
    /// pre-filters by divmask only — the caller still has to run
    /// the exact `divides` check on each yielded element.
    ///
    /// On the heap backend this is a trivial wrapper around
    /// [`iter_live`] with a scalar per-element filter. The flat
    /// backend (`lset_flat::LSet`, behind `cargo --features
    /// flat_lset`) implements the same API with a SIMD-batched
    /// scan over a parallel `lcm_divmasks: Vec<u64>` array — same
    /// semantics, much higher throughput. Call sites work against
    /// either backend without `#[cfg]`.
    pub fn iter_filtered_subset(
        &self,
        subset_mask: u64,
    ) -> impl Iterator<Item = &Pair> + '_ {
        self.iter_live()
            .filter(move |p| (subset_mask & !p.lcm_divmask) == 0)
    }

    /// Debug-only invariant check.
    pub fn assert_canonical(&self, ring: &crate::ring::Ring) {
        // Every live pair in the heap has a matching by_indices entry.
        // The by_indices hashes are all live (not deleted).
        let mut live_in_heap = 0usize;
        for Reverse(entry) in self.heap.iter() {
            if self.deleted.contains(&entry.key) {
                continue;
            }
            live_in_heap += 1;
            entry.pair.assert_canonical(ring);
            let got = self.by_indices.get(&(entry.pair.i, entry.pair.j));
            assert_eq!(
                got,
                Some(&entry.key),
                "by_indices disagreement for live pair {:?}",
                (entry.pair.i, entry.pair.j)
            );
        }
        assert_eq!(
            live_in_heap, self.live,
            "live count {} disagrees with heap live-entry count {}",
            self.live, live_in_heap
        );
        for key in self.by_indices.values() {
            assert!(
                !self.deleted.contains(key),
                "by_indices holds a deleted key"
            );
        }
    }
}

// Public API contract tests live in `tests/lset_contract.rs` so
// the same test bodies exercise whichever backend `cargo
// --features flat_lset` selects. Backend-specific invariant
// tests (none today; the heap backend has nothing analogous to
// the flat backend's tombstone-vector lockstep) would live here
// under `#[cfg(test)] mod tests`.
