//! `BSet` — transient per-survivor pair set.
//!
//! Design reference: `~/project/docs/rust-bba-port-plan.md` §7.4.
//!
//! The bba driver builds a `BSet` per newly-reduced survivor `h`:
//! one candidate pair per live basis element (filtered by the
//! product criterion). The chain criterion then dedups within the
//! set and the survivors merge into the main [`LSet`](crate::lset::LSet).
//!
//! `BSet` is unsorted (append-only `Vec<Pair>`) because the driver
//! walks it linearly and the hash index is enough for
//! remove-by-indices. No priority queue here.

use std::collections::HashMap;

use crate::pair::Pair;

/// Transient pair set built during `enterpairs`.
///
/// `Send + Sync` — just plain data. The driver owns it for the
/// duration of a single `enterpairs` call and drops it afterward.
#[derive(Debug, Default)]
pub struct BSet {
    pairs: Vec<Pair>,
    /// (i, j) → index into `pairs`. Never holds a stale mapping: on
    /// removal the last element is swapped in and the hash for the
    /// swapped-in pair is updated.
    by_indices: HashMap<(u32, u32), usize>,
}

impl BSet {
    /// Empty B set.
    pub fn new() -> Self {
        Self {
            pairs: Vec::new(),
            by_indices: HashMap::new(),
        }
    }

    /// Number of pairs.
    #[inline]
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// Whether the set is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// Push a pair. Panics in debug builds if `(i, j)` is already
    /// present — the caller is expected to build B without
    /// duplicates (the chain criterion handles chain-implied pairs,
    /// not literal duplicates).
    pub fn push(&mut self, pair: Pair) {
        let idx = self.pairs.len();
        let key = (pair.i, pair.j);
        debug_assert!(
            !self.by_indices.contains_key(&key),
            "BSet duplicate push of {:?}",
            key
        );
        self.by_indices.insert(key, idx);
        self.pairs.push(pair);
    }

    /// Borrow the raw pair slice.
    #[inline]
    pub fn pairs(&self) -> &[Pair] {
        &self.pairs
    }

    /// Remove the pair at `at` (swap-remove). Returns the removed
    /// pair. Keeps `by_indices` consistent.
    pub fn swap_remove(&mut self, at: usize) -> Pair {
        let removed = self.pairs.swap_remove(at);
        self.by_indices.remove(&(removed.i, removed.j));
        if at < self.pairs.len() {
            // The element previously at the end now lives at `at`.
            let moved = &self.pairs[at];
            self.by_indices.insert((moved.i, moved.j), at);
        }
        removed
    }

    /// Drain the entire set, consuming it into its constituent pairs.
    pub fn into_pairs(self) -> Vec<Pair> {
        self.pairs
    }

    /// Debug-only invariant check.
    pub fn assert_canonical(&self, ring: &crate::ring::Ring) {
        assert_eq!(self.pairs.len(), self.by_indices.len(), "index size");
        for (idx, pair) in self.pairs.iter().enumerate() {
            pair.assert_canonical(ring);
            let got = self.by_indices.get(&(pair.i, pair.j));
            assert_eq!(got, Some(&idx), "by_indices mismatch for {idx}");
        }
    }
}

#[cfg(test)]
mod tests {
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
        Pair::new(i, j, lcm, sugar, arrival)
    }

    #[test]
    fn push_and_swap_remove_maintain_index() {
        let r = mk_ring(3);
        let mut b = BSet::new();
        b.push(mk_pair(&r, 0, 1, 3, 0));
        b.push(mk_pair(&r, 0, 2, 4, 1));
        b.push(mk_pair(&r, 0, 3, 5, 2));
        b.assert_canonical(&r);
        let removed = b.swap_remove(1);
        assert_eq!((removed.i, removed.j), (0, 2));
        b.assert_canonical(&r);
        assert_eq!(b.len(), 2);
    }
}
