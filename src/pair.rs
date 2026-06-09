//! S-pairs (`Pair`) and pair keys (`PairKey`).
//!
//! A [`Pair`] describes a potential S-polynomial pair `(S[i], S[j])`
//! where `j > i`. It carries the LCM of the two leading monomials
//! (with its sev pre-computed), the pair's sugar degree and an
//! arrival counter that lets the heap order "older pairs first" on
//! sugar ties.
//!
//! The ordering is designed for a `BinaryHeap<Reverse<Pair>>`:
//! `Pair: Ord` is ascending on `(sugar, arrival, i, j)`, so wrapping
//! in `Reverse` makes `pop()` yield the smallest-sugar / oldest-
//! arrival pair first. This matches Singular's `posInL17` behaviour
//! used by `std` / `bba` — see `~/Singular/kernel/GBEngine/kutil.cc`
//! (the `posInL17` selector) and the port plan §7.3.
//!
//! [`PairKey`] is a fresh identity assigned at insert time. The
//! `LSet` keeps it in the heap entries so tombstone-on-pop can tell
//! two distinct pairs apart even if they somehow share `(i, j,
//! sugar)` (which the G-M code can request when regenerating a pair
//! whose earlier instance was deleted).

use std::cmp::Ordering;

use crate::monomial::Monomial;

/// Opaque identity of a pair in an [`LSet`](crate::lset::LSet).
///
/// Fresh per insert; never recycled within the lifetime of a single
/// `LSet`. Used as the tombstone key, so the heap entry and the
/// hash index both know which exact pair they refer to.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct PairKey(pub u64);

/// An S-pair candidate.
///
/// The pair represents the polynomial
/// `c_j · m_i · S[i] − c_i · m_j · S[j]` where
/// `m_i = lcm / lm(S[i])`, `m_j = lcm / lm(S[j])`,
/// `c_i = lm_coeff(S[i])`, `c_j = lm_coeff(S[j])`. The S-polynomial
/// itself is built by [`LObject::from_spoly`](crate::lobject::LObject::from_spoly)
/// using this data.
#[derive(Clone, Debug)]
pub struct Pair {
    /// Smaller basis index. `i < j` by construction.
    pub i: u32,
    /// Larger basis index.
    pub j: u32,
    /// LCM of `lm(S[i])` and `lm(S[j])`.
    pub lcm: Monomial,
    /// Cached short exponent vector of `lcm` — pre-computed (via
    /// `Monomial::compute_sev`, per ADR-019) so the chain criterion's
    /// sev pre-filter is a direct u64 load.
    pub lcm_sev: u64,
    /// Cached divmask of `lcm` (ADR-025). Pre-computed via
    /// `Ring::divmask_of`. Used by the chain criterion's divmask
    /// fast-reject — strictly stronger than `lcm_sev` (encodes
    /// exponent ranges, not just nonzero/zero).
    pub lcm_divmask: u64,
    /// Order-preserving degrevlex key of `lcm` (ADR-034). Pre-computed
    /// via `Monomial::degrevlex_key`. The L-queue uses this as the
    /// tie-break among equal-sugar pairs when the `pairorder_lm`
    /// feature is enabled, so that `pop()` yields the
    /// smallest-degrevlex LCM (Singular's `compareL15` top()). Computed
    /// unconditionally (it's cheap and the ring is in hand here); the
    /// feature only chooses whether the L-queue *reads* it.
    pub lcm_ord_key: [u64; 4],
    /// Sugar degree of the pair: `max(sugar(S[i]) + deg(m_i),
    /// sugar(S[j]) + deg(m_j))`. For the bootstrap where inputs
    /// carry `sugar = lm_deg`, this is equivalent to the LCM's total
    /// degree; we still store an explicit field so the future bba
    /// driver can carry a sharper sugar through reductions.
    pub sugar: u32,
    /// Monotonic insertion counter. On ties in `sugar`, the pair
    /// with the smaller `arrival` comes out first.
    pub arrival: u64,
    /// Opaque key assigned when the pair enters an `LSet`. The
    /// constructor [`Pair::new`] sets this to a sentinel value; the
    /// `LSet` overwrites it at insert time.
    pub key: PairKey,
}

impl Pair {
    /// Build a fresh pair. `i < j` is a precondition; the constructor
    /// swaps them if the caller gave them in the wrong order so that
    /// downstream code can rely on `i < j`.
    ///
    /// `sugar` and `arrival` must be supplied by the caller — they
    /// depend on the basis state at the moment the pair is created.
    ///
    /// The `key` field is initialised to `PairKey(0)` and will be
    /// overwritten by [`LSet::insert`](crate::lset::LSet::insert);
    /// callers that never hand the pair to an `LSet` may read a stale
    /// key, which is harmless.
    pub fn new(
        i: u32,
        j: u32,
        lcm: Monomial,
        ring: &crate::ring::Ring,
        sugar: u32,
        arrival: u64,
    ) -> Self {
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        debug_assert!(i != j, "degenerate pair with i == j");
        // ADR-019: SEV computed on demand from lcm; ring required.
        // ADR-025: divmask alongside SEV.
        let lcm_sev = lcm.compute_sev(ring);
        let lcm_divmask = ring.divmask_of(&lcm);
        // ADR-034: order-preserving degrevlex key for the (sugar, LCM)
        // pair tie-break. Computed here unconditionally — the ring is
        // in hand and the cost is one XOR-reorder of four words.
        let lcm_ord_key = lcm.degrevlex_key(ring);
        Self {
            i,
            j,
            lcm,
            lcm_sev,
            lcm_divmask,
            lcm_ord_key,
            sugar,
            arrival,
            key: PairKey(0),
        }
    }

    /// Debug-only invariant check.
    pub fn assert_canonical(&self, ring: &crate::ring::Ring) {
        assert!(self.i < self.j, "pair indices not ordered");
        self.lcm.assert_canonical(ring);
        assert_eq!(
            self.lcm_sev,
            self.lcm.compute_sev(ring),
            "lcm_sev cache mismatch"
        );
        assert_eq!(
            self.lcm_divmask,
            ring.divmask_of(&self.lcm),
            "lcm_divmask cache mismatch (ADR-025)"
        );
        assert_eq!(
            self.lcm_ord_key,
            self.lcm.degrevlex_key(ring),
            "lcm_ord_key cache mismatch (ADR-034)"
        );
    }
}

// Ordering: ascending on (sugar, <tie-break>, i, j). Wrap in `Reverse`
// when using `BinaryHeap` so the smallest comes out first.
//
// The tie-break among equal-sugar pairs depends on the `pairorder_lm`
// feature (ADR-034):
// * OFF (default): `arrival` — insertion order, byte-for-byte the
//   prior behaviour.
// * ON: `lcm_ord_key` — the LCM's order-preserving degrevlex key, so
//   `pop()` (the minimum) yields the smallest-degrevlex LCM, matching
//   Singular's `compareL15`. `arrival` is folded in after `lcm_ord_key`
//   as a final deterministic tie so two pairs with the same
//   `(sugar, lcm)` still order stably (the LCM equality case the
//   product/chain criteria allow).
//
// Both LSet backends must agree: `lset.rs`'s `HeapEntry` delegates here
// via `Pair::cmp`, and `lset_flat.rs`'s `SortedKey` mirrors this exact
// key under the same feature gate. Keeping them consistent ensures the
// two backends produce the same pair sequence (the cross-backend
// contract tests rely on it).
impl Ord for Pair {
    fn cmp(&self, other: &Self) -> Ordering {
        let by_sugar = self.sugar.cmp(&other.sugar);
        #[cfg(not(feature = "pairorder_lm"))]
        let tie = || self.arrival.cmp(&other.arrival);
        #[cfg(feature = "pairorder_lm")]
        let tie = || {
            self.lcm_ord_key
                .cmp(&other.lcm_ord_key)
                .then_with(|| self.arrival.cmp(&other.arrival))
        };
        by_sugar
            .then_with(tie)
            .then_with(|| self.i.cmp(&other.i))
            .then_with(|| self.j.cmp(&other.j))
    }
}
impl PartialOrd for Pair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for Pair {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Pair {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Field;
    use crate::monomial::Monomial;
    use crate::ordering::MonoOrder;
    use crate::ring::Ring;
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    fn mk_ring(nvars: u32) -> Ring {
        Ring::new(nvars, MonoOrder::DegRevLex, Field::new(32003).unwrap()).unwrap()
    }

    fn lcm_mono(r: &Ring, exps: &[u32]) -> Monomial {
        Monomial::from_exponents(r, exps).unwrap()
    }

    #[test]
    fn new_swaps_indices() {
        let r = mk_ring(3);
        let l = lcm_mono(&r, &[1, 1, 0]);
        let p = Pair::new(5, 2, l, &r, 4, 0);
        assert_eq!(p.i, 2);
        assert_eq!(p.j, 5);
    }

    #[test]
    fn binary_heap_pops_smallest_sugar_first() {
        let r = mk_ring(3);
        let l = lcm_mono(&r, &[1, 1, 0]);
        let mut h = BinaryHeap::new();
        h.push(Reverse(Pair::new(0, 1, l.clone(), &r, 7, 0)));
        h.push(Reverse(Pair::new(0, 2, l.clone(), &r, 3, 1)));
        h.push(Reverse(Pair::new(1, 2, l.clone(), &r, 5, 2)));
        let first = h.pop().unwrap().0;
        assert_eq!(first.sugar, 3);
        let second = h.pop().unwrap().0;
        assert_eq!(second.sugar, 5);
        let third = h.pop().unwrap().0;
        assert_eq!(third.sugar, 7);
    }

    #[test]
    fn arrival_breaks_sugar_tie() {
        let r = mk_ring(3);
        let l = lcm_mono(&r, &[1, 1, 0]);
        let mut h = BinaryHeap::new();
        h.push(Reverse(Pair::new(0, 3, l.clone(), &r, 5, 10)));
        h.push(Reverse(Pair::new(0, 2, l.clone(), &r, 5, 5)));
        h.push(Reverse(Pair::new(0, 4, l.clone(), &r, 5, 20)));
        let a = h.pop().unwrap().0;
        assert_eq!(a.arrival, 5);
        let b = h.pop().unwrap().0;
        assert_eq!(b.arrival, 10);
        let c = h.pop().unwrap().0;
        assert_eq!(c.arrival, 20);
    }
}
