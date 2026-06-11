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
    /// ADR-040 (`input_tiebreak`): order-preserving degrevlex key of the
    /// **ordering monomial** — the leading monomial of the actual
    /// S-polynomial (the `ksCreateShortSpoly` result), NOT the LCM. For
    /// an input L-entry this is just the input's leading monomial (=
    /// `lcm`, since `lcm` holds the input LM for inputs), so
    /// `spoly_ord_key == lcm_ord_key` for inputs. For an S-pair it is the
    /// short-spoly LM's key, which is strictly below the LCM. The L-queue
    /// uses this instead of `lcm_ord_key` when `input_tiebreak` is on, so
    /// the pop order matches Singular's `compareL15`
    /// (`pLmCmp(strat->P.p)`, where `strat->P.p` is the short spoly).
    /// Without the feature this field is absent and the queue keys on
    /// `lcm_ord_key` (ADR-034) exactly as before.
    #[cfg(feature = "input_tiebreak")]
    pub spoly_ord_key: [u64; 4],
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
    /// ADR-039 (`seed_in_l` feature): the input-generator payload for
    /// an **input L-entry**. `None` for a real S-pair; `Some(poly)`
    /// for an input seeded into `L` (Singular's `initSL` push of a
    /// `p1==NULL` LObject). When `Some`, the entry's `i`/`j` are the
    /// sentinel `(u32::MAX, input_seq)` (so `by_indices` keys never
    /// collide across distinct inputs and never alias a real pair),
    /// `lcm` is the input's own leading monomial (so `lcm_ord_key`
    /// yields the `compareL15` `pLmCmp` tie-break), and `sugar` is the
    /// input's `pFDeg` = leading-monomial total degree (matching
    /// `initEcartBBA`, which sets `ecart = 0`, so the L-order key is
    /// `pFDeg + 0`). The main loop pops it, reduces the carried `Poly`
    /// against the current basis, and inserts the survivor — without
    /// emitting a `POP` trace event (Singular suppresses POP for
    /// `p1==NULL` entries) and without ever being chain-pruned
    /// (Singular's L-side chain crit guards `it->p1 != NULL`).
    #[cfg(feature = "seed_in_l")]
    pub input: Option<crate::poly::Poly>,
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
    ///
    /// ADR-035: `lcm_sev` / `lcm_divmask` are recomputed from the
    /// LCM's exponents here. The OR-composed fast path
    /// ([`Pair::new_from_masks`], `pair_mask_or` feature) instead
    /// passes them in as `mask(a) | mask(b)`; see that constructor and
    /// ADR-035. This base constructor is the canonical recompute path
    /// and is always available regardless of feature state.
    pub fn new(
        i: u32,
        j: u32,
        lcm: Monomial,
        ring: &crate::ring::Ring,
        sugar: u32,
        arrival: u64,
    ) -> Self {
        // ADR-019: SEV computed on demand from lcm; ring required.
        // ADR-025: divmask alongside SEV.
        let lcm_sev = lcm.compute_sev(ring);
        let lcm_divmask = ring.divmask_of(&lcm);
        Self::from_parts(i, j, lcm, ring, lcm_sev, lcm_divmask, sugar, arrival)
    }

    /// ADR-035: build a pair with the LCM's masks supplied by the
    /// caller as the OR of the two operands' cached masks.
    ///
    /// Because both the SEV (ADR-019/029) and divmask (ADR-025)
    /// schemes are *threshold-monotone* — a bit is set iff the
    /// exponent meets a fixed per-variable threshold — and the LCM is
    /// the componentwise max of the operands, the LCM's masks are
    /// **exactly** the bitwise OR of the operands' masks:
    ///
    /// ```text
    /// sev(lcm(a,b))     = sev(a)     | sev(b)
    /// divmask(lcm(a,b)) = divmask(a) | divmask(b)
    /// ```
    ///
    /// (`max(e_a, e_b) ≥ t  ⟺  e_a ≥ t ∨ e_b ≥ t`.) This replaces the
    /// per-pair `compute_sev` + `divmask_of` recompute with two ORs at
    /// the call site, where both operand masks are already cached.
    /// Debug builds re-verify the identity via
    /// [`Pair::assert_canonical`].
    #[inline]
    pub fn new_from_masks(
        i: u32,
        j: u32,
        lcm: Monomial,
        ring: &crate::ring::Ring,
        lcm_sev: u64,
        lcm_divmask: u64,
        sugar: u32,
        arrival: u64,
    ) -> Self {
        Self::from_parts(i, j, lcm, ring, lcm_sev, lcm_divmask, sugar, arrival)
    }

    /// Shared body of [`Pair::new`] and [`Pair::new_from_masks`]:
    /// index-swap, `lcm_ord_key` computation, struct assembly. The two
    /// public constructors differ only in how `lcm_sev` / `lcm_divmask`
    /// are obtained (recompute vs OR).
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn from_parts(
        i: u32,
        j: u32,
        lcm: Monomial,
        ring: &crate::ring::Ring,
        lcm_sev: u64,
        lcm_divmask: u64,
        sugar: u32,
        arrival: u64,
    ) -> Self {
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        debug_assert!(i != j, "degenerate pair with i == j");
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
            // ADR-040: default the spoly ordering key to the LCM key.
            // For S-pairs the pair-creation path overwrites this via
            // `set_spoly_ord_key` with the short-spoly LM key; if it
            // never does, the queue order falls back to the LCM key
            // (the pre-ADR-040 behaviour), which is a safe, total order.
            #[cfg(feature = "input_tiebreak")]
            spoly_ord_key: lcm_ord_key,
            sugar,
            arrival,
            key: PairKey(0),
            #[cfg(feature = "seed_in_l")]
            input: None,
        }
    }

    /// ADR-040 (`input_tiebreak`): set the S-polynomial ordering key
    /// (the short-spoly LM's degrevlex key). Called by the pair-creation
    /// path after computing the short spoly's leading monomial via
    /// [`crate::lobject::LObject::short_spoly_lm`]. Input L-entries keep
    /// the default (their own LM key) and never call this.
    #[cfg(feature = "input_tiebreak")]
    #[inline]
    pub fn set_spoly_ord_key(&mut self, key: [u64; 4]) {
        self.spoly_ord_key = key;
    }

    /// ADR-039 (`seed_in_l`): build an **input L-entry** carrying the
    /// pre-reduction input generator `poly`.
    ///
    /// `input_seq` is a fresh per-input sentinel (0, 1, 2, …) used as
    /// the entry's `j` while `i == u32::MAX`. The `(u32::MAX,
    /// input_seq)` index pair is unique per input and disjoint from
    /// every real pair's `(i, j)` (real basis indices are `< u32::MAX`),
    /// so the `LSet`'s `by_indices` map never tombstones one input
    /// against another, never aliases a pair, and the chain
    /// criterion's `delete(i, j)` / `contains(i, j)` (only ever called
    /// with real basis indices) never touch input entries.
    ///
    /// `lm` is the input's leading monomial; it becomes the entry's
    /// `lcm`, so the `(sugar, lcm_ord_key)` L-ordering reproduces
    /// `compareL15` = `(pFDeg + ecart, pLmCmp)` with `ecart = 0`
    /// (`initEcartBBA`). `sugar` must be the input's `pFDeg` (its
    /// leading-monomial total degree).
    ///
    /// `arrival` is handed out from the same monotonic counter as real
    /// pairs, so the final `(sugar, lcm_ord_key, arrival)` tie-break is
    /// deterministic.
    #[cfg(feature = "seed_in_l")]
    pub fn new_input(
        input_seq: u32,
        lm: Monomial,
        poly: crate::poly::Poly,
        ring: &crate::ring::Ring,
        sugar: u32,
        arrival: u64,
    ) -> Self {
        let lcm_sev = lm.compute_sev(ring);
        let lcm_divmask = ring.divmask_of(&lm);
        let lcm_ord_key = lm.degrevlex_key(ring);
        Self {
            i: u32::MAX,
            j: input_seq,
            lcm: lm,
            lcm_sev,
            lcm_divmask,
            lcm_ord_key,
            // ADR-040: an input L-entry's ordering monomial IS its own
            // leading monomial (Singular keys `p1==NULL` entries on
            // `pLmCmp(P.p)` = the input poly's LM), which is exactly
            // `lcm` here. So the spoly key equals the LCM key for inputs.
            #[cfg(feature = "input_tiebreak")]
            spoly_ord_key: lcm_ord_key,
            sugar,
            arrival,
            key: PairKey(0),
            input: Some(poly),
        }
    }

    /// ADR-039: `true` iff this entry is an input L-entry (carries an
    /// input-generator `Poly` rather than describing an S-pair).
    #[cfg(feature = "seed_in_l")]
    #[inline]
    pub fn is_input(&self) -> bool {
        self.input.is_some()
    }

    /// Debug-only invariant check.
    ///
    /// The recompute-and-compare on `lcm_sev` / `lcm_divmask` doubles
    /// as the ADR-035 OR-identity check: when the pair was built via
    /// [`Pair::new_from_masks`] (`pair_mask_or` feature) the cached
    /// masks were obtained by OR-ing the operands' masks, so these
    /// asserts verify `mask(a) | mask(b) == mask(lcm(a,b))` at every
    /// debug-build construction. A failure here means the mask scheme
    /// is not threshold-monotone (which would invalidate ADR-035).
    pub fn assert_canonical(&self, ring: &crate::ring::Ring) {
        // ADR-039: an input L-entry uses the sentinel `i == u32::MAX`
        // and is not subject to the `i < j` pair invariant.
        #[cfg(feature = "seed_in_l")]
        if self.is_input() {
            self.lcm.assert_canonical(ring);
            assert_eq!(self.lcm_sev, self.lcm.compute_sev(ring), "input lcm_sev cache mismatch");
            assert_eq!(self.lcm_divmask, ring.divmask_of(&self.lcm), "input lcm_divmask cache mismatch");
            assert_eq!(self.lcm_ord_key, self.lcm.degrevlex_key(ring), "input lcm_ord_key cache mismatch");
            // ADR-040: an input entry's spoly ordering key is its own LM
            // key (Singular keys `p1==NULL` entries on the input poly's
            // LM), which equals `lcm_ord_key` here.
            #[cfg(feature = "input_tiebreak")]
            assert_eq!(self.spoly_ord_key, self.lcm_ord_key, "input spoly_ord_key must equal lcm_ord_key");
            return;
        }
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
// The tie-break among equal-sugar pairs depends on the feature stack:
//
// * `pairorder_lm` OFF (default-of-defaults): `arrival` — insertion
//   order, byte-for-byte the prior behaviour.
// * `pairorder_lm` ON, `input_tiebreak` OFF (ADR-034): `lcm_ord_key` —
//   the LCM's order-preserving degrevlex key; `pop()` (the minimum)
//   yields the smallest-degrevlex LCM. `arrival` ascending is the final
//   stabilizer. This approximates `compareL15` but keys on the LCM.
// * `input_tiebreak` ON (ADR-040): `spoly_ord_key` — the degrevlex key
//   of the **S-polynomial's actual leading monomial** (the
//   `ksCreateShortSpoly` result for S-pairs; the input's own LM for
//   input entries), with `arrival` DESCENDING (LIFO) as the final
//   stabilizer. This mirrors Singular's `compareL15` EXACTLY:
//   `compareL15` is `(GetpFDeg+ecart, then pLmCmp(P.p)*OrdSgn)` where
//   `P.p` is the short spoly, NOT the LCM — and on the staging workload
//   the short-spoly LM differs from the LCM for 100 % of pairs, so
//   keying on the LCM (the `pairorder_lm`-only path) systematically
//   mis-orders pairs relative to Singular (task 394 root-cause). The
//   `std::multiset<LObject*>` Singular uses stores equal-`compareL15`
//   elements LIFO (`CompareLObject` returns `lhs.seq > rhs.seq` for the
//   non-FIFO comparators; `kInline.h:986-988`, next-opt), so descending
//   arrival reproduces the seq-LIFO stabilizer. `arrival` is globally
//   unique, so the `(i, j)` suffix never decides; it keeps `Ord` total.
//
// Both LSet backends must agree: `lset.rs`'s `HeapEntry` delegates here
// via `Pair::cmp`, and `lset_flat.rs`'s `SortedKey` mirrors this exact
// key under the same feature gate. Keeping them consistent ensures the
// two backends produce the same pair sequence (the cross-backend
// contract tests rely on it).
impl Ord for Pair {
    fn cmp(&self, other: &Self) -> Ordering {
        let by_sugar = self.sugar.cmp(&other.sugar);
        // ADR-040: arrival stabilizer is LIFO (descending) under
        // `input_tiebreak`, FIFO (ascending) otherwise.
        #[cfg(not(feature = "input_tiebreak"))]
        let arrival_tie = self.arrival.cmp(&other.arrival);
        #[cfg(feature = "input_tiebreak")]
        let arrival_tie = other.arrival.cmp(&self.arrival);
        #[cfg(not(feature = "pairorder_lm"))]
        let tie = || arrival_tie;
        // ADR-034 path (LCM key) — only when input_tiebreak is OFF.
        #[cfg(all(feature = "pairorder_lm", not(feature = "input_tiebreak")))]
        let tie = || self.lcm_ord_key.cmp(&other.lcm_ord_key).then(arrival_tie);
        // ADR-040 path (short-spoly LM key).
        #[cfg(feature = "input_tiebreak")]
        let tie = || self.spoly_ord_key.cmp(&other.spoly_ord_key).then(arrival_tie);
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
        // All three share sugar 5 and the same LCM (hence the same
        // lcm_ord_key / default spoly_ord_key), so the arrival
        // stabilizer alone orders them.
        // ADR-040: `input_tiebreak` flips the stabilizer to LIFO
        // (descending arrival); otherwise it is FIFO (ascending).
        #[cfg(not(feature = "input_tiebreak"))]
        let expected = [5u64, 10, 20];
        #[cfg(feature = "input_tiebreak")]
        let expected = [20u64, 10, 5];
        let a = h.pop().unwrap().0;
        assert_eq!(a.arrival, expected[0]);
        let b = h.pop().unwrap().0;
        assert_eq!(b.arrival, expected[1]);
        let c = h.pop().unwrap().0;
        assert_eq!(c.arrival, expected[2]);
    }

    /// ADR-039: an input L-entry is recognised via `is_input`, uses the
    /// sentinel `i == u32::MAX`, and orders by `(sugar, lcm_ord_key)`
    /// alongside real pairs (so a lower-sugar input pops before a
    /// higher-sugar pair). `assert_canonical` tolerates the sentinel.
    #[cfg(feature = "seed_in_l")]
    #[test]
    fn input_entry_orders_by_sugar_and_is_recognised() {
        use crate::field::{Coeff, Field};
        let r = Ring::new(3, MonoOrder::DegRevLex, Field::new(32003).unwrap()).unwrap();
        // A degree-1 input (sugar 1) and a degree-2 pair (sugar 2).
        let in_lm = Monomial::from_exponents(&r, &[1, 0, 0]).unwrap();
        let in_poly = crate::poly::Poly::monomial(&r, 1 as Coeff, in_lm.clone());
        let input = Pair::new_input(0, in_lm, in_poly, &r, 1, 0);
        assert!(input.is_input());
        assert_eq!(input.i, u32::MAX);
        input.assert_canonical(&r);

        let pair_lcm = Monomial::from_exponents(&r, &[1, 1, 0]).unwrap();
        let pair = Pair::new(0, 1, pair_lcm, &r, 2, 1);
        assert!(!pair.is_input());

        let mut h = BinaryHeap::new();
        h.push(Reverse(pair));
        h.push(Reverse(input));
        // Lower-sugar input pops first.
        let first = h.pop().unwrap().0;
        assert!(first.is_input(), "sugar-1 input must pop before sugar-2 pair");
        let second = h.pop().unwrap().0;
        assert!(!second.is_input());
    }
}
