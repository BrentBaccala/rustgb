//! `SBasis` — growing basis of polynomials.
//!
//! Design reference: `~/project/docs/rust-bba-port-plan.md` §7.1.
//!
//! This is the single-threaded version of the design described
//! there. Polynomials live in a `Vec<Box<Poly>>` so `&Poly`
//! references stay valid across subsequent `insert` calls (the
//! future parallel sweep depends on this). Parallel arrays
//! `sevs`, `lm_degs`, `redundant`, `arrival` cache the per-element
//! metadata that the sweep and the Gebauer–Möller pair-criterion
//! machinery hot-read without touching the `Poly` itself.
//!
//! A follow-up task will swap `redundant: Vec<bool>` for
//! `Vec<AtomicBool>` and wrap `next_arrival` in `AtomicU64` once
//! parallelism lands; the current types satisfy the single-threaded
//! contract.

use crate::monomial::Monomial;
use crate::poly::Poly;
use crate::ring::Ring;

/// The running basis of a Groebner-basis computation.
///
/// Polynomials are owned; leading metadata (`sevs`, `lms`,
/// `lm_degs`) is cached in parallel arrays for the sweep's fast
/// path. `Send + Sync` by construction (only plain arrays of
/// owned data).
#[derive(Debug, Default)]
pub struct SBasis {
    /// The polynomials, in insertion order. `Box` so `&Poly` remains
    /// stable across vector growth — the port plan §7.1 specifies
    /// this layout so the future parallel sweep can hold `&Poly`
    /// across concurrent `insert` calls. Clippy's `vec_box` lint
    /// doesn't know about that requirement.
    #[allow(clippy::vec_box)]
    polys: Vec<Box<Poly>>,
    /// Leading short-exponent vectors. `sevs[i] == polys[i].lm_sev()`
    /// when `polys[i]` is nonzero, else 0.
    sevs: Vec<u64>,
    /// Leading monomial cache, kept in lockstep with `polys`.
    /// `lms[i] == polys[i].leading().unwrap().1.clone()`. Used by
    /// the divisor sweep in `bba::find_divisor_idx` (ADR-010) to
    /// avoid the `Vec<Box<Poly>>` pointer chase per probed
    /// candidate. Each `Monomial` is 48 bytes, so for ~3000-element
    /// staging bases the cache totals ~144 KB — fits in L2,
    /// streams cleanly during the sweep. See ADR-010 in
    /// `~/rustgb/docs/design-decisions.md`.
    lms: Vec<Monomial>,
    /// Leading total degrees. `lm_degs[i] == polys[i].lm_deg()`.
    lm_degs: Vec<u32>,
    /// Redundancy flags. `redundant[i] == true` means `polys[i]`'s
    /// leading monomial is divisible by some later `polys[j]`'s
    /// leading monomial, so `polys[i]` no longer produces useful
    /// divisors. The poly itself is retained (not compacted) for
    /// index stability.
    redundant: Vec<bool>,
    /// Insertion arrival IDs.
    arrival: Vec<u64>,
    /// Next arrival counter to hand out.
    next_arrival: u64,
}

impl SBasis {
    /// Empty basis.
    pub fn new() -> Self {
        Self {
            polys: Vec::new(),
            sevs: Vec::new(),
            lms: Vec::new(),
            lm_degs: Vec::new(),
            redundant: Vec::new(),
            arrival: Vec::new(),
            next_arrival: 0,
        }
    }

    /// Number of polynomials in the basis (redundant or otherwise).
    #[inline]
    pub fn len(&self) -> usize {
        self.polys.len()
    }

    /// Whether the basis has no polynomials.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.polys.is_empty()
    }

    /// Borrow the polynomial at `idx`.
    #[inline]
    pub fn poly(&self, idx: usize) -> &Poly {
        &self.polys[idx]
    }

    /// Whether the polynomial at `idx` has been marked redundant.
    #[inline]
    pub fn is_redundant(&self, idx: usize) -> bool {
        self.redundant[idx]
    }

    /// Slice of cached leading-term sevs. Length equals [`len`](Self::len).
    #[inline]
    pub fn sevs(&self) -> &[u64] {
        &self.sevs
    }

    /// Slice of cached leading monomials. Length equals [`len`](Self::len).
    /// `lms()[i] == polys[i].leading().unwrap().1.clone()`. ADR-010.
    #[inline]
    pub fn lms(&self) -> &[Monomial] {
        &self.lms
    }

    /// Slice of cached leading-term total degrees.
    #[inline]
    pub fn lm_degs(&self) -> &[u32] {
        &self.lm_degs
    }

    /// Slice of redundancy flags.
    #[inline]
    pub fn redundant_flags(&self) -> &[bool] {
        &self.redundant
    }

    /// Slice of arrival IDs.
    #[inline]
    pub fn arrivals(&self) -> &[u64] {
        &self.arrival
    }

    /// Iterate non-redundant `(idx, &Poly)` pairs.
    pub fn iter_active(&self) -> impl Iterator<Item = (usize, &Poly)> + '_ {
        self.polys
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.redundant[*i])
            .map(|(i, p)| (i, p.as_ref()))
    }

    /// Insert `h` into the basis and mark any existing basis element
    /// whose leading monomial is divisible by `lm(h)` as redundant.
    ///
    /// This mirrors Singular's `clearS` + `enterS` pattern. Returns
    /// the index at which `h` was placed. The polynomial must be
    /// nonzero; inserting zero panics in debug builds and no-ops in
    /// release (returns the would-be index without actually pushing).
    pub fn insert(&mut self, ring: &Ring, h: Poly) -> usize {
        let idx = self.insert_no_clear(h);
        self.clear_redundant_for(ring, idx);
        idx
    }

    /// Like [`insert`](Self::insert) but **does not** run the
    /// "mark older elements redundant" sweep. The bba driver uses
    /// this so it can generate pairs against not-yet-marked-redundant
    /// older elements first (matching Singular's `initenterpairs`
    /// before `clearS` ordering), then call
    /// [`clear_redundant_for`](Self::clear_redundant_for) to do the
    /// sweep.
    ///
    /// Returns the index at which `h` was placed. Debug-only
    /// assertion: `h` is nonzero.
    pub fn insert_no_clear(&mut self, h: Poly) -> usize {
        debug_assert!(!h.is_zero(), "SBasis::insert of zero polynomial");
        if h.is_zero() {
            return self.polys.len();
        }
        let lm_sev = h.lm_sev();
        let lm_deg = h.lm_deg();
        // Capture the leading monomial before the poly moves into
        // the Box. `unwrap` is safe: we just checked is_zero above.
        let lm = h.leading().expect("non-zero").1.clone();
        let idx = self.polys.len();
        let arrival = self.next_arrival;
        self.next_arrival += 1;

        self.polys.push(Box::new(h));
        self.sevs.push(lm_sev);
        self.lms.push(lm);
        self.lm_degs.push(lm_deg);
        self.redundant.push(false);
        self.arrival.push(arrival);
        idx
    }

    /// Mark every older element (`i < idx`) whose leading monomial is
    /// divisible by `polys[idx]`'s leading monomial as redundant.
    ///
    /// This is the "clearS" half of insert, split out so the bba
    /// driver can run pair generation between the two halves. Safe
    /// to call multiple times — redundancy is monotonic.
    pub fn clear_redundant_for(&mut self, ring: &Ring, idx: usize) {
        debug_assert!(idx < self.polys.len());
        let lm_sev = self.sevs[idx];
        // ADR-010: read leader from the lms cache rather than
        // dereferencing polys[idx].leading() — the lms cache lives
        // contiguous with sevs, no Box pointer chase.
        let h_lm = self.lms[idx].clone();
        for i in 0..idx {
            if self.redundant[i] {
                continue;
            }
            if (lm_sev & !self.sevs[i]) != 0 {
                continue;
            }
            let s_i_lm = &self.lms[i];
            if h_lm.divides(s_i_lm, ring) {
                self.redundant[i] = true;
            }
        }
    }

    /// Force the redundancy flag for `idx` to `flag`. The bba
    /// driver's tail-reduction pass uses this to temporarily hide a
    /// basis element from the reducer (so it doesn't reduce a poly
    /// by itself) and to un-hide afterward.
    ///
    /// Note: the ordinary `insert` flow sets `redundant[i]` from
    /// `true`-only; flipping back to `false` is only safe during
    /// tail reduction, where the caller has manually verified that
    /// the basis element is still *algebraically* non-redundant. Do
    /// not use this to "unmark" an element that genuinely became
    /// redundant via an `insert` of a dividing new element.
    #[inline]
    pub fn set_redundant(&mut self, idx: usize, flag: bool) {
        self.redundant[idx] = flag;
    }

    /// Replace the polynomial at `idx` with `new_poly`. The leading-
    /// term metadata (`sevs[idx]`, `lm_degs[idx]`) is refreshed from
    /// `new_poly`'s leading term. Used by the tail-reduction pass,
    /// which produces reduced-normal-form versions of existing basis
    /// elements while preserving their leading monomials.
    ///
    /// Precondition: `new_poly.leading()` has the same leading
    /// monomial as the old poly (tail reduction never changes the
    /// leader); this is checked in debug builds and silently trusted
    /// in release. `new_poly` must be nonzero.
    pub fn replace_poly(&mut self, ring: &Ring, idx: usize, new_poly: Poly) {
        debug_assert!(!new_poly.is_zero(), "replace_poly with zero poly");
        debug_assert!(idx < self.polys.len());
        // Debug-only check that leading monomial is preserved.
        #[cfg(debug_assertions)]
        {
            let old_lm = self.polys[idx]
                .leading()
                .expect("non-redundant is nonzero")
                .1
                .clone();
            let new_lm = new_poly.leading().unwrap().1;
            assert!(
                old_lm.cmp(new_lm, ring).is_eq(),
                "replace_poly must preserve leading monomial"
            );
        }
        let _ = ring; // suppress unused warning in release
        self.sevs[idx] = new_poly.lm_sev();
        self.lm_degs[idx] = new_poly.lm_deg();
        // ADR-010: refresh lms cache. Per the precondition above
        // the new leading monomial equals the old one, so this is
        // a no-op semantically; we update anyway in case a future
        // caller relaxes the invariant.
        self.lms[idx] = new_poly.leading().expect("non-zero").1.clone();
        *self.polys[idx] = new_poly;
    }

    /// Next arrival ID the next `insert` will stamp. Exposed so
    /// callers (e.g. `gm::enterpairs`) can generate pair arrival IDs
    /// that share the same monotonic counter without actually
    /// inserting.
    ///
    /// The bba driver owns whether pair `arrival` and SBasis
    /// `arrival` share a counter or not; we default to separate
    /// counters so the Pair's arrival is its own sequence. Callers
    /// that want a unified stream can build their own `u64` ticker
    /// around this value.
    #[inline]
    pub fn peek_next_arrival(&self) -> u64 {
        self.next_arrival
    }

    /// Debug-only invariant check.
    ///
    /// - All parallel arrays have length `self.polys.len()`.
    /// - For every `i`, `sevs[i] == polys[i].lm_sev()`.
    /// - For every `i`, `lm_degs[i] == polys[i].lm_deg()`.
    /// - No polynomial is zero.
    /// - `arrival[i]` is strictly ascending.
    pub fn assert_canonical(&self, ring: &Ring) {
        let n = self.polys.len();
        assert_eq!(self.sevs.len(), n);
        assert_eq!(self.lms.len(), n, "lms cache length mismatch (ADR-010)");
        assert_eq!(self.lm_degs.len(), n);
        assert_eq!(self.redundant.len(), n);
        assert_eq!(self.arrival.len(), n);
        for (i, p) in self.polys.iter().enumerate() {
            p.assert_canonical(ring);
            assert!(!p.is_zero(), "SBasis holds zero at index {i}");
            assert_eq!(self.sevs[i], p.lm_sev(), "sevs mismatch at {i}");
            assert_eq!(self.lm_degs[i], p.lm_deg(), "lm_degs mismatch at {i}");
            // ADR-010: lms cache must agree with the poly's
            // own leading monomial.
            let actual_lm = p.leading().expect("non-zero").1;
            assert!(
                self.lms[i].cmp(actual_lm, ring).is_eq(),
                "lms cache mismatch at {i}"
            );
            if i > 0 {
                assert!(
                    self.arrival[i] > self.arrival[i - 1],
                    "arrival not ascending at {i}"
                );
            }
        }
        assert!(self.next_arrival >= self.arrival.last().copied().unwrap_or(0));
    }
}

/// Helper: `m_lm.divides(other_lm)` with sev pre-filter. Lives on
/// `Monomial` logically but we re-implement it here so callers that
/// already hold both `sev` values don't pay the hash-map-friendly
/// `Monomial::divides` walk until the sev check passes.
#[inline]
pub fn divides_with_sev(
    m_sev: u64,
    other_sev: u64,
    m: &Monomial,
    other: &Monomial,
    ring: &Ring,
) -> bool {
    if (m_sev & !other_sev) != 0 {
        return false;
    }
    m.divides(other, ring)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{Coeff, Field};
    use crate::ordering::MonoOrder;

    fn mk_ring(nvars: u32, p: u32) -> Ring {
        Ring::new(nvars, MonoOrder::DegRevLex, Field::new(p).unwrap()).unwrap()
    }

    fn mono(r: &Ring, e: &[u32]) -> Monomial {
        Monomial::from_exponents(r, e).unwrap()
    }

    fn poly1(r: &Ring, c: Coeff, e: &[u32]) -> Poly {
        Poly::monomial(r, c, mono(r, e))
    }

    #[test]
    fn empty_basis_is_empty() {
        let r = mk_ring(3, 13);
        let s = SBasis::new();
        s.assert_canonical(&r);
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn insert_preserves_order() {
        let r = mk_ring(3, 13);
        let mut s = SBasis::new();
        let a = s.insert(&r, poly1(&r, 1, &[2, 0, 0]));
        let b = s.insert(&r, poly1(&r, 1, &[0, 3, 0]));
        let c = s.insert(&r, poly1(&r, 1, &[0, 0, 4]));
        assert_eq!((a, b, c), (0, 1, 2));
        s.assert_canonical(&r);
        assert_eq!(s.len(), 3);
        assert_eq!(s.sevs().len(), 3);
        assert_eq!(s.arrivals(), &[0, 1, 2]);
    }

    #[test]
    fn enters_marks_older_redundant_on_lm_divide() {
        // Insert x^2 first, then x: x | x^2, so the earlier entry
        // becomes redundant.
        let r = mk_ring(3, 13);
        let mut s = SBasis::new();
        s.insert(&r, poly1(&r, 1, &[2, 0, 0]));
        s.insert(&r, poly1(&r, 1, &[1, 0, 0]));
        s.assert_canonical(&r);
        assert!(s.is_redundant(0));
        assert!(!s.is_redundant(1));
    }

    #[test]
    fn coprime_leading_monomials_do_not_mark_redundant() {
        let r = mk_ring(3, 13);
        let mut s = SBasis::new();
        s.insert(&r, poly1(&r, 1, &[2, 0, 0])); // x^2
        s.insert(&r, poly1(&r, 1, &[0, 2, 0])); // y^2
        s.insert(&r, poly1(&r, 1, &[0, 0, 2])); // z^2
        s.assert_canonical(&r);
        for i in 0..s.len() {
            assert!(!s.is_redundant(i));
        }
    }

    #[test]
    fn iter_active_skips_redundant() {
        let r = mk_ring(3, 13);
        let mut s = SBasis::new();
        s.insert(&r, poly1(&r, 1, &[2, 0, 0]));
        s.insert(&r, poly1(&r, 1, &[1, 0, 0]));
        s.insert(&r, poly1(&r, 1, &[0, 1, 0]));
        let live: Vec<usize> = s.iter_active().map(|(i, _)| i).collect();
        assert_eq!(live, vec![1, 2]);
    }
}
