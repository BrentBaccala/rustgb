//! Polynomials as parallel `Vec<Coeff>` + `Vec<Monomial>`.
//!
//! Invariants (checked by [`Poly::assert_canonical`]):
//!
//! 1. `coeffs.len() == terms.len()`.
//! 2. All coefficients are in canonical form `0 < c < p` (zeros excluded).
//! 3. Monomials are strictly descending under the ring's ordering (no
//!    duplicates, no unsorted runs).
//! 4. `lm_*` fields match `terms[0]` / `coeffs[0]` when nonempty.
//!
//! Mathicgb's `Poly.hpp` is the structural template (parallel arrays,
//! leading-term at index 0). This implementation does not copy mathicgb
//! code.

use crate::field::Coeff;
use crate::monomial::Monomial;
use crate::ring::Ring;

/// A sparse polynomial in a [`Ring`].
///
/// See module documentation for invariants. `Send + Sync`: all fields
/// are owned vectors of plain integer data.
#[derive(Clone, Debug)]
pub struct Poly {
    /// Coefficients in the same order as `terms`. All nonzero.
    coeffs: Vec<Coeff>,
    /// Monomials in strictly descending order.
    terms: Vec<Monomial>,
    /// Cached leading-term sev (`terms[0].sev()`); 0 when empty.
    lm_sev: u64,
    /// Cached leading coefficient (`coeffs[0]`); 0 when empty.
    lm_coeff: Coeff,
    /// Cached leading monomial degree (`terms[0].total_deg()`); 0 when
    /// empty.
    lm_deg: u32,
}

impl Poly {
    // ----- Constructors -----

    /// The zero polynomial.
    pub fn zero() -> Self {
        Self {
            coeffs: Vec::new(),
            terms: Vec::new(),
            lm_sev: 0,
            lm_coeff: 0,
            lm_deg: 0,
        }
    }

    /// A polynomial with a single term `c * m`. Returns the zero
    /// polynomial if `c == 0`. `c` must already be reduced mod `p`.
    pub fn monomial(ring: &Ring, c: Coeff, m: Monomial) -> Self {
        debug_assert!(c < ring.field().p(), "coeff {c} not reduced");
        if c == 0 {
            return Self::zero();
        }
        let lm_sev = m.sev();
        let lm_deg = m.total_deg();
        Self {
            coeffs: vec![c],
            terms: vec![m],
            lm_sev,
            lm_coeff: c,
            lm_deg,
        }
    }

    /// Build a polynomial from a sequence of `(coeff, monomial)` pairs
    /// that is already in strictly-descending monomial order with no
    /// duplicates and no zero coefficients.
    ///
    /// This is the zero-overhead fast path used by hot loops that
    /// already know they produced descending, deduped, nonzero terms:
    /// `kbucket::build_neg_cmp` (monomial multiplication by a fixed m
    /// preserves descending order in degrevlex), and `bba::reduce_tail`
    /// (whose `done` vector accumulates in descending order by
    /// construction from the bucket's `extract_leading`).
    ///
    /// Preconditions (checked only in debug):
    /// * `terms` is strictly descending under the ring's ordering.
    /// * No monomial appears twice.
    /// * Every coefficient is in `(0, p)`.
    pub fn from_descending_terms_unchecked(
        ring: &Ring,
        terms: Vec<(Coeff, Monomial)>,
    ) -> Self {
        if terms.is_empty() {
            return Self::zero();
        }
        let p = ring.field().p();
        let mut coeffs: Vec<Coeff> = Vec::with_capacity(terms.len());
        let mut mons: Vec<Monomial> = Vec::with_capacity(terms.len());
        for (c, m) in terms {
            debug_assert!(c != 0, "from_descending_terms_unchecked: zero coeff");
            debug_assert!(c < p, "from_descending_terms_unchecked: unreduced coeff");
            let _ = p;
            if cfg!(debug_assertions)
                && let Some(prev) = mons.last()
            {
                debug_assert!(
                    prev.cmp(&m, ring).is_gt(),
                    "from_descending_terms_unchecked: not strictly descending"
                );
            }
            coeffs.push(c);
            mons.push(m);
        }
        let mut out = Self {
            coeffs,
            terms: mons,
            lm_sev: 0,
            lm_coeff: 0,
            lm_deg: 0,
        };
        out.refresh_cache();
        out
    }

    /// Build a polynomial directly from pre-built parallel vectors of
    /// coefficients and monomials. Same preconditions as
    /// [`from_descending_terms_unchecked`] but skips the tuple
    /// unpacking: the vectors are moved in place. Used by the hot
    /// `kbucket::build_neg_cmp` loop where building parallel vectors
    /// is cheaper than pushing `(Coeff, Monomial)` tuples one at a
    /// time into a single vec (tuple pushes move 48 bytes each; the
    /// parallel vec writes move 4 bytes + 48 bytes to separate cache
    /// lines and vectorise more cleanly).
    pub fn from_descending_parallel_unchecked(
        ring: &Ring,
        coeffs: Vec<Coeff>,
        terms: Vec<Monomial>,
    ) -> Self {
        debug_assert_eq!(coeffs.len(), terms.len());
        if terms.is_empty() {
            return Self::zero();
        }
        #[cfg(debug_assertions)]
        {
            let p = ring.field().p();
            for &c in &coeffs {
                debug_assert!(c != 0 && c < p);
            }
            for w in terms.windows(2) {
                debug_assert!(w[0].cmp(&w[1], ring).is_gt());
            }
        }
        let _ = ring;
        let mut out = Self {
            coeffs,
            terms,
            lm_sev: 0,
            lm_coeff: 0,
            lm_deg: 0,
        };
        out.refresh_cache();
        out
    }

    /// Build a polynomial from an unsorted sequence of `(coeff, monomial)`
    /// pairs. Duplicates are summed, zeros are dropped, the result is
    /// sorted into descending order. Primarily for tests and round-trip
    /// from textual input.
    pub fn from_terms(ring: &Ring, terms: Vec<(Coeff, Monomial)>) -> Self {
        // Sort descending by monomial.
        let mut terms = terms;
        terms.sort_by(|a, b| b.1.cmp(&a.1, ring));

        let mut coeffs: Vec<Coeff> = Vec::with_capacity(terms.len());
        let mut mons: Vec<Monomial> = Vec::with_capacity(terms.len());

        for (c, m) in terms {
            let c = if c >= ring.field().p() {
                ring.field().reduce(c as u64)
            } else {
                c
            };
            if c == 0 {
                continue;
            }
            if let Some(last) = mons.last()
                && last == &m
            {
                // Merge with previous term.
                let idx = coeffs.len() - 1;
                coeffs[idx] = ring.field().add(coeffs[idx], c);
                if coeffs[idx] == 0 {
                    coeffs.pop();
                    mons.pop();
                }
                continue;
            }
            coeffs.push(c);
            mons.push(m);
        }

        let mut p = Self {
            coeffs,
            terms: mons,
            lm_sev: 0,
            lm_coeff: 0,
            lm_deg: 0,
        };
        p.refresh_cache();
        p
    }

    // ----- Cache maintenance -----

    fn refresh_cache(&mut self) {
        if let Some(m) = self.terms.first() {
            self.lm_sev = m.sev();
            self.lm_deg = m.total_deg();
            self.lm_coeff = self.coeffs[0];
        } else {
            self.lm_sev = 0;
            self.lm_coeff = 0;
            self.lm_deg = 0;
        }
    }

    // ----- Accessors -----

    /// Number of nonzero terms.
    #[allow(clippy::len_without_is_empty)] // `is_zero` is the domain-natural spelling; see below.
    #[inline]
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Whether this is the zero polynomial.
    ///
    /// We deliberately expose `is_zero` instead of `is_empty`: a
    /// polynomial with zero terms *is* the zero polynomial, so this
    /// spells the same thing in ring-theoretic terminology.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Iterate over `(coeff, monomial)` pairs in descending order.
    pub fn iter(&self) -> impl Iterator<Item = (Coeff, &Monomial)> + '_ {
        self.coeffs.iter().copied().zip(self.terms.iter())
    }

    /// Leading term `(coeff, &monomial)`, or `None` if zero.
    pub fn leading(&self) -> Option<(Coeff, &Monomial)> {
        if self.is_zero() {
            None
        } else {
            Some((self.coeffs[0], &self.terms[0]))
        }
    }

    /// Leading short exponent vector. 0 when zero (caller should check
    /// `is_zero` before interpreting).
    #[inline]
    pub fn lm_sev(&self) -> u64 {
        self.lm_sev
    }

    /// Leading coefficient. 0 when zero.
    #[inline]
    pub fn lm_coeff(&self) -> Coeff {
        self.lm_coeff
    }

    /// Leading monomial total degree. 0 when zero.
    #[inline]
    pub fn lm_deg(&self) -> u32 {
        self.lm_deg
    }

    /// Borrow the coefficient slice (descending order).
    #[inline]
    pub fn coeffs(&self) -> &[Coeff] {
        &self.coeffs
    }

    /// Borrow the monomial slice (descending order).
    #[inline]
    pub fn terms(&self) -> &[Monomial] {
        &self.terms
    }

    /// Return a new polynomial with the leading term removed. If `self`
    /// is zero or a single term, returns the zero polynomial.
    ///
    /// The tail of a canonical polynomial is itself canonical (terms are
    /// strictly descending and coefficients all nonzero), so this skips
    /// the sort+dedup pass that `from_terms` would do.
    pub fn drop_leading(&self) -> Poly {
        if self.terms.len() <= 1 {
            return Self::zero();
        }
        let mut out = Self {
            coeffs: self.coeffs[1..].to_vec(),
            terms: self.terms[1..].to_vec(),
            lm_sev: 0,
            lm_coeff: 0,
            lm_deg: 0,
        };
        out.refresh_cache();
        out
    }

    /// In-place variant of [`drop_leading`](Self::drop_leading). Shifts
    /// the tail down by one, keeping the same `Vec` allocation. Leaves
    /// `self` as the zero polynomial if it was a single term. O(n) due
    /// to the shift, but avoids the clone + allocation that
    /// `drop_leading` does for the tail.
    ///
    /// Used by the geobucket's cancellation-peel path in
    /// `KBucket::leading` / `extract_leading`, where the poly is taken
    /// out of its slot, the leader removed, and the result put back in
    /// the same slot (no caller holds a reference to the old value).
    pub fn drop_leading_in_place(&mut self) {
        if self.terms.is_empty() {
            return;
        }
        self.coeffs.remove(0);
        self.terms.remove(0);
        self.refresh_cache();
    }

    // ----- Arithmetic -----

    /// In-place: `self = self + other`. Linear merge by monomial order.
    pub fn add_assign(&mut self, other: &Poly, ring: &Ring) {
        if other.is_zero() {
            return;
        }
        if self.is_zero() {
            *self = other.clone();
            return;
        }
        *self = merge(ring, self, other, /* subtract = */ false);
    }

    /// Out-of-place addition. `other` may alias self (Rust borrow rules
    /// prevent true aliasing at the type level, but we accept any two
    /// references including `&a, &a`).
    pub fn add(&self, other: &Poly, ring: &Ring) -> Poly {
        if other.is_zero() {
            return self.clone();
        }
        if self.is_zero() {
            return other.clone();
        }
        merge(ring, self, other, false)
    }

    /// Out-of-place subtraction.
    pub fn sub(&self, other: &Poly, ring: &Ring) -> Poly {
        if other.is_zero() {
            return self.clone();
        }
        if self.is_zero() {
            return other.neg(ring);
        }
        merge(ring, self, other, true)
    }

    /// Negation (flip every coefficient).
    pub fn neg(&self, ring: &Ring) -> Poly {
        let f = ring.field();
        let coeffs: Vec<Coeff> = self.coeffs.iter().map(|&c| f.neg(c)).collect();
        let mut out = Self {
            coeffs,
            terms: self.terms.clone(),
            lm_sev: 0,
            lm_coeff: 0,
            lm_deg: 0,
        };
        out.refresh_cache();
        out
    }

    /// Multiply every coefficient by a scalar. Returns zero if `c == 0`.
    pub fn scale(&self, c: Coeff, ring: &Ring) -> Poly {
        let f = ring.field();
        debug_assert!(c < f.p());
        if c == 0 || self.is_zero() {
            return Self::zero();
        }
        let coeffs: Vec<Coeff> = self.coeffs.iter().map(|&ci| f.mul(ci, c)).collect();
        let mut out = Self {
            coeffs,
            terms: self.terms.clone(),
            lm_sev: 0,
            lm_coeff: 0,
            lm_deg: 0,
        };
        out.refresh_cache();
        out
    }

    /// Multiply every monomial by `m` (no scalar scaling). Requires
    /// that all products fit in the 8-bit exponent range.
    pub fn shift(&self, m: &Monomial, ring: &Ring) -> Option<Poly> {
        if self.is_zero() {
            return Some(Self::zero());
        }
        let mut terms = Vec::with_capacity(self.len());
        for t in &self.terms {
            terms.push(t.mul(m, ring)?);
        }
        // Descending order is preserved: monomial multiplication by a
        // fixed m is monotone in degrevlex.
        let mut out = Self {
            coeffs: self.coeffs.clone(),
            terms,
            lm_sev: 0,
            lm_coeff: 0,
            lm_deg: 0,
        };
        out.refresh_cache();
        Some(out)
    }

    /// Standard multiplication. Straightforward O(|f|·|g|) using a
    /// merge-based accumulator; fine for tests. A heap-based Johnson
    /// multiplication is future work.
    pub fn mul(&self, other: &Poly, ring: &Ring) -> Option<Poly> {
        if self.is_zero() || other.is_zero() {
            return Some(Self::zero());
        }
        let f = ring.field();
        let mut acc: Vec<(Coeff, Monomial)> = Vec::with_capacity(self.len() * other.len());
        for (ca, ma) in self.iter() {
            for (cb, mb) in other.iter() {
                let m = ma.mul(mb, ring)?;
                let c = f.mul(ca, cb);
                if c != 0 {
                    acc.push((c, m));
                }
            }
        }
        Some(Self::from_terms(ring, acc))
    }

    /// The inner reduction step `p - c * m * q`.
    ///
    /// This is the single hottest operation inside the bba driver — it
    /// is the equivalent of Singular's `p_Minus_mm_Mult_qq` (see
    /// `~/Singular/libpolys/polys/pInline2.h`) and of mathicgb's
    /// `Poly::combineInto`. We materialise `m*q` lazily during the
    /// merge so no intermediate polynomial is allocated.
    pub fn sub_mul_term(&self, c: Coeff, m: &Monomial, q: &Poly, ring: &Ring) -> Option<Poly> {
        debug_assert!(c < ring.field().p());
        if c == 0 || q.is_zero() {
            return Some(self.clone());
        }
        let f = ring.field();

        // Walk `self` and `c * m * q` with a two-pointer merge.
        let mut out_c: Vec<Coeff> = Vec::with_capacity(self.len() + q.len());
        let mut out_m: Vec<Monomial> = Vec::with_capacity(self.len() + q.len());

        let mut i = 0usize; // index into self
        let mut j = 0usize; // index into q

        while i < self.len() && j < q.len() {
            // Next term from `c*m*q` is (c * q.coeffs[j], m * q.terms[j]).
            let mq_term_mon = m.mul(&q.terms[j], ring)?;
            match self.terms[i].cmp(&mq_term_mon, ring) {
                std::cmp::Ordering::Greater => {
                    out_c.push(self.coeffs[i]);
                    out_m.push(self.terms[i].clone());
                    i += 1;
                }
                std::cmp::Ordering::Less => {
                    // subtract: output is 0 - (c * q.coeffs[j]).
                    let neg = f.neg(f.mul(c, q.coeffs[j]));
                    if neg != 0 {
                        out_c.push(neg);
                        out_m.push(mq_term_mon);
                    }
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    let cmq = f.mul(c, q.coeffs[j]);
                    let diff = f.sub(self.coeffs[i], cmq);
                    if diff != 0 {
                        out_c.push(diff);
                        out_m.push(self.terms[i].clone());
                    }
                    i += 1;
                    j += 1;
                }
            }
        }
        while i < self.len() {
            out_c.push(self.coeffs[i]);
            out_m.push(self.terms[i].clone());
            i += 1;
        }
        while j < q.len() {
            let neg = f.neg(f.mul(c, q.coeffs[j]));
            if neg != 0 {
                out_c.push(neg);
                out_m.push(m.mul(&q.terms[j], ring)?);
            }
            j += 1;
        }

        let mut out = Self {
            coeffs: out_c,
            terms: out_m,
            lm_sev: 0,
            lm_coeff: 0,
            lm_deg: 0,
        };
        out.refresh_cache();
        Some(out)
    }

    /// Return a scalar multiple that makes the leading coefficient 1.
    /// Zero is returned unchanged. Requires a nonzero leading coefficient
    /// that's invertible (always true over a prime field for nonzero lc).
    pub fn monic(&self, ring: &Ring) -> Option<Poly> {
        if self.is_zero() {
            return Some(Self::zero());
        }
        let lc = self.lm_coeff;
        if lc == 1 {
            return Some(self.clone());
        }
        let inv = ring.field().inv(lc)?;
        Some(self.scale(inv, ring))
    }

    // ----- Invariants -----

    /// Panic if any internal invariant is violated.
    pub fn assert_canonical(&self, ring: &Ring) {
        assert_eq!(self.coeffs.len(), self.terms.len(), "length mismatch");
        let p = ring.field().p();
        for (k, (&c, m)) in self.coeffs.iter().zip(self.terms.iter()).enumerate() {
            assert!(c > 0 && c < p, "coeff[{k}] = {c} not in 1..{p}");
            m.assert_canonical(ring);
        }
        for w in self.terms.windows(2) {
            let ord = w[0].cmp(&w[1], ring);
            assert!(
                ord == std::cmp::Ordering::Greater,
                "terms not strictly descending: got {ord:?}"
            );
        }
        // Cached leading fields agree with the stored leading term.
        if self.is_zero() {
            assert_eq!(self.lm_sev, 0);
            assert_eq!(self.lm_coeff, 0);
            assert_eq!(self.lm_deg, 0);
        } else {
            assert_eq!(self.lm_sev, self.terms[0].sev());
            assert_eq!(self.lm_coeff, self.coeffs[0]);
            assert_eq!(self.lm_deg, self.terms[0].total_deg());
        }
    }
}

impl Default for Poly {
    fn default() -> Self {
        Self::zero()
    }
}

impl PartialEq for Poly {
    fn eq(&self, other: &Self) -> bool {
        self.coeffs == other.coeffs && self.terms == other.terms
    }
}
impl Eq for Poly {}

/// Merge two sorted polynomials into one. If `subtract` is true, the
/// second operand's coefficients are negated.
fn merge(ring: &Ring, a: &Poly, b: &Poly, subtract: bool) -> Poly {
    let f = ring.field();
    let mut out_c = Vec::with_capacity(a.len() + b.len());
    let mut out_m = Vec::with_capacity(a.len() + b.len());
    let mut i = 0usize;
    let mut j = 0usize;
    while i < a.len() && j < b.len() {
        match a.terms[i].cmp(&b.terms[j], ring) {
            std::cmp::Ordering::Greater => {
                out_c.push(a.coeffs[i]);
                out_m.push(a.terms[i].clone());
                i += 1;
            }
            std::cmp::Ordering::Less => {
                let c = if subtract {
                    f.neg(b.coeffs[j])
                } else {
                    b.coeffs[j]
                };
                if c != 0 {
                    out_c.push(c);
                    out_m.push(b.terms[j].clone());
                }
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                let bc = if subtract {
                    f.neg(b.coeffs[j])
                } else {
                    b.coeffs[j]
                };
                let s = f.add(a.coeffs[i], bc);
                if s != 0 {
                    out_c.push(s);
                    out_m.push(a.terms[i].clone());
                }
                i += 1;
                j += 1;
            }
        }
    }
    while i < a.len() {
        out_c.push(a.coeffs[i]);
        out_m.push(a.terms[i].clone());
        i += 1;
    }
    while j < b.len() {
        let c = if subtract {
            f.neg(b.coeffs[j])
        } else {
            b.coeffs[j]
        };
        if c != 0 {
            out_c.push(c);
            out_m.push(b.terms[j].clone());
        }
        j += 1;
    }
    let mut out = Poly {
        coeffs: out_c,
        terms: out_m,
        lm_sev: 0,
        lm_coeff: 0,
        lm_deg: 0,
    };
    out.refresh_cache();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Field;
    use crate::ordering::MonoOrder;

    fn mk_ring(nvars: u32, p: u32) -> Ring {
        Ring::new(nvars, MonoOrder::DegRevLex, Field::new(p).unwrap()).unwrap()
    }

    fn mono(r: &Ring, e: &[u32]) -> Monomial {
        Monomial::from_exponents(r, e).unwrap()
    }

    #[test]
    fn zero_is_zero() {
        let p = Poly::zero();
        assert!(p.is_zero());
        assert_eq!(p.len(), 0);
        assert!(p.leading().is_none());
    }

    #[test]
    fn from_terms_sorts_and_dedups() {
        let r = mk_ring(3, 13);
        // y^2 has higher total degree than x, so it becomes leading
        // under degrevlex.
        let terms = vec![
            (3, mono(&r, &[1, 0, 0])),
            (5, mono(&r, &[0, 2, 0])),
            (7, mono(&r, &[1, 0, 0])), // duplicate: merges -> 3+7 = 10
            (0, mono(&r, &[0, 0, 1])), // zero coeff: dropped
        ];
        let p = Poly::from_terms(&r, terms);
        p.assert_canonical(&r);
        assert_eq!(p.len(), 2);
        let (c0, m0) = p.leading().unwrap();
        assert_eq!(c0, 5);
        assert_eq!(*m0, mono(&r, &[0, 2, 0]));
        // The non-leading term has the combined coefficient 10.
        assert_eq!(p.coeffs()[1], 10);
        assert_eq!(p.terms()[1], mono(&r, &[1, 0, 0]));
    }

    #[test]
    fn add_and_sub_cancel() {
        let r = mk_ring(3, 13);
        let f = Poly::from_terms(
            &r,
            vec![
                (3, mono(&r, &[1, 0, 0])),
                (5, mono(&r, &[0, 2, 0])),
                (1, mono(&r, &[0, 0, 1])),
            ],
        );
        let g = f.sub(&f, &r);
        g.assert_canonical(&r);
        assert!(g.is_zero());
    }

    #[test]
    fn sub_mul_term_matches_slow_path() {
        let r = mk_ring(3, 13);
        let p = Poly::from_terms(
            &r,
            vec![
                (3, mono(&r, &[2, 1, 0])),
                (7, mono(&r, &[1, 0, 1])),
                (1, mono(&r, &[0, 0, 2])),
            ],
        );
        let q = Poly::from_terms(
            &r,
            vec![(4, mono(&r, &[1, 1, 0])), (5, mono(&r, &[0, 0, 1]))],
        );
        let m = mono(&r, &[1, 0, 0]);
        let c: Coeff = 2;

        // Slow path.
        let mq = q.shift(&m, &r).unwrap().scale(c, &r);
        let slow = p.sub(&mq, &r);
        let fast = p.sub_mul_term(c, &m, &q, &r).unwrap();
        slow.assert_canonical(&r);
        fast.assert_canonical(&r);
        assert_eq!(slow, fast);
    }

    #[test]
    fn monic_is_idempotent() {
        let r = mk_ring(2, 32003);
        let p = Poly::from_terms(
            &r,
            vec![
                (17, mono(&r, &[3, 0])),
                (2, mono(&r, &[1, 1])),
                (9, mono(&r, &[0, 2])),
            ],
        );
        let once = p.monic(&r).unwrap();
        let twice = once.monic(&r).unwrap();
        assert_eq!(once, twice);
        assert_eq!(once.lm_coeff(), 1);
    }

    #[test]
    fn leading_invariants() {
        let r = mk_ring(2, 7);
        let p = Poly::from_terms(&r, vec![(3, mono(&r, &[2, 0])), (4, mono(&r, &[1, 1]))]);
        let (c, m) = p.leading().unwrap();
        assert_eq!(c, 3);
        assert_eq!(m.total_deg(), 2);
        assert_eq!(p.lm_sev(), m.sev());
        assert_eq!(p.lm_coeff(), 3);
        assert_eq!(p.lm_deg(), 2);
    }

    #[test]
    fn drop_leading_basic() {
        let r = mk_ring(3, 13);
        let p = Poly::from_terms(
            &r,
            vec![
                (3, mono(&r, &[2, 1, 0])),
                (7, mono(&r, &[1, 0, 1])),
                (1, mono(&r, &[0, 0, 2])),
            ],
        );
        let tail = p.drop_leading();
        tail.assert_canonical(&r);
        assert_eq!(tail.len(), 2);
        // New leading is the old second term.
        let (c, m) = tail.leading().unwrap();
        assert_eq!(c, 7);
        assert_eq!(m, &mono(&r, &[1, 0, 1]));
        // Cache fields agree.
        assert_eq!(tail.lm_coeff(), 7);
        assert_eq!(tail.lm_sev(), m.sev());
        assert_eq!(tail.lm_deg(), m.total_deg());
    }

    #[test]
    fn drop_leading_edge_cases() {
        let r = mk_ring(2, 7);
        // Zero in, zero out.
        let z = Poly::zero();
        let z_tail = z.drop_leading();
        assert!(z_tail.is_zero());
        z_tail.assert_canonical(&r);
        // Single term in, zero out.
        let single = Poly::monomial(&r, 3, mono(&r, &[1, 0]));
        let single_tail = single.drop_leading();
        assert!(single_tail.is_zero());
        single_tail.assert_canonical(&r);
    }

    #[test]
    fn drop_leading_matches_from_terms_tail() {
        // The optimised drop_leading should match what from_terms
        // would produce for the same tail, just without re-sorting.
        let r = mk_ring(4, 32003);
        let p = Poly::from_terms(
            &r,
            vec![
                (5, mono(&r, &[3, 0, 0, 0])),
                (2, mono(&r, &[2, 1, 0, 0])),
                (9, mono(&r, &[1, 0, 1, 0])),
                (4, mono(&r, &[0, 0, 0, 2])),
            ],
        );
        let fast = p.drop_leading();
        let slow_tail: Vec<(Coeff, Monomial)> = p.coeffs()[1..]
            .iter()
            .zip(p.terms()[1..].iter())
            .map(|(&c, m)| (c, m.clone()))
            .collect();
        let slow = Poly::from_terms(&r, slow_tail);
        assert_eq!(fast, slow);
    }
}
