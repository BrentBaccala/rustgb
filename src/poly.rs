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
///
/// The live terms are `coeffs[head..]` / `terms[head..]`. `head`
/// advances when [`drop_leading_in_place`](Self::drop_leading_in_place)
/// peels a leader — that's an O(1) cursor bump rather than the O(n)
/// `Vec::remove(0)` that the previous representation required. The
/// dead prefix is reclaimed when the `Poly` is cloned (see the
/// custom `Clone` impl) or when it falls out of scope. All accessors
/// (`coeffs()`, `terms()`, `iter()`, `leading()`, `len()`,
/// `is_zero()`) operate on the live region.
#[derive(Debug)]
pub struct Poly {
    /// Coefficients in the same order as `terms`. All nonzero.
    coeffs: Vec<Coeff>,
    /// Monomials in strictly descending order.
    terms: Vec<Monomial>,
    /// Index of the live leading term. `coeffs[head..]` and
    /// `terms[head..]` are the algebraically meaningful slice;
    /// indices below `head` are abandoned but not yet freed.
    /// Always `<= terms.len()`; equality means the polynomial is zero.
    head: usize,
    /// Cached leading-term sev (`terms[head].sev()`); 0 when empty.
    lm_sev: u64,
    /// Cached leading coefficient (`coeffs[head]`); 0 when empty.
    lm_coeff: Coeff,
    /// Cached leading monomial degree (`terms[head].total_deg()`);
    /// 0 when empty.
    lm_deg: u32,
}

impl Clone for Poly {
    /// Clone only the live tail (`coeffs[head..]` / `terms[head..]`)
    /// so the dead prefix is dropped at every clone site. This keeps
    /// the bucket-slot reuse pattern (drop_leading repeatedly, then
    /// absorb a new poly via `merge` which constructs fresh) from
    /// accumulating dead memory across the chain of intermediate
    /// clones a caller might make.
    fn clone(&self) -> Self {
        let coeffs = self.coeffs[self.head..].to_vec();
        let terms = self.terms[self.head..].to_vec();
        Self {
            coeffs,
            terms,
            head: 0,
            lm_sev: self.lm_sev,
            lm_coeff: self.lm_coeff,
            lm_deg: self.lm_deg,
        }
    }
}

impl Poly {
    // ----- Constructors -----

    /// The zero polynomial.
    pub fn zero() -> Self {
        Self {
            coeffs: Vec::new(),
            terms: Vec::new(),
            head: 0,
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
            head: 0,
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
            head: 0,
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
            head: 0,
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
            head: 0,
            lm_sev: 0,
            lm_coeff: 0,
            lm_deg: 0,
        };
        p.refresh_cache();
        p
    }

    // ----- Cache maintenance -----

    fn refresh_cache(&mut self) {
        if let Some(m) = self.terms.get(self.head) {
            self.lm_sev = m.sev();
            self.lm_deg = m.total_deg();
            self.lm_coeff = self.coeffs[self.head];
        } else {
            self.lm_sev = 0;
            self.lm_coeff = 0;
            self.lm_deg = 0;
        }
    }

    // ----- Accessors -----

    /// Number of live (post-`head`) terms.
    #[allow(clippy::len_without_is_empty)] // `is_zero` is the domain-natural spelling; see below.
    #[inline]
    pub fn len(&self) -> usize {
        self.terms.len() - self.head
    }

    /// Whether this is the zero polynomial.
    ///
    /// We deliberately expose `is_zero` instead of `is_empty`: a
    /// polynomial with zero terms *is* the zero polynomial, so this
    /// spells the same thing in ring-theoretic terminology.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.head == self.terms.len()
    }

    /// Iterate over `(coeff, monomial)` pairs in descending order.
    pub fn iter(&self) -> impl Iterator<Item = (Coeff, &Monomial)> + '_ {
        self.coeffs[self.head..]
            .iter()
            .copied()
            .zip(self.terms[self.head..].iter())
    }

    /// Leading term `(coeff, &monomial)`, or `None` if zero.
    pub fn leading(&self) -> Option<(Coeff, &Monomial)> {
        if self.is_zero() {
            None
        } else {
            Some((self.coeffs[self.head], &self.terms[self.head]))
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
        &self.coeffs[self.head..]
    }

    /// Borrow the monomial slice (descending order).
    #[inline]
    pub fn terms(&self) -> &[Monomial] {
        &self.terms[self.head..]
    }

    /// A cursor positioned at the leading term (or at end if zero).
    ///
    /// Both the `Vec`-backed and the future `List`-backed `Poly`
    /// expose the same cursor API so callers can walk a polynomial
    /// in descending order without assuming random access. The
    /// cursor is cheap (`Copy`) and carries only a reference to the
    /// underlying storage plus its position; the reducer stores one
    /// per in-flight reducer.
    #[inline]
    pub fn cursor(&self) -> PolyCursor<'_> {
        PolyCursor {
            poly: self,
            idx: self.head,
        }
    }

    /// Return a new polynomial with the leading term removed. If `self`
    /// is zero or a single term, returns the zero polynomial.
    ///
    /// The tail of a canonical polynomial is itself canonical (terms are
    /// strictly descending and coefficients all nonzero), so this skips
    /// the sort+dedup pass that `from_terms` would do.
    pub fn drop_leading(&self) -> Poly {
        if self.len() <= 1 {
            return Self::zero();
        }
        let mut out = Self {
            coeffs: self.coeffs[self.head + 1..].to_vec(),
            terms: self.terms[self.head + 1..].to_vec(),
            head: 0,
            lm_sev: 0,
            lm_coeff: 0,
            lm_deg: 0,
        };
        out.refresh_cache();
        out
    }

    /// In-place variant of [`drop_leading`](Self::drop_leading).
    ///
    /// Bumps the internal `head` cursor by one — O(1). Does not free
    /// the old leader's slot in the underlying `Vec`s; the dead prefix
    /// is reclaimed when this `Poly` is cloned or dropped.
    ///
    /// Used by the geobucket's cancellation peel in
    /// [`KBucket::leading`](crate::kbucket::KBucket::leading) and by
    /// [`KBucket::extract_leading`](crate::kbucket::KBucket::extract_leading).
    /// In a long bucket-slot lifetime (many drops without an
    /// intervening absorb) the slot can hold a Poly whose live region
    /// is much smaller than its allocation; the next `merge`/`absorb`
    /// constructs a fresh Poly with `head == 0`, returning that memory
    /// to bounded use.
    pub fn drop_leading_in_place(&mut self) {
        if self.is_zero() {
            return;
        }
        self.head += 1;
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
        let coeffs: Vec<Coeff> = self.coeffs[self.head..].iter().map(|&c| f.neg(c)).collect();
        let mut out = Self {
            coeffs,
            terms: self.terms[self.head..].to_vec(),
            head: 0,
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
        let coeffs: Vec<Coeff> = self.coeffs[self.head..]
            .iter()
            .map(|&ci| f.mul(ci, c))
            .collect();
        let mut out = Self {
            coeffs,
            terms: self.terms[self.head..].to_vec(),
            head: 0,
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
        for t in &self.terms[self.head..] {
            terms.push(t.mul(m, ring)?);
        }
        // Descending order is preserved: monomial multiplication by a
        // fixed m is monotone in degrevlex.
        let mut out = Self {
            coeffs: self.coeffs[self.head..].to_vec(),
            terms,
            head: 0,
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

        let s_c = self.coeffs();
        let s_m = self.terms();
        let q_c = q.coeffs();
        let q_m = q.terms();

        // Walk `self` and `c * m * q` with a two-pointer merge.
        let mut out_c: Vec<Coeff> = Vec::with_capacity(s_c.len() + q_c.len());
        let mut out_m: Vec<Monomial> = Vec::with_capacity(s_m.len() + q_m.len());

        let mut i = 0usize; // index into self.live
        let mut j = 0usize; // index into q.live

        while i < s_m.len() && j < q_m.len() {
            // Next term from `c*m*q` is (c * q_c[j], m * q_m[j]).
            let mq_term_mon = m.mul(&q_m[j], ring)?;
            match s_m[i].cmp(&mq_term_mon, ring) {
                std::cmp::Ordering::Greater => {
                    out_c.push(s_c[i]);
                    out_m.push(s_m[i].clone());
                    i += 1;
                }
                std::cmp::Ordering::Less => {
                    // subtract: output is 0 - (c * q_c[j]).
                    let neg = f.neg(f.mul(c, q_c[j]));
                    if neg != 0 {
                        out_c.push(neg);
                        out_m.push(mq_term_mon);
                    }
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    let cmq = f.mul(c, q_c[j]);
                    let diff = f.sub(s_c[i], cmq);
                    if diff != 0 {
                        out_c.push(diff);
                        out_m.push(s_m[i].clone());
                    }
                    i += 1;
                    j += 1;
                }
            }
        }
        while i < s_m.len() {
            out_c.push(s_c[i]);
            out_m.push(s_m[i].clone());
            i += 1;
        }
        while j < q_m.len() {
            let neg = f.neg(f.mul(c, q_c[j]));
            if neg != 0 {
                out_c.push(neg);
                out_m.push(m.mul(&q_m[j], ring)?);
            }
            j += 1;
        }

        let mut out = Self {
            coeffs: out_c,
            terms: out_m,
            head: 0,
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
        assert!(
            self.head <= self.terms.len(),
            "head {} out of range (terms.len = {})",
            self.head,
            self.terms.len()
        );
        let p = ring.field().p();
        let live_c = self.coeffs();
        let live_m = self.terms();
        for (k, (&c, m)) in live_c.iter().zip(live_m.iter()).enumerate() {
            assert!(c > 0 && c < p, "live coeff[{k}] = {c} not in 1..{p}");
            m.assert_canonical(ring);
        }
        for w in live_m.windows(2) {
            let ord = w[0].cmp(&w[1], ring);
            assert!(
                ord == std::cmp::Ordering::Greater,
                "live terms not strictly descending: got {ord:?}"
            );
        }
        // Cached leading fields agree with the live leading term.
        if self.is_zero() {
            assert_eq!(self.lm_sev, 0);
            assert_eq!(self.lm_coeff, 0);
            assert_eq!(self.lm_deg, 0);
        } else {
            assert_eq!(self.lm_sev, self.terms[self.head].sev());
            assert_eq!(self.lm_coeff, self.coeffs[self.head]);
            assert_eq!(self.lm_deg, self.terms[self.head].total_deg());
        }
    }
}

impl Default for Poly {
    fn default() -> Self {
        Self::zero()
    }
}

/// A cursor walking a [`Poly`]'s terms in descending order.
///
/// Obtain one with [`Poly::cursor`]. The cursor is `Copy` and cheap;
/// it holds a reference to the polynomial and a position. Both
/// `Vec`-backed and `List`-backed `Poly` back-ends expose this same
/// cursor shape, so consumers (notably [`crate::reducer::Reducer`])
/// work uniformly regardless of storage choice.
///
/// On the `Vec` backend the position is an index into the parallel
/// arrays; on the `List` backend it is an `Option<&Node>`. Callers
/// never observe the difference: [`term`](Self::term) always returns
/// `Some((coeff, &monomial))` when live and `None` once exhausted,
/// [`advance`](Self::advance) steps one term forward, and
/// [`is_done`](Self::is_done) reports exhaustion.
#[derive(Clone, Copy, Debug)]
pub struct PolyCursor<'a> {
    poly: &'a Poly,
    /// Index into the parallel-array storage. Always in `[head, terms.len()]`;
    /// equality with `terms.len()` means exhausted.
    idx: usize,
}

impl<'a> PolyCursor<'a> {
    /// Current term `(coeff, &monomial)`, or `None` if exhausted.
    #[inline]
    pub fn term(&self) -> Option<(Coeff, &'a Monomial)> {
        if self.idx < self.poly.terms.len() {
            Some((self.poly.coeffs[self.idx], &self.poly.terms[self.idx]))
        } else {
            None
        }
    }

    /// Advance one term. No-op once exhausted (`is_done` stays true).
    #[inline]
    pub fn advance(&mut self) {
        if self.idx < self.poly.terms.len() {
            self.idx += 1;
        }
    }

    /// True once all terms have been walked.
    #[inline]
    pub fn is_done(&self) -> bool {
        self.idx >= self.poly.terms.len()
    }
}

impl PartialEq for Poly {
    fn eq(&self, other: &Self) -> bool {
        // Compare the live regions only — `head` may differ between
        // two algebraically equal polys depending on their drop history.
        self.coeffs() == other.coeffs() && self.terms() == other.terms()
    }
}
impl Eq for Poly {}

/// Merge two sorted polynomials into one. If `subtract` is true, the
/// second operand's coefficients are negated.
///
/// Implementation per ADR-006: pre-allocates the output to the
/// upper-bound capacity `a.len() + b.len()`, writes through
/// `Vec::spare_capacity_mut()` with `MaybeUninit::write`, and finalises
/// the length once via `set_len` instead of pushing one term at a
/// time. Cancellation is branch-free: the write happens
/// unconditionally and the cursor is incremented only when the
/// resulting coefficient is nonzero. This mirrors FLINT's
/// `_nmod_mpoly_add` (`~/flint/src/nmod_mpoly/add.c:16-124`).
fn merge(ring: &Ring, a: &Poly, b: &Poly, subtract: bool) -> Poly {
    let f = ring.field();
    let a_c = a.coeffs();
    let a_m = a.terms();
    let b_c = b.coeffs();
    let b_m = b.terms();
    let cap = a_c.len() + b_c.len();
    let mut out_c: Vec<Coeff> = Vec::with_capacity(cap);
    let mut out_m: Vec<Monomial> = Vec::with_capacity(cap);

    // Number of initialised slots written so far. Tracked outside the
    // inner block so the final `set_len` can read it after the
    // spare-capacity borrows have been dropped.
    let mut k: usize = 0;
    {
        let spare_c = out_c.spare_capacity_mut();
        let spare_m = out_m.spare_capacity_mut();
        let mut i = 0usize;
        let mut j = 0usize;

        while i < a_m.len() && j < b_m.len() {
            match a_m[i].cmp(&b_m[j], ring) {
                std::cmp::Ordering::Greater => {
                    spare_c[k].write(a_c[i]);
                    spare_m[k].write(a_m[i].clone());
                    k += 1;
                    i += 1;
                }
                std::cmp::Ordering::Less => {
                    let c = if subtract { f.neg(b_c[j]) } else { b_c[j] };
                    spare_c[k].write(c);
                    spare_m[k].write(b_m[j].clone());
                    // Branch-free cancellation skip: write
                    // unconditionally; advance k only on nonzero.
                    // The next iteration overwrites the same slot if
                    // k didn't advance.
                    k += (c != 0) as usize;
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    let bc = if subtract { f.neg(b_c[j]) } else { b_c[j] };
                    let s = f.add(a_c[i], bc);
                    spare_c[k].write(s);
                    spare_m[k].write(a_m[i].clone());
                    k += (s != 0) as usize;
                    i += 1;
                    j += 1;
                }
            }
        }
        while i < a_m.len() {
            spare_c[k].write(a_c[i]);
            spare_m[k].write(a_m[i].clone());
            k += 1;
            i += 1;
        }
        while j < b_m.len() {
            let c = if subtract { f.neg(b_c[j]) } else { b_c[j] };
            spare_c[k].write(c);
            spare_m[k].write(b_m[j].clone());
            k += (c != 0) as usize;
            j += 1;
        }
    }
    // SAFETY: We have written exactly `k` initialised slots starting
    // at index 0 in both `out_c` and `out_m`. Slots in [k, capacity)
    // may have been written to by the branch-free cancellation pattern
    // (their bytes are non-canonical garbage), but `set_len(k)`
    // truncates them out of the live region so they are never read
    // and never dropped via the Vec's destructor. Both `Coeff` (u32)
    // and `Monomial` (POD struct of u64s and u32s) have no Drop side
    // effects, so no resource leak occurs.
    unsafe {
        out_c.set_len(k);
        out_m.set_len(k);
    }
    let mut out = Poly {
        coeffs: out_c,
        terms: out_m,
        head: 0,
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
    fn drop_leading_in_place_walks_head_cursor() {
        // Locks in the head-cursor mechanic: repeated in-place drops
        // must produce the same logical poly as the same number of
        // out-of-place drops, and arithmetic on the partly-dropped
        // poly must agree with arithmetic on the equivalent fresh poly.
        let r = mk_ring(3, 32003);
        let original = Poly::from_terms(
            &r,
            vec![
                (5, mono(&r, &[3, 0, 0])),
                (4, mono(&r, &[2, 1, 0])),
                (3, mono(&r, &[1, 0, 1])),
                (2, mono(&r, &[0, 0, 2])),
                (1, mono(&r, &[0, 1, 0])),
            ],
        );

        let mut peeled = original.clone();
        peeled.drop_leading_in_place();
        peeled.drop_leading_in_place();
        peeled.assert_canonical(&r);
        assert_eq!(peeled.len(), 3);

        let fresh = original.drop_leading().drop_leading();
        fresh.assert_canonical(&r);

        // Logical equality across the head boundary.
        assert_eq!(peeled, fresh);
        assert_eq!(peeled.coeffs(), fresh.coeffs());
        assert_eq!(peeled.terms(), fresh.terms());
        assert_eq!(peeled.lm_coeff(), fresh.lm_coeff());
        assert_eq!(peeled.lm_sev(), fresh.lm_sev());
        assert_eq!(peeled.lm_deg(), fresh.lm_deg());

        // Arithmetic across the head boundary: subtracting peeled
        // from itself yields zero, and adding it to fresh doubles it.
        let zero = peeled.sub(&peeled, &r);
        zero.assert_canonical(&r);
        assert!(zero.is_zero());

        let doubled_via_peeled = peeled.add(&fresh, &r);
        let doubled_via_fresh = fresh.add(&fresh, &r);
        assert_eq!(doubled_via_peeled, doubled_via_fresh);

        // Cloning a peeled poly must drop the dead prefix.
        let cloned = peeled.clone();
        cloned.assert_canonical(&r);
        assert_eq!(cloned, peeled);
        assert_eq!(cloned.coeffs().len(), cloned.len());

        // Drop the rest one at a time; final state is zero with cache
        // cleared.
        let mut p = peeled;
        for _ in 0..3 {
            p.drop_leading_in_place();
            p.assert_canonical(&r);
        }
        assert!(p.is_zero());
        assert_eq!(p.lm_coeff(), 0);
        assert_eq!(p.lm_sev(), 0);
        assert_eq!(p.lm_deg(), 0);
        // Extra drop on a zero poly is a no-op.
        p.drop_leading_in_place();
        assert!(p.is_zero());
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
