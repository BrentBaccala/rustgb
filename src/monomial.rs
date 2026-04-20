//! Packed-exponent monomials.
//!
//! A [`Monomial`] represents `x_0^{e_0} * x_1^{e_1} * ... * x_{n-1}^{e_{n-1}}`
//! for a ring with `nvars = n`. Exponents are packed 8 bits each into
//! four `u64` words — 32 bytes total. For degrevlex comparison we use
//! a layout tuned so that lex-comparing the four `u64` words (most
//! significant word first) yields degrevlex order; total degree is
//! cached in a separate `u32` field so the effective degree can exceed
//! the 255 that a single 8-bit exponent slot allows.
//!
//! ## Packing (degrevlex, 8 bits/var)
//!
//! ```text
//! word 3 (MSB) : [ byte31 | byte30 | byte29 | ... | byte24 ]
//! word 2       : [ byte23 | byte22 | ....................  ]
//! word 1       : [ byte15 | byte14 | ....................  ]
//! word 0 (LSB) : [  byte7 |  byte6 | ....................  ]
//! ```
//!
//! Byte 31 holds `min(total_deg, 255)` (the "capped" total degree).
//! Bytes 30..(31 - nvars) hold the *complemented* exponent of variable
//! `(31 - byte_index)`'s — i.e. variable `nvars - 1` is stored at byte
//! `30`, variable `nvars - 2` at byte `29`, etc., down to variable `0`
//! at byte `31 - nvars`. Bytes below `31 - nvars` are always zero and
//! do not affect comparison results (they contribute identical bits to
//! every monomial in the same ring).
//!
//! Why complemented? Lex-compare of u64 words (MSB first) puts the
//! largest-index variable in the highest byte. Degrevlex's tie-break
//! rule says: on equal total degree, find the largest index `k` at
//! which exponents differ; the monomial with *smaller* exponent at
//! index `k` is greater. Storing `255 - e_k` flips the direction:
//! larger complement wins lex compare, which is smaller `e_k` wins
//! degrevlex. That is exactly the rule.
//!
//! Caveat: total degrees that exceed 255 are stored capped at 255.
//! [`Monomial::cmp`] falls back on the cached `total_deg: u32` when
//! either operand's cap is saturated.
//!
//! ## Caches
//!
//! * `sev` (short exponent vector): a 64-bit bloom filter with bit
//!   `i mod 64` set iff `e_i > 0`. Used by the bba sweep as a
//!   divisibility pre-filter.
//! * `total_deg`: sum of exponents, not capped.
//! * `component`: always 0 today. Reserved for future module support.
//!
//! Reference: mathicgb `MonoMonoid.hpp`, Singular's `ExpL`, and the
//! FLINT `mpoly_monomial_*` helpers. Algorithms re-derived, not copied.

use crate::ordering::MonoOrder;
use crate::ring::{BITS_PER_VAR, Ring};
use std::cmp::Ordering;

/// Number of u64 words in the packed exponent block.
pub const WORDS_PER_MONO: usize = 4;

const _BITS_PER_VAR_IS_8: () = assert!(BITS_PER_VAR == 8);

/// Packed-exponent monomial. See module documentation for layout.
///
/// `Send + Sync`: only owns integer arrays.
#[derive(Clone, Debug)]
pub struct Monomial {
    /// Four u64 words; word 3 is most significant.
    packed: [u64; WORDS_PER_MONO],
    /// Short exponent vector.
    sev: u64,
    /// True total degree, uncapped.
    total_deg: u32,
    /// Component index. Always 0 today.
    component: u32,
}

impl Monomial {
    // ----- Construction -----

    /// Build a monomial from an exponent slice of length `ring.nvars()`.
    ///
    /// Returns `None` if the length is wrong or any exponent exceeds
    /// 255 (the 8-bit per-variable limit).
    pub fn from_exponents(ring: &Ring, exps: &[u32]) -> Option<Self> {
        let n = ring.nvars() as usize;
        if exps.len() != n {
            return None;
        }
        for &e in exps {
            if e > u8::MAX as u32 {
                return None;
            }
        }

        let mut packed = [0u64; WORDS_PER_MONO];
        let mut total: u64 = 0;
        let mut sev: u64 = 0;

        for (i, &e) in exps.iter().enumerate() {
            total += e as u64;
            if e > 0 {
                sev |= 1u64 << (i % 64);
            }
            // Always write the complemented byte, even when the
            // exponent is zero (then complement == 255): leaving a
            // variable byte at 0 would misread as exponent 255.
            let byte_idx = byte_index_for_var(n, i);
            let (word, shift) = split_byte_index(byte_idx);
            let complement = (u8::MAX as u32 - e) as u64;
            packed[word] |= complement << shift;
        }

        // Top byte: capped total degree.
        let capped = total.min(u8::MAX as u64);
        packed[WORDS_PER_MONO - 1] |= capped << 56;

        if total > u32::MAX as u64 {
            return None;
        }

        Some(Self {
            packed,
            sev,
            total_deg: total as u32,
            component: 0,
        })
    }

    /// The identity monomial (all exponents zero).
    pub fn one(ring: &Ring) -> Self {
        let zeros = vec![0u32; ring.nvars() as usize];
        Self::from_exponents(ring, &zeros).expect("identity monomial fits trivially")
    }

    // ----- Accessors -----

    /// Short exponent vector.
    #[inline]
    pub fn sev(&self) -> u64 {
        self.sev
    }

    /// Total degree (uncapped).
    #[inline]
    pub fn total_deg(&self) -> u32 {
        self.total_deg
    }

    /// Component index. Always 0 in this bootstrap.
    #[inline]
    pub fn component(&self) -> u32 {
        self.component
    }

    /// Exponent of variable `i`. Returns `None` if `i >= ring.nvars()`.
    pub fn exponent(&self, ring: &Ring, i: u32) -> Option<u32> {
        if i >= ring.nvars() {
            return None;
        }
        Some(self.exponent_raw(ring.nvars() as usize, i as usize))
    }

    #[inline]
    fn exponent_raw(&self, nvars: usize, i: usize) -> u32 {
        let byte_idx = byte_index_for_var(nvars, i);
        let (word, shift) = split_byte_index(byte_idx);
        let complement = ((self.packed[word] >> shift) & 0xFF) as u32;
        u8::MAX as u32 - complement
    }

    /// Copy the exponent vector into a `Vec<u32>`.
    pub fn exponents(&self, ring: &Ring) -> Vec<u32> {
        let n = ring.nvars() as usize;
        (0..n).map(|i| self.exponent_raw(n, i)).collect()
    }

    // ----- Arithmetic -----

    /// Multiply two monomials. Returns `None` if any resulting exponent
    /// exceeds 255.
    pub fn mul(&self, other: &Self, ring: &Ring) -> Option<Self> {
        let n = ring.nvars() as usize;
        let mut exps = vec![0u32; n];
        for (i, slot) in exps.iter_mut().enumerate() {
            let sum = self.exponent_raw(n, i) + other.exponent_raw(n, i);
            if sum > u8::MAX as u32 {
                return None;
            }
            *slot = sum;
        }
        Self::from_exponents(ring, &exps)
    }

    /// `true` iff `self | other` (each `e_i(self) ≤ e_i(other)`).
    pub fn divides(&self, other: &Self, ring: &Ring) -> bool {
        let n = ring.nvars() as usize;
        (0..n).all(|i| self.exponent_raw(n, i) <= other.exponent_raw(n, i))
    }

    /// Divide. Precondition `other.divides(self)`; returns `None`
    /// otherwise.
    pub fn div(&self, other: &Self, ring: &Ring) -> Option<Self> {
        let n = ring.nvars() as usize;
        let mut exps = vec![0u32; n];
        for (i, slot) in exps.iter_mut().enumerate() {
            let a = self.exponent_raw(n, i);
            let b = other.exponent_raw(n, i);
            if a < b {
                return None;
            }
            *slot = a - b;
        }
        Self::from_exponents(ring, &exps)
    }

    /// Componentwise maximum (least common multiple of monomials).
    pub fn lcm(&self, other: &Self, ring: &Ring) -> Self {
        let n = ring.nvars() as usize;
        let mut exps = vec![0u32; n];
        for (i, slot) in exps.iter_mut().enumerate() {
            *slot = self.exponent_raw(n, i).max(other.exponent_raw(n, i));
        }
        // Each per-var exponent stays ≤ 255; total is ≤ 2·255·n, fits u32.
        Self::from_exponents(ring, &exps).expect("lcm per-var exponents ≤ 255")
    }

    // ----- Ordering -----

    /// Compare under the ring's ordering.
    pub fn cmp(&self, other: &Self, ring: &Ring) -> Ordering {
        match ring.ordering() {
            MonoOrder::DegRevLex => self.cmp_degrevlex(other),
        }
    }

    /// Degrevlex comparison.
    ///
    /// Fast path: lex-compare the four packed words MSB first. The
    /// layout guarantees this yields the degrevlex order *provided*
    /// neither top-byte total-degree cap is saturated (i.e. both
    /// total degrees are ≤ 255). When either operand saturates, the
    /// top byte is uninformative and we fall back on the `total_deg`
    /// cache, then on the remaining bytes below the top.
    fn cmp_degrevlex(&self, other: &Self) -> Ordering {
        let a_cap = (self.packed[WORDS_PER_MONO - 1] >> 56) & 0xFF;
        let b_cap = (other.packed[WORDS_PER_MONO - 1] >> 56) & 0xFF;
        let saturated = a_cap == u8::MAX as u64 || b_cap == u8::MAX as u64;

        if saturated {
            match self.total_deg.cmp(&other.total_deg) {
                Ordering::Equal => {}
                ord => return ord,
            }
            // Equal (uncapped) total degrees: compare the lower three
            // words, then the low 56 bits of the top word.
            for i in (0..WORDS_PER_MONO - 1).rev() {
                match self.packed[i].cmp(&other.packed[i]) {
                    Ordering::Equal => {}
                    ord => return ord,
                }
            }
            let mask = (1u64 << 56) - 1;
            let a_lo = self.packed[WORDS_PER_MONO - 1] & mask;
            let b_lo = other.packed[WORDS_PER_MONO - 1] & mask;
            return a_lo.cmp(&b_lo);
        }

        // Fast path: lex compare of all four words, MSB first.
        for i in (0..WORDS_PER_MONO).rev() {
            match self.packed[i].cmp(&other.packed[i]) {
                Ordering::Equal => {}
                ord => return ord,
            }
        }
        Ordering::Equal
    }

    // ----- Invariants -----

    /// Panic if any internal invariant is violated. Intended for
    /// `debug_assert!` guards and for tests.
    pub fn assert_canonical(&self, ring: &Ring) {
        let n = ring.nvars() as usize;
        let mut total: u64 = 0;
        let mut sev: u64 = 0;

        for i in 0..n {
            let e = self.exponent_raw(n, i);
            assert!(e <= u8::MAX as u32, "exponent {e} at var {i} > 255");
            total += e as u64;
            if e > 0 {
                sev |= 1u64 << (i % 64);
            }
        }

        assert!(total <= u32::MAX as u64, "total degree overflows u32");
        assert_eq!(total as u32, self.total_deg, "total_deg cache mismatch");
        assert_eq!(sev, self.sev, "sev cache mismatch");

        // Top byte is min(total, 255).
        let expected_cap = total.min(u8::MAX as u64);
        let cap = (self.packed[WORDS_PER_MONO - 1] >> 56) & 0xFF;
        assert_eq!(cap, expected_cap, "top-byte total-degree cap mismatch");

        // Bytes outside the [31-n, 30] range (the variable bytes) plus
        // bits outside those byte slots in the top word must be zero.
        // Rather than reason byte-by-byte, reconstruct the expected
        // packed block from exponents and compare.
        let expected = Self::from_exponents(ring, &self.exponents(ring))
            .expect("re-canonicalising from our own exponents must succeed");
        assert_eq!(
            self.packed, expected.packed,
            "packed representation differs from canonical re-build"
        );

        assert_eq!(self.component, 0, "non-zero component not yet supported");
    }
}

impl PartialEq for Monomial {
    fn eq(&self, other: &Self) -> bool {
        self.packed == other.packed && self.component == other.component
    }
}
impl Eq for Monomial {}

// ----- packing helpers -----

/// Byte index of variable `i` in the 32-byte packed block.
///
/// Placement (for `nvars = n`):
///
/// * Byte 31: total-degree cap.
/// * Byte 30: variable `n - 1` (the highest-index variable, most
///   significant in degrevlex tie-break).
/// * Byte 29: variable `n - 2`.
/// * ...
/// * Byte `31 - n`: variable `0`.
/// * Bytes `0 .. 31 - n`: unused, always zero.
///
/// So `byte_index_for_var(n, i) = 31 - (n - i) = i + 31 - n`.
#[inline]
fn byte_index_for_var(nvars: usize, i: usize) -> usize {
    debug_assert!(i < nvars);
    debug_assert!(nvars < WORDS_PER_MONO * 8); // at least one byte for total_deg
    i + (WORDS_PER_MONO * 8 - 1) - nvars
}

/// Split a byte index in `[0, 32)` into `(word_idx, bit_shift)`.
#[inline]
fn split_byte_index(byte_idx: usize) -> (usize, u32) {
    debug_assert!(byte_idx < WORDS_PER_MONO * 8);
    let word = byte_idx / 8;
    let shift = ((byte_idx % 8) * 8) as u32;
    (word, shift)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Field;

    fn mk_ring(nvars: u32) -> Ring {
        Ring::new(nvars, MonoOrder::DegRevLex, Field::new(32003).unwrap()).unwrap()
    }

    #[test]
    fn round_trip_exponents() {
        let r = mk_ring(5);
        let exps = vec![0u32, 3, 7, 0, 12];
        let m = Monomial::from_exponents(&r, &exps).unwrap();
        assert_eq!(m.exponents(&r), exps);
        assert_eq!(m.total_deg(), 22);
        m.assert_canonical(&r);
    }

    #[test]
    fn one_is_canonical() {
        let r = mk_ring(7);
        let one = Monomial::one(&r);
        assert_eq!(one.total_deg(), 0);
        assert_eq!(one.sev(), 0);
        one.assert_canonical(&r);
    }

    #[test]
    fn sev_matches_nonzero_vars() {
        let r = mk_ring(10);
        let m = Monomial::from_exponents(&r, &[0, 2, 0, 0, 5, 0, 0, 1, 0, 0]).unwrap();
        let expected = (1u64 << 1) | (1u64 << 4) | (1u64 << 7);
        assert_eq!(m.sev(), expected);
    }

    #[test]
    fn divides_is_componentwise_le() {
        let r = mk_ring(3);
        let a = Monomial::from_exponents(&r, &[1, 2, 3]).unwrap();
        let b = Monomial::from_exponents(&r, &[2, 2, 4]).unwrap();
        assert!(a.divides(&b, &r));
        assert!(!b.divides(&a, &r));
    }

    #[test]
    fn div_after_mul_roundtrip() {
        let r = mk_ring(4);
        let a = Monomial::from_exponents(&r, &[1, 2, 0, 5]).unwrap();
        let b = Monomial::from_exponents(&r, &[3, 0, 4, 1]).unwrap();
        let p = a.mul(&b, &r).unwrap();
        let back = p.div(&b, &r).unwrap();
        assert_eq!(back, a);
    }

    #[test]
    fn lcm_is_max_componentwise() {
        let r = mk_ring(3);
        let a = Monomial::from_exponents(&r, &[1, 5, 3]).unwrap();
        let b = Monomial::from_exponents(&r, &[4, 2, 3]).unwrap();
        let l = a.lcm(&b, &r);
        assert_eq!(l.exponents(&r), vec![4, 5, 3]);
    }

    #[test]
    fn degrevlex_cmp_basic() {
        let r = mk_ring(3);
        let x2 = Monomial::from_exponents(&r, &[2, 0, 0]).unwrap();
        let xy = Monomial::from_exponents(&r, &[1, 1, 0]).unwrap();
        let y2 = Monomial::from_exponents(&r, &[0, 2, 0]).unwrap();
        let xz = Monomial::from_exponents(&r, &[1, 0, 1]).unwrap();
        let yz = Monomial::from_exponents(&r, &[0, 1, 1]).unwrap();
        let z2 = Monomial::from_exponents(&r, &[0, 0, 2]).unwrap();

        // Standard degrevlex ordering for 3 variables at total degree 2:
        // x^2 > x*y > y^2 > x*z > y*z > z^2
        assert_eq!(x2.cmp(&xy, &r), Ordering::Greater);
        assert_eq!(xy.cmp(&y2, &r), Ordering::Greater);
        assert_eq!(y2.cmp(&xz, &r), Ordering::Greater);
        assert_eq!(xz.cmp(&yz, &r), Ordering::Greater);
        assert_eq!(yz.cmp(&z2, &r), Ordering::Greater);
    }

    #[test]
    fn degrevlex_cmp_by_total_deg() {
        let r = mk_ring(3);
        let a = Monomial::from_exponents(&r, &[3, 0, 0]).unwrap();
        let b = Monomial::from_exponents(&r, &[0, 0, 2]).unwrap();
        assert_eq!(a.cmp(&b, &r), Ordering::Greater);
    }

    #[test]
    fn degrevlex_cmp_equal() {
        let r = mk_ring(4);
        let a = Monomial::from_exponents(&r, &[1, 2, 3, 4]).unwrap();
        let b = Monomial::from_exponents(&r, &[1, 2, 3, 4]).unwrap();
        assert_eq!(a.cmp(&b, &r), Ordering::Equal);
    }

    #[test]
    fn large_total_deg_cap_still_orders_correctly() {
        let r = mk_ring(2);
        // total_deg = 255 + 200 = 455 > 255; top byte saturates.
        let a = Monomial::from_exponents(&r, &[255, 200]).unwrap();
        let b = Monomial::from_exponents(&r, &[200, 255]).unwrap();
        // Total degrees are equal. Largest index with differing
        // exponent is 1: a_1 = 200, b_1 = 255. Smaller exponent wins
        // degrevlex, so a > b.
        assert_eq!(a.cmp(&b, &r), Ordering::Greater);
    }

    #[test]
    fn degrevlex_tiebreak_on_last_variable() {
        // Classic Cox-Little-O'Shea example: for nvars=3,
        // total deg 3, we want x*y*z < x^2*z. Actually let's test the
        // canonical example: x*y^2 > y^3 > x*y*z > y^2*z > x*z^2 > y*z^2 > z^3.
        let r = mk_ring(3);
        let xy2 = Monomial::from_exponents(&r, &[1, 2, 0]).unwrap();
        let y3 = Monomial::from_exponents(&r, &[0, 3, 0]).unwrap();
        let xyz = Monomial::from_exponents(&r, &[1, 1, 1]).unwrap();
        let y2z = Monomial::from_exponents(&r, &[0, 2, 1]).unwrap();
        let xz2 = Monomial::from_exponents(&r, &[1, 0, 2]).unwrap();
        let yz2 = Monomial::from_exponents(&r, &[0, 1, 2]).unwrap();
        let z3 = Monomial::from_exponents(&r, &[0, 0, 3]).unwrap();
        let sequence = [&xy2, &y3, &xyz, &y2z, &xz2, &yz2, &z3];
        for w in sequence.windows(2) {
            let ord = w[0].cmp(w[1], &r);
            assert_eq!(ord, Ordering::Greater, "{:?} should be > {:?}", w[0].exponents(&r), w[1].exponents(&r));
        }
    }
}
