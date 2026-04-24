//! Packed-exponent monomials.
//!
//! A [`Monomial`] represents `x_0^{e_0} * x_1^{e_1} * ... * x_{n-1}^{e_{n-1}}`
//! for a ring with `nvars = n`. Exponents are packed into four `u64`
//! words — 32 bytes total — at 8 bytes per slot, of which the low 7
//! bits hold the exponent (max value 127) and bit 7 is reserved as an
//! overflow guard. See `~/rustgb/docs/design-decisions.md` ADR-005 for
//! the rationale; in short: this matches Singular's `p_LmExpVectorAddIsOk`
//! divmask trick (a single AND-and-test per word detects per-byte
//! overflow on the result of a packed-word add) and FLINT's flat
//! `mpoly_monomial_add`.
//!
//! ## Packing (degrevlex, 7 bits/var + guard)
//!
//! ```text
//! word 3 (MSB) : [ byte31 | byte30 | byte29 | ... | byte24 ]
//! word 2       : [ byte23 | byte22 | ....................  ]
//! word 1       : [ byte15 | byte14 | ....................  ]
//! word 0 (LSB) : [  byte7 |  byte6 | ....................  ]
//! ```
//!
//! Byte 31 holds `min(total_deg, 255)` (the "capped" total degree —
//! 8 bits, no guard, since this byte is rewritten cleanly after every
//! mul rather than incremented in place). Bytes 30..(31 - nvars) hold
//! the **direct** exponent of the variables: variable `nvars - 1` at
//! byte `30`, variable `nvars - 2` at byte `29`, ..., variable `0` at
//! byte `31 - nvars`. In each variable byte, bits 0-6 hold the
//! exponent and bit 7 is the overflow guard (always 0 in a canonical
//! monomial). Bytes below `31 - nvars` are always zero.
//!
//! ## Multiplication and overflow detection
//!
//! `Monomial::mul` is a plain word-wise wrapping-add of the four
//! packed words (LLVM auto-vectorises this into one `vpaddq`). Because
//! each variable byte's value is ≤ 127, byte-wise sums fit in 8 bits
//! and no carry can propagate from one variable byte into the next —
//! the word-add is byte-isolated for the purposes of correctness.
//! Overflow is detected by examining the guard bits of the result
//! word: `(a + b) & ring.overflow_mask` is nonzero iff some byte's
//! sum exceeded 127 (and either set bit 7 of that byte itself, or
//! would have carried further but had nowhere to go).
//!
//! ## Comparison
//!
//! Degrevlex compare is still a lex compare of the four packed words
//! MSB first, but with each word XOR'd against `ring.cmp_flip_mask`
//! before the compare. The mask has `0x7F` in each variable byte slot
//! and `0x00` in the total-degree byte, so:
//!
//! * Top byte (total-deg cap) compares directly: larger byte wins,
//!   matching degrevlex's primary "higher total degree wins".
//! * Variable bytes: `e ^ 0x7F = 127 - e` (since bit 7 is always 0
//!   in canonical form). Larger `e` becomes smaller after XOR, so
//!   smaller exponent wins the byte compare — matching degrevlex's
//!   tie-break rule "smaller exponent at the largest-index differing
//!   variable wins".
//!
//! Caveat: total degrees that exceed 255 saturate the top byte. When
//! either operand's cap is saturated, `cmp_degrevlex` falls back on
//! the cached `total_deg: u32` first, then on the variable bytes
//! through the same XOR-flipped compare.
//!
//! ## Caches
//!
//! * `sev` (short exponent vector): a 64-bit bloom filter with bit
//!   `i mod 64` set iff `e_i > 0`. Used by the bba sweep as a
//!   divisibility pre-filter.
//! * `total_deg`: sum of exponents, not capped.
//! * `component`: always 0 today. Reserved for future module support.
//!
//! Reference: mathicgb `MonoMonoid.hpp`, Singular's `p_ExpVectorAdd` /
//! `p_LmExpVectorAddIsOk`, and FLINT's `mpoly_monomial_add`.
//! Algorithms re-derived, not copied. See ADR-005 for the
//! Singular/FLINT comparison and the rationale for the 7+1 layout.

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
    /// Returns `None` if the length is wrong, any exponent exceeds
    /// [`crate::ring::MAX_VAR_EXP`] (= 127, the 7-bit per-variable
    /// limit), or the total degree exceeds `u32::MAX`.
    pub fn from_exponents(ring: &Ring, exps: &[u32]) -> Option<Self> {
        let n = ring.nvars() as usize;
        if exps.len() != n {
            return None;
        }
        for &e in exps {
            if e > crate::ring::MAX_VAR_EXP {
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
            // Direct storage: the byte is the exponent itself, in
            // bits 0-6. Bit 7 (the overflow guard) stays clear in
            // canonical form. A zero exponent leaves the byte at 0,
            // which is the natural reading.
            let byte_idx = byte_index_for_var(n, i);
            let (word, shift) = split_byte_index(byte_idx);
            packed[word] |= (e as u64) << shift;
        }

        // Top byte: capped total degree (full 8 bits, no guard).
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

    /// Borrow the packed exponent block (4 × u64 = 32 bytes).
    /// Layout per the module documentation: byte 31 = capped
    /// total degree, bytes [31-nvars, 30] = direct exponents
    /// (per ADR-005), low bytes always zero.
    ///
    /// Used by [`crate::reducer`] (ADR-008) to construct heap
    /// `cmp_key`s by XOR'ing against the ring's `cmp_flip_mask`.
    #[inline]
    pub fn packed(&self) -> &[u64; 4] {
        &self.packed
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
        // Direct storage: bits 0-6 hold the exponent; bit 7 is the
        // (always-zero in canonical form) overflow guard, masked off.
        ((self.packed[word] >> shift) & 0x7F) as u32
    }

    /// Copy the exponent vector into a `Vec<u32>`.
    pub fn exponents(&self, ring: &Ring) -> Vec<u32> {
        let n = ring.nvars() as usize;
        (0..n).map(|i| self.exponent_raw(n, i)).collect()
    }

    // ----- Arithmetic -----

    /// Multiply two monomials. Returns `None` if any resulting exponent
    /// exceeds [`crate::ring::MAX_VAR_EXP`] (= 127), or the resulting
    /// total degree exceeds `u32::MAX`.
    ///
    /// Implementation per ADR-005, with the codegen-motivated split
    /// in ADR-017: the word-wise wrapping-add is emitted as an
    /// explicit 4-element array literal (so LLVM sees four
    /// independent reads+writes with no loop structure and, under
    /// `-C target-cpu=native` on an AVX2 host, folds the four u64
    /// adds into one `vpaddq ymm`); the divmask overflow check is
    /// then a separate OR-reduction over the four words with a
    /// single final branch, rather than a per-word `any()` early
    /// exit that would block the vectorizer from batching the adds.
    /// The top byte (total-degree cap) is rewritten cleanly from
    /// the cached u32 total rather than relying on the wrap-add
    /// result.
    pub fn mul(&self, other: &Self, ring: &Ring) -> Option<Self> {
        // The explicit unroll below assumes exactly four words; if
        // WORDS_PER_MONO ever changes, update the literal.
        const _: () = assert!(WORDS_PER_MONO == 4);

        // Word-wise add. Explicit 4-element literal so LLVM sees
        // four independent reads+writes; overflow within a variable
        // byte sets the guard bit (bit 7) of that byte, no carry can
        // propagate to the neighbouring byte because both inputs are
        // ≤ 127 in their low 7 bits and bit 7 is always 0 in canonical
        // form, so each byte sum is ≤ 254 — always fits in 8 bits.
        let mut packed: [u64; WORDS_PER_MONO] = [
            self.packed[0].wrapping_add(other.packed[0]),
            self.packed[1].wrapping_add(other.packed[1]),
            self.packed[2].wrapping_add(other.packed[2]),
            self.packed[3].wrapping_add(other.packed[3]),
        ];

        // Divmask overflow check, OR-reduced into a single branch so
        // the add above stays a pure data dependency — no per-word
        // early-exit between the adds. Any guard bit set in the
        // result signals per-byte exponent > 127. The mask zeroes
        // the total-degree byte, so a wrapped top byte does not
        // trigger a false positive here.
        let m = ring.overflow_mask();
        let ovf = (packed[0] & m[0])
            | (packed[1] & m[1])
            | (packed[2] & m[2])
            | (packed[3] & m[3]);
        if ovf != 0 {
            return None;
        }

        // Total degree (uncapped, exact via the cached u32 sum).
        let total = self.total_deg as u64 + other.total_deg as u64;
        if total > u32::MAX as u64 {
            return None;
        }
        // Rewrite the top byte: clear the wrap-add result and write
        // the actual cap.
        let capped = total.min(u8::MAX as u64);
        packed[WORDS_PER_MONO - 1] = (packed[WORDS_PER_MONO - 1] & !(0xFFu64 << 56)) | (capped << 56);

        // SEV update: e_new > 0 iff e_a > 0 OR e_b > 0; so OR the sevs.
        let sev = self.sev | other.sev;

        Some(Self {
            packed,
            sev,
            total_deg: total as u32,
            component: 0,
        })
    }

    /// `true` iff `self | other` (each `e_i(self) ≤ e_i(other)`).
    ///
    /// With direct exponent storage (ADR-005), this is a per-byte
    /// `≤` test. Implemented byte-by-byte over the variable bytes;
    /// could be SIMD'd later if it shows up in a profile.
    pub fn divides(&self, other: &Self, ring: &Ring) -> bool {
        let n = ring.nvars() as usize;
        let first_var_byte = (WORDS_PER_MONO * 8 - 1) - n; // = 31 - n
        let last_var_byte = WORDS_PER_MONO * 8 - 2; // 30
        for byte_idx in first_var_byte..=last_var_byte {
            let (word, shift) = split_byte_index(byte_idx);
            // Mask to 0x7F to ignore the (always-zero in canonical
            // form) guard bit. Direct storage: byte value == exponent.
            let ea = (self.packed[word] >> shift) & 0x7F;
            let eb = (other.packed[word] >> shift) & 0x7F;
            if ea > eb {
                return false;
            }
        }
        true
    }

    /// Divide. Precondition `other.divides(self)`; returns `None`
    /// otherwise.
    ///
    /// With direct storage, the per-byte op is `e_new = e_self - e_other`,
    /// rejecting when `e_other > e_self`. Per-byte loop maintained for
    /// the same reason as `divides`.
    pub fn div(&self, other: &Self, ring: &Ring) -> Option<Self> {
        let n = ring.nvars() as usize;
        let first_var_byte = (WORDS_PER_MONO * 8 - 1) - n;
        let last_var_byte = WORDS_PER_MONO * 8 - 2;
        let mut packed = [0u64; WORDS_PER_MONO];
        let mut sev: u64 = 0;
        for byte_idx in first_var_byte..=last_var_byte {
            let (word, shift) = split_byte_index(byte_idx);
            let ea = (self.packed[word] >> shift) & 0x7F;
            let eb = (other.packed[word] >> shift) & 0x7F;
            if eb > ea {
                return None;
            }
            let new_e = ea - eb;
            packed[word] |= new_e << shift;
            if new_e > 0 {
                // Recover variable index from byte position:
                // var i has byte index `i + 31 - n`, so `i = byte - 31 + n`.
                let var_i = byte_idx + n - (WORDS_PER_MONO * 8 - 1);
                sev |= 1u64 << (var_i % 64);
            }
        }
        // Total degree: self.total_deg - other.total_deg.
        if other.total_deg > self.total_deg {
            return None;
        }
        let total = self.total_deg - other.total_deg;
        let capped = (total as u64).min(u8::MAX as u64);
        packed[WORDS_PER_MONO - 1] |= capped << 56;
        Some(Self {
            packed,
            sev,
            total_deg: total,
            component: 0,
        })
    }

    /// Componentwise maximum (least common multiple of monomials).
    pub fn lcm(&self, other: &Self, ring: &Ring) -> Self {
        let n = ring.nvars() as usize;
        let mut exps = vec![0u32; n];
        for (i, slot) in exps.iter_mut().enumerate() {
            *slot = self.exponent_raw(n, i).max(other.exponent_raw(n, i));
        }
        // Each per-var exponent stays ≤ MAX_VAR_EXP (127); total fits u32.
        Self::from_exponents(ring, &exps).expect("lcm per-var exponents ≤ MAX_VAR_EXP")
    }

    // ----- Ordering -----

    /// Compare under the ring's ordering.
    pub fn cmp(&self, other: &Self, ring: &Ring) -> Ordering {
        match ring.ordering() {
            MonoOrder::DegRevLex => self.cmp_degrevlex(other, ring),
        }
    }

    /// Degrevlex comparison.
    ///
    /// With direct exponent storage (ADR-005) the lex order of the raw
    /// packed words would compare variable bytes the wrong way (larger
    /// exponent = greater), so we XOR each word against
    /// `ring.cmp_flip_mask` first. The mask is `0x7F` in each variable
    /// byte slot and `0x00` in the total-degree byte and unused bytes,
    /// so:
    /// * Top byte (capped total-deg) compares directly: larger byte
    ///   wins (degrevlex's primary "higher total degree" rule).
    /// * Variable bytes after XOR: `127 - e`. Smaller `e` becomes
    ///   greater after the flip, matching degrevlex's tie-break
    ///   "smaller exponent at the largest-index differing variable
    ///   wins".
    ///
    /// Saturation fallback: when either operand's capped top byte is
    /// 255, the cap byte is uninformative; we fall back on the cached
    /// `total_deg: u32` first, then on the variable bytes through the
    /// same XOR-flipped compare.
    fn cmp_degrevlex(&self, other: &Self, ring: &Ring) -> Ordering {
        let a_cap = (self.packed[WORDS_PER_MONO - 1] >> 56) & 0xFF;
        let b_cap = (other.packed[WORDS_PER_MONO - 1] >> 56) & 0xFF;
        let saturated = a_cap == u8::MAX as u64 || b_cap == u8::MAX as u64;
        let mask = ring.cmp_flip_mask();

        if saturated {
            match self.total_deg.cmp(&other.total_deg) {
                Ordering::Equal => {}
                ord => return ord,
            }
            // Equal (uncapped) total degrees: compare the lower three
            // words via the flip mask, then the low 56 bits of the top
            // word (also via the flip mask, which has the top byte
            // zeroed).
            for i in (0..WORDS_PER_MONO - 1).rev() {
                let av = self.packed[i] ^ mask[i];
                let bv = other.packed[i] ^ mask[i];
                match av.cmp(&bv) {
                    Ordering::Equal => {}
                    ord => return ord,
                }
            }
            let lo_mask = (1u64 << 56) - 1;
            let av_top = (self.packed[WORDS_PER_MONO - 1] ^ mask[WORDS_PER_MONO - 1]) & lo_mask;
            let bv_top = (other.packed[WORDS_PER_MONO - 1] ^ mask[WORDS_PER_MONO - 1]) & lo_mask;
            return av_top.cmp(&bv_top);
        }

        // Fast path: lex compare of all four words via the flip mask,
        // MSB first.
        for i in (0..WORDS_PER_MONO).rev() {
            let av = self.packed[i] ^ mask[i];
            let bv = other.packed[i] ^ mask[i];
            match av.cmp(&bv) {
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
            assert!(
                e <= crate::ring::MAX_VAR_EXP,
                "exponent {e} at var {i} exceeds 7-bit limit ({})",
                crate::ring::MAX_VAR_EXP
            );
            total += e as u64;
            if e > 0 {
                sev |= 1u64 << (i % 64);
            }
        }

        // Guard bits must all be zero in canonical form.
        for word in 0..WORDS_PER_MONO {
            assert_eq!(
                self.packed[word] & ring.overflow_mask()[word],
                0,
                "overflow guard bit set in word {word} (packed = {:#018x}, mask = {:#018x})",
                self.packed[word],
                ring.overflow_mask()[word]
            );
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
    fn from_exponents_rejects_above_max_var_exp() {
        // ADR-005: per-variable exponents are 7-bit, max 127.
        let r = mk_ring(3);
        assert!(Monomial::from_exponents(&r, &[127, 0, 0]).is_some());
        assert!(Monomial::from_exponents(&r, &[128, 0, 0]).is_none());
        assert!(Monomial::from_exponents(&r, &[0, 200, 0]).is_none());
        assert!(Monomial::from_exponents(&r, &[0, 0, 255]).is_none());
    }

    #[test]
    fn mul_overflow_detected_via_guard_bit() {
        // ADR-005: overflow on a single variable is caught by the
        // divmask check on the result of the packed-word add.
        let r = mk_ring(4);
        // 100 + 50 = 150 > 127: overflow on var 1.
        let a = Monomial::from_exponents(&r, &[1, 100, 0, 0]).unwrap();
        let b = Monomial::from_exponents(&r, &[1, 50, 0, 0]).unwrap();
        assert!(a.mul(&b, &r).is_none(), "150 should overflow 7-bit slot");

        // 64 + 64 = 128: smallest possible overflow (sets the guard
        // bit exactly).
        let a = Monomial::from_exponents(&r, &[64, 0, 0, 0]).unwrap();
        let b = Monomial::from_exponents(&r, &[64, 0, 0, 0]).unwrap();
        assert!(a.mul(&b, &r).is_none(), "exactly 128 should overflow");

        // 63 + 64 = 127: just under the limit, must succeed.
        let a = Monomial::from_exponents(&r, &[63, 0, 0, 0]).unwrap();
        let b = Monomial::from_exponents(&r, &[64, 0, 0, 0]).unwrap();
        let p = a.mul(&b, &r).expect("127 should fit");
        p.assert_canonical(&r);
        assert_eq!(p.exponent(&r, 0).unwrap(), 127);
    }

    #[test]
    fn mul_no_carry_propagation_between_neighbouring_bytes() {
        // The whole point of the guard bit (and the ≤127 invariant)
        // is that a per-byte sum can never exceed 254, so it cannot
        // carry into the next byte and corrupt a neighbour. Verify
        // that two non-overflowing sums in adjacent variables both
        // come out correctly.
        let r = mk_ring(5);
        let a = Monomial::from_exponents(&r, &[60, 70, 80, 90, 100]).unwrap();
        let b = Monomial::from_exponents(&r, &[60, 50, 40, 30, 20]).unwrap();
        // Per-var sums: 120, 120, 120, 120, 120 — all ≤127, no
        // overflow on any byte.
        let p = a.mul(&b, &r).expect("all per-var sums = 120 ≤ 127");
        p.assert_canonical(&r);
        for i in 0..5 {
            assert_eq!(
                p.exponent(&r, i).unwrap(),
                120,
                "exponent of var {i} corrupted by neighbour byte"
            );
        }
        assert_eq!(p.total_deg(), 600);
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
        // With 7-bit-per-var packing (ADR-005), per-variable max is
        // 127. Use nvars = 3 so total_deg can still saturate the
        // 8-bit top-byte cap (>255): 127 + 127 + 50 = 304.
        let r = mk_ring(3);
        let a = Monomial::from_exponents(&r, &[127, 50, 127]).unwrap();
        let b = Monomial::from_exponents(&r, &[50, 127, 127]).unwrap();
        // Total degrees are equal (304). Largest index with differing
        // exponent is 1: a_1 = 50, b_1 = 127. Smaller exponent at
        // largest differing index wins degrevlex, so a > b.
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
            assert_eq!(
                ord,
                Ordering::Greater,
                "{:?} should be > {:?}",
                w[0].exponents(&r),
                w[1].exponents(&r)
            );
        }
    }
}
