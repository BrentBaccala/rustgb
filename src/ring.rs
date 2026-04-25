//! Polynomial ring definition.
//!
//! A [`Ring`] bundles the immutable data every polynomial operation needs:
//! number of variables, monomial ordering, coefficient field, and the
//! precomputed bitmasks used by [`crate::monomial::Monomial`]'s mul and
//! cmp routines. Rings are shared between threads via `Arc<Ring>` (see
//! `~/project/docs/rust-bba-port-plan.md` §6.1); the type is `Send + Sync`
//! because it holds only immutable data.
//!
//! This bootstrap fixes two representation parameters:
//!
//! * **Ordering**: [`MonoOrder::DegRevLex`] only.
//! * **Bits per variable**: 7 + 1 guard. Variables are packed 8 bits per
//!   slot in a 4 × u64 (32-byte) block; bits 0-6 of each variable byte
//!   hold the exponent (max value 127), and bit 7 is reserved as an
//!   overflow guard so a Singular-style divmask check on the result of
//!   a packed-word add can detect per-byte overflow in O(words) ops.
//!   See `~/rustgb/docs/design-decisions.md` ADR-005. The total degree
//!   byte (byte 31) keeps the full 8 bits since it is rewritten cleanly
//!   after every mul rather than incremented, so it needs no guard.
//!
//! Future widths (wider per-variable packing, dynamic ring widening
//! mirroring Singular's `kStratChangeTailRing`) are listed as deferred
//! enhancements in ADR-005.

use crate::field::Field;
use crate::monomial::Monomial;
use crate::ordering::MonoOrder;

/// Bits used to store each variable's exponent in the packed monomial.
/// Eight bytes per slot, of which 7 hold the exponent and 1 is the
/// overflow guard.
pub const BITS_PER_VAR: u8 = 8;

/// Number of bits in a divmask (fixed-width 64-bit bloom filter).
/// See `Ring::divmask_of` and ADR-025 for the rationale.
pub const DIVMASK_BITS: u32 = 64;

/// Maximum value a single variable's exponent may take. Bit 7 of each
/// variable byte is the overflow guard (see ADR-005), so the usable
/// range is [0, 127]. Per ADR-018, ring construction is responsible
/// for ensuring no bba-step product exceeds this bound; release-build
/// [`crate::monomial::Monomial::mul`] does not check.
pub const MAX_VAR_EXP: u32 = 0x7F;

/// Maximum number of variables supported by the 8-bit packing.
///
/// One 8-bit byte is reserved at the front of the packed representation
/// for total degree, leaving 31 bytes of a 256-bit (four-word)
/// exponent block for variables. The port plan aims at 25-variable
/// staging workloads, so 31 gives comfortable headroom.
pub const MAX_VARS: u32 = 31;

/// An immutable polynomial ring.
///
/// Construct via [`Ring::new`]. Share via `Arc<Ring>`. Never mutated
/// after construction; every method takes `&self`.
#[derive(Debug, Clone)]
pub struct Ring {
    /// Number of variables. `1 ≤ nvars ≤ MAX_VARS`.
    nvars: u32,
    /// Monomial ordering. Currently always [`MonoOrder::DegRevLex`].
    ordering: MonoOrder,
    /// Coefficient field Z/pZ.
    field: Field,
    /// Per-word overflow guard mask: bit 7 set in each variable byte
    /// slot, 0 elsewhere (top "total-degree" byte and any unused
    /// low bytes). Used by `Monomial::assert_canonical` and by
    /// `Monomial::mul`'s `debug_assert!` invariant (ADR-018). Release
    /// builds of `mul` no longer consult this mask — matching
    /// Singular's PDEBUG-gated check. See ADR-005 / ADR-018 in
    /// `~/rustgb/docs/design-decisions.md`.
    overflow_mask: [u64; 4],
    /// Per-word XOR mask used to flip the degrevlex tie-break direction
    /// at compare time: `0x7F` in each variable byte slot, `0x00` in
    /// the total-degree byte and unused low bytes. XOR'd into packed
    /// words before the lex compare in `Monomial::cmp_degrevlex`. With
    /// direct exponent storage (ADR-005), the variable byte direction
    /// has to be flipped to encode "smaller exponent at the
    /// largest-index differing variable wins"; the top byte is left
    /// untouched so larger total degree wins directly.
    cmp_flip_mask: [u64; 4],
    /// Divmask layout: per-bit `(var, threshold)` pairs.
    ///
    /// Bit `k` of a monomial's divmask is set iff `exp[var_k] > threshold_k`.
    /// The arrays are length [`DIVMASK_BITS`] = 64. Built once at ring
    /// construction time by [`compute_divmask_layout`]; immutable
    /// thereafter. See [`Ring::divmask_of`] for the divmask invariant
    /// and ADR-025 for the design rationale.
    divmask_vars: [u8; DIVMASK_BITS as usize],
    divmask_thresholds: [u32; DIVMASK_BITS as usize],
    /// Per-variable divmask bit-range: `(start_bit, n_bits)` for
    /// variable `v`. Used by the hot-path `divmask_of` to iterate
    /// variables once and unpack bits per variable, rather than
    /// looping over all 64 divmask bits and re-fetching each
    /// variable's exponent. Length is `MAX_VARS as usize + 1` (we
    /// keep the slot for `nvars` itself zero, which lets the hot
    /// path skip variables outside the ring without a separate
    /// length check).
    divmask_var_ranges: [(u8, u8); MAX_VARS as usize + 1],
}

impl Ring {
    /// Construct a new ring.
    ///
    /// Returns `None` if `nvars` is out of range (`0` or `> MAX_VARS`)
    /// or if the caller passes an unsupported ordering. Today only
    /// `DegRevLex` is supported.
    ///
    /// **Caller contract (ADR-018, mirroring Singular's `rComplete`):**
    /// the caller must ensure that every `Monomial::mul` product
    /// arising in the intended computation stays within
    /// [`MAX_VAR_EXP`] (= 127) per variable and within `u32::MAX`
    /// in total degree. Release builds of [`crate::monomial::Monomial::mul`]
    /// do not check this; violating the contract produces silent
    /// exponent corruption (matching Singular's release-mode
    /// `p_ExpVectorAdd` at
    /// `~/Singular/libpolys/polys/monomials/p_polys.h:1432`). Debug
    /// builds catch the violation via `debug_assert!`. If a future
    /// FFI caller admits rings whose bba-step products could
    /// overflow, the dispatch filter (in Singular-rustgb's
    /// `rustgb-dispatch.lib`) must tighten to exclude them before
    /// the ring reaches this constructor.
    pub fn new(nvars: u32, ordering: MonoOrder, field: Field) -> Option<Self> {
        if nvars == 0 || nvars > MAX_VARS {
            return None;
        }
        // Ordering is an exhaustive match; kept as `match` so future
        // variants must consciously opt in.
        match ordering {
            MonoOrder::DegRevLex => {}
        }
        let (overflow_mask, cmp_flip_mask) = compute_packing_masks(nvars);
        let (divmask_vars, divmask_thresholds, divmask_var_ranges) =
            compute_divmask_layout(nvars);
        Some(Self {
            nvars,
            ordering,
            field,
            overflow_mask,
            cmp_flip_mask,
            divmask_vars,
            divmask_thresholds,
            divmask_var_ranges,
        })
    }

    /// Number of variables.
    #[inline]
    pub fn nvars(&self) -> u32 {
        self.nvars
    }

    /// Monomial ordering.
    #[inline]
    pub fn ordering(&self) -> MonoOrder {
        self.ordering
    }

    /// Coefficient field.
    #[inline]
    pub fn field(&self) -> &Field {
        &self.field
    }

    /// Per-word overflow guard mask. See struct docstring.
    #[inline]
    pub fn overflow_mask(&self) -> &[u64; 4] {
        &self.overflow_mask
    }

    /// Per-word degrevlex compare flip mask. See struct docstring.
    #[inline]
    pub fn cmp_flip_mask(&self) -> &[u64; 4] {
        &self.cmp_flip_mask
    }

    /// Compute the divmask of a monomial in this ring.
    ///
    /// The divmask is a 64-bit bloom filter encoding exponent ranges
    /// per variable. Bit `k` is set iff the monomial's `exp[divmask_vars[k]]`
    /// strictly exceeds `divmask_thresholds[k]`.
    ///
    /// # Divmask invariant
    ///
    /// For any pair of monomials `a, b` in this ring with
    /// `a.divides(b, ring) == true` (i.e. `e_a[v] <= e_b[v]` for all
    /// variables `v`), we have:
    ///
    /// ```text
    /// (divmask_of(a) & !divmask_of(b)) == 0
    /// ```
    ///
    /// Proof: each bit `(v, t)` of the divmask is set iff `exp[v] > t`.
    /// If `e_a[v] > t` and `e_a[v] <= e_b[v]`, then `e_b[v] > t` also.
    /// So every bit set in `divmask_of(a)` is also set in
    /// `divmask_of(b)`, hence `divmask(a) & ~divmask(b) == 0`.
    ///
    /// The contrapositive is the fast-reject: if
    /// `(divmask_a & !divmask_b) != 0`, then `a` does not divide `b`,
    /// without needing the full byte-by-byte `Monomial::divides` walk.
    ///
    /// # Layout
    ///
    /// Determined at ring construction time by
    /// [`compute_divmask_layout`]. For `nvars` variables, each variable
    /// gets `64 / nvars` bits (rounded down), with the remainder bits
    /// going to the lowest-indexed variables. Per-variable thresholds
    /// follow a geometric scale `0, 1, 2, 4, 8, 16, ...` (the standard
    /// mathicgb default — bit 0 is "exp > 0", bit 1 is "exp > 1", bit
    /// 2 is "exp > 2", and so on). Variables with zero allocated bits
    /// (only possible if `nvars > 64`, which we forbid) contribute
    /// nothing to the mask.
    ///
    /// # Singular precedent
    ///
    /// Singular's `r->divmask` (set by `rGetDivMask` in
    /// `~/Singular/libpolys/polys/monomials/ring.cc:4200`) is a
    /// per-word top-bit mask used by `_p_LmDivisibleByNoComp` for
    /// per-word borrow detection during the divisibility test —
    /// structurally different from the bloom filter here. The
    /// per-monomial-cached richer scheme matches mathicgb's `DivMask`
    /// (`~/mathic/src/mathic/DivMask.h`) and is what ADR-025 ports.
    #[inline]
    pub fn divmask_of(&self, m: &Monomial) -> u64 {
        let mut mask = 0u64;
        let nvars = self.nvars as usize;
        // Walk variables once; for each, count how many of this
        // variable's geometric thresholds (0, 1, 2, 4, 8, ...) are
        // strictly less than the exponent, and shift in that many
        // 1-bits at the variable's start position.
        //
        // Branchless count formula (matching the threshold scale
        // 0, 1, 2, 4, ..., 2^(b-2)):
        //   * e == 0          → count = 0
        //   * e >= 1          → count >= 1 (bit 0: e > 0)
        //   * e >= 2          → count >= 2 (bit 1: e > 1)
        //   * e > 2^(j-1) for j ≥ 2 → bit j set
        //
        // Equivalently: count = min(n_bits, ceil_log2(e+1)) where
        // ceil_log2 here is the index of the high bit + 1. Concretely,
        // for e>=1: count = 1 + 32 - (e-1).leading_zeros() if e>=2 else 1.
        // Simplest correct form: clamp to n_bits and use u32::checked_log2.
        for v in 0..nvars {
            let e = m.exponent_raw_pub(nvars, v);
            let (start, n_bits) = self.divmask_var_ranges[v];
            // Set count = number of thresholds 0, 1, 2, 4, 8, ...
            // strictly less than e. Walk while it's cheaper than
            // computing a log; at n_bits ≤ 4 (the common case for
            // nvars ≥ 16), the loop runs ≤ 4 times and LLVM unrolls
            // it. For nvars = 5 or smaller, n_bits can be up to 12,
            // and the per-iteration cost is still a single `cmp`
            // and a `setne` plus an OR.
            let mut t: u32 = 0;
            let mut bit = start;
            for _ in 0..n_bits {
                if e > t {
                    mask |= 1u64 << bit;
                } else {
                    // No subsequent threshold (which doubles or
                    // increments t) will be reached either, so we
                    // can break early. This shortens the inner loop
                    // for the common case `e == 0` to one cmp.
                    break;
                }
                bit += 1;
                t = if t == 0 { 1 } else { t * 2 };
            }
        }
        mask
    }

    /// Number of bits dedicated to the highest-bit-budget variable.
    /// Exposed for ADR-025 documentation tests.
    #[doc(hidden)]
    pub fn divmask_bits_for_var(&self, var: u32) -> u32 {
        let mut count = 0u32;
        for k in 0..DIVMASK_BITS as usize {
            if u32::from(self.divmask_vars[k]) == var {
                count += 1;
            }
        }
        count
    }

    /// Borrow the per-bit `var` array of the divmask layout. Length
    /// [`DIVMASK_BITS`]. Each entry is the variable index whose
    /// exponent that bit examines, or `nvars` (sentinel) for an
    /// unused bit.
    #[doc(hidden)]
    #[inline]
    pub fn divmask_vars(&self) -> &[u8; DIVMASK_BITS as usize] {
        &self.divmask_vars
    }

    /// Borrow the per-bit threshold array of the divmask layout.
    /// Length [`DIVMASK_BITS`]. Bit `k` is set in
    /// [`Self::divmask_of`] iff `exp[divmask_vars[k]] > thresholds[k]`.
    #[doc(hidden)]
    #[inline]
    pub fn divmask_thresholds(&self) -> &[u32; DIVMASK_BITS as usize] {
        &self.divmask_thresholds
    }

    /// Predicate for the **hand-specialised dispatch fingerprint**
    /// (ADR-023): characteristic Z/p (always true today since [`Field`]
    /// only models Z/pZ), monomial ordering [`MonoOrder::DegRevLex`],
    /// and `nvars ≤ MAX_VARS` (= 31, the WORDS_PER_MONO=4 length-4
    /// packing limit). When this predicate returns `true`, the
    /// `Poly::*_zp_degrevlex_len4` specialisations may be used in
    /// place of the generic implementations.
    ///
    /// The check is constant-foldable in practice: a `Ring` is built
    /// once at FFI boundary and the answer is invariant for the
    /// lifetime of a `bba()` call. LLVM tends to hoist the dispatch
    /// branch out of the inner loop entirely once the caller's
    /// `&Ring` parameter is realised.
    ///
    /// Mirrors Singular's per-ring procs-table lookup: at `rComplete`
    /// time Singular picks the single
    /// `p_Minus_mm_Mult_qq__FieldZp_LengthFour_OrdRevDeg`
    /// instantiation and stamps a function pointer; rustgb does the
    /// same selection inline, with the predicate as the compile-time
    /// gate.
    #[inline]
    pub fn is_zp_degrevlex(&self) -> bool {
        // `MonoOrder::DegRevLex` is the only variant; the match folds
        // to a constant. `Field` only models Z/pZ today, so the
        // characteristic check is trivially true. The `nvars` bound
        // is enforced at construction (`Ring::new` rejects nvars >
        // MAX_VARS), so this also folds to true. Kept as an explicit
        // method so future variants (e.g. a different ordering) must
        // consciously update the dispatch fingerprint.
        match self.ordering {
            MonoOrder::DegRevLex => self.nvars <= MAX_VARS,
        }
    }
}

impl PartialEq for Ring {
    fn eq(&self, other: &Self) -> bool {
        // The masks are a pure function of nvars + ordering, so we don't
        // need to compare them explicitly.
        self.nvars == other.nvars && self.ordering == other.ordering && self.field == other.field
    }
}
impl Eq for Ring {}

/// Compute the packing masks for a ring with the given number of
/// variables. The variable bytes occupy positions `[31 - nvars, 30]`
/// of the 32-byte packed block (byte 31 = total-degree, low bytes
/// always zero). See `monomial::byte_index_for_var`.
fn compute_packing_masks(nvars: u32) -> ([u64; 4], [u64; 4]) {
    let n = nvars as usize;
    let mut overflow = [0u64; 4];
    let mut flip = [0u64; 4];
    let first_var_byte = 31 - n; // (4*8 - 1) - n
    let last_var_byte = 30;
    for byte_idx in first_var_byte..=last_var_byte {
        let word = byte_idx / 8;
        let shift = ((byte_idx % 8) * 8) as u32;
        overflow[word] |= 0x80u64 << shift;
        flip[word] |= 0x7Fu64 << shift;
    }
    (overflow, flip)
}

/// Compute the per-bit `(variable, threshold)` mapping for the divmask.
///
/// Distributes [`DIVMASK_BITS`] (= 64) bits across `nvars` variables.
/// Each variable gets `64 / nvars` bits; the remainder `64 % nvars`
/// goes to the lowest-indexed variables.
///
/// Per-variable thresholds follow the geometric scale
/// `0, 1, 2, 4, 8, 16, ...`:
/// * bit 0 of variable `v` ⇔ `exp[v] > 0` (i.e. exp >= 1)
/// * bit 1 ⇔ `exp[v] > 1` (i.e. exp >= 2)
/// * bit 2 ⇔ `exp[v] > 2` (i.e. exp >= 3)
/// * bit 3 ⇔ `exp[v] > 4` (i.e. exp >= 5)
/// * bit `k` ⇔ `exp[v] > 2^(k-1)` for `k >= 1`
///
/// This is mathicgb's default (`DivMask::Calculator::rebuildDefault`,
/// `~/mathic/src/mathic/DivMask.h:312-327`): coarse for low exponents
/// (which dominate in typical bba intermediates), exponential for
/// higher ones (so a single bit covers a wide range and false-positive
/// rates stay bounded for outlier exponent values).
///
/// Returns `(vars, thresholds)` with length [`DIVMASK_BITS`]. Slots
/// allocated to variable `v < nvars` set `vars[k] = v as u8`; any
/// remaining slots (only when `nvars > DIVMASK_BITS`, which the
/// constructor rejects) are filled with the sentinel `vars[k] = nvars
/// as u8` so [`Ring::divmask_of`] skips them. With our `MAX_VARS = 31`
/// constraint, every slot is allocated.
fn compute_divmask_layout(
    nvars: u32,
) -> (
    [u8; DIVMASK_BITS as usize],
    [u32; DIVMASK_BITS as usize],
    [(u8, u8); MAX_VARS as usize + 1],
) {
    let mut vars = [0u8; DIVMASK_BITS as usize];
    let mut thresholds = [0u32; DIVMASK_BITS as usize];
    let mut var_ranges = [(0u8, 0u8); MAX_VARS as usize + 1];

    if nvars == 0 || nvars > DIVMASK_BITS {
        // Caller should never reach here (Ring::new rejects). Fill
        // sentinels and return.
        for v in vars.iter_mut() {
            *v = nvars as u8;
        }
        return (vars, thresholds, var_ranges);
    }

    let nvars_us = nvars as usize;
    let bits_per_var_floor = DIVMASK_BITS as usize / nvars_us;
    let extra = DIVMASK_BITS as usize % nvars_us;

    let mut k: usize = 0;
    for v in 0..nvars_us {
        // Lower-indexed variables get one extra bit when the bit count
        // doesn't divide evenly. (The order is irrelevant for
        // correctness, but fixing it keeps the layout reproducible.)
        let bits_for_v = bits_per_var_floor + if v < extra { 1 } else { 0 };
        var_ranges[v] = (k as u8, bits_for_v as u8);
        for j in 0..bits_for_v {
            vars[k] = v as u8;
            // Threshold scale: bit 0 → 0, bit 1 → 1, bit j → 2^(j-1).
            thresholds[k] = if j == 0 { 0 } else { 1u32 << (j - 1) };
            k += 1;
        }
    }
    debug_assert_eq!(k, DIVMASK_BITS as usize);
    (vars, thresholds, var_ranges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructs_valid_ring() {
        let f = Field::new(32003).unwrap();
        let r = Ring::new(5, MonoOrder::DegRevLex, f).unwrap();
        assert_eq!(r.nvars(), 5);
    }

    #[test]
    fn rejects_out_of_range_nvars() {
        let f = Field::new(5).unwrap();
        assert!(Ring::new(0, MonoOrder::DegRevLex, f).is_none());
        assert!(Ring::new(MAX_VARS + 1, MonoOrder::DegRevLex, f).is_none());
        assert!(Ring::new(MAX_VARS, MonoOrder::DegRevLex, f).is_some());
    }

    #[test]
    fn packing_masks_cover_variable_bytes_only() {
        // For nvars = 3: variable bytes are 28, 29, 30 (in word 3,
        // shifts 32, 40, 48). Top byte 31 (shift 56, total-degree)
        // and all of words 0..3 should be zero.
        let f = Field::new(2).unwrap();
        let r = Ring::new(3, MonoOrder::DegRevLex, f).unwrap();
        let ovf = r.overflow_mask();
        let flip = r.cmp_flip_mask();
        assert_eq!(ovf, &[0, 0, 0, 0x0080808000000000]);
        assert_eq!(flip, &[0, 0, 0, 0x007F7F7F00000000]);

        // For nvars = 25: variable bytes 6..=30. Crosses word
        // boundaries at byte 8, 16, 24. Top byte (shift 56) excluded.
        let r = Ring::new(25, MonoOrder::DegRevLex, Field::new(2).unwrap()).unwrap();
        let ovf = r.overflow_mask();
        // Word 0: bytes 6, 7 → shifts 48, 56.
        assert_eq!(ovf[0], 0x8080_0000_0000_0000);
        // Word 1, 2: all eight bytes have the guard.
        assert_eq!(ovf[1], 0x8080_8080_8080_8080);
        assert_eq!(ovf[2], 0x8080_8080_8080_8080);
        // Word 3: bytes 24..=30 (shifts 0..=48) get the guard;
        //         byte 31 (shift 56, total-deg) does NOT.
        assert_eq!(ovf[3], 0x0080_8080_8080_8080);
    }

    #[test]
    fn packing_masks_have_no_overlap_with_unused_bytes() {
        // For nvars = 1: only byte 30. Verify no other byte set.
        let r = Ring::new(1, MonoOrder::DegRevLex, Field::new(2).unwrap()).unwrap();
        let ovf = r.overflow_mask();
        assert_eq!(ovf, &[0, 0, 0, 0x0080_0000_0000_0000]);
    }

    #[test]
    fn divmask_layout_distributes_bits() {
        // nvars = 1 → 64 bits all to var 0.
        let r = Ring::new(1, MonoOrder::DegRevLex, Field::new(2).unwrap()).unwrap();
        assert_eq!(r.divmask_bits_for_var(0), 64);
        for k in 0..64usize {
            assert_eq!(r.divmask_vars()[k], 0);
        }
        // First bit threshold 0; second 1; third 2; fourth 4; ...
        assert_eq!(r.divmask_thresholds()[0], 0);
        assert_eq!(r.divmask_thresholds()[1], 1);
        assert_eq!(r.divmask_thresholds()[2], 2);
        assert_eq!(r.divmask_thresholds()[3], 4);
        assert_eq!(r.divmask_thresholds()[4], 8);

        // nvars = 16 → 4 bits per var, even split.
        let r = Ring::new(16, MonoOrder::DegRevLex, Field::new(2).unwrap()).unwrap();
        for v in 0..16 {
            assert_eq!(r.divmask_bits_for_var(v), 4, "var {v}");
        }

        // nvars = 31 → 2 bits per var × 31 = 62, with the 2 leftover
        // bits going to the lowest-indexed two vars (so vars 0 and 1
        // get 3 bits each, vars 2..=30 get 2 bits each).
        let r = Ring::new(31, MonoOrder::DegRevLex, Field::new(2).unwrap()).unwrap();
        assert_eq!(r.divmask_bits_for_var(0), 3);
        assert_eq!(r.divmask_bits_for_var(1), 3);
        for v in 2..31 {
            assert_eq!(r.divmask_bits_for_var(v), 2, "var {v}");
        }
        // Total bits accounted for: 2*3 + 29*2 = 64.
        let total: u32 = (0..31).map(|v| r.divmask_bits_for_var(v)).sum();
        assert_eq!(total, 64);
    }

    #[test]
    fn divmask_invariant_simple_examples() {
        // nvars = 3, with thresholds spread per var.
        // a = (1, 0, 0) divides b = (2, 1, 0). Verify the divmask
        // invariant: (mask_a & !mask_b) == 0.
        let f = Field::new(32003).unwrap();
        let r = Ring::new(3, MonoOrder::DegRevLex, f).unwrap();
        let a = Monomial::from_exponents(&r, &[1, 0, 0]).unwrap();
        let b = Monomial::from_exponents(&r, &[2, 1, 0]).unwrap();
        assert!(a.divides(&b, &r));
        let ma = r.divmask_of(&a);
        let mb = r.divmask_of(&b);
        assert_eq!(ma & !mb, 0, "divmask invariant failed for a | b");

        // Identity: 1 | anything → mask(1) = 0 ⊆ everything.
        let one = Monomial::one(&r);
        assert_eq!(r.divmask_of(&one), 0);

        // Doubled exponents: a doubles in every var; (mask_a & !mask_b) == 0.
        let a = Monomial::from_exponents(&r, &[3, 5, 7]).unwrap();
        let b = Monomial::from_exponents(&r, &[6, 10, 14]).unwrap();
        assert!(a.divides(&b, &r));
        assert_eq!(r.divmask_of(&a) & !r.divmask_of(&b), 0);
    }

    #[test]
    fn divmask_negative_examples() {
        // Non-divisor: a = (5, 0, 0), b = (3, 0, 0). a does not divide b.
        // We expect the divmask to detect this with high probability.
        let f = Field::new(32003).unwrap();
        let r = Ring::new(3, MonoOrder::DegRevLex, f).unwrap();
        let a = Monomial::from_exponents(&r, &[5, 0, 0]).unwrap();
        let b = Monomial::from_exponents(&r, &[3, 0, 0]).unwrap();
        assert!(!a.divides(&b, &r));
        // a's exp[0] = 5 > 4 sets bit "exp[0] > 4"; b's exp[0] = 3
        // does not. So mask(a) has a bit not in mask(b).
        let ma = r.divmask_of(&a);
        let mb = r.divmask_of(&b);
        assert_ne!(
            ma & !mb,
            0,
            "divmask should reject a=(5,0,0) ∤ b=(3,0,0); mask_a = {ma:#x}, mask_b = {mb:#x}"
        );
    }
}
