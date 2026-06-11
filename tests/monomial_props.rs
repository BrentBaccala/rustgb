//! Property-based tests for monomials.

use proptest::prelude::*;
use rustgb::{Field, MonoOrder, Monomial, Ring};
use std::cmp::Ordering;

/// Generate a random ring with `nvars ∈ [1, 25]`, fixed prime.
///
/// 25 is the staging workload size; it also leaves room (we permit up
/// to 31) for the packing overflow cases.
fn ring_strategy() -> impl Strategy<Value = Ring> {
    (1u32..=25).prop_map(|nvars| {
        let f = Field::new(32003).unwrap();
        Ring::new(nvars, MonoOrder::DegRevLex, f).unwrap()
    })
}

/// Generate a monomial in the given ring with per-variable exponents
/// small enough that products stay within the 8-bit limit.
fn mono_strategy(ring: Ring) -> impl Strategy<Value = (Ring, Monomial)> {
    let n = ring.nvars() as usize;
    // Cap at 30 so sums of up to ~8 monomials stay within 255.
    prop::collection::vec(0u32..30, n).prop_map(move |exps| {
        let m = Monomial::from_exponents(&ring, &exps).unwrap();
        (ring.clone(), m)
    })
}

/// Generate a ring and three monomials sharing it.
fn ring_mono3_strategy() -> impl Strategy<Value = (Ring, Monomial, Monomial, Monomial)> {
    ring_strategy().prop_flat_map(|r| {
        let n = r.nvars() as usize;
        (
            Just(r),
            prop::collection::vec(0u32..20, n),
            prop::collection::vec(0u32..20, n),
            prop::collection::vec(0u32..20, n),
        )
            .prop_map(|(r, ae, be, ce)| {
                let a = Monomial::from_exponents(&r, &ae).unwrap();
                let b = Monomial::from_exponents(&r, &be).unwrap();
                let c = Monomial::from_exponents(&r, &ce).unwrap();
                (r, a, b, c)
            })
    })
}

fn ring_mono2_strategy() -> impl Strategy<Value = (Ring, Monomial, Monomial)> {
    ring_strategy().prop_flat_map(|r| {
        let n = r.nvars() as usize;
        (
            Just(r),
            prop::collection::vec(0u32..25, n),
            prop::collection::vec(0u32..25, n),
        )
            .prop_map(|(r, ae, be)| {
                let a = Monomial::from_exponents(&r, &ae).unwrap();
                let b = Monomial::from_exponents(&r, &be).unwrap();
                (r, a, b)
            })
    })
}

/// Ring strategy that sweeps the packing regimes explicitly:
/// `nvars ∈ {5, 25, 31}`. 5 vars gives a wide divmask budget
/// (`64/5 = 12` bits/var); 25 is the staging size; 31 is the
/// dispatch-shim maximum (`64/31 = 2` bits/var, the tightest divmask
/// packing). ADR-035 / ADR-036 identities are exponent-threshold and
/// componentwise-max facts, so they must hold across all three.
fn packing_regime_ring_strategy() -> impl Strategy<Value = Ring> {
    prop::sample::select(vec![5u32, 25u32, 31u32]).prop_map(|nvars| {
        let f = Field::new(32003).unwrap();
        Ring::new(nvars, MonoOrder::DegRevLex, f).unwrap()
    })
}

/// Two monomials over a packing-regime ring (nvars ∈ {5, 25, 31}).
fn regime_mono2_strategy() -> impl Strategy<Value = (Ring, Monomial, Monomial)> {
    packing_regime_ring_strategy().prop_flat_map(|r| {
        let n = r.nvars() as usize;
        (
            Just(r),
            prop::collection::vec(0u32..25, n),
            prop::collection::vec(0u32..25, n),
        )
            .prop_map(|(r, ae, be)| {
                let a = Monomial::from_exponents(&r, &ae).unwrap();
                let b = Monomial::from_exponents(&r, &be).unwrap();
                (r, a, b)
            })
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1024))]

    #[test]
    fn mul_associative((r, a, b, c) in ring_mono3_strategy()) {
        // Exponents capped at 20 to keep products in range.
        let ab = a.mul(&b, &r);
        let ab_c = ab.mul(&c, &r);
        let bc = b.mul(&c, &r);
        let a_bc = a.mul(&bc, &r);
        prop_assert_eq!(ab_c, a_bc);
    }

    #[test]
    fn mul_commutative((r, a, b) in ring_mono2_strategy()) {
        let ab = a.mul(&b, &r);
        let ba = b.mul(&a, &r);
        prop_assert_eq!(ab, ba);
    }

    #[test]
    fn a_divides_ab((r, a, b) in ring_mono2_strategy()) {
        let ab = a.mul(&b, &r);
        prop_assert!(a.divides(&ab, &r));
        prop_assert!(b.divides(&ab, &r));
    }

    #[test]
    fn div_after_mul_recovers((r, a, b) in ring_mono2_strategy()) {
        let ab = a.mul(&b, &r);
        prop_assert_eq!(ab.div(&b, &r).unwrap(), a);
    }

    #[test]
    fn lcm_commutative((r, a, b) in ring_mono2_strategy()) {
        let ab = a.lcm(&b, &r);
        let ba = b.lcm(&a, &r);
        prop_assert_eq!(ab, ba);
    }

    #[test]
    fn lcm_absorbs_both((r, a, b) in ring_mono2_strategy()) {
        let l = a.lcm(&b, &r);
        prop_assert!(a.divides(&l, &r));
        prop_assert!(b.divides(&l, &r));
    }

    #[test]
    fn total_deg_is_sum((r, a, b) in ring_mono2_strategy()) {
        // ADR-020: total_deg is the byte cap (saturates at 255).
        // The sum rule holds exactly when the result doesn't
        // saturate; above 255 the product's cap reads 255.
        let ab = a.mul(&b, &r);
        let raw_sum = a.total_deg() + b.total_deg();
        let expected = raw_sum.min(255);
        prop_assert_eq!(ab.total_deg(), expected);
    }

    #[test]
    fn sev_of_product_is_or((r, a, b) in ring_mono2_strategy()) {
        let ab = a.mul(&b, &r);
        prop_assert_eq!(ab.compute_sev(&r), a.compute_sev(&r) | b.compute_sev(&r));
    }

    /// sev pre-filter soundness: `a | b` implies every bit set in `sev(a)`
    /// is also set in `sev(b)`.  The sweep relies on the contrapositive
    /// (bits in `sev(a)` not in `sev(b)` => a ∤ b) to reject non-divisors
    /// cheaply; if this ever fails, the sweep is unsound.
    #[test]
    fn sev_prefilter_sound((r, a, b) in ring_mono2_strategy()) {
        if a.divides(&b, &r) {
            prop_assert_eq!(a.compute_sev(&r) & !b.compute_sev(&r), 0,
                "a | b but sev(a) has bits not in sev(b)");
        }
    }

    /// Divmask fast-reject soundness (ADR-025): `a | b` implies every
    /// bit set in `divmask(a)` is also set in `divmask(b)`. The
    /// divisor sweep and chain criterion rely on the contrapositive
    /// `(divmask(a) & ~divmask(b)) != 0 ⇒ a ∤ b` to reject
    /// non-divisors cheaply; if this ever fails, the sweeps are
    /// unsound. This is the property that the divmask layout
    /// `compute_divmask_layout` is engineered to satisfy.
    #[test]
    fn divmask_prefilter_sound((r, a, b) in ring_mono2_strategy()) {
        if a.divides(&b, &r) {
            prop_assert_eq!(r.divmask_of(&a) & !r.divmask_of(&b), 0,
                "a | b but divmask(a) has bits not in divmask(b)");
        }
    }

    #[test]
    fn cmp_is_total((r, a, b) in ring_mono2_strategy()) {
        let ord_ab = a.cmp(&b, &r);
        let ord_ba = b.cmp(&a, &r);
        match (ord_ab, ord_ba) {
            (Ordering::Less, Ordering::Greater)
            | (Ordering::Greater, Ordering::Less)
            | (Ordering::Equal, Ordering::Equal) => {}
            other => prop_assert!(false, "cmp not antisymmetric: {:?}", other),
        }
    }

    #[test]
    fn cmp_equal_iff_same_exponents((r, a, b) in ring_mono2_strategy()) {
        let equal_exps = a.exponents(&r) == b.exponents(&r);
        prop_assert_eq!(a.cmp(&b, &r) == Ordering::Equal, equal_exps);
    }

    #[test]
    fn round_trip_exponents((r, m) in ring_strategy().prop_flat_map(mono_strategy)) {
        let es = m.exponents(&r);
        let m2 = Monomial::from_exponents(&r, &es).unwrap();
        prop_assert_eq!(m, m2);
    }

    #[test]
    fn assert_canonical_after_ops((r, a, b) in ring_mono2_strategy()) {
        a.assert_canonical(&r);
        b.assert_canonical(&r);
        let ab = a.mul(&b, &r);
        ab.assert_canonical(&r);
        let l = a.lcm(&b, &r);
        l.assert_canonical(&r);
        if a.divides(&b, &r) {
            let q = b.div(&a, &r).unwrap();
            q.assert_canonical(&r);
        }
    }
}

// Divmask invariant proptest at higher case count (4096) per ADR-025
// task spec. The bit-layout is the load-bearing piece; we want a
// thorough random-coverage sample.
proptest! {
    #![proptest_config(ProptestConfig::with_cases(4096))]

    /// ADR-025 divmask invariant, restated for emphasis at 4×
    /// case count. Generates random `(a, b)` pairs, conditions on
    /// `a.divides(&b, ring)`, and asserts the bit-subset relation
    /// on the divmasks.
    #[test]
    fn divmask_invariant_high_volume((r, a, b) in ring_mono2_strategy()) {
        if a.divides(&b, &r) {
            prop_assert_eq!(r.divmask_of(&a) & !r.divmask_of(&b), 0,
                "a | b but divmask(a) has bits not in divmask(b); \
                 a={:?} b={:?} mask_a={:#x} mask_b={:#x}",
                a.exponents(&r), b.exponents(&r),
                r.divmask_of(&a), r.divmask_of(&b));
        }
    }

    /// Pairwise consistency: `divmask_of` over the multiplication of
    /// two monomials must be a *superset* of each operand's mask.
    /// This is implied by the divmask invariant (since `a` divides
    /// `a*b` and `b` divides `a*b`), but worth checking directly.
    #[test]
    fn divmask_of_product_supersets_operands((r, a, b) in ring_mono2_strategy()) {
        let ab = a.mul(&b, &r);
        let ma = r.divmask_of(&a);
        let mb = r.divmask_of(&b);
        let mab = r.divmask_of(&ab);
        prop_assert_eq!(ma & !mab, 0,
            "mask(a) has bit not in mask(a*b)");
        prop_assert_eq!(mb & !mab, 0,
            "mask(b) has bit not in mask(a*b)");
    }
}

// ADR-035 / ADR-036 identities. These are exact (not fast-reject
// approximations): if any of these falsifies, the corresponding lever's
// premise is wrong and the feature must NOT ship. The packing-regime
// sweep (nvars ∈ {5, 25, 31}) covers the divmask budget extremes.
proptest! {
    #![proptest_config(ProptestConfig::with_cases(4096))]

    /// ADR-035: `sev(lcm(a,b)) == sev(a) | sev(b)` exactly. Both schemes
    /// are threshold-monotone and lcm is the componentwise max, so the
    /// OR composition is exact, not just a sound over-approximation.
    #[test]
    fn lcm_sev_is_or_of_operands((r, a, b) in regime_mono2_strategy()) {
        let l = a.lcm(&b, &r);
        prop_assert_eq!(
            l.compute_sev(&r),
            a.compute_sev(&r) | b.compute_sev(&r),
            "sev(lcm) != sev(a) | sev(b); a={:?} b={:?}",
            a.exponents(&r), b.exponents(&r)
        );
    }

    /// ADR-035: `divmask(lcm(a,b)) == divmask(a) | divmask(b)` exactly.
    /// This is the load-bearing identity for finding A — if it fails,
    /// the divmask scheme is not threshold-monotone and the profile
    /// report's finding A is wrong.
    #[test]
    fn lcm_divmask_is_or_of_operands((r, a, b) in regime_mono2_strategy()) {
        let l = a.lcm(&b, &r);
        prop_assert_eq!(
            r.divmask_of(&l),
            r.divmask_of(&a) | r.divmask_of(&b),
            "divmask(lcm) != divmask(a) | divmask(b); a={:?} b={:?}",
            a.exponents(&r), b.exponents(&r)
        );
    }

}

// =====================================================================
// ADR-038: byte-parallel SWAR/SIMD monomial kernels (lcm, divides,
// coprime). Each kernel is a provably-exact drop-in for its scalar
// predecessor; these proptests assert SIMD == scalar-reference over
// the full packing-regime sweep, with explicit coverage of the
// max-exponent (127) edge and the degree-cap (>255) boundary.
// =====================================================================

/// Two monomials whose per-variable exponents span the *full* 0..=127
/// range — exercising the max-exponent edge and (for small nvars) the
/// total-degree cap-saturation boundary (>255) that the lcm degree-byte
/// recompute must get exactly right. `nvars ∈ {5, 25, 31}`.
fn regime_mono2_full_range() -> impl Strategy<Value = (Ring, Monomial, Monomial)> {
    prop::sample::select(vec![5u32, 25u32, 31u32])
        .prop_flat_map(|nvars| {
            let f = Field::new(32003).unwrap();
            let r = Ring::new(nvars, MonoOrder::DegRevLex, f).unwrap();
            let n = nvars as usize;
            (
                Just(r),
                prop::collection::vec(0u32..=127, n),
                prop::collection::vec(0u32..=127, n),
            )
        })
        .prop_map(|(r, ae, be)| {
            let a = Monomial::from_exponents(&r, &ae).unwrap();
            let b = Monomial::from_exponents(&r, &be).unwrap();
            (r, a, b)
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8192))]

    /// `lcm_simd == lcm_scalar` over the full 0..=127 range, including
    /// the cases where the lcm's total degree saturates the 8-bit cap
    /// (>255). The scalar reference round-trips through `from_exponents`
    /// (the pre-ADR-038 path); equality of the resulting `Monomial`
    /// (full `PartialEq` over packed words + component) means the SIMD
    /// path reproduces the canonical block byte-for-byte, including the
    /// recomputed degree cap.
    #[test]
    fn lcm_simd_equals_scalar((r, a, b) in regime_mono2_full_range()) {
        let simd = a.lcm(&b, &r);
        let scalar = a.lcm_scalar(&b, &r);
        prop_assert_eq!(&simd, &scalar,
            "lcm mismatch: a={:?} b={:?} simd={:?} scalar={:?}",
            a.exponents(&r), b.exponents(&r),
            simd.exponents(&r), scalar.exponents(&r));
        // Degree cap explicitly: capped sum of the componentwise max.
        let want_cap: u32 = (0..r.nvars())
            .map(|i| a.exponent(&r, i).unwrap().max(b.exponent(&r, i).unwrap()))
            .sum::<u32>()
            .min(255);
        prop_assert_eq!(simd.total_deg(), want_cap, "lcm degree-cap mismatch");
    }

    /// `divides_simd == componentwise_le`. The scalar reference is the
    /// per-variable byte `≤` loop (`divides_scalar`); the SIMD path uses
    /// saturating-subtract. Full 0..=127 range so the degree-cap byte is
    /// frequently nonzero in both operands (confirming the kernel does
    /// not let it false-negative — it is masked out).
    #[test]
    fn divides_simd_equals_componentwise_le((r, a, b) in regime_mono2_full_range()) {
        prop_assert_eq!(a.divides(&b, &r), a.divides_scalar(&b, &r),
            "divides mismatch a|b: a={:?} b={:?}", a.exponents(&r), b.exponents(&r));
        prop_assert_eq!(b.divides(&a, &r), b.divides_scalar(&a, &r),
            "divides mismatch b|a: a={:?} b={:?}", a.exponents(&r), b.exponents(&r));
        // Reflexive: a | a always.
        prop_assert!(a.divides(&a, &r));
        // a | lcm(a,b) and b | lcm(a,b) — the lcm absorbs both even at
        // the 127 edge.
        let l = a.lcm(&b, &r);
        prop_assert!(a.divides(&l, &r));
        prop_assert!(b.divides(&l, &r));
    }

    /// `coprime_simd == coprime_scalar`. Degree bytes are nonzero in
    /// both operands across this range, so the test confirms the kernel
    /// masks them out (a naive componentwise-min that included byte 31
    /// would always report not-coprime).
    #[test]
    fn coprime_simd_equals_scalar((r, a, b) in regime_mono2_full_range()) {
        prop_assert_eq!(
            rustgb::gm::monomials_are_coprime(&a, &b, &r),
            rustgb::gm::monomials_are_coprime_scalar(&a, &b, &r),
            "coprime mismatch: a={:?} b={:?}", a.exponents(&r), b.exponents(&r));
    }
}

/// Hand-picked degree-cap boundary cases for `lcm` (deterministic, not
/// proptest): the recomputed cap must saturate at exactly 255.
#[test]
fn lcm_degree_cap_boundary() {
    let f = Field::new(32003).unwrap();
    // nvars = 3, all at 127: max(127,..)=127 each; sum 381 → cap 255.
    let r = Ring::new(3, MonoOrder::DegRevLex, f).unwrap();
    let a = Monomial::from_exponents(&r, &[127, 0, 127]).unwrap();
    let b = Monomial::from_exponents(&r, &[0, 127, 127]).unwrap();
    let l = a.lcm(&b, &r);
    assert_eq!(l.exponents(&r), vec![127, 127, 127]);
    assert_eq!(l.total_deg(), 255, "381 must saturate to 255");
    l.assert_canonical(&r);

    // Exactly-255 boundary: 85 * 3 = 255, no saturation.
    let a = Monomial::from_exponents(&r, &[85, 85, 0]).unwrap();
    let b = Monomial::from_exponents(&r, &[0, 0, 85]).unwrap();
    let l = a.lcm(&b, &r);
    assert_eq!(l.total_deg(), 255);
    l.assert_canonical(&r);

    // Just under: 85 + 85 + 84 = 254.
    let a = Monomial::from_exponents(&r, &[85, 85, 0]).unwrap();
    let b = Monomial::from_exponents(&r, &[0, 0, 84]).unwrap();
    let l = a.lcm(&b, &r);
    assert_eq!(l.total_deg(), 254);
    l.assert_canonical(&r);
}

/// `divides` must use only the variable bytes, not the degree cap:
/// construct a | b at the variable level and confirm divides holds even
/// though the degree bytes differ.
#[test]
fn divides_ignores_degree_byte() {
    let f = Field::new(32003).unwrap();
    let r = Ring::new(4, MonoOrder::DegRevLex, f).unwrap();
    // a = (1,1,1,1) deg 4; b = (2,2,2,2) deg 8. a | b; degree bytes
    // differ (4 vs 8) but divides looks only at variable bytes.
    let a = Monomial::from_exponents(&r, &[1, 1, 1, 1]).unwrap();
    let b = Monomial::from_exponents(&r, &[2, 2, 2, 2]).unwrap();
    assert!(a.divides(&b, &r));
    assert!(!b.divides(&a, &r));
}
