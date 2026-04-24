//! Property-based tests for polynomials.
//!
//! Many of these are direct adaptations of FLINT's `nmod_mpoly`
//! tests (see `~/flint/src/nmod_mpoly/test/`), translated into
//! Rust `proptest` style.

use proptest::prelude::*;
use rustgb::{Coeff, Field, MonoOrder, Monomial, Poly, Ring};

/// A small prime keeps the tests fast while catching arithmetic bugs.
const DEFAULT_PRIME: u32 = 32003;

fn ring_strategy(max_nvars: u32) -> impl Strategy<Value = Ring> {
    (1u32..=max_nvars).prop_map(|n| {
        Ring::new(n, MonoOrder::DegRevLex, Field::new(DEFAULT_PRIME).unwrap()).unwrap()
    })
}

fn poly_strategy(
    ring: Ring,
    max_terms: usize,
    max_exp: u32,
) -> impl Strategy<Value = (Ring, Poly)> {
    let n = ring.nvars() as usize;
    let p = ring.field().p();
    prop::collection::vec(
        (1u32..p, prop::collection::vec(0u32..max_exp, n)),
        0..=max_terms,
    )
    .prop_map(move |terms| {
        let converted: Vec<(Coeff, Monomial)> = terms
            .into_iter()
            .map(|(c, e)| (c, Monomial::from_exponents(&ring, &e).unwrap()))
            .collect();
        let p = Poly::from_terms(&ring, converted);
        (ring.clone(), p)
    })
}

/// Ring + three polys. Small caps so products stay within the 8-bit
/// exponent budget (we sum exponents of up to two operands, so
/// max_exp * 2 ≤ 255 → max_exp ≤ 127).
fn ring_poly3_strategy() -> impl Strategy<Value = (Ring, Poly, Poly, Poly)> {
    ring_strategy(5).prop_flat_map(|r| {
        (
            Just(r.clone()),
            poly_strategy(r.clone(), 6, 8),
            poly_strategy(r.clone(), 6, 8),
            poly_strategy(r, 6, 8),
        )
            .prop_map(|(r, (_, f), (_, g), (_, h))| (r, f, g, h))
    })
}

fn ring_poly2_strategy() -> impl Strategy<Value = (Ring, Poly, Poly)> {
    ring_strategy(5).prop_flat_map(|r| {
        (
            Just(r.clone()),
            poly_strategy(r.clone(), 8, 10),
            poly_strategy(r, 8, 10),
        )
            .prop_map(|(r, (_, f), (_, g))| (r, f, g))
    })
}

fn ring_poly1_strategy() -> impl Strategy<Value = (Ring, Poly)> {
    ring_strategy(5).prop_flat_map(|r| poly_strategy(r, 10, 15))
}

fn ring_poly_term_strategy() -> impl Strategy<Value = (Ring, Poly, Poly, Coeff, Monomial)> {
    ring_strategy(4).prop_flat_map(|r| {
        let n = r.nvars() as usize;
        let p = r.field().p();
        (
            Just(r.clone()),
            poly_strategy(r.clone(), 6, 8),
            poly_strategy(r.clone(), 6, 8),
            1u32..p,
            prop::collection::vec(0u32..20, n),
        )
            .prop_map(move |(r, (_, p1), (_, p2), c, exps)| {
                let m = Monomial::from_exponents(&r, &exps).unwrap();
                (r, p1, p2, c, m)
            })
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1024))]

    #[test]
    fn add_associative((r, f, g, h) in ring_poly3_strategy()) {
        let lhs = f.add(&g, &r).add(&h, &r);
        let rhs = f.add(&g.add(&h, &r), &r);
        prop_assert_eq!(lhs, rhs);
    }

    #[test]
    fn add_commutative((r, f, g) in ring_poly2_strategy()) {
        let fg = f.add(&g, &r);
        let gf = g.add(&f, &r);
        prop_assert_eq!(fg, gf);
    }

    #[test]
    fn add_zero((r, f) in ring_poly1_strategy()) {
        let zero = Poly::zero();
        prop_assert_eq!(f.add(&zero, &r), f.clone());
    }

    #[test]
    fn sub_self_zero((r, f) in ring_poly1_strategy()) {
        let z = f.sub(&f, &r);
        prop_assert!(z.is_zero());
    }

    #[test]
    fn sub_then_add_round_trip((r, f, g) in ring_poly2_strategy()) {
        let rebuilt = f.sub(&g, &r).add(&g, &r);
        prop_assert_eq!(rebuilt, f);
    }

    #[test]
    fn mul_zero_is_zero((r, f) in ring_poly1_strategy()) {
        let z = Poly::zero();
        prop_assert!(f.mul(&z, &r).is_zero());
        prop_assert!(z.mul(&f, &r).is_zero());
    }

    #[test]
    fn mul_one_is_identity((r, f) in ring_poly1_strategy()) {
        let one_mono = Monomial::one(&r);
        let one = Poly::monomial(&r, 1, one_mono);
        prop_assert_eq!(f.mul(&one, &r), f.clone());
    }

    #[test]
    fn mul_distributes_over_add((r, f, g, h) in ring_poly3_strategy()) {
        // ADR-018: Poly::mul is infallible; ring_poly3_strategy's
        // per-var cap (8) keeps all product exponents ≤ 16 < 127.
        let lhs = f.mul(&g.add(&h, &r), &r);
        let rhs_g = f.mul(&g, &r);
        let rhs_h = f.mul(&h, &r);
        let rhs = rhs_g.add(&rhs_h, &r);
        prop_assert_eq!(lhs, rhs);
    }

    #[test]
    fn aliasing_add_assign_matches_add((r, f, g) in ring_poly2_strategy()) {
        let out_of_place = f.add(&g, &r);
        let mut in_place = f.clone();
        in_place.add_assign(&g, &r);
        prop_assert_eq!(in_place, out_of_place);
    }

    #[test]
    fn assert_canonical_after_ops((r, f, g) in ring_poly2_strategy()) {
        f.assert_canonical(&r);
        g.assert_canonical(&r);
        let s = f.add(&g, &r);
        s.assert_canonical(&r);
        let d = f.sub(&g, &r);
        d.assert_canonical(&r);
        let p = f.mul(&g, &r);
        p.assert_canonical(&r);
        if let Some(mf) = f.monic(&r) {
            mf.assert_canonical(&r);
        }
    }

    #[test]
    fn leading_of_product_is_product_of_leadings((r, f, g) in ring_poly2_strategy()) {
        if f.is_zero() || g.is_zero() { return Ok(()); }
        // ADR-018: Poly::mul and Monomial::mul are infallible in
        // release; ring_poly2_strategy's per-var cap (10) keeps sums
        // well within the 7-bit budget.
        let prod = f.mul(&g, &r);
        let (fc, fm) = f.leading().unwrap();
        let (gc, gm) = g.leading().unwrap();
        let expected_m = fm.mul(gm, &r);
        let expected_c = r.field().mul(fc, gc);
        if prod.is_zero() {
            // Only possible if expected_c == 0, which needs a zero
            // divisor — impossible in Z/p.
            prop_assert!(expected_c == 0);
        } else {
            let (pc, pm) = prod.leading().unwrap();
            prop_assert_eq!(pc, expected_c);
            prop_assert_eq!(pm.clone(), expected_m);
        }
    }

    #[test]
    fn sub_mul_term_matches_slow_path((r, p, q, c, m) in ring_poly_term_strategy()) {
        // ADR-018: release-build sub_mul_term is infallible. Strategy
        // parameters are bounded (ring_poly_term_strategy keeps
        // per-var exponents well below the 7-bit cap) so no overflow
        // can arise.
        let fast = p.sub_mul_term(c, &m, &q, &r);
        let scaled_m = q.shift(&m, &r);
        let slow = p.sub(&scaled_m.scale(c, &r), &r);
        fast.assert_canonical(&r);
        slow.assert_canonical(&r);
        prop_assert_eq!(fast, slow);
    }

    #[test]
    fn monic_idempotent((r, f) in ring_poly1_strategy()) {
        if f.is_zero() { return Ok(()); }
        let once = f.monic(&r).unwrap();
        let twice = once.monic(&r).unwrap();
        prop_assert_eq!(once.lm_coeff(), 1);
        prop_assert_eq!(once, twice);
    }
}

// --- Fixed-input fixtures drawn from the Groebner basis literature --

/// The ideal of the cyclic-3 system over Z/32003.
#[test]
fn cyclic3_polynomials_are_canonical() {
    let r = Ring::new(3, MonoOrder::DegRevLex, Field::new(32003).unwrap()).unwrap();
    let mono = |e: &[u32]| Monomial::from_exponents(&r, e).unwrap();
    // f1 = x + y + z
    let f1 = Poly::from_terms(
        &r,
        vec![
            (1, mono(&[1, 0, 0])),
            (1, mono(&[0, 1, 0])),
            (1, mono(&[0, 0, 1])),
        ],
    );
    // f2 = x*y + y*z + x*z
    let f2 = Poly::from_terms(
        &r,
        vec![
            (1, mono(&[1, 1, 0])),
            (1, mono(&[0, 1, 1])),
            (1, mono(&[1, 0, 1])),
        ],
    );
    // f3 = x*y*z - 1
    let f3 = Poly::from_terms(&r, vec![(1, mono(&[1, 1, 1])), (32002, mono(&[0, 0, 0]))]);
    f1.assert_canonical(&r);
    f2.assert_canonical(&r);
    f3.assert_canonical(&r);
    // Sanity: f1 has 3 terms, all degree 1.
    assert_eq!(f1.len(), 3);
    assert_eq!(f1.lm_deg(), 1);
}

/// A small ideal over several small moduli. Just builds and canonicalises.
#[test]
fn small_moduli_fixture() {
    for &p in &[2u32, 3, 5, 32003] {
        let r = Ring::new(1, MonoOrder::DegRevLex, Field::new(p).unwrap()).unwrap();
        let one_mono = Monomial::one(&r);
        let x = Monomial::from_exponents(&r, &[1]).unwrap();
        // f = x - 1
        let f = Poly::from_terms(&r, vec![(1, x), (p - 1, one_mono)]);
        f.assert_canonical(&r);
        assert_eq!(f.len(), 2);
    }
}

#[test]
fn single_term_edge_cases() {
    let r = Ring::new(3, MonoOrder::DegRevLex, Field::new(7).unwrap()).unwrap();
    // The multiplicative identity polynomial.
    let one_poly = Poly::monomial(&r, 1, Monomial::one(&r));
    one_poly.assert_canonical(&r);
    assert_eq!(one_poly.len(), 1);
    assert_eq!(one_poly.lm_deg(), 0);
    // The zero polynomial.
    let z = Poly::zero();
    z.assert_canonical(&r);
    assert!(z.is_zero());
    // Single variable.
    let x = Poly::monomial(&r, 3, Monomial::from_exponents(&r, &[1, 0, 0]).unwrap());
    x.assert_canonical(&r);
    assert_eq!(x.lm_coeff(), 3);
}

#[test]
fn drop_100k_term_poly_does_not_overflow_stack() {
    // Regression guard on the linked-list backend's iterative Drop
    // impl (ADR-014). A naive recursive drop on a 100 000-term chain
    // would blow the default 8 MB thread stack; the Vec backend's
    // destructor is a simple loop and doesn't care.
    //
    // Either way we want to confirm the public `Poly` contract holds
    // at this scale on whichever backend cargo selected for this
    // test run. Scaling to 100 K terms costs ~10 ms on the Vec
    // backend and ~40 ms on the linked-list backend — cheap enough
    // to keep in the default suite.
    let r = Ring::new(4, MonoOrder::DegRevLex, Field::new(32003).unwrap()).unwrap();
    let n: usize = 100_000;

    // Generate N distinct monomials by sweeping exponents in base-64
    // across four variables; sort descending so
    // `from_descending_terms_unchecked`'s contract holds.
    let mut distinct: Vec<Monomial> = Vec::with_capacity(n);
    'outer: for d in 0u32..64 {
        for c in 0u32..64 {
            for b in 0u32..64 {
                for a in 0u32..64 {
                    if distinct.len() >= n {
                        break 'outer;
                    }
                    distinct.push(Monomial::from_exponents(&r, &[a, b, c, d]).unwrap());
                }
            }
        }
    }
    distinct.sort_by(|x, y| y.cmp(x, &r));
    let terms: Vec<(Coeff, Monomial)> = distinct.into_iter().map(|m| (1u32, m)).collect();
    let p = Poly::from_descending_terms_unchecked(&r, terms);
    assert_eq!(p.len(), n);
    // Dropping the huge poly at scope exit must not overflow the
    // stack. If this test ever regresses by stack-overflow on the
    // linked-list backend, Drop has gone back to recursive.
    drop(p);
}
