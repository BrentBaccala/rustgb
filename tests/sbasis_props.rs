//! Property tests for [`SBasis`].
//!
//! The headline property is the one called out in the task prompt:
//! for random descending-sev polynomial inserts, the redundancy
//! marking `SBasis` produces must agree with the O(n²) naive "scan
//! every earlier element for LM divisibility by a later element".

use proptest::prelude::*;
use rustgb::{Coeff, Field, MonoOrder, Monomial, Poly, Ring, SBasis};

const P: u32 = 32003;
const MAX_VARS: u32 = 4;
const MAX_EXP: u32 = 4;
const MAX_BASIS: usize = 12;

fn ring_strategy() -> impl Strategy<Value = Ring> {
    (1u32..=MAX_VARS)
        .prop_map(|n| Ring::new(n, MonoOrder::DegRevLex, Field::new(P).unwrap()).unwrap())
}

fn lm_strategy(ring: Ring) -> impl Strategy<Value = Monomial> {
    let n = ring.nvars() as usize;
    prop::collection::vec(0u32..=MAX_EXP, n)
        .prop_map(move |e| Monomial::from_exponents(&ring, &e).unwrap())
}

/// A basis-worth of single-term polynomials (one term = its LM).
/// Single terms are enough to study redundancy — `SBasis::insert`
/// reads only the leading monomial.
fn basis_strategy() -> impl Strategy<Value = (Ring, Vec<Monomial>)> {
    ring_strategy().prop_flat_map(|r| {
        let lms = prop::collection::vec(lm_strategy(r.clone()), 1..=MAX_BASIS);
        lms.prop_map(move |v| (r.clone(), v))
    })
}

/// O(n²) reference: for each element `i`, it is redundant iff some
/// later element `j > i` has `lm(j) | lm(i)`.
fn naive_redundant(ring: &Ring, lms: &[Monomial]) -> Vec<bool> {
    let n = lms.len();
    let mut red = vec![false; n];
    // Replicate the insert-order incremental logic: when element
    // `j` arrives, every earlier non-redundant `i` with `lm(j) | lm(i)`
    // becomes redundant.
    for j in 0..n {
        for i in 0..j {
            if red[i] {
                continue;
            }
            if lms[j].divides(&lms[i], ring) {
                red[i] = true;
            }
        }
    }
    red
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn redundancy_matches_naive((r, lms) in basis_strategy()) {
        let mut s = SBasis::new();
        for m in &lms {
            s.insert(&r, Poly::monomial(&r, 1 as Coeff, m.clone()));
        }
        s.assert_canonical(&r);
        let got: Vec<bool> = (0..lms.len()).map(|i| s.is_redundant(i)).collect();
        let want = naive_redundant(&r, &lms);
        prop_assert_eq!(got, want);
    }

    #[test]
    fn len_counts_all_polys((r, lms) in basis_strategy()) {
        let mut s = SBasis::new();
        for m in &lms {
            s.insert(&r, Poly::monomial(&r, 1 as Coeff, m.clone()));
        }
        // Every insert produces exactly one basis element, regardless
        // of redundancy.
        prop_assert_eq!(s.len(), lms.len());
    }

    #[test]
    fn sevs_and_lm_degs_match_polys((r, lms) in basis_strategy()) {
        let mut s = SBasis::new();
        for m in &lms {
            s.insert(&r, Poly::monomial(&r, 1 as Coeff, m.clone()));
        }
        for (i, m) in lms.iter().enumerate() {
            prop_assert_eq!(s.sevs()[i], m.sev());
            prop_assert_eq!(s.lm_degs()[i], m.total_deg());
        }
    }
}
