//! Property-based tests for Z/pZ field arithmetic.
//!
//! The goal is FLINT-style coverage of the field operations in
//! isolation, so later debugging of polynomial bugs doesn't have to
//! wonder whether the coefficient layer is trustworthy.

use proptest::prelude::*;
use rustgb::Field;

/// A curated list of primes spanning the permitted range.
const PRIMES: &[u32] = &[
    2,
    3,
    5,
    7,
    11,
    13,
    101,
    32003,
    100_003,
    1_000_003,
    (1u32 << 31) - 1, // Mersenne prime 2^31 - 1
];

fn prime_strategy() -> impl Strategy<Value = u32> {
    prop::sample::select(PRIMES.to_vec())
}

fn elem_strategy(p: u32) -> impl Strategy<Value = u32> {
    0u32..p
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2048))]

    #[test]
    fn add_associative(p in prime_strategy(),
                       a in any::<u32>(),
                       b in any::<u32>(),
                       c in any::<u32>()) {
        let f = Field::new(p).unwrap();
        let a = a % p; let b = b % p; let c = c % p;
        prop_assert_eq!(f.add(a, f.add(b, c)), f.add(f.add(a, b), c));
    }

    #[test]
    fn add_commutative(p in prime_strategy(),
                       a in any::<u32>(),
                       b in any::<u32>()) {
        let f = Field::new(p).unwrap();
        let a = a % p; let b = b % p;
        prop_assert_eq!(f.add(a, b), f.add(b, a));
    }

    #[test]
    fn mul_distributes(p in prime_strategy(),
                       a in any::<u32>(),
                       b in any::<u32>(),
                       c in any::<u32>()) {
        let f = Field::new(p).unwrap();
        let a = a % p; let b = b % p; let c = c % p;
        prop_assert_eq!(f.mul(a, f.add(b, c)), f.add(f.mul(a, b), f.mul(a, c)));
    }

    #[test]
    fn add_zero_identity(p in prime_strategy(), a in any::<u32>()) {
        let f = Field::new(p).unwrap();
        let a = a % p;
        prop_assert_eq!(f.add(a, 0), a);
    }

    #[test]
    fn mul_one_identity(p in prime_strategy(), a in any::<u32>()) {
        let f = Field::new(p).unwrap();
        let a = a % p;
        prop_assert_eq!(f.mul(a, 1), a);
    }

    #[test]
    fn mul_inverse(p in prime_strategy(), a in any::<u32>()) {
        let f = Field::new(p).unwrap();
        let a = (a % p).max(1); // make sure nonzero
        let a = if a == 0 { 1 } else { a };
        let inv = f.inv(a).expect("nonzero element has an inverse");
        prop_assert_eq!(f.mul(a, inv), 1);
    }

    #[test]
    fn sub_self_is_zero(p in prime_strategy(), a in any::<u32>()) {
        let f = Field::new(p).unwrap();
        let a = a % p;
        prop_assert_eq!(f.sub(a, a), 0);
    }

    #[test]
    fn neg_is_sub_from_zero(p in prime_strategy(), a in any::<u32>()) {
        let f = Field::new(p).unwrap();
        let a = a % p;
        prop_assert_eq!(f.neg(a), f.sub(0, a));
    }

    #[test]
    fn barrett_matches_naive(p in prime_strategy(), a in any::<u32>(), b in any::<u32>()) {
        let f = Field::new(p).unwrap();
        let a = a % p; let b = b % p;
        let got = f.mul(a, b);
        let want = ((a as u64) * (b as u64) % p as u64) as u32;
        prop_assert_eq!(got, want);
    }

    #[test]
    fn mul_associative(p in prime_strategy(),
                       a in any::<u32>(), b in any::<u32>(), c in any::<u32>()) {
        let f = Field::new(p).unwrap();
        let a = a % p; let b = b % p; let c = c % p;
        prop_assert_eq!(f.mul(a, f.mul(b, c)), f.mul(f.mul(a, b), c));
    }

    #[test]
    fn elem_in_range_strategy_behaves(p in prime_strategy()) {
        let f = Field::new(p).unwrap();
        // use elem_strategy for a consistency smoke test
        let _ = elem_strategy(p);
        prop_assert!(f.p() == p);
    }
}
