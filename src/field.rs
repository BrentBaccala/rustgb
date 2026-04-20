//! Finite field Z/pZ with p prime, p < 2^31.
//!
//! Modular multiplication uses Barrett reduction: we precompute
//! `barrett_mu = floor(2^62 / p)`, then for a 62-bit product `x` the
//! quotient estimate `q = (x * barrett_mu) >> 62` is within 1 of
//! `floor(x / p)`. Two subtractions produce the reduced result without
//! a division instruction.
//!
//! Reference implementations consulted (none copied):
//! feanor-math `zn_64`
//! (<https://github.com/FeanorTheElf/feanor-math/blob/master/src/rings/zn/zn_64.rs>),
//! mathicgb `PrimeField.hpp` in `~/mathicgb/src/mathicgb/`.
//!
//! `Field` is `Send + Sync`: it holds only immutable data.

/// Coefficient type. `u32` is enough for primes `p < 2^31`. Elements
/// are always in the canonical range `0 ≤ c < p`.
pub type Coeff = u32;

/// A finite field Z/pZ.
#[derive(Clone, Copy, Debug)]
pub struct Field {
    /// The prime characteristic. Must satisfy `2 ≤ p < 2^31`.
    p: u32,
    /// Barrett reduction multiplier: `floor(2^62 / p)`. Used to compute
    /// `a * b mod p` without a division instruction on the hot path.
    barrett_mu: u64,
}

impl Field {
    /// Construct a field Z/pZ. Returns `None` unless `2 ≤ p < 2^31`.
    ///
    /// This constructor does *not* verify primality — that's the
    /// caller's responsibility. Non-prime moduli give a ring, not a
    /// field, and [`Field::inv`] will produce garbage.
    pub fn new(p: u32) -> Option<Self> {
        if !(2..(1u32 << 31)).contains(&p) {
            return None;
        }
        // Barrett: mu = floor(2^62 / p).
        let mu = ((1u128 << 62) / p as u128) as u64;
        Some(Self { p, barrett_mu: mu })
    }

    /// The prime characteristic.
    #[inline]
    pub fn p(&self) -> u32 {
        self.p
    }

    /// The zero element.
    #[inline]
    pub fn zero(&self) -> Coeff {
        0
    }

    /// The one element. (`p >= 2` is enforced at construction.)
    #[inline]
    pub fn one(&self) -> Coeff {
        1
    }

    /// Equality on already-reduced elements. Provided for symmetry with
    /// `zero` / `one`; callers can equally well use `==`.
    #[inline]
    pub fn eq(&self, a: Coeff, b: Coeff) -> bool {
        a == b
    }

    /// Modular reduction of an unsigned integer that may exceed `p`.
    /// Used only for canonicalising user-supplied values; the fast
    /// arithmetic paths below avoid generic reduction.
    #[inline]
    pub fn reduce(&self, x: u64) -> Coeff {
        (x % self.p as u64) as u32
    }

    /// Addition in Z/pZ. Inputs must already be reduced (`< p`).
    #[inline]
    pub fn add(&self, a: Coeff, b: Coeff) -> Coeff {
        debug_assert!(a < self.p && b < self.p);
        let s = a as u64 + b as u64;
        if s >= self.p as u64 {
            (s - self.p as u64) as u32
        } else {
            s as u32
        }
    }

    /// Subtraction in Z/pZ. Inputs must already be reduced.
    #[inline]
    pub fn sub(&self, a: Coeff, b: Coeff) -> Coeff {
        debug_assert!(a < self.p && b < self.p);
        if a >= b { a - b } else { a + self.p - b }
    }

    /// Negation in Z/pZ. Input must be reduced.
    #[inline]
    pub fn neg(&self, a: Coeff) -> Coeff {
        debug_assert!(a < self.p);
        if a == 0 { 0 } else { self.p - a }
    }

    /// Multiplication in Z/pZ via Barrett reduction. Inputs must be
    /// reduced (`< p < 2^31`), so the product fits in `u64`.
    #[inline]
    pub fn mul(&self, a: Coeff, b: Coeff) -> Coeff {
        debug_assert!(a < self.p && b < self.p);
        // prod < 2^62 since a, b < 2^31.
        let prod = (a as u64) * (b as u64);
        // q = floor(prod / p) or one less.
        let q = ((prod as u128 * self.barrett_mu as u128) >> 62) as u64;
        let r = prod.wrapping_sub(q.wrapping_mul(self.p as u64));
        // At most one correction step because barrett_mu under-
        // approximates 2^62 / p by strictly less than 1.
        let r = if r >= self.p as u64 {
            r - self.p as u64
        } else {
            r
        };
        debug_assert!(r < self.p as u64);
        r as u32
    }

    /// Multiplicative inverse via Fermat's little theorem: `a^(p-2) mod p`.
    /// Returns `None` when `a == 0`. Correct only when `p` is prime.
    pub fn inv(&self, a: Coeff) -> Option<Coeff> {
        debug_assert!(a < self.p);
        if a == 0 {
            return None;
        }
        Some(self.pow(a, (self.p - 2) as u64))
    }

    /// Modular exponentiation by repeated squaring. Exposed primarily
    /// so [`Field::inv`] can use it; handy in tests too.
    pub fn pow(&self, mut base: Coeff, mut exp: u64) -> Coeff {
        let mut acc: Coeff = 1;
        while exp > 0 {
            if exp & 1 == 1 {
                acc = self.mul(acc, base);
            }
            exp >>= 1;
            if exp > 0 {
                base = self.mul(base, base);
            }
        }
        acc
    }
}

impl PartialEq for Field {
    fn eq(&self, other: &Self) -> bool {
        self.p == other.p
    }
}
impl Eq for Field {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Barrett multiplication must match the naive reduction.
    #[test]
    fn barrett_matches_naive_sweep() {
        // A spread of small, medium, and large primes under 2^31.
        for &p in &[
            2u32,
            3,
            5,
            7,
            11,
            13,
            101,
            32003,
            100_003,
            1_000_003,
            2_147_483_647,
        ] {
            let f = Field::new(p).unwrap();
            // A deterministic but "spread-out" generator.
            let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
            for _ in 0..10_000 {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let a = ((x >> 32) as u32) % p;
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let b = ((x >> 32) as u32) % p;
                let got = f.mul(a, b);
                let want = ((a as u64 * b as u64) % p as u64) as u32;
                assert_eq!(got, want, "p={p} a={a} b={b}");
            }
        }
    }

    #[test]
    fn zero_and_one() {
        let f = Field::new(13).unwrap();
        assert_eq!(f.zero(), 0);
        assert_eq!(f.one(), 1);
        for a in 0..13 {
            assert_eq!(f.add(a, 0), a);
            assert_eq!(f.mul(a, 1), a);
            assert_eq!(f.mul(a, 0), 0);
            assert_eq!(f.sub(a, a), 0);
        }
    }

    #[test]
    fn inverse_matches_naive() {
        let f = Field::new(32003).unwrap();
        for a in 1..200 {
            let inv = f.inv(a).unwrap();
            assert_eq!(f.mul(a, inv), 1);
        }
        assert!(f.inv(0).is_none());
    }

    #[test]
    fn rejects_out_of_range_p() {
        assert!(Field::new(0).is_none());
        assert!(Field::new(1).is_none());
        assert!(Field::new(1u32 << 31).is_none());
        assert!(Field::new(u32::MAX).is_none());
        assert!(Field::new((1u32 << 31) - 1).is_some());
    }
}
