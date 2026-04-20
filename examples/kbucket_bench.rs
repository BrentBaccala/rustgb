//! Synthetic bba-shaped benchmark for [`KBucket`] vs. the slow-path
//! `Poly::sub_mul_term` fold.
//!
//! Workload:
//!
//! * Seed: one random ~200-term polynomial `p0`.
//! * Reducers: 200 random ~30-term polynomials `q_i` with random
//!   leading-monomial multipliers `m_i` and scalars `c_i`.
//! * Slow path: fold `Poly::sub_mul_term` over all reducers.
//! * Bucket path: `KBucket::from_poly(p0)` then 200
//!   `minus_m_mult_p` calls, then `into_poly`.
//! * Report both timings and the ratio.
//!
//! Run with:
//!
//! ```sh
//! cargo run --release --example kbucket_bench
//! ```
//!
//! This is a first-pass benchmark; no warm-up, single run. For a
//! rigorous benchmark use `criterion` in a follow-up task.

use std::sync::Arc;
use std::time::Instant;

use rustgb::{Coeff, Field, KBucket, MonoOrder, Monomial, Poly, Ring};

/// Deterministic LCG used to generate all random data.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
}

fn build_ring() -> Arc<Ring> {
    Arc::new(Ring::new(8, MonoOrder::DegRevLex, Field::new(32003).unwrap()).unwrap())
}

fn random_poly(
    ring: &Ring,
    rng: &mut Lcg,
    nterms: usize,
    max_exp: u32,
) -> Poly {
    let n = ring.nvars() as usize;
    let p = ring.field().p();
    let mut pairs = Vec::with_capacity(nterms);
    for _ in 0..nterms {
        let mut exps = vec![0u32; n];
        for slot in exps.iter_mut() {
            *slot = ((rng.next() >> 32) as u32) % (max_exp + 1);
        }
        let c = ((rng.next() >> 32) as u32) % (p - 1) + 1;
        let m = Monomial::from_exponents(ring, &exps).unwrap();
        pairs.push((c, m));
    }
    Poly::from_terms(ring, pairs)
}

fn random_mono(ring: &Ring, rng: &mut Lcg, max_exp: u32) -> Monomial {
    let n = ring.nvars() as usize;
    let mut exps = vec![0u32; n];
    for slot in exps.iter_mut() {
        *slot = ((rng.next() >> 32) as u32) % (max_exp + 1);
    }
    Monomial::from_exponents(ring, &exps).unwrap()
}

fn main() {
    let ring = build_ring();
    let p_field = ring.field().p();

    // Generate the workload.
    let mut rng = Lcg::new(0xDEADBEEF);
    let seed = random_poly(&ring, &mut rng, 200, 3);
    let reducer_count = 200;
    let mut reducers: Vec<(Monomial, Coeff, Poly)> = Vec::with_capacity(reducer_count);
    for _ in 0..reducer_count {
        // Small multipliers so the product fits in the 8-bit budget.
        let m = random_mono(&ring, &mut rng, 2);
        let c = ((rng.next() >> 32) as u32) % (p_field - 1) + 1;
        let q = random_poly(&ring, &mut rng, 30, 3);
        reducers.push((m, c, q));
    }

    // Quick sanity check that both paths succeed before we time them.
    {
        let mut acc = seed.clone();
        let mut overflowed = false;
        for (m, c, q) in &reducers {
            match acc.sub_mul_term(*c, m, q, &ring) {
                Some(p) => acc = p,
                None => {
                    overflowed = true;
                    break;
                }
            }
        }
        if overflowed {
            eprintln!(
                "warning: workload overflows 8-bit budget; adjust max_exp in the benchmark"
            );
            std::process::exit(1);
        }
        println!(
            "seed terms: {}, final slow-path terms: {}",
            seed.len(),
            acc.len()
        );
    }

    // Slow path.
    let t0 = Instant::now();
    let mut slow_acc = seed.clone();
    for (m, c, q) in &reducers {
        slow_acc = slow_acc
            .sub_mul_term(*c, m, q, &ring)
            .expect("no overflow");
    }
    let slow_elapsed = t0.elapsed();
    let slow_terms = slow_acc.len();

    // Bucket path.
    let t0 = Instant::now();
    let mut bucket = KBucket::from_poly(Arc::clone(&ring), seed.clone());
    for (m, c, q) in &reducers {
        bucket.minus_m_mult_p(m, *c, q);
    }
    let bucket_result = bucket.into_poly();
    let bucket_elapsed = t0.elapsed();
    let bucket_terms = bucket_result.len();

    // Correctness check.
    assert_eq!(
        slow_acc, bucket_result,
        "bucket result disagrees with slow-path result"
    );

    // Report.
    println!(
        "slow path (Poly::sub_mul_term fold): {slow_terms} terms, elapsed {:?}",
        slow_elapsed
    );
    println!(
        "bucket path (KBucket::minus_m_mult_p fold): {bucket_terms} terms, elapsed {:?}",
        bucket_elapsed
    );
    let ratio = slow_elapsed.as_secs_f64() / bucket_elapsed.as_secs_f64();
    println!("speedup: {:.2}x", ratio);
}
