//! Rough, non-rigorous timing for `Poly::add` and `Poly::sub_mul_term`.
//!
//! This is *not* a benchmark (no warm-up, no variance reporting). It
//! exists so the bootstrap report can state a rough order-of-magnitude
//! for these two hot paths. For real benchmarks use `criterion` in a
//! follow-up task.
//!
//! Run with `cargo run --release --example sanity`.

use rustgb::{Field, MonoOrder, Monomial, Poly, Ring};
use std::time::Instant;

fn build_ring() -> Ring {
    Ring::new(10, MonoOrder::DegRevLex, Field::new(32003).unwrap()).unwrap()
}

fn random_poly(ring: &Ring, nterms: usize, seed: u64) -> Poly {
    let n = ring.nvars() as usize;
    let mut s = seed;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s
    };
    let mut pairs = Vec::with_capacity(nterms);
    for _ in 0..nterms {
        let mut exps = vec![0u32; n];
        for slot in exps.iter_mut() {
            *slot = ((next() >> 32) as u32) % 6;
        }
        let c = ((next() >> 32) as u32) % (ring.field().p() - 1) + 1;
        let m = Monomial::from_exponents(ring, &exps).unwrap();
        pairs.push((c, m));
    }
    Poly::from_terms(ring, pairs)
}

fn main() {
    let ring = build_ring();
    let f = random_poly(&ring, 200, 0x1234);
    let g = random_poly(&ring, 200, 0xabcd);

    // Poly::add
    let iters = 10_000;
    let t0 = Instant::now();
    let mut acc = f.clone();
    for _ in 0..iters {
        acc = acc.add(&g, &ring);
        if acc.len() > 2000 {
            acc = f.clone();
        }
    }
    let elapsed = t0.elapsed();
    let per = elapsed / iters;
    println!(
        "Poly::add: 200+200 terms, {iters} iters, total {:?}, per-op {:?}",
        elapsed, per
    );

    // sub_mul_term
    let m = Monomial::from_exponents(&ring, &[1, 0, 1, 0, 0, 2, 0, 0, 1, 0]).unwrap();
    let c = 7u32;
    let q = random_poly(&ring, 150, 0xdead);
    let p0 = random_poly(&ring, 300, 0xbeef);
    let t0 = Instant::now();
    let mut acc = p0.clone();
    for _ in 0..iters {
        acc = acc.sub_mul_term(c, &m, &q, &ring).unwrap();
        if acc.len() > 2000 {
            acc = p0.clone();
        }
    }
    let elapsed = t0.elapsed();
    let per = elapsed / iters;
    println!(
        "Poly::sub_mul_term: p=300, q=150 terms, {iters} iters, total {:?}, per-op {:?}",
        elapsed, per
    );
}
