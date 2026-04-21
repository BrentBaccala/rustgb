//! Long-running cyclic-5 driver, suitable for `perf record`.
//!
//! Runs cyclic-5 over F_32003 N times (default 200) under whatever
//! RUSTGB_THREADS value the environment carries. Prints the mean and
//! stdev of per-iteration wall time.

use std::sync::Arc;
use std::time::Instant;

use rustgb::compute_gb;
use rustgb::field::{Coeff, Field};
use rustgb::monomial::Monomial;
use rustgb::ordering::MonoOrder;
use rustgb::poly::Poly;
use rustgb::ring::Ring;

fn mk_ring(nvars: u32, p: u32) -> Arc<Ring> {
    Arc::new(Ring::new(nvars, MonoOrder::DegRevLex, Field::new(p).unwrap()).unwrap())
}

fn mono(r: &Ring, e: &[u32]) -> Monomial {
    Monomial::from_exponents(r, e).unwrap()
}

fn cyclic5_input(ring: &Arc<Ring>) -> Vec<Poly> {
    let m = |e: &[u32]| mono(ring, e);
    let p_minus_one: Coeff = 32002;
    vec![
        Poly::from_terms(
            ring,
            vec![
                (1, m(&[1, 0, 0, 0, 0])),
                (1, m(&[0, 1, 0, 0, 0])),
                (1, m(&[0, 0, 1, 0, 0])),
                (1, m(&[0, 0, 0, 1, 0])),
                (1, m(&[0, 0, 0, 0, 1])),
            ],
        ),
        Poly::from_terms(
            ring,
            vec![
                (1, m(&[1, 1, 0, 0, 0])),
                (1, m(&[0, 1, 1, 0, 0])),
                (1, m(&[0, 0, 1, 1, 0])),
                (1, m(&[0, 0, 0, 1, 1])),
                (1, m(&[1, 0, 0, 0, 1])),
            ],
        ),
        Poly::from_terms(
            ring,
            vec![
                (1, m(&[1, 1, 1, 0, 0])),
                (1, m(&[0, 1, 1, 1, 0])),
                (1, m(&[0, 0, 1, 1, 1])),
                (1, m(&[1, 0, 0, 1, 1])),
                (1, m(&[1, 1, 0, 0, 1])),
            ],
        ),
        Poly::from_terms(
            ring,
            vec![
                (1, m(&[1, 1, 1, 1, 0])),
                (1, m(&[0, 1, 1, 1, 1])),
                (1, m(&[1, 0, 1, 1, 1])),
                (1, m(&[1, 1, 0, 1, 1])),
                (1, m(&[1, 1, 1, 0, 1])),
            ],
        ),
        Poly::from_terms(
            ring,
            vec![(1, m(&[1, 1, 1, 1, 1])), (p_minus_one, m(&[0, 0, 0, 0, 0]))],
        ),
    ]
}

fn main() {
    let n_iter: usize = std::env::var("PERF_ITER")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let r = mk_ring(5, 32003);
    // Warm-up
    let _ = compute_gb(Arc::clone(&r), cyclic5_input(&r));

    let mut times_us: Vec<u64> = Vec::with_capacity(n_iter);
    for _ in 0..n_iter {
        let input = cyclic5_input(&r);
        let t = Instant::now();
        let gb = compute_gb(Arc::clone(&r), input);
        let elapsed = t.elapsed();
        std::hint::black_box(gb);
        times_us.push(elapsed.as_micros() as u64);
    }
    let sum: u64 = times_us.iter().sum();
    let n = times_us.len() as f64;
    let mean = sum as f64 / n;
    let var = times_us
        .iter()
        .map(|&t| {
            let d = t as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    let stdev = var.sqrt();
    // sorted for median/min
    let mut sorted = times_us.clone();
    sorted.sort_unstable();
    let median = sorted[sorted.len() / 2];
    let min = sorted[0];
    println!(
        "cyclic-5 (N={}): mean={:.1}us stdev={:.1}us median={}us min={}us",
        n_iter, mean, stdev, median, min
    );
}
