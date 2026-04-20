//! Minimal usage example for `rustgb::compute_gb`.
//!
//! Builds the cyclic-3 ideal in `Z/32003[x, y, z]` under degrevlex,
//! runs [`rustgb::compute_gb`], and prints the reduced Gröbner
//! basis in a Singular-ish textual form.
//!
//! Also prints timings for cyclic-3, cyclic-4, and cyclic-5 so the
//! example doubles as the task's performance-sanity run:
//!
//! ```console
//! $ cargo run --release --example compute_gb
//! ```

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

/// Render one polynomial as a Singular-ish string using the supplied
/// variable names. Good enough for human inspection; not a parser-
/// compatible round-trip (see the fixture parser in the tests for
/// that).
fn poly_to_string(p: &Poly, ring: &Ring, var_names: &[&str]) -> String {
    if p.is_zero() {
        return "0".to_string();
    }
    let field_p = ring.field().p();
    let mut out = String::new();
    for (i, (c, m)) in p.iter().enumerate() {
        let (sign, mag) = if c > field_p / 2 {
            (-1i64, field_p - c)
        } else {
            (1i64, c)
        };
        if i == 0 {
            if sign < 0 {
                out.push('-');
            }
        } else if sign < 0 {
            out.push('-');
        } else {
            out.push('+');
        }
        // Check if monomial is 1 (all exponents zero).
        let exps = m.exponents(ring);
        let is_one = exps.iter().all(|&e| e == 0);
        if mag != 1 || is_one {
            out.push_str(&mag.to_string());
        }
        if !is_one {
            let mut first_var = true;
            for (vi, &e) in exps.iter().enumerate() {
                if e == 0 {
                    continue;
                }
                // Put an explicit `*` before any variable factor
                // that has something preceding it on this term:
                // either the printed coefficient, or a prior var.
                if !first_var || mag != 1 {
                    out.push('*');
                }
                out.push_str(var_names[vi]);
                if e > 1 {
                    out.push('^');
                    out.push_str(&e.to_string());
                }
                first_var = false;
            }
        }
    }
    out
}

fn run_cyclic(ring: Arc<Ring>, name: &str, input: Vec<Poly>, var_names: &[&str]) {
    println!("=== {} ===", name);
    let t = Instant::now();
    let gb = compute_gb(Arc::clone(&ring), input);
    let elapsed = t.elapsed();
    println!("basis size: {}", gb.len());
    println!("time:       {:?}", elapsed);
    for (i, p) in gb.iter().enumerate() {
        println!("  [{:>2}] {}", i, poly_to_string(p, &ring, var_names));
    }
    println!();
}

fn cyclic3() {
    let r = mk_ring(3, 32003);
    let m = |e: &[u32]| mono(&r, e);
    let p_minus_one: Coeff = 32002;
    let gens = vec![
        Poly::from_terms(
            &r,
            vec![(1, m(&[1, 0, 0])), (1, m(&[0, 1, 0])), (1, m(&[0, 0, 1]))],
        ),
        Poly::from_terms(
            &r,
            vec![(1, m(&[1, 1, 0])), (1, m(&[0, 1, 1])), (1, m(&[1, 0, 1]))],
        ),
        Poly::from_terms(&r, vec![(1, m(&[1, 1, 1])), (p_minus_one, m(&[0, 0, 0]))]),
    ];
    run_cyclic(r, "cyclic-3 over F_32003", gens, &["x", "y", "z"]);
}

fn cyclic4() {
    let r = mk_ring(4, 32003);
    let m = |e: &[u32]| mono(&r, e);
    let p_minus_one: Coeff = 32002;
    let gens = vec![
        Poly::from_terms(
            &r,
            vec![
                (1, m(&[1, 0, 0, 0])),
                (1, m(&[0, 1, 0, 0])),
                (1, m(&[0, 0, 1, 0])),
                (1, m(&[0, 0, 0, 1])),
            ],
        ),
        Poly::from_terms(
            &r,
            vec![
                (1, m(&[1, 1, 0, 0])),
                (1, m(&[0, 1, 1, 0])),
                (1, m(&[0, 0, 1, 1])),
                (1, m(&[1, 0, 0, 1])),
            ],
        ),
        Poly::from_terms(
            &r,
            vec![
                (1, m(&[1, 1, 1, 0])),
                (1, m(&[0, 1, 1, 1])),
                (1, m(&[1, 0, 1, 1])),
                (1, m(&[1, 1, 0, 1])),
            ],
        ),
        Poly::from_terms(
            &r,
            vec![(1, m(&[1, 1, 1, 1])), (p_minus_one, m(&[0, 0, 0, 0]))],
        ),
    ];
    run_cyclic(r, "cyclic-4 over F_32003", gens, &["a", "b", "c", "d"]);
}

fn cyclic5() {
    let r = mk_ring(5, 32003);
    let m = |e: &[u32]| mono(&r, e);
    let p_minus_one: Coeff = 32002;
    let gens = vec![
        Poly::from_terms(
            &r,
            vec![
                (1, m(&[1, 0, 0, 0, 0])),
                (1, m(&[0, 1, 0, 0, 0])),
                (1, m(&[0, 0, 1, 0, 0])),
                (1, m(&[0, 0, 0, 1, 0])),
                (1, m(&[0, 0, 0, 0, 1])),
            ],
        ),
        Poly::from_terms(
            &r,
            vec![
                (1, m(&[1, 1, 0, 0, 0])),
                (1, m(&[0, 1, 1, 0, 0])),
                (1, m(&[0, 0, 1, 1, 0])),
                (1, m(&[0, 0, 0, 1, 1])),
                (1, m(&[1, 0, 0, 0, 1])),
            ],
        ),
        Poly::from_terms(
            &r,
            vec![
                (1, m(&[1, 1, 1, 0, 0])),
                (1, m(&[0, 1, 1, 1, 0])),
                (1, m(&[0, 0, 1, 1, 1])),
                (1, m(&[1, 0, 0, 1, 1])),
                (1, m(&[1, 1, 0, 0, 1])),
            ],
        ),
        Poly::from_terms(
            &r,
            vec![
                (1, m(&[1, 1, 1, 1, 0])),
                (1, m(&[0, 1, 1, 1, 1])),
                (1, m(&[1, 0, 1, 1, 1])),
                (1, m(&[1, 1, 0, 1, 1])),
                (1, m(&[1, 1, 1, 0, 1])),
            ],
        ),
        Poly::from_terms(
            &r,
            vec![(1, m(&[1, 1, 1, 1, 1])), (p_minus_one, m(&[0, 0, 0, 0, 0]))],
        ),
    ];
    run_cyclic(r, "cyclic-5 over F_32003", gens, &["a", "b", "c", "d", "e"]);
}

fn main() {
    cyclic3();
    cyclic4();
    cyclic5();
}
