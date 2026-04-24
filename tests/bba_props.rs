//! Property-based tests for the bba driver.
//!
//! These exercise three invariants of the reduced Gröbner basis
//! returned by [`rustgb::compute_gb`]:
//!
//! 1. **Determinism** — running `compute_gb` twice on the same input
//!    produces bit-identical output.
//! 2. **Input-order invariance** — shuffling the order of the input
//!    generators does not change the output (the reduced GB is
//!    unique up to permutation, and our canonical sort removes the
//!    permutation ambiguity).
//! 3. **Idempotence** — feeding the output of one `compute_gb` call
//!    back as input must reproduce the same output.
//! 4. **Inclusion** — every input polynomial reduces to zero against
//!    the output basis.
//!
//! A small random-ideal generator keeps the tests varied without
//! requiring proptest's full shrinking machinery for this task.
//!
//! Test-case generators use plain PRNG — the goal is coverage
//! breadth, not shrink-path minimisation. If a specific failure
//! crops up later, we can promote it to a hand-crafted `#[test]`.

use std::sync::Arc;

use rustgb::compute_gb;
use rustgb::compute_gb_parallel;
use rustgb::field::Field;
use rustgb::monomial::Monomial;
use rustgb::ordering::MonoOrder;
use rustgb::poly::Poly;
use rustgb::ring::Ring;

/// Tiny LCG for deterministic "random" inputs. Same one used in
/// `field.rs`'s tests.
struct Prng(u64);
impl Prng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn in_range(&mut self, lo: u32, hi: u32) -> u32 {
        let span = hi - lo + 1;
        lo + self.next_u32() % span
    }
}

fn mk_ring(nvars: u32, p: u32) -> Arc<Ring> {
    Arc::new(Ring::new(nvars, MonoOrder::DegRevLex, Field::new(p).unwrap()).unwrap())
}

fn mono(r: &Ring, e: &[u32]) -> Monomial {
    Monomial::from_exponents(r, e).unwrap()
}

/// Generate a random polynomial in `r` with at most `max_terms`
/// monomials of per-variable exponent at most `max_exp`.
fn random_poly(rng: &mut Prng, ring: &Ring, max_terms: u32, max_exp: u32) -> Poly {
    let n = ring.nvars() as usize;
    let t = rng.in_range(1, max_terms);
    let mut terms = Vec::with_capacity(t as usize);
    for _ in 0..t {
        let mut exps = vec![0u32; n];
        for slot in &mut exps {
            *slot = rng.in_range(0, max_exp);
        }
        let c = rng.in_range(1, ring.field().p() - 1);
        terms.push((c, mono(ring, &exps)));
    }
    Poly::from_terms(ring, terms)
}

/// Reduce `p` to normal form against `gb` (assumed to be a GB of
/// some ideal I). Returns the normal form.
///
/// This is a transparent polynomial reducer, used by the inclusion
/// property test. It doesn't reach into any private rustgb API —
/// it uses `Poly::sub_mul_term` as the single reduction step.
fn normal_form(p: &Poly, gb: &[Poly], ring: &Ring) -> Poly {
    let mut cur = p.clone();
    'outer: loop {
        if cur.is_zero() {
            return cur;
        }
        let (c, m) = {
            let (c, m) = cur.leading().expect("nonzero");
            (c, m.clone())
        };
        for s in gb {
            let (s_c, s_m) = s.leading().expect("gb element nonzero");
            if s_m.divides(&m, ring) {
                let mult = m.div(s_m, ring).expect("divisibility");
                let f = ring.field();
                let inv = f.inv(s_c).expect("invertible");
                let coeff = f.mul(c, inv);
                cur = cur.sub_mul_term(coeff, &mult, s, ring);
                continue 'outer;
            }
        }
        // Leader has no divisor. Try to reduce non-leading terms.
        // If every term has no divisor, we're done.
        let terms: Vec<(rustgb::field::Coeff, Monomial)> =
            cur.iter().map(|(c, m)| (c, m.clone())).collect();
        let mut made_progress = false;
        let mut rebuilt = vec![];
        for (c, m) in terms {
            let mut reduced = false;
            for s in gb {
                let (s_c, s_m) = s.leading().expect("nonzero");
                if s_m.divides(&m, ring) {
                    let mult = m.div(s_m, ring).expect("div");
                    let f = ring.field();
                    let inv = f.inv(s_c).expect("inv");
                    let coeff = f.mul(c, inv);
                    // single-term working poly
                    let t = Poly::monomial(ring, c, m.clone());
                    let r = t.sub_mul_term(coeff, &mult, s, ring);
                    for (rc, rm) in r.iter() {
                        rebuilt.push((rc, rm.clone()));
                    }
                    reduced = true;
                    made_progress = true;
                    break;
                }
            }
            if !reduced {
                rebuilt.push((c, m));
            }
        }
        if !made_progress {
            return cur;
        }
        cur = Poly::from_terms(ring, rebuilt);
    }
}

/// Shuffle `input` in place via Fisher-Yates using `rng`.
fn shuffle<T>(rng: &mut Prng, input: &mut [T]) {
    let n = input.len();
    if n < 2 {
        return;
    }
    for i in (1..n).rev() {
        let j = (rng.next_u32() as usize) % (i + 1);
        input.swap(i, j);
    }
}

#[test]
fn determinism_small_ideals() {
    let r = mk_ring(3, 32003);
    let mut rng = Prng::new(0x00C0_FFEE_1234_5678);
    for _ in 0..50 {
        let ngens = rng.in_range(1, 4);
        let gens: Vec<Poly> = (0..ngens)
            .map(|_| random_poly(&mut rng, &r, 4, 2))
            .collect();
        let gb1 = compute_gb(Arc::clone(&r), gens.clone());
        let gb2 = compute_gb(Arc::clone(&r), gens.clone());
        assert_eq!(gb1, gb2, "determinism violated on input {:?}", gens);
    }
}

#[test]
fn input_order_invariance_small_ideals() {
    let r = mk_ring(3, 32003);
    let mut rng = Prng::new(0xF00D_BABE);
    for _ in 0..40 {
        let ngens = rng.in_range(2, 4);
        let gens: Vec<Poly> = (0..ngens)
            .map(|_| random_poly(&mut rng, &r, 3, 2))
            .collect();
        let gb_orig = compute_gb(Arc::clone(&r), gens.clone());
        let mut shuffled = gens.clone();
        shuffle(&mut rng, &mut shuffled);
        let gb_sh = compute_gb(Arc::clone(&r), shuffled);
        assert_eq!(gb_orig, gb_sh, "order-invariance violated");
    }
}

#[test]
fn idempotence_small_ideals() {
    let r = mk_ring(3, 32003);
    let mut rng = Prng::new(0xDEAD_BEEF);
    for _ in 0..30 {
        let ngens = rng.in_range(1, 3);
        let gens: Vec<Poly> = (0..ngens)
            .map(|_| random_poly(&mut rng, &r, 3, 2))
            .collect();
        let gb_once = compute_gb(Arc::clone(&r), gens);
        let gb_twice = compute_gb(Arc::clone(&r), gb_once.clone());
        assert_eq!(gb_once, gb_twice, "idempotence violated");
    }
}

#[test]
fn every_input_reduces_to_zero() {
    let r = mk_ring(3, 32003);
    let mut rng = Prng::new(0xBADF00D);
    for _ in 0..25 {
        let ngens = rng.in_range(1, 4);
        let gens: Vec<Poly> = (0..ngens)
            .map(|_| random_poly(&mut rng, &r, 3, 2))
            .collect();
        let gb = compute_gb(Arc::clone(&r), gens.clone());
        for g in &gens {
            let nf = normal_form(g, &gb, &r);
            assert!(
                nf.is_zero(),
                "input {:?} did not reduce to zero against basis of size {}",
                g,
                gb.len()
            );
        }
    }
}

#[test]
fn cyclic3_order_permutations_all_agree() {
    // Stronger check: the six permutations of cyclic-3 inputs must
    // all produce the same reduced GB.
    let r = mk_ring(3, 32003);
    let f1 = Poly::from_terms(
        &r,
        vec![
            (1, mono(&r, &[1, 0, 0])),
            (1, mono(&r, &[0, 1, 0])),
            (1, mono(&r, &[0, 0, 1])),
        ],
    );
    let f2 = Poly::from_terms(
        &r,
        vec![
            (1, mono(&r, &[1, 1, 0])),
            (1, mono(&r, &[0, 1, 1])),
            (1, mono(&r, &[1, 0, 1])),
        ],
    );
    let f3 = Poly::from_terms(
        &r,
        vec![(1, mono(&r, &[1, 1, 1])), (32002, mono(&r, &[0, 0, 0]))],
    );
    let base = compute_gb(Arc::clone(&r), vec![f1.clone(), f2.clone(), f3.clone()]);
    let perms: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let fs = [f1, f2, f3];
    for p in perms {
        let input = p.iter().map(|&i| fs[i].clone()).collect::<Vec<_>>();
        let gb = compute_gb(Arc::clone(&r), input);
        assert_eq!(gb, base, "permutation {:?} gave different GB", p);
    }
}

// ===== Parallel-driver property tests =====
//
// The parallel driver must:
//   1. Produce the reduced GB at T>1 — bit-equal to the T=1 serial
//      output after canonical sort.
//   2. Be stable under re-execution: compute_gb_parallel(…, T=4) twice
//      must produce bit-equal output.
//   3. Match the serial path across every ideal in the random set
//      used by determinism_small_ideals.

#[test]
fn parallel_matches_serial_small_ideals() {
    let r = mk_ring(3, 32003);
    let mut rng = Prng::new(0x5A5A_A5A5_1111_2222);
    for iter in 0..30 {
        let ngens = rng.in_range(1, 4);
        let gens: Vec<Poly> = (0..ngens)
            .map(|_| random_poly(&mut rng, &r, 4, 2))
            .collect();
        let gb_serial = compute_gb(Arc::clone(&r), gens.clone());
        let gb_par = compute_gb_parallel(Arc::clone(&r), gens.clone(), 4).unwrap();
        assert_eq!(
            gb_serial, gb_par,
            "iter {}: serial vs parallel mismatch on input {:?}",
            iter, gens
        );
    }
}

#[test]
fn parallel_reduced_gb_invariant() {
    // Same input twice at T=4 — outputs must be bitwise equal.
    let r = mk_ring(3, 32003);
    let mut rng = Prng::new(0xDEAD_D00D);
    for _ in 0..20 {
        let ngens = rng.in_range(2, 4);
        let gens: Vec<Poly> = (0..ngens)
            .map(|_| random_poly(&mut rng, &r, 4, 2))
            .collect();
        let gb_a = compute_gb_parallel(Arc::clone(&r), gens.clone(), 4).unwrap();
        let gb_b = compute_gb_parallel(Arc::clone(&r), gens.clone(), 4).unwrap();
        assert_eq!(gb_a, gb_b, "parallel outputs not equal across runs");
    }
}

#[test]
fn parallel_idempotence_t4() {
    let r = mk_ring(3, 32003);
    let mut rng = Prng::new(0x1234_BEEF_BABE);
    for _ in 0..15 {
        let ngens = rng.in_range(1, 3);
        let gens: Vec<Poly> = (0..ngens)
            .map(|_| random_poly(&mut rng, &r, 3, 2))
            .collect();
        let gb_once = compute_gb_parallel(Arc::clone(&r), gens, 4).unwrap();
        let gb_twice =
            compute_gb_parallel(Arc::clone(&r), gb_once.clone(), 4).unwrap();
        assert_eq!(gb_once, gb_twice, "parallel idempotence violated at T=4");
    }
}

#[test]
fn parallel_every_input_reduces_to_zero_t4() {
    let r = mk_ring(3, 32003);
    let mut rng = Prng::new(0xF00F_B00F);
    for _ in 0..15 {
        let ngens = rng.in_range(1, 4);
        let gens: Vec<Poly> = (0..ngens)
            .map(|_| random_poly(&mut rng, &r, 3, 2))
            .collect();
        let gb = compute_gb_parallel(Arc::clone(&r), gens.clone(), 4).unwrap();
        for g in &gens {
            let nf = normal_form(g, &gb, &r);
            assert!(nf.is_zero(), "input {:?} did not reduce to zero", g);
        }
    }
}

#[test]
fn parallel_stable_across_thread_counts() {
    // T=2, T=4, T=8 must all produce the same reduced GB.
    let r = mk_ring(3, 32003);
    let mut rng = Prng::new(0xA1B2_C3D4);
    for _ in 0..10 {
        let ngens = rng.in_range(2, 4);
        let gens: Vec<Poly> = (0..ngens)
            .map(|_| random_poly(&mut rng, &r, 4, 2))
            .collect();
        let gb2 = compute_gb_parallel(Arc::clone(&r), gens.clone(), 2).unwrap();
        let gb4 = compute_gb_parallel(Arc::clone(&r), gens.clone(), 4).unwrap();
        let gb8 = compute_gb_parallel(Arc::clone(&r), gens.clone(), 8).unwrap();
        assert_eq!(gb2, gb4, "T=2 != T=4");
        assert_eq!(gb4, gb8, "T=4 != T=8");
    }
}
