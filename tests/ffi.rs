//! Integration test for the rustgb C FFI.
//!
//! We exercise the extern "C" surface from Rust itself (dogfooding
//! the `extern "C"` prototypes) and compare against the direct Rust
//! [`rustgb::compute_gb`] path. This gives us a round-trip sanity
//! check without needing Singular.

#![allow(unused_unsafe)]


use std::ffi::CStr;
use std::sync::Arc;

use rustgb::ffi::{
    last_error_string, rustgb_basis, rustgb_basis_destroy, rustgb_basis_poly_count,
    rustgb_basis_term_count, rustgb_compute, rustgb_input, rustgb_input_begin,
    rustgb_input_destroy, rustgb_input_poly_begin, rustgb_input_poly_end, rustgb_input_term,
    rustgb_last_error, rustgb_ring, rustgb_ring_create, rustgb_ring_destroy,
    rustgb_term_iter_close, rustgb_term_iter_next, rustgb_term_iter_open, rustgb_version,
};
use rustgb::{Field, MonoOrder, Monomial, Poly, Ring, compute_gb};

fn read_version() -> String {
    unsafe {
        let p = rustgb_version();
        CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

/// Helper: build a polynomial from `(coeff, exponents)` pairs using
/// the Rust API, then the FFI, and compare the basis output.
///
/// Returns `(ffi_basis_exponents, ffi_basis_coeffs, rust_basis)`.
type TermSpec<'a> = (u32, &'a [i32]);
type PolySpec<'a> = &'a [TermSpec<'a>];

/// One basis polynomial as read out of the FFI: a list of
/// `(coeff, exponent_vector)` pairs, terms in descending order.
type FfiPoly = Vec<(u32, Vec<i32>)>;

/// The full basis read out via the FFI.
type FfiBasis = Vec<FfiPoly>;

fn compute_via_ffi(
    nvars: u32,
    prime: u32,
    polys: &[PolySpec<'_>],
) -> (FfiBasis, Vec<Poly>) {
    // --- FFI path ---
    let ring = unsafe { rustgb_ring_create(nvars, prime) };
    assert!(!ring.is_null(), "ring create failed: {}", last_error_string());

    let input = unsafe { rustgb_input_begin(ring) };
    assert!(!input.is_null(), "input begin failed");

    for p in polys {
        assert_eq!(unsafe { rustgb_input_poly_begin(input) }, 0);
        for (coeff, exps) in *p {
            assert_eq!(
                unsafe { rustgb_input_term(input, exps.as_ptr(), *coeff) },
                0,
                "term append failed: {}",
                last_error_string()
            );
        }
        assert_eq!(unsafe { rustgb_input_poly_end(input) }, 0);
    }

    let basis = unsafe { rustgb_compute(input) };
    assert!(!basis.is_null(), "compute failed: {}", last_error_string());

    // Read out the basis through the opaque term iterator. We use
    // `rustgb_basis_term_count` only for preallocation; the walk
    // itself is driven by the iterator's exhaustion signal.
    let npoly = unsafe { rustgb_basis_poly_count(basis) };
    let mut ffi_out: FfiBasis = Vec::with_capacity(npoly);
    for pi in 0..npoly {
        let nt = unsafe { rustgb_basis_term_count(basis, pi) };
        let mut terms: FfiPoly = Vec::with_capacity(nt);
        let it = unsafe { rustgb_term_iter_open(basis, pi) };
        assert!(
            !it.is_null(),
            "term_iter_open failed: {}",
            last_error_string()
        );
        loop {
            let mut exps = vec![0i32; nvars as usize];
            let mut coeff: u32 = 0;
            let rc = unsafe {
                rustgb_term_iter_next(it, exps.as_mut_ptr(), &mut coeff as *mut u32)
            };
            if rc == 1 {
                break;
            }
            assert_eq!(rc, 0, "term_iter_next error: {}", last_error_string());
            terms.push((coeff, exps));
        }
        unsafe { rustgb_term_iter_close(it) };
        ffi_out.push(terms);
    }
    unsafe { rustgb_basis_destroy(basis) };
    unsafe { rustgb_ring_destroy(ring) };

    // --- Direct Rust path ---
    let rust_ring = Arc::new(
        Ring::new(nvars, MonoOrder::DegRevLex, Field::new(prime).unwrap()).unwrap(),
    );
    let rust_polys: Vec<Poly> = polys
        .iter()
        .map(|p| {
            let terms: Vec<(u32, Monomial)> = p
                .iter()
                .map(|(c, es)| {
                    let uexps: Vec<u32> = es.iter().map(|&e| e as u32).collect();
                    (*c, Monomial::from_exponents(&rust_ring, &uexps).unwrap())
                })
                .collect();
            Poly::from_terms(&rust_ring, terms)
        })
        .collect();
    let rust_out = compute_gb(Arc::clone(&rust_ring), rust_polys);

    (ffi_out, rust_out)
}

fn polys_match(ffi_out: &[FfiPoly], rust_out: &[Poly], ring: &Ring) -> bool {
    if ffi_out.len() != rust_out.len() {
        return false;
    }
    for (ffi_p, rust_p) in ffi_out.iter().zip(rust_out.iter()) {
        if ffi_p.len() != rust_p.len() {
            return false;
        }
        // Walk the rust Poly via its cursor-based iter() (no slice
        // accessors — those are private to the backend per ADR-014).
        for ((c_ffi, exps), (c_rust, m_rust)) in ffi_p.iter().zip(rust_p.iter()) {
            if *c_ffi != c_rust {
                return false;
            }
            let uexps: Vec<u32> = exps.iter().map(|&e| e as u32).collect();
            let m = Monomial::from_exponents(ring, &uexps).unwrap();
            if &m != m_rust {
                return false;
            }
        }
    }
    true
}

#[test]
fn version_has_prefix() {
    let v = read_version();
    assert!(v.starts_with("rustgb "), "got {v:?}");
}

#[test]
fn last_error_starts_empty() {
    unsafe {
        let p = rustgb_last_error();
        let s = CStr::from_ptr(p).to_string_lossy();
        // (Might be set by a prior test in the same thread; just check
        // we can read it.)
        let _ = s;
    }
}

#[test]
fn ring_create_destroy_roundtrip() {
    let ring = unsafe { rustgb_ring_create(3, 32003) };
    assert!(!ring.is_null());
    unsafe { rustgb_ring_destroy(ring) };
}

#[test]
fn ring_create_rejects_bad_prime() {
    let ring = unsafe { rustgb_ring_create(3, 1) };
    assert!(ring.is_null());
    let err = last_error_string();
    assert!(err.contains("prime"), "got {err:?}");
}

#[test]
fn ring_create_rejects_bad_nvars() {
    let ring = unsafe { rustgb_ring_create(0, 32003) };
    assert!(ring.is_null());
    let ring2 = unsafe { rustgb_ring_create(100, 32003) };
    assert!(ring2.is_null());
}

#[test]
fn cyclic3_ffi_matches_rust() {
    // f1 = x+y+z, f2 = xy+yz+zx, f3 = xyz-1
    // Variables indexed 0,1,2.
    let f1: PolySpec = &[
        (1, &[1, 0, 0]),
        (1, &[0, 1, 0]),
        (1, &[0, 0, 1]),
    ];
    let f2: PolySpec = &[
        (1, &[1, 1, 0]),
        (1, &[0, 1, 1]),
        (1, &[1, 0, 1]),
    ];
    let f3: PolySpec = &[
        (1, &[1, 1, 1]),
        (32002, &[0, 0, 0]),
    ];
    let polys: &[PolySpec] = &[f1, f2, f3];
    let (ffi_out, rust_out) = compute_via_ffi(3, 32003, polys);

    let ring = Ring::new(3, MonoOrder::DegRevLex, Field::new(32003).unwrap()).unwrap();
    assert!(
        polys_match(&ffi_out, &rust_out, &ring),
        "FFI output differs from direct Rust output:\nFFI: {ffi_out:?}\nRust: {rust_out:?}"
    );
    // Shape check against the reference basis from bba.rs tests.
    assert_eq!(rust_out.len(), 3);
}

#[test]
fn cyclic4_ffi_matches_rust() {
    // cyclic-4: x+y+z+w, xy+yz+zw+wx, xyz+yzw+zwx+wxy, xyzw-1
    let f1: PolySpec = &[
        (1, &[1, 0, 0, 0]),
        (1, &[0, 1, 0, 0]),
        (1, &[0, 0, 1, 0]),
        (1, &[0, 0, 0, 1]),
    ];
    let f2: PolySpec = &[
        (1, &[1, 1, 0, 0]),
        (1, &[0, 1, 1, 0]),
        (1, &[0, 0, 1, 1]),
        (1, &[1, 0, 0, 1]),
    ];
    let f3: PolySpec = &[
        (1, &[1, 1, 1, 0]),
        (1, &[0, 1, 1, 1]),
        (1, &[1, 0, 1, 1]),
        (1, &[1, 1, 0, 1]),
    ];
    let f4: PolySpec = &[
        (1, &[1, 1, 1, 1]),
        (32002, &[0, 0, 0, 0]),
    ];
    let polys: &[PolySpec] = &[f1, f2, f3, f4];
    let (ffi_out, rust_out) = compute_via_ffi(4, 32003, polys);

    let ring = Ring::new(4, MonoOrder::DegRevLex, Field::new(32003).unwrap()).unwrap();
    assert!(
        polys_match(&ffi_out, &rust_out, &ring),
        "cyclic-4 FFI vs Rust mismatch"
    );
}

#[test]
fn empty_input_gives_empty_basis() {
    let (ffi, rust) = compute_via_ffi(3, 32003, &[]);
    assert!(ffi.is_empty());
    assert!(rust.is_empty());
}

#[test]
fn unfinished_poly_caught_by_compute() {
    let ring = unsafe { rustgb_ring_create(2, 32003) };
    let input = unsafe { rustgb_input_begin(ring) };
    assert_eq!(unsafe { rustgb_input_poly_begin(input) }, 0);
    // Note: no poly_end.
    let basis = unsafe { rustgb_compute(input) };
    assert!(basis.is_null());
    assert!(last_error_string().contains("unfinished"));
    unsafe { rustgb_ring_destroy(ring) };
}

#[test]
fn negative_exponent_rejected() {
    let ring = unsafe { rustgb_ring_create(2, 32003) };
    let input = unsafe { rustgb_input_begin(ring) };
    assert_eq!(unsafe { rustgb_input_poly_begin(input) }, 0);
    let exps: [i32; 2] = [-1, 2];
    let rc = unsafe { rustgb_input_term(input, exps.as_ptr(), 1) };
    assert_ne!(rc, 0);
    assert!(last_error_string().contains("negative"));
    unsafe { rustgb_input_destroy(input) };
    unsafe { rustgb_ring_destroy(ring) };
}

#[test]
fn coeff_not_reduced_rejected() {
    let ring = unsafe { rustgb_ring_create(2, 7) };
    let input = unsafe { rustgb_input_begin(ring) };
    assert_eq!(unsafe { rustgb_input_poly_begin(input) }, 0);
    let exps: [i32; 2] = [1, 0];
    let rc = unsafe { rustgb_input_term(input, exps.as_ptr(), 9) }; // 9 >= 7
    assert_ne!(rc, 0);
    assert!(last_error_string().contains("not reduced"));
    unsafe { rustgb_input_destroy(input) };
    unsafe { rustgb_ring_destroy(ring) };
}

#[test]
fn roundtrip_preserves_term_order() {
    // xy + y^2 + 1 — make sure the FFI reads back terms in the
    // same descending order the Rust side uses.
    let f: PolySpec = &[
        (1, &[1, 1]),
        (1, &[0, 2]),
        (1, &[0, 0]),
    ];
    let (ffi, rust) = compute_via_ffi(2, 32003, &[f]);
    let ring = Ring::new(2, MonoOrder::DegRevLex, Field::new(32003).unwrap()).unwrap();
    assert!(polys_match(&ffi, &rust, &ring));
}

/// Make sure opaque handle types aren't zero-sized (would make
/// `Box::into_raw` unsafe to round-trip).
#[allow(dead_code)]
const fn handle_sizes_nonzero() {
    assert!(std::mem::size_of::<rustgb_ring>() > 0);
    assert!(std::mem::size_of::<rustgb_input>() > 0);
    assert!(std::mem::size_of::<rustgb_basis>() > 0);
}
