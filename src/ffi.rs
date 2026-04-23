//! C FFI for the rustgb crate.
//!
//! The layout of this module mirrors the public header
//! [`include/rustgb.h`](../../include/rustgb.h). Each exported
//! function wraps its body in [`std::panic::catch_unwind`] so that
//! a panic on the Rust side is converted into an error return
//! instead of unwinding across the C boundary (which is UB).
//!
//! Thread-local state:
//!
//! * `LAST_ERROR` holds a `CString` with the most recent error
//!   message on the current thread; [`rustgb_last_error`] returns
//!   a pointer into it.
//!
//! All handles (`RustgbRing`, `RustgbInput`, `RustgbBasis`) are
//! zero-cost newtypes around the corresponding Rust struct. They
//! are leaked (via `Box::into_raw`) into C ownership and must be
//! reclaimed with the matching `_destroy` function; `rustgb_compute`
//! additionally consumes the input handle.

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use crate::bba::compute_gb;
use crate::field::{Coeff, Field};
use crate::monomial::Monomial;
use crate::ordering::MonoOrder;
use crate::poly::Poly;
use crate::ring::Ring;

// ---------------------------------------------------------------
//  Thread-local error string
// ---------------------------------------------------------------

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap());
}

fn set_last_error(msg: &str) {
    // Replace any embedded NUL so the conversion always succeeds.
    let sanitised: String = msg.chars().map(|c| if c == '\0' { '?' } else { c }).collect();
    let cstr = CString::new(sanitised).unwrap_or_else(|_| CString::new("unknown error").unwrap());
    LAST_ERROR.with(|cell| *cell.borrow_mut() = cstr);
}

fn clear_last_error() {
    LAST_ERROR.with(|cell| *cell.borrow_mut() = CString::new("").unwrap());
}

// ---------------------------------------------------------------
//  Opaque handle types
// ---------------------------------------------------------------

/// Opaque ring handle. The C header declares `typedef struct rustgb_ring`
/// so this name matters.
#[allow(non_camel_case_types)]
#[repr(C)]
pub struct rustgb_ring {
    inner: Arc<Ring>,
}

/// Opaque input-ideal builder.
#[allow(non_camel_case_types)]
#[repr(C)]
pub struct rustgb_input {
    ring: Arc<Ring>,
    /// Completed polynomials so far.
    polys: Vec<Poly>,
    /// In-progress polynomial: accumulated as `(coeff, monomial)`
    /// pairs; sorted / deduped when `poly_end` is called. `None`
    /// means we're between polynomials.
    current: Option<Vec<(Coeff, Monomial)>>,
}

/// Opaque basis output.
#[allow(non_camel_case_types)]
#[repr(C)]
pub struct rustgb_basis {
    ring: Arc<Ring>,
    polys: Vec<Poly>,
}

// Helper macros to translate `catch_unwind` outcomes. We keep them
// inline for clarity at each call site rather than abstracting over
// return types.

const VERSION: &str = concat!("rustgb ", env!("CARGO_PKG_VERSION"), "\0");

// ---------------------------------------------------------------
//  Version / diagnostics
// ---------------------------------------------------------------

/// Return a static version string (null-terminated).
///
/// # Safety
/// No preconditions; always safe to call.
#[unsafe(no_mangle)]
pub extern "C" fn rustgb_version() -> *const c_char {
    // `VERSION` is a compile-time literal with a trailing NUL.
    VERSION.as_ptr() as *const c_char
}

/// Return the current thread's last error message (may be empty).
///
/// # Safety
/// The returned pointer is valid until the next rustgb call on
/// the same thread.
#[unsafe(no_mangle)]
pub extern "C" fn rustgb_last_error() -> *const c_char {
    // We leak a pointer into the thread-local CString; it stays
    // valid until the next `set_last_error` or `clear_last_error`.
    LAST_ERROR.with(|cell| cell.borrow().as_ptr())
}

// ---------------------------------------------------------------
//  Ring
// ---------------------------------------------------------------

/// Construct a ring over Z/`prime` with `nvars` variables, degrevlex.
///
/// Returns `NULL` on error; see `rustgb_last_error()`.
///
/// # Safety
/// No preconditions; returns an owned handle the caller must
/// release with [`rustgb_ring_destroy`].
#[unsafe(no_mangle)]
pub extern "C" fn rustgb_ring_create(nvars: u32, prime: u32) -> *mut rustgb_ring {
    clear_last_error();
    let r = catch_unwind(|| {
        let field = match Field::new(prime) {
            Some(f) => f,
            None => {
                set_last_error(&format!(
                    "rustgb_ring_create: prime {prime} out of range (need 2 <= p < 2^31)"
                ));
                return std::ptr::null_mut();
            }
        };
        let ring = match Ring::new(nvars, MonoOrder::DegRevLex, field) {
            Some(r) => r,
            None => {
                set_last_error(&format!(
                    "rustgb_ring_create: nvars {nvars} out of range (need 1..=31)"
                ));
                return std::ptr::null_mut();
            }
        };
        let handle = Box::new(rustgb_ring {
            inner: Arc::new(ring),
        });
        Box::into_raw(handle)
    });
    match r {
        Ok(p) => p,
        Err(_) => {
            set_last_error("rustgb_ring_create: panic");
            std::ptr::null_mut()
        }
    }
}

/// Release a ring handle. Passing NULL is a no-op.
///
/// # Safety
/// `r` must have been obtained from [`rustgb_ring_create`] or be NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustgb_ring_destroy(r: *mut rustgb_ring) {
    if r.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: caller contract.
        unsafe {
            drop(Box::from_raw(r));
        }
    }));
}

// ---------------------------------------------------------------
//  Input streaming
// ---------------------------------------------------------------

/// Begin streaming an input ideal in `ring`.
///
/// # Safety
/// `ring` must have been obtained from [`rustgb_ring_create`] and
/// must remain alive until the returned input is consumed (by
/// [`rustgb_compute`]) or destroyed (by [`rustgb_input_destroy`]).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustgb_input_begin(ring: *mut rustgb_ring) -> *mut rustgb_input {
    clear_last_error();
    let r = catch_unwind(AssertUnwindSafe(|| {
        if ring.is_null() {
            set_last_error("rustgb_input_begin: ring is NULL");
            return std::ptr::null_mut();
        }
        // SAFETY: caller contract.
        let ring_ref = unsafe { &*ring };
        let handle = Box::new(rustgb_input {
            ring: Arc::clone(&ring_ref.inner),
            polys: Vec::new(),
            current: None,
        });
        Box::into_raw(handle)
    }));
    match r {
        Ok(p) => p,
        Err(_) => {
            set_last_error("rustgb_input_begin: panic");
            std::ptr::null_mut()
        }
    }
}

/// Free an input stream that will NOT be passed to `rustgb_compute`.
///
/// # Safety
/// `input` must have been obtained from [`rustgb_input_begin`] and
/// must not have been consumed by [`rustgb_compute`], or must be NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustgb_input_destroy(input: *mut rustgb_input) {
    if input.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: caller contract.
        unsafe {
            drop(Box::from_raw(input));
        }
    }));
}

/// Start a new polynomial.
///
/// # Safety
/// `input` must be a live handle from [`rustgb_input_begin`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustgb_input_poly_begin(input: *mut rustgb_input) -> i32 {
    clear_last_error();
    let r = catch_unwind(AssertUnwindSafe(|| {
        if input.is_null() {
            set_last_error("rustgb_input_poly_begin: input is NULL");
            return 1;
        }
        // SAFETY: caller contract.
        let inp = unsafe { &mut *input };
        if inp.current.is_some() {
            set_last_error("rustgb_input_poly_begin: previous poly not ended");
            return 1;
        }
        inp.current = Some(Vec::new());
        0
    }));
    match r {
        Ok(code) => code,
        Err(_) => {
            set_last_error("rustgb_input_poly_begin: panic");
            1
        }
    }
}

/// Append a term to the current polynomial.
///
/// # Safety
/// `input` must be a live handle; `exps` must point to `nvars`
/// `i32` exponents.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustgb_input_term(
    input: *mut rustgb_input,
    exps: *const i32,
    coefficient: u32,
) -> i32 {
    clear_last_error();
    let r = catch_unwind(AssertUnwindSafe(|| {
        if input.is_null() {
            set_last_error("rustgb_input_term: input is NULL");
            return 1;
        }
        if exps.is_null() {
            set_last_error("rustgb_input_term: exps is NULL");
            return 1;
        }
        // SAFETY: caller contract.
        let inp = unsafe { &mut *input };
        let Some(ref mut terms) = inp.current else {
            set_last_error("rustgb_input_term: no poly in progress");
            return 1;
        };
        let nvars = inp.ring.nvars() as usize;
        // SAFETY: caller promises `exps` has length `nvars`.
        let slice = unsafe { std::slice::from_raw_parts(exps, nvars) };
        let mut u_exps: Vec<u32> = Vec::with_capacity(nvars);
        for (i, &e) in slice.iter().enumerate() {
            if e < 0 {
                set_last_error(&format!(
                    "rustgb_input_term: exponent[{i}] = {e} is negative"
                ));
                return 1;
            }
            if e > u8::MAX as i32 {
                set_last_error(&format!(
                    "rustgb_input_term: exponent[{i}] = {e} exceeds 255"
                ));
                return 1;
            }
            u_exps.push(e as u32);
        }
        let mono = match Monomial::from_exponents(&inp.ring, &u_exps) {
            Some(m) => m,
            None => {
                set_last_error("rustgb_input_term: failed to construct monomial");
                return 1;
            }
        };
        let p = inp.ring.field().p();
        if coefficient >= p {
            set_last_error(&format!(
                "rustgb_input_term: coefficient {coefficient} is not reduced mod {p}"
            ));
            return 1;
        }
        if coefficient != 0 {
            terms.push((coefficient, mono));
        }
        0
    }));
    match r {
        Ok(code) => code,
        Err(_) => {
            set_last_error("rustgb_input_term: panic");
            1
        }
    }
}

/// Finish the current polynomial and append it to the ideal.
///
/// # Safety
/// `input` must be a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustgb_input_poly_end(input: *mut rustgb_input) -> i32 {
    clear_last_error();
    let r = catch_unwind(AssertUnwindSafe(|| {
        if input.is_null() {
            set_last_error("rustgb_input_poly_end: input is NULL");
            return 1;
        }
        // SAFETY: caller contract.
        let inp = unsafe { &mut *input };
        let Some(terms) = inp.current.take() else {
            set_last_error("rustgb_input_poly_end: no poly in progress");
            return 1;
        };
        // `from_terms` sorts descending, merges dupes, drops zeros.
        let poly = Poly::from_terms(&inp.ring, terms);
        inp.polys.push(poly);
        0
    }));
    match r {
        Ok(code) => code,
        Err(_) => {
            set_last_error("rustgb_input_poly_end: panic");
            1
        }
    }
}

// ---------------------------------------------------------------
//  Computation
// ---------------------------------------------------------------

/// Compute the reduced Gröbner basis. Consumes `input`.
///
/// # Safety
/// `input` must be a live handle. After this call the pointer is
/// invalid regardless of success or failure; do not reuse it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustgb_compute(input: *mut rustgb_input) -> *mut rustgb_basis {
    clear_last_error();
    let r = catch_unwind(AssertUnwindSafe(|| {
        if input.is_null() {
            set_last_error("rustgb_compute: input is NULL");
            return std::ptr::null_mut();
        }
        // SAFETY: caller contract — we take ownership.
        let inp = unsafe { Box::from_raw(input) };
        if inp.current.is_some() {
            set_last_error("rustgb_compute: unfinished poly in progress");
            return std::ptr::null_mut();
        }
        let ring = Arc::clone(&inp.ring);
        let basis = compute_gb(Arc::clone(&ring), inp.polys);
        let handle = Box::new(rustgb_basis {
            ring,
            polys: basis,
        });
        Box::into_raw(handle)
    }));
    match r {
        Ok(p) => p,
        Err(_) => {
            set_last_error("rustgb_compute: panic");
            std::ptr::null_mut()
        }
    }
}

/// Release a basis handle.
///
/// # Safety
/// `b` must have been obtained from [`rustgb_compute`] or be NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustgb_basis_destroy(b: *mut rustgb_basis) {
    if b.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: caller contract.
        unsafe {
            drop(Box::from_raw(b));
        }
    }));
}

// ---------------------------------------------------------------
//  Output accessors
// ---------------------------------------------------------------

/// Number of polynomials in the basis.
///
/// # Safety
/// `b` must be a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustgb_basis_poly_count(b: *const rustgb_basis) -> usize {
    clear_last_error();
    if b.is_null() {
        set_last_error("rustgb_basis_poly_count: b is NULL");
        return 0;
    }
    // SAFETY: caller contract.
    let b = unsafe { &*b };
    b.polys.len()
}

/// Number of terms in polynomial `poly_idx`.
///
/// # Safety
/// `b` must be a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustgb_basis_term_count(b: *const rustgb_basis, poly_idx: usize) -> usize {
    clear_last_error();
    if b.is_null() {
        set_last_error("rustgb_basis_term_count: b is NULL");
        return 0;
    }
    // SAFETY: caller contract.
    let b = unsafe { &*b };
    if poly_idx >= b.polys.len() {
        set_last_error(&format!(
            "rustgb_basis_term_count: poly_idx {poly_idx} out of range (nel={})",
            b.polys.len()
        ));
        return 0;
    }
    b.polys[poly_idx].len()
}

// ---------------------------------------------------------------
//  Term iterator
// ---------------------------------------------------------------
//
// The iterator is an opaque boxed struct that borrows the basis
// behind a raw pointer. The caller contract (documented in the C
// header) is: the basis (and its ring) must outlive the iterator,
// and must not be mutated while the iterator is live.
//
// Internal shape is backend-specific but hidden: we hold a
// `PolyCursor<'static>` obtained from the basis's poly and lifetime-
// extended to `'static` at the FFI boundary. The caller's
// basis-outlives-iterator contract justifies the extension — the
// cursor is never dereferenced after the basis is freed because
// the caller must close the iterator first. Both the `Vec`-backed
// and linked-list-backed `Poly` produce a `PolyCursor` of the same
// opaque shape, so this handle is backend-agnostic.
//
// The C surface does not expose any of this.

/// Opaque term-iterator handle. See the C header for the caller
/// contract; do not rely on the field layout.
#[allow(non_camel_case_types)]
#[repr(C)]
pub struct rustgb_term_iter {
    /// Borrowed pointer to the basis the iterator was opened against.
    /// Retained for defensive null / index re-validation on the next
    /// call; the actual read goes through `cursor`.
    basis: *const rustgb_basis,
    /// Which polynomial in the basis we're walking. Retained for
    /// diagnostics on the error path.
    poly_idx: usize,
    /// Cursor into that polynomial, lifetime-extended to `'static`.
    /// Must not be dereferenced after the basis is freed (caller
    /// contract: the basis outlives the iterator). The reader
    /// re-validates `basis` / `poly_idx` on each `_next` call so a
    /// corrupt handle reports an error rather than dereferencing a
    /// stale pointer into the cursor — but once that validation
    /// passes, `cursor.term()` is the sole source of truth for the
    /// current term.
    cursor: crate::poly::PolyCursor<'static>,
}

/// Open an iterator over the terms of polynomial `poly_idx` in `b`.
///
/// Returns `NULL` on error (see `rustgb_last_error()`). Terms are
/// yielded in the ring's descending order (same order as the
/// underlying `Poly`).
///
/// # Safety
/// `b` must be a live handle, and must outlive the returned
/// iterator. The basis must not be destroyed or mutated while the
/// iterator is outstanding. The caller must release the iterator
/// with [`rustgb_term_iter_close`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustgb_term_iter_open(
    b: *const rustgb_basis,
    poly_idx: usize,
) -> *mut rustgb_term_iter {
    clear_last_error();
    let r = catch_unwind(AssertUnwindSafe(|| {
        if b.is_null() {
            set_last_error("rustgb_term_iter_open: b is NULL");
            return std::ptr::null_mut();
        }
        // SAFETY: caller contract.
        let basis_ref = unsafe { &*b };
        if poly_idx >= basis_ref.polys.len() {
            set_last_error(&format!(
                "rustgb_term_iter_open: poly_idx {poly_idx} out of range (nel={})",
                basis_ref.polys.len()
            ));
            return std::ptr::null_mut();
        }
        // SAFETY: We extend the cursor's lifetime to `'static`. The
        // caller's contract (basis outlives the iterator, not mutated
        // while live) is what makes this sound; the cursor is closed
        // by `rustgb_term_iter_close` (or dropped if the caller
        // forgets — still safe because `PolyCursor` holds only
        // references, no owned resources).
        let cursor_static: crate::poly::PolyCursor<'static> = unsafe {
            std::mem::transmute::<crate::poly::PolyCursor<'_>, crate::poly::PolyCursor<'static>>(
                basis_ref.polys[poly_idx].cursor(),
            )
        };
        let handle = Box::new(rustgb_term_iter {
            basis: b,
            poly_idx,
            cursor: cursor_static,
        });
        Box::into_raw(handle)
    }));
    match r {
        Ok(p) => p,
        Err(_) => {
            set_last_error("rustgb_term_iter_open: panic");
            std::ptr::null_mut()
        }
    }
}

/// Read the next term from an iterator.
///
/// On success (return 0) writes `ring.nvars()` exponents into
/// `exps_out` and the coefficient into `*coeff_out`, then advances
/// the cursor. When the iterator is exhausted returns 1 and leaves
/// the output buffers untouched. Returns 2 on error (with
/// `rustgb_last_error` set).
///
/// # Safety
/// `it` must be a live iterator handle (or NULL — treated as
/// error). `exps_out` must be writable for `nvars` `i32` slots;
/// `coeff_out` must be a valid pointer to a `u32`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustgb_term_iter_next(
    it: *mut rustgb_term_iter,
    exps_out: *mut i32,
    coeff_out: *mut u32,
) -> i32 {
    clear_last_error();
    let r = catch_unwind(AssertUnwindSafe(|| {
        if it.is_null() {
            set_last_error("rustgb_term_iter_next: it is NULL");
            return 2;
        }
        if exps_out.is_null() || coeff_out.is_null() {
            set_last_error("rustgb_term_iter_next: output pointer is NULL");
            return 2;
        }
        // SAFETY: caller contract.
        let iter = unsafe { &mut *it };
        if iter.basis.is_null() {
            set_last_error("rustgb_term_iter_next: iterator has NULL basis");
            return 2;
        }
        // SAFETY: caller promised the basis outlives the iterator.
        let basis_ref = unsafe { &*iter.basis };
        if iter.poly_idx >= basis_ref.polys.len() {
            set_last_error("rustgb_term_iter_next: poly_idx out of range");
            return 2;
        }
        let Some((coeff, mono)) = iter.cursor.term() else {
            // Exhausted: leave output untouched.
            return 1;
        };
        let nvars = basis_ref.ring.nvars() as usize;
        // SAFETY: caller contract.
        let slice = unsafe { std::slice::from_raw_parts_mut(exps_out, nvars) };
        for (i, slot) in slice.iter_mut().enumerate() {
            *slot = mono.exponent(&basis_ref.ring, i as u32).expect("i < nvars") as i32;
        }
        // SAFETY: caller contract.
        unsafe {
            *coeff_out = coeff;
        }
        iter.cursor.advance();
        0
    }));
    match r {
        Ok(code) => code,
        Err(_) => {
            set_last_error("rustgb_term_iter_next: panic");
            2
        }
    }
}

/// Release an iterator handle. Passing NULL is a no-op.
///
/// # Safety
/// `it` must have been obtained from [`rustgb_term_iter_open`] or
/// be NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustgb_term_iter_close(it: *mut rustgb_term_iter) {
    if it.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: caller contract.
        unsafe {
            drop(Box::from_raw(it));
        }
    }));
}

// ---------------------------------------------------------------
//  Internal: used by tests/ffi.rs
// ---------------------------------------------------------------

/// Test-only helper: convert a C-style error string into a Rust
/// `String`. Not part of the public header.
#[doc(hidden)]
pub fn last_error_string() -> String {
    // SAFETY: rustgb_last_error always returns a valid C string.
    unsafe {
        let ptr = rustgb_last_error();
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}
