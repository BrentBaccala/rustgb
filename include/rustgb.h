/*
 * rustgb — C FFI header for the rustgb Groebner-basis engine.
 *
 * Hand-written to avoid a `cbindgen` build-time dependency. The
 * layout is documented in `~/project/docs/rust-bba-port-plan.md`
 * §§11–12 and `~/project/docs/rustgb-singular-ffi-report.md`.
 *
 * Thread-safety: the library is NOT thread-safe across calls.
 * Every handle (ring / input / basis) must be used by one thread
 * at a time. `rustgb_last_error()` is thread-local.
 *
 * Error protocol: functions returning `int` return 0 on success,
 * nonzero on failure. Functions returning pointers return NULL
 * on failure. In either case `rustgb_last_error()` holds a
 * human-readable message.
 *
 * Ownership: opaque handles returned by `*_create` / `*_begin` /
 * `*_compute` must be released by the matching `*_destroy`.
 *   - `rustgb_input_begin`   → `rustgb_input_destroy`  OR
 *                              `rustgb_compute` (consumes input)
 *   - `rustgb_compute`       → `rustgb_basis_destroy`
 *   - `rustgb_ring_create`   → `rustgb_ring_destroy`
 *
 * The rustgb cdylib panics are caught via `catch_unwind` at every
 * FFI boundary and converted into error returns; however panicking
 * across an FFI boundary is undefined behaviour in general, so
 * callers should still treat a nonzero return as "the library is
 * in a recoverable but possibly suspect state".
 */
#ifndef RUSTGB_H
#define RUSTGB_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------
 *  Version / diagnostics
 * ------------------------------------------------------------------ */

/*
 * Returns a static version string, e.g. "rustgb 0.0.1". The pointer
 * is valid for the lifetime of the process. Used by autoconf to
 * probe linkage.
 */
const char* rustgb_version(void);

/*
 * Returns the last error message set on the current thread, or an
 * empty string if no error is pending. The pointer is valid until
 * the next rustgb call on the same thread.
 */
const char* rustgb_last_error(void);

/* ------------------------------------------------------------------
 *  Rings
 * ------------------------------------------------------------------ */

typedef struct rustgb_ring rustgb_ring;

/*
 * Construct a ring over Z/prime with `nvars` variables, degrevlex
 * ordering. `prime` must satisfy 2 <= prime < 2^31 and be prime;
 * primality is NOT verified. `nvars` must satisfy 1 <= nvars <= 31
 * (the current 8-bit-per-var packing limit).
 *
 * Returns NULL on error.
 */
rustgb_ring* rustgb_ring_create(uint32_t nvars, uint32_t prime);

/* Release a ring handle. Passing NULL is a no-op. */
void rustgb_ring_destroy(rustgb_ring* r);

/* ------------------------------------------------------------------
 *  Input streaming
 * ------------------------------------------------------------------ */

typedef struct rustgb_input rustgb_input;

/*
 * Begin streaming an input ideal in `ring`. Returns a fresh handle
 * on success, NULL on error. The ring pointer is borrowed — it must
 * outlive the returned input handle (until the handle is destroyed
 * or consumed by `rustgb_compute`).
 */
rustgb_input* rustgb_input_begin(rustgb_ring* ring);

/* Free an input stream that will NOT be passed to `rustgb_compute`. */
void rustgb_input_destroy(rustgb_input* in);

/*
 * Start a new polynomial in the input. Must be matched by a
 * `rustgb_input_poly_end`. Zero or more `rustgb_input_term` calls
 * may appear between them (zero terms = zero polynomial).
 *
 * Returns 0 on success, nonzero on error.
 */
int rustgb_input_poly_begin(rustgb_input* in);

/*
 * Append a term to the polynomial currently being built.
 *
 *   `exps` is an array of length `nvars` of non-negative integer
 *   exponents (each must be <= 255 given the current 8-bit
 *   packing).
 *   `coefficient` is in the canonical range [0, prime). A zero
 *   coefficient is silently ignored (the term is dropped).
 *
 * Returns 0 on success, nonzero on error.
 */
int rustgb_input_term(rustgb_input* in,
                      const int32_t* exps,
                      uint32_t coefficient);

/*
 * Finish the polynomial currently being built. Returns 0 on
 * success, nonzero on error. The polynomial is added to the
 * ideal being streamed in.
 */
int rustgb_input_poly_end(rustgb_input* in);

/* ------------------------------------------------------------------
 *  Computation
 * ------------------------------------------------------------------ */

typedef struct rustgb_basis rustgb_basis;

/*
 * Compute the reduced Gröbner basis of the streamed ideal.
 *
 * ** Consumes the input stream. ** On success the caller should
 * NOT call `rustgb_input_destroy(in)`. On failure (NULL return)
 * the input stream is still consumed; the pointer must not be
 * reused.
 *
 * Returns NULL on error.
 */
rustgb_basis* rustgb_compute(rustgb_input* in);

/* Release a basis handle. Passing NULL is a no-op. */
void rustgb_basis_destroy(rustgb_basis* b);

/* ------------------------------------------------------------------
 *  Output accessors
 * ------------------------------------------------------------------ */

/* Number of polynomials in the basis. */
size_t rustgb_basis_poly_count(const rustgb_basis* b);

/* Number of terms in polynomial `poly_idx` (0-based). */
size_t rustgb_basis_term_count(const rustgb_basis* b, size_t poly_idx);

/*
 * Read term `term_idx` of polynomial `poly_idx`. Writes the
 * exponent vector into `exps_out` (length `nvars`) and the
 * coefficient into `*coeff_out`.
 *
 * Returns 0 on success, nonzero on error (out-of-bounds indices).
 */
int rustgb_basis_term(const rustgb_basis* b,
                      size_t poly_idx,
                      size_t term_idx,
                      int32_t* exps_out,
                      uint32_t* coeff_out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RUSTGB_H */
