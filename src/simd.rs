//! SIMD helpers shared across the crate.
//!
//! Two primitives, each in the same shape (unrolled main loop,
//! narrow tail, scalar tail):
//!
//! * [`find_sev_match`] — original SEV pre-filter (ADR-007), also
//!   reused as [`find_divmask_match`] for the divmask fast-reject
//!   (ADR-025) in `bba::find_divisor_idx`.
//! * [`find_sev_superset_match`] — superset variant for the
//!   chain-criterion sweep (ADR-009), reused as
//!   [`find_divmask_superset_match`] (ADR-025).
//!
//! Both take a flat `&[u64]` of cached bloom filters and a mask,
//! returning the first index where the per-bit predicate holds.
//!
//! Mirrors Singular's `kSevScanAVX2` / `kSevScanSSE4` from
//! `~/Singular-next-opt/kernel/GBEngine/kstd2.cc`.
//!
//! ## Dispatch (ADR-027)
//!
//! Backend selection is **runtime**, not build-time: the first call
//! resolves the best available path once (cached in a `OnceLock`)
//! and every later call matches on the cached [`SimdLevel`]:
//!
//! * **AVX2** — 16-entry-per-iteration unrolled main loop (four
//!   4-wide `AND + CMPEQ_EPI64 + MOVEMASK` batches), 4-wide tail.
//! * **SSE4.1** — the same shape halved: 8-entry main loop (four
//!   2-wide batches), 2-wide tail. Active on the SSE4.2-only Cisco
//!   C200 fleet (c200-1 / edge / ragazzo), where the AVX2 path is
//!   unavailable.
//! * **Scalar** — plain linear scan; non-x86_64 path and the
//!   reference oracle for the SIMD unit tests.
//!
//! The `#[target_feature(enable = ...)]` functions always compile
//! on x86_64 regardless of the crate's global codegen flags, so a
//! single portable `cargo build --release` (no `target-cpu=native`
//! needed) runs AVX2 on a Zen/Haswell host and SSE4.1 on a Westmere
//! host. This replaces the previous build-time `cfg(target_feature)`
//! gating, under which a plain build silently fell back to scalar.

#[cfg(target_arch = "x86_64")]
use std::sync::OnceLock;

/// Best SIMD backend available on the current CPU, resolved once.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SimdLevel {
    Avx2,
    Sse41,
    Scalar,
}

#[cfg(target_arch = "x86_64")]
fn detect_simd_level() -> SimdLevel {
    if std::is_x86_feature_detected!("avx2") {
        SimdLevel::Avx2
    } else if std::is_x86_feature_detected!("sse4.1") {
        SimdLevel::Sse41
    } else {
        SimdLevel::Scalar
    }
}

/// Cached SIMD backend. The detection (an atomic-load-backed CPUID
/// check) runs once; subsequent calls are a single acquire load.
#[cfg(target_arch = "x86_64")]
#[inline]
fn simd_level() -> SimdLevel {
    static LEVEL: OnceLock<SimdLevel> = OnceLock::new();
    *LEVEL.get_or_init(detect_simd_level)
}

// =====================================================================
// find_sev_match: first i >= start with (sevs[i] & not_sev) == 0.
// =====================================================================

/// Return the smallest index `i >= start` with
/// `(sevs[i] & not_sev) == 0`, or `sevs.len()` if no such index
/// exists.
///
/// Dispatches at runtime (ADR-027) to the AVX2, SSE4.1, or scalar
/// implementation. All three produce identical results on identical
/// inputs; the cargo test suite asserts agreement per backend.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn find_sev_match(sevs: &[u64], not_sev: u64, start: usize) -> usize {
    // SAFETY: each arm is gated on the matching runtime feature
    // detection, satisfying the `#[target_feature]` precondition.
    match simd_level() {
        SimdLevel::Avx2 => unsafe { find_sev_match_avx2(sevs, not_sev, start) },
        SimdLevel::Sse41 => unsafe { find_sev_match_sse41(sevs, not_sev, start) },
        SimdLevel::Scalar => find_sev_match_scalar(sevs, not_sev, start),
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub(crate) fn find_sev_match(sevs: &[u64], not_sev: u64, start: usize) -> usize {
    find_sev_match_scalar(sevs, not_sev, start)
}

/// Return the smallest index `i >= start` with
/// `(divmasks[i] & not_divmask) == 0`, or `divmasks.len()` if no
/// such index exists.
///
/// Algorithmically identical to [`find_sev_match`] — the mask
/// semantics differ (per-variable exponent ranges instead of just
/// "nonzero?"), but the per-u64 subset check is the same. Kept as a
/// distinct symbol so the call site in `bba::find_divisor_idx` reads
/// as "scan divmasks", and a future divergence (e.g. a 128-bit
/// divmask) lands in just this function. ADR-025.
#[inline]
pub(crate) fn find_divmask_match(divmasks: &[u64], not_divmask: u64, start: usize) -> usize {
    find_sev_match(divmasks, not_divmask, start)
}

/// Scalar implementation of [`find_sev_match`]. Non-x86_64 path and
/// the reference oracle for the SIMD unit tests.
#[inline]
pub(crate) fn find_sev_match_scalar(sevs: &[u64], not_sev: u64, start: usize) -> usize {
    let len = sevs.len();
    let mut idx = start;
    while idx < len {
        if (sevs[idx] & not_sev) == 0 {
            return idx;
        }
        idx += 1;
    }
    len
}

/// AVX2 implementation of [`find_sev_match`]. Mirrors Singular's
/// `kSevScanAVX2` (`~/Singular-next-opt/kernel/GBEngine/kstd2.cc:74`).
///
/// # Safety
/// Requires AVX2 at runtime; the caller in [`find_sev_match`] gates
/// this behind `is_x86_feature_detected!("avx2")`. Every load is
/// bounds-checked against `sevs.len()` before being issued.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn find_sev_match_avx2(sevs: &[u64], not_sev: u64, start: usize) -> usize {
    use std::arch::x86_64::*;
    let len = sevs.len();
    let mut j = start;
    let ptr = sevs.as_ptr();

    // SAFETY: every load is bounded by the surrounding `while j + N
    // <= len` check before issuing, so the 256-bit loads stay inside
    // the slice.
    unsafe {
        let vnot_sev = _mm256_set1_epi64x(not_sev as i64);
        let vzero = _mm256_setzero_si256();

        // Main loop: 16 entries per iteration. Singular's pattern:
        // four 4-wide AND+CMPEQ operations, OR the masks together,
        // and only branch out when *some* batch matched.
        while j + 16 <= len {
            let v1 = _mm256_loadu_si256(ptr.add(j) as *const __m256i);
            let v2 = _mm256_loadu_si256(ptr.add(j + 4) as *const __m256i);
            let v3 = _mm256_loadu_si256(ptr.add(j + 8) as *const __m256i);
            let v4 = _mm256_loadu_si256(ptr.add(j + 12) as *const __m256i);
            let a1 = _mm256_and_si256(v1, vnot_sev);
            let a2 = _mm256_and_si256(v2, vnot_sev);
            let a3 = _mm256_and_si256(v3, vnot_sev);
            let a4 = _mm256_and_si256(v4, vnot_sev);
            let c1 = _mm256_cmpeq_epi64(a1, vzero);
            let c2 = _mm256_cmpeq_epi64(a2, vzero);
            let c3 = _mm256_cmpeq_epi64(a3, vzero);
            let c4 = _mm256_cmpeq_epi64(a4, vzero);
            let m1 = _mm256_movemask_epi8(c1) as u32;
            let m2 = _mm256_movemask_epi8(c2) as u32;
            let m3 = _mm256_movemask_epi8(c3) as u32;
            let m4 = _mm256_movemask_epi8(c4) as u32;
            if (m1 | m2 | m3 | m4) != 0 {
                // Find the first matching qword across the four batches.
                // Each set qword has all 8 of its movemask bits set,
                // so trailing_zeros / 8 is the qword index.
                if m1 != 0 {
                    return j + (m1.trailing_zeros() / 8) as usize;
                }
                if m2 != 0 {
                    return j + 4 + (m2.trailing_zeros() / 8) as usize;
                }
                if m3 != 0 {
                    return j + 8 + (m3.trailing_zeros() / 8) as usize;
                }
                return j + 12 + (m4.trailing_zeros() / 8) as usize;
            }
            j += 16;
        }

        // Tail: one 4-wide batch at a time.
        while j + 4 <= len {
            let v = _mm256_loadu_si256(ptr.add(j) as *const __m256i);
            let a = _mm256_and_si256(v, vnot_sev);
            let c = _mm256_cmpeq_epi64(a, vzero);
            let m = _mm256_movemask_epi8(c) as u32;
            if m != 0 {
                return j + (m.trailing_zeros() / 8) as usize;
            }
            j += 4;
        }
    }

    // Scalar tail (0..3 elements).
    find_sev_match_scalar(sevs, not_sev, j)
}

/// SSE4.1 implementation of [`find_sev_match`] — the AVX2 shape at
/// half width (8-entry main loop of four 2-wide batches, 2-wide
/// tail). Mirrors Singular's `kSevScanSSE4`. `_mm_cmpeq_epi64` is the
/// SSE4.1 instruction that gates this path.
///
/// # Safety
/// Requires SSE4.1 at runtime; the caller in [`find_sev_match`] gates
/// this behind `is_x86_feature_detected!("sse4.1")`. Every load is
/// bounds-checked against `sevs.len()` before being issued.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn find_sev_match_sse41(sevs: &[u64], not_sev: u64, start: usize) -> usize {
    use std::arch::x86_64::*;
    let len = sevs.len();
    let mut j = start;
    let ptr = sevs.as_ptr();

    // SAFETY: every 128-bit load is bounded by the surrounding
    // `while j + N <= len` check before issuing.
    unsafe {
        let vnot_sev = _mm_set1_epi64x(not_sev as i64);
        let vzero = _mm_setzero_si128();

        // Main loop: 8 entries per iteration (four 2-wide batches).
        while j + 8 <= len {
            let v1 = _mm_loadu_si128(ptr.add(j) as *const __m128i);
            let v2 = _mm_loadu_si128(ptr.add(j + 2) as *const __m128i);
            let v3 = _mm_loadu_si128(ptr.add(j + 4) as *const __m128i);
            let v4 = _mm_loadu_si128(ptr.add(j + 6) as *const __m128i);
            let a1 = _mm_and_si128(v1, vnot_sev);
            let a2 = _mm_and_si128(v2, vnot_sev);
            let a3 = _mm_and_si128(v3, vnot_sev);
            let a4 = _mm_and_si128(v4, vnot_sev);
            let c1 = _mm_cmpeq_epi64(a1, vzero);
            let c2 = _mm_cmpeq_epi64(a2, vzero);
            let c3 = _mm_cmpeq_epi64(a3, vzero);
            let c4 = _mm_cmpeq_epi64(a4, vzero);
            // movemask_epi8 yields 16 bits over 16 bytes; a matching
            // 64-bit lane sets all 8 of its bytes, so trailing_zeros
            // / 8 is the in-batch qword index (0 or 1).
            let m1 = _mm_movemask_epi8(c1) as u32;
            let m2 = _mm_movemask_epi8(c2) as u32;
            let m3 = _mm_movemask_epi8(c3) as u32;
            let m4 = _mm_movemask_epi8(c4) as u32;
            if (m1 | m2 | m3 | m4) != 0 {
                if m1 != 0 {
                    return j + (m1.trailing_zeros() / 8) as usize;
                }
                if m2 != 0 {
                    return j + 2 + (m2.trailing_zeros() / 8) as usize;
                }
                if m3 != 0 {
                    return j + 4 + (m3.trailing_zeros() / 8) as usize;
                }
                return j + 6 + (m4.trailing_zeros() / 8) as usize;
            }
            j += 8;
        }

        // Tail: one 2-wide batch at a time.
        while j + 2 <= len {
            let v = _mm_loadu_si128(ptr.add(j) as *const __m128i);
            let a = _mm_and_si128(v, vnot_sev);
            let c = _mm_cmpeq_epi64(a, vzero);
            let m = _mm_movemask_epi8(c) as u32;
            if m != 0 {
                return j + (m.trailing_zeros() / 8) as usize;
            }
            j += 2;
        }
    }

    // Scalar tail (0..1 elements).
    find_sev_match_scalar(sevs, not_sev, j)
}

// =====================================================================
// Superset variant for "subset_mask ⊆ sevs[idx]" — used by ADR-009's
// chain-criterion sweep, where the question is "does fixed pair a's
// sev fit inside iterating pair c's sev" (i.e., a divides c). This is
// the dual of `find_sev_match` and uses `andnot` to test
// `(~c_sev & subset_mask) == 0` per qword.
// =====================================================================

/// Return the smallest index `i >= start` with
/// `(subset_mask & !sevs[i]) == 0`, or `sevs.len()` if no such index
/// exists.
///
/// Equivalent to "find the first `sevs[i]` that is a *superset* of
/// `subset_mask`" — every set bit in `subset_mask` is also set in
/// `sevs[i]`. This is exactly the sev pre-filter for the divides
/// predicate: a candidate may pass `divides` only if its sev is a
/// superset of the divider's sev. Used by `gm::chain_crit_normal`'s
/// ADR-009 B-internal sweep. Runtime-dispatched (ADR-027).
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn find_sev_superset_match(sevs: &[u64], subset_mask: u64, start: usize) -> usize {
    // SAFETY: each arm is gated on the matching runtime feature
    // detection, satisfying the `#[target_feature]` precondition.
    match simd_level() {
        SimdLevel::Avx2 => unsafe { find_sev_superset_match_avx2(sevs, subset_mask, start) },
        SimdLevel::Sse41 => unsafe { find_sev_superset_match_sse41(sevs, subset_mask, start) },
        SimdLevel::Scalar => find_sev_superset_match_scalar(sevs, subset_mask, start),
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub(crate) fn find_sev_superset_match(sevs: &[u64], subset_mask: u64, start: usize) -> usize {
    find_sev_superset_match_scalar(sevs, subset_mask, start)
}

/// Divmask analogue of [`find_sev_superset_match`] (ADR-025). Used by
/// the chain-criterion B-internal sweep, which asks "does the outer
/// pair's lcm divide the iterating inner pair's lcm" — i.e.
/// `divmask(outer) ⊆ divmask(inner)`. Algorithmically identical;
/// named separately so the call site reads as the bitmap semantic it
/// consults.
#[inline]
pub(crate) fn find_divmask_superset_match(
    divmasks: &[u64],
    subset_mask: u64,
    start: usize,
) -> usize {
    find_sev_superset_match(divmasks, subset_mask, start)
}

/// Scalar implementation of [`find_sev_superset_match`].
#[inline]
pub(crate) fn find_sev_superset_match_scalar(
    sevs: &[u64],
    subset_mask: u64,
    start: usize,
) -> usize {
    let len = sevs.len();
    let mut idx = start;
    while idx < len {
        if (subset_mask & !sevs[idx]) == 0 {
            return idx;
        }
        idx += 1;
    }
    len
}

/// AVX2 implementation of [`find_sev_superset_match`].
///
/// Same shape as [`find_sev_match_avx2`] but the per-batch op is
/// `_mm256_andnot_si256(c_sev, vsubset)` (computes
/// `~c_sev & subset_mask`), zero per qword iff `subset_mask`'s bits
/// are a subset of that qword's bits.
///
/// # Safety
/// Same as [`find_sev_match_avx2`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn find_sev_superset_match_avx2(
    sevs: &[u64],
    subset_mask: u64,
    start: usize,
) -> usize {
    use std::arch::x86_64::*;
    let len = sevs.len();
    let mut j = start;
    let ptr = sevs.as_ptr();
    unsafe {
        let vsubset = _mm256_set1_epi64x(subset_mask as i64);
        let vzero = _mm256_setzero_si256();

        while j + 16 <= len {
            let v1 = _mm256_loadu_si256(ptr.add(j) as *const __m256i);
            let v2 = _mm256_loadu_si256(ptr.add(j + 4) as *const __m256i);
            let v3 = _mm256_loadu_si256(ptr.add(j + 8) as *const __m256i);
            let v4 = _mm256_loadu_si256(ptr.add(j + 12) as *const __m256i);
            // _mm256_andnot_si256(a, b) = (~a) & b → `~sevs & subset`.
            let a1 = _mm256_andnot_si256(v1, vsubset);
            let a2 = _mm256_andnot_si256(v2, vsubset);
            let a3 = _mm256_andnot_si256(v3, vsubset);
            let a4 = _mm256_andnot_si256(v4, vsubset);
            let c1 = _mm256_cmpeq_epi64(a1, vzero);
            let c2 = _mm256_cmpeq_epi64(a2, vzero);
            let c3 = _mm256_cmpeq_epi64(a3, vzero);
            let c4 = _mm256_cmpeq_epi64(a4, vzero);
            let m1 = _mm256_movemask_epi8(c1) as u32;
            let m2 = _mm256_movemask_epi8(c2) as u32;
            let m3 = _mm256_movemask_epi8(c3) as u32;
            let m4 = _mm256_movemask_epi8(c4) as u32;
            if (m1 | m2 | m3 | m4) != 0 {
                if m1 != 0 {
                    return j + (m1.trailing_zeros() / 8) as usize;
                }
                if m2 != 0 {
                    return j + 4 + (m2.trailing_zeros() / 8) as usize;
                }
                if m3 != 0 {
                    return j + 8 + (m3.trailing_zeros() / 8) as usize;
                }
                return j + 12 + (m4.trailing_zeros() / 8) as usize;
            }
            j += 16;
        }
        while j + 4 <= len {
            let v = _mm256_loadu_si256(ptr.add(j) as *const __m256i);
            let a = _mm256_andnot_si256(v, vsubset);
            let c = _mm256_cmpeq_epi64(a, vzero);
            let m = _mm256_movemask_epi8(c) as u32;
            if m != 0 {
                return j + (m.trailing_zeros() / 8) as usize;
            }
            j += 4;
        }
    }
    find_sev_superset_match_scalar(sevs, subset_mask, j)
}

/// SSE4.1 implementation of [`find_sev_superset_match`] — the AVX2
/// superset shape at half width. Mirrors `kSevScanSSE4`'s superset
/// form.
///
/// # Safety
/// Same as [`find_sev_match_sse41`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn find_sev_superset_match_sse41(
    sevs: &[u64],
    subset_mask: u64,
    start: usize,
) -> usize {
    use std::arch::x86_64::*;
    let len = sevs.len();
    let mut j = start;
    let ptr = sevs.as_ptr();
    unsafe {
        let vsubset = _mm_set1_epi64x(subset_mask as i64);
        let vzero = _mm_setzero_si128();

        while j + 8 <= len {
            let v1 = _mm_loadu_si128(ptr.add(j) as *const __m128i);
            let v2 = _mm_loadu_si128(ptr.add(j + 2) as *const __m128i);
            let v3 = _mm_loadu_si128(ptr.add(j + 4) as *const __m128i);
            let v4 = _mm_loadu_si128(ptr.add(j + 6) as *const __m128i);
            // _mm_andnot_si128(a, b) = (~a) & b → `~sevs & subset`.
            let a1 = _mm_andnot_si128(v1, vsubset);
            let a2 = _mm_andnot_si128(v2, vsubset);
            let a3 = _mm_andnot_si128(v3, vsubset);
            let a4 = _mm_andnot_si128(v4, vsubset);
            let c1 = _mm_cmpeq_epi64(a1, vzero);
            let c2 = _mm_cmpeq_epi64(a2, vzero);
            let c3 = _mm_cmpeq_epi64(a3, vzero);
            let c4 = _mm_cmpeq_epi64(a4, vzero);
            let m1 = _mm_movemask_epi8(c1) as u32;
            let m2 = _mm_movemask_epi8(c2) as u32;
            let m3 = _mm_movemask_epi8(c3) as u32;
            let m4 = _mm_movemask_epi8(c4) as u32;
            if (m1 | m2 | m3 | m4) != 0 {
                if m1 != 0 {
                    return j + (m1.trailing_zeros() / 8) as usize;
                }
                if m2 != 0 {
                    return j + 2 + (m2.trailing_zeros() / 8) as usize;
                }
                if m3 != 0 {
                    return j + 4 + (m3.trailing_zeros() / 8) as usize;
                }
                return j + 6 + (m4.trailing_zeros() / 8) as usize;
            }
            j += 8;
        }
        while j + 2 <= len {
            let v = _mm_loadu_si128(ptr.add(j) as *const __m128i);
            let a = _mm_andnot_si128(v, vsubset);
            let c = _mm_cmpeq_epi64(a, vzero);
            let m = _mm_movemask_epi8(c) as u32;
            if m != 0 {
                return j + (m.trailing_zeros() / 8) as usize;
            }
            j += 2;
        }
    }
    find_sev_superset_match_scalar(sevs, subset_mask, j)
}

// =====================================================================
// Byte-parallel monomial kernels (ADR-038).
//
// The three hot per-variable monomial loops — `lcm` (componentwise
// max + capped-degree recompute), `divides` (componentwise ≤), and
// `monomials_are_coprime` (no variable positive in both) — operate on
// the 32-byte packed `[u64; 4]` block (ADR-005). With direct
// per-variable byte storage, each is a byte-parallel SIMD primitive:
//
// * `lcm`   → `_mm_max_epu8` per variable byte, then rewrite byte 31
//             (capped total-degree) from `_mm_sad_epu8` of the variable
//             bytes.
// * `divides` → `_mm_subs_epu8(self, other)` is all-zero over the
//             variable bytes iff every `e_i(self) ≤ e_i(other)`.
// * `coprime` → `_mm_min_epu8(a, b)` is all-zero over the variable
//             bytes iff no variable is positive in both.
//
// All three are SSE2 (baseline x86-64 — no runtime dispatch needed:
// `_mm_max_epu8`, `_mm_subs_epu8`, `_mm_min_epu8`, `_mm_sad_epu8` are
// all SSE2). A scalar-SWAR fallback keeps the crate portable to
// non-x86 targets. Each is a provably-exact drop-in for its scalar
// predecessor, validated by the proptests in `tests/monomial_props.rs`.
//
// `var_mask` is the ring's `cmp_flip_mask`: `0x7F` in every variable
// byte slot, `0x00` in byte 31 (total-degree cap) and the unused low
// bytes. ANDing a canonical packed block against it isolates the
// variable bytes (guard bit 7 is already zero, so `0x7F` loses
// nothing) and zeroes the degree byte — exactly the selector these
// kernels need.
//
// Singular comparison: `p_Lcm` (`libpolys/polys/monomials/p_polys.cc`)
// is a scalar per-variable loop; `p_LmShortDivisibleBy` is a SEV
// pre-filter plus a scalar confirm. Rust goes *past* Singular here
// (like ADR-035's divscan): the confirm step itself is vectorized.
// FLINT comparison: `mpoly` monomial ops (`mpoly_monomial_max`,
// `mpoly_monomials_cmp`) operate word-at-a-time over a packed limb
// array but FLINT's packing crosses field boundaries within a limb,
// so it cannot use saturating byte ops — it carries an explicit
// overflow-bit mask instead. Our fixed one-byte-per-variable layout
// (ADR-005) lets us use the byte-saturating SSE2 ops directly, which
// FLINT's variable-width bit-packing cannot.
// =====================================================================

/// Componentwise byte-max of two packed monomial blocks, returning the
/// raw `[u64; 4]` (variable bytes maxed; the degree byte is **not**
/// fixed up here — the caller recomputes it). Low/unused bytes are zero
/// in both inputs so their max stays zero.
#[inline]
pub(crate) fn packed_byte_max(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    #[cfg(target_arch = "x86_64")]
    {
        // SSE2 is baseline on x86-64; `_mm_max_epu8` is unconditionally
        // available, so no runtime dispatch.
        // SAFETY: SSE2 is guaranteed on every x86_64 target.
        return unsafe { packed_byte_max_sse2(a, b) };
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        packed_byte_max_swar(a, b)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn packed_byte_max_sse2(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    use std::arch::x86_64::*;
    // SAFETY: the two 16-byte loads cover exactly the 32-byte packed
    // blocks; `a`/`b` are `&[u64; 4]` so the reads are in-bounds and
    // aligned to at least 8 bytes (loadu tolerates any alignment).
    unsafe {
        let a0 = _mm_loadu_si128(a.as_ptr() as *const __m128i);
        let a1 = _mm_loadu_si128(a.as_ptr().add(2) as *const __m128i);
        let b0 = _mm_loadu_si128(b.as_ptr() as *const __m128i);
        let b1 = _mm_loadu_si128(b.as_ptr().add(2) as *const __m128i);
        let m0 = _mm_max_epu8(a0, b0);
        let m1 = _mm_max_epu8(a1, b1);
        let mut out = [0u64; 4];
        _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, m0);
        _mm_storeu_si128(out.as_mut_ptr().add(2) as *mut __m128i, m1);
        out
    }
}

/// SWAR byte-max fallback: per-byte unsigned max of two u64 words.
/// Used on non-x86 targets and as the kernel's portable reference.
/// Clarity over cleverness — this path never runs on the perf-critical
/// (x86_64) workload, so a straightforward per-byte loop is preferred
/// to a fragile bit-trick.
///
/// On x86_64 the production path is the SSE2 kernel, so this and its
/// `word_byte_*` helpers are only reached from the unit tests there
/// (which cross-check SIMD against SWAR) — hence the dead-code allow.
#[cfg_attr(target_arch = "x86_64", allow(dead_code))]
#[inline]
pub(crate) fn packed_byte_max_swar(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let mut out = [0u64; 4];
    for i in 0..4 {
        out[i] = word_byte_max(a[i], b[i]);
    }
    out
}

/// Per-byte unsigned max of two u64 words.
#[cfg_attr(target_arch = "x86_64", allow(dead_code))]
#[inline]
fn word_byte_max(a: u64, b: u64) -> u64 {
    let mut out = 0u64;
    for k in 0..8 {
        let sh = k * 8;
        let ba = (a >> sh) & 0xFF;
        let bb = (b >> sh) & 0xFF;
        out |= ba.max(bb) << sh;
    }
    out
}

/// Per-byte unsigned saturating subtract `a - b`: each byte is
/// `max(a_byte - b_byte, 0)`.
#[cfg_attr(target_arch = "x86_64", allow(dead_code))]
#[inline]
fn word_byte_subs(a: u64, b: u64) -> u64 {
    let mut out = 0u64;
    for k in 0..8 {
        let sh = k * 8;
        let ba = (a >> sh) & 0xFF;
        let bb = (b >> sh) & 0xFF;
        out |= ba.saturating_sub(bb) << sh;
    }
    out
}

/// Per-byte unsigned saturating subtract over the whole packed block.
#[inline]
pub(crate) fn packed_byte_subs(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: SSE2 guaranteed on x86_64.
        return unsafe { packed_byte_subs_sse2(a, b) };
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let mut out = [0u64; 4];
        for i in 0..4 {
            out[i] = word_byte_subs(a[i], b[i]);
        }
        out
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn packed_byte_subs_sse2(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    use std::arch::x86_64::*;
    // SAFETY: 16-byte loads over the 32-byte packed blocks; in-bounds.
    unsafe {
        let a0 = _mm_loadu_si128(a.as_ptr() as *const __m128i);
        let a1 = _mm_loadu_si128(a.as_ptr().add(2) as *const __m128i);
        let b0 = _mm_loadu_si128(b.as_ptr() as *const __m128i);
        let b1 = _mm_loadu_si128(b.as_ptr().add(2) as *const __m128i);
        let s0 = _mm_subs_epu8(a0, b0);
        let s1 = _mm_subs_epu8(a1, b1);
        let mut out = [0u64; 4];
        _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, s0);
        _mm_storeu_si128(out.as_mut_ptr().add(2) as *mut __m128i, s1);
        out
    }
}

/// Per-byte unsigned min over the whole packed block.
#[inline]
pub(crate) fn packed_byte_min(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: SSE2 guaranteed on x86_64.
        return unsafe { packed_byte_min_sse2(a, b) };
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        // min(a,b) = a - subs(a, b) per byte (a - max(a-b,0)).
        let mut out = [0u64; 4];
        for i in 0..4 {
            out[i] = a[i].wrapping_sub(word_byte_subs(a[i], b[i]));
        }
        out
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn packed_byte_min_sse2(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    use std::arch::x86_64::*;
    // SAFETY: 16-byte loads over the 32-byte packed blocks; in-bounds.
    unsafe {
        let a0 = _mm_loadu_si128(a.as_ptr() as *const __m128i);
        let a1 = _mm_loadu_si128(a.as_ptr().add(2) as *const __m128i);
        let b0 = _mm_loadu_si128(b.as_ptr() as *const __m128i);
        let b1 = _mm_loadu_si128(b.as_ptr().add(2) as *const __m128i);
        let m0 = _mm_min_epu8(a0, b0);
        let m1 = _mm_min_epu8(a1, b1);
        let mut out = [0u64; 4];
        _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, m0);
        _mm_storeu_si128(out.as_mut_ptr().add(2) as *mut __m128i, m1);
        out
    }
}

/// Sum of all bytes of a packed block (used after masking to the
/// variable bytes, to recompute the lcm's capped total degree). Uses
/// `_mm_sad_epu8` against zero on x86 (two 8-byte-lane sums), or a
/// SWAR byte sum on other targets. Caller must pre-mask the degree
/// byte and any non-variable bytes to zero.
#[inline]
pub(crate) fn packed_byte_sum(p: &[u64; 4]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: SSE2 guaranteed on x86_64.
        return unsafe { packed_byte_sum_sse2(p) };
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let mut s: u32 = 0;
        for &w in p.iter() {
            for k in 0..8 {
                s += ((w >> (k * 8)) & 0xFF) as u32;
            }
        }
        s
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn packed_byte_sum_sse2(p: &[u64; 4]) -> u32 {
    use std::arch::x86_64::*;
    // SAFETY: two 16-byte loads over the 32-byte packed block.
    unsafe {
        let v0 = _mm_loadu_si128(p.as_ptr() as *const __m128i);
        let v1 = _mm_loadu_si128(p.as_ptr().add(2) as *const __m128i);
        let zero = _mm_setzero_si128();
        // _mm_sad_epu8 sums the 8 absolute differences per 64-bit lane
        // into the low 16 bits of that lane; against zero it is the
        // byte-sum of each 8-byte lane.
        let s0 = _mm_sad_epu8(v0, zero);
        let s1 = _mm_sad_epu8(v1, zero);
        let s = _mm_add_epi64(s0, s1);
        // Two lane sums in bits [0..16) of each 64-bit half.
        let lo = _mm_cvtsi128_si64(s) as u64 & 0xFFFF;
        let hi = _mm_cvtsi128_si64(_mm_unpackhi_epi64(s, s)) as u64 & 0xFFFF;
        (lo + hi) as u32
    }
}

/// True iff `packed_byte_subs(a, b)` masked to the variable bytes is
/// all-zero — i.e. every variable byte of `a` is ≤ the corresponding
/// byte of `b`. `var_mask` selects the variable bytes (the ring's
/// `cmp_flip_mask`: 0x7F on variable bytes, 0 elsewhere).
#[inline]
pub(crate) fn packed_divides(a: &[u64; 4], b: &[u64; 4], var_mask: &[u64; 4]) -> bool {
    let d = packed_byte_subs(a, b);
    // subs is ≤ 0x7F per variable byte (both inputs ≤ 0x7F there), so
    // ANDing with the 0x7F var_mask keeps the whole difference on the
    // variable bytes and discards the degree/low bytes.
    (d[0] & var_mask[0])
        | (d[1] & var_mask[1])
        | (d[2] & var_mask[2])
        | (d[3] & var_mask[3])
        == 0
}

/// True iff no variable byte is positive in both `a` and `b` — the
/// componentwise min over the variable bytes is all-zero. `var_mask`
/// selects the variable bytes (degree bytes, generally nonzero in
/// both, are masked out).
#[inline]
pub(crate) fn packed_coprime(a: &[u64; 4], b: &[u64; 4], var_mask: &[u64; 4]) -> bool {
    let m = packed_byte_min(a, b);
    (m[0] & var_mask[0])
        | (m[1] & var_mask[1])
        | (m[2] & var_mask[2])
        | (m[3] & var_mask[3])
        == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random `u64` array spanning all three
    /// loop bodies (main / narrow tail / scalar tail) of every
    /// backend.
    fn sample_sevs(len: usize) -> Vec<u64> {
        let mut sevs = Vec::with_capacity(len);
        let mut state: u64 = 0x00c0_ffee_d00d_face;
        for _ in 0..len {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            sevs.push(state);
        }
        sevs
    }

    const MASKS: [u64; 5] = [
        0u64,
        !0u64,
        0x0000_0000_FFFF_FFFFu64,
        0xAAAA_AAAA_5555_5555u64,
        0x0000_0000_0000_0001u64,
    ];
    const STARTS: [usize; 10] = [0, 1, 7, 8, 16, 17, 32, 96, 192, 199];

    /// Every available backend (scalar always; SSE4.1 / AVX2 when the
    /// running CPU supports them) must agree with the scalar oracle
    /// for both primitives, across masks, start indices, and the
    /// dispatched entry point. Exercising the backends explicitly —
    /// not only through dispatch — keeps coverage real on hosts where
    /// dispatch would pick a single path.
    #[test]
    fn simd_backends_match_scalar() {
        let sevs = sample_sevs(200);
        for &mask in &MASKS {
            for &start in &STARTS {
                let sub_ref = find_sev_match_scalar(&sevs, mask, start);
                let sup_ref = find_sev_superset_match_scalar(&sevs, mask, start);

                // Dispatched entry points (whatever the host picked).
                assert_eq!(find_sev_match(&sevs, mask, start), sub_ref);
                assert_eq!(find_sev_superset_match(&sevs, mask, start), sup_ref);

                #[cfg(target_arch = "x86_64")]
                {
                    if std::is_x86_feature_detected!("sse4.1") {
                        // SAFETY: gated on runtime sse4.1 detection.
                        unsafe {
                            assert_eq!(
                                find_sev_match_sse41(&sevs, mask, start),
                                sub_ref,
                                "sse41 sub mask={mask:#x} start={start}"
                            );
                            assert_eq!(
                                find_sev_superset_match_sse41(&sevs, mask, start),
                                sup_ref,
                                "sse41 sup mask={mask:#x} start={start}"
                            );
                        }
                    }
                    if std::is_x86_feature_detected!("avx2") {
                        // SAFETY: gated on runtime avx2 detection.
                        unsafe {
                            assert_eq!(
                                find_sev_match_avx2(&sevs, mask, start),
                                sub_ref,
                                "avx2 sub mask={mask:#x} start={start}"
                            );
                            assert_eq!(
                                find_sev_superset_match_avx2(&sevs, mask, start),
                                sup_ref,
                                "avx2 sup mask={mask:#x} start={start}"
                            );
                        }
                    }
                }
            }
        }

        // Edge cases on the dispatched path.
        assert_eq!(find_sev_superset_match(&[], 0u64, 0), 0);
        assert_eq!(find_sev_superset_match(&[5u64], 0u64, 0), 0); // empty mask ⊆ anything
        assert_eq!(find_sev_superset_match(&[5u64], !0u64, 0), 1); // full mask ⊄ partial
        assert_eq!(find_sev_superset_match(&[7u64, 5u64], 1u64, 0), 0); // 1 ⊆ 7
        assert_eq!(find_sev_superset_match(&[2u64, 5u64], 4u64, 0), 1); // 4 ⊄ 2; 4 ⊆ 5
        assert_eq!(find_sev_match(&[1u64, 2u64], 1u64, 0), 1); // 1&1≠0; 2&1==0
    }

    // ---- ADR-038 byte-parallel monomial kernels ----

    /// Naive per-byte reference for the byte ops, independent of both
    /// the SIMD and the SWAR production paths.
    fn ref_byte_op(a: &[u64; 4], b: &[u64; 4], op: impl Fn(u8, u8) -> u8) -> [u64; 4] {
        let mut out = [0u64; 4];
        for i in 0..4 {
            for k in 0..8 {
                let sh = k * 8;
                let ba = ((a[i] >> sh) & 0xFF) as u8;
                let bb = ((b[i] >> sh) & 0xFF) as u8;
                out[i] |= (op(ba, bb) as u64) << sh;
            }
        }
        out
    }

    fn sample_packed(seed: u64) -> [u64; 4] {
        let mut s = seed;
        let mut out = [0u64; 4];
        for w in out.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *w = s;
        }
        out
    }

    #[test]
    fn byte_ops_match_reference() {
        for seed in 0u64..256 {
            let a = sample_packed(seed);
            let b = sample_packed(seed ^ 0xdead_beef);

            let want_max = ref_byte_op(&a, &b, |x, y| x.max(y));
            let want_min = ref_byte_op(&a, &b, |x, y| x.min(y));
            let want_subs = ref_byte_op(&a, &b, |x, y| x.saturating_sub(y));

            // SIMD / dispatched production path.
            assert_eq!(packed_byte_max(&a, &b), want_max, "max seed={seed}");
            assert_eq!(packed_byte_min(&a, &b), want_min, "min seed={seed}");
            assert_eq!(packed_byte_subs(&a, &b), want_subs, "subs seed={seed}");

            // SWAR reference path (also exercised on x86, where it is
            // otherwise the non-default arm) — must agree byte-for-byte.
            assert_eq!(packed_byte_max_swar(&a, &b), want_max, "swar-max seed={seed}");
            assert_eq!(word_byte_max(a[0], b[0]), want_max[0], "word-max seed={seed}");
            assert_eq!(
                word_byte_subs(a[0], b[0]),
                want_subs[0],
                "word-subs seed={seed}"
            );

            // Byte-sum: sum every byte of `a`.
            let want_sum: u32 = (0..4)
                .flat_map(|i| (0..8).map(move |k| ((a[i] >> (k * 8)) & 0xFF) as u32))
                .sum();
            assert_eq!(packed_byte_sum(&a), want_sum, "sum seed={seed}");
        }
    }

    #[test]
    fn byte_sum_handles_max_bytes() {
        // All bytes 0xFF: 32 * 255 = 8160, fits the two 16-bit SAD
        // lane accumulators (each lane sums 8*255 = 2040).
        let all = [!0u64; 4];
        assert_eq!(packed_byte_sum(&all), 32 * 255);
    }
}
