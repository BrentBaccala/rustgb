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
}
