//! Minimal FxHash (the rustc-hash algorithm) for the small
//! fixed-size integer keys used by the L-set / B-set pair indices
//! (ADR-028).
//!
//! The std default `HashMap`/`HashSet` hasher is `RandomState` =
//! SipHash-1-3, a keyed cryptographic hash sized for HashDoS
//! resistance. The pair indices key on a non-adversarial,
//! engine-internal `(u32, u32)` (8 bytes); SipHash's per-key cost
//! dominated `by_indices` lookup/insert in the staging-5101449
//! profile (~3.7 % wall, `hash_one` + `DefaultHasher::write`).
//!
//! Singular's `next-opt` keys the analogous `LSet::pair_index`
//! (`std::unordered_map<std::pair<poly,poly>, iterator,
//! PolyPairHash>`, `kernel/GBEngine/kutil.h`) with a trivial
//! `h1 ^ (h2 << 1)` over two pointer hashes — essentially free, which
//! is why it never surfaces in the C++ profile. FxHash gives rustgb
//! the same: one multiply-rotate per machine word, no external crate
//! (the crate stays stdlib-only).
//!
//! Not collision-resistant against an adversary — fine here, the keys
//! are basis indices the engine generates itself.

use std::hash::{BuildHasherDefault, Hasher};

/// `HashMap` specialised to the [`FxHasher`].
pub(crate) type FxHashMap<K, V> =
    std::collections::HashMap<K, V, BuildHasherDefault<FxHasher>>;

/// `HashSet` specialised to the [`FxHasher`].
pub(crate) type FxHashSet<T> =
    std::collections::HashSet<T, BuildHasherDefault<FxHasher>>;

/// rustc-hash's constant (the golden-ratio odd multiplier).
const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

/// FxHasher: `hash = (hash.rotate_left(5) ^ word).wrapping_mul(SEED)`
/// per word consumed. `Default` yields a zeroed state, so
/// `BuildHasherDefault<FxHasher>` is a complete `BuildHasher`.
#[derive(Default)]
pub(crate) struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }

    /// Generic byte path — consumes 8 bytes at a time. The hot keys
    /// (`(u32, u32)`, `usize`) route through the integer
    /// specialisations below, not here.
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut buf = [0u8; 8];
            buf[..chunk.len()].copy_from_slice(chunk);
            self.add(u64::from_le_bytes(buf));
        }
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(i as u64);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }
}
