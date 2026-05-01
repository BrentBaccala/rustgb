//! # rustgb
//!
//! Polynomial layer for the Singular Groebner-basis port.
//!
//! This crate is an early milestone of the port described in
//! `~/project/docs/rust-bba-port-plan.md`. It supplies the ring,
//! field, monomial, polynomial, and geobucket primitives that a
//! later `bba` driver will build on. There is deliberately **no**
//! S/T/L, **no** S-pair queue, **no** FFI, and **no** parallelism
//! in this crate yet.
//!
//! ## Current scope
//!
//! * **Field:** Z/p with `2 ≤ p < 2^31`, Barrett-reduced modular mul.
//! * **Ordering:** `DegRevLex` only.
//! * **Exponent width:** 8 bits per variable, up to
//!   [`MAX_VARS`](ring::MAX_VARS) variables.
//! * **Polynomial:** parallel `Vec<Coeff>` / `Vec<Monomial>` with
//!   cached leading-term metadata.
//!
//! Public types are `Send + Sync` and intended to be shared through
//! `Arc<Ring>` once the driver lands.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod bba;
pub mod bset;
pub mod computation;
pub mod ffi;
pub mod field;
pub mod gm;
pub mod kbucket;
pub mod lobject;
pub mod lset;
#[cfg(feature = "flat_lset")]
pub mod lset_flat;
pub mod monomial;
pub mod ordering;
pub mod pair;
pub mod parallel;
pub mod poly;
pub mod reducer;
pub mod ring;
pub mod sbasis;
mod simd;

pub use bba::compute_gb;
pub use computation::{Computation, SharedLSet, SharedSBasis};
pub use parallel::{CancelHandle, Cancelled, compute_gb_parallel};
pub use bset::BSet;
pub use field::{Coeff, Field};
pub use kbucket::KBucket;
pub use lobject::LObject;
#[cfg(not(feature = "flat_lset"))]
pub use lset::LSet;
#[cfg(feature = "flat_lset")]
pub use lset_flat::LSet;
pub use monomial::Monomial;
pub use ordering::MonoOrder;
pub use pair::{Pair, PairKey};
pub use poly::Poly;
pub use ring::Ring;
pub use sbasis::SBasis;

// Compile-time Send + Sync check on the key public types.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Ring>();
    assert_send_sync::<Field>();
    assert_send_sync::<Monomial>();
    assert_send_sync::<Poly>();
    assert_send_sync::<Pair>();
    assert_send_sync::<SBasis>();
    assert_send_sync::<LSet>();
    assert_send_sync::<BSet>();
};

// KBucket and LObject are Send but deliberately not Sync
// (per-thread ownership).
const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<KBucket>();
    assert_send::<LObject>();
};
