//! # rustgb
//!
//! Polynomial layer for the Singular Groebner-basis port.
//!
//! This crate is the first milestone of the port described in
//! `~/project/docs/rust-bba-port-plan.md`. It supplies the ring,
//! field, monomial, and polynomial primitives that a later `bba`
//! driver will build on. There is deliberately **no** S/T/L,
//! **no** S-pair queue, **no** kBucket reducer, **no** FFI, and
//! **no** parallelism in this crate yet.
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

pub mod field;
pub mod monomial;
pub mod ordering;
pub mod poly;
pub mod ring;

pub use field::{Coeff, Field};
pub use monomial::Monomial;
pub use ordering::MonoOrder;
pub use poly::Poly;
pub use ring::Ring;

// Compile-time Send + Sync check on the key public types.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Ring>();
    assert_send_sync::<Field>();
    assert_send_sync::<Monomial>();
    assert_send_sync::<Poly>();
};
