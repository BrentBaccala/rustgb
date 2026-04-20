# rustgb

Polynomial layer for the Singular Groebner-basis port.

This crate is the **first milestone** of the port described in
[`~/project/docs/rust-bba-port-plan.md`](../project/docs/rust-bba-port-plan.md)
(§4–6). It supplies the ring, field, monomial, and polynomial
primitives that a later `bba` driver will build on.

It deliberately does **not** yet include:

- the bba driver, S/T/L data structures, S-pair queue, or
  `enterpairs` / `chainCritNormal`
- a kBucket / geobucket reducer (polynomial primitives only)
- an FFI surface or Singular integration
- parallelism or SIMD
- any monomial ordering other than degrevlex
- any coefficient ring other than Z/p for a prime `p < 2^31`

Those are all follow-up tasks; see the port plan for the broader
roadmap.

## Design choices

- **Field: Z/p, Barrett reduction.** `p` is a user-supplied prime less
  than 2^31. Barrett reduction avoids a division on every modular
  multiplication. See `src/field.rs`.
- **Ordering: degrevlex only.** Hard-coded in the comparator for now.
  The `MonoOrder` enum is public so call sites learn the name, but
  only `DegRevLex` is accepted. See `src/ordering.rs`.
- **Monomial: 8 bits per variable, 4×u64 words.** This accommodates up
  to 31 variables (one byte is reserved for a capped total-degree).
  Layout is tuned so that a lex-compare of the four words (MSB first)
  yields the degrevlex order; a cached `u32` total-degree handles the
  rare case where an individual exponent would exceed 255. See
  `src/monomial.rs`.
- **Polynomial: parallel `Vec<Coeff>` + `Vec<Monomial>`.** Cached
  leading-term metadata (`lm_sev`, `lm_coeff`, `lm_deg`) lives on the
  struct so the bba sweep can peek at a candidate without touching the
  term arrays. Terms are strictly descending under the ring's
  ordering. See `src/poly.rs`.
- **`assert_canonical` on every type.** Invariants are checked from
  tests and `debug_assert!` sites the way FLINT's
  `nmod_mpoly_assert_canonical` does.
- **`Send + Sync` everywhere.** The public types are ready for the
  parallel driver to share through `Arc<Ring>`, but no actual
  threading code lives here.

## Layout

```
src/
├── lib.rs        crate root and re-exports
├── ring.rs       Ring struct, BITS_PER_VAR=8, MAX_VARS=31
├── ordering.rs   MonoOrder::DegRevLex
├── field.rs      Z/p with Barrett reduction (u32 coefficient)
├── monomial.rs   packed-exponent Monomial + arithmetic + cmp
└── poly.rs       Poly + add / sub / mul / sub_mul_term / monic

tests/
├── field_props.rs     ~11 proptest properties, 2048 cases each
├── monomial_props.rs  ~13 proptest properties, 1024 cases each
└── poly_props.rs      ~13 proptest properties + fixed fixtures

examples/
└── sanity.rs     rough timing for Poly::add and Poly::sub_mul_term
```

## Reference reading

- [`~/project/docs/rust-bba-port-plan.md`](../project/docs/rust-bba-port-plan.md) —
  full roadmap and architectural rationale.
- [`~/project/docs/rust-polynomial-crates-survey.md`](../project/docs/rust-polynomial-crates-survey.md) —
  the build-vs-buy analysis that settled the pure-Rust choice.
- Mathicgb's `~/mathicgb/src/mathicgb/{Poly,MonoMonoid,PrimeField}.hpp` —
  structural templates (GPL-2+).
- FLINT's `~/flint/src/nmod_mpoly/` — test patterns used as a model
  for the proptest suite.
- feanor-math's `zn_64` — structural model for the Z/p implementation.

None of the above was vendored or directly copied; algorithms are
re-derived in Rust. License: GPL-3.0-or-later.

## Building

```bash
cargo build --release
cargo test
cargo clippy --all-targets -- -D warnings
cargo run --release --example sanity
```
