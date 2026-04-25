//! Codegen probe for `Monomial::cmp_degrevlex`.
//!
//! Mirrors `mul_probe.rs`. Build with `cargo build --release --example
//! cmp_probe` and disassemble the resulting binary's `probe_cmp` symbol
//! (e.g. `objdump -d --no-show-raw-insn target/release/examples/cmp_probe |
//! awk '/<probe_cmp>:/,/^$/'`) to inspect the codegen of the length-4
//! degrevlex compare.
//!
//! See ADR-022 (codegen audit + tightening for `cmp_degrevlex`) and the
//! length-4 specialisation task notes in
//! `~/project/docs/profile-rustgb-cmp-len4-staging-5101449.md`.

use rustgb::field::Field;
use rustgb::monomial::Monomial;
use rustgb::ordering::MonoOrder;
use rustgb::ring::Ring;
use std::cmp::Ordering;

#[inline(never)]
#[unsafe(no_mangle)]
pub extern "C" fn probe_cmp(a: &Monomial, b: &Monomial, ring: &Ring) -> i32 {
    match a.cmp(b, ring) {
        Ordering::Less => -1,
        Ordering::Equal => 0,
        Ordering::Greater => 1,
    }
}

fn main() {
    let ring = Ring::new(25, MonoOrder::DegRevLex, Field::new(32003).unwrap()).unwrap();
    let a = Monomial::from_exponents(&ring, &vec![1u32; 25]).unwrap();
    let b = Monomial::from_exponents(&ring, &vec![2u32; 25]).unwrap();
    let r = probe_cmp(&a, &b, &ring);
    eprintln!("{}", r);
}
