use rustgb::field::Field;
use rustgb::monomial::Monomial;
use rustgb::ordering::MonoOrder;
use rustgb::ring::Ring;

#[inline(never)]
#[unsafe(no_mangle)]
pub extern "C" fn probe_mul(a: &Monomial, b: &Monomial, ring: &Ring, out: &mut Option<Monomial>) {
    *out = a.mul(b, ring);
}

fn main() {
    let ring = Ring::new(25, MonoOrder::DegRevLex, Field::new(32003).unwrap()).unwrap();
    let a = Monomial::from_exponents(&ring, &vec![1u32; 25]).unwrap();
    let b = Monomial::from_exponents(&ring, &vec![2u32; 25]).unwrap();
    let mut out: Option<Monomial> = None;
    probe_mul(&a, &b, &ring, &mut out);
    eprintln!("{:?}", out.is_some());
}
