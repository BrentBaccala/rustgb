//! Monomial orderings.
//!
//! For this bootstrap task we only support `DegRevLex` (degree-reverse-lex).
//! Other orderings (`Lex`, block orderings) are enumerated in the port plan
//! (`~/project/docs/rust-bba-port-plan.md` §6.1) but are deferred until the
//! bba driver is operational.

/// The monomial ordering of a [`Ring`](crate::ring::Ring).
///
/// Currently only [`MonoOrder::DegRevLex`] is implemented; operations on
/// other variants will panic. The enum is defined now so call sites get
/// used to naming it — future variants can be added without rewriting
/// every signature.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MonoOrder {
    /// Graded reverse lexicographic order: compare total degree first
    /// (larger = greater); break ties by the leftmost differing variable,
    /// where the *smaller* exponent on that variable wins.
    DegRevLex,
}
