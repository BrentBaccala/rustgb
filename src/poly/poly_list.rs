//! Singly-linked-list backend for [`Poly`].
//!
//! Enabled by the `linked_list_poly` Cargo feature. The default is
//! the flat-array backend in [`poly_vec`](super::poly_vec); see ADR-001
//! for the profile evidence that favours it, and ADR-014 for the
//! rationale behind keeping this second backend available.
//!
//! The shape here is close to Singular's `spolyrec` storage: each
//! node owns a coefficient, a monomial, and a `Box<Node>` pointing
//! at the next node (or `None` at the tail). `drop_leading_in_place`
//! is O(1) via a take-and-replace on the head slot. A custom
//! [`Drop`] impl walks the chain iteratively so a million-term poly
//! doesn't overflow the stack.
//!
//! Invariants (checked by [`Poly::assert_canonical`]):
//!
//! 1. `len` equals the number of reachable nodes from `head`.
//! 2. All coefficients are in canonical form `0 < c < p` (zeros excluded).
//! 3. Monomials are strictly descending under the ring's ordering (no
//!    duplicates, no unsorted runs).
//! 4. `lm_*` fields match the head node's coefficient / monomial / deg
//!    when nonempty.

use crate::field::Coeff;
use crate::monomial::Monomial;
use crate::ring::Ring;

/// A sparse polynomial in a [`Ring`], stored as a singly linked list.
///
/// See module documentation for invariants. `Send + Sync`: the
/// recursive `Box<Node>` chain contains only `Coeff` (u32) and
/// `Monomial` (POD struct) plus pointer fields, all of which are
/// themselves `Send + Sync`.
///
/// The head node is the leading term; descendants are in strictly-
/// descending order under the ring's monomial ordering.
pub struct Poly {
    /// First node (the leading term), or `None` for the zero poly.
    head: Option<Box<Node>>,
    /// Number of live nodes reachable from `head`. Maintained on
    /// every mutation so `len()` stays O(1).
    len: usize,
    /// Cached leading-term sev (`head.mono.sev()`); 0 when empty.
    lm_sev: u64,
    /// Cached leading coefficient (`head.coeff`); 0 when empty.
    lm_coeff: Coeff,
    /// Cached leading monomial degree (`head.mono.total_deg()`);
    /// 0 when empty.
    lm_deg: u32,
}

/// One term's worth of storage in the linked list.
struct Node {
    coeff: Coeff,
    mono: Monomial,
    next: Option<Box<Node>>,
}

impl std::fmt::Debug for Poly {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut dbg = f.debug_struct("Poly");
        dbg.field("len", &self.len)
            .field("lm_coeff", &self.lm_coeff)
            .field("lm_sev", &self.lm_sev)
            .field("lm_deg", &self.lm_deg);
        // Walk and collect terms for debugging.
        let mut terms: Vec<(Coeff, &Monomial)> = Vec::with_capacity(self.len);
        let mut node = self.head.as_deref();
        while let Some(n) = node {
            terms.push((n.coeff, &n.mono));
            node = n.next.as_deref();
        }
        dbg.field("terms", &terms).finish()
    }
}

impl Drop for Poly {
    /// Iterative drop so a very long chain does not blow the stack
    /// via the default recursive `Box<Node>` destructor. Mirrors the
    /// canonical pattern from the Rust nomicon's linked-list exercise.
    fn drop(&mut self) {
        let mut cur = self.head.take();
        while let Some(mut boxed) = cur {
            // Detach `boxed.next` into `cur` before the outer drop
            // runs, so the boxed node we are about to release no
            // longer owns its successor.
            cur = boxed.next.take();
            // `boxed` is dropped here with `next == None`, so its
            // destructor is O(1) and non-recursive.
        }
    }
}

impl Clone for Poly {
    /// Deep clone: walks the chain, allocating fresh nodes.
    fn clone(&self) -> Self {
        // Build in reverse from a tail-first construction pattern?
        // Simpler: collect the terms into a temporary Vec so we can
        // reconstruct head-to-tail via tail-append. But we want to
        // avoid the Vec allocation. Instead, we do a forward walk
        // and maintain a pointer to the tail's `next` slot so each
        // new node attaches there.
        let mut out_head: Option<Box<Node>> = None;
        // `tail_slot` is the `&mut Option<Box<Node>>` where the next
        // fresh node should land. We start by pointing it at
        // `out_head`; after each insert we reborrow through the new
        // node's `next` field.
        let mut tail_slot: &mut Option<Box<Node>> = &mut out_head;
        let mut node = self.head.as_deref();
        while let Some(n) = node {
            let fresh = Box::new(Node {
                coeff: n.coeff,
                mono: n.mono.clone(),
                next: None,
            });
            *tail_slot = Some(fresh);
            // Reborrow: move `tail_slot` to point at the new node's
            // `next` slot. `as_mut()` gives us `&mut Box<Node>`, and
            // `&mut box.next` gives the slot we want.
            tail_slot = &mut tail_slot.as_mut().unwrap().next;
            node = n.next.as_deref();
        }
        Poly {
            head: out_head,
            len: self.len,
            lm_sev: self.lm_sev,
            lm_coeff: self.lm_coeff,
            lm_deg: self.lm_deg,
        }
    }
}

impl Poly {
    // ----- Constructors -----

    /// The zero polynomial.
    pub fn zero() -> Self {
        Self {
            head: None,
            len: 0,
            lm_sev: 0,
            lm_coeff: 0,
            lm_deg: 0,
        }
    }

    /// A polynomial with a single term `c * m`. Returns the zero
    /// polynomial if `c == 0`. `c` must already be reduced mod `p`.
    pub fn monomial(ring: &Ring, c: Coeff, m: Monomial) -> Self {
        debug_assert!(c < ring.field().p(), "coeff {c} not reduced");
        if c == 0 {
            return Self::zero();
        }
        let lm_sev = m.sev();
        let lm_deg = m.total_deg();
        Self {
            head: Some(Box::new(Node {
                coeff: c,
                mono: m,
                next: None,
            })),
            len: 1,
            lm_sev,
            lm_coeff: c,
            lm_deg,
        }
    }

    /// Build a polynomial from a sequence of `(coeff, monomial)` pairs
    /// already in strictly-descending monomial order with no
    /// duplicates and no zero coefficients. See the `poly_vec`
    /// counterpart for the caller contract.
    pub fn from_descending_terms_unchecked(
        ring: &Ring,
        terms: Vec<(Coeff, Monomial)>,
    ) -> Self {
        if terms.is_empty() {
            return Self::zero();
        }
        let p = ring.field().p();
        let len = terms.len();
        // Build head-to-tail by walking forward and reborrowing the
        // tail slot (same pattern as `Clone`).
        let mut head: Option<Box<Node>> = None;
        let mut tail_slot: &mut Option<Box<Node>> = &mut head;
        let mut prev_mono: Option<&Monomial> = None;
        let mut first_coeff: Coeff = 0;
        let mut first_mono_sev: u64 = 0;
        let mut first_mono_deg: u32 = 0;
        let mut first = true;
        for (c, m) in terms.iter() {
            debug_assert!(
                *c != 0,
                "from_descending_terms_unchecked: zero coeff"
            );
            debug_assert!(
                *c < p,
                "from_descending_terms_unchecked: unreduced coeff"
            );
            if let Some(prev) = prev_mono {
                debug_assert!(
                    prev.cmp(m, ring).is_gt(),
                    "from_descending_terms_unchecked: not strictly descending"
                );
            }
            let _ = prev_mono; // consumed below when re-bound
            if first {
                first_coeff = *c;
                first_mono_sev = m.sev();
                first_mono_deg = m.total_deg();
                first = false;
            }
            prev_mono = Some(m);
        }
        // Second pass: actually construct the nodes. (We could have
        // built them in the first pass, but the borrow-checker gets
        // confused if we both hold `prev_mono: &Monomial` and push
        // into the chain in the same loop. Two passes is fine for a
        // path that's not on the hot loop.)
        let _ = p;
        for (c, m) in terms {
            let fresh = Box::new(Node {
                coeff: c,
                mono: m,
                next: None,
            });
            *tail_slot = Some(fresh);
            tail_slot = &mut tail_slot.as_mut().unwrap().next;
        }
        Poly {
            head,
            len,
            lm_sev: first_mono_sev,
            lm_coeff: first_coeff,
            lm_deg: first_mono_deg,
        }
    }

    /// Build from parallel vectors in descending order. Mirrors the
    /// `poly_vec` signature verbatim. Iterates both vectors once to
    /// chain up nodes.
    pub fn from_descending_parallel_unchecked(
        ring: &Ring,
        coeffs: Vec<Coeff>,
        terms: Vec<Monomial>,
    ) -> Self {
        debug_assert_eq!(coeffs.len(), terms.len());
        if terms.is_empty() {
            return Self::zero();
        }
        #[cfg(debug_assertions)]
        {
            let p = ring.field().p();
            for &c in &coeffs {
                debug_assert!(c != 0 && c < p);
            }
            for w in terms.windows(2) {
                debug_assert!(w[0].cmp(&w[1], ring).is_gt());
            }
        }
        let _ = ring;
        let len = coeffs.len();
        let lm_coeff = coeffs[0];
        let lm_sev = terms[0].sev();
        let lm_deg = terms[0].total_deg();

        let mut head: Option<Box<Node>> = None;
        let mut tail_slot: &mut Option<Box<Node>> = &mut head;
        for (c, m) in coeffs.into_iter().zip(terms) {
            let fresh = Box::new(Node {
                coeff: c,
                mono: m,
                next: None,
            });
            *tail_slot = Some(fresh);
            tail_slot = &mut tail_slot.as_mut().unwrap().next;
        }
        Poly {
            head,
            len,
            lm_sev,
            lm_coeff,
            lm_deg,
        }
    }

    /// Build from unsorted terms. Sorts descending, de-dupes via sum,
    /// drops zeros. Semantics match `poly_vec::Poly::from_terms`.
    pub fn from_terms(ring: &Ring, terms: Vec<(Coeff, Monomial)>) -> Self {
        let mut terms = terms;
        terms.sort_by(|a, b| b.1.cmp(&a.1, ring));

        // Normalise: merge adjacent equal monomials, reduce coeffs
        // mod p, drop zeros. We do this into a Vec first to keep the
        // merge logic straightforward, then chain the survivors into
        // the linked list.
        let mut surviving: Vec<(Coeff, Monomial)> = Vec::with_capacity(terms.len());
        for (c, m) in terms {
            let c = if c >= ring.field().p() {
                ring.field().reduce(c as u64)
            } else {
                c
            };
            if c == 0 {
                continue;
            }
            if let Some(last) = surviving.last_mut()
                && last.1 == m
            {
                last.0 = ring.field().add(last.0, c);
                if last.0 == 0 {
                    surviving.pop();
                }
                continue;
            }
            surviving.push((c, m));
        }
        if surviving.is_empty() {
            return Self::zero();
        }
        Self::from_descending_terms_unchecked(ring, surviving)
    }

    // ----- Cache maintenance -----

    fn refresh_cache(&mut self) {
        if let Some(h) = self.head.as_deref() {
            self.lm_sev = h.mono.sev();
            self.lm_deg = h.mono.total_deg();
            self.lm_coeff = h.coeff;
        } else {
            self.lm_sev = 0;
            self.lm_coeff = 0;
            self.lm_deg = 0;
        }
    }

    // ----- Accessors -----

    /// Number of live terms.
    #[allow(clippy::len_without_is_empty)]
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether this is the zero polynomial.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.head.is_none()
    }

    /// Iterate over `(coeff, &monomial)` pairs in descending order.
    pub fn iter(&self) -> impl Iterator<Item = (Coeff, &Monomial)> + '_ {
        let mut node = self.head.as_deref();
        std::iter::from_fn(move || {
            let n = node?;
            node = n.next.as_deref();
            Some((n.coeff, &n.mono))
        })
    }

    /// Leading term `(coeff, &monomial)`, or `None` if zero.
    pub fn leading(&self) -> Option<(Coeff, &Monomial)> {
        self.head.as_deref().map(|n| (n.coeff, &n.mono))
    }

    /// Leading short exponent vector. 0 when zero.
    #[inline]
    pub fn lm_sev(&self) -> u64 {
        self.lm_sev
    }

    /// Leading coefficient. 0 when zero.
    #[inline]
    pub fn lm_coeff(&self) -> Coeff {
        self.lm_coeff
    }

    /// Leading monomial total degree. 0 when zero.
    #[inline]
    pub fn lm_deg(&self) -> u32 {
        self.lm_deg
    }

    /// A cursor positioned at the leading term (or at end if zero).
    /// Both backends expose the same cursor shape — see the parent
    /// module's dispatcher for context.
    #[inline]
    pub fn cursor(&self) -> PolyCursor<'_> {
        PolyCursor {
            node: self.head.as_deref(),
        }
    }

    /// Return a new polynomial with the leading term removed. If
    /// `self` is zero or a single term, returns the zero polynomial.
    /// Implemented by walking the tail and cloning each node (O(n)
    /// like the Vec version).
    pub fn drop_leading(&self) -> Poly {
        if self.len <= 1 {
            return Self::zero();
        }
        // New head is the first node of self.head.next.
        let mut out_head: Option<Box<Node>> = None;
        let mut tail_slot: &mut Option<Box<Node>> = &mut out_head;
        // Skip the leading node.
        let mut node = self.head.as_deref().and_then(|n| n.next.as_deref());
        while let Some(n) = node {
            let fresh = Box::new(Node {
                coeff: n.coeff,
                mono: n.mono.clone(),
                next: None,
            });
            *tail_slot = Some(fresh);
            tail_slot = &mut tail_slot.as_mut().unwrap().next;
            node = n.next.as_deref();
        }
        let mut out = Poly {
            head: out_head,
            len: self.len - 1,
            lm_sev: 0,
            lm_coeff: 0,
            lm_deg: 0,
        };
        out.refresh_cache();
        out
    }

    /// In-place leading-term drop. O(1): takes the head, replaces it
    /// with `head.next`, drops the detached node.
    pub fn drop_leading_in_place(&mut self) {
        if let Some(mut boxed) = self.head.take() {
            // Detach `next` before the old box drops, so the
            // detached head's destructor is non-recursive even if a
            // custom Node::drop is added later.
            self.head = boxed.next.take();
            self.len -= 1;
            // `boxed` is dropped here with next already None.
        }
        self.refresh_cache();
    }

    // ----- Arithmetic -----

    /// In-place: `self = self + other`.
    pub fn add_assign(&mut self, other: &Poly, ring: &Ring) {
        if other.is_zero() {
            return;
        }
        if self.is_zero() {
            *self = other.clone();
            return;
        }
        *self = merge(ring, self, other, false);
    }

    /// Out-of-place addition.
    pub fn add(&self, other: &Poly, ring: &Ring) -> Poly {
        if other.is_zero() {
            return self.clone();
        }
        if self.is_zero() {
            return other.clone();
        }
        merge(ring, self, other, false)
    }

    /// Destructive addition: `self + other`, reusing both inputs' list
    /// nodes in the output chain rather than allocating fresh ones.
    /// Mirrors Singular's `p_Add_q` template
    /// (`~/Singular/libpolys/polys/templates/p_Add_q__T.cc`) — see
    /// ADR-015 for the contract mapping. Both operands are consumed
    /// (ownership transfer enforces Singular's "Destroys: p, q"
    /// comment at the Rust type level).
    pub fn add_consuming(self, other: Poly, ring: &Ring) -> Poly {
        if other.is_zero() {
            return self;
        }
        if self.is_zero() {
            return other;
        }
        merge_consuming(ring, self, other, false)
    }

    /// Out-of-place subtraction.
    pub fn sub(&self, other: &Poly, ring: &Ring) -> Poly {
        if other.is_zero() {
            return self.clone();
        }
        if self.is_zero() {
            return other.neg(ring);
        }
        merge(ring, self, other, true)
    }

    /// Negation (flip every coefficient).
    pub fn neg(&self, ring: &Ring) -> Poly {
        let f = ring.field();
        // Build head-to-tail by walking and cloning with negated coeffs.
        let mut out_head: Option<Box<Node>> = None;
        let mut tail_slot: &mut Option<Box<Node>> = &mut out_head;
        let mut node = self.head.as_deref();
        while let Some(n) = node {
            let fresh = Box::new(Node {
                coeff: f.neg(n.coeff),
                mono: n.mono.clone(),
                next: None,
            });
            *tail_slot = Some(fresh);
            tail_slot = &mut tail_slot.as_mut().unwrap().next;
            node = n.next.as_deref();
        }
        let mut out = Poly {
            head: out_head,
            len: self.len,
            lm_sev: 0,
            lm_coeff: 0,
            lm_deg: 0,
        };
        out.refresh_cache();
        out
    }

    /// Multiply every coefficient by a scalar. Returns zero if
    /// `c == 0`.
    pub fn scale(&self, c: Coeff, ring: &Ring) -> Poly {
        let f = ring.field();
        debug_assert!(c < f.p());
        if c == 0 || self.is_zero() {
            return Self::zero();
        }
        let mut out_head: Option<Box<Node>> = None;
        let mut tail_slot: &mut Option<Box<Node>> = &mut out_head;
        let mut node = self.head.as_deref();
        while let Some(n) = node {
            let fresh = Box::new(Node {
                coeff: f.mul(n.coeff, c),
                mono: n.mono.clone(),
                next: None,
            });
            *tail_slot = Some(fresh);
            tail_slot = &mut tail_slot.as_mut().unwrap().next;
            node = n.next.as_deref();
        }
        let mut out = Poly {
            head: out_head,
            len: self.len,
            lm_sev: 0,
            lm_coeff: 0,
            lm_deg: 0,
        };
        out.refresh_cache();
        out
    }

    /// Multiply every monomial by `m`. Requires the products fit in
    /// the exponent range.
    pub fn shift(&self, m: &Monomial, ring: &Ring) -> Option<Poly> {
        if self.is_zero() {
            return Some(Self::zero());
        }
        let mut out_head: Option<Box<Node>> = None;
        let mut tail_slot: &mut Option<Box<Node>> = &mut out_head;
        let mut node = self.head.as_deref();
        while let Some(n) = node {
            let new_mono = n.mono.mul(m, ring)?;
            let fresh = Box::new(Node {
                coeff: n.coeff,
                mono: new_mono,
                next: None,
            });
            *tail_slot = Some(fresh);
            tail_slot = &mut tail_slot.as_mut().unwrap().next;
            node = n.next.as_deref();
        }
        // Descending order preserved by degrevlex monotonicity, same
        // as the Vec backend.
        let mut out = Poly {
            head: out_head,
            len: self.len,
            lm_sev: 0,
            lm_coeff: 0,
            lm_deg: 0,
        };
        out.refresh_cache();
        Some(out)
    }

    /// Standard multiplication via an accumulator (same strategy as
    /// the Vec backend).
    pub fn mul(&self, other: &Poly, ring: &Ring) -> Option<Poly> {
        if self.is_zero() || other.is_zero() {
            return Some(Self::zero());
        }
        let f = ring.field();
        let mut acc: Vec<(Coeff, Monomial)> = Vec::with_capacity(self.len * other.len);
        for (ca, ma) in self.iter() {
            for (cb, mb) in other.iter() {
                let m = ma.mul(mb, ring)?;
                let c = f.mul(ca, cb);
                if c != 0 {
                    acc.push((c, m));
                }
            }
        }
        Some(Self::from_terms(ring, acc))
    }

    /// The inner reduction step `self - c * m * q`. Splice-style
    /// two-pointer merge along both inputs' linked chains.
    pub fn sub_mul_term(&self, c: Coeff, m: &Monomial, q: &Poly, ring: &Ring) -> Option<Poly> {
        debug_assert!(c < ring.field().p());
        if c == 0 || q.is_zero() {
            return Some(self.clone());
        }
        let f = ring.field();

        let mut out_head: Option<Box<Node>> = None;
        let mut tail_slot: &mut Option<Box<Node>> = &mut out_head;
        let mut out_len: usize = 0;

        let mut left = self.head.as_deref();
        let mut right = q.head.as_deref();

        while let (Some(l), Some(r)) = (left, right) {
            let r_mono = m.mul(&r.mono, ring)?;
            match l.mono.cmp(&r_mono, ring) {
                std::cmp::Ordering::Greater => {
                    let fresh = Box::new(Node {
                        coeff: l.coeff,
                        mono: l.mono.clone(),
                        next: None,
                    });
                    *tail_slot = Some(fresh);
                    tail_slot = &mut tail_slot.as_mut().unwrap().next;
                    out_len += 1;
                    left = l.next.as_deref();
                }
                std::cmp::Ordering::Less => {
                    let neg = f.neg(f.mul(c, r.coeff));
                    if neg != 0 {
                        let fresh = Box::new(Node {
                            coeff: neg,
                            mono: r_mono,
                            next: None,
                        });
                        *tail_slot = Some(fresh);
                        tail_slot = &mut tail_slot.as_mut().unwrap().next;
                        out_len += 1;
                    }
                    right = r.next.as_deref();
                }
                std::cmp::Ordering::Equal => {
                    let cmq = f.mul(c, r.coeff);
                    let diff = f.sub(l.coeff, cmq);
                    if diff != 0 {
                        let fresh = Box::new(Node {
                            coeff: diff,
                            mono: l.mono.clone(),
                            next: None,
                        });
                        *tail_slot = Some(fresh);
                        tail_slot = &mut tail_slot.as_mut().unwrap().next;
                        out_len += 1;
                    }
                    left = l.next.as_deref();
                    right = r.next.as_deref();
                }
            }
        }
        while let Some(l) = left {
            let fresh = Box::new(Node {
                coeff: l.coeff,
                mono: l.mono.clone(),
                next: None,
            });
            *tail_slot = Some(fresh);
            tail_slot = &mut tail_slot.as_mut().unwrap().next;
            out_len += 1;
            left = l.next.as_deref();
        }
        while let Some(r) = right {
            let neg = f.neg(f.mul(c, r.coeff));
            if neg != 0 {
                let fresh = Box::new(Node {
                    coeff: neg,
                    mono: m.mul(&r.mono, ring)?,
                    next: None,
                });
                *tail_slot = Some(fresh);
                tail_slot = &mut tail_slot.as_mut().unwrap().next;
                out_len += 1;
            }
            right = r.next.as_deref();
        }

        let mut out = Poly {
            head: out_head,
            len: out_len,
            lm_sev: 0,
            lm_coeff: 0,
            lm_deg: 0,
        };
        out.refresh_cache();
        Some(out)
    }

    /// Destructive variant of [`sub_mul_term`](Self::sub_mul_term):
    /// `self - c * m * q`, destroying `self` and reusing its list nodes
    /// in the output chain. `m` and `q` are read-only (const in
    /// Singular's terminology).
    ///
    /// Mirrors Singular's `p_Minus_mm_Mult_qq` template
    /// (`~/Singular/libpolys/polys/templates/p_Minus_mm_Mult_qq__T.cc`).
    /// See ADR-015 for the contract mapping. Returns `None` if any
    /// `m * q[i]` product overflows the 7-bit-per-variable exponent
    /// budget.
    ///
    /// Includes the tail-splice fast path: when `self` exhausts before
    /// `q`, the remainder of `-c * m * q` is produced in a single pass
    /// and appended, rather than going through the compare loop.
    pub fn sub_mm_mult_qq_consuming(
        mut self,
        c: Coeff,
        m: &Monomial,
        q: &Poly,
        ring: &Ring,
    ) -> Option<Poly> {
        debug_assert!(c < ring.field().p());
        if c == 0 || q.is_zero() {
            return Some(self);
        }
        let f = ring.field();

        // Sentinel-slot pattern: `sentinel_slot` is a stack-local
        // `Option<Box<Node>>` that will end up containing the output
        // chain's head. The `tail_slot` raw pointer tracks the
        // `Option<Box<Node>>` slot where the next node attaches (starts
        // at &mut sentinel_slot, moves to &mut new_node.next after each
        // append). Safe Rust can't express this append-to-tail pattern
        // directly because each `&mut Option<Box<Node>>` alias-blocks
        // everything else reachable from the sentinel; see ADR-015 for
        // the soundness argument.
        let mut sentinel_slot: Option<Box<Node>> = None;
        let mut tail_slot: *mut Option<Box<Node>> = &mut sentinel_slot;
        let mut out_len: usize = 0;

        // Take ownership of self's chain. From here on, `self` is not
        // used until we reconstruct a Poly at the end.
        let mut left: Option<Box<Node>> = self.head.take();
        let mut right = q.head.as_deref();

        // SAFETY: Every write through `tail_slot` targets an
        // `Option<Box<Node>>` that belongs to either `sentinel_slot`
        // (on the first iteration) or the `next` field of the node
        // most recently appended (which `sentinel_slot` transitively
        // owns). `sentinel_slot` is a stack-local that outlives every
        // `tail_slot` update in this function. No other live reference
        // to any tail slot exists during the writes: the input chains
        // (`left`, `right`) are walked separately, and on each
        // iteration we transfer ownership of any spliced input node
        // into the output before dereferencing `tail_slot`. At the
        // end, `sentinel_slot.take()` moves the chain out.
        //
        // On early-return via `?` (monomial overflow), Rust's drop
        // glue runs: `sentinel_slot` drops the partial output chain,
        // `left` drops the unvisited self-tail, and `self` drops with
        // `head = None`. No leak, no double-free.
        unsafe {
            loop {
                let (l_ref, r_node) = match (left.as_ref(), right) {
                    (Some(l), Some(r)) => (l, r),
                    _ => break,
                };
                let r_mono = m.mul(&r_node.mono, ring)?;
                match l_ref.mono.cmp(&r_mono, ring) {
                    std::cmp::Ordering::Greater => {
                        // Splice left's head into the output tail.
                        let mut node = left.take().unwrap();
                        left = node.next.take();
                        // Install `node` at *tail_slot and advance
                        // `tail_slot` to point at `node.next`.
                        *tail_slot = Some(node);
                        let next_slot: *mut Option<Box<Node>> =
                            &mut (*tail_slot).as_mut().unwrap().next;
                        tail_slot = next_slot;
                        out_len += 1;
                    }
                    std::cmp::Ordering::Less => {
                        // Emit a fresh node for `-c * q_coeff * (m * q_mono)`.
                        // `r_mono` is already the product.
                        let neg = f.neg(f.mul(c, r_node.coeff));
                        right = r_node.next.as_deref();
                        if neg != 0 {
                            let fresh = Box::new(Node {
                                coeff: neg,
                                mono: r_mono,
                                next: None,
                            });
                            *tail_slot = Some(fresh);
                            let next_slot: *mut Option<Box<Node>> =
                                &mut (*tail_slot).as_mut().unwrap().next;
                            tail_slot = next_slot;
                            out_len += 1;
                        }
                    }
                    std::cmp::Ordering::Equal => {
                        // Reuse left's node if the difference is
                        // nonzero; otherwise free it.
                        let cmq = f.mul(c, r_node.coeff);
                        let diff = f.sub(l_ref.coeff, cmq);
                        right = r_node.next.as_deref();
                        let mut node = left.take().unwrap();
                        left = node.next.take();
                        if diff != 0 {
                            node.coeff = diff;
                            *tail_slot = Some(node);
                            let next_slot: *mut Option<Box<Node>> =
                                &mut (*tail_slot).as_mut().unwrap().next;
                            tail_slot = next_slot;
                            out_len += 1;
                        }
                        // else: `node` drops here (freed).
                    }
                }
            }

            // Tail splices. At most one of `left` / `right` is nonempty.
            if let Some(l_head) = left.take() {
                // Self still has terms; q is exhausted. Splice the
                // entire remaining chain in one assignment and count
                // its length by walking.
                let mut remaining_len = 1usize;
                let mut node = l_head.next.as_deref();
                while let Some(x) = node {
                    remaining_len += 1;
                    node = x.next.as_deref();
                }
                *tail_slot = Some(l_head);
                out_len += remaining_len;
                // `tail_slot` is no longer used after this point.
            } else {
                // Self exhausted; q may still have terms. Build the
                // remainder of `-c * m * q` fresh.
                while let Some(r_node) = right {
                    let neg = f.neg(f.mul(c, r_node.coeff));
                    right = r_node.next.as_deref();
                    if neg == 0 {
                        continue;
                    }
                    let prod_m = m.mul(&r_node.mono, ring)?;
                    let fresh = Box::new(Node {
                        coeff: neg,
                        mono: prod_m,
                        next: None,
                    });
                    *tail_slot = Some(fresh);
                    let next_slot: *mut Option<Box<Node>> =
                        &mut (*tail_slot).as_mut().unwrap().next;
                    tail_slot = next_slot;
                    out_len += 1;
                }
            }
        }

        // Move the chain out of the sentinel slot.
        self.head = sentinel_slot.take();
        self.len = out_len;
        self.refresh_cache();
        Some(self)
    }

    /// Scale so the leading coefficient becomes 1.
    pub fn monic(&self, ring: &Ring) -> Option<Poly> {
        if self.is_zero() {
            return Some(Self::zero());
        }
        let lc = self.lm_coeff;
        if lc == 1 {
            return Some(self.clone());
        }
        let inv = ring.field().inv(lc)?;
        Some(self.scale(inv, ring))
    }

    // ----- Invariants -----

    /// Panic if any internal invariant is violated.
    pub fn assert_canonical(&self, ring: &Ring) {
        let p = ring.field().p();
        let mut node = self.head.as_deref();
        let mut prev: Option<&Monomial> = None;
        let mut count: usize = 0;
        while let Some(n) = node {
            assert!(
                n.coeff > 0 && n.coeff < p,
                "coeff[{count}] = {} not in 1..{p}",
                n.coeff
            );
            n.mono.assert_canonical(ring);
            if let Some(p_m) = prev {
                let ord = p_m.cmp(&n.mono, ring);
                assert!(
                    ord == std::cmp::Ordering::Greater,
                    "terms not strictly descending at [{count}]: got {ord:?}"
                );
            }
            prev = Some(&n.mono);
            count += 1;
            node = n.next.as_deref();
        }
        assert_eq!(self.len, count, "cached len disagrees with walk");
        if self.is_zero() {
            assert_eq!(self.lm_sev, 0);
            assert_eq!(self.lm_coeff, 0);
            assert_eq!(self.lm_deg, 0);
        } else {
            let h = self.head.as_deref().unwrap();
            assert_eq!(self.lm_sev, h.mono.sev());
            assert_eq!(self.lm_coeff, h.coeff);
            assert_eq!(self.lm_deg, h.mono.total_deg());
        }
    }
}

impl Default for Poly {
    fn default() -> Self {
        Self::zero()
    }
}

/// A cursor walking a [`Poly`]'s terms in descending order.
///
/// Obtain one with [`Poly::cursor`]. Cheap and `Copy`: it holds
/// a single reference to the current node (or `None` when
/// exhausted). On the linked-list backend `advance` chases the
/// `next` pointer; on the flat-array backend (see
/// [`super::poly_vec::PolyCursor`]) it bumps an index. The same
/// shape on both backends lets [`crate::reducer::Reducer`] work
/// uniformly.
#[derive(Clone, Copy)]
pub struct PolyCursor<'a> {
    node: Option<&'a Node>,
}

impl<'a> std::fmt::Debug for PolyCursor<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PolyCursor")
            .field("exhausted", &self.node.is_none())
            .finish()
    }
}

impl<'a> PolyCursor<'a> {
    /// Current term `(coeff, &monomial)`, or `None` if exhausted.
    #[inline]
    pub fn term(&self) -> Option<(Coeff, &'a Monomial)> {
        self.node.map(|n| (n.coeff, &n.mono))
    }

    /// Advance one term. No-op once exhausted.
    #[inline]
    pub fn advance(&mut self) {
        self.node = self.node.and_then(|n| n.next.as_deref());
    }

    /// True once all terms have been walked.
    #[inline]
    pub fn is_done(&self) -> bool {
        self.node.is_none()
    }
}

impl PartialEq for Poly {
    fn eq(&self, other: &Self) -> bool {
        // Same length + same terms in order. We compare through
        // cursors so the comparison stays O(n) (no materialised
        // Vecs).
        if self.len != other.len {
            return false;
        }
        let mut a = self.head.as_deref();
        let mut b = other.head.as_deref();
        while let (Some(x), Some(y)) = (a, b) {
            if x.coeff != y.coeff || x.mono != y.mono {
                return false;
            }
            a = x.next.as_deref();
            b = y.next.as_deref();
        }
        a.is_none() && b.is_none()
    }
}
impl Eq for Poly {}

/// Destructive merge: walk both chains, splicing input nodes
/// directly into the output chain rather than allocating fresh
/// ones. Mirrors Singular's `p_Add_q` node-splice pattern (see
/// `~/Singular/libpolys/polys/templates/p_Add_q__T.cc` lines 57,
/// 65, 71, 60-61, 67, 73). Both operands are consumed; `subtract`
/// flag chooses add vs sub semantics on the second operand's
/// coefficients.
///
/// Both inputs must be nonempty (callers guard zero operands).
fn merge_consuming(ring: &Ring, a: Poly, b: Poly, subtract: bool) -> Poly {
    debug_assert!(!a.is_zero());
    debug_assert!(!b.is_zero());
    let f = ring.field();

    // Sentinel-slot pattern (see `sub_mm_mult_qq_consuming` for the
    // soundness argument).
    let mut sentinel_slot: Option<Box<Node>> = None;
    let mut tail_slot: *mut Option<Box<Node>> = &mut sentinel_slot;
    let mut out_len: usize = 0;

    // Move both input chains into local owned variables. We can't
    // destructure Poly because it has a custom Drop; `head.take()`
    // leaves the Poly with `head = None`, so the subsequent implicit
    // drop of `a` / `b` is O(1).
    let mut a = a;
    let mut b = b;
    let mut left: Option<Box<Node>> = a.head.take();
    let mut right: Option<Box<Node>> = b.head.take();

    // SAFETY: Identical argument to `sub_mm_mult_qq_consuming`. Every
    // write through `tail_slot` targets an `Option<Box<Node>>` that
    // belongs to the sentinel chain. `sentinel_slot` outlives every
    // update. Input nodes are moved into the output before any
    // dereference of `tail_slot`.
    unsafe {
        loop {
            let (l_ref, r_ref) = match (left.as_ref(), right.as_ref()) {
                (Some(l), Some(r)) => (l, r),
                _ => break,
            };
            match l_ref.mono.cmp(&r_ref.mono, ring) {
                std::cmp::Ordering::Greater => {
                    // Splice `left`'s head.
                    let mut node = left.take().unwrap();
                    left = node.next.take();
                    *tail_slot = Some(node);
                    let next_slot: *mut Option<Box<Node>> =
                        &mut (*tail_slot).as_mut().unwrap().next;
                    tail_slot = next_slot;
                    out_len += 1;
                }
                std::cmp::Ordering::Less => {
                    // Splice `right`'s head (negating coeff if subtracting).
                    let mut node = right.take().unwrap();
                    right = node.next.take();
                    if subtract {
                        node.coeff = f.neg(node.coeff);
                    }
                    *tail_slot = Some(node);
                    let next_slot: *mut Option<Box<Node>> =
                        &mut (*tail_slot).as_mut().unwrap().next;
                    tail_slot = next_slot;
                    out_len += 1;
                }
                std::cmp::Ordering::Equal => {
                    // Combine coefficients; reuse left's node if the
                    // sum is nonzero, otherwise free both.
                    let bc = if subtract {
                        f.neg(r_ref.coeff)
                    } else {
                        r_ref.coeff
                    };
                    let s = f.add(l_ref.coeff, bc);
                    // Consume both heads.
                    let mut l_node = left.take().unwrap();
                    left = l_node.next.take();
                    let mut r_node = right.take().unwrap();
                    right = r_node.next.take();
                    if s != 0 {
                        l_node.coeff = s;
                        *tail_slot = Some(l_node);
                        let next_slot: *mut Option<Box<Node>> =
                            &mut (*tail_slot).as_mut().unwrap().next;
                        tail_slot = next_slot;
                        out_len += 1;
                    }
                    // `l_node` (if dropped here) and `r_node` are freed.
                    // Drop `r_node` explicitly to make the intent clear.
                    drop(r_node);
                }
            }
        }

        // Tail splices: one side is exhausted. Splice the remainder
        // of the other side in a single pointer assignment.
        if let Some(l_head) = left.take() {
            let mut remaining_len = 1usize;
            let mut node = l_head.next.as_deref();
            while let Some(x) = node {
                remaining_len += 1;
                node = x.next.as_deref();
            }
            *tail_slot = Some(l_head);
            out_len += remaining_len;
        } else if let Some(r_head) = right.take() {
            // If subtracting, every node's coeff must be negated.
            // Otherwise the remainder can be spliced as-is.
            if subtract {
                let mut cur: Option<Box<Node>> = Some(r_head);
                while let Some(mut node) = cur {
                    let next = node.next.take();
                    node.coeff = f.neg(node.coeff);
                    *tail_slot = Some(node);
                    let next_slot: *mut Option<Box<Node>> =
                        &mut (*tail_slot).as_mut().unwrap().next;
                    tail_slot = next_slot;
                    out_len += 1;
                    cur = next;
                }
            } else {
                let mut remaining_len = 1usize;
                let mut node = r_head.next.as_deref();
                while let Some(x) = node {
                    remaining_len += 1;
                    node = x.next.as_deref();
                }
                *tail_slot = Some(r_head);
                out_len += remaining_len;
            }
        }
    }

    let mut out = Poly {
        head: sentinel_slot.take(),
        len: out_len,
        lm_sev: 0,
        lm_coeff: 0,
        lm_deg: 0,
    };
    out.refresh_cache();
    out
}

/// Merge two polynomials into one via a splice-style two-pointer
/// walk along both chains. If `subtract` is true, the second
/// operand's coefficients are negated. Allocates fresh nodes for
/// every output term (list-splice node reuse is a future
/// optimisation).
fn merge(ring: &Ring, a: &Poly, b: &Poly, subtract: bool) -> Poly {
    let f = ring.field();
    let mut out_head: Option<Box<Node>> = None;
    let mut tail_slot: &mut Option<Box<Node>> = &mut out_head;
    let mut out_len: usize = 0;

    let mut left = a.head.as_deref();
    let mut right = b.head.as_deref();

    while let (Some(l), Some(r)) = (left, right) {
        match l.mono.cmp(&r.mono, ring) {
            std::cmp::Ordering::Greater => {
                let fresh = Box::new(Node {
                    coeff: l.coeff,
                    mono: l.mono.clone(),
                    next: None,
                });
                *tail_slot = Some(fresh);
                tail_slot = &mut tail_slot.as_mut().unwrap().next;
                out_len += 1;
                left = l.next.as_deref();
            }
            std::cmp::Ordering::Less => {
                let c = if subtract { f.neg(r.coeff) } else { r.coeff };
                // c is nonzero as long as r.coeff is nonzero (which
                // it always is by the canonical invariant): negating
                // preserves nonzeroness.
                let fresh = Box::new(Node {
                    coeff: c,
                    mono: r.mono.clone(),
                    next: None,
                });
                *tail_slot = Some(fresh);
                tail_slot = &mut tail_slot.as_mut().unwrap().next;
                out_len += 1;
                right = r.next.as_deref();
            }
            std::cmp::Ordering::Equal => {
                let bc = if subtract { f.neg(r.coeff) } else { r.coeff };
                let s = f.add(l.coeff, bc);
                if s != 0 {
                    let fresh = Box::new(Node {
                        coeff: s,
                        mono: l.mono.clone(),
                        next: None,
                    });
                    *tail_slot = Some(fresh);
                    tail_slot = &mut tail_slot.as_mut().unwrap().next;
                    out_len += 1;
                }
                left = l.next.as_deref();
                right = r.next.as_deref();
            }
        }
    }
    while let Some(l) = left {
        let fresh = Box::new(Node {
            coeff: l.coeff,
            mono: l.mono.clone(),
            next: None,
        });
        *tail_slot = Some(fresh);
        tail_slot = &mut tail_slot.as_mut().unwrap().next;
        out_len += 1;
        left = l.next.as_deref();
    }
    while let Some(r) = right {
        let c = if subtract { f.neg(r.coeff) } else { r.coeff };
        let fresh = Box::new(Node {
            coeff: c,
            mono: r.mono.clone(),
            next: None,
        });
        *tail_slot = Some(fresh);
        tail_slot = &mut tail_slot.as_mut().unwrap().next;
        out_len += 1;
        right = r.next.as_deref();
    }

    let mut out = Poly {
        head: out_head,
        len: out_len,
        lm_sev: 0,
        lm_coeff: 0,
        lm_deg: 0,
    };
    out.refresh_cache();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Field;
    use crate::ordering::MonoOrder;

    fn mk_ring(nvars: u32, p: u32) -> Ring {
        Ring::new(nvars, MonoOrder::DegRevLex, Field::new(p).unwrap()).unwrap()
    }

    fn mono(r: &Ring, e: &[u32]) -> Monomial {
        Monomial::from_exponents(r, e).unwrap()
    }

    #[test]
    fn zero_is_zero() {
        let p = Poly::zero();
        assert!(p.is_zero());
        assert_eq!(p.len(), 0);
        assert!(p.leading().is_none());
    }

    #[test]
    fn from_terms_sorts_and_dedups() {
        let r = mk_ring(3, 13);
        let terms = vec![
            (3, mono(&r, &[1, 0, 0])),
            (5, mono(&r, &[0, 2, 0])),
            (7, mono(&r, &[1, 0, 0])),
            (0, mono(&r, &[0, 0, 1])),
        ];
        let p = Poly::from_terms(&r, terms);
        p.assert_canonical(&r);
        assert_eq!(p.len(), 2);
        let (c0, m0) = p.leading().unwrap();
        assert_eq!(c0, 5);
        assert_eq!(*m0, mono(&r, &[0, 2, 0]));
        // Second term via iter().
        let second = p.iter().nth(1).unwrap();
        assert_eq!(second.0, 10);
        assert_eq!(*second.1, mono(&r, &[1, 0, 0]));
    }

    #[test]
    fn add_and_sub_cancel() {
        let r = mk_ring(3, 13);
        let f = Poly::from_terms(
            &r,
            vec![
                (3, mono(&r, &[1, 0, 0])),
                (5, mono(&r, &[0, 2, 0])),
                (1, mono(&r, &[0, 0, 1])),
            ],
        );
        let g = f.sub(&f, &r);
        g.assert_canonical(&r);
        assert!(g.is_zero());
    }

    #[test]
    fn sub_mul_term_matches_slow_path() {
        let r = mk_ring(3, 13);
        let p = Poly::from_terms(
            &r,
            vec![
                (3, mono(&r, &[2, 1, 0])),
                (7, mono(&r, &[1, 0, 1])),
                (1, mono(&r, &[0, 0, 2])),
            ],
        );
        let q = Poly::from_terms(
            &r,
            vec![(4, mono(&r, &[1, 1, 0])), (5, mono(&r, &[0, 0, 1]))],
        );
        let m = mono(&r, &[1, 0, 0]);
        let c: Coeff = 2;

        let mq = q.shift(&m, &r).unwrap().scale(c, &r);
        let slow = p.sub(&mq, &r);
        let fast = p.sub_mul_term(c, &m, &q, &r).unwrap();
        slow.assert_canonical(&r);
        fast.assert_canonical(&r);
        assert_eq!(slow, fast);
    }

    #[test]
    fn monic_is_idempotent() {
        let r = mk_ring(2, 32003);
        let p = Poly::from_terms(
            &r,
            vec![
                (17, mono(&r, &[3, 0])),
                (2, mono(&r, &[1, 1])),
                (9, mono(&r, &[0, 2])),
            ],
        );
        let once = p.monic(&r).unwrap();
        let twice = once.monic(&r).unwrap();
        assert_eq!(once, twice);
        assert_eq!(once.lm_coeff(), 1);
    }

    #[test]
    fn leading_invariants() {
        let r = mk_ring(2, 7);
        let p = Poly::from_terms(&r, vec![(3, mono(&r, &[2, 0])), (4, mono(&r, &[1, 1]))]);
        let (c, m) = p.leading().unwrap();
        assert_eq!(c, 3);
        assert_eq!(m.total_deg(), 2);
        assert_eq!(p.lm_sev(), m.sev());
        assert_eq!(p.lm_coeff(), 3);
        assert_eq!(p.lm_deg(), 2);
    }

    #[test]
    fn drop_leading_basic() {
        let r = mk_ring(3, 13);
        let p = Poly::from_terms(
            &r,
            vec![
                (3, mono(&r, &[2, 1, 0])),
                (7, mono(&r, &[1, 0, 1])),
                (1, mono(&r, &[0, 0, 2])),
            ],
        );
        let tail = p.drop_leading();
        tail.assert_canonical(&r);
        assert_eq!(tail.len(), 2);
        let (c, m) = tail.leading().unwrap();
        assert_eq!(c, 7);
        assert_eq!(m, &mono(&r, &[1, 0, 1]));
    }

    #[test]
    fn drop_leading_in_place_o1() {
        // Peel leaders repeatedly, checking the cache and length
        // stay consistent and the poly ends up zero.
        let r = mk_ring(3, 32003);
        let mut p = Poly::from_terms(
            &r,
            vec![
                (5, mono(&r, &[3, 0, 0])),
                (4, mono(&r, &[2, 1, 0])),
                (3, mono(&r, &[1, 0, 1])),
                (2, mono(&r, &[0, 0, 2])),
                (1, mono(&r, &[0, 1, 0])),
            ],
        );
        for expected_len in (0..5).rev() {
            p.drop_leading_in_place();
            p.assert_canonical(&r);
            assert_eq!(p.len(), expected_len);
        }
        // Extra drop on a zero poly is a no-op.
        p.drop_leading_in_place();
        assert!(p.is_zero());
    }

    #[test]
    fn cursor_walks_all_terms() {
        let r = mk_ring(3, 32003);
        let p = Poly::from_terms(
            &r,
            vec![
                (5, mono(&r, &[3, 0, 0])),
                (4, mono(&r, &[2, 1, 0])),
                (3, mono(&r, &[0, 0, 2])),
            ],
        );
        let mut c = p.cursor();
        assert!(!c.is_done());
        let (c0, m0) = c.term().unwrap();
        assert_eq!(c0, 5);
        assert_eq!(*m0, mono(&r, &[3, 0, 0]));
        c.advance();
        let (c1, _) = c.term().unwrap();
        assert_eq!(c1, 4);
        c.advance();
        let (c2, _) = c.term().unwrap();
        assert_eq!(c2, 3);
        c.advance();
        assert!(c.is_done());
        assert!(c.term().is_none());
        // advance past end is a no-op.
        c.advance();
        assert!(c.is_done());
    }

    #[test]
    fn add_consuming_matches_add() {
        // Round-trip: destructive add matches non-destructive add on
        // the same inputs.
        let r = mk_ring(3, 13);
        let a = Poly::from_terms(
            &r,
            vec![
                (3, mono(&r, &[2, 0, 0])),
                (5, mono(&r, &[1, 1, 0])),
                (1, mono(&r, &[0, 0, 2])),
            ],
        );
        let b = Poly::from_terms(
            &r,
            vec![
                (2, mono(&r, &[1, 1, 0])),
                (4, mono(&r, &[0, 2, 0])),
                (9, mono(&r, &[0, 0, 1])),
            ],
        );
        let expected = a.add(&b, &r);
        let got = a.clone().add_consuming(b.clone(), &r);
        got.assert_canonical(&r);
        assert_eq!(got, expected);
    }

    #[test]
    fn add_consuming_cancellation_middle() {
        // Middle-of-chain cancellation: a has x^2 + x + 1, b has -x.
        // Output should be x^2 + 1.
        let r = mk_ring(1, 13);
        let a = Poly::from_terms(
            &r,
            vec![
                (1, mono(&r, &[2])),
                (1, mono(&r, &[1])),
                (1, mono(&r, &[0])),
            ],
        );
        let b = Poly::from_terms(&r, vec![(12, mono(&r, &[1]))]); // -1
        let expected = a.add(&b, &r);
        let got = a.clone().add_consuming(b.clone(), &r);
        got.assert_canonical(&r);
        assert_eq!(got, expected);
        assert_eq!(got.len(), 2);
    }

    #[test]
    fn add_consuming_full_cancellation() {
        // f + (-f) = 0.
        let r = mk_ring(3, 13);
        let f = Poly::from_terms(
            &r,
            vec![
                (3, mono(&r, &[1, 0, 0])),
                (5, mono(&r, &[0, 2, 0])),
                (1, mono(&r, &[0, 0, 1])),
            ],
        );
        let neg_f = f.neg(&r);
        let got = f.clone().add_consuming(neg_f, &r);
        got.assert_canonical(&r);
        assert!(got.is_zero());
    }

    #[test]
    fn add_consuming_tail_splice_left_longer() {
        // Left has more terms past the last b-term, exercising the
        // tail-splice path for `left`.
        let r = mk_ring(3, 13);
        let a = Poly::from_terms(
            &r,
            vec![
                (3, mono(&r, &[3, 0, 0])),
                (5, mono(&r, &[2, 0, 0])),
                (7, mono(&r, &[1, 0, 0])),
                (2, mono(&r, &[0, 1, 0])),
                (4, mono(&r, &[0, 0, 1])),
            ],
        );
        let b = Poly::from_terms(&r, vec![(1, mono(&r, &[3, 0, 0]))]);
        let expected = a.add(&b, &r);
        let got = a.clone().add_consuming(b.clone(), &r);
        got.assert_canonical(&r);
        assert_eq!(got, expected);
    }

    #[test]
    fn add_consuming_tail_splice_right_longer() {
        // Right has more terms past the last a-term.
        let r = mk_ring(3, 13);
        let a = Poly::from_terms(&r, vec![(3, mono(&r, &[3, 0, 0]))]);
        let b = Poly::from_terms(
            &r,
            vec![
                (5, mono(&r, &[2, 0, 0])),
                (7, mono(&r, &[1, 0, 0])),
                (2, mono(&r, &[0, 1, 0])),
                (4, mono(&r, &[0, 0, 1])),
            ],
        );
        let expected = a.add(&b, &r);
        let got = a.clone().add_consuming(b.clone(), &r);
        got.assert_canonical(&r);
        assert_eq!(got, expected);
    }

    #[test]
    fn add_consuming_single_terms() {
        let r = mk_ring(2, 7);
        let a = Poly::monomial(&r, 3, mono(&r, &[1, 0]));
        let b = Poly::monomial(&r, 4, mono(&r, &[0, 1]));
        let expected = a.add(&b, &r);
        let got = a.clone().add_consuming(b.clone(), &r);
        got.assert_canonical(&r);
        assert_eq!(got, expected);
    }

    #[test]
    fn add_consuming_zero_operands() {
        let r = mk_ring(2, 7);
        let f = Poly::from_terms(&r, vec![(3, mono(&r, &[2, 0])), (4, mono(&r, &[1, 1]))]);
        let z = Poly::zero();
        // f + 0 = f.
        let got = f.clone().add_consuming(z.clone(), &r);
        got.assert_canonical(&r);
        assert_eq!(got, f);
        // 0 + f = f.
        let got = Poly::zero().add_consuming(f.clone(), &r);
        got.assert_canonical(&r);
        assert_eq!(got, f);
        // 0 + 0 = 0.
        let got = Poly::zero().add_consuming(Poly::zero(), &r);
        assert!(got.is_zero());
    }

    #[test]
    fn sub_mm_mult_qq_consuming_matches_sub_mul_term() {
        // Round-trip: destructive equals non-destructive on the same
        // inputs.
        let r = mk_ring(3, 13);
        let p = Poly::from_terms(
            &r,
            vec![
                (3, mono(&r, &[2, 1, 0])),
                (7, mono(&r, &[1, 0, 1])),
                (1, mono(&r, &[0, 0, 2])),
            ],
        );
        let q = Poly::from_terms(
            &r,
            vec![(4, mono(&r, &[1, 1, 0])), (5, mono(&r, &[0, 0, 1]))],
        );
        let m = mono(&r, &[1, 0, 0]);
        let c: Coeff = 2;

        let expected = p.sub_mul_term(c, &m, &q, &r).unwrap();
        let got = p.clone().sub_mm_mult_qq_consuming(c, &m, &q, &r).unwrap();
        got.assert_canonical(&r);
        assert_eq!(got, expected);
    }

    #[test]
    fn sub_mm_mult_qq_consuming_cancellation() {
        // Choose inputs so a specific term cancels.
        let r = mk_ring(2, 13);
        // p = x^2 + xy + y^2. q = x + y. m = x, c = 1.
        // p - 1 * x * (x + y) = p - (x^2 + xy) = y^2.
        let p = Poly::from_terms(
            &r,
            vec![
                (1, mono(&r, &[2, 0])),
                (1, mono(&r, &[1, 1])),
                (1, mono(&r, &[0, 2])),
            ],
        );
        let q = Poly::from_terms(&r, vec![(1, mono(&r, &[1, 0])), (1, mono(&r, &[0, 1]))]);
        let m = mono(&r, &[1, 0]);
        let got = p.clone().sub_mm_mult_qq_consuming(1, &m, &q, &r).unwrap();
        got.assert_canonical(&r);
        let expected = p.sub_mul_term(1, &m, &q, &r).unwrap();
        assert_eq!(got, expected);
        assert_eq!(got.len(), 1);
    }

    #[test]
    fn sub_mm_mult_qq_consuming_tail_splice_self_exhausts() {
        // `p` exhausts before `q*m*c` does — exercises the
        // "self exhausts; q may still have terms" path.
        let r = mk_ring(2, 13);
        // p = x^3 (leading only). q has many smaller terms after
        // matching x^2 via m = x.
        let p = Poly::from_terms(&r, vec![(3, mono(&r, &[3, 0]))]);
        let q = Poly::from_terms(
            &r,
            vec![
                (1, mono(&r, &[2, 0])),
                (2, mono(&r, &[1, 1])),
                (4, mono(&r, &[0, 2])),
                (5, mono(&r, &[0, 0])),
            ],
        );
        let m = mono(&r, &[1, 0]);
        let c: Coeff = 1;
        let expected = p.sub_mul_term(c, &m, &q, &r).unwrap();
        let got = p.clone().sub_mm_mult_qq_consuming(c, &m, &q, &r).unwrap();
        got.assert_canonical(&r);
        assert_eq!(got, expected);
    }

    #[test]
    fn sub_mm_mult_qq_consuming_tail_splice_q_exhausts() {
        // `q*m*c` exhausts before `p` does.
        let r = mk_ring(2, 13);
        let p = Poly::from_terms(
            &r,
            vec![
                (3, mono(&r, &[5, 0])),
                (2, mono(&r, &[4, 0])),
                (1, mono(&r, &[3, 0])),
                (4, mono(&r, &[0, 1])),
                (5, mono(&r, &[0, 0])),
            ],
        );
        let q = Poly::from_terms(&r, vec![(1, mono(&r, &[4, 0]))]);
        let m = mono(&r, &[1, 0]);
        let c: Coeff = 1;
        let expected = p.sub_mul_term(c, &m, &q, &r).unwrap();
        let got = p.clone().sub_mm_mult_qq_consuming(c, &m, &q, &r).unwrap();
        got.assert_canonical(&r);
        assert_eq!(got, expected);
    }

    #[test]
    fn sub_mm_mult_qq_consuming_zero_coeff() {
        // c = 0 returns self unchanged.
        let r = mk_ring(2, 13);
        let p = Poly::from_terms(&r, vec![(3, mono(&r, &[2, 0])), (4, mono(&r, &[1, 1]))]);
        let q = Poly::from_terms(&r, vec![(1, mono(&r, &[1, 0]))]);
        let m = mono(&r, &[0, 1]);
        let got = p.clone().sub_mm_mult_qq_consuming(0, &m, &q, &r).unwrap();
        got.assert_canonical(&r);
        assert_eq!(got, p);
    }

    #[test]
    fn sub_mm_mult_qq_consuming_zero_q() {
        // q = 0 returns self unchanged.
        let r = mk_ring(2, 13);
        let p = Poly::from_terms(&r, vec![(3, mono(&r, &[2, 0])), (4, mono(&r, &[1, 1]))]);
        let q = Poly::zero();
        let m = mono(&r, &[0, 1]);
        let got = p.clone().sub_mm_mult_qq_consuming(2, &m, &q, &r).unwrap();
        got.assert_canonical(&r);
        assert_eq!(got, p);
    }

    #[test]
    fn sub_mm_mult_qq_consuming_self_zero() {
        // self = 0; result is -c*m*q, same as Poly::zero().sub_mul_term(...).
        let r = mk_ring(2, 13);
        let q = Poly::from_terms(&r, vec![(3, mono(&r, &[1, 0])), (2, mono(&r, &[0, 1]))]);
        let m = mono(&r, &[1, 0]);
        let c: Coeff = 2;
        let expected = Poly::zero().sub_mul_term(c, &m, &q, &r).unwrap();
        let got = Poly::zero()
            .sub_mm_mult_qq_consuming(c, &m, &q, &r)
            .unwrap();
        got.assert_canonical(&r);
        assert_eq!(got, expected);
    }

    #[test]
    fn iterative_drop_survives_long_chain() {
        // A recursive Box<Node> destructor would overflow the stack
        // on a chain of this length. The custom iterative Drop must
        // handle it.
        //
        // Exponents must fit the 7-bit-per-variable budget
        // (ADR-005), so we spread the chain across enough variables
        // to give us N distinct monomials at or below that cap. With
        // 4 variables each up to exponent 63 we get 64^4 = 16.8M
        // possible monomials — plenty for a 100 000-term chain, and
        // every pair is distinct so the descending-order contract
        // is trivial.
        let r = mk_ring(4, 32003);
        let n: usize = 100_000;

        // Generate N distinct monomials in descending degrevlex
        // order by sweeping exponents. Simplest: take monomials of
        // the form `x0^a` for a in [0..n). That fits in variable 0
        // alone if n <= 64 — not enough. So use base-64 digits
        // across the 4 variables instead, and sort descending.
        let mut distinct: Vec<Monomial> = Vec::with_capacity(n);
        // Walk (a, b, c, d) in lex order where each is 0..63; stop
        // after n entries.
        'outer: for d in 0u32..64 {
            for c in 0u32..64 {
                for b in 0u32..64 {
                    for a in 0u32..64 {
                        if distinct.len() >= n {
                            break 'outer;
                        }
                        distinct.push(mono(&r, &[a, b, c, d]));
                    }
                }
            }
        }
        // Sort descending under the ring's ordering.
        distinct.sort_by(|x, y| y.cmp(x, &r));
        let terms: Vec<(Coeff, Monomial)> =
            distinct.into_iter().map(|m| (1u32, m)).collect();
        let p = Poly::from_descending_terms_unchecked(&r, terms);
        assert_eq!(p.len(), n);
        // When `p` is dropped at scope exit, iterative Drop should
        // walk the chain without recursing. If this test ever starts
        // overflowing the stack, Drop has regressed to recursive.
        drop(p);
    }
}
