//! Thread-local `Node` allocator for the [`poly_list`](super::poly_list)
//! backend.
//!
//! Two variants of [`NodePool`] live here, selected at compile time by
//! the `linked_list_poly_pool` Cargo feature:
//!
//! * **`linked_list_poly_pool` on** — pool-backed. `alloc` pops from a
//!   thread-local free list when possible and falls back to a single
//!   `Box::leak` on miss. `dealloc` pushes the node's storage onto the
//!   free list; storage is **never** returned to the system during the
//!   lifetime of the thread. Peak memory is bounded by the peak
//!   in-flight `Node` count during the run (for staging-5101449 that is
//!   order ~3M nodes × 48 bytes ≈ ~140 MB; see ADR-016).
//!
//! * **`linked_list_poly_pool` off** — forwarder. `alloc` is a single
//!   `Box::new` + `Box::into_raw`; `dealloc` is a single
//!   `Box::from_raw` + implicit drop. No free list, no reuse. The API
//!   surface is identical to the pool-backed variant so the caller
//!   ([`poly_list`](super::poly_list)) is `#[cfg]`-free on the alloc
//!   path.
//!
//! The thread-local [`POOL`] is unconditional; only its inner type's
//! behaviour changes with the feature flag. At `--release`, the
//! compiler inlines the RefCell / thread_local indirection of the
//! forwarder variant so the overhead vs a plain `Box::new` is
//! negligible — the pool-off configuration is a valid performance
//! baseline for the A/B comparison, not just an API-compatibility
//! layer.
//!
//! See ADR-016 for the motivation (Singular's omalloc PolyBin
//! equivalent), the design trade-offs (unbounded growth, thread-local
//! rather than shared), and the safety argument for
//! `unsafe impl Send + Sync for Poly`.

use std::cell::RefCell;
use std::ptr::NonNull;

use crate::field::Coeff;
use crate::monomial::Monomial;

use super::poly_list::Node;

// --- Pool-backed variant: under the linked_list_poly_pool feature. ---

#[cfg(feature = "linked_list_poly_pool")]
pub(super) struct NodePool {
    free: Vec<NonNull<Node>>,
}

#[cfg(feature = "linked_list_poly_pool")]
impl NodePool {
    const fn new() -> Self {
        Self { free: Vec::new() }
    }

    /// Allocate a `Node` with the given fields. Reuses a node from the
    /// free list if one is available; otherwise falls through to a
    /// single `Box::leak`. O(1) in the steady state.
    pub(super) fn alloc(
        &mut self,
        coeff: Coeff,
        mono: Monomial,
        next: Option<NonNull<Node>>,
    ) -> NonNull<Node> {
        if let Some(ptr) = self.free.pop() {
            // SAFETY: `ptr` was produced by an earlier `Box::leak` or
            // `Box::into_raw` and handed back to us via `dealloc` with
            // its `next` field already taken. Its storage is a
            // well-aligned `Node` slot that nothing else references,
            // so overwriting the whole `Node` struct is sound.
            unsafe {
                std::ptr::write(ptr.as_ptr(), Node { coeff, mono, next });
            }
            ptr
        } else {
            let b = Box::new(Node { coeff, mono, next });
            // SAFETY: `Box::into_raw` never returns null.
            unsafe { NonNull::new_unchecked(Box::into_raw(b)) }
        }
    }

    /// # Safety
    ///
    /// * `ptr` must point to a `Node` that is no longer reachable from
    ///   any live `Poly`.
    /// * The caller must have already `take()`-d `ptr`'s `next` field
    ///   (or equivalently, cleared it). We do **not** chain-free to
    ///   avoid stack-recursive drop on long lists.
    /// * `ptr` must have been obtained from an earlier
    ///   [`NodePool::alloc`] call on this thread (it is pushed back
    ///   onto the same thread's free list).
    pub(super) unsafe fn dealloc(&mut self, ptr: NonNull<Node>) {
        // No in-place drop needed: `Node` holds only `Coeff` (u32),
        // `Monomial` (a POD-shape struct with no allocating `Drop`),
        // and `Option<NonNull<Node>>` (trivially `Drop`). Pushing the
        // slot straight onto the free list is sound because any
        // subsequent `alloc` overwrites the whole struct via
        // `std::ptr::write`, which does **not** call drop on the old
        // contents (the old contents were logically consumed at this
        // `dealloc` call by moving them out through the caller's
        // field-level accesses).
        //
        // If `Node` ever grows a field whose drop has side effects
        // (e.g. an `Arc<...>` or a heap-owning `Monomial`), this path
        // needs an explicit `ptr::drop_in_place(ptr.as_ptr())` here.
        self.free.push(ptr);
    }

    /// Number of nodes currently on the free list. Test-only helper.
    #[cfg(test)]
    pub(super) fn free_len(&self) -> usize {
        self.free.len()
    }
}

// --- Forwarder variant: when linked_list_poly_pool is off. ---

#[cfg(not(feature = "linked_list_poly_pool"))]
pub(super) struct NodePool;

#[cfg(not(feature = "linked_list_poly_pool"))]
impl NodePool {
    const fn new() -> Self {
        Self
    }

    /// Allocate a `Node` via `Box::new`; hand back the raw pointer so
    /// the caller's API matches the pool-backed variant exactly.
    pub(super) fn alloc(
        &mut self,
        coeff: Coeff,
        mono: Monomial,
        next: Option<NonNull<Node>>,
    ) -> NonNull<Node> {
        let b = Box::new(Node { coeff, mono, next });
        // SAFETY: `Box::into_raw` never returns null.
        unsafe { NonNull::new_unchecked(Box::into_raw(b)) }
    }

    /// # Safety
    ///
    /// Same contract as the pool-backed variant's `dealloc`:
    /// * `ptr` must point to a `Node` no longer reachable from any
    ///   live `Poly`.
    /// * The caller must have already taken `ptr`'s `next` field so
    ///   the implicit `Box` drop here does not recurse down a chain.
    /// * `ptr` must have been obtained from an earlier
    ///   [`NodePool::alloc`] call (the `Box::from_raw` here reverses
    ///   the `Box::into_raw` there).
    pub(super) unsafe fn dealloc(&mut self, ptr: NonNull<Node>) {
        // Safety-checked by contract above: reclaim the `Box` and drop
        // it. The implicit drop walks only this one node because the
        // caller has already cleared `next`.
        unsafe { drop(Box::from_raw(ptr.as_ptr())); }
    }

    /// Test-only stub so the same test code compiles under either
    /// variant. Always zero because the forwarder keeps no free list.
    #[cfg(test)]
    pub(super) fn free_len(&self) -> usize {
        0
    }
}

// --- Shared thread-local handle (both variants). ---

thread_local! {
    /// Per-thread [`NodePool`]. All `Poly` allocations and
    /// deallocations in the List backend route through this.
    ///
    /// The pool-backed variant holds a free-list `Vec<NonNull<Node>>`
    /// and is owned by this thread for the thread's lifetime. Nodes
    /// allocated on one thread must be freed on the same thread (see
    /// the `unsafe impl Send + Sync for Poly` safety comment in
    /// `poly_list.rs`). The forwarder variant is stateless and its
    /// thread locality is an implementation detail of the inlined
    /// `Box::new` / `Box::from_raw` path.
    pub(super) static POOL: RefCell<NodePool> = const { RefCell::new(NodePool::new()) };
}
