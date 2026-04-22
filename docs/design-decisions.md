# rustgb Design Decisions

A running record of architectural decisions in the rustgb crate
(`~/rustgb`), the Singular dyn_module that loads it (`~/Singular-rustgb`),
and the dispatch shim (`rustgb-dispatch.lib`).

This is a decisions ledger, not a how-to. The port plan
(`~/project/docs/rust-bba-port-plan.md`) covers the algorithm;
profile reports (`~/project/docs/profile-rustgb-*.md`) cover
measurements; status reports (`~/project/docs/rustgb-*-report.md`)
cover individual task outcomes. This file captures the *why*
behind structural choices that the code itself will not explain.

This file used to live at `~/project/docs/rustgb-design-decisions.md`;
it was moved to `~/rustgb/docs/design-decisions.md` on 2026-04-22 so
that ADRs commit alongside the code that implements them.

## Format

Each decision is a numbered ADR-style entry with these sections:

- **Status** — Accepted / Superseded by #N / Under review
- **Date** — when the decision was made
- **Context** — what problem we are deciding about
- **Singular's approach** — what `~/Singular` does here. Always
  filled in: Singular is the reference implementation we are
  porting from.
- **FLINT's approach** — what `~/flint` does here, when applicable.
  FLINT does not implement Gröbner bases, so for high-level GB
  decisions (pair criteria, sugar strategy, redundancy marking,
  etc.) this section is **N/A — FLINT has no GB engine**. For
  polynomial-layer decisions (storage, multiplication, division,
  geobuckets), FLINT is consulted as a second reference point.
- **Decision** — what rustgb does and why.
- **Consequences** — costs, follow-ups, things that have to be
  true elsewhere as a result.
- **References** — file:line citations, profile reports, prior
  discussion in transcripts (`~/.claude/projects/-home-claude/*.jsonl`).

The Singular/FLINT comparison is a standing requirement, not an
optional section. If a decision diverges from both references, the
ADR must say so explicitly and justify it. If FLINT genuinely does
not address the question (no GB engine), say so explicitly with the
N/A wording above — silence is not allowed because it could mean
either "doesn't apply" or "wasn't checked."

---

## ADR-001: Polynomial storage — flat parallel arrays with a head cursor

**Status:** Accepted
**Date:** 2026-04-21

### Context

A polynomial in rustgb needs to support: descending iteration over
terms, O(1) leading-term access, in-place leading-term drop (called
millions of times per `bba()` from the geobucket cancellation peel),
arithmetic via merge (linear in `|f| + |g|`), and `Send + Sync` for
later parallelisation.

The choice of storage shape is load-bearing for the reducer's wall
clock — the staging-5101449 profile (`profile-rustgb-staging-5101449.md`)
showed that a naive choice put 62.6 % of total cycles into a single
`memmove` instruction.

### Singular's approach

Polynomials are **singly linked lists** of `spolyrec` nodes
(`~/Singular/libpolys/polys/monomials/p_polys.h`). Each node carries
a coefficient (`number`) and an inline exponent buffer; `pNext(p)`
walks the list. "Drop the leading term" is `pIter(p)` —
`p = pNext(p)` plus a `p_FreeBinAddr` of the old head; both O(1).
Iteration follows next-pointers; allocation is via `omalloc` bins
sized for `spolyrec`.

Cost: O(1) per drop-leading and O(1) per term-allocate (omalloc bin).
Trade-off: poor cache locality across long polys, pointer overhead
per term (~8 bytes / term wasted), allocator round-trip per term.

### FLINT's approach

FLINT's `nmod_mpoly` (`~/flint/src/nmod_mpoly`) stores polynomials
as **flat parallel arrays**: `mp_limb_t * coeffs` and
`ulong * exps` (packed), plus a `length` field. Iteration is array
indexing; arithmetic outputs are written term-by-term into freshly
allocated arrays.

FLINT never needs an O(1) "drop the front term" operation because
the reducer is heap-based (Monagan-Pearce, see ADR-003) and
operates on *indices into source arrays*, not on a partial-sum
structure that gets mutated.

### Decision

rustgb stores polynomials as parallel `Vec<Coeff>` and
`Vec<Monomial>` (`~/rustgb/src/poly.rs`), plus a `head: usize`
cursor. The live region is `coeffs[head..] / terms[head..]`.
`drop_leading_in_place` is `self.head += 1; self.refresh_cache();`
— O(1). Dead prefix is reclaimed when the `Poly` is cloned (custom
`Clone` impl copies only `[head..]`) or dropped.

This combines FLINT's storage shape (flat, cache-friendly,
allocator-light) with Singular's O(1) drop-leading semantics
(needed because we do use a geobucket reducer — see ADR-002).

### Consequences

- All `Poly` accessors (`coeffs()`, `terms()`, `iter()`, `leading()`,
  `len()`, `is_zero()`) return / operate on the live region.
- Internal arithmetic (`merge`, `sub_mul_term`) takes the live
  slice once at the top via `coeffs()` / `terms()`, avoiding
  per-iteration `+ self.head` arithmetic.
- `PartialEq` compares live regions, not raw vectors — two
  algebraically equal polys with different drop histories must
  still be `==`.
- Custom `Clone` is required: `derive(Clone)` would clone the dead
  prefix forever across the bucket-slot reuse pattern, bounding
  memory poorly.
- A bucket slot that is dropped many times without an intervening
  `merge` keeps the original allocation pinned. The next `merge` /
  `absorb` produces a fresh `Poly` with `head: 0`, returning that
  memory to bounded use. Worst case is a slot that drops to near-empty
  without ever being absorbed — bounded by the size of the original
  poly that was put in.
- The `head: usize` mechanic is the kind of subtle invariant that
  decays without targeted tests. Locked in by
  `poly::tests::drop_leading_in_place_walks_head_cursor`.

### References

- `~/rustgb/src/poly.rs` (struct definition lines 24-71, accessors
  lines 269-330, `drop_leading_in_place` lines 352-372)
- `~/project/docs/profile-rustgb-staging-5101449.md` (62.6 % memmove
  before this fix)
- `~/Singular/libpolys/polys/templates/p_kBucketSetLm__T.cc:60-63,80-83`
  (Singular's `pIter` peel)
- `~/flint/src/nmod_mpoly/divrem_monagan_pearce.c:139-344` (FLINT's
  array-based polys + heap reducer)
- Decision driven by perf profile of 2026-04-21; superseded the
  earlier `Vec::remove(0)` implementation (commit prior to this ADR).

---

## ADR-002: Reducer architecture — geobucket with cancellation peel

**Status:** Accepted, but flagged for re-evaluation if performance
work shifts the bottleneck (see ADR-003 alternative).
**Date:** 2026-04-21 (formalising a decision implicit in the port plan)

### Context

The bba inner loop reduces an L-object against the current basis
by repeatedly subtracting `c * m * g_i` for matching basis elements
`g_i`. Done naively this allocates intermediate polys whose length
is `O(|h| + |g_i|)` per step. Real reducers buffer these subtractions
into a structure that can answer "what is the current leading term?"
without flushing the whole partial sum.

### Singular's approach

**Geobucket** (`~/Singular/libpolys/polys/kbuckets.cc`,
`p_kBucketSetLm__T.cc`). A bucket has `BIT_SIZEOF_LONG` slots; slot
`i` holds a poly whose length is in roughly `(2^(2(i-1)), 2^(2i)]`.
Adding a poly puts it into the slot matching its length, with carry
to higher slots when slots fill. The leading term is found by
scanning all non-empty slots (O(slots) = ~32) for the maximum head
monomial; if multiple slots have the same head and their coefficients
sum to zero, those heads are peeled off and the scan repeats.

Drop-leading inside the peel is `pIter` on a linked-list poly —
O(1) per peel. The popped leader is parked in slot 0 so the next
`kBucketGetLm` reads it directly without rescanning.

### FLINT's approach

**N/A — FLINT does not use a geobucket reducer.** FLINT *has* a
geobucket data structure (`~/flint/src/nmod_mpoly/geobuckets.c`),
but it has no `extract_leading` operation: the API is
`init / clear / empty / set / add / sub`. It is used purely as
a staging buffer for sums (e.g. polynomial composition), with
`empty()` collapsing slots into a single output poly via repeated
binary additions.

FLINT's actual reducer is heap-based — see ADR-003.

### Decision

rustgb uses a Singular-style geobucket
(`~/rustgb/src/kbucket.rs`) with the same cancellation-peel
algorithm as `p_kBucketSetLm__T`. `KBucket::leading()` is the
non-extracting probe (returns the current leader, mutates only to
peel cancellations); `KBucket::extract_leading()` pops the leader
algebraically. Both call `Poly::drop_leading_in_place` on slots,
which under ADR-001 is O(1).

Reasons we picked geobucket over a heap reducer:
1. The port plan (`rust-bba-port-plan.md` §5) specified mirroring
   Singular's algorithm closely so the validation surface (Singular
   regression suite + helium staging suite) compares apples to
   apples on outputs.
2. The geobucket integrates cleanly with redHomog and the sugar
   strategy already in `lobject.rs`. A heap reducer would require
   rethinking how sugar is propagated through pending products.
3. The peel cost was misattributed to the algorithm in the
   2026-04-21 profile; the real culprit was poly storage, fixed
   by ADR-001 without touching the reducer.

### Consequences

- 32 slots, single-poly per slot. No `coef[]` lazy-multiply
  optimisation (Singular's `USE_COEF_BUCKETS`). May want to revisit
  if scalar multiplications dominate later.
- The `lm_cache` field caches the most recent `leading()` result
  and is invalidated by `dirty` mask. Plays the role of Singular's
  "park leader in slot 0," but without splitting the poly.
- If a future profile shows the reducer remains the bottleneck even
  after ADR-001's memmove fix, a Monagan-Pearce heap reducer
  (ADR-003 candidate) is the obvious next target — but that's a
  multi-week rewrite, not a tweak.

### References

- `~/rustgb/src/kbucket.rs` (`leading()` lines 288-373,
  `extract_leading()` lines 379-410)
- `~/Singular/libpolys/polys/templates/p_kBucketSetLm__T.cc` (full
  comparison in `profile-rustgb-staging-5101449.md`)
- `~/flint/src/nmod_mpoly/geobuckets.c` (FLINT's sum-only
  geobucket, no `extract_leading`)
- `~/project/docs/rust-bba-port-plan.md` §5

---

## ADR-003 (candidate, not yet adopted): Heap-based reducer (Monagan-Pearce)

**Status:** Under review — listed for visibility, not active.
**Date:** placeholder

### Context

If the geobucket cancellation peel becomes the bottleneck again
after polynomial-layer optimisations are exhausted, the alternative
is a heap-based reducer.

### Singular's approach

Singular has a heap reducer for some operations (`kspoly.cc` paths
with `kStratHeap`), but `bba` uses the geobucket path by default.

### FLINT's approach

FLINT's `divrem_monagan_pearce` (`~/flint/src/nmod_mpoly/divrem_monagan_pearce.c`,
726 lines) maintains a min-heap of pending products `(c_i * c_j,
e_i + e_j)` indexed by `(i, j)` into source arrays. Each iteration
pops the maximum monomial off the heap; if it matches the current
target leader, the term is consumed and the next `(i, j+1)` product
is pushed back onto the heap. There is no partial-sum structure to
peel — "drop the leading term" is `j++`.

### Decision (deferred)

Not adopted. To be reconsidered if a future profile shows the
reducer still dominates after ADR-001's storage fix and any further
polynomial-layer tuning.

### Consequences (if adopted)

Would obsolete `kbucket.rs` entirely. Would require redesigning
sugar propagation through heap nodes. Would integrate poorly with
the current `LObject::refresh` + `KBucket::leading` split — heap
nodes are the unit of state, not buckets.

### References

- `~/flint/src/nmod_mpoly/divrem_monagan_pearce.c`
- See ADR-002 for current decision and rationale.

---

## ADR-004: Threading dispatch via `RUSTGB_THREADS` env var

**Status:** Accepted (provisional — interface may change once the FFI
gains a thread-count parameter).
**Date:** 2026-04-21 (formalising existing behaviour)

### Context

The serial and parallel reducers live in the same crate
(`bba::compute_gb_serial`, `parallel::compute_gb_parallel`). The
FFI presents a single entry point (`rustgb_compute`); we need a way
to choose between them without changing the C ABI.

### Singular's approach

Singular's parallel-bba branch (`~/Singular-parallel-bba`) reads
`SINGULAR_THREADS` from the environment and threads it through the
strategy struct. Same shape: env var → driver-internal int.

### FLINT's approach

FLINT uses an explicit `nthreads` argument on functions that can
parallelise (e.g. `nmod_mpoly_mul_threaded`, `divides_heap_threaded`).
No global env-var convention — the caller is responsible.

### Decision

rustgb reads `RUSTGB_THREADS` from the environment in
`bba::rustgb_threads()` (default 1, clamped to `[1, 256]`). At
`T == 1` the serial path runs; at `T >= 2` it dispatches to
`parallel::compute_gb_parallel`. The FFI does not expose a thread
parameter — Singular sets the env var before calling the dispatch
shim if the user wants threading.

This matches Singular's convention so users who set
`SINGULAR_THREADS` can additionally set `RUSTGB_THREADS` with the
same mental model. It diverges from FLINT's explicit-arg style.

### Consequences

- Cancellation from the FFI side is not yet wired; `compute_gb`
  `expect`s the parallel computation to complete. The `parallel`
  module exposes `CancelHandle` for callers that need it, but the
  FFI path doesn't use it.
- The `parallel` path has not been validated against the staging
  suite as of 2026-04-21 (that is task 318). The 890s
  staging-5101449 run from 2026-04-21 was serial because the
  validation runner doesn't set the env var.
- If we ever want per-call thread control, the FFI would need a
  new entry point (`rustgb_compute_threaded(input, n)`); the env
  var path can stay as the default.

### References

- `~/rustgb/src/bba.rs:75-80` (`rustgb_threads()`)
- `~/rustgb/src/bba.rs:56-70` (dispatch in `compute_gb`)
- `~/rustgb/src/parallel.rs:86-91` (`compute_gb_parallel` signature)

---

## ADR-005: Monomial representation — direct exponents, 7 bits/var, divmask overflow guard

**Status:** Accepted and implemented (supersedes the original
complemented-storage representation that the initial commit shipped
with). Landed alongside this ADR's commit in `~/rustgb`.
**Date:** 2026-04-21 (decision); 2026-04-22 (implementation)

### Context

Profile v2 (`profile-rustgb-v2-staging-5101449.md`) showed
`Monomial::mul` at 30 % of total cycles after ADR-001's
head-cursor fix removed the prior memmove bottleneck. The cost is
not the multiplication itself — it's the per-byte loop with an
overflow check on every byte:

```rust
for b_idx in low_byte..=high_byte {                  // 5.93% loop control
    let ca = (a >> shift) & 0xFF;
    let cb = (b >> shift) & 0xFF;
    if ca + cb < 0xFF { return None; }                // 3.73% per-byte check
    let cnew = ca + cb - 0xFF;
    new_word |= cnew << shift;
}
```

The byte-by-byte structure exists because rustgb stored
**complemented** exponents (`255 − e`) so that lex-comparison of
the packed words encoded degrevlex order directly. That choice
made `cmp` cheap (~6 % of cycles) at the price of making `mul`
expensive: `(255 − a) + (255 − b) ≠ 255 − (a + b)`, so each byte
needs a `−0xFF` correction and an overflow check.

The question this ADR answers: how should monomial multiplication,
exponent storage, and overflow handling be structured?

### Singular's approach

**Storage:** direct (`e_v`, not complemented). Packed into u64
words at a per-ring configurable bits-per-variable, with one
**guard bit** reserved per variable slot so overflow can be
detected by examining that bit after addition.

**Multiplication:** plain word-wise add. `p_ExpVectorAdd`
(`p_polys.h:1432-1444`) reduces to `p_MemAdd_LengthGeneral`
(`templates/p_MemAdd.h`) which is `for (i=0; i<ExpL_Size; i++)
p1->exp[i] += p2->exp[i]`. Length-specialised macros unroll for
small word counts.

**Overflow handling — three layers:**

1. *Word-level mul itself is unchecked.* Just the plain add. The
   PDEBUG check (`pAssume1((unsigned long)(...) <= r->bitmask)`) is
   debug-only.

2. *Cheap pre-check using the divmask trick.*
   `p_LmExpVectorAddIsOk` (`p_polys.h:2020-2038`) is called before
   every spoly creation and every reducer step (call sites in
   `kspoly.cc:125, 260, 403, 540, 662, 876, 1123`):
   ```c
   if ( (l1 > ULONG_MAX - l2) ||
        (((l1 & divmask) ^ (l2 & divmask)) != ((l1 + l2) & divmask)))
     return FALSE;
   ```
   `divmask` has the guard bit set in each variable slot. If a
   byte overflows out of its slot, the carry corrupts the
   guard-bit pattern and the XOR check catches it. Branch-free,
   O(words) per check.

3. *Dynamic tail-ring widening.* When the pre-check fails,
   `kStratChangeTailRing` (`kutil.cc:10939-11034`) doubles the
   bitmask, builds a new ring via `rModifyRing`, and migrates
   every entry in `strat->T`, `strat->L`, `strat->P` into the
   wider representation via `ShallowCopyDelete`. Returns
   `FALSE` only if `expbound >= currRing->bitmask` (the absolute
   user-declared ceiling), at which point the bba driver emits
   `WerrorS("OVERFLOW...")` and bails out.

**Comparison:** uses `p_MemCmp__T` plus an `ordsgn` (sign vector)
so each ordering type can flip word-direction without dispatch
overhead.

### FLINT's approach

**Storage:** direct exponents, packed into `ulong` limbs at
per-poly bits-per-field. The `bits` field travels with the poly
(not the ring), so distinct polys can have different packings.

**Multiplication:** plain word-wise add. `mpoly_monomial_add`
(`flint/src/mpoly.h:233-240`):
```c
FLINT_FORCE_INLINE
void mpoly_monomial_add(ulong * exp_ptr, const ulong * exp2,
                                         const ulong * exp3, slong N)
{
   for (slong i = 0; i < N; i++)
      exp_ptr[i] = exp2[i] + exp3[i];
}
```
With a multi-limb variant (`_mp`) deferring to `__gmpn_add_n` and
multiply-add variants (`madd`/`msub`) for the heap reducer's
pending-product accumulation.

**Overflow handling:** post-hoc detection plus repack.
`mpoly_monomials_overflow_test` is run as a separate verification
pass on the result; on overflow `repack_monomials` widens
bits-per-field and re-encodes the poly. Less aggressive than
Singular: there's no per-multiply pre-check; FLINT relies on
either a generous initial `bits` or running the test after batch
operations and repacking once.

**Comparison:** per-ordering routines selected at compile time
(`monomials_cmp.c`); direct storage means each ordering's compare
encodes the direction in its own code, no per-word sign lookup at
runtime.

### Decision

Replace the current Monomial representation with:

1. **Direct storage of exponents** — store `e_v`, not `255 − e_v`.
   `Monomial::mul` becomes plain wrapping-add per u64 word.
2. **7 bits per variable, top bit as overflow guard.** Maximum
   single-variable exponent drops from 255 to 127. For the helium
   workload (max degree ~30) this is comfortable headroom; for any
   workload that exceeds it we error early.
3. **Cheap divmask-style overflow detection at the multiply
   site,** matching Singular's pre-check: a precomputed guard-bit
   mask in the `Ring` (`overflow_mask: [u64; 4]` with bit 7 set in
   each variable byte slot, 0 in the total-deg byte and unused
   slots), checked via `(a & mask) ^ (b & mask) != (a + b) & mask`
   per word. Auto-vectorisable; ~2 ops per word, ~8 ops for the
   whole packed block.
4. **Per-spoly `max_exp` caching is deferred** but the
   architecture leaves room for it: `Poly` can grow a
   `max_exp: Monomial` cache later, computed incrementally on
   `add_assign` / `merge`. When that lands, `KBucket::minus_m_mult_p`
   can do the overflow check **once** per reducer step against
   `multiplier + g.max_exp` and skip the per-term check entirely
   inside the inner loop (matching Singular's `kspoly.cc`
   pattern).
5. **`cmp_degrevlex` keeps word-level lex compare,** but applies a
   precomputed XOR mask to flip variable-byte direction at compare
   time. The mask (`cmp_flip_mask: [u64; 4]`, `0x7F` in variable
   byte slots, `0x00` in the total-deg byte and unused slots) is
   stored in the `Ring` and applied as `a.packed[i] ^ mask[i]`
   before each per-word compare. Cost: one extra XOR per word per
   cmp.
6. **No dynamic ring widening yet.** On overflow, panic with a
   clear "exponent exceeds 7-bit packing" message. Listed as a
   deferred enhancement (see Consequences).

### Consequences

**Performance:** Profile v2 hotspot reshuffle prediction was:

| Function | v2 (current) | After ADR-005 | Notes |
|---|---|---|---|
| `Monomial::mul` | 30.0 % | ~3-5 % | word-add + cheap check |
| `Monomial::cmp` (under merge) | ~6 % | ~9 % | XOR per word added |
| Net effect on wall | — | **~ −22 %** | residue stays the same shape |

**Measured (post-implementation, 2026-04-22):**

| Test | Pre-ADR-005 wall | Post-ADR-005 wall | Δ |
|---|---|---|---|
| staging-5101449 | 255 s | **204 s** | −20 % (matches prediction) |
| staging-5104053 | 311 s | **262 s** | −16 % |
| staging-5106746 | 484 s | **348 s** | −28 % |

All three staging tests pass with exact fixture matches. A v3 perf
profile (next ADR work) should confirm the predicted hotspot shift
to `poly::merge` and `KBucket::leading`.

**Safety:** strictly stronger than the original "trust silently
in release" sketch. The divmask check catches every overflow at
the multiply site with negligible cost. Worst case is a clear
panic, never silent corruption.

**Capacity:** max single-variable exponent 127, max total degree
still bounded by the cached `total_deg: u32`. The helium workload
peaks well under both limits.

**Implementation surface (all in `monomial.rs` + a constant in
`Ring`):**
- `Monomial::from_exponents`: write `e_v` directly. Validate
  `e_v < 128`.
- `Monomial::mul`: 4×u64 wrapping-add + divmask overflow check +
  total_deg / sev update. Returns `Option` only because of
  `total_deg` u32 sum overflow (extremely unlikely; could become
  infallible).
- `Monomial::cmp_degrevlex`: XOR with `ring.cmp_flip_mask` per
  word before comparison.
- `Monomial::div`, `Monomial::lcm`, `Monomial::divides`: also
  per-byte today; rewrite to word-level (`div` = wrapping-sub;
  `divides` = "every byte of `a` ≤ corresponding byte of `b`",
  expressible as `(a + (~b & mask)) & mask == 0` style trick).
- `Ring`: add `overflow_mask: [u64; 4]` and
  `cmp_flip_mask: [u64; 4]`, both computed once at `Ring::new`
  from `nvars`.
- `assert_canonical` / `sev` computation: straightforward update
  to read direct exponents.
- All existing tests should pass unchanged (public API is
  identical); a new test should explicitly exercise the overflow
  detection panic.

**Deferred enhancement: per-spoly max_exp caching.** Once
`Poly::max_exp` is plumbed through `add_assign` / `merge`,
`KBucket::minus_m_mult_p` can hoist the overflow check out of the
inner loop. Skipping it in the inner loop is worth maybe another
1-2 % wall, not urgent.

**Deferred enhancement: dynamic ring widening.** Multi-week
project mirroring `kStratChangeTailRing`: requires a mutable
`Ring`, polynomial migration via `ShallowCopyDelete`, and proc-table
swaps. Not currently needed; revisit if a future workload exceeds
7-bit per-variable packing.

**Supersession:** This ADR overturns the original choice (made
implicitly when `monomial.rs` was first written) of
complemented-exponent storage for cheap comparison. The original
choice optimised the wrong half of the tradeoff; profile evidence
showed cmp was always cheap relative to mul, regardless of
representation.

### References

- `~/rustgb/src/monomial.rs:185-225` (current `Monomial::mul`,
  per-byte loop with overflow check — the code being replaced)
- `~/rustgb/src/monomial.rs:370-402` (current `cmp_degrevlex`,
  word-level lex on complemented exponents)
- `~/Singular-rustgb/libpolys/polys/monomials/p_polys.h:1432-1444`
  (`p_ExpVectorAdd`)
- `~/Singular-rustgb/libpolys/polys/templates/p_MemAdd.h`
  (`p_MemSum_LengthGeneral` and length-specialised macros)
- `~/Singular-rustgb/libpolys/polys/monomials/p_polys.h:2020-2038`
  (`p_LmExpVectorAddIsOk`, the divmask trick)
- `~/Singular-rustgb/kernel/GBEngine/kutil.cc:10939-11062`
  (`kStratChangeTailRing`, `kStratInitChangeTailRing`)
- `~/Singular-rustgb/kernel/GBEngine/kstd2.cc:2706-2748`
  (overflow handling in the bba main loop)
- `~/Singular-rustgb/kernel/GBEngine/kspoly.cc:120-138` (per-spoly
  pre-check + retry pattern)
- `~/flint/src/mpoly.h:233-282` (`mpoly_monomial_add` and
  `madd` / `msub` family)
- `~/project/docs/profile-rustgb-v2-staging-5101449.md` (the
  30 % `Monomial::mul` evidence that motivated this ADR)

---

## ADR-006: poly::merge — pre-allocated output, FLINT-style index writes

**Status:** Accepted and implemented. Landed alongside this ADR's
commit in `~/rustgb`.
**Date:** 2026-04-22

### Context

After ADR-005 collapsed `Monomial::mul` from 30 % of cycles to
fully-inlined-out, the v3 profile
(`~/project/docs/profile-rustgb-v3-staging-5101449.md`) showed
`poly::merge` as the new top concentrated function at 21.2 % of
total cycles. Inside `merge`, `Vec::push` accounted for 7.0 % of
total cycles (3.4 % of which was `core::ptr::write` itself), with
the remainder split between `Monomial::cmp` (7.0 %) and the loop
body (~7 %).

`merge` is hot because every reducer step that absorbs a non-empty
`build_neg_cmp` result into a non-empty geobucket slot fires it
(via `KBucket::absorb` → `Poly::add` → `merge`). Across an
entire bba run on staging-5101449 that's millions of merge calls,
each one constructing a fresh `Vec<Coeff>` and `Vec<Monomial>`
output by repeated `push`-ing.

The question this ADR answers: how should the merge emit its
output terms?

### Singular's approach

Singular's `p_Add_q__T` (`~/Singular-rustgb/libpolys/polys/templates/p_Add_q__T.cc`,
86 lines) sidesteps the question entirely by **never allocating
output nodes**. Polynomials are linked lists of `spolyrec` nodes;
emitting a term is one pointer write that splices an existing input
node into the output list:

```c
Greater:
  a = pNext(a) = p;     // splice existing node into output
  pIter(p);             // advance source pointer
  if (p==NULL) { pNext(a) = q; goto Finish; }   // O(1) tail splice
  goto Top;
```

When one input is exhausted, `pNext(a) = q` joins the entire
remaining tail of the other in **one pointer write**, regardless of
length. The input lists are explicitly destroyed by the call (the
docstring says `Destroys: p, q`). The "Equal" path adds
coefficients in place via `n_InpAdd__T`, freeing the q-side node;
on cancellation, `n_Delete__T` + `p_LmFreeAndNext` consume both
nodes. Cmp uses `p_MemCmp__T` (word-wise compare with `ordsgn`).

Per-emitted-term cost: **one pointer write + one pointer chase +
one cmp**. No allocation, no copy of coefficient or exponent data.

This is structurally inaccessible to rustgb: we picked flat-array
storage in ADR-001 (head-cursor over a `Vec`), so we don't have
linked-list nodes to splice and there is no "tail splice" trick
available. Adopting Singular's design here would require redoing
ADR-001.

### FLINT's approach

FLINT's `_nmod_mpoly_add` (`~/flint/src/nmod_mpoly/add.c:16-67` and
the general-N variant at lines 69-124) uses flat parallel arrays
with **pre-allocated output and direct index writes**:

```c
if ((Bexps[i]^maskhi) > (Cexps[j]^maskhi)) {
    Aexps[k] = Bexps[i];
    Acoeffs[k] = Bcoeffs[i];
    i++;
}
else if ((Bexps[i]^maskhi) == (Cexps[j]^maskhi)) {
    Aexps[k] = Bexps[i];
    Acoeffs[k] = nmod_add(Bcoeffs[i], Ccoeffs[j], fctx);
    k -= (Acoeffs[k] == 0);   // branch-free cancellation skip
    i++; j++;
}
else { /* mirror */ }
k++;
```

Three structural choices:

1. **Pre-allocated output.** The wrapper `nmod_mpoly_add`
   (`add.c:169-186`) calls `nmod_mpoly_init3(T, B->length + C->length, ...)`
   to size the output to the worst case (no cancellation) before
   the inner loop begins.
2. **Index-and-write.** The inner loop writes into pre-sized array
   slots (`Aexps[k] = Bexps[i]; Acoeffs[k] = Bcoeffs[i]`). No bounds
   check, no length update per write.
3. **Branch-free cancellation.** `k -= (Acoeffs[k] == 0)` decrements
   the write cursor when the result was zero, "uncommitting" the
   slot. No `if`-then-skip-push branch.

Length is recovered at the end: `return k` → caller assigns
`T->length = k`.

There is also a hot-path specialisation `_nmod_mpoly_add1` for
`N == 1` (single-limb exponents), which inlines the cmp as
`(Bexps[i] ^ maskhi) > (Cexps[j] ^ maskhi)` rather than calling
`mpoly_monomial_cmp`. rustgb's exponent block is always 4 u64s, so
this specialisation does not apply directly, though the cmp is
already inlined via `cmp_degrevlex`.

### Decision

Adopt FLINT's pattern verbatim. Concretely:

1. **Pre-allocate** `out_c` and `out_m` to the upper-bound capacity
   `a.len() + b.len()` (already done in the existing code, but
   currently followed by `push`).
2. **Write via `Vec::spare_capacity_mut()` + `MaybeUninit::write`**
   instead of `Vec::push`. This skips the per-push bounds-check
   against `len < capacity` and the per-push length increment.
   Writing into the spare-capacity slice is safe; only the final
   `set_len` call is `unsafe`.
3. **Branch-free cancellation** in the Less and Equal arms:
   ```rust
   spare_c[k].write(c);
   spare_m[k].write(m.clone());
   k += (c != 0) as usize;
   ```
   The write to slot `k` is wasted on cancellation (the next
   iteration overwrites the same slot), but the *branch* is gone —
   matching FLINT's `k -= (acc == 0)` shape with a `+=` instead of
   `-=` (we never speculatively bumped k, so we conditionally hold
   it back rather than conditionally back it off).
4. **Single `set_len`** at the end. The Vec is now logically
   length-`k` with `capacity - k` slots in spare; the wasted writes
   (if any) leak their bytes when the Vec eventually drops, which
   is fine because both `Coeff` (u32) and `Monomial` (POD struct)
   have no Drop side effects.

`sub_mul_term` (`poly.rs:503`) has the same structural pattern
(2-pointer merge with materialised `c·m·q` terms) and would
benefit from the same change, but it is not in the v3 hot path.
Defer until profile evidence shows it matters.

### Consequences

**Performance prediction:** the v3 profile attributed 7.0 % of
total cycles to `Vec::push` inside `merge`. Eliminating the
per-push bounds-check + length-increment pair (keeping the
underlying `ptr::write`) should cut that to ~2-3 %, for **~4-5 %
wall reduction**. The `Monomial::clone()` cost (32-byte struct
copy) is unaffected; that's an unavoidable cost of value-move
into an array slot.

**Measured (post-implementation, v4 profile, samsung):**
`Vec::push` is **completely gone** from the v4 profile.
`poly::merge`'s share dropped from 21.2 % (v3) to 17.2 % (v4) —
the predicted ~4 percentage point reduction. Inside the new merge,
`Monomial::cmp` is 8.4 % and the loop body accounts for the rest;
no `Vec::push` line at all. Wall under perf load went 3:34 (v3) →
2:52 (v4), a 19 % reduction; the cleaner steady-state metric is
the ~4 pp share drop, which translates to roughly the predicted
wall improvement once contention noise is averaged out.
All staging tests still produce exact fixture matches.

The branch-free cancellation removes one branch per Equal-with-cancel
or Less-with-zero-coefficient case. Cancellation is rare in general
but happens reliably for the leading term in every `KBucket::absorb`
call (that's the algorithmic point of `minus_m_mult_p`), so the
saving is at least one branch per merge call.

**Safety:** the only `unsafe` is the final `set_len(k)`. The
write-through-spare-capacity pattern via `MaybeUninit::write` is
safe at compile time. The wasted-write slots beyond `k` are not
considered initialised by the Vec (`set_len` truncates), so they
are not dropped — but neither `Coeff` nor `Monomial` has a Drop
impl with side effects we care about, so this is correct.

**Capacity invariant:** `out.coeffs.capacity()` may exceed
`out.coeffs.len()` after the merge, by up to (number of cancelled
terms). For the typical bucket-absorb workload that's a single-digit
overhang, well within ordinary `Vec` slop. Not worth shrinking.

**API:** the `merge` signature is unchanged. Callers
(`Poly::add`, `Poly::sub`, `Poly::add_assign`) need no changes.

### References

- `~/rustgb/src/poly.rs:648-708` (current `merge` — the code
  being replaced)
- `~/Singular-rustgb/libpolys/polys/templates/p_Add_q__T.cc`
  (Singular's linked-list merge with O(1) tail splice; reference
  but structurally inapplicable to flat-array storage)
- `~/flint/src/nmod_mpoly/add.c:16-124` (FLINT's
  `_nmod_mpoly_add1` and `_nmod_mpoly_add` — the model adopted
  here)
- `~/flint/src/nmod_mpoly/add.c:126-196` (the wrapper
  `nmod_mpoly_add` showing the pre-allocation pattern)
- `~/project/docs/profile-rustgb-v3-staging-5101449.md` (the
  21.2 % `poly::merge` and 7.0 % `Vec::push` evidence that
  motivated this ADR)

---

## ADR-007: SIMD-batched sev pre-filter for the basis-sweep in `reduce_lobject`

**Status:** Accepted and implemented. Landed alongside this ADR's
commit in `~/rustgb`.
**Date:** 2026-04-22

### Context

The v4 profile (`~/project/docs/profile-rustgb-v4-staging-5101449.md`)
showed `bba::reduce_lobject` at 30.0 % of total cycles. A
per-instruction `perf annotate` revealed that the single hottest
instruction in the entire program is the sev pre-filter `jne`
inside the divisor-search loop:

```asm
                       :  if (s_sev & !lm_sev) != 0 {
0.68 :   2c18f:  test   %r15,(%rax,%r13,8)        ; sevs[idx] & !lm_sev
18.70 :  2c193:  jne    2c160                      ; if hits, skip
```

That `jne` alone is 18.70 % of within-function cycles ≈ 5.6 % of
total cycles. Including the rest of the per-iteration overhead
(loop bound check, redundant-flag check, sevs bounds check, sevs
load), the "skip-fast-path" of the basis sweep totals 41.78 %
within-function ≈ **12.5 % of total program cycles**. The actual
`Monomial::divides` call when the sev pre-filter passes adds
another ~6.7 % of total.

So roughly **19 % of the entire program's runtime is the
basis-sweep inside `reduce_lobject`** — the loop at `bba.rs:241-258`
that walks `s_basis` looking for a divisor of the current leader.
The sweep itself is simple, but it fires on every reduction step
across millions of reduction steps in a typical bba run, with the
basis growing to ~3000 elements by the end of staging-5101449.

The sev pre-filter is doing its algorithmic job (most candidates
get rejected). The cost is per-iteration *fixed overhead* — load,
test, branch — and three structural sources stall it:

1. The sev load misses L1 frequently as the basis grows past a
   few thousand u64s (cache thrashing during sweep).
2. The data-dependent branch is hard to predict.
3. Bounds checks Rust inserts for the indexed array accesses
   (`sevs[idx]`, `redund[idx]`) cost one branch per iteration.

This is the same pathology Singular's `next-opt` branch had on the
same workload — and the same fix.

### Singular's approach

Singular's `next-opt` branch introduced `kSevScanAVX2`
(`~/Singular-next-opt/kernel/GBEngine/kstd2.cc:74-121`):

```c
__attribute__((target("avx2")))
static inline int kSevScanAVX2(const unsigned long* sevT,
                               unsigned long not_sev,
                               int j, int tl)
{
  const __m256i vnot_sev = _mm256_set1_epi64x((long long)not_sev);
  const __m256i vzero    = _mm256_setzero_si256();
  // Main loop: 16 entries (4 batches of 4) per iteration
  while (j + 15 <= tl) {
    __builtin_prefetch(sevT + j + 16, 0, 1);
    __m256i vand1 = _mm256_and_si256(_mm256_loadu_si256(...), vnot_sev);
    __m256i vand2 = _mm256_and_si256(_mm256_loadu_si256(...), vnot_sev);
    __m256i vand3 = _mm256_and_si256(_mm256_loadu_si256(...), vnot_sev);
    __m256i vand4 = _mm256_and_si256(_mm256_loadu_si256(...), vnot_sev);
    __m256i vcmp1 = _mm256_cmpeq_epi64(vand1, vzero);
    /* ... vcmp2, vcmp3, vcmp4 ... */
    int combined = mask1 | mask2 | mask3 | mask4;
    if (__builtin_expect(combined != 0, 0)) {
      if (mask1) return j;
      if (mask2) return j + 4;
      if (mask3) return j + 8;
      return j + 12;
    }
    j += 16;
  }
  /* tail loop: 4 entries at a time, then scalar */
}
```

The pattern: load 4 sevs at a time via `_mm256_loadu_si256`,
compute `sevs & not_sev` per element via `_mm256_and_si256`,
compare against zero with `_mm256_cmpeq_epi64`, extract a 32-bit
mask (8 bits per qword) via `_mm256_movemask_epi8`. The main
loop is unrolled 4× to amortise the loop overhead and let the
common all-miss case branch only once per 16 entries.

There's also a SSE4.1 fallback (`kSevScanSSE4`,
`kstd2.cc:127-`) for non-AVX2 CPUs and a scalar fallback
(`kSevScan`). Runtime dispatch chooses the best available path
once per `bba` invocation.

The function returns the **first index** where the sev pre-filter
passes (or `tl` past end if none found). The caller checks the
redundant flag and runs the actual `divides` separately. This
keeps the SIMD code pure-sev and easy to reason about.

Singular's measured impact: kSevScanAVX2 took 7-11 % of total
cycles in the v6 profile (it absorbed the time previously paid
by `chainCritNormal` and the inner sweep), and the cumulative
optimisation (sev_flat + AVX2 scan) was the largest single
contributor to next-opt's ~36 % cumulative speedup.

### FLINT's approach

**N/A — FLINT has no GB engine.** The closest analogue is FLINT's
heap-based reducer's "process the next pending product" loop
(`divrem_monagan_pearce.c`), but that walks a heap, not a basis,
and the structure is fundamentally different. There is no
"sweep T-set looking for a divisor" idiom in FLINT to compare
against.

### Decision

Adopt the Singular pattern verbatim, gated by Rust's compile-time
`target_feature = "avx2"`. Concretely:

1. **Refactor the divisor search out of `reduce_lobject`** into a
   helper `find_divisor_idx(s_basis, lm_sev, lm, ring)` so the
   sweep code can be optimised independently and unit-tested in
   isolation.
2. **Inside the helper, dispatch on `cfg(target_feature = "avx2")`**:
   - **AVX2 path:** mirror `kSevScanAVX2`. Process the basis in
     batches; each batch loads 4 sevs, ANDs with `vnot_sev`, compares
     against zero, extracts a movemask, and (if any bit set) finds
     the first hit. For each hit, check `redundant[idx]` and call
     `Monomial::divides` exactly as the scalar path does. Manually
     unroll 4× as Singular does, for the same all-miss-fast-path
     reason.
   - **Scalar fallback:** the original loop, unchanged. Used when
     building without AVX2 enabled (e.g., on older CPUs like
     c200-1, which is Westmere-era).
3. **No SSE4.1 fallback for this first cut.** Adding SSE4.1 is
   straightforward (mirror Singular's `kSevScanSSE4`) but not
   necessary for correctness. Defer until measurement on c200-1
   shows it's worth the complexity.
4. **Runtime feature detection deferred.** Compile-time gating is
   simplest. Document in the rustgb README that release builds on
   AVX2 hardware should use `RUSTFLAGS="-C target-cpu=native"` to
   pick up the AVX2 path.

Note that the redundant-flag check and the `Monomial::divides`
probe stay scalar in the caller. Singular does the same — its
SIMD function returns just an index, and the caller checks
`redundant` and calls `divides`. Folding redund into the SIMD
batch is possible (read 4 redund bytes alongside the 4 sevs, AND
into the candidate mask) but adds complexity for marginal gain
since `redund[idx]` is itself a single byte load, already cheap.

### Consequences

**Performance prediction:** Singular's measurement suggests this
optimisation can cut the basis-sweep cost from ~12.5 % of total
cycles (rustgb v4) to the ~7-11 % range that Singular's `kSevScanAVX2`
occupies (which is a smaller share because Singular's reducer
also has other costs we don't have). Conservatively, **~5-8 %
wall reduction** on staging workloads. For a v5 profile,
`reduce_lobject`'s self-time should drop from 30 % to ~22-25 %,
with the freed share spreading proportionally.

**Measured (post-implementation, 2026-04-22, samsung):**
- Correctness: all three staging tests pass with exact fixture
  matches under both AVX2-enabled and scalar builds.
- SIMD activation verified: `objdump` shows 20 AVX2 instructions
  (vpand, vpcmpeq, vpmovmskb, vpbroadcastq, vmovdqu) inlined into
  `reduce_lobject` in the AVX2 build.
- Wall numbers under perf load are noisy on samsung due to
  background contention from concurrent processes; staging-5101449
  un-profiled walls ranged 123-226 s across runs of the same code,
  making per-ADR attribution unreliable. A clean v5 perf profile
  comparison (under controlled load) would settle the cycle-count
  attribution; deferred as the obvious next step.

**Build configuration:** the AVX2 path is compile-time gated. A
default `cargo build --release` on a non-AVX2-enabled rustc
configuration will use the scalar path. To opt in:
```
RUSTFLAGS="-C target-cpu=native" cargo build --release
```
This should be added to the README / build instructions. CI builds
on x86_64 hosts with AVX2 (the dev laptop and edge / c200-1's
successor systems) will pick it up automatically.

**Portability:** the scalar path is identical in behaviour. Both
paths are unit-tested against each other (a property test that
runs the same input through both and checks identical results).
Cross-compilation to non-x86 (e.g., ARM) falls through to the
scalar path with no special handling.

**Safety:** the AVX2 intrinsics live in `unsafe` blocks. The
unsafety is local: each intrinsic call wraps a `_mm256_loadu_si256`
on a slice we have already bounds-checked at the top of the
batch. The function's external API is safe.

**Why not SIMD `Monomial::divides`?** It's the next thing in line
(6.7 % of total) but lower priority for two reasons:
1. The sev-sweep work is the larger absolute cost.
2. SIMD-divides would need to handle two monomials' packed words
   simultaneously, which is more complex than the single-stream
   sev scan.

Listed as a possible follow-up; not adopted now.

### References

- `~/rustgb/src/bba.rs:220-286` (current `reduce_lobject` — the
  divisor search at lines 241-258 is the surface being changed)
- `~/rustgb/src/sbasis.rs:33-47` (the `SBasis` struct showing
  `sevs: Vec<u64>` is already a contiguous flat array, ready
  for SIMD without layout changes)
- `~/Singular-next-opt/kernel/GBEngine/kstd2.cc:74-121`
  (`kSevScanAVX2` — the model adopted here)
- `~/Singular-next-opt/kernel/GBEngine/kstd2.cc:127-` 
  (`kSevScanSSE4` — the SSE4.1 fallback we are deferring)
- `~/project/docs/profile-rustgb-v4-staging-5101449.md` (the
  v4 profile showing `reduce_lobject` at 30 % and identifying
  the sweep as the largest concentrated target)
- This conversation's `perf annotate` of `reduce_lobject` showing
  the per-instruction breakdown (the 18.70 % `jne` at offset
  0x2c193 was the smoking gun)

---

## ADR-008: Heap-based Monagan-Pearce reducer (supersedes ADR-002, ADR-003)

**Status:** Accepted. Implementation in progress (multi-phase plan
landing across several commits). Phase 1 (this commit) lands the
ADR + scaffold; ADR-002's geobucket reducer remains the active
runtime path until Phase 5 plumbs the heap reducer in behind a
feature flag, and is fully retired in Phase 7 if heap-based
validation succeeds.

**Date:** 2026-04-22

### Context

The v5 profile (`~/project/docs/profile-rustgb-v5-staging-5101449.md`)
showed the four largest concentrated functions to be:

| Function | v5 % of total |
|---|---|
| `bba::reduce_lobject` (self + inlined) | 23.5 |
| `poly::merge` | 20.1 |
| `KBucket::minus_m_mult_p` | 16.1 |
| `KBucket::leading` | 15.3 |
| libc allocator (combined) | 5.3 |
| `Monomial::cmp` (combined under merge + leading) | ~7 |

Of those, `poly::merge` (20.1 %), `KBucket::minus_m_mult_p`
(16.1 %), `KBucket::leading` (15.3 %), and a large share of
the allocator cost (~5 %) are **all costs imposed by the geobucket
reducer architecture chosen in ADR-002**. They exist because the
geobucket maintains the partial reduction as a materialised set of
polynomial slots that must be merged, scanned, and absorbed step
by step. About **57 % of v5's total cycles** live in functions
either eliminated or replaced by an alternative reducer
architecture.

The cost shape that's *not* a function of the geobucket — basis
sweep (~7 %), `Monomial::div` for multiplier construction (~7 %),
sugar bookkeeping, and the various cmp/divides primitives — is
unchanged regardless of which reducer architecture we pick.

ADR-003 listed the heap-based Monagan-Pearce reducer as a deferred
candidate; the v5 profile evidence has now made the case strong
enough to adopt it. This ADR promotes ADR-003 from "deferred" to
"accepted" and supersedes ADR-002.

### Singular's approach

Singular has **both** reducer architectures available. The default
bba path (`~/Singular-rustgb/kernel/GBEngine/kstd2.cc:bba()`)
uses the geobucket via `kBucket_pt`; a separate path
(`kStratHeap`-style strategies in `kstd1.cc`) uses a heap reducer
for situations where the geobucket's slot-management overhead is
known to lose. The choice was made decades ago when typical
workloads were larger than today's helium examples; for our scale
the geobucket's O(slot_count) leader scan cost amortises poorly
because the slot scan happens on *every* reducer step, not per
emitted term.

Singular's heap reducer machinery (when used) carries its own
`max_exp` cache per source poly and pushes new heap nodes as
divisors are added — structurally identical to the design adopted
here.

### FLINT's approach

FLINT uses **only** the heap-based Monagan-Pearce reducer for its
mpoly division (`~/flint/src/nmod_mpoly/divrem_monagan_pearce.c`,
726 lines, plus a similar `divides_monagan_pearce.c` for the
"is-it-divisible" specialisation). There is no geobucket reducer
in FLINT (the geobucket data structure exists but is used only as
a sum buffer with no `extract_leading` operation — see ADR-002).

FLINT's heap-node design is the closest reference. Each node
carries a chain of `(i, j)` indices: `i` identifies a source
polynomial (the input being divided, plus each pending divisor
times its multiplier), and `j` is the current term index in that
source. Pop the max, accumulate same-monomial chain entries, sum
coefficients, emit-or-cancel. When a quotient term is produced,
the corresponding divisor's tail terms get pushed onto the heap
(scaled by the quotient term).

The relevant detail FLINT documents but is easy to miss: each
source contributes **at most one heap node at a time**. When you
pop term j from source i, you push term j+1 from the same source,
keeping the heap size bounded by `1 + number_of_active_reducers`.
This is what keeps the heap log factor manageable.

### Mathicgb's approach

Mathicgb (the reference for ADR-001's polynomial layout) ships
**both** reducer architectures as runtime-selectable strategies
via the `Reducer::Type` enum (`~/mathicgb/src/mathicgb/Reducer.hpp`).
Implementations: `ReducerHeap.cpp`, `ReducerHashTable.cpp`,
`ReducerNoDedupHashTable.cpp`, `ReducerPackedDedupList.cpp`,
`ReducerHashPack.cpp`. The default for typical workloads is
`ReducerHeap`. The fact that mathicgb's authors made heap the
default after extensive benchmarking on Gröbner-basis-shaped
problems is a strong signal for our adoption.

Mathicgb's heap reducer also carries a per-reducer "monomial slab"
that pre-allocates the multiplier monomials in a side table
indexed by reducer-id. The heap nodes themselves stay small (~24
bytes: source poly pointer + index + reducer-id), which keeps
cache pressure low. We will mirror this pattern.

### Decision

Adopt a heap-based Monagan-Pearce reducer for `bba::reduce_lobject`,
matching the pattern documented above (FLINT's bookkeeping +
mathicgb's slab layout + lazy divisor addition driven by the
existing `find_divisor_idx`).

**Algorithmic shape** (the four mechanics that have to all line
up correctly):

1. **In-flight reducers**, stored in a per-LObject slab:
   ```text
   Reducer { poly: &Poly, multiplier: Monomial, coeff: Coeff, index: usize, sugar: u32 }
   ```
   `coeff` is pre-negated for cancellation (so summing the heap-top
   chain naturally produces the cancellation result). `index`
   tracks the next term in `poly` that hasn't yet been emitted into
   the heap.

2. **Heap nodes**, max-heap by degrevlex:
   ```text
   HeapNode { cmp_key: [u64; 4], reducer_idx: usize }
   ```
   `cmp_key` is the packed monomial `multiplier * poly.terms[index]`,
   pre-XORed against `ring.cmp_flip_mask` so plain lex compare on
   `[u64; 4]` is the correct max-heap ordering. Cached at push
   time, so the heap's internal compares need no `&Ring`.

3. **Pop-with-cancellation**: pop the max; while the next pop has
   the same `cmp_key`, accumulate. After draining the equal chain,
   if the summed coeff is zero, advance all contributing reducers'
   indices and recurse (or loop). If nonzero, that's the new
   leader / output term.

4. **Lazy divisor addition**: when a new leader emerges, run the
   existing `find_divisor_idx` against `s_basis`. If it returns
   `Some(idx)`, push a new `Reducer` with `index = 0` and the
   appropriate negated coefficient onto both the slab and the
   heap. The next pop-with-cancellation will (by construction)
   sum the old leader and the new reducer's first term to zero,
   driving us toward the next non-cancelled term.

5. **Survivor materialisation**: the LObject reduces to zero iff
   the heap becomes empty. Otherwise we have a survivor — drain
   the remaining heap entries via repeated pop-with-cancellation,
   pushing each emitted term into a fresh `Poly`. **This is the
   only place `Monomial::clone` happens** in the entire reduction
   chain (excluding the multiplier-monomial clones, one per added
   reducer, paid into the slab).

**Sugar tracking**: the LObject's sugar at any moment is `max(
initial_sugar, max over all in-flight reducers of (reducer.sugar))`.
Since reducers are only added (never removed mid-reduction), this
is just the running max of `(g_i.sugar + multiplier_deg)` over
adds, plus the initial. One `u32` slot on the LObject; updated on
each `push_reducer`.

**What this removes**:
- `KBucket` entirely (the geobucket struct, its `absorb`,
  `leading`, `extract_leading`, `minus_m_mult_p`, `is_zero`,
  `dirty` mask, `lm_cache`, the 32 NUM_SLOTS array and the
  `slot_for_len` length-bucketing).
- `poly::merge` (replaced by heap-pop emission); `Poly::add` and
  `Poly::sub` retain their public API but now go through a much
  simpler implementation that constructs from the heap output.
- `Poly::sub_mul_term` (subsumed by adding-as-reducer + heap
  iteration).
- The `LObject`'s embedded geobucket; replaced by a heap state
  (`Vec<Reducer>` slab + `BinaryHeap<HeapNode>` or similar).
- `kbucket.rs` as a module (retained in the repo through Phase 6
  for A/B comparison; deleted in Phase 7 if heap wins).

**What this preserves**:
- `find_divisor_idx` (unchanged — basis sweep still happens on
  each new leader).
- `Monomial` and its arithmetic (unchanged).
- `Poly` and its arithmetic outside the reducer (unchanged).
- `SBasis` and its sevs/redund parallel arrays (unchanged).
- `gm` pair criteria (unchanged).
- Sugar strategy (preserved with simpler bookkeeping).
- `bba::compute_gb` public API (unchanged).
- Output: bit-for-bit identical reduced GB on identical inputs
  (algorithmically Monagan-Pearce produces the same reduced
  Gröbner basis as geobucket-based bba).

### Consequences

**Performance prediction** (rough, conservative):

| Cost source | v5 share | Post-ADR-008 share |
|---|---|---|
| `poly::merge` | 20.1 % | ~3-5 % (only survivor materialisation) |
| `KBucket::minus_m_mult_p` | 16.1 % | gone |
| `KBucket::leading` | 15.3 % | gone (replaced by heap-pop, ~5-8 %) |
| Heap operations (new) | — | ~10-15 % (push, pop, log factor) |
| `Monomial::clone` per emitted term | ~3-5 % (inside merge's "loop body") | gone for non-survivors |
| libc allocator | 5.3 % | ~1-2 % (slab is amortised) |

Net wall prediction: **30-50 % wall reduction** on staging
workloads, putting staging-5101449 at roughly 60-80 s
(from v5's 115 s). This would close maybe a third of the
gap to the C++ next-opt target (~5 s on samsung).

**Measured (post-implementation, v6 profile, samsung, 2026-04-22):**

The actual win exceeded the prediction by 2×.

| Test | v5 (geobucket, AVX2) | v6 (heap, AVX2) | Wall reduction |
|---|---|---|---|
| staging-5101449 | 115 s | **38 s** | **−67 %** |
| staging-5104053 | 186 s | **52 s** | **−72 %** |
| staging-5106746 | 225 s | **58 s** | **−74 %** |

All three staging tests pass with exact fixture matches.

**Cumulative wall on staging-5101449 across the optimisation
series (un-profiled, AVX2 build, low contention):**

| Profile | Wall | Cumulative speedup |
|---|---|---|
| v1 (raw memmove) | ~870 s | 1.0× |
| v2 (ADR-001 head cursor) | ~225 s | 3.9× |
| v3 (ADR-005 direct exp) | ~204 s | 4.3× |
| v4 (ADR-006 FLINT merge) | ~140 s | 6.2× |
| v5 (ADR-007 SIMD sev) | 115 s | 7.6× |
| **v6 (ADR-008 heap reducer)** | **38 s** | **23×** |

vs C++ next-opt baseline (~5-7 s on samsung): rustgb went from
~17-25× slower (v5) to ~5-8× slower (v6). The heap reducer alone
closed roughly two-thirds of the remaining gap.

**v6 profile cost shape** (~/project/docs/profile-rustgb-v6-staging-5101449.md):

| Function | v5 % | v6 % | Notes |
|---|---|---|---|
| `bba::reduce_lobject` (geo path, self+inlined) | 23.5 | gone | replaced by heap path |
| `poly::merge` | 20.1 | **gone** | matches prediction |
| `KBucket::minus_m_mult_p` | 16.1 | **gone** | matches prediction |
| `KBucket::leading` | 15.3 | **gone** | matches prediction |
| `ReducerHeap::reduce_to_normal_form` | — | 19.8 | new; mostly find_divisor |
| `gm::chain_crit_normal` | 3.2 | **19.8** | **6× share growth** |
| `ReducerHeap::pop_with_cancellation` | — | 11.2 | new |
| `BinaryHeap::pop` | — | 10.5 | new (std-lib heap log factor) |
| Hashing (gm pair-criterion) | <1 | ~7 | new visibility |
| libc allocator (combined) | 5.3 | ~3 | matches prediction |

The huge surprise: `gm::chain_crit_normal` jumped from 3.2 % to
19.8 % share — the same Amdahl's law effect Singular saw when
they reduced the reducer cost. The pair criterion is now the
plurality-share concentrated function (tied with reducer's outer
loop). The natural next ADR target.

**Risks**:
- **Sugar regression**: the heap-based design's sugar bookkeeping
  is structurally simpler but easy to get wrong. Validation gate:
  the cargo test suite's reduction-result comparisons will catch
  algorithm-level bugs; staging-validation against fixtures will
  catch end-to-end correctness regressions.
- **Heap log factor on large bases**: for very long reductions
  with many active reducers, the O(log n) heap ops could
  cumulatively cost more than the geobucket's O(slot_count)
  leader scan. Singular's choice of geobucket-by-default was made
  for this case. For helium-staging-shaped workloads (moderate
  basis, short reductions), the trade-off goes the other way —
  but worth confirming with a v6 profile.
- **Cache locality of interleaved source poly access**: the heap
  pops dereference different source polys in interleaved order.
  Sequential within one source poly (cache-friendly) but
  interleaved across many sources during the pop sequence
  (cache-unfriendly). Possibly mitigated by L2's larger size
  relative to the working set (a few thousand polys × ~24 bytes
  per heap-active term = small).
- **Significant code-volume change** (~500-1000 lines new in
  `reducer.rs`, ~300 lines retired from `kbucket.rs`, ~50 lines
  changed in `bba.rs`, similar in `lobject.rs`). Phased landing
  with feature flag (Phase 5) lets the cargo test suite validate
  each phase independently.

**Migration plan** (the "implementation roadmap"; each phase is
a separate commit):

- **Phase 1** (this ADR's commit): scaffold `reducer.rs`, define
  `Reducer` and `HeapNode` types, register the module. No
  algorithm yet; cargo test still passes.
- **Phase 2**: heap data structure (push, pop_max, peek). Unit
  tests against a known-correct slow reference (sort all entries,
  pick max repeatedly).
- **Phase 3**: pop-with-cancellation (sum same-cmp_key chains,
  return non-zero or recurse). Tests on hand-crafted small heaps.
- **Phase 4**: lazy-add-divisor and survivor materialisation.
  Tests against tiny reductions (manually verified).
- **Phase 5**: plumb into `reduce_lobject` behind
  `cfg!(feature = "heap_reducer")`. Both reducers compile; cargo
  test runs both and asserts identical output on shared fixtures.
- **Phase 6**: full staging-validation under heap-reducer flag.
  Three-test fixture comparison is the correctness gate.
- **Phase 7**: profile v6 under heap. If wall improves
  meaningfully (≥ 15 % wall reduction), retire `kbucket.rs` (move
  to `examples/legacy/`) and make heap the default.

### References

- `~/rustgb/src/bba.rs:220-286` (current `reduce_lobject` —
  the integration target)
- `~/rustgb/src/kbucket.rs` (the geobucket being superseded;
  retired in Phase 7)
- `~/Singular-rustgb/kernel/GBEngine/kstd2.cc` (Singular's bba
  using geobucket — the historical default)
- `~/Singular-rustgb/kernel/GBEngine/kstd1.cc` (Singular's
  alternative `kStratHeap` strategies — heap-reducer evidence)
- `~/flint/src/nmod_mpoly/divrem_monagan_pearce.c` (FLINT's
  heap reducer; the closest reference implementation)
- `~/mathicgb/src/mathicgb/ReducerHeap.cpp` (mathicgb's heap
  reducer; the reference for the slab + small heap node pattern)
- `~/mathicgb/src/mathicgb/Reducer.hpp` (mathicgb's runtime
  reducer-selector enum)
- `~/project/docs/profile-rustgb-v5-staging-5101449.md` (the
  v5 profile evidence motivating this ADR)
- ADR-002 (geobucket reducer — superseded by this ADR)
- ADR-003 (heap reducer candidate — promoted to accepted by
  this ADR)

---

## How to add a new ADR

1. Pick the next number. Don't reuse retired numbers.
2. Fill in every section. If FLINT genuinely does not address the
   question (e.g. "how does `bba` handle Gebauer-Möller chain
   criterion?"), write **N/A — FLINT has no GB engine** in the FLINT
   section. Don't omit the section.
3. Cite source files with `path:line` ranges where possible.
4. If the decision changes later, add a new ADR rather than editing
   the old one. Mark the old one **Superseded by #N**.
5. Commit the ADR in the same change as the code that implements it,
   so `git blame` lines up.
