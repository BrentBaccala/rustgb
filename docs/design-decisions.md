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

**Status:** Accepted (proposed; supersedes the original complemented-storage
representation that the initial commit shipped with).
**Date:** 2026-04-21

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

**Performance:** Profile v2 hotspot reshuffle prediction:

| Function | v2 (current) | After ADR-005 | Notes |
|---|---|---|---|
| `Monomial::mul` | 30.0 % | ~3-5 % | word-add + cheap check |
| `Monomial::cmp` (under merge) | ~6 % | ~9 % | XOR per word added |
| Net effect on wall | — | **~ −22 %** | residue stays the same shape |

Expected wall on staging-5101449: ~3:00 (was 4:02). After this
fix the new top function will be `poly::merge` (~13.7 % today,
likely the new plurality-share hotspot).

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
