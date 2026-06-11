#!/usr/bin/env python3
"""trace-diff.py — align two bba operation traces and report divergences.

Task 392 (rustgb-step-trace). Inputs: two trace files emitted by the
rust (`RUSTGB_TRACE_FILE`) and Singular (`SING_TRACE_FILE`) engines in
the canonical, index-free event format:

    POP  <lcm-exps> <sugar>     pair selected from L
    RES  zero                   reduction zero-reduced
    RES  <lm-exps> <len>        nonzero reduction outcome (pre-redtail)
    INS  <lm-exps> <len>        element enters the basis (post-redtail)
    NEW  <lcm-exps> <sugar>     candidate pair created (phase B)
    KILL <lcm-exps> <tag>       pair eliminated (phase B)

Both engines render a monomial as its exponent vector, variable 0
first, dot-separated, no total-degree field — so events are content-
addressed and comparable across engines whose internal indices differ.

## Alignment rules

The trajectory skeleton is the POP/RES/INS stream. These are compared
**in strict stream order** — they encode the actual order the engine
selected and processed pairs and inserted basis elements, which is the
whole point of the "same ops same order" question.

NEW/KILL events (phase B) occur in *batches*: all NEW/KILL emitted
between two consecutive skeleton events (POP/RES/INS) form one
enterpairs batch. Within a batch the two engines legitimately differ
in intra-batch order (they enumerate candidate s-indices and run the
chain criterion in their own orders), so for phase-B comparison this
tool **canonicalizes (sorts) within each batch** before comparing.

## Outputs

(a) First divergence in the skeleton stream, with +/-10 events of
    context from both streams.
(b) Re-convergence analysis: the longest common subsequence (LCS)
    share of the two skeleton streams — does an early divergence wash
    out (streams re-converge) or compound?
(c) Full divergence inventory: per-event-type counts and direction
    (rust-only / singular-only), plus phase-B batch KILL-tag deltas.
(d) Summary stats: total events per type per engine (this re-measures
    the pair gap — POP counts — as a side effect).

Usage:
    trace-diff.py RUST.trace SING.trace [--phase-b] [--context N]
                  [--lcs-cap N]
"""

import argparse
import sys
from collections import Counter


SKELETON = ("POP", "RES", "INS")


def load(path):
    """Load a trace file into a list of (kind, key) tuples.

    `kind` is the event type (POP/RES/INS/NEW/KILL). `key` is the rest
    of the line — the canonical content payload (exps + sugar/len/tag).
    We keep the payload verbatim so two events compare equal iff their
    rendered text is identical.
    """
    events = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            sp = line.split(" ", 1)
            kind = sp[0]
            key = sp[1] if len(sp) > 1 else ""
            events.append((kind, key))
    return events


def skeleton(events):
    """The POP/RES/INS sub-stream, in order (phase-A trajectory)."""
    return [e for e in events if e[0] in SKELETON]


def batches(events):
    """Split into (skeleton_event_or_None, [phase-b events]) batches.

    A batch is the run of NEW/KILL events that follow a skeleton event
    (or precede the first skeleton event). Returns a list of
    (anchor, [phaseb...]) where anchor is the preceding skeleton event
    (or None for the leading batch).
    """
    out = []
    anchor = None
    pending = []
    for kind, key in events:
        if kind in SKELETON:
            out.append((anchor, pending))
            anchor = (kind, key)
            pending = []
        else:
            pending.append((kind, key))
    out.append((anchor, pending))
    return out


def first_divergence(a, b):
    """Index of the first position where skeleton streams a, b differ.
    Returns len(min) if one is a prefix of the other; None if identical
    and same length."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None


def lcs_length(a, b, cap=None):
    """Length of the longest common subsequence of a and b.

    O(n*m) DP, rolling row. For ~30K-event skeletons this is ~10^9
    cell-ops worst case — too slow in pure Python. We Hunt–Szymanski
    via difflib's matcher which is fast on highly-similar sequences and
    near the realistic case here; if streams are nearly disjoint it can
    be slow, so `cap` truncates both to the first `cap` events.
    """
    import difflib

    if cap is not None:
        a = a[:cap]
        b = b[:cap]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    return sum(blk.size for blk in sm.get_matching_blocks())


def context_block(stream, idx, ctx, label):
    lo = max(0, idx - ctx)
    hi = min(len(stream), idx + ctx + 1)
    lines = []
    for i in range(lo, hi):
        marker = ">>" if i == idx else "  "
        kind, key = stream[i]
        lines.append(f"  {marker} [{i:6d}] {kind} {key}")
    return f"{label} (events {lo}..{hi-1}):\n" + "\n".join(lines)


def type_counts(events):
    c = Counter()
    for kind, key in events:
        if kind == "RES":
            c["RES zero" if key == "zero" else "RES nonzero"] += 1
        elif kind == "KILL":
            tag = key.rsplit(" ", 1)[-1] if key else "?"
            c[f"KILL {tag}"] += 1
        else:
            c[kind] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rust", help="rust trace file")
    ap.add_argument("sing", help="singular trace file")
    ap.add_argument("--phase-b", action="store_true",
                    help="also compare phase-B (NEW/KILL) batches")
    ap.add_argument("--context", type=int, default=10,
                    help="events of context around first divergence")
    ap.add_argument("--lcs-cap", type=int, default=None,
                    help="truncate skeletons to N events before LCS "
                         "(LCS is O(n*m); cap keeps it tractable)")
    args = ap.parse_args()

    rust = load(args.rust)
    sing = load(args.sing)

    print("=" * 70)
    print("SUMMARY — total events per type per engine")
    print("=" * 70)
    rc, sc = type_counts(rust), type_counts(sing)
    keys = sorted(set(rc) | set(sc))
    print(f"{'event':<18} {'rust':>10} {'singular':>10} {'delta':>10}")
    for k in keys:
        r, s = rc.get(k, 0), sc.get(k, 0)
        print(f"{k:<18} {r:>10} {s:>10} {r - s:>+10}")
    # The pair-gap headline:
    rp, sp = rc.get("POP", 0), sc.get("POP", 0)
    if sp:
        print(f"\nPOP (pair) gap: rust/singular = {rp/sp:.4f}  "
              f"({rp - sp:+d} pairs, {100*(rp-sp)/sp:+.1f}%)")

    rsk, ssk = skeleton(rust), skeleton(sing)
    print(f"\nskeleton lengths: rust={len(rsk)}  singular={len(ssk)}")

    print("\n" + "=" * 70)
    print("FIRST DIVERGENCE (skeleton stream, strict order)")
    print("=" * 70)
    fd = first_divergence(rsk, ssk)
    if fd is None:
        print("IDENTICAL — the two skeleton streams match exactly.")
    else:
        print(f"first mismatch at skeleton index {fd}\n")
        print(context_block(rsk, fd, args.context, "RUST"))
        print()
        print(context_block(ssk, fd, args.context, "SINGULAR"))

    print("\n" + "=" * 70)
    print("RE-CONVERGENCE (longest common subsequence share)")
    print("=" * 70)
    cap = args.lcs_cap
    if cap is None and max(len(rsk), len(ssk)) > 6000:
        cap = 6000
        print(f"(skeletons large; capping LCS to first {cap} events — "
              f"pass --lcs-cap to override)")
    lcs = lcs_length(rsk, ssk, cap=cap)
    base = min(len(rsk), len(ssk)) if cap is None else min(cap, len(rsk), len(ssk))
    print(f"LCS length = {lcs}  ({100*lcs/base:.1f}% of the shorter "
          f"stream{' (capped)' if cap else ''})")
    if fd is not None and lcs > 0.5 * base:
        print("→ streams largely re-converge after the first divergence "
              "(LCS > 50%): the early divergence does NOT fully compound.")
    elif fd is not None:
        print("→ streams do NOT strongly re-converge (LCS <= 50%): the "
              "divergence compounds.")

    if args.phase_b:
        print("\n" + "=" * 70)
        print("PHASE-B batch comparison (NEW/KILL, intra-batch sorted)")
        print("=" * 70)
        rb, sb = batches(rust), batches(sing)
        print(f"rust batches={len(rb)}  singular batches={len(sb)}")
        # Aggregate phase-B by tag rather than per-batch alignment (the
        # skeleton already diverges, so batch i on each side is not the
        # same enterpairs call). Report the tag totals as the inventory.
        rkill = Counter()
        skill = Counter()
        rnew = snew = 0
        for kind, key in rust:
            if kind == "NEW":
                rnew += 1
            elif kind == "KILL":
                rkill[key.rsplit(" ", 1)[-1]] += 1
        for kind, key in sing:
            if kind == "NEW":
                snew += 1
            elif kind == "KILL":
                skill[key.rsplit(" ", 1)[-1]] += 1
        print(f"\nNEW: rust={rnew}  singular={snew}  delta={rnew-snew:+d}")
        print("\nKILL by tag:")
        tags = sorted(set(rkill) | set(skill))
        print(f"  {'tag':<14} {'rust':>10} {'singular':>10}")
        for t in tags:
            print(f"  {t:<14} {rkill.get(t,0):>10} {skill.get(t,0):>10}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
