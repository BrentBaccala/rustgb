//! step_trace — canonical per-event operation trace of a bba run
//! (task 392). Diff against an equivalently-instrumented Singular run
//! to localize where the two engines' operation streams diverge.
//!
//! Compile-gated behind the default-OFF `step_trace` cargo feature,
//! same zero-cost-when-off discipline as `scan_stats`: every call site
//! is `#[cfg(feature = "step_trace")]`, so the production build has no
//! trace code at all.
//!
//! ## Event format (engine-independent, index-free)
//!
//! The two engines' internal element indices never match, so events
//! identify elements by **content** — the leading/LCM exponent vector
//! rendered as compact dotted text, one field per ring variable in
//! ring-variable order (variable 0 first). E.g. a monomial with
//! exponents `[2,0,0,1,3,…]` renders as `2.0.0.1.3.…`. Both engines
//! must render identically; see [`render_exps`].
//!
//! Phase A (the trajectory skeleton — small, ~30 K events):
//!   * `POP <lcm-exps> <sugar>`     — pair selected from L for reduction.
//!   * `RES zero` / `RES <lm-exps> <len>` — reduction outcome.
//!   * `INS <lm-exps> <len>`        — element enters the basis
//!                                    (post-redtail shape).
//!
//! Phase B (drill-down — gated behind verbosity level 2, ~100s of K):
//!   * `NEW <lcm-exps> <sugar>`     — candidate pair created (survived
//!                                    the product criterion).
//!   * `KILL <lcm-exps> <tag>`      — pair eliminated; tag names the
//!                                    criterion site.
//!
//! ## Output
//!
//! `RUSTGB_TRACE_FILE` names the output path (one file per run). The
//! Singular dispatch swallows a loaded dylib's stderr (probe-report
//! gotcha), so the trace MUST go to a file, never stderr. The
//! verbosity level is `RUSTGB_TRACE_LEVEL` (default 1 = phase A only;
//! 2 = phase A + B).
//!
//! ## Alignment note for the diff tool
//!
//! POP/RES/INS are emitted in strict stream order and compared
//! strictly. NEW/KILL events between two POPs form one *enterpairs
//! batch*; their intra-batch order is an implementation detail (the
//! two engines enumerate candidate s-indices and run the chain
//! criterion in their own orders), so the diff tool canonicalizes
//! (sorts) within a batch before comparing. This module emits NEW/KILL
//! in rust's natural order; the batch boundary is implicit (the next
//! POP/INS).

use std::cell::RefCell;
use std::fs::File;
use std::io::{BufWriter, Write};

use crate::monomial::Monomial;
use crate::ring::Ring;

thread_local! {
    /// The trace sink, opened lazily on first event if `RUSTGB_TRACE_FILE`
    /// is set. `None` once we've decided tracing is off (no env var or
    /// the file could not be opened).
    static SINK: RefCell<Option<BufWriter<File>>> = const { RefCell::new(None) };
    /// Whether we've already attempted to open the sink this run.
    static OPENED: RefCell<bool> = const { RefCell::new(false) };
    /// Cached verbosity level (1 = phase A, 2 = +phase B).
    static LEVEL: RefCell<u8> = const { RefCell::new(0) };
}

/// Render an exponent vector as compact dotted text, one field per
/// ring variable in ring-variable order (variable 0 first). This is
/// the canonical, index-free element identity shared with the Singular
/// side — both engines MUST produce byte-identical output for the same
/// monomial. The trailing capped-degree byte is NOT included (it is a
/// derived field, not part of the exponent vector).
pub fn render_exps(m: &Monomial, ring: &Ring) -> String {
    let n = ring.nvars();
    let mut s = String::with_capacity(n as usize * 2);
    for i in 0..n {
        if i > 0 {
            s.push('.');
        }
        let e = m.exponent(ring, i).expect("i < nvars");
        s.push_str(&e.to_string());
    }
    s
}

/// Open the sink on first use. Returns the active verbosity level
/// (0 means tracing is off for this run).
fn ensure_open() -> u8 {
    let already = OPENED.with(|o| *o.borrow());
    if !already {
        OPENED.with(|o| *o.borrow_mut() = true);
        let path = std::env::var("RUSTGB_TRACE_FILE").ok();
        if let Some(p) = path {
            if let Ok(f) = File::create(&p) {
                SINK.with(|s| *s.borrow_mut() = Some(BufWriter::new(f)));
                let lvl = std::env::var("RUSTGB_TRACE_LEVEL")
                    .ok()
                    .and_then(|v| v.parse::<u8>().ok())
                    .unwrap_or(1)
                    .max(1);
                LEVEL.with(|l| *l.borrow_mut() = lvl);
            }
        }
    }
    LEVEL.with(|l| *l.borrow())
}

/// Reset the trace state at the start of a counted computation, so a
/// process running multiple `std()` calls writes a fresh file per call.
/// (We truncate via `File::create` on next open.)
pub fn reset() {
    SINK.with(|s| {
        if let Some(w) = s.borrow_mut().as_mut() {
            let _ = w.flush();
        }
        *s.borrow_mut() = None;
    });
    OPENED.with(|o| *o.borrow_mut() = false);
    LEVEL.with(|l| *l.borrow_mut() = 0);
}

#[inline]
fn emit(line: &str) {
    SINK.with(|s| {
        if let Some(w) = s.borrow_mut().as_mut() {
            let _ = writeln!(w, "{line}");
        }
    });
}

/// `POP <lcm-exps> <sugar>` — a pair is selected from L for reduction.
/// `lcm` is the pair's LCM monomial; `sugar` its sugar.
pub fn pop(lcm: &Monomial, sugar: u32, ring: &Ring) {
    if ensure_open() == 0 {
        return;
    }
    emit(&format!("POP {} {}", render_exps(lcm, ring), sugar));
}

/// `RES zero` — the reduction zero-reduced.
pub fn res_zero() {
    if ensure_open() == 0 {
        return;
    }
    emit("RES zero");
}

/// `RES <lm-exps> <len>` — the reduction produced a nonzero survivor
/// with leading monomial `lm` and `len` terms (pre-redtail shape, i.e.
/// the reducer output before the per-step redTail tail reduction).
pub fn res_nonzero(lm: &Monomial, len: usize, ring: &Ring) {
    if ensure_open() == 0 {
        return;
    }
    emit(&format!("RES {} {}", render_exps(lm, ring), len));
}

/// `INS <lm-exps> <len>` — an element enters the basis (post-redtail
/// shape, i.e. what becomes a reducer). `lm` is its leading monomial,
/// `len` its term count.
pub fn ins(lm: &Monomial, len: usize, ring: &Ring) {
    if ensure_open() == 0 {
        return;
    }
    emit(&format!("INS {} {}", render_exps(lm, ring), len));
}

/// `NEW <lcm-exps> <sugar>` — a candidate pair was created (survived
/// the product criterion, entering B). Phase B only (level >= 2).
pub fn new_pair(lcm: &Monomial, sugar: u32, ring: &Ring) {
    if ensure_open() < 2 {
        return;
    }
    emit(&format!("NEW {} {}", render_exps(lcm, ring), sugar));
}

/// `KILL <lcm-exps> <tag>` — a pair was eliminated. `tag` names the
/// criterion site (e.g. `prodcrit`, `chainB-div`, `chainB-eq`,
/// `chainL`). Phase B only (level >= 2).
pub fn kill_pair(lcm: &Monomial, tag: &str, ring: &Ring) {
    if ensure_open() < 2 {
        return;
    }
    emit(&format!("KILL {} {}", render_exps(lcm, ring), tag));
}

/// Flush the sink at the end of the computation.
pub fn flush() {
    SINK.with(|s| {
        if let Some(w) = s.borrow_mut().as_mut() {
            let _ = w.flush();
        }
    });
}
