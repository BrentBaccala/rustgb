//! scan_stats — divisor-scan volume counters (task 389, finding C).
//!
//! Compile-gated behind the default-OFF `scan_stats` cargo feature.
//! When the feature is off, every symbol here is absent and the call
//! sites compile to nothing (`#[cfg(feature = "scan_stats")]` on the
//! instrumentation lines, not runtime branches), so there is zero
//! cost on the production build.
//!
//! Purpose: attribute the reduction divisor-scan excess (rust ~1.04 s
//! vs Singular next-opt ~0.65 s on staging-5101449, "finding C" of the
//! 10 Jun 2026 comparative profile) to scan *volume*, split per call
//! site of `bba::find_divisor_idx`:
//!
//!   1. `Head`     — head reduction (`reduce_lobject_geobucket`)
//!   2. `Redtail`  — per-step redtail (`reduce_tail` via `reduce_h_tail`)
//!   3. `Tailall`  — final `tail_reduce_all` pass (`reduce_tail`)
//!
//! Per site we count seven quantities (see `SiteCounters`), plus a
//! separate count of `ring.divmask_of` calls in `reduce_tail`'s
//! per-parked-leader path (finding-A-shaped sub-hypothesis).
//!
//! Dump: at the end of `compute_gb_serial`, when `RUSTGB_SCAN_STATS=1`,
//! one machine-greppable line per site to stderr.

use std::cell::RefCell;

/// Which call site of `find_divisor_idx` issued a scan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanSite {
    /// Head reduction — `reduce_lobject_geobucket` / `reduce_lobject_heap`.
    Head,
    /// Per-step redtail (ADR-024) — `reduce_tail` reached via `reduce_h_tail`.
    Redtail,
    /// Final post-hoc pass — `reduce_tail` reached via `tail_reduce_all`.
    Tailall,
}

/// The seven per-site scan counters plus the redtail divmask probe.
#[derive(Clone, Copy, Debug, Default)]
pub struct SiteCounters {
    /// Calls to `find_divisor_idx` from this site.
    pub scans: u64,
    /// Basis size (`divmasks.len()`) at each scan — elements *eligible*.
    pub sum_len: u64,
    /// Elements the sweep actually advanced past before returning. With
    /// the `li <= 2` early-out (and first-match when shortest_reducer is
    /// off) the sweep can stop early; this is what was actually covered
    /// (the final `idx` reached, capped at `len`).
    pub sum_swept: u64,
    /// Candidates that passed the divmask SIMD pre-filter (each distinct
    /// divmask-match the driver examined).
    pub divmask_hits: u64,
    /// Confirmed divisors (`Monomial::divides` returned true).
    pub divides_hits: u64,
    /// `li <= 2` early-out returns.
    pub earlyouts: u64,
    /// Sum of the sweep position (idx) at which an early-out fired.
    pub sum_earlyout_pos: u64,
}

/// All per-site counters plus the redtail-path divmask-compute count.
#[derive(Clone, Copy, Debug, Default)]
pub struct ScanStats {
    /// Head-reduction site counters.
    pub head: SiteCounters,
    /// Per-step redtail (ADR-024) site counters.
    pub redtail: SiteCounters,
    /// Final `tail_reduce_all` pass site counters.
    pub tailall: SiteCounters,
    /// Separate (finding-A-shaped) count: `ring.divmask_of` calls in
    /// `reduce_tail`'s per-parked-leader path (`bba.rs`, once per
    /// distinct bucket leader of a tail reduction).
    pub redtail_divmask_of: u64,
}

impl ScanStats {
    fn site_mut(&mut self, site: ScanSite) -> &mut SiteCounters {
        match site {
            ScanSite::Head => &mut self.head,
            ScanSite::Redtail => &mut self.redtail,
            ScanSite::Tailall => &mut self.tailall,
        }
    }
}

thread_local! {
    static STATS: RefCell<ScanStats> = RefCell::new(ScanStats::default());
}

/// Reset all counters (call at the start of a counted computation so a
/// single process running multiple `std()` calls reports per-call).
pub fn reset() {
    STATS.with(|s| *s.borrow_mut() = ScanStats::default());
}

/// Record one completed scan from `site`.
///
/// * `len`          — basis size (`divmasks.len()`).
/// * `swept`        — final `idx` reached (elements covered), ≤ `len`.
/// * `divmask_hits` — candidates that passed the divmask pre-filter.
/// * `divides_hits` — candidates confirmed by `Monomial::divides`.
/// * `earlyout_pos` — `Some(idx)` if a `li<=2` early-out fired at `idx`,
///                     else `None`.
#[allow(clippy::too_many_arguments)]
pub fn record_scan(
    site: ScanSite,
    len: usize,
    swept: usize,
    divmask_hits: u64,
    divides_hits: u64,
    earlyout_pos: Option<usize>,
) {
    STATS.with(|s| {
        let mut g = s.borrow_mut();
        let c = g.site_mut(site);
        c.scans += 1;
        c.sum_len += len as u64;
        c.sum_swept += swept as u64;
        c.divmask_hits += divmask_hits;
        c.divides_hits += divides_hits;
        if let Some(pos) = earlyout_pos {
            c.earlyouts += 1;
            c.sum_earlyout_pos += pos as u64;
        }
    });
}

/// Record one `ring.divmask_of` call in the redtail per-leader path.
pub fn record_redtail_divmask_of() {
    STATS.with(|s| s.borrow_mut().redtail_divmask_of += 1);
}

/// If `RUSTGB_SCAN_STATS=1`, dump the counters to stderr, one labelled
/// line per site (machine-greppable), then a redtail-divmask line.
pub fn dump_if_enabled() {
    if std::env::var("RUSTGB_SCAN_STATS").as_deref() != Ok("1") {
        return;
    }
    STATS.with(|s| {
        let g = s.borrow();
        for (label, c) in [
            ("head", &g.head),
            ("redtail", &g.redtail),
            ("tailall", &g.tailall),
        ] {
            eprintln!(
                "SCANSTAT site={label} scans={} sum_len={} sum_swept={} \
                 divmask_hits={} divides_hits={} earlyouts={} sum_earlyout_pos={}",
                c.scans,
                c.sum_len,
                c.sum_swept,
                c.divmask_hits,
                c.divides_hits,
                c.earlyouts,
                c.sum_earlyout_pos,
            );
        }
        eprintln!(
            "SCANSTAT redtail_divmask_of={}",
            g.redtail_divmask_of
        );
    });
}
