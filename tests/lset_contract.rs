//! Cross-backend contract tests for [`LSet`].
//!
//! These tests exercise the public alias `rustgb::LSet`, which
//! resolves to the heap backend (`src/lset.rs`) by default and to
//! the flat-Vec backend (`src/lset_flat.rs`) when the crate is
//! built with `--features flat_lset`. The same test bodies must
//! pass on either backend — that's the contract.
//!
//! ADR-026 is the rationale for the dual-backend setup.
//!
//! Test inventory:
//!
//! * `insert_pop_orders_by_sugar` — basic priority-queue ordering.
//! * `delete_by_indices_tombstones` — deletion is honoured by `pop`
//!   and by `contains`.
//! * `reinsert_same_indices_replaces` — the most-recent insert for
//!   `(i, j)` shadows older live pairs on those indices.
//! * `contains_agrees_with_delete` — `contains` follows `delete`
//!   and tolerates swapped `(i, j)`.
//! * `iter_filtered_subset_matches_brute_force` — the new
//!   `iter_filtered_subset` API yields exactly the live pairs whose
//!   `lcm_divmask` is a divmask-superset of the query mask. Checked
//!   against a brute-force `iter_live().filter(...)` reference for
//!   a randomly-generated pair set across multiple query masks.

use rustgb::{Field, LSet, MonoOrder, Monomial, Pair, Ring};

fn mk_ring(nvars: u32) -> Ring {
    Ring::new(nvars, MonoOrder::DegRevLex, Field::new(32003).unwrap()).unwrap()
}

fn mk_pair(r: &Ring, i: u32, j: u32, sugar: u32, arrival: u64) -> Pair {
    let lcm = Monomial::from_exponents(r, &vec![1u32; r.nvars() as usize]).unwrap();
    Pair::new(i, j, lcm, r, sugar, arrival)
}

#[test]
fn insert_pop_orders_by_sugar() {
    let r = mk_ring(3);
    let mut l = LSet::new();
    l.insert(mk_pair(&r, 0, 1, 7, 0));
    l.insert(mk_pair(&r, 0, 2, 3, 1));
    l.insert(mk_pair(&r, 1, 2, 5, 2));
    assert_eq!(l.len(), 3);
    l.assert_canonical(&r);
    assert_eq!(l.pop().unwrap().sugar, 3);
    assert_eq!(l.pop().unwrap().sugar, 5);
    assert_eq!(l.pop().unwrap().sugar, 7);
    assert!(l.pop().is_none());
    assert_eq!(l.len(), 0);
}

#[test]
fn delete_by_indices_tombstones() {
    let r = mk_ring(3);
    let mut l = LSet::new();
    l.insert(mk_pair(&r, 0, 1, 7, 0));
    l.insert(mk_pair(&r, 0, 2, 3, 1));
    l.insert(mk_pair(&r, 1, 2, 5, 2));
    assert!(l.delete(0, 2));
    assert!(!l.contains(0, 2));
    l.assert_canonical(&r);
    // Remaining pop order: 5 then 7.
    assert_eq!(l.pop().unwrap().sugar, 5);
    assert_eq!(l.pop().unwrap().sugar, 7);
    assert!(l.pop().is_none());
}

#[test]
fn reinsert_same_indices_replaces() {
    let r = mk_ring(3);
    let mut l = LSet::new();
    l.insert(mk_pair(&r, 0, 1, 7, 0));
    l.insert(mk_pair(&r, 0, 1, 3, 1)); // same indices, lower sugar
    assert_eq!(l.len(), 1);
    assert_eq!(l.pop().unwrap().sugar, 3);
    assert!(l.pop().is_none());
}

#[test]
fn contains_agrees_with_delete() {
    let r = mk_ring(3);
    let mut l = LSet::new();
    l.insert(mk_pair(&r, 2, 5, 4, 0));
    assert!(l.contains(2, 5));
    assert!(l.contains(5, 2)); // swap tolerated
    assert!(l.delete(5, 2));
    assert!(!l.contains(2, 5));
    assert!(!l.delete(2, 5));
}

/// `iter_filtered_subset(mask)` must yield exactly the live pairs
/// whose `lcm_divmask` is a divmask-superset of `mask` — i.e.
/// `(mask & !pair.lcm_divmask) == 0`. We build a small LSet with
/// varied LCMs (so the divmasks differ), then for a basket of
/// query masks compare the iterator's output to a brute-force
/// `iter_live().filter(...)` reference.
///
/// Includes tombstone churn (delete + re-insert) so the test also
/// catches the flat backend's "tombstoned slot must not match"
/// bookkeeping.
#[test]
fn iter_filtered_subset_matches_brute_force() {
    let r = mk_ring(5); // 5 vars → divmask layout has multiple windows
    let mut l = LSet::new();

    // 12 pairs with varied LCM exponent vectors.
    let exps: Vec<[u32; 5]> = vec![
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [2, 0, 0, 1, 0],
        [0, 2, 0, 0, 1],
        [3, 0, 0, 0, 0],
        [0, 0, 0, 3, 0],
        [1, 1, 1, 0, 0],
        [2, 2, 1, 0, 0],
        [4, 0, 4, 0, 0],
    ];
    let mut next_arrival: u64 = 0;
    for (k, e) in exps.iter().enumerate() {
        let lcm = Monomial::from_exponents(&r, e).unwrap();
        // Use distinct (i, j) per pair so no insert collides with
        // a previous live slot.
        let i = k as u32;
        let j = (k as u32) + 100;
        let pair = Pair::new(i, j, lcm, &r, k as u32 + 1, next_arrival);
        next_arrival += 1;
        l.insert(pair);
    }
    l.assert_canonical(&r);

    // Tombstone churn: pop the smallest, delete one in the middle,
    // re-insert one of the popped/deleted (i, j) with a fresh LCM.
    let _ = l.pop().unwrap();
    let _ = l.delete(5, 105);
    let lcm_extra = Monomial::from_exponents(&r, &[1, 1, 1, 1, 1]).unwrap();
    l.insert(Pair::new(5, 105, lcm_extra, &r, 99, next_arrival));
    // `next_arrival` is intentionally not bumped further — the test
    // body ends here, so a redundant `+= 1` would just be dead code.
    l.assert_canonical(&r);

    // For each query mask, compare iter_filtered_subset output
    // (collected as keys) against a brute-force scan.
    //
    // Pair::cmp ties on (sugar, arrival) so we collect by `key` to
    // get a canonical, order-insensitive identity. The two
    // iterators do NOT have to yield in the same order — only the
    // *set* of yielded pairs has to match.
    let collect_keys = |mask: u64| -> std::collections::BTreeSet<u64> {
        l.iter_filtered_subset(mask).map(|p| p.key.0).collect()
    };
    let brute_force_keys = |mask: u64| -> std::collections::BTreeSet<u64> {
        l.iter_live()
            .filter(|p| (mask & !p.lcm_divmask) == 0)
            .map(|p| p.key.0)
            .collect()
    };

    // Sample query masks: every live pair's own divmask (each must
    // match itself), plus a few adversarial values.
    let mut sample_masks: Vec<u64> = Vec::new();
    for p in l.iter_live() {
        sample_masks.push(p.lcm_divmask);
    }
    sample_masks.extend([
        0u64,
        !0u64,
        0x1u64,
        0x3u64,
        0xFFu64,
        0xFFFF_FFFFu64,
        0xAAAA_AAAA_AAAA_AAAAu64,
    ]);

    for mask in sample_masks {
        let got = collect_keys(mask);
        let want = brute_force_keys(mask);
        assert_eq!(
            got, want,
            "iter_filtered_subset disagreed with brute force for mask = {mask:#018x}"
        );
    }
}

/// `iter_filtered_subset(mask)` after a long mixed stream of
/// inserts, deletes, and pops still agrees with brute force. This
/// is a stronger version of the previous test: more mutations,
/// higher chance of catching a bookkeeping bug in the flat
/// backend's tombstone vector / sorted-pop heap interaction.
#[test]
fn iter_filtered_subset_after_mixed_stream() {
    let r = mk_ring(4);
    let mut l = LSet::new();

    // Stream of operations driven by a tiny LCG so the test is
    // deterministic.
    let mut state: u64 = 0xc0ff_ee00_0000_0001;
    let mut step = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state
    };

    for (arrival, round) in (0..40u64).enumerate() {
        // Insert one pair per round with a varied LCM.
        let s = step();
        let exps: [u32; 4] = [
            (s & 0x3) as u32,
            ((s >> 2) & 0x3) as u32,
            ((s >> 4) & 0x3) as u32,
            ((s >> 6) & 0x3) as u32,
        ];
        let lcm = Monomial::from_exponents(&r, &exps).unwrap();
        let i = (round % 5) as u32;
        let j = i + 50 + ((round / 5) as u32);
        let pair = Pair::new(i, j, lcm, &r, (s & 0xff) as u32, arrival as u64);
        l.insert(pair);

        // Every 4 rounds, pop one.
        if round % 4 == 3 {
            let _ = l.pop();
        }
        // Every 7 rounds, delete a (possibly nonexistent) (i, j).
        if round % 7 == 6 {
            let _ = l.delete(i, j);
        }
        l.assert_canonical(&r);
    }

    let collect_keys = |mask: u64| -> std::collections::BTreeSet<u64> {
        l.iter_filtered_subset(mask).map(|p| p.key.0).collect()
    };
    let brute_force_keys = |mask: u64| -> std::collections::BTreeSet<u64> {
        l.iter_live()
            .filter(|p| (mask & !p.lcm_divmask) == 0)
            .map(|p| p.key.0)
            .collect()
    };

    for &mask in &[
        0u64,
        !0u64,
        0x1u64,
        0xFFu64,
        0x10u64,
        0x101_0101u64,
        0x55_55_55_55u64,
    ] {
        let got = collect_keys(mask);
        let want = brute_force_keys(mask);
        assert_eq!(
            got, want,
            "mixed-stream iter_filtered_subset disagreed for mask = {mask:#018x}"
        );
    }
}
