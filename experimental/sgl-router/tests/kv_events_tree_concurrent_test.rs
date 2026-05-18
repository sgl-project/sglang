// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Concurrent-mutation stress test for `HashTree`.
//!
//! The 19 inline tests in `policies::kv_events::tree` are all
//! single-threaded.  Under production load, multiple worker subscribers
//! drive `insert` / `remove` / `clear_worker` against the same tree from
//! tokio worker threads while the chat handler simultaneously calls
//! `match_prefix` from many concurrent requests.
//!
//! The tree is documented as taking a write-lock for mutations and a
//! read-lock for `match_prefix`; this test exercises that contract under
//! heavy contention to catch:
//!
//!   * Deadlocks between the reverse index and the arena's RwLock.
//!   * Logical races where a removed worker still appears in the reverse
//!     index (or vice versa).
//!   * Panics from a node arena being mutated mid-read.
//!
//! After the storm settles, the tree must be self-consistent: every
//! worker that was fully cleared must be absent from every node's worker
//! set, and `node_count()` must converge to zero.

use std::sync::Arc;
use std::thread;

use sgl_router::policies::kv_events::{HashTree, KvWorkerId};

fn worker(i: usize) -> KvWorkerId {
    KvWorkerId {
        url: format!("http://w{i}:30000"),
        dp_rank: 0,
    }
}

/// 8 mutator threads × 200 ops + 4 reader threads × 500 match queries.
/// Each mutator inserts a chain, queries it, then clears the worker; the
/// invariant is that after every thread joins, the tree is empty (every
/// worker was cleared) and no thread panicked.
#[test]
fn tree_survives_concurrent_inserts_removes_and_matches() {
    let tree = Arc::new(HashTree::new());

    let mut handles = Vec::new();

    for tid in 0..8 {
        let tree = tree.clone();
        handles.push(thread::spawn(move || {
            let w = worker(tid);
            for round in 0..200_u64 {
                // Each round uses a fresh chain so different mutators
                // don't trample each other's nodes — we want contention
                // on the lock, not contention on the keys (those are
                // covered by the single-threaded reinsert/remove tests).
                let chain: Vec<i64> = (0..4)
                    .map(|i| ((tid as i64) << 32) | ((round as i64) << 8) | i as i64)
                    .collect();
                tree.insert(&w, None, &chain);

                let m = tree.match_prefix(None, &chain);
                assert!(
                    m.matched_blocks <= chain.len(),
                    "match must never exceed query length",
                );

                // Half the rounds use remove(&chain); the rest use
                // clear_worker — both must leave a consistent tree.
                if round % 2 == 0 {
                    tree.remove(&w, &chain);
                } else {
                    tree.clear_worker(&w);
                }
            }
            // Final blanket clear in case the last iteration used `remove`
            // on only part of the chain.
            tree.clear_worker(&w);
        }));
    }

    for tid in 0..4 {
        let tree = tree.clone();
        handles.push(thread::spawn(move || {
            for round in 0..500_u64 {
                let probe: Vec<i64> = (0..3)
                    .map(|i| ((tid as i64) << 40) | ((round as i64) << 8) | i as i64)
                    .collect();
                // Readers must never block-walk and must never panic.
                let _ = tree.match_prefix(None, &probe);
            }
        }));
    }

    for h in handles {
        h.join()
            .expect("worker thread panicked under concurrent load");
    }

    assert_eq!(
        tree.node_count(),
        0,
        "tree must be empty after every worker was cleared; \
         residual nodes indicate a missed clear_worker path",
    );

    // The arena and the reverse index must agree: zero non-root nodes
    // means zero `by_hash` entries. A bug that prunes the arena but not
    // the reverse index would leak memory and corrupt future inserts;
    // this assertion turns that into an immediate test failure.
    assert_eq!(
        tree.reverse_index_size(),
        0,
        "by_hash reverse index must be empty when no non-root nodes remain",
    );
}

/// A mutator races `clear_worker` against a reader that is mid-`match_prefix`
/// on a deep chain.  The reader must never see a partially-mutated tree
/// (no panic, no double-counted workers in the result set).
#[test]
fn match_prefix_is_consistent_with_concurrent_clear() {
    let tree = Arc::new(HashTree::new());
    let w = worker(0);
    let chain: Vec<i64> = (0..32).map(|i| 1_000 + i).collect();

    // Pre-populate so the reader has something to walk.
    tree.insert(&w, None, &chain);

    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));

    let mutator = {
        let tree = tree.clone();
        let stop = stop.clone();
        let w = w.clone();
        let chain = chain.clone();
        thread::spawn(move || {
            let mut round = 0u64;
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                if round.is_multiple_of(2) {
                    tree.clear_worker(&w);
                } else {
                    tree.insert(&w, None, &chain);
                }
                round += 1;
            }
        })
    };

    for _ in 0..2_000 {
        let m = tree.match_prefix(None, &chain);
        // Either the worker was present (matched_blocks == chain.len(),
        // workers set contains w) or it was cleared mid-walk (matched_blocks
        // == 0 OR matched_blocks > 0 with empty workers if the chain is
        // partially present).  Whichever — the result must be internally
        // consistent.
        if m.matched_blocks == chain.len() {
            assert!(
                m.workers.contains(&w),
                "full match must include worker; got {:?}",
                m.workers,
            );
        }
    }

    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    mutator.join().unwrap();
}
