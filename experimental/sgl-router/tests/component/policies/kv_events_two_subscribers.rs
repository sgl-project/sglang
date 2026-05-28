// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Two independent `KvEventIndex` instances subscribed to the same PUB
//! socket — the in-process surrogate for "two router replicas watching
//! the same SGLang worker's KV publisher."
//!
//! Why this matters: sgl-router v1 explicitly omits multi-replica state
//! sync (deferred to v2 in the slim-design spec). Independent ZMQ
//! subscription is the **only** mechanism by which two routers arrive at
//! a consistent cache-aware view today. If a future change accidentally
//! degraded that property — e.g. a worker that only allows one subscriber,
//! a switch from PUB/SUB to PUSH/PULL, or a teardown bug that drops
//! events to one of N subscribers — this test fails loudly.
//!
//! Property pinned: after publishing N `BlockStored` events, both trees
//! report the same `match_prefix(matched_blocks, workers)` for the
//! published key, and an unpublished key remains absent from both.

use std::sync::Arc;
use std::time::Duration;

use zeromq::SocketSend;

use sgl_router::policies::kv_events::discovery::EventConfig;
use sgl_router::policies::kv_events::{compute_block_hashes, KvEventIndex, KvWorkerId};

use super::zmq_helpers::{
    build_multipart, encode_block_stored_event, encode_event_batch, make_pub_bound,
};

#[tokio::test]
async fn two_independent_subscribers_converge_to_same_tree_state() {
    // 1. One PUB socket — the worker. Both router surrogates connect to it.
    let (mut publisher, port) = make_pub_bound().await;
    let worker_url = "http://127.0.0.1:30000";
    let block_size = 4u32;
    let cfg = EventConfig {
        host: "127.0.0.1".into(),
        port_base: port,
        topic: String::new(),
        block_size,
        dp_size: 1,
    };

    // 2. Two independent router-process surrogates, each with its own
    //    `KvEventIndex` (own tree, own subscriber, own pump task). Both
    //    call `add_worker` with the same preresolved `EventConfig` — the
    //    same shape production wires through `WorkerManager`.
    let router_a = KvEventIndex::new();
    let router_b = KvEventIndex::new();
    router_a.add_worker(worker_url, Some(cfg.clone())).await;
    router_b.add_worker(worker_url, Some(cfg.clone())).await;

    // SUB-side handshake settle. Publishing before the subscribers
    // finish their initial connect loses messages in PUB/SUB semantics;
    // the polling loop below would then never converge.
    tokio::time::sleep(Duration::from_millis(200)).await;

    // 3. Publish a deterministic, multi-block event chain.
    let tokens: Vec<u32> = (0..16).collect();
    let hashes = compute_block_hashes(&tokens, block_size as usize);
    assert!(
        hashes.len() >= 3,
        "test needs ≥3 blocks; got {}",
        hashes.len()
    );
    let event_bytes = encode_block_stored_event(&hashes, None, &tokens, block_size);
    let payload = encode_event_batch(0.0, vec![event_bytes], Some(0));
    publisher
        .send(build_multipart(1, payload))
        .await
        .expect("publish BlockStored");

    // 4. Poll both trees until both report the FULL chain matched. The
    //    SUB→mpsc→pump→tree pipeline is async; loopback delivery is
    //    reliable but not instantaneous.
    let target = hashes.len();
    let key = KvWorkerId {
        url: worker_url.into(),
        dp_rank: 0,
    };
    let start = std::time::Instant::now();
    loop {
        let ma = router_a.tree().match_prefix(None, &hashes);
        let mb = router_b.tree().match_prefix(None, &hashes);
        let converged = ma.matched_blocks == target
            && mb.matched_blocks == target
            && ma.workers.contains(&key)
            && mb.workers.contains(&key);
        if converged {
            // Both trees agree on count AND on the worker that holds the
            // prefix. This is what the cache-aware-zmq policy reads to
            // pick a worker; both routers picking the same key here
            // means they would route the same prompt to the same worker.
            assert_eq!(
                ma.matched_blocks, mb.matched_blocks,
                "subscribers disagreed on matched_blocks",
            );
            assert_eq!(
                ma.workers, mb.workers,
                "subscribers disagreed on worker set",
            );
            break;
        }
        if start.elapsed() > Duration::from_secs(3) {
            panic!(
                "subscribers did not converge within 3s: \
                 router_a={{matched={}, workers={:?}}}, \
                 router_b={{matched={}, workers={:?}}}, target={target}",
                ma.matched_blocks, ma.workers, mb.matched_blocks, mb.workers,
            );
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // 5. Negative leg: a key that was never published must not appear in
    //    either tree. Guards against a future bug where one subscriber
    //    accidentally inherits another's state (shared static, etc.).
    let unseen: Vec<i64> = vec![999_999_999_001, 999_999_999_002, 999_999_999_003];
    let na = router_a.tree().match_prefix(None, &unseen);
    let nb = router_b.tree().match_prefix(None, &unseen);
    assert_eq!(na.matched_blocks, 0, "router_a leaked unpublished key");
    assert_eq!(nb.matched_blocks, 0, "router_b leaked unpublished key");

    // 6. Both shutdowns must complete cleanly — no hang from the second
    //    subscriber holding a reference to a shared resource. The first
    //    drains under a generous ceiling (worker thread joins, mpsc
    //    receiver drop); the second has nothing left to wait on and
    //    must complete promptly. A slow second shutdown indicates the
    //    two subscribers were sharing a resource that serialized them.
    let r = tokio::time::timeout(Duration::from_secs(2), Arc::clone(&router_a).shutdown()).await;
    assert!(r.is_ok(), "router_a shutdown hung");

    let t = std::time::Instant::now();
    let r = tokio::time::timeout(Duration::from_secs(2), Arc::clone(&router_b).shutdown()).await;
    assert!(r.is_ok(), "router_b shutdown hung");
    let elapsed = t.elapsed();
    assert!(
        elapsed < Duration::from_millis(100),
        "router_b shutdown after router_a drained took {elapsed:?}; \
         expected <100ms (no shared-resource contention)",
    );
}

/// Two PUB sockets (two workers) + two `KvEventIndex` instances (two
/// routers), each subscribed to **both** publishers. This is the real
/// v1 HA shape: each router replica fans out subscriptions across the
/// worker pool and merges every publisher's `BlockStored` stream into
/// its own tree. The companion 1-PUB test above only verifies broadcast
/// fan-out; this test verifies the per-worker attribution stays correct
/// when events arrive from multiple sources concurrently.
///
/// Property pinned: after publishing prefix `X` on `pub_x` and prefix
/// `Y` on `pub_y`, both trees report
///   * `match_prefix(X) = {full, workers={worker_x}}`
///   * `match_prefix(Y) = {full, workers={worker_y}}`
/// with no cross-attribution (worker_x must NOT appear in match(Y)).
/// A regression that wires both subscribers to the same internal
/// channel — or that mis-keys events by their arrival socket rather
/// than their announced worker URL — would surface here as cross-
/// contamination of the worker sets.
#[tokio::test]
async fn two_subscribers_merge_events_from_two_publishers() {
    let (mut pub_x, port_x) = make_pub_bound().await;
    let (mut pub_y, port_y) = make_pub_bound().await;
    let worker_x = "http://127.0.0.1:30001";
    let worker_y = "http://127.0.0.1:30002";
    let block_size = 4u32;
    let cfg_x = EventConfig {
        host: "127.0.0.1".into(),
        port_base: port_x,
        topic: String::new(),
        block_size,
        dp_size: 1,
    };
    let cfg_y = EventConfig {
        host: "127.0.0.1".into(),
        port_base: port_y,
        topic: String::new(),
        block_size,
        dp_size: 1,
    };

    // Both routers subscribe to BOTH workers — the production fan-out.
    let router_a = KvEventIndex::new();
    let router_b = KvEventIndex::new();
    router_a.add_worker(worker_x, Some(cfg_x.clone())).await;
    router_a.add_worker(worker_y, Some(cfg_y.clone())).await;
    router_b.add_worker(worker_x, Some(cfg_x.clone())).await;
    router_b.add_worker(worker_y, Some(cfg_y.clone())).await;

    // Four SUB→PUB handshakes need to settle before publishing; missed
    // SUBSCRIBE frames lose messages forever in PUB/SUB semantics.
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Two non-overlapping token streams → two distinct hash chains. The
    // gap between them (0..16 vs 1000..1016) keeps `compute_block_hashes`
    // outputs disjoint so a cross-attribution bug can't be masked by
    // hash collision.
    let tokens_x: Vec<u32> = (0..16).collect();
    let tokens_y: Vec<u32> = (1000..1016).collect();
    let hashes_x = compute_block_hashes(&tokens_x, block_size as usize);
    let hashes_y = compute_block_hashes(&tokens_y, block_size as usize);
    assert!(hashes_x.len() >= 3 && hashes_y.len() >= 3);

    let payload_x = encode_event_batch(
        0.0,
        vec![encode_block_stored_event(
            &hashes_x, None, &tokens_x, block_size,
        )],
        Some(0),
    );
    let payload_y = encode_event_batch(
        0.0,
        vec![encode_block_stored_event(
            &hashes_y, None, &tokens_y, block_size,
        )],
        Some(0),
    );
    pub_x
        .send(build_multipart(1, payload_x))
        .await
        .expect("publish on pub_x");
    pub_y
        .send(build_multipart(1, payload_y))
        .await
        .expect("publish on pub_y");

    let key_x = KvWorkerId {
        url: worker_x.into(),
        dp_rank: 0,
    };
    let key_y = KvWorkerId {
        url: worker_y.into(),
        dp_rank: 0,
    };
    let target_x = hashes_x.len();
    let target_y = hashes_y.len();

    let start = std::time::Instant::now();
    loop {
        let ax = router_a.tree().match_prefix(None, &hashes_x);
        let ay = router_a.tree().match_prefix(None, &hashes_y);
        let bx = router_b.tree().match_prefix(None, &hashes_x);
        let by = router_b.tree().match_prefix(None, &hashes_y);
        let converged = ax.matched_blocks == target_x
            && ay.matched_blocks == target_y
            && bx.matched_blocks == target_x
            && by.matched_blocks == target_y
            && ax.workers.contains(&key_x)
            && ay.workers.contains(&key_y)
            && bx.workers.contains(&key_x)
            && by.workers.contains(&key_y);
        if converged {
            // Negative attribution: prefix X must not be attributed to
            // worker_y in either tree, and vice versa. A regression that
            // keyed events by arriving socket rather than announced
            // worker URL would set BOTH worker keys on each prefix.
            assert!(
                !ax.workers.contains(&key_y),
                "router_a cross-attributed worker_y to prefix X: {:?}",
                ax.workers,
            );
            assert!(
                !ay.workers.contains(&key_x),
                "router_a cross-attributed worker_x to prefix Y: {:?}",
                ay.workers,
            );
            assert!(
                !bx.workers.contains(&key_y),
                "router_b cross-attributed worker_y to prefix X: {:?}",
                bx.workers,
            );
            assert!(
                !by.workers.contains(&key_x),
                "router_b cross-attributed worker_x to prefix Y: {:?}",
                by.workers,
            );
            break;
        }
        if start.elapsed() > Duration::from_secs(3) {
            panic!(
                "trees did not converge within 3s:\n  \
                 router_a: X={{matched={}, workers={:?}}}, Y={{matched={}, workers={:?}}}\n  \
                 router_b: X={{matched={}, workers={:?}}}, Y={{matched={}, workers={:?}}}\n  \
                 targets: X={target_x}, Y={target_y}",
                ax.matched_blocks,
                ax.workers,
                ay.matched_blocks,
                ay.workers,
                bx.matched_blocks,
                bx.workers,
                by.matched_blocks,
                by.workers,
            );
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    let r = tokio::time::timeout(Duration::from_secs(2), Arc::clone(&router_a).shutdown()).await;
    assert!(r.is_ok(), "router_a shutdown hung");
    let r = tokio::time::timeout(Duration::from_secs(2), Arc::clone(&router_b).shutdown()).await;
    assert!(r.is_ok(), "router_b shutdown hung");
}
