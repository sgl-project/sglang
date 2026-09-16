// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! A worker whose heartbeat lapses must lose its placements: through a key
//! expiry notification when the server emits them, through the sweep
//! otherwise. A worker that never heartbeated is never touched, and a bridge
//! whose worker probe fails must not heartbeat on its behalf.

#[allow(dead_code)]
#[path = "common/kv.rs"]
mod test_kv;
#[allow(dead_code)]
#[path = "common/valkey.rs"]
mod test_valkey;

use std::time::{Duration, Instant};

use sgl_kv_indexer::liveness::{alive_key, marker_key};
use sgl_kv_indexer::pb::{ExternalKvActionType, MatchExternalKvRequest};
use sgl_kv_indexer::{Heartbeat, KvIndexerBackend, LivenessWatcher, ValkeyKvIndexerBackend};
use test_kv::{action_with_parent, apply_request, hbm};
use test_valkey::{fresh_prefix, ValkeyServer};

macro_rules! require_valkey {
    ($($arg:expr),*) => {
        match ValkeyServer::start_with(&[$($arg),*]) {
            Some(server) => server,
            None => {
                eprintln!("skipping: no valkey-server on PATH and KV_INDEXER_TEST_VALKEY_URL unset");
                return;
            }
        }
    };
}

async fn held_hashes(backend: &ValkeyKvIndexerBackend, worker: &str, hashes: &[i64]) -> Vec<i64> {
    let resp = backend
        .match_external_kv(MatchExternalKvRequest {
            hashes: hashes.to_vec(),
            count_as_hit: false,
        })
        .await
        .unwrap();
    let mut held: Vec<i64> = resp
        .matches
        .iter()
        .filter(|node| node.worker_id == worker)
        .flat_map(|node| node.hashes_by_tier.iter().flat_map(|t| t.hashes.clone()))
        .collect();
    held.sort();
    held
}

async fn seed(backend: &ValkeyKvIndexerBackend, worker: &str, hashes: &[i64]) {
    backend
        .apply_external_kv_batch(apply_request(
            worker,
            &format!("http://{worker}"),
            1,
            vec![action_with_parent(
                ExternalKvActionType::ActionReport,
                hbm(),
                None,
                hashes,
            )],
        ))
        .await
        .unwrap();
}

async fn wait_until_cleared(
    backend: &ValkeyKvIndexerBackend,
    worker: &str,
    hashes: &[i64],
    what: &str,
) {
    let deadline = Instant::now() + Duration::from_secs(6);
    loop {
        if held_hashes(backend, worker, hashes).await.is_empty() {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "{what}: placements of {worker} were never cleared"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[tokio::test]
async fn expired_heartbeat_clears_the_worker_through_notifications() {
    let server = require_valkey!("--notify-keyspace-events", "Ex");
    let prefix = fresh_prefix();
    let backend = server.backend(&prefix).await;
    seed(&backend, "w0", &[1, 2, 3]).await;
    seed(&backend, "w1", &[1, 2]).await;
    seed(&backend, "w2", &[5]).await;

    // w2 keeps heartbeating throughout: the watcher clears a worker only while
    // some other heartbeating worker is still alive.
    let steady = Heartbeat::connect(&server.config(&prefix), "w2", "", Duration::from_secs(60))
        .await
        .unwrap();
    let (steady_tx, steady_rx) = tokio::sync::oneshot::channel::<()>();
    let steady_task = tokio::spawn(steady.run(async move {
        let _ = steady_rx.await;
    }));

    // Heartbeat for w0 only, then stop it and let the key expire.
    let ttl = Duration::from_millis(400);
    let heartbeat = Heartbeat::connect(&server.config(&prefix), "w0", "", ttl)
        .await
        .unwrap();
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
    let beating = tokio::spawn(heartbeat.run(async move {
        let _ = stop_rx.await;
    }));
    // Sweep interval far away: only the notification can clear in time.
    let watcher = LivenessWatcher::new(
        backend.clone(),
        server.config(&prefix),
        Duration::from_secs(600),
    );
    let (watch_tx, watch_rx) = tokio::sync::oneshot::channel::<()>();
    let watching = tokio::spawn(watcher.run(async move {
        let _ = watch_rx.await;
    }));
    tokio::time::sleep(Duration::from_millis(600)).await;
    assert_eq!(
        held_hashes(&backend, "w0", &[1, 2, 3]).await,
        vec![1, 2, 3],
        "alive worker was cleared"
    );

    let _ = stop_tx.send(());
    let _ = beating.await;
    wait_until_cleared(&backend, "w0", &[1, 2, 3], "notification").await;
    // A worker without heartbeats is legacy: never declared dead.
    assert_eq!(held_hashes(&backend, "w1", &[1, 2]).await, vec![1, 2]);
    let _ = watch_tx.send(());
    let _ = watching.await;
    let _ = steady_tx.send(());
    let _ = steady_task.await;
}

#[tokio::test]
async fn sweep_clears_marked_workers_without_a_live_key() {
    let server = require_valkey!();
    let prefix = fresh_prefix();
    let backend = server.backend(&prefix).await;
    seed(&backend, "w0", &[1, 2, 3]).await;
    seed(&backend, "w1", &[1, 2]).await;
    seed(&backend, "w2", &[5]).await;
    let mut raw = server.raw().await;
    // w0 heartbeated once and its key is gone; w1 never did; w2 is alive, which
    // is what tells the watcher this is one dead worker and not a Valkey event.
    let _: () = redis::cmd("SET")
        .arg(marker_key(&prefix, "w0"))
        .arg(1)
        .query_async(&mut raw)
        .await
        .unwrap();
    for key in [marker_key(&prefix, "w2"), alive_key(&prefix, "w2")] {
        let _: () = redis::cmd("SET")
            .arg(key)
            .arg(1)
            .arg("PX")
            .arg(60_000)
            .query_async(&mut raw)
            .await
            .unwrap();
    }

    let watcher = LivenessWatcher::new(
        backend.clone(),
        server.config(&prefix),
        Duration::from_secs(600),
    );
    assert_eq!(watcher.sweep().await.unwrap(), 1);
    assert!(held_hashes(&backend, "w0", &[1, 2, 3]).await.is_empty());
    assert_eq!(held_hashes(&backend, "w1", &[1, 2]).await, vec![1, 2]);
    // Idempotent: nothing left to clear.
    assert_eq!(watcher.sweep().await.unwrap(), 0);

    // A live key protects a marked worker.
    seed(&backend, "w0", &[1, 2, 3]).await;
    let _: () = redis::cmd("SET")
        .arg(alive_key(&prefix, "w0"))
        .arg(1)
        .arg("PX")
        .arg(60_000)
        .query_async(&mut raw)
        .await
        .unwrap();
    assert_eq!(watcher.sweep().await.unwrap(), 0);
    assert_eq!(held_hashes(&backend, "w0", &[1, 2, 3]).await, vec![1, 2, 3]);
}

#[tokio::test]
async fn heartbeat_does_not_vouch_for_an_unreachable_worker() {
    let server = require_valkey!();
    let prefix = fresh_prefix();
    // Port 1 refuses connections: the probe fails on every beat.
    let heartbeat = Heartbeat::connect(
        &server.config(&prefix),
        "w0",
        "http://127.0.0.1:1",
        Duration::from_millis(300),
    )
    .await
    .unwrap();
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
    let beating = tokio::spawn(heartbeat.run(async move {
        let _ = stop_rx.await;
    }));
    tokio::time::sleep(Duration::from_millis(500)).await;
    let _ = stop_tx.send(());
    let _ = beating.await;

    let mut raw = server.raw().await;
    let alive: i64 = redis::cmd("EXISTS")
        .arg(alive_key(&prefix, "w0"))
        .query_async(&mut raw)
        .await
        .unwrap();
    let marked: i64 = redis::cmd("EXISTS")
        .arg(marker_key(&prefix, "w0"))
        .query_async(&mut raw)
        .await
        .unwrap();
    assert_eq!((alive, marked), (0, 0));
}

/// A Valkey failover or a restore that brings back the markers but not the
/// volatile heartbeats makes every worker look dead at once. Clearing then turns
/// one infrastructure event into a fleet-wide loss of cache affinity, so the
/// watcher must refuse and say so.
#[tokio::test]
async fn a_fleet_that_looks_entirely_dead_is_not_cleared() {
    let server = require_valkey!();
    let prefix = fresh_prefix();
    let backend = server.backend(&prefix).await;
    seed(&backend, "w0", &[1, 2, 3]).await;
    seed(&backend, "w1", &[4, 5]).await;
    let mut raw = server.raw().await;
    // Both heartbeated before; neither has a live key now.
    for worker in ["w0", "w1"] {
        let _: () = redis::cmd("SET")
            .arg(marker_key(&prefix, worker))
            .arg(1)
            .query_async(&mut raw)
            .await
            .unwrap();
    }

    let watcher = LivenessWatcher::new(
        backend.clone(),
        server.config(&prefix),
        Duration::from_secs(600),
    );
    assert_eq!(watcher.sweep().await.unwrap(), 0);
    assert_eq!(held_hashes(&backend, "w0", &[1, 2, 3]).await, vec![1, 2, 3]);
    assert_eq!(held_hashes(&backend, "w1", &[4, 5]).await, vec![4, 5]);

    // One worker coming back is the evidence that the others really are gone.
    let _: () = redis::cmd("SET")
        .arg(alive_key(&prefix, "w1"))
        .arg(1)
        .arg("PX")
        .arg(60_000)
        .query_async(&mut raw)
        .await
        .unwrap();
    assert_eq!(watcher.sweep().await.unwrap(), 1);
    assert!(held_hashes(&backend, "w0", &[1, 2, 3]).await.is_empty());
    assert_eq!(held_hashes(&backend, "w1", &[4, 5]).await, vec![4, 5]);
}

/// A cleared worker must be replayed from the start of its buffer, so the clear
/// also drops the bridge's sequence checkpoint; leaving it means the bridge
/// resumes past events the index no longer has and the worker stays empty until
/// its cache churns.
#[tokio::test]
async fn clearing_a_worker_drops_its_bridge_checkpoint() {
    let server = require_valkey!();
    let prefix = fresh_prefix();
    let backend = server.backend(&prefix).await;
    seed(&backend, "w0", &[1, 2, 3]).await;
    seed(&backend, "w1", &[9]).await;
    let mut raw = server.raw().await;
    let checkpoint = format!("{prefix}seq:w0");
    let _: () = redis::cmd("SET")
        .arg(&checkpoint)
        .arg(4242)
        .query_async(&mut raw)
        .await
        .unwrap();
    let _: () = redis::cmd("SET")
        .arg(marker_key(&prefix, "w0"))
        .arg(1)
        .query_async(&mut raw)
        .await
        .unwrap();
    for key in [marker_key(&prefix, "w1"), alive_key(&prefix, "w1")] {
        let _: () = redis::cmd("SET")
            .arg(key)
            .arg(1)
            .arg("PX")
            .arg(60_000)
            .query_async(&mut raw)
            .await
            .unwrap();
    }

    let watcher = LivenessWatcher::new(
        backend.clone(),
        server.config(&prefix),
        Duration::from_secs(600),
    );
    assert_eq!(watcher.sweep().await.unwrap(), 1);
    let left: i64 = redis::cmd("EXISTS")
        .arg(&checkpoint)
        .query_async(&mut raw)
        .await
        .unwrap();
    assert_eq!(left, 0, "the cleared worker's checkpoint must be gone");
}
