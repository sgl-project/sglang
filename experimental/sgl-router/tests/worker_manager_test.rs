// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use axum::{routing::get, Json, Router};
use serde_json::{json, Value};
use sgl_router::discovery::{DiscoveryEvent, ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::workers::{manager, WorkerRegistry};
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot};

/// Spin up a tiny fake worker that returns `body` on `GET /server_info`.
/// Returns the worker base URL and a shutdown channel.
async fn spawn_fake_worker(body: Value) -> (String, oneshot::Sender<()>) {
    let body = Arc::new(body);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let app = Router::new().route(
        "/server_info",
        get(move || {
            let body = body.clone();
            async move { Json((*body).clone()) }
        }),
    );
    let (tx, rx) = oneshot::channel::<()>();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = rx.await;
            })
            .await;
    });
    (format!("http://127.0.0.1:{port}"), tx)
}

fn spec_for(id: &str, url: &str, mode: WorkerMode) -> WorkerSpec {
    // model_ids are intentionally empty: the manager resolves them via
    // /server_info introspection.  Pre-populating here would lie about
    // what discovery backends actually emit.
    WorkerSpec {
        id: WorkerId(id.into()),
        url: url.into(),
        mode,
        model_ids: Vec::new(),
    }
}

#[tokio::test]
async fn manager_processes_added_then_removed() {
    let (url_a, _s_a) = spawn_fake_worker(json!({"served_model_name": "m"})).await;
    let (url_b, _s_b) = spawn_fake_worker(json!({"served_model_name": "m"})).await;

    let (tx, rx) = mpsc::channel(16);
    let registry = Arc::new(WorkerRegistry::default());
    let h = tokio::spawn(manager::run(rx, registry.clone()));

    tx.send(DiscoveryEvent::Added(spec_for("w1", &url_a, WorkerMode::Plain)))
        .await
        .unwrap();
    tx.send(DiscoveryEvent::Added(spec_for("w2", &url_b, WorkerMode::Plain)))
        .await
        .unwrap();

    // Give the manager time to drain.
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(registry.workers_for(&ModelId("m".into())).len(), 2);

    tx.send(DiscoveryEvent::Removed {
        id: WorkerId("w1".into()),
    })
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(registry.workers_for(&ModelId("m".into())).len(), 1);

    drop(tx);
    h.await.unwrap();
}

#[tokio::test]
async fn manager_handles_mode_changed() {
    let (url, _s) = spawn_fake_worker(json!({"served_model_name": "m"})).await;

    let (tx, rx) = mpsc::channel(16);
    let registry = Arc::new(WorkerRegistry::default());
    let h = tokio::spawn(manager::run(rx, registry.clone()));

    tx.send(DiscoveryEvent::Added(spec_for("w1", &url, WorkerMode::Prefill)))
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(
        registry
            .workers_for_mode(&ModelId("m".into()), WorkerMode::Prefill)
            .len(),
        1
    );

    tx.send(DiscoveryEvent::ModeChanged {
        id: WorkerId("w1".into()),
        mode: WorkerMode::Decode,
    })
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(
        registry
            .workers_for_mode(&ModelId("m".into()), WorkerMode::Prefill)
            .len(),
        0
    );
    assert_eq!(
        registry
            .workers_for_mode(&ModelId("m".into()), WorkerMode::Decode)
            .len(),
        1
    );

    drop(tx);
    h.await.unwrap();
}

#[tokio::test]
async fn mode_changed_preserves_active_requests_and_breaker() {
    let (tx, rx) = mpsc::channel(16);
    let registry = Arc::new(WorkerRegistry::default());
    let h = tokio::spawn(manager::run(rx, registry.clone()));

    tx.send(DiscoveryEvent::Added(spec(
        "w1",
        "http://x:30000",
        WorkerMode::Prefill,
        &["m"],
    )))
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Grab a handle, bump active_requests, and open the breaker.
    let w = registry.get(&WorkerId("w1".into())).unwrap();
    w.active_requests
        .fetch_add(5, std::sync::atomic::Ordering::Relaxed);
    // Default threshold is 3 — record 10 failures to guarantee Open state.
    for _ in 0..10 {
        w.breaker.record_failure();
    }
    let breaker_open_before = !w.breaker.allow();
    assert!(
        breaker_open_before,
        "breaker should be open after 10 failures"
    );

    // Flip mode via ModeChanged.
    tx.send(DiscoveryEvent::ModeChanged {
        id: WorkerId("w1".into()),
        mode: WorkerMode::Decode,
    })
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Re-fetch the Worker handle from the registry.
    let w_after = registry.get(&WorkerId("w1".into())).unwrap();

    assert_eq!(
        w_after.mode(),
        WorkerMode::Decode,
        "mode should have flipped to Decode"
    );
    assert_eq!(
        w_after
            .active_requests
            .load(std::sync::atomic::Ordering::Relaxed),
        5,
        "active_requests should be preserved across mode change"
    );
    assert!(
        !w_after.breaker.allow(),
        "breaker open state should be preserved across mode change"
    );

    // Critical: the Arc identity must be the same — mutation in place.
    assert!(
        Arc::ptr_eq(&w, &w_after),
        "Worker handle should be the SAME Arc, not a fresh replacement"
    );

    drop(tx);
    h.await.unwrap();
}

/// An out-of-order `ModeChanged` for a worker the registry does not know
/// about (e.g. a buggy discovery backend reordered `Removed` and
/// `ModeChanged`) must not panic, must not silently log INFO claiming the
/// mode flip happened, and must leave the registry untouched.
#[tokio::test]
async fn manager_handles_orphan_mode_changed_without_panic() {
    let (tx, rx) = mpsc::channel(16);
    let registry = Arc::new(WorkerRegistry::default());
    let h = tokio::spawn(manager::run(rx, registry.clone()));

    tx.send(DiscoveryEvent::ModeChanged {
        id: WorkerId("ghost".into()),
        mode: WorkerMode::Decode,
    })
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    assert!(
        registry.get(&WorkerId("ghost".into())).is_none(),
        "an orphan ModeChanged must not create a phantom worker",
    );
    assert_eq!(
        registry.workers_for(&ModelId("m".into())).len(),
        0,
        "registry must be empty after an orphan event",
    );

    drop(tx);
    h.await.unwrap();
}

/// A `Removed` for an unknown id is a no-op — registry stays empty, manager
/// keeps running.
#[tokio::test]
async fn manager_handles_orphan_removed_without_panic() {
    let (tx, rx) = mpsc::channel(16);
    let registry = Arc::new(WorkerRegistry::default());
    let h = tokio::spawn(manager::run(rx, registry.clone()));

    tx.send(DiscoveryEvent::Removed {
        id: WorkerId("ghost".into()),
    })
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(registry.is_empty());

    drop(tx);
    h.await.unwrap();
}

/// Duplicate `Added` for the same id is an upsert — the registry must end
/// up with exactly one worker carrying the latest spec's `model_ids`.
#[tokio::test]
async fn manager_handles_duplicate_added_as_upsert() {
    let (tx, rx) = mpsc::channel(16);
    let registry = Arc::new(WorkerRegistry::default());
    let h = tokio::spawn(manager::run(rx, registry.clone()));

    tx.send(DiscoveryEvent::Added(spec(
        "w1",
        "http://x:30000",
        WorkerMode::Plain,
        &["m1", "m2"],
    )))
    .await
    .unwrap();
    tx.send(DiscoveryEvent::Added(spec(
        "w1",
        "http://x:30000",
        WorkerMode::Plain,
        &["m1"],
    )))
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    assert_eq!(
        registry.workers_for(&ModelId("m1".into())).len(),
        1,
        "w1 still serves m1 after the second Added",
    );
    assert_eq!(
        registry.workers_for(&ModelId("m2".into())).len(),
        0,
        "w1 must drop out of m2's index after the model set shrank",
    );

    drop(tx);
    h.await.unwrap();
}
