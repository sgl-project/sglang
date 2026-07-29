// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use axum::{routing::get, Json, Router};
use serde_json::{json, Value};
use sgl_router::discovery::{DiscoveryEvent, ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::workers::{manager, WorkerRegistry};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
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
        bootstrap_port: None,
    }
}

#[tokio::test]
async fn manager_processes_added_then_removed() {
    let (url_a, _s_a) = spawn_fake_worker(json!({"served_model_name": "m"})).await;
    let (url_b, _s_b) = spawn_fake_worker(json!({"served_model_name": "m"})).await;

    let (tx, rx) = mpsc::channel(16);
    let registry = Arc::new(WorkerRegistry::default());
    let h = tokio::spawn(manager::run(rx, registry.clone()));

    tx.send(DiscoveryEvent::Added(spec_for(
        "w1",
        &url_a,
        WorkerMode::Plain,
    )))
    .await
    .unwrap();
    tx.send(DiscoveryEvent::Added(spec_for(
        "w2",
        &url_b,
        WorkerMode::Plain,
    )))
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

    tx.send(DiscoveryEvent::Added(spec_for(
        "w1",
        &url,
        WorkerMode::Prefill,
    )))
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
    let (url, _s) = spawn_fake_worker(json!({"served_model_name": "m"})).await;

    let (tx, rx) = mpsc::channel(16);
    let registry = Arc::new(WorkerRegistry::default());
    let h = tokio::spawn(manager::run(rx, registry.clone()));

    tx.send(DiscoveryEvent::Added(spec_for(
        "w1",
        &url,
        WorkerMode::Prefill,
    )))
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Grab a handle, bump active_requests, and open the breaker.
    let w = registry.get(&WorkerId("w1".into())).unwrap();
    w.active_requests.fetch_add(5, Ordering::Relaxed);
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
        w_after.active_requests.load(Ordering::Relaxed),
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

/// Duplicate `Added` for the same id is an upsert — the registry ends up
/// with exactly one worker. The model resolved by /server_info wins on
/// re-add (a different worker may advertise a different served model).
#[tokio::test]
async fn manager_handles_duplicate_added_as_upsert() {
    let (url_first, _s_first) = spawn_fake_worker(json!({"served_model_name": "m1"})).await;
    let (url_second, _s_second) = spawn_fake_worker(json!({"served_model_name": "m1"})).await;

    let (tx, rx) = mpsc::channel(16);
    let registry = Arc::new(WorkerRegistry::default());
    let h = tokio::spawn(manager::run(rx, registry.clone()));

    tx.send(DiscoveryEvent::Added(spec_for(
        "w1",
        &url_first,
        WorkerMode::Plain,
    )))
    .await
    .unwrap();
    tx.send(DiscoveryEvent::Added(spec_for(
        "w1",
        &url_second,
        WorkerMode::Plain,
    )))
    .await
    .unwrap();
    tokio::time::sleep(Duration::from_millis(300)).await;

    assert_eq!(
        registry.workers_for(&ModelId("m1".into())).len(),
        1,
        "w1 still serves m1 after the second Added",
    );

    drop(tx);
    h.await.unwrap();
}

/// Spawn a fake worker whose `/server_info` returns `body` only after
/// sleeping for `delay`. Returns the worker URL and a shutdown channel.
async fn spawn_slow_worker(body: Value, delay: Duration) -> (String, oneshot::Sender<()>) {
    let body = Arc::new(body);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let app = Router::new().route(
        "/server_info",
        get(move || {
            let body = body.clone();
            async move {
                tokio::time::sleep(delay).await;
                Json((*body).clone())
            }
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

/// Spawn a fake worker that counts each `GET /server_info` hit in the
/// returned `AtomicUsize`.  Used to assert the manager makes exactly
/// one round-trip per worker.
async fn spawn_counting_worker(body: Value) -> (String, Arc<AtomicUsize>, oneshot::Sender<()>) {
    let body = Arc::new(body);
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = counter.clone();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let app = Router::new().route(
        "/server_info",
        get(move || {
            let body = body.clone();
            let counter = counter_clone.clone();
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Json((*body).clone())
            }
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
    (format!("http://127.0.0.1:{port}"), counter, tx)
}

/// Registration must run in parallel across multiple `Added` events.
/// Each fake worker delays its `/server_info` by 200ms; with sequential
/// processing the manager would take ≥1000ms for 5 workers. We allow
/// up to 600ms (3x the per-fetch delay) as a generous bound that still
/// rejects the sequential implementation.
#[tokio::test]
async fn added_events_run_in_parallel() {
    let delay = Duration::from_millis(200);
    let n = 5;
    let mut workers = Vec::new();
    for _ in 0..n {
        workers.push(spawn_slow_worker(json!({"served_model_name": "m"}), delay).await);
    }

    let (tx, rx) = mpsc::channel(16);
    let registry = Arc::new(WorkerRegistry::default());
    let h = tokio::spawn(manager::run(rx, registry.clone()));

    let start = Instant::now();
    for (i, (url, _s)) in workers.iter().enumerate() {
        tx.send(DiscoveryEvent::Added(spec_for(
            &format!("w{i}"),
            url,
            WorkerMode::Plain,
        )))
        .await
        .unwrap();
    }
    let registered = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if registry.workers_for(&ModelId("m".into())).len() == n {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
    let elapsed = start.elapsed();
    assert!(registered.is_ok(), "manager failed to register {n} workers");
    assert!(
        elapsed < Duration::from_millis(600),
        "registration of {n} workers took {elapsed:?}; sequential per-worker /server_info \
         fetches would take ≥1000ms — parallel spawn is required"
    );

    drop(tx);
    h.await.unwrap();
}

/// A `Removed` issued while the matching `Added` is still mid-fetch
/// must await the in-flight registration handle before removing.
/// Without that ordering the removal runs first (registry has nothing
/// to remove), then the Added's deferred registry write leaks the
/// worker.
#[tokio::test]
async fn removed_awaits_pending_added() {
    let (url, _s) = spawn_slow_worker(
        json!({"served_model_name": "m"}),
        Duration::from_millis(300),
    )
    .await;

    let (tx, rx) = mpsc::channel(16);
    let registry = Arc::new(WorkerRegistry::default());
    let h = tokio::spawn(manager::run(rx, registry.clone()));

    tx.send(DiscoveryEvent::Added(spec_for(
        "w-slow",
        &url,
        WorkerMode::Plain,
    )))
    .await
    .unwrap();
    tx.send(DiscoveryEvent::Removed {
        id: WorkerId("w-slow".into()),
    })
    .await
    .unwrap();

    // Wait long enough for the Added's /server_info to complete (300ms),
    // then assert the worker is gone. If Removed ran before Added's
    // registry write, the post-fetch write would leak the entry.
    tokio::time::sleep(Duration::from_millis(600)).await;
    assert!(
        registry.get(&WorkerId("w-slow".into())).is_none(),
        "Removed must await the in-flight Added; otherwise the deferred \
         registry write leaks the worker"
    );

    drop(tx);
    h.await.unwrap();
}

/// The manager must make exactly ONE `/server_info` request per worker.
/// Before this fix the worker manager fetched `served_model_name` and
/// `KvEventIndex::add_worker` fetched the `kv_events` block
/// independently — 2N round-trips for N workers.
#[tokio::test]
async fn manager_emits_single_server_info_fetch_per_worker() {
    use sgl_router::policies::kv_events::KvEventIndex;

    let body = json!({
        "served_model_name": "m",
        "kv_events": {
            "publisher": "zmq",
            "endpoint_host": "127.0.0.1",
            "endpoint_port_base": 60100,
            "topic": "",
            "block_size": 64,
            "dp_size": 1,
        }
    });
    let (url, counter, _s) = spawn_counting_worker(body).await;

    let (tx, rx) = mpsc::channel(16);
    let registry = Arc::new(WorkerRegistry::default());
    let kv_index = KvEventIndex::new();
    let h = tokio::spawn(manager::run_with_config(
        rx,
        registry.clone(),
        None,
        Some(kv_index.clone()),
        None,
    ));

    tx.send(DiscoveryEvent::Added(spec_for(
        "w1",
        &url,
        WorkerMode::Plain,
    )))
    .await
    .unwrap();

    // Wait for both the registry and kv-events index to reflect the worker.
    let ready = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if registry.get(&WorkerId("w1".into())).is_some() && kv_index.known_worker_count() == 1
            {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await;
    assert!(
        ready.is_ok(),
        "manager did not finish onboarding the worker"
    );

    let hits = counter.load(Ordering::SeqCst);
    assert_eq!(
        hits, 1,
        "manager must fetch /server_info exactly once per worker (got {hits})"
    );

    drop(tx);
    h.await.unwrap();
    kv_index.shutdown().await;
}
