// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use sgl_router::config::StaticUrlsDiscoveryConfig;
use sgl_router::discovery::{DiscoveryEvent, WorkerMode};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

#[tokio::test]
async fn emits_one_added_per_url_with_plain_seed() {
    let cfg = StaticUrlsDiscoveryConfig {
        urls: vec!["http://x:30000".into(), "http://y:30000".into()],
    };
    let (tx, mut rx) = mpsc::channel(16);
    let _h = sgl_router::discovery::static_urls::spawn(cfg, tx)
        .await
        .unwrap();

    let mut seen = std::collections::HashSet::new();
    for _ in 0..2 {
        let event = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .unwrap()
            .unwrap();
        match event {
            DiscoveryEvent::Added(spec) => {
                // mode / model_ids / bootstrap_port are seeded as Plain/empty/None;
                // the worker manager fills them from /server_info post-discovery.
                assert_eq!(spec.mode, WorkerMode::Plain);
                assert!(spec.model_ids.is_empty());
                assert_eq!(spec.bootstrap_port, None);
                // The URL doubles as the worker id — strings already have to
                // be unique (rejected at config-load otherwise).
                assert_eq!(spec.id.0, spec.url);
                seen.insert(spec.url);
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }
    assert_eq!(
        seen,
        ["http://x:30000".to_string(), "http://y:30000".to_string()].into(),
    );
}

/// Single-URL list — the common dev deployment shape. The producer
/// emits exactly one event and then parks until the receiver is
/// dropped. Earlier versions exited as soon as fan-out completed,
/// which tripped `server::supervisor::supervise_critical_tasks` →
/// `mark_unready` → `/readyz` 503; the lib-side
/// `stays_alive_after_fanout_until_receiver_dropped` pins that
/// invariant in isolation, while this test pins the same contract
/// through the public `spawn` entry point used by the binary.
#[tokio::test]
async fn emits_one_event_and_parks_until_receiver_dropped() {
    let cfg = StaticUrlsDiscoveryConfig {
        urls: vec!["http://x:30000".into()],
    };
    let (tx, mut rx) = mpsc::channel(16);
    let h = sgl_router::discovery::static_urls::spawn(cfg, tx)
        .await
        .unwrap();
    let event = rx.recv().await.unwrap();
    assert!(matches!(event, DiscoveryEvent::Added(_)));
    assert!(rx.try_recv().is_err(), "exactly one event expected");

    // Drop the receiver → producer's `tx.closed()` resolves → task
    // exits cleanly.
    drop(rx);
    tokio::time::timeout(Duration::from_secs(2), h)
        .await
        .expect("static_urls task should exit after receiver is dropped")
        .expect("join handle should not panic");
}

/// Spin up a fake worker that advertises
/// `disaggregation_mode = "prefill"` + `disaggregation_bootstrap_port`,
/// pipe it through `spawn_discovery` (StaticUrls backend) into
/// `manager::run_with_config`, and assert the worker lands in the
/// registry with `WorkerMode::Prefill` + the disclosed port.
///
/// This is the load-bearing end-to-end assertion for the refactor's
/// central claim — "prefill, decode, and plain workers can all appear
/// in the same `urls` list and end up classified correctly" — exercised
/// against the full discovery → introspect → registry pipeline rather
/// than just the in-isolation `register_one` unit test.
#[tokio::test]
async fn static_urls_pd_role_resolved_end_to_end() {
    use axum::{routing::get, Json, Router};
    use serde_json::json;
    use sgl_router::config::{
        ActiveLoadConfig, Config, DiscoveryBackend, ObservabilityConfig, ProxyConfig, ServerConfig,
    };
    use sgl_router::discovery::{spawn_discovery, WorkerId};
    use sgl_router::workers::{manager, WorkerRegistry};
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;

    // Fake worker advertising a prefill role + bootstrap port.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let url = format!("http://127.0.0.1:{port}");
    let app = Router::new().route(
        "/server_info",
        get(|| async {
            Json(json!({
                "served_model_name": "tiny",
                "disaggregation_mode": "prefill",
                "disaggregation_bootstrap_port": 8998,
            }))
        }),
    );
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await;
    });

    let cfg = Config {
        server: ServerConfig {
            host: "127.0.0.1".into(),
            port: 0,
            ..Default::default()
        },
        observability: ObservabilityConfig::default(),
        model: sgl_router::config::ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            tokenizer_shards: 1,
            tokenizer_backend: Default::default(),
            tokenizer_l1_cache_mb: 0,
            policy: sgl_router::config::PolicyKind::RoundRobin,
            circuit_breaker: None,
            cache_aware: None,
            sticky: None,
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec![url.clone()],
        }),
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
        admission: sgl_router::config::AdmissionConfig::default(),
        retry: sgl_router::config::RetryConfig::default(),
    };

    let registry = Arc::new(WorkerRegistry::default());
    let (event_rx, _disc) = spawn_discovery(&cfg).await.unwrap();
    let _mgr = tokio::spawn(manager::run_with_config(
        event_rx,
        registry.clone(),
        Some(Arc::new(cfg)),
        None,
        None,
        None,
    ));

    let id = WorkerId(url);
    let resolved = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if let Some(w) = registry.get(&id) {
                if w.mode() == WorkerMode::Prefill && w.bootstrap_port() == Some(8998) {
                    return true;
                }
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await;
    assert!(
        resolved.is_ok(),
        "expected mode=Prefill bootstrap_port=Some(8998); got {:?}",
        registry.get(&id).map(|w| (w.mode(), w.bootstrap_port()))
    );

    let _ = shutdown_tx.send(());
}
