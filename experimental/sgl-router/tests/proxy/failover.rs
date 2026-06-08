// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use axum::body::Body;
use axum::http::Request;
use sgl_router::config::*;
use sgl_router::discovery::{spawn_discovery, ModelId};
use sgl_router::policies::factory::build_registry_with_defaults as build_policy_registry;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::manager;
use sgl_router::workers::WorkerRegistry;
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

#[tokio::test]
async fn failover_when_one_worker_dies() {
    // Three mock workers. Each advertises served_model_name = "tiny" on
    // /server_info, so the worker manager's introspect step resolves the
    // registry's model_ids without us having to hand-declare them here.
    let w1 = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let w2 = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let w3 = crate::common::mock_worker::MockWorker::start(vec![]).await;

    let cfg = Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
        },
        observability: Default::default(),
        model: ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            policy: PolicyKind::RoundRobin,
            circuit_breaker: Some(CircuitBreakerConfig {
                threshold: std::num::NonZeroU32::new(1).unwrap(), // open after first failure
                cool_down_secs: 30,
            }),
            cache_aware: None,
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec![w1.url.clone(), w2.url.clone(), w3.url.clone()],
        }),
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
    };

    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    let policies = Arc::new(build_policy_registry(&cfg).unwrap());

    let (event_rx, _disc) = spawn_discovery(&cfg).await.unwrap();
    let _mgr = tokio::spawn(manager::run_with_config(
        event_rx,
        registry.clone(),
        Some(Arc::new(cfg.clone())),
        None,
        None,
    ));

    // Poll for the registry to converge — `register_one` introspect is
    // a per-task spawn (manager.rs:127), so order of registration is
    // non-deterministic under load. Cap the wait so a real hang surfaces
    // instead of becoming a flake.
    let converged = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if registry.workers_for(&ModelId("tiny".into())).len() == 3 {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await;
    assert!(
        converged.is_ok(),
        "registry should contain all 3 workers after discovery; have {}",
        registry.workers_for(&ModelId("tiny".into())).len()
    );

    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    let ctx = Arc::new(AppContext::new(
        cfg,
        tokenizers,
        proxy,
        registry.clone(),
        policies,
    ));
    ctx.mark_ready();
    let app = build_router(ctx);

    // Kill w2 by dropping its handle, then poll until its socket
    // actually refuses connections. Without this, the first request
    // routed to w2 can race against the listener's graceful shutdown
    // and succeed, masking the failover assertion below.
    let w2_url = w2.url.clone();
    drop(w2);
    let host_port = w2_url.trim_start_matches("http://");
    let down = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if tokio::net::TcpStream::connect(host_port).await.is_err() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await;
    assert!(down.is_ok(), "w2 socket never went down");

    // Send 6 requests; round-robin would route 2 to w2 → connection refused →
    // breaker opens (threshold=1); subsequent round-robin picks rotate among
    // the 2 healthy workers (#1 and #3) because healthy_workers_for filters out w2.
    let mut errs = 0usize;
    let mut oks = 0usize;
    for i in 0..6 {
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "tiny",
            "messages": [{"role": "user", "content": format!("hi {i}")}],
        }))
        .unwrap();
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();
        let res = app.clone().oneshot(req).await.unwrap();
        if res.status().is_success() {
            oks += 1;
        } else {
            errs += 1;
        }
    }
    // We expect exactly 1 error — the first call routed to w2 fails and opens
    // its breaker; subsequent round-robin picks rotate among the 2 healthy
    // workers since registry.healthy_workers_for filters out the open breaker.
    assert_eq!(errs, 1, "exactly the first w2 pick should error");
    assert_eq!(oks, 5, "remaining 5 picks should succeed via filtered RR");
}
