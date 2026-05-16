// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

mod common;

use axum::body::Body;
use axum::http::Request;
use sgl_router::config::*;
use sgl_router::discovery::{spawn_discovery, ModelId};
use sgl_router::policies::factory::build_registry as build_policy_registry;
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
    // Three mock workers.
    let w1 = common::mock_worker::MockWorker::start(vec![]).await;
    let w2 = common::mock_worker::MockWorker::start(vec![]).await;
    let w3 = common::mock_worker::MockWorker::start(vec![]).await;

    // Write a workers.toml with all 3.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("workers.toml");
    tokio::fs::write(
        &path,
        format!(
            r#"
[[workers]]
id = "w1"
url = "{}"
mode = "plain"
model_ids = ["tiny"]

[[workers]]
id = "w2"
url = "{}"
mode = "plain"
model_ids = ["tiny"]

[[workers]]
id = "w3"
url = "{}"
mode = "plain"
model_ids = ["tiny"]
"#,
            w1.url, w2.url, w3.url
        ),
    )
    .await
    .unwrap();

    let cfg = Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
        },
        observability: Default::default(),
        models: vec![ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            policy: "round_robin".into(),
            circuit_breaker: Some(CircuitBreakerConfig {
                threshold: 1, // open after first failure
                cool_down_secs: 30,
            }),
        }],
        discovery: DiscoveryConfig {
            backend: DiscoveryBackend::StaticFile(StaticFileDiscoveryConfig {
                path: path.to_string_lossy().into_owned(),
                poll_interval_ms: 50,
            }),
        },
    };

    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    let policies = Arc::new(build_policy_registry(&cfg).unwrap());

    let (event_rx, _disc) = spawn_discovery(&cfg).await.unwrap();
    let _mgr = tokio::spawn(manager::run_with_config(
        event_rx,
        registry.clone(),
        Some(Arc::new(cfg.clone())),
    ));

    // Give time for the registry to receive Added events.
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(
        registry.workers_for(&ModelId("tiny".into())).len(),
        3,
        "registry should contain all 3 workers after discovery"
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

    // Kill w2 by dropping its handle.
    drop(w2);
    // Give time for OS to release the port.
    tokio::time::sleep(Duration::from_millis(100)).await;

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
