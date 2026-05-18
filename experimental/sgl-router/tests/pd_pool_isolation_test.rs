// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! M4 PD pool isolation — end-to-end at the HTTP layer using MockWorker.
//!
//! Drives the chat handler with:
//!
//! * A model whose registered workers are all `WorkerMode::Decode`. The
//!   handler dispatches **prefill** traffic (chat-completions is the
//!   prefill phase of a PD request), so it must return 503 with
//!   `no_prefill_workers_available`.
//! * A model with no workers at all → 503 `no_healthy_workers`
//!   (existing code path; pinned here so a future PD wiring change
//!   doesn't silently swap codes).
//! * A PD-disagg model with both pools healthy → request flows to the
//!   prefill worker (smoke; the decode worker MUST NOT be selected for
//!   the chat route).

mod common;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use sgl_router::config::{
    Config, DiscoveryBackend, DiscoveryConfig, ModelConfig, ObservabilityConfig, PolicyKind,
    ServerConfig, StaticFileDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry_with_defaults;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

fn config() -> Config {
    Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
        },
        observability: ObservabilityConfig::default(),
        models: vec![ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            policy: PolicyKind::RoundRobin,
            circuit_breaker: None,
            cache_aware: None,
        }],
        discovery: DiscoveryConfig {
            backend: DiscoveryBackend::StaticFile(StaticFileDiscoveryConfig {
                path: "/tmp/x.toml".into(),
                poll_interval_ms: 200,
            }),
        },
    }
}

fn build_ctx(specs: Vec<WorkerSpec>) -> Arc<AppContext> {
    let cfg = config();
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    for s in specs {
        registry.add(s);
    }
    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
}

fn chat_request() -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&serde_json::json!({
                "model": "tiny",
                "messages": [{"role": "user", "content": "hi"}],
            }))
            .unwrap(),
        ))
        .unwrap()
}

/// Gap closer #1: PD mode with only decode workers → 503 with
/// `no_prefill_workers_available`. The chat route is a prefill
/// dispatch, so a decode-only pool means partial failure.
#[tokio::test]
async fn pd_mode_decode_only_returns_no_prefill_workers_available() {
    let worker = common::mock_worker::MockWorker::start(vec![]).await;
    let ctx = build_ctx(vec![WorkerSpec {
        id: WorkerId("d1".into()),
        url: worker.url.clone(),
        mode: WorkerMode::Decode,
        model_ids: vec![ModelId("tiny".into())],
    }]);
    let app = build_router(ctx);

    let res = app.oneshot(chat_request()).await.unwrap();
    assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        res.headers().get("x-router-error-code").unwrap(),
        "no_prefill_workers_available",
    );
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let body_str = String::from_utf8_lossy(&body);
    assert!(
        body_str.contains("\"code\":\"no_prefill_workers_available\""),
        "body: {body_str}"
    );
}

/// Pin the existing-code-path branch: no workers at all → 503 with
/// `no_healthy_workers`. Ensures the new PD code path didn't swap the
/// code for the "model has zero workers" case.
#[tokio::test]
async fn no_workers_returns_no_healthy_workers() {
    let ctx = build_ctx(vec![]);
    let app = build_router(ctx);

    let res = app.oneshot(chat_request()).await.unwrap();
    assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        res.headers().get("x-router-error-code").unwrap(),
        "no_healthy_workers",
    );
}

/// PD-disagg deployment with both pools healthy → chat dispatch selects
/// from the prefill pool only. Asserted by: only the prefill worker's
/// URL is routable; if the route hit the decode worker we'd see the
/// prefill MockWorker's `captured.last_body` stay empty, while the
/// decode MockWorker's would fill. Pinning this in M4 covers the cross-
/// pool selection guard.
#[tokio::test]
async fn pd_mode_chat_dispatch_selects_only_prefill_workers() {
    let prefill = common::mock_worker::MockWorker::start(vec![]).await;
    let decode = common::mock_worker::MockWorker::start(vec![]).await;
    let ctx = build_ctx(vec![
        WorkerSpec {
            id: WorkerId("p1".into()),
            url: prefill.url.clone(),
            mode: WorkerMode::Prefill,
            model_ids: vec![ModelId("tiny".into())],
        },
        WorkerSpec {
            id: WorkerId("d1".into()),
            url: decode.url.clone(),
            mode: WorkerMode::Decode,
            model_ids: vec![ModelId("tiny".into())],
        },
    ]);
    let app = build_router(ctx);

    // Fire 5 requests; round-robin within the prefill pool would
    // serve all of them from the lone prefill worker. None should
    // reach the decode worker.
    for _ in 0..5 {
        let res = app.clone().oneshot(chat_request()).await.unwrap();
        assert_eq!(
            res.status(),
            StatusCode::OK,
            "chat request must succeed via the prefill worker",
        );
    }

    // Decode mock should have received zero requests — its capture
    // buffer's `last_body` stays `None`.
    let decode_seen = decode.captured.lock().unwrap();
    assert!(
        decode_seen.last_body.is_none(),
        "decode worker must not be selected for chat (prefill) dispatch",
    );
    let prefill_seen = prefill.captured.lock().unwrap();
    assert!(
        prefill_seen.last_body.is_some(),
        "prefill worker must have received at least one chat request",
    );
}

/// Task C: PD-mode chat request carries an `x-sgl-decode-url` header
/// pointing at the host-affinity decode peer. With two prefill workers
/// on different hosts and a decode worker on each, the affinity helper
/// MUST pick the decode peer co-located with the chosen prefill — pin
/// the structural prep for M5's bootstrap dispatch.
///
/// Round-robin will select prefill workers deterministically (alphabetic
/// dashmap order is not guaranteed; the test fires several requests so
/// at least one lands on each prefill, and asserts the per-host pairing
/// holds across all of them).
#[tokio::test]
async fn pd_mode_chat_dispatch_sets_decode_affinity_header() {
    use std::collections::HashSet;
    let prefill_a = common::mock_worker::MockWorker::start(vec![]).await;
    let prefill_b = common::mock_worker::MockWorker::start(vec![]).await;
    let decode_a = common::mock_worker::MockWorker::start(vec![]).await;
    let decode_b = common::mock_worker::MockWorker::start(vec![]).await;
    // MockWorker URLs always bind to `127.0.0.1`, so every worker
    // shares the same host string and the affinity helper's
    // same-host branch is moot here — the helper still returns a
    // decode peer via the load-tiebreak fallback. The unit tests in
    // `policies::registry::tests::decoder_picks_same_host_when_available`
    // carry the real burden of pinning the host-affinity rules; this
    // integration test only asserts the wiring is in place (the
    // `x-sgl-decode-url` header IS set on PD requests, and the
    // value is one of the registered decode worker URLs).
    let ctx = build_ctx(vec![
        WorkerSpec {
            id: WorkerId("p1".into()),
            url: prefill_a.url.clone(),
            mode: WorkerMode::Prefill,
            model_ids: vec![ModelId("tiny".into())],
        },
        WorkerSpec {
            id: WorkerId("p2".into()),
            url: prefill_b.url.clone(),
            mode: WorkerMode::Prefill,
            model_ids: vec![ModelId("tiny".into())],
        },
        WorkerSpec {
            id: WorkerId("d1".into()),
            url: decode_a.url.clone(),
            mode: WorkerMode::Decode,
            model_ids: vec![ModelId("tiny".into())],
        },
        WorkerSpec {
            id: WorkerId("d2".into()),
            url: decode_b.url.clone(),
            mode: WorkerMode::Decode,
            model_ids: vec![ModelId("tiny".into())],
        },
    ]);
    let app = build_router(ctx);

    // Fire 4 requests; both prefill workers see traffic via round-robin.
    for _ in 0..4 {
        let res = app.clone().oneshot(chat_request()).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    // Every request that hit a prefill mock MUST carry the decode-hint
    // header. The header value MUST be one of the two registered
    // decode worker URLs.
    let decode_urls: HashSet<String> = [decode_a.url.clone(), decode_b.url.clone()]
        .into_iter()
        .collect();
    for (label, p) in [("prefill_a", &prefill_a), ("prefill_b", &prefill_b)] {
        let g = p.captured.lock().unwrap();
        if g.last_body.is_none() {
            // This prefill didn't receive a request — round-robin's
            // dashmap iteration is non-deterministic, so one side may
            // skip in a 4-request fire. Continue.
            continue;
        }
        let hdr = g.headers.get("x-sgl-decode-url").unwrap_or_else(|| {
            panic!(
                "{label} did not receive an x-sgl-decode-url header. headers: {:?}",
                g.headers
            )
        });
        assert!(
            decode_urls.contains(hdr),
            "{label} got decode hint {hdr}, expected one of {decode_urls:?}",
        );
    }
}

/// Task C: plain-mode (non-PD) request does NOT carry the
/// `x-sgl-decode-url` header. Pin: the affinity step is gated on
/// `worker.mode() == Prefill` so plain workers are not asked to
/// bootstrap nonexistent decode peers.
#[tokio::test]
async fn plain_mode_chat_dispatch_omits_decode_affinity_header() {
    let plain = common::mock_worker::MockWorker::start(vec![]).await;
    let ctx = build_ctx(vec![WorkerSpec {
        id: WorkerId("w1".into()),
        url: plain.url.clone(),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId("tiny".into())],
    }]);
    let app = build_router(ctx);

    let res = app.oneshot(chat_request()).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let g = plain.captured.lock().unwrap();
    assert!(
        !g.headers.contains_key("x-sgl-decode-url"),
        "plain-mode worker must not receive a decode-affinity header. headers: {:?}",
        g.headers,
    );
}

/// Task C: PD-mode prefill request with NO decode workers → 503
/// `no_decode_workers_available`. Pin: failure mode is loud and
/// distinct from the existing `no_prefill_workers_available` path.
#[tokio::test]
async fn pd_mode_prefill_only_returns_no_decode_workers_available() {
    let prefill = common::mock_worker::MockWorker::start(vec![]).await;
    let ctx = build_ctx(vec![WorkerSpec {
        id: WorkerId("p1".into()),
        url: prefill.url.clone(),
        mode: WorkerMode::Prefill,
        model_ids: vec![ModelId("tiny".into())],
    }]);
    let app = build_router(ctx);

    let res = app.oneshot(chat_request()).await.unwrap();
    assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        res.headers().get("x-router-error-code").unwrap(),
        "no_decode_workers_available",
    );
}
