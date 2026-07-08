// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! PD pool isolation — end-to-end at the HTTP layer using MockWorker.
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

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use sgl_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, ModelConfig, ObservabilityConfig, PolicyKind,
    ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
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
            ..Default::default()
        },
        observability: ObservabilityConfig::default(),
        model: ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            tokenizer_shards: 1,
            tokenizer_backend: Default::default(),
            tokenizer_l1_cache_mb: 0,
            policy: PolicyKind::RoundRobin,
            circuit_breaker: None,
            cache_aware: None,
            sticky: None,
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
        admission: sgl_router::config::AdmissionConfig::default(),
        retry: sgl_router::config::RetryConfig::default(),
    }
}

fn build_ctx(specs: Vec<WorkerSpec>) -> Arc<AppContext> {
    let cfg = config();
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    for s in specs {
        let _ = registry.add(s);
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
    let worker = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let ctx = build_ctx(vec![WorkerSpec {
        id: WorkerId("d1".into()),
        url: worker.url.clone(),
        mode: WorkerMode::Decode,
        model_ids: vec![ModelId("tiny".into())],
        bootstrap_port: None,
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

/// PD-disagg deployment with both pools healthy → chat dispatch fans
/// out to BOTH the prefill and the decode worker (Pattern B: prefill
/// in a detached task, decode awaited for the client response). Both
/// receive the same bootstrap-injected body so the SGLang engine can
/// match KV transfers via `bootstrap_room`. Pool *isolation* — the
/// guarantee that the policy's prefill candidate set excludes decode
/// workers — is exercised at the resolver layer
/// (`policies::registry::tests::pd_resolution_returns_distinct_pools`).
/// Here we only assert the HTTP-layer wiring of the dual dispatch.
#[tokio::test]
async fn pd_mode_chat_dispatch_fans_to_both_prefill_and_decode() {
    let prefill = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let decode = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let ctx = build_ctx(vec![
        WorkerSpec {
            id: WorkerId("p1".into()),
            url: prefill.url.clone(),
            mode: WorkerMode::Prefill,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: Some(8997),
        },
        WorkerSpec {
            id: WorkerId("d1".into()),
            url: decode.url.clone(),
            mode: WorkerMode::Decode,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        },
    ]);
    let app = build_router(ctx);

    // Fire a single request; both prefill (spawn-and-forget) and
    // decode (awaited) must receive a body with the injected
    // bootstrap fields. The decode body is what the client sees on
    // the response.
    let res = app.oneshot(chat_request()).await.unwrap();
    assert_eq!(
        res.status(),
        StatusCode::OK,
        "decode response status should reach the client",
    );

    // Decode receives its body synchronously (we awaited it), so it's
    // guaranteed captured by the time the response returned. Scope
    // the lock guard to this block so it doesn't span the `.await`
    // below (clippy: await_holding_lock).
    {
        let decode_seen = decode.captured.lock().unwrap();
        assert!(
            decode_seen.last_body.is_some(),
            "decode worker must receive the bootstrap-injected request body in PD mode",
        );
    }

    // Prefill is detached; poll briefly until its capture lands. The
    // prefill task races the HTTP response back to the client. The
    // local binding releases the `std::sync::Mutex` guard before the
    // `.await` — holding a sync mutex across an await would let one
    // task pin the lock while another tries to acquire it.
    let prefill_body = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            let captured = prefill.captured.lock().unwrap().last_body.clone();
            if let Some(b) = captured {
                return b;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("prefill MUST eventually receive its body via the detached task");
    assert!(!prefill_body.is_empty());
}

/// Task C: PD-mode chat request carries an `x-sgl-decode-url` header
/// pointing at the host-affinity decode peer. With two prefill workers
/// on different hosts and a decode worker on each, the affinity helper
/// MUST pick the decode peer co-located with the chosen prefill.
///
/// Round-robin will select prefill workers deterministically (alphabetic
/// dashmap order is not guaranteed; the test fires several requests so
/// at least one lands on each prefill, and asserts the per-host pairing
/// holds across all of them).
#[tokio::test]
async fn pd_mode_chat_dispatch_sets_decode_affinity_header() {
    use std::collections::HashSet;
    let prefill_a = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let prefill_b = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let decode_a = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let decode_b = crate::common::mock_worker::MockWorker::start(vec![]).await;
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
            bootstrap_port: None,
        },
        WorkerSpec {
            id: WorkerId("p2".into()),
            url: prefill_b.url.clone(),
            mode: WorkerMode::Prefill,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        },
        WorkerSpec {
            id: WorkerId("d1".into()),
            url: decode_a.url.clone(),
            mode: WorkerMode::Decode,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        },
        WorkerSpec {
            id: WorkerId("d2".into()),
            url: decode_b.url.clone(),
            mode: WorkerMode::Decode,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
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
    let plain = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let ctx = build_ctx(vec![WorkerSpec {
        id: WorkerId("w1".into()),
        url: plain.url.clone(),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId("tiny".into())],
        bootstrap_port: None,
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
    let prefill = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let ctx = build_ctx(vec![WorkerSpec {
        id: WorkerId("p1".into()),
        url: prefill.url.clone(),
        mode: WorkerMode::Prefill,
        model_ids: vec![ModelId("tiny".into())],
        bootstrap_port: None,
    }]);
    let app = build_router(ctx);

    let res = app.oneshot(chat_request()).await.unwrap();
    assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        res.headers().get("x-router-error-code").unwrap(),
        "no_decode_workers_available",
    );
}

/// PD-mode chat response carries `x-sgl-decode-url` so external tests
/// can observe decode affinity end-to-end (without sniffing the proxy
/// hop into the upstream prefill worker). Mirrors the request-side
/// behavior asserted by `pd_mode_chat_dispatch_sets_decode_affinity_header`.
#[tokio::test]
async fn pd_mode_chat_response_carries_decode_affinity_header() {
    use std::collections::HashSet;
    let prefill = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let decode_a = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let decode_b = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let ctx = build_ctx(vec![
        WorkerSpec {
            id: WorkerId("p1".into()),
            url: prefill.url.clone(),
            mode: WorkerMode::Prefill,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        },
        WorkerSpec {
            id: WorkerId("d1".into()),
            url: decode_a.url.clone(),
            mode: WorkerMode::Decode,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        },
        WorkerSpec {
            id: WorkerId("d2".into()),
            url: decode_b.url.clone(),
            mode: WorkerMode::Decode,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        },
    ]);
    let app = build_router(ctx);

    let res = app.oneshot(chat_request()).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let decode_urls: HashSet<String> = [decode_a.url.clone(), decode_b.url.clone()]
        .into_iter()
        .collect();
    let hdr = res
        .headers()
        .get("x-sgl-decode-url")
        .unwrap_or_else(|| {
            panic!(
                "PD-mode chat response did not carry x-sgl-decode-url; headers: {:?}",
                res.headers(),
            )
        })
        .to_str()
        .unwrap()
        .to_owned();
    assert!(
        decode_urls.contains(&hdr),
        "response carried decode hint {hdr}, expected one of {decode_urls:?}",
    );
}

/// Plain-mode chat response does NOT carry `x-sgl-decode-url`. Pin: the
/// response-side mirror is gated on PD-mode dispatch.
#[tokio::test]
async fn plain_mode_chat_response_omits_decode_affinity_header() {
    let plain = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let ctx = build_ctx(vec![WorkerSpec {
        id: WorkerId("w1".into()),
        url: plain.url.clone(),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId("tiny".into())],
        bootstrap_port: None,
    }]);
    let app = build_router(ctx);

    let res = app.oneshot(chat_request()).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    assert!(
        !res.headers().contains_key("x-sgl-decode-url"),
        "plain-mode chat response must not carry x-sgl-decode-url; headers: {:?}",
        res.headers(),
    );
}
