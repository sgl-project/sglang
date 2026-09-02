// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! PD transfer groups at the HTTP layer with MockWorkers.
//!
//! * The decode peer is always taken from the chosen prefill's group, even
//!   when every worker shares a host (so host affinity alone would not
//!   separate them).
//! * A group with a prefill but no decode contributes no prefill candidate.
//! * When no group is complete the request fails with
//!   `503 no_compatible_pd_group`.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use sgl_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, LoadMonitorConfig, ModelConfig,
    ObservabilityConfig, PolicyKind, ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry_with_defaults;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;
use std::collections::HashSet;
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
        model: ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            policy: PolicyKind::RoundRobin,
            circuit_breaker: None,
            cache_aware: None,
            decode_policy: None,
            sticky: None,
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig::default(),
        load_monitor: LoadMonitorConfig::default(),
        active_load: ActiveLoadConfig::default(),
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

fn spec(id: &str, url: &str, mode: WorkerMode, group: Option<&str>) -> WorkerSpec {
    WorkerSpec {
        id: WorkerId(id.into()),
        url: url.into(),
        mode,
        model_ids: vec![ModelId("tiny".into())],
        bootstrap_port: (mode == WorkerMode::Prefill).then_some(8997),
        transfer_group: group.map(str::to_string),
    }
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

fn decode_url(res: &axum::http::Response<Body>) -> String {
    res.headers()
        .get("x-sgl-decode-url")
        .and_then(|v| v.to_str().ok())
        .expect("pd response carries x-sgl-decode-url")
        .to_string()
}

/// Two complete groups on the same host: round-robin alternates prefill
/// a/b, and the decode peer always comes from the same group.
#[tokio::test]
async fn decode_peer_is_taken_from_the_prefill_transfer_group() {
    let m = |_: &str| crate::common::mock_worker::MockWorker::start(vec![]);
    let (pa, da, pb, db) = (m("pa").await, m("da").await, m("pb").await, m("db").await);
    let ctx = build_ctx(vec![
        spec("pa", &pa.url, WorkerMode::Prefill, Some("a")),
        spec("da", &da.url, WorkerMode::Decode, Some("a")),
        spec("pb", &pb.url, WorkerMode::Prefill, Some("b")),
        spec("db", &db.url, WorkerMode::Decode, Some("b")),
    ]);
    let app = build_router(ctx);

    let mut seen_decodes = HashSet::new();
    for _ in 0..8 {
        let res = app.clone().oneshot(chat_request()).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        seen_decodes.insert(decode_url(&res));
    }
    // Both groups were used (round-robin over two prefills) …
    assert_eq!(
        seen_decodes.len(),
        2,
        "expected both decode peers to be used"
    );

    // … and each decode only ever received bodies bootstrapped by ITS OWN
    // group's prefill. Bootstrap host is identical here (all loopback), so
    // check pairing through which mocks got traffic: da/db both did, and
    // every request's decode came from the chosen prefill's group — that
    // is what the per-request assertion below pins.
    for _ in 0..8 {
        // Reset captured bodies so we can attribute one request precisely.
        for w in [&pa, &da, &pb, &db] {
            w.captured.lock().unwrap().last_body = None;
        }
        let res = app.clone().oneshot(chat_request()).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let d = decode_url(&res);
        // Give the detached prefill leg a moment to land.
        tokio::time::sleep(Duration::from_millis(20)).await;
        let pa_hit = pa.captured.lock().unwrap().last_body.is_some();
        let pb_hit = pb.captured.lock().unwrap().last_body.is_some();
        if d == da.url {
            assert!(pa_hit && !pb_hit, "decode a must pair with prefill a");
        } else if d == db.url {
            assert!(pb_hit && !pa_hit, "decode b must pair with prefill b");
        } else {
            panic!("unexpected decode url {d}");
        }
    }
}

/// Group `b` has a prefill but no decode: its prefill is never chosen.
#[tokio::test]
async fn prefill_in_group_without_decode_is_never_selected() {
    let m = |_: &str| crate::common::mock_worker::MockWorker::start(vec![]);
    let (pa, da, pb) = (m("pa").await, m("da").await, m("pb").await);
    let ctx = build_ctx(vec![
        spec("pa", &pa.url, WorkerMode::Prefill, Some("a")),
        spec("da", &da.url, WorkerMode::Decode, Some("a")),
        spec("pb", &pb.url, WorkerMode::Prefill, Some("b")),
    ]);
    let app = build_router(ctx);
    for _ in 0..6 {
        let res = app.clone().oneshot(chat_request()).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(decode_url(&res), da.url);
    }
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(pb.captured.lock().unwrap().last_body.is_none());
    assert!(pa.captured.lock().unwrap().last_body.is_some());
}

/// Prefill and decode exist but in different groups → 503
/// `no_compatible_pd_group`, nothing dispatched.
#[tokio::test]
async fn mismatched_groups_yield_503_no_compatible_pd_group() {
    let m = |_: &str| crate::common::mock_worker::MockWorker::start(vec![]);
    let (pa, db) = (m("pa").await, m("db").await);
    let ctx = build_ctx(vec![
        spec("pa", &pa.url, WorkerMode::Prefill, Some("a")),
        spec("db", &db.url, WorkerMode::Decode, Some("b")),
    ]);
    let app = build_router(ctx);
    let res = app.oneshot(chat_request()).await.unwrap();
    assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        res.headers()
            .get("x-router-error-code")
            .and_then(|v| v.to_str().ok()),
        Some("no_compatible_pd_group"),
    );
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(pa.captured.lock().unwrap().last_body.is_none());
    assert!(db.captured.lock().unwrap().last_body.is_none());
}

/// Ungrouped workers pair only with ungrouped workers: a labelled decode
/// is not a peer for an unlabelled prefill.
#[tokio::test]
async fn ungrouped_prefill_pairs_only_with_ungrouped_decode() {
    let m = |_: &str| crate::common::mock_worker::MockWorker::start(vec![]);
    let (p, d_none, d_a) = (m("p").await, m("dn").await, m("da").await);
    let ctx = build_ctx(vec![
        spec("p", &p.url, WorkerMode::Prefill, None),
        spec("dn", &d_none.url, WorkerMode::Decode, None),
        spec("da", &d_a.url, WorkerMode::Decode, Some("a")),
    ]);
    let app = build_router(ctx);
    for _ in 0..4 {
        let res = app.clone().oneshot(chat_request()).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(decode_url(&res), d_none.url);
    }
}
