// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Load Monitor (pull mode) at the HTTP layer with MockWorkers:
//!
//! * A worker becomes routable only once its first `GET /v1/loads` report
//!   has landed (`fresh`); a worker that cannot be polled is excluded and,
//!   alone, yields 503 `no_fresh_worker_load`.
//! * In PD mode both pools are freshness-gated, and `--decode-policy`
//!   selects the decode peer through a real policy.
//! * `/metrics` exposes the pull counters and reported load gauges.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use sgl_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, LoadMonitorConfig as LoadMonitorSettings,
    ModelConfig, ObservabilityConfig, PolicyKind, ProxyConfig, ServerConfig,
    StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::load_monitor::{Freshness, LoadMonitor, LoadMonitorConfig};
use sgl_router::policies::factory::build_registry_with_defaults;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tower::ServiceExt;

fn config(policy: PolicyKind, decode_policy: Option<PolicyKind>) -> Config {
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
            policy,
            circuit_breaker: None,
            cache_aware: None,
            decode_policy,
            sticky: None,
            max_output_tokens: None,
            sampling_overrides: Default::default(),
            forward_input_ids: true,
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig::default(),
        load_monitor: LoadMonitorSettings {
            enabled: true,
            ..LoadMonitorSettings::default()
        },
        active_load: ActiveLoadConfig::default(),
        admission: sgl_router::config::AdmissionConfig::default(),
        retry: sgl_router::config::RetryConfig::default(),
    }
}

/// Build a context with the Load Monitor enabled and every spec'd worker
/// registered AND tracked (what the worker manager does in production).
fn build_ctx(
    specs: Vec<WorkerSpec>,
    policy: PolicyKind,
    decode_policy: Option<PolicyKind>,
) -> (Arc<AppContext>, Arc<LoadMonitor>) {
    let cfg = config(policy, decode_policy);
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    let lm = LoadMonitor::new(LoadMonitorConfig {
        report_interval: Duration::from_millis(50),
        stale_after: Duration::from_millis(2000),
        request_timeout: Duration::from_millis(500),
    });
    for s in specs {
        lm.track(&s.url);
        let _ = registry.add(s);
    }
    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    let mut ctx = AppContext::new(cfg, tokenizers, proxy, registry, policies);
    lm.attach_metrics(Arc::clone(&ctx.metrics));
    ctx.load_monitor = Some(Arc::clone(&lm));
    (Arc::new(ctx), lm)
}

fn spec(id: &str, url: &str, mode: WorkerMode, port: Option<u16>) -> WorkerSpec {
    WorkerSpec {
        id: WorkerId(id.into()),
        url: url.into(),
        mode,
        model_ids: vec![ModelId("tiny".into())],
        bootstrap_port: port,
        transfer_group: None,
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

async fn wait_for(lm: &LoadMonitor, url: &str, want: Freshness) {
    let start = Instant::now();
    while lm.snapshot(url).freshness != want {
        assert!(
            start.elapsed() < Duration::from_secs(3),
            "{url} did not reach {want:?}; got {:?}",
            lm.snapshot(url)
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

fn error_code(res: &axum::http::Response<Body>) -> Option<&str> {
    res.headers()
        .get("x-router-error-code")
        .and_then(|v| v.to_str().ok())
}

/// Plain mode: a worker is routable once its first report is fresh; a
/// worker without `/v1/loads` (404 → unreachable) is excluded.
#[tokio::test]
async fn plain_mode_routes_only_to_fresh_workers() {
    let good = crate::common::mock_worker::MockWorker::start(vec![]).await;
    // `start_hanging` serves /v1/chat/completions + /server_info only —
    // no /v1/loads — so the monitor marks it unreachable.
    let no_loads =
        crate::common::mock_worker::MockWorker::start_hanging(Duration::from_millis(1)).await;
    let (ctx, lm) = build_ctx(
        vec![
            spec("good", &good.url, WorkerMode::Plain, None),
            spec("noloads", &no_loads.url, WorkerMode::Plain, None),
        ],
        PolicyKind::LoadBased,
        None,
    );
    wait_for(&lm, &good.url, Freshness::Fresh).await;
    wait_for(&lm, &no_loads.url, Freshness::Unreachable).await;
    let app = build_router(ctx);

    // Every request lands on the fresh worker, never the unreachable one.
    for _ in 0..5 {
        let res = app.clone().oneshot(chat_request()).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }
    assert!(good.captured.lock().unwrap().last_body.is_some());
    assert!(no_loads.captured.lock().unwrap().last_body.is_none());

    // Pre-deduction is visible until the next pull resets it.
    let snap = lm.snapshot(&good.url);
    assert!(snap.sequence >= 1);

    // Metrics: pull counters + reported gauges are exposed.
    let metrics = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let text = String::from_utf8(
        BodyExt::collect(metrics.into_body())
            .await
            .unwrap()
            .to_bytes()
            .to_vec(),
    )
    .unwrap();
    assert!(
        text.contains(&format!(
            "sgl_router_load_pulls_total{{worker_url=\"{}\",outcome=\"ok\"}}",
            good.url
        )),
        "missing ok pull counter:\n{text}"
    );
    assert!(
        text.contains(&format!(
            "sgl_router_load_pulls_total{{worker_url=\"{}\",outcome=\"unreachable\"}}",
            no_loads.url
        )),
        "missing unreachable pull counter:\n{text}"
    );
    assert!(
        text.contains(&format!(
            "sgl_router_reported_load{{worker_url=\"{}\",kind=\"free_tokens\"}} 4096",
            good.url
        )),
        "missing reported free_tokens gauge:\n{text}"
    );
}

/// No fresh worker at all → 503 `no_fresh_worker_load`, not a dispatch
/// to a worker whose load the router cannot see.
#[tokio::test]
async fn no_fresh_worker_yields_503_no_fresh_worker_load() {
    let no_loads =
        crate::common::mock_worker::MockWorker::start_hanging(Duration::from_millis(1)).await;
    let (ctx, lm) = build_ctx(
        vec![spec("noloads", &no_loads.url, WorkerMode::Plain, None)],
        PolicyKind::RoundRobin,
        None,
    );
    wait_for(&lm, &no_loads.url, Freshness::Unreachable).await;
    let app = build_router(ctx);
    let res = app.oneshot(chat_request()).await.unwrap();
    assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(error_code(&res), Some("no_fresh_worker_load"));
    assert!(no_loads.captured.lock().unwrap().last_body.is_none());
}

/// PD mode with `--decode-policy load_based`: the decode peer is chosen by
/// the policy over the fresh decode pool (not by host affinity), and a
/// decode worker without a fresh report is never chosen.
#[tokio::test]
async fn pd_mode_decode_policy_selects_from_fresh_decode_pool() {
    let prefill = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let decode_fresh = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let decode_stale =
        crate::common::mock_worker::MockWorker::start_hanging(Duration::from_millis(1)).await;
    let (ctx, lm) = build_ctx(
        vec![
            spec("p1", &prefill.url, WorkerMode::Prefill, Some(8997)),
            spec("d-fresh", &decode_fresh.url, WorkerMode::Decode, None),
            spec("d-stale", &decode_stale.url, WorkerMode::Decode, None),
        ],
        PolicyKind::RoundRobin,
        Some(PolicyKind::LoadBased),
    );
    wait_for(&lm, &prefill.url, Freshness::Fresh).await;
    wait_for(&lm, &decode_fresh.url, Freshness::Fresh).await;
    wait_for(&lm, &decode_stale.url, Freshness::Unreachable).await;
    let app = build_router(ctx);

    for _ in 0..3 {
        let res = app.clone().oneshot(chat_request()).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(
            res.headers()
                .get("x-sgl-decode-url")
                .and_then(|v| v.to_str().ok()),
            Some(decode_fresh.url.as_str()),
        );
    }
    assert!(decode_stale.captured.lock().unwrap().last_body.is_none());
}

/// PD mode: the prefill pool is fresh but no decode worker has a fresh
/// report → 503 `no_fresh_worker_load` (role decode), and nothing is
/// dispatched to either side.
#[tokio::test]
async fn pd_mode_no_fresh_decode_yields_503_and_no_dispatch() {
    let prefill = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let decode =
        crate::common::mock_worker::MockWorker::start_hanging(Duration::from_millis(1)).await;
    let (ctx, lm) = build_ctx(
        vec![
            spec("p1", &prefill.url, WorkerMode::Prefill, Some(8997)),
            spec("d1", &decode.url, WorkerMode::Decode, None),
        ],
        PolicyKind::RoundRobin,
        None,
    );
    wait_for(&lm, &prefill.url, Freshness::Fresh).await;
    wait_for(&lm, &decode.url, Freshness::Unreachable).await;
    let app = build_router(ctx);
    let res = app.oneshot(chat_request()).await.unwrap();
    assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(error_code(&res), Some("no_fresh_worker_load"));
    // Give any stray dispatch a moment to land, then assert none did.
    tokio::time::sleep(Duration::from_millis(30)).await;
    assert!(prefill.captured.lock().unwrap().last_body.is_none());
    assert!(decode.captured.lock().unwrap().last_body.is_none());
}
