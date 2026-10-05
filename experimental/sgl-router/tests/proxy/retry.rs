// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! `--retry-max-attempts`: a dispatch that fails before any response reaches
//! the client is retried on a worker the request has not tried yet, under both
//! legacy and reorg routing.

use crate::common::mock_worker::MockWorker;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use sgl_router::config::{
    Config, DiscoveryBackend, InflightLoadConfig, ModelConfig, ObservabilityConfig, PolicyKind,
    ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry_with_defaults;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::{AppContext, ChatRouting};
use sgl_router::state::load_monitor::router_inflight_load::{
    RouterInflightLoadRegistry, SystemTimeClock,
};
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

fn config(max_attempts: u32) -> Config {
    Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
            ..Default::default()
        },
        observability: ObservabilityConfig::default(),
        model: ModelConfig {
            id: "tiny".into(),
            tokenizer_path: Some("tests/fixtures/tiny_tokenizer.json".into()),
            disable_input_ids_forwarding: false,
            tokenizer: Default::default(),
            policy: PolicyKind::RoundRobin,
            decode_policy: Default::default(),
            dp_aware: false,
            bucket_config: None,
            reorg_buckets: None,
            reorg_admission: Default::default(),
            circuit_breaker: None,
            cache_aware: None,
            sticky: None,
            affinity: None,
            fused: None,
            eligibility: None,
            sampling_overrides: Default::default(),
            default_chat_template_kwargs: Default::default(),
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig {
            max_attempts: NonZeroU32::new(max_attempts).unwrap(),
            ..Default::default()
        },
        router_inflight_load: InflightLoadConfig::default(),
    }
}

fn spec(id: &str, url: &str, mode: WorkerMode) -> WorkerSpec {
    WorkerSpec {
        id: WorkerId(id.into()),
        url: url.into(),
        mode,
        model_ids: vec![ModelId("tiny".into())],
        bootstrap_port: (mode == WorkerMode::Prefill).then_some(8997),
        ..Default::default()
    }
}

/// A router over `workers`, with legacy round-robin or reorg power-of-two routing.
fn router_ctx(
    workers: &[(&str, &str, WorkerMode)],
    max_attempts: u32,
    reorg: bool,
) -> Arc<AppContext> {
    let mut cfg = config(max_attempts);
    if reorg {
        cfg.model.policy = PolicyKind::PowerOfTwo;
    }
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    for (id, url, mode) in workers {
        registry.add(spec(id, url, *mode)).unwrap();
    }
    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    let mut ctx = AppContext::new(cfg, tokenizers, proxy, registry, policies);
    if reorg {
        let state = sgl_router::state::kv_events::KvEventIndex::new();
        let (resolver, _) =
            sgl_router::policies_reorg::factory::build_resolver(&ctx.config.model, &state, None)
                .unwrap();
        ctx.chat_routing = ChatRouting::Reorg([(ModelId("tiny".into()), resolver)].into());
    }
    Arc::new(ctx)
}

fn chat(stream: bool) -> Request<Body> {
    chat_with(json!({"stream": stream}))
}

/// A chat request carrying `fields` on top of the model and one message.
fn chat_with(mut fields: Value) -> Request<Body> {
    fields["model"] = json!("tiny");
    fields["messages"] = json!([{"role": "user", "content": "hi"}]);
    Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(fields.to_string()))
        .unwrap()
}

fn hit(worker: &MockWorker) -> bool {
    worker.captured.lock().unwrap().last_body.is_some()
}

fn retries(ctx: &AppContext) -> u64 {
    let metrics = ctx.metrics.render();
    metrics
        .lines()
        .find_map(|line| line.strip_prefix(r#"sgl_router_retries_total{model_id="tiny"} "#))
        .map_or(0, |count| count.parse().unwrap())
}

/// A URL nothing listens on, so a dispatch to it fails to connect.
async fn dead_url() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    format!("http://{}", listener.local_addr().unwrap())
}

fn rejected() -> Value {
    json!({"error": "rejected"})
}

/// Requests until the failing worker has been picked; every one still succeeds.
#[tokio::test]
async fn failed_attempt_succeeds_on_another_worker() {
    for reorg in [false, true] {
        for stream in [false, true] {
            let ok = MockWorker::start(vec!["data: {}\n\n", "data: [DONE]\n\n"]).await;
            let unavailable =
                MockWorker::start_returning_error(StatusCode::SERVICE_UNAVAILABLE, rejected())
                    .await;
            let dead = dead_url().await;
            let ctx = router_ctx(
                &[
                    ("ok", &ok.url, WorkerMode::Plain),
                    ("unavailable", &unavailable.url, WorkerMode::Plain),
                    ("dead", &dead, WorkerMode::Plain),
                ],
                3,
                reorg,
            );
            let app = build_router(ctx.clone());
            // Both failing workers are picked first at least once, so two retries show up.
            for _ in 0..50 {
                let response = app.clone().oneshot(chat(stream)).await.unwrap();
                assert_eq!(response.status(), StatusCode::OK, "reorg={reorg}");
                if hit(&unavailable) && retries(&ctx) >= 2 {
                    break;
                }
            }
            assert!(hit(&unavailable), "reorg={reorg}: 503 worker never picked");
            assert!(
                retries(&ctx) >= 2,
                "reorg={reorg}: retries {}",
                retries(&ctx)
            );
        }
    }
}

/// With every worker failing, each is tried once and the client sees the last failure.
#[tokio::test]
async fn exhausted_workers_return_the_last_failure() {
    for reorg in [false, true] {
        for stream in [false, true] {
            let a = MockWorker::start_returning_error(StatusCode::SERVICE_UNAVAILABLE, rejected())
                .await;
            let b = MockWorker::start_returning_error(StatusCode::SERVICE_UNAVAILABLE, rejected())
                .await;
            let ctx = router_ctx(
                &[
                    ("a", &a.url, WorkerMode::Plain),
                    ("b", &b.url, WorkerMode::Plain),
                ],
                3,
                reorg,
            );
            let response = build_router(ctx.clone())
                .oneshot(chat(stream))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
            assert!(hit(&a) && hit(&b), "reorg={reorg}: both workers tried");
            assert_eq!(retries(&ctx), 1, "reorg={reorg}");
        }
    }
}

/// Client errors and the default single attempt are never retried.
#[tokio::test]
async fn client_errors_and_the_default_are_not_retried() {
    for reorg in [false, true] {
        for (status, max_attempts) in [
            (StatusCode::BAD_REQUEST, 3),
            (StatusCode::SERVICE_UNAVAILABLE, 1),
        ] {
            let a = MockWorker::start_returning_error(status, rejected()).await;
            let b = MockWorker::start_returning_error(status, rejected()).await;
            let ctx = router_ctx(
                &[
                    ("a", &a.url, WorkerMode::Plain),
                    ("b", &b.url, WorkerMode::Plain),
                ],
                max_attempts,
                reorg,
            );
            let response = build_router(ctx.clone())
                .oneshot(chat(false))
                .await
                .unwrap();
            assert_eq!(response.status(), status);
            assert_eq!(
                [hit(&a), hit(&b)].iter().filter(|hit| **hit).count(),
                1,
                "reorg={reorg} status={status}"
            );
            assert_eq!(retries(&ctx), 0);
        }
    }
}

/// In PD mode, a failed prefill or decode is excluded and the request is
/// re-paired; the retried pair shares one fresh bootstrap room.
#[tokio::test]
async fn pd_retry_repairs_around_the_failed_side() {
    for reorg in [false, true] {
        // A failing prefill answers at once; decode waits for KV, as an engine would.
        let failing_prefill =
            MockWorker::start_returning_error(StatusCode::INTERNAL_SERVER_ERROR, rejected()).await;
        let prefill = MockWorker::start(vec![]).await;
        let decode = MockWorker::start_hanging(Duration::from_millis(100)).await;
        let ctx = router_ctx(
            &[
                ("p-failing", &failing_prefill.url, WorkerMode::Prefill),
                ("p", &prefill.url, WorkerMode::Prefill),
                ("d", &decode.url, WorkerMode::Decode),
            ],
            3,
            reorg,
        );
        let app = build_router(ctx.clone());
        for _ in 0..50 {
            let response = app.clone().oneshot(chat(false)).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK, "reorg={reorg}");
            if hit(&failing_prefill) {
                break;
            }
        }
        assert!(
            hit(&failing_prefill),
            "reorg={reorg}: failing prefill never picked"
        );
        assert!(retries(&ctx) >= 1);
        // The last request was the retried one: its decode carries the new prefill's room.
        let room = |body: Value| body["bootstrap_room"].clone();
        assert_eq!(
            room(prefill.captured_json().await),
            room(decode.captured_json().await)
        );

        // A failing decode is excluded the same way.
        let prefill = MockWorker::start(vec![]).await;
        let failing_decode =
            MockWorker::start_returning_error(StatusCode::SERVICE_UNAVAILABLE, rejected()).await;
        let decode = MockWorker::start(vec![]).await;
        let ctx = router_ctx(
            &[
                ("p", &prefill.url, WorkerMode::Prefill),
                ("d-failing", &failing_decode.url, WorkerMode::Decode),
                ("d", &decode.url, WorkerMode::Decode),
            ],
            3,
            reorg,
        );
        let app = build_router(ctx.clone());
        for _ in 0..50 {
            let response = app.clone().oneshot(chat(false)).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK, "reorg={reorg}");
            if hit(&failing_decode) {
                break;
            }
        }
        assert!(
            hit(&failing_decode),
            "reorg={reorg}: failing decode never picked"
        );
        assert!(retries(&ctx) >= 1);
    }
}

/// A caller's rid may still be live on the failed pair's prefill, so the retry
/// avoids both sides; with a single prefill there is nothing left to retry on.
#[tokio::test]
async fn pd_retry_with_a_caller_rid_avoids_the_whole_pair() {
    for reorg in [false, true] {
        let prefill = MockWorker::start(vec![]).await;
        let failing_decode =
            MockWorker::start_returning_error(StatusCode::SERVICE_UNAVAILABLE, rejected()).await;
        let decode = MockWorker::start(vec![]).await;
        let ctx = router_ctx(
            &[
                ("p", &prefill.url, WorkerMode::Prefill),
                ("d-failing", &failing_decode.url, WorkerMode::Decode),
                ("d", &decode.url, WorkerMode::Decode),
            ],
            3,
            reorg,
        );
        let app = build_router(ctx.clone());
        for _ in 0..50 {
            let request = chat_with(json!({"rid": "caller-rid"}));
            let status = app.clone().oneshot(request).await.unwrap().status();
            if hit(&failing_decode) {
                assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "reorg={reorg}");
                break;
            }
            assert_eq!(status, StatusCode::OK, "reorg={reorg}");
        }
        assert!(
            hit(&failing_decode),
            "reorg={reorg}: failing decode never picked"
        );
        assert_eq!(retries(&ctx), 0, "reorg={reorg}");
    }
}

/// Retries share the request's stale deadline: none starts once it has passed.
#[tokio::test]
async fn retries_stop_at_the_request_deadline() {
    let workers = [
        MockWorker::start_hanging(Duration::from_secs(5)).await,
        MockWorker::start_hanging(Duration::from_secs(5)).await,
        MockWorker::start_hanging(Duration::from_secs(5)).await,
    ];
    let mut ctx = router_ctx(
        &[
            ("a", &workers[0].url, WorkerMode::Plain),
            ("b", &workers[1].url, WorkerMode::Plain),
            ("c", &workers[2].url, WorkerMode::Plain),
        ],
        3,
        false,
    );
    // Each attempt times out after 200 ms; the request may live 300 ms.
    let context = Arc::get_mut(&mut ctx).unwrap();
    context.proxy = Arc::new(Proxy::new(Duration::from_millis(200)).unwrap());
    context.router_inflight_load =
        RouterInflightLoadRegistry::new(Arc::new(SystemTimeClock), Duration::from_millis(300));
    let response = build_router(ctx.clone())
        .oneshot(chat(false))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    assert_eq!(workers.iter().filter(|worker| hit(worker)).count(), 2);
    assert_eq!(retries(&ctx), 1);
}
