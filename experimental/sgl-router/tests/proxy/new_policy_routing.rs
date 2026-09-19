// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use sgl_router::buckets_reorg::{Bucket, BucketResolver, EngineGroup, TokenLimits};
use sgl_router::config::PolicyKind;
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::PolicyRegistry;
use sgl_router::policies_reorg::admission::{Admission, InFlightLimit};
use sgl_router::policies_reorg::power_of_two::PowerOfTwoPolicy;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::{build_router, build_router_with_new_policy};
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;
use tower::ServiceExt;

use super::common::mock_worker::MockWorker;

fn context(workers: &[(&str, &MockWorker, WorkerMode)]) -> Arc<AppContext> {
    let mut config = super::common::cache_aware_fixture::config();
    config.model.id = "tiny".into();
    config.model.policy = PolicyKind::RoundRobin;
    config.model.cache_aware = None;
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&config).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    for (id, worker, mode) in workers {
        registry
            .add(WorkerSpec {
                id: WorkerId((*id).into()),
                url: worker.url.clone(),
                mode: *mode,
                model_ids: vec![ModelId("tiny".into())],
                bootstrap_port: (*mode == WorkerMode::Prefill).then_some(8997),
            })
            .unwrap();
    }
    Arc::new(AppContext::new(
        config,
        tokenizers,
        Arc::new(Proxy::new(Duration::from_secs(5)).unwrap()),
        registry,
        Arc::new(PolicyRegistry::default()),
    ))
}

fn group(ids: &[&str]) -> EngineGroup {
    EngineGroup {
        worker_ids: Some(ids.iter().map(|id| WorkerId((*id).into())).collect()),
        ..EngineGroup::new(Arc::new(PowerOfTwoPolicy::default()))
    }
}

fn request(body: Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap()
}

fn chat(stream: bool) -> Value {
    // The tiny byte-level fixture produces two input tokens for "hi".
    json!({"model": "tiny", "messages": [{"role": "user", "content": "hi"}], "stream": stream})
}

#[tokio::test]
async fn new_policy_selects_lower_load_and_forwards_json_and_sse() {
    let busy = MockWorker::start(vec![]).await;
    let idle = MockWorker::start(vec![
        "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n",
        "data: [DONE]\n\n",
    ])
    .await;
    let ctx = context(&[
        ("busy", &busy, WorkerMode::Plain),
        ("idle", &idle, WorkerMode::Plain),
    ]);
    let busy_worker = ctx.registry.get(&WorkerId("busy".into())).unwrap();
    let _busy = busy_worker.load_guard();
    let resolver = BucketResolver::new(
        ctx.registry.clone(),
        vec![Bucket {
            id: "two-tokens".into(),
            max_context_tokens: Some(2),
            plain: Some(EngineGroup {
                limits: TokenLimits {
                    min: Some(2),
                    max: Some(2),
                },
                ..group(&["busy", "idle"])
            }),
            ..Default::default()
        }],
    );
    let app = build_router_with_new_policy(ctx.clone(), resolver);
    for stream in [false, true] {
        let response = app.clone().oneshot(request(chat(stream))).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        if stream {
            assert!(String::from_utf8_lossy(&body).contains("[DONE]"));
        } else {
            assert_eq!(
                serde_json::from_slice::<Value>(&body).unwrap()["choices"][0]["message"]["content"],
                "ok"
            );
        }
        assert_eq!(ctx.active_load.inflight_count(), 0);
        assert_eq!(
            ctx.registry
                .get(&WorkerId("idle".into()))
                .unwrap()
                .active_load(),
            0
        );
    }
    assert!(busy.captured.lock().unwrap().last_body.is_none());
    assert!(idle.captured.lock().unwrap().last_body.is_some());
    assert_eq!(ctx.inflight_http.count(), 0);
    assert!(ctx
        .metrics
        .render()
        .contains("route=\"/v1/chat/completions\""));

    // No legacy policy was installed; the default endpoint still uses that path.
    let response = build_router(ctx)
        .oneshot(request(chat(false)))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn new_policy_pd_selects_different_buckets_and_reuses_bootstrap_forwarding() {
    let prefill = MockWorker::start(vec![]).await;
    let short_decode = MockWorker::start(vec![]).await;
    let long_decode = MockWorker::start(vec![]).await;
    let ctx = context(&[
        ("p", &prefill, WorkerMode::Prefill),
        ("short-d", &short_decode, WorkerMode::Decode),
        ("long-d", &long_decode, WorkerMode::Decode),
    ]);
    let resolver = BucketResolver::new(
        ctx.registry.clone(),
        vec![
            Bucket {
                id: "short".into(),
                max_context_tokens: Some(2),
                prefill: Some(group(&["p"])),
                decode: Some(group(&["short-d"])),
                ..Default::default()
            },
            Bucket {
                id: "long".into(),
                decode: Some(EngineGroup {
                    limits: TokenLimits {
                        min: Some(10),
                        max: None,
                    },
                    ..group(&["long-d"])
                }),
                ..Default::default()
            },
        ],
    );
    let app = build_router_with_new_policy(ctx.clone(), resolver);
    let mut body = chat(false);
    body["max_completion_tokens"] = json!(8);
    let response = app.oneshot(request(body)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["x-sgl-decode-url"], long_decode.url);
    response.into_body().collect().await.unwrap();
    tokio::time::timeout(Duration::from_secs(2), async {
        while prefill.captured.lock().unwrap().last_body.is_none()
            || ctx.active_load.inflight_count() != 0
        {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    let prefill_body: Value =
        serde_json::from_slice(prefill.captured.lock().unwrap().last_body.as_ref().unwrap())
            .unwrap();
    let decode_body: Value = serde_json::from_slice(
        long_decode
            .captured
            .lock()
            .unwrap()
            .last_body
            .as_ref()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        prefill_body["bootstrap_room"],
        decode_body["bootstrap_room"]
    );
    assert!(prefill_body["bootstrap_room"].is_u64());
    assert_eq!(prefill_body["bootstrap_port"], 8997);
    assert!(short_decode.captured.lock().unwrap().last_body.is_none());
    assert_eq!(
        ctx.registry
            .get(&WorkerId("p".into()))
            .unwrap()
            .active_load(),
        0
    );
    assert_eq!(
        ctx.registry
            .get(&WorkerId("long-d".into()))
            .unwrap()
            .active_load(),
        0
    );
}

#[tokio::test]
async fn new_policy_admission_rejection_returns_503_without_dispatch() {
    let worker = MockWorker::start(vec![]).await;
    let ctx = context(&[("w", &worker, WorkerMode::Plain)]);
    let resolver = BucketResolver::new(
        ctx.registry.clone(),
        vec![Bucket {
            id: "full".into(),
            plain: Some(EngineGroup::new(Arc::new(PowerOfTwoPolicy {
                admission: Admission::before(InFlightLimit(0)),
            }))),
            ..Default::default()
        }],
    );
    let response = build_router_with_new_policy(ctx.clone(), resolver)
        .oneshot(request(chat(false)))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        response.headers()["x-router-error-code"],
        "policy_selection_failed"
    );
    assert!(worker.captured.lock().unwrap().last_body.is_none());
    assert_eq!(ctx.active_load.inflight_count(), 0);
}

#[tokio::test]
async fn new_policy_missing_decode_group_never_dispatches_prefill() {
    let prefill = MockWorker::start(vec![]).await;
    let decode = MockWorker::start(vec![]).await;
    let ctx = context(&[
        ("p", &prefill, WorkerMode::Prefill),
        ("d", &decode, WorkerMode::Decode),
    ]);
    let resolver = BucketResolver::new(
        ctx.registry.clone(),
        vec![Bucket {
            id: "prefill-only".into(),
            prefill: Some(group(&["p"])),
            ..Default::default()
        }],
    );
    let response = build_router_with_new_policy(ctx.clone(), resolver)
        .oneshot(request(chat(false)))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        response.headers()["x-router-error-code"],
        "no_decode_workers_available"
    );
    assert!(prefill.captured.lock().unwrap().last_body.is_none());
    assert!(decode.captured.lock().unwrap().last_body.is_none());
    assert_eq!(ctx.active_load.inflight_count(), 0);
}

#[tokio::test]
async fn new_policy_invalid_requests_fail_before_dispatch() {
    let worker = MockWorker::start(vec![]).await;
    let ctx = context(&[("w", &worker, WorkerMode::Plain)]);
    let resolver = BucketResolver::new(
        ctx.registry.clone(),
        vec![Bucket {
            id: "default".into(),
            plain: Some(group(&["w"])),
            ..Default::default()
        }],
    );
    let app = build_router_with_new_policy(ctx.clone(), resolver);
    let mut unknown = chat(false);
    unknown["model"] = json!("unknown");
    let mut overflow = chat(false);
    overflow["max_tokens"] = json!(u64::MAX);
    for (body, status, code) in [
        (json!({}), StatusCode::BAD_REQUEST, "bad_request"),
        (unknown, StatusCode::NOT_FOUND, "model_not_found"),
        (overflow, StatusCode::BAD_REQUEST, "bad_request"),
    ] {
        let response = app.clone().oneshot(request(body)).await.unwrap();
        assert_eq!(response.status(), status);
        assert_eq!(response.headers()["x-router-error-code"], code);
        response.into_body().collect().await.unwrap();
    }
    assert!(worker.captured.lock().unwrap().last_body.is_none());
    assert_eq!(ctx.active_load.inflight_count(), 0);
    assert_eq!(ctx.inflight_http.count(), 0);
}

#[tokio::test]
async fn new_policy_slo_headers_select_pd_groups_independently() {
    use sgl_router::buckets_reorg::SloPreference;

    let fast_p = MockWorker::start(vec![]).await;
    let slow_p = MockWorker::start(vec![]).await;
    let fast_d = MockWorker::start(vec![]).await;
    let slow_d = MockWorker::start(vec![]).await;
    let ctx = context(&[
        ("fast-p", &fast_p, WorkerMode::Prefill),
        ("slow-p", &slow_p, WorkerMode::Prefill),
        ("fast-d", &fast_d, WorkerMode::Decode),
        ("slow-d", &slow_d, WorkerMode::Decode),
    ]);
    let mut resolver = BucketResolver::new(
        ctx.registry.clone(),
        vec![
            Bucket {
                id: "fast-prefill".into(),
                ttft_ms: Some(50),
                tokens_per_second: Some(10.0),
                prefill: Some(group(&["fast-p"])),
                decode: Some(group(&["slow-d"])),
                ..Default::default()
            },
            Bucket {
                id: "fast-decode".into(),
                ttft_ms: Some(100),
                tokens_per_second: Some(100.0),
                prefill: Some(group(&["slow-p"])),
                decode: Some(group(&["fast-d"])),
                ..Default::default()
            },
        ],
    );
    resolver.prefill_slo = SloPreference::SloFirst;
    resolver.decode_slo = SloPreference::SloFirst;
    let app = build_router_with_new_policy(ctx, resolver);
    for (ttft, tps) in [("0", "100"), ("50", "NaN"), ("50", "0")] {
        let mut req = request(chat(false));
        req.headers_mut()
            .insert("x-sgl-ttft-slo-ms", ttft.parse().unwrap());
        req.headers_mut()
            .insert("x-sgl-tps-slo", tps.parse().unwrap());
        let response = app.clone().oneshot(req).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
    for worker in [&fast_p, &slow_p, &fast_d, &slow_d] {
        assert!(worker.captured.lock().unwrap().last_body.is_none());
    }
    let mut req = request(chat(false));
    req.headers_mut()
        .insert("x-sgl-ttft-slo-ms", "50".parse().unwrap());
    req.headers_mut()
        .insert("x-sgl-tps-slo", "100".parse().unwrap());
    let response = app.oneshot(req).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["x-sgl-decode-url"], fast_d.url);
    response.into_body().collect().await.unwrap();
    tokio::time::timeout(Duration::from_secs(2), async {
        while fast_p.captured.lock().unwrap().last_body.is_none() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    assert!(slow_p.captured.lock().unwrap().last_body.is_none());
    assert!(slow_d.captured.lock().unwrap().last_body.is_none());
}
