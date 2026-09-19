// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use sgl_router::buckets_reorg::{Bucket, BucketGroups, BucketResolver, EngineGroup, TokenLimits};
use sgl_router::config::PolicyKind;
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::PolicyRegistry;
use sgl_router::policies_reorg::admission::{AllowAll, EngineAdmission, InFlightLimit};
use sgl_router::policies_reorg::power_of_two::PowerOfTwoPolicy;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::{AppContext, ChatRouting};
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;
use tower::ServiceExt;

use super::common::mock_worker::MockWorker;

fn context(workers: &[(&str, &MockWorker, WorkerMode)]) -> AppContext {
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
    AppContext::new(
        config,
        tokenizers,
        Arc::new(Proxy::new(Duration::from_secs(5)).unwrap()),
        registry,
        Arc::new(PolicyRegistry::default()),
    )
}

fn group(ctx: &AppContext, ids: &[&str], admission: impl EngineAdmission + 'static) -> EngineGroup {
    let mut policy = PowerOfTwoPolicy::new(ctx.engine_load.clone());
    policy.admission = Arc::new(admission);
    EngineGroup {
        worker_ids: Some(ids.iter().map(|id| WorkerId((*id).into())).collect()),
        policy: Arc::new(policy),
    }
}

fn router(mut ctx: AppContext, buckets: Vec<Bucket>) -> (Arc<AppContext>, axum::Router) {
    ctx.chat_routing =
        ChatRouting::Reorg([(ModelId("tiny".into()), BucketResolver::new(buckets))].into());
    let ctx = Arc::new(ctx);
    let app = build_router(ctx.clone());
    (ctx, app)
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
    let mut bucket = Bucket::new(
        "two-tokens",
        BucketGroups::Plain(group(&ctx, &["busy", "idle"], AllowAll)),
    );
    bucket.max_context_tokens = Some(2);
    bucket.limits = TokenLimits {
        min: Some(2),
        max: Some(2),
    };
    let (ctx, app) = router(ctx, vec![bucket]);
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
}

#[tokio::test]
async fn new_policy_pd_selects_one_bucket_and_reuses_bootstrap_forwarding() {
    let short_p = MockWorker::start(vec![]).await;
    let short_d = MockWorker::start(vec![]).await;
    let long_p = MockWorker::start(vec![]).await;
    let long_d = MockWorker::start(vec![]).await;
    let ctx = context(&[
        ("short-p", &short_p, WorkerMode::Prefill),
        ("short-d", &short_d, WorkerMode::Decode),
        ("long-p", &long_p, WorkerMode::Prefill),
        ("long-d", &long_d, WorkerMode::Decode),
    ]);
    let mut short = Bucket::new(
        "short",
        BucketGroups::Pd {
            prefill: group(&ctx, &["short-p"], AllowAll),
            decode: group(&ctx, &["short-d"], AllowAll),
        },
    );
    short.max_context_tokens = Some(2);
    let long = Bucket::new(
        "long",
        BucketGroups::Pd {
            prefill: group(&ctx, &["long-p"], AllowAll),
            decode: group(&ctx, &["long-d"], AllowAll),
        },
    );
    let (ctx, app) = router(ctx, vec![short, long]);
    let mut body = chat(false);
    body["max_completion_tokens"] = json!(8);
    let response = app.oneshot(request(body)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["x-sgl-decode-url"], long_d.url);
    response.into_body().collect().await.unwrap();
    tokio::time::timeout(Duration::from_secs(2), async {
        while long_p.captured.lock().unwrap().last_body.is_none()
            || ctx.active_load.inflight_count() != 0
        {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    let prefill_body: Value =
        serde_json::from_slice(long_p.captured.lock().unwrap().last_body.as_ref().unwrap())
            .unwrap();
    let decode_body: Value =
        serde_json::from_slice(long_d.captured.lock().unwrap().last_body.as_ref().unwrap())
            .unwrap();
    assert_eq!(
        prefill_body["bootstrap_room"],
        decode_body["bootstrap_room"]
    );
    assert!(prefill_body["bootstrap_room"].is_u64());
    assert_eq!(prefill_body["bootstrap_port"], 8997);
    assert!(short_p.captured.lock().unwrap().last_body.is_none());
    assert!(short_d.captured.lock().unwrap().last_body.is_none());
}

#[tokio::test]
async fn new_policy_admission_rejection_returns_503_without_dispatch() {
    let worker = MockWorker::start(vec![]).await;
    let ctx = context(&[("w", &worker, WorkerMode::Plain)]);
    let bucket = Bucket::new(
        "full",
        BucketGroups::Plain(group(&ctx, &["w"], InFlightLimit(0))),
    );
    let (ctx, app) = router(ctx, vec![bucket]);
    let response = app.oneshot(request(chat(false))).await.unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        response.headers()["x-router-error-code"],
        "policy_selection_failed"
    );
    assert!(worker.captured.lock().unwrap().last_body.is_none());
    assert_eq!(ctx.active_load.inflight_count(), 0);
}

#[tokio::test]
async fn new_policy_decode_admission_rejection_retries_the_entire_bucket() {
    let rejected_p = MockWorker::start(vec![]).await;
    let rejected_d = MockWorker::start(vec![]).await;
    let accepted_p = MockWorker::start(vec![]).await;
    let accepted_d = MockWorker::start(vec![]).await;
    let ctx = context(&[
        ("p1", &rejected_p, WorkerMode::Prefill),
        ("d1", &rejected_d, WorkerMode::Decode),
        ("p2", &accepted_p, WorkerMode::Prefill),
        ("d2", &accepted_d, WorkerMode::Decode),
    ]);
    let rejected = Bucket::new(
        "a-rejected",
        BucketGroups::Pd {
            prefill: group(&ctx, &["p1"], AllowAll),
            decode: group(&ctx, &["d1"], InFlightLimit(0)),
        },
    );
    let accepted = Bucket::new(
        "b-accepted",
        BucketGroups::Pd {
            prefill: group(&ctx, &["p2"], AllowAll),
            decode: group(&ctx, &["d2"], InFlightLimit(1)),
        },
    );
    let (_, app) = router(ctx, vec![rejected, accepted]);
    let response = app.oneshot(request(chat(false))).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["x-sgl-decode-url"], accepted_d.url);
    response.into_body().collect().await.unwrap();
    tokio::time::timeout(Duration::from_secs(2), async {
        while accepted_p.captured.lock().unwrap().last_body.is_none() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    assert!(rejected_p.captured.lock().unwrap().last_body.is_none());
    assert!(rejected_d.captured.lock().unwrap().last_body.is_none());
}

#[tokio::test]
async fn new_policy_invalid_requests_fail_before_dispatch() {
    let worker = MockWorker::start(vec![]).await;
    let ctx = context(&[("w", &worker, WorkerMode::Plain)]);
    let bucket = Bucket::new(
        "default",
        BucketGroups::Plain(group(&ctx, &["w"], AllowAll)),
    );
    let (ctx, app) = router(ctx, vec![bucket]);
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
async fn slo_headers_order_complete_pd_buckets_and_validate_before_dispatch() {
    use sgl_router::buckets_reorg::SloPreference;
    let fast_p = MockWorker::start(vec![]).await;
    let slow_p = MockWorker::start(vec![]).await;
    let fast_d = MockWorker::start(vec![]).await;
    let slow_d = MockWorker::start(vec![]).await;
    let mut ctx = context(&[
        ("fast-p", &fast_p, WorkerMode::Prefill),
        ("slow-p", &slow_p, WorkerMode::Prefill),
        ("fast-d", &fast_d, WorkerMode::Decode),
        ("slow-d", &slow_d, WorkerMode::Decode),
    ]);
    let mut prefill = Bucket::new(
        "a-prefill",
        BucketGroups::Pd {
            prefill: group(&ctx, &["fast-p"], AllowAll),
            decode: group(&ctx, &["slow-d"], AllowAll),
        },
    );
    prefill.ttft_ms = Some(50);
    prefill.tokens_per_second = Some(10.0);
    let mut decode = Bucket::new(
        "b-decode",
        BucketGroups::Pd {
            prefill: group(&ctx, &["slow-p"], AllowAll),
            decode: group(&ctx, &["fast-d"], AllowAll),
        },
    );
    decode.ttft_ms = Some(100);
    decode.tokens_per_second = Some(100.0);
    let mut resolver = BucketResolver::new(vec![decode, prefill]);
    resolver.ttft_slo = SloPreference::SloFirst;
    resolver.tps_slo = SloPreference::SloFirst;
    ctx.chat_routing = ChatRouting::Reorg([(ModelId("tiny".into()), resolver)].into());
    let app = build_router(Arc::new(ctx));
    for (ttft, tps) in [("0", "100"), ("50", "NaN"), ("50", "0"), ("abc", "100")] {
        let mut req = request(chat(false));
        req.headers_mut()
            .insert("x-sgl-ttft-slo-ms", ttft.parse().unwrap());
        req.headers_mut()
            .insert("x-sgl-tps-slo", tps.parse().unwrap());
        assert_eq!(
            app.clone().oneshot(req).await.unwrap().status(),
            StatusCode::BAD_REQUEST
        );
    }
    for worker in [&fast_p, &slow_p, &fast_d, &slow_d] {
        assert!(worker.captured.lock().unwrap().last_body.is_none());
    }
    // With both targets unmet once, the existing bucket order breaks the tie.
    // With TTFT unconstrained, throughput chooses the other complete bucket.
    for (ttft, expected_d, expected_p, unused_d, unused_p) in [
        ("50", &slow_d, &fast_p, &fast_d, &slow_p),
        ("100", &fast_d, &slow_p, &slow_d, &fast_p),
    ] {
        for worker in [&fast_p, &slow_p, &fast_d, &slow_d] {
            worker.captured.lock().unwrap().last_body = None;
        }
        let mut req = request(chat(false));
        req.headers_mut()
            .insert("x-sgl-ttft-slo-ms", ttft.parse().unwrap());
        req.headers_mut()
            .insert("x-sgl-tps-slo", "100".parse().unwrap());
        let response = app.clone().oneshot(req).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["x-sgl-decode-url"], expected_d.url);
        response.into_body().collect().await.unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            while expected_p.captured.lock().unwrap().last_body.is_none() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        assert!(unused_p.captured.lock().unwrap().last_body.is_none());
        assert!(unused_d.captured.lock().unwrap().last_body.is_none());
    }
}

#[tokio::test]
async fn preferred_bucket_rejection_falls_back_and_disabled_headers_are_ignored() {
    use sgl_router::buckets_reorg::SloPreference;
    for enabled in [false, true] {
        let rejected = MockWorker::start(vec![]).await;
        let accepted = MockWorker::start(vec![]).await;
        let mut ctx = context(&[
            ("rejected", &rejected, WorkerMode::Plain),
            ("accepted", &accepted, WorkerMode::Plain),
        ]);
        let mut fast = Bucket::new(
            "a-fast",
            BucketGroups::Plain(group(&ctx, &["rejected"], InFlightLimit(0))),
        );
        fast.ttft_ms = Some(10);
        let mut slow = Bucket::new(
            "b-slow",
            BucketGroups::Plain(group(&ctx, &["accepted"], AllowAll)),
        );
        slow.ttft_ms = Some(100);
        let mut resolver = BucketResolver::new(vec![slow, fast]);
        if enabled {
            resolver.ttft_slo = SloPreference::SloFirst;
        }
        ctx.chat_routing = ChatRouting::Reorg([(ModelId("tiny".into()), resolver)].into());
        let app = build_router(Arc::new(ctx));
        let mut req = request(chat(false));
        req.headers_mut().insert(
            "x-sgl-ttft-slo-ms",
            if enabled { "50" } else { "invalid" }.parse().unwrap(),
        );
        // TPS is disabled even when TTFT is enabled.
        req.headers_mut()
            .insert("x-sgl-tps-slo", "NaN".parse().unwrap());
        let response = app.oneshot(req).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        response.into_body().collect().await.unwrap();
        assert!(rejected.captured.lock().unwrap().last_body.is_none());
        assert!(accepted.captured.lock().unwrap().last_body.is_some());
    }
}
