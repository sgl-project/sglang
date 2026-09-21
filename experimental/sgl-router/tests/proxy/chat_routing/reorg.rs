// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::common::mock_worker::MockWorker;
use futures::future::BoxFuture;
use sgl_router::buckets_reorg::{Bucket, BucketGroups, BucketResolver, EngineGroup};
use sgl_router::policies::PolicyRegistry;
use sgl_router::policies_reorg::admission::{AllowAll, Decision, EngineAdmission};
use sgl_router::policies_reorg::{Pick, PickError, PickRequest, Policy, Rejection, Stage};
use sgl_router::server::app_context::ChatRouting;
use sgl_router::state::load_monitor::engine_reported_load::EngineReportedWorkerLoad;
use std::sync::Mutex;

mod session_aware;

type PickCall = (String, Stage, u64, Option<u64>);

#[derive(Debug)]
struct FirstPolicy {
    calls: Mutex<Vec<PickCall>>,
    admission: Arc<dyn EngineAdmission>,
    miss: bool,
    invalid: bool,
}

impl Default for FirstPolicy {
    fn default() -> Self {
        Self {
            admission: Arc::new(AllowAll),
            miss: false,
            invalid: false,
            calls: Mutex::new(Vec::new()),
        }
    }
}

impl Policy for FirstPolicy {
    fn pick<'a>(
        &'a self,
        engines: &'a [Arc<Worker>],
        request: &'a PickRequest<'a>,
    ) -> BoxFuture<'a, Result<Pick, PickError>> {
        Box::pin(async move {
            self.calls.lock().unwrap().push((
                request.bucket.to_owned(),
                request.stage,
                request.input_tokens,
                request.expected_peak_tokens,
            ));
            if self.invalid {
                return Err(PickError::InvalidSignal("invalid policy input".into()));
            }
            if self.miss {
                return Err(PickError::NoCandidates);
            }
            if engines.is_empty() {
                return Err(PickError::NoCandidates);
            }
            let engine = engines[0].clone();
            if let Decision::Reject(reason) = self.admission.check(&engine, request, None)? {
                return Err(PickError::AdmissionRejected(Rejection {
                    engine: engine.id.clone(),
                    reason,
                }));
            }
            Ok(Pick {
                engine,
                reason: "test",
            })
        })
    }
}

#[derive(Debug)]
struct RejectAll;

impl EngineAdmission for RejectAll {
    fn check(
        &self,
        _: &Worker,
        _: &PickRequest<'_>,
        _: Option<&EngineReportedWorkerLoad>,
    ) -> Result<Decision, PickError> {
        Ok(Decision::Reject("full".into()))
    }
}

fn rejecting_policy() -> Arc<FirstPolicy> {
    Arc::new(FirstPolicy {
        admission: Arc::new(RejectAll),
        ..Default::default()
    })
}

fn group(id: &str, policy: Arc<FirstPolicy>) -> EngineGroup {
    EngineGroup {
        worker_ids: Some([WorkerId(id.into())].into_iter().collect()),
        policy,
    }
}

fn context(workers: &[(&str, Stage, &MockWorker)], buckets: Vec<Bucket>) -> Arc<AppContext> {
    let config = config_for("");
    let registry = Arc::new(WorkerRegistry::default());
    for &(id, mode, worker) in workers {
        registry
            .add(WorkerSpec {
                id: WorkerId(id.into()),
                url: worker.url.clone(),
                mode,
                model_ids: vec![ModelId("tiny".into())],
                bootstrap_port: Some(8998),
            })
            .unwrap();
    }
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&config).unwrap());
    let mut ctx = AppContext::new(
        config,
        tokenizers,
        Arc::new(Proxy::new(TEST_TIMEOUT).unwrap()),
        registry,
        // The reorg route must not require the legacy policy registry.
        Arc::new(PolicyRegistry::default()),
    );
    ctx.chat_routing = ChatRouting::Reorg(
        [(ModelId("tiny".into()), BucketResolver::new(buckets))]
            .into_iter()
            .collect(),
    );
    Arc::new(ctx)
}

fn request(value: serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&value).unwrap()))
        .unwrap()
}

fn body(content: &str) -> serde_json::Value {
    serde_json::json!({"model": "tiny", "messages": [{"role": "user", "content": content}]})
}

#[tokio::test]
async fn length_selects_plain_bucket_before_engine_selection() {
    let short_worker = MockWorker::start(vec![]).await;
    let long_worker = MockWorker::start(vec![]).await;
    let policy = Arc::new(FirstPolicy::default());
    let mut short = Bucket::new("short", BucketGroups::Plain(group("short", policy.clone())));
    short.limits.max = Some(4);
    let long = Bucket::new("long", BucketGroups::Plain(group("long", policy.clone())));
    let ctx = context(
        &[
            ("short", Stage::Plain, &short_worker),
            ("long", Stage::Plain, &long_worker),
        ],
        vec![long, short],
    );
    let app = build_router(ctx);
    let response = app.clone().oneshot(request(body("hi"))).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let _ = response.into_body().collect().await.unwrap();
    assert!(short_worker.captured.lock().unwrap().last_body.is_some());
    assert!(long_worker.captured.lock().unwrap().last_body.is_none());

    let response = app
        .oneshot(request(body(&"hello ".repeat(30))))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let _ = response.into_body().collect().await.unwrap();
    assert!(long_worker.captured.lock().unwrap().last_body.is_some());
    let calls = policy.calls.lock().unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!((&*calls[0].0, calls[0].1), ("short", Stage::Plain));
    assert_eq!((&*calls[1].0, calls[1].1), ("long", Stage::Plain));
}

#[tokio::test]
async fn pd_picks_both_groups_from_selected_bucket_and_shares_bootstrap() {
    let prefill = MockWorker::start(vec![]).await;
    let decode = MockWorker::start(vec![]).await;
    let policy = Arc::new(FirstPolicy::default());
    let mut selected = Bucket::new(
        "selected",
        BucketGroups::Pd {
            prefill: group("p", policy.clone()),
            decode: group("d", policy.clone()),
        },
    );
    selected.limits.max = Some(100);
    let other_policy = Arc::new(FirstPolicy::default());
    let other = Bucket::new(
        "other",
        BucketGroups::Pd {
            prefill: group("p", other_policy.clone()),
            decode: group("d", other_policy.clone()),
        },
    );
    let ctx = context(
        &[
            ("p", Stage::Prefill, &prefill),
            ("d", Stage::Decode, &decode),
        ],
        vec![other, selected],
    );
    let mut body = body("hello");
    body["max_completion_tokens"] = 10.into();
    let response = build_router(ctx).oneshot(request(body)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["x-sgl-decode-url"], decode.url);
    let _ = response.into_body().collect().await.unwrap();
    tokio::time::timeout(TEST_TIMEOUT, async {
        while prefill.captured.lock().unwrap().last_body.is_none() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let p: serde_json::Value =
        serde_json::from_slice(prefill.captured.lock().unwrap().last_body.as_ref().unwrap())
            .unwrap();
    let d: serde_json::Value =
        serde_json::from_slice(decode.captured.lock().unwrap().last_body.as_ref().unwrap())
            .unwrap();
    assert!(p["bootstrap_room"].is_number());
    assert_eq!(p["bootstrap_room"], d["bootstrap_room"]);
    let calls = policy.calls.lock().unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!((&*calls[0].0, calls[0].1), ("selected", Stage::Prefill));
    assert_eq!((&*calls[1].0, calls[1].1), ("selected", Stage::Decode));
    assert_eq!(calls[0].3, Some(calls[0].2 + 10));
    assert_eq!(calls[1].3, calls[0].3);
    assert!(other_policy.calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn missing_decode_in_all_buckets_does_not_dispatch_prefill() {
    let prefill = MockWorker::start(vec![]).await;
    let decode = MockWorker::start(vec![]).await;
    let policy = Arc::new(FirstPolicy::default());
    let mut selected = Bucket::new(
        "selected",
        BucketGroups::Pd {
            prefill: group("p", policy.clone()),
            decode: group("missing", policy.clone()),
        },
    );
    selected.limits.max = Some(100);
    let other = Bucket::new(
        "other",
        BucketGroups::Pd {
            prefill: group("p", policy.clone()),
            decode: group("also-missing", policy.clone()),
        },
    );
    let ctx = context(
        &[
            ("p", Stage::Prefill, &prefill),
            ("d", Stage::Decode, &decode),
        ],
        vec![other, selected],
    );
    let response = build_router(ctx.clone())
        .oneshot(request(body("hi")))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        response.headers()["x-router-error-code"],
        "no_decode_workers_available"
    );
    assert!(prefill.captured.lock().unwrap().last_body.is_none());
    assert!(decode.captured.lock().unwrap().last_body.is_none());
    assert_eq!(policy.calls.lock().unwrap().len(), 2);
    assert_eq!(ctx.router_inflight_load.inflight_count(), 0);
    assert_eq!(
        ctx.registry
            .get(&WorkerId("p".into()))
            .unwrap()
            .router_inflight_load(),
        0
    );
}

#[tokio::test]
async fn rejects_unsupported_length_unknown_model_and_overflow_before_policy() {
    let worker = MockWorker::start(vec![]).await;
    let policy = Arc::new(FirstPolicy::default());
    let mut bucket = Bucket::new("short", BucketGroups::Plain(group("w", policy.clone())));
    bucket.max_context_tokens = Some(4);
    let app = build_router(context(&[("w", Stage::Plain, &worker)], vec![bucket]));
    let mut long = body("hi");
    long["max_tokens"] = 100.into();
    let mut overflow = body("hi");
    overflow["max_tokens"] = u64::MAX.into();
    let mut unknown = body("hi");
    unknown["model"] = "unknown".into();
    for (body, status) in [
        (long, StatusCode::BAD_REQUEST),
        (overflow, StatusCode::BAD_REQUEST),
        (unknown, StatusCode::NOT_FOUND),
        (serde_json::json!({}), StatusCode::BAD_REQUEST),
    ] {
        let response = app.clone().oneshot(request(body)).await.unwrap();
        assert_eq!(response.status(), status);
    }
    assert!(policy.calls.lock().unwrap().is_empty());
    assert!(worker.captured.lock().unwrap().last_body.is_none());
}

#[tokio::test]
async fn streaming_uses_existing_forwarder() {
    let worker = MockWorker::start(vec!["data: {\"choices\":[]}\n\n", "data: [DONE]\n\n"]).await;
    let policy = Arc::new(FirstPolicy::default());
    let bucket = Bucket::new("plain", BucketGroups::Plain(group("w", policy)));
    let app = build_router(context(&[("w", Stage::Plain, &worker)], vec![bucket]));
    let mut body = body("hi");
    body["stream"] = true.into();
    let response = app.oneshot(request(body)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert!(response.headers()["content-type"]
        .to_str()
        .unwrap()
        .starts_with("text/event-stream"));
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    assert!(String::from_utf8_lossy(&bytes).contains("data: [DONE]"));
}

#[tokio::test]
async fn reorg_route_keeps_chat_body_limit() {
    let app = build_router(context(&[], vec![]));
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(vec![b' '; MAX_CHAT_BODY_BYTES + 1]))
        .unwrap();
    assert_eq!(
        app.oneshot(request).await.unwrap().status(),
        StatusCode::PAYLOAD_TOO_LARGE
    );
}

#[tokio::test]
async fn plain_fallback_skips_empty_missed_and_rejected_buckets_then_stops_on_success() {
    let rejected_worker = MockWorker::start(vec![]).await;
    let winner = MockWorker::start(vec![]).await;
    let skipped = Arc::new(FirstPolicy::default());
    let rejected = rejecting_policy();
    let missed = Arc::new(FirstPolicy {
        miss: true,
        ..Default::default()
    });
    let accepted = Arc::new(FirstPolicy::default());
    let buckets = vec![
        Bucket::new(
            "a-empty",
            BucketGroups::Plain(group("missing", skipped.clone())),
        ),
        Bucket::new(
            "b-miss",
            BucketGroups::Plain(group("rejected", missed.clone())),
        ),
        Bucket::new(
            "c-rejected",
            BucketGroups::Plain(group("rejected", rejected.clone())),
        ),
        Bucket::new(
            "d-winner",
            BucketGroups::Plain(group("winner", accepted.clone())),
        ),
        Bucket::new(
            "e-unused",
            BucketGroups::Plain(group("winner", skipped.clone())),
        ),
    ];
    let ctx = context(
        &[
            ("rejected", Stage::Plain, &rejected_worker),
            ("winner", Stage::Plain, &winner),
        ],
        buckets,
    );
    let response = build_router(ctx)
        .oneshot(request(body("hi")))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let _ = response.into_body().collect().await.unwrap();
    assert!(rejected_worker.captured.lock().unwrap().last_body.is_none());
    assert!(winner.captured.lock().unwrap().last_body.is_some());
    assert!(skipped.calls.lock().unwrap().is_empty());
    assert_eq!(missed.calls.lock().unwrap().len(), 1);
    assert_eq!(rejected.calls.lock().unwrap().len(), 1);
    assert_eq!(accepted.calls.lock().unwrap()[0].0, "d-winner");
}

#[tokio::test]
async fn decode_failure_retries_both_groups_in_next_bucket_without_dispatching_first_prefill() {
    // Cover an empty decode group and rejection of a selected decode engine.
    for reject_decode in [false, true] {
        let first_prefill = MockWorker::start(vec![]).await;
        let second_prefill = MockWorker::start(vec![]).await;
        let decode = MockWorker::start(vec![]).await;
        let first = Arc::new(FirstPolicy::default());
        let rejected = rejecting_policy();
        let accepted = Arc::new(FirstPolicy::default());
        let buckets = vec![
            Bucket::new(
                "a-first",
                BucketGroups::Pd {
                    prefill: group("p1", first.clone()),
                    decode: group(if reject_decode { "d" } else { "missing" }, rejected),
                },
            ),
            Bucket::new(
                "b-second",
                BucketGroups::Pd {
                    prefill: group("p2", accepted.clone()),
                    decode: group("d", accepted.clone()),
                },
            ),
        ];
        let ctx = context(
            &[
                ("p1", Stage::Prefill, &first_prefill),
                ("p2", Stage::Prefill, &second_prefill),
                ("d", Stage::Decode, &decode),
            ],
            buckets,
        );
        let response = build_router(ctx.clone())
            .oneshot(request(body("hi")))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let _ = response.into_body().collect().await.unwrap();
        tokio::time::timeout(TEST_TIMEOUT, async {
            while second_prefill.captured.lock().unwrap().last_body.is_none() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert!(first_prefill.captured.lock().unwrap().last_body.is_none());
        assert!(decode.captured.lock().unwrap().last_body.is_some());
        assert_eq!(
            ctx.registry
                .get(&WorkerId("p1".into()))
                .unwrap()
                .router_inflight_load(),
            0
        );
        assert_eq!(first.calls.lock().unwrap().len(), 1);
        let calls = accepted.calls.lock().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!((&*calls[0].0, calls[0].1), ("b-second", Stage::Prefill));
        assert_eq!((&*calls[1].0, calls[1].1), ("b-second", Stage::Decode));
    }
}

#[tokio::test]
async fn admission_exhaustion_is_preserved_when_later_buckets_are_empty() {
    let worker = MockWorker::start(vec![]).await;
    let first = rejecting_policy();
    let second = rejecting_policy();
    let empty = Arc::new(FirstPolicy::default());
    let ctx = context(
        &[("w", Stage::Plain, &worker)],
        vec![
            Bucket::new("a-first", BucketGroups::Plain(group("w", first.clone()))),
            Bucket::new("b-second", BucketGroups::Plain(group("w", second.clone()))),
            Bucket::new(
                "c-empty",
                BucketGroups::Plain(group("missing", empty.clone())),
            ),
        ],
    );
    let response = build_router(ctx.clone())
        .oneshot(request(body("hi")))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        response.headers()["x-router-error-code"],
        "policy_selection_failed"
    );
    assert_eq!(first.calls.lock().unwrap().len(), 1);
    assert_eq!(second.calls.lock().unwrap().len(), 1);
    assert!(empty.calls.lock().unwrap().is_empty());
    assert!(worker.captured.lock().unwrap().last_body.is_none());
    assert_eq!(ctx.router_inflight_load.inflight_count(), 0);
}

#[tokio::test]
async fn invalid_policy_signal_stops_bucket_iteration() {
    let worker = MockWorker::start(vec![]).await;
    let invalid = Arc::new(FirstPolicy {
        invalid: true,
        ..Default::default()
    });
    let later = Arc::new(FirstPolicy::default());
    let ctx = context(
        &[("w", Stage::Plain, &worker)],
        vec![
            Bucket::new(
                "a-invalid",
                BucketGroups::Plain(group("w", invalid.clone())),
            ),
            Bucket::new("b-later", BucketGroups::Plain(group("w", later.clone()))),
        ],
    );
    let response = build_router(ctx)
        .oneshot(request(body("hi")))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(invalid.calls.lock().unwrap().len(), 1);
    assert!(later.calls.lock().unwrap().is_empty());
    assert!(worker.captured.lock().unwrap().last_body.is_none());
}
