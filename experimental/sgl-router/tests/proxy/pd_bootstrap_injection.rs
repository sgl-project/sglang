// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! PD-disagg bootstrap-room injection + dual-dispatch — end-to-end
//! at the HTTP layer using MockWorkers.
//!
//! Asserts the router-side contract for SGLang disagg-prefill HTTP mode:
//!
//! * Every PD-mode `/v1/chat/completions` request fans out to BOTH a
//!   prefill and a decode worker (the prefill is `tokio::spawn`'d in
//!   the background; the decode is awaited for the client response).
//! * Both bodies carry the SAME flat top-level fields:
//!     - `bootstrap_host` = the chosen prefill worker's host
//!     - `bootstrap_port` = the chosen prefill worker's bootstrap port
//!     - `bootstrap_room` = a random u64 in `[0, i64::MAX]` (63-bit)
//! * Plain-mode requests do NOT carry any `bootstrap_*` field — the
//!   injection step is gated on `worker.mode() == Prefill`.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use bytes::Bytes;
use serde_json::{json, Value};
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

/// Pattern-B dispatch: prefill is `tokio::spawn`'d as a detached task
/// so the client response can return as soon as decode is reachable —
/// the prefill body is captured *eventually* but may not be present
/// when the handler returns. Poll with a short bound rather than
/// sleeping a fixed duration.
async fn await_captured_body(
    mock: &crate::common::mock_worker::MockWorker,
    timeout: Duration,
    label: &str,
) -> Bytes {
    let start = std::time::Instant::now();
    loop {
        // Release the `std::sync::Mutex` guard before the sleep.await
        // (clippy: await_holding_lock).
        let captured = mock.captured.lock().unwrap().last_body.clone();
        if let Some(b) = captured {
            return b;
        }
        if start.elapsed() > timeout {
            panic!("{label}: no request body captured within {timeout:?}");
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

fn parse_body(b: &Bytes) -> Value {
    serde_json::from_slice(b).expect("body must be valid JSON")
}

/// Helper: extract bootstrap_host as &str.
fn bootstrap_host(v: &Value) -> Option<&str> {
    v.get("bootstrap_host").and_then(|x| x.as_str())
}
/// Helper: extract bootstrap_port as u16.
fn bootstrap_port(v: &Value) -> Option<u16> {
    v.get("bootstrap_port")
        .and_then(|x| x.as_u64())
        .map(|p| p as u16)
}
/// Helper: extract bootstrap_room as u64.
fn bootstrap_room(v: &Value) -> Option<u64> {
    v.get("bootstrap_room").and_then(|x| x.as_u64())
}

/// PD-mode chat fans out to BOTH prefill and decode with identical
/// bootstrap fields injected into both bodies.
#[tokio::test]
async fn pd_mode_chat_injects_bootstrap_fields_into_both_bodies() {
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

    let res = app.oneshot(chat_request()).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK, "decode side should 200");

    let prefill_body = await_captured_body(&prefill, Duration::from_secs(2), "prefill").await;
    let decode_body = await_captured_body(&decode, Duration::from_secs(2), "decode").await;
    let pj = parse_body(&prefill_body);
    let dj = parse_body(&decode_body);

    // Same bootstrap_room on both sides (one room minted per request).
    let p_room = bootstrap_room(&pj).expect("prefill body missing bootstrap_room");
    let d_room = bootstrap_room(&dj).expect("decode body missing bootstrap_room");
    assert_eq!(
        p_room, d_room,
        "prefill and decode must share the same bootstrap_room"
    );

    // Room must be in [0, i64::MAX]: the SGLang prefill stores it as
    // i64 internally, so values with the top bit set wrap negative.
    assert!(
        p_room <= i64::MAX as u64,
        "bootstrap_room {p_room} exceeds 63-bit range; SGLang would mis-store as negative i64",
    );

    // bootstrap_host on both sides == prefill worker's hostname
    // (MockWorker binds to 127.0.0.1).
    assert_eq!(bootstrap_host(&pj), Some("127.0.0.1"));
    assert_eq!(bootstrap_host(&dj), Some("127.0.0.1"));

    // bootstrap_port on both sides == prefill's configured bootstrap_port.
    assert_eq!(bootstrap_port(&pj), Some(8997));
    assert_eq!(bootstrap_port(&dj), Some(8997));
}

/// Plain-mode (non-PD) requests do NOT carry any `bootstrap_*` field.
/// The injection step is gated on `worker.mode() == Prefill`; plain
/// workers serve the chat route directly without disagg bootstrapping.
#[tokio::test]
async fn plain_mode_chat_does_not_inject_bootstrap_fields() {
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

    let body = await_captured_body(&plain, Duration::from_secs(2), "plain").await;
    let v = parse_body(&body);
    assert!(
        v.get("bootstrap_room").is_none(),
        "plain-mode request must not carry bootstrap_room; got {v}"
    );
    assert!(
        v.get("bootstrap_host").is_none(),
        "plain-mode request must not carry bootstrap_host; got {v}"
    );
    assert!(
        v.get("bootstrap_port").is_none(),
        "plain-mode request must not carry bootstrap_port; got {v}"
    );
}

/// PD-mode with multiple prefill workers + different `bootstrap_port`
/// values: the bootstrap_port injected MUST match the actually-chosen
/// prefill (not e.g. the first registered or a global config value).
#[tokio::test]
async fn pd_mode_bootstrap_port_matches_chosen_prefill_worker() {
    let prefill_a = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let prefill_b = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let decode = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let ctx = build_ctx(vec![
        WorkerSpec {
            id: WorkerId("pA".into()),
            url: prefill_a.url.clone(),
            mode: WorkerMode::Prefill,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: Some(11111),
        },
        WorkerSpec {
            id: WorkerId("pB".into()),
            url: prefill_b.url.clone(),
            mode: WorkerMode::Prefill,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: Some(22222),
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

    // Fire enough requests to ensure round-robin hits both prefill workers.
    for _ in 0..6 {
        let res = app.clone().oneshot(chat_request()).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    // Wait until both prefill workers have captured at least one body.
    let body_a = await_captured_body(&prefill_a, Duration::from_secs(2), "prefill_a").await;
    let body_b = await_captured_body(&prefill_b, Duration::from_secs(2), "prefill_b").await;
    let va = parse_body(&body_a);
    let vb = parse_body(&body_b);
    // Each prefill must see its OWN bootstrap_port — never the other's.
    assert_eq!(
        bootstrap_port(&va),
        Some(11111),
        "prefill_a body should carry its own bootstrap_port"
    );
    assert_eq!(
        bootstrap_port(&vb),
        Some(22222),
        "prefill_b body should carry its own bootstrap_port"
    );
}

/// Pin Pattern B's "prefill failure is invisible to the client"
/// contract: when the spawned prefill task gets a 5xx (or any other
/// upstream error), the decode response still reaches the client
/// unmodified. The router intentionally does not wire fail-fast here —
/// the decode side will eventually hang on `bootstrap_room` and time
/// out, but the chat handler itself doesn't propagate the prefill
/// error. Matches llm-d / aibrix behaviour.
#[tokio::test]
async fn pd_mode_prefill_5xx_does_not_poison_decode_response() {
    let prefill = crate::common::mock_worker::MockWorker::start_returning_error(
        StatusCode::INTERNAL_SERVER_ERROR,
        json!({"error": "simulated prefill failure"}),
    )
    .await;
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

    // Client must see decode's 200 — the failing prefill is invisible.
    let res = app.oneshot(chat_request()).await.unwrap();
    assert_eq!(
        res.status(),
        StatusCode::OK,
        "decode response should reach the client even when prefill returned 5xx",
    );

    // Decode received its body (proves dual dispatch fired despite
    // the prefill failure).
    let decode_body = await_captured_body(&decode, Duration::from_secs(2), "decode").await;
    let v = parse_body(&decode_body);
    assert_eq!(bootstrap_port(&v), Some(8997));

    // Prefill also received its body — it just returned 5xx. The
    // bootstrap fields are present so the engine WOULD have honoured
    // the bootstrap_room if the mock had succeeded.
    let prefill_body = await_captured_body(&prefill, Duration::from_secs(2), "prefill").await;
    let pv = parse_body(&prefill_body);
    assert_eq!(bootstrap_port(&pv), Some(8997));
}

fn streaming_chat_request() -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&serde_json::json!({
                "model": "tiny",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true,
            }))
            .unwrap(),
        ))
        .unwrap()
}

/// PD-disagg mode deliberately excludes the abort-on-disconnect feature (see
/// `chat_completions_inner`'s `request_id` comment): prefill is detached to
/// outlive the client for KV-transfer correctness, and aborting only the
/// decode half mid-transfer is out of scope. A client disconnect mid-decode-
/// stream must NOT send `/abort_request` to either worker, and neither
/// forwarded body should carry an injected `rid`.
#[tokio::test]
async fn pd_mode_disconnect_does_not_abort_either_worker() {
    let prefill = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let decode = crate::common::mock_worker::MockWorker::start_slow_stream(
        vec!["data: a\n\n", "data: b\n\n", "data: c\n\n"],
        Duration::from_millis(50),
    )
    .await;
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

    let res = app.oneshot(streaming_chat_request()).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    // Disconnect mid-decode-stream — the same shape as the plain-mode abort
    // test, but here it must be a no-op.
    use futures::StreamExt;
    let mut data_stream = res.into_body().into_data_stream();
    assert!(data_stream.next().await.is_some());
    drop(data_stream);

    // No fixed event to poll for ("nothing happens"), so wait out a window
    // comfortably longer than the plain-mode abort tests' detection latency.
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        decode.abort_log.lock().unwrap().is_empty(),
        "PD mode must never send /abort_request to the decode worker"
    );
    assert!(
        prefill.abort_log.lock().unwrap().is_empty(),
        "PD mode must never send /abort_request to the prefill worker"
    );

    // Neither forwarded body carries an injected `rid` — `rid_to_inject` is
    // `None` whenever `decode_peer.is_some()`.
    let decode_body = await_captured_body(&decode, Duration::from_secs(2), "decode").await;
    let prefill_body = await_captured_body(&prefill, Duration::from_secs(2), "prefill").await;
    assert!(
        parse_body(&decode_body).get("rid").is_none(),
        "PD mode must not inject a rid into the decode body"
    );
    assert!(
        parse_body(&prefill_body).get("rid").is_none(),
        "PD mode must not inject a rid into the prefill body"
    );
}

/// PD-mode STREAMING: the stream-outcome hook must be installed on the
/// decode arm (the client-facing stream) and labelled with the DECODE
/// worker's URL — the label is load-bearing, since the metric exists to
/// finger the specific engine pod that accepted-then-in-band-rejected.
#[tokio::test]
async fn pd_mode_streaming_inband_error_labels_decode_worker() {
    use http_body_util::BodyExt;

    let prefill = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let decode_chunks: Vec<&'static str> = vec![
        "data: {\"error\": {\"message\": \"The request queue is full.\", \"code\": 503}}\n\n",
        "data: [DONE]\n\n",
    ];
    let decode = crate::common::mock_worker::MockWorker::start(decode_chunks).await;
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
    let app = build_router(ctx.clone());

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({
                "model": "tiny",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true
            }))
            .unwrap(),
        ))
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let _ = res.into_body().collect().await.unwrap().to_bytes();

    // The metric records from the SSE pump's spawned task — poll briefly.
    let expected = format!(
        r#"sgl_router_stream_outcome_total{{worker_url="{}",model_id="tiny",outcome="inband_error"}} 1"#,
        decode.url,
    );
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    let m = loop {
        let m = ctx.metrics.render();
        if m.contains(&expected) || std::time::Instant::now() > deadline {
            break m;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    };
    assert!(
        m.contains(&expected),
        "PD streaming in-band error must be labelled with the decode worker; got:\n{m}",
    );
    // The prefill worker must NOT appear in the stream-outcome family — its
    // spawned side-request is a buffered JSON forward, not a client stream.
    assert!(
        !m.contains(&format!(
            r#"sgl_router_stream_outcome_total{{worker_url="{}""#,
            prefill.url,
        )),
        "prefill side must not record a stream outcome; got:\n{m}",
    );
}
