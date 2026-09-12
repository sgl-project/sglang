// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! `POST /generate` routing end to end: the sglang-native surface proxies
//! to the worker's `/generate`, falls back to the configured model id,
//! rejects batch bodies before dispatch, never rewrites client-supplied
//! `input_ids`, and injects the flat PD `bootstrap_*` fields — all asserted
//! against what the `MockWorker` actually received.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use bytes::Bytes;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use sgl_router::config::{Cli, Config};
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

use crate::common::mock_worker::MockWorker;

const MODEL: &str = "tiny";

fn config(flags: &[&str]) -> Config {
    let mut argv = vec![
        "sgl-router",
        "--model-id",
        MODEL,
        "--tokenizer-path",
        "tests/fixtures/tiny_tokenizer.json",
        "--worker-urls",
        "http://placeholder:0",
    ];
    argv.extend_from_slice(flags);
    <Cli as clap::Parser>::parse_from(argv)
        .into_config()
        .expect("flags must parse")
}

fn build_ctx(specs: Vec<WorkerSpec>, flags: &[&str]) -> Arc<AppContext> {
    let cfg = config(flags);
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    for s in specs {
        let _ = registry.add(s);
    }
    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
}

fn plain_spec(mock: &MockWorker) -> WorkerSpec {
    WorkerSpec {
        id: WorkerId(mock.url.clone()),
        url: mock.url.clone(),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId(MODEL.into())],
        bootstrap_port: None,
    }
}

async fn send(ctx: Arc<AppContext>, path: &str, body: Value) -> (StatusCode, Bytes) {
    let req = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    let resp = build_router(ctx).oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, bytes)
}

fn captured(mock: &MockWorker) -> Option<Value> {
    let b = mock.captured.lock().unwrap().last_body.clone()?;
    Some(serde_json::from_slice(&b).expect("captured body is valid JSON"))
}

/// A minimal non-streaming `/generate` request proxies through to the
/// worker's `/generate` path with the body intact.
#[tokio::test]
async fn non_streaming_returns_200_and_proxies_to_generate() {
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(vec![plain_spec(&mock)], &[]);
    let (status, _) = send(ctx, "/generate", json!({"model": MODEL, "text": "hi"})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        mock.captured.lock().unwrap().last_path.as_deref(),
        Some("/generate"),
    );
    let body = captured(&mock).expect("worker received a request");
    assert_eq!(body.get("text"), Some(&json!("hi")), "{body}");
}

/// A `stream: true` request gets the worker's SSE stream passed through.
#[tokio::test]
async fn streaming_sse_passthrough() {
    let chunks = vec![
        "data: {\"text\":\"hel\"}\n\n",
        "data: {\"text\":\"lo\"}\n\n",
        "data: [DONE]\n\n",
    ];
    let mock = MockWorker::start(chunks.clone()).await;
    let ctx = build_ctx(vec![plain_spec(&mock)], &[]);
    let (status, body) = send(
        ctx,
        "/generate",
        json!({"model": MODEL, "text": "hi", "stream": true}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let text = String::from_utf8(body.to_vec()).unwrap();
    for chunk in chunks {
        assert!(text.contains(chunk.trim_end()), "missing chunk: {text}");
    }
}

/// `/generate` is the sglang-native surface, where `model` is a
/// gateway-borrowed key sglang's own clients don't send: an omitted `model`
/// routes on the configured id. The same omission on
/// `/v1/chat/completions` stays a 400.
#[tokio::test]
async fn missing_model_routes_on_the_configured_id() {
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(vec![plain_spec(&mock)], &[]);
    let (status, _) = send(ctx, "/generate", json!({"text": "hi"})).await;
    assert_eq!(status, StatusCode::OK);
    assert!(captured(&mock).is_some(), "worker received a request");

    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(vec![plain_spec(&mock)], &[]);
    let (status, _) = send(
        ctx,
        "/v1/chat/completions",
        json!({"messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "chat keeps its missing-model 400"
    );
    assert!(captured(&mock).is_none());
}

/// Batch spellings are rejected before dispatch: `text` as an array of
/// strings, and `input_ids` as an array of ARRAYS. A flat integer
/// `input_ids` is the supported single-prompt shape and must NOT be
/// rejected.
#[tokio::test]
async fn batch_bodies_are_400_before_dispatch() {
    for body in [
        json!({"model": MODEL, "text": ["a", "b"]}),
        json!({"model": MODEL, "input_ids": [[1, 2], [3, 4]]}),
    ] {
        let mock = MockWorker::start(vec![]).await;
        let ctx = build_ctx(vec![plain_spec(&mock)], &[]);
        let (status, _) = send(ctx, "/generate", body.clone()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(
            captured(&mock).is_none(),
            "a rejected batch must not reach the engine: {body}"
        );
    }
}

/// A flat integer `input_ids` is a single prompt: it proxies through, and
/// the router does NOT overwrite the client's ids with its own tokenization
/// (router-computed ids are never forwarded on `/generate` — the engine
/// would prefer them and silently drop `text`).
#[tokio::test]
async fn flat_input_ids_is_a_single_prompt_and_not_rewritten() {
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(vec![plain_spec(&mock)], &[]);
    let (status, _) = send(
        ctx,
        "/generate",
        json!({"model": MODEL, "text": "hi", "input_ids": [450, 12, 99]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let body = captured(&mock).expect("worker received a request");
    assert_eq!(body.get("input_ids"), Some(&json!([450, 12, 99])), "{body}");
}

/// PD mode: `/generate` fans out to prefill and decode with the SAME three
/// flat top-level `bootstrap_*` fields, and the prefill spawn also targets
/// `/generate` on the worker.
#[tokio::test]
async fn pd_mode_generate_injects_flat_bootstrap_fields() {
    let prefill = MockWorker::start(vec![]).await;
    let decode = MockWorker::start(vec![]).await;
    let ctx = build_ctx(
        vec![
            WorkerSpec {
                id: WorkerId("p1".into()),
                url: prefill.url.clone(),
                mode: WorkerMode::Prefill,
                model_ids: vec![ModelId(MODEL.into())],
                bootstrap_port: Some(8997),
            },
            WorkerSpec {
                id: WorkerId("d1".into()),
                url: decode.url.clone(),
                mode: WorkerMode::Decode,
                model_ids: vec![ModelId(MODEL.into())],
                bootstrap_port: None,
            },
        ],
        &[],
    );
    let (status, _) = send(ctx, "/generate", json!({"model": MODEL, "text": "hi"})).await;
    assert_eq!(status, StatusCode::OK, "decode side should 200");

    // The prefill leg is spawned in the background; poll briefly.
    let mut prefill_body = None;
    for _ in 0..400 {
        if let Some(b) = captured(&prefill) {
            prefill_body = Some(b);
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let prefill_body = prefill_body.expect("prefill received a request");
    let decode_body = captured(&decode).expect("decode received a request");

    for (label, body) in [("prefill", &prefill_body), ("decode", &decode_body)] {
        assert_eq!(
            body.get("bootstrap_host").and_then(Value::as_str),
            Some("127.0.0.1"),
            "{label}: {body}"
        );
        assert_eq!(
            body.get("bootstrap_port").and_then(Value::as_u64),
            Some(8997),
            "{label}: {body}"
        );
        assert!(
            body.get("bootstrap_room").and_then(Value::as_u64).is_some(),
            "{label}: {body}"
        );
    }
    assert_eq!(
        prefill_body.get("bootstrap_room"),
        decode_body.get("bootstrap_room"),
        "prefill and decode must share the room"
    );
    assert_eq!(
        prefill.captured.lock().unwrap().last_path.as_deref(),
        Some("/generate"),
        "the prefill spawn must also target /generate"
    );
}

/// A `/v1/chat/completions` body carrying client `input_ids` still routes
/// via the chat encoder, so the broken-offload counter does NOT bump — the
/// `input_ids` branch is `Generate`-only, and this is the test that would
/// catch it leaking into the shared token production.
#[tokio::test]
async fn chat_with_client_input_ids_does_not_bump_the_offload_error_metric() {
    // A registry whose model has a chat encoder: the tiny tokenizer plus a
    // sibling tokenizer_config.json carrying a chat template.
    let dir = tempfile::tempdir().unwrap();
    std::fs::copy(
        "tests/fixtures/tiny_tokenizer.json",
        dir.path().join("tokenizer.json"),
    )
    .unwrap();
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        r#"{"chat_template": "{{ bos_token }}{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}{% endfor %}",
            "bos_token": "<|endoftext|>"}"#,
    )
    .unwrap();

    let mock = MockWorker::start(vec![]).await;
    let cfg = <Cli as clap::Parser>::parse_from([
        "sgl-router",
        "--model-id",
        MODEL,
        "--tokenizer-path",
        dir.path().to_str().unwrap(),
        "--worker-urls",
        "http://placeholder:0",
    ])
    .into_config()
    .expect("flags must parse");
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    assert!(
        tokenizers.has_chat_encoder(MODEL),
        "the fixture must attach a Jinja chat encoder"
    );
    let registry = Arc::new(WorkerRegistry::default());
    let _ = registry.add(plain_spec(&mock));
    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    let ctx = Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies));

    let (status, _) = send(
        Arc::clone(&ctx),
        "/v1/chat/completions",
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "input_ids": [1, 2, 3],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let m = ctx.metrics.render();
    assert!(
        !m.contains(r#"sgl_router_ingress_tokenize_errors_total{model_id="tiny"}"#),
        "the broken-offload counter must not bump for a chat request with input_ids; got:\n{m}"
    );
}
