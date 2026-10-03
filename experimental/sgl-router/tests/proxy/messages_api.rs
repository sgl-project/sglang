// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! `POST /v1/messages` end to end.

use axum::body::Body;
use axum::http::{Request, StatusCode};
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
use crate::common::streaming::collect_body;

const MODEL: &str = "tiny";

fn config() -> Config {
    <Cli as clap::Parser>::parse_from([
        "sgl-router",
        "--model-id",
        MODEL,
        "--tokenizer-path",
        "tests/fixtures/tiny_tokenizer.json",
        "--worker-urls",
        "http://placeholder:0",
    ])
    .into_config()
    .expect("flags must parse")
}

fn build_ctx(url: String) -> Arc<AppContext> {
    let cfg = config();
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    let _ = registry.add(WorkerSpec {
        id: WorkerId(url.clone()),
        url,
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId(MODEL.into())],
        bootstrap_port: None,
        version_group: None,
    });
    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
}

async fn post(ctx: Arc<AppContext>, path: &str, body: Value) -> (StatusCode, String, Vec<u8>) {
    let req = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .header("anthropic-version", "2023-06-01")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    let res = build_router(ctx).oneshot(req).await.unwrap();
    let status = res.status();
    let ct = res
        .headers()
        .get("content-type")
        .map(|v| v.to_str().unwrap().to_owned())
        .unwrap_or_default();
    (status, ct, collect_body(res.into_body()).await.to_vec())
}

fn captured(mock: &MockWorker) -> Option<Value> {
    let b = mock.captured.lock().unwrap().last_body.clone()?;
    Some(serde_json::from_slice(&b).unwrap())
}

fn assert_anthropic_error(body: &[u8], want_type: &str) {
    let v: Value = serde_json::from_slice(body).expect("error body is JSON");
    assert_eq!(v["type"], "error", "{v}");
    assert_eq!(v["error"]["type"], want_type, "{v}");
    assert!(!v["error"]["message"].as_str().unwrap().is_empty(), "{v}");
}

#[tokio::test]
async fn buffered_message() {
    let mock = MockWorker::start(vec![]).await;
    let (status, ct, body) = post(
        build_ctx(mock.url.clone()),
        "/v1/messages",
        json!({"model": MODEL, "max_tokens": 512, "system": [{"type": "text", "text": "be brief"}],
               "messages": [{"role": "user", "content": [{"type": "text", "text": "reply OK"}]}]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    assert!(ct.starts_with("application/json"));
    let m: Value = serde_json::from_slice(&body).unwrap();
    assert!(m["id"].as_str().unwrap().starts_with("msg_"));
    assert_eq!(m["type"], "message");
    assert_eq!(m["role"], "assistant");
    assert_eq!(m["model"], MODEL);
    assert_eq!(m["content"], json!([{"type": "text", "text": "ok"}]));
    assert_eq!(m["stop_reason"], "end_turn");
    assert!(m["usage"]["input_tokens"].is_u64());
    assert!(m["usage"]["output_tokens"].is_u64());

    let sent = captured(&mock).unwrap();
    assert_eq!(
        sent["messages"],
        json!([{"role": "system", "content": "be brief"}, {"role": "user", "content": "reply OK"}])
    );
    assert_eq!(sent["max_tokens"], 512);
    assert!(sent.get("system").is_none());
}

#[tokio::test]
async fn streaming_event_sequence() {
    let mock = MockWorker::start(vec![
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"O\"},\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":1}}\n\ndata: {\"choices\":[{\"index\":0,\"delta\":{\"con",
        "tent\":\"K\"},\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":2}}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":2}}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":2}}\n\n",
        "data: [DONE]\n\n",
    ])
    .await;
    let (status, ct, body) = post(
        build_ctx(mock.url.clone()),
        "/v1/messages",
        json!({"model": MODEL, "max_tokens": 512, "stream": true,
               "messages": [{"role": "user", "content": "reply OK"}]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(ct.starts_with("text/event-stream"), "{ct}");
    let text = String::from_utf8(body).unwrap();
    let events: Vec<(String, Value)> = text
        .split("\n\n")
        .filter(|b| !b.is_empty())
        .map(|b| {
            let mut l = b.lines();
            let ev = l
                .next()
                .unwrap()
                .strip_prefix("event: ")
                .unwrap()
                .to_owned();
            let data =
                serde_json::from_str(l.next().unwrap().strip_prefix("data: ").unwrap()).unwrap();
            (ev, data)
        })
        .collect();
    let names: Vec<&str> = events.iter().map(|(e, _)| e.as_str()).collect();
    assert_eq!(
        names,
        [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
    );
    for (ev, data) in &events {
        assert_eq!(data["type"], ev.as_str());
    }
    assert_eq!(events[0].1["message"]["usage"]["input_tokens"], 9);
    assert_eq!(events[5].1["delta"]["stop_reason"], "end_turn");
    assert_eq!(events[5].1["usage"]["output_tokens"], 2);
    assert!(!text.contains("[DONE]"));

    let sent = captured(&mock).unwrap();
    assert_eq!(
        sent["stream_options"],
        json!({"include_usage": true, "continuous_usage_stats": true})
    );
}

#[tokio::test]
async fn engine_error_is_rewrapped() {
    let mock = MockWorker::start_returning_error(
        StatusCode::BAD_REQUEST,
        json!({"object": "error", "message": "invalid json_schema", "type": "BadRequestError",
               "param": null, "code": 400}),
    )
    .await;
    let (status, _, body) = post(
        build_ctx(mock.url.clone()),
        "/v1/messages",
        json!({"model": MODEL, "max_tokens": 16, "messages": [{"role": "user", "content": "x"}],
               "output_config": {"format": {"type": "json_schema", "schema": "nope"}}}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_anthropic_error(&body, "invalid_request_error");
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["error"]["message"], "invalid json_schema");
}

#[tokio::test]
async fn router_side_rejections_use_the_anthropic_envelope() {
    let mock = MockWorker::start(vec![]).await;
    for req in [
        json!({"model": MODEL, "max_tokens": 16, "messages": []}),
        json!({"model": MODEL, "max_tokens": 16, "messages": [{"content": "x"}]}),
        json!({"model": MODEL, "messages": [{"role": "user", "content": "x"}]}),
    ] {
        let (status, _, body) =
            post(build_ctx(mock.url.clone()), "/v1/messages", req.clone()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{req}");
        assert_anthropic_error(&body, "invalid_request_error");
    }
    let (status, _, body) = post(
        build_ctx(mock.url.clone()),
        "/v1/messages",
        json!({"model": "no-such-model", "max_tokens": 16,
               "messages": [{"role": "user", "content": "x"}]}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_anthropic_error(&body, "not_found_error");
    assert!(
        captured(&mock).is_none(),
        "rejected requests must not reach the engine"
    );
}

#[tokio::test]
async fn count_tokens() {
    let mock = MockWorker::start(vec![]).await;
    let (status, _, body) = post(
        build_ctx(mock.url.clone()),
        "/v1/messages/count_tokens",
        json!({"model": MODEL, "messages": [{"role": "user", "content": "hello world"}]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert!(v["input_tokens"].as_u64().unwrap() > 0, "{v}");
    assert!(
        captured(&mock).is_none(),
        "count_tokens is answered by the router"
    );
}

#[tokio::test]
async fn oversized_body_uses_the_anthropic_envelope() {
    let mock = MockWorker::start(vec![]).await;
    let big = "x".repeat(sgl_router::server::routes::chat::MAX_CHAT_BODY_BYTES);
    let (status, _, body) = post(
        build_ctx(mock.url.clone()),
        "/v1/messages",
        json!({"model": MODEL, "max_tokens": 16,
               "messages": [{"role": "user", "content": big}]}),
    )
    .await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_anthropic_error(&body, "request_too_large");
}
