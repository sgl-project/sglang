// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! `POST /v1/responses` end to end.

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
        transfer_group: None,
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

fn captured(mock: &MockWorker) -> Value {
    let b = mock.captured.lock().unwrap().last_body.clone().unwrap();
    serde_json::from_slice(&b).unwrap()
}

#[tokio::test]
async fn buffered_response_object() {
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone());
    let (status, ct, body) = post(
        ctx,
        "/v1/responses",
        json!({"model": MODEL, "input": "reply OK", "instructions": "be brief",
               "max_output_tokens": 512}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    assert!(ct.starts_with("application/json"));
    let r: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(r["object"], "response");
    assert!(r["id"].as_str().unwrap().starts_with("resp_"));
    assert_eq!(r["model"], MODEL);
    assert_eq!(r["status"], "completed");
    assert_eq!(r["instructions"], "be brief");
    assert_eq!(r["max_output_tokens"], 512);
    assert_eq!(r["output"][0]["type"], "message");
    assert_eq!(r["output"][0]["content"][0]["text"], "ok");
    assert!(r["usage"].is_object());

    let sent = captured(&mock);
    assert_eq!(
        sent["messages"],
        json!([{"role": "system", "content": "be brief"},
               {"role": "user", "content": "reply OK"}])
    );
    assert_eq!(sent["max_completion_tokens"], 512);
    assert!(sent.get("input").is_none());
    assert!(sent.get("instructions").is_none());
}

#[tokio::test]
async fn streaming_event_sequence() {
    let mock = MockWorker::start(vec![
        "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"hm\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"O\"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"con",
        "tent\":\"K\"},\"finish_reason\":\"stop\"}]}\n\n",
        "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":3,\"total_tokens\":8}}\n\n",
        "data: [DONE]\n\n",
    ])
    .await;
    let ctx = build_ctx(mock.url.clone());
    let (status, ct, body) = post(
        ctx,
        "/v1/responses",
        json!({"model": MODEL, "input": "reply OK", "stream": true}),
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
    for (i, (ev, data)) in events.iter().enumerate() {
        assert_eq!(data["type"], ev.as_str());
        assert_eq!(data["sequence_number"], i as u64);
    }
    assert_eq!(events[0].0, "response.created");
    assert_eq!(events[1].0, "response.in_progress");
    let deltas: String = events
        .iter()
        .filter(|(e, _)| e == "response.output_text.delta")
        .map(|(_, d)| d["delta"].as_str().unwrap())
        .collect();
    assert_eq!(deltas, "OK");
    let (last, done) = events.last().unwrap();
    assert_eq!(last, "response.completed");
    assert_eq!(done["response"]["output"][0]["type"], "reasoning");
    assert_eq!(done["response"]["output"][1]["content"][0]["text"], "OK");
    assert_eq!(done["response"]["usage"]["total_tokens"], 8);
    assert!(!text.contains("[DONE]"));

    assert_eq!(captured(&mock)["stream_options"]["include_usage"], true);
}

#[tokio::test]
async fn engine_error_is_rewrapped() {
    let mock = MockWorker::start_returning_error(
        StatusCode::BAD_REQUEST,
        json!({"object": "error", "message": "invalid json_schema", "type": "BadRequestError",
               "param": null, "code": 400}),
    )
    .await;
    let ctx = build_ctx(mock.url.clone());
    let (status, _, body) = post(
        ctx,
        "/v1/responses",
        json!({"model": MODEL, "input": "x",
               "text": {"format": {"type": "json_schema", "name": "s", "schema": "nope"}}}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["error"]["message"], "invalid json_schema");
}

#[tokio::test]
async fn router_side_rejections_are_structured() {
    let mock = MockWorker::start(vec![]).await;
    for (req, want) in [
        (
            json!({"model": MODEL, "input": "x", "previous_response_id": "resp_1"}),
            StatusCode::BAD_REQUEST,
        ),
        (
            json!({"model": MODEL, "input": []}),
            StatusCode::BAD_REQUEST,
        ),
        (
            json!({"model": "no-such-model", "input": "x"}),
            StatusCode::BAD_REQUEST,
        ),
    ] {
        let (status, _, body) =
            post(build_ctx(mock.url.clone()), "/v1/responses", req.clone()).await;
        assert_eq!(status, want, "{req}");
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert!(!v["error"]["message"].as_str().unwrap().is_empty(), "{req}");
    }
    assert!(
        mock.captured.lock().unwrap().last_body.is_none(),
        "rejected requests must not reach the engine"
    );
}

#[tokio::test]
async fn unknown_path_has_structured_404() {
    let mock = MockWorker::start(vec![]).await;
    let (status, _, body) = post(
        build_ctx(mock.url.clone()),
        "/v1/responses/not-found",
        json!({"model": MODEL, "input": "x"}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert!(v["error"]["message"]
        .as_str()
        .unwrap()
        .contains("/v1/responses/not-found"));
}
