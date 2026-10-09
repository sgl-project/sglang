// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! API profiles end to end, through the `stepfun-step5` preset and a file.

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

fn ctx(url: &str, flags: &[&str]) -> Arc<AppContext> {
    let cfg = config(flags);
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    let _ = registry.add(WorkerSpec {
        id: WorkerId(url.into()),
        url: url.into(),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId(MODEL.into())],
        bootstrap_port: None,
        transfer_group: None,
    });
    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
}

async fn post(ctx: Arc<AppContext>, path: &str, body: Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let res = build_router(ctx).oneshot(req).await.unwrap();
    let status = res.status();
    let bytes = collect_body(res.into_body()).await;
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

fn sent(mock: &MockWorker) -> Option<Value> {
    let b = mock.captured.lock().unwrap().last_body.clone()?;
    Some(serde_json::from_slice(&b).unwrap())
}

const STEP5: &[&str] = &["--api-profile", "stepfun-step5"];

#[tokio::test]
async fn chat_defaults_normalize_and_clamp() {
    let mock = MockWorker::start(vec![]).await;
    let (status, _) = post(
        ctx(&mock.url, STEP5),
        "/v1/chat/completions",
        json!({"model": MODEL, "messages": [{"role": "user", "content": "hi"}],
               "top_k": 0, "max_tokens": 70000}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let body = sent(&mock).unwrap();
    assert_eq!(body["top_k"], -1);
    assert_eq!(body["max_tokens"], 64000);
    assert_eq!(body["temperature"], 1.0);
    assert_eq!(body["top_p"], 0.95);
}

#[tokio::test]
async fn out_of_range_is_400_with_the_profile_error_type_on_every_protocol() {
    let mock = MockWorker::start(vec![]).await;
    let user = json!([{"role": "user", "content": "hi"}]);
    for (path, body) in [
        (
            "/v1/chat/completions",
            json!({"model": MODEL, "messages": user, "temperature": 2.1}),
        ),
        (
            "/v1/responses",
            json!({"model": MODEL, "input": "hi", "temperature": 2.1}),
        ),
        (
            "/v1/messages",
            json!({"model": MODEL, "max_tokens": 8, "messages": user, "temperature": 2.1}),
        ),
    ] {
        let (status, v) = post(ctx(&mock.url, STEP5), path, body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{path}");
        assert_eq!(v["error"]["type"], "request_params_invalid", "{path}: {v}");
        assert!(
            v["error"]["message"]
                .as_str()
                .unwrap()
                .contains("between 0 and 2"),
            "{path}: {v}"
        );
    }
    assert!(sent(&mock).is_none(), "rejected before the engine");
}

#[tokio::test]
async fn responses_budget_is_clamped() {
    let mock = MockWorker::start(vec![]).await;
    let (status, _) = post(
        ctx(&mock.url, STEP5),
        "/v1/responses",
        json!({"model": MODEL, "input": "hi", "max_output_tokens": 70000}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(sent(&mock).unwrap()["max_completion_tokens"], 64000);
}

#[tokio::test]
async fn engine_400_is_retyped() {
    let mock = MockWorker::start_returning_error(
        StatusCode::BAD_REQUEST,
        json!({"object": "error", "message": "bad schema", "type": "BadRequestError", "code": 400}),
    )
    .await;
    let (status, v) = post(
        ctx(&mock.url, STEP5),
        "/v1/chat/completions",
        json!({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(v["type"], "request_params_invalid");
    assert_eq!(v["message"], "bad schema");
}

#[tokio::test]
async fn image_limit() {
    let mock = MockWorker::start(vec![]).await;
    let img = json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}});
    let (status, v) = post(
        ctx(&mock.url, STEP5),
        "/v1/chat/completions",
        json!({"model": MODEL, "messages": [{"role": "user", "content": vec![img; 61]}]}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("too many images: 61"),
        "{v}"
    );
}

#[tokio::test]
async fn reasoning_format_general_renames_the_stream_field() {
    let mock = MockWorker::start(vec![
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"hm\"}}]}\n\n",
        "data: [DONE]\n\n",
    ])
    .await;
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"model": MODEL, "stream": true, "reasoning_format": "general",
                   "messages": [{"role": "user", "content": "hi"}]})
            .to_string(),
        ))
        .unwrap();
    let res = build_router(ctx(&mock.url, STEP5))
        .oneshot(req)
        .await
        .unwrap();
    let text = String::from_utf8(collect_body(res.into_body()).await.to_vec()).unwrap();
    assert!(
        text.contains("\"reasoning\":\"hm\"") && !text.contains("reasoning_content"),
        "{text}"
    );

    let (status, _) = post(
        ctx(&mock.url, STEP5),
        "/v1/chat/completions",
        json!({"model": MODEL, "reasoning_format": "bogus", "messages": []}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn profile_file_extends_a_preset_and_legacy_flags_win() {
    let dir = std::env::temp_dir().join(format!("api-profile-proxy-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("profile.yaml");
    std::fs::write(
        &file,
        "extends: stepfun-step5\nmodels: {aliases: [step-5-preview]}\nprotocols: {responses: false}\n",
    )
    .unwrap();
    let path = file.to_str().unwrap();
    let mock = MockWorker::start(vec![]).await;
    let flags = ["--api-profile-file", path, "--max-output-tokens", "100"];

    let (status, _) = post(
        ctx(&mock.url, &flags),
        "/v1/chat/completions",
        json!({"model": "step-5-preview", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 500}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let body = sent(&mock).unwrap();
    assert_eq!(body["model"], MODEL, "alias rewritten to the served id");
    assert_eq!(
        body["max_tokens"], 100,
        "flag cap wins, preset's clamp mode kept"
    );

    let (status, _) = post(
        ctx(&mock.url, &flags),
        "/v1/responses",
        json!({"model": MODEL, "input": "x"}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "disabled protocol");
}

#[tokio::test]
async fn oversized_body_gets_a_json_413_in_each_protocol_envelope() {
    let path = std::env::temp_dir().join(format!("api-profile-413-{}.yaml", std::process::id()));
    std::fs::write(&path, "limits: {max_body_bytes: 1KiB}\n").unwrap();
    let mock = MockWorker::start(vec![]).await;
    let flags = ["--api-profile-file", path.to_str().unwrap()];
    let user = json!([{"role": "user", "content": "x".repeat(2048)}]);

    let (status, v) = post(
        ctx(&mock.url, &flags),
        "/v1/chat/completions",
        json!({"model": MODEL, "messages": user}),
    )
    .await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(v["error"]["code"], "request_too_large", "{v}");
    assert!(
        v["error"]["message"].as_str().unwrap().contains("1 KiB"),
        "{v}"
    );

    let (status, v) = post(
        ctx(&mock.url, &flags),
        "/v1/messages",
        json!({"model": MODEL, "max_tokens": 8, "messages": user}),
    )
    .await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(v["type"], "error", "{v}");
    assert_eq!(v["error"]["type"], "request_too_large", "{v}");
    assert!(sent(&mock).is_none());
}

#[tokio::test]
async fn thinking_blocks_on_request_hides_unrequested_thinking() {
    let path = std::env::temp_dir().join(format!("api-profile-think-{}.yaml", std::process::id()));
    std::fs::write(
        &path,
        "extends: stepfun-step5\nmessages: {thinking_blocks: on_request}\n",
    )
    .unwrap();
    let mock = MockWorker::start(vec![
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"hm\"},\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":1}}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"OK\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2}}\n\n",
        "data: [DONE]\n\n",
    ])
    .await;
    let flags = ["--api-profile-file", path.to_str().unwrap()];
    let stream_text = |thinking: Option<Value>| {
        let mut body = json!({"model": MODEL, "max_tokens": 64, "stream": true,
                              "messages": [{"role": "user", "content": "hi"}]});
        if let Some(t) = thinking {
            body["thinking"] = t;
        }
        let req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let app = build_router(ctx(&mock.url, &flags));
        async move {
            String::from_utf8(
                collect_body(app.oneshot(req).await.unwrap().into_body())
                    .await
                    .to_vec(),
            )
            .unwrap()
        }
    };
    let hidden = stream_text(None).await;
    assert!(
        !hidden.contains("thinking") && hidden.contains("\"text\":\"OK\""),
        "{hidden}"
    );
    let shown = stream_text(Some(json!({"type": "enabled", "budget_tokens": 1024}))).await;
    assert!(shown.contains("thinking_delta"), "{shown}");
}
