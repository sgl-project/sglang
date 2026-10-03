// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! API profiles end to end: from the YAML an operator mounts to what the
//! client gets back and what the engine receives.

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

fn config(profile: &str, flags: &[&str]) -> Config {
    let path = std::env::temp_dir().join(format!(
        "api-profile-{}.yaml",
        uuid::Uuid::new_v4().simple()
    ));
    std::fs::write(&path, profile).unwrap();
    let path = path.display().to_string();
    let mut argv = vec![
        "sgl-router",
        "--model-id",
        MODEL,
        "--tokenizer-path",
        "tests/fixtures/tiny_tokenizer.json",
        "--worker-urls",
        "http://placeholder:0",
        "--api-profile-file",
        &path,
    ];
    argv.extend_from_slice(flags);
    <Cli as clap::Parser>::parse_from(argv)
        .into_config()
        .expect("flags must parse")
}

fn build_ctx(url: String, profile: &str, flags: &[&str]) -> Arc<AppContext> {
    let cfg = config(profile, flags);
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    let _ = registry.add(WorkerSpec {
        id: WorkerId(url.clone()),
        url,
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId(MODEL.into())],
        ..Default::default()
    });
    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
}

async fn chat(ctx: Arc<AppContext>, body: Vec<u8>) -> (StatusCode, Vec<u8>) {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let res = build_router(ctx).oneshot(req).await.unwrap();
    let status = res.status();
    (status, collect_body(res.into_body()).await.to_vec())
}

fn request(model: &str, extra: Value) -> Vec<u8> {
    let mut body = json!({"model": model, "messages": [{"role": "user", "content": "hi"}]});
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    serde_json::to_vec(&body).unwrap()
}

#[tokio::test]
async fn output_budget_is_injected_rejected_or_clamped() {
    let profile = "output: {max_tokens: {cap: 100, default: 10}}";
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone(), profile, &[]);
    let (status, _) = chat(ctx.clone(), request(MODEL, json!({}))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(mock.captured_json().await["max_tokens"], 10);

    let (status, body) = chat(ctx, request(MODEL, json!({"max_tokens": 101}))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(String::from_utf8_lossy(&body).contains("max_tokens must be at most 100"));

    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(
        mock.url.clone(),
        "output: {max_tokens: {cap: 100, on_exceed: clamp}}",
        &[],
    );
    let (status, _) = chat(ctx, request(MODEL, json!({"max_completion_tokens": 500}))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(mock.captured_json().await["max_completion_tokens"], 100);
}

#[tokio::test]
async fn alias_routes_to_the_served_model() {
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone(), "models: {aliases: [tiny-preview]}", &[]);
    let (status, _) = chat(ctx.clone(), request("tiny-preview", json!({}))).await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = chat(ctx, request("other", json!({}))).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn limits_and_error_type() {
    let profile = "limits: {max_images: 1, max_body_bytes: 2KiB}\n\
                   errors: {bad_request_type: request_params_invalid}";
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone(), profile, &[]);
    let image = json!({"type": "image_url", "image_url": {"url": "https://x/i.png"}});
    let body = serde_json::to_vec(&json!({"model": MODEL, "messages": [
        {"role": "user", "content": [image.clone(), image]}]}))
    .unwrap();
    let (status, body) = chat(ctx.clone(), body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["error"]["type"], "request_params_invalid");
    assert!(v["error"]["message"].as_str().unwrap().contains("got 2"));

    let big = request(MODEL, json!({"user": "x".repeat(4096)}));
    let (status, _) = chat(ctx, big).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
    assert!(mock.captured.lock().unwrap().last_body.is_none());
}

#[tokio::test]
async fn sampling_section_and_flag_precedence() {
    let profile = "sampling: {params: {top_p: 0.95}}";
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone(), profile, &[]);
    let (status, _) = chat(ctx.clone(), request(MODEL, json!({}))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(mock.captured_json().await["top_p"], 0.95);
    let (status, _) = chat(ctx, request(MODEL, json!({"top_p": 0.5}))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // The flag replaces the profile's sampling section as a whole.
    let cfg = config(profile, &["--override-sampling-params", r#"{"top_k": 5}"#]);
    assert!(cfg.model.profile.sampling.is_none());
    let params = &cfg.model.sampling_overrides.params;
    assert_eq!(params.len(), 1);
    assert!(params.keys().all(|f| f.wire_name() == "top_k"));
}
