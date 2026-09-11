// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Verify forwarding eligibility and message preservation through the HTTP handler.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::{json, Value};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry;
use sgl_router::policies::kv_events::{BlockSizeOracle, HashTree};
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

use crate::common::cache_aware_fixture::{config, MODEL};
use crate::common::mock_worker::MockWorker;

fn build_ctx(url: String) -> Arc<AppContext> {
    let cfg = config();
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    assert!(
        tokenizers.has_chat_encoder(MODEL),
        "deepseek-v4 model id must auto-attach the built-in chat encoder"
    );
    let registry = Arc::new(WorkerRegistry::default());
    let _ = registry.add(WorkerSpec {
        id: WorkerId(url.clone()),
        url,
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId(MODEL.into())],
        bootstrap_port: None,
    });
    // Use the configured tokenizer so the chat path can emit input_ids.
    let policies =
        Arc::new(build_registry(&cfg, Arc::new(HashTree::new()), BlockSizeOracle::new()).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
}

async fn send(ctx: Arc<AppContext>, body: Value) -> StatusCode {
    let app = build_router(ctx);
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    app.oneshot(req).await.unwrap().status()
}

fn captured(mock: &MockWorker) -> Value {
    let b = mock
        .captured
        .lock()
        .unwrap()
        .last_body
        .clone()
        .expect("worker captured a request body");
    serde_json::from_slice(&b).expect("captured body is valid JSON")
}

#[tokio::test]
async fn plain_chat_forwards_input_ids_and_keeps_messages() {
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone());
    let status = send(
        ctx,
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hello there friend"}],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let body = captured(&mock);
    let ids = body.get("input_ids").and_then(|v| v.as_array());
    assert!(
        ids.is_some_and(|a| !a.is_empty()),
        "engine must receive non-empty input_ids; got {body}"
    );
    assert!(
        body.get("messages").is_some(),
        "messages must be retained alongside input_ids; got {body}"
    );
}

#[tokio::test]
async fn guarded_requests_render_without_forwarding_ids() {
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone());
    for options in [
        json!({"tools": [{"type": "function", "function": {"name": "f"}}]}),
        json!({"chat_template_kwargs": {"thinking": true}}),
        json!({"reasoning_effort": "none"}),
        json!({"chat_template": "custom"}),
    ] {
        let mut request = json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
        });
        request
            .as_object_mut()
            .unwrap()
            .extend(options.as_object().unwrap().clone());
        let tokens = sgl_router::policies::request_tokens_for(
            &ctx.tokenizers,
            &ModelId(MODEL.into()),
            &request,
        )
        .expect("request renders for routing");
        assert!(tokens.chat_rendered);
        assert_eq!(send(ctx.clone(), request.clone()).await, StatusCode::OK);
        assert_eq!(captured(&mock), request);
    }
}

#[tokio::test]
async fn reasoning_history_omits_input_ids_and_preserves_messages() {
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone());
    let messages = json!([
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
        {"role": "user", "content": "U2"}
    ]);
    let status = send(ctx, json!({"model": MODEL, "messages": messages})).await;
    assert_eq!(status, StatusCode::OK);

    let body = captured(&mock);
    assert!(
        body.get("input_ids").is_none(),
        "the engine must render reasoning history with its own template; got {body}"
    );
    assert_eq!(body["messages"], messages);
}

#[tokio::test]
async fn multimodal_request_omits_input_ids() {
    let mock = MockWorker::start(vec![]).await;
    let ctx = build_ctx(mock.url.clone());
    let status = send(
        ctx,
        json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": "x"}]}],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let body = captured(&mock);
    assert!(
        body.get("input_ids").is_none(),
        "multimodal requests must not forward input_ids; got {body}"
    );
}
