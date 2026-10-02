// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! SGLang's `/v1/rerank`: the engine's request and response, forwarded as sent.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::json;
use sgl_router::discovery::WorkerMode;
use sgl_router::state::kv_events::HashTree;
use tower::ServiceExt;

use crate::common::cache_aware_fixture::radix_router;
use crate::common::mock_worker::MockWorker;
use crate::common::streaming::collect_body;

#[tokio::test]
async fn rerank_forwards_the_body_as_sent() {
    let engine = MockWorker::start(vec![]).await;
    let (prefill, decode) = (
        MockWorker::start(vec![]).await,
        MockWorker::start(vec![]).await,
    );
    let plain = radix_router(&[(&engine, WorkerMode::Plain)], HashTree::new());
    let pd = radix_router(
        &[
            (&prefill, WorkerMode::Prefill),
            (&decode, WorkerMode::Decode),
        ],
        HashTree::new(),
    );
    let body =
        json!({"query": "hi", "documents": ["yo", [{"type": "text", "text": "x"}]], "top_n": 1});
    let send = |app: &axum::Router, method: &str| {
        let request = Request::builder()
            .method(method)
            .uri("/v1/rerank")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        app.clone().oneshot(request)
    };

    for method in ["POST", "PUT"] {
        let response = send(&plain, method).await.unwrap();
        let response = collect_body(response.into_body()).await;
        assert_eq!(response, json!([{"score": 0.5, "index": 0}]).to_string());
        assert_eq!(engine.captured_json().await, body);
    }
    // Prefill and decode engines serve no rerank.
    let response = send(&pd, "POST").await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(prefill.captured.lock().unwrap().last_body.is_none());
}
