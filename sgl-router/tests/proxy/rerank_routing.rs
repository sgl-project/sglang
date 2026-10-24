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
async fn rerank_forwards_put_and_rejects_prefill_decode_workers() {
    let (engine, prefill, decode) = (
        MockWorker::start(vec![]).await,
        MockWorker::start(vec![]).await,
        MockWorker::start(vec![]).await,
    );
    let body = json!({"query": "hi", "documents": ["yo", [{"type": "text", "text": "x"}]]});
    let send = |workers: &[(&MockWorker, WorkerMode)]| {
        let request = Request::put("/v1/rerank")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        radix_router(workers, HashTree::new()).oneshot(request)
    };

    let response = send(&[(&engine, WorkerMode::Plain)]).await.unwrap();
    let response = collect_body(response.into_body()).await;
    assert_eq!(response, json!([{"score": 0.5, "index": 0}]).to_string());
    assert_eq!(engine.captured_json().await, body);

    let pd = [
        (&prefill, WorkerMode::Prefill),
        (&decode, WorkerMode::Decode),
    ];
    let response = send(&pd).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(prefill.captured.lock().unwrap().last_body.is_none());
}
