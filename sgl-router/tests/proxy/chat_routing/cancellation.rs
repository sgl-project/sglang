// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;
use axum::{extract::State, http::HeaderMap, routing::post, Json, Router};
use serde_json::{json, Value};
use tokio::sync::mpsc;

type Event = (&'static str, Value);
type Events = mpsc::UnboundedSender<Event>;

async fn chat(State(events): State<Events>, Json(body): Json<Value>) -> (StatusCode, Body) {
    events.send(("chat", body.clone())).unwrap();
    if body["before_headers"] == true || (body["hold"] == true && body["stream"] != true) {
        std::future::pending::<()>().await;
    }
    let response = if body["hold"] == true {
        Body::from_stream(futures::stream::pending::<
            Result<bytes::Bytes, std::io::Error>,
        >())
    } else if body["stream"] == true {
        Body::from("data: [DONE]\n\n")
    } else {
        Body::from("{}")
    };
    (
        StatusCode::from_u16(body["status"].as_u64().unwrap_or(200) as u16).unwrap(),
        response,
    )
}

async fn abort(
    State(events): State<Events>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> StatusCode {
    assert_eq!(headers["authorization"], "Bearer test");
    events.send(("abort", body)).unwrap();
    StatusCode::INTERNAL_SERVER_ERROR // Abort failures must not affect the worker's breaker.
}

struct Harness {
    ctx: Arc<AppContext>,
    events: mpsc::UnboundedReceiver<Event>,
    server: tokio::task::JoinHandle<()>,
}

impl Harness {
    async fn new() -> Self {
        let (events, rx) = mpsc::unbounded_channel();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let ctx = build_ctx_with_worker(&format!("http://{}", listener.local_addr().unwrap()));
        let server = tokio::spawn(async move {
            axum::serve(
                listener,
                Router::new()
                    .route("/v1/chat/completions", post(chat))
                    .route("/abort_request", post(abort))
                    .with_state(events),
            )
            .await
            .unwrap();
        });
        Self {
            ctx,
            events: rx,
            server,
        }
    }

    async fn event(&mut self, expected: &str) -> Value {
        let (kind, body) = tokio::time::timeout(TEST_TIMEOUT, self.events.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(kind, expected);
        body
    }

    async fn quiet(&mut self) {
        assert!(
            tokio::time::timeout(Duration::from_millis(50), self.events.recv())
                .await
                .is_err()
        );
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        self.server.abort();
    }
}

pub(super) fn request(mut body: Value) -> Request<Body> {
    body["model"] = json!("tiny");
    Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("authorization", "Bearer test")
        .header("x-request-id", "reused-gateway-id")
        .body(Body::from(body.to_string()))
        .unwrap()
}

#[tokio::test]
async fn only_unfinished_requests_abort() {
    let mut h = Harness::new().await;
    for (stream, hold, before_headers) in [
        (false, false, false),
        (true, false, false),
        (false, true, false),
        (true, true, true),
        (true, true, false),
    ] {
        let task = tokio::spawn(build_router(h.ctx.clone()).oneshot(request(json!({
            "stream": stream, "hold": hold, "before_headers": before_headers,
        }))));
        let forwarded = h.event("chat").await;
        assert!(crate::common::is_engine_shaped_rid(
            forwarded["rid"].as_str().unwrap()
        ));
        let worker = h.ctx.registry.get(&WorkerId("w1".into())).unwrap();
        if hold && (!stream || before_headers) {
            worker.breaker.record_failure();
            worker.breaker.record_failure();
            task.abort();
            assert!(task.await.unwrap_err().is_cancelled());
        } else {
            let response = tokio::time::timeout(TEST_TIMEOUT, task)
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            if hold {
                drop(response); // Silent upstream: cancellation must not wait for a token.
            } else {
                response.into_body().collect().await.unwrap();
            }
        }
        if hold {
            assert_eq!(
                h.event("abort").await,
                json!({"rid": forwarded["rid"], "abort_all": false})
            );
        }
        h.quiet().await;
        assert_eq!(worker.breaker.snapshot().state_code, 0);
        worker.breaker.record_success();
    }
}

#[tokio::test]
async fn caller_ids_fan_out_and_rejected_streams_do_not_abort() {
    let mut h = Harness::new().await;
    for fields in [
        json!({"rid":"a"}),
        json!({"rid":["a","b"]}),
        json!({"n":2}),
        json!({"status":400}),
        json!({"status":429}),
        json!({"status":500}),
        json!({"status":503}),
    ] {
        let mut body = json!({"stream":true, "hold":true});
        body.as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        let response = build_router(h.ctx.clone())
            .oneshot(request(body))
            .await
            .unwrap();
        let forwarded = h.event("chat").await;
        if fields.get("status").is_none() {
            assert_eq!(forwarded.get("rid"), fields.get("rid"));
        }
        drop(response);
        h.quiet().await;
    }
}

#[tokio::test]
async fn concurrent_requests_with_the_same_header_get_distinct_abort_ids() {
    let mut h = Harness::new().await;
    let mut tasks = Vec::new();
    for _ in 0..2 {
        tasks.push(tokio::spawn(
            build_router(h.ctx.clone()).oneshot(request(json!({"hold":true}))),
        ));
    }
    let mut rids = Vec::new();
    for _ in 0..2 {
        rids.push(h.event("chat").await["rid"].clone());
    }
    assert_ne!(rids[0], rids[1]);
    for task in tasks {
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
    }
    for _ in 0..2 {
        let aborted = h.event("abort").await;
        let index = rids.iter().position(|rid| rid == &aborted["rid"]).unwrap();
        rids.remove(index);
    }
    h.quiet().await;
}
