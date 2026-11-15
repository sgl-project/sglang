// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Seeded HTTP chat fuzz smoke tests; no model, GPU, or network downloads.
//!
//! Run with `cargo test --locked --test proxy chat_routing::fuzz_smoke:: -- --ignored`.
//! Failures print the seed, case index, and complete input. Replay the same run with
//! `SGL_ROUTER_FUZZ_SEED=<seed>`; `SGL_ROUTER_FUZZ_CASES` defaults to 64 (maximum 256).
//! This is generated integration coverage, not coverage-guided fuzzing. It drives
//! the legacy regular HTTP route, with real outbound HTTP to a local mock engine.
//! Mock body frames may be coalesced by HTTP; they are not TCP packet boundaries.

use super::*;
use axum::extract::State;
use axum::routing::post;
use bytes::Bytes;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::VecDeque;
use std::sync::Mutex;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

const CASE_TIMEOUT: Duration = Duration::from_secs(2);
const RUN_TIMEOUT: Duration = Duration::from_secs(30);
const VALID_CHAT_RESPONSE: &[u8] =
    br#"{"choices":[{"message":{"role":"assistant","content":"ok"}}]}"#;

struct RequestCase {
    name: &'static str,
    body: Vec<u8>,
    expected_status: StatusCode,
}

struct ResponseCase {
    name: &'static str,
    streaming: bool,
    payload: Vec<u8>,
}

#[derive(Clone, Debug)]
struct MockResponse {
    streaming: bool,
    chunks: Vec<Vec<u8>>,
}

impl MockResponse {
    fn valid_chat_response() -> Self {
        Self {
            streaming: false,
            chunks: vec![VALID_CHAT_RESPONSE.to_vec()],
        }
    }
}

#[derive(Clone)]
struct MockEngineState {
    responses: Arc<Mutex<VecDeque<MockResponse>>>,
    requests: Arc<Mutex<Vec<Bytes>>>,
}

/// A local HTTP engine that returns the queued responses in request order.
struct MockEngine {
    url: String,
    state: MockEngineState,
    shutdown: Option<oneshot::Sender<()>>,
    task: JoinHandle<std::io::Result<()>>,
}

impl MockEngine {
    async fn start(responses: Vec<MockResponse>) -> Self {
        async fn chat(
            State(state): State<MockEngineState>,
            body: Bytes,
        ) -> axum::response::Response {
            state.requests.lock().unwrap().push(body);
            let reply = state
                .responses
                .lock()
                .unwrap()
                .pop_front()
                .expect("unexpected upstream request");
            let frames = reply
                .chunks
                .into_iter()
                .map(|chunk| Ok::<_, std::io::Error>(Bytes::from(chunk)));
            axum::response::Response::builder()
                .header(
                    "content-type",
                    if reply.streaming {
                        "text/event-stream"
                    } else {
                        "application/json"
                    },
                )
                .body(Body::from_stream(futures::stream::iter(frames)))
                .unwrap()
        }
        let state = MockEngineState {
            responses: Arc::new(Mutex::new(responses.into())),
            requests: Arc::new(Mutex::new(Vec::new())),
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let app = axum::Router::new()
            .route("/v1/chat/completions", post(chat))
            .with_state(state.clone());
        let (shutdown, rx) = oneshot::channel();
        let task = tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = rx.await;
                })
                .await
        });
        Self {
            url,
            state,
            shutdown: Some(shutdown),
            task,
        }
    }

    fn request_count(&self) -> usize {
        self.state.requests.lock().unwrap().len()
    }

    async fn stop(mut self) {
        let _ = self.shutdown.take().unwrap().send(());
        tokio::time::timeout(CASE_TIMEOUT, &mut self.task)
            .await
            .expect("mock shutdown hung")
            .expect("mock task panicked")
            .expect("mock server failed");
        assert!(
            self.state.responses.lock().unwrap().is_empty(),
            "not all scripted responses were used"
        );
    }
}

impl Drop for MockEngine {
    fn drop(&mut self) {
        // Also runs when a case fails or the enclosing timeout drops its future.
        self.task.abort();
    }
}

fn run_config() -> (u64, usize) {
    let seed = std::env::var("SGL_ROUTER_FUZZ_SEED")
        .map(|s| s.parse().expect("seed must be a u64"))
        .unwrap_or(50045);
    let cases = std::env::var("SGL_ROUTER_FUZZ_CASES")
        .map(|s| s.parse().expect("cases must be an integer"))
        .unwrap_or(64);
    assert!(
        (1..=256).contains(&cases),
        "cases must be between 1 and 256"
    );
    (seed, cases)
}

fn chat_request_body(streaming: bool, content: &str) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "model": "tiny", "messages": [{"role": "user", "content": content}], "stream": streaming,
    }))
    .unwrap()
}

async fn send_chat_request(app: &axum::Router, body: Vec<u8>) -> (StatusCode, Bytes) {
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    // Body errors must fail the test, including panics caught by the SSE pump.
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("router response body failed")
        .to_bytes();
    (status, bytes)
}

async fn assert_router_recovers_and_becomes_idle(app: &axum::Router, ctx: &AppContext) {
    let (status, body) = send_chat_request(app, chat_request_body(false, "recovery")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body.as_ref(), VALID_CHAT_RESPONSE);
    loop {
        let workers_idle = ctx
            .registry
            .workers_for(&ModelId("tiny".into()))
            .iter()
            .all(|w| w.router_inflight_load() == 0);
        if workers_idle && ctx.router_inflight_load.inflight_count() == 0 {
            break;
        }
        tokio::task::yield_now().await;
    }
}

#[tokio::test]
#[ignore = "run explicitly in the time-limited HTTP chat fuzz CI step"]
async fn generated_requests_are_validated_and_router_recovers() {
    let (seed, cases) = run_config();
    let mut rng = StdRng::seed_from_u64(seed);
    tokio::time::timeout(RUN_TIMEOUT, async {
        for index in 0..cases {
            let case = generate_request_case(&mut rng, index);
            eprintln!(
                "request seed={seed} case={index} kind={} body={:?}",
                case.name, case.body
            );
            tokio::time::timeout(CASE_TIMEOUT, async {
                let forwarded_requests = usize::from(case.expected_status == StatusCode::OK);
                let responses = vec![MockResponse::valid_chat_response(); forwarded_requests + 1];
                let engine = MockEngine::start(responses).await;
                let ctx = build_ctx_with_worker(&engine.url);
                let app = build_router(ctx.clone());

                let (status, _) = send_chat_request(&app, case.body).await;
                assert_eq!(status, case.expected_status, "{}", case.name);
                assert_eq!(engine.request_count(), forwarded_requests, "{}", case.name);

                assert_router_recovers_and_becomes_idle(&app, &ctx).await;
                assert_eq!(engine.request_count(), forwarded_requests + 1);
                engine.stop().await;
            })
            .await
            .expect("generated request case hung");
        }
    })
    .await
    .expect("request fuzz run exceeded its time budget");
}

#[tokio::test]
#[ignore = "run explicitly in the time-limited HTTP chat fuzz CI step"]
async fn generated_responses_preserve_bytes_and_router_recovers() {
    let (seed, cases) = run_config();
    let mut rng = StdRng::seed_from_u64(seed);
    tokio::time::timeout(RUN_TIMEOUT, async {
        for index in 0..cases {
            let case = generate_response_case(&mut rng, index);
            let chunks = split_into_random_chunks(&mut rng, &case.payload);
            eprintln!(
                "response seed={seed} case={index} kind={} streaming={} payload={:?} chunks={chunks:?}",
                case.name, case.streaming, case.payload
            );
            tokio::time::timeout(CASE_TIMEOUT, async {
                let engine = MockEngine::start(vec![
                    MockResponse {
                        streaming: case.streaming,
                        chunks: vec![case.payload.clone()],
                    },
                    MockResponse {
                        streaming: case.streaming,
                        chunks,
                    },
                    MockResponse::valid_chat_response(),
                ])
                .await;
                let ctx = build_ctx_with_worker(&engine.url);
                let app = build_router(ctx.clone());

                for delivery in ["whole body", "chunked body"] {
                    let body = chat_request_body(case.streaming, "generated response");
                    let (status, received) = send_chat_request(&app, body).await;
                    assert_eq!(status, StatusCode::OK, "{} / {delivery}", case.name);
                    // This route forwards upstream bodies without validating JSON or SSE.
                    assert_eq!(received.as_ref(), case.payload, "{} / {delivery}", case.name);
                }

                assert_router_recovers_and_becomes_idle(&app, &ctx).await;
                assert_eq!(engine.request_count(), 3);
                engine.stop().await;
            })
            .await
            .expect("generated response case hung");
        }
    })
    .await
    .expect("response fuzz run exceeded its time budget");
}

fn generate_request_case(rng: &mut StdRng, index: usize) -> RequestCase {
    let content: String = (0..rng.gen_range(0..64))
        .map(|_| char::from(rng.gen_range(32u8..127)))
        .collect();
    let mut body = chat_request_body(false, &content);
    let (name, expected_status) = match index % 4 {
        0 => ("valid request", StatusCode::OK),
        1 => {
            body.truncate(rng.gen_range(0..body.len()));
            ("truncated JSON", StatusCode::BAD_REQUEST)
        }
        2 => {
            body = vec![b'['];
            body.extend((0..rng.gen_range(0..128)).map(|_| rng.gen::<u8>()));
            body.push(0xff);
            ("invalid UTF-8", StatusCode::BAD_REQUEST)
        }
        _ => {
            body = serde_json::to_vec(&serde_json::json!([content])).unwrap();
            ("array instead of chat request", StatusCode::BAD_REQUEST)
        }
    };
    RequestCase {
        name,
        body,
        expected_status,
    }
}

fn generate_response_case(rng: &mut StdRng, index: usize) -> ResponseCase {
    let (name, payload) = match index % 6 {
        0 => (
            "valid SSE",
            concat!(
                "data: ",
                r#"{"choices":[{"delta":{"content":"你好🌍"}}]}"#,
                "\n\ndata: [DONE]\n\n",
            )
            .as_bytes()
            .to_vec(),
        ),
        1 => ("valid JSON", VALID_CHAT_RESPONSE.to_vec()),
        2 => (
            "malformed SSE",
            b"data: {not json}\n\ninvalid SSE\r\ndata: [DONE]".to_vec(),
        ),
        3 => ("truncated JSON", br#"{"choices":["#.to_vec()),
        _ => (
            "arbitrary bytes",
            (0..rng.gen_range(0..512))
                .map(|_| rng.gen::<u8>())
                .collect(),
        ),
    };
    ResponseCase {
        name,
        streaming: index.is_multiple_of(2),
        payload,
    }
}

fn split_into_random_chunks(rng: &mut StdRng, payload: &[u8]) -> Vec<Vec<u8>> {
    let mut chunks = Vec::new();
    let mut offset = 0;
    while offset < payload.len() {
        let end = (offset + rng.gen_range(1..=16)).min(payload.len());
        chunks.push(payload[offset..end].to_vec());
        offset = end;
    }
    chunks
}
