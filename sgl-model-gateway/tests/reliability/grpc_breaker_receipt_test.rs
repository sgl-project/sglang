use std::{
    collections::VecDeque,
    convert::Infallible,
    net::SocketAddr,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use anyhow::Result;
use axum::{
    body::{to_bytes, Body},
    http::{header::CONTENT_TYPE, Request, Response, StatusCode},
    routing::post,
    Router,
};
use bytes::Bytes;
use futures_util::stream;
use http_body::Frame;
use http_body_util::StreamBody;
use prost::Message;
use smg::{
    config::{RetryConfig, RouterConfig},
    core::{BasicWorkerBuilder, ConnectionMode, ModelCard, RuntimeType, Worker, WorkerType},
    routers::{grpc::client::GrpcClient, RouterFactory, RouterTrait},
    tokenizer::traits::{Decoder, Encoder, Encoding, SpecialTokens, TokenIdType, Tokenizer},
};
use smg_grpc_client::sglang_proto as sglang;
use tokio::{net::TcpListener, task::JoinHandle, time};
use tower::ServiceExt;

use crate::common;

const GENERATE_PATH: &str = "/sglang.grpc.scheduler.SglangScheduler/Generate";
const ABORT_PATH: &str = "/sglang.grpc.scheduler.SglangScheduler/Abort";

#[derive(Default)]
struct PlainTokenizer {
    special: SpecialTokens,
}

impl Encoder for PlainTokenizer {
    fn encode(&self, input: &str, _add_special_tokens: bool) -> Result<Encoding> {
        Ok(Encoding::Plain(input.bytes().map(u32::from).collect()))
    }

    fn encode_batch(&self, inputs: &[&str], add_special_tokens: bool) -> Result<Vec<Encoding>> {
        inputs
            .iter()
            .map(|input| self.encode(input, add_special_tokens))
            .collect()
    }
}

impl Decoder for PlainTokenizer {
    fn decode(&self, token_ids: &[TokenIdType], _skip_special_tokens: bool) -> Result<String> {
        Ok(token_ids
            .iter()
            .map(|id| char::from_u32(*id).unwrap_or('\u{fffd}'))
            .collect())
    }
}

impl Tokenizer for PlainTokenizer {
    fn vocab_size(&self) -> usize {
        256
    }

    fn get_special_tokens(&self) -> &SpecialTokens {
        &self.special
    }

    fn token_to_id(&self, token: &str) -> Option<TokenIdType> {
        token.bytes().next().map(u32::from)
    }

    fn id_to_token(&self, id: TokenIdType) -> Option<String> {
        char::from_u32(id).map(|character| character.to_string())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Clone)]
enum Behavior {
    HeadersStatus(u32),
    Messages { messages: Vec<Vec<u8>>, status: u32 },
}

struct Upstream {
    address: SocketAddr,
    generate_calls: Arc<AtomicUsize>,
    abort_calls: Arc<AtomicUsize>,
    server: JoinHandle<()>,
}

impl Drop for Upstream {
    fn drop(&mut self) {
        self.server.abort();
    }
}

fn grpc_frame(payload: Vec<u8>) -> Bytes {
    let mut frame = Vec::with_capacity(payload.len() + 5);
    frame.push(0);
    frame.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    frame.extend_from_slice(&payload);
    Bytes::from(frame)
}

fn response(behavior: Behavior) -> Response<Body> {
    let builder = Response::builder()
        .status(200)
        .header("content-type", "application/grpc");
    match behavior {
        Behavior::HeadersStatus(status) => builder
            .header("grpc-status", status.to_string())
            .body(Body::empty())
            .unwrap(),
        Behavior::Messages { messages, status } => {
            let mut trailers = http::HeaderMap::new();
            trailers.insert("grpc-status", status.to_string().parse().unwrap());
            let frames = messages
                .into_iter()
                .map(|message| Ok::<_, Infallible>(Frame::data(grpc_frame(message))))
                .chain(std::iter::once(Ok(Frame::trailers(trailers))));
            builder
                .body(Body::new(StreamBody::new(stream::iter(frames))))
                .unwrap()
        }
    }
}

async fn start_upstream(script: Vec<Behavior>) -> Upstream {
    let script = Arc::new(Mutex::new(VecDeque::from(script)));
    let generate_calls = Arc::new(AtomicUsize::new(0));
    let abort_calls = Arc::new(AtomicUsize::new(0));
    let generate_counter = Arc::clone(&generate_calls);
    let abort_counter = Arc::clone(&abort_calls);
    let app = Router::new()
        .route(
            GENERATE_PATH,
            post(move || {
                let script = Arc::clone(&script);
                let generate_counter = Arc::clone(&generate_counter);
                async move {
                    generate_counter.fetch_add(1, Ordering::SeqCst);
                    let behavior = script
                        .lock()
                        .unwrap()
                        .pop_front()
                        .expect("unexpected generate attempt");
                    response(behavior)
                }
            }),
        )
        .route(
            ABORT_PATH,
            post(move || {
                let abort_counter = Arc::clone(&abort_counter);
                async move {
                    abort_counter.fetch_add(1, Ordering::SeqCst);
                    Response::builder()
                        .status(200)
                        .header("content-type", "application/grpc")
                        .header("grpc-status", "0")
                        .body(Body::empty())
                        .unwrap()
                }
            }),
        );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    Upstream {
        address,
        generate_calls,
        abort_calls,
        server,
    }
}

fn complete() -> Vec<u8> {
    sglang::GenerateResponse {
        response: Some(sglang::generate_response::Response::Complete(
            sglang::GenerateComplete {
                output_ids: vec![65],
                finish_reason: "stop".to_string(),
                index: 0,
                ..Default::default()
            },
        )),
        ..Default::default()
    }
    .encode_to_vec()
}

fn chunk() -> Vec<u8> {
    sglang::GenerateResponse {
        response: Some(sglang::generate_response::Response::Chunk(
            sglang::GenerateStreamChunk {
                token_ids: vec![65],
                index: 0,
                ..Default::default()
            },
        )),
        ..Default::default()
    }
    .encode_to_vec()
}

struct Harness {
    app: Router,
    worker: Arc<dyn Worker>,
    upstream: Upstream,
}

async fn harness(script: Vec<Behavior>, max_attempts: u32) -> Harness {
    let upstream = start_upstream(script).await;
    let config = RouterConfig::builder()
        .regular_mode(vec![])
        .round_robin_policy()
        .grpc_connection_default()
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(1024 * 1024)
        .request_timeout_secs(10)
        .worker_startup_timeout_secs(1)
        .worker_startup_check_interval_secs(1)
        .max_concurrent_requests(8)
        .queue_timeout_secs(2)
        .retry_config(RetryConfig {
            max_retries: max_attempts,
            initial_backoff_ms: 1,
            max_backoff_ms: 1,
            backoff_multiplier: 1.0,
            jitter_factor: 0.0,
        })
        .build_unchecked();
    let context = common::create_test_context_with_parsers(config).await;
    context
        .tokenizer_registry
        .load("receipt-test", "mock-model", "inline", || async {
            Ok(Arc::new(PlainTokenizer::default()) as Arc<dyn Tokenizer>)
        })
        .await
        .unwrap();

    let client = GrpcClient::connect(&format!("grpc://{}", upstream.address), "sglang")
        .await
        .unwrap();
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(format!("grpc://{}", upstream.address))
            .worker_type(WorkerType::Regular)
            .connection_mode(ConnectionMode::Grpc { port: None })
            .runtime_type(RuntimeType::Sglang)
            .models(vec![ModelCard::new("mock-model")])
            .grpc_client(client)
            .build(),
    );
    context.worker_registry.register(Arc::clone(&worker));
    let router: Arc<dyn RouterTrait> =
        Arc::from(RouterFactory::create_router(&context).await.unwrap());
    let app = common::test_app::create_test_app_with_context(router, context);
    Harness {
        app,
        worker,
        upstream,
    }
}

fn generate_request(stream: bool) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/generate")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(format!(
            r#"{{"model":"mock-model","input_ids":[65],"stream":{stream},"rid":"receipt-public"}}"#
        )))
        .unwrap()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Snapshot {
    successes: u64,
    failures: u64,
    generates: usize,
    aborts: usize,
}

fn snapshot(worker: &Arc<dyn Worker>, upstream: &Upstream) -> Snapshot {
    Snapshot {
        successes: worker.circuit_breaker().total_successes(),
        failures: worker.circuit_breaker().total_failures(),
        generates: upstream.generate_calls.load(Ordering::SeqCst),
        aborts: upstream.abort_calls.load(Ordering::SeqCst),
    }
}

async fn assert_snapshot_stays(worker: &Arc<dyn Worker>, upstream: &Upstream, expected: Snapshot) {
    time::timeout(Duration::from_secs(2), async {
        while snapshot(worker, upstream) != expected {
            time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("attempt state did not reach the expected snapshot");
    time::sleep(Duration::from_secs(2)).await;
    assert_eq!(snapshot(worker, upstream), expected);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn public_streaming_cancelled_body_is_one_failed_attempt_without_retry() {
    let Harness {
        app,
        worker,
        upstream,
    } = harness(
        vec![Behavior::Messages {
            messages: vec![chunk()],
            status: 1,
        }],
        2,
    )
    .await;
    let response = app.oneshot(generate_request(true)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    assert!(String::from_utf8_lossy(&body).contains("Stream error"));
    assert_snapshot_stays(
        &worker,
        &upstream,
        Snapshot {
            successes: 0,
            failures: 1,
            generates: 1,
            aborts: 1,
        },
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn public_non_streaming_invalid_argument_body_retries_then_succeeds() {
    let Harness {
        app,
        worker,
        upstream,
    } = harness(
        vec![
            Behavior::Messages {
                messages: vec![chunk()],
                status: 3,
            },
            Behavior::Messages {
                messages: vec![complete()],
                status: 0,
            },
        ],
        2,
    )
    .await;
    let response = app.oneshot(generate_request(false)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(
        body.pointer("/0/text").and_then(|value| value.as_str()),
        Some("A")
    );
    assert_snapshot_stays(
        &worker,
        &upstream,
        Snapshot {
            successes: 1,
            failures: 1,
            generates: 2,
            aborts: 1,
        },
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn public_non_streaming_empty_completion_set_is_failed_attempt_without_abort() {
    let Harness {
        app,
        worker,
        upstream,
    } = harness(
        vec![
            Behavior::Messages {
                messages: vec![chunk()],
                status: 0,
            },
            Behavior::Messages {
                messages: vec![complete()],
                status: 0,
            },
        ],
        2,
    )
    .await;
    let response = app.oneshot(generate_request(false)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(
        body.pointer("/0/text").and_then(|value| value.as_str()),
        Some("A")
    );
    assert_snapshot_stays(
        &worker,
        &upstream,
        Snapshot {
            successes: 1,
            failures: 1,
            generates: 2,
            aborts: 0,
        },
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn public_start_failure_retries_without_creating_a_body_receipt() {
    let Harness {
        app,
        worker,
        upstream,
    } = harness(
        vec![
            Behavior::HeadersStatus(14),
            Behavior::Messages {
                messages: vec![complete()],
                status: 0,
            },
        ],
        2,
    )
    .await;
    let response = app.oneshot(generate_request(false)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(
        body.pointer("/0/text").and_then(|value| value.as_str()),
        Some("A")
    );
    assert_snapshot_stays(
        &worker,
        &upstream,
        Snapshot {
            successes: 1,
            failures: 1,
            generates: 2,
            aborts: 0,
        },
    )
    .await;
}
