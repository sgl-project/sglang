use std::{
    convert::Infallible,
    net::SocketAddr,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use axum::{body::Body, http::Response, routing::post, Router};
use bytes::Bytes;
use futures_util::stream;
use http_body::Frame;
use http_body_util::StreamBody;
use prost::Message;
use smg_grpc_client::{sglang_proto as sglang, vllm_proto as vllm};
use tokio::{net::TcpListener, task::JoinHandle, time};

use super::{ExecutionMode, RequestExecutionStage};
use crate::{
    core::{BasicWorkerBuilder, Worker},
    routers::grpc::{
        client::GrpcClient,
        context::{ClientSelection, ExecutionResult, WorkerSelection},
        proto_wrapper::ProtoGenerateRequest,
    },
};

const GENERATE_PATH: &str = "/sglang.grpc.scheduler.SglangScheduler/Generate";
const ABORT_PATH: &str = "/sglang.grpc.scheduler.SglangScheduler/Abort";
const VLLM_GENERATE_PATH: &str = "/vllm.grpc.engine.VllmEngine/Generate";
const VLLM_ABORT_PATH: &str = "/vllm.grpc.engine.VllmEngine/Abort";

#[derive(Clone)]
enum Behavior {
    HeadersStatus(u32),
    Messages {
        messages: Vec<Vec<u8>>,
        status: u32,
    },
    MessagesWithStatusMessage {
        messages: Vec<Vec<u8>>,
        status: u32,
        message: &'static str,
    },
    Malformed,
}

struct Fixture {
    address: SocketAddr,
    generate_calls: Arc<AtomicUsize>,
    abort_calls: Arc<AtomicUsize>,
    server: JoinHandle<()>,
}

impl Drop for Fixture {
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
        Behavior::Malformed => builder.body(Body::from(vec![0, 0, 0, 0, 1, 0xff])).unwrap(),
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
        Behavior::MessagesWithStatusMessage {
            messages,
            status,
            message,
        } => {
            let mut trailers = http::HeaderMap::new();
            trailers.insert("grpc-status", status.to_string().parse().unwrap());
            trailers.insert("grpc-message", message.parse().unwrap());
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

async fn fixture(behavior: Behavior) -> Fixture {
    let generate_calls = Arc::new(AtomicUsize::new(0));
    let abort_calls = Arc::new(AtomicUsize::new(0));
    let generate_counter = Arc::clone(&generate_calls);
    let vllm_generate_counter = Arc::clone(&generate_calls);
    let abort_counter = Arc::clone(&abort_calls);
    let vllm_abort_counter = Arc::clone(&abort_calls);
    let vllm_behavior = behavior.clone();
    let app = Router::new()
        .route(
            GENERATE_PATH,
            post(move || {
                let behavior = behavior.clone();
                let generate_counter = Arc::clone(&generate_counter);
                async move {
                    generate_counter.fetch_add(1, Ordering::SeqCst);
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
        )
        .route(
            VLLM_GENERATE_PATH,
            post(move || {
                let behavior = vllm_behavior.clone();
                let generate_counter = Arc::clone(&vllm_generate_counter);
                async move {
                    generate_counter.fetch_add(1, Ordering::SeqCst);
                    response(behavior)
                }
            }),
        )
        .route(
            VLLM_ABORT_PATH,
            post(move || {
                let abort_counter = Arc::clone(&vllm_abort_counter);
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
    Fixture {
        address,
        generate_calls,
        abort_calls,
        server,
    }
}

fn encoded(response: sglang::generate_response::Response) -> Vec<u8> {
    sglang::GenerateResponse {
        response: Some(response),
        ..Default::default()
    }
    .encode_to_vec()
}

fn chunk() -> Vec<u8> {
    encoded(sglang::generate_response::Response::Chunk(
        sglang::GenerateStreamChunk::default(),
    ))
}

fn complete(index: u32) -> Vec<u8> {
    encoded(sglang::generate_response::Response::Complete(
        sglang::GenerateComplete {
            index,
            finish_reason: "stop".to_string(),
            ..Default::default()
        },
    ))
}

fn in_band_error(http_status_code: &str) -> Vec<u8> {
    encoded(sglang::generate_response::Response::Error(
        sglang::GenerateError {
            http_status_code: http_status_code.to_string(),
            ..Default::default()
        },
    ))
}

fn vllm_complete() -> Vec<u8> {
    vllm::GenerateResponse {
        response: Some(vllm::generate_response::Response::Complete(
            vllm::GenerateComplete {
                finish_reason: "stop".to_string(),
                ..Default::default()
            },
        )),
    }
    .encode_to_vec()
}

fn worker(name: &str) -> Arc<dyn Worker> {
    Arc::new(BasicWorkerBuilder::new(format!("grpc://{name}")).build())
}

fn request(id: &str) -> ProtoGenerateRequest {
    ProtoGenerateRequest::Sglang(Box::new(sglang::GenerateRequest {
        request_id: id.to_string(),
        stream: true,
        ..Default::default()
    }))
}

async fn selections(
    fixture: &Fixture,
    selected: Arc<dyn Worker>,
) -> (ClientSelection, WorkerSelection) {
    let client = GrpcClient::connect(&format!("grpc://{}", fixture.address), "sglang")
        .await
        .unwrap();
    (
        ClientSelection::Single { client },
        WorkerSelection::Single { worker: selected },
    )
}

async fn execute(fixture: &Fixture, selected: Arc<dyn Worker>) -> ExecutionResult {
    let (mut clients, workers) = selections(fixture, selected).await;
    RequestExecutionStage::new(ExecutionMode::Single)
        .execute_single(request("receipt-test"), &mut clients, &workers)
        .await
        .unwrap()
}

async fn execute_vllm(fixture: &Fixture, selected: Arc<dyn Worker>) -> ExecutionResult {
    let client = GrpcClient::connect(&format!("grpc://{}", fixture.address), "vllm")
        .await
        .unwrap();
    let mut clients = ClientSelection::Single { client };
    let workers = WorkerSelection::Single { worker: selected };
    let request = ProtoGenerateRequest::Vllm(Box::new(vllm::GenerateRequest {
        request_id: "vllm-receipt-test".to_string(),
        stream: true,
        ..Default::default()
    }));
    RequestExecutionStage::new(ExecutionMode::Single)
        .execute_single(request, &mut clients, &workers)
        .await
        .unwrap()
}

fn into_stream(result: ExecutionResult) -> crate::routers::grpc::proto_wrapper::ProtoStream {
    match result {
        ExecutionResult::Single { stream } => stream,
        _ => panic!("expected a single stream"),
    }
}

fn outcomes(worker: &Arc<dyn Worker>) -> (u64, u64) {
    (
        worker.circuit_breaker().total_successes(),
        worker.circuit_breaker().total_failures(),
    )
}

async fn assert_abort_stays(fixture: &Fixture, expected: usize) {
    time::timeout(Duration::from_secs(2), async {
        while fixture.abort_calls.load(Ordering::SeqCst) != expected {
            time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("abort count did not reach the expected value");
    time::sleep(Duration::from_secs(2)).await;
    assert_eq!(fixture.abort_calls.load(Ordering::SeqCst), expected);
}

#[tokio::test]
async fn start_failure_is_published_by_the_dispatch_stage() {
    for code in [3, 14] {
        let fixture = fixture(Behavior::HeadersStatus(code)).await;
        let selected = worker(&format!("start-{code}"));
        let (mut clients, workers) = selections(&fixture, Arc::clone(&selected)).await;
        let result = RequestExecutionStage::new(ExecutionMode::Single)
            .execute_single(request("start-status"), &mut clients, &workers)
            .await;
        assert!(result.is_err());
        assert_eq!(outcomes(&selected), (0, 1));
    }
}

#[tokio::test]
async fn every_body_status_is_one_failed_attempt_without_origin_guessing() {
    for code in 1..=16 {
        let fixture = fixture(Behavior::Messages {
            messages: vec![chunk()],
            status: code,
        })
        .await;
        let selected = worker(&format!("body-{code}"));
        let mut stream = into_stream(execute(&fixture, Arc::clone(&selected)).await);
        assert!(matches!(stream.next().await, Some(Ok(_))));
        assert!(matches!(stream.next().await, Some(Err(_))));
        assert_eq!(outcomes(&selected), (0, 1), "gRPC status code {code}");
        stream.mark_completed();
        drop(stream);
        assert_eq!(outcomes(&selected), (0, 1), "gRPC status code {code}");
    }
}

#[tokio::test]
async fn terminal_status_message_is_not_used_as_origin() {
    let fixture = fixture(Behavior::MessagesWithStatusMessage {
        messages: vec![chunk()],
        status: 13,
        message: "cancelled by caller",
    })
    .await;
    let selected = worker("message-origin");
    let mut stream = into_stream(execute(&fixture, Arc::clone(&selected)).await);

    assert!(matches!(stream.next().await, Some(Ok(_))));
    let status = match stream.next().await {
        Some(Err(status)) => status,
        _ => panic!("expected a terminal status"),
    };
    assert_eq!(status.message(), "cancelled by caller");
    assert_eq!(outcomes(&selected), (0, 1));
}

#[tokio::test]
async fn start_success_transfers_ownership_without_eager_publication() {
    let fixture = fixture(Behavior::Malformed).await;
    let selected = worker("selected");
    let replacement = worker("replacement");
    let mut stream = into_stream(execute(&fixture, Arc::clone(&selected)).await);

    assert_eq!(outcomes(&selected), (0, 0));
    assert!(stream
        .attach_breaker_receipt(Arc::clone(&replacement))
        .is_err());
    assert!(matches!(stream.next().await, Some(Err(_))));
    assert_eq!(outcomes(&selected), (0, 1));
    assert_eq!(outcomes(&replacement), (0, 0));
}

#[tokio::test]
async fn clean_drain_requires_the_consumers_completion_acknowledgement() {
    let fixture = fixture(Behavior::Messages {
        messages: vec![complete(0)],
        status: 0,
    })
    .await;
    let selected = worker("clean");
    let mut stream = into_stream(execute(&fixture, Arc::clone(&selected)).await);

    assert!(matches!(stream.next().await, Some(Ok(_))));
    assert!(stream.next().await.is_none());
    assert_eq!(outcomes(&selected), (0, 0));

    stream.mark_completed();
    stream.mark_completed();
    assert_eq!(outcomes(&selected), (1, 0));
    drop(stream);
    assert_eq!(outcomes(&selected), (1, 0));
}

#[tokio::test]
async fn clean_drain_rejected_by_consumer_is_failure_without_abort() {
    let fixture = fixture(Behavior::Messages {
        messages: vec![chunk()],
        status: 0,
    })
    .await;
    let selected = worker("rejected-body");
    let mut stream = into_stream(execute(&fixture, Arc::clone(&selected)).await);

    assert!(matches!(stream.next().await, Some(Ok(_))));
    assert!(stream.next().await.is_none());
    assert_eq!(outcomes(&selected), (0, 0));

    stream.reject_completed_body();
    stream.reject_completed_body();
    drop(stream);
    assert_eq!(outcomes(&selected), (0, 1));
    assert_abort_stays(&fixture, 0).await;
}

#[tokio::test]
async fn normal_consumer_observes_a_late_transport_failure_before_acknowledging() {
    let fixture = fixture(Behavior::Messages {
        messages: vec![complete(0)],
        status: 13,
    })
    .await;
    let selected = worker("late-status");
    let mut stream = into_stream(execute(&fixture, Arc::clone(&selected)).await);

    assert!(matches!(stream.next().await, Some(Ok(_))));
    assert_eq!(outcomes(&selected), (0, 0));
    assert!(matches!(stream.next().await, Some(Err(ref e)) if e.code() == tonic::Code::Internal));
    assert_eq!(outcomes(&selected), (0, 1));
    stream.mark_completed();
    assert_eq!(outcomes(&selected), (0, 1));
}

#[tokio::test]
async fn explicit_completion_is_absorbing_even_if_a_late_error_is_polled() {
    let fixture = fixture(Behavior::Messages {
        messages: vec![complete(0)],
        status: 13,
    })
    .await;
    let selected = worker("explicit-completion");
    let mut stream = into_stream(execute(&fixture, Arc::clone(&selected)).await);

    assert!(matches!(stream.next().await, Some(Ok(_))));
    stream.mark_completed();
    assert_eq!(outcomes(&selected), (1, 0));
    assert!(matches!(stream.next().await, Some(Err(_))));
    assert_eq!(outcomes(&selected), (1, 0));
}

#[tokio::test]
async fn legacy_in_band_error_is_one_failed_attempt_independent_of_http_hint() {
    for http_status_code in ["499", "400", "500", "", "not-a-status"] {
        let fixture = fixture(Behavior::Messages {
            messages: vec![in_band_error(http_status_code)],
            status: 0,
        })
        .await;
        let selected = worker(&format!("in-band-{http_status_code}"));
        let mut stream = into_stream(execute(&fixture, Arc::clone(&selected)).await);

        assert!(matches!(stream.next().await, Some(Ok(_))));
        assert_eq!(outcomes(&selected), (0, 1));
        assert!(stream.next().await.is_none());
        stream.mark_completed();
        drop(stream);
        assert_eq!(outcomes(&selected), (0, 1));
    }
}

#[tokio::test]
async fn vllm_stream_uses_the_same_terminal_contract() {
    let status_fixture = fixture(Behavior::Messages {
        messages: vec![vllm_complete()],
        status: 3,
    })
    .await;
    let status_worker = worker("vllm-status");
    let mut status_stream =
        into_stream(execute_vllm(&status_fixture, Arc::clone(&status_worker)).await);
    assert!(matches!(status_stream.next().await, Some(Ok(_))));
    assert!(matches!(status_stream.next().await, Some(Err(_))));
    assert_eq!(outcomes(&status_worker), (0, 1));

    let failing_fixture = fixture(Behavior::Malformed).await;
    let failing_worker = worker("vllm-failure");
    let mut failing_stream =
        into_stream(execute_vllm(&failing_fixture, Arc::clone(&failing_worker)).await);
    assert_eq!(outcomes(&failing_worker), (0, 0));
    assert!(matches!(failing_stream.next().await, Some(Err(_))));
    assert_eq!(outcomes(&failing_worker), (0, 1));

    let success_fixture = fixture(Behavior::Messages {
        messages: vec![vllm_complete()],
        status: 0,
    })
    .await;
    let success_worker = worker("vllm-success");
    let mut success_stream =
        into_stream(execute_vllm(&success_fixture, Arc::clone(&success_worker)).await);
    assert!(matches!(success_stream.next().await, Some(Ok(_))));
    assert_eq!(outcomes(&success_worker), (0, 0));
    assert!(success_stream.next().await.is_none());
    assert_eq!(outcomes(&success_worker), (0, 0));
    success_stream.mark_completed();
    assert_eq!(outcomes(&success_worker), (1, 0));
}

#[tokio::test]
async fn single_shape_mismatch_fails_before_dispatch() {
    let fixture = fixture(Behavior::Malformed).await;
    let selected = worker("client-worker");
    let other = worker("other-worker");
    let (mut clients, _) = selections(&fixture, Arc::clone(&selected)).await;
    let workers = WorkerSelection::Dual {
        prefill: selected,
        decode: other,
    };

    let result = RequestExecutionStage::new(ExecutionMode::Single)
        .execute_single(request("shape-mismatch"), &mut clients, &workers)
        .await;
    assert!(result.is_err());
    assert_eq!(fixture.generate_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn dual_streams_remain_detached_from_body_terminal_accounting() {
    let fixture = fixture(Behavior::Malformed).await;
    let prefill_worker = worker("prefill");
    let decode_worker = worker("decode");
    let prefill = GrpcClient::connect(&format!("grpc://{}", fixture.address), "sglang")
        .await
        .unwrap();
    let decode = GrpcClient::connect(&format!("grpc://{}", fixture.address), "sglang")
        .await
        .unwrap();
    let mut clients = ClientSelection::Dual { prefill, decode };
    let workers = WorkerSelection::Dual {
        prefill: Arc::clone(&prefill_worker),
        decode: Arc::clone(&decode_worker),
    };

    let result = RequestExecutionStage::new(ExecutionMode::DualDispatch)
        .execute_dual_dispatch(request("detached-dual"), &mut clients, &workers)
        .await;
    let ExecutionResult::Dual {
        mut prefill,
        decode,
    } = result.unwrap()
    else {
        panic!("expected dual streams");
    };
    let mut decode = *decode;
    assert!(matches!(prefill.next().await, Some(Err(_))));
    assert!(matches!(decode.next().await, Some(Err(_))));
    drop(prefill);
    drop(decode);
    assert_eq!(outcomes(&prefill_worker), (1, 0));
    assert_eq!(outcomes(&decode_worker), (1, 0));
}

#[tokio::test]
async fn early_drop_is_abandoned_and_abort_is_exactly_once_after_settle() {
    let fixture = fixture(Behavior::Messages {
        messages: vec![complete(0)],
        status: 0,
    })
    .await;
    let selected = worker("drop");
    let stream = into_stream(execute(&fixture, Arc::clone(&selected)).await);

    drop(stream);
    assert_eq!(outcomes(&selected), (0, 0));
    assert_abort_stays(&fixture, 1).await;
}
