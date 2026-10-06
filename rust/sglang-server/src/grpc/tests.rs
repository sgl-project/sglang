use std::time::Duration;

use futures::StreamExt;
use sglang_grpc_types::sglang::runtime::v1 as proto;
use sglang_grpc_types::sglang::runtime::v1::sglang_service_server::SglangService;
use tonic::{Code, Request};

use super::GrpcService;
use crate::frontend::{FrontendConfig, FrontendHandle, FrontendMetadata};
use crate::message::finish_reason::FinishReason;
use crate::message::ids::Rid;
use crate::message::request::{GenerateRequest, Request as RuntimeRequest, RequestKind};
use crate::message::response::{ChunkEvent, ResponseItem, ResponseSink};
use crate::tokenizer_manager::wiring::{AbortSource, RequestAdmission, TmEvent};

struct Harness {
    service: GrpcService,
    intake_rx: flume::Receiver<TmEvent>,
    abort_rx: flume::Receiver<AbortSource>,
}

struct GenerationIntake {
    request: Box<GenerateRequest>,
    rid: Rid,
    sink: ResponseSink,
    admission: RequestAdmission,
}

impl Harness {
    fn new(response_capacity: usize, incremental: bool, response_timeout: Duration) -> Self {
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, abort_rx) = flume::unbounded();
        let frontend = FrontendHandle::new(
            intake_tx,
            abort_tx,
            FrontendConfig {
                response_capacity,
                response_activity: Default::default(),
                startup_ready: true,
                is_disaggregation: false,
                mm_limits: Default::default(),
                metadata: FrontendMetadata::default(),
            },
        );
        Self {
            service: GrpcService::for_test(frontend, None, incremental, response_timeout),
            intake_rx,
            abort_rx,
        }
    }

    async fn next_generation(&self) -> GenerationIntake {
        let TmEvent::Intake { request, admission } = self
            .intake_rx
            .recv_async()
            .await
            .expect("request submitted")
        else {
            panic!("expected generation intake");
        };
        let RuntimeRequest {
            rid,
            sink,
            kind,
            state: _,
        } = request;
        let RequestKind::Generate(request) = kind else {
            panic!("expected generation request");
        };
        GenerationIntake {
            request,
            rid,
            sink,
            admission,
        }
    }
}

fn chunk(rid: &Rid, text: &str, token_id: i64, finished: bool) -> ChunkEvent {
    ChunkEvent {
        rid: rid.clone(),
        token_ids: vec![token_id].into(),
        finish_reason: finished.then(stop_reason),
        prompt_tokens: 3,
        text: text.into(),
        completion_tokens: 1,
        extras: None,
    }
}

fn stop_reason() -> FinishReason {
    serde_json::from_value(serde_json::json!({"type": "stop"})).unwrap()
}

#[tokio::test]
async fn text_generate_maps_request_and_streams_cumulative_responses() {
    let harness = Harness::new(2, false, Duration::from_secs(1));
    let response = harness
        .service
        .text_generate(Request::new(proto::TextGenerateRequest {
            text: "prompt".into(),
            stream: Some(true),
            rid: Some("request-1".into()),
            ..Default::default()
        }))
        .await
        .unwrap();
    let mut stream = response.into_inner();

    let intake = harness.next_generation().await;
    assert_eq!(
        intake.request.output_mode,
        crate::message::request::OutputMode::TextAndTokenIds
    );
    assert!(intake.admission.try_accept());
    assert_eq!(intake.request.text.as_deref(), Some("prompt"));
    assert!(intake.request.input_ids.is_none());
    assert!(intake.request.stream);
    assert_eq!(intake.rid.client_facing(), "request-1");
    intake
        .sink
        .try_send(ResponseItem::Frame(chunk(&intake.rid, "Hel", 1, false)))
        .unwrap();
    intake
        .sink
        .try_send(ResponseItem::Done(chunk(&intake.rid, "lo", 2, true)))
        .unwrap();

    let first = stream.next().await.unwrap().unwrap();
    assert_eq!(first.text, "Hel");
    assert!(!first.finished);
    assert_eq!(first.meta_info["id"], r#""request-1""#);
    assert_eq!(first.meta_info["prompt_tokens"], "3");
    assert_eq!(first.meta_info["completion_tokens"], "1");
    assert_eq!(first.meta_info["finish_reason"], "null");

    let finished = stream.next().await.unwrap().unwrap();
    assert_eq!(finished.text, "Hello");
    assert!(finished.finished);
    assert_eq!(finished.meta_info["completion_tokens"], "2");
    assert!(finished.meta_info["finish_reason"].contains(r#""type":"stop""#));
    assert!(stream.next().await.is_none());
    assert!(harness.abort_rx.try_recv().is_err());
}

#[tokio::test]
async fn generate_streams_incremental_token_ids_with_cumulative_count() {
    let harness = Harness::new(2, true, Duration::from_secs(1));
    let response = harness
        .service
        .generate(Request::new(proto::GenerateRequest {
            input_ids: vec![4, 5],
            stream: Some(true),
            rid: Some("tokens".into()),
            ..Default::default()
        }))
        .await
        .unwrap();
    let mut stream = response.into_inner();

    let intake = harness.next_generation().await;
    assert_eq!(
        intake.request.output_mode,
        crate::message::request::OutputMode::TokenIds
    );
    assert!(intake.admission.try_accept());
    assert_eq!(intake.request.input_ids.as_deref(), Some(&[4, 5][..]));
    assert!(intake.request.text.is_none());
    intake
        .sink
        .try_send(ResponseItem::Frame(chunk(&intake.rid, "A", 10, false)))
        .unwrap();
    intake
        .sink
        .try_send(ResponseItem::Done(chunk(&intake.rid, "B", 11, true)))
        .unwrap();

    let first = stream.next().await.unwrap().unwrap();
    let finished = stream.next().await.unwrap().unwrap();
    assert_eq!(first.output_ids, vec![10]);
    assert_eq!(finished.output_ids, vec![11]);
    assert_eq!(finished.meta_info["completion_tokens"], "2");
    assert!(finished.finished);
}

#[tokio::test]
async fn stream_drop_aborts_admitted_request() {
    let harness = Harness::new(1, false, Duration::from_secs(1));
    let mut stream = harness
        .service
        .generate(Request::new(proto::GenerateRequest {
            input_ids: vec![1],
            rid: Some("cancel-me".into()),
            ..Default::default()
        }))
        .await
        .unwrap()
        .into_inner();
    let intake = harness.next_generation().await;
    assert!(intake.admission.try_accept());
    intake
        .sink
        .try_send(ResponseItem::Frame(chunk(&intake.rid, "partial", 1, false)))
        .unwrap();
    assert!(stream.next().await.unwrap().is_ok());
    drop(stream);
    let abort = harness.abort_rx.recv_async().await.unwrap();
    assert_eq!(abort.rid().client_facing(), "cancel-me");
    assert!(harness.abort_rx.try_recv().is_err());
}

#[tokio::test]
async fn runtime_failure_is_an_in_stream_status_and_disarms_abort() {
    let harness = Harness::new(1, false, Duration::from_secs(1));
    let mut stream = harness
        .service
        .generate(Request::new(proto::GenerateRequest {
            input_ids: vec![1],
            ..Default::default()
        }))
        .await
        .unwrap()
        .into_inner();
    let intake = harness.next_generation().await;
    assert!(intake.admission.try_accept());
    intake
        .sink
        .try_send(ResponseItem::Error(crate::utils::error::Error::Validation(
            "bad sampling".into(),
        )))
        .unwrap();

    let error = stream.next().await.unwrap().unwrap_err();
    assert_eq!(error.code(), Code::InvalidArgument);
    drop(stream);
    assert!(harness.abort_rx.try_recv().is_err());
}

#[tokio::test]
async fn per_chunk_timeout_after_progress_returns_deadline_exceeded_and_aborts() {
    let harness = Harness::new(1, false, Duration::from_millis(10));
    let mut stream = harness
        .service
        .generate(Request::new(proto::GenerateRequest {
            input_ids: vec![1],
            rid: Some("too-slow".into()),
            ..Default::default()
        }))
        .await
        .unwrap()
        .into_inner();
    let intake = harness.next_generation().await;
    assert!(intake.admission.try_accept());
    intake
        .sink
        .try_send(ResponseItem::Frame(chunk(&intake.rid, "partial", 1, false)))
        .unwrap();
    assert!(stream.next().await.unwrap().is_ok());

    let error = stream.next().await.unwrap().unwrap_err();
    assert_eq!(error.code(), Code::DeadlineExceeded);
    drop(stream);
    let abort = harness.abort_rx.recv_async().await.unwrap();
    assert_eq!(abort.rid().client_facing(), "too-slow");
}

#[tokio::test]
async fn closed_intake_is_a_top_level_unavailable_status() {
    let (intake_tx, intake_rx) = flume::unbounded();
    drop(intake_rx);
    let (abort_tx, _abort_rx) = flume::unbounded();
    let frontend = FrontendHandle::new(
        intake_tx,
        abort_tx,
        FrontendConfig {
            response_capacity: 1,
            response_activity: Default::default(),
            startup_ready: true,
            is_disaggregation: false,
            mm_limits: Default::default(),
            metadata: FrontendMetadata::default(),
        },
    );
    let service = GrpcService::for_test(frontend, None, false, Duration::from_secs(1));

    let result = service
        .generate(Request::new(proto::GenerateRequest {
            input_ids: vec![1],
            ..Default::default()
        }))
        .await;
    let error = match result {
        Ok(_) => panic!("closed intake must reject the RPC"),
        Err(error) => error,
    };
    assert_eq!(error.code(), Code::Unavailable);
}

#[tokio::test]
async fn new_proto_fields_and_engine_watch_are_explicitly_unsupported() {
    let harness = Harness::new(1, false, Duration::from_secs(1));
    let text_error = harness
        .service
        .text_generate(Request::new(proto::TextGenerateRequest {
            text: "hello".into(),
            kv_hints: Some(Default::default()),
            ..Default::default()
        }))
        .await
        .err()
        .expect("KV hints must not be silently ignored");
    assert_eq!(text_error.code(), Code::Unimplemented);
    let token_error = harness
        .service
        .generate(Request::new(proto::GenerateRequest {
            input_ids: vec![1],
            kv_hints: Some(Default::default()),
            ..Default::default()
        }))
        .await
        .err()
        .expect("KV hints must not be silently ignored");
    assert_eq!(token_error.code(), Code::Unimplemented);
    let watch_error = harness
        .service
        .watch_engine_state(Request::new(proto::WatchEngineStateRequest {}))
        .await
        .err()
        .expect("engine watch is outside the generation adapter");
    assert_eq!(watch_error.code(), Code::Unimplemented);
    assert!(harness.intake_rx.try_recv().is_err());
}

#[tokio::test]
async fn output_token_overflow_returns_error_instead_of_truncating() {
    let harness = Harness::new(1, true, Duration::from_secs(1));
    let mut stream = harness
        .service
        .generate(Request::new(proto::GenerateRequest {
            input_ids: vec![i32::MAX],
            ..Default::default()
        }))
        .await
        .unwrap()
        .into_inner();
    let intake = harness.next_generation().await;
    assert!(intake.admission.try_accept());
    assert_eq!(
        intake.request.input_ids.as_deref(),
        Some(&[i64::from(i32::MAX)][..])
    );
    intake
        .sink
        .try_send(ResponseItem::Frame(chunk(
            &intake.rid,
            "",
            i64::from(i32::MAX) + 1,
            false,
        )))
        .unwrap();
    assert_eq!(
        stream.next().await.unwrap().unwrap_err().code(),
        Code::Internal
    );
    assert!(stream.next().await.is_none());
    assert!(
        matches!(harness.abort_rx.recv_async().await.unwrap(), AbortSource::Guard(rid) if rid == intake.rid)
    );
}
