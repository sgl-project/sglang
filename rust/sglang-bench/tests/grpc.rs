//! End to end over the api.v1 gRPC transport: a tonic mock of the server's
//! service answers model info, server info and a token-id generate stream,
//! and the client is driven through its public entry point with ids on both
//! ends and no tokenizer. This is the one path the unit tests cannot reach,
//! since they stop at the frame parser.

use std::net::TcpListener;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::Stream;
use sglang_api_types::api::v1 as api;
use sglang_api_types::api::v1::sglang_service_server::{SglangService, SglangServiceServer};
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{Request, Response, Status};

/// Serves `tokens_per_request` ids per generate call, two per frame, with an
/// empty `text` as a `--skip-tokenizer-init` server would.
struct Mock {
    tokens_per_request: usize,
    /// Requests that arrived with `input_ids` rather than `text`.
    id_requests: Arc<AtomicUsize>,
}

type ItemStream = Pin<Box<dyn Stream<Item = Result<api::GenerateStreamItem, Status>> + Send>>;

#[tonic::async_trait]
impl SglangService for Mock {
    type GenerateStream = ItemStream;

    async fn generate(
        &self,
        request: Request<api::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let request = request.into_inner();
        if request.input_ids.is_some() && request.text.is_none() {
            self.id_requests.fetch_add(1, Ordering::Relaxed);
        }
        let want = self.tokens_per_request;
        let mut frames = Vec::new();
        let mut ids: Vec<i64> = Vec::new();
        while ids.len() < want {
            ids.push(ids.len() as i64 + 100);
            if ids.len() < want {
                ids.push(ids.len() as i64 + 100);
            }
            let done = ids.len() == want;
            frames.push(Ok(api::GenerateStreamItem {
                item: Some(api::generate_stream_item::Item::Frame(
                    api::GenerateResponse {
                        text: String::new(),
                        meta_info: Some(api::GenerateMetaInfo {
                            id: "mock".into(),
                            prompt_tokens: 9,
                            completion_tokens: ids.len() as u64,
                            finish_reason: done.then_some(api::FinishReason {
                                kind: Some(api::finish_reason::Kind::Length(api::FinishLength {
                                    length: Some(want as u64),
                                })),
                            }),
                            ..Default::default()
                        }),
                        output_ids: Some(api::TokenIds { ids: ids.clone() }),
                        index: None,
                    },
                )),
            }));
        }
        Ok(Response::new(Box::pin(futures::stream::iter(frames))))
    }

    async fn health_check(
        &self,
        _: Request<api::HealthCheckRequest>,
    ) -> Result<Response<api::HealthCheckResponse>, Status> {
        Ok(Response::new(api::HealthCheckResponse { healthy: true }))
    }

    async fn get_model_info(
        &self,
        _: Request<api::GetModelInfoRequest>,
    ) -> Result<Response<api::GetModelInfoResponse>, Status> {
        Ok(Response::new(api::GetModelInfoResponse {
            model_path: "/models/mock".into(),
            served_model_name: "mock-model".into(),
            tokenizer_path: "/models/mock".into(),
            is_generation: true,
            ..Default::default()
        }))
    }

    async fn get_server_info(
        &self,
        _: Request<api::GetServerInfoRequest>,
    ) -> Result<Response<api::GetServerInfoResponse>, Status> {
        Ok(Response::new(api::GetServerInfoResponse {
            model_path: "/models/mock".into(),
            served_model_name: "mock-model".into(),
            tokenizer_path: "/models/mock".into(),
            max_context_length: 4096,
            max_total_num_tokens: Some(8192),
            version: "test".into(),
            internal_states: vec![r#"{"avg_spec_accept_length": 2.5}"#.into()],
            ..Default::default()
        }))
    }

    async fn list_models(
        &self,
        _: Request<api::ListModelsRequest>,
    ) -> Result<Response<api::ListModelsResponse>, Status> {
        Err(Status::unimplemented("list_models"))
    }

    async fn detokenize(
        &self,
        _: Request<api::DetokenizeRequest>,
    ) -> Result<Response<api::DetokenizeResponse>, Status> {
        Err(Status::unimplemented("detokenize"))
    }

    async fn abort(
        &self,
        _: Request<api::AbortRequest>,
    ) -> Result<Response<api::AbortResponse>, Status> {
        Err(Status::unimplemented("abort"))
    }
}

/// Bind on a free port and serve the mock on its own runtime thread, so the
/// client's own Tokio runtime is what the test exercises.
fn serve_mock(mock: Mock) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    listener.set_nonblocking(true).unwrap();
    std::thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                let listener = tokio::net::TcpListener::from_std(listener).unwrap();
                tonic::transport::Server::builder()
                    .add_service(SglangServiceServer::new(mock))
                    .serve_with_incoming(TcpListenerStream::new(listener))
                    .await
                    .unwrap();
            });
    });
    port
}

/// The whole client, ids in and ids out, with no tokenizer loaded: every
/// request completes, tokens are counted from the ids the frames carry, the
/// accept length is read from the gRPC server info, and the per-run result
/// lands in the file with the Python client's keys.
#[test]
fn grpc_backend_runs_end_to_end_on_token_ids() {
    let id_requests = Arc::new(AtomicUsize::new(0));
    let port = serve_mock(Mock {
        tokens_per_request: 6,
        id_requests: id_requests.clone(),
    });
    let output_file =
        std::env::temp_dir().join(format!("sglang-bench-grpc-{}.jsonl", std::process::id()));

    let args: sglang_bench::Args = serde_json::from_value(serde_json::json!({
        "backend": "sglang",
        "protocol": "grpc",
        "base_url": format!("http://127.0.0.1:{port}"),
        "dataset_name": "random-ids",
        "tokenize_prompt": true,
        "vocab_size": 1000,
        "num_prompts": 12,
        "random_input_len": 8,
        "random_output_len": 6,
        "random_range_ratio": 1.0,
        "max_concurrency": 4,
        "warmup_requests": 1,
        "ready_check_timeout_sec": 10,
        "disable_tqdm": true,
        "output_file": output_file.to_str().unwrap(),
    }))
    .unwrap();

    let result = sglang_bench::run_blocking(args).expect("benchmark runs");

    assert_eq!(result["completed"], 12);
    assert_eq!(result["backend"], "sglang");
    assert_eq!(result["protocol"], "grpc");
    // Every prompt is 8 ids and every reply 6 ids; no text crossed the wire,
    // so the retokenized total is 0 by design.
    assert_eq!(result["total_input_tokens"], 12 * 8);
    assert_eq!(result["total_output_tokens"], 12 * 6);
    assert_eq!(result["total_output_tokens_retokenized"], 0);
    assert_eq!(result["accept_length"], 2.5);
    assert!(result["mean_ttft_ms"].as_f64().unwrap() > 0.0);
    // 12 measured requests plus the warmup, all as input_ids.
    assert_eq!(id_requests.load(Ordering::Relaxed), 13);

    let written = std::fs::read_to_string(&output_file).unwrap();
    let line: serde_json::Value = serde_json::from_str(written.trim()).unwrap();
    assert_eq!(line["completed"], 12);
    std::fs::remove_file(&output_file).unwrap();
}
