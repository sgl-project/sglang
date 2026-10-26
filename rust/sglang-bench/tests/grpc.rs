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

type ItemStreamOf<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send>>;
type ItemStream = ItemStreamOf<api::GenerateStreamItem>;

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
                        prompt_token_ids: None,
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

    async fn get_hi_cache_storage(
        &self,
        _: Request<api::GetHiCacheStorageRequest>,
    ) -> Result<Response<api::HiCacheStatus>, Status> {
        Err(Status::unimplemented("get_hi_cache_storage"))
    }

    async fn attach_hi_cache_storage(
        &self,
        _: Request<api::AttachHiCacheStorageRequest>,
    ) -> Result<Response<api::HiCacheStorageResponse>, Status> {
        Err(Status::unimplemented("attach_hi_cache_storage"))
    }

    async fn detach_hi_cache_storage(
        &self,
        _: Request<api::DetachHiCacheStorageRequest>,
    ) -> Result<Response<api::HiCacheStorageResponse>, Status> {
        Err(Status::unimplemented("detach_hi_cache_storage"))
    }

    async fn clear_hi_cache_storage(
        &self,
        _: Request<api::ClearHiCacheStorageRequest>,
    ) -> Result<Response<api::HiCacheStorageResponse>, Status> {
        Err(Status::unimplemented("clear_hi_cache_storage"))
    }

    async fn set_internal_state(
        &self,
        _: Request<api::SetInternalStateRequest>,
    ) -> Result<Response<api::SetInternalStateResponse>, Status> {
        Err(Status::unimplemented("set_internal_state"))
    }
    async fn start_profile(
        &self,
        _: Request<api::StartProfileRequest>,
    ) -> Result<Response<api::ProfileResponse>, Status> {
        Err(Status::unimplemented("start_profile"))
    }
    async fn stop_profile(
        &self,
        _: Request<api::StopProfileRequest>,
    ) -> Result<Response<api::ProfileResponse>, Status> {
        Err(Status::unimplemented("stop_profile"))
    }
    async fn start_expert_distribution_record(
        &self,
        _: Request<api::ExpertDistributionRecordRequest>,
    ) -> Result<Response<api::ExpertDistributionRecordResponse>, Status> {
        Err(Status::unimplemented("start_expert_distribution_record"))
    }
    async fn stop_expert_distribution_record(
        &self,
        _: Request<api::ExpertDistributionRecordRequest>,
    ) -> Result<Response<api::ExpertDistributionRecordResponse>, Status> {
        Err(Status::unimplemented("stop_expert_distribution_record"))
    }
    async fn dump_expert_distribution_record(
        &self,
        _: Request<api::ExpertDistributionRecordRequest>,
    ) -> Result<Response<api::ExpertDistributionRecordResponse>, Status> {
        Err(Status::unimplemented("dump_expert_distribution_record"))
    }
    async fn slow_down(
        &self,
        _: Request<api::SlowDownRequest>,
    ) -> Result<Response<api::SlowDownResponse>, Status> {
        Err(Status::unimplemented("slow_down"))
    }
    async fn release_memory_occupation(
        &self,
        _: Request<api::ReleaseMemoryOccupationRequest>,
    ) -> Result<Response<api::MemoryOccupationResponse>, Status> {
        Err(Status::unimplemented("release_memory_occupation"))
    }
    async fn resume_memory_occupation(
        &self,
        _: Request<api::ResumeMemoryOccupationRequest>,
    ) -> Result<Response<api::MemoryOccupationResponse>, Status> {
        Err(Status::unimplemented("resume_memory_occupation"))
    }
    async fn get_weights_by_name(
        &self,
        _: Request<api::GetWeightsByNameRequest>,
    ) -> Result<Response<api::GetWeightsByNameResponse>, Status> {
        Err(Status::unimplemented("get_weights_by_name"))
    }
    async fn init_weights_update_group(
        &self,
        _: Request<api::InitWeightsUpdateGroupRequest>,
    ) -> Result<Response<api::WeightGroupResponse>, Status> {
        Err(Status::unimplemented("init_weights_update_group"))
    }
    async fn destroy_weights_update_group(
        &self,
        _: Request<api::DestroyWeightsUpdateGroupRequest>,
    ) -> Result<Response<api::WeightGroupResponse>, Status> {
        Err(Status::unimplemented("destroy_weights_update_group"))
    }
    async fn init_weights_send_group_for_remote_instance(
        &self,
        _: Request<api::InitWeightsSendGroupForRemoteInstanceRequest>,
    ) -> Result<Response<api::WeightGroupResponse>, Status> {
        Err(Status::unimplemented(
            "init_weights_send_group_for_remote_instance",
        ))
    }
    async fn send_weights_to_remote_instance(
        &self,
        _: Request<api::SendWeightsToRemoteInstanceRequest>,
    ) -> Result<Response<api::WeightGroupResponse>, Status> {
        Err(Status::unimplemented("send_weights_to_remote_instance"))
    }
    async fn get_remote_instance_transfer_engine_info(
        &self,
        _: Request<api::RemoteInstanceTransferEngineInfoRequest>,
    ) -> Result<Response<api::RemoteInstanceTransferEngineInfoResponse>, Status> {
        Err(Status::unimplemented(
            "get_remote_instance_transfer_engine_info",
        ))
    }
    async fn parse_function_call(
        &self,
        _: Request<api::ParseFunctionCallRequest>,
    ) -> Result<Response<api::ParseFunctionCallResponse>, Status> {
        Err(Status::unimplemented("parse_function_call"))
    }
    type OllamaChatStream = ItemStreamOf<api::OllamaChatResponse>;

    async fn ollama_chat(
        &self,
        _: Request<api::OllamaChatRequest>,
    ) -> Result<Response<Self::OllamaChatStream>, Status> {
        unimplemented!()
    }

    type OllamaGenerateStream = ItemStreamOf<api::OllamaGenerateResponse>;

    async fn ollama_generate(
        &self,
        _: Request<api::OllamaGenerateRequest>,
    ) -> Result<Response<Self::OllamaGenerateStream>, Status> {
        unimplemented!()
    }

    async fn ollama_tags(
        &self,
        _: Request<api::OllamaTagsRequest>,
    ) -> Result<Response<api::OllamaTagsResponse>, Status> {
        unimplemented!()
    }

    async fn ollama_show(
        &self,
        _: Request<api::OllamaShowRequest>,
    ) -> Result<Response<api::OllamaShowResponse>, Status> {
        unimplemented!()
    }

    async fn anthropic_messages(
        &self,
        _: Request<api::AnthropicMessagesRequest>,
    ) -> Result<Response<api::AnthropicMessagesResponse>, Status> {
        unimplemented!()
    }

    type AnthropicStreamMessagesStream = ItemStreamOf<api::AnthropicStreamEvent>;

    async fn anthropic_stream_messages(
        &self,
        _: Request<api::AnthropicMessagesRequest>,
    ) -> Result<Response<Self::AnthropicStreamMessagesStream>, Status> {
        unimplemented!()
    }

    async fn anthropic_count_tokens(
        &self,
        _: Request<api::AnthropicCountTokensRequest>,
    ) -> Result<Response<api::AnthropicCountTokensResponse>, Status> {
        unimplemented!()
    }

    async fn open_ai_completions(
        &self,
        _: Request<api::OpenAiCompletionRequest>,
    ) -> Result<Response<api::OpenAiCompletionResponse>, Status> {
        unimplemented!()
    }

    async fn open_ai_chat_completions(
        &self,
        _: Request<api::OpenAiChatCompletionRequest>,
    ) -> Result<Response<api::OpenAiChatCompletionResponse>, Status> {
        unimplemented!()
    }

    async fn open_ai_embeddings(
        &self,
        _: Request<api::OpenAiEmbeddingRequest>,
    ) -> Result<Response<api::OpenAiEmbeddingResponse>, Status> {
        unimplemented!()
    }

    async fn open_ai_classify(
        &self,
        _: Request<api::OpenAiClassifyRequest>,
    ) -> Result<Response<api::OpenAiClassifyResponse>, Status> {
        unimplemented!()
    }

    async fn open_ai_score(
        &self,
        _: Request<api::OpenAiScoreRequest>,
    ) -> Result<Response<api::OpenAiScoreResponse>, Status> {
        unimplemented!()
    }

    async fn open_ai_rerank(
        &self,
        _: Request<api::OpenAiRerankRequest>,
    ) -> Result<Response<api::OpenAiRerankResponse>, Status> {
        unimplemented!()
    }

    async fn open_ai_tokenize(
        &self,
        _: Request<api::OpenAiTokenizeRequest>,
    ) -> Result<Response<api::OpenAiTokenizeResponse>, Status> {
        unimplemented!()
    }

    async fn open_ai_detokenize(
        &self,
        _: Request<api::OpenAiDetokenizeRequest>,
    ) -> Result<Response<api::OpenAiDetokenizeResponse>, Status> {
        unimplemented!()
    }

    async fn open_ai_responses(
        &self,
        _: Request<api::OpenAiResponsesRequest>,
    ) -> Result<Response<api::OpenAiResponsesResponse>, Status> {
        unimplemented!()
    }

    async fn open_ai_list_models(
        &self,
        _: Request<api::OpenAiListModelsRequest>,
    ) -> Result<Response<api::OpenAiModelList>, Status> {
        unimplemented!()
    }

    async fn open_ai_retrieve_model(
        &self,
        _: Request<api::OpenAiRetrieveModelRequest>,
    ) -> Result<Response<api::OpenAiModelCard>, Status> {
        unimplemented!()
    }

    type OpenAiStreamCompletionsStream = ItemStreamOf<api::OpenAiCompletionStreamResponse>;

    async fn open_ai_stream_completions(
        &self,
        _: Request<api::OpenAiCompletionRequest>,
    ) -> Result<Response<Self::OpenAiStreamCompletionsStream>, Status> {
        unimplemented!()
    }

    type OpenAiStreamChatCompletionsStream = ItemStreamOf<api::OpenAiChatCompletionStreamResponse>;

    async fn open_ai_stream_chat_completions(
        &self,
        _: Request<api::OpenAiChatCompletionRequest>,
    ) -> Result<Response<Self::OpenAiStreamChatCompletionsStream>, Status> {
        unimplemented!()
    }

    async fn separate_reasoning(
        &self,
        _: Request<api::SeparateReasoningRequest>,
    ) -> Result<Response<api::SeparateReasoningResponse>, Status> {
        Err(Status::unimplemented("separate_reasoning"))
    }
    async fn add_external_corpus(
        &self,
        _: Request<api::AddExternalCorpusRequest>,
    ) -> Result<Response<api::AddExternalCorpusResponse>, Status> {
        Err(Status::unimplemented("add_external_corpus"))
    }
    async fn remove_external_corpus(
        &self,
        _: Request<api::RemoveExternalCorpusRequest>,
    ) -> Result<Response<api::RemoveExternalCorpusResponse>, Status> {
        Err(Status::unimplemented("remove_external_corpus"))
    }
    async fn list_external_corpora(
        &self,
        _: Request<api::ListExternalCorporaRequest>,
    ) -> Result<Response<api::ListExternalCorporaResponse>, Status> {
        Err(Status::unimplemented("list_external_corpora"))
    }
    async fn freeze_gc(
        &self,
        _: Request<api::FreezeGcRequest>,
    ) -> Result<Response<api::FreezeGcResponse>, Status> {
        Err(Status::unimplemented("freeze_gc"))
    }
    async fn pause_generation(
        &self,
        _: Request<api::PauseGenerationRequest>,
    ) -> Result<Response<api::GenerationPauseResponse>, Status> {
        Err(Status::unimplemented("pause_generation"))
    }
    async fn continue_generation(
        &self,
        _: Request<api::ContinueGenerationRequest>,
    ) -> Result<Response<api::GenerationPauseResponse>, Status> {
        Err(Status::unimplemented("continue_generation"))
    }
    async fn pd_role_switch(
        &self,
        _: Request<api::PdRoleSwitchRequest>,
    ) -> Result<Response<api::PdRoleSwitchResponse>, Status> {
        Err(Status::unimplemented("pd_role_switch"))
    }
    async fn update_weight_version(
        &self,
        _: Request<api::UpdateWeightVersionRequest>,
    ) -> Result<Response<api::UpdateWeightVersionResponse>, Status> {
        Err(Status::unimplemented("update_weight_version"))
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
