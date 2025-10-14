// PD (Prefill-Decode) gRPC Router Implementation

use crate::config::types::RetryConfig;
use crate::core::{ConnectionMode, Worker, WorkerRegistry, WorkerType};
use crate::grpc_client::proto;
use crate::grpc_client::SglangSchedulerClient;
use crate::policies::PolicyRegistry;
use crate::protocols::spec::{
    ChatCompletionRequest, CompletionRequest, EmbeddingRequest, GenerateRequest, InputIds,
    RerankRequest, ResponsesGetParams, ResponsesRequest,
};
use crate::reasoning_parser::ReasoningParserFactory;
use crate::routers::http::pd_types::generate_room_id;
use crate::routers::{grpc, RouterTrait};
use crate::server::AppContext;
use crate::tokenizer::traits::Tokenizer;
use crate::tokenizer::SequenceDecoderOutput;
use crate::tool_parser::ToolParserFactory;
use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use grpc::utils;
use proto::generate_response::Response::{Chunk, Complete, Error};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc::unbounded_channel;
use tokio::sync::mpsc::UnboundedSender;
use tokio_stream::Stream;
use tokio_stream::StreamExt;
use tracing::{debug, error};
use uuid::Uuid;

/// gRPC PD (Prefill-Decode) router implementation for SGLang
#[derive(Clone)]
#[allow(dead_code)] // Fields will be used once implementation is complete
pub struct GrpcPDRouter {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    tokenizer: Arc<dyn Tokenizer>,
    reasoning_parser_factory: ReasoningParserFactory,
    tool_parser_factory: ToolParserFactory,
    dp_aware: bool,
    api_key: Option<String>,
    retry_config: RetryConfig,
    configured_reasoning_parser: Option<String>,
    configured_tool_parser: Option<String>,
    // Pipeline for non-streaming requests
    pipeline: super::pipeline::ChatCompletionPipeline,
    // Shared components for pipeline
    shared_components: Arc<super::context::SharedComponents>,
}

impl GrpcPDRouter {
    /// Create a new gRPC PD router
    pub async fn new(ctx: &Arc<AppContext>) -> Result<Self, String> {
        // Get registries from context
        let worker_registry = ctx.worker_registry.clone();
        let policy_registry = ctx.policy_registry.clone();

        // Extract necessary components from context
        let tokenizer = ctx
            .tokenizer
            .as_ref()
            .ok_or_else(|| "gRPC PD router requires tokenizer".to_string())?
            .clone();
        let reasoning_parser_factory = ctx
            .reasoning_parser_factory
            .as_ref()
            .ok_or_else(|| "gRPC PD router requires reasoning parser factory".to_string())?
            .clone();
        let tool_parser_factory = ctx
            .tool_parser_factory
            .as_ref()
            .ok_or_else(|| "gRPC PD router requires tool parser factory".to_string())?
            .clone();

        // Create shared components for pipeline
        let shared_components = Arc::new(super::context::SharedComponents {
            tokenizer: tokenizer.clone(),
            tool_parser_factory: tool_parser_factory.clone(),
            reasoning_parser_factory: reasoning_parser_factory.clone(),
        });

        // Create response processor
        let processor = super::processing::ResponseProcessor::new(
            tokenizer.clone(),
            tool_parser_factory.clone(),
            reasoning_parser_factory.clone(),
            ctx.configured_tool_parser.clone(),
            ctx.configured_reasoning_parser.clone(),
        );

        // Create streaming processor
        let streaming_processor = Arc::new(super::streaming::StreamingProcessor::new(
            tokenizer.clone(),
            tool_parser_factory.clone(),
            reasoning_parser_factory.clone(),
            ctx.configured_tool_parser.clone(),
            ctx.configured_reasoning_parser.clone(),
        ));

        // Create PD pipeline
        let pipeline = super::pipeline::ChatCompletionPipeline::new_pd(
            worker_registry.clone(),
            policy_registry.clone(),
            processor,
            streaming_processor,
        );

        Ok(GrpcPDRouter {
            worker_registry,
            policy_registry,
            tokenizer,
            reasoning_parser_factory,
            tool_parser_factory,
            dp_aware: ctx.router_config.dp_aware,
            api_key: ctx.router_config.api_key.clone(),
            retry_config: ctx.router_config.effective_retry_config(),
            configured_reasoning_parser: ctx.configured_reasoning_parser.clone(),
            configured_tool_parser: ctx.configured_tool_parser.clone(),
            pipeline,
            shared_components,
        })
    }

    /// Select a prefill-decode worker pair using load balancing policies
    async fn select_pd_pair(
        &self,
        request_text: Option<&str>,
        model_id: Option<&str>,
    ) -> Result<(Arc<dyn Worker>, Arc<dyn Worker>), String> {
        let effective_model_id = if !self.dp_aware { None } else { model_id };

        debug!(
            "Selecting PD pair: dp_aware={}, model_id={:?}, effective_model_id={:?}",
            self.dp_aware, model_id, effective_model_id
        );

        // Get prefill workers
        let prefill_workers = if let Some(model) = effective_model_id {
            self.worker_registry
                .get_by_model_fast(model)
                .into_iter()
                .filter(|w| matches!(w.worker_type(), WorkerType::Prefill { .. }))
                .collect()
        } else {
            self.worker_registry.get_workers_filtered(
                None,
                Some(WorkerType::Prefill {
                    bootstrap_port: None,
                }),
                Some(ConnectionMode::Grpc { port: None }),
                true, // only healthy workers
            )
        };

        // Get decode workers
        let decode_workers = if let Some(model) = effective_model_id {
            self.worker_registry
                .get_by_model_fast(model)
                .into_iter()
                .filter(|w| matches!(w.worker_type(), WorkerType::Decode))
                .collect()
        } else {
            self.worker_registry.get_workers_filtered(
                None,
                Some(WorkerType::Decode),
                Some(ConnectionMode::Grpc { port: None }),
                true, // only healthy workers
            )
        };

        if prefill_workers.is_empty() {
            return Err("No healthy prefill workers available".to_string());
        }
        if decode_workers.is_empty() {
            return Err("No healthy decode workers available".to_string());
        }

        debug!(
            "Found {} prefill workers and {} decode workers",
            prefill_workers.len(),
            decode_workers.len()
        );

        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();

        let prefill_idx = prefill_policy
            .select_worker(&prefill_workers, request_text)
            .ok_or_else(|| "Failed to select prefill worker".to_string())?;

        let decode_idx = decode_policy
            .select_worker(&decode_workers, request_text)
            .ok_or_else(|| "Failed to select decode worker".to_string())?;

        let prefill = prefill_workers[prefill_idx].clone();
        let decode = decode_workers[decode_idx].clone();

        debug!(
            "Selected PD pair: prefill={}, decode={}",
            prefill.url(),
            decode.url()
        );

        Ok((prefill, decode))
    }

    /// Main route_generate implementation with PD dual dispatch
    async fn route_generate_impl(
        &self,
        _headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        debug!(
            "Processing generate request for model: {:?} (PD mode)",
            model_id
        );

        // Step 1: Resolve input (text or input_ids)
        let (original_text, token_ids) = match self.resolve_generate_input(body) {
            Ok(res) => res,
            Err(msg) => {
                return utils::bad_request_error(msg);
            }
        };

        debug!("Resolved input with {} tokens", token_ids.len());

        // Step 2: Select prefill-decode worker pair
        let (prefill_worker, decode_worker) = match self
            .select_pd_pair(original_text.as_deref(), model_id)
            .await
        {
            Ok(pair) => pair,
            Err(e) => {
                return utils::service_unavailable_error(e);
            }
        };

        debug!(
            "Selected PD pair: prefill={}, decode={}",
            prefill_worker.url(),
            decode_worker.url()
        );

        // Step 3: Get gRPC clients for both workers
        let prefill_client = match utils::get_grpc_client_from_worker(&prefill_worker).await {
            Ok(client) => client,
            Err(response) => return response,
        };

        let decode_client = match utils::get_grpc_client_from_worker(&decode_worker).await {
            Ok(client) => client,
            Err(response) => return response,
        };

        // Step 4: Build the gRPC request
        let request_id = body
            .rid
            .clone()
            .unwrap_or_else(|| format!("gen-{}", Uuid::new_v4()));

        let mut request = match prefill_client.build_plain_generate_request(
            request_id.clone(),
            body,
            original_text.clone(),
            token_ids,
        ) {
            Ok(req) => req,
            Err(e) => {
                return utils::bad_request_error(e);
            }
        };

        // Step 5: Inject bootstrap metadata
        if let Err(e) = Self::inject_bootstrap_metadata(&mut request, &*prefill_worker) {
            return utils::internal_error_message(e);
        }

        // Step 6: Get weight version for response metadata
        let weight_version = decode_worker
            .metadata()
            .labels
            .get("weight_version")
            .cloned()
            .unwrap_or_else(|| "default".to_string());

        // Step 7: Handle streaming vs non-streaming
        if body.stream {
            self.handle_streaming_generate(
                prefill_client,
                decode_client,
                request,
                body,
                request_id,
                weight_version,
            )
            .await
        } else {
            self.handle_non_streaming_generate(
                prefill_client,
                decode_client,
                request,
                body,
                request_id,
                weight_version,
            )
            .await
        }
    }

    /// Inject bootstrap metadata into a protobuf GenerateRequest
    fn inject_bootstrap_metadata(
        request: &mut proto::GenerateRequest,
        prefill_worker: &dyn Worker,
    ) -> Result<(), String> {
        let hostname = prefill_worker.bootstrap_host();
        let bootstrap_port = prefill_worker.bootstrap_port().unwrap_or(8998);

        let room_id = generate_room_id();

        // Create DisaggregatedParams
        let disagg_params = proto::DisaggregatedParams {
            bootstrap_host: hostname.to_string(),
            bootstrap_port: bootstrap_port as i32,
            bootstrap_room: room_id as i32,
        };

        // Inject metadata
        request.disaggregated_params = Some(disagg_params);

        debug!(
            "Injected bootstrap metadata: host={}, port={}, room={}",
            hostname, bootstrap_port, room_id
        );

        Ok(())
    }

    /// Main route_chat implementation with PD dual dispatch
    async fn route_chat_impl(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        debug!(
            "Processing chat completion request for model: {:?} (PD mode)",
            model_id
        );

        // Use pipeline for ALL requests (streaming and non-streaming)
        self.pipeline
            .execute_chat(
                body.clone(),
                headers.cloned(),
                model_id.map(|s| s.to_string()),
                self.shared_components.clone(),
            )
            .await
    }

    /// Resolve the generate input into optional original text and token IDs
    fn resolve_generate_input(
        &self,
        request: &GenerateRequest,
    ) -> Result<(Option<String>, Vec<u32>), String> {
        if let Some(text) = &request.text {
            let encoding = self
                .tokenizer
                .encode(text)
                .map_err(|e| format!("Tokenization failed: {}", e))?;
            return Ok((Some(text.to_string()), encoding.token_ids().to_vec()));
        }

        // Handle input_ids - validate and convert
        if let Some(input_ids) = &request.input_ids {
            return match input_ids {
                InputIds::Single(ids) => ids
                    .iter()
                    .map(|&id| u32::try_from(id))
                    .collect::<Result<Vec<u32>, _>>()
                    .map(|converted| (None, converted))
                    .map_err(|_| "input_ids must be non-negative".to_string()),
                InputIds::Batch(_) => {
                    Err("Batch input_ids are not supported in PD mode".to_string())
                }
            };
        }

        Err("Either `text` or `input_ids` must be provided".to_string())
    }

    /// Submit request and handle streaming response for generate endpoint (PD mode)
    async fn handle_streaming_generate(
        &self,
        mut prefill_client: SglangSchedulerClient,
        mut decode_client: SglangSchedulerClient,
        request: proto::GenerateRequest,
        original_request: &GenerateRequest,
        request_id: String,
        weight_version: String,
    ) -> Response {
        // Create channel for SSE streaming
        let (tx, rx) = unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();

        // Send requests in parallel to both prefill and decode workers
        debug!("Starting concurrent streaming generate requests to prefill and decode workers");
        let prefill_request = request.clone();
        let decode_request = request;

        let (prefill_result, decode_result) = tokio::join!(
            prefill_client.generate(prefill_request),
            decode_client.generate(decode_request)
        );

        // Get prefill stream (for input_logprobs if needed)
        let prefill_stream = match prefill_result {
            Ok(s) => s,
            Err(e) => {
                return utils::internal_error_message(format!(
                    "Prefill worker failed to start: {}",
                    e
                ));
            }
        };

        // Get decode stream - this is what we'll process for output
        let decode_stream = match decode_result {
            Ok(s) => s,
            Err(e) => {
                return utils::internal_error_message(format!(
                    "Decode worker failed to start: {}",
                    e
                ));
            }
        };

        // Spawn processing task for both streams
        let tokenizer = self.tokenizer.clone();
        let return_logprob = original_request.return_logprob;
        tokio::spawn(async move {
            let result = Self::process_generate_streaming(
                tokenizer,
                prefill_stream,
                decode_stream,
                request_id,
                weight_version,
                return_logprob,
                &tx,
            )
            .await;

            if let Err(e) = result {
                let error_chunk = format!(
                    "data: {}\n\n",
                    serde_json::json!({
                        "error": {
                            "message": e,
                            "type": "internal_error"
                        }
                    })
                );
                let _ = tx.send(Ok(bytes::Bytes::from(error_chunk)));
            }

            // Send DONE marker
            let _ = tx.send(Ok(bytes::Bytes::from("data: [DONE]\n\n")));
        });

        // Create response with SSE headers
        let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);
        let mut response = Response::new(Body::from_stream(stream));
        *response.status_mut() = StatusCode::OK;
        response.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream"),
        );
        response
            .headers_mut()
            .insert("Cache-Control", HeaderValue::from_static("no-cache"));
        response
            .headers_mut()
            .insert("Connection", HeaderValue::from_static("keep-alive"));
        response
    }

    /// Process generate streaming (simplified - no tool calls or reasoning)
    #[allow(clippy::too_many_arguments)]
    async fn process_generate_streaming(
        tokenizer: Arc<dyn Tokenizer>,
        mut prefill_stream: impl Stream<Item = Result<proto::GenerateResponse, tonic::Status>> + Unpin,
        mut decode_stream: impl Stream<Item = Result<proto::GenerateResponse, tonic::Status>> + Unpin,
        request_id: String,
        weight_version: String,
        include_logprobs: bool,
        tx: &UnboundedSender<Result<bytes::Bytes, std::io::Error>>,
    ) -> Result<(), String> {
        let start_time = Instant::now();

        // Phase 1: Collect input_logprobs from prefill stream if requested
        // TODO: Store and emit input_logprobs when implementing prompt logprobs in streaming
        if include_logprobs {
            while let Some(response) = prefill_stream.next().await {
                let gen_response = response.map_err(|e| format!("Prefill stream error: {}", e))?;
                match gen_response.response {
                    Some(Complete(_complete)) => {
                        // Input logprobs collected but not yet used in streaming
                        break;
                    }
                    Some(Error(error)) => {
                        return Err(format!("Prefill error: {}", error.message));
                    }
                    _ => continue,
                }
            }
        }

        // Phase 2: Main streaming loop (decode stream)
        // Track state per index for n>1 case
        let mut accumulated_texts: HashMap<u32, String> = HashMap::new();
        let mut completion_tokens_map: HashMap<u32, u32> = HashMap::new();
        let mut current_index: u32 = 0;

        while let Some(response) = decode_stream.next().await {
            let gen_response = response.map_err(|e| format!("Decode stream error: {}", e))?;

            match gen_response.response {
                Some(Chunk(chunk)) => {
                    // Use our tracked index instead of chunk.index (PD backend bug workaround)
                    let index = current_index;
                    debug!(
                        "Received chunk with backend_index={}, using_index={}, tokens={:?}",
                        chunk.index, index, chunk.token_ids
                    );

                    let completion_tokens = completion_tokens_map.entry(index).or_insert(0);
                    *completion_tokens += chunk.token_ids.len() as u32;

                    let chunk_text = tokenizer.decode(&chunk.token_ids, true).unwrap_or_default();

                    let accumulated_text = accumulated_texts.entry(index).or_default();
                    accumulated_text.push_str(&chunk_text);

                    let index_id = format!("{}-{}", request_id, index);

                    let chunk_response = serde_json::json!({
                        "text": accumulated_text.clone(),
                        "output_ids": chunk.token_ids,
                        "meta_info": {
                            "id": index_id,
                            "finish_reason": null,
                            "prompt_tokens": chunk.prompt_tokens,
                            "weight_version": weight_version,
                            "completion_tokens": *completion_tokens,
                            "cached_tokens": chunk.cached_tokens
                        },
                        "index": index
                    });

                    let sse_chunk = format!(
                        "data: {}\n\n",
                        serde_json::to_string(&chunk_response).unwrap()
                    );
                    tx.send(Ok(bytes::Bytes::from(sse_chunk)))
                        .map_err(|_| "Failed to send chunk".to_string())?;
                }
                Some(Complete(complete)) => {
                    let index = current_index;
                    debug!(
                        "Received Complete with backend_index={}, using_index={}, finish_reason={}",
                        complete.index, index, complete.finish_reason
                    );
                    let accumulated_text =
                        accumulated_texts.get(&index).cloned().unwrap_or_default();
                    let completion_tokens = *completion_tokens_map.get(&index).unwrap_or(&0);
                    let index_id = format!("{}-{}", request_id, index);
                    let e2e_latency = start_time.elapsed().as_secs_f64();

                    // Send final chunk with finish_reason (no new tokens in Complete, they were already sent in Chunks)
                    let finish_response = serde_json::json!({
                        "text": accumulated_text,
                        "output_ids": complete.output_ids[complete.output_ids.len().saturating_sub(1)..].to_vec(),
                        "meta_info": {
                            "id": index_id,
                            "finish_reason": complete.finish_reason,
                            "prompt_tokens": complete.prompt_tokens,
                            "weight_version": weight_version,
                            "completion_tokens": completion_tokens,
                            "cached_tokens": complete.cached_tokens,
                            "e2e_latency": e2e_latency
                        },
                        "index": index
                    });

                    let sse_chunk = format!(
                        "data: {}\n\n",
                        serde_json::to_string(&finish_response).unwrap()
                    );
                    tx.send(Ok(bytes::Bytes::from(sse_chunk)))
                        .map_err(|_| "Failed to send finish chunk".to_string())?;

                    // Move to next completion
                    current_index += 1;
                }
                Some(Error(error)) => {
                    return Err(error.message);
                }
                None => continue,
            }
        }

        Ok(())
    }

    /// Submit request and handle non-streaming response for generate endpoint (PD mode)
    async fn handle_non_streaming_generate(
        &self,
        mut prefill_client: SglangSchedulerClient,
        mut decode_client: SglangSchedulerClient,
        request: proto::GenerateRequest,
        original_request: &GenerateRequest,
        request_id: String,
        weight_version: String,
    ) -> Response {
        use std::time::Instant;

        let start_time = Instant::now();

        // Send requests in parallel
        debug!("Sending concurrent generate requests to prefill and decode workers");
        let prefill_request = request.clone();
        let decode_request = request;

        let (prefill_result, decode_result) = tokio::join!(
            prefill_client.generate(prefill_request),
            decode_client.generate(decode_request)
        );

        // Process prefill stream
        let prefill_stream = match prefill_result {
            Ok(s) => s,
            Err(e) => {
                error!("Failed to start prefill generation: {}", e);
                return utils::internal_error_message(format!(
                    "Prefill worker failed to start: {}",
                    e
                ));
            }
        };

        let decode_stream = match decode_result {
            Ok(s) => s,
            Err(e) => {
                error!("Failed to start decode generation: {}", e);
                return utils::internal_error_message(format!(
                    "Decode worker failed to start: {}",
                    e
                ));
            }
        };

        // Collect prefill responses
        // TODO add logprob for generate
        let _prefill_responses =
            match utils::collect_stream_responses(prefill_stream, "Prefill").await {
                Ok(responses) => responses,
                Err(error_response) => return error_response,
            };

        // Collect decode responses
        let decode_responses = match utils::collect_stream_responses(decode_stream, "Decode").await
        {
            Ok(responses) => responses,
            Err(error_response) => return error_response,
        };

        if decode_responses.is_empty() {
            return utils::internal_error_static("No completion received from decode worker");
        }

        // Create stop decoder from sampling params
        let params = original_request.sampling_params.as_ref();
        let mut stop_decoder = utils::create_stop_decoder(
            &self.tokenizer,
            params.and_then(|p| p.stop.as_ref()),
            params.and_then(|p| p.stop_token_ids.as_ref()),
            params.and_then(|p| p.skip_special_tokens).unwrap_or(true),
            params.and_then(|p| p.no_stop_trim).unwrap_or(false),
        );

        // Process each completion
        let mut result_array = Vec::new();
        for mut complete in decode_responses {
            stop_decoder.reset();

            // Process tokens through stop decoder
            let outputs = match stop_decoder.process_tokens(&complete.output_ids) {
                Ok(outputs) => outputs,
                Err(e) => {
                    return utils::internal_error_message(format!(
                        "Failed to process tokens: {}",
                        e
                    ))
                }
            };

            // Accumulate text with early breaks
            let mut decoded_text = String::new();
            for output in outputs {
                match output {
                    SequenceDecoderOutput::Text(t) => decoded_text.push_str(&t),
                    SequenceDecoderOutput::StoppedWithText(t) => {
                        decoded_text.push_str(&t);
                        break;
                    }
                    SequenceDecoderOutput::Stopped => break,
                    SequenceDecoderOutput::Held => {}
                }
            }

            // Flush remaining text
            if let SequenceDecoderOutput::Text(t) = stop_decoder.flush() {
                decoded_text.push_str(&t);
            }

            let output_ids = complete.output_ids.clone();

            // Build base meta_info
            let mut meta_info = serde_json::json!({
                "id": request_id.clone(),
                "finish_reason": complete.finish_reason.clone(),
                "prompt_tokens": complete.prompt_tokens,
                "weight_version": weight_version.clone(),
                "completion_tokens": complete.completion_tokens,
                "cached_tokens": complete.cached_tokens,
                "e2e_latency": start_time.elapsed().as_secs_f64(),
            });

            let meta_obj = meta_info.as_object_mut().unwrap();

            // Add matched_stop if present
            if let Some(matched) = complete.matched_stop.take() {
                use proto::generate_complete::MatchedStop;
                let matched_value = match matched {
                    MatchedStop::MatchedTokenId(id) => serde_json::json!(id),
                    MatchedStop::MatchedStopStr(s) => serde_json::json!(s),
                };
                meta_obj.insert("matched_stop".to_string(), matched_value);
            }

            result_array.push(serde_json::json!({
                "text": decoded_text,
                "output_ids": output_ids,
                "meta_info": meta_info,
            }));
        }

        Json(result_array).into_response()
    }
}

impl std::fmt::Debug for GrpcPDRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let prefill_workers = self.worker_registry.get_workers_filtered(
            None,
            Some(WorkerType::Prefill {
                bootstrap_port: None,
            }),
            Some(ConnectionMode::Grpc { port: None }),
            false,
        );
        let decode_workers = self.worker_registry.get_workers_filtered(
            None,
            Some(WorkerType::Decode),
            Some(ConnectionMode::Grpc { port: None }),
            false,
        );
        f.debug_struct("GrpcPDRouter")
            .field("prefill_workers_count", &prefill_workers.len())
            .field("decode_workers_count", &decode_workers.len())
            .field("dp_aware", &self.dp_aware)
            .finish()
    }
}

#[async_trait]
impl RouterTrait for GrpcPDRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        // TODO: Implement actual generation test for gRPC PD mode
        (
            StatusCode::NOT_IMPLEMENTED,
            "Health generate not yet implemented for gRPC PD",
        )
            .into_response()
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn get_models(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn get_model_info(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_generate_impl(headers, body, model_id).await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_chat_impl(headers, body, model_id).await
    }

    async fn route_completion(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &CompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_responses(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &ResponsesRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn get_response(
        &self,
        _headers: Option<&HeaderMap>,
        _response_id: &str,
        _params: &ResponsesGetParams,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn cancel_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_embeddings(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &EmbeddingRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_rerank(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &RerankRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    fn router_type(&self) -> &'static str {
        "grpc_pd"
    }
}
