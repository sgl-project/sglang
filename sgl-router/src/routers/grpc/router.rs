// gRPC Router Implementation

use std::sync::Arc;

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use tracing::debug;

use crate::config::types::RetryConfig;
use crate::core::{ConnectionMode, Worker, WorkerRegistry, WorkerType};
use crate::grpc_client::{proto, SglangSchedulerClient};
use crate::policies::PolicyRegistry;
use crate::protocols::spec::{
    ChatCompletionRequest, CompletionRequest, EmbeddingRequest, GenerateRequest, InputIds,
    RerankRequest, ResponsesGetParams, ResponsesRequest,
};
use crate::reasoning_parser::ReasoningParserFactory;
use crate::routers::{grpc, RouterTrait};
use crate::server::AppContext;
use crate::tokenizer::stop::SequenceDecoderOutput;
use crate::tokenizer::traits::Tokenizer;
use crate::tool_parser::ToolParserFactory;
use grpc::utils;
use serde_json::json;
use std::time::Instant;
use uuid::Uuid;

/// gRPC router implementation for SGLang
#[derive(Clone)]
#[allow(dead_code)]
pub struct GrpcRouter {
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

impl GrpcRouter {
    /// Create a new gRPC router
    pub async fn new(ctx: &Arc<AppContext>) -> Result<Self, String> {
        // Extract necessary components from context
        let tokenizer = ctx
            .tokenizer
            .as_ref()
            .ok_or_else(|| "gRPC router requires tokenizer".to_string())?
            .clone();
        let reasoning_parser_factory = ctx
            .reasoning_parser_factory
            .as_ref()
            .ok_or_else(|| "gRPC router requires reasoning parser factory".to_string())?
            .clone();
        let tool_parser_factory = ctx
            .tool_parser_factory
            .as_ref()
            .ok_or_else(|| "gRPC router requires tool parser factory".to_string())?
            .clone();

        let worker_registry = ctx.worker_registry.clone();
        let policy_registry = ctx.policy_registry.clone();

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

        // Create pipeline
        let pipeline = super::pipeline::ChatCompletionPipeline::new_regular(
            worker_registry.clone(),
            policy_registry.clone(),
            processor,
            streaming_processor,
        );

        Ok(GrpcRouter {
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

    /// Main route_chat implementation
    async fn route_chat_impl(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        debug!(
            "Processing chat completion request for model: {:?}",
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

    /// Main route_generate implementation
    async fn route_generate_impl(
        &self,
        _headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        debug!("Processing generate request for model: {:?}", model_id);

        // Step 1: Resolve input (text, prompt, or input_ids)
        let (original_text, token_ids) = match self.resolve_generate_input(body) {
            Ok(res) => res,
            Err(msg) => {
                return utils::bad_request_error(msg);
            }
        };

        debug!("Resolved input with {} tokens", token_ids.len());

        // Step 2: Select worker (fail fast if no workers available)
        let worker = match self.select_worker_for_request(model_id, original_text.as_deref()) {
            Some(w) => w,
            None => {
                return utils::service_unavailable_error(format!(
                    "No available workers for model: {:?}",
                    model_id
                ));
            }
        };

        debug!("Selected worker: {}", worker.url());

        // Step 3: Get gRPC client from worker
        let client = match utils::get_grpc_client_from_worker(&worker).await {
            Ok(client) => client,
            Err(response) => return response,
        };

        // Step 4: Build the gRPC request
        let request_id = body
            .rid
            .clone()
            .unwrap_or_else(|| format!("gen-{}", Uuid::new_v4()));

        let request = match client.build_plain_generate_request(
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

        // Step 5: Get weight version for response metadata
        let weight_version = worker
            .metadata()
            .labels
            .get("weight_version")
            .cloned()
            .unwrap_or_else(|| "default".to_string());

        // Step 6: Handle streaming vs non-streaming
        if body.stream {
            self.handle_streaming_generate(client, request, body, request_id, weight_version)
                .await
        } else {
            self.handle_non_streaming_generate(client, request, body, request_id, weight_version)
                .await
        }
    }

    /// Select a worker for the request
    fn select_worker_for_request(
        &self,
        model_id: Option<&str>,
        text: Option<&str>,
    ) -> Option<Arc<dyn Worker>> {
        // Get workers for the specified model, filtered by connection mode
        let workers = self.worker_registry.get_workers_filtered(
            model_id,
            Some(WorkerType::Regular),
            Some(ConnectionMode::Grpc { port: None }),
            false, // get all workers, we'll filter by is_available() next
        );

        // Filter by availability (health + circuit breaker)
        let available: Vec<Arc<dyn Worker>> = workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();

        if available.is_empty() {
            return None;
        }

        // Get the appropriate policy for this model
        let policy = match model_id {
            Some(model) => self.policy_registry.get_policy_or_default(model),
            None => self.policy_registry.get_default_policy(),
        };

        // Select worker using the policy
        let idx = policy.select_worker(&available, text)?;
        Some(available[idx].clone())
    }

    /// Resolve the generate input into optional original text and token IDs
    fn resolve_generate_input(
        &self,
        request: &GenerateRequest,
    ) -> Result<(Option<String>, Vec<u32>), String> {
        if let Some(text) = &request.text {
            return self
                .tokenize_single_text(text)
                .map(|(original, ids)| (Some(original), ids));
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
                    Err("Batch input_ids are not supported over gRPC generate yet".to_string())
                }
            };
        }

        Err("Either `text` or `input_ids` must be provided".to_string())
    }

    fn tokenize_single_text(&self, text: &str) -> Result<(String, Vec<u32>), String> {
        let encoding = self
            .tokenizer
            .encode(text)
            .map_err(|e| format!("Tokenization failed: {}", e))?;
        Ok((text.to_string(), encoding.token_ids().to_vec()))
    }

    /// Submit request and handle non-streaming response for the `/generate` endpoint
    async fn handle_non_streaming_generate(
        &self,
        mut client: SglangSchedulerClient,
        request: proto::GenerateRequest,
        original_request: &GenerateRequest,
        request_id: String,
        weight_version: String,
    ) -> Response {
        let start_time = Instant::now();

        let stream = match client.generate(request).await {
            Ok(stream) => stream,
            Err(e) => {
                return utils::internal_error_message(format!("Failed to start generation: {}", e))
            }
        };

        // Collect all responses using utils helper
        let responses = match utils::collect_stream_responses(stream, "Generate").await {
            Ok(responses) => responses,
            Err(error_response) => return error_response,
        };

        if responses.is_empty() {
            return utils::internal_error_static("No completion received from scheduler");
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
        for mut complete in responses {
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

            let output_ids = std::mem::take(&mut complete.output_ids);
            let finish_reason = std::mem::take(&mut complete.finish_reason);

            // Build base meta_info using json! macro
            let mut meta_info = json!({
                "id": request_id.clone(),
                "finish_reason": finish_reason,
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
                    MatchedStop::MatchedTokenId(id) => json!(id),
                    MatchedStop::MatchedStopStr(s) => json!(s),
                };
                meta_obj.insert("matched_stop".to_string(), matched_value);
            }

            result_array.push(json!({
                "text": decoded_text,
                "output_ids": output_ids,
                "meta_info": meta_info,
            }));
        }

        Json(result_array).into_response()
    }

    /// Submit request and handle streaming response for the `/generate` endpoint
    async fn handle_streaming_generate(
        &self,
        mut client: SglangSchedulerClient,
        request: proto::GenerateRequest,
        original_request: &GenerateRequest,
        request_id: String,
        weight_version: String,
    ) -> Response {
        let tokenizer = self.tokenizer.clone();
        let return_logprob = original_request.return_logprob;

        // Create channel for SSE streaming
        let (tx, rx) =
            tokio::sync::mpsc::unbounded_channel::<Result<bytes::Bytes, std::io::Error>>();

        // Start the stream
        let stream = match client.generate(request).await {
            Ok(stream) => stream,
            Err(e) => {
                return utils::internal_error_message(format!("Failed to start generation: {}", e))
            }
        };

        // Spawn async task to process stream
        tokio::spawn(async move {
            let result = Self::process_generate_streaming(
                tokenizer,
                stream,
                request_id,
                weight_version,
                return_logprob,
                &tx,
            )
            .await;

            if let Err(e) = result {
                let error_chunk = format!("data: {{\"error\": \"{}\"}}\n\n", e);
                let _ = tx.send(Ok(bytes::Bytes::from(error_chunk)));
            }

            // Send [DONE] marker
            let _ = tx.send(Ok(bytes::Bytes::from("data: [DONE]\n\n")));
        });

        // Create SSE response stream
        let body_stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);
        Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "text/event-stream")
            .header("Cache-Control", "no-cache")
            .header("Connection", "keep-alive")
            .body(axum::body::Body::from_stream(body_stream))
            .unwrap()
    }

    /// Process streaming chunks for generate endpoint
    async fn process_generate_streaming(
        tokenizer: Arc<dyn Tokenizer>,
        mut stream: impl tokio_stream::Stream<Item = Result<proto::GenerateResponse, tonic::Status>>
            + Unpin,
        request_id: String,
        weight_version: String,
        _include_logprobs: bool,
        tx: &tokio::sync::mpsc::UnboundedSender<Result<bytes::Bytes, std::io::Error>>,
    ) -> Result<(), String> {
        use proto::generate_response::Response::{Chunk, Complete, Error};
        use std::time::Instant;
        use tokio_stream::StreamExt;

        let start_time = Instant::now();

        // Track state per index for n>1 case
        use std::collections::HashMap;
        let mut accumulated_texts: HashMap<u32, String> = HashMap::new();
        let mut completion_tokens_map: HashMap<u32, u32> = HashMap::new();

        while let Some(response) = stream.next().await {
            let gen_response = response.map_err(|e| format!("Stream error: {}", e))?;

            match gen_response.response {
                Some(Chunk(chunk)) => {
                    let index = chunk.index;

                    // Update completion tokens for this index
                    let completion_tokens = completion_tokens_map.entry(index).or_insert(0);
                    *completion_tokens += chunk.token_ids.len() as u32;

                    // Decode tokens to text (skip_special_tokens=true to handle newlines correctly)
                    let chunk_text = tokenizer.decode(&chunk.token_ids, true).unwrap_or_default();

                    // Accumulate text for this index
                    let accumulated_text = accumulated_texts.entry(index).or_default();
                    accumulated_text.push_str(&chunk_text);

                    // Generate unique ID per index
                    let index_id = format!("{}-{}", request_id, index);

                    // Build streaming response chunk (SGLang format)
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
                    let index = complete.index;
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

                    // Continue to process all completions if n>1
                }
                Some(Error(error)) => {
                    return Err(error.message);
                }
                None => continue,
            }
        }

        Ok(())
    }
}

impl std::fmt::Debug for GrpcRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.worker_registry.stats();
        f.debug_struct("GrpcRouter")
            .field("workers_count", &stats.total_workers)
            .field("dp_aware", &self.dp_aware)
            .finish()
    }
}

#[async_trait]
impl RouterTrait for GrpcRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        // TODO: Implement actual generation test for gRPC
        (
            StatusCode::NOT_IMPLEMENTED,
            "Health generate not yet implemented for gRPC",
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
        "grpc"
    }
}
