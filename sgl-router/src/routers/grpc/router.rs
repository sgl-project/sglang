// gRPC Router Implementation

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use bytes::Bytes;
use std::io;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error, warn};

use crate::config::types::RetryConfig;
use crate::core::{ConnectionMode, Worker, WorkerRegistry, WorkerType};
use crate::grpc_client::{proto, SglangSchedulerClient};
use crate::policies::PolicyRegistry;
use crate::protocols::spec::{
    ChatChoice, ChatCompletionMessage, ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionStreamResponse, ChatMessage, ChatMessageDelta, ChatStreamChoice,
    CompletionRequest, EmbeddingRequest, FunctionCallDelta, FunctionCallResponse, GenerateRequest,
    RerankRequest, ResponsesGetParams, ResponsesRequest, StringOrArray, ToolCall, ToolCallDelta,
    ToolChoice, ToolChoiceValue, Usage,
};
use crate::reasoning_parser::{ParserResult, ReasoningParserFactory};
use crate::routers::{grpc, RouterTrait};
use crate::server::AppContext;
use crate::tokenizer::stop::{SequenceDecoderOutput, StopSequenceDecoder};
use crate::tokenizer::traits::Tokenizer;
use crate::tool_parser::{StreamingParseResult, ToolParserFactory};
use grpc::utils;
use proto::generate_response::Response::{Chunk, Complete, Error};
use serde_json::{json, Value};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio_stream::StreamExt;
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
        })
    }

    /// Main route_chat implementation
    async fn route_chat_impl(
        &self,
        _headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        debug!(
            "Processing chat completion request for model: {:?}",
            model_id
        );

        // Step 1: Filter tools if needed for allowed_tools or specific function
        let body_ref = utils::filter_tools_for_request(body);

        // Step 2: Process messages and apply chat template
        let processed_messages = match utils::process_chat_messages(&body_ref, &*self.tokenizer) {
            Ok(msgs) => msgs,
            Err(e) => {
                return utils::bad_request_error(e.to_string());
            }
        };

        // Step 3: Tokenize the processed text
        let encoding = match self.tokenizer.encode(&processed_messages.text) {
            Ok(encoding) => encoding,
            Err(e) => {
                return utils::internal_error_message(format!("Tokenization failed: {}", e));
            }
        };

        let token_ids = encoding.token_ids().to_vec();
        debug!("Tokenized {} tokens from input", token_ids.len());

        // Step 4: Build tool constraints if needed
        // body_ref already has filtered tools if needed
        let tool_call_constraint = body_ref.tools.as_ref().and_then(|tools| {
            utils::generate_tool_constraints(tools, &body.tool_choice, &body.model)
        });

        // Step 5: Select worker
        let worker = match self.select_worker_for_request(model_id, Some(&processed_messages.text))
        {
            Some(w) => w,
            None => {
                return utils::service_unavailable_error(format!(
                    "No available workers for model: {:?}",
                    model_id
                ));
            }
        };

        debug!("Selected worker: {}", worker.url());

        // Step 6: Get gRPC client from worker
        let client = match utils::get_grpc_client_from_worker(&worker).await {
            Ok(client) => client,
            Err(response) => return response,
        };

        // Step 7: Build the base gRPC request (use body_ref with filtered tools if applicable)
        let request_id = format!("chatcmpl-{}", Uuid::new_v4());
        let request = match client.build_generate_request(
            request_id,
            &body_ref,
            processed_messages.text.clone(),
            token_ids,
            processed_messages.multimodal_inputs,
            tool_call_constraint, // Pass the full tuple (type, value)
        ) {
            Ok(request) => request,
            Err(e) => {
                return utils::bad_request_error(format!("Invalid request parameters: {}", e));
            }
        };

        // Step 7: Handle streaming vs non-streaming
        if body.stream {
            self.handle_streaming_chat(client, request, body).await
        } else {
            self.handle_non_streaming_chat(client, request, body).await
        }
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

    /// Parse tool calls using model-specific parser
    async fn parse_tool_calls(
        &self,
        processed_text: &str,
        model: &str,
        history_tool_calls_count: usize,
    ) -> (Option<Vec<ToolCall>>, String) {
        // Get pooled parser for this model
        let pooled_parser = utils::get_tool_parser(
            &self.tool_parser_factory,
            self.configured_tool_parser.as_ref(),
            model,
        );

        // Check format detection first
        let can_parse = {
            let parser = pooled_parser.lock().await;
            parser.has_tool_markers(processed_text)
            // Lock is dropped here
        };

        if !can_parse {
            return (None, processed_text.to_string());
        }

        // Lock again for async parsing
        let result = {
            let parser = pooled_parser.lock().await;
            parser.parse_complete(processed_text).await
            // Lock is dropped here
        };

        match result {
            Ok((normal_text, parsed_tool_calls)) => {
                if parsed_tool_calls.is_empty() {
                    return (None, normal_text);
                }

                let spec_tool_calls = parsed_tool_calls
                    .into_iter()
                    .enumerate()
                    .map(|(index, tc)| {
                        // Generate ID for this tool call
                        let id = Self::generate_tool_call_id(
                            model,
                            &tc.function.name,
                            index,
                            history_tool_calls_count,
                        );
                        ToolCall {
                            id,
                            tool_type: "function".to_string(),
                            function: FunctionCallResponse {
                                name: tc.function.name,
                                arguments: Some(
                                    serde_json::to_string(&tc.function.arguments)
                                        .unwrap_or_else(|_| "{}".to_string()),
                                ),
                            },
                        }
                    })
                    .collect();
                (Some(spec_tool_calls), normal_text)
            }
            Err(e) => {
                error!("Tool call parsing error: {}", e);
                (None, processed_text.to_string())
            }
        }
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
                crate::protocols::spec::InputIds::Single(ids) => ids
                    .iter()
                    .map(|&id| u32::try_from(id))
                    .collect::<Result<Vec<u32>, _>>()
                    .map(|converted| (None, converted))
                    .map_err(|_| "input_ids must be non-negative".to_string()),
                crate::protocols::spec::InputIds::Batch(_) => {
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

    /// Count the number of tool calls in the request message history
    /// This is used for KimiK2 format which needs globally unique indices
    fn get_history_tool_calls_count(request: &ChatCompletionRequest) -> usize {
        request
            .messages
            .iter()
            .filter_map(|msg| {
                if let ChatMessage::Assistant { tool_calls, .. } = msg {
                    tool_calls.as_ref().map(|calls| calls.len())
                } else {
                    None
                }
            })
            .sum()
    }

    /// Generate a tool call ID based on model format
    ///
    /// # Arguments
    /// * `model` - Model name to determine ID format
    /// * `tool_name` - Name of the tool being called
    /// * `tool_index` - Index of this tool call within the current message
    /// * `history_count` - Number of tool calls in previous messages
    ///
    /// # Returns
    /// A unique ID string. KimiK2 uses `functions.{name}:{global_index}`, others use `call_{uuid}`
    fn generate_tool_call_id(
        model: &str,
        tool_name: &str,
        tool_index: usize,
        history_count: usize,
    ) -> String {
        if model.to_lowercase().contains("kimi") {
            // KimiK2 format: functions.{name}:{global_index}
            format!("functions.{}:{}", tool_name, history_count + tool_index)
        } else {
            // Standard OpenAI format: call_{24-char-uuid}
            format!("call_{}", &Uuid::new_v4().simple().to_string()[..24])
        }
    }

    /// Process a chunk of tokens through the stop decoder
    fn process_chunk_tokens(
        stop_decoder: &mut StopSequenceDecoder,
        token_ids: &[u32],
    ) -> (String, bool) {
        let mut chunk_text = String::new();

        for &token_id in token_ids {
            match stop_decoder.process_token(token_id).unwrap_or_else(|e| {
                debug!(
                    "Error processing token {}: {}. Treating as Held.",
                    token_id, e
                );
                SequenceDecoderOutput::Held
            }) {
                SequenceDecoderOutput::Text(text) => {
                    chunk_text.push_str(&text);
                }
                SequenceDecoderOutput::StoppedWithText(text) => {
                    chunk_text.push_str(&text);
                    return (chunk_text, true); // Return text and signal to stop
                }
                SequenceDecoderOutput::Stopped => {
                    return (chunk_text, true); // Return text and signal to stop
                }
                SequenceDecoderOutput::Held => {
                    // Text held for potential stop sequence match
                }
            }
        }
        (chunk_text, false) // Return text and continue processing
    }

    /// Helper: Process reasoning content in streaming mode
    /// Returns (modified_delta, optional_reasoning_chunk)
    fn process_reasoning_stream(
        &self,
        delta: &str,
        index: u32,
        reasoning_parsers: &mut HashMap<
            u32,
            Arc<std::sync::Mutex<Box<dyn crate::reasoning_parser::ReasoningParser>>>,
        >,
        request_id: &str,
        model: &str,
        created: u64,
    ) -> (String, Option<ChatCompletionStreamResponse>, bool) {
        // Get or create parser for this index
        reasoning_parsers.entry(index).or_insert_with(|| {
            utils::get_reasoning_parser(
                &self.reasoning_parser_factory,
                self.configured_reasoning_parser.as_ref(),
                model,
            )
        });

        if let Some(pooled_parser) = reasoning_parsers.get(&index) {
            let (parse_result, in_reasoning) = {
                let mut parser = pooled_parser.lock().unwrap();
                let result = parser.parse_reasoning_streaming_incremental(delta);
                let in_reasoning = parser.is_in_reasoning();
                (result, in_reasoning)
            };

            match parse_result {
                Ok(ParserResult {
                    reasoning_text,
                    normal_text,
                }) => {
                    let chunk = if !reasoning_text.is_empty() {
                        Some(ChatCompletionStreamResponse {
                            id: request_id.to_string(),
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model: model.to_string(),
                            system_fingerprint: None,
                            choices: vec![ChatStreamChoice {
                                index,
                                delta: ChatMessageDelta {
                                    role: Some("assistant".to_string()),
                                    content: None,
                                    tool_calls: None,
                                    reasoning_content: Some(reasoning_text),
                                },
                                logprobs: None,
                                finish_reason: None,
                                matched_stop: None,
                            }],
                            usage: None,
                        })
                    } else {
                        None
                    };
                    return (normal_text, chunk, in_reasoning);
                }
                Err(e) => {
                    warn!("Reasoning parsing error: {}", e);
                }
            }
        }

        (delta.to_string(), None, false)
    }

    /// Helper: Process tool calls in streaming mode
    /// Returns (should_skip_content, chunks_to_emit)
    #[allow(clippy::too_many_arguments)]
    async fn process_tool_calls_stream(
        &self,
        delta: &str,
        index: u32,
        tool_parsers: &mut HashMap<
            u32,
            Arc<tokio::sync::Mutex<Box<dyn crate::tool_parser::ToolParser>>>,
        >,
        has_tool_calls: &mut HashMap<u32, bool>,
        tools: &[crate::protocols::spec::Tool],
        request_id: &str,
        model: &str,
        created: u64,
        history_tool_calls_count: usize,
    ) -> (bool, Vec<ChatCompletionStreamResponse>) {
        let mut chunks = Vec::new();

        // Get or create parser for this index
        tool_parsers.entry(index).or_insert_with(|| {
            utils::get_tool_parser(
                &self.tool_parser_factory,
                self.configured_tool_parser.as_ref(),
                model,
            )
        });

        if let Some(pooled_parser) = tool_parsers.get(&index) {
            let mut parser = pooled_parser.lock().await;
            match parser.parse_incremental(delta, tools).await {
                Ok(StreamingParseResult { normal_text, calls }) => {
                    // Emit normal text if present
                    if !normal_text.is_empty() {
                        chunks.push(ChatCompletionStreamResponse {
                            id: request_id.to_string(),
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model: model.to_string(),
                            system_fingerprint: None,
                            choices: vec![ChatStreamChoice {
                                index,
                                delta: ChatMessageDelta {
                                    role: Some("assistant".to_string()),
                                    content: Some(normal_text),
                                    tool_calls: None,
                                    reasoning_content: None,
                                },
                                logprobs: None,
                                finish_reason: None,
                                matched_stop: None,
                            }],
                            usage: None,
                        });
                    }

                    // Emit tool call chunks
                    for tool_call_item in calls {
                        has_tool_calls.insert(index, true);

                        let tool_call_id = if let Some(ref name) = tool_call_item.name {
                            Some(Self::generate_tool_call_id(
                                model,
                                name,
                                tool_call_item.tool_index,
                                history_tool_calls_count,
                            ))
                        } else {
                            None
                        };

                        let tool_call_delta = ToolCallDelta {
                            index: tool_call_item.tool_index as u32,
                            id: tool_call_id,
                            tool_type: if tool_call_item.name.is_some() {
                                Some("function".to_string())
                            } else {
                                None
                            },
                            function: Some(FunctionCallDelta {
                                name: tool_call_item.name,
                                arguments: if !tool_call_item.parameters.is_empty() {
                                    Some(tool_call_item.parameters)
                                } else {
                                    None
                                },
                            }),
                        };

                        chunks.push(ChatCompletionStreamResponse {
                            id: request_id.to_string(),
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model: model.to_string(),
                            system_fingerprint: None,
                            choices: vec![ChatStreamChoice {
                                index,
                                delta: ChatMessageDelta {
                                    role: Some("assistant".to_string()),
                                    content: None,
                                    tool_calls: Some(vec![tool_call_delta]),
                                    reasoning_content: None,
                                },
                                logprobs: None,
                                finish_reason: None,
                                matched_stop: None,
                            }],
                            usage: None,
                        });
                    }

                    // If we emitted chunks, skip regular content
                    return (!chunks.is_empty(), chunks);
                }
                Err(e) => {
                    warn!("Tool call parsing error: {}", e);
                }
            }
        }

        (false, chunks)
    }

    /// Helper: Create content chunk
    fn create_content_chunk(
        content: String,
        index: u32,
        request_id: &str,
        model: &str,
        created: u64,
        logprobs: Option<crate::protocols::spec::ChatLogProbs>,
    ) -> ChatCompletionStreamResponse {
        ChatCompletionStreamResponse {
            id: request_id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.to_string(),
            system_fingerprint: None,
            choices: vec![ChatStreamChoice {
                index,
                delta: ChatMessageDelta {
                    role: Some("assistant".to_string()),
                    content: Some(content),
                    tool_calls: None,
                    reasoning_content: None,
                },
                logprobs,
                finish_reason: None,
                matched_stop: None,
            }],
            usage: None,
        }
    }

    /// Helper: Format response as SSE chunk
    fn format_sse_chunk(response: &ChatCompletionStreamResponse) -> String {
        format!(
            "data: {}\n\n",
            serde_json::to_string(response).unwrap_or_default()
        )
    }

    /// Submit request and handle streaming response for chat completions route
    async fn handle_streaming_chat(
        &self,
        mut client: SglangSchedulerClient,
        request: proto::GenerateRequest,
        original_request: &ChatCompletionRequest,
    ) -> Response {
        let request_id = request.request_id.clone();
        let model = original_request.model.clone();

        // Create channel for SSE streaming
        let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, io::Error>>();

        // Start the gRPC stream
        let mut grpc_stream = match client.generate(request).await {
            Ok(stream) => stream,
            Err(e) => {
                return utils::internal_error_message(format!("Generation failed: {}", e));
            }
        };

        let stop_params = (
            original_request.stop.clone(),
            original_request.stop_token_ids.clone(),
            original_request.skip_special_tokens,
            original_request.no_stop_trim,
        );

        // Spawn processing task
        let self_clone = self.clone();
        let original_request_clone = original_request.clone();
        tokio::spawn(async move {
            let result = Self::process_streaming_chunks(
                &self_clone,
                &mut grpc_stream,
                request_id,
                model,
                stop_params,
                original_request_clone,
                &tx,
            )
            .await;

            if let Err(e) = result {
                let error_chunk = format!(
                    "data: {}\n\n",
                    json!({
                        "error": {
                            "message": e,
                            "type": "internal_error"
                        }
                    })
                );
                let _ = tx.send(Ok(Bytes::from(error_chunk)));
            }

            // Send DONE marker
            let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
        });

        // Create response with SSE headers
        let stream = UnboundedReceiverStream::new(rx);
        let mut response = Response::new(Body::from_stream(stream));
        *response.status_mut() = StatusCode::OK;
        response
            .headers_mut()
            .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        response
            .headers_mut()
            .insert("Cache-Control", HeaderValue::from_static("no-cache"));
        response
            .headers_mut()
            .insert("Connection", HeaderValue::from_static("keep-alive"));
        response
    }

    /// Process streaming chunks and send SSE events
    async fn process_streaming_chunks(
        router: &GrpcRouter,
        grpc_stream: &mut (impl tokio_stream::Stream<Item = Result<proto::GenerateResponse, tonic::Status>>
                  + Unpin),
        request_id: String,
        model: String,
        stop_params: (Option<StringOrArray>, Option<Vec<u32>>, bool, bool),
        original_request: ChatCompletionRequest,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    ) -> Result<(), String> {
        // Extract request parameters
        let separate_reasoning = original_request.separate_reasoning;
        let tool_choice = &original_request.tool_choice;
        let tools = &original_request.tools;
        let history_tool_calls_count = Self::get_history_tool_calls_count(&original_request);
        let stream_options = &original_request.stream_options;

        // Phase 1: Initialize state tracking (per-index for n>1 support)
        let mut is_firsts: HashMap<u32, bool> = HashMap::new();
        let mut stream_buffers: HashMap<u32, String> = HashMap::new();
        let mut finish_reasons: HashMap<u32, String> = HashMap::new();
        let mut matched_stops: HashMap<u32, Option<Value>> = HashMap::new();
        let mut prompt_tokens: HashMap<u32, u32> = HashMap::new();
        let mut completion_tokens: HashMap<u32, u32> = HashMap::new();
        let mut cached_tokens: HashMap<u32, u32> = HashMap::new();

        // Parser state (lazy initialization per index)
        type PooledReasoningParser =
            Arc<std::sync::Mutex<Box<dyn crate::reasoning_parser::ReasoningParser>>>;
        let mut reasoning_parsers: HashMap<u32, PooledReasoningParser> = HashMap::new();

        type PooledToolParser = Arc<tokio::sync::Mutex<Box<dyn crate::tool_parser::ToolParser>>>;
        let mut tool_parsers: HashMap<u32, PooledToolParser> = HashMap::new();
        let mut has_tool_calls: HashMap<u32, bool> = HashMap::new();

        // Create stop decoder
        let (stop, stop_token_ids, skip_special_tokens, no_stop_trim) = stop_params;
        let mut stop_decoder = utils::create_stop_decoder(
            &router.tokenizer,
            stop.as_ref(),
            stop_token_ids.as_ref(),
            skip_special_tokens,
            no_stop_trim,
        );

        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Phase 2: Main streaming loop
        while let Some(response) = grpc_stream.next().await {
            let gen_response = response.map_err(|e| format!("Stream error: {}", e))?;

            match gen_response.response {
                Some(Chunk(chunk)) => {
                    let index = chunk.index;

                    // Process tokens through stop decoder
                    let (chunk_text, _should_stop) =
                        Self::process_chunk_tokens(&mut stop_decoder, &chunk.token_ids);

                    if chunk_text.is_empty() {
                        continue;
                    }

                    // Process logprobs if present
                    let choice_logprobs = if let Some(ref proto_logprobs) = chunk.output_logprobs {
                        match router.convert_proto_to_openai_logprobs(proto_logprobs) {
                            Ok(logprobs) => Some(logprobs),
                            Err(e) => {
                                warn!("Failed to process logprobs: {}", e);
                                None
                            }
                        }
                    } else {
                        None
                    };

                    // Initialize stream buffer if first time
                    let stream_buffer = stream_buffers.entry(index).or_default();

                    // Send first chunk with role
                    if is_firsts.get(&index).copied().unwrap_or(true) {
                        let first_chunk = ChatCompletionStreamResponse {
                            id: request_id.clone(),
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model: model.clone(),
                            system_fingerprint: None,
                            choices: vec![ChatStreamChoice {
                                index,
                                delta: ChatMessageDelta {
                                    role: Some("assistant".to_string()),
                                    content: None,
                                    tool_calls: None,
                                    reasoning_content: None,
                                },
                                logprobs: None,
                                finish_reason: None,
                                matched_stop: None,
                            }],
                            usage: None,
                        };
                        tx.send(Ok(Bytes::from(Self::format_sse_chunk(&first_chunk))))
                            .map_err(|_| "Failed to send first chunk".to_string())?;
                        is_firsts.insert(index, false);
                    }

                    // Calculate delta
                    let mut delta = chunk_text;
                    stream_buffer.push_str(&delta);

                    // Reasoning content handling
                    let in_reasoning = if separate_reasoning {
                        let (normal_text, reasoning_chunk, in_reasoning) = router
                            .process_reasoning_stream(
                                &delta,
                                index,
                                &mut reasoning_parsers,
                                &request_id,
                                &model,
                                created,
                            );
                        if let Some(chunk) = reasoning_chunk {
                            tx.send(Ok(Bytes::from(Self::format_sse_chunk(&chunk))))
                                .map_err(|_| "Failed to send reasoning chunk".to_string())?;
                        }
                        delta = normal_text;
                        in_reasoning
                    } else {
                        false
                    };

                    // Tool call handling
                    let tool_choice_enabled =
                        !matches!(tool_choice, Some(ToolChoice::Value(ToolChoiceValue::None)));

                    if !in_reasoning && tool_choice_enabled && tools.is_some() {
                        let (should_skip, tool_chunks) = router
                            .process_tool_calls_stream(
                                &delta,
                                index,
                                &mut tool_parsers,
                                &mut has_tool_calls,
                                tools.as_ref().unwrap(),
                                &request_id,
                                &model,
                                created,
                                history_tool_calls_count,
                            )
                            .await;

                        for chunk in tool_chunks {
                            tx.send(Ok(Bytes::from(Self::format_sse_chunk(&chunk))))
                                .map_err(|_| "Failed to send tool call chunk".to_string())?;
                        }

                        if should_skip {
                            continue;
                        }
                    }

                    // Regular content emission
                    if !delta.is_empty() {
                        let content_chunk = Self::create_content_chunk(
                            delta,
                            index,
                            &request_id,
                            &model,
                            created,
                            choice_logprobs,
                        );
                        tx.send(Ok(Bytes::from(Self::format_sse_chunk(&content_chunk))))
                            .map_err(|_| "Failed to send content chunk".to_string())?;
                    }
                }
                Some(Complete(complete)) => {
                    // Flush any remaining text
                    if let SequenceDecoderOutput::Text(text) = stop_decoder.flush() {
                        if !text.is_empty() {
                            let index = complete.index;
                            let stream_buffer = stream_buffers.entry(index).or_default();
                            stream_buffer.push_str(&text);

                            let content_chunk = ChatCompletionStreamResponse {
                                id: request_id.clone(),
                                object: "chat.completion.chunk".to_string(),
                                created,
                                model: model.clone(),
                                system_fingerprint: None,
                                choices: vec![ChatStreamChoice {
                                    index,
                                    delta: ChatMessageDelta {
                                        role: Some("assistant".to_string()),
                                        content: Some(text),
                                        tool_calls: None,
                                        reasoning_content: None,
                                    },
                                    logprobs: None,
                                    finish_reason: None,
                                    matched_stop: None,
                                }],
                                usage: None,
                            };

                            let sse_chunk = serde_json::to_string(&content_chunk)
                                .map_err(|e| format!("Failed to serialize content chunk: {}", e))?;
                            tx.send(Ok(Bytes::from(format!("data: {}\n\n", sse_chunk))))
                                .map_err(|_| "Failed to send flushed content".to_string())?;
                        }
                    }

                    // Store metadata
                    let index = complete.index;
                    prompt_tokens.insert(index, complete.prompt_tokens as u32);
                    completion_tokens.insert(index, complete.completion_tokens as u32);
                    cached_tokens.insert(index, complete.cached_tokens as u32);
                    finish_reasons.insert(index, complete.finish_reason.clone());

                    // Extract matched_stop
                    let matched_stop_value = match &complete.matched_stop {
                        Some(proto::generate_complete::MatchedStop::MatchedTokenId(token_id)) => {
                            Some(Value::Number(serde_json::Number::from(*token_id)))
                        }
                        Some(proto::generate_complete::MatchedStop::MatchedStopStr(stop_str)) => {
                            Some(Value::String(stop_str.clone()))
                        }
                        None => None,
                    };
                    matched_stops.insert(index, matched_stop_value);

                    break;
                }
                Some(Error(error)) => {
                    return Err(error.message);
                }
                None => continue,
            }
        }

        // Phase 3: Check unstreamed tool args
        // Check if parsers have any remaining arguments that haven't been streamed yet
        for (index, parser) in &tool_parsers {
            let parser_guard = parser.lock().await;
            if let Some(unstreamed_items) = parser_guard.get_unstreamed_tool_args() {
                for tool_call_item in unstreamed_items {
                    let tool_call_delta = ToolCallDelta {
                        index: tool_call_item.tool_index as u32,
                        id: None,
                        tool_type: None, // No type for argument deltas
                        function: Some(FunctionCallDelta {
                            name: None, // No name for argument deltas
                            arguments: if !tool_call_item.parameters.is_empty() {
                                Some(tool_call_item.parameters)
                            } else {
                                None
                            },
                        }),
                    };

                    let tool_chunk = ChatCompletionStreamResponse {
                        id: request_id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model.clone(),
                        system_fingerprint: None,
                        choices: vec![ChatStreamChoice {
                            index: *index,
                            delta: ChatMessageDelta {
                                role: Some("assistant".to_string()),
                                content: None,
                                tool_calls: Some(vec![tool_call_delta]),
                                reasoning_content: None,
                            },
                            logprobs: None,
                            finish_reason: None,
                            matched_stop: None,
                        }],
                        usage: None,
                    };

                    let sse_chunk = serde_json::to_string(&tool_chunk)
                        .map_err(|e| format!("Failed to serialize tool chunk: {}", e))?;
                    tx.send(Ok(Bytes::from(format!("data: {}\n\n", sse_chunk))))
                        .map_err(|_| "Failed to send unstreamed tool args".to_string())?;
                }
            }
        }

        // Phase 4: Finish reason chunks
        for (index, finish_reason) in finish_reasons.iter() {
            let final_finish_reason =
                if has_tool_calls.get(index).copied().unwrap_or(false) && finish_reason == "stop" {
                    "tool_calls".to_string()
                } else {
                    finish_reason.clone()
                };

            let matched_stop_value = matched_stops.get(index).and_then(|v| v.clone());

            let finish_chunk = ChatCompletionStreamResponse {
                id: request_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model.clone(),
                system_fingerprint: None,
                choices: vec![ChatStreamChoice {
                    index: *index,
                    delta: ChatMessageDelta {
                        role: Some("assistant".to_string()),
                        content: None,
                        tool_calls: None,
                        reasoning_content: None,
                    },
                    logprobs: None,
                    finish_reason: Some(final_finish_reason),
                    matched_stop: matched_stop_value,
                }],
                usage: None,
            };

            let sse_chunk = serde_json::to_string(&finish_chunk)
                .map_err(|e| format!("Failed to serialize finish chunk: {}", e))?;
            tx.send(Ok(Bytes::from(format!("data: {}\n\n", sse_chunk))))
                .map_err(|_| "Failed to send finish chunk".to_string())?;
        }

        // Phase 5: Usage chunk
        if let Some(stream_opts) = stream_options {
            if stream_opts.include_usage.unwrap_or(false) {
                let total_prompt: u32 = prompt_tokens.values().sum();
                let total_completion: u32 = completion_tokens.values().sum();

                let usage_chunk = ChatCompletionStreamResponse {
                    id: request_id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model.clone(),
                    system_fingerprint: None,
                    choices: vec![],
                    usage: Some(Usage {
                        prompt_tokens: total_prompt,
                        completion_tokens: total_completion,
                        total_tokens: total_prompt + total_completion,
                        completion_tokens_details: None,
                    }),
                };

                let sse_chunk = serde_json::to_string(&usage_chunk)
                    .map_err(|e| format!("Failed to serialize usage chunk: {}", e))?;
                tx.send(Ok(Bytes::from(format!("data: {}\n\n", sse_chunk))))
                    .map_err(|_| "Failed to send usage chunk".to_string())?;
            }
        }

        Ok(())
    }

    /// Submit request and handle non-streaming response for chat completions route
    async fn handle_non_streaming_chat(
        &self,
        mut client: SglangSchedulerClient,
        request: proto::GenerateRequest,
        original_request: &ChatCompletionRequest,
    ) -> Response {
        let mut stop_decoder = utils::create_stop_decoder(
            &self.tokenizer,
            original_request.stop.as_ref(),
            original_request.stop_token_ids.as_ref(),
            original_request.skip_special_tokens,
            original_request.no_stop_trim,
        );

        // Start generation
        let stream = match client.generate(request).await {
            Ok(s) => s,
            Err(e) => {
                return utils::internal_error_message(format!("Failed to start generation: {}", e))
            }
        };

        let all_responses = match utils::collect_stream_responses(stream, "Regular").await {
            Ok(responses) => responses,
            Err(err_response) => return err_response,
        };

        if all_responses.is_empty() {
            return utils::internal_error_static("No responses from server");
        }

        // Process each response into a ChatChoice
        let history_tool_calls_count = Self::get_history_tool_calls_count(original_request);
        let mut choices = Vec::new();
        for (index, complete) in all_responses.iter().enumerate() {
            match self
                .process_single_choice(
                    complete,
                    index,
                    original_request,
                    &mut stop_decoder,
                    history_tool_calls_count,
                )
                .await
            {
                Ok(choice) => choices.push(choice),
                Err(e) => {
                    return utils::internal_error_message(format!(
                        "Failed to process choice {}: {}",
                        index, e
                    ));
                }
            }
        }

        // Aggregate usage information from all responses
        let total_prompt_tokens: u32 = all_responses.iter().map(|r| r.prompt_tokens as u32).sum();
        let total_completion_tokens: u32 = all_responses
            .iter()
            .map(|r| r.completion_tokens as u32)
            .sum();
        let usage = Usage {
            prompt_tokens: total_prompt_tokens,
            completion_tokens: total_completion_tokens,
            total_tokens: total_prompt_tokens + total_completion_tokens,
            completion_tokens_details: None,
        };

        // Build final ChatCompletionResponse
        let response = ChatCompletionResponse {
            id: format!("chatcmpl-{}", Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            model: original_request.model.clone(),
            choices,
            usage: Some(usage),
            system_fingerprint: None,
        };

        // Serialize and return JSON response
        Json(response).into_response()
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

    /// Convert proto LogProbs to OpenAI ChatLogProbs format
    /// Note: Always decodes with skip_special_tokens=false to show actual tokens generated
    fn convert_proto_to_openai_logprobs(
        &self,
        proto_logprobs: &proto::OutputLogProbs,
    ) -> Result<crate::protocols::spec::ChatLogProbs, String> {
        let mut content_items = Vec::new();

        // Decode token IDs to text (always with skip_special_tokens=false for logprobs)
        let token_texts: Vec<String> = proto_logprobs
            .token_ids
            .iter()
            .map(|&token_id| {
                self.tokenizer
                    .decode(&[token_id as u32], false)
                    .unwrap_or_else(|_| format!("<token_{}>", token_id))
            })
            .collect();

        // Build ChatLogProbsContent for each token (consume iterator to avoid clones)
        for (i, (&logprob, token_text)) in proto_logprobs
            .token_logprobs
            .iter()
            .zip(token_texts.into_iter())
            .enumerate()
        {
            let bytes = Some(token_text.as_bytes().to_vec());

            // Build top_logprobs for this position
            let mut top_logprobs = Vec::new();
            if let Some(top_logprobs_entry) = proto_logprobs.top_logprobs.get(i) {
                // Decode top token IDs (always with skip_special_tokens=false)
                let top_token_texts: Vec<String> = top_logprobs_entry
                    .token_ids
                    .iter()
                    .map(|&tid| {
                        self.tokenizer
                            .decode(&[tid as u32], false)
                            .unwrap_or_else(|_| format!("<token_{}>", tid))
                    })
                    .collect();

                for (j, (&top_logprob, &_top_token_id)) in top_logprobs_entry
                    .values
                    .iter()
                    .zip(top_logprobs_entry.token_ids.iter())
                    .enumerate()
                {
                    if let Some(top_token_text) = top_token_texts.get(j) {
                        top_logprobs.push(crate::protocols::spec::TopLogProb {
                            token: top_token_text.clone(),
                            logprob: top_logprob,
                            bytes: Some(top_token_text.as_bytes().to_vec()),
                        });
                    }
                }
            }

            content_items.push(crate::protocols::spec::ChatLogProbsContent {
                token: token_text,
                logprob,
                bytes,
                top_logprobs,
            });
        }

        Ok(crate::protocols::spec::ChatLogProbs::Detailed {
            content: (!content_items.is_empty()).then_some(content_items),
        })
    }

    /// Process a single GenerateComplete response into a ChatChoice
    async fn process_single_choice(
        &self,
        complete: &proto::GenerateComplete,
        index: usize,
        original_request: &ChatCompletionRequest,
        stop_decoder: &mut StopSequenceDecoder,
        history_tool_calls_count: usize,
    ) -> Result<ChatChoice, String> {
        stop_decoder.reset();
        // Decode tokens
        let outputs = stop_decoder
            .process_tokens(&complete.output_ids)
            .map_err(|e| format!("Failed to process tokens: {}", e))?;

        // Accumulate text with early breaks
        let mut final_text = String::new();
        for output in outputs {
            match output {
                SequenceDecoderOutput::Text(t) => final_text.push_str(&t),
                SequenceDecoderOutput::StoppedWithText(t) => {
                    final_text.push_str(&t);
                    break;
                }
                SequenceDecoderOutput::Stopped => break,
                SequenceDecoderOutput::Held => {}
            }
        }

        // Flush remaining text
        if let SequenceDecoderOutput::Text(t) = stop_decoder.flush() {
            final_text.push_str(&t);
        }

        // Step 1: Handle reasoning content parsing
        let mut reasoning_text: Option<String> = None;
        let mut processed_text = final_text;

        // Check if reasoning parsing is enabled and separate_reasoning is requested
        if original_request.separate_reasoning {
            let pooled_parser = utils::get_reasoning_parser(
                &self.reasoning_parser_factory,
                self.configured_reasoning_parser.as_ref(),
                &original_request.model,
            );

            let mut parser = pooled_parser
                .lock()
                .map_err(|e| format!("Failed to acquire reasoning parser lock: {}", e))?;
            match parser.detect_and_parse_reasoning(&processed_text) {
                Ok(result) => {
                    if !result.reasoning_text.is_empty() {
                        reasoning_text = Some(result.reasoning_text);
                    }
                    processed_text = result.normal_text;
                }
                Err(e) => {
                    return Err(format!("Reasoning parsing error: {}", e));
                }
            }
        }

        // Step 2: Handle tool call parsing
        let mut tool_calls: Option<Vec<ToolCall>> = None;

        // Check if tool calls should be processed
        let tool_choice_enabled = !matches!(
            &original_request.tool_choice,
            Some(ToolChoice::Value(ToolChoiceValue::None))
        );

        if tool_choice_enabled && original_request.tools.is_some() {
            // Check if JSON schema constraint was used (specific function or required mode)
            let used_json_schema = match &original_request.tool_choice {
                Some(ToolChoice::Function { .. }) => true,
                Some(ToolChoice::Value(ToolChoiceValue::Required)) => true,
                Some(ToolChoice::AllowedTools { mode, .. }) => mode == "required",
                _ => false,
            };

            if used_json_schema {
                (tool_calls, processed_text) = utils::parse_json_schema_response(
                    &processed_text,
                    &original_request.tool_choice,
                );
            } else {
                (tool_calls, processed_text) = self
                    .parse_tool_calls(
                        &processed_text,
                        &original_request.model,
                        history_tool_calls_count,
                    )
                    .await;
            }
        }

        // Step 3: Use finish reason directly from proto (already OpenAI-compatible string)
        let finish_reason_str = &complete.finish_reason;

        // Override finish reason if we have tool calls
        let final_finish_reason_str = if tool_calls.is_some() {
            "tool_calls"
        } else {
            finish_reason_str
        };

        // Extract matched_stop information from proto
        let matched_stop = match &complete.matched_stop {
            Some(proto::generate_complete::MatchedStop::MatchedTokenId(token_id)) => {
                Some(Value::Number(serde_json::Number::from(*token_id)))
            }
            Some(proto::generate_complete::MatchedStop::MatchedStopStr(stop_str)) => {
                Some(Value::String(stop_str.clone()))
            }
            None => None,
        };

        // Step 4: Convert output logprobs if present
        // Note: complete.input_logprobs exists in proto but is not used for chat completions
        //       (input logprobs are only used in /v1/completions endpoint with echo=true)
        let logprobs = if let Some(proto_logprobs) = &complete.output_logprobs {
            match self.convert_proto_to_openai_logprobs(proto_logprobs) {
                Ok(logprobs) => Some(logprobs),
                Err(e) => {
                    error!("Failed to convert logprobs: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Step 5: Build ChatCompletionMessage (proper response message type)
        let chat_message = ChatCompletionMessage {
            role: "assistant".to_string(),
            content: if processed_text.is_empty() {
                None
            } else {
                Some(processed_text)
            },
            tool_calls,
            reasoning_content: reasoning_text,
        };

        // Step 6: Build ChatChoice
        let choice = ChatChoice {
            index: index as u32,
            message: chat_message,
            logprobs,
            finish_reason: Some(final_finish_reason_str.to_string()),
            matched_stop,
            hidden_states: None,
        };

        Ok(choice)
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
