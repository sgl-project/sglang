//! Streaming response processor for gRPC routers
//!
//! This module contains shared streaming logic for both Regular and PD routers,
//! eliminating ~600 lines of duplication.

use axum::response::Response;
use axum::{body::Body, http::StatusCode};
use bytes::Bytes;
use http::header::{HeaderValue, CONTENT_TYPE};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io;
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedSender;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;
use tonic::codec::Streaming;
use tracing::{debug, error, warn};

use super::context;
use super::utils;
use crate::grpc_client::proto;
use crate::protocols::spec::*;
use crate::reasoning_parser::ReasoningParser;
use crate::tokenizer::stop::{SequenceDecoderOutput, StopSequenceDecoder};
use crate::tokenizer::traits::Tokenizer;
use crate::tool_parser::ToolParser;
use proto::generate_complete::MatchedStop::{MatchedStopStr, MatchedTokenId};
use proto::generate_response::Response::{Chunk, Complete, Error};
use std::time::Instant;
use tokio::sync::mpsc;

/// Shared streaming processor for both single and dual dispatch modes
#[derive(Clone)]
pub struct StreamingProcessor {
    tokenizer: Arc<dyn Tokenizer>,
    tool_parser_factory: crate::tool_parser::ParserFactory,
    reasoning_parser_factory: crate::reasoning_parser::ParserFactory,
    configured_tool_parser: Option<String>,
    configured_reasoning_parser: Option<String>,
}

impl StreamingProcessor {
    pub fn new(
        tokenizer: Arc<dyn Tokenizer>,
        tool_parser_factory: crate::tool_parser::ParserFactory,
        reasoning_parser_factory: crate::reasoning_parser::ParserFactory,
        configured_tool_parser: Option<String>,
        configured_reasoning_parser: Option<String>,
    ) -> Self {
        Self {
            tokenizer,
            tool_parser_factory,
            reasoning_parser_factory,
            configured_tool_parser,
            configured_reasoning_parser,
        }
    }

    /// Process streaming chat response and return SSE response
    ///
    /// This is the high-level entry point for streaming responses, handling:
    /// - Channel creation
    /// - Background task spawning
    /// - SSE response building
    pub fn process_streaming_response(
        self: Arc<Self>,
        execution_result: context::ExecutionResult,
        chat_request: Arc<ChatCompletionRequest>,
        dispatch: context::DispatchMetadata,
    ) -> Response {
        use bytes::Bytes;
        use tokio::sync::mpsc;

        let stop_params = (
            chat_request.stop.clone(),
            chat_request.stop_token_ids.clone(),
            chat_request.skip_special_tokens,
            chat_request.no_stop_trim,
        );

        // Create SSE channel
        let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, io::Error>>();

        // Spawn background task based on execution mode
        match execution_result {
            context::ExecutionResult::Single { stream } => {
                let processor = self.clone();
                let dispatch_clone = dispatch.clone();
                tokio::spawn(async move {
                    let result = processor
                        .process_streaming_chunks(
                            stream,
                            dispatch_clone,
                            stop_params,
                            chat_request,
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

                    let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
                });
            }
            context::ExecutionResult::Dual { prefill, decode } => {
                let processor = self.clone();
                tokio::spawn(async move {
                    let result = processor
                        .process_dual_streaming_chunks(
                            prefill,
                            *decode,
                            dispatch,
                            stop_params,
                            chat_request,
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

                    let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
                });
            }
        }

        // Return SSE response
        build_sse_response(rx)
    }

    /// Process streaming chunks from a single stream (Regular mode)
    pub async fn process_streaming_chunks(
        &self,
        mut grpc_stream: Streaming<proto::GenerateResponse>,
        dispatch: context::DispatchMetadata,
        stop_params: (Option<StringOrArray>, Option<Vec<u32>>, bool, bool),
        original_request: Arc<ChatCompletionRequest>,
        tx: &UnboundedSender<Result<Bytes, io::Error>>,
    ) -> Result<(), String> {
        // Extract request parameters
        let separate_reasoning = original_request.separate_reasoning;
        let tool_choice = &original_request.tool_choice;
        let tools = &original_request.tools;
        let history_tool_calls_count = utils::get_history_tool_calls_count(&original_request);
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
        type PooledReasoningParser = Arc<tokio::sync::Mutex<Box<dyn ReasoningParser>>>;
        let mut reasoning_parsers: HashMap<u32, PooledReasoningParser> = HashMap::new();

        type PooledToolParser = Arc<tokio::sync::Mutex<Box<dyn ToolParser>>>;
        let mut tool_parsers: HashMap<u32, PooledToolParser> = HashMap::new();
        let mut has_tool_calls: HashMap<u32, bool> = HashMap::new();

        // Per-index stop decoders (each index needs its own state for n>1 support)
        let mut stop_decoders: HashMap<u32, StopSequenceDecoder> = HashMap::new();

        // Reusable SSE formatting buffer to avoid allocations per chunk
        let mut sse_buffer = Vec::with_capacity(512);

        // Use dispatch metadata for consistent response fields
        let request_id = &dispatch.request_id;
        let model = &dispatch.model;
        let created = dispatch.created;
        let system_fingerprint = dispatch.weight_version.as_deref();

        // Check parser availability once upfront (log warning only once per request)
        let reasoning_parser_available = if separate_reasoning {
            if let Some(parser_name) = self.configured_reasoning_parser.as_ref() {
                self.reasoning_parser_factory
                    .registry()
                    .has_parser(parser_name)
            } else {
                self.reasoning_parser_factory
                    .registry()
                    .has_parser_for_model(model)
            }
        } else {
            false
        };

        let tool_parser_available = if tools.is_some() {
            if let Some(parser_name) = self.configured_tool_parser.as_ref() {
                self.tool_parser_factory.registry().has_parser(parser_name)
            } else {
                self.tool_parser_factory
                    .registry()
                    .has_parser_for_model(model)
            }
        } else {
            false
        };

        if separate_reasoning && !reasoning_parser_available {
            warn!(
                "No reasoning parser found for model '{}', skipping reasoning parsing",
                model
            );
        }

        if tools.is_some() && !tool_parser_available {
            warn!(
                "No tool parser found for model '{}', skipping tool call parsing",
                model
            );
        }

        // Phase 2: Main streaming loop
        while let Some(response) = grpc_stream.next().await {
            let gen_response = response.map_err(|e| format!("Stream error: {}", e))?;

            match gen_response.response {
                Some(Chunk(chunk)) => {
                    let index = chunk.index;

                    // Get or create stop decoder for this index
                    let stop_decoder = stop_decoders.entry(index).or_insert_with(|| {
                        let (ref stop, ref stop_token_ids, skip_special_tokens, no_stop_trim) =
                            stop_params;
                        utils::create_stop_decoder(
                            &self.tokenizer,
                            stop.as_ref(),
                            stop_token_ids.as_ref(),
                            skip_special_tokens,
                            no_stop_trim,
                        )
                    });

                    // Process tokens through stop decoder
                    let (chunk_text, _should_stop) =
                        Self::process_chunk_tokens(stop_decoder, &chunk.token_ids);

                    if chunk_text.is_empty() {
                        continue;
                    }

                    // Process logprobs if present
                    let choice_logprobs = if let Some(ref proto_logprobs) = chunk.output_logprobs {
                        match utils::convert_proto_to_openai_logprobs(
                            proto_logprobs,
                            &self.tokenizer,
                        ) {
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
                            system_fingerprint: system_fingerprint.map(|s| s.to_string()),
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
                        Self::format_sse_chunk_into(&mut sse_buffer, &first_chunk);
                        tx.send(Ok(Bytes::from(sse_buffer.clone())))
                            .map_err(|_| "Failed to send first chunk".to_string())?;
                        is_firsts.insert(index, false);
                    }

                    // Calculate delta
                    let mut delta = chunk_text;
                    stream_buffer.push_str(&delta);

                    // Reasoning content handling
                    let in_reasoning = if separate_reasoning && reasoning_parser_available {
                        let (normal_text, reasoning_chunk, in_reasoning) = self
                            .process_reasoning_stream(
                                &delta,
                                index,
                                &mut reasoning_parsers,
                                request_id,
                                model,
                                created,
                                system_fingerprint,
                            )
                            .await;
                        if let Some(chunk) = reasoning_chunk {
                            Self::format_sse_chunk_into(&mut sse_buffer, &chunk);
                            tx.send(Ok(Bytes::from(sse_buffer.clone())))
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

                    if !in_reasoning
                        && tool_choice_enabled
                        && tools.is_some()
                        && tool_parser_available
                    {
                        let tool_chunks = self
                            .process_tool_calls_stream(
                                &delta,
                                index,
                                &mut tool_parsers,
                                &mut has_tool_calls,
                                tools.as_ref().unwrap(),
                                request_id,
                                model,
                                created,
                                system_fingerprint,
                                history_tool_calls_count,
                            )
                            .await;

                        for chunk in tool_chunks {
                            Self::format_sse_chunk_into(&mut sse_buffer, &chunk);
                            tx.send(Ok(Bytes::from(sse_buffer.clone())))
                                .map_err(|_| "Failed to send tool call chunk".to_string())?;
                        }

                        // Always skip regular content when tool parsing is active
                        // Parser either emitted chunks or buffered content
                        continue;
                    }

                    // Regular content emission
                    if !delta.is_empty() {
                        let content_chunk = Self::create_content_chunk(
                            delta,
                            index,
                            request_id,
                            model,
                            created,
                            system_fingerprint,
                            choice_logprobs,
                        );
                        Self::format_sse_chunk_into(&mut sse_buffer, &content_chunk);
                        tx.send(Ok(Bytes::from(sse_buffer.clone())))
                            .map_err(|_| "Failed to send content chunk".to_string())?;
                    }
                }
                Some(Complete(complete)) => {
                    let index = complete.index;

                    // Flush any remaining text for this index's stop_decoder
                    if let Some(decoder) = stop_decoders.get_mut(&index) {
                        if let SequenceDecoderOutput::Text(text) = decoder.flush() {
                            if !text.is_empty() {
                                let stream_buffer = stream_buffers.entry(index).or_default();
                                stream_buffer.push_str(&text);

                                let content_chunk = ChatCompletionStreamResponse {
                                    id: request_id.clone(),
                                    object: "chat.completion.chunk".to_string(),
                                    created,
                                    model: model.clone(),
                                    system_fingerprint: system_fingerprint.map(|s| s.to_string()),
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

                                let sse_chunk =
                                    serde_json::to_string(&content_chunk).map_err(|e| {
                                        format!("Failed to serialize content chunk: {}", e)
                                    })?;
                                tx.send(Ok(Bytes::from(format!("data: {}\n\n", sse_chunk))))
                                    .map_err(|_| "Failed to send flushed content".to_string())?;
                            }
                        }
                    }

                    // Store metadata
                    prompt_tokens.insert(index, complete.prompt_tokens as u32);
                    completion_tokens.insert(index, complete.completion_tokens as u32);
                    cached_tokens.insert(index, complete.cached_tokens as u32);
                    finish_reasons.insert(index, complete.finish_reason.clone());

                    // Extract matched_stop
                    let matched_stop_value = match &complete.matched_stop {
                        Some(MatchedTokenId(token_id)) => {
                            Some(Value::Number(serde_json::Number::from(*token_id)))
                        }
                        Some(MatchedStopStr(stop_str)) => Some(Value::String(stop_str.clone())),
                        None => None,
                    };
                    matched_stops.insert(index, matched_stop_value);

                    // Don't break - continue reading all Complete messages for n>1
                }
                Some(Error(error)) => {
                    return Err(error.message);
                }
                None => continue,
            }
        }

        // Phase 3: Check unstreamed tool args
        for (index, parser) in &tool_parsers {
            let parser_guard = parser.lock().await;
            if let Some(unstreamed_items) = parser_guard.get_unstreamed_tool_args() {
                for tool_call_item in unstreamed_items {
                    let tool_call_delta = ToolCallDelta {
                        index: tool_call_item.tool_index as u32,
                        id: None,
                        tool_type: None,
                        function: Some(FunctionCallDelta {
                            name: None,
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
                        system_fingerprint: system_fingerprint.map(|s| s.to_string()),
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
                system_fingerprint: system_fingerprint.map(|s| s.to_string()),
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
                    system_fingerprint: system_fingerprint.map(|s| s.to_string()),
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

    /// Process dual streaming chunks (prefill + decode) - PD mode
    pub async fn process_dual_streaming_chunks(
        &self,
        mut prefill_stream: Streaming<proto::GenerateResponse>,
        decode_stream: Streaming<proto::GenerateResponse>,
        dispatch: context::DispatchMetadata,
        stop_params: (Option<StringOrArray>, Option<Vec<u32>>, bool, bool),
        original_request: Arc<ChatCompletionRequest>,
        tx: &UnboundedSender<Result<Bytes, io::Error>>,
    ) -> Result<(), String> {
        // Phase 1.5: Collect input_logprobs from prefill stream if requested
        if original_request.logprobs {
            while let Some(response) = prefill_stream.next().await {
                let gen_response = response.map_err(|e| format!("Prefill stream error: {}", e))?;
                match gen_response.response {
                    Some(Complete(_complete)) => {
                        // Input logprobs collected but not yet used in streaming
                        // (OpenAI spec doesn't require prompt logprobs in streaming responses)
                        break;
                    }
                    Some(Error(error)) => {
                        return Err(format!("Prefill error: {}", error.message));
                    }
                    _ => continue,
                }
            }
        }

        // Phase 2-5: Process decode stream (same as single mode)
        self.process_streaming_chunks(decode_stream, dispatch, stop_params, original_request, tx)
            .await
    }

    /// Process streaming generate response and return SSE response
    ///
    /// Simpler than chat - no tool/reasoning parsing, just text accumulation
    pub fn process_streaming_generate(
        self: Arc<Self>,
        execution_result: context::ExecutionResult,
        generate_request: Arc<GenerateRequest>,
        dispatch: context::DispatchMetadata,
    ) -> Response {
        let return_logprob = generate_request.return_logprob;

        // Create SSE channel
        let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, io::Error>>();

        // Spawn background task based on execution mode
        match execution_result {
            context::ExecutionResult::Single { stream } => {
                let tokenizer = self.tokenizer.clone();
                let request_id = dispatch.request_id.clone();
                let weight_version = dispatch
                    .weight_version
                    .clone()
                    .unwrap_or_else(|| "default".to_string());
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
                        let _ = tx.send(Ok(Bytes::from(error_chunk)));
                    }

                    let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
                });
            }
            context::ExecutionResult::Dual { prefill, decode } => {
                // For PD mode, need to handle prefill stream for input_logprobs
                let tokenizer = self.tokenizer.clone();
                let request_id = dispatch.request_id.clone();
                let weight_version = dispatch
                    .weight_version
                    .clone()
                    .unwrap_or_else(|| "default".to_string());
                tokio::spawn(async move {
                    let result = Self::process_generate_streaming_dual(
                        tokenizer,
                        prefill,
                        *decode,
                        request_id,
                        weight_version,
                        return_logprob,
                        &tx,
                    )
                    .await;

                    if let Err(e) = result {
                        let error_chunk = format!("data: {{\"error\": \"{}\"}}\n\n", e);
                        let _ = tx.send(Ok(Bytes::from(error_chunk)));
                    }

                    let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
                });
            }
        }

        // Return SSE response
        build_sse_response(rx)
    }

    //TODO add streaming logprob support
    /// Process streaming chunks for generate endpoint (no tool/reasoning parsing)
    async fn process_generate_streaming(
        tokenizer: Arc<dyn Tokenizer>,
        mut stream: Streaming<proto::GenerateResponse>,
        request_id: String,
        weight_version: String,
        _include_logprobs: bool,
        tx: &UnboundedSender<Result<Bytes, io::Error>>,
    ) -> Result<(), String> {
        let start_time = Instant::now();

        // Track state per index for n>1 case
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
                            "weight_version": &weight_version,
                            "completion_tokens": *completion_tokens,
                            "cached_tokens": chunk.cached_tokens
                        },
                        "index": index
                    });

                    let sse_chunk = format!(
                        "data: {}\n\n",
                        serde_json::to_string(&chunk_response).unwrap()
                    );
                    tx.send(Ok(Bytes::from(sse_chunk)))
                        .map_err(|_| "Failed to send chunk".to_string())?;
                }
                Some(Complete(complete)) => {
                    let index = complete.index;
                    let accumulated_text =
                        accumulated_texts.get(&index).cloned().unwrap_or_default();
                    let completion_tokens = *completion_tokens_map.get(&index).unwrap_or(&0);
                    let index_id = format!("{}-{}", request_id, index);
                    let e2e_latency = start_time.elapsed().as_secs_f64();

                    // Send final chunk with finish_reason
                    let finish_response = serde_json::json!({
                        "text": accumulated_text,
                        "output_ids": complete.output_ids[complete.output_ids.len().saturating_sub(1)..].to_vec(),
                        "meta_info": {
                            "id": index_id,
                            "finish_reason": complete.finish_reason,
                            "prompt_tokens": complete.prompt_tokens,
                            "weight_version": &weight_version,
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
                    tx.send(Ok(Bytes::from(sse_chunk)))
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

    /// Process dual streaming for generate endpoint (PD mode with logprobs support)
    async fn process_generate_streaming_dual(
        tokenizer: Arc<dyn Tokenizer>,
        mut prefill_stream: Streaming<proto::GenerateResponse>,
        decode_stream: Streaming<proto::GenerateResponse>,
        request_id: String,
        weight_version: String,
        return_logprob: bool,
        tx: &UnboundedSender<Result<Bytes, io::Error>>,
    ) -> Result<(), String> {
        // Collect input_logprobs from prefill stream if requested
        let input_token_logprobs = if return_logprob {
            let mut input_logprobs = None;
            while let Some(response) = prefill_stream.next().await {
                let gen_response = response.map_err(|e| format!("Prefill stream error: {}", e))?;
                match gen_response.response {
                    Some(Complete(complete)) => {
                        // Extract input_logprobs from prefill Complete message (convert proto to SGLang format)
                        input_logprobs = complete
                            .input_logprobs
                            .as_ref()
                            .map(utils::convert_generate_input_logprobs);
                        break;
                    }
                    Some(Error(error)) => {
                        return Err(format!("Prefill error: {}", error.message));
                    }
                    _ => continue,
                }
            }
            input_logprobs
        } else {
            None
        };

        // Process decode stream with input_logprobs prepended
        Self::process_generate_streaming_with_input_logprobs(
            tokenizer,
            decode_stream,
            request_id,
            weight_version,
            return_logprob,
            input_token_logprobs,
            tx,
        )
        .await
    }

    /// Process generate streaming with optional input_logprobs
    async fn process_generate_streaming_with_input_logprobs(
        tokenizer: Arc<dyn Tokenizer>,
        mut stream: Streaming<proto::GenerateResponse>,
        request_id: String,
        weight_version: String,
        _include_logprobs: bool,
        input_token_logprobs: Option<Vec<Vec<Option<f64>>>>,
        tx: &UnboundedSender<Result<Bytes, io::Error>>,
    ) -> Result<(), String> {
        let start_time = Instant::now();

        // Track state per index for n>1 case
        let mut accumulated_texts: HashMap<u32, String> = HashMap::new();
        let mut accumulated_output_logprobs: HashMap<u32, Option<Vec<Vec<Option<f64>>>>> =
            HashMap::new();
        let mut completion_tokens_map: HashMap<u32, u32> = HashMap::new();

        while let Some(response) = stream.next().await {
            let gen_response = response.map_err(|e| format!("Stream error: {}", e))?;

            match gen_response.response {
                Some(Chunk(chunk)) => {
                    let index = chunk.index;

                    // Update completion tokens for this index
                    let completion_tokens = completion_tokens_map.entry(index).or_insert(0);
                    *completion_tokens += chunk.token_ids.len() as u32;

                    // Decode tokens to text
                    let chunk_text = tokenizer.decode(&chunk.token_ids, true).unwrap_or_default();

                    // Accumulate text for this index
                    let accumulated_text = accumulated_texts.entry(index).or_default();
                    accumulated_text.push_str(&chunk_text);

                    // Store latest output logprobs (cumulative from proto, convert to SGLang format)
                    if let Some(ref output_logprobs) = chunk.output_logprobs {
                        let converted = utils::convert_generate_output_logprobs(output_logprobs);
                        accumulated_output_logprobs.insert(index, Some(converted));
                    }

                    // Generate unique ID per index
                    let index_id = format!("{}-{}", request_id, index);

                    // Build streaming response chunk with cumulative logprobs
                    let current_output_logprobs = accumulated_output_logprobs
                        .get(&index)
                        .and_then(|o| o.as_ref());

                    let chunk_response = serde_json::json!({
                        "text": accumulated_text.clone(),
                        "output_ids": chunk.token_ids,
                        "meta_info": {
                            "id": index_id,
                            "finish_reason": null,
                            "prompt_tokens": chunk.prompt_tokens,
                            "weight_version": &weight_version,
                            "input_token_logprobs": input_token_logprobs.as_ref(),
                            "output_token_logprobs": current_output_logprobs,
                            "completion_tokens": *completion_tokens,
                            "cached_tokens": chunk.cached_tokens
                        },
                        "index": index
                    });

                    let sse_chunk = format!(
                        "data: {}\n\n",
                        serde_json::to_string(&chunk_response).unwrap()
                    );
                    tx.send(Ok(Bytes::from(sse_chunk)))
                        .map_err(|_| "Failed to send chunk".to_string())?;
                }
                Some(Complete(complete)) => {
                    let index = complete.index;
                    let accumulated_text =
                        accumulated_texts.get(&index).cloned().unwrap_or_default();
                    let completion_tokens = *completion_tokens_map.get(&index).unwrap_or(&0);
                    let final_output_logprobs = accumulated_output_logprobs
                        .get(&index)
                        .and_then(|o| o.as_ref());
                    let index_id = format!("{}-{}", request_id, index);
                    let e2e_latency = start_time.elapsed().as_secs_f64();

                    // Parse finish_reason
                    let finish_reason = utils::parse_finish_reason(
                        &complete.finish_reason,
                        complete.completion_tokens,
                    );

                    // Send final chunk with finish_reason
                    let finish_response = serde_json::json!({
                        "text": accumulated_text,
                        "output_ids": complete.output_ids[complete.output_ids.len().saturating_sub(1)..].to_vec(),
                        "meta_info": {
                            "id": index_id,
                            "finish_reason": finish_reason,
                            "prompt_tokens": complete.prompt_tokens,
                            "weight_version": &weight_version,
                            "input_token_logprobs": input_token_logprobs.as_ref(),
                            "output_token_logprobs": final_output_logprobs,
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
                    tx.send(Ok(Bytes::from(sse_chunk)))
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

    // ========================================================================
    // Helper Methods
    // ========================================================================

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
                    return (chunk_text, true);
                }
                SequenceDecoderOutput::Stopped => {
                    return (chunk_text, true);
                }
                SequenceDecoderOutput::Held => {}
            }
        }
        (chunk_text, false)
    }

    /// Helper: Process reasoning content in streaming mode
    #[allow(clippy::too_many_arguments)]
    async fn process_reasoning_stream(
        &self,
        delta: &str,
        index: u32,
        reasoning_parsers: &mut HashMap<u32, Arc<tokio::sync::Mutex<Box<dyn ReasoningParser>>>>,
        request_id: &str,
        model: &str,
        created: u64,
        system_fingerprint: Option<&str>,
    ) -> (String, Option<ChatCompletionStreamResponse>, bool) {
        // Create fresh parser for this index (not pooled, to avoid state pollution)
        reasoning_parsers.entry(index).or_insert_with(|| {
            let parser = utils::create_reasoning_parser(
                &self.reasoning_parser_factory,
                self.configured_reasoning_parser.as_ref(),
                model,
            )
            .expect("Parser should be available - checked upfront");
            Arc::new(tokio::sync::Mutex::new(parser))
        });

        if let Some(pooled_parser) = reasoning_parsers.get(&index) {
            let (parse_result, in_reasoning) = {
                let mut parser = pooled_parser.lock().await;
                let result = parser.parse_reasoning_streaming_incremental(delta);
                let in_reasoning = parser.is_in_reasoning();
                (result, in_reasoning)
            };

            match parse_result {
                Ok(crate::reasoning_parser::ParserResult {
                    reasoning_text,
                    normal_text,
                }) => {
                    let chunk = if !reasoning_text.is_empty() {
                        Some(ChatCompletionStreamResponse {
                            id: request_id.to_string(),
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model: model.to_string(),
                            system_fingerprint: system_fingerprint.map(|s| s.to_string()),
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
    #[allow(clippy::too_many_arguments)]
    async fn process_tool_calls_stream(
        &self,
        delta: &str,
        index: u32,
        tool_parsers: &mut HashMap<u32, Arc<tokio::sync::Mutex<Box<dyn ToolParser>>>>,
        has_tool_calls: &mut HashMap<u32, bool>,
        tools: &[Tool],
        request_id: &str,
        model: &str,
        created: u64,
        system_fingerprint: Option<&str>,
        history_tool_calls_count: usize,
    ) -> Vec<ChatCompletionStreamResponse> {
        let mut chunks = Vec::new();

        // Create fresh parser for this index (not pooled, to avoid state pollution)
        tool_parsers.entry(index).or_insert_with(|| {
            let parser = utils::create_tool_parser(
                &self.tool_parser_factory,
                self.configured_tool_parser.as_ref(),
                model,
            )
            .expect("Parser should be available - checked upfront");
            Arc::new(tokio::sync::Mutex::new(parser))
        });

        if let Some(pooled_parser) = tool_parsers.get(&index) {
            let mut parser = pooled_parser.lock().await;

            match parser.parse_incremental(delta, tools).await {
                Ok(crate::tool_parser::StreamingParseResult { normal_text, calls }) => {
                    // Emit normal text if present
                    if !normal_text.is_empty() {
                        chunks.push(ChatCompletionStreamResponse {
                            id: request_id.to_string(),
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model: model.to_string(),
                            system_fingerprint: system_fingerprint.map(|s| s.to_string()),
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
                            Some(utils::generate_tool_call_id(
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
                            system_fingerprint: system_fingerprint.map(|s| s.to_string()),
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

                    return chunks;
                }
                Err(e) => {
                    error!("Tool call parsing error: {}", e);
                }
            }
        }

        chunks
    }

    /// Format a response as SSE chunk into a reusable buffer
    /// This avoids allocations by reusing the same buffer across multiple chunks
    #[inline]
    fn format_sse_chunk_into(buffer: &mut Vec<u8>, chunk: &ChatCompletionStreamResponse) {
        buffer.clear();
        buffer.extend_from_slice(b"data: ");
        if let Err(e) = serde_json::to_writer(&mut *buffer, chunk) {
            error!("Failed to serialize SSE chunk: {}", e);
            buffer.clear();
            buffer.extend_from_slice(b"data: ");
            let error_msg = json!({"error": "serialization_failed"}).to_string();
            buffer.extend_from_slice(error_msg.as_bytes());
        }
        buffer.extend_from_slice(b"\n\n");
    }

    /// Create a content chunk response
    fn create_content_chunk(
        content: String,
        index: u32,
        request_id: &str,
        model: &str,
        created: u64,
        system_fingerprint: Option<&str>,
        logprobs: Option<ChatLogProbs>,
    ) -> ChatCompletionStreamResponse {
        ChatCompletionStreamResponse {
            id: request_id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.to_string(),
            system_fingerprint: system_fingerprint.map(|s| s.to_string()),
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
}

/// Build SSE response with proper headers
pub fn build_sse_response(rx: mpsc::UnboundedReceiver<Result<Bytes, io::Error>>) -> Response {
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
