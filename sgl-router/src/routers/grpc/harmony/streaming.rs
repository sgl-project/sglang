//! Harmony streaming response processor

use std::{
    collections::{hash_map::Entry::Vacant, HashMap},
    io,
    sync::Arc,
};

use axum::{body::Body, http::StatusCode, response::Response};
use bytes::Bytes;
use http::header::{HeaderValue, CONTENT_TYPE};
use proto::{
    generate_complete::MatchedStop::{MatchedStopStr, MatchedTokenId},
    generate_response::Response::{Chunk, Complete},
};
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::{wrappers::UnboundedReceiverStream, StreamExt};
use tracing::error;

use super::{types::HarmonyChannelDelta, HarmonyParserAdapter};
use crate::{
    grpc_client::{proto, sglang_scheduler::AbortOnDropStream},
    protocols::{
        chat::{
            ChatCompletionRequest, ChatCompletionStreamResponse, ChatMessageDelta, ChatStreamChoice,
        },
        common::{FunctionCallDelta, ToolCallDelta, Usage},
    },
    routers::grpc::context,
};

/// Processor for streaming Harmony responses
///
/// Returns an SSE stream that parses Harmony tokens incrementally and
/// emits ChatCompletionChunk events for streaming responses.
pub struct HarmonyStreamingProcessor;

impl HarmonyStreamingProcessor {
    /// Create a new Harmony streaming processor
    pub fn new() -> Self {
        Self
    }

    /// Process a streaming Harmony Chat Completion response
    ///
    /// Returns an SSE response with streaming token updates.
    pub fn process_streaming_chat_response(
        self: Arc<Self>,
        execution_result: context::ExecutionResult,
        chat_request: Arc<ChatCompletionRequest>,
        dispatch: context::DispatchMetadata,
    ) -> Response {
        // Create SSE channel
        let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, io::Error>>();

        // Spawn background task based on execution mode
        match execution_result {
            context::ExecutionResult::Single { stream } => {
                tokio::spawn(async move {
                    let result =
                        Self::process_single_stream(stream, dispatch, chat_request, &tx).await;

                    if let Err(e) = result {
                        error!("Harmony streaming error: {}", e);
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
                tokio::spawn(async move {
                    let result =
                        Self::process_dual_stream(prefill, *decode, dispatch, chat_request, &tx)
                            .await;

                    if let Err(e) = result {
                        error!("Harmony dual streaming error: {}", e);
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
        Self::build_sse_response(rx)
    }

    /// Process streaming chunks from a single stream
    async fn process_single_stream(
        mut grpc_stream: AbortOnDropStream,
        dispatch: context::DispatchMetadata,
        original_request: Arc<ChatCompletionRequest>,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    ) -> Result<(), String> {
        // Per-index state management (for n>1 support)
        let mut parsers: HashMap<u32, HarmonyParserAdapter> = HashMap::new();
        let mut is_firsts: HashMap<u32, bool> = HashMap::new();
        let mut finish_reasons: HashMap<u32, Option<String>> = HashMap::new();
        let mut matched_stops: HashMap<u32, Option<serde_json::Value>> = HashMap::new();
        let mut prompt_tokens: HashMap<u32, u32> = HashMap::new();
        let mut completion_tokens: HashMap<u32, u32> = HashMap::new();

        let stream_options = &original_request.stream_options;

        // Process stream
        while let Some(result) = grpc_stream.next().await {
            let response = result.map_err(|e| format!("Stream error: {}", e))?;

            match response.response {
                Some(Chunk(chunk)) => {
                    let index = chunk.index;

                    // Initialize parser for this index if needed
                    if let Vacant(e) = parsers.entry(index) {
                        e.insert(
                            HarmonyParserAdapter::new()
                                .map_err(|e| format!("Failed to create parser: {}", e))?,
                        );
                        is_firsts.insert(index, true);
                    }

                    // Track token counts
                    *completion_tokens.entry(index).or_insert(0) += 1;

                    // Parse chunk via Harmony parser
                    let parser = parsers
                        .get_mut(&index)
                        .ok_or("Parser not found for index")?;

                    let delta_result = parser
                        .parse_chunk(&chunk.token_ids)
                        .map_err(|e| format!("Parse error: {}", e))?;

                    // Emit SSE event if there's a delta
                    if let Some(delta) = delta_result {
                        let is_first = is_firsts.get(&index).copied().unwrap_or(false);
                        Self::emit_chunk_delta(
                            &delta,
                            index,
                            is_first,
                            &dispatch,
                            &original_request,
                            tx,
                        )?;

                        if is_first {
                            is_firsts.insert(index, false);
                        }
                    }
                }
                Some(Complete(complete)) => {
                    let index = complete.index;

                    // Store final metadata
                    finish_reasons.insert(index, Some(complete.finish_reason.clone()));
                    matched_stops.insert(
                        index,
                        complete.matched_stop.as_ref().map(|m| match m {
                            MatchedTokenId(id) => {
                                serde_json::json!(id)
                            }
                            MatchedStopStr(s) => {
                                serde_json::json!(s)
                            }
                        }),
                    );
                    prompt_tokens.insert(index, complete.prompt_tokens as u32);
                    *completion_tokens.entry(index).or_insert(0) =
                        complete.completion_tokens as u32;

                    // Finalize parser and emit final chunk
                    if let Some(parser) = parsers.get_mut(&index) {
                        let matched_stop = matched_stops.get(&index).and_then(|m| m.clone());
                        let final_output = parser
                            .finalize(complete.finish_reason.clone(), matched_stop.clone())
                            .map_err(|e| format!("Finalize error: {}", e))?;

                        Self::emit_final_chunk(
                            index,
                            &final_output.finish_reason,
                            final_output.matched_stop.as_ref(),
                            &dispatch,
                            &original_request,
                            tx,
                        )?;
                    }
                }
                Some(proto::generate_response::Response::Error(err)) => {
                    return Err(format!("Server error: {}", err.message));
                }
                None => {}
            }
        }

        // Emit final usage if requested
        if let Some(true) = stream_options.as_ref().and_then(|so| so.include_usage) {
            let total_prompt: u32 = prompt_tokens.values().sum();
            let total_completion: u32 = completion_tokens.values().sum();

            Self::emit_usage_chunk(
                total_prompt,
                total_completion,
                &dispatch,
                &original_request,
                tx,
            )?;
        }

        // Mark stream as completed successfully to prevent abort on drop
        grpc_stream.mark_completed();

        Ok(())
    }

    /// Process streaming chunks from dual streams (prefill + decode)
    async fn process_dual_stream(
        mut prefill_stream: AbortOnDropStream,
        mut decode_stream: AbortOnDropStream,
        dispatch: context::DispatchMetadata,
        original_request: Arc<ChatCompletionRequest>,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    ) -> Result<(), String> {
        // Phase 1: Process prefill stream (collect metadata)
        let mut prompt_tokens: HashMap<u32, u32> = HashMap::new();

        while let Some(result) = prefill_stream.next().await {
            let response = result.map_err(|e| format!("Prefill stream error: {}", e))?;

            if let Some(Complete(complete)) = response.response {
                prompt_tokens.insert(complete.index, complete.prompt_tokens as u32);
            }
        }

        // Phase 2: Process decode stream (same as single stream)
        let mut parsers: HashMap<u32, HarmonyParserAdapter> = HashMap::new();
        let mut is_firsts: HashMap<u32, bool> = HashMap::new();
        let mut finish_reasons: HashMap<u32, Option<String>> = HashMap::new();
        let mut matched_stops: HashMap<u32, Option<serde_json::Value>> = HashMap::new();
        let mut completion_tokens: HashMap<u32, u32> = HashMap::new();

        let stream_options = &original_request.stream_options;

        while let Some(result) = decode_stream.next().await {
            let response = result.map_err(|e| format!("Decode stream error: {}", e))?;

            match response.response {
                Some(Chunk(chunk)) => {
                    let index = chunk.index;

                    // Initialize parser for this index if needed
                    if let Vacant(e) = parsers.entry(index) {
                        e.insert(
                            HarmonyParserAdapter::new()
                                .map_err(|e| format!("Failed to create parser: {}", e))?,
                        );
                        is_firsts.insert(index, true);
                    }

                    *completion_tokens.entry(index).or_insert(0) += 1;

                    let parser = parsers
                        .get_mut(&index)
                        .ok_or("Parser not found for index")?;

                    let delta_result = parser
                        .parse_chunk(&chunk.token_ids)
                        .map_err(|e| format!("Parse error: {}", e))?;

                    if let Some(delta) = delta_result {
                        let is_first = is_firsts.get(&index).copied().unwrap_or(false);
                        Self::emit_chunk_delta(
                            &delta,
                            index,
                            is_first,
                            &dispatch,
                            &original_request,
                            tx,
                        )?;

                        if is_first {
                            is_firsts.insert(index, false);
                        }
                    }
                }
                Some(Complete(complete)) => {
                    let index = complete.index;

                    finish_reasons.insert(index, Some(complete.finish_reason.clone()));
                    matched_stops.insert(
                        index,
                        complete.matched_stop.as_ref().map(|m| match m {
                            MatchedTokenId(id) => {
                                json!(id)
                            }
                            MatchedStopStr(s) => {
                                json!(s)
                            }
                        }),
                    );
                    *completion_tokens.entry(index).or_insert(0) =
                        complete.completion_tokens as u32;

                    if let Some(parser) = parsers.get_mut(&index) {
                        let matched_stop = matched_stops.get(&index).and_then(|m| m.clone());
                        let final_output = parser
                            .finalize(complete.finish_reason.clone(), matched_stop.clone())
                            .map_err(|e| format!("Finalize error: {}", e))?;

                        Self::emit_final_chunk(
                            index,
                            &final_output.finish_reason,
                            final_output.matched_stop.as_ref(),
                            &dispatch,
                            &original_request,
                            tx,
                        )?;
                    }
                }
                Some(proto::generate_response::Response::Error(err)) => {
                    return Err(format!("Server error: {}", err.message));
                }
                None => {}
            }
        }

        decode_stream.mark_completed();

        // Mark prefill stream as completed AFTER decode completes successfully
        // This ensures that if client disconnects during decode, BOTH streams send abort
        prefill_stream.mark_completed();

        // Emit final usage if requested
        if let Some(true) = stream_options.as_ref().and_then(|so| so.include_usage) {
            let total_prompt: u32 = prompt_tokens.values().sum();
            let total_completion: u32 = completion_tokens.values().sum();

            Self::emit_usage_chunk(
                total_prompt,
                total_completion,
                &dispatch,
                &original_request,
                tx,
            )?;
        }

        Ok(())
    }

    /// Emit a chunk delta from Harmony channels
    fn emit_chunk_delta(
        delta: &HarmonyChannelDelta,
        index: u32,
        is_first: bool,
        dispatch: &context::DispatchMetadata,
        original_request: &ChatCompletionRequest,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    ) -> Result<(), String> {
        // On first chunk, emit role announcement separately
        if is_first {
            let role_chunk = ChatCompletionStreamResponse {
                id: dispatch.request_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created: dispatch.created,
                model: original_request.model.clone(),
                system_fingerprint: dispatch.weight_version.clone(),
                choices: vec![ChatStreamChoice {
                    index,
                    delta: ChatMessageDelta {
                        role: Some("assistant".to_string()),
                        content: Some(String::new()),
                        tool_calls: None,
                        reasoning_content: None,
                    },
                    logprobs: None,
                    finish_reason: None,
                    matched_stop: None,
                }],
                usage: None,
            };

            let chunk_json = serde_json::to_string(&role_chunk)
                .map_err(|e| format!("JSON serialization error: {}", e))?;
            let sse_data = format!("data: {}\n\n", chunk_json);

            tx.send(Ok(Bytes::from(sse_data)))
                .map_err(|_| "Failed to send role chunk".to_string())?;
        }

        // Emit content delta (role is always None for content chunks)
        let chat_delta = ChatMessageDelta {
            role: None,
            content: delta.final_delta.clone(),
            tool_calls: delta.commentary_delta.as_ref().map(|tc_delta| {
                vec![ToolCallDelta {
                    index: tc_delta.index as u32,
                    id: tc_delta.id.clone(),
                    tool_type: tc_delta.id.as_ref().map(|_| "function".to_string()),
                    function: tc_delta.function.as_ref().map(|f| FunctionCallDelta {
                        name: f.name.clone(),
                        arguments: f.arguments.clone(),
                    }),
                }]
            }),
            reasoning_content: delta.analysis_delta.clone(),
        };

        // Build and emit chunk
        let chunk = ChatCompletionStreamResponse {
            id: dispatch.request_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: dispatch.created,
            model: original_request.model.clone(),
            system_fingerprint: dispatch.weight_version.clone(),
            choices: vec![ChatStreamChoice {
                index,
                delta: chat_delta,
                logprobs: None,
                finish_reason: None,
                matched_stop: None,
            }],
            usage: None,
        };

        let chunk_json = serde_json::to_string(&chunk)
            .map_err(|e| format!("JSON serialization error: {}", e))?;
        let sse_data = format!("data: {}\n\n", chunk_json);

        tx.send(Ok(Bytes::from(sse_data)))
            .map_err(|_| "Failed to send chunk".to_string())?;

        Ok(())
    }

    /// Emit final chunk with finish_reason
    fn emit_final_chunk(
        index: u32,
        finish_reason: &str,
        matched_stop: Option<&serde_json::Value>,
        dispatch: &context::DispatchMetadata,
        original_request: &ChatCompletionRequest,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    ) -> Result<(), String> {
        let chunk = ChatCompletionStreamResponse {
            id: dispatch.request_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: dispatch.created,
            model: original_request.model.clone(),
            system_fingerprint: dispatch.weight_version.clone(),
            choices: vec![ChatStreamChoice {
                index,
                delta: ChatMessageDelta {
                    role: None,
                    content: None,
                    tool_calls: None,
                    reasoning_content: None,
                },
                logprobs: None,
                finish_reason: Some(finish_reason.to_string()),
                matched_stop: matched_stop.cloned(),
            }],
            usage: None,
        };

        let chunk_json = serde_json::to_string(&chunk)
            .map_err(|e| format!("JSON serialization error: {}", e))?;
        let sse_data = format!("data: {}\n\n", chunk_json);

        tx.send(Ok(Bytes::from(sse_data)))
            .map_err(|_| "Failed to send final chunk".to_string())?;

        Ok(())
    }

    /// Emit usage chunk at the end
    fn emit_usage_chunk(
        prompt_tokens: u32,
        completion_tokens: u32,
        dispatch: &context::DispatchMetadata,
        original_request: &ChatCompletionRequest,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    ) -> Result<(), String> {
        let usage_chunk = ChatCompletionStreamResponse {
            id: dispatch.request_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: dispatch.created,
            model: original_request.model.clone(),
            system_fingerprint: dispatch.weight_version.clone(),
            choices: vec![],
            usage: Some(Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
                completion_tokens_details: None,
            }),
        };

        let chunk_json = serde_json::to_string(&usage_chunk)
            .map_err(|e| format!("JSON serialization error: {}", e))?;
        let sse_data = format!("data: {}\n\n", chunk_json);

        tx.send(Ok(Bytes::from(sse_data)))
            .map_err(|_| "Failed to send usage chunk".to_string())?;

        Ok(())
    }

    /// Build SSE response from receiver
    fn build_sse_response(rx: mpsc::UnboundedReceiver<Result<Bytes, io::Error>>) -> Response {
        let stream = UnboundedReceiverStream::new(rx);
        let body = Body::from_stream(stream);

        Response::builder()
            .status(StatusCode::OK)
            .header(
                CONTENT_TYPE,
                HeaderValue::from_static("text/event-stream; charset=utf-8"),
            )
            .header("Cache-Control", HeaderValue::from_static("no-cache"))
            .header("Connection", HeaderValue::from_static("keep-alive"))
            .body(body)
            .unwrap()
    }
}

impl Default for HarmonyStreamingProcessor {
    fn default() -> Self {
        Self::new()
    }
}
