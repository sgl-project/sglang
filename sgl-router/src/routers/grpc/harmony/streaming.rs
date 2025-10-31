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
use tracing::{debug, error};

use super::{
    processor::ResponsesIterationResult, types::HarmonyChannelDelta, HarmonyParserAdapter,
};
use crate::{
    grpc_client::{proto, sglang_scheduler::AbortOnDropStream},
    protocols::{
        chat::{
            ChatCompletionRequest, ChatCompletionStreamResponse, ChatMessageDelta, ChatStreamChoice,
        },
        common::{FunctionCallDelta, ToolCall, ToolCallDelta, Usage},
        responses::{ResponseStatus, ResponseUsage, ResponsesResponse, ResponsesUsage},
    },
    routers::grpc::{
        context,
        responses::streaming::{OutputItemType, ResponseStreamEventEmitter},
    },
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

    /// Common decode stream processing logic for both single and dual stream modes
    ///
    /// This helper function contains the shared logic for processing the decode stream,
    /// parsing Harmony tokens, emitting SSE events, and tracking state.
    async fn process_decode_stream_common(
        mut decode_stream: AbortOnDropStream,
        emitter: &mut ResponseStreamEventEmitter,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    ) -> Result<ResponsesIterationResult, String> {
        // Initialize Harmony parser for this iteration
        let mut parser =
            HarmonyParserAdapter::new().map_err(|e| format!("Failed to create parser: {}", e))?;

        // State tracking for channels
        let mut has_analysis = false;
        let mut accumulated_final_text = String::new();
        let mut accumulated_tool_calls: Option<Vec<ToolCall>> = None;

        // Track which items we've started
        let mut reasoning_output_index: Option<usize> = None;
        let mut message_output_index: Option<usize> = None;
        let mut message_item_id: Option<String> = None;
        let mut has_emitted_content_part_added = false;

        // MCP tool call tracking (call_index -> (output_index, item_id))
        let mut mcp_call_tracking: HashMap<usize, (usize, String)> = HashMap::new();

        // Metadata from Complete message
        let mut finish_reason = String::from("stop");
        let mut matched_stop: Option<serde_json::Value> = None;

        // Process stream
        let mut chunk_count = 0;
        while let Some(result) = decode_stream.next().await {
            chunk_count += 1;
            let response = result.map_err(|e| format!("Decode stream error: {}", e))?;

            match response.response {
                Some(Chunk(chunk)) => {
                    // Parse chunk via Harmony parser
                    let delta_result = parser
                        .parse_chunk(&chunk.token_ids)
                        .map_err(|e| format!("Parse error: {}", e))?;

                    // Emit SSE events if there's a delta
                    if let Some(delta) = delta_result {
                        // Analysis channel → Reasoning item (wrapper events only, emitted once)
                        if let Some(_analysis_text) = &delta.analysis_delta {
                            if reasoning_output_index.is_none() {
                                // Allocate reasoning item and emit wrapper events
                                let (output_index, _item_id) =
                                    emitter.allocate_output_index(OutputItemType::Reasoning);
                                reasoning_output_index = Some(output_index);

                                // Emit reasoning item (added + done in one call)
                                // Note: reasoning_content will be provided at finalize
                                emitter
                                    .emit_reasoning_item(tx, None)
                                    .map_err(|e| format!("Failed to emit reasoning item: {}", e))?;

                                has_analysis = true;
                            }
                        }

                        // Final channel → Message item (WITH text streaming)
                        if let Some(final_delta) = &delta.final_delta {
                            if !final_delta.is_empty() {
                                // Allocate message item if needed
                                if message_output_index.is_none() {
                                    let (output_index, item_id) =
                                        emitter.allocate_output_index(OutputItemType::Message);
                                    message_output_index = Some(output_index);
                                    message_item_id = Some(item_id.clone());

                                    // Build message item structure
                                    let item = json!({
                                        "id": item_id,
                                        "type": "message",
                                        "role": "assistant",
                                        "content": []
                                    });

                                    // Emit output_item.added
                                    let event = emitter.emit_output_item_added(output_index, &item);
                                    emitter.send_event_best_effort(&event, tx);
                                }

                                let output_index = message_output_index.unwrap();
                                let item_id = message_item_id.as_ref().unwrap();
                                let content_index = 0; // Single content part

                                // Emit content_part.added before first delta
                                if !has_emitted_content_part_added {
                                    let event = emitter.emit_content_part_added(
                                        output_index,
                                        item_id,
                                        content_index,
                                    );
                                    emitter.send_event_best_effort(&event, tx);
                                    has_emitted_content_part_added = true;
                                }

                                // Emit text delta
                                let event = emitter.emit_text_delta(
                                    final_delta,
                                    output_index,
                                    item_id,
                                    content_index,
                                );
                                emitter.send_event_best_effort(&event, tx);

                                accumulated_final_text.push_str(final_delta);
                            }
                        }

                        // Commentary channel → MCP tool call streaming
                        if let Some(tc_delta) = &delta.commentary_delta {
                            let call_index = tc_delta.index;

                            // Check if this is a new tool call (has id and name)
                            if tc_delta.id.is_some() {
                                // NEW MCP CALL: Allocate output item and emit in_progress
                                let (output_index, item_id) =
                                    emitter.allocate_output_index(OutputItemType::McpCall);

                                // Store tracking info
                                mcp_call_tracking
                                    .insert(call_index, (output_index, item_id.clone()));

                                // Emit mcp_call.in_progress
                                let event =
                                    emitter.emit_mcp_call_in_progress(output_index, &item_id);
                                emitter.send_event_best_effort(&event, tx);

                                // If we have function name, emit initial mcp_call_arguments.delta
                                if let Some(func) = &tc_delta.function {
                                    if func.name.is_some() {
                                        let event = emitter.emit_mcp_call_arguments_delta(
                                            output_index,
                                            &item_id,
                                            "",
                                        );
                                        emitter.send_event_best_effort(&event, tx);
                                    }
                                }
                            } else {
                                // CONTINUING MCP CALL: Emit arguments delta
                                if let Some((output_index, item_id)) =
                                    mcp_call_tracking.get(&call_index)
                                {
                                    if let Some(args) = tc_delta
                                        .function
                                        .as_ref()
                                        .and_then(|f| f.arguments.as_ref())
                                        .filter(|a| !a.is_empty())
                                    {
                                        let event = emitter.emit_mcp_call_arguments_delta(
                                            *output_index,
                                            item_id,
                                            args,
                                        );
                                        emitter.send_event_best_effort(&event, tx);
                                    }
                                }
                            }
                        }
                    }
                }
                Some(Complete(complete)) => {
                    // Store final metadata
                    finish_reason = complete.finish_reason.clone();
                    matched_stop = complete.matched_stop.as_ref().map(|m| match m {
                        MatchedTokenId(id) => {
                            serde_json::json!(id)
                        }
                        MatchedStopStr(s) => {
                            serde_json::json!(s)
                        }
                    });

                    // Finalize parser and get complete output
                    let final_output = parser
                        .finalize(finish_reason.clone(), matched_stop.clone())
                        .map_err(|e| format!("Finalize error: {}", e))?;

                    // Store finalized tool calls
                    accumulated_tool_calls = final_output.commentary.clone();

                    // Complete all MCP tool calls if we have commentary
                    if let Some(ref tool_calls) = accumulated_tool_calls {
                        for (call_idx, tool_call) in tool_calls.iter().enumerate() {
                            if let Some((output_index, item_id)) = mcp_call_tracking.get(&call_idx)
                            {
                                // Emit mcp_call_arguments.done with final arguments
                                let args_str =
                                    tool_call.function.arguments.as_deref().unwrap_or("");
                                let event = emitter.emit_mcp_call_arguments_done(
                                    *output_index,
                                    item_id,
                                    args_str,
                                );
                                emitter.send_event_best_effort(&event, tx);

                                // Emit mcp_call.completed
                                let event = emitter.emit_mcp_call_completed(*output_index, item_id);
                                emitter.send_event_best_effort(&event, tx);

                                // Mark output item as completed
                                emitter.complete_output_item(*output_index);
                            }
                        }
                    }

                    // Close message item if we opened one
                    if let Some(output_index) = message_output_index {
                        let item_id = message_item_id.as_ref().unwrap();
                        let content_index = 0;

                        // Emit text_done
                        let event = emitter.emit_text_done(output_index, item_id, content_index);
                        emitter.send_event_best_effort(&event, tx);

                        // Emit content_part.done
                        let event =
                            emitter.emit_content_part_done(output_index, item_id, content_index);
                        emitter.send_event_best_effort(&event, tx);

                        // Emit output_item.done
                        let item = json!({
                            "id": item_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [{
                                "type": "text",
                                "text": accumulated_final_text.clone()
                            }]
                        });
                        let event = emitter.emit_output_item_done(output_index, &item);
                        emitter.send_event_best_effort(&event, tx);

                        emitter.complete_output_item(output_index);
                    }
                }
                Some(proto::generate_response::Response::Error(err)) => {
                    return Err(format!("Server error: {}", err.message));
                }
                None => {}
            }
        }

        debug!(
            "Stream loop ended. Total chunks received: {}, has_analysis: {}, tool_calls: {}, final_text_len: {}",
            chunk_count,
            has_analysis,
            accumulated_tool_calls.as_ref().map(|tc| tc.len()).unwrap_or(0),
            accumulated_final_text.len()
        );

        // Extract tool calls from completed messages or incomplete commentary
        if chunk_count > 0 && accumulated_tool_calls.is_none() {
            let messages = parser.get_messages();

            // Try extracting from completed messages first
            let (analysis_opt, commentary_opt, final_text_extracted) =
                HarmonyParserAdapter::parse_messages(&messages);
            accumulated_tool_calls = commentary_opt.clone();

            // If no tool calls found, check for incomplete commentary in parser state
            if accumulated_tool_calls.is_none() {
                accumulated_tool_calls = parser.extract_incomplete_commentary();
            }

            debug!(
                "Tool call extraction: completed_msgs={}, tool_calls={}, has_analysis={}, final_text_len={}",
                messages.len(),
                accumulated_tool_calls.as_ref().map(|tc| tc.len()).unwrap_or(0),
                analysis_opt.is_some(),
                final_text_extracted.len()
            );

            // Complete any pending MCP tool calls with data from completed messages
            if let Some(ref tool_calls) = accumulated_tool_calls {
                for (call_idx, tool_call) in tool_calls.iter().enumerate() {
                    if let Some((output_index, item_id)) = mcp_call_tracking.get(&call_idx) {
                        // Emit mcp_call_arguments.done with final arguments
                        let args_str = tool_call.function.arguments.as_deref().unwrap_or("");
                        let event =
                            emitter.emit_mcp_call_arguments_done(*output_index, item_id, args_str);
                        emitter.send_event_best_effort(&event, tx);

                        // Emit mcp_call.completed
                        let event = emitter.emit_mcp_call_completed(*output_index, item_id);
                        emitter.send_event_best_effort(&event, tx);
                    }
                }
            }
        }

        // Mark stream as completed successfully to prevent abort on drop
        decode_stream.mark_completed();

        // Return result based on whether tool calls were found
        if let Some(tool_calls) = accumulated_tool_calls {
            if !tool_calls.is_empty() {
                let analysis_content = if has_analysis {
                    // Get analysis from finalized parser output by calling finalize again
                    // This is safe because finalize can be called multiple times
                    let output = parser.finalize(finish_reason.clone(), matched_stop.clone())?;
                    output.analysis
                } else {
                    None
                };

                return Ok(ResponsesIterationResult::ToolCallsFound {
                    tool_calls,
                    analysis: analysis_content,
                    partial_text: accumulated_final_text,
                });
            }
        }

        // For streaming, we don't build the full ResponsesResponse here
        // The caller will build it from the SSE events
        // Return a placeholder Completed result (caller ignores these fields in streaming mode)
        Ok(ResponsesIterationResult::Completed {
            response: Box::new(ResponsesResponse {
                id: String::new(),
                object: "response".to_string(),
                created_at: 0,
                status: ResponseStatus::Completed,
                error: None,
                incomplete_details: None,
                instructions: None,
                max_output_tokens: None,
                model: String::new(),
                output: vec![],
                parallel_tool_calls: true,
                previous_response_id: None,
                reasoning: None,
                store: true,
                temperature: None,
                text: None,
                tool_choice: "auto".to_string(),
                tools: vec![],
                top_p: None,
                truncation: None,
                user: None,
                safety_identifier: None,
                metadata: HashMap::new(),
                usage: Some(ResponsesUsage::Modern(ResponseUsage {
                    input_tokens: 0,
                    output_tokens: 0,
                    total_tokens: 0,
                    input_tokens_details: None,
                    output_tokens_details: None,
                })),
            }),
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
                completion_tokens_details: None,
            },
        })
    }

    /// Process streaming chunks for Responses API iteration
    ///
    /// Returns ResponsesIterationResult indicating whether tool calls were found
    /// (requiring MCP loop continuation) or if the iteration is complete.
    pub async fn process_responses_iteration_stream(
        execution_result: context::ExecutionResult,
        emitter: &mut ResponseStreamEventEmitter,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    ) -> Result<ResponsesIterationResult, String> {
        match execution_result {
            context::ExecutionResult::Single { stream } => {
                debug!("Processing Responses API single stream mode");
                Self::process_responses_single_stream(stream, emitter, tx).await
            }
            context::ExecutionResult::Dual { prefill, decode } => {
                debug!("Processing Responses API dual stream mode");
                Self::process_responses_dual_stream(prefill, *decode, emitter, tx).await
            }
        }
    }

    /// Process streaming chunks from a single stream (Responses API)
    async fn process_responses_single_stream(
        grpc_stream: AbortOnDropStream,
        emitter: &mut ResponseStreamEventEmitter,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    ) -> Result<ResponsesIterationResult, String> {
        // Delegate to common helper
        Self::process_decode_stream_common(grpc_stream, emitter, tx).await
    }

    /// Process streaming chunks from dual streams (Responses API)
    async fn process_responses_dual_stream(
        mut prefill_stream: AbortOnDropStream,
        decode_stream: AbortOnDropStream,
        emitter: &mut ResponseStreamEventEmitter,
        tx: &mpsc::UnboundedSender<Result<Bytes, io::Error>>,
    ) -> Result<ResponsesIterationResult, String> {
        // Phase 1: Process prefill stream (collect metadata, no output)
        while let Some(result) = prefill_stream.next().await {
            let _response = result.map_err(|e| format!("Prefill stream error: {}", e))?;
            // No-op for prefill in Responses API (just metadata collection)
        }

        // Phase 2: Process decode stream using common helper
        let result = Self::process_decode_stream_common(decode_stream, emitter, tx).await;

        // Mark prefill stream as completed AFTER decode completes successfully
        // This ensures that if client disconnects during decode, BOTH streams send abort
        prefill_stream.mark_completed();

        result
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
