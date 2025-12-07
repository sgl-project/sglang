//! gRPC response converter FFI functions

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::ptr;
use std::sync::Arc;
use std::collections::HashMap;
use serde_json::Value;
use tokio::runtime::Runtime;
use once_cell::sync::Lazy;

use sgl_model_gateway::tokenizer::traits::Tokenizer;
use sgl_model_gateway::tokenizer::stream::DecodeStream;
use sgl_model_gateway::tool_parser::ToolParser;
use sgl_model_gateway::protocols::common::{Tool, ToolChoice, ToolChoiceValue, ToolCallDelta, FunctionCallDelta, Usage, StringOrArray};
use sgl_model_gateway::tokenizer::stop::StopSequenceDecoder;
use sgl_model_gateway::grpc_client::sglang_proto as proto;

use super::error::{SglErrorCode, set_error_message, clear_error_message};
use super::tokenizer::TokenizerHandle;
use super::utils::generate_tool_call_id;

/// Global parser factory (initialized once)
// Use the re-exported ParserFactory from tool_parser module
static PARSER_FACTORY: Lazy<sgl_model_gateway::tool_parser::ParserFactory> = Lazy::new(|| {
    // ParserFactory is re-exported from tool_parser::factory, so we can use it directly
    sgl_model_gateway::tool_parser::ParserFactory::default()
});

/// Global tokio runtime for async operations
static RUNTIME: Lazy<Runtime> = Lazy::new(|| {
    Runtime::new().expect("Failed to create tokio runtime for gRPC converter FFI")
});

/// Handle for gRPC response converter (maintains state for streaming)
#[repr(C)]
pub struct GrpcResponseConverterHandle {
    pub(crate) tokenizer: Arc<dyn Tokenizer>,
    pub(crate) tool_parser: Option<Arc<tokio::sync::Mutex<Box<dyn ToolParser>>>>,
    pub(crate) stop_decoder: Option<Arc<tokio::sync::Mutex<StopSequenceDecoder>>>,
    pub(crate) model: String,
    pub(crate) request_id: String,
    pub(crate) created: u64,
    pub(crate) system_fingerprint: Option<String>,
    pub(crate) tools: Option<Vec<Tool>>,
    pub(crate) tool_choice: Option<ToolChoice>,
    pub(crate) history_tool_calls_count: usize,
    pub(crate) stream_buffers: HashMap<u32, String>, // Per-index text buffers
    pub(crate) decode_streams: HashMap<u32, DecodeStream>, // Per-index incremental decoders
    pub(crate) has_tool_calls: HashMap<u32, bool>, // Track if tool calls were emitted
    pub(crate) is_first_chunk: HashMap<u32, bool>, // Track first chunk per index
    pub(crate) prompt_tokens: HashMap<u32, i32>, // Track prompt tokens per index (from chunks)
    pub(crate) completion_tokens: HashMap<u32, i32>, // Track completion tokens per index (cumulative)
    pub(crate) initial_prompt_tokens: Option<i32>, // Initial prompt tokens from request (if available)
    pub(crate) skip_special_tokens: bool, // Whether to skip special tokens when decoding
}

/// Create a gRPC response converter handle
///
/// # Arguments
/// * `tokenizer_handle` - Tokenizer handle (must be valid)
/// * `model` - Model name
/// * `request_id` - Request ID
/// * `tools_json` - Optional JSON array of tools
/// * `tool_choice_json` - Optional JSON object for tool_choice
/// * `stop` - Optional stop sequences (JSON array)
/// * `stop_token_ids` - Optional stop token IDs (JSON array)
/// * `skip_special_tokens` - Whether to skip special tokens
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * Pointer to GrpcResponseConverterHandle on success, null on failure
#[no_mangle]
pub unsafe extern "C" fn sgl_grpc_response_converter_create(
    tokenizer_handle: *mut TokenizerHandle,
    model: *const c_char,
    request_id: *const c_char,
    tools_json: *const c_char,
    tool_choice_json: *const c_char,
    stop: *const c_char,
    stop_token_ids: *const c_char,
    skip_special_tokens: c_int,
    error_out: *mut *mut c_char,
) -> *mut GrpcResponseConverterHandle {
    if tokenizer_handle.is_null() || model.is_null() || request_id.is_null() {
        set_error_message(error_out, "Invalid arguments: null pointer");
        return ptr::null_mut();
    }

    let model_str = match CStr::from_ptr(model).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in model");
            return ptr::null_mut();
        }
    };

    let request_id_str = match CStr::from_ptr(request_id).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in request_id");
            return ptr::null_mut();
        }
    };

    let handle_ref = &*tokenizer_handle;
    let tokenizer = Arc::clone(&handle_ref.tokenizer);

    // Parse tools if provided
    let tools: Option<Vec<Tool>> = if !tools_json.is_null() {
        match CStr::from_ptr(tools_json).to_str() {
            Ok(s) => serde_json::from_str::<Vec<Tool>>(s).ok(),
            Err(_) => None,
        }
    } else {
        None
    };

    // Parse tool_choice if provided
    let tool_choice: Option<ToolChoice> = if !tool_choice_json.is_null() {
        match CStr::from_ptr(tool_choice_json).to_str() {
            Ok(s) => serde_json::from_str::<ToolChoice>(s).ok(),
            Err(_) => None,
        }
    } else {
        None
    };

    // Parse stop sequences
    let stop: Option<StringOrArray> = if !stop.is_null() {
        let stop_str = match CStr::from_ptr(stop).to_str() {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        };
        serde_json::from_str::<StringOrArray>(stop_str).ok()
    } else {
        None
    };

    // Parse stop token IDs
    let stop_token_ids: Option<Vec<u32>> = if !stop_token_ids.is_null() {
        let ids_str = match CStr::from_ptr(stop_token_ids).to_str() {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        };
        serde_json::from_str::<Vec<u32>>(ids_str).ok()
    } else {
        None
    };

    // Create stop decoder if needed
    let stop_decoder = if stop.is_some() || stop_token_ids.is_some() {
        Some(Arc::new(tokio::sync::Mutex::new(
            sgl_model_gateway::routers::grpc::utils::create_stop_decoder(
                &tokenizer,
                stop.as_ref(),
                stop_token_ids.as_ref(),
                skip_special_tokens != 0,
                false, // no_stop_trim
            ),
        )))
    } else {
        None
    };

    // Create tool parser if tools are provided
    let tool_parser = if tools.is_some() {
        PARSER_FACTORY.registry().create_for_model(model_str)
            .map(|p| Arc::new(tokio::sync::Mutex::new(p)))
    } else {
        None
    };

    // Get system fingerprint from model (simplified)
    let system_fingerprint = Some("fp_placeholder".to_string()); // TODO: Get actual fingerprint

    Box::into_raw(Box::new(GrpcResponseConverterHandle {
        tokenizer,
        tool_parser,
        stop_decoder,
        model: model_str.to_string(),
        request_id: request_id_str.to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        system_fingerprint,
        tools,
        tool_choice,
        history_tool_calls_count: 0,
        stream_buffers: HashMap::new(),
        decode_streams: HashMap::new(),
        has_tool_calls: HashMap::new(),
        is_first_chunk: HashMap::new(),
        prompt_tokens: HashMap::new(),
        completion_tokens: HashMap::new(),
        initial_prompt_tokens: None, // Will be set from stream handle
        skip_special_tokens: skip_special_tokens != 0,
    }))
}

/// Convert a gRPC GenerateResponse chunk to OpenAI format
///
/// # Arguments
/// * `handle` - Converter handle
/// * `response_json` - JSON string of proto.GenerateResponse
/// * `result_json_out` - Pointer to receive OpenAI format JSON (must be freed with sgl_free_string)
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * SglErrorCode::Success on success, error code on failure
#[no_mangle]
pub unsafe extern "C" fn sgl_grpc_response_converter_convert_chunk(
    handle: *mut GrpcResponseConverterHandle,
    response_json: *const c_char,
    result_json_out: *mut *mut c_char,
    error_out: *mut *mut c_char,
) -> SglErrorCode {
    if handle.is_null() || response_json.is_null() || result_json_out.is_null() {
        set_error_message(error_out, "Invalid arguments: null pointer");
        return SglErrorCode::InvalidArgument;
    }

    let response_str = match CStr::from_ptr(response_json).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in response_json");
            return SglErrorCode::InvalidArgument;
        }
    };

    // Parse proto.GenerateResponse from JSON
    let json_value: Value = match serde_json::from_str(response_str) {
        Ok(v) => v,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to parse response JSON: {}", e));
            return SglErrorCode::ParsingError;
        }
    };

    // Build proto::GenerateResponse from JSON value
    let mut proto_response = proto::GenerateResponse {
        request_id: json_value.get("request_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        response: None,
    };

    // Parse the response oneof field
    if let Some(chunk_json) = json_value.get("chunk") {
        let chunk = proto::GenerateStreamChunk {
            token_ids: chunk_json.get("token_ids")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_u64().map(|n| n as u32)).collect())
                .unwrap_or_default(),
            prompt_tokens: chunk_json.get("prompt_tokens")
                .and_then(|v| v.as_i64())
                .map(|n| n as i32)
                .unwrap_or(0),
            completion_tokens: chunk_json.get("completion_tokens")
                .and_then(|v| v.as_i64())
                .map(|n| n as i32)
                .unwrap_or(0),
            cached_tokens: chunk_json.get("cached_tokens")
                .and_then(|v| v.as_i64())
                .map(|n| n as i32)
                .unwrap_or(0),
            output_logprobs: None,
            hidden_states: vec![],
            input_logprobs: None,
            index: 0,
        };
        proto_response.response = Some(proto::generate_response::Response::Chunk(chunk));
    } else if let Some(complete_json) = json_value.get("complete") {
        let complete = proto::GenerateComplete {
            output_ids: complete_json.get("output_ids")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_u64().map(|n| n as u32)).collect())
                .unwrap_or_default(),
            finish_reason: complete_json.get("finish_reason")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            prompt_tokens: complete_json.get("prompt_tokens")
                .and_then(|v| v.as_i64())
                .map(|n| n as i32)
                .unwrap_or(0),
            completion_tokens: complete_json.get("completion_tokens")
                .and_then(|v| v.as_i64())
                .map(|n| n as i32)
                .unwrap_or(0),
            cached_tokens: complete_json.get("cached_tokens")
                .and_then(|v| v.as_i64())
                .map(|n| n as i32)
                .unwrap_or(0),
            output_logprobs: None,
            all_hidden_states: vec![],
            input_logprobs: None,
            matched_stop: None,
            index: 0,
        };
        proto_response.response = Some(proto::generate_response::Response::Complete(complete));
    } else if let Some(error_json) = json_value.get("error") {
        let error = proto::GenerateError {
            message: error_json.get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            http_status_code: error_json.get("http_status_code")
                .and_then(|v| v.as_str())
                .unwrap_or("500")
                .to_string(),
            details: error_json.get("details")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
        };
        proto_response.response = Some(proto::generate_response::Response::Error(error));
    } else {
        set_error_message(error_out, "Response JSON must contain 'chunk', 'complete', or 'error' field");
        return SglErrorCode::ParsingError;
    }

    let handle_ref = &mut *handle;
    let tokenizer = Arc::clone(&handle_ref.tokenizer);
    let model = handle_ref.model.clone();
    let request_id = handle_ref.request_id.clone();
    let created = handle_ref.created;
    let system_fingerprint = handle_ref.system_fingerprint.clone();

    // Use tokio runtime to run async code
    let result = RUNTIME.block_on(async {
        convert_proto_chunk_to_openai(
            proto_response,
            handle_ref,
            &tokenizer,
            &model,
            &request_id,
            created,
            system_fingerprint.as_deref(),
        )
        .await
    });

    match result {
        Ok(Some(openai_response)) => {
            // Serialize to JSON
            let result_str = match serde_json::to_string(&openai_response) {
                Ok(s) => s,
                Err(e) => {
                    set_error_message(error_out, &format!("Failed to serialize response: {}", e));
                    return SglErrorCode::ParsingError;
                }
            };

            let result_cstr = match CString::new(result_str) {
                Ok(s) => s,
                Err(e) => {
                    set_error_message(error_out, &format!("Failed to create result string: {}", e));
                    return SglErrorCode::MemoryError;
                }
            };

            *result_json_out = result_cstr.into_raw();
            clear_error_message(error_out);
            SglErrorCode::Success
        }
        Ok(None) => {
            // No response to send (e.g., empty chunk)
            let empty = CString::new("").unwrap();
            *result_json_out = empty.into_raw();
            clear_error_message(error_out);
            SglErrorCode::Success
        }
        Err(e) => {
            set_error_message(error_out, &format!("Conversion error: {}", e));
            SglErrorCode::ParsingError
        }
    }
}

/// Helper function to convert proto chunk to OpenAI format
pub(crate) async fn convert_proto_chunk_to_openai(
    proto_response: proto::GenerateResponse,
    handle: &mut GrpcResponseConverterHandle,
    tokenizer: &Arc<dyn Tokenizer>,
    model: &str,
    request_id: &str,
    created: u64,
    system_fingerprint: Option<&str>,
) -> Result<Option<sgl_model_gateway::protocols::chat::ChatCompletionStreamResponse>, String> {
    use sgl_model_gateway::grpc_client::sglang_proto::generate_response::Response::*;
    use sgl_model_gateway::protocols::chat::{ChatCompletionStreamResponse, ChatMessageDelta, ChatStreamChoice};

    match proto_response.response {
        Some(Chunk(chunk)) => {
            let index = chunk.index;

            // Mark as not first chunk if we've seen this index before
            let is_first = handle.is_first_chunk.entry(index).or_insert(true);
            let first_chunk = *is_first;
            *is_first = false;

            // Track token counts from chunks (cumulative values from proto)
            // These are cumulative values, so we always use the latest value
            // For prompt_tokens, if chunk value is 0, preserve existing value or use initial_prompt_tokens
            // This prevents overwriting valid prompt_tokens with 0
            if chunk.prompt_tokens > 0 {
                handle.prompt_tokens.insert(index, chunk.prompt_tokens);
            } else {
                // If chunk.prompt_tokens is 0, try to preserve existing value or use initial_prompt_tokens
                if !handle.prompt_tokens.contains_key(&index) {
                    // No existing value, try to use initial_prompt_tokens
                    if let Some(initial_prompt) = handle.initial_prompt_tokens {
                        handle.prompt_tokens.insert(index, initial_prompt);
                    }
                }
                // If existing value exists, keep it (don't overwrite with 0)
            }
            // For completion_tokens, always update (even if 0) as it's cumulative
            handle.completion_tokens.insert(index, chunk.completion_tokens);

            // Process tokens through stop decoder if available, otherwise use incremental decoder
            let chunk_text = if let Some(ref stop_decoder) = handle.stop_decoder {
                let mut decoder_guard = stop_decoder.lock().await;
                let mut text = String::new();
                for &token_id in &chunk.token_ids {
                    match decoder_guard.process_token(token_id).unwrap_or_else(|_| {
                        sgl_model_gateway::tokenizer::stop::SequenceDecoderOutput::Held
                    }) {
                        sgl_model_gateway::tokenizer::stop::SequenceDecoderOutput::Text(t) => {
                            text.push_str(&t);
                        }
                        sgl_model_gateway::tokenizer::stop::SequenceDecoderOutput::StoppedWithText(t) => {
                            text.push_str(&t);
                            break;
                        }
                        sgl_model_gateway::tokenizer::stop::SequenceDecoderOutput::Stopped => {
                            break;
                        }
                        sgl_model_gateway::tokenizer::stop::SequenceDecoderOutput::Held => {}
                    }
                }
                text
            } else {
                // Use incremental decoder to handle multi-byte character boundaries
                let decode_stream = handle.decode_streams.entry(index).or_insert_with(|| {
                    DecodeStream::new(
                        Arc::clone(&tokenizer),
                        &[], // No prompt tokens for completion
                        handle.skip_special_tokens,
                    )
                });

                // Process tokens incrementally
                let mut text_parts = Vec::new();
                for &token_id in &chunk.token_ids {
                    if let Ok(Some(text)) = decode_stream.step(token_id) {
                        text_parts.push(text);
                    }
                }
                text_parts.join("")
            };

            if chunk_text.is_empty() {
                return Ok(None);
            }

            // Send first chunk with role
            if first_chunk {
                let first_response = ChatCompletionStreamResponse {
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
                            reasoning_content: None,
                        },
                        logprobs: None,
                        finish_reason: None,
                        matched_stop: None,
                    }],
                    usage: None,
                };
                return Ok(Some(first_response));
            }

            // Update stream buffer
            let stream_buffer = handle.stream_buffers.entry(index).or_default();
            stream_buffer.push_str(&chunk_text);

            // Handle tool calls if tools are provided
            if let (Some(ref tools), Some(ref tool_parser)) = (handle.tools.as_ref(), handle.tool_parser.as_ref()) {
                let tool_choice_enabled = !matches!(
                    handle.tool_choice,
                    Some(ToolChoice::Value(ToolChoiceValue::None))
                );

                if tool_choice_enabled {
                    let mut parser_guard = tool_parser.lock().await;
                    match parser_guard.parse_incremental(&chunk_text, tools).await {
                        Ok(streaming_result) => {
                            if !streaming_result.calls.is_empty() {
                                handle.has_tool_calls.insert(index, true);
                                // Convert tool call items to OpenAI format
                                let tool_call_deltas: Vec<_> = streaming_result
                                    .calls
                                    .into_iter()
                                    .map(|item| {
                                        let id = if let Some(ref name) = item.name {
                                            generate_tool_call_id(
                                                model,
                                                name,
                                                item.tool_index,
                                                handle.history_tool_calls_count,
                                            )
                                        } else {
                                            format!("call_{}", item.tool_index)
                                        };

                                        ToolCallDelta {
                                            index: item.tool_index as u32,
                                            id: Some(id),
                                            tool_type: if item.name.is_some() {
                                                Some("function".to_string())
                                            } else {
                                                None
                                            },
                                            function: Some(FunctionCallDelta {
                                                name: item.name,
                                                arguments: if !item.parameters.is_empty() {
                                                    Some(item.parameters)
                                                } else {
                                                    None
                                                },
                                            }),
                                        }
                                    })
                                    .collect();

                                let tool_response = ChatCompletionStreamResponse {
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
                                            tool_calls: Some(tool_call_deltas),
                                            reasoning_content: None,
                                        },
                                        logprobs: None,
                                        finish_reason: None,
                                        matched_stop: None,
                                    }],
                                    usage: None,
                                };
                                return Ok(Some(tool_response));
                            }
                        }
                        Err(e) => {
                            // Log error but continue with regular content
                            tracing::warn!("Tool parser error: {}", e);
                        }
                    }
                }
            }

            // Regular content emission
            let content_response = ChatCompletionStreamResponse {
                id: request_id.to_string(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model.to_string(),
                system_fingerprint: system_fingerprint.map(|s| s.to_string()),
                choices: vec![ChatStreamChoice {
                    index,
                    delta: ChatMessageDelta {
                        role: Some("assistant".to_string()),
                        content: Some(chunk_text),
                        tool_calls: None,
                        reasoning_content: None,
                    },
                    logprobs: None,
                    finish_reason: None,
                    matched_stop: None,
                }],
                usage: None,
            };

            Ok(Some(content_response))
        }
        Some(Complete(complete)) => {
            let index = complete.index;

            // Flush any remaining text
            // Flush any remaining text from decode stream
            let mut final_text = handle.stream_buffers.remove(&index).unwrap_or_default();
            if let Some(ref mut decode_stream) = handle.decode_streams.get_mut(&index) {
                if let Ok(Some(remaining)) = decode_stream.flush() {
                    final_text.push_str(&remaining);
                }
            }
            handle.decode_streams.remove(&index);

            // Determine finish reason - ensure it's never empty
            // If finish_reason is empty, try to infer from other fields or use default
            let finish_reason = if handle.has_tool_calls.get(&index).copied().unwrap_or(false)
                && (complete.finish_reason == "stop" || complete.finish_reason.is_empty())
            {
                "tool_calls".to_string()
            } else if complete.finish_reason.is_empty() || complete.finish_reason.trim().is_empty() {
                // If finish_reason is empty, try to infer from completion_tokens or use default
                if complete.completion_tokens > 0 {
                    // If we have completion tokens, likely stopped normally
                    "stop".to_string()
                } else if !complete.output_ids.is_empty() {
                    // If we have output_ids, likely stopped normally
                    "stop".to_string()
                } else {
                    // Default fallback - always ensure we have a value
                    "stop".to_string()
                }
            } else {
                complete.finish_reason.clone()
            };

            // Ensure finish_reason is never empty (defensive check)
            let finish_reason = if finish_reason.is_empty() || finish_reason.trim().is_empty() {
                "stop".to_string()
            } else {
                finish_reason
            };

            // Extract matched_stop
            let matched_stop = match &complete.matched_stop {
                Some(proto::generate_complete::MatchedStop::MatchedTokenId(token_id)) => {
                    Some(Value::Number(serde_json::Number::from(*token_id)))
                }
                Some(proto::generate_complete::MatchedStop::MatchedStopStr(stop_str)) => {
                    Some(Value::String(stop_str.clone()))
                }
                None => None,
            };

            // Build usage - prefer values from complete message, but fallback to accumulated values from chunks
            // Complete message should have the final values, but sometimes they might be 0 or missing
            // Always use the latest cumulative value from chunks if available, otherwise use complete message value
            let mut prompt_tokens = handle.prompt_tokens.get(&index)
                .copied()
                .filter(|&v| v > 0)
                .unwrap_or(complete.prompt_tokens);
            let mut completion_tokens = handle.completion_tokens.get(&index)
                .copied()
                .filter(|&v| v > 0)
                .unwrap_or(complete.completion_tokens);

            // Always try to use initial_prompt_tokens if prompt_tokens is 0 or missing
            // This is the most reliable source for prompt tokens since we calculate it from the request
            if prompt_tokens == 0 {
                if let Some(initial_prompt) = handle.initial_prompt_tokens {
                    prompt_tokens = initial_prompt;
                }
            }

            // If completion_tokens is 0, try to infer from output_ids or accumulated chunks
            if completion_tokens == 0 {
                // Try to use completion_tokens from complete message even if 0
                // Or calculate from output_ids
                if complete.completion_tokens > 0 {
                    completion_tokens = complete.completion_tokens;
                } else if !complete.output_ids.is_empty() {
                    completion_tokens = complete.output_ids.len() as i32;
                } else if let Some(&last_completion) = handle.completion_tokens.get(&index) {
                    completion_tokens = last_completion;
                }
            }

            // Final fallback: if both are still 0, try to use initial_prompt_tokens for prompt
            // and calculate completion from output_ids
            if prompt_tokens == 0 && completion_tokens == 0 {
                // Try to infer from output_ids if available
                let output_ids_len = complete.output_ids.len() as i32;
                if output_ids_len > 0 {
                    completion_tokens = output_ids_len;
                    // Always try to use initial_prompt_tokens for prompt
                    if let Some(initial_prompt) = handle.initial_prompt_tokens {
                        prompt_tokens = initial_prompt;
                    }
                }
            }

            // Final defensive check: ensure prompt_tokens is set if we have initial_prompt_tokens
            if prompt_tokens == 0 {
                if let Some(initial_prompt) = handle.initial_prompt_tokens {
                    prompt_tokens = initial_prompt;
                }
            }

            // Always create usage, even if values are 0 (defensive)
            let usage = Some(Usage {
                prompt_tokens: prompt_tokens.max(0) as u32,
                completion_tokens: completion_tokens.max(0) as u32,
                total_tokens: (prompt_tokens.max(0) + completion_tokens.max(0)) as u32,
                completion_tokens_details: None,
            });

            let finish_response = ChatCompletionStreamResponse {
                id: request_id.to_string(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model.to_string(),
                system_fingerprint: system_fingerprint.map(|s| s.to_string()),
                choices: vec![ChatStreamChoice {
                    index,
                    delta: ChatMessageDelta {
                        role: Some("assistant".to_string()),
                        content: if !final_text.is_empty() {
                            Some(final_text)
                        } else {
                            None
                        },
                        tool_calls: None,
                        reasoning_content: None,
                    },
                    logprobs: None,
                    finish_reason: Some(finish_reason),
                    matched_stop,
                }],
                usage,
            };

            Ok(Some(finish_response))
        }
        Some(Error(error)) => {
            Err(format!("Server error: {} (status: {})", error.message, error.http_status_code))
        }
        None => Ok(None),
    }
}

/// Free a gRPC response converter handle
#[no_mangle]
pub unsafe extern "C" fn sgl_grpc_response_converter_free(handle: *mut GrpcResponseConverterHandle) {
    if !handle.is_null() {
        let _ = Box::from_raw(handle);
    }
}
