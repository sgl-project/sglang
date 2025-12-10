//! Postprocessing FFI functions for gRPC stream chunks
//!
//! This module provides C-compatible functions for postprocessing gRPC stream chunks:
//! - Parse tool calls from model output
//! - Convert proto format to OpenAI format
//! - Handle reasoning content parsing
//!
//! These functions are designed to be called for each stream chunk, but can be optimized
//! with batching in the future.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::ptr;
use std::sync::Arc;
use serde_json::Value;

use sgl_model_gateway::grpc_client::sglang_proto as proto;

use super::error::{SglErrorCode, set_error_message};
use super::grpc_converter::GrpcResponseConverterHandle;

use tokio::runtime::Runtime;
use once_cell::sync::Lazy;

/// Global tokio runtime for async operations
static RUNTIME: Lazy<Runtime> = Lazy::new(|| {
    Runtime::new().expect("Failed to create tokio runtime for postprocessor FFI")
});

/// Postprocess a gRPC stream chunk to OpenAI format
///
/// This function:
/// 1. Parses the proto chunk from JSON
/// 2. Converts it to OpenAI format using the converter handle
/// 3. Returns the OpenAI format JSON
///
/// # Arguments
/// * `converter_handle` - Converter handle (created with sgl_grpc_response_converter_create)
/// * `proto_chunk_json` - JSON string of proto.GenerateResponse
/// * `openai_json_out` - Pointer to receive OpenAI format JSON (must be freed with sgl_free_string)
/// * `is_done_out` - Pointer to receive is_done flag (1 if stream is complete, 0 otherwise)
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * SglErrorCode::Success on success, error code on failure
#[no_mangle]
pub unsafe extern "C" fn sgl_postprocess_stream_chunk(
    converter_handle: *mut GrpcResponseConverterHandle,
    proto_chunk_json: *const c_char,
    openai_json_out: *mut *mut c_char,
    is_done_out: *mut c_int,
    error_out: *mut *mut c_char,
) -> SglErrorCode {
    if converter_handle.is_null()
        || proto_chunk_json.is_null()
        || openai_json_out.is_null()
        || is_done_out.is_null()
    {
        set_error_message(error_out, "Invalid arguments: null pointer");
        return SglErrorCode::InvalidArgument;
    }

    let proto_chunk_str = match CStr::from_ptr(proto_chunk_json).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in proto_chunk_json");
            return SglErrorCode::InvalidArgument;
        }
    };

    // Parse proto.GenerateResponse from JSON
    let json_value: Value = match serde_json::from_str(proto_chunk_str) {
        Ok(v) => v,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to parse proto chunk JSON: {}", e));
            return SglErrorCode::ParsingError;
        }
    };

    // Build proto::GenerateResponse from JSON value
    let mut proto_response = proto::GenerateResponse {
        request_id: json_value
            .get("request_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        response: None,
    };

    // Parse the response oneof field
    let is_done = if let Some(chunk_json) = json_value.get("chunk") {
        let chunk = proto::GenerateStreamChunk {
            token_ids: chunk_json
                .get("token_ids")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as u32))
                        .collect()
                })
                .unwrap_or_default(),
            prompt_tokens: chunk_json
                .get("prompt_tokens")
                .and_then(|v| v.as_i64())
                .map(|n| n as i32)
                .unwrap_or(0),
            completion_tokens: chunk_json
                .get("completion_tokens")
                .and_then(|v| v.as_i64())
                .map(|n| n as i32)
                .unwrap_or(0),
            cached_tokens: chunk_json
                .get("cached_tokens")
                .and_then(|v| v.as_i64())
                .map(|n| n as i32)
                .unwrap_or(0),
            output_logprobs: None,
            hidden_states: vec![],
            input_logprobs: None,
            index: 0,
        };
        proto_response.response = Some(proto::generate_response::Response::Chunk(chunk));
        false
    } else if let Some(complete_json) = json_value.get("complete") {
        let complete = proto::GenerateComplete {
            output_ids: complete_json
                .get("output_ids")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as u32))
                        .collect()
                })
                .unwrap_or_default(),
            finish_reason: complete_json
                .get("finish_reason")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            prompt_tokens: complete_json
                .get("prompt_tokens")
                .and_then(|v| v.as_i64())
                .map(|n| n as i32)
                .unwrap_or(0),
            completion_tokens: complete_json
                .get("completion_tokens")
                .and_then(|v| v.as_i64())
                .map(|n| n as i32)
                .unwrap_or(0),
            cached_tokens: complete_json
                .get("cached_tokens")
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
        true
    } else if let Some(error_json) = json_value.get("error") {
        let error = proto::GenerateError {
            message: error_json
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            http_status_code: error_json
                .get("http_status_code")
                .and_then(|v| v.as_str())
                .unwrap_or("500")
                .to_string(),
            details: error_json
                .get("details")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
        };
        proto_response.response = Some(proto::generate_response::Response::Error(error));
        true
    } else {
        set_error_message(
            error_out,
            "Proto chunk JSON must contain 'chunk', 'complete', or 'error' field",
        );
        return SglErrorCode::ParsingError;
    };

    // Convert proto chunk to OpenAI format using the converter's convert_chunk function
    // We'll use the existing converter API instead of calling the internal function directly
    let proto_chunk_json_cstr = match CString::new(proto_chunk_str) {
        Ok(s) => s,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to create C string: {}", e));
            return SglErrorCode::MemoryError;
        }
    };

    // Use the existing converter API
    let mut openai_json_ptr: *mut c_char = ptr::null_mut();
    let result = super::grpc_converter::sgl_grpc_response_converter_convert_chunk(
        converter_handle,
        proto_chunk_json_cstr.as_ptr(),
        &mut openai_json_ptr,
        error_out,
    );

    if result == SglErrorCode::Success {
        *openai_json_out = openai_json_ptr;
        *is_done_out = if is_done { 1 } else { 0 };
        SglErrorCode::Success
    } else {
        *openai_json_out = ptr::null_mut();
        *is_done_out = if is_done { 1 } else { 0 };
        result
    }
}

/// Postprocess multiple gRPC stream chunks in batch (reduces FFI overhead)
///
/// This function processes multiple chunks in a single FFI call, significantly reducing
/// FFI overhead in streaming scenarios.
///
/// # Arguments
/// * `converter_handle` - Converter handle (created with sgl_grpc_response_converter_create)
/// * `proto_chunks_json_array` - JSON array string of proto.GenerateResponse chunks
/// * `max_chunks` - Maximum number of chunks to process (for safety)
/// * `openai_chunks_json_array_out` - Pointer to receive JSON array of OpenAI format chunks (must be freed with sgl_free_string)
/// * `chunks_count_out` - Pointer to receive number of processed chunks
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * SglErrorCode::Success on success, error code on failure
#[no_mangle]
pub unsafe extern "C" fn sgl_postprocess_stream_chunks_batch(
    converter_handle: *mut GrpcResponseConverterHandle,
    proto_chunks_json_array: *const c_char,
    max_chunks: c_int,
    openai_chunks_json_array_out: *mut *mut c_char,
    chunks_count_out: *mut c_int,
    error_out: *mut *mut c_char,
) -> SglErrorCode {
    if converter_handle.is_null()
        || proto_chunks_json_array.is_null()
        || openai_chunks_json_array_out.is_null()
        || chunks_count_out.is_null()
    {
        set_error_message(error_out, "Invalid arguments: null pointer");
        return SglErrorCode::InvalidArgument;
    }

    let chunks_array_str = match CStr::from_ptr(proto_chunks_json_array).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in proto_chunks_json_array");
            return SglErrorCode::InvalidArgument;
        }
    };

    // Parse JSON array of chunks
    let chunks_array: Vec<Value> = match serde_json::from_str(chunks_array_str) {
        Ok(arr) => arr,
        Err(e) => {
            set_error_message(
                error_out,
                &format!("Failed to parse chunks JSON array: {}", e),
            );
            return SglErrorCode::ParsingError;
        }
    };

    // Limit batch size for safety
    let max_chunks_usize = max_chunks as usize;
    let chunks_to_process = if chunks_array.len() > max_chunks_usize {
        &chunks_array[..max_chunks_usize]
    } else {
        &chunks_array
    };

    let handle_ref = &mut *converter_handle;
    let tokenizer = Arc::clone(&handle_ref.tokenizer);
    let model = handle_ref.model.clone();
    let request_id = handle_ref.request_id.clone();
    let created = handle_ref.created;
    let system_fingerprint = handle_ref.system_fingerprint.clone();

    // Process chunks in batch
    let mut results = Vec::new();
    let mut has_error = false;
    let mut error_msg = String::new();

    for chunk_json in chunks_to_process {
        // Parse proto.GenerateResponse from JSON
        let mut proto_response = proto::GenerateResponse {
            request_id: chunk_json
                .get("request_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            response: None,
        };

        // Parse the response oneof field (same logic as single chunk processing)
        let _is_done = if let Some(chunk_json) = chunk_json.get("chunk") {
            let chunk = proto::GenerateStreamChunk {
                token_ids: chunk_json
                    .get("token_ids")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_u64().map(|n| n as u32))
                            .collect()
                    })
                    .unwrap_or_default(),
                prompt_tokens: chunk_json
                    .get("prompt_tokens")
                    .and_then(|v| v.as_i64())
                    .map(|n| n as i32)
                    .unwrap_or(0),
                completion_tokens: chunk_json
                    .get("completion_tokens")
                    .and_then(|v| v.as_i64())
                    .map(|n| n as i32)
                    .unwrap_or(0),
                cached_tokens: chunk_json
                    .get("cached_tokens")
                    .and_then(|v| v.as_i64())
                    .map(|n| n as i32)
                    .unwrap_or(0),
                output_logprobs: None,
                hidden_states: vec![],
                input_logprobs: None,
                index: 0,
            };
            proto_response.response = Some(proto::generate_response::Response::Chunk(chunk));
            false
        } else if let Some(complete_json) = chunk_json.get("complete") {
            let complete = proto::GenerateComplete {
                output_ids: complete_json
                    .get("output_ids")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_u64().map(|n| n as u32))
                            .collect()
                    })
                    .unwrap_or_default(),
                finish_reason: complete_json
                    .get("finish_reason")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                prompt_tokens: complete_json
                    .get("prompt_tokens")
                    .and_then(|v| v.as_i64())
                    .map(|n| n as i32)
                    .unwrap_or(0),
                completion_tokens: complete_json
                    .get("completion_tokens")
                    .and_then(|v| v.as_i64())
                    .map(|n| n as i32)
                    .unwrap_or(0),
                cached_tokens: complete_json
                    .get("cached_tokens")
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
            true
        } else if let Some(error_json) = chunk_json.get("error") {
            let error = proto::GenerateError {
                message: error_json
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                http_status_code: error_json
                    .get("http_status_code")
                    .and_then(|v| v.as_str())
                    .unwrap_or("500")
                    .to_string(),
                details: error_json
                    .get("details")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
            };
            proto_response.response = Some(proto::generate_response::Response::Error(error));
            true
        } else {
            error_msg = format!(
                "Chunk JSON must contain 'chunk', 'complete', or 'error' field: {}",
                chunk_json
            );
            has_error = true;
            break;
        };

        // Convert proto chunk to OpenAI format
        let result = RUNTIME.block_on(async {
            super::grpc_converter::convert_proto_chunk_to_openai(
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
                results.push(openai_response);
            }
            Ok(None) => {
                // Empty response, skip
            }
            Err(e) => {
                error_msg = format!("Postprocessing failed for chunk: {}", e);
                has_error = true;
                break;
            }
        }
    }

    if has_error {
        set_error_message(error_out, &error_msg);
        return SglErrorCode::ParsingError;
    }

    // Serialize results to JSON array
    let results_json = match serde_json::to_string(&results) {
        Ok(s) => s,
        Err(e) => {
            set_error_message(
                error_out,
                &format!("Failed to serialize results JSON array: {}", e),
            );
            return SglErrorCode::ParsingError;
        }
    };

    let results_cstr = match CString::new(results_json) {
        Ok(s) => s,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to create C string: {}", e));
            return SglErrorCode::MemoryError;
        }
    };

    *openai_chunks_json_array_out = results_cstr.into_raw();
    *chunks_count_out = results.len() as c_int;

    SglErrorCode::Success
}
