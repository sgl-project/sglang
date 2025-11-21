//! Stream handling FFI functions
//!
//! This module provides FFI (Foreign Function Interface) functions for managing
//! streaming responses from the SGLang gRPC API. It handles:
//!
//! - Creating and managing stream handles
//! - Reading chunks from streams and converting them to OpenAI format
//! - Managing automatic abort on stream drop (via AbortOnDropStream)
//! - Thread-safe access to streams and response converters
//!
//! # Safety
//!
//! All FFI functions are marked `unsafe` as per Rust FFI conventions. Callers must:
//! - Pass valid pointers
//! - Ensure proper pointer lifetime management
//! - Call corresponding free functions for cleanup

use std::ffi::CString;
use std::os::raw::{c_char, c_int};
use std::ptr;
use std::sync::Arc;
use tokio::runtime::Runtime;
use once_cell::sync::Lazy;
use futures_util::StreamExt;

use sglang_router::grpc_client::{sglang_proto as proto, sglang_scheduler::{SglangSchedulerClient, AbortOnDropStream}};

use super::error::{SglErrorCode, set_error_message};
use super::grpc_converter::{GrpcResponseConverterHandle, convert_proto_chunk_to_openai};

/// Global tokio runtime for async operations
static RUNTIME: Lazy<Runtime> = Lazy::new(|| {
    Runtime::new().expect("Failed to create tokio runtime for stream FFI")
});

/// Handle for an active streaming request.
///
/// This struct manages the stream and response converter for a single request.
/// It is wrapped in Arc and Mutex for thread-safe concurrent access.
///
/// # Fields
///
/// * `stream` - The gRPC stream wrapped in AbortOnDropStream for automatic cleanup
/// * `converter` - Response converter that transforms proto messages to OpenAI format
/// * `client` - The underlying gRPC client connection
/// * `prompt_tokens` - Number of prompt tokens from the original request
pub struct SglangStreamHandle {
    pub(crate) stream: Arc<tokio::sync::Mutex<AbortOnDropStream>>,
    pub(crate) converter: Arc<tokio::sync::Mutex<GrpcResponseConverterHandle>>,
    #[allow(dead_code)]
    pub(crate) client: Arc<SglangSchedulerClient>,
    #[allow(dead_code)]
    pub(crate) prompt_tokens: i32, // Number of prompt tokens for this request
}

/// Read next chunk from stream and convert to OpenAI format.
///
/// This function reads the next chunk from the gRPC stream, converts it from the
/// internal protocol format to OpenAI-compatible JSON format, and returns it via
/// the output parameters.
///
/// # Arguments
///
/// * `stream_handle` - Mutable pointer to the stream handle
/// * `response_json_out` - Pointer to receive OpenAI format JSON string
///   - Caller must free this with `sgl_free_string`
///   - May be NULL if no data available
/// * `is_done_out` - Pointer to receive completion status
///   - 0 = stream has more data
///   - 1 = stream is complete
/// * `error_out` - Optional pointer to receive error message
///   - Only set if function returns an error code
///   - Must be freed with `sgl_free_string` if not NULL
///
/// # Returns
///
/// * `SglErrorCode::Success` - Successfully read a chunk or reached end of stream
/// * Other error codes - See `SglErrorCode` for details
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - `stream_handle` must point to a valid `SglangStreamHandle`
/// - Output pointers must be writable
///
/// # Notes
///
/// - Complete messages are identified by the presence of `proto::GenerateResponse::Complete`
/// - When is_done=1, this may be the last readable chunk or the stream may be ending
/// - Subsequent calls after is_done=1 will mark the stream as complete internally
#[no_mangle]
pub unsafe extern "C" fn sgl_stream_read_next(
    stream_handle: *mut SglangStreamHandle,
    response_json_out: *mut *mut c_char,
    is_done_out: *mut c_int,
    error_out: *mut *mut c_char,
) -> SglErrorCode {
    if stream_handle.is_null() || response_json_out.is_null() || is_done_out.is_null() {
        set_error_message(error_out, "Invalid arguments: null pointer");
        return SglErrorCode::InvalidArgument;
    }

    let handle_ref = &*stream_handle;
    let stream = Arc::clone(&handle_ref.stream);
    let converter = Arc::clone(&handle_ref.converter);

    // Read next chunk from stream
    let chunk_result = RUNTIME.block_on(async {
        let mut stream_guard = stream.lock().await;
        stream_guard.next().await
    });

    match chunk_result {
        Some(Ok(proto_response)) => {
            // Convert proto response to OpenAI format
            // We need to get the converter lock first
            let conversion_result = RUNTIME.block_on(async {
                let mut converter_guard = converter.lock().await;

                // Clone necessary fields for conversion
                let tokenizer = Arc::clone(&converter_guard.tokenizer);
                let model = converter_guard.model.clone();
                let request_id = converter_guard.request_id.clone();
                let created = converter_guard.created;
                let system_fingerprint = converter_guard.system_fingerprint.clone();

                // Call the conversion function
                convert_proto_chunk_to_openai(
                    proto_response.clone(),
                    &mut *converter_guard,
                    &tokenizer,
                    &model,
                    &request_id,
                    created,
                    system_fingerprint.as_deref(),
                )
                .await
            });

            match conversion_result {
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

                    // Check if this is a complete response (stream done)
                    let is_complete = matches!(proto_response.response, Some(proto::generate_response::Response::Complete(_)) | Some(proto::generate_response::Response::Error(_)));

                    *response_json_out = result_cstr.into_raw();
                    *is_done_out = if is_complete { 1 } else { 0 };

                    if is_complete {
                        // Mark stream as completed
                        // Ensure mark_completed() completes and is visible before returning
                        // Use yield_now to ensure Release ordering is fully propagated
                        RUNTIME.block_on(async {
                            let stream_guard = stream.lock().await;
                            stream_guard.mark_completed();
                            // Keep the guard until mark_completed() is fully executed
                            drop(stream_guard);
                            // Yield to ensure Release ordering is propagated before returning
                            // This prevents race condition where Free() is called immediately
                            // and Drop might not see the mark_completed() write
                            tokio::task::yield_now().await;
                        });
                    }

                    SglErrorCode::Success
                }
                Ok(None) => {
                    // No response to send (e.g., empty chunk)
                    // Don't mark as completed - stream might continue
                    // Just return null and let caller read more
                    *response_json_out = ptr::null_mut();
                    *is_done_out = 0; // Keep stream open, not done yet
                    SglErrorCode::Success
                }
                Err(e) => {
                    // Conversion error - don't mark as completed
                    // Let the stream end naturally or return error without stopping stream
                    set_error_message(error_out, &format!("Conversion error: {}", e));
                    *response_json_out = ptr::null_mut();
                    *is_done_out = 0; // Don't mark as done - let caller decide
                    SglErrorCode::ParsingError
                }
            }
        }
        Some(Err(e)) => {
            // Stream error - mark as completed to prevent abort
            RUNTIME.block_on(async {
                let stream_guard = stream.lock().await;
                stream_guard.mark_completed();
                drop(stream_guard);
                // Yield to ensure Release ordering is propagated
                tokio::task::yield_now().await;
            });

            set_error_message(error_out, &format!("Stream error: {}", e));
            *is_done_out = 1;
            SglErrorCode::UnknownError
        }
        None => {
            // Stream ended naturally (no more chunks)
            // Mark stream as completed before returning to prevent abort
            RUNTIME.block_on(async {
                let stream_guard = stream.lock().await;
                stream_guard.mark_completed();
                drop(stream_guard);
                // Yield to ensure Release ordering is propagated
                tokio::task::yield_now().await;
            });

            *response_json_out = ptr::null_mut();
            *is_done_out = 1;
            SglErrorCode::Success
        }
    }
}

/// Free a stream handle and release all associated resources.
///
/// This function must be called exactly once for each stream handle returned by
/// `sgl_client_chat_completion_stream`. It marks the stream as completed internally
/// to prevent abort signals from being sent when resources are cleaned up.
///
/// # Arguments
///
/// * `handle` - Mutable pointer to the stream handle to free
///   - If NULL, this function does nothing
///
/// # Safety
///
/// - Must be called only once per handle
/// - Handle must not be used after calling this function
/// - After this call, the stream is no longer valid
///
/// # Notes
///
/// - This function internally calls `mark_completed()` before freeing to ensure
///   the stream cleanup doesn't trigger an abort RPC to the server
/// - Memory fences are used to ensure visibility across threads
#[no_mangle]
pub unsafe extern "C" fn sgl_stream_free(handle: *mut SglangStreamHandle) {
    if !handle.is_null() {
        let handle_ref = Box::from_raw(handle);

        // Mark stream as completed to prevent abort on drop
        // By this point, the stream should already be completed by ReadNext()
        // but we call it again to be safe
        RUNTIME.block_on(async {
            let stream_guard = handle_ref.stream.lock().await;
            stream_guard.mark_completed();
            // Keep guard alive to ensure mark_completed() write completes
            drop(stream_guard);
            // Yield to ensure the atomic write is visible
            tokio::task::yield_now().await;
        });

        // Use a strong memory fence to ensure mark_completed()'s Release write
        // is visible before we drop the last Arc reference
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

        // Now drop all references - if mark_completed() was called successfully,
        // the drop won't send an abort
        drop(handle_ref.stream);

        // Free converter
        let converter = Arc::try_unwrap(handle_ref.converter)
            .ok()
            .map(|m| m.into_inner());
        if let Some(conv) = converter {
            super::grpc_converter::sgl_grpc_response_converter_free(Box::into_raw(Box::new(conv)));
        }
    }
}
