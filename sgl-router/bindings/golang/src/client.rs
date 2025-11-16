//! Client SDK FFI functions

use std::ffi::{CStr, CString};
use std::os::raw::{c_char};
use std::ptr;
use std::sync::Arc;
use tokio::runtime::Runtime;
use once_cell::sync::Lazy;
use uuid::Uuid;

use sglang_router::tokenizer::create_tokenizer_from_file;
use sglang_router::tokenizer::traits::Tokenizer;
use sglang_router::grpc_client::sglang_scheduler::SglangSchedulerClient;
use sglang_router::protocols::chat::ChatCompletionRequest;
use sglang_router::routers::grpc::utils::{process_chat_messages, generate_tool_constraints};

use super::error::{SglErrorCode, set_error_message};
use super::grpc_converter::sgl_grpc_response_converter_create;
use super::tokenizer::TokenizerHandle;
use super::stream::SglangStreamHandle;

/// Global tokio runtime for async operations
static RUNTIME: Lazy<Runtime> = Lazy::new(|| {
    Runtime::new().expect("Failed to create tokio runtime for client FFI")
});

/// Handle for complete client SDK (gRPC client + tokenizer)
/// This handle manages the connection to sglang and provides a complete SDK interface
pub struct SglangClientHandle {
    pub(crate) client: Arc<SglangSchedulerClient>,
    pub(crate) tokenizer: Arc<dyn Tokenizer>,
}

/// Handle for streaming request (includes prompt token count)
#[allow(dead_code)]
pub struct StreamRequestState {
    pub(crate) prompt_tokens: i32, // Number of prompt tokens for this request
}

/// Create a new SGLang client handle
///
/// # Arguments
/// * `endpoint` - gRPC endpoint (e.g., "grpc://localhost:20000")
/// * `tokenizer_path` - Path to tokenizer directory
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * Pointer to SglangClientHandle on success, null on failure
#[no_mangle]
pub unsafe extern "C" fn sgl_client_create(
    endpoint: *const c_char,
    tokenizer_path: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut SglangClientHandle {
    if endpoint.is_null() || tokenizer_path.is_null() {
        set_error_message(error_out, "Invalid arguments: null pointer");
        return ptr::null_mut();
    }

    let endpoint_str = match CStr::from_ptr(endpoint).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in endpoint");
            return ptr::null_mut();
        }
    };

    let tokenizer_path_str = match CStr::from_ptr(tokenizer_path).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in tokenizer_path");
            return ptr::null_mut();
        }
    };

    // Create tokenizer
    let tokenizer = match create_tokenizer_from_file(tokenizer_path_str) {
        Ok(t) => t,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to create tokenizer: {}", e));
            return ptr::null_mut();
        }
    };

    // Create gRPC client
    let client = match RUNTIME.block_on(async {
        SglangSchedulerClient::connect(endpoint_str).await
    }) {
        Ok(c) => Arc::new(c),
        Err(e) => {
            set_error_message(error_out, &format!("Failed to connect to endpoint: {}", e));
            return ptr::null_mut();
        }
    };

    Box::into_raw(Box::new(SglangClientHandle {
        client,
        tokenizer,
    }))
}

/// Free a client handle
#[no_mangle]
pub unsafe extern "C" fn sgl_client_free(handle: *mut SglangClientHandle) {
    if !handle.is_null() {
        let _ = Box::from_raw(handle);
    }
}

/// Send a chat completion request and start streaming
///
/// # Arguments
/// * `client_handle` - Client handle
/// * `request_json` - OpenAI ChatCompletionRequest as JSON string
/// * `stream_handle_out` - Pointer to receive stream handle
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * SglErrorCode::Success on success, error code on failure
#[no_mangle]
pub unsafe extern "C" fn sgl_client_chat_completion_stream(
    client_handle: *mut SglangClientHandle,
    request_json: *const c_char,
    stream_handle_out: *mut *mut SglangStreamHandle,
    error_out: *mut *mut c_char,
) -> SglErrorCode {
    if client_handle.is_null() || request_json.is_null() || stream_handle_out.is_null() {
        set_error_message(error_out, "Invalid arguments: null pointer");
        return SglErrorCode::InvalidArgument;
    }

    let request_str = match CStr::from_ptr(request_json).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in request_json");
            return SglErrorCode::InvalidArgument;
        }
    };

    let client_ref = &*client_handle;
    let client = Arc::clone(&client_ref.client);
    let tokenizer = Arc::clone(&client_ref.tokenizer);

    // Parse OpenAI ChatCompletionRequest
    let chat_request: ChatCompletionRequest = match serde_json::from_str(request_str) {
        Ok(req) => req,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to parse request JSON: {}", e));
            return SglErrorCode::ParsingError;
        }
    };

    // Process messages and apply chat template
    let processed_messages = match process_chat_messages(&chat_request, tokenizer.as_ref()) {
        Ok(msgs) => msgs,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to process messages: {}", e));
            return SglErrorCode::TokenizationError;
        }
    };

    // Tokenize
    let token_ids = match tokenizer.encode(&processed_messages.text) {
        Ok(encoding) => encoding.token_ids().to_vec(),
        Err(e) => {
            set_error_message(error_out, &format!("Failed to tokenize: {}", e));
            return SglErrorCode::TokenizationError;
        }
    };
    let prompt_tokens = token_ids.len() as i32; // Save prompt token count

    // Generate tool constraints if needed
    let tool_constraint = if let Some(tools) = chat_request.tools.as_ref() {
        match generate_tool_constraints(tools, &chat_request.tool_choice, &chat_request.model) {
            Ok(Some((constraint_type, constraint_value))) => Some((constraint_type, constraint_value)),
            Ok(None) => None,
            Err(e) => {
                set_error_message(error_out, &format!("Failed to generate tool constraints: {}", e));
                return SglErrorCode::ParsingError;
            }
        }
    } else {
        None
    };

    // Build GenerateRequest
    let request_id = format!("chatcmpl-{}", Uuid::new_v4());
    let proto_request = match client.build_generate_request_from_chat(
        request_id.clone(),
        &chat_request,
        processed_messages.text,
        token_ids,
        processed_messages.multimodal_inputs,
        tool_constraint,
    ) {
        Ok(req) => req,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to build generate request: {}", e));
            return SglErrorCode::ParsingError;
        }
    };

    // Send request and get stream
    let stream = match RUNTIME.block_on(async {
        client.generate(proto_request).await
    }) {
        Ok(s) => s,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to send request: {}", e));
            return SglErrorCode::UnknownError;
        }
    };

    // Create response converter
    let tools_json = chat_request.tools.as_ref()
        .and_then(|t| serde_json::to_string(t).ok())
        .map(|s| CString::new(s).unwrap().into_raw());
    let tool_choice_json = chat_request.tool_choice.as_ref()
        .and_then(|tc| serde_json::to_string(tc).ok())
        .map(|s| CString::new(s).unwrap().into_raw());
    let stop_json = chat_request.stop.as_ref()
        .and_then(|s| serde_json::to_string(s).ok())
        .map(|s| CString::new(s).unwrap().into_raw());
    let stop_token_ids_json = chat_request.stop_token_ids.as_ref()
        .and_then(|ids| serde_json::to_string(ids).ok())
        .map(|s| CString::new(s).unwrap().into_raw());

    // Create tokenizer handle for converter (we'll create a temporary one)
    let tokenizer_handle = Box::into_raw(Box::new(TokenizerHandle {
        tokenizer: Arc::clone(&tokenizer),
    }));

    let converter = sgl_grpc_response_converter_create(
        tokenizer_handle,
        CString::new(chat_request.model.clone()).unwrap().as_ptr(),
        CString::new(request_id.clone()).unwrap().as_ptr(),
        tools_json.unwrap_or(ptr::null_mut()),
        tool_choice_json.unwrap_or(ptr::null_mut()),
        stop_json.unwrap_or(ptr::null_mut()),
        stop_token_ids_json.unwrap_or(ptr::null_mut()),
        if chat_request.skip_special_tokens { 1 } else { 0 },
        error_out,
    );

    // Free temporary tokenizer handle (converter now owns the tokenizer)
    let _ = Box::from_raw(tokenizer_handle);

    if converter.is_null() {
        return SglErrorCode::MemoryError;
    }

    // Clean up temporary CStrings
    if let Some(ptr) = tools_json {
        let _ = CString::from_raw(ptr);
    }
    if let Some(ptr) = tool_choice_json {
        let _ = CString::from_raw(ptr);
    }
    if let Some(ptr) = stop_json {
        let _ = CString::from_raw(ptr);
    }
    if let Some(ptr) = stop_token_ids_json {
        let _ = CString::from_raw(ptr);
    }

    // Create converter handle and set initial_prompt_tokens immediately
    let mut converter_handle = *Box::from_raw(converter);
    converter_handle.initial_prompt_tokens = Some(prompt_tokens);

    // Create stream handle with prompt_tokens
    *stream_handle_out = Box::into_raw(Box::new(SglangStreamHandle {
        stream: Arc::new(tokio::sync::Mutex::new(stream)),
        converter: Arc::new(tokio::sync::Mutex::new(converter_handle)),
        client: Arc::clone(&client),
        prompt_tokens,
    }));

    SglErrorCode::Success
}
