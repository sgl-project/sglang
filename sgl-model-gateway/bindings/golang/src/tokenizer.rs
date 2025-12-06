//! Tokenizer FFI functions

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::ptr;
use std::sync::Arc;
use serde_json::Value;

use sgl_model_gateway::tokenizer::{
    create_tokenizer_from_file,
    traits::Tokenizer as TokenizerTrait,
    chat_template::ChatTemplateParams,
    huggingface::HuggingFaceTokenizer,
};

use super::error::{SglErrorCode, set_error_message, clear_error_message};

/// Opaque handle for a tokenizer instance
#[repr(C)]
pub struct TokenizerHandle {
    pub(crate) tokenizer: Arc<dyn TokenizerTrait>,
}

/// Create a tokenizer from a file path
///
/// # Arguments
/// * `path` - Path to tokenizer.json file (null-terminated C string)
/// * `error_out` - Optional pointer to receive error message (must be freed with sgl_free_string)
///
/// # Returns
/// * Pointer to TokenizerHandle on success, null on failure
///
/// # Safety
/// The returned handle must be freed with `sgl_tokenizer_free`.
#[no_mangle]
pub unsafe extern "C" fn sgl_tokenizer_create_from_file(
    path: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut TokenizerHandle {
    if path.is_null() {
        set_error_message(error_out, "path cannot be null");
        return ptr::null_mut();
    }

    let path_str = match CStr::from_ptr(path).to_str() {
        Ok(s) => s,
        Err(e) => {
            set_error_message(error_out, &format!("Invalid UTF-8 in path: {}", e));
            return ptr::null_mut();
        }
    };

    match create_tokenizer_from_file(path_str) {
        Ok(tokenizer) => {
            clear_error_message(error_out);
            Box::into_raw(Box::new(TokenizerHandle {
                tokenizer,
            }))
        }
        Err(e) => {
            set_error_message(error_out, &e.to_string());
            ptr::null_mut()
        }
    }
}

/// Encode text to token IDs
///
/// # Arguments
/// * `handle` - Tokenizer handle (must not be null)
/// * `text` - Input text (null-terminated C string)
/// * `token_ids_out` - Pointer to receive array of token IDs (must be freed with sgl_free_token_ids)
/// * `token_count_out` - Pointer to receive token count
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * SglErrorCode::Success on success, error code on failure
///
/// # Safety
/// The token_ids_out array must be freed with sgl_free_token_ids() after use.
#[no_mangle]
pub unsafe extern "C" fn sgl_tokenizer_encode(
    handle: *mut TokenizerHandle,
    text: *const c_char,
    token_ids_out: *mut *mut u32,
    token_count_out: *mut usize,
    error_out: *mut *mut c_char,
) -> SglErrorCode {
    if handle.is_null() || text.is_null() || token_ids_out.is_null() || token_count_out.is_null() {
        set_error_message(error_out, "Invalid arguments: null pointer");
        return SglErrorCode::InvalidArgument;
    }

    let text_str = match CStr::from_ptr(text).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in text");
            return SglErrorCode::InvalidArgument;
        }
    };

    let tokenizer = &(*handle).tokenizer;
    match tokenizer.encode(text_str) {
        Ok(encoding) => {
            let token_ids = encoding.token_ids();
            let count = token_ids.len();

            // Allocate memory for token IDs using Vec, then leak to give ownership to C
            let vec = token_ids.to_vec();
            let ptr = vec.as_ptr() as *mut u32;
            let _ = std::mem::ManuallyDrop::new(vec);

            *token_ids_out = ptr;
            *token_count_out = count;
            clear_error_message(error_out);
            SglErrorCode::Success
        }
        Err(e) => {
            set_error_message(error_out, &e.to_string());
            SglErrorCode::TokenizationError
        }
    }
}

/// Apply chat template to messages with tools support
///
/// # Arguments
/// * `handle` - Tokenizer handle
/// * `messages_json` - JSON string of messages array
/// * `tools_json` - Optional JSON string of tools array (null or empty string for no tools)
/// * `result_out` - Pointer to receive result string (must be freed with sgl_free_string)
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * SglErrorCode::Success on success, error code on failure
#[no_mangle]
pub unsafe extern "C" fn sgl_tokenizer_apply_chat_template_with_tools(
    handle: *mut TokenizerHandle,
    messages_json: *const c_char,
    tools_json: *const c_char,
    result_out: *mut *mut c_char,
    error_out: *mut *mut c_char,
) -> SglErrorCode {
    if handle.is_null() || messages_json.is_null() || result_out.is_null() {
        set_error_message(error_out, "Invalid arguments: null pointer");
        return SglErrorCode::InvalidArgument;
    }

    let messages_str = match CStr::from_ptr(messages_json).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in messages_json");
            return SglErrorCode::InvalidArgument;
        }
    };

    // Parse JSON messages
    let messages: Vec<Value> = match serde_json::from_str(messages_str) {
        Ok(msgs) => msgs,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to parse messages JSON: {}", e));
            return SglErrorCode::InvalidArgument;
        }
    };

    // Parse tools JSON if provided
    let tools: Option<Vec<Value>> = if tools_json.is_null() {
        None
    } else {
        let tools_str = match CStr::from_ptr(tools_json).to_str() {
            Ok(s) => {
                if s.is_empty() {
                    None
                } else {
                    match serde_json::from_str::<Vec<Value>>(s) {
                        Ok(t) => Some(t),
                        Err(e) => {
                            set_error_message(error_out, &format!("Failed to parse tools JSON: {}", e));
                            return SglErrorCode::InvalidArgument;
                        }
                    }
                }
            }
            Err(_) => {
                set_error_message(error_out, "Invalid UTF-8 in tools_json");
                return SglErrorCode::InvalidArgument;
            }
        };
        tools_str
    };

    // Get the tokenizer from handle
    let handle_ref = &*handle;
    let tokenizer = &handle_ref.tokenizer;

    // Try to downcast to HuggingFaceTokenizer
    if let Some(hf_tokenizer) = tokenizer.as_any().downcast_ref::<HuggingFaceTokenizer>() {
        // Apply chat template with tools
        let empty_docs: [Value; 0] = [];
        let tools_slice = tools.as_ref().map(|t| t.as_slice());
        let params = ChatTemplateParams {
            add_generation_prompt: true,
            tools: tools_slice,
            documents: Some(&empty_docs),
            template_kwargs: None,
        };

        match hf_tokenizer.apply_chat_template(&messages, params) {
            Ok(result) => {
                let result_cstr = match CString::new(result) {
                    Ok(s) => s,
                    Err(e) => {
                        set_error_message(error_out, &format!("Failed to create result string: {}", e));
                        return SglErrorCode::MemoryError;
                    }
                };
                *result_out = result_cstr.into_raw();
                clear_error_message(error_out);
                SglErrorCode::Success
            }
            Err(e) => {
                set_error_message(error_out, &format!("Failed to apply chat template: {}", e));
                SglErrorCode::TokenizationError
            }
        }
    } else {
        set_error_message(error_out, "Chat template is only supported for HuggingFace tokenizers");
        SglErrorCode::TokenizationError
    }
}

/// Apply chat template to messages
///
/// # Arguments
/// * `handle` - Tokenizer handle
/// * `messages_json` - JSON string of messages array
/// * `result_out` - Pointer to receive result string (must be freed with sgl_free_string)
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * SglErrorCode::Success on success, error code on failure
#[no_mangle]
pub unsafe extern "C" fn sgl_tokenizer_apply_chat_template(
    handle: *mut TokenizerHandle,
    messages_json: *const c_char,
    result_out: *mut *mut c_char,
    error_out: *mut *mut c_char,
) -> SglErrorCode {
    if handle.is_null() || messages_json.is_null() || result_out.is_null() {
        set_error_message(error_out, "Invalid arguments: null pointer");
        return SglErrorCode::InvalidArgument;
    }

    let messages_str = match CStr::from_ptr(messages_json).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in messages_json");
            return SglErrorCode::InvalidArgument;
        }
    };

    // Parse JSON messages
    let messages: Vec<Value> = match serde_json::from_str(messages_str) {
        Ok(msgs) => msgs,
        Err(e) => {
            set_error_message(error_out, &format!("Failed to parse messages JSON: {}", e));
            return SglErrorCode::InvalidArgument;
        }
    };

    // Get the tokenizer from handle
    let handle_ref = &*handle;
    let tokenizer = &handle_ref.tokenizer;

    // Try to downcast to HuggingFaceTokenizer
    if let Some(hf_tokenizer) = tokenizer.as_any().downcast_ref::<HuggingFaceTokenizer>() {
        // Apply chat template with default parameters
        // Use empty arrays instead of None to avoid template errors
        // Set add_generation_prompt to true so the model knows to start generating
        let empty_tools: [Value; 0] = [];
        let empty_docs: [Value; 0] = [];
        let params = ChatTemplateParams {
            add_generation_prompt: true,  // Important: tells the model to start generating
            tools: Some(&empty_tools),
            documents: Some(&empty_docs),
            template_kwargs: None,
        };

        match hf_tokenizer.apply_chat_template(&messages, params) {
            Ok(result) => {
                let result_cstr = match CString::new(result) {
                    Ok(s) => s,
                    Err(e) => {
                        set_error_message(error_out, &format!("Failed to create result string: {}", e));
                        return SglErrorCode::MemoryError;
                    }
                };
                *result_out = result_cstr.into_raw();
                clear_error_message(error_out);
                SglErrorCode::Success
            }
            Err(e) => {
                set_error_message(error_out, &format!("Failed to apply chat template: {}", e));
                SglErrorCode::TokenizationError
            }
        }
    } else {
        set_error_message(error_out, "Chat template is only supported for HuggingFace tokenizers");
        SglErrorCode::TokenizationError
    }
}

/// Decode token IDs to text
///
/// # Arguments
/// * `handle` - Tokenizer handle
/// * `token_ids` - Array of token IDs
/// * `token_count` - Number of tokens
/// * `skip_special_tokens` - Whether to skip special tokens
/// * `result_out` - Pointer to receive result string (must be freed with sgl_free_string)
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * SglErrorCode::Success on success, error code on failure
#[no_mangle]
pub unsafe extern "C" fn sgl_tokenizer_decode(
    handle: *mut TokenizerHandle,
    token_ids: *const u32,
    token_count: usize,
    skip_special_tokens: c_int,
    result_out: *mut *mut c_char,
    error_out: *mut *mut c_char,
) -> SglErrorCode {
    if handle.is_null() || token_ids.is_null() || result_out.is_null() {
        set_error_message(error_out, "Invalid arguments: null pointer");
        return SglErrorCode::InvalidArgument;
    }

    if token_count == 0 {
        let empty = CString::new("").unwrap();
        *result_out = empty.into_raw();
        clear_error_message(error_out);
        return SglErrorCode::Success;
    }

    // Convert C array to Rust slice
    let token_slice = std::slice::from_raw_parts(token_ids, token_count);

    let tokenizer = &(*handle).tokenizer;
    match tokenizer.decode(token_slice, skip_special_tokens != 0) {
        Ok(text) => {
            let result_cstr = match CString::new(text) {
                Ok(s) => s,
                Err(e) => {
                    set_error_message(error_out, &format!("Failed to create result string: {}", e));
                    return SglErrorCode::MemoryError;
                }
            };
            *result_out = result_cstr.into_raw();
            clear_error_message(error_out);
            SglErrorCode::Success
        }
        Err(e) => {
            set_error_message(error_out, &e.to_string());
            SglErrorCode::TokenizationError
        }
    }
}

/// Free a tokenizer handle
///
/// # Safety
/// This function must only be called once per handle, and the handle must not be used after calling.
#[no_mangle]
pub unsafe extern "C" fn sgl_tokenizer_free(handle: *mut TokenizerHandle) {
    if !handle.is_null() {
        let _ = Box::from_raw(handle);
    }
}
