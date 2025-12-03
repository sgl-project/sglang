//! Tool parser FFI functions

use std::ffi::{CStr, CString};
use std::os::raw::{c_char};
use std::ptr;
use std::sync::Arc;
use std::collections::HashMap;
use serde_json::{json, Value};
use tokio::runtime::Runtime;
use once_cell::sync::Lazy;

use sgl_model_gateway::tool_parser::{ParserFactory, ToolParser};
use sgl_model_gateway::protocols::common::Tool;

use super::error::{SglErrorCode, set_error_message, clear_error_message};
use super::utils::generate_tool_call_id;

/// Global parser factory (initialized once)
static PARSER_FACTORY: Lazy<ParserFactory> = Lazy::new(|| ParserFactory::new());

/// Global tokio runtime for async operations
static RUNTIME: Lazy<Runtime> = Lazy::new(|| {
    Runtime::new().expect("Failed to create tokio runtime for tool parser FFI")
});

/// Opaque handle for a tool parser instance
/// Note: For streaming, we need mutable access, so we use Arc<Mutex<>> internally
/// Note: This is an opaque handle, C code doesn't access fields directly
pub struct ToolParserHandle {
    parser: Arc<tokio::sync::Mutex<Box<dyn ToolParser>>>,
    model: String, // Store model name for ID generation
    history_tool_calls_count: usize, // Track tool call count for ID generation
    tool_index_to_id: HashMap<usize, String>, // Map tool_index to ID for incremental updates
}

/// Create a tool parser
///
/// # Arguments
/// * `parser_type` - Parser type name (e.g., "json", "llama", "mistral") or model name (e.g., "gpt-4")
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * Pointer to ToolParserHandle on success, null on failure
#[no_mangle]
pub unsafe extern "C" fn sgl_tool_parser_create(
    parser_type: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut ToolParserHandle {
    if parser_type.is_null() {
        set_error_message(error_out, "parser_type cannot be null");
        return ptr::null_mut();
    }

    let type_str = match CStr::from_ptr(parser_type).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in parser_type");
            return ptr::null_mut();
        }
    };

    // Create parser using factory
    // The factory will determine the parser type based on model name or use the provided type
    let parser = if let Some(parser_box) = PARSER_FACTORY.registry().create_for_model(type_str) {
        parser_box
    } else if let Some(parser_box) = PARSER_FACTORY.registry().create_parser(type_str) {
        parser_box
    } else {
        set_error_message(error_out, &format!("Unknown parser type: {}", type_str));
        return ptr::null_mut();
    };

    Box::into_raw(Box::new(ToolParserHandle {
        parser: Arc::new(tokio::sync::Mutex::new(parser)),
        model: type_str.to_string(),
        history_tool_calls_count: 0,
        tool_index_to_id: HashMap::new(),
    }))
}

/// Parse complete tool calls from text
///
/// # Arguments
/// * `handle` - Tool parser handle
/// * `text` - Input text to parse
/// * `result_json_out` - Pointer to receive JSON result (must be freed with sgl_free_string)
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * SglErrorCode::Success on success, error code on failure
#[no_mangle]
pub unsafe extern "C" fn sgl_tool_parser_parse_complete(
    handle: *mut ToolParserHandle,
    text: *const c_char,
    result_json_out: *mut *mut c_char,
    error_out: *mut *mut c_char,
) -> SglErrorCode {
    if handle.is_null() || text.is_null() || result_json_out.is_null() {
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

    let handle_ref = &*handle;
    let parser = Arc::clone(&handle_ref.parser);
    let model = handle_ref.model.clone();
    let history_count = handle_ref.history_tool_calls_count;

    // Use tokio runtime to run async code
    let result = RUNTIME.block_on(async {
        let parser_guard = parser.lock().await;
        parser_guard.parse_complete(text_str).await
    });

    match result {
        Ok((normal_text, tool_calls)) => {
            // Convert Rust ToolCall to OpenAI format
            let openai_tool_calls: Vec<Value> = tool_calls
                .into_iter()
                .enumerate()
                .map(|(index, tc)| {
                    // Generate ID for this tool call
                    let id = generate_tool_call_id(&model, &tc.function.name, index, history_count);
                    json!({
                        "id": id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    })
                })
                .collect();

            // Build result JSON
            let result_json = json!({
                "normal_text": normal_text,
                "tool_calls": openai_tool_calls
            });

            let result_str = match serde_json::to_string(&result_json) {
                Ok(s) => s,
                Err(e) => {
                    set_error_message(error_out, &format!("Failed to serialize JSON: {}", e));
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
        Err(e) => {
            set_error_message(error_out, &format!("Parse error: {}", e));
            SglErrorCode::ParsingError
        }
    }
}

/// Parse tool calls incrementally from streaming chunks
///
/// # Arguments
/// * `handle` - Tool parser handle
/// * `chunk` - New text chunk from stream
/// * `tools_json` - JSON array of available tools (for validation, can be null/empty)
/// * `result_json_out` - Pointer to receive JSON result (must be freed with sgl_free_string)
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * SglErrorCode::Success on success, error code on failure
#[no_mangle]
pub unsafe extern "C" fn sgl_tool_parser_parse_incremental(
    handle: *mut ToolParserHandle,
    chunk: *const c_char,
    tools_json: *const c_char,
    result_json_out: *mut *mut c_char,
    error_out: *mut *mut c_char,
) -> SglErrorCode {
    if handle.is_null() || chunk.is_null() || result_json_out.is_null() {
        set_error_message(error_out, "Invalid arguments: null pointer");
        return SglErrorCode::InvalidArgument;
    }

    let chunk_str = match CStr::from_ptr(chunk).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error_message(error_out, "Invalid UTF-8 in chunk");
            return SglErrorCode::InvalidArgument;
        }
    };

    // Parse tools JSON if provided
    let tools: Vec<Tool> = if !tools_json.is_null() {
        let tools_str = match CStr::from_ptr(tools_json).to_str() {
            Ok(s) => s,
            Err(_) => {
                set_error_message(error_out, "Invalid UTF-8 in tools_json");
                return SglErrorCode::InvalidArgument;
            }
        };
        match serde_json::from_str::<Vec<Tool>>(tools_str) {
            Ok(t) => t,
            Err(_) => vec![], // If parsing fails, use empty tools
        }
    } else {
        vec![]
    };

    let handle_ref = &*handle;
    let parser = Arc::clone(&handle_ref.parser);
    let model = handle_ref.model.clone();
    let history_count = handle_ref.history_tool_calls_count;

    // Use tokio runtime to run async code
    let result = RUNTIME.block_on(async {
        let mut parser_guard = parser.lock().await;
        parser_guard.parse_incremental(chunk_str, &tools).await
    });

    match result {
        Ok(streaming_result) => {
            // Convert StreamingParseResult to OpenAI format
            let handle_mut = &mut *handle;
            let openai_tool_calls: Vec<Value> = streaming_result
                .calls
                .into_iter()
                .map(|item| {
                    // For incremental parsing, we may not have complete tool calls yet
                    // Generate or reuse ID based on tool_index
                    let id = if let Some(ref name) = item.name {
                        // New tool call with name - generate ID and store it
                        let id = generate_tool_call_id(&model, name, item.tool_index, history_count);
                        handle_mut.tool_index_to_id.insert(item.tool_index, id.clone());
                        id
                    } else {
                        // Parameter update - reuse existing ID for this tool_index
                        handle_mut.tool_index_to_id
                            .get(&item.tool_index)
                            .cloned()
                            .unwrap_or_else(|| format!("call_{}", item.tool_index))
                    };

                    json!({
                        "id": id,
                        "type": "function",
                        "function": {
                            "name": item.name.unwrap_or_default(),
                            "arguments": item.parameters
                        }
                    })
                })
                .collect();

            // Build result JSON
            let result_json = json!({
                "normal_text": streaming_result.normal_text,
                "tool_calls": openai_tool_calls
            });

            let result_str = match serde_json::to_string(&result_json) {
                Ok(s) => s,
                Err(e) => {
                    set_error_message(error_out, &format!("Failed to serialize JSON: {}", e));
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
        Err(e) => {
            set_error_message(error_out, &format!("Parse incremental error: {}", e));
            SglErrorCode::ParsingError
        }
    }
}

/// Reset the parser state for reuse
#[no_mangle]
pub unsafe extern "C" fn sgl_tool_parser_reset(handle: *mut ToolParserHandle) {
    if handle.is_null() {
        return;
    }

    let handle_ref = &mut *handle;
    let parser = Arc::clone(&handle_ref.parser);

    // Reset parser state
    RUNTIME.block_on(async {
        let mut parser_guard = parser.lock().await;
        parser_guard.reset();
    });

    // Reset history count and tool index mapping
    handle_ref.history_tool_calls_count = 0;
    handle_ref.tool_index_to_id.clear();
}

/// Free a tool parser handle
#[no_mangle]
pub unsafe extern "C" fn sgl_tool_parser_free(handle: *mut ToolParserHandle) {
    if !handle.is_null() {
        let _ = Box::from_raw(handle);
    }
}
