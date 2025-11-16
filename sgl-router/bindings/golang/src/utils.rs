//! Utility functions for FFI

use uuid::Uuid;

/// Helper function to generate tool call ID (matches router implementation)
pub fn generate_tool_call_id(
    model: &str,
    function_name: &str,
    index: usize,
    history_tool_calls_count: usize,
) -> String {
    if model.to_lowercase().contains("kimi") {
        // KimiK2 format: functions.{name}:{global_index}
        format!("functions.{}:{}", function_name, history_tool_calls_count + index)
    } else {
        // Standard OpenAI format: call_{24-char-uuid}
        format!("call_{}", &Uuid::new_v4().simple().to_string()[..24])
    }
}

/// Convert proto::GenerateResponse to JSON Value (since prost types don't support serde)
#[allow(dead_code)]
pub fn proto_response_to_json(response: &sglang_router::grpc_client::sglang_proto::GenerateResponse) -> String {
    use serde_json::{Value, Map};
    
    let mut json_obj = Map::new();
    json_obj.insert("request_id".to_string(), Value::String(response.request_id.clone()));
    
    match &response.response {
        Some(sglang_router::grpc_client::sglang_proto::generate_response::Response::Chunk(chunk)) => {
            let mut chunk_obj = Map::new();
            chunk_obj.insert("token_ids".to_string(), 
                Value::Array(chunk.token_ids.iter().map(|&id| Value::Number(id.into())).collect()));
            chunk_obj.insert("prompt_tokens".to_string(), Value::Number(chunk.prompt_tokens.into()));
            chunk_obj.insert("completion_tokens".to_string(), Value::Number(chunk.completion_tokens.into()));
            chunk_obj.insert("cached_tokens".to_string(), Value::Number(chunk.cached_tokens.into()));
            chunk_obj.insert("index".to_string(), Value::Number(chunk.index.into()));
            json_obj.insert("chunk".to_string(), Value::Object(chunk_obj));
        }
        Some(sglang_router::grpc_client::sglang_proto::generate_response::Response::Complete(complete)) => {
            let mut complete_obj = Map::new();
            complete_obj.insert("output_ids".to_string(), 
                Value::Array(complete.output_ids.iter().map(|&id| Value::Number(id.into())).collect()));
            complete_obj.insert("finish_reason".to_string(), Value::String(complete.finish_reason.clone()));
            complete_obj.insert("prompt_tokens".to_string(), Value::Number(complete.prompt_tokens.into()));
            complete_obj.insert("completion_tokens".to_string(), Value::Number(complete.completion_tokens.into()));
            complete_obj.insert("cached_tokens".to_string(), Value::Number(complete.cached_tokens.into()));
            complete_obj.insert("index".to_string(), Value::Number(complete.index.into()));
            json_obj.insert("complete".to_string(), Value::Object(complete_obj));
        }
        Some(sglang_router::grpc_client::sglang_proto::generate_response::Response::Error(err)) => {
            let mut error_obj = Map::new();
            error_obj.insert("message".to_string(), Value::String(err.message.clone()));
            error_obj.insert("http_status_code".to_string(), Value::String(err.http_status_code.clone()));
            error_obj.insert("details".to_string(), Value::String(err.details.clone()));
            json_obj.insert("error".to_string(), Value::Object(error_obj));
        }
        None => {}
    }
    
    serde_json::to_string(&Value::Object(json_obj)).unwrap_or_default()
}

/// Generate tool constraints (placeholder implementation)
///
/// # Arguments
/// * `tools_json` - JSON array of tools
/// * `tool_choice_json` - JSON object representing tool_choice
/// * `constraint_type_out` - Pointer to receive constraint type (e.g., "json_schema")
/// * `constraint_schema_out` - Pointer to receive constraint schema JSON
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * SglErrorCode::Success on success, error code on failure
#[no_mangle]
pub unsafe extern "C" fn sgl_generate_tool_constraints(
    _tools_json: *const std::os::raw::c_char,
    _tool_choice_json: *const std::os::raw::c_char,
    _constraint_type_out: *mut *mut std::os::raw::c_char,
    _constraint_schema_out: *mut *mut std::os::raw::c_char,
    error_out: *mut *mut std::os::raw::c_char,
) -> super::error::SglErrorCode {
    // Implementation would parse JSON and call generate_tool_constraints
    // This is a placeholder
    super::error::set_error_message(error_out, "Tool constraint generation not yet implemented in FFI");
    super::error::SglErrorCode::UnknownError
}

