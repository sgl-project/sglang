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
