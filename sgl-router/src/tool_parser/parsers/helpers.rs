use crate::protocols::spec::Tool;
use serde_json::Value;
use std::collections::HashMap;

use crate::tool_parser::errors::{ToolParserError, ToolParserResult};
use crate::tool_parser::types::{StreamingParseResult, ToolCallItem};

/// Get a mapping of tool names to their indices
pub fn get_tool_indices(tools: &[Tool]) -> HashMap<String, usize> {
    tools
        .iter()
        .enumerate()
        .map(|(i, tool)| (tool.function.name.clone(), i))
        .collect()
}

/// Check if a buffer ends with a partial occurrence of a token
/// Returns Some(length) if there's a partial match, None otherwise
pub fn ends_with_partial_token(buffer: &str, token: &str) -> Option<usize> {
    if buffer.is_empty() || token.is_empty() {
        return None;
    }

    (1..token.len()).find(|&i| buffer.ends_with(&token[..i]))
}

/// Reset state for the current tool being parsed (used when skipping invalid tools).
/// This preserves the parser's overall state (current_tool_id, prev_tool_call_arr)
/// but clears the state specific to the current incomplete tool.
pub fn reset_current_tool_state(
    buffer: &mut String,
    current_tool_name_sent: &mut bool,
    streamed_args_for_tool: &mut Vec<String>,
    prev_tool_call_arr: &[Value],
) {
    buffer.clear();
    *current_tool_name_sent = false;

    // Only pop if we added an entry for the current (invalid) tool
    // streamed_args_for_tool should match prev_tool_call_arr length for completed tools
    if streamed_args_for_tool.len() > prev_tool_call_arr.len() {
        streamed_args_for_tool.pop();
    }
}

/// Reset the entire parser state (used at the start of a new request).
/// Clears all accumulated tool calls and resets all state to initial values.
pub fn reset_parser_state(
    buffer: &mut String,
    prev_tool_call_arr: &mut Vec<Value>,
    current_tool_id: &mut i32,
    current_tool_name_sent: &mut bool,
    streamed_args_for_tool: &mut Vec<String>,
) {
    buffer.clear();
    prev_tool_call_arr.clear();
    *current_tool_id = 0;
    *current_tool_name_sent = false;
    streamed_args_for_tool.clear();
}

/// Ensure arrays have capacity for the given tool ID
pub fn ensure_capacity(
    current_tool_id: i32,
    prev_tool_call_arr: &mut Vec<Value>,
    streamed_args_for_tool: &mut Vec<String>,
) {
    if current_tool_id < 0 {
        return;
    }
    let needed = (current_tool_id + 1) as usize;

    if prev_tool_call_arr.len() < needed {
        prev_tool_call_arr.resize_with(needed, || Value::Null);
    }
    if streamed_args_for_tool.len() < needed {
        streamed_args_for_tool.resize_with(needed, String::new);
    }
}

/// Check if a string contains complete, valid JSON
pub fn is_complete_json(input: &str) -> bool {
    serde_json::from_str::<Value>(input).is_ok()
}

/// Normalize the arguments/parameters field in a tool call object.
/// If the object has "parameters" but not "arguments", copy parameters to arguments.
///
/// # Background
/// Different LLM formats use different field names:
/// - Llama and JSON parsers use "parameters" (correct per JSON Schema spec)
/// - Mistral and Qwen use "arguments"
///
/// This function normalizes to "arguments" for consistent downstream processing.
pub fn normalize_arguments_field(mut obj: Value) -> Value {
    if obj.get("arguments").is_none() {
        if let Some(params) = obj.get("parameters").cloned() {
            if let Value::Object(ref mut map) = obj {
                map.insert("arguments".to_string(), params);
            }
        }
    }
    obj
}

/// Handle the entire JSON tool call streaming process for JSON-based parsers.
///
/// This unified function handles all aspects of streaming tool calls:
/// - Parsing partial JSON from the buffer
/// - Validating tool names against available tools
/// - Streaming tool names (Case 1)
/// - Streaming tool arguments (Case 2)
/// - Managing parser state and buffer updates
///
/// Used by JSON, Llama, Mistral, and Qwen parsers.
///
/// # Parameters
/// - `current_text`: The current buffered text being parsed
/// - `start_idx`: Start index of JSON content in current_text
/// - `partial_json`: Mutable reference to partial JSON parser
/// - `tool_indices`: Map of valid tool names to their indices
/// - `buffer`: Mutable parser buffer
/// - `current_tool_id`: Mutable current tool index (-1 means no active tool)
/// - `current_tool_name_sent`: Mutable flag for whether current tool's name was sent
/// - `streamed_args_for_tool`: Mutable accumulator of streamed arguments per tool
/// - `prev_tool_call_arr`: Mutable array of previous tool call states
///
/// # Returns
/// - `Ok(StreamingParseResult)` with any tool call items to stream
/// - `Err(ToolParserError)` if JSON parsing or serialization fails
#[allow(clippy::too_many_arguments)]
pub fn handle_json_tool_streaming(
    current_text: &str,
    start_idx: usize,
    partial_json: &mut crate::tool_parser::partial_json::PartialJson,
    tool_indices: &HashMap<String, usize>,
    buffer: &mut String,
    current_tool_id: &mut i32,
    current_tool_name_sent: &mut bool,
    streamed_args_for_tool: &mut Vec<String>,
    prev_tool_call_arr: &mut Vec<Value>,
) -> ToolParserResult<StreamingParseResult> {
    // Check if we have content to parse
    if start_idx >= current_text.len() {
        return Ok(StreamingParseResult::default());
    }

    // Extract JSON string from current position
    let json_str = &current_text[start_idx..];

    // Parse partial JSON
    let (obj, end_idx) = match partial_json.parse_value(json_str) {
        Ok(result) => result,
        Err(_) => {
            return Ok(StreamingParseResult::default());
        }
    };

    // Check if JSON is complete
    let is_complete = end_idx == json_str.len() && serde_json::from_str::<Value>(json_str).is_ok();

    // Validate tool name if present
    if let Some(name) = obj.get("name").and_then(|v| v.as_str()) {
        if !tool_indices.contains_key(name) {
            // Invalid tool name - skip this tool, preserve indexing for next tool
            tracing::warn!("Invalid tool name '{}' - skipping", name);
            reset_current_tool_state(
                buffer,
                current_tool_name_sent,
                streamed_args_for_tool,
                prev_tool_call_arr,
            );
            return Ok(StreamingParseResult::default());
        }
    }

    // Normalize parameters/arguments field
    let current_tool_call = normalize_arguments_field(obj);

    let mut result = StreamingParseResult::default();

    // Case 1: Handle tool name streaming
    if !*current_tool_name_sent {
        if let Some(function_name) = current_tool_call.get("name").and_then(|v| v.as_str()) {
            if tool_indices.contains_key(function_name) {
                // Initialize if first tool
                if *current_tool_id == -1 {
                    *current_tool_id = 0;
                    streamed_args_for_tool.push(String::new());
                } else if *current_tool_id as usize >= streamed_args_for_tool.len() {
                    // Ensure capacity for subsequent tools
                    ensure_capacity(*current_tool_id, prev_tool_call_arr, streamed_args_for_tool);
                }

                // Send tool name with empty parameters
                *current_tool_name_sent = true;
                result.calls.push(ToolCallItem {
                    tool_index: *current_tool_id as usize,
                    name: Some(function_name.to_string()),
                    parameters: String::new(),
                });
            }
        }
    }
    // Case 2: Handle streaming arguments
    else if let Some(cur_arguments) = current_tool_call.get("arguments") {
        let tool_id = *current_tool_id as usize;
        let sent = streamed_args_for_tool
            .get(tool_id)
            .map(|s| s.len())
            .unwrap_or(0);
        let cur_args_json = serde_json::to_string(cur_arguments)
            .map_err(|e| ToolParserError::ParsingFailed(e.to_string()))?;

        // Compute diff: everything after what we've already sent
        let diff = cur_args_json[sent..].to_string();

        // Send diff if there's new content
        if !diff.is_empty() {
            // Only accumulate if not complete
            if !is_complete && tool_id < streamed_args_for_tool.len() {
                streamed_args_for_tool[tool_id].push_str(&diff);
            }

            result.calls.push(ToolCallItem {
                tool_index: tool_id,
                name: None,
                parameters: diff,
            });
        }

        // If JSON is complete, advance to next tool
        if is_complete {
            // Remove processed portion, keep unprocessed content
            *buffer = current_text[start_idx + end_idx..].to_string();

            // Clear completed tool data
            if tool_id < prev_tool_call_arr.len() {
                prev_tool_call_arr[tool_id] = Value::Null;
            }
            *current_tool_name_sent = false;
            if tool_id < streamed_args_for_tool.len() {
                streamed_args_for_tool[tool_id].clear();
            }
            *current_tool_id += 1;
        }
    }

    // Update prev_tool_call_arr with current state
    if *current_tool_id >= 0 {
        ensure_capacity(*current_tool_id, prev_tool_call_arr, streamed_args_for_tool);
        let tool_id = *current_tool_id as usize;

        if tool_id < prev_tool_call_arr.len() {
            prev_tool_call_arr[tool_id] = current_tool_call;
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ends_with_partial_token() {
        assert!(ends_with_partial_token("hello <|py", "<|python_tag|>").is_some());
        assert!(ends_with_partial_token("hello <|python_tag", "<|python_tag|>").is_some());
        assert!(ends_with_partial_token("hello <|python_tag|>", "<|python_tag|>").is_none());
        assert!(ends_with_partial_token("", "<|python_tag|>").is_none());
        assert!(ends_with_partial_token("hello world", "<|python_tag|>").is_none());
    }

    #[test]
    fn test_reset_current_tool_state() {
        let mut buffer = String::from("partial json");
        let mut current_tool_name_sent = true;
        let mut streamed_args = vec!["tool0_args".to_string(), "tool1_partial".to_string()];
        let prev_tools = vec![serde_json::json!({"name": "tool0"})];

        reset_current_tool_state(
            &mut buffer,
            &mut current_tool_name_sent,
            &mut streamed_args,
            &prev_tools,
        );

        assert_eq!(buffer, "");
        assert!(!current_tool_name_sent);
        assert_eq!(streamed_args.len(), 1); // Popped the partial tool1 args
        assert_eq!(streamed_args[0], "tool0_args");
    }

    #[test]
    fn test_reset_current_tool_state_no_pop_when_synced() {
        let mut buffer = String::from("partial json");
        let mut current_tool_name_sent = true;
        let mut streamed_args = vec!["tool0_args".to_string()];
        let prev_tools = vec![serde_json::json!({"name": "tool0"})];

        reset_current_tool_state(
            &mut buffer,
            &mut current_tool_name_sent,
            &mut streamed_args,
            &prev_tools,
        );

        assert_eq!(buffer, "");
        assert!(!current_tool_name_sent);
        assert_eq!(streamed_args.len(), 1); // No pop, lengths matched
    }

    #[test]
    fn test_reset_parser_state() {
        let mut buffer = String::from("some buffer");
        let mut prev_tools = vec![serde_json::json!({"name": "tool0"})];
        let mut current_tool_id = 5;
        let mut current_tool_name_sent = true;
        let mut streamed_args = vec!["args".to_string()];

        reset_parser_state(
            &mut buffer,
            &mut prev_tools,
            &mut current_tool_id,
            &mut current_tool_name_sent,
            &mut streamed_args,
        );

        assert_eq!(buffer, "");
        assert_eq!(prev_tools.len(), 0);
        assert_eq!(current_tool_id, 0);
        assert!(!current_tool_name_sent);
        assert_eq!(streamed_args.len(), 0);
    }

    #[test]
    fn test_ensure_capacity() {
        let mut prev_tools = vec![];
        let mut streamed_args = vec![];

        ensure_capacity(2, &mut prev_tools, &mut streamed_args);

        assert_eq!(prev_tools.len(), 3);
        assert_eq!(streamed_args.len(), 3);
        assert_eq!(prev_tools[0], Value::Null);
        assert_eq!(streamed_args[0], "");
    }

    #[test]
    fn test_ensure_capacity_negative_id() {
        let mut prev_tools = vec![];
        let mut streamed_args = vec![];

        ensure_capacity(-1, &mut prev_tools, &mut streamed_args);

        // Should not resize for negative ID
        assert_eq!(prev_tools.len(), 0);
        assert_eq!(streamed_args.len(), 0);
    }

    #[test]
    fn test_is_complete_json() {
        assert!(is_complete_json(r#"{"name": "test"}"#));
        assert!(is_complete_json("[1, 2, 3]"));
        assert!(is_complete_json("42"));
        assert!(is_complete_json("true"));
        assert!(!is_complete_json(r#"{"name": "#));
        assert!(!is_complete_json("[1, 2,"));
    }

    #[test]
    fn test_normalize_arguments_field() {
        // Case 1: Has parameters, no arguments
        let obj = serde_json::json!({
            "name": "test",
            "parameters": {"key": "value"}
        });
        let normalized = normalize_arguments_field(obj);
        assert_eq!(
            normalized.get("arguments").unwrap(),
            &serde_json::json!({"key": "value"})
        );

        // Case 2: Already has arguments
        let obj = serde_json::json!({
            "name": "test",
            "arguments": {"key": "value"}
        });
        let normalized = normalize_arguments_field(obj.clone());
        assert_eq!(normalized, obj);

        // Case 3: No parameters or arguments
        let obj = serde_json::json!({"name": "test"});
        let normalized = normalize_arguments_field(obj.clone());
        assert_eq!(normalized, obj);
    }
}
