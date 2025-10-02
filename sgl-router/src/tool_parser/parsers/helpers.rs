use crate::protocols::spec::Tool;
use serde_json::Value;
use std::collections::HashMap;

/// Get a mapping of tool names to their indices
pub fn get_tool_indices(tools: &[Tool]) -> HashMap<String, usize> {
    tools
        .iter()
        .enumerate()
        .map(|(i, tool)| (tool.function.name.clone(), i))
        .collect()
}

/// Check if a buffer ends with a partial occurrence of a token
pub fn ends_with_partial_token(buffer: &str, token: &str) -> bool {
    if buffer.is_empty() || token.is_empty() {
        return false;
    }

    for i in 1..token.len() {
        if buffer.ends_with(&token[..i]) {
            return true;
        }
    }
    false
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ends_with_partial_token() {
        assert!(ends_with_partial_token("hello <|py", "<|python_tag|>"));
        assert!(ends_with_partial_token(
            "hello <|python_tag",
            "<|python_tag|>"
        ));
        assert!(!ends_with_partial_token(
            "hello <|python_tag|>",
            "<|python_tag|>"
        ));
        assert!(!ends_with_partial_token("", "<|python_tag|>"));
        assert!(!ends_with_partial_token("hello world", "<|python_tag|>"));
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
