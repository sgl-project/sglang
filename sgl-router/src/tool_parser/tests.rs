use super::*;
use crate::tool_parser::partial_json::{
    compute_diff, find_common_prefix, is_complete_json, PartialJson,
};

#[test]
fn test_parse_state_new() {
    let state = ParseState::new();
    assert_eq!(state.phase, ParsePhase::Searching);
    assert_eq!(state.buffer, "");
    assert_eq!(state.consumed, 0);
    assert_eq!(state.bracket_depth, 0);
    assert!(!state.in_string);
    assert!(!state.escape_next);
}

#[test]
fn test_parse_state_process_char() {
    let mut state = ParseState::new();

    // Test bracket tracking
    state.process_char('{');
    assert_eq!(state.bracket_depth, 1);

    state.process_char('}');
    assert_eq!(state.bracket_depth, 0);

    // Test string tracking
    state.process_char('"');
    assert!(state.in_string);

    state.process_char('"');
    assert!(!state.in_string);

    // Test escape handling
    state.process_char('"');
    state.process_char('\\');
    assert!(state.escape_next);

    state.process_char('"');
    assert!(!state.escape_next);
    assert!(state.in_string); // Still in string because quote was escaped
}

#[test]
fn test_token_config() {
    let config = TokenConfig {
        start_tokens: vec!["<start>".to_string(), "[".to_string()],
        end_tokens: vec!["</end>".to_string(), "]".to_string()],
        separator: ", ".to_string(),
    };

    let pairs: Vec<_> = config.iter_pairs().collect();
    assert_eq!(pairs.len(), 2);
    assert_eq!(pairs[0], ("<start>", "</end>"));
    assert_eq!(pairs[1], ("[", "]"));
}

#[test]
fn test_parser_registry() {
    let registry = ParserRegistry::new();

    // Test has default mappings
    assert!(!registry.list_mappings().is_empty());

    // Test model pattern matching
    let mappings = registry.list_mappings();
    let has_gpt = mappings.iter().any(|(m, _)| m.starts_with("gpt"));
    assert!(has_gpt);
}

#[test]
fn test_parser_registry_pattern_matching() {
    let mut registry = ParserRegistry::new();

    // Test that model mappings work by checking the list
    registry.map_model("test-model", "json");

    // Verify through list_mappings
    let mappings = registry.list_mappings();
    let has_test = mappings
        .iter()
        .any(|(m, p)| *m == "test-model" && *p == "json");
    assert!(has_test);
}

#[test]
fn test_tool_call_serialization() {
    let tool_call = ToolCall {
        id: "call-123".to_string(),
        r#type: "function".to_string(),
        function: FunctionCall {
            name: "search".to_string(),
            arguments: r#"{"query": "rust programming"}"#.to_string(),
        },
    };

    let json = serde_json::to_string(&tool_call).unwrap();
    assert!(json.contains("call-123"));
    assert!(json.contains("search"));
    assert!(json.contains("rust programming"));

    let parsed: ToolCall = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.id, "call-123");
    assert_eq!(parsed.function.name, "search");
}

#[test]
fn test_partial_json_parser() {
    let parser = PartialJson::default();

    // Test complete JSON
    let input = r#"{"name": "test", "value": 42}"#;
    let (value, consumed) = parser.parse_value(input).unwrap();
    assert_eq!(value["name"], "test");
    assert_eq!(value["value"], 42);
    assert_eq!(consumed, input.len());

    // Test incomplete JSON object
    let input = r#"{"name": "test", "value": "#;
    let (value, _consumed) = parser.parse_value(input).unwrap();
    assert_eq!(value["name"], "test");
    assert!(value["value"].is_null());

    // Test incomplete string
    let input = r#"{"name": "tes"#;
    let (value, _consumed) = parser.parse_value(input).unwrap();
    assert_eq!(value["name"], "tes");

    // Test incomplete array
    let input = r#"[1, 2, "#;
    let (value, _consumed) = parser.parse_value(input).unwrap();
    assert!(value.is_array());
    assert_eq!(value[0], 1);
    assert_eq!(value[1], 2);
}

#[test]
fn test_partial_json_depth_limit() {
    // max_depth of 3 allows nesting up to 3 levels
    // Set allow_incomplete to false to get errors instead of partial results
    let parser = PartialJson::new(3, false);

    // This should work (simple object)
    let input = r#"{"a": 1}"#;
    let result = parser.parse_value(input);
    assert!(result.is_ok());

    // This should work (nested to depth 3)
    let input = r#"{"a": {"b": {"c": 1}}}"#;
    let result = parser.parse_value(input);
    assert!(result.is_ok());

    // This should fail (nested to depth 4, exceeds limit)
    let input = r#"{"a": {"b": {"c": {"d": 1}}}}"#;
    let result = parser.parse_value(input);
    assert!(result.is_err());
}

#[test]
fn test_is_complete_json() {
    assert!(is_complete_json(r#"{"name": "test"}"#));
    assert!(is_complete_json(r#"[1, 2, 3]"#));
    assert!(is_complete_json(r#""string""#));
    assert!(is_complete_json("42"));
    assert!(is_complete_json("true"));
    assert!(is_complete_json("null"));

    assert!(!is_complete_json(r#"{"name": "#));
    assert!(!is_complete_json(r#"[1, 2, "#));
    assert!(!is_complete_json(r#""unclosed"#));
}

#[test]
fn test_find_common_prefix() {
    assert_eq!(find_common_prefix("hello", "hello"), 5);
    assert_eq!(find_common_prefix("hello", "help"), 3);
    assert_eq!(find_common_prefix("hello", "world"), 0);
    assert_eq!(find_common_prefix("", "hello"), 0);
    assert_eq!(find_common_prefix("hello", ""), 0);
}

#[test]
fn test_compute_diff() {
    assert_eq!(compute_diff("hello", "hello world"), " world");
    assert_eq!(compute_diff("", "hello"), "hello");
    assert_eq!(compute_diff("hello", "hello"), "");
    assert_eq!(compute_diff("test", "hello"), "hello");
}

#[test]
fn test_stream_result_variants() {
    // Test Incomplete
    let result = StreamResult::Incomplete;
    matches!(result, StreamResult::Incomplete);

    // Test ToolName
    let result = StreamResult::ToolName {
        index: 0,
        name: "test".to_string(),
    };
    if let StreamResult::ToolName { index, name } = result {
        assert_eq!(index, 0);
        assert_eq!(name, "test");
    } else {
        panic!("Expected ToolName variant");
    }

    // Test ToolComplete
    let tool = ToolCall {
        id: "123".to_string(),
        r#type: "function".to_string(),
        function: FunctionCall {
            name: "test".to_string(),
            arguments: "{}".to_string(),
        },
    };
    let result = StreamResult::ToolComplete(tool.clone());
    if let StreamResult::ToolComplete(t) = result {
        assert_eq!(t.id, "123");
    } else {
        panic!("Expected ToolComplete variant");
    }
}

#[test]
fn test_partial_tool_call() {
    let mut partial = PartialToolCall {
        name: None,
        arguments_buffer: String::new(),
        start_position: 0,
        name_sent: false,
        streamed_args: String::new(),
    };

    // Set name
    partial.name = Some("test_function".to_string());
    assert_eq!(partial.name.as_ref().unwrap(), "test_function");

    // Append arguments
    partial.arguments_buffer.push_str(r#"{"key": "value"}"#);
    assert_eq!(partial.arguments_buffer, r#"{"key": "value"}"#);

    // Update streaming state
    partial.name_sent = true;
    partial.streamed_args = r#"{"key": "#.to_string();
    assert!(partial.name_sent);
    assert_eq!(partial.streamed_args, r#"{"key": "#);
}
