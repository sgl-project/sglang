//! Tests for structural tag generation by tool parsers
//!
//! Tests each parser's `get_format_info()` and `build_structural_tag()` methods
//! according to GitHub issue #13032 requirements

use smg::protocols::common::Tool;
use smg::tool_parser::{
    parsers::{DeepSeekParser, JsonParser, LlamaParser, MistralParser, QwenParser},
    traits::ToolParser,
};

/// Helper to create a test tool
fn create_test_tool(name: &str) -> Tool {
    Tool {
        tool_type: "function".to_string(),
        function: smg::protocols::common::Function {
            name: name.to_string(),
            description: Some(format!("Test function: {}", name)),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "param1": {"type": "string"},
                    "param2": {"type": "number"}
                }
            }),
            strict: None,
        },
    }
}

// ============================================================================
// get_format_info() Tests - Each parser's format-specific patterns
// ============================================================================

#[test]
fn test_llama_get_format_info() {
    let parser = LlamaParser::new();
    let (begin, end, trigger) = parser.get_format_info("test_function");

    assert_eq!(begin, r#"{"name":"test_function", "arguments":"#);
    assert_eq!(end, "}");
    assert_eq!(trigger, "<|python_tag|>");
}

#[test]
fn test_mistral_get_format_info() {
    let parser = MistralParser::new();
    let (begin, end, trigger) = parser.get_format_info("my_tool");

    assert_eq!(begin, "[TOOL_CALLS] [{");
    assert_eq!(end, "}]");
    assert_eq!(trigger, "[TOOL_CALLS]");
}

#[test]
fn test_qwen_get_format_info() {
    let parser = QwenParser::new();
    let (begin, end, trigger) = parser.get_format_info("search_tool");

    assert_eq!(begin, "{\"name\": \"");
    assert_eq!(end, "\"}");
    assert_eq!(trigger, "<|tool_call|>");
}

#[test]
fn test_json_get_format_info() {
    let parser = JsonParser::new();
    let (begin, end, trigger) = parser.get_format_info("json_func");

    assert_eq!(begin, "{\"name\": \"");
    assert_eq!(end, "\"}");
    assert_eq!(trigger, "[TOOL_CALLS]");
}

// ============================================================================
// build_structural_tag() Tests - Default implementation (triggered_tags format)
// ============================================================================

#[test]
fn test_llama_build_structural_tag_default() {
    let parser = LlamaParser::new();
    let tools = vec![create_test_tool("get_weather")];

    let tag = parser.build_structural_tag(&tools, false, false).unwrap();
    let tag_json: serde_json::Value = serde_json::from_str(&tag).unwrap();

    assert_eq!(tag_json["format"], "triggered_tags");
    assert!(tag_json["triggers"]
        .as_array()
        .unwrap()
        .contains(&"<|python_tag|>".into()));
    assert_eq!(tag_json["at_least_one"], false);
    assert_eq!(tag_json["stop_after_first"], false);

    let tags = tag_json["tags"].as_array().unwrap();
    assert_eq!(tags.len(), 1);
    assert_eq!(tags[0]["format"], "tag");
    assert!(tags[0]["begin"].as_str().unwrap().contains("get_weather"));
    assert_eq!(tags[0]["content"]["format"], "json_schema");
}

#[test]
fn test_mistral_build_structural_tag_default() {
    let parser = MistralParser::new();
    let tools = vec![create_test_tool("search")];

    let tag = parser.build_structural_tag(&tools, false, false).unwrap();
    let tag_json: serde_json::Value = serde_json::from_str(&tag).unwrap();

    assert_eq!(tag_json["format"], "triggered_tags");
    assert!(tag_json["triggers"]
        .as_array()
        .unwrap()
        .contains(&"[TOOL_CALLS]".into()));

    let tags = tag_json["tags"].as_array().unwrap();
    assert_eq!(tags.len(), 1);
    assert_eq!(tags[0]["format"], "tag");
    assert!(tags[0]["begin"].as_str().unwrap().contains("[TOOL_CALLS]"));
}

#[test]
fn test_multiple_tools_default() {
    let parser = LlamaParser::new();
    let tools = vec![
        create_test_tool("tool1"),
        create_test_tool("tool2"),
        create_test_tool("tool3"),
    ];

    let tag = parser.build_structural_tag(&tools, false, false).unwrap();
    let tag_json: serde_json::Value = serde_json::from_str(&tag).unwrap();

    let tags = tag_json["tags"].as_array().unwrap();
    assert_eq!(tags.len(), 3);
}

// ============================================================================
// at_least_one and stop_after_first parameter tests
// ============================================================================

#[test]
fn test_at_least_one_true() {
    let parser = LlamaParser::new();
    let tools = vec![create_test_tool("weather")];

    let tag = parser.build_structural_tag(&tools, true, false).unwrap();
    let tag_json: serde_json::Value = serde_json::from_str(&tag).unwrap();

    assert_eq!(tag_json["at_least_one"], true);
}

#[test]
fn test_stop_after_first_true() {
    let parser = JsonParser::new();
    let tools = vec![create_test_tool("calc")];

    let tag = parser.build_structural_tag(&tools, false, true).unwrap();
    let tag_json: serde_json::Value = serde_json::from_str(&tag).unwrap();

    assert_eq!(tag_json["stop_after_first"], true);
}

// ============================================================================
// DeepSeek Parser - Special case with tags_with_separator override
// ============================================================================

#[test]
fn test_deepseek_build_structural_tag_override() {
    let parser = DeepSeekParser::new();
    let tools = vec![create_test_tool("deepseek_tool")];

    let tag = parser.build_structural_tag(&tools, false, false).unwrap();
    let tag_json: serde_json::Value = serde_json::from_str(&tag).unwrap();

    assert_eq!(tag_json["format"], "triggered_tags");
    assert!(tag_json["triggers"]
        .as_array()
        .unwrap()
        .contains(&"<｜tool▁calls▁begin｜>".into()));

    let outer_tags = tag_json["tags"].as_array().unwrap();
    assert_eq!(outer_tags.len(), 1);
    assert!(outer_tags[0]["begin"]
        .as_str()
        .unwrap()
        .contains("<｜tool▁calls▁begin｜>"));
    assert_eq!(
        outer_tags[0]["end"].as_str().unwrap(),
        "\\n<｜tool▁calls▁end｜>"
    );

    let inner_content = &outer_tags[0]["content"];
    assert_eq!(inner_content["format"], "tags_with_separator");
    assert_eq!(inner_content["separator"], "\\n");

    let inner_tags = inner_content["tags"].as_array().unwrap();
    assert_eq!(inner_tags.len(), 1);
    assert_eq!(inner_tags[0]["format"], "tag");
}

#[test]
fn test_deepseek_multiple_tools_with_separator() {
    let parser = DeepSeekParser::new();
    let tools = vec![
        create_test_tool("tool1"),
        create_test_tool("tool2"),
        create_test_tool("tool3"),
    ];

    let tag = parser.build_structural_tag(&tools, false, false).unwrap();
    let tag_json: serde_json::Value = serde_json::from_str(&tag).unwrap();

    let outer_tags = tag_json["tags"].as_array().unwrap();
    let inner_content = &outer_tags[0]["content"];
    let inner_tags = inner_content["tags"].as_array().unwrap();

    assert_eq!(inner_tags.len(), 3);
    assert_eq!(inner_content["separator"], "\\n");
}

#[test]
fn test_deepseek_with_constraints() {
    let parser = DeepSeekParser::new();
    let tools = vec![create_test_tool("constrained_tool")];

    let tag = parser.build_structural_tag(&tools, true, true).unwrap();
    let tag_json: serde_json::Value = serde_json::from_str(&tag).unwrap();

    assert_eq!(tag_json["at_least_one"], false);
    assert_eq!(tag_json["stop_after_first"], false);

    let inner_content = &tag_json["tags"][0]["content"];
    assert_eq!(inner_content["at_least_one"], true);
    assert_eq!(inner_content["stop_after_first"], true);
}

// ============================================================================
// Schema handling in structural tags
// ============================================================================

#[test]
fn test_schema_included_in_structural_tag() {
    let parser = JsonParser::new();
    let tools = vec![create_test_tool("schema_tool")];

    let tag = parser.build_structural_tag(&tools, false, false).unwrap();
    let tag_json: serde_json::Value = serde_json::from_str(&tag).unwrap();

    let schema = &tag_json["tags"][0]["content"]["schema"];
    assert!(schema.is_object());
    assert_eq!(schema["type"], "object");
    assert!(schema["properties"].is_object());
}

#[test]
fn test_complex_schema_in_structural_tag() {
    let parser = LlamaParser::new();
    let complex_schema = serde_json::json!({
        "type": "object",
        "properties": {
            "nested": {
                "type": "object",
                "properties": {
                    "inner": {"type": "string"}
                }
            },
            "array_param": {"type": "array", "items": {"type": "number"}},
            "optional_param": {"type": "boolean"}
        },
        "required": ["nested"]
    });

    let complex_tool = Tool {
        tool_type: "function".to_string(),
        function: smg::protocols::common::Function {
            name: "complex_func".to_string(),
            description: Some("Complex test function: complex_func".to_string()),
            parameters: complex_schema,
            strict: None,
        },
    };

    let tag = parser
        .build_structural_tag(&[complex_tool], false, false)
        .unwrap();
    let tag_json: serde_json::Value = serde_json::from_str(&tag).unwrap();

    let schema = &tag_json["tags"][0]["content"]["schema"];
    assert_eq!(schema["type"], "object");
    assert!(schema["properties"]["nested"]["properties"]["inner"].is_object());
    assert_eq!(
        schema["properties"]["array_param"]["items"]["type"],
        "number"
    );
    assert!(schema["required"]
        .as_array()
        .unwrap()
        .contains(&"nested".into()));
}
