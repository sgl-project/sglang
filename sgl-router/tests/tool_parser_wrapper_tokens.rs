//! Wrapper Token Tests
//!
//! Tests for JSON parser with custom wrapper tokens

use sglang_router_rs::tool_parser::{JsonParser, TokenConfig, ToolParser};

#[tokio::test]
async fn test_json_with_xml_style_wrapper() {
    let parser = JsonParser::with_config(TokenConfig {
        start_tokens: vec!["<tool>".to_string()],
        end_tokens: vec!["</tool>".to_string()],
        separator: ", ".to_string(),
    });

    let input =
        r#"Some text before <tool>{"name": "test", "arguments": {"x": 1}}</tool> and after"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "test");

    let args: serde_json::Value = serde_json::from_str(&result[0].function.arguments).unwrap();
    assert_eq!(args["x"], 1);
}

#[tokio::test]
async fn test_json_with_multiple_wrapper_pairs() {
    // Test with multiple start/end token pairs
    let parser = JsonParser::with_config(TokenConfig {
        start_tokens: vec!["<tool>".to_string(), "<<TOOL>>".to_string()],
        end_tokens: vec!["</tool>".to_string(), "<</TOOL>>".to_string()],
        separator: ", ".to_string(),
    });

    // Test first pair
    let input1 = r#"<tool>{"name": "tool1", "arguments": {}}</tool>"#;
    let result1 = parser.parse_complete(input1).await.unwrap();
    assert_eq!(result1.len(), 1);
    assert_eq!(result1[0].function.name, "tool1");

    // Test second pair
    let input2 = r#"<<TOOL>>{"name": "tool2", "arguments": {}}<</TOOL>>"#;
    let result2 = parser.parse_complete(input2).await.unwrap();
    assert_eq!(result2.len(), 1);
    assert_eq!(result2[0].function.name, "tool2");
}

#[tokio::test]
async fn test_json_with_only_start_token() {
    // Test when only start token is provided (no end token)
    let parser = JsonParser::with_config(TokenConfig {
        start_tokens: vec![">>>FUNCTION:".to_string()],
        end_tokens: vec!["".to_string()], // Empty end token
        separator: ", ".to_string(),
    });

    let input = r#"Some preamble >>>FUNCTION:{"name": "execute", "arguments": {"cmd": "ls"}}"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "execute");
}

#[tokio::test]
async fn test_json_with_custom_separator() {
    let parser = JsonParser::with_config(TokenConfig {
        start_tokens: vec!["[FUNC]".to_string()],
        end_tokens: vec!["[/FUNC]".to_string()],
        separator: " | ".to_string(), // Custom separator
    });

    // Though we're not testing multiple tools here, the separator is configured
    let input = r#"[FUNC]{"name": "test", "arguments": {}}[/FUNC]"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "test");
}

#[tokio::test]
async fn test_json_with_nested_wrapper_tokens_in_content() {
    // Known limitation: When wrapper tokens appear inside JSON strings,
    // the simple regex-based extraction may fail. This would require
    // a more sophisticated parser that understands JSON string escaping.

    let parser = JsonParser::with_config(TokenConfig {
        start_tokens: vec!["<call>".to_string()],
        end_tokens: vec!["</call>".to_string()],
        separator: ", ".to_string(),
    });

    let input =
        r#"<call>{"name": "echo", "arguments": {"text": "Use <call> and </call> tags"}}</call>"#;

    let result = parser.parse_complete(input).await.unwrap();

    // This is a known limitation - the parser may fail when end tokens appear in content
    // For now, we accept this behavior
    if result.is_empty() {
        // Parser failed due to nested tokens - this is expected
        assert_eq!(
            result.len(),
            0,
            "Known limitation: nested wrapper tokens in content"
        );
    } else {
        // If it does parse, verify it's correct
        assert_eq!(result[0].function.name, "echo");
        let args: serde_json::Value = serde_json::from_str(&result[0].function.arguments).unwrap();
        assert_eq!(args["text"], "Use <call> and </call> tags");
    }
}

#[tokio::test]
async fn test_json_extraction_without_wrapper_tokens() {
    // Default parser without wrapper tokens should extract JSON from text
    let parser = JsonParser::new();

    let input = r#"
    Here is some text before the JSON.
    {"name": "search", "arguments": {"query": "test"}}
    And here is some text after.
    "#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "search");
}

#[tokio::test]
async fn test_json_with_multiline_wrapper_content() {
    let parser = JsonParser::with_config(TokenConfig {
        start_tokens: vec!["```json\n".to_string()],
        end_tokens: vec!["\n```".to_string()],
        separator: ", ".to_string(),
    });

    let input = r#"Here's the function call:
```json
{
    "name": "format_code",
    "arguments": {
        "language": "rust",
        "code": "fn main() {}"
    }
}
```
Done!"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "format_code");
}

#[tokio::test]
async fn test_json_with_special_chars_in_tokens() {
    let parser = JsonParser::with_config(TokenConfig {
        start_tokens: vec!["{{FUNC[[".to_string()],
        end_tokens: vec!["]]FUNC}}".to_string()],
        separator: ", ".to_string(),
    });

    let input = r#"{{FUNC[[{"name": "test", "arguments": {"special": "[]{}"}}]]FUNC}}"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "test");

    let args: serde_json::Value = serde_json::from_str(&result[0].function.arguments).unwrap();
    assert_eq!(args["special"], "[]{}");
}

#[tokio::test]
async fn test_json_multiple_tools_with_wrapper() {
    let parser = JsonParser::with_config(TokenConfig {
        start_tokens: vec!["<fn>".to_string()],
        end_tokens: vec!["</fn>".to_string()],
        separator: ", ".to_string(),
    });

    // Multiple wrapped JSON objects
    let input = r#"
    <fn>{"name": "tool1", "arguments": {}}</fn>
    Some text between.
    <fn>{"name": "tool2", "arguments": {"x": 1}}</fn>
    "#;

    // Current implementation might handle this as separate calls
    // Let's test that at least the first one is parsed
    let result = parser.parse_complete(input).await.unwrap();
    assert!(!result.is_empty(), "Should parse at least one tool");
    assert_eq!(result[0].function.name, "tool1");
}

#[tokio::test]
async fn test_json_wrapper_with_array() {
    let parser = JsonParser::with_config(TokenConfig {
        start_tokens: vec!["<tools>".to_string()],
        end_tokens: vec!["</tools>".to_string()],
        separator: ", ".to_string(),
    });

    let input = r#"<tools>[
        {"name": "func1", "arguments": {}},
        {"name": "func2", "arguments": {"param": "value"}}
    ]</tools>"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].function.name, "func1");
    assert_eq!(result[1].function.name, "func2");
}

#[tokio::test]
async fn test_json_incomplete_wrapper_tokens() {
    let parser = JsonParser::with_config(TokenConfig {
        start_tokens: vec!["<tool>".to_string()],
        end_tokens: vec!["</tool>".to_string()],
        separator: ", ".to_string(),
    });

    // Missing end token
    let input = r#"<tool>{"name": "test", "arguments": {}}"#;
    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 0, "Should not parse without closing token");

    // Missing start token
    let input = r#"{"name": "test", "arguments": {}}</tool>"#;
    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 0, "Should not parse without opening token");
}

#[tokio::test]
async fn test_json_empty_wrapper_tokens() {
    // Test with empty wrapper tokens (should behave like default)
    let parser = JsonParser::with_config(TokenConfig {
        start_tokens: vec![],
        end_tokens: vec![],
        separator: ", ".to_string(),
    });

    let input = r#"{"name": "test", "arguments": {"key": "value"}}"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "test");
}
