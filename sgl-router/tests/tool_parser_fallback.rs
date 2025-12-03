//! Tests for tool parser fallback behavior
//!
//! When tool call parsing fails, the original text should be preserved as normal text
//! rather than being lost. This ensures graceful degradation.

use sgl_model_gateway::tool_parser::{
    DeepSeekParser, JsonParser, LlamaParser, MistralParser, QwenParser, ToolParser,
};

#[tokio::test]
async fn test_json_parser_invalid_json_returns_as_normal_text() {
    let parser = JsonParser::new();

    // Malformed JSON should be returned as normal text (note: commas may be processed)
    let input = r#"{"name": "test", "arguments": invalid json here}"#;
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);
    assert_eq!(
        normal_text,
        r#"{"name": "test", "arguments": invalid json here}"#
    );

    // Plain text with no JSON structure should be returned as normal text
    let input = "This is just plain text that should not be parsed as a tool call";
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);
    assert_eq!(normal_text, input);

    // Text that looks like it might have JSON but doesn't should be returned as normal text
    let input = "The user said: {something} but it's not valid JSON";
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);
    assert_eq!(normal_text, input);
}

#[tokio::test]
async fn test_qwen_parser_invalid_format_returns_as_normal_text() {
    let parser = QwenParser::new();

    // Missing closing tag
    let input = r#"<tool_call>
{"name": "test", "arguments": {}}
This text is missing the closing tag"#;
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);
    assert_eq!(normal_text, input); // Should preserve original text when no valid tools found

    // Malformed JSON inside valid tags
    let input = r#"<tool_call>
{"name": "test", "arguments": invalid}
</tool_call>"#;
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);
    // When JSON parsing fails but tags are present, it should preserve the original text
    assert_eq!(normal_text, input);

    // Plain text without any tool markers
    let input = "This is a regular response without any tool calls.";
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);
    assert_eq!(normal_text, input); // Should return original text when no markers found
}

#[tokio::test]
async fn test_llama_parser_invalid_format_returns_as_normal_text() {
    let parser = LlamaParser::new();

    // Invalid JSON after python_tag
    let input = r#"<|python_tag|>{"name": "test", "arguments": invalid}"#;
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);
    assert_eq!(normal_text, input); // Should preserve original text when parsing fails

    // Plain text without markers or JSON
    let input = "Just explaining something without any function calls.";
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);
    assert_eq!(normal_text, input); // Should return original text

    // Text with python_tag but completely invalid content
    let input = r#"Here's my response <|python_tag|>not even close to JSON"#;
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);
    assert_eq!(normal_text, input); // Should preserve everything when parsing fails
}

#[tokio::test]
async fn test_mistral_parser_invalid_format_returns_as_normal_text() {
    let parser = MistralParser::new();

    // Missing closing bracket
    let input = r#"[TOOL_CALLS] [{"name": "test", "arguments": {}"#;
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);
    assert_eq!(normal_text, input); // Should preserve original text when parsing fails

    // Invalid JSON in tool calls section
    let input = r#"[TOOL_CALLS] [{"name": invalid json}]"#;
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);
    assert_eq!(normal_text, input); // Should preserve original text when parsing fails

    // Plain text
    let input = "No tool calls here, just regular text.";
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);
    assert_eq!(normal_text, input); // Should return original text
}

#[tokio::test]
async fn test_deepseek_parser_invalid_format_returns_as_normal_text() {
    let parser = DeepSeekParser::new();

    // Invalid JSON in tool call
    let input = r#"Some text<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>test
```json
{"name": "test", "arguments": malformed}
```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"#;
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);
    assert_eq!(normal_text, input); // Should preserve original text when parsing fails

    // Missing function marker
    let input = r#"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>notfunction<｜tool▁sep｜>test
```json
{"x": 1}
```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"#;
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);
    assert_eq!(normal_text, input); // Should return original text when parsing fails

    // No tool markers at all
    let input = "Regular response without any special markers.";
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);
    assert_eq!(normal_text, input); // Should return original text
}

#[tokio::test]
async fn test_mixed_valid_and_invalid_content() {
    let parser = QwenParser::new();

    // Text with one valid tool call and one invalid
    let input = r#"Let me help you with that.
<tool_call>
{"name": "valid_tool", "arguments": {"x": 1}}
</tool_call>
And here's another one:
<tool_call>
{"name": "invalid_tool", "arguments": malformed}
</tool_call>
That's all!"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1); // Should extract the valid tool
    assert_eq!(tools[0].function.name, "valid_tool");
    // Normal text should contain text before the first tool call
    assert_eq!(normal_text, "Let me help you with that.\n");
}

#[tokio::test]
async fn test_partial_tool_markers() {
    // Test cases where tool markers are incomplete or cut off

    let parser = QwenParser::new();
    let input = "<tool_call>\nThis looks like it might be a tool call but it's not";
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);
    assert_eq!(normal_text, input);

    let parser = MistralParser::new();
    let input = "[TOOL_CALLS] But then nothing follows...";
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);
    assert_eq!(normal_text, input);

    let parser = LlamaParser::new();
    let input = "Starting a response <|python_tag|> but no JSON";
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);
    assert_eq!(normal_text, input);
}

#[tokio::test]
async fn test_escaped_json_like_content() {
    // Test that JSON-like content in regular text doesn't get parsed as tools

    let parser = JsonParser::new();
    let input = r#"The user typed: {"name": "example"} but this is just quoted text"#;
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    // JsonParser should extract the valid JSON and return normal text
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "example");
    assert_eq!(normal_text, "The user typed:  but this is just quoted text");

    let parser = QwenParser::new();
    let input = r#"The syntax is: <tool_call>
{"name": "example"}
</tool_call> - that's how you format it"#;
    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    // This actually contains valid tool call syntax, so it should parse
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "example");
}

#[tokio::test]
async fn test_unicode_and_special_chars_in_failed_parsing() {
    let parser = QwenParser::new();

    // Unicode in malformed tool calls
    let input = r#"<tool_call>
{"name": "测试", "arguments": 🚀 invalid}
</tool_call>"#;
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);
    // Should handle Unicode properly in the fallback text - malformed content should be preserved
    assert_eq!(normal_text, input);

    // Special characters that might confuse parsers
    let input = r#"Response: <tool_call>{"name": "test\n\t", "arguments": {"]}"}</tool_call>"#;
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    // This might or might not parse depending on JSON handling of escape sequences
    if tools.is_empty() {
        assert!(!normal_text.is_empty() || normal_text == input);
    }
}

#[tokio::test]
async fn test_very_long_invalid_input() {
    let parser = JsonParser::new();

    // Generate a very long string that looks like it might be JSON but isn't
    let mut input = String::from("{\"name\": \"test\", \"arguments\": {");
    for i in 0..1000 {
        input.push_str(&format!("\"field{}\": \"value{}\", ", i, i));
    }
    input.push_str("\"final\": incomplete"); // Don't close the JSON properly

    let (normal_text, tools) = parser.parse_complete(&input).await.unwrap();
    assert_eq!(tools.len(), 0);
    assert_eq!(normal_text, input); // Invalid JSON should be returned as normal text
}

#[tokio::test]
async fn test_almost_valid_tool_calls() {
    // Test tool calls that are almost valid but have small issues

    let parser = JsonParser::new();

    // Missing closing quote should be returned as normal text
    let input = r#"{"name": "test", "arguments": {"key": "value}}"#;
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);
    assert_eq!(
        normal_text,
        r#"{"name": "test", "arguments": {"key": "value}}"#
    );

    // Extra comma
    let input = r#"{"name": "test", "arguments": {},}"#;
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    // Some JSON parsers might accept trailing commas
    if tools.is_empty() {
        assert_eq!(normal_text, r#"{"name": "test", "arguments": {},}"#);
    }

    // Wrong quote types
    let input = r#"{'name': 'test', 'arguments': {}}"#;
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0); // Standard JSON requires double quotes
    assert_eq!(normal_text, r#"{'name': 'test', 'arguments': {}}"#);
}
