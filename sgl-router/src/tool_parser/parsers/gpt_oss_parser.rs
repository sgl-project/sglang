use async_trait::async_trait;
use regex::Regex;
use serde_json::Value;

use crate::tool_parser::{
    errors::{ToolParserError, ToolParserResult},
    partial_json::PartialJson,
    state::ParseState,
    traits::ToolParser,
    types::{FunctionCall, StreamResult, ToolCall},
};

/// GPT-OSS format parser for tool calls
///
/// Handles the GPT-OSS specific channel format:
/// `<|channel|>commentary to={namespace.function}<|constrain|>json<|message|>{json_args}<|call|>`
///
/// Features:
/// - Channel-based format with commentary
/// - Namespaced function calls
/// - JSON arguments
pub struct GptOssParser {
    /// Parser for handling incomplete JSON during streaming
    partial_json: PartialJson,
    /// Regex for extracting complete function calls
    function_call_extractor: Regex,
    /// Regex for extracting streaming function calls
    streaming_extractor: Regex,
}

impl GptOssParser {
    /// Create a new GPT-OSS parser
    pub fn new() -> Self {
        // Pattern for complete function calls with to= parameter
        // Handles optional <|start|>assistant prefix and whitespace after function name
        let function_call_pattern = r"(?s)(?:<\|start\|>assistant)?<\|channel\|>commentary to=([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*<\|constrain\|>json<\|message\|>(.*?)<\|call\|>(?:commentary)?";
        let function_call_extractor =
            Regex::new(function_call_pattern).expect("Valid regex pattern");

        // Pattern for streaming function calls (incomplete)
        let streaming_pattern = r"(?s)(?:<\|start\|>assistant)?<\|channel\|>commentary to=([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*<\|constrain\|>json<\|message\|>(.*)";
        let streaming_extractor = Regex::new(streaming_pattern).expect("Valid regex pattern");

        Self {
            partial_json: PartialJson::default(),
            function_call_extractor,
            streaming_extractor,
        }
    }

    /// Check if text contains GPT-OSS tool markers
    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains("<|channel|>commentary to=")
    }

    /// Extract function name from full namespace (e.g., "functions.get_weather" -> "get_weather")
    fn extract_function_name(&self, full_name: &str) -> String {
        if let Some(dot_pos) = full_name.rfind('.') {
            full_name[dot_pos + 1..].to_string()
        } else {
            full_name.to_string()
        }
    }
}

impl Default for GptOssParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for GptOssParser {
    async fn parse_complete(&self, text: &str) -> ToolParserResult<Vec<ToolCall>> {
        // Check if text contains GPT-OSS format
        if !self.has_tool_markers(text) {
            return Ok(vec![]);
        }

        let mut tools = Vec::new();
        let mut _tool_index = 0;

        // Extract all function calls
        for captures in self.function_call_extractor.captures_iter(text) {
            if let (Some(name_match), Some(args_match)) = (captures.get(1), captures.get(2)) {
                let full_function_name = name_match.as_str();
                let args_content = args_match.as_str().trim();

                // Extract actual function name
                let function_name = self.extract_function_name(full_function_name);

                // Parse JSON arguments
                let arguments = if args_content.is_empty() {
                    "{}".to_string()
                } else {
                    match serde_json::from_str::<Value>(args_content) {
                        Ok(value) => serde_json::to_string(&value)
                            .map_err(|e| ToolParserError::ParsingFailed(e.to_string()))?,
                        Err(_) => {
                            // Skip malformed JSON
                            continue;
                        }
                    }
                };

                // Generate unique ID
                let id = format!("gpt_oss_call_{}", uuid::Uuid::new_v4());

                tools.push(ToolCall {
                    id,
                    r#type: "function".to_string(),
                    function: FunctionCall {
                        name: function_name,
                        arguments,
                    },
                });

                _tool_index += 1;
            }
        }

        Ok(tools)
    }

    async fn parse_incremental(
        &self,
        chunk: &str,
        state: &mut ParseState,
    ) -> ToolParserResult<StreamResult> {
        state.buffer.push_str(chunk);

        // Check for tool markers
        if !self.has_tool_markers(&state.buffer) {
            // No markers found, clear buffer and return
            state.buffer.clear();
            return Ok(StreamResult::Incomplete);
        }

        // Try to match streaming pattern
        if let Some(captures) = self.streaming_extractor.captures(&state.buffer) {
            if let (Some(name_match), Some(args_match)) = (captures.get(1), captures.get(2)) {
                let full_function_name = name_match.as_str();
                let partial_args = args_match.as_str();

                // Extract actual function name
                let function_name = self.extract_function_name(full_function_name);

                // Send function name if not sent yet
                if !state.in_string {
                    state.in_string = true; // Mark name as sent
                    return Ok(StreamResult::ToolName {
                        index: 0,
                        name: function_name.clone(),
                    });
                }

                // Check if we have a complete function call
                if let Some(complete_match) = self.function_call_extractor.captures(&state.buffer) {
                    if let Some(args_match) = complete_match.get(2) {
                        let args_content = args_match.as_str().trim();

                        // Parse JSON arguments
                        let arguments = if args_content.is_empty() {
                            "{}".to_string()
                        } else {
                            match serde_json::from_str::<Value>(args_content) {
                                Ok(value) => serde_json::to_string(&value)
                                    .unwrap_or_else(|_| "{}".to_string()),
                                Err(_) => "{}".to_string(),
                            }
                        };

                        // Generate unique ID
                        let id = format!("gpt_oss_call_{}", uuid::Uuid::new_v4());

                        let tool = ToolCall {
                            id,
                            r#type: "function".to_string(),
                            function: FunctionCall {
                                name: function_name,
                                arguments,
                            },
                        };

                        // Remove the processed part from buffer
                        let complete_end = complete_match.get(0).unwrap().end();
                        state.buffer.drain(..complete_end);

                        // Reset state for next tool
                        state.in_string = false;

                        return Ok(StreamResult::ToolComplete(tool));
                    }
                } else {
                    // Try to parse partial JSON for streaming arguments
                    if !partial_args.is_empty() {
                        // Look for the end of JSON (before <|call|>)
                        let json_part = if let Some(call_pos) = partial_args.find("<|call|>") {
                            &partial_args[..call_pos]
                        } else {
                            partial_args
                        };

                        match self.partial_json.parse_value(json_part) {
                            Ok((value, _consumed)) => {
                                let args_str = serde_json::to_string(&value)
                                    .unwrap_or_else(|_| "{}".to_string());

                                return Ok(StreamResult::ToolArguments {
                                    index: 0,
                                    arguments: args_str,
                                });
                            }
                            Err(_) => {
                                // Can't parse yet, keep buffering
                            }
                        }
                    }
                }
            }
        }

        Ok(StreamResult::Incomplete)
    }

    fn detect_format(&self, text: &str) -> bool {
        self.has_tool_markers(text) || text.contains("<|channel|>commentary")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_parse_gpt_oss_single_tool() {
        let parser = GptOssParser::new();
        let input = r#"Some text
<|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{"location": "San Francisco"}<|call|>
More text"#;

        let result = parser.parse_complete(input).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].function.name, "get_weather");
        assert!(result[0].function.arguments.contains("San Francisco"));
    }

    #[tokio::test]
    async fn test_parse_gpt_oss_multiple_tools() {
        let parser = GptOssParser::new();
        let input = r#"<|channel|>commentary to=functions.get_weather<|constrain|>json<|message|>{"location": "Paris"}<|call|>commentary
<|channel|>commentary to=functions.search<|constrain|>json<|message|>{"query": "Paris tourism"}<|call|>"#;

        let result = parser.parse_complete(input).await.unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].function.name, "get_weather");
        assert_eq!(result[1].function.name, "search");
        assert!(result[0].function.arguments.contains("Paris"));
        assert!(result[1].function.arguments.contains("Paris tourism"));
    }

    #[tokio::test]
    async fn test_parse_gpt_oss_with_prefix() {
        let parser = GptOssParser::new();
        let input = r#"<|start|>assistant<|channel|>commentary to=functions.test<|constrain|>json<|message|>{"key": "value"}<|call|>"#;

        let result = parser.parse_complete(input).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].function.name, "test");
    }

    #[tokio::test]
    async fn test_parse_gpt_oss_empty_args() {
        let parser = GptOssParser::new();
        let input =
            r#"<|channel|>commentary to=functions.get_time<|constrain|>json<|message|>{}<|call|>"#;

        let result = parser.parse_complete(input).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].function.name, "get_time");
        assert_eq!(result[0].function.arguments, "{}");
    }

    #[test]
    fn test_detect_format() {
        let parser = GptOssParser::new();
        assert!(parser.detect_format("<|channel|>commentary to="));
        assert!(parser.detect_format("<|channel|>commentary"));
        assert!(!parser.detect_format("plain text"));
        assert!(!parser.detect_format("[TOOL_CALLS]"));
    }
}
