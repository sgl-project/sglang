use async_trait::async_trait;
use regex::Regex;

use crate::tool_parser::{
    errors::ToolParserResult,
    partial_json::PartialJson,
    state::ParseState,
    traits::ToolParser,
    types::{FunctionCall, StreamResult, ToolCall},
};

/// Kimi K2 format parser for tool calls
///
/// Handles the Kimi K2 specific format:
/// `<|tool_calls_section_begin|><|tool_call_begin|>functions.{name}:{index}<|tool_call_argument_begin|>{json_args}<|tool_call_end|><|tool_calls_section_end|>`
///
/// Features:
/// - Token-based delimiters
/// - Function calls with explicit indexing
/// - JSON arguments
pub struct KimiK2Parser {
    /// Parser for handling incomplete JSON during streaming
    partial_json: PartialJson,
    /// Regex for extracting complete tool calls
    tool_call_extractor: Regex,
    /// Regex for extracting partial tool calls (streaming)
    stream_tool_call_extractor: Regex,
}

impl KimiK2Parser {
    /// Create a new Kimi K2 parser
    pub fn new() -> Self {
        // Pattern for complete tool calls
        let tool_call_pattern = r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[\w\.]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>\{.*?\})\s*<\|tool_call_end\|>";
        let tool_call_extractor = Regex::new(tool_call_pattern).expect("Valid regex pattern");

        // Pattern for streaming (partial) tool calls
        let stream_pattern = r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[\w\.]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>\{.*)";
        let stream_tool_call_extractor = Regex::new(stream_pattern).expect("Valid regex pattern");

        Self {
            partial_json: PartialJson::default(),
            tool_call_extractor,
            stream_tool_call_extractor,
        }
    }

    /// Check if text contains Kimi K2 tool markers
    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains("<|tool_calls_section_begin|>")
    }

    /// Parse function ID to extract name and index
    fn parse_function_id(&self, id: &str) -> Option<(String, usize)> {
        // Format: functions.{name}:{index} or namespace.functions.{name}:{index}
        // Extract everything after the last dot before the colon as the function name
        if let Some(colon_pos) = id.rfind(':') {
            let before_colon = &id[..colon_pos];
            let index_str = &id[colon_pos + 1..];

            // Find the last dot to extract the function name
            if let Some(dot_pos) = before_colon.rfind('.') {
                let func_name = &before_colon[dot_pos + 1..];

                if let Ok(index) = index_str.parse::<usize>() {
                    return Some((func_name.to_string(), index));
                }
            }
        }
        None
    }
}

impl Default for KimiK2Parser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for KimiK2Parser {
    async fn parse_complete(&self, text: &str) -> ToolParserResult<(String, Vec<ToolCall>)> {
        // Check if text contains Kimi K2 format
        if !self.has_tool_markers(text) {
            return Ok((text.to_string(), vec![]));
        }

        // Collect matches with positions and parse tools in one pass
        let matches: Vec<_> = self.tool_call_extractor.captures_iter(text).collect();
        let mut tools = Vec::new();

        // Extract all tool calls using collected matches
        for captures in matches.iter() {
            if let (Some(id_match), Some(args_match)) = (
                captures.name("tool_call_id"),
                captures.name("function_arguments"),
            ) {
                let function_id = id_match.as_str();
                let function_args = args_match.as_str();

                // Parse function ID
                if let Some((func_name, _index)) = self.parse_function_id(function_id) {
                    // Validate JSON arguments
                    if serde_json::from_str::<serde_json::Value>(function_args).is_ok() {
                        // Generate unique ID
                        let id = format!("kimi_call_{}", uuid::Uuid::new_v4());

                        tools.push(ToolCall {
                            id,
                            r#type: "function".to_string(),
                            function: FunctionCall {
                                name: func_name,
                                arguments: function_args.to_string(),
                            },
                        });
                    }
                }
            }
        }

        // Extract normal text using first and last match positions
        let normal_text = if tools.is_empty() || matches.is_empty() {
            text.to_string()
        } else {
            let first_start = matches[0].get(0).unwrap().start();
            let last_end = matches.last().unwrap().get(0).unwrap().end();
            let before = if first_start > 0 {
                &text[..first_start]
            } else {
                ""
            };
            let after = if last_end < text.len() {
                &text[last_end..]
            } else {
                ""
            };
            format!("{}{}", before, after)
        };

        Ok((normal_text, tools))
    }

    async fn parse_incremental(
        &self,
        chunk: &str,
        state: &mut ParseState,
    ) -> ToolParserResult<StreamResult> {
        state.buffer.push_str(chunk);

        // Check for tool markers
        let has_tool_call =
            self.has_tool_markers(&state.buffer) || state.buffer.contains("<|tool_call_begin|>");

        if !has_tool_call {
            // No tool markers detected - return all buffered content as normal text
            let normal_text = std::mem::take(&mut state.buffer);
            return Ok(StreamResult::NormalText(normal_text));
        }

        // Check for text before tool markers and extract it as normal text
        let marker1_pos = state.buffer.find("<|tool_calls_section_begin|>");
        let marker2_pos = state.buffer.find("<|tool_call_begin|>");
        let marker_pos = marker1_pos.iter().chain(marker2_pos.iter()).min().copied();

        if let Some(pos) = marker_pos {
            if pos > 0 {
                // We have text before the tool marker - extract it as normal text
                let normal_text: String = state.buffer.drain(..pos).collect();
                return Ok(StreamResult::NormalText(normal_text));
            }
        }

        // Try to match streaming pattern
        if let Some(captures) = self.stream_tool_call_extractor.captures(&state.buffer) {
            if let (Some(id_match), Some(args_match)) = (
                captures.name("tool_call_id"),
                captures.name("function_arguments"),
            ) {
                let function_id = id_match.as_str();
                let partial_args = args_match.as_str();

                // Parse function ID
                if let Some((func_name, _index)) = self.parse_function_id(function_id) {
                    // Send function name if not sent yet
                    if !state.in_string {
                        state.in_string = true; // Mark name as sent
                        return Ok(StreamResult::ToolName {
                            index: 0,
                            name: func_name.clone(),
                        });
                    }

                    // Check if we have a complete tool call
                    if let Some(end_pos) = partial_args.find("<|tool_call_end|>") {
                        // Extract just the JSON part
                        let json_args = &partial_args[..end_pos];

                        // Validate and parse JSON
                        if serde_json::from_str::<serde_json::Value>(json_args).is_ok() {
                            // Generate unique ID
                            let id = format!("kimi_call_{}", uuid::Uuid::new_v4());

                            let tool = ToolCall {
                                id,
                                r#type: "function".to_string(),
                                function: FunctionCall {
                                    name: func_name,
                                    arguments: json_args.to_string(),
                                },
                            };

                            // Find where this tool call ends in the buffer
                            if let Some(tool_end) = state.buffer.find("<|tool_call_end|>") {
                                let end_pos = tool_end + "<|tool_call_end|>".len();
                                state.buffer.drain(..end_pos);
                            }

                            // Reset state for next tool
                            state.in_string = false;

                            return Ok(StreamResult::ToolComplete(tool));
                        }
                    } else {
                        // Try to parse partial JSON for streaming arguments
                        match self.partial_json.parse_value(partial_args) {
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
        self.has_tool_markers(text) || text.contains("<|tool_call_begin|>")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_parse_kimi_single_tool() {
        let parser = KimiK2Parser::new();
        let input = r#"Some text
<|tool_calls_section_begin|>
<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location": "Tokyo", "units": "celsius"}<|tool_call_end|>
<|tool_calls_section_end|>More text"#;

        let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "get_weather");
        assert!(tools[0].function.arguments.contains("Tokyo"));
    }

    #[tokio::test]
    async fn test_parse_kimi_multiple_tools() {
        let parser = KimiK2Parser::new();
        let input = r#"<|tool_calls_section_begin|>
<|tool_call_begin|>functions.search:0<|tool_call_argument_begin|>{"query": "rust"}<|tool_call_end|>
<|tool_call_begin|>functions.calculate:1<|tool_call_argument_begin|>{"expression": "2+2"}<|tool_call_end|>
<|tool_calls_section_end|>"#;

        let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].function.name, "search");
        assert_eq!(tools[1].function.name, "calculate");
    }

    #[tokio::test]
    async fn test_parse_kimi_with_whitespace() {
        let parser = KimiK2Parser::new();
        let input = r#"<|tool_calls_section_begin|>
<|tool_call_begin|> functions.test:0 <|tool_call_argument_begin|> {"key": "value"} <|tool_call_end|>
<|tool_calls_section_end|>"#;

        let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "test");
    }

    #[test]
    fn test_detect_format() {
        let parser = KimiK2Parser::new();
        assert!(parser.detect_format("<|tool_calls_section_begin|>"));
        assert!(parser.detect_format("<|tool_call_begin|>"));
        assert!(!parser.detect_format("plain text"));
        assert!(!parser.detect_format("[TOOL_CALLS]"));
    }
}
