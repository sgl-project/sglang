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

/// DeepSeek V3 format parser for tool calls
///
/// Handles the DeepSeek V3 specific format that uses Unicode tokens:
/// `<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>{name}\n```json\n{args}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>`
///
/// Features:
/// - Unicode token delimiters
/// - JSON arguments in code blocks
/// - Support for multiple sequential tool calls
pub struct DeepSeekParser {
    /// Parser for handling incomplete JSON during streaming
    partial_json: PartialJson,
    /// Regex for extracting complete tool calls
    tool_call_extractor: Regex,
    /// Regex for extracting function details
    func_detail_extractor: Regex,
}

impl DeepSeekParser {
    /// Create a new DeepSeek parser
    pub fn new() -> Self {
        // Use (?s) flag for DOTALL mode to handle newlines
        let tool_call_pattern = r"(?s)<｜tool▁call▁begin｜>.*?<｜tool▁call▁end｜>";
        let tool_call_extractor = Regex::new(tool_call_pattern).expect("Valid regex pattern");

        let func_detail_pattern = r"(?s)<｜tool▁call▁begin｜>(.*?)<｜tool▁sep｜>(.*?)\n```json\n(.*?)\n```<｜tool▁call▁end｜>";
        let func_detail_extractor = Regex::new(func_detail_pattern).expect("Valid regex pattern");

        Self {
            partial_json: PartialJson::default(),
            tool_call_extractor,
            func_detail_extractor,
        }
    }

    /// Check if text contains DeepSeek tool markers
    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains("<｜tool▁calls▁begin｜>")
    }

    /// Parse a single tool call block
    fn parse_tool_call(&self, block: &str) -> ToolParserResult<Option<ToolCall>> {
        if let Some(captures) = self.func_detail_extractor.captures(block) {
            // Get function type (should be "function")
            let func_type = captures.get(1).map_or("", |m| m.as_str());
            if func_type != "function" {
                return Ok(None);
            }

            // Get function name
            let func_name = captures.get(2).map_or("", |m| m.as_str()).trim();

            // Get JSON arguments
            let json_args = captures.get(3).map_or("{}", |m| m.as_str()).trim();

            // Parse JSON arguments
            match serde_json::from_str::<Value>(json_args) {
                Ok(value) => {
                    // Create arguments object
                    let args = if value.is_object() {
                        value
                    } else {
                        // If not an object, wrap it
                        serde_json::json!({ "value": value })
                    };

                    let arguments = serde_json::to_string(&args)
                        .map_err(|e| ToolParserError::ParsingFailed(e.to_string()))?;

                    // Generate ID
                    let id = format!("deepseek_call_{}", uuid::Uuid::new_v4());

                    Ok(Some(ToolCall {
                        id,
                        r#type: "function".to_string(),
                        function: FunctionCall {
                            name: func_name.to_string(),
                            arguments,
                        },
                    }))
                }
                Err(_) => Ok(None),
            }
        } else {
            Ok(None)
        }
    }
}

impl Default for DeepSeekParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for DeepSeekParser {
    async fn parse_complete(&self, text: &str) -> ToolParserResult<(String, Vec<ToolCall>)> {
        // Check if text contains DeepSeek format
        if !self.has_tool_markers(text) {
            return Ok((text.to_string(), vec![]));
        }

        // Collect matches with positions and parse tools in one pass
        let matches: Vec<_> = self.tool_call_extractor.find_iter(text).collect();
        let mut tools = Vec::new();

        for mat in matches.iter() {
            if let Some(tool) = self.parse_tool_call(mat.as_str())? {
                tools.push(tool);
            }
        }

        // Extract normal text using first and last match positions
        let normal_text = if tools.is_empty() || matches.is_empty() {
            text.to_string()
        } else {
            let first_start = matches[0].start();
            let last_end = matches.last().unwrap().end();
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
        if !self.has_tool_markers(&state.buffer) {
            // No tool markers detected - return all buffered content as normal text
            let normal_text = std::mem::take(&mut state.buffer);
            return Ok(StreamResult::NormalText(normal_text));
        }

        // Check for text before tool markers and extract it as normal text
        if let Some(marker_pos) = state.buffer.find("<｜tool▁calls▁begin｜>") {
            if marker_pos > 0 {
                // We have text before the tool marker - extract it as normal text
                let normal_text: String = state.buffer.drain(..marker_pos).collect();
                return Ok(StreamResult::NormalText(normal_text));
            }
        }

        // Look for start of tool calls
        if let Some(start_pos) = state.buffer.find("<｜tool▁calls▁begin｜>") {
            // Look for individual tool call start
            let search_from = start_pos + "<｜tool▁calls▁begin｜>".len();
            if let Some(call_start) = state.buffer[search_from..].find("<｜tool▁call▁begin｜>")
            {
                let call_start_abs = search_from + call_start;

                // Look for the end of this tool call
                let search_end_from = call_start_abs + "<｜tool▁call▁begin｜>".len();
                if let Some(call_end) = state.buffer[search_end_from..].find("<｜tool▁call▁end｜>")
                {
                    let call_end_abs = search_end_from + call_end + "<｜tool▁call▁end｜>".len();

                    // Extract and parse the complete tool call
                    let tool_call_text = &state.buffer[call_start_abs..call_end_abs];

                    if let Some(tool) = self.parse_tool_call(tool_call_text)? {
                        // Remove the processed part from buffer
                        state.buffer.drain(..call_end_abs);

                        return Ok(StreamResult::ToolComplete(tool));
                    }
                } else {
                    // Tool call not complete yet, try to extract partial info
                    let partial = &state.buffer[search_end_from..];

                    // Try to extract function name
                    if let Some(sep_pos) = partial.find("<｜tool▁sep｜>") {
                        if let Some(_func_start) = partial[..sep_pos].rfind("function") {
                            // We have the function type marker
                            let after_sep = &partial[sep_pos + "<｜tool▁sep｜>".len()..];

                            // Look for function name (ends at newline before ```json)
                            if let Some(name_end) = after_sep.find("\n```json\n") {
                                let func_name = after_sep[..name_end].trim();

                                if !state.in_string {
                                    state.in_string = true; // Mark name as sent
                                    return Ok(StreamResult::ToolName {
                                        index: 0,
                                        name: func_name.to_string(),
                                    });
                                }

                                // Try to extract partial arguments
                                let args_start = name_end + "\n```json\n".len();
                                let partial_args = &after_sep[args_start..];

                                // Check if we can parse partial JSON
                                if !partial_args.is_empty() {
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
                                            // Can't parse yet, continue waiting for more data
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(StreamResult::Incomplete)
    }

    fn detect_format(&self, text: &str) -> bool {
        self.has_tool_markers(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_parse_deepseek_single_tool() {
        let parser = DeepSeekParser::new();
        let input = r#"Some text
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
```json
{"location": "Tokyo", "units": "celsius"}
```<｜tool▁call▁end｜><｜tool▁calls▁end｜>More text"#;

        let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "get_weather");
        assert!(tools[0].function.arguments.contains("Tokyo"));
    }

    #[tokio::test]
    async fn test_parse_deepseek_multiple_tools() {
        let parser = DeepSeekParser::new();
        let input = r#"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
```json
{"location": "Tokyo"}
```<｜tool▁call▁end｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
```json
{"location": "Paris"}
```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"#;

        let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].function.name, "get_weather");
        assert_eq!(tools[1].function.name, "get_weather");
        assert!(tools[0].function.arguments.contains("Tokyo"));
        assert!(tools[1].function.arguments.contains("Paris"));
    }

    #[test]
    fn test_detect_format() {
        let parser = DeepSeekParser::new();
        assert!(parser.detect_format("<｜tool▁calls▁begin｜>"));
        assert!(!parser.detect_format("plain text"));
        assert!(!parser.detect_format("[TOOL_CALLS]"));
    }
}
