use async_trait::async_trait;
use regex::Regex;
use serde_json::Value;

use crate::tool_parser::{
    errors::{ToolParserError, ToolParserResult},
    state::ParseState,
    traits::ToolParser,
    types::{FunctionCall, StreamResult, ToolCall},
};

/// GLM-4 MoE format parser for tool calls
///
/// Handles the GLM-4 MoE specific format:
/// `<tool_call>{name}\n<arg_key>{key}</arg_key>\n<arg_value>{value}</arg_value>\n</tool_call>`
///
/// Features:
/// - XML-style tags for tool calls
/// - Key-value pairs for arguments
/// - Support for multiple sequential tool calls
pub struct Glm4MoeParser {
    /// Regex for extracting complete tool calls
    tool_call_extractor: Regex,
    /// Regex for extracting function details
    func_detail_extractor: Regex,
    /// Regex for extracting argument key-value pairs
    arg_extractor: Regex,
}

impl Glm4MoeParser {
    /// Create a new GLM-4 MoE parser
    pub fn new() -> Self {
        // Use (?s) flag for DOTALL mode to handle newlines
        let tool_call_pattern = r"(?s)<tool_call>.*?</tool_call>";
        let tool_call_extractor = Regex::new(tool_call_pattern).expect("Valid regex pattern");

        let func_detail_pattern = r"(?s)<tool_call>([^\n]*)\n(.*)</tool_call>";
        let func_detail_extractor = Regex::new(func_detail_pattern).expect("Valid regex pattern");

        let arg_pattern = r"(?s)<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>";
        let arg_extractor = Regex::new(arg_pattern).expect("Valid regex pattern");

        Self {
            tool_call_extractor,
            func_detail_extractor,
            arg_extractor,
        }
    }

    /// Check if text contains GLM-4 MoE tool markers
    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains("<tool_call>")
    }

    /// Parse arguments from key-value pairs
    fn parse_arguments(&self, args_text: &str) -> ToolParserResult<serde_json::Map<String, Value>> {
        let mut arguments = serde_json::Map::new();

        for capture in self.arg_extractor.captures_iter(args_text) {
            let key = capture.get(1).map_or("", |m| m.as_str()).trim();
            let value_str = capture.get(2).map_or("", |m| m.as_str()).trim();

            // Try to parse the value as JSON first, fallback to string
            let value = if let Ok(json_val) = serde_json::from_str::<Value>(value_str) {
                json_val
            } else {
                // Try parsing as Python literal (similar to Python's ast.literal_eval)
                if value_str == "true" || value_str == "True" {
                    Value::Bool(true)
                } else if value_str == "false" || value_str == "False" {
                    Value::Bool(false)
                } else if value_str == "null" || value_str == "None" {
                    Value::Null
                } else if let Ok(num) = value_str.parse::<i64>() {
                    Value::Number(num.into())
                } else if let Ok(num) = value_str.parse::<f64>() {
                    if let Some(n) = serde_json::Number::from_f64(num) {
                        Value::Number(n)
                    } else {
                        Value::String(value_str.to_string())
                    }
                } else {
                    Value::String(value_str.to_string())
                }
            };

            arguments.insert(key.to_string(), value);
        }

        Ok(arguments)
    }

    /// Parse a single tool call block
    fn parse_tool_call(&self, block: &str) -> ToolParserResult<Option<ToolCall>> {
        if let Some(captures) = self.func_detail_extractor.captures(block) {
            // Get function name
            let func_name = captures.get(1).map_or("", |m| m.as_str()).trim();

            // Get arguments text
            let args_text = captures.get(2).map_or("", |m| m.as_str());

            // Parse arguments
            let arguments = self.parse_arguments(args_text)?;

            let arguments_str = serde_json::to_string(&arguments)
                .map_err(|e| ToolParserError::ParsingFailed(e.to_string()))?;

            // Generate ID
            let id = format!("glm4_call_{}", uuid::Uuid::new_v4());

            Ok(Some(ToolCall {
                id,
                r#type: "function".to_string(),
                function: FunctionCall {
                    name: func_name.to_string(),
                    arguments: arguments_str,
                },
            }))
        } else {
            Ok(None)
        }
    }
}

impl Default for Glm4MoeParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for Glm4MoeParser {
    async fn parse_complete(&self, text: &str) -> ToolParserResult<Vec<ToolCall>> {
        // Check if text contains GLM-4 MoE format
        if !self.has_tool_markers(text) {
            return Ok(vec![]);
        }

        // Extract all tool call blocks
        let mut tools = Vec::new();
        for mat in self.tool_call_extractor.find_iter(text) {
            if let Some(tool) = self.parse_tool_call(mat.as_str())? {
                tools.push(tool);
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
            // No markers found, return as incomplete
            return Ok(StreamResult::Incomplete);
        }

        // Look for start of tool call
        if let Some(start_pos) = state.buffer.find("<tool_call>") {
            // Look for the end of this tool call
            let search_from = start_pos + "<tool_call>".len();
            if let Some(end_pos) = state.buffer[search_from..].find("</tool_call>") {
                let end_abs = search_from + end_pos + "</tool_call>".len();

                // Extract and parse the complete tool call
                let tool_call_text = &state.buffer[start_pos..end_abs];

                if let Some(tool) = self.parse_tool_call(tool_call_text)? {
                    // Remove the processed part from buffer
                    state.buffer.drain(..end_abs);

                    return Ok(StreamResult::ToolComplete(tool));
                }
            } else {
                // Tool call not complete yet, try to extract partial info
                let partial = &state.buffer[search_from..];

                // Try to extract function name (first line after <tool_call>)
                if let Some(name_end) = partial.find('\n') {
                    let func_name = partial[..name_end].trim();

                    if !func_name.is_empty() && !state.in_string {
                        state.in_string = true; // Mark name as sent
                        return Ok(StreamResult::ToolName {
                            index: 0,
                            name: func_name.to_string(),
                        });
                    }

                    // Try to extract partial arguments
                    let args_text = &partial[name_end + 1..];
                    let partial_args = self.parse_arguments(args_text)?;

                    if !partial_args.is_empty() {
                        let args_str = serde_json::to_string(&partial_args)
                            .unwrap_or_else(|_| "{}".to_string());

                        return Ok(StreamResult::ToolArguments {
                            index: 0,
                            arguments: args_str,
                        });
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
    async fn test_parse_glm4_single_tool() {
        let parser = Glm4MoeParser::new();
        let input = r#"Some text
<tool_call>get_weather
<arg_key>city</arg_key>
<arg_value>Beijing</arg_value>
<arg_key>date</arg_key>
<arg_value>2024-06-27</arg_value>
</tool_call>More text"#;

        let result = parser.parse_complete(input).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].function.name, "get_weather");
        assert!(result[0].function.arguments.contains("Beijing"));
        assert!(result[0].function.arguments.contains("2024-06-27"));
    }

    #[tokio::test]
    async fn test_parse_glm4_multiple_tools() {
        let parser = Glm4MoeParser::new();
        let input = r#"<tool_call>get_weather
<arg_key>city</arg_key>
<arg_value>Beijing</arg_value>
</tool_call>
<tool_call>get_weather
<arg_key>city</arg_key>
<arg_value>Shanghai</arg_value>
</tool_call>"#;

        let result = parser.parse_complete(input).await.unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].function.name, "get_weather");
        assert_eq!(result[1].function.name, "get_weather");
        assert!(result[0].function.arguments.contains("Beijing"));
        assert!(result[1].function.arguments.contains("Shanghai"));
    }

    #[tokio::test]
    async fn test_parse_glm4_mixed_types() {
        let parser = Glm4MoeParser::new();
        let input = r#"<tool_call>process_data
<arg_key>count</arg_key>
<arg_value>42</arg_value>
<arg_key>active</arg_key>
<arg_value>true</arg_value>
<arg_key>name</arg_key>
<arg_value>test</arg_value>
</tool_call>"#;

        let result = parser.parse_complete(input).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].function.name, "process_data");

        // Parse arguments to check types
        let args: serde_json::Value = serde_json::from_str(&result[0].function.arguments).unwrap();
        assert_eq!(args["count"], 42);
        assert_eq!(args["active"], true);
        assert_eq!(args["name"], "test");
    }

    #[test]
    fn test_detect_format() {
        let parser = Glm4MoeParser::new();
        assert!(parser.detect_format("<tool_call>"));
        assert!(!parser.detect_format("plain text"));
        assert!(!parser.detect_format("[TOOL_CALLS]"));
    }
}
