use async_trait::async_trait;
use regex::Regex;
use serde_json::Value;

use crate::tool_parser::{
    errors::{ToolParserError, ToolParserResult},
    state::ParseState,
    traits::ToolParser,
    types::{FunctionCall, StreamResult, ToolCall},
};

/// Step3 format parser for tool calls
///
/// Handles the Step3 specific format with steptml XML:
/// `<｜tool_calls_begin｜><｜tool_call_begin｜>function<｜tool_sep｜><steptml:invoke name="{name}"><steptml:parameter name="{k}">{v}</steptml:parameter></steptml:invoke><｜tool_call_end｜><｜tool_calls_end｜>`
///
/// Features:
/// - Unicode token delimiters
/// - StepTML XML format for invocations
/// - Support for multiple sequential tool calls
pub struct Step3Parser {
    /// Regex for extracting tool call blocks
    tool_call_extractor: Regex,
    /// Regex for extracting steptml invocations
    invoke_extractor: Regex,
    /// Regex for extracting parameters
    param_extractor: Regex,
}

impl Step3Parser {
    /// Create a new Step3 parser
    pub fn new() -> Self {
        // Pattern for individual tool calls
        let tool_call_pattern = r"(?s)<｜tool_call_begin｜>.*?<｜tool_call_end｜>";
        let tool_call_extractor = Regex::new(tool_call_pattern).expect("Valid regex pattern");

        // Pattern for steptml invocations
        let invoke_pattern = r#"(?s)<steptml:invoke name="([^"]+)">(.+?)</steptml:invoke>"#;
        let invoke_extractor = Regex::new(invoke_pattern).expect("Valid regex pattern");

        // Pattern for steptml parameters - using non-greedy match for values to handle < characters
        let param_pattern = r#"(?s)<steptml:parameter name="([^"]+)">(.+?)</steptml:parameter>"#;
        let param_extractor = Regex::new(param_pattern).expect("Valid regex pattern");

        Self {
            tool_call_extractor,
            invoke_extractor,
            param_extractor,
        }
    }

    /// Check if text contains Step3 tool markers
    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains("<｜tool_calls_begin｜>")
    }

    /// Parse parameters from steptml format
    fn parse_steptml_parameters(
        &self,
        params_text: &str,
    ) -> ToolParserResult<serde_json::Map<String, Value>> {
        let mut parameters = serde_json::Map::new();

        for capture in self.param_extractor.captures_iter(params_text) {
            let param_name = capture.get(1).map_or("", |m| m.as_str()).trim();
            let param_value_str = capture.get(2).map_or("", |m| m.as_str()).trim();

            // Try to parse the value as JSON first, fallback to string
            let param_value = if let Ok(json_val) = serde_json::from_str::<Value>(param_value_str) {
                json_val
            } else {
                // Try parsing as Python literal
                if param_value_str == "true" || param_value_str == "True" {
                    Value::Bool(true)
                } else if param_value_str == "false" || param_value_str == "False" {
                    Value::Bool(false)
                } else if param_value_str == "null" || param_value_str == "None" {
                    Value::Null
                } else if let Ok(num) = param_value_str.parse::<i64>() {
                    Value::Number(num.into())
                } else if let Ok(num) = param_value_str.parse::<f64>() {
                    if let Some(n) = serde_json::Number::from_f64(num) {
                        Value::Number(n)
                    } else {
                        Value::String(param_value_str.to_string())
                    }
                } else {
                    Value::String(param_value_str.to_string())
                }
            };

            parameters.insert(param_name.to_string(), param_value);
        }

        Ok(parameters)
    }

    /// Parse a single tool call block
    fn parse_tool_call(&self, block: &str) -> ToolParserResult<Option<ToolCall>> {
        // Check if it contains function marker and tool separator
        if !block.contains("function") || !block.contains("<｜tool_sep｜>") {
            return Ok(None);
        }

        // Split by tool separator
        let parts: Vec<&str> = block.split("<｜tool_sep｜>").collect();
        if parts.len() != 2 {
            return Ok(None);
        }

        // Check if it's a function type
        if !parts[0].contains("function") {
            return Ok(None);
        }

        let invoke_part = parts[1];

        // Extract steptml invoke
        if let Some(captures) = self.invoke_extractor.captures(invoke_part) {
            let func_name = captures.get(1).map_or("", |m| m.as_str()).trim();

            // Validate function name is not empty
            if func_name.is_empty() {
                return Ok(None);
            }

            let params_text = captures.get(2).map_or("", |m| m.as_str());

            // Parse parameters
            let parameters = self.parse_steptml_parameters(params_text)?;

            let arguments_str = serde_json::to_string(&parameters)
                .map_err(|e| ToolParserError::ParsingFailed(e.to_string()))?;

            // Generate ID
            let id = format!("step3_call_{}", uuid::Uuid::new_v4());

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

impl Default for Step3Parser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for Step3Parser {
    async fn parse_complete(&self, text: &str) -> ToolParserResult<Vec<ToolCall>> {
        // Check if text contains Step3 format
        if !self.has_tool_markers(text) {
            return Ok(vec![]);
        }

        // Find the tool calls section
        if let Some(start_pos) = text.find("<｜tool_calls_begin｜>") {
            let search_from = start_pos + "<｜tool_calls_begin｜>".len();

            // Find the end of tool calls section
            if let Some(end_pos) = text[search_from..].find("<｜tool_calls_end｜>") {
                let tool_section = &text[search_from..search_from + end_pos];

                // Extract all tool call blocks
                let mut tools = Vec::new();
                for mat in self.tool_call_extractor.find_iter(tool_section) {
                    if let Some(tool) = self.parse_tool_call(mat.as_str())? {
                        tools.push(tool);
                    }
                }

                return Ok(tools);
            }
        }

        Ok(vec![])
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

        // Look for start of tool calls
        if let Some(start_pos) = state.buffer.find("<｜tool_calls_begin｜>") {
            let search_from = start_pos + "<｜tool_calls_begin｜>".len();

            // Look for individual tool call start
            if let Some(call_start) = state.buffer[search_from..].find("<｜tool_call_begin｜>") {
                let call_start_abs = search_from + call_start;

                // Look for the end of this tool call
                let search_end_from = call_start_abs + "<｜tool_call_begin｜>".len();
                if let Some(call_end) = state.buffer[search_end_from..].find("<｜tool_call_end｜>")
                {
                    let call_end_abs = search_end_from + call_end + "<｜tool_call_end｜>".len();

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

                    // Check for tool separator
                    if let Some(sep_pos) = partial.find("<｜tool_sep｜>") {
                        // Check if it's a function
                        if partial[..sep_pos].contains("function") {
                            let after_sep = &partial[sep_pos + "<｜tool_sep｜>".len()..];

                            // Try to extract function name from steptml:invoke
                            if let Some(name_match) = self.invoke_extractor.captures(after_sep) {
                                let func_name = name_match.get(1).map_or("", |m| m.as_str()).trim();

                                if !state.in_string && !func_name.is_empty() {
                                    state.in_string = true; // Mark name as sent
                                    return Ok(StreamResult::ToolName {
                                        index: 0,
                                        name: func_name.to_string(),
                                    });
                                }

                                // Try to extract partial parameters
                                if let Some(params_text) = name_match.get(2) {
                                    let parameters =
                                        self.parse_steptml_parameters(params_text.as_str())?;

                                    if !parameters.is_empty() {
                                        let args_str = serde_json::to_string(&parameters)
                                            .unwrap_or_else(|_| "{}".to_string());

                                        return Ok(StreamResult::ToolArguments {
                                            index: 0,
                                            arguments: args_str,
                                        });
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
    async fn test_parse_step3_single_tool() {
        let parser = Step3Parser::new();
        let input = r#"Some text
<｜tool_calls_begin｜>
<｜tool_call_begin｜>function<｜tool_sep｜><steptml:invoke name="get_weather">
<steptml:parameter name="location">Tokyo</steptml:parameter>
<steptml:parameter name="units">celsius</steptml:parameter>
</steptml:invoke><｜tool_call_end｜>
<｜tool_calls_end｜>More text"#;

        let result = parser.parse_complete(input).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].function.name, "get_weather");
        assert!(result[0].function.arguments.contains("Tokyo"));
        assert!(result[0].function.arguments.contains("celsius"));
    }

    #[tokio::test]
    async fn test_parse_step3_multiple_tools() {
        let parser = Step3Parser::new();
        let input = r#"<｜tool_calls_begin｜>
<｜tool_call_begin｜>function<｜tool_sep｜><steptml:invoke name="search">
<steptml:parameter name="query">rust programming</steptml:parameter>
</steptml:invoke><｜tool_call_end｜>
<｜tool_call_begin｜>function<｜tool_sep｜><steptml:invoke name="calculate">
<steptml:parameter name="expression">2 + 2</steptml:parameter>
</steptml:invoke><｜tool_call_end｜>
<｜tool_calls_end｜>"#;

        let result = parser.parse_complete(input).await.unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].function.name, "search");
        assert_eq!(result[1].function.name, "calculate");
    }

    #[tokio::test]
    async fn test_parse_step3_mixed_types() {
        let parser = Step3Parser::new();
        let input = r#"<｜tool_calls_begin｜>
<｜tool_call_begin｜>function<｜tool_sep｜><steptml:invoke name="process_data">
<steptml:parameter name="count">42</steptml:parameter>
<steptml:parameter name="active">true</steptml:parameter>
<steptml:parameter name="rate">1.5</steptml:parameter>
<steptml:parameter name="name">test</steptml:parameter>
</steptml:invoke><｜tool_call_end｜>
<｜tool_calls_end｜>"#;

        let result = parser.parse_complete(input).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].function.name, "process_data");

        // Parse arguments to check types
        let args: serde_json::Value = serde_json::from_str(&result[0].function.arguments).unwrap();
        assert_eq!(args["count"], 42);
        assert_eq!(args["active"], true);
        assert_eq!(args["rate"], 1.5);
        assert_eq!(args["name"], "test");
    }

    #[test]
    fn test_detect_format() {
        let parser = Step3Parser::new();
        assert!(parser.detect_format("<｜tool_calls_begin｜>"));
        assert!(!parser.detect_format("plain text"));
        assert!(!parser.detect_format("[TOOL_CALLS]"));
    }
}
