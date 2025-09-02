/// Pythonic format parser for tool calls
///
/// Handles Python function call syntax within square brackets:
/// ```text
/// [tool1(arg1=val1, arg2=val2), tool2(arg1=val3)]
/// ```
///
/// This format is used by Llama-4 models and uses Python literals
/// rather than JSON for arguments.
use async_trait::async_trait;
use regex::Regex;
use serde_json::{json, Value};

use crate::tool_parser::{
    errors::ToolParserResult,
    python_literal_parser::parse_python_literal,
    state::ParseState,
    traits::ToolParser,
    types::{FunctionCall, StreamResult, ToolCall},
};

/// Parser for Pythonic tool call format
pub struct PythonicParser {
    /// Regex to detect tool calls in Pythonic format
    tool_call_regex: Regex,
    /// Regex to parse function calls - cached for reuse
    call_regex: Regex,
}

impl PythonicParser {
    /// Create a new Pythonic parser
    pub fn new() -> Self {
        // Simple regex to detect start of Pythonic tool calls
        // We'll use manual parsing for the actual extraction
        let pattern = r"\[[a-zA-Z_]\w*\(";
        let tool_call_regex = Regex::new(pattern).expect("Valid regex pattern");

        // Compile the function call regex once
        let call_regex = Regex::new(r"(?s)^([a-zA-Z_]\w*)\((.*)\)$").expect("Valid regex pattern");

        Self {
            tool_call_regex,
            call_regex,
        }
    }

    /// Extract tool calls using bracket counting (similar to MistralParser)
    fn extract_tool_calls(&self, text: &str) -> Option<String> {
        // Find the start of a tool call list - look for [ followed by a function name
        let chars: Vec<char> = text.chars().collect();

        for start_idx in 0..chars.len() {
            if chars[start_idx] != '[' {
                continue;
            }

            // Check if this looks like a tool call
            // Skip whitespace after [
            let mut check_idx = start_idx + 1;
            while check_idx < chars.len() && chars[check_idx].is_whitespace() {
                check_idx += 1;
            }

            // Check if we have a function name (starts with letter or underscore)
            if check_idx >= chars.len()
                || (!chars[check_idx].is_alphabetic() && chars[check_idx] != '_')
            {
                continue;
            }

            // Now count brackets to find the matching ]
            let mut bracket_count = 0;
            let mut _paren_count = 0;
            let mut _brace_count = 0;
            let mut in_string = false;
            let mut string_char = ' ';
            let mut escape_next = false;

            for i in start_idx..chars.len() {
                let ch = chars[i];

                if escape_next {
                    escape_next = false;
                    continue;
                }

                if ch == '\\' && in_string {
                    escape_next = true;
                    continue;
                }

                if !in_string && (ch == '"' || ch == '\'') {
                    in_string = true;
                    string_char = ch;
                } else if in_string && ch == string_char && !escape_next {
                    in_string = false;
                } else if !in_string {
                    match ch {
                        '[' => bracket_count += 1,
                        ']' => {
                            bracket_count -= 1;
                            if bracket_count == 0 {
                                // Found the matching bracket
                                let extracted: String = chars[start_idx..=i].iter().collect();
                                // Verify this actually contains a function call
                                if extracted.contains('(') && extracted.contains(')') {
                                    return Some(extracted);
                                }
                            }
                        }
                        '(' => _paren_count += 1,
                        ')' => _paren_count -= 1,
                        '{' => _brace_count += 1,
                        '}' => _brace_count -= 1,
                        _ => {}
                    }
                }
            }
        }
        None
    }

    /// Strip special tokens that Llama 4 might output
    fn strip_special_tokens(text: &str) -> String {
        text.replace("<|python_start|>", "")
            .replace("<|python_end|>", "")
    }

    /// Parse a single function call from Python syntax
    fn parse_function_call(&self, call_str: &str) -> ToolParserResult<Option<ToolCall>> {
        // Use cached regex instead of creating new one
        if let Some(captures) = self.call_regex.captures(call_str.trim()) {
            let function_name = captures.get(1).unwrap().as_str();
            let args_str = captures.get(2).unwrap().as_str();

            // Parse arguments
            let arguments = self.parse_arguments(args_str)?;

            Ok(Some(ToolCall {
                id: format!("call_{}", uuid::Uuid::new_v4()),
                r#type: "function".to_string(),
                function: FunctionCall {
                    name: function_name.to_string(),
                    arguments: serde_json::to_string(&arguments)?,
                },
            }))
        } else {
            Ok(None)
        }
    }

    /// Parse Python-style arguments into JSON
    fn parse_arguments(&self, args_str: &str) -> ToolParserResult<Value> {
        if args_str.trim().is_empty() {
            return Ok(json!({}));
        }

        let mut result = serde_json::Map::new();
        let mut current_key = String::new();
        let mut current_value = String::new();
        let mut in_key = true;
        let mut depth = 0;
        let mut in_string = false;
        let mut string_char = ' ';
        let mut escape_next = false;

        let chars: Vec<char> = args_str.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let ch = chars[i];

            if escape_next {
                if in_key {
                    current_key.push(ch);
                } else {
                    current_value.push(ch);
                }
                escape_next = false;
                i += 1;
                continue;
            }

            if ch == '\\' && in_string {
                escape_next = true;
                current_value.push(ch);
                i += 1;
                continue;
            }

            // Handle string literals
            if !in_string && (ch == '"' || ch == '\'') {
                in_string = true;
                string_char = ch;
                if !in_key {
                    current_value.push(ch);
                }
            } else if in_string && ch == string_char && !escape_next {
                in_string = false;
                if !in_key {
                    current_value.push(ch);
                }
            } else if in_string {
                if in_key {
                    current_key.push(ch);
                } else {
                    current_value.push(ch);
                }
            } else {
                // Not in string
                match ch {
                    '=' if in_key && depth == 0 => {
                        in_key = false;
                    }
                    ',' if depth == 0 => {
                        // End of current argument
                        if !current_key.is_empty() {
                            let value = parse_python_literal(current_value.trim())?;
                            result.insert(current_key.trim().to_string(), value);
                        }
                        current_key.clear();
                        current_value.clear();
                        in_key = true;
                    }
                    '[' | '{' | '(' => {
                        depth += 1;
                        if !in_key {
                            current_value.push(ch);
                        }
                    }
                    ']' | '}' | ')' => {
                        depth -= 1;
                        if !in_key {
                            current_value.push(ch);
                        }
                    }
                    _ => {
                        if in_key {
                            if !ch.is_whitespace() || !current_key.is_empty() {
                                current_key.push(ch);
                            }
                        } else {
                            current_value.push(ch);
                        }
                    }
                }
            }

            i += 1;
        }

        // Handle the last argument
        if !current_key.is_empty() {
            let value = parse_python_literal(current_value.trim())?;
            result.insert(current_key.trim().to_string(), value);
        }

        Ok(Value::Object(result))
    }
}

#[async_trait]
impl ToolParser for PythonicParser {
    async fn parse_complete(&self, text: &str) -> ToolParserResult<Vec<ToolCall>> {
        let cleaned = Self::strip_special_tokens(text);

        // Extract tool calls using bracket counting
        if let Some(tool_calls_text) = self.extract_tool_calls(&cleaned) {
            // Remove the outer brackets
            let tool_calls_str = &tool_calls_text[1..tool_calls_text.len() - 1];

            // Split into individual function calls
            let mut calls = Vec::new();
            let mut current_call = String::new();
            let mut paren_depth = 0;
            let mut in_string = false;
            let mut string_char = ' ';

            for ch in tool_calls_str.chars() {
                if !in_string && (ch == '"' || ch == '\'') {
                    in_string = true;
                    string_char = ch;
                    current_call.push(ch);
                } else if in_string && ch == string_char {
                    in_string = false;
                    current_call.push(ch);
                } else if in_string {
                    current_call.push(ch);
                } else {
                    match ch {
                        '(' => {
                            paren_depth += 1;
                            current_call.push(ch);
                        }
                        ')' => {
                            paren_depth -= 1;
                            current_call.push(ch);
                        }
                        ',' if paren_depth == 0 => {
                            // End of current function call
                            if let Some(call) = self.parse_function_call(current_call.trim())? {
                                calls.push(call);
                            }
                            current_call.clear();
                        }
                        _ => {
                            if !ch.is_whitespace() || !current_call.is_empty() {
                                current_call.push(ch);
                            }
                        }
                    }
                }
            }

            // Handle the last call (important for single calls or the last call in a list)
            if !current_call.trim().is_empty() {
                if let Some(call) = self.parse_function_call(current_call.trim())? {
                    calls.push(call);
                }
            }

            Ok(calls)
        } else {
            Ok(vec![])
        }
    }

    async fn parse_incremental(
        &self,
        chunk: &str,
        state: &mut ParseState,
    ) -> ToolParserResult<StreamResult> {
        // For Pythonic format, we accumulate until we have a complete tool call
        // This is a simplified implementation
        state.buffer.push_str(chunk);

        // Try to parse if we have a complete tool call
        let cleaned = Self::strip_special_tokens(&state.buffer);
        if self.extract_tool_calls(&cleaned).is_some() {
            let result = self.parse_complete(&state.buffer).await?;
            if !result.is_empty() {
                state.buffer.clear();
                return Ok(StreamResult::ToolComplete(
                    result.into_iter().next().unwrap(),
                ));
            }
        }

        Ok(StreamResult::Incomplete)
    }

    fn detect_format(&self, text: &str) -> bool {
        let cleaned = Self::strip_special_tokens(text);
        self.tool_call_regex.is_match(&cleaned)
    }
}

impl Default for PythonicParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_single_function_call() {
        let parser = PythonicParser::new();
        let input = r#"[search_web(query="Rust programming", max_results=5)]"#;

        let result = parser.parse_complete(input).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].function.name, "search_web");

        let args: Value = serde_json::from_str(&result[0].function.arguments).unwrap();
        assert_eq!(args["query"], "Rust programming");
        assert_eq!(args["max_results"], 5);
    }

    #[tokio::test]
    async fn test_multiple_function_calls() {
        let parser = PythonicParser::new();
        let input = r#"[get_weather(city="Tokyo"), search(query="news")]"#;

        let result = parser.parse_complete(input).await.unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].function.name, "get_weather");
        assert_eq!(result[1].function.name, "search");
    }

    #[tokio::test]
    async fn test_python_literals() {
        let parser = PythonicParser::new();
        let input = r#"[test(flag=True, disabled=False, optional=None)]"#;

        let result = parser.parse_complete(input).await.unwrap();
        assert_eq!(result.len(), 1);

        let args: Value = serde_json::from_str(&result[0].function.arguments).unwrap();
        assert_eq!(args["flag"], true);
        assert_eq!(args["disabled"], false);
        assert_eq!(args["optional"], Value::Null);
    }

    #[tokio::test]
    async fn test_special_tokens() {
        let parser = PythonicParser::new();
        let input = r#"<|python_start|>[calculate(x=10, y=20)]<|python_end|>"#;

        let result = parser.parse_complete(input).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].function.name, "calculate");

        let args: Value = serde_json::from_str(&result[0].function.arguments).unwrap();
        assert_eq!(args["x"], 10);
        assert_eq!(args["y"], 20);
    }

    #[tokio::test]
    async fn test_llama4_format() {
        let parser = PythonicParser::new();
        let input = r#"[get_weather(city="London", units="celsius")]"#;

        let result = parser.parse_complete(input).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].function.name, "get_weather");

        let args: Value = serde_json::from_str(&result[0].function.arguments).unwrap();
        assert_eq!(args["city"], "London");
        assert_eq!(args["units"], "celsius");
    }
}
