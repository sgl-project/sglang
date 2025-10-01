use async_trait::async_trait;
use serde_json::Value;

use crate::tool_parser::{
    errors::{ToolParserError, ToolParserResult},
    partial_json::PartialJson,
    state::ParseState,
    traits::ToolParser,
    types::{FunctionCall, StreamResult, ToolCall},
};

/// JSON format parser for tool calls
///
/// Handles pure JSON formats for function calling:
/// - Single tool call: {"name": "fn", "arguments": {...}}
/// - Multiple tool calls: [{"name": "fn1", "arguments": {...}}, ...]
/// - With parameters instead of arguments: {"name": "fn", "parameters": {...}}
pub struct JsonParser {
    /// Parser for handling incomplete JSON during streaming
    partial_json: PartialJson,
}

impl JsonParser {
    /// Create a new JSON parser
    pub fn new() -> Self {
        Self {
            partial_json: PartialJson::default(),
        }
    }

    /// Try to extract a first valid JSON object or array from text that may contain other content
    /// Returns (json_string, normal_text) where normal_text is text before and after the JSON
    fn extract_json_from_text(&self, text: &str) -> Option<(String, String)> {
        let mut in_string = false;
        let mut escape = false;
        let mut stack: Vec<char> = Vec::with_capacity(8);
        let mut start: Option<usize> = None;

        for (i, ch) in text.char_indices() {
            if escape {
                escape = false;
                continue;
            }

            match ch {
                '\\' if in_string => escape = true,
                '"' => in_string = !in_string,
                _ if in_string => {}
                '{' | '[' => {
                    if start.is_none() {
                        start = Some(i);
                    }
                    stack.push(ch);
                }
                '}' | ']' => {
                    let Some(open) = stack.pop() else {
                        // Stray closer - reset and continue looking for next valid JSON
                        start = None;
                        continue;
                    };

                    let valid = (open == '{' && ch == '}') || (open == '[' && ch == ']');
                    if !valid {
                        // Mismatch - reset and continue looking
                        start = None;
                        stack.clear();
                        continue;
                    }

                    if stack.is_empty() {
                        let s = start.unwrap();
                        let e = i + ch.len_utf8();
                        let potential_json = &text[s..e];

                        // Validate that this is actually valid JSON before returning
                        if serde_json::from_str::<Value>(potential_json).is_ok() {
                            let json = potential_json.to_string();
                            let normal = format!("{}{}", &text[..s], &text[e..]);
                            return Some((json, normal));
                        } else {
                            // Not valid JSON, reset and continue looking
                            start = None;
                            continue;
                        }
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// Parse a single JSON object into a ToolCall
    fn parse_single_object(&self, obj: &Value) -> ToolParserResult<Option<ToolCall>> {
        // Check if this looks like a tool call
        let name = obj
            .get("name")
            .or_else(|| obj.get("function"))
            .and_then(|v| v.as_str());

        if let Some(name) = name {
            // Get arguments - support both "arguments" and "parameters" keys
            let empty_obj = Value::Object(serde_json::Map::new());
            let args = obj
                .get("arguments")
                .or_else(|| obj.get("parameters"))
                .unwrap_or(&empty_obj);

            // Convert arguments to JSON string
            let arguments = serde_json::to_string(args)
                .map_err(|e| ToolParserError::ParsingFailed(e.to_string()))?;

            // Generate a unique ID if not provided
            let id = obj
                .get("id")
                .and_then(|v| v.as_str())
                .map(String::from)
                .unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4()));

            Ok(Some(ToolCall {
                id,
                r#type: "function".to_string(),
                function: FunctionCall {
                    name: name.to_string(),
                    arguments,
                },
            }))
        } else {
            Ok(None)
        }
    }

    /// Parse JSON value(s) into tool calls
    fn parse_json_value(&self, value: &Value) -> ToolParserResult<Vec<ToolCall>> {
        let mut tools = Vec::new();

        match value {
            Value::Array(arr) => {
                // Parse each element in the array
                for item in arr {
                    if let Some(tool) = self.parse_single_object(item)? {
                        tools.push(tool);
                    }
                }
            }
            Value::Object(_) => {
                // Single tool call
                if let Some(tool) = self.parse_single_object(value)? {
                    tools.push(tool);
                }
            }
            _ => {
                // Not a valid tool call format
                return Ok(vec![]);
            }
        }

        Ok(tools)
    }

    /// Check if text contains JSON tool call markers (complete markers)
    fn has_tool_markers(&self, text: &str) -> bool {
        (text.contains('{') || text.contains('[')) && text.contains("name")
    }

    /// Check if buffer could be building toward a tool call pattern
    fn has_partial_start_token(&self, buffer: &str) -> bool {
        // Check if buffer ends with a partial match of tool call patterns
        let patterns = [r#"{"name""#, r#"[{"name""#];

        for pattern in &patterns {
            // Check if buffer ends with any partial of this pattern
            for i in 1..=buffer.len().min(pattern.len()) {
                if pattern.starts_with(&buffer[buffer.len() - i..]) {
                    return true;
                }
            }
        }
        false
    }
}

impl Default for JsonParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for JsonParser {
    async fn parse_complete(&self, text: &str) -> ToolParserResult<(String, Vec<ToolCall>)> {
        // Always use extract_json_from_text to handle both pure JSON and mixed content
        if let Some((extracted_json, normal_text)) = self.extract_json_from_text(text) {
            let parsed = serde_json::from_str::<Value>(&extracted_json)
                .map_err(|e| ToolParserError::ParsingFailed(e.to_string()))
                .and_then(|v| self.parse_json_value(&v));

            match parsed {
                Ok(tools) => return Ok((normal_text, tools)),
                Err(e) => tracing::warn!("parse_complete failed: {:?}", e),
            }
        }

        // No valid JSON found, return original text as normal text
        Ok((text.to_string(), vec![]))
    }

    async fn parse_incremental(
        &self,
        chunk: &str,
        state: &mut ParseState,
    ) -> ToolParserResult<StreamResult> {
        state.buffer.push_str(chunk);
        let trimmed = state.buffer.trim();

        // If no tool markers and not a partial token, return as normal text                                                                                                                                                                                        │ │
        if !self.has_tool_markers(trimmed) && !self.has_partial_start_token(trimmed) {
            let normal_text = std::mem::take(&mut state.buffer);
            return Ok(StreamResult::NormalText(normal_text));
        }

        // Try to parse with partial JSON parser
        match self.partial_json.parse_value(trimmed) {
            Ok((value, consumed)) => {
                // Check if we have a complete JSON structure
                if consumed == trimmed.len() {
                    // Check if this is truly complete
                    let looks_complete = trimmed.ends_with('}') || trimmed.ends_with(']');

                    if looks_complete {
                        // Complete JSON, parse tool calls
                        let tools = self.parse_json_value(&value)?;
                        if !tools.is_empty() {
                            // Clear buffer since we consumed everything
                            state.buffer.clear();

                            // Return the first tool as complete
                            // TODO simplified version, address more complex version
                            if let Some(tool) = tools.into_iter().next() {
                                return Ok(StreamResult::ToolComplete(tool));
                            }
                        }
                    }
                } else {
                    // Partial JSON, try to extract tool name
                    if let Some(name) = value.get("name").and_then(|v| v.as_str()) {
                        // TODO simplified version, address more complex version
                        // Just return the tool name once we see it
                        if !state.in_string {
                            state.in_string = true; // Use as a flag for "name sent"
                            return Ok(StreamResult::ToolName {
                                index: 0,
                                name: name.to_string(),
                            });
                        }

                        // Check for complete arguments
                        if let Some(args) =
                            value.get("arguments").or_else(|| value.get("parameters"))
                        {
                            if let Ok(args_str) = serde_json::to_string(args) {
                                // Return arguments as a single update
                                return Ok(StreamResult::ToolArguments {
                                    index: 0,
                                    arguments: args_str,
                                });
                            }
                        }
                    }
                }
            }
            Err(_) => {
                // Failed to parse even as partial JSON
                // Continue waiting for more data
            }
        }

        Ok(StreamResult::Incomplete)
    }

    fn detect_format(&self, text: &str) -> bool {
        self.has_tool_markers(text)
    }
}
