use async_trait::async_trait;
use serde_json::Value;
use uuid;

use crate::tool_parser::{
    errors::{ToolParserError, ToolParserResult},
    partial_json::PartialJson,
    state::ParseState,
    traits::ToolParser,
    types::{FunctionCall, StreamResult, ToolCall},
};

/// Llama 3.2 format parser for tool calls
///
/// Handles the Llama 3.2 specific format:
/// `<|python_tag|>{"name": "func", "arguments": {...}}`
///
/// Also supports plain JSON without the python_tag prefix
pub struct LlamaParser {
    /// Parser for handling incomplete JSON during streaming
    partial_json: PartialJson,
}

impl LlamaParser {
    /// Create a new Llama parser
    pub fn new() -> Self {
        Self {
            partial_json: PartialJson::default(),
        }
    }

    /// Extract content after python_tag token
    fn extract_content_after_python_tag(&self, text: &str) -> Option<(String, String)> {
        const PYTHON_TAG: &str = "<|python_tag|>";

        if let Some(tag_pos) = text.find(PYTHON_TAG) {
            let normal_text = text[..tag_pos].to_string();
            let json_content = text[tag_pos + PYTHON_TAG.len()..].to_string();
            Some((normal_text, json_content))
        } else {
            None
        }
    }

    /// Parse a single JSON object into a ToolCall (Llama format: name + parameters)
    fn parse_single_object(&self, obj: &Value) -> ToolParserResult<Option<ToolCall>> {
        // Llama format only: {"name": "function_name", "parameters": {...}}
        let name = obj.get("name").and_then(|v| v.as_str());

        if let Some(name) = name {
            // Llama uses "parameters" key
            let empty_obj = Value::Object(serde_json::Map::new());
            let parameters = obj.get("parameters").unwrap_or(&empty_obj);

            // Convert parameters to JSON string
            let arguments = serde_json::to_string(parameters)
                .map_err(|e| ToolParserError::ParsingFailed(e.to_string()))?;

            // Generate a unique ID for Llama calls
            let id = obj
                .get("id")
                .and_then(|v| v.as_str())
                .map(String::from)
                .unwrap_or_else(|| format!("llama_call_{}", uuid::Uuid::new_v4()));

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

    /// Check if text contains potential tool call markers
    fn has_python_tag(&self, text: &str) -> bool {
        text.contains("<|python_tag|>")
    }

    /// Parse semicolon-separated JSON objects
    fn parse_semicolon_separated(&self, content: &str) -> ToolParserResult<Vec<ToolCall>> {
        let mut all_tools = Vec::new();

        // Split by semicolon and parse each JSON object
        for part in content.split(';') {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                continue;
            }

            // Try to parse this part as a single JSON object
            match serde_json::from_str::<Value>(trimmed) {
                Ok(value) => {
                    if let Some(tool) = self.parse_single_object(&value)? {
                        all_tools.push(tool);
                    }
                }
                Err(e) => {
                    // Skip invalid JSON parts in semicolon-separated list
                    tracing::warn!("Failed to parse tool call: {}", e);
                }
            }
        }

        Ok(all_tools)
    }
}

impl Default for LlamaParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for LlamaParser {
    async fn parse_complete(&self, text: &str) -> ToolParserResult<(String, Vec<ToolCall>)> {
        // Extract normal text and JSON content
        let (normal_text, json_content) =
            if let Some((normal, json)) = self.extract_content_after_python_tag(text) {
                (normal, json)
            } else if text.trim_start().starts_with('{') {
                (String::new(), text.to_string())
            } else {
                // No JSON structure found
                return Ok((text.to_string(), vec![]));
            };

        // Parse the JSON content (may contain semicolon-separated objects)
        let tools = if json_content.contains(';') {
            self.parse_semicolon_separated(&json_content)?
        } else {
            // Try single JSON object
            let parsed = serde_json::from_str::<Value>(json_content.trim())
                .map_err(|e| ToolParserError::ParsingFailed(e.to_string()))
                .and_then(|v| {
                    self.parse_single_object(&v)
                        .map(|opt| opt.map_or_else(Vec::new, |tool| vec![tool]))
                });

            parsed.unwrap_or_else(|e| {
                tracing::warn!("Failed to parse tool call: {:?}", e);
                vec![]
            })
        };

        // If we couldn't parse any tools, return the original text
        if tools.is_empty() {
            return Ok((text.to_string(), vec![]));
        }

        Ok((normal_text, tools))
    }

    async fn parse_incremental(
        &self,
        chunk: &str,
        state: &mut ParseState,
    ) -> ToolParserResult<StreamResult> {
        state.buffer.push_str(chunk);

        // In streaming mode, be more lenient - check for potential JSON start
        let has_potential_json = state.buffer.contains('{');
        let has_tag = self.has_python_tag(&state.buffer);

        // If we have neither python_tag nor potential JSON structure, return as normal text
        if !has_tag && !has_potential_json {
            // No relevant markers detected - return all buffered content as normal text
            let normal_text = std::mem::take(&mut state.buffer);
            return Ok(StreamResult::NormalText(normal_text));
        }

        // If we only have '{' without more content, wait for more data
        let trimmed = state.buffer.trim();
        if (trimmed == "{") && !has_tag {
            return Ok(StreamResult::Incomplete);
        }

        // Check for text before python_tag and extract it as normal text
        if let Some(tag_pos) = state.buffer.find("<|python_tag|>") {
            if tag_pos > 0 {
                // We have text before the python_tag - extract it as normal text
                let normal_text: String = state.buffer.drain(..tag_pos).collect();
                return Ok(StreamResult::NormalText(normal_text));
            }
        } else {
            // For JSON without python_tag, look for the start of JSON structure
            let brace_pos = state.buffer.find('{');
            let bracket_pos = state.buffer.find('[');
            let json_pos = brace_pos.iter().chain(bracket_pos.iter()).min().copied();

            if let Some(pos) = json_pos {
                if pos > 0 {
                    // We have text before JSON structure - extract it as normal text
                    let normal_text: String = state.buffer.drain(..pos).collect();
                    return Ok(StreamResult::NormalText(normal_text));
                }
            }
        }

        // Extract JSON content based on whether we have python_tag
        let (json_content, content_start_pos) = if self.has_python_tag(&state.buffer) {
            // Extract content after python_tag
            if let Some(tag_pos) = state.buffer.find("<|python_tag|>") {
                let start = tag_pos + "<|python_tag|>".len();
                (&state.buffer[start..], start)
            } else {
                (&state.buffer[..], 0)
            }
        } else {
            // Find where the actual content starts after trimming
            let trimmed = state.buffer.trim_start();
            let trim_offset = state.buffer.len() - trimmed.len();
            (trimmed.trim_end(), trim_offset)
        };

        // Check if we have a semicolon separator (multiple tools)
        if let Some(semicolon_pos) = json_content.find(';') {
            // We have multiple tools - try to parse the first one
            let first_json = &json_content[..semicolon_pos];

            if let Ok(value) = serde_json::from_str::<Value>(first_json.trim()) {
                if let Some(tool) = self.parse_single_object(&value)? {
                    // Remove the parsed JSON and semicolon from the buffer
                    let end_pos = content_start_pos + semicolon_pos + 1; // +1 to include the semicolon
                    state.buffer.drain(content_start_pos..end_pos);

                    return Ok(StreamResult::ToolComplete(tool));
                }
            }
        }

        // Try to parse with partial JSON parser
        match self.partial_json.parse_value(json_content) {
            Ok((value, consumed)) => {
                // Check if we have a complete JSON structure
                if consumed == json_content.len() {
                    // Check if this is truly complete
                    let looks_complete = json_content.ends_with('}') || json_content.ends_with(']');

                    if looks_complete {
                        // Complete JSON, parse tool calls
                        let tools = self.parse_json_value(&value)?;
                        if !tools.is_empty() {
                            // Clear buffer since we consumed everything
                            state.buffer.clear();

                            // Return the first tool as complete
                            if let Some(tool) = tools.into_iter().next() {
                                return Ok(StreamResult::ToolComplete(tool));
                            }
                        }
                    }
                } else {
                    // Partial JSON, try to extract tool name for streaming
                    if let Some(name) = value.get("name").and_then(|v| v.as_str()) {
                        // Return tool name once we see it
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
        // Llama format if contains python_tag or starts with JSON object
        text.contains("<|python_tag|>")
            || (text.trim_start().starts_with('{') && text.contains(r#""name""#))
    }
}
