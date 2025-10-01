use async_trait::async_trait;
use serde_json::Value;

use crate::tool_parser::{
    errors::{ToolParserError, ToolParserResult},
    partial_json::PartialJson,
    state::ParseState,
    traits::ToolParser,
    types::{FunctionCall, StreamResult, ToolCall},
};

/// Mistral format parser for tool calls
///
/// Handles the Mistral-specific format:
/// `[TOOL_CALLS] [{"name": "func", "arguments": {...}}, ...]`
///
/// Features:
/// - Bracket counting for proper JSON array extraction
/// - Support for multiple tool calls in a single array
/// - String-aware parsing to handle nested brackets in JSON
pub struct MistralParser {
    /// Parser for handling incomplete JSON during streaming
    partial_json: PartialJson,
}

impl MistralParser {
    /// Create a new Mistral parser
    pub fn new() -> Self {
        Self {
            partial_json: PartialJson::default(),
        }
    }

    /// Extract JSON array using bracket counting
    ///
    /// Handles nested brackets in JSON content by tracking:
    /// - String boundaries (quotes)
    /// - Escape sequences
    /// - Bracket depth
    fn extract_json_array<'a>(&self, text: &'a str) -> Option<&'a str> {
        self.extract_json_array_with_pos(text).map(|(_, json)| json)
    }

    fn extract_json_array_with_pos<'a>(&self, text: &'a str) -> Option<(usize, &'a str)> {
        const BOT_TOKEN: &str = "[TOOL_CALLS] [";

        // Find the start of the token
        let start_idx = text.find(BOT_TOKEN)?;

        // Start from the opening bracket after [TOOL_CALLS]
        // The -1 is to include the opening bracket that's part of the token
        let json_start = start_idx + BOT_TOKEN.len() - 1;

        let mut bracket_count = 0;
        let mut in_string = false;
        let mut escape_next = false;

        let bytes = text.as_bytes();

        for i in json_start..text.len() {
            let char = bytes[i];

            if escape_next {
                escape_next = false;
                continue;
            }

            if char == b'\\' {
                escape_next = true;
                continue;
            }

            if char == b'"' && !escape_next {
                in_string = !in_string;
                continue;
            }

            if !in_string {
                if char == b'[' {
                    bracket_count += 1;
                } else if char == b']' {
                    bracket_count -= 1;
                    if bracket_count == 0 {
                        // Found the matching closing bracket
                        return Some((start_idx, &text[json_start..=i]));
                    }
                }
            }
        }

        // Incomplete array (no matching closing bracket found)
        None
    }

    /// Parse tool calls from a JSON array
    fn parse_json_array(&self, json_str: &str) -> ToolParserResult<Vec<ToolCall>> {
        let value: Value = serde_json::from_str(json_str)
            .map_err(|e| ToolParserError::ParsingFailed(e.to_string()))?;

        let mut tools = Vec::new();

        if let Value::Array(arr) = value {
            for (index, item) in arr.iter().enumerate() {
                if let Some(tool) = self.parse_single_object(item, index)? {
                    tools.push(tool);
                }
            }
        } else {
            // Single object case (shouldn't happen with Mistral format, but handle it)
            if let Some(tool) = self.parse_single_object(&value, 0)? {
                tools.push(tool);
            }
        }

        Ok(tools)
    }

    /// Parse a single JSON object into a ToolCall
    fn parse_single_object(&self, obj: &Value, index: usize) -> ToolParserResult<Option<ToolCall>> {
        let name = obj.get("name").and_then(|v| v.as_str());

        if let Some(name) = name {
            // Get arguments - Mistral uses "arguments" key
            let empty_obj = Value::Object(serde_json::Map::new());
            let args = obj.get("arguments").unwrap_or(&empty_obj);

            // Convert arguments to JSON string
            let arguments = serde_json::to_string(args)
                .map_err(|e| ToolParserError::ParsingFailed(e.to_string()))?;

            // Generate ID with index for multiple tools
            let id = format!("mistral_call_{}", index);

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

    /// Check if text contains Mistral tool markers
    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains("[TOOL_CALLS]")
    }
}

impl Default for MistralParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for MistralParser {
    async fn parse_complete(&self, text: &str) -> ToolParserResult<(String, Vec<ToolCall>)> {
        // Check if text contains Mistral format
        if !self.has_tool_markers(text) {
            return Ok((text.to_string(), vec![]));
        }

        // Extract JSON array from Mistral format with position
        if let Some((start_idx, json_array)) = self.extract_json_array_with_pos(text) {
            // Extract normal text before BOT_TOKEN
            let normal_text_before = if start_idx > 0 {
                text[..start_idx].to_string()
            } else {
                String::new()
            };

            match self.parse_json_array(json_array) {
                Ok(tools) => Ok((normal_text_before, tools)),
                Err(e) => {
                    // If JSON parsing fails, return the original text as normal text
                    tracing::warn!("Failed to parse tool call: {}", e);
                    Ok((text.to_string(), vec![]))
                }
            }
        } else {
            // Markers present but no complete array found
            Ok((text.to_string(), vec![]))
        }
    }

    async fn parse_incremental(
        &self,
        chunk: &str,
        state: &mut ParseState,
    ) -> ToolParserResult<StreamResult> {
        state.buffer.push_str(chunk);

        // Check if we have the start marker
        if !self.has_tool_markers(&state.buffer) {
            // No tool markers detected - return all buffered content as normal text
            let normal_text = std::mem::take(&mut state.buffer);
            return Ok(StreamResult::NormalText(normal_text));
        }

        // Check for text before [TOOL_CALLS] and extract it as normal text
        if let Some(marker_pos) = state.buffer.find("[TOOL_CALLS]") {
            if marker_pos > 0 {
                // We have text before the tool marker - extract it as normal text
                let normal_text: String = state.buffer.drain(..marker_pos).collect();
                return Ok(StreamResult::NormalText(normal_text));
            }
        }

        // Try to extract complete JSON array
        if let Some(json_array) = self.extract_json_array(&state.buffer) {
            // Parse with partial JSON to handle incomplete content
            match self.partial_json.parse_value(json_array) {
                Ok((value, consumed)) => {
                    // Check if we have a complete JSON structure
                    if consumed == json_array.len() {
                        // Complete JSON, parse tool calls
                        let tools = if let Value::Array(arr) = value {
                            let mut result = Vec::new();
                            for (index, item) in arr.iter().enumerate() {
                                if let Some(tool) = self.parse_single_object(item, index)? {
                                    result.push(tool);
                                }
                            }
                            result
                        } else {
                            vec![]
                        };

                        if !tools.is_empty() {
                            // Clear buffer since we consumed everything
                            state.buffer.clear();

                            // Return the first tool (simplified for Phase 3)
                            // Full multi-tool streaming will be implemented later
                            if let Some(tool) = tools.into_iter().next() {
                                return Ok(StreamResult::ToolComplete(tool));
                            }
                        }
                    } else {
                        // Partial JSON - try to extract tool name for streaming
                        if let Value::Array(arr) = value {
                            if let Some(first_tool) = arr.first() {
                                if let Some(name) = first_tool.get("name").and_then(|v| v.as_str())
                                {
                                    // Check if we've already sent the name
                                    if !state.in_string {
                                        state.in_string = true; // Use as flag for "name sent"
                                        return Ok(StreamResult::ToolName {
                                            index: 0,
                                            name: name.to_string(),
                                        });
                                    }

                                    // Check for arguments
                                    if let Some(args) = first_tool.get("arguments") {
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
                    }
                }
                Err(_) => {
                    // Failed to parse even as partial JSON
                    // Keep buffering
                }
            }
        }

        Ok(StreamResult::Incomplete)
    }

    fn detect_format(&self, text: &str) -> bool {
        self.has_tool_markers(text)
    }
}
