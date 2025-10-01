use async_trait::async_trait;
use serde_json::Value;
use uuid;

use crate::tool_parser::{
    errors::{ToolParserError, ToolParserResult},
    partial_json::PartialJson,
    state::ParseState,
    traits::ToolParser,
    types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
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
    ) -> ToolParserResult<StreamingParseResult> {
        state.buffer.push_str(chunk);
        let mut result = StreamingParseResult::new();

        const PYTHON_TAG: &str = "<|python_tag|>";

        // Phase 1: Check for normal text before tool markers
        if !state.in_tool_section {
            if let Some(tag_pos) = state.buffer.find(PYTHON_TAG) {
                if tag_pos > 0 {
                    result.normal_text = state.buffer.drain(..tag_pos).collect();
                    state.in_tool_section = true;
                    return Ok(result);
                }
                state.in_tool_section = true;
                // Remove the python_tag from buffer
                state.buffer.drain(..PYTHON_TAG.len());
            } else {
                // Check if we might have a partial python_tag at the end
                if self.has_partial_python_tag(&state.buffer) {
                    return Ok(result);
                }

                // No python_tag, try parsing as direct JSON
                if let Some(json_start) = self.find_json_start(&state.buffer) {
                    if json_start > 0 {
                        result.normal_text = state.buffer.drain(..json_start).collect();
                        state.in_tool_section = true;
                        return Ok(result);
                    }
                    state.in_tool_section = true;
                } else if !self.might_be_json_start(&state.buffer) {
                    // Not JSON either, return as normal text
                    result.normal_text = std::mem::take(&mut state.buffer);
                    return Ok(result);
                }
            }
        }

        // Phase 2: Process tool calls
        if state.in_tool_section {
            self.process_tool_json(state, &mut result)?;
        }

        Ok(result)
    }

    fn detect_format(&self, text: &str) -> bool {
        self.has_python_tag(text) || text.contains('{')
    }
}

impl LlamaParser {
    fn has_partial_python_tag(&self, buffer: &str) -> bool {
        const PYTHON_TAG: &str = "<|python_tag|>";
        for i in 1..PYTHON_TAG.len().min(buffer.len()) {
            if PYTHON_TAG.starts_with(&buffer[buffer.len() - i..]) {
                return true;
            }
        }
        false
    }

    fn find_json_start(&self, buffer: &str) -> Option<usize> {
        let mut in_string = false;
        let mut escape = false;

        for (i, ch) in buffer.char_indices() {
            if escape {
                escape = false;
                continue;
            }

            match ch {
                '\\' if in_string => escape = true,
                '"' => in_string = !in_string,
                '{' | '[' if !in_string => return Some(i),
                _ => {}
            }
        }
        None
    }

    fn might_be_json_start(&self, buffer: &str) -> bool {
        buffer.ends_with('{') || buffer.ends_with('[') || buffer.ends_with('"')
    }

    fn process_tool_json(
        &self,
        state: &mut ParseState,
        result: &mut StreamingParseResult,
    ) -> ToolParserResult<()> {
        // Try to parse partial JSON
        match self.partial_json.parse_value(&state.buffer) {
            Ok((value, consumed)) => {
                // Check if it's an array or single object
                let tools = if value.is_array() {
                    value.as_array().unwrap().clone()
                } else {
                    vec![value]
                };

                for (idx, tool_value) in tools.iter().enumerate() {
                    self.process_single_tool(idx, tool_value, state, result)?;
                }

                // Check if we've consumed everything
                if consumed == state.buffer.len() {
                    state.buffer.clear();
                    state.mode = crate::tool_parser::state::ParseMode::Complete;
                }
            }
            Err(_) => {
                // Incomplete JSON, try to extract partial tool info
                self.try_extract_partial_tools(state, result)?;
            }
        }

        Ok(())
    }

    fn process_single_tool(
        &self,
        index: usize,
        tool_value: &Value,
        state: &mut ParseState,
        result: &mut StreamingParseResult,
    ) -> ToolParserResult<()> {
        // Ensure we have a partial tool entry
        let partial = state.ensure_partial_tool(index);

        // Extract and send tool name if not sent
        if !partial.name_sent {
            if let Some(name) = tool_value.get("name").and_then(|v| v.as_str()) {
                partial.name = Some(name.to_string());
                partial.id = Some(format!("llama_call_{}", uuid::Uuid::new_v4()));
                partial.name_sent = true;

                result.tool_calls.push(ToolCallItem {
                    tool_index: index,
                    id: partial.id.clone(),
                    name: partial.name.clone(),
                    arguments_delta: String::new(),
                });
                return Ok(());
            }
        }

        // Stream arguments (Llama uses "parameters" instead of "arguments")
        if partial.name_sent {
            let args = tool_value.get("parameters")
                .or_else(|| tool_value.get("arguments"));

            if let Some(args_value) = args {
                let args_str = serde_json::to_string(args_value)?;

                if args_str.len() > partial.streamed_arguments.len() {
                    let delta = &args_str[partial.streamed_arguments.len()..];

                    result.tool_calls.push(ToolCallItem {
                        tool_index: index,
                        id: None,
                        name: None,
                        arguments_delta: delta.to_string(),
                    });

                    partial.streamed_arguments = args_str;
                }
            }
        }

        Ok(())
    }

    fn try_extract_partial_tools(
        &self,
        state: &mut ParseState,
        result: &mut StreamingParseResult,
    ) -> ToolParserResult<()> {
        // Try regex-based extraction for partial tools
        let tool_pattern = r#"\{\s*"name"\s*:\s*"([^"]+)"[^}]*\}"#;
        let re = regex::Regex::new(tool_pattern).unwrap();

        let buffer = state.buffer.clone();
        for (idx, mat) in re.captures_iter(&buffer).enumerate() {
            // Ensure we have partial tool entry
            let partial = state.ensure_partial_tool(idx);

            if !partial.name_sent {
                if let Some(name_match) = mat.get(1) {
                    partial.name = Some(name_match.as_str().to_string());
                    partial.id = Some(format!("llama_call_{}", uuid::Uuid::new_v4()));
                    partial.name_sent = true;

                    result.tool_calls.push(ToolCallItem {
                        tool_index: idx,
                        id: partial.id.clone(),
                        name: partial.name.clone(),
                        arguments_delta: String::new(),
                    });
                }
            }
        }

        Ok(())
    }
}
