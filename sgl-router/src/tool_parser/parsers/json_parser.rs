use async_trait::async_trait;
use regex::Regex;
use serde_json::Value;

use crate::tool_parser::{
    errors::{ToolParserError, ToolParserResult},
    partial_json::PartialJson,
    state::ParseState,
    traits::ToolParser,
    types::{FunctionCall, StreamResult, TokenConfig, ToolCall},
};

/// JSON format parser for tool calls
///
/// Handles various JSON formats for function calling:
/// - Single tool call: {"name": "fn", "arguments": {...}}
/// - Multiple tool calls: [{"name": "fn1", "arguments": {...}}, ...]
/// - With parameters instead of arguments: {"name": "fn", "parameters": {...}}
///
/// Supports configurable token markers for different models
pub struct JsonParser {
    /// Token configuration for parsing
    token_config: TokenConfig,
    /// Parser for handling incomplete JSON during streaming
    partial_json: PartialJson,
    /// Regex patterns for extracting content between tokens
    extractors: Vec<Regex>,
}

impl JsonParser {
    /// Create a new JSON parser with default configuration
    pub fn new() -> Self {
        Self::with_config(TokenConfig {
            start_tokens: vec![],
            end_tokens: vec![],
            separator: ", ".to_string(),
        })
    }

    /// Create a parser with custom token configuration
    pub fn with_config(config: TokenConfig) -> Self {
        // Build extraction patterns for each token pair
        let extractors: Vec<Regex> = config
            .iter_pairs()
            .filter_map(|(start, end)| {
                if !start.is_empty() && !end.is_empty() {
                    // Use (?s) flag to enable DOTALL mode so . matches newlines
                    let pattern =
                        format!(r"(?s){}(.*?){}", regex::escape(start), regex::escape(end));
                    Regex::new(&pattern).ok()
                } else {
                    None
                }
            })
            .collect();

        Self {
            token_config: config,
            partial_json: PartialJson::default(),
            extractors,
        }
    }

    /// Extract JSON content from text, handling wrapper tokens if configured
    fn extract_json_content<'a>(&self, text: &'a str) -> &'a str {
        let mut content = text;

        // Try each extractor pattern (for tokens with both start and end)
        for extractor in &self.extractors {
            if let Some(captures) = extractor.captures(content) {
                if let Some(matched) = captures.get(1) {
                    return matched.as_str().trim();
                }
            }
        }

        // Handle special case where there's a start token but no end token
        for (start, end) in self.token_config.iter_pairs() {
            if !start.is_empty() && end.is_empty() {
                // Find the start token and extract everything after it
                if let Some(pos) = content.find(start) {
                    content = &content[pos + start.len()..];
                    return content.trim();
                }
            }
        }

        content.trim()
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

    /// Check if text contains potential tool call markers
    fn has_tool_markers(&self, text: &str) -> bool {
        // If no start tokens configured, check for JSON structure
        if self.token_config.start_tokens.is_empty() {
            // For JSON, we just need to see the start of an object or array
            return text.contains('{') || text.contains('[');
        }

        // Check for any start token
        let has_start_token = self
            .token_config
            .start_tokens
            .iter()
            .any(|token| text.contains(token));

        // Also check if we have what looks like JSON even without start token
        // This handles cases where we've already processed the start token
        // and are working on subsequent tools
        has_start_token || (text.trim_start().starts_with('{') && text.contains(r#""name""#))
    }

    /// Check if text might contain a partial start token (for streaming)
    fn has_partial_start_token(&self, text: &str) -> bool {
        if self.token_config.start_tokens.is_empty() {
            return false;
        }

        // Check if the end of the buffer could be the start of any start token
        for start_token in &self.token_config.start_tokens {
            for i in 1..start_token.len() {
                let token_prefix = &start_token[..i];
                if text.ends_with(token_prefix) {
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
        // Check if we have multiple start tokens (e.g., multiple <|python_tag|> markers)
        if !self.token_config.start_tokens.is_empty() {
            let start_token = &self.token_config.start_tokens[0];
            if !start_token.is_empty() && text.matches(start_token).count() > 1 {
                // We have multiple occurrences of the start token
                let mut all_tools = Vec::new();
                let mut all_normal_text = String::new();
                let mut remaining = text;

                while let Some(start_pos) = remaining.find(start_token.as_str()) {
                    // Add text before this start token to normal text
                    all_normal_text.push_str(&remaining[..start_pos]);

                    // Extract content after this start token
                    let after_token = &remaining[start_pos + start_token.len()..];

                    // Find where this JSON ends (look for the next start token or end of string)
                    let end_pos = if let Some(next_start) = after_token.find(start_token.as_str()) {
                        next_start
                    } else {
                        after_token.len()
                    };

                    let json_content = &after_token[..end_pos];

                    // Try to extract and parse JSON from this segment
                    if let Some((extracted, segment_normal_text)) =
                        self.extract_json_from_text(json_content)
                    {
                        if let Ok(value) = serde_json::from_str::<Value>(&extracted) {
                            if let Ok(tools) = self.parse_json_value(&value) {
                                all_tools.extend(tools);
                            }
                        }
                        // Add the normal text from this segment
                        all_normal_text.push_str(&segment_normal_text);
                    } else {
                        // If no JSON found, add the entire content as normal text
                        all_normal_text.push_str(json_content);
                    }

                    // Move to the next segment
                    remaining = &remaining[start_pos + start_token.len() + end_pos..];
                    if remaining.is_empty() {
                        break;
                    }
                }

                // Add any remaining text
                all_normal_text.push_str(remaining);

                return Ok((all_normal_text, all_tools));
            }
        }

        // Extract JSON content from wrapper tokens if present
        let json_content = self.extract_json_content(text);

        // Try to parse as JSON first
        match serde_json::from_str::<Value>(json_content) {
            Ok(value) => {
                let tools = self.parse_json_value(&value)?;
                Ok((String::new(), tools))
            }
            Err(_) => {
                // If parse failed, check if we have multiple JSON objects separated by the configured separator
                // Only do this if we can reasonably expect multiple complete JSON objects
                // (i.e., text starts and ends with JSON-like structure)
                if !self.token_config.separator.is_empty()
                    && json_content.contains(&self.token_config.separator)
                    && json_content.trim().starts_with('{')
                    && json_content.trim().ends_with('}')
                {
                    let mut all_tools = Vec::new();

                    // Split by separator and try to parse each part
                    let parts: Vec<&str> =
                        json_content.split(&self.token_config.separator).collect();
                    let mut normal_parts = Vec::new();

                    for part in parts {
                        let trimmed = part.trim();
                        if trimmed.is_empty() {
                            normal_parts.push(trimmed.to_string());
                            continue;
                        }

                        // Try to parse this part as JSON
                        if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
                            if let Ok(tools) = self.parse_json_value(&value) {
                                all_tools.extend(tools);
                            }
                            normal_parts.push(trimmed.to_string());
                        } else if let Some((extracted, part_normal_text)) =
                            self.extract_json_from_text(trimmed)
                        {
                            // Try extracting JSON from this part
                            if let Ok(value) = serde_json::from_str::<Value>(&extracted) {
                                if let Ok(tools) = self.parse_json_value(&value) {
                                    all_tools.extend(tools);
                                }
                            }
                            normal_parts.push(part_normal_text);
                        } else {
                            normal_parts.push(trimmed.to_string());
                        }
                    }

                    // Rejoin with the original separator to preserve it
                    let all_normal_text = normal_parts.join(&self.token_config.separator);

                    return Ok((all_normal_text, all_tools));
                }

                // If no wrapper tokens configured and parse failed, try to extract JSON from mixed text
                if self.token_config.start_tokens.is_empty() {
                    if let Some((extracted_json, normal_text)) = self.extract_json_from_text(text) {
                        if let Ok(value) = serde_json::from_str::<Value>(&extracted_json) {
                            let tools = self.parse_json_value(&value)?;
                            return Ok((normal_text, tools));
                        }
                    }
                }

                // No valid JSON found, return original text as normal text
                Ok((text.to_string(), vec![]))
            }
        }
    }

    async fn parse_incremental(
        &self,
        chunk: &str,
        state: &mut ParseState,
    ) -> ToolParserResult<StreamResult> {
        state.buffer.push_str(chunk);

        // Check if we have potential tool calls
        if !self.has_tool_markers(&state.buffer) {
            if self.has_partial_start_token(&state.buffer) {
                // We might be in the middle of receiving a start token, wait for more data
                return Ok(StreamResult::Incomplete);
            }

            // No tool markers and no partial tokens - return all buffered content as normal text
            let normal_text = std::mem::take(&mut state.buffer);
            return Ok(StreamResult::NormalText(normal_text));
        }

        // Check for text before tool markers and extract it as normal text
        if !self.token_config.start_tokens.is_empty() {
            let start_token = &self.token_config.start_tokens[0];
            if !start_token.is_empty() {
                if let Some(marker_pos) = state.buffer.find(start_token) {
                    if marker_pos > 0 {
                        // We have text before the tool marker - extract it as normal text
                        let normal_text: String = state.buffer.drain(..marker_pos).collect();
                        return Ok(StreamResult::NormalText(normal_text));
                    }
                }
            }
        } else {
            // For JSON without start tokens, look for the start of JSON structure
            // Find whichever comes first: '{' or '['
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

        // Extract JSON content first to check for separators
        let extracted_json = self.extract_json_content(&state.buffer);

        // Handle multiple JSON objects with separators
        // Check if we have a separator and potentially multiple JSON objects
        let separator = &self.token_config.separator;
        if !separator.is_empty() && extracted_json.contains(separator.as_str()) {
            // Try to find a complete JSON object before the separator
            if let Some(separator_pos) = extracted_json.find(separator.as_str()) {
                // Get JSON before separator
                let before_separator = &extracted_json[..separator_pos];

                // Try to parse the JSON before the separator
                match serde_json::from_str::<Value>(before_separator) {
                    Ok(value) => {
                        // Parse tool calls from this JSON
                        let tools = self.parse_json_value(&value)?;
                        if !tools.is_empty() {
                            // We need to figure out how much to remove from the original buffer
                            // Find where the separator is in the original buffer and remove up to and including it
                            if let Some(sep_in_original) = state.buffer.find(separator.as_str()) {
                                // Remove processed content up to and including separator
                                state.buffer.drain(..=sep_in_original + separator.len() - 1);
                            }

                            // Return the first tool as complete
                            if let Some(tool) = tools.into_iter().next() {
                                return Ok(StreamResult::ToolComplete(tool));
                            }
                        }
                    }
                    Err(_) => {
                        // Failed to parse, continue to try other methods
                    }
                }
            }
        }

        // Handle multiple start tokens (e.g., multiple <|python_tag|> markers)
        if !self.token_config.start_tokens.is_empty() {
            let start_token = &self.token_config.start_tokens[0];
            if !start_token.is_empty() {
                // Find all occurrences of start token
                let occurrences: Vec<_> =
                    state.buffer.match_indices(start_token.as_str()).collect();
                if occurrences.len() > 1 {
                    // We have multiple start tokens, try to process the first complete one
                    let first_pos = occurrences[0].0;
                    let second_pos = occurrences[1].0;

                    // Extract content between first and second start token
                    let first_json_section = &state.buffer[first_pos..second_pos];
                    let json_content = self.extract_json_content(first_json_section);

                    // Try to parse this as complete JSON
                    if let Ok(value) = serde_json::from_str::<Value>(json_content) {
                        // Parse tool calls from this JSON
                        let tools = self.parse_json_value(&value)?;
                        if !tools.is_empty() {
                            // Remove the processed section from buffer
                            let remaining = state.buffer[second_pos..].to_string();
                            state.buffer = remaining;

                            // Return the first tool as complete
                            if let Some(tool) = tools.into_iter().next() {
                                return Ok(StreamResult::ToolComplete(tool));
                            }
                        }
                    }
                }
            }
        }

        // Regular single JSON parsing
        // Extract JSON content
        let json_content = self.extract_json_content(&state.buffer);

        // Try to parse with partial JSON parser
        match self.partial_json.parse_value(json_content) {
            Ok((value, consumed)) => {
                // Check if we have a complete JSON structure
                if consumed == json_content.len() {
                    // Check if this is truly complete or just has null from incomplete parsing
                    // We need to ensure the JSON actually ends properly (not cut off mid-key)
                    let trimmed = json_content.trim();
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
        // Check if text contains JSON-like structure
        if self.has_tool_markers(text) {
            // Try to extract and parse
            let json_content = self.extract_json_content(text);

            // Check if it looks like valid JSON for tool calls
            if let Ok(value) = serde_json::from_str::<Value>(json_content) {
                match value {
                    Value::Object(ref obj) => {
                        // Check for tool call structure
                        obj.contains_key("name") || obj.contains_key("function")
                    }
                    Value::Array(ref arr) => {
                        // Check if array contains tool-like objects
                        arr.iter().any(|v| {
                            if let Some(obj) = v.as_object() {
                                obj.contains_key("name") || obj.contains_key("function")
                            } else {
                                false
                            }
                        })
                    }
                    _ => false,
                }
            } else {
                false
            }
        } else {
            false
        }
    }
}
