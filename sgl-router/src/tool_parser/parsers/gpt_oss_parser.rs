use async_trait::async_trait;
use regex::Regex;
use serde_json::Value;

use crate::{
    protocols::common::Tool,
    tool_parser::{
        errors::{ParserError, ParserResult},
        parsers::helpers,
        partial_json::PartialJson,
        traits::ToolParser,
        types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
    },
};

/// GPT-OSS format parser for tool calls
///
/// Handles the GPT-OSS specific channel format:
/// `<|channel|>commentary to={namespace.function}<|constrain|>json<|message|>{json_args}<|call|>`
///
/// Features:
/// - Channel-based format with commentary
/// - Namespaced function calls
/// - JSON arguments
pub struct GptOssParser {
    /// Parser for handling incomplete JSON during streaming
    partial_json: PartialJson,
    /// Regex for extracting complete function calls
    function_call_extractor: Regex,
    /// Regex for extracting streaming function calls
    streaming_extractor: Regex,

    /// Buffer for accumulating chunks
    buffer: String,
    /// Whether the tool name has been sent (for streaming)
    name_sent: bool,
}

impl GptOssParser {
    /// Create a new GPT-OSS parser
    pub fn new() -> Self {
        // Pattern for complete function calls with to= parameter
        // Handles optional <|start|>assistant prefix and whitespace after function name
        let function_call_pattern = r"(?s)(?:<\|start\|>assistant)?<\|channel\|>commentary to=([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_-]*)*)\s*<\|constrain\|>json<\|message\|>(.*?)<\|call\|>(?:commentary)?";
        let function_call_extractor =
            Regex::new(function_call_pattern).expect("Valid regex pattern");

        // Pattern for streaming function calls (incomplete)
        let streaming_pattern = r"(?s)(?:<\|start\|>assistant)?<\|channel\|>commentary to=([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_-]*)*)\s*<\|constrain\|>json<\|message\|>(.*)";
        let streaming_extractor = Regex::new(streaming_pattern).expect("Valid regex pattern");

        Self {
            partial_json: PartialJson::default(),
            function_call_extractor,
            streaming_extractor,

            buffer: String::new(),
            name_sent: false,
        }
    }

    /// Extract function name from full namespace (e.g., "functions.get_weather" -> "get_weather")
    fn extract_function_name(&self, full_name: &str) -> String {
        if let Some(dot_pos) = full_name.rfind('.') {
            full_name[dot_pos + 1..].to_string()
        } else {
            full_name.to_string()
        }
    }
}

impl Default for GptOssParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for GptOssParser {
    async fn parse_complete(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        // Check if text contains GPT-OSS format
        if !self.has_tool_markers(text) {
            return Ok((text.to_string(), vec![]));
        }

        let mut tools = Vec::new();
        let mut _tool_index = 0;

        // Extract all function calls
        for captures in self.function_call_extractor.captures_iter(text) {
            if let (Some(name_match), Some(args_match)) = (captures.get(1), captures.get(2)) {
                let full_function_name = name_match.as_str();
                let args_content = args_match.as_str().trim();

                // Extract actual function name
                let function_name = self.extract_function_name(full_function_name);

                // Parse JSON arguments
                let arguments = if args_content.is_empty() {
                    "{}".to_string()
                } else {
                    match serde_json::from_str::<Value>(args_content) {
                        Ok(value) => serde_json::to_string(&value)
                            .map_err(|e| ParserError::ParsingFailed(e.to_string()))?,
                        Err(_) => {
                            // Skip malformed JSON
                            continue;
                        }
                    }
                };

                tools.push(ToolCall {
                    function: FunctionCall {
                        name: function_name,
                        arguments,
                    },
                });

                _tool_index += 1;
            }
        }

        Ok((String::new(), tools)) // GPT-OSS parser returns empty normal text
    }

    async fn parse_incremental(
        &mut self,
        chunk: &str,
        tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        self.buffer.push_str(chunk);

        // Check for tool markers
        if !self.has_tool_markers(&self.buffer) {
            // No markers found, clear buffer and return
            self.buffer.clear();
            return Ok(StreamingParseResult::default());
        }

        // Try to match streaming pattern
        if let Some(captures) = self.streaming_extractor.captures(&self.buffer) {
            if let (Some(name_match), Some(args_match)) = (captures.get(1), captures.get(2)) {
                let full_function_name = name_match.as_str();
                let partial_args = args_match.as_str();

                // Extract actual function name
                let function_name = self.extract_function_name(full_function_name);

                // Send function name if not sent yet
                if !self.name_sent {
                    // Validate tool name
                    let tool_indices = helpers::get_tool_indices(tools);
                    if !tool_indices.contains_key(&function_name) {
                        // Invalid tool name - skip
                        tracing::warn!("Invalid tool name '{}' - skipping", function_name);
                        self.buffer.clear();
                        self.name_sent = false;
                        return Ok(StreamingParseResult::default());
                    }

                    self.name_sent = true; // Mark name as sent
                    return Ok(StreamingParseResult {
                        normal_text: String::new(),
                        calls: vec![ToolCallItem {
                            tool_index: 0,
                            name: Some(function_name.clone()),
                            parameters: String::new(),
                        }],
                    });
                }

                // Check if we have a complete function call
                if let Some(complete_match) = self.function_call_extractor.captures(&self.buffer) {
                    if let Some(args_match) = complete_match.get(2) {
                        let args_content = args_match.as_str().trim();

                        // Parse JSON arguments
                        let arguments = if args_content.is_empty() {
                            "{}".to_string()
                        } else {
                            match serde_json::from_str::<Value>(args_content) {
                                Ok(value) => serde_json::to_string(&value)
                                    .unwrap_or_else(|_| "{}".to_string()),
                                Err(_) => "{}".to_string(),
                            }
                        };

                        // Remove the processed part from buffer
                        let complete_end = complete_match.get(0).unwrap().end();
                        self.buffer.drain(..complete_end);

                        // Reset state for next tool
                        self.name_sent = false;

                        // Return final arguments
                        return Ok(StreamingParseResult {
                            normal_text: String::new(),
                            calls: vec![ToolCallItem {
                                tool_index: 0,
                                name: None,
                                parameters: arguments,
                            }],
                        });
                    }
                } else {
                    // Try to parse partial JSON for streaming arguments
                    if !partial_args.is_empty() {
                        // Look for the end of JSON (before <|call|>)
                        let json_part = if let Some(call_pos) = partial_args.find("<|call|>") {
                            &partial_args[..call_pos]
                        } else {
                            partial_args
                        };

                        match self.partial_json.parse_value(json_part, true) {
                            Ok((value, _consumed)) => {
                                let args_str = serde_json::to_string(&value)
                                    .unwrap_or_else(|_| "{}".to_string());

                                return Ok(StreamingParseResult {
                                    normal_text: String::new(),
                                    calls: vec![ToolCallItem {
                                        tool_index: 0,
                                        name: None,
                                        parameters: args_str,
                                    }],
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

        Ok(StreamingParseResult::default())
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains("<|channel|>commentary")
    }
}
