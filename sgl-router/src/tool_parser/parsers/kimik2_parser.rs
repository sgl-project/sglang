use async_trait::async_trait;
use regex::Regex;
use serde_json::Value;

use crate::{
    protocols::common::Tool,
    tool_parser::{
        errors::ParserResult,
        parsers::helpers,
        traits::ToolParser,
        types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
    },
};

/// Kimi K2 format parser for tool calls
///
/// Handles the Kimi K2 specific format:
/// `<|tool_calls_section_begin|><|tool_call_begin|>functions.{name}:{index}<|tool_call_argument_begin|>{json_args}<|tool_call_end|><|tool_calls_section_end|>`
///
/// Features:
/// - Token-based delimiters
/// - Function calls with explicit indexing
/// - JSON arguments
pub struct KimiK2Parser {
    /// Regex for extracting complete tool calls
    tool_call_extractor: Regex,
    /// Regex for extracting partial tool calls (streaming)
    stream_tool_call_extractor: Regex,
    /// Regex pattern for removing completed tool calls from buffer
    tool_call_end_pattern: Regex,
    /// Robust parser for ids like "functions.search:0" or fallback "search:0"
    tool_call_id_regex: Regex,

    /// Buffer for accumulating incomplete patterns across chunks
    buffer: String,

    /// Stores complete tool call info (name and arguments) for each tool being parsed
    prev_tool_call_arr: Vec<Value>,

    /// Index of currently streaming tool call (-1 means no active tool)
    current_tool_id: i32,

    /// Flag for whether current tool's name has been sent to client
    current_tool_name_sent: bool,

    /// Tracks raw JSON string content streamed to client for each tool's arguments
    streamed_args_for_tool: Vec<String>,

    /// Tracks the last arguments sent for incremental diffing
    last_arguments: String,
}

impl KimiK2Parser {
    /// Create a new Kimi K2 parser
    pub fn new() -> Self {
        // Pattern for complete tool calls
        let tool_call_pattern = r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[\w\.]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>\{.*?\})\s*<\|tool_call_end\|>";
        let tool_call_extractor = Regex::new(tool_call_pattern).expect("Valid regex pattern");

        // Pattern for streaming (partial) tool calls
        let stream_pattern = r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[\w\.]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>\{.*)";
        let stream_tool_call_extractor = Regex::new(stream_pattern).expect("Valid regex pattern");

        // Pattern for removing completed tool calls
        let end_pattern = r"<\|tool_call_begin\|>.*?<\|tool_call_end\|>";
        let tool_call_end_pattern = Regex::new(end_pattern).expect("Valid regex pattern");

        // Robust parser for ids like "functions.search:0" or fallback "search:0"
        let id_pattern = r"^(?:functions\.)?(?P<name>[\w\.]+):(?P<index>\d+)$";
        let tool_call_id_regex = Regex::new(id_pattern).expect("Valid regex pattern");

        Self {
            tool_call_extractor,
            stream_tool_call_extractor,
            tool_call_end_pattern,
            tool_call_id_regex,
            buffer: String::new(),
            prev_tool_call_arr: Vec::new(),
            current_tool_id: -1,
            current_tool_name_sent: false,
            streamed_args_for_tool: Vec::new(),
            last_arguments: String::new(),
        }
    }

    /// Parse function ID to extract name and index
    fn parse_function_id(&self, id: &str) -> Option<(String, usize)> {
        if let Some(captures) = self.tool_call_id_regex.captures(id) {
            let name = captures.name("name")?.as_str().to_string();
            let index = captures.name("index")?.as_str().parse::<usize>().ok()?;
            Some((name, index))
        } else {
            None
        }
    }
}

impl Default for KimiK2Parser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for KimiK2Parser {
    async fn parse_complete(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        if !self.has_tool_markers(text) {
            return Ok((text.to_string(), vec![]));
        }

        // Find where tool calls begin
        let idx = text.find("<|tool_calls_section_begin|>").unwrap();
        let normal_text = text[..idx].to_string();

        // Try to extract tool calls
        let mut tools = Vec::new();
        for captures in self.tool_call_extractor.captures_iter(text) {
            if let (Some(id_match), Some(args_match)) = (
                captures.name("tool_call_id"),
                captures.name("function_arguments"),
            ) {
                let function_id = id_match.as_str();
                let function_args = args_match.as_str();

                // Parse function ID
                if let Some((func_name, _index)) = self.parse_function_id(function_id) {
                    // Try to parse JSON arguments
                    match serde_json::from_str::<Value>(function_args) {
                        Ok(_) => {
                            tools.push(ToolCall {
                                function: FunctionCall {
                                    name: func_name,
                                    arguments: function_args.to_string(),
                                },
                            });
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Failed to parse JSON arguments for {}: {}",
                                func_name,
                                e
                            );
                            continue;
                        }
                    }
                } else {
                    tracing::warn!("Failed to parse function ID: {}", function_id);
                    continue;
                }
            }
        }

        // If no tools were successfully parsed despite having markers, return entire text as fallback
        if tools.is_empty() {
            return Ok((text.to_string(), vec![]));
        }

        Ok((normal_text, tools))
    }

    async fn parse_incremental(
        &mut self,
        chunk: &str,
        tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        self.buffer.push_str(chunk);
        let current_text = &self.buffer.clone();

        // Check if we have a tool call (either the start token or individual tool call)
        let has_tool_call =
            self.has_tool_markers(current_text) || current_text.contains("<|tool_call_begin|>");

        if !has_tool_call {
            // No tool markers detected - return all buffered content as normal text
            let mut normal_text = std::mem::take(&mut self.buffer);
            // Remove end tokens if present
            for e_token in ["<|tool_calls_section_end|>", "<|tool_call_end|>"] {
                normal_text = normal_text.replace(e_token, "");
            }
            return Ok(StreamingParseResult {
                normal_text,
                calls: vec![],
            });
        }

        // Build tool indices for validation
        let tool_indices = helpers::get_tool_indices(tools);

        let mut calls: Vec<ToolCallItem> = Vec::new();

        // Try to match streaming pattern
        if let Some(captures) = self.stream_tool_call_extractor.captures(current_text) {
            if let (Some(id_match), Some(args_match)) = (
                captures.name("tool_call_id"),
                captures.name("function_arguments"),
            ) {
                let function_id = id_match.as_str();
                let function_args = args_match.as_str();

                // Parse function ID
                if let Some((func_name, _index)) = self.parse_function_id(function_id) {
                    // Validate tool name
                    if !tool_indices.contains_key(&func_name) {
                        // Invalid tool name - skip this tool, preserve indexing for next tool
                        tracing::warn!("Invalid tool name '{}' - skipping", func_name);
                        helpers::reset_current_tool_state(
                            &mut self.buffer,
                            &mut self.current_tool_name_sent,
                            &mut self.streamed_args_for_tool,
                            &self.prev_tool_call_arr,
                        );
                        return Ok(StreamingParseResult::default());
                    }

                    // Initialize state if this is the first tool call
                    if self.current_tool_id == -1 {
                        self.current_tool_id = 0;
                        self.prev_tool_call_arr = Vec::new();
                        self.streamed_args_for_tool = vec![String::new()];
                    }

                    // Ensure we have enough entries in our tracking arrays
                    helpers::ensure_capacity(
                        self.current_tool_id,
                        &mut self.prev_tool_call_arr,
                        &mut self.streamed_args_for_tool,
                    );

                    // Send tool name if not sent yet
                    if !self.current_tool_name_sent {
                        calls.push(ToolCallItem {
                            tool_index: self.current_tool_id as usize,
                            name: Some(func_name.clone()),
                            parameters: String::new(),
                        });
                        self.current_tool_name_sent = true;

                        // Store the tool call info for serving layer completions endpoint
                        let tool_id = self.current_tool_id as usize;
                        if self.prev_tool_call_arr.len() <= tool_id {
                            self.prev_tool_call_arr
                                .resize_with(tool_id + 1, || Value::Null);
                        }
                        self.prev_tool_call_arr[tool_id] = serde_json::json!({
                            "name": func_name,
                            "arguments": {},
                        });
                    } else {
                        // Compute incremental diff
                        let argument_diff = if function_args.starts_with(&self.last_arguments) {
                            &function_args[self.last_arguments.len()..]
                        } else {
                            function_args
                        };

                        // Split by end token before sending (like Python does)
                        let parsed_args_diff =
                            if let Some(pos) = argument_diff.find("<|tool_call_end|>") {
                                &argument_diff[..pos]
                            } else {
                                argument_diff
                            };

                        if !parsed_args_diff.is_empty() {
                            calls.push(ToolCallItem {
                                tool_index: self.current_tool_id as usize,
                                name: None,
                                parameters: parsed_args_diff.to_string(),
                            });
                            // Note: Python adds full diff to _last_arguments, not just parsed part
                            self.last_arguments.push_str(argument_diff);
                            let tool_id = self.current_tool_id as usize;
                            if tool_id < self.streamed_args_for_tool.len() {
                                self.streamed_args_for_tool[tool_id].push_str(parsed_args_diff);
                            }
                        }

                        // Check completeness - split by end token first
                        let parsed_args = if let Some(pos) = function_args.find("<|tool_call_end|>")
                        {
                            &function_args[..pos]
                        } else {
                            function_args
                        };

                        if helpers::is_complete_json(parsed_args) {
                            // Update the stored arguments
                            if let Ok(parsed_args_value) =
                                serde_json::from_str::<Value>(parsed_args)
                            {
                                let tool_id = self.current_tool_id as usize;
                                if tool_id < self.prev_tool_call_arr.len() {
                                    if let Some(obj) =
                                        self.prev_tool_call_arr[tool_id].as_object_mut()
                                    {
                                        obj.insert("arguments".to_string(), parsed_args_value);
                                    }
                                }
                            }

                            // Find the end of the current tool call and remove only that part from buffer
                            if let Some(mat) = self.tool_call_end_pattern.find(current_text) {
                                // Remove the completed tool call from buffer, keep any remaining content
                                self.buffer = current_text[mat.end()..].to_string();
                            } else {
                                self.buffer.clear();
                            }

                            let result = StreamingParseResult {
                                normal_text: String::new(),
                                calls,
                            };

                            self.current_tool_id += 1;
                            self.last_arguments.clear();
                            self.current_tool_name_sent = false;
                            return Ok(result);
                        }
                    }
                }
            }
        }

        Ok(StreamingParseResult {
            normal_text: String::new(),
            calls,
        })
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains("<|tool_calls_section_begin|>")
    }

    fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallItem>> {
        helpers::get_unstreamed_args(&self.prev_tool_call_arr, &self.streamed_args_for_tool)
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.prev_tool_call_arr.clear();
        self.current_tool_id = -1;
        self.current_tool_name_sent = false;
        self.streamed_args_for_tool.clear();
        self.last_arguments.clear();
    }
}
