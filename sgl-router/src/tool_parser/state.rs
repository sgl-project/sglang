use crate::tool_parser::types::{PartialToolCall, ToolCall};

/// Current phase of parsing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParsePhase {
    /// Looking for start of tool call
    Searching,
    /// Parsing function name
    InName,
    /// Parsing function arguments
    InArguments,
    /// Tool call complete
    Complete,
}

/// State for streaming parser
#[derive(Debug, Clone)]
pub struct ParseState {
    /// Buffer for accumulating input
    pub buffer: String,
    /// Position of last consumed character
    pub consumed: usize,
    /// Current partial tool being parsed
    pub partial_tool: Option<PartialToolCall>,
    /// Completed tool calls
    pub completed_tools: Vec<ToolCall>,
    /// Current parsing phase
    pub phase: ParsePhase,
    /// Bracket/brace depth for JSON parsing
    pub bracket_depth: i32,
    /// Whether currently inside a string literal
    pub in_string: bool,
    /// Whether next character should be escaped
    pub escape_next: bool,
    /// Current tool index (for streaming)
    pub tool_index: usize,
    /// Optional Harmony-specific streaming state (populated by token-aware parsers)
    pub harmony_stream: Option<HarmonyStreamState>,

    // vLLM-style stateful streaming tracking (following Python detectors pattern)
    /// Index of currently streaming tool call (-1 means no active tool)
    /// Tracks which tool's arguments are currently being streamed
    pub current_tool_id: i32,
    /// Flag for whether current tool's name has been sent to client
    /// Tool names are sent first with empty parameters, then arguments stream incrementally
    pub current_tool_name_sent: bool,
    /// Tracks raw JSON string content streamed to client for each tool's arguments
    /// Critical for calculating remaining content when streaming ends
    /// Each index corresponds to a tool_id (e.g., ['{"location": "San Francisco"', '{"temp": 72'])
    pub streamed_args_for_tool: Vec<String>,
    /// Stores complete tool call info (name and arguments) for each tool being parsed
    /// Used for calculating incremental diffs between parsing calls
    /// Format: [{"name": str, "arguments": dict}, ...]
    pub prev_tool_call_arr: Vec<serde_json::Map<String, serde_json::Value>>,
}

impl ParseState {
    /// Create a new parse state
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            consumed: 0,
            partial_tool: None,
            completed_tools: Vec::new(),
            phase: ParsePhase::Searching,
            bracket_depth: 0,
            in_string: false,
            escape_next: false,
            tool_index: 0,
            harmony_stream: None,
            current_tool_id: -1,
            current_tool_name_sent: false,
            streamed_args_for_tool: Vec::new(),
            prev_tool_call_arr: Vec::new(),
        }
    }

    /// Reset state for parsing next tool
    pub fn reset(&mut self) {
        self.partial_tool = None;
        self.phase = ParsePhase::Searching;
        self.bracket_depth = 0;
        self.in_string = false;
        self.escape_next = false;
        self.harmony_stream = None;
        self.current_tool_id = -1;
        self.current_tool_name_sent = false;
        self.streamed_args_for_tool.clear();
        self.prev_tool_call_arr.clear();
    }

    /// Process a single character for JSON parsing
    pub fn process_char(&mut self, ch: char) {
        // Handle escape sequences
        if self.escape_next {
            self.escape_next = false;
            self.buffer.push(ch);
            return;
        }

        if ch == '\\' && self.in_string {
            self.escape_next = true;
            self.buffer.push(ch);
            return;
        }

        // Track string boundaries
        if ch == '"' && !self.escape_next {
            self.in_string = !self.in_string;
        }

        // Track bracket depth for JSON
        if !self.in_string {
            match ch {
                '{' | '[' => {
                    self.bracket_depth += 1;
                }
                '}' | ']' => {
                    self.bracket_depth -= 1;
                    if self.bracket_depth == 0 && self.partial_tool.is_some() {
                        // Complete tool call found
                        self.phase = ParsePhase::Complete;
                    }
                }
                _ => {}
            }
        }

        self.buffer.push(ch);
    }

    /// Check if we have a complete JSON object/array
    pub fn has_complete_json(&self) -> bool {
        self.bracket_depth == 0 && !self.in_string && !self.buffer.is_empty()
    }

    /// Extract content from buffer starting at position
    pub fn extract_from(&self, start: usize) -> &str {
        if start >= self.buffer.len() {
            return "";
        }

        // Find the nearest character boundary at or after start
        let mut safe_start = start;
        while safe_start < self.buffer.len() && !self.buffer.is_char_boundary(safe_start) {
            safe_start += 1;
        }

        if safe_start < self.buffer.len() {
            &self.buffer[safe_start..]
        } else {
            ""
        }
    }

    /// Mark content as consumed up to position
    pub fn consume_to(&mut self, position: usize) {
        if position > self.consumed {
            self.consumed = position;
        }
    }

    /// Get unconsumed content
    pub fn unconsumed(&self) -> &str {
        if self.consumed >= self.buffer.len() {
            return "";
        }

        // Find the nearest character boundary at or after consumed
        let mut safe_consumed = self.consumed;
        while safe_consumed < self.buffer.len() && !self.buffer.is_char_boundary(safe_consumed) {
            safe_consumed += 1;
        }

        if safe_consumed < self.buffer.len() {
            &self.buffer[safe_consumed..]
        } else {
            ""
        }
    }

    /// Clear consumed content from buffer
    pub fn clear_consumed(&mut self) {
        if self.consumed > 0 {
            // Find the nearest character boundary at or before consumed
            let mut safe_consumed = self.consumed;
            while safe_consumed > 0 && !self.buffer.is_char_boundary(safe_consumed) {
                safe_consumed -= 1;
            }

            if safe_consumed > 0 {
                self.buffer.drain(..safe_consumed);
                self.consumed = self.consumed.saturating_sub(safe_consumed);
            }
        }
    }

    /// Add completed tool
    pub fn add_completed_tool(&mut self, tool: ToolCall) {
        self.completed_tools.push(tool);
        self.tool_index += 1;
    }
}

impl Default for ParseState {
    fn default() -> Self {
        Self::new()
    }
}

/// Placeholder for Harmony streaming metadata captured during token-aware parsing.
#[derive(Debug, Clone, Default)]
pub struct HarmonyStreamState {
    /// All tokens observed so far for the current assistant response.
    pub tokens: Vec<u32>,
    /// Number of tokens that have already been processed by the Harmony parser.
    pub processed_tokens: usize,
    /// Number of tool calls emitted downstream.
    pub emitted_calls: usize,
    /// Pending analysis-channel content awaiting flush into normal text output.
    pub analysis_buffer: String,
    /// Whether the tool name has been surfaced for the current call.
    pub emitted_name: bool,
    /// Whether arguments have been surfaced for the current call.
    pub emitted_args: bool,
}
