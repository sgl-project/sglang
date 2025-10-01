use std::collections::HashMap;
pub use crate::tool_parser::types::PartialToolCall;

/// Current parsing mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseMode {
    /// Looking for tool markers
    Scanning,
    /// Found tool start, accumulating content
    InToolCall,
    /// Parsing tool arguments
    InArguments,
    /// Tool section completed
    Complete,
}


/// State for streaming parser
#[derive(Debug, Clone)]
pub struct ParseState {
    /// Accumulated text buffer
    pub buffer: String,

    /// Current parsing mode
    pub mode: ParseMode,

    /// Active partial tool calls being streamed
    pub partial_tools: Vec<PartialToolCall>,

    /// Text before tool markers (normal content)
    pub prefix_text: String,

    /// Whether we've started streaming tool calls
    pub in_tool_section: bool,

    /// Current tool ID for tracking sequential tools
    pub current_tool_id: usize,

    /// Parser-specific state (for complex formats)
    pub parser_state: HashMap<String, String>,
}

impl ParseState {
    /// Create a new parse state
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            mode: ParseMode::Scanning,
            partial_tools: Vec::new(),
            prefix_text: String::new(),
            in_tool_section: false,
            current_tool_id: 0,
            parser_state: HashMap::new(),
        }
    }

    /// Reset state for parsing next tool
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.mode = ParseMode::Scanning;
        self.partial_tools.clear();
        self.prefix_text.clear();
        self.in_tool_section = false;
        self.current_tool_id = 0;
        self.parser_state.clear();
    }

    /// Ensure we have a PartialToolCall entry at the given index
    /// Returns a mutable reference to the partial tool at that index
    pub fn ensure_partial_tool(&mut self, index: usize) -> &mut PartialToolCall {
        // Grow the vector as needed
        while self.partial_tools.len() <= index {
            self.partial_tools.push(PartialToolCall {
                index: self.partial_tools.len(),
                name: None,
                name_sent: false,
                arguments_buffer: String::new(),
                streamed_arguments: String::new(),
                id: None,
            });
        }
        &mut self.partial_tools[index]
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
