use std::fmt;

/// Response lifecycle events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResponseEvent {
    Created,
    InProgress,
    Completed,
}

impl ResponseEvent {
    pub const CREATED: &'static str = "response.created";
    pub const IN_PROGRESS: &'static str = "response.in_progress";
    pub const COMPLETED: &'static str = "response.completed";

    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Created => Self::CREATED,
            Self::InProgress => Self::IN_PROGRESS,
            Self::Completed => Self::COMPLETED,
        }
    }
}

impl fmt::Display for ResponseEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Output item events for streaming
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputItemEvent {
    Added,
    Done,
    Delta,
}

impl OutputItemEvent {
    pub const ADDED: &'static str = "response.output_item.added";
    pub const DONE: &'static str = "response.output_item.done";
    pub const DELTA: &'static str = "response.output_item.delta";

    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Added => Self::ADDED,
            Self::Done => Self::DONE,
            Self::Delta => Self::DELTA,
        }
    }
}

impl fmt::Display for OutputItemEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Function call argument streaming events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FunctionCallEvent {
    ArgumentsDelta,
    ArgumentsDone,
}

impl FunctionCallEvent {
    pub const ARGUMENTS_DELTA: &'static str = "response.function_call_arguments.delta";
    pub const ARGUMENTS_DONE: &'static str = "response.function_call_arguments.done";

    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::ArgumentsDelta => Self::ARGUMENTS_DELTA,
            Self::ArgumentsDone => Self::ARGUMENTS_DONE,
        }
    }
}

impl fmt::Display for FunctionCallEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Content part streaming events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContentPartEvent {
    Added,
    Done,
}

impl ContentPartEvent {
    pub const ADDED: &'static str = "response.content_part.added";
    pub const DONE: &'static str = "response.content_part.done";

    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Added => Self::ADDED,
            Self::Done => Self::DONE,
        }
    }
}

impl fmt::Display for ContentPartEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Output text streaming events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputTextEvent {
    Delta,
    Done,
}

impl OutputTextEvent {
    pub const DELTA: &'static str = "response.output_text.delta";
    pub const DONE: &'static str = "response.output_text.done";

    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Delta => Self::DELTA,
            Self::Done => Self::DONE,
        }
    }
}

impl fmt::Display for OutputTextEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ============================================================================
// MCP Events
// ============================================================================

/// MCP (Model Context Protocol) call events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum McpEvent {
    CallArgumentsDelta,
    CallArgumentsDone,
    CallInProgress,
    CallCompleted,
    CallFailed,
    ListToolsInProgress,
    ListToolsCompleted,
}

impl McpEvent {
    pub const CALL_ARGUMENTS_DELTA: &'static str = "response.mcp_call_arguments.delta";
    pub const CALL_ARGUMENTS_DONE: &'static str = "response.mcp_call_arguments.done";
    pub const CALL_IN_PROGRESS: &'static str = "response.mcp_call.in_progress";
    pub const CALL_COMPLETED: &'static str = "response.mcp_call.completed";
    pub const CALL_FAILED: &'static str = "response.mcp_call.failed";
    pub const LIST_TOOLS_IN_PROGRESS: &'static str = "response.mcp_list_tools.in_progress";
    pub const LIST_TOOLS_COMPLETED: &'static str = "response.mcp_list_tools.completed";

    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::CallArgumentsDelta => Self::CALL_ARGUMENTS_DELTA,
            Self::CallArgumentsDone => Self::CALL_ARGUMENTS_DONE,
            Self::CallInProgress => Self::CALL_IN_PROGRESS,
            Self::CallCompleted => Self::CALL_COMPLETED,
            Self::CallFailed => Self::CALL_FAILED,
            Self::ListToolsInProgress => Self::LIST_TOOLS_IN_PROGRESS,
            Self::ListToolsCompleted => Self::LIST_TOOLS_COMPLETED,
        }
    }
}

impl fmt::Display for McpEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Item type discriminators used in output items
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ItemType {
    FunctionCall,
    FunctionToolCall,
    McpCall,
    Function,
    McpListTools,
}

impl ItemType {
    pub const FUNCTION_CALL: &'static str = "function_call";
    pub const FUNCTION_TOOL_CALL: &'static str = "function_tool_call";
    pub const MCP_CALL: &'static str = "mcp_call";
    pub const FUNCTION: &'static str = "function";
    pub const MCP_LIST_TOOLS: &'static str = "mcp_list_tools";

    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::FunctionCall => Self::FUNCTION_CALL,
            Self::FunctionToolCall => Self::FUNCTION_TOOL_CALL,
            Self::McpCall => Self::MCP_CALL,
            Self::Function => Self::FUNCTION,
            Self::McpListTools => Self::MCP_LIST_TOOLS,
        }
    }

    /// Check if this is a function call variant (FunctionCall or FunctionToolCall)
    pub const fn is_function_call(&self) -> bool {
        matches!(self, Self::FunctionCall | Self::FunctionToolCall)
    }
}

impl fmt::Display for ItemType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Check if an event type string matches any response lifecycle event
pub fn is_response_event(event_type: &str) -> bool {
    matches!(
        event_type,
        ResponseEvent::CREATED | ResponseEvent::IN_PROGRESS | ResponseEvent::COMPLETED
    )
}

/// Check if an item type string is a function call variant
pub fn is_function_call_type(item_type: &str) -> bool {
    item_type == ItemType::FUNCTION_CALL || item_type == ItemType::FUNCTION_TOOL_CALL
}
