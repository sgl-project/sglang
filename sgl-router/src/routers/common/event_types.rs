//! SSE Event Type Constants
//!
//! Single source of truth for all /v1/responses SSE event type strings.
//! Used by both OpenAI router (for transformation) and gRPC router (for generation).

// ============================================================================
// Response Lifecycle Events
// ============================================================================

pub const RESPONSE_CREATED: &str = "response.created";
pub const RESPONSE_IN_PROGRESS: &str = "response.in_progress";
pub const RESPONSE_COMPLETED: &str = "response.completed";
pub const RESPONSE_ERROR: &str = "response.error";

// ============================================================================
// Output Item Events
// ============================================================================

pub const OUTPUT_ITEM_ADDED: &str = "response.output_item.added";
pub const OUTPUT_ITEM_DONE: &str = "response.output_item.done";
pub const OUTPUT_ITEM_DELTA: &str = "response.output_item.delta";

// ============================================================================
// Function Call Events (OpenAI upstream format)
// ============================================================================

pub const FUNCTION_CALL_ARGUMENTS_DELTA: &str = "response.function_call_arguments.delta";
pub const FUNCTION_CALL_ARGUMENTS_DONE: &str = "response.function_call_arguments.done";

// ============================================================================
// MCP Events (client-facing format)
// ============================================================================

pub const MCP_CALL_IN_PROGRESS: &str = "response.mcp_call.in_progress";
pub const MCP_CALL_ARGUMENTS_DELTA: &str = "response.mcp_call_arguments.delta";
pub const MCP_CALL_ARGUMENTS_DONE: &str = "response.mcp_call_arguments.done";
pub const MCP_CALL_COMPLETED: &str = "response.mcp_call.completed";
pub const MCP_CALL_FAILED: &str = "response.mcp_call.failed";

pub const MCP_LIST_TOOLS_IN_PROGRESS: &str = "response.mcp_list_tools.in_progress";
pub const MCP_LIST_TOOLS_COMPLETED: &str = "response.mcp_list_tools.completed";

// ============================================================================
// Content Events
// ============================================================================

pub const CONTENT_PART_ADDED: &str = "response.content_part.added";
pub const CONTENT_PART_DONE: &str = "response.content_part.done";

pub const OUTPUT_TEXT_DELTA: &str = "response.output_text.delta";
pub const OUTPUT_TEXT_DONE: &str = "response.output_text.done";

// ============================================================================
// Item Types
// ============================================================================

pub const ITEM_TYPE_FUNCTION_CALL: &str = "function_call";
pub const ITEM_TYPE_FUNCTION_TOOL_CALL: &str = "function_tool_call";
pub const ITEM_TYPE_FUNCTION: &str = "function";
pub const ITEM_TYPE_MCP_CALL: &str = "mcp_call";
pub const ITEM_TYPE_MCP_LIST_TOOLS: &str = "mcp_list_tools";
pub const ITEM_TYPE_MESSAGE: &str = "message";
pub const ITEM_TYPE_REASONING: &str = "reasoning";
