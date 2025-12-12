//! Utility types and constants for OpenAI router

use std::collections::HashMap;

// ============================================================================
// SSE Event Type Constants
// ============================================================================

/// SSE event type constants - single source of truth for event type strings
pub(crate) mod event_types {
    // Response lifecycle events
    pub const RESPONSE_CREATED: &str = "response.created";
    pub const RESPONSE_IN_PROGRESS: &str = "response.in_progress";
    pub const RESPONSE_COMPLETED: &str = "response.completed";

    // Output item events
    pub const OUTPUT_ITEM_ADDED: &str = "response.output_item.added";
    pub const OUTPUT_ITEM_DONE: &str = "response.output_item.done";
    pub const OUTPUT_ITEM_DELTA: &str = "response.output_item.delta";

    // Function call events
    pub const FUNCTION_CALL_ARGUMENTS_DELTA: &str = "response.function_call_arguments.delta";
    pub const FUNCTION_CALL_ARGUMENTS_DONE: &str = "response.function_call_arguments.done";

    // MCP call events
    pub const MCP_CALL_ARGUMENTS_DELTA: &str = "response.mcp_call_arguments.delta";
    pub const MCP_CALL_ARGUMENTS_DONE: &str = "response.mcp_call_arguments.done";
    pub const MCP_CALL_IN_PROGRESS: &str = "response.mcp_call.in_progress";
    pub const MCP_CALL_COMPLETED: &str = "response.mcp_call.completed";
    pub const MCP_LIST_TOOLS_IN_PROGRESS: &str = "response.mcp_list_tools.in_progress";
    pub const MCP_LIST_TOOLS_COMPLETED: &str = "response.mcp_list_tools.completed";

    // Item types
    pub const ITEM_TYPE_FUNCTION_CALL: &str = "function_call";
    pub const ITEM_TYPE_FUNCTION_TOOL_CALL: &str = "function_tool_call";
    pub const ITEM_TYPE_MCP_CALL: &str = "mcp_call";
    pub const ITEM_TYPE_FUNCTION: &str = "function";
    pub const ITEM_TYPE_MCP_LIST_TOOLS: &str = "mcp_list_tools";
}

// ============================================================================
// Stream Action Enum
// ============================================================================

/// Action to take based on streaming event processing
#[derive(Debug)]
pub(crate) enum StreamAction {
    Forward,      // Pass event to client
    Buffer,       // Accumulate for tool execution
    ExecuteTools, // Function call complete, execute now
}

// ============================================================================
// Output Index Mapper
// ============================================================================

/// Maps upstream output indices to sequential downstream indices
#[derive(Debug, Default)]
pub(crate) struct OutputIndexMapper {
    next_index: usize,
    // Map upstream output_index -> remapped output_index
    assigned: HashMap<usize, usize>,
}

impl OutputIndexMapper {
    pub fn with_start(next_index: usize) -> Self {
        Self {
            next_index,
            assigned: HashMap::new(),
        }
    }

    pub fn ensure_mapping(&mut self, upstream_index: usize) -> usize {
        *self.assigned.entry(upstream_index).or_insert_with(|| {
            let assigned = self.next_index;
            self.next_index += 1;
            assigned
        })
    }

    pub fn lookup(&self, upstream_index: usize) -> Option<usize> {
        self.assigned.get(&upstream_index).copied()
    }

    pub fn allocate_synthetic(&mut self) -> usize {
        let assigned = self.next_index;
        self.next_index += 1;
        assigned
    }

    pub fn next_index(&self) -> usize {
        self.next_index
    }
}

// ============================================================================
// Re-export FunctionCallInProgress from mcp module
// ============================================================================

pub(crate) use super::mcp::FunctionCallInProgress;
