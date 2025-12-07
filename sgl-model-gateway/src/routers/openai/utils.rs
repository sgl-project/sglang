//! Utility types for OpenAI router

use std::collections::HashMap;

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
