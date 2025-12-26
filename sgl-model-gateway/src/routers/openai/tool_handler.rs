//! Streaming tool call handling for MCP interception.

use std::collections::HashMap;

use serde_json::Value;
use tracing::warn;

use super::{
    accumulator::StreamingResponseAccumulator,
    mcp::FunctionCallInProgress,
    streaming::{extract_output_index, get_event_type},
};
use crate::protocols::event_types::{
    is_function_call_type, FunctionCallEvent, OutputItemEvent, ResponseEvent,
};

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
// Streaming Tool Handler
// ============================================================================

/// Handles streaming responses with MCP tool call interception
pub(super) struct StreamingToolHandler {
    /// Accumulator for response persistence
    pub accumulator: StreamingResponseAccumulator,
    /// Function calls being built from deltas
    pub pending_calls: Vec<FunctionCallInProgress>,
    /// Track if we're currently in a function call
    in_function_call: bool,
    /// Manage output_index remapping so they increment per item
    output_index_mapper: OutputIndexMapper,
    /// Original response id captured from the first response.created event
    pub original_response_id: Option<String>,
}

impl StreamingToolHandler {
    pub fn with_starting_index(start: usize) -> Self {
        Self {
            accumulator: StreamingResponseAccumulator::new(),
            pending_calls: Vec::new(),
            in_function_call: false,
            output_index_mapper: OutputIndexMapper::with_start(start),
            original_response_id: None,
        }
    }

    pub fn ensure_output_index(&mut self, upstream_index: usize) -> usize {
        self.output_index_mapper.ensure_mapping(upstream_index)
    }

    pub fn mapped_output_index(&self, upstream_index: usize) -> Option<usize> {
        self.output_index_mapper.lookup(upstream_index)
    }

    pub fn allocate_synthetic_output_index(&mut self) -> usize {
        self.output_index_mapper.allocate_synthetic()
    }

    pub fn next_output_index(&self) -> usize {
        self.output_index_mapper.next_index()
    }

    pub fn original_response_id(&self) -> Option<&str> {
        self.original_response_id
            .as_deref()
            .or_else(|| self.accumulator.original_response_id())
    }

    pub fn snapshot_final_response(&self) -> Option<Value> {
        self.accumulator.snapshot_final_response()
    }

    /// Process an SSE event and determine what action to take
    pub fn process_event(&mut self, event_name: Option<&str>, data: &str) -> StreamAction {
        // Always feed to accumulator for storage
        self.accumulator.ingest_block(&format!(
            "{}data: {}",
            event_name
                .map(|n| format!("event: {}\n", n))
                .unwrap_or_default(),
            data
        ));

        let parsed: Value = match serde_json::from_str(data) {
            Ok(v) => v,
            Err(_) => return StreamAction::Forward,
        };

        match get_event_type(event_name, &parsed) {
            ResponseEvent::CREATED => {
                if self.original_response_id.is_none() {
                    self.original_response_id = parsed
                        .get("response")
                        .and_then(|v| v.get("id"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                }
                StreamAction::Forward
            }
            ResponseEvent::COMPLETED => StreamAction::Forward,
            OutputItemEvent::ADDED => self.handle_output_item_added(&parsed),
            FunctionCallEvent::ARGUMENTS_DELTA => self.handle_arguments_delta(&parsed),
            FunctionCallEvent::ARGUMENTS_DONE => self.handle_arguments_done(&parsed),
            OutputItemEvent::DELTA => self.process_output_delta(&parsed),
            OutputItemEvent::DONE => {
                if let Some(output_index) = extract_output_index(&parsed) {
                    self.ensure_output_index(output_index);
                }
                if self.has_complete_calls() {
                    StreamAction::ExecuteTools
                } else {
                    StreamAction::Forward
                }
            }
            _ => StreamAction::Forward,
        }
    }

    fn handle_output_item_added(&mut self, parsed: &Value) -> StreamAction {
        if let Some(output_index) = extract_output_index(parsed) {
            self.ensure_output_index(output_index);
        }

        // Check if this is a function_call item being added
        let Some(item) = parsed.get("item") else {
            return StreamAction::Forward;
        };
        let Some(item_type) = item.get("type").and_then(|v| v.as_str()) else {
            return StreamAction::Forward;
        };

        if !is_function_call_type(item_type) {
            return StreamAction::Forward;
        }

        let Some(output_index) = extract_output_index(parsed) else {
            warn!(
                "Missing output_index in function_call added event, \
                 forwarding without processing for tool execution"
            );
            return StreamAction::Forward;
        };

        let assigned_index = self.ensure_output_index(output_index);
        let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
        let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");

        let call = self.get_or_create_call(output_index, item);
        call.call_id = call_id.to_string();
        call.name = name.to_string();
        call.assigned_output_index = Some(assigned_index);
        self.in_function_call = true;

        StreamAction::Forward
    }

    fn handle_arguments_delta(&mut self, parsed: &Value) -> StreamAction {
        let Some(output_index) = extract_output_index(parsed) else {
            return StreamAction::Forward;
        };

        let assigned_index = self.ensure_output_index(output_index);

        if let Some(delta) = parsed.get("delta").and_then(|v| v.as_str()) {
            if let Some(call) = self.find_call_mut(output_index) {
                call.arguments_buffer.push_str(delta);
                if let Some(obfuscation) = parsed.get("obfuscation").and_then(|v| v.as_str()) {
                    call.last_obfuscation = Some(obfuscation.to_string());
                }
                if call.assigned_output_index.is_none() {
                    call.assigned_output_index = Some(assigned_index);
                }
            }
        }
        StreamAction::Forward
    }

    fn handle_arguments_done(&mut self, parsed: &Value) -> StreamAction {
        if let Some(output_index) = extract_output_index(parsed) {
            let assigned_index = self.ensure_output_index(output_index);
            if let Some(call) = self.find_call_mut(output_index) {
                if call.assigned_output_index.is_none() {
                    call.assigned_output_index = Some(assigned_index);
                }
            }
        }

        if self.has_complete_calls() {
            StreamAction::ExecuteTools
        } else {
            StreamAction::Forward
        }
    }

    fn find_call_mut(&mut self, output_index: usize) -> Option<&mut FunctionCallInProgress> {
        self.pending_calls
            .iter_mut()
            .find(|c| c.output_index == output_index)
    }

    /// Process output delta events to detect and accumulate function calls
    fn process_output_delta(&mut self, event: &Value) -> StreamAction {
        let output_index = extract_output_index(event).unwrap_or(0);
        let assigned_index = self.ensure_output_index(output_index);

        let delta = match event.get("delta") {
            Some(d) => d,
            None => return StreamAction::Forward,
        };

        // Check if this is a function call delta
        let item_type = delta.get("type").and_then(|v| v.as_str());

        if item_type.is_some_and(is_function_call_type) {
            self.in_function_call = true;

            // Get or create function call for this output index
            let call = self.get_or_create_call(output_index, delta);
            call.assigned_output_index = Some(assigned_index);

            // Accumulate call_id if present
            if let Some(call_id) = delta.get("call_id").and_then(|v| v.as_str()) {
                call.call_id = call_id.to_string();
            }

            // Accumulate name if present
            if let Some(name) = delta.get("name").and_then(|v| v.as_str()) {
                call.name.push_str(name);
            }

            // Accumulate arguments if present
            if let Some(args) = delta.get("arguments").and_then(|v| v.as_str()) {
                call.arguments_buffer.push_str(args);
            }

            if let Some(obfuscation) = delta.get("obfuscation").and_then(|v| v.as_str()) {
                call.last_obfuscation = Some(obfuscation.to_string());
            }

            // Buffer this event, don't forward to client
            return StreamAction::Buffer;
        }

        // Forward non-function-call events
        StreamAction::Forward
    }

    fn get_or_create_call(
        &mut self,
        output_index: usize,
        delta: &Value,
    ) -> &mut FunctionCallInProgress {
        // Find existing call for this output index
        if let Some(pos) = self
            .pending_calls
            .iter()
            .position(|c| c.output_index == output_index)
        {
            return &mut self.pending_calls[pos];
        }

        // Create new call
        let call_id = delta
            .get("call_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let mut call = FunctionCallInProgress::new(call_id, output_index);
        if let Some(obfuscation) = delta.get("obfuscation").and_then(|v| v.as_str()) {
            call.last_obfuscation = Some(obfuscation.to_string());
        }

        self.pending_calls.push(call);
        self.pending_calls
            .last_mut()
            .expect("Just pushed to pending_calls, must have at least one element")
    }

    fn has_complete_calls(&self) -> bool {
        !self.pending_calls.is_empty() && self.pending_calls.iter().all(|c| c.is_complete())
    }

    pub fn take_pending_calls(&mut self) -> Vec<FunctionCallInProgress> {
        std::mem::take(&mut self.pending_calls)
    }
}
