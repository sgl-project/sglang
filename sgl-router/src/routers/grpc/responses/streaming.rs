//! Streaming infrastructure for /v1/responses endpoint

use std::collections::HashMap;

use bytes::Bytes;
use serde_json::json;
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::protocols::chat::ChatCompletionStreamResponse;

pub(super) enum OutputItemType {
    Message,
    McpListTools,
    McpCall,
    Reasoning,
}

/// Status of an output item
#[derive(Debug, Clone, PartialEq)]
enum ItemStatus {
    InProgress,
    Completed,
}

/// State tracking for a single output item
#[derive(Debug, Clone)]
struct OutputItemState {
    output_index: usize,
    status: ItemStatus,
}

// ============================================================================
// Streaming Event Emitter
// ============================================================================

/// OpenAI-compatible event emitter for /v1/responses streaming
///
/// Manages state and sequence numbers to emit proper event types:
/// - response.created
/// - response.in_progress
/// - response.output_item.added
/// - response.content_part.added
/// - response.output_text.delta (multiple)
/// - response.output_text.done
/// - response.content_part.done
/// - response.output_item.done
/// - response.completed
/// - response.mcp_list_tools.in_progress
/// - response.mcp_list_tools.completed
/// - response.mcp_call.in_progress
/// - response.mcp_call_arguments.delta
/// - response.mcp_call_arguments.done
/// - response.mcp_call.completed
/// - response.mcp_call.failed
pub(super) struct ResponseStreamEventEmitter {
    sequence_number: u64,
    response_id: String,
    model: String,
    created_at: u64,
    message_id: String,
    accumulated_text: String,
    has_emitted_created: bool,
    has_emitted_in_progress: bool,
    has_emitted_output_item_added: bool,
    has_emitted_content_part_added: bool,
    // MCP call tracking
    mcp_call_accumulated_args: HashMap<String, String>,
    // Output item tracking (NEW)
    output_items: Vec<OutputItemState>,
    next_output_index: usize,
    current_message_output_index: Option<usize>, // Tracks output_index of current message
    current_item_id: Option<String>,             // Tracks item_id of current item
}

impl ResponseStreamEventEmitter {
    pub(super) fn new(response_id: String, model: String, created_at: u64) -> Self {
        let message_id = format!("msg_{}", Uuid::new_v4());

        Self {
            sequence_number: 0,
            response_id,
            model,
            created_at,
            message_id,
            accumulated_text: String::new(),
            has_emitted_created: false,
            has_emitted_in_progress: false,
            has_emitted_output_item_added: false,
            has_emitted_content_part_added: false,
            mcp_call_accumulated_args: HashMap::new(),
            output_items: Vec::new(),
            next_output_index: 0,
            current_message_output_index: None,
            current_item_id: None,
        }
    }

    fn next_sequence(&mut self) -> u64 {
        let seq = self.sequence_number;
        self.sequence_number += 1;
        seq
    }

    pub(super) fn emit_created(&mut self) -> serde_json::Value {
        self.has_emitted_created = true;
        json!({
            "type": "response.created",
            "sequence_number": self.next_sequence(),
            "response": {
                "id": self.response_id,
                "object": "response",
                "created_at": self.created_at,
                "status": "in_progress",
                "model": self.model,
                "output": []
            }
        })
    }

    pub(super) fn emit_in_progress(&mut self) -> serde_json::Value {
        self.has_emitted_in_progress = true;
        json!({
            "type": "response.in_progress",
            "sequence_number": self.next_sequence(),
            "response": {
                "id": self.response_id,
                "object": "response",
                "status": "in_progress"
            }
        })
    }

    pub(super) fn emit_content_part_added(
        &mut self,
        output_index: usize,
        item_id: &str,
        content_index: usize,
    ) -> serde_json::Value {
        self.has_emitted_content_part_added = true;
        json!({
            "type": "response.content_part.added",
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "content_index": content_index,
            "part": {
                "type": "text",
                "text": ""
            }
        })
    }

    pub(super) fn emit_text_delta(
        &mut self,
        delta: &str,
        output_index: usize,
        item_id: &str,
        content_index: usize,
    ) -> serde_json::Value {
        self.accumulated_text.push_str(delta);
        json!({
            "type": "response.output_text.delta",
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "content_index": content_index,
            "delta": delta
        })
    }

    pub(super) fn emit_text_done(
        &mut self,
        output_index: usize,
        item_id: &str,
        content_index: usize,
    ) -> serde_json::Value {
        json!({
            "type": "response.output_text.done",
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "content_index": content_index,
            "text": self.accumulated_text.clone()
        })
    }

    pub(super) fn emit_content_part_done(
        &mut self,
        output_index: usize,
        item_id: &str,
        content_index: usize,
    ) -> serde_json::Value {
        json!({
            "type": "response.content_part.done",
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "content_index": content_index,
            "part": {
                "type": "text",
                "text": self.accumulated_text.clone()
            }
        })
    }

    pub(super) fn emit_completed(
        &mut self,
        usage: Option<&serde_json::Value>,
    ) -> serde_json::Value {
        let mut response = json!({
            "type": "response.completed",
            "sequence_number": self.next_sequence(),
            "response": {
                "id": self.response_id,
                "object": "response",
                "created_at": self.created_at,
                "status": "completed",
                "model": self.model,
                "output": [{
                    "id": self.message_id.clone(),
                    "type": "message",
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": self.accumulated_text.clone()
                    }]
                }]
            }
        });

        if let Some(usage_val) = usage {
            response["response"]["usage"] = usage_val.clone();
        }

        response
    }

    // ========================================================================
    // MCP Event Emission Methods
    // ========================================================================

    pub(super) fn emit_mcp_list_tools_in_progress(
        &mut self,
        output_index: usize,
    ) -> serde_json::Value {
        json!({
            "type": "response.mcp_list_tools.in_progress",
            "sequence_number": self.next_sequence(),
            "output_index": output_index
        })
    }

    pub(super) fn emit_mcp_list_tools_completed(
        &mut self,
        output_index: usize,
        tools: &[crate::mcp::ToolInfo],
    ) -> serde_json::Value {
        let tool_items: Vec<_> = tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters.clone().unwrap_or_else(|| json!({
                        "type": "object",
                        "properties": {},
                        "required": []
                    }))
                })
            })
            .collect();

        json!({
            "type": "response.mcp_list_tools.completed",
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "tools": tool_items
        })
    }

    pub(super) fn emit_mcp_call_in_progress(
        &mut self,
        output_index: usize,
        item_id: &str,
    ) -> serde_json::Value {
        json!({
            "type": "response.mcp_call.in_progress",
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id
        })
    }

    pub(super) fn emit_mcp_call_arguments_delta(
        &mut self,
        output_index: usize,
        item_id: &str,
        delta: &str,
    ) -> serde_json::Value {
        // Accumulate arguments for this call
        self.mcp_call_accumulated_args
            .entry(item_id.to_string())
            .or_default()
            .push_str(delta);

        json!({
            "type": "response.mcp_call_arguments.delta",
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "delta": delta
        })
    }

    pub(super) fn emit_mcp_call_arguments_done(
        &mut self,
        output_index: usize,
        item_id: &str,
        arguments: &str,
    ) -> serde_json::Value {
        json!({
            "type": "response.mcp_call_arguments.done",
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "arguments": arguments
        })
    }

    pub(super) fn emit_mcp_call_completed(
        &mut self,
        output_index: usize,
        item_id: &str,
    ) -> serde_json::Value {
        json!({
            "type": "response.mcp_call.completed",
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id
        })
    }

    pub(super) fn emit_mcp_call_failed(
        &mut self,
        output_index: usize,
        item_id: &str,
        error: &str,
    ) -> serde_json::Value {
        json!({
            "type": "response.mcp_call.failed",
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "error": error
        })
    }

    // ========================================================================
    // Output Item Wrapper Events
    // ========================================================================

    /// Emit response.output_item.added event
    pub(super) fn emit_output_item_added(
        &mut self,
        output_index: usize,
        item: &serde_json::Value,
    ) -> serde_json::Value {
        json!({
            "type": "response.output_item.added",
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item": item
        })
    }

    /// Emit response.output_item.done event
    pub(super) fn emit_output_item_done(
        &mut self,
        output_index: usize,
        item: &serde_json::Value,
    ) -> serde_json::Value {
        json!({
            "type": "response.output_item.done",
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item": item
        })
    }

    /// Generate unique ID for item type
    fn generate_item_id(prefix: &str) -> String {
        format!("{}_{}", prefix, Uuid::new_v4().to_string().replace("-", ""))
    }

    /// Allocate next output index and track item
    pub(super) fn allocate_output_index(&mut self, item_type: OutputItemType) -> (usize, String) {
        let index = self.next_output_index;
        self.next_output_index += 1;

        let id_prefix = match &item_type {
            OutputItemType::McpListTools => "mcpl",
            OutputItemType::McpCall => "mcp",
            OutputItemType::Message => "msg",
            OutputItemType::Reasoning => "rs",
        };

        let id = Self::generate_item_id(id_prefix);

        self.output_items.push(OutputItemState {
            output_index: index,
            status: ItemStatus::InProgress,
        });

        (index, id)
    }

    /// Mark output item as completed
    pub(super) fn complete_output_item(&mut self, output_index: usize) {
        if let Some(item) = self
            .output_items
            .iter_mut()
            .find(|i| i.output_index == output_index)
        {
            item.status = ItemStatus::Completed;
        }
    }

    /// Emit reasoning item wrapper events (added + done)
    ///
    /// Reasoning items in OpenAI format are simple placeholders emitted between tool iterations.
    /// They don't have streaming content - just wrapper events with empty/null content.
    pub(super) fn emit_reasoning_item(
        &mut self,
        tx: &mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
        reasoning_content: Option<String>,
    ) -> Result<(), String> {
        // Allocate output index and generate ID
        let (output_index, item_id) = self.allocate_output_index(OutputItemType::Reasoning);

        // Build reasoning item structure
        let item = json!({
            "id": item_id,
            "type": "reasoning",
            "summary": [],
            "content": reasoning_content,
            "encrypted_content": null,
            "status": null
        });

        // Emit output_item.added
        let added_event = self.emit_output_item_added(output_index, &item);
        self.send_event(&added_event, tx)?;

        // Immediately emit output_item.done (no streaming for reasoning)
        let done_event = self.emit_output_item_done(output_index, &item);
        self.send_event(&done_event, tx)?;

        // Mark as completed
        self.complete_output_item(output_index);

        Ok(())
    }

    /// Process a chunk and emit appropriate events
    pub(super) fn process_chunk(
        &mut self,
        chunk: &ChatCompletionStreamResponse,
        tx: &mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
    ) -> Result<(), String> {
        // Process content if present
        if let Some(choice) = chunk.choices.first() {
            if let Some(content) = &choice.delta.content {
                if !content.is_empty() {
                    // Allocate output_index and item_id for this message item (once per message)
                    if self.current_item_id.is_none() {
                        let (output_index, item_id) =
                            self.allocate_output_index(OutputItemType::Message);

                        // Build message item structure
                        let item = json!({
                            "id": item_id,
                            "type": "message",
                            "role": "assistant",
                            "content": []
                        });

                        // Emit output_item.added
                        let event = self.emit_output_item_added(output_index, &item);
                        self.send_event(&event, tx)?;
                        self.has_emitted_output_item_added = true;

                        // Store for subsequent events
                        self.current_item_id = Some(item_id);
                        self.current_message_output_index = Some(output_index);
                    }

                    let output_index = self.current_message_output_index.unwrap();
                    let item_id = self.current_item_id.clone().unwrap(); // Clone to avoid borrow checker issues
                    let content_index = 0; // Single content part for now

                    // Emit content_part.added before first delta
                    if !self.has_emitted_content_part_added {
                        let event =
                            self.emit_content_part_added(output_index, &item_id, content_index);
                        self.send_event(&event, tx)?;
                        self.has_emitted_content_part_added = true;
                    }

                    // Emit text delta
                    let event =
                        self.emit_text_delta(content, output_index, &item_id, content_index);
                    self.send_event(&event, tx)?;
                }
            }

            // Check for finish_reason to emit completion events
            if let Some(reason) = &choice.finish_reason {
                if reason == "stop" || reason == "length" {
                    let output_index = self.current_message_output_index.unwrap();
                    let item_id = self.current_item_id.clone().unwrap(); // Clone to avoid borrow checker issues
                    let content_index = 0;

                    // Emit closing events
                    if self.has_emitted_content_part_added {
                        let event = self.emit_text_done(output_index, &item_id, content_index);
                        self.send_event(&event, tx)?;
                        let event =
                            self.emit_content_part_done(output_index, &item_id, content_index);
                        self.send_event(&event, tx)?;
                    }

                    if self.has_emitted_output_item_added {
                        // Build complete message item for output_item.done
                        let item = json!({
                            "id": item_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [{
                                "type": "text",
                                "text": self.accumulated_text.clone()
                            }]
                        });
                        let event = self.emit_output_item_done(output_index, &item);
                        self.send_event(&event, tx)?;
                    }

                    // Mark item as completed
                    self.complete_output_item(output_index);
                }
            }
        }

        Ok(())
    }

    pub(super) fn send_event(
        &self,
        event: &serde_json::Value,
        tx: &mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
    ) -> Result<(), String> {
        let event_json = serde_json::to_string(event)
            .map_err(|e| format!("Failed to serialize event: {}", e))?;

        if tx
            .send(Ok(Bytes::from(format!("data: {}\n\n", event_json))))
            .is_err()
        {
            return Err("Client disconnected".to_string());
        }

        Ok(())
    }
}
