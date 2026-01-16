//! Streaming infrastructure for /v1/responses endpoint

use std::collections::HashMap;

use axum::{body::Body, http::StatusCode, response::Response};
use bytes::Bytes;
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use uuid::Uuid;

use crate::{
    mcp,
    protocols::{
        chat::ChatCompletionStreamResponse,
        common::{Usage, UsageInfo},
        event_types::{
            ContentPartEvent, FunctionCallEvent, McpEvent, OutputItemEvent, OutputTextEvent,
            ResponseEvent,
        },
        responses::{
            ResponseOutputItem, ResponseStatus, ResponsesRequest, ResponsesResponse, ResponsesUsage,
        },
    },
    routers::grpc::harmony::responses::ToolResult,
};

pub(crate) enum OutputItemType {
    Message,
    McpListTools,
    McpCall,
    FunctionCall,
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
    item_data: Option<serde_json::Value>,
}

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
pub(crate) struct ResponseStreamEventEmitter {
    sequence_number: u64,
    pub response_id: String,
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
    pub(crate) mcp_tool_labels: Option<HashMap<String, String>>,
    // Output item tracking
    output_items: Vec<OutputItemState>,
    next_output_index: usize,
    current_message_output_index: Option<usize>, // Tracks output_index of current message
    current_item_id: Option<String>,             // Tracks item_id of current item
    original_request: Option<ResponsesRequest>,
}

impl ResponseStreamEventEmitter {
    pub fn new(response_id: String, model: String, created_at: u64) -> Self {
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
            mcp_tool_labels: None,
            output_items: Vec::new(),
            next_output_index: 0,
            current_message_output_index: None,
            current_item_id: None,
            original_request: None,
        }
    }

    /// Set the original request for including all fields in response.completed
    pub fn set_original_request(&mut self, request: ResponsesRequest) {
        self.original_request = Some(request);
    }

    /// Set MCP tool labels for per-tool server labeling
    pub fn set_mcp_tool_labels(&mut self, tool_labels: HashMap<String, String>) {
        self.mcp_tool_labels = Some(tool_labels);
    }

    /// Update mcp_call output items with tool execution results
    ///
    /// After MCP tools are executed, this updates the stored output items
    /// to include the output field from the tool results.
    pub(crate) fn update_mcp_call_outputs(&mut self, tool_results: &[ToolResult]) {
        for tool_result in tool_results {
            // Find the output item with matching call_id
            for item_state in self.output_items.iter_mut() {
                if let Some(ref mut item_data) = item_state.item_data {
                    // Check if this is an mcp_call item with matching call_id
                    if item_data.get("type").and_then(|t| t.as_str()) == Some("mcp_call")
                        && item_data.get("call_id").and_then(|c| c.as_str())
                            == Some(&tool_result.call_id)
                    {
                        // Add output field
                        let output_str = serde_json::to_string(&tool_result.output)
                            .unwrap_or_else(|_| "{}".to_string());
                        item_data["output"] = json!(output_str);

                        // Update status based on success
                        if tool_result.is_error {
                            item_data["status"] = json!("failed");
                        }
                        break;
                    }
                }
            }
        }
    }

    fn next_sequence(&mut self) -> u64 {
        let seq = self.sequence_number;
        self.sequence_number += 1;
        seq
    }

    pub fn emit_created(&mut self) -> serde_json::Value {
        self.has_emitted_created = true;
        json!({
            "type": ResponseEvent::CREATED,
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

    pub fn emit_in_progress(&mut self) -> serde_json::Value {
        self.has_emitted_in_progress = true;
        json!({
            "type": ResponseEvent::IN_PROGRESS,
            "sequence_number": self.next_sequence(),
            "response": {
                "id": self.response_id,
                "object": "response",
                "status": "in_progress"
            }
        })
    }

    pub fn emit_content_part_added(
        &mut self,
        output_index: usize,
        item_id: &str,
        content_index: usize,
    ) -> serde_json::Value {
        self.has_emitted_content_part_added = true;
        json!({
            "type": ContentPartEvent::ADDED,
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

    pub fn emit_text_delta(
        &mut self,
        delta: &str,
        output_index: usize,
        item_id: &str,
        content_index: usize,
    ) -> serde_json::Value {
        self.accumulated_text.push_str(delta);
        json!({
            "type": OutputTextEvent::DELTA,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "content_index": content_index,
            "delta": delta
        })
    }

    pub fn emit_text_done(
        &mut self,
        output_index: usize,
        item_id: &str,
        content_index: usize,
    ) -> serde_json::Value {
        json!({
            "type": OutputTextEvent::DONE,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "content_index": content_index,
            "text": self.accumulated_text.clone()
        })
    }

    pub fn emit_content_part_done(
        &mut self,
        output_index: usize,
        item_id: &str,
        content_index: usize,
    ) -> serde_json::Value {
        json!({
            "type": ContentPartEvent::DONE,
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

    pub fn emit_completed(&mut self, usage: Option<&serde_json::Value>) -> serde_json::Value {
        // Build output array from tracked items
        let output: Vec<serde_json::Value> = self
            .output_items
            .iter()
            .filter_map(|item| {
                if item.status == ItemStatus::Completed {
                    item.item_data.clone()
                } else {
                    None
                }
            })
            .collect();

        // If no items were tracked (legacy path), fall back to generic message
        let output = if output.is_empty() {
            vec![json!({
                "id": self.message_id.clone(),
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": self.accumulated_text.clone()
                }]
            })]
        } else {
            output
        };

        // Build base response object
        let mut response_obj = json!({
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": "completed",
            "model": self.model,
            "output": output
        });

        // Add usage if provided
        if let Some(usage_val) = usage {
            response_obj["usage"] = usage_val.clone();
        }

        // Add all original request fields if available
        if let Some(ref req) = self.original_request {
            Self::add_optional_field(&mut response_obj, "instructions", &req.instructions);
            Self::add_optional_field(
                &mut response_obj,
                "max_output_tokens",
                &req.max_output_tokens,
            );
            Self::add_optional_field(&mut response_obj, "max_tool_calls", &req.max_tool_calls);
            Self::add_optional_field(
                &mut response_obj,
                "previous_response_id",
                &req.previous_response_id,
            );
            Self::add_optional_field(&mut response_obj, "reasoning", &req.reasoning);
            Self::add_optional_field(&mut response_obj, "temperature", &req.temperature);
            Self::add_optional_field(&mut response_obj, "top_p", &req.top_p);
            Self::add_optional_field(&mut response_obj, "truncation", &req.truncation);
            Self::add_optional_field(&mut response_obj, "user", &req.user);

            response_obj["parallel_tool_calls"] = json!(req.parallel_tool_calls.unwrap_or(true));
            response_obj["store"] = json!(req.store.unwrap_or(true));
            response_obj["tools"] = json!(req.tools.as_ref().unwrap_or(&vec![]));
            response_obj["metadata"] = json!(req.metadata.as_ref().unwrap_or(&Default::default()));

            // tool_choice: serialize if present, otherwise use "auto"
            if let Some(ref tc) = req.tool_choice {
                response_obj["tool_choice"] = json!(tc);
            } else {
                response_obj["tool_choice"] = json!("auto");
            }
        }

        json!({
            "type": ResponseEvent::COMPLETED,
            "sequence_number": self.next_sequence(),
            "response": response_obj
        })
    }

    /// Helper to add optional fields to JSON object
    fn add_optional_field<T: serde::Serialize>(
        obj: &mut serde_json::Value,
        key: &str,
        value: &Option<T>,
    ) {
        if let Some(val) = value {
            obj[key] = json!(val);
        }
    }

    // ========================================================================
    // MCP Event Emission Methods
    // ========================================================================

    pub fn emit_mcp_list_tools_in_progress(&mut self, output_index: usize) -> serde_json::Value {
        json!({
            "type": McpEvent::LIST_TOOLS_IN_PROGRESS,
            "sequence_number": self.next_sequence(),
            "output_index": output_index
        })
    }

    pub fn emit_mcp_list_tools_completed(
        &mut self,
        output_index: usize,
        tools: &[mcp::Tool],
    ) -> serde_json::Value {
        let tool_items: Vec<_> = tools
            .iter()
            .map(|t| {
                json!({
                    "name": &t.name,
                    "description": &t.description,
                    "input_schema": t.input_schema.clone()
                })
            })
            .collect();

        json!({
            "type": McpEvent::LIST_TOOLS_COMPLETED,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "tools": tool_items
        })
    }

    pub fn emit_mcp_call_in_progress(
        &mut self,
        output_index: usize,
        item_id: &str,
    ) -> serde_json::Value {
        json!({
            "type": McpEvent::CALL_IN_PROGRESS,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id
        })
    }

    pub fn emit_mcp_call_arguments_delta(
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
            "type": McpEvent::CALL_ARGUMENTS_DELTA,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "delta": delta
        })
    }

    pub fn emit_mcp_call_arguments_done(
        &mut self,
        output_index: usize,
        item_id: &str,
        arguments: &str,
    ) -> serde_json::Value {
        json!({
            "type": McpEvent::CALL_ARGUMENTS_DONE,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "arguments": arguments
        })
    }

    pub fn emit_mcp_call_completed(
        &mut self,
        output_index: usize,
        item_id: &str,
    ) -> serde_json::Value {
        json!({
            "type": McpEvent::CALL_COMPLETED,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id
        })
    }

    pub fn emit_mcp_call_failed(
        &mut self,
        output_index: usize,
        item_id: &str,
        error: &str,
    ) -> serde_json::Value {
        json!({
            "type": McpEvent::CALL_FAILED,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "error": error
        })
    }

    // ========================================================================
    // Function Call Event Emission Methods
    // ========================================================================

    pub fn emit_function_call_arguments_delta(
        &mut self,
        output_index: usize,
        item_id: &str,
        delta: &str,
    ) -> serde_json::Value {
        json!({
            "type": FunctionCallEvent::ARGUMENTS_DELTA,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "delta": delta
        })
    }

    pub fn emit_function_call_arguments_done(
        &mut self,
        output_index: usize,
        item_id: &str,
        arguments: &str,
    ) -> serde_json::Value {
        json!({
            "type": FunctionCallEvent::ARGUMENTS_DONE,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item_id": item_id,
            "arguments": arguments
        })
    }

    // ========================================================================
    // Output Item Wrapper Events
    // ========================================================================

    /// Emit response.output_item.added event
    pub fn emit_output_item_added(
        &mut self,
        output_index: usize,
        item: &serde_json::Value,
    ) -> serde_json::Value {
        json!({
            "type": OutputItemEvent::ADDED,
            "sequence_number": self.next_sequence(),
            "output_index": output_index,
            "item": item
        })
    }

    /// Emit response.output_item.done event
    pub fn emit_output_item_done(
        &mut self,
        output_index: usize,
        item: &serde_json::Value,
    ) -> serde_json::Value {
        // Store the item data for later use in emit_completed
        self.store_output_item_data(output_index, item.clone());

        json!({
            "type": OutputItemEvent::DONE,
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
    pub fn allocate_output_index(&mut self, item_type: OutputItemType) -> (usize, String) {
        let index = self.next_output_index;
        self.next_output_index += 1;

        let id_prefix = match &item_type {
            OutputItemType::McpListTools => "mcpl",
            OutputItemType::McpCall => "mcp",
            OutputItemType::FunctionCall => "fc",
            OutputItemType::Message => "msg",
            OutputItemType::Reasoning => "rs",
        };

        let id = Self::generate_item_id(id_prefix);

        self.output_items.push(OutputItemState {
            output_index: index,
            status: ItemStatus::InProgress,
            item_data: None,
        });

        (index, id)
    }

    /// Mark output item as completed and store its data
    pub fn complete_output_item(&mut self, output_index: usize) {
        if let Some(item) = self
            .output_items
            .iter_mut()
            .find(|i| i.output_index == output_index)
        {
            item.status = ItemStatus::Completed;
        }
    }

    /// Store output item data when emitting output_item.done
    pub fn store_output_item_data(&mut self, output_index: usize, item_data: serde_json::Value) {
        if let Some(item) = self
            .output_items
            .iter_mut()
            .find(|i| i.output_index == output_index)
        {
            item.item_data = Some(item_data);
        }
    }

    /// Finalize and return the complete ResponsesResponse
    ///
    /// This constructs the final ResponsesResponse from all accumulated output items
    /// for persistence. Should be called after streaming is complete.
    pub fn finalize(&self, usage: Option<Usage>) -> ResponsesResponse {
        // Build output array from tracked items
        let output: Vec<ResponseOutputItem> = self
            .output_items
            .iter()
            .filter_map(|item| {
                item.item_data
                    .as_ref()
                    .and_then(|data| serde_json::from_value(data.clone()).ok())
            })
            .collect();

        // Convert Usage to ResponsesUsage
        let responses_usage = usage.map(|u| {
            let usage_info = UsageInfo {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
                reasoning_tokens: u
                    .completion_tokens_details
                    .as_ref()
                    .and_then(|d| d.reasoning_tokens),
                prompt_tokens_details: None,
            };
            ResponsesUsage::Classic(usage_info)
        });

        // Build response using builder
        ResponsesResponse::builder(&self.response_id, &self.model)
            .created_at(self.created_at as i64)
            .status(ResponseStatus::Completed)
            .output(output)
            .maybe_copy_from_request(self.original_request.as_ref())
            .maybe_usage(responses_usage)
            .build()
    }

    /// Emit reasoning item wrapper events (added + done)
    ///
    /// Reasoning items in OpenAI format are simple placeholders emitted between tool iterations.
    /// They don't have streaming content - just wrapper events with empty/null content.
    pub fn emit_reasoning_item(
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
    pub fn process_chunk(
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

    pub fn send_event(
        &self,
        event: &serde_json::Value,
        tx: &mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
    ) -> Result<(), String> {
        let event_json = serde_json::to_string(event)
            .map_err(|e| format!("Failed to serialize event: {}", e))?;

        // Extract event type from the JSON for SSE event field
        let event_type = event
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("message");

        // Format as SSE with event: field
        let sse_message = format!("event: {}\ndata: {}\n\n", event_type, event_json);

        if tx.send(Ok(Bytes::from(sse_message))).is_err() {
            return Err("Client disconnected".to_string());
        }

        Ok(())
    }

    /// Send event and log any errors (typically client disconnect)
    ///
    /// This is a convenience method for streaming scenarios where client
    /// disconnection is expected and should be logged but not fail the operation.
    /// Returns true if sent successfully, false if client disconnected.
    pub fn send_event_best_effort(
        &self,
        event: &serde_json::Value,
        tx: &mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
    ) -> bool {
        match self.send_event(event, tx) {
            Ok(()) => true,
            Err(e) => {
                tracing::debug!("Failed to send event (likely client disconnect): {}", e);
                false
            }
        }
    }

    /// Emit an error event
    ///
    /// Creates and sends an error event with the given error message.
    /// Uses OpenAI's error event format.
    /// Use this for terminal errors that should abort the streaming response.
    pub fn emit_error(
        &mut self,
        error_msg: &str,
        error_code: Option<&str>,
        tx: &mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
    ) {
        let event = json!({
            "type": "error",
            "code": error_code.unwrap_or("internal_error"),
            "message": error_msg,
            "param": null,
            "sequence_number": self.next_sequence()
        });
        let sse_data = format!("data: {}\n\n", serde_json::to_string(&event).unwrap());
        let _ = tx.send(Ok(Bytes::from(sse_data)));
    }
}

/// Build a Server-Sent Events (SSE) response
///
/// Creates a Response with proper SSE headers and streaming body.
pub(crate) fn build_sse_response(
    rx: mpsc::UnboundedReceiver<Result<Bytes, std::io::Error>>,
) -> Response {
    let stream = UnboundedReceiverStream::new(rx);
    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .header("Connection", "keep-alive")
        .body(Body::from_stream(stream))
        .unwrap()
}
