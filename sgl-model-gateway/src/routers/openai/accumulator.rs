//! Streaming response accumulator for persisting responses.

use serde_json::Value;
use tracing::warn;

use super::streaming::{extract_output_index, get_event_type};
use crate::protocols::event_types::{OutputItemEvent, ResponseEvent};

// ============================================================================
// Streaming Response Accumulator
// ============================================================================

/// Helper that parses SSE frames from the OpenAI responses stream and
/// accumulates enough information to persist the final response locally.
pub(super) struct StreamingResponseAccumulator {
    /// The initial `response.created` payload (if emitted).
    initial_response: Option<Value>,
    /// The final `response.completed` payload (if emitted).
    completed_response: Option<Value>,
    /// Collected output items keyed by the upstream output index, used when
    /// a final response payload is absent and we need to synthesize one.
    output_items: Vec<(usize, Value)>,
    /// Captured error payload (if the upstream stream fails midway).
    encountered_error: Option<Value>,
}

impl StreamingResponseAccumulator {
    pub fn new() -> Self {
        Self {
            initial_response: None,
            completed_response: None,
            output_items: Vec::new(),
            encountered_error: None,
        }
    }

    /// Feed the accumulator with the next SSE chunk.
    pub fn ingest_block(&mut self, block: &str) {
        if block.trim().is_empty() {
            return;
        }
        self.process_block(block);
    }

    /// Consume the accumulator and produce the best-effort final response value.
    pub fn into_final_response(mut self) -> Option<Value> {
        if self.completed_response.is_some() {
            return self.completed_response;
        }

        self.build_fallback_response()
    }

    pub fn encountered_error(&self) -> Option<&Value> {
        self.encountered_error.as_ref()
    }

    pub fn original_response_id(&self) -> Option<&str> {
        self.initial_response
            .as_ref()
            .and_then(|response| response.get("id"))
            .and_then(|id| id.as_str())
    }

    pub fn snapshot_final_response(&self) -> Option<Value> {
        if let Some(resp) = &self.completed_response {
            return Some(resp.clone());
        }
        self.build_fallback_response_snapshot()
    }

    fn build_fallback_response_snapshot(&self) -> Option<Value> {
        let mut response = self.initial_response.clone()?;

        if let Some(obj) = response.as_object_mut() {
            obj.insert("status".to_string(), Value::String("completed".to_string()));

            let mut output_items = self.output_items.clone();
            output_items.sort_by_key(|(index, _)| *index);
            let outputs: Vec<Value> = output_items.into_iter().map(|(_, item)| item).collect();
            obj.insert("output".to_string(), Value::Array(outputs));
        }

        Some(response)
    }

    fn process_block(&mut self, block: &str) {
        let trimmed = block.trim();
        if trimmed.is_empty() {
            return;
        }

        let mut event_name: Option<String> = None;
        let mut data_lines: Vec<String> = Vec::new();

        for line in trimmed.lines() {
            if let Some(rest) = line.strip_prefix("event:") {
                event_name = Some(rest.trim().to_string());
            } else if let Some(rest) = line.strip_prefix("data:") {
                data_lines.push(rest.trim_start().to_string());
            }
        }

        let data_payload = data_lines.join("\n");
        if data_payload.is_empty() {
            return;
        }

        self.handle_event(event_name.as_deref(), &data_payload);
    }

    fn handle_event(&mut self, event_name: Option<&str>, data_payload: &str) {
        let parsed: Value = match serde_json::from_str(data_payload) {
            Ok(value) => value,
            Err(err) => {
                warn!("Failed to parse streaming event JSON: {}", err);
                return;
            }
        };

        match get_event_type(event_name, &parsed) {
            ResponseEvent::CREATED => {
                if self.initial_response.is_none() {
                    if let Some(response) = parsed.get("response") {
                        self.initial_response = Some(response.clone());
                    }
                }
            }
            ResponseEvent::COMPLETED => {
                if let Some(response) = parsed.get("response") {
                    self.completed_response = Some(response.clone());
                }
            }
            OutputItemEvent::DONE => {
                if let (Some(index), Some(item)) =
                    (extract_output_index(&parsed), parsed.get("item"))
                {
                    self.output_items.push((index, item.clone()));
                }
            }
            "response.error" => {
                self.encountered_error = Some(parsed);
            }
            _ => {}
        }
    }

    fn build_fallback_response(&mut self) -> Option<Value> {
        let mut response = self.initial_response.clone()?;

        if let Some(obj) = response.as_object_mut() {
            obj.insert("status".to_string(), Value::String("completed".to_string()));

            self.output_items.sort_by_key(|(index, _)| *index);
            let outputs: Vec<Value> = std::mem::take(&mut self.output_items)
                .into_iter()
                .map(|(_, item)| item)
                .collect();
            obj.insert("output".to_string(), Value::Array(outputs));
        }

        Some(response)
    }
}
