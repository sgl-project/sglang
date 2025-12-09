//! Builder for ResponsesResponse
//!
//! Provides an ergonomic fluent API for constructing ResponsesResponse instances.

use std::collections::HashMap;

use serde_json::Value;

use crate::protocols::responses::*;

/// Builder for ResponsesResponse
///
/// Provides a fluent interface for constructing responses with sensible defaults.
#[must_use = "Builder does nothing until .build() is called"]
#[derive(Clone, Debug)]
pub struct ResponsesResponseBuilder {
    id: String,
    object: String,
    created_at: i64,
    status: ResponseStatus,
    error: Option<Value>,
    incomplete_details: Option<Value>,
    instructions: Option<String>,
    max_output_tokens: Option<u32>,
    model: String,
    output: Vec<ResponseOutputItem>,
    parallel_tool_calls: bool,
    previous_response_id: Option<String>,
    reasoning: Option<ReasoningInfo>,
    store: bool,
    temperature: Option<f32>,
    text: Option<TextConfig>,
    tool_choice: String,
    tools: Vec<ResponseTool>,
    top_p: Option<f32>,
    truncation: Option<String>,
    usage: Option<ResponsesUsage>,
    user: Option<String>,
    safety_identifier: Option<String>,
    metadata: HashMap<String, Value>,
}

impl ResponsesResponseBuilder {
    /// Create a new builder with required fields
    ///
    /// # Arguments
    /// - `id`: Response ID (e.g., "resp_abc123")
    /// - `model`: Model name used for generation
    pub fn new(id: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            object: "response".to_string(),
            created_at: chrono::Utc::now().timestamp(),
            status: ResponseStatus::InProgress,
            error: None,
            incomplete_details: None,
            instructions: None,
            max_output_tokens: None,
            model: model.into(),
            output: Vec::new(),
            parallel_tool_calls: true,
            previous_response_id: None,
            reasoning: None,
            store: true,
            temperature: None,
            text: None,
            tool_choice: "auto".to_string(),
            tools: Vec::new(),
            top_p: None,
            truncation: None,
            usage: None,
            user: None,
            safety_identifier: None,
            metadata: HashMap::new(),
        }
    }

    /// Copy common fields from a ResponsesRequest
    ///
    /// This populates fields like instructions, max_output_tokens, temperature, etc.
    /// from the original request, making it easy to construct a response that mirrors
    /// the request parameters.
    ///
    /// Note: `safety_identifier` is intentionally NOT copied as it is for content moderation
    /// and should be set independently from the request's `user` field (which is for billing/tracking).
    pub fn copy_from_request(mut self, request: &ResponsesRequest) -> Self {
        self.instructions = request.instructions.clone();
        self.max_output_tokens = request.max_output_tokens;
        self.parallel_tool_calls = request.parallel_tool_calls.unwrap_or(true);
        self.previous_response_id = request.previous_response_id.clone();
        self.store = request.store.unwrap_or(true);
        self.temperature = request.temperature;
        self.tool_choice = if let Some(ref tc) = request.tool_choice {
            serde_json::to_string(tc).unwrap_or_else(|_| "auto".to_string())
        } else {
            "auto".to_string()
        };
        self.tools = request.tools.clone().unwrap_or_default();
        self.top_p = request.top_p;
        self.user = request.user.clone();
        self.metadata = request.metadata.clone().unwrap_or_default();
        self
    }

    /// Set the object type (default: "response")
    pub fn object(mut self, object: impl Into<String>) -> Self {
        self.object = object.into();
        self
    }

    /// Set the creation timestamp (default: current time)
    pub fn created_at(mut self, timestamp: i64) -> Self {
        self.created_at = timestamp;
        self
    }

    /// Set the response status
    pub fn status(mut self, status: ResponseStatus) -> Self {
        self.status = status;
        self
    }

    /// Set error information (if status is failed)
    pub fn error(mut self, error: Value) -> Self {
        self.error = Some(error);
        self
    }

    /// Set incomplete details (if response was truncated)
    pub fn incomplete_details(mut self, details: Value) -> Self {
        self.incomplete_details = Some(details);
        self
    }

    /// Set system instructions
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Set max output tokens
    pub fn max_output_tokens(mut self, tokens: u32) -> Self {
        self.max_output_tokens = Some(tokens);
        self
    }

    /// Set output items
    pub fn output(mut self, output: Vec<ResponseOutputItem>) -> Self {
        self.output = output;
        self
    }

    /// Add a single output item
    pub fn add_output(mut self, item: ResponseOutputItem) -> Self {
        self.output.push(item);
        self
    }

    /// Set whether parallel tool calls are enabled
    pub fn parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.parallel_tool_calls = enabled;
        self
    }

    /// Set previous response ID (if continuation)
    pub fn previous_response_id(mut self, id: impl Into<String>) -> Self {
        self.previous_response_id = Some(id.into());
        self
    }

    /// Set reasoning information
    pub fn reasoning(mut self, reasoning: ReasoningInfo) -> Self {
        self.reasoning = Some(reasoning);
        self
    }

    /// Set whether the response is stored
    pub fn store(mut self, store: bool) -> Self {
        self.store = store;
        self
    }

    /// Set temperature setting
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set text format settings if provided (handles Option)
    pub fn maybe_text(mut self, text: Option<TextConfig>) -> Self {
        if let Some(t) = text {
            self.text = Some(t);
        }
        self
    }

    /// Set tool choice setting
    pub fn tool_choice(mut self, tool_choice: impl Into<String>) -> Self {
        self.tool_choice = tool_choice.into();
        self
    }

    /// Set available tools
    pub fn tools(mut self, tools: Vec<ResponseTool>) -> Self {
        self.tools = tools;
        self
    }

    /// Set top-p setting
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set truncation strategy
    pub fn truncation(mut self, truncation: impl Into<String>) -> Self {
        self.truncation = Some(truncation.into());
        self
    }

    /// Set usage information
    pub fn usage(mut self, usage: ResponsesUsage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Set usage if provided (handles Option)
    pub fn maybe_usage(mut self, usage: Option<ResponsesUsage>) -> Self {
        if let Some(u) = usage {
            self.usage = Some(u);
        }
        self
    }

    /// Copy from request if provided (handles Option)
    pub fn maybe_copy_from_request(mut self, request: Option<&ResponsesRequest>) -> Self {
        if let Some(req) = request {
            self = self.copy_from_request(req);
        }
        self
    }

    /// Set user identifier
    pub fn user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    /// Set safety identifier
    pub fn safety_identifier(mut self, identifier: impl Into<String>) -> Self {
        self.safety_identifier = Some(identifier.into());
        self
    }

    /// Set metadata
    pub fn metadata(mut self, metadata: HashMap<String, Value>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Add a single metadata entry
    pub fn add_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Build the ResponsesResponse
    pub fn build(self) -> ResponsesResponse {
        ResponsesResponse {
            id: self.id,
            object: self.object,
            created_at: self.created_at,
            status: self.status,
            error: self.error,
            incomplete_details: self.incomplete_details,
            instructions: self.instructions,
            max_output_tokens: self.max_output_tokens,
            model: self.model,
            output: self.output,
            parallel_tool_calls: self.parallel_tool_calls,
            previous_response_id: self.previous_response_id,
            reasoning: self.reasoning,
            store: self.store,
            temperature: self.temperature,
            text: self.text,
            tool_choice: self.tool_choice,
            tools: self.tools,
            top_p: self.top_p,
            truncation: self.truncation,
            usage: self.usage,
            user: self.user,
            safety_identifier: self.safety_identifier,
            metadata: self.metadata,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_minimal() {
        let response = ResponsesResponse::builder("resp_123", "gpt-4").build();

        assert_eq!(response.id, "resp_123");
        assert_eq!(response.model, "gpt-4");
        assert_eq!(response.object, "response");
        assert_eq!(response.status, ResponseStatus::InProgress);
        assert!(response.output.is_empty());
        assert!(response.parallel_tool_calls);
        assert!(response.store);
    }

    #[test]
    fn test_build_complete() {
        let response = ResponsesResponse::builder("resp_123", "gpt-4")
            .status(ResponseStatus::Completed)
            .instructions("You are a helpful assistant")
            .max_output_tokens(1000)
            .temperature(0.7)
            .top_p(0.9)
            .parallel_tool_calls(false)
            .store(false)
            .build();

        assert_eq!(response.status, ResponseStatus::Completed);
        assert_eq!(
            response.instructions.as_ref().unwrap(),
            "You are a helpful assistant"
        );
        assert_eq!(response.max_output_tokens, Some(1000));
        assert_eq!(response.temperature, Some(0.7));
        assert_eq!(response.top_p, Some(0.9));
        assert!(!response.parallel_tool_calls);
        assert!(!response.store);
    }

    #[test]
    fn test_copy_from_request() {
        let request = ResponsesRequest {
            model: "gpt-4".to_string(),
            input: ResponseInput::Text("test".to_string()),
            instructions: Some("Be helpful".to_string()),
            max_output_tokens: Some(500),
            temperature: Some(0.8),
            top_p: Some(0.95),
            parallel_tool_calls: Some(false),
            store: Some(false),
            user: Some("user_123".to_string()),
            metadata: Some(HashMap::from([(
                "key".to_string(),
                serde_json::json!("value"),
            )])),
            ..Default::default()
        };

        let response = ResponsesResponse::builder("resp_456", "gpt-4")
            .copy_from_request(&request)
            .status(ResponseStatus::Completed)
            .build();

        assert_eq!(response.instructions.as_ref().unwrap(), "Be helpful");
        assert_eq!(response.max_output_tokens, Some(500));
        assert_eq!(response.temperature, Some(0.8));
        assert_eq!(response.top_p, Some(0.95));
        assert!(!response.parallel_tool_calls);
        assert!(!response.store);
        assert_eq!(response.user.as_ref().unwrap(), "user_123");
        assert_eq!(
            response.metadata.get("key").unwrap(),
            &serde_json::json!("value")
        );
    }

    #[test]
    fn test_add_output_items() {
        let response = ResponsesResponse::builder("resp_789", "gpt-4")
            .add_output(ResponseOutputItem::Message {
                id: "msg_1".to_string(),
                role: "assistant".to_string(),
                content: vec![],
                status: "completed".to_string(),
            })
            .add_output(ResponseOutputItem::Message {
                id: "msg_2".to_string(),
                role: "assistant".to_string(),
                content: vec![],
                status: "completed".to_string(),
            })
            .build();

        assert_eq!(response.output.len(), 2);
    }

    #[test]
    fn test_add_metadata() {
        let response = ResponsesResponse::builder("resp_101", "gpt-4")
            .add_metadata("key1", serde_json::json!("value1"))
            .add_metadata("key2", serde_json::json!(42))
            .build();

        assert_eq!(response.metadata.len(), 2);
        assert_eq!(response.metadata.get("key1").unwrap(), "value1");
        assert_eq!(response.metadata.get("key2").unwrap(), 42);
    }
}
