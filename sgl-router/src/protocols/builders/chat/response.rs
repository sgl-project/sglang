//! Builder for ChatCompletionResponse
//!
//! Provides an ergonomic fluent API for constructing chat completion responses.

use crate::protocols::{chat::*, common::Usage};

/// Builder for ChatCompletionResponse
///
/// Provides a fluent interface for constructing chat completion responses with sensible defaults.
#[must_use = "Builder does nothing until .build() is called"]
#[derive(Clone, Debug)]
pub struct ChatCompletionResponseBuilder {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Option<Usage>,
    system_fingerprint: Option<String>,
}

impl ChatCompletionResponseBuilder {
    /// Create a new builder with required fields
    ///
    /// # Arguments
    /// - `id`: Completion ID (e.g., "chatcmpl_abc123")
    /// - `model`: Model name used for generation
    pub fn new(id: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp() as u64,
            model: model.into(),
            choices: Vec::new(),
            usage: None,
            system_fingerprint: None,
        }
    }

    /// Copy common fields from a ChatCompletionRequest
    ///
    /// This populates the model field from the request.
    pub fn copy_from_request(mut self, request: &ChatCompletionRequest) -> Self {
        self.model = request.model.clone();
        self
    }

    /// Set the object type (default: "chat.completion")
    pub fn object(mut self, object: impl Into<String>) -> Self {
        self.object = object.into();
        self
    }

    /// Set the creation timestamp (default: current time)
    pub fn created(mut self, timestamp: u64) -> Self {
        self.created = timestamp;
        self
    }

    /// Set the choices
    pub fn choices(mut self, choices: Vec<ChatChoice>) -> Self {
        self.choices = choices;
        self
    }

    /// Add a single choice
    pub fn add_choice(mut self, choice: ChatChoice) -> Self {
        self.choices.push(choice);
        self
    }

    /// Set usage information
    pub fn usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Set usage if provided (handles Option)
    pub fn maybe_usage(mut self, usage: Option<Usage>) -> Self {
        if let Some(u) = usage {
            self.usage = Some(u);
        }
        self
    }

    /// Set system fingerprint if provided (handles Option)
    pub fn maybe_system_fingerprint(mut self, fingerprint: Option<impl Into<String>>) -> Self {
        if let Some(fp) = fingerprint {
            self.system_fingerprint = Some(fp.into());
        }
        self
    }

    /// Build the ChatCompletionResponse
    pub fn build(self) -> ChatCompletionResponse {
        ChatCompletionResponse {
            id: self.id,
            object: self.object,
            created: self.created,
            model: self.model,
            choices: self.choices,
            usage: self.usage,
            system_fingerprint: self.system_fingerprint,
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
        let response = ChatCompletionResponse::builder("chatcmpl_123", "gpt-4").build();

        assert_eq!(response.id, "chatcmpl_123");
        assert_eq!(response.model, "gpt-4");
        assert_eq!(response.object, "chat.completion");
        assert!(response.choices.is_empty());
        assert!(response.usage.is_none());
        assert!(response.system_fingerprint.is_none());
    }

    #[test]
    fn test_build_complete() {
        let choice = ChatChoice {
            index: 0,
            message: ChatCompletionMessage {
                role: "assistant".to_string(),
                content: Some("Hello!".to_string()),
                tool_calls: None,
                reasoning_content: None,
            },
            logprobs: None,
            finish_reason: Some("stop".to_string()),
            matched_stop: None,
            hidden_states: None,
        };

        let usage = Usage {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
            completion_tokens_details: None,
        };

        let response = ChatCompletionResponse::builder("chatcmpl_456", "gpt-4")
            .choices(vec![choice.clone()])
            .maybe_usage(Some(usage))
            .maybe_system_fingerprint(Some("fp_123abc"))
            .build();

        assert_eq!(response.id, "chatcmpl_456");
        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].index, 0);
        assert!(response.usage.is_some());
        assert_eq!(response.system_fingerprint.as_ref().unwrap(), "fp_123abc");
    }

    #[test]
    fn test_add_multiple_choices() {
        let choice1 = ChatChoice {
            index: 0,
            message: ChatCompletionMessage {
                role: "assistant".to_string(),
                content: Some("Option 1".to_string()),
                tool_calls: None,
                reasoning_content: None,
            },
            logprobs: None,
            finish_reason: Some("stop".to_string()),
            matched_stop: None,
            hidden_states: None,
        };

        let choice2 = ChatChoice {
            index: 1,
            message: ChatCompletionMessage {
                role: "assistant".to_string(),
                content: Some("Option 2".to_string()),
                tool_calls: None,
                reasoning_content: None,
            },
            logprobs: None,
            finish_reason: Some("stop".to_string()),
            matched_stop: None,
            hidden_states: None,
        };

        let response = ChatCompletionResponse::builder("chatcmpl_789", "gpt-4")
            .add_choice(choice1)
            .add_choice(choice2)
            .build();

        assert_eq!(response.choices.len(), 2);
        assert_eq!(response.choices[0].index, 0);
        assert_eq!(response.choices[1].index, 1);
    }

    #[test]
    fn test_copy_from_request() {
        let request = ChatCompletionRequest {
            messages: vec![],
            model: "gpt-3.5-turbo".to_string(),
            ..Default::default()
        };

        let response = ChatCompletionResponse::builder("chatcmpl_101", "gpt-4")
            .copy_from_request(&request)
            .build();

        assert_eq!(response.model, "gpt-3.5-turbo"); // Copied from request
    }
}
