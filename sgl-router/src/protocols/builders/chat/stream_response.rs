//! Builder for ChatCompletionStreamResponse
//!
//! Provides an ergonomic fluent API for constructing streaming chat completion responses.

use std::borrow::Cow;

use crate::protocols::{
    chat::*,
    common::{FunctionCallDelta, ToolCallDelta, Usage},
};

/// Builder for ChatCompletionStreamResponse
///
/// Provides a fluent interface for constructing streaming chat completion chunks with sensible defaults.
#[must_use = "Builder does nothing until .build() is called"]
#[derive(Clone, Debug)]
pub struct ChatCompletionStreamResponseBuilder {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ChatStreamChoice>,
    usage: Option<Usage>,
    system_fingerprint: Option<String>,
}

impl ChatCompletionStreamResponseBuilder {
    /// Create a new builder with required fields
    ///
    /// # Arguments
    /// - `id`: Completion ID (e.g., "chatcmpl_abc123")
    /// - `model`: Model name used for generation
    pub fn new(id: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            object: "chat.completion.chunk".to_string(),
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

    /// Set the object type (default: "chat.completion.chunk")
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
    pub fn choices(mut self, choices: Vec<ChatStreamChoice>) -> Self {
        self.choices = choices;
        self
    }

    /// Add a single choice (delta)
    pub fn add_choice(mut self, choice: ChatStreamChoice) -> Self {
        self.choices.push(choice);
        self
    }

    /// Set usage information (typically sent in final chunk)
    pub fn usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Set system fingerprint if provided (handles Option)
    pub fn maybe_system_fingerprint(mut self, fingerprint: Option<impl Into<String>>) -> Self {
        if let Some(fp) = fingerprint {
            self.system_fingerprint = Some(fp.into());
        }
        self
    }

    /// Set usage if provided (handles Option)
    pub fn maybe_usage(mut self, usage: Option<Usage>) -> Self {
        if let Some(u) = usage {
            self.usage = Some(u);
        }
        self
    }

    /// Add a choice delta that sets `role` and `content`
    pub fn add_choice_content(
        mut self,
        index: u32,
        role: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        self.choices.push(ChatStreamChoice {
            index,
            delta: ChatMessageDelta {
                role: Some(role.into()),
                content: Some(content.into()),
                tool_calls: None,
                reasoning_content: None,
            },
            logprobs: None,
            finish_reason: None,
            matched_stop: None,
        });
        self
    }

    /// Add a choice delta that sets `role`, `content`, and `logprobs`
    pub fn add_choice_content_with_logprobs(
        mut self,
        index: u32,
        role: impl Into<String>,
        content: impl Into<String>,
        logprobs: Option<crate::protocols::common::ChatLogProbs>,
    ) -> Self {
        self.choices.push(ChatStreamChoice {
            index,
            delta: ChatMessageDelta {
                role: Some(role.into()),
                content: Some(content.into()),
                tool_calls: None,
                reasoning_content: None,
            },
            logprobs,
            finish_reason: None,
            matched_stop: None,
        });
        self
    }

    /// Add a choice delta that only sets `role`
    pub fn add_choice_role(mut self, index: u32, role: impl Into<String>) -> Self {
        self.choices.push(ChatStreamChoice {
            index,
            delta: ChatMessageDelta {
                role: Some(role.into()),
                content: None,
                tool_calls: None,
                reasoning_content: None,
            },
            logprobs: None,
            finish_reason: None,
            matched_stop: None,
        });
        self
    }

    /// Add a choice delta that appends a tool-call *arguments delta*
    /// Uses `Cow` so you can pass `&str` or `String` without extra clones
    pub fn add_choice_tool_args(
        mut self,
        index: u32,
        args_delta: impl Into<Cow<'static, str>>,
    ) -> Self {
        self.choices.push(ChatStreamChoice {
            index,
            delta: ChatMessageDelta {
                role: Some("assistant".to_string()),
                content: None,
                tool_calls: Some(vec![ToolCallDelta {
                    index: 0,
                    id: None,
                    tool_type: None,
                    function: Some(FunctionCallDelta {
                        name: None,
                        arguments: Some(args_delta.into().into_owned()),
                    }),
                }]),
                reasoning_content: None,
            },
            logprobs: None,
            finish_reason: None,
            matched_stop: None,
        });
        self
    }

    /// Add a choice delta that sets reasoning content (for models that stream reasoning)
    pub fn add_choice_reasoning(mut self, index: u32, reasoning: impl Into<String>) -> Self {
        self.choices.push(ChatStreamChoice {
            index,
            delta: ChatMessageDelta {
                role: Some("assistant".to_string()),
                content: None,
                tool_calls: None,
                reasoning_content: Some(reasoning.into()),
            },
            logprobs: None,
            finish_reason: None,
            matched_stop: None,
        });
        self
    }

    /// Add a choice delta for tool call with function name and ID
    pub fn add_choice_tool_name(
        mut self,
        index: u32,
        tool_call_id: impl Into<String>,
        function_name: impl Into<String>,
    ) -> Self {
        self.choices.push(ChatStreamChoice {
            index,
            delta: ChatMessageDelta {
                role: Some("assistant".to_string()),
                content: None,
                tool_calls: Some(vec![ToolCallDelta {
                    index: 0,
                    id: Some(tool_call_id.into()),
                    tool_type: Some("function".to_string()),
                    function: Some(FunctionCallDelta {
                        name: Some(function_name.into()),
                        arguments: None,
                    }),
                }]),
                reasoning_content: None,
            },
            logprobs: None,
            finish_reason: None,
            matched_stop: None,
        });
        self
    }

    /// Add a choice delta with a pre-constructed ToolCallDelta
    /// Useful when you already have a ToolCallDelta object to emit
    pub fn add_choice_tool_call_delta(
        mut self,
        index: u32,
        tool_call_delta: ToolCallDelta,
    ) -> Self {
        self.choices.push(ChatStreamChoice {
            index,
            delta: ChatMessageDelta {
                role: Some("assistant".to_string()),
                content: None,
                tool_calls: Some(vec![tool_call_delta]),
                reasoning_content: None,
            },
            logprobs: None,
            finish_reason: None,
            matched_stop: None,
        });
        self
    }

    /// Add a choice with finish_reason (final chunk)
    /// This is used for the last chunk in a stream to signal completion
    pub fn add_choice_finish_reason(
        mut self,
        index: u32,
        finish_reason: impl Into<String>,
        matched_stop: Option<serde_json::Value>,
    ) -> Self {
        self.choices.push(ChatStreamChoice {
            index,
            delta: ChatMessageDelta {
                role: None,
                content: None,
                tool_calls: None,
                reasoning_content: None,
            },
            logprobs: None,
            finish_reason: Some(finish_reason.into()),
            matched_stop,
        });
        self
    }

    /// Build the ChatCompletionStreamResponse
    pub fn build(self) -> ChatCompletionStreamResponse {
        ChatCompletionStreamResponse {
            id: self.id,
            object: self.object,
            created: self.created,
            model: self.model,
            system_fingerprint: self.system_fingerprint,
            choices: self.choices,
            usage: self.usage,
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
        let chunk = ChatCompletionStreamResponseBuilder::new("chatcmpl_123", "gpt-4").build();

        assert_eq!(chunk.id, "chatcmpl_123");
        assert_eq!(chunk.model, "gpt-4");
        assert_eq!(chunk.object, "chat.completion.chunk");
        assert!(chunk.choices.is_empty());
        assert!(chunk.usage.is_none());
    }

    #[test]
    fn test_with_content_delta() {
        let chunk = ChatCompletionStreamResponseBuilder::new("chatcmpl_456", "gpt-4")
            .add_choice_content(0, "assistant", "Hello")
            .build();

        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(chunk.choices[0].index, 0);
        assert_eq!(chunk.choices[0].delta.content.as_ref().unwrap(), "Hello");
        assert_eq!(chunk.choices[0].delta.role.as_ref().unwrap(), "assistant");
        assert!(chunk.choices[0].finish_reason.is_none());
    }

    #[test]
    fn test_with_role_delta() {
        let chunk = ChatCompletionStreamResponseBuilder::new("chatcmpl_789", "gpt-4")
            .add_choice_role(0, "assistant")
            .build();

        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(chunk.choices[0].delta.role.as_ref().unwrap(), "assistant");
        assert!(chunk.choices[0].delta.content.is_none());
    }

    #[test]
    fn test_with_finish_reason() {
        let chunk = ChatCompletionStreamResponseBuilder::new("chatcmpl_101", "gpt-4")
            .add_choice_finish_reason(0, "stop", None)
            .build();

        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(chunk.choices[0].finish_reason.as_ref().unwrap(), "stop");
        assert!(chunk.choices[0].delta.content.is_none());
        assert!(chunk.choices[0].delta.role.is_none());
    }

    #[test]
    fn test_multiple_deltas() {
        let chunk = ChatCompletionStreamResponseBuilder::new("chatcmpl_202", "gpt-4")
            .add_choice_role(0, "assistant")
            .add_choice_content(0, "assistant", "Hello")
            .add_choice_content(0, "assistant", " world")
            .add_choice_finish_reason(0, "stop", None)
            .build();

        assert_eq!(chunk.choices.len(), 4); // role + 2 content + finish
    }

    #[test]
    fn test_with_usage() {
        let usage = Usage {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
            completion_tokens_details: None,
        };

        let chunk = ChatCompletionStreamResponseBuilder::new("chatcmpl_303", "gpt-4")
            .add_choice_finish_reason(0, "stop", None)
            .usage(usage)
            .build();

        assert!(chunk.usage.is_some());
        assert_eq!(chunk.usage.as_ref().unwrap().total_tokens, 30);
    }

    #[test]
    fn test_copy_from_request() {
        let request = ChatCompletionRequest {
            messages: vec![],
            model: "gpt-3.5-turbo".to_string(),
            ..Default::default()
        };

        let chunk = ChatCompletionStreamResponseBuilder::new("chatcmpl_404", "gpt-4")
            .copy_from_request(&request)
            .add_choice_content(0, "assistant", "test")
            .build();

        assert_eq!(chunk.model, "gpt-3.5-turbo"); // Copied from request
    }

    #[test]
    fn test_add_choice_explicit() {
        let choice = ChatStreamChoice {
            index: 0,
            delta: ChatMessageDelta {
                role: Some("assistant".to_string()),
                content: Some("Hello".to_string()),
                tool_calls: None,
                reasoning_content: None,
            },
            logprobs: None,
            finish_reason: None,
            matched_stop: None,
        };

        let chunk = ChatCompletionStreamResponseBuilder::new("chatcmpl_505", "gpt-4")
            .add_choice(choice)
            .build();

        assert_eq!(chunk.choices.len(), 1);
        assert_eq!(chunk.choices[0].delta.role.as_ref().unwrap(), "assistant");
        assert_eq!(chunk.choices[0].delta.content.as_ref().unwrap(), "Hello");
    }
}
