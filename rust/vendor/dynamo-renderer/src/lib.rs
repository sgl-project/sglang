// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Modified by SGLang: exposes the renderer's existing content-array classification through
// OAIPromptFormatter and PromptFormatter; other changes are comment-only provenance or spelling metadata.

//! Prompt Formatting
//!
//! Standalone, runtime-free chat-template / prompt formatting for
//! OpenAI-compatible inference frontends. Renders HuggingFace `chat_template`
//! jinja2 (via `minijinja` + `minijinja-contrib` pycompat), handles tool
//! usage formatting and generation-prompt handling.
//!
//! Consumers implement [`OAIChatLikeRequest`] for their request type (or use
//! the ready-made impl for `dynamo-protocols`' OpenAI chat request) and render
//! with a [`PromptFormatter`] built from a HuggingFace `tokenizer_config.json`
//! ([`ChatTemplate`]).
//!
//! This crate is a *bridge* between OpenAI request types ([`dynamo_protocols`])
//! and prompt rendering. Most formatters return text; segment-sensitive native
//! formats can preserve tokenizer policy through [`RenderedPrompt`].

// TODO:
// 1. Query if `add_generation_prompt` is present in the prompt template
// 2. Support for models with add_generation_prompt:
//    - PALS (Prefix-Assisted Language Sampling)
//    - Continuation - Detected on user turns, where we can return
//      partial assistant responses without add_generation_prompt

use anyhow::Result;
use minijinja::value::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// Re-export of `dynamo-tokenizers` as a one-import convenience: consumers that
/// want both tokenization and chat templating can reach the tokenizer types via
/// `dynamo_renderer::dynamo_tokenizers::*` without adding a second dependency.
pub use dynamo_tokenizers;

pub mod deepseek;
pub mod inkling;
pub mod kimi_k3;
mod template;

pub use template::{
    ChatTemplate, ChatTemplateValue, ContextMixins, deepseek_formatter_for, kimi_k3_formatter_for,
    may_be_fix_tool_schema, native_formatter_for,
};

/// Selects which context-mixin behaviors a template renders with.
///
/// Carried on the model deployment card (`prompt_context`) and consumed by the
/// chat-template renderer via [`ContextMixins`].
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum PromptContextMixin {
    /// Support OAI Chat Messages and Tools
    OaiChat,

    /// Enables templates with `{{datetime}}` to be rendered with the current date and time.
    Llama3DateTime,
}

/// Shared helper: extract a boolean thinking toggle from `chat_template_args`.
///
/// Reads the two equivalent keys (`thinking`, `enable_thinking` — vLLM's
/// canonical kwarg) in order and returns the first bool value found, or `None`
/// if neither key is present (or neither carries a bool). Used by the V4
/// formatter's `resolve_thinking_mode` and by reasoning-parser gating in
/// consumers so both paths agree on the signal interpretation.
pub fn thinking_bool_from_args(args: Option<&HashMap<String, serde_json::Value>>) -> Option<bool> {
    let args = args?;
    for key in ["thinking", "enable_thinking"] {
        if let Some(v) = args.get(key).and_then(|x| x.as_bool()) {
            return Some(v);
        }
    }
    None
}

#[derive(Debug)]
pub enum TokenInput {
    Single(Vec<u32>),
    Batch(Vec<Vec<u32>>),
}

#[derive(Debug)]
pub enum TextInput {
    Single(String),
    Batch(Vec<String>),
}

#[derive(Debug)]
pub enum PromptInput {
    Tokens(TokenInput),
    Text(TextInput),
}

/// One owned prompt segment with an explicit special-token trust boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedSegment {
    pub text: String,
    pub allow_special: bool,
}

impl RenderedSegment {
    pub fn new(text: impl Into<String>, allow_special: bool) -> Self {
        Self {
            text: text.into(),
            allow_special,
        }
    }

    pub fn as_encode_segment(&self) -> dynamo_tokenizers::EncodeSegment<'_> {
        dynamo_tokenizers::EncodeSegment::new(&self.text, self.allow_special)
    }
}

/// A rendered prompt plus its optional tokenization boundaries.
///
/// The prompt owns its segment text while `dynamo-tokenizers` borrows that text
/// during encoding. Keeping the types separate preserves the tokenizer crate's
/// published zero-copy `EncodeSegment<'_>` API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedPrompt {
    text: String,
    segments: Option<Vec<RenderedSegment>>,
}

impl RenderedPrompt {
    pub fn text(text: String) -> Self {
        Self {
            text,
            segments: None,
        }
    }

    pub fn segmented(segments: Vec<RenderedSegment>) -> Self {
        let text = segments
            .iter()
            .map(|segment| segment.text.as_str())
            .collect();
        Self {
            text,
            segments: Some(segments),
        }
    }

    pub fn as_str(&self) -> &str {
        &self.text
    }

    pub fn segments(&self) -> Option<&[RenderedSegment]> {
        self.segments.as_deref()
    }

    pub fn encode_segments(&self) -> Option<Vec<dynamo_tokenizers::EncodeSegment<'_>>> {
        Some(
            self.segments()?
                .iter()
                .map(RenderedSegment::as_encode_segment)
                .collect(),
        )
    }

    pub fn into_text(self) -> String {
        self.text
    }
}

/// A prompt-rendering failure caused by the request rather than server state.
///
/// Callers can downcast an [`anyhow::Error`] to this type and map it to their
/// protocol's invalid-request status without treating every template failure as
/// a client error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptRenderError {
    InvalidRequest(String),
}

impl PromptRenderError {
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::InvalidRequest(message.into())
    }
}

impl std::fmt::Display for PromptRenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRequest(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for PromptRenderError {}

/// Trait that defines a request that can map to an OpenAI-like request.
///
/// Implement this for your request type to render it through a
/// [`PromptFormatter`]. Media/multimodal IO config is intentionally *not* part
/// of this trait — it is a preprocessing concern owned by the consumer, kept
/// off the rendering surface so this crate stays runtime-free.
pub trait OAIChatLikeRequest {
    fn model(&self) -> String;
    fn messages(&self) -> Value;
    fn typed_messages(&self) -> Option<&[dynamo_protocols::types::ChatCompletionRequestMessage]> {
        None
    }
    fn tools(&self) -> Option<Value> {
        None
    }
    fn tool_choice(&self) -> Option<Value> {
        None
    }
    fn response_format(&self) -> Option<Value> {
        None
    }

    /// OpenAI-compatible reasoning-effort control, when the request type
    /// exposes it as a top-level field.
    fn reasoning_effort(&self) -> Option<Value> {
        None
    }

    fn should_add_generation_prompt(&self) -> bool;

    /// Optional additional args to merge into the chat template context
    fn chat_template_args(&self) -> Option<&HashMap<String, serde_json::Value>> {
        None
    }

    /// Returns the type of input for the prompt. Default is Text.
    fn prompt_input_type(&self) -> PromptInput {
        PromptInput::Text(TextInput::Single(String::new()))
    }

    /// Extract tokens if the input is pre-tokenized
    fn extract_tokens(&self) -> Option<TokenInput> {
        None
    }

    fn extract_text(&self) -> Option<TextInput> {
        None
    }

    fn mm_processor_kwargs(&self) -> Option<&serde_json::Value> {
        None
    }
}

pub trait OAIPromptFormatter: Send + Sync + 'static {
    fn supports_add_generation_prompt(&self) -> bool;

    /// Whether this formatter requires OpenAI message content arrays.
    ///
    /// The concrete Jinja formatter determines this once at load time. Other
    /// formatters preserve the historical string-content default.
    fn requires_content_arrays(&self) -> bool {
        false
    }

    fn render(&self, req: &dyn OAIChatLikeRequest) -> Result<String>;

    fn render_prompt(&self, req: &dyn OAIChatLikeRequest) -> Result<RenderedPrompt> {
        self.render(req).map(RenderedPrompt::text)
    }
}

#[derive(Clone)]
pub enum PromptFormatter {
    OAI(Arc<dyn OAIPromptFormatter>),
}

// No-op formatter: used for models without chat_template
#[derive(Debug, Default)]
pub struct NoOpFormatter;

impl OAIPromptFormatter for NoOpFormatter {
    fn supports_add_generation_prompt(&self) -> bool {
        false
    }

    fn render(&self, req: &dyn OAIChatLikeRequest) -> Result<String> {
        let messages = req.messages();

        let first_message = messages
            .get_item_by_index(0)
            .map_err(|_| anyhow::Error::msg("No message at index 0 or messages array is empty"))?;

        let content = first_message
            .get_attr("content")
            .map_err(|_| anyhow::Error::msg("First message has no 'content' field"))?;

        let content_str = content
            .as_str()
            .ok_or_else(|| anyhow::Error::msg("Message content is not a string"))?
            .to_string();
        Ok(content_str)
    }
}

impl PromptFormatter {
    pub fn no_op() -> Self {
        Self::OAI(Arc::new(NoOpFormatter))
    }

    /// Expose the formatter's load-time OpenAI content-shape requirement.
    pub fn requires_content_arrays(&self) -> bool {
        match self {
            Self::OAI(formatter) => formatter.requires_content_arrays(),
        }
    }
}

#[cfg(test)]
mod rendered_prompt_tests {
    use super::{RenderedPrompt, RenderedSegment};

    #[test]
    fn owned_segments_borrow_into_tokenizer_segments() {
        let prompt = RenderedPrompt::segmented(vec![
            RenderedSegment::new("<|open|>", true),
            RenderedSegment::new("user text", false),
        ]);

        let segments = prompt.encode_segments().expect("segmented prompt");
        assert_eq!(segments[0].text, "<|open|>");
        assert!(segments[0].allow_special);
        assert_eq!(segments[1].text, "user text");
        assert!(!segments[1].allow_special);
        assert_eq!(prompt.as_str(), "<|open|>user text");
    }
}
