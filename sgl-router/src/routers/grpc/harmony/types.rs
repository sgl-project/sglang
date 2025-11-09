//! Shared types for Harmony pipeline

use openai_harmony::chat::Content;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::protocols::common::ToolCall;

/// Harmony message format
///
/// Represents messages in the Harmony encoding format with role and content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonyMessage {
    pub role: String,
    pub content: String,
}

impl HarmonyMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }

    /// Convert from openai_harmony::chat::Message to our simplified HarmonyMessage
    pub fn from_openai_harmony(msg: openai_harmony::chat::Message) -> Self {
        // Extract role as string
        let role = match msg.author.role {
            openai_harmony::chat::Role::User => "user",
            openai_harmony::chat::Role::Assistant => "assistant",
            openai_harmony::chat::Role::System => "system",
            openai_harmony::chat::Role::Developer => "developer",
            openai_harmony::chat::Role::Tool => "tool",
        }
        .to_string();

        // Extract text content from all Content::Text parts
        let content = msg
            .content
            .iter()
            .filter_map(|c| match c {
                Content::Text(tc) => Some(tc.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");

        Self { role, content }
    }
}

/// Output from Harmony encoding process
///
/// Contains the encoded input_ids, stop tokens, selection text for worker routing,
/// and the Harmony message history.
#[derive(Debug, Clone)]
pub struct HarmonyBuildOutput {
    /// Encoded token IDs to send to the model
    pub input_ids: Vec<u32>,

    /// Stop token IDs for this model (injected into sampling params)
    pub stop_token_ids: Vec<u32>,

    /// Selection text for worker routing (concise snippet from last user message)
    pub selection_text: String,

    /// Harmony messages for this conversation (used for history tracking)
    pub harmony_messages: Vec<HarmonyMessage>,
}

/// Parsed output from all three Harmony channels
///
/// Represents the complete response after parsing analysis, commentary, and final channels.
#[derive(Debug, Clone)]
pub struct HarmonyChannelOutput {
    /// Analysis/reasoning content (from analysis channel)
    pub analysis: Option<String>,

    /// Tool calls (from commentary channel)
    pub commentary: Option<Vec<ToolCall>>,

    /// Final text content (from final channel)
    pub final_text: String,

    /// Finish reason
    pub finish_reason: String,

    /// Matched stop token (if any)
    pub matched_stop: Option<Value>,
}

/// Streaming delta for SSE responses
///
/// Represents incremental updates as tokens are parsed from the stream.
#[derive(Debug, Clone)]
pub struct HarmonyChannelDelta {
    /// Delta for analysis/reasoning content
    pub analysis_delta: Option<String>,

    /// Delta for tool calls
    pub commentary_delta: Option<ToolCallDelta>,

    /// Delta for final text content
    pub final_delta: Option<String>,

    /// Whether this is the final delta
    pub is_final: bool,
}

/// Tool call delta for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDelta {
    pub index: usize,
    pub id: Option<String>,
    pub function: Option<FunctionDelta>,
}

/// Function call delta for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDelta {
    pub name: Option<String>,
    pub arguments: Option<String>,
}
