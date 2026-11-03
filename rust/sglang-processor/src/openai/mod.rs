//! OpenAI endpoints served through SGLang's native `/generate`, matching the
//! Python server's OpenAI layer (`python/sglang/srt/entrypoints/openai/`).
//!
//! A host lowers the OpenAI body into a typed `/generate` request, sends it, and feeds
//! the engine's response back to the returned responder. Requests whose
//! behavior is not reproduced here come back as [`Unsupported`], and the host
//! sends them to the engine's own OpenAI route.

mod chat;
mod completions;
mod kimi_k3_tag;
mod pieces;
mod request;
mod wire;

use serde::Deserialize;
use sglang_api_types::api::v1::{GenerateResponse, GenerateStreamError};

pub use chat::{ChatModel, ChatResponder, DeepSeekOutputError, ReasoningToggle, lower_chat};
pub use completions::{CompletionResponder, lower_completion};
pub use pieces::TokenPieces;
pub use sglang_api_types::api::v1::GenerateRequest;
pub use wire::Reply;

/// Builds the OpenAI response of a lowered request from the engine's `/generate` output.
pub enum Responder {
    Completion(Box<CompletionResponder>),
    Chat(Box<ChatResponder>),
}

impl Responder {
    /// The buffered `/generate` response; `None` passes the engine's reply through.
    pub fn unary(
        self,
        output: Vec<Result<GenerateResponse, GenerateStreamError>>,
    ) -> Option<Reply> {
        let output = match output.into_iter().collect::<Result<Vec<_>, _>>() {
            Ok(output) => output,
            Err(error) => return Some(Self::rejected(error)),
        };
        match self {
            Self::Completion(responder) => responder.unary(output),
            Self::Chat(responder) => responder.unary(output),
        }
    }

    /// One decoded `/generate` stream item. `Err` replaces the whole stream,
    /// before anything was sent.
    pub fn stream_data(
        &mut self,
        data: Result<GenerateResponse, GenerateStreamError>,
    ) -> Result<Vec<String>, Reply> {
        match self {
            Self::Completion(responder) => responder.stream_data(data),
            Self::Chat(responder) => responder.stream_data(data),
        }
    }

    /// Finish the OpenAI stream after the engine's `[DONE]` marker.
    pub fn stream_end(&mut self) -> Vec<String> {
        match self {
            Self::Completion(responder) => responder.stream_end(),
            Self::Chat(responder) => responder.stream_end(),
        }
    }

    /// Whether the OpenAI stream has ended, so the rest of `/generate` can be dropped.
    pub fn done(&self) -> bool {
        match self {
            Self::Completion(responder) => responder.done(),
            Self::Chat(responder) => responder.done(),
        }
    }

    /// A `/generate` response with an error status.
    pub fn rejected(error: GenerateStreamError) -> Reply {
        wire::engine_error_reply(error)
    }

    /// The engine returned an output missing a field required by OpenAI.
    pub fn malformed_output(missing: &str) -> Reply {
        wire::malformed_output_reply(missing)
    }
}

/// Engine server args that change its OpenAI layer, as `/server_info` reports them.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(default)]
pub struct OpenAiSettings {
    pub allow_auto_truncate: bool,
    pub completion_template: Option<String>,
    /// `--context-length`; the model's own when unset.
    pub context_length: Option<u64>,
    pub default_chat_template_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
    pub enable_cache_report: bool,
    pub incremental_streaming_output: bool,
    pub reasoning_parser: Option<String>,
    pub return_input_ids: bool,
    pub return_output_ids: bool,
    pub sampling_defaults: Option<String>,
    /// `--served-model-name`, which a `model:adapter` request may contain.
    pub served_model_name: Option<String>,
    pub stream_response_default_include_usage: bool,
    pub tokenizer_metrics_custom_labels_header: Option<String>,
    pub tokenizer_metrics_allowed_custom_labels: Option<Vec<String>>,
    pub tool_call_parser: Option<String>,
}

/// The request headers Python's OpenAI layer reads.
#[derive(Debug, Clone, Copy, Default)]
pub struct OpenAiHeaders<'a> {
    pub routing_key: Option<&'a str>,
    /// The value of `settings.tokenizer_metrics_custom_labels_header`.
    pub custom_labels: Option<&'a str>,
    /// `x-sglext-return-input-ids` or `x-sglext-return-output-ids` is `1`.
    pub sglext_ids: bool,
}

/// The tokenizer queries Python's OpenAI layer makes.
pub trait OpenAiTokenizer: Send + Sync {
    /// `tokenizer.decode(ids, skip_special_tokens=True)`.
    fn decode(&self, ids: &[u32]) -> Option<String>;
    /// The bytes of `tokenizer.convert_ids_to_tokens(id)` for a byte-level
    /// BPE tokenizer, else `None`.
    fn byte_level_bytes(&self, id: u32) -> Option<Vec<u8>>;
}

/// Why a request is left to the engine's own OpenAI route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Unsupported(pub &'static str);
