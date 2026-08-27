//! Reusable request preprocessing for SGLang.
//!
//! The core renders normalized chat requests, lowers textual completions,
//! tokenizes prompts, and produces the token-in contract consumed by SGLang.
//! The optional `http` feature owns OpenAI wire lowering and a standalone
//! frontend backed by SGLang's `/generate` endpoint.

mod chat;
mod config;
mod error;
#[cfg(feature = "http")]
mod generation;
#[cfg(feature = "http")]
mod http;
mod output;
#[cfg(feature = "http")]
mod protocol;
mod request;
mod sampling;
mod service;
mod template;
mod tokenizer;
mod types;

mod regex;

pub(crate) use chat::ChatPreprocessor;
pub use chat::{ChatRequest, LoweredChat};
pub use config::{RendererConfig, RendererLimits, SamplingDefaults};
pub use error::{RendererError, RendererErrorKind};
#[cfg(feature = "http")]
pub(crate) use generation::{
    FrontendError, GenerationEvent, GenerationFinishReason, GenerationOutput,
    GenerationOutputExtras, GenerationStream, MatchedStop,
};
#[cfg(feature = "http")]
pub use http::{RendererRuntimeConfig, serve};
pub use output::{
    ChatEvent, ChatFinishReason, ChatResponseError, ChatResponseInput, ChatResponseItem,
    ChatResponseProcessor, ChatToolCall, ChatToolCallDelta, DecodedChatEvent, ParsedChatChoice,
};
pub use request::{
    CompletionRequest, GenerateRequest, GenerateRequestMetadata, GenerateSamplingParams,
    GenerationOptions, TextRequest, TokenIdsRequest,
};
pub use sampling::SamplingParams;
#[cfg(feature = "http")]
pub(crate) use sampling::SamplingParamsOverrides;
pub use service::{PreparedChat, RendererService, TokenizationBackend};
pub(crate) use template::ChatFormatter;
pub use tokenizer::{DynamoTokenizer, NoTokenizer, PooledTokenizer, TextTokenizer, load_tokenizer};
pub use types::{OneOrMany, OneOrManyItem, TokenIds};
