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
pub use chat::ChatRequest;
pub(crate) use chat::LoweredChat;
pub use config::{RendererConfig, RendererLimits, SamplingDefaults};
pub use error::{RendererError, RendererErrorKind};
#[cfg(feature = "http")]
pub(crate) use generation::{
    GenerationFinishReason, GenerationOutput, GenerationOutputExtras, GenerationStream, MatchedStop,
};
#[cfg(feature = "http")]
pub use http::{RendererRuntimeConfig, serve};
pub use output::{
    ChatEvent, ChatFinishReason, ChatResponseProcessor, ChatToolCallDelta, DecodedChatEvent,
    ResponseError,
};
pub use request::{
    GenerateRequest, GenerateRequestMetadata, GenerateSamplingParams, GenerationOptions,
    TextRequest, TokenIdsRequest,
};
pub use sampling::SamplingParams;
#[cfg(feature = "http")]
pub(crate) use sampling::SamplingParamsOverrides;
pub(crate) use service::TokenizationBackend;
pub use service::{PreparedChat, RendererService};
pub(crate) use template::ChatFormatter;
#[cfg(test)]
pub(crate) use tokenizer::NoTokenizer;
#[cfg(test)]
pub(crate) use tokenizer::PooledTokenizer;
pub use tokenizer::{DynamoTokenizer, TextTokenizer, load_tokenizer};
pub use types::{OneOrMany, OneOrManyItem, TokenIds};
