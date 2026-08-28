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
mod kimi_k25;
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
pub(crate) use chat::LoweredChat;
pub use chat::{ChatRequest, ReasoningEffort};
pub use config::{RendererConfig, RendererLimits, SamplingDefaults};
pub use error::{RendererError, RendererErrorKind};
#[cfg(feature = "http")]
pub(crate) use generation::{
    GenerationFinishReason, GenerationOutput, GenerationOutputExtras, GenerationStream,
    MatchedStop, PositionLogprobs, TokenLogprob,
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
pub use service::{PreparedChat, RendererService};
pub(crate) use template::ChatFormatter;
pub use tokenizer::{DynamoTokenizer, TextTokenizer, load_tokenizer};
pub use types::{OneOrMany, TokenIds};
