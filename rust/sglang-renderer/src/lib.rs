//! Reusable request preprocessing for SGLang.
//!
//! The core renders normalized chat requests, lowers textual completions,
//! tokenizes prompts, and produces the token-in contract consumed by SGLang.
//! The optional `http` feature owns OpenAI wire lowering, an engine-free
//! rendering frontend, and optional inference backed by SGLang's `/generate`
//! endpoint.

mod config;
#[cfg(feature = "http")]
mod engine;
mod error;
#[cfg(feature = "http")]
mod launcher;
#[cfg(feature = "http")]
mod openai;
mod preprocessing;
mod response;
#[cfg(feature = "http")]
mod runtime;
mod types;

pub use config::{RendererConfig, RendererLimits, SamplingDefaults};
#[cfg(feature = "http")]
pub(crate) use engine::{
    GenerationFinishReason, GenerationOutput, GenerationOutputExtras, GenerationStream,
    MatchedStop, PositionLogprobs, TokenLogprob,
};
pub use error::{RendererError, RendererErrorKind};
#[cfg(feature = "http")]
pub use launcher::run_cli;
pub(crate) use preprocessing::ChatFormatter;
#[cfg(feature = "http")]
pub(crate) use preprocessing::SamplingParamsOverrides;
pub(crate) use preprocessing::{ChatPreprocessor, LoweredChat};
pub use preprocessing::{
    ChatRequest, DynamoTokenizer, PreparedChat, ReasoningEffort, RendererService, SamplingParams,
    TextTokenizer, load_tokenizer,
};
pub use preprocessing::{
    GenerateRequest, GenerateRequestMetadata, GenerateSamplingParams, GenerationOptions,
    TextRequest, TokenIdsRequest,
};
pub use response::{
    ChatEvent, ChatFinishReason, ChatResponseProcessor, ChatToolCallDelta, DecodedChatEvent,
    ResponseError,
};
#[cfg(feature = "http")]
pub use runtime::{RendererRuntimeConfig, serve};
pub use types::{OneOrMany, TokenIds};
