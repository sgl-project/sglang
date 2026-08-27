//! Reusable request preprocessing for SGLang.
//!
//! The core lowers OpenAI requests, renders chat templates, tokenizes prompts,
//! and produces the token-in contract consumed by SGLang. The optional `http`
//! feature adds a standalone OpenAI frontend backed by SGLang's `/generate`
//! endpoint.

mod chat;
mod config;
mod error;
mod generation;
#[cfg(feature = "http")]
mod http;
mod output;
mod protocol;
mod request;
mod sampling;
mod service;
mod template;
mod tokenizer;
mod types;

mod regex;

pub(crate) use chat::{ChatPreprocessor, ChatRequest, LoweredChat};
pub use config::{RendererConfig, RendererLimits, SamplingDefaults};
pub use error::{RendererError, RendererErrorKind};
pub(crate) use generation::{
    FrontendError, GenerationEvent, GenerationFinishReason, GenerationOutput,
    GenerationOutputExtras, GenerationStream, MatchedStop,
};
#[cfg(feature = "http")]
pub use http::{RendererRuntimeConfig, serve};
pub(crate) use output::{
    ChatEvent, ChatFinishReason, ChatResponseError, ChatResponseInput, ChatResponseItem,
    ChatResponseProcessor, ChatToolCallDelta, DecodedChatEvent,
};
pub use request::{
    GenerateRequest, GenerateRequestMetadata, GenerateSamplingParams, GenerationOptions,
    TextRequest, TokenIdsRequest,
};
pub use sampling::SamplingParams;
pub(crate) use sampling::SamplingParamsOverrides;
pub use service::{RendererService, TokenizationBackend};
pub(crate) use template::ChatFormatter;
pub use tokenizer::{DynamoTokenizer, NoTokenizer, PooledTokenizer, TextTokenizer, load_tokenizer};
pub use types::{OneOrMany, OneOrManyItem, TokenIds};
