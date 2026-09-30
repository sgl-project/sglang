//! Reusable request preprocessing for SGLang.
//!
//! The core renders normalized chat requests, lowers textual completions,
//! tokenizes prompts, and produces the token-in contract consumed by SGLang.
//! OpenAI operations and generation decoding are independent of transport.
//! Templates, tokenizer loading, and output parsing come from `sglang-processor`.
//! Protocol adapters own middleware and framing; shared services own request
//! preparation, submission policy, and decoding.

mod config;
mod engine;
mod error;
mod frontend;
mod launcher;
mod openai;
mod preprocessing;
mod runtime;
mod types;

pub use config::{RendererConfig, RendererLimits, SamplingDefaults};
pub(crate) use engine::{
    GenerationFinishReason, GenerationOutput, GenerationOutputExtras, GenerationStream,
    MatchedStop, PositionLogprobs, TokenLogprob,
};
pub use error::{
    RendererError, RendererErrorKind, ResponseError, ResponseErrorKind, UpstreamErrorCode,
};
pub use launcher::run_cli;
pub(crate) use preprocessing::ChatFormatter;
pub use preprocessing::RegexPattern;
pub(crate) use preprocessing::{ChatPreprocessor, LoweredChat};
pub use preprocessing::{
    ChatRequest, DynamoTokenizer, PreparedChat, ReasoningEffort, RendererService, SamplingParams,
    TextTokenizer, load_tokenizer,
};
pub use preprocessing::{
    GenerateRequest, GenerateRequestMetadata, GenerateSamplingParams, GenerationOptions,
    TextRequest, TokenIdsRequest,
};
pub use runtime::{RendererRuntimeConfig, serve};
pub use sglang_processor::{
    ChatEvent, ChatFinishReason, ChatResponseProcessor, ChatToolCallDelta, DecodedChatEvent,
};
pub use types::{OneOrMany, TokenIds};
