//! Reusable request preprocessing for SGLang.
//!
//! The core renders normalized chat requests, lowers textual completions,
//! tokenizes prompts, and produces the token-in contract consumed by SGLang.
//! OpenAI operations and generation decoding are independent of transport.
//! The optional `http` feature adds HTTP adapters, the SGLang HTTP engine client,
//! and the process runtime. Protocol adapters own middleware and framing;
//! shared services own request preparation, submission policy, and decoding.

mod config;
// Shared serving code is compiled without HTTP; production adapters are optional.
#[cfg_attr(not(feature = "http"), allow(dead_code))]
mod engine;
mod error;
mod frontend;
#[cfg(feature = "http")]
mod launcher;
#[cfg_attr(not(feature = "http"), allow(dead_code))]
mod openai;
mod postprocessing;
mod preprocessing;
#[cfg(feature = "http")]
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
#[cfg(feature = "http")]
pub use launcher::run_cli;
pub use postprocessing::{
    ChatEvent, ChatFinishReason, ChatResponseProcessor, ChatToolCallDelta, DecodedChatEvent,
};
pub(crate) use preprocessing::ChatFormatter;
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
#[cfg(feature = "http")]
pub use runtime::{RendererRuntimeConfig, serve};
pub use types::{OneOrMany, TokenIds};
