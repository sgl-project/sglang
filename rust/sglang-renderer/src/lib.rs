//! Standalone SGLang renderer service: OpenAI routes, request preparation,
//! and the SGLang HTTP `/generate` engine client over `sglang-processor`.
//!
//! Temporary: see README.md. Protocol adapters own middleware and framing;
//! shared services own submission policy and decoding.

mod config;
mod engine;
mod error;
mod frontend;
mod launcher;
mod openai;
mod preprocessing;
mod runtime;
mod types;

pub(crate) use config::{RendererConfig, RendererLimits, SamplingDefaults};
pub(crate) use engine::{
    GenerationFinishReason, GenerationOutput, GenerationOutputExtras, GenerationStream,
    MatchedStop, PositionLogprobs, TokenLogprob,
};
pub(crate) use error::{RendererError, ResponseError, ResponseErrorKind, UpstreamErrorCode};
pub use launcher::run_cli;
pub(crate) use preprocessing::{
    ChatGenerateRequest, GenerateRequest, GenerateRequestIdentity, GenerateRequestMetadata,
    GenerationOptions, PreparedChat, RendererService, SamplingParams, TextRequest,
    TextRequestGroup, TokenIdsRequest,
};
pub(crate) use runtime::{RendererRuntimeConfig, serve};
pub(crate) use types::{OneOrMany, TokenIds};
