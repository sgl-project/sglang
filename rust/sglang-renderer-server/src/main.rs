//! Standalone SGLang renderer service: OpenAI routes, request preparation,
//! and the SGLang HTTP `/generate` engine client over `sglang-processor`.
//!
//! Temporary: see README.md. Protocol adapters own middleware and framing;
//! shared services own submission policy and decoding.

mod api;
mod config;
mod engine;
mod error;
mod http;
mod launcher;
mod preprocessing;
mod runtime;
mod types;

pub(crate) use config::{RendererConfig, RendererLimits, SamplingDefaults};
pub(crate) use engine::{
    GenerationFinishReason, GenerationOutput, GenerationOutputExtras, GenerationStream,
    MatchedStop, PositionLogprobs, TokenLogprob,
};
pub(crate) use error::{RendererError, ResponseError, ResponseErrorKind, UpstreamErrorCode};
pub(crate) use preprocessing::{
    ChatGenerateRequest, GenerateRequest, GenerateRequestIdentity, GenerateRequestMetadata,
    GenerationOptions, PreparedChat, RendererService, SamplingParams, TextRequest,
    TextRequestGroup, TokenIdsRequest,
};
pub(crate) use runtime::{RendererRuntimeConfig, serve};
pub(crate) use types::{OneOrMany, TokenIds};

fn main() {
    crate::launcher::run_cli().unwrap_or_else(|error| exit(error));
}

fn exit(message: impl std::fmt::Display) -> ! {
    eprintln!("sglang-renderer: {message}");
    std::process::exit(2)
}
