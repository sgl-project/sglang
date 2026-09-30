//! Standalone SGLang renderer service: OpenAI routes over `sglang-processor`
//! and the SGLang HTTP `/generate` engine client.
//!
//! Temporary: see README.md. Protocol adapters own middleware and framing;
//! shared services own submission policy and decoding.

mod api;
mod engine;
mod http;
mod launcher;
mod runtime;

pub(crate) use engine::{
    GenerationFinishReason, GenerationOutput, GenerationOutputExtras, GenerationStream,
    MatchedStop, PositionLogprobs, TokenLogprob,
};
pub(crate) use runtime::{RendererRuntimeConfig, serve};

fn main() {
    crate::launcher::run_cli().unwrap_or_else(|error| exit(error));
}

fn exit(message: impl std::fmt::Display) -> ! {
    eprintln!("sglang-renderer: {message}");
    std::process::exit(2)
}
