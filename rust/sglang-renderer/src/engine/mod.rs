//! Token-only generation transport and decoded engine output.

mod client;
mod types;

pub(crate) use client::HttpGenerateClient;
pub(crate) use types::{
    GenerationFinishReason, GenerationOutput, GenerationOutputExtras, GenerationStream,
    MatchedStop, PositionLogprobs, TokenLogprob,
};
