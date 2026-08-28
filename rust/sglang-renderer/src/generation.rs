//! Generated output shared by the OpenAI response paths.

use futures::stream::BoxStream;

use crate::{ResponseError, TokenIds};

#[derive(Debug, Clone, PartialEq)]
pub enum MatchedStop {
    Token(i64),
    Text(String),
    Tokens(Vec<i64>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum GenerationFinishReason {
    Stop(Option<MatchedStop>),
    Length,
    Abort,
    ContentFilter,
    Other(String),
}

#[derive(Debug, Clone, Default)]
pub struct GenerationOutputExtras {
    pub output_logprobs: Vec<f32>,
    pub output_logprob_token_ids: TokenIds,
    pub output_logprob_text: Vec<String>,
    pub input_logprobs: Vec<f32>,
    pub input_logprob_token_ids: TokenIds,
    pub input_logprob_text: Vec<String>,
    pub output_top_logprobs: Vec<f32>,
    pub output_top_logprob_token_ids: TokenIds,
    pub output_top_logprob_lengths: Vec<u32>,
    pub output_top_logprob_text: Vec<String>,
    pub input_top_logprobs: Vec<f32>,
    pub input_top_logprob_token_ids: TokenIds,
    pub input_top_logprob_lengths: Vec<u32>,
    pub input_top_logprob_text: Vec<String>,
}

/// One decoded engine delta. All owned buffers are moved across the boundary.
#[derive(Debug, Clone, Default)]
pub struct GenerationOutput {
    pub text: String,
    pub token_ids: TokenIds,
    pub finish_reason: Option<GenerationFinishReason>,
    pub prompt_tokens: u32,
    pub completion_tokens: u64,
    pub extras: Option<Box<GenerationOutputExtras>>,
}

pub type GenerationStream = BoxStream<'static, Result<GenerationOutput, ResponseError>>;
