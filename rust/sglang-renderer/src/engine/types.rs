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

#[derive(Debug, Clone, PartialEq)]
pub struct TokenLogprob {
    pub logprob: Option<f32>,
    pub token_id: i32,
    pub text: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PositionLogprobs {
    pub token: TokenLogprob,
    pub top: Vec<TokenLogprob>,
}

#[derive(Debug, Clone, Default)]
pub struct GenerationOutputExtras {
    pub output_logprobs: Vec<PositionLogprobs>,
    pub input_logprobs: Vec<PositionLogprobs>,
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
