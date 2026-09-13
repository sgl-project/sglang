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

/// Normalized engine token delta, before renderer-owned text decoding.
/// Completion counts are deltas; prompt counts describe the complete prompt.
/// A successful stream includes a terminal finish reason.
#[derive(Debug, Clone, Default)]
pub(crate) struct TokenDelta {
    pub token_ids: TokenIds,
    pub finish_reason: Option<GenerationFinishReason>,
    pub prompt_tokens: u32,
    pub completion_tokens: u64,
    pub extras: Option<Box<GenerationOutputExtras>>,
}

impl From<TokenDelta> for GenerationOutput {
    fn from(delta: TokenDelta) -> Self {
        Self {
            text: String::new(),
            token_ids: delta.token_ids,
            finish_reason: delta.finish_reason,
            prompt_tokens: delta.prompt_tokens,
            completion_tokens: delta.completion_tokens,
            extras: delta.extras,
        }
    }
}
