//! Engine-neutral boundary used by protocol frontends during generation.

use futures::future::BoxFuture;
use futures::stream::BoxStream;

use crate::{TokenIds, TokenIdsRequest};

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

pub enum GenerationEvent {
    Frame(GenerationOutput),
    Done(GenerationOutput),
}

#[derive(Debug, Clone)]
pub struct FrontendError {
    pub status_code: u16,
    pub message: String,
}

pub struct GenerationSubmission {
    /// Opaque host identity used only to mark terminal lifecycle.
    pub id: String,
    pub events: BoxStream<'static, Result<GenerationEvent, FrontendError>>,
}

/// Host generation implementation. A session spans every choice derived from
/// one OpenAI request so dropping it cancels all unfinished submissions.
pub trait InferenceBackend: Clone + Send + Sync + 'static {
    type Session: InferenceSession;

    fn begin_session(&self) -> Self::Session;
}

pub trait InferenceSession: Send + 'static {
    fn submit(
        &mut self,
        request: TokenIdsRequest,
        stream: bool,
    ) -> BoxFuture<'_, Result<GenerationSubmission, FrontendError>>;

    fn detokenize(&mut self, token_ids: TokenIds) -> BoxFuture<'_, Result<String, FrontendError>>;

    /// Remove a naturally terminal request from cancellation tracking.
    fn complete(&mut self, submission_id: &str);
}
