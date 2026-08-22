//! Protocol buffer type wrappers for SGLang and vLLM backends
//!
//! This module provides unified enums that wrap proto types from both SGLang and vLLM,
//! allowing the router to work with either backend transparently.

use std::sync::Arc;

use futures_util::StreamExt;
use smg_grpc_client::{
    sglang_proto::{self as sglang, generate_complete::MatchedStop},
    sglang_scheduler::AbortOnDropStream as SglangStream,
    vllm_engine::AbortOnDropStream as VllmStream,
    vllm_proto as vllm,
};

use crate::core::Worker;

/// Unified ProtoRequest
#[derive(Clone)]
pub enum ProtoRequest {
    Generate(ProtoGenerateRequest),
    Embed(ProtoEmbedRequest),
}

impl ProtoRequest {
    /// Get request ID from either variant
    pub fn request_id(&self) -> &str {
        match self {
            Self::Generate(req) => req.request_id(),
            Self::Embed(req) => req.request_id(),
        }
    }
}

/// Unified GenerateRequest that works with both backends
#[derive(Clone)]
pub enum ProtoGenerateRequest {
    Sglang(Box<sglang::GenerateRequest>),
    Vllm(Box<vllm::GenerateRequest>),
}

impl ProtoGenerateRequest {
    /// Get SGLang variant (panics if vLLM)
    pub fn as_sglang(&self) -> &sglang::GenerateRequest {
        match self {
            Self::Sglang(req) => req,
            Self::Vllm(_) => panic!("Expected SGLang GenerateRequest, got vLLM"),
        }
    }

    /// Get mutable SGLang variant (panics if vLLM)
    pub fn as_sglang_mut(&mut self) -> &mut sglang::GenerateRequest {
        match self {
            Self::Sglang(req) => req,
            Self::Vllm(_) => panic!("Expected SGLang GenerateRequest, got vLLM"),
        }
    }

    /// Get vLLM variant (panics if SGLang)
    pub fn as_vllm(&self) -> &vllm::GenerateRequest {
        match self {
            Self::Vllm(req) => req,
            Self::Sglang(_) => panic!("Expected vLLM GenerateRequest, got SGLang"),
        }
    }

    /// Get mutable vLLM variant (panics if SGLang)
    pub fn as_vllm_mut(&mut self) -> &mut vllm::GenerateRequest {
        match self {
            Self::Vllm(req) => req,
            Self::Sglang(_) => panic!("Expected vLLM GenerateRequest, got SGLang"),
        }
    }

    /// Check if this is SGLang
    pub fn is_sglang(&self) -> bool {
        matches!(self, Self::Sglang(_))
    }

    /// Check if this is vLLM
    pub fn is_vllm(&self) -> bool {
        matches!(self, Self::Vllm(_))
    }

    /// Clone the inner request (for passing to generate())
    pub fn clone_inner(&self) -> Self {
        self.clone()
    }

    /// Get request ID
    pub fn request_id(&self) -> &str {
        match self {
            Self::Sglang(req) => &req.request_id,
            Self::Vllm(req) => &req.request_id,
        }
    }
}

/// Unified GenerateResponse from stream
pub enum ProtoGenerateResponse {
    Sglang(Box<sglang::GenerateResponse>),
    Vllm(vllm::GenerateResponse),
}

impl ProtoGenerateResponse {
    /// Whether a legacy/custom SGLang producer sent an in-band error.
    ///
    /// Official `smg-grpc-servicer` releases supported by SGLang report
    /// errors with gRPC status instead. The pinned legacy client schema still
    /// exposes this variant, so treat it as a failed attempt without inferring
    /// provenance from its untyped string fields.
    fn legacy_in_band_attempt_outcome(&self) -> Option<AttemptOutcome> {
        match self {
            Self::Sglang(response) => matches!(
                response.response,
                Some(sglang::generate_response::Response::Error(_))
            )
            .then_some(AttemptOutcome::AttemptFailure),
            Self::Vllm(_) => None,
        }
    }

    /// Get the response variant (chunk, complete, or error)
    ///
    /// Consumes self to avoid cloning large proto messages in hot streaming path
    pub fn into_response(self) -> ProtoResponseVariant {
        match self {
            Self::Sglang(resp) => match resp.response {
                Some(sglang::generate_response::Response::Chunk(chunk)) => {
                    ProtoResponseVariant::Chunk(ProtoGenerateStreamChunk::Sglang(chunk))
                }
                Some(sglang::generate_response::Response::Complete(complete)) => {
                    ProtoResponseVariant::Complete(ProtoGenerateComplete::Sglang(complete))
                }
                Some(sglang::generate_response::Response::Error(error)) => {
                    ProtoResponseVariant::Error(ProtoGenerateError::Sglang(error))
                }
                None => ProtoResponseVariant::None,
            },
            Self::Vllm(resp) => match resp.response {
                Some(vllm::generate_response::Response::Chunk(chunk)) => {
                    ProtoResponseVariant::Chunk(ProtoGenerateStreamChunk::Vllm(chunk))
                }
                Some(vllm::generate_response::Response::Complete(complete)) => {
                    ProtoResponseVariant::Complete(ProtoGenerateComplete::Vllm(complete))
                }
                // Note: vLLM proto no longer has Error variant in GenerateResponse
                None => ProtoResponseVariant::None,
            },
        }
    }
}

/// Response variant extracted from GenerateResponse
pub enum ProtoResponseVariant {
    Chunk(ProtoGenerateStreamChunk),
    Complete(ProtoGenerateComplete),
    Error(ProtoGenerateError),
    None,
}

/// Unified GenerateStreamChunk
#[derive(Clone)]
pub enum ProtoGenerateStreamChunk {
    Sglang(sglang::GenerateStreamChunk),
    Vllm(vllm::GenerateStreamChunk),
}

impl ProtoGenerateStreamChunk {
    /// Get SGLang variant (panics if vLLM)
    pub fn as_sglang(&self) -> &sglang::GenerateStreamChunk {
        match self {
            Self::Sglang(chunk) => chunk,
            Self::Vllm(_) => panic!("Expected SGLang GenerateStreamChunk, got vLLM"),
        }
    }

    /// Get vLLM variant (panics if SGLang)
    pub fn as_vllm(&self) -> &vllm::GenerateStreamChunk {
        match self {
            Self::Vllm(chunk) => chunk,
            Self::Sglang(_) => panic!("Expected vLLM GenerateStreamChunk, got SGLang"),
        }
    }

    /// Check if this is SGLang
    pub fn is_sglang(&self) -> bool {
        matches!(self, Self::Sglang(_))
    }

    /// Check if this is vLLM
    pub fn is_vllm(&self) -> bool {
        matches!(self, Self::Vllm(_))
    }

    /// Get token IDs from chunk (common field)
    pub fn token_ids(&self) -> &[u32] {
        match self {
            Self::Sglang(c) => &c.token_ids,
            Self::Vllm(c) => &c.token_ids,
        }
    }

    /// Get index (for n>1 support)
    /// vLLM doesn't support n>1, so always returns 0
    pub fn index(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.index,
            Self::Vllm(_) => 0, // vLLM doesn't support n>1
        }
    }

    /// Get output logprobs (SGLang only, returns None for vLLM)
    pub fn output_logprobs(&self) -> Option<&sglang::OutputLogProbs> {
        match self {
            Self::Sglang(c) => c.output_logprobs.as_ref(),
            Self::Vllm(_) => None, // TODO: vLLM logprobs mapping
        }
    }

    /// Get prompt tokens (cumulative)
    pub fn prompt_tokens(&self) -> i32 {
        match self {
            Self::Sglang(c) => c.prompt_tokens,
            Self::Vllm(c) => c.prompt_tokens as i32,
        }
    }

    /// Get completion tokens (cumulative)
    pub fn completion_tokens(&self) -> i32 {
        match self {
            Self::Sglang(c) => c.completion_tokens,
            Self::Vllm(c) => c.completion_tokens as i32,
        }
    }

    /// Get cached tokens (cumulative)
    pub fn cached_tokens(&self) -> i32 {
        match self {
            Self::Sglang(c) => c.cached_tokens,
            Self::Vllm(c) => c.cached_tokens as i32,
        }
    }
}

/// Unified GenerateComplete response
#[derive(Clone)]
pub enum ProtoGenerateComplete {
    Sglang(sglang::GenerateComplete),
    Vllm(vllm::GenerateComplete),
}

impl ProtoGenerateComplete {
    /// Get SGLang variant (panics if vLLM)
    pub fn as_sglang(&self) -> &sglang::GenerateComplete {
        match self {
            Self::Sglang(complete) => complete,
            Self::Vllm(_) => panic!("Expected SGLang GenerateComplete, got vLLM"),
        }
    }

    /// Get mutable SGLang variant (panics if vLLM)
    pub fn as_sglang_mut(&mut self) -> &mut sglang::GenerateComplete {
        match self {
            Self::Sglang(complete) => complete,
            Self::Vllm(_) => panic!("Expected SGLang GenerateComplete, got vLLM"),
        }
    }

    /// Get vLLM variant (panics if SGLang)
    pub fn as_vllm(&self) -> &vllm::GenerateComplete {
        match self {
            Self::Vllm(complete) => complete,
            Self::Sglang(_) => panic!("Expected vLLM GenerateComplete, got SGLang"),
        }
    }

    /// Check if this is SGLang
    pub fn is_sglang(&self) -> bool {
        matches!(self, Self::Sglang(_))
    }

    /// Check if this is vLLM
    pub fn is_vllm(&self) -> bool {
        matches!(self, Self::Vllm(_))
    }

    /// Get token IDs from either backend (output_ids in proto)
    pub fn token_ids(&self) -> &[u32] {
        match self {
            Self::Sglang(c) => &c.output_ids,
            Self::Vllm(c) => &c.output_ids,
        }
    }

    /// Get prompt tokens
    pub fn prompt_tokens(&self) -> i32 {
        match self {
            Self::Sglang(c) => c.prompt_tokens,
            Self::Vllm(c) => c.prompt_tokens as i32,
        }
    }

    /// Get completion tokens
    pub fn completion_tokens(&self) -> i32 {
        match self {
            Self::Sglang(c) => c.completion_tokens,
            Self::Vllm(c) => c.completion_tokens as i32,
        }
    }

    /// Get finish reason
    pub fn finish_reason(&self) -> &str {
        match self {
            Self::Sglang(c) => &c.finish_reason,
            Self::Vllm(c) => &c.finish_reason,
        }
    }

    /// Get index (for n>1 support)
    /// vLLM doesn't support n>1, so always returns 0
    pub fn index(&self) -> u32 {
        match self {
            Self::Sglang(c) => c.index,
            Self::Vllm(_) => 0, // vLLM doesn't have index field (n>1 not supported)
        }
    }

    /// Get matched stop (SGLang only, returns oneof)
    /// vLLM doesn't have matched_stop, returns None
    pub fn matched_stop(&self) -> Option<&MatchedStop> {
        match self {
            Self::Sglang(c) => c.matched_stop.as_ref(),
            Self::Vllm(_) => None, // vLLM doesn't have matched_stop
        }
    }

    /// Get output IDs (decode tokens only)
    pub fn output_ids(&self) -> &[u32] {
        match self {
            Self::Sglang(c) => &c.output_ids,
            Self::Vllm(c) => &c.output_ids,
        }
    }

    /// Get cached tokens
    pub fn cached_tokens(&self) -> i32 {
        match self {
            Self::Sglang(c) => c.cached_tokens,
            Self::Vllm(c) => c.cached_tokens as i32,
        }
    }

    /// Get input logprobs (SGLang only)
    pub fn input_logprobs(&self) -> Option<&sglang::InputLogProbs> {
        match self {
            Self::Sglang(c) => c.input_logprobs.as_ref(),
            Self::Vllm(_) => None, // vLLM doesn't have input_logprobs
        }
    }

    /// Get output logprobs
    pub fn output_logprobs(&self) -> Option<&sglang::OutputLogProbs> {
        match self {
            Self::Sglang(c) => c.output_logprobs.as_ref(),
            Self::Vllm(_) => None, // TODO: vLLM logprobs mapping
        }
    }
}

/// Unified GenerateError
/// Note: vLLM proto no longer has GenerateError - errors are returned via gRPC status
#[derive(Clone)]
pub enum ProtoGenerateError {
    Sglang(sglang::GenerateError),
}

impl ProtoGenerateError {
    /// Get error message
    pub fn message(&self) -> &str {
        match self {
            Self::Sglang(e) => &e.message,
        }
    }
}

/// Terminal outcome of a selected single-worker attempt, decided at body terminal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttemptOutcome {
    /// The consumer accepted the fully drained response body.
    Success,
    /// The selected upstream attempt ended with a transport/protocol error.
    ///
    /// This is an attempt result, not a claim about which component caused it.
    AttemptFailure,
    /// The downstream stopped before an upstream terminal verdict was known.
    Abandoned,
}

impl AttemptOutcome {
    fn publish_to(self, worker: &dyn Worker) {
        match self {
            Self::Success => worker.record_outcome(true),
            Self::AttemptFailure => worker.record_outcome(false),
            Self::Abandoned => {}
        }
    }
}

/// Single-owner right to publish the breaker outcome of one selected attempt.
///
/// The right starts Detached, is transferred to the finally selected worker at
/// stream-start success, and is consumed exactly once by the first legal
/// body-terminal observation. Duplicate and late observations are absorbed.
enum BreakerReceipt {
    /// No breaker publication will happen (e.g. dual PD legs stay detached).
    Detached,
    /// The stream holds the publication right for this selected worker.
    Active(Arc<dyn Worker>),
    /// The right has been consumed; later observations are no-ops.
    Resolved,
}

impl BreakerReceipt {
    /// Consume the receipt, publishing `outcome` at most once (absorbing).
    fn resolve(&mut self, outcome: AttemptOutcome) {
        if let BreakerReceipt::Active(worker) = std::mem::replace(self, BreakerReceipt::Resolved) {
            outcome.publish_to(worker.as_ref());
        }
    }
}

/// Unified stream wrapper carrying the breaker publication receipt.
pub struct ProtoStream {
    inner: ProtoStreamInner,
    receipt: BreakerReceipt,
}

enum ProtoStreamInner {
    Sglang(SglangStream),
    Vllm(VllmStream),
}

impl ProtoStream {
    pub(crate) fn sglang(stream: SglangStream) -> Self {
        Self {
            inner: ProtoStreamInner::Sglang(stream),
            receipt: BreakerReceipt::Detached,
        }
    }

    pub(crate) fn vllm(stream: VllmStream) -> Self {
        Self {
            inner: ProtoStreamInner::Vllm(stream),
            receipt: BreakerReceipt::Detached,
        }
    }

    /// Attach the selected worker breaker publication right (crate-private).
    ///
    /// Called exactly once by the dispatch stage after stream-start success
    /// and final worker selection; attaching twice is a programming error.
    pub(crate) fn attach_breaker_receipt(
        &mut self,
        worker: Arc<dyn Worker>,
    ) -> Result<(), &'static str> {
        if !matches!(self.receipt, BreakerReceipt::Detached) {
            return Err("breaker receipt is already attached or resolved");
        }
        self.receipt = BreakerReceipt::Active(worker);
        Ok(())
    }

    /// Get next item from stream.
    ///
    /// Transport and legacy in-band failures resolve the receipt before they
    /// are returned. The consumer acknowledges a successfully drained body
    /// through `mark_completed`.
    pub async fn next(&mut self) -> Option<Result<ProtoGenerateResponse, tonic::Status>> {
        let item = match &mut self.inner {
            ProtoStreamInner::Sglang(stream) => stream
                .next()
                .await
                .map(|result| result.map(|r| ProtoGenerateResponse::Sglang(Box::new(r)))),
            ProtoStreamInner::Vllm(stream) => stream
                .next()
                .await
                .map(|result| result.map(ProtoGenerateResponse::Vllm)),
        };
        match &item {
            None => {}
            Some(Err(_)) => {
                self.receipt.resolve(AttemptOutcome::AttemptFailure);
            }
            Some(Ok(response)) => {
                if let Some(outcome) = response.legacy_in_band_attempt_outcome() {
                    self.receipt.resolve(outcome);
                }
            }
        }
        item
    }

    /// Mark stream as completed (no abort needed).
    ///
    /// This is the consumer's enforced terminal declaration: it publishes one
    /// success and suppresses the dependency's abort-on-drop behavior.
    pub fn mark_completed(&mut self) {
        self.receipt.resolve(AttemptOutcome::Success);
        self.mark_inner_completed();
    }

    /// Reject a fully drained body that violates the consumer contract.
    ///
    /// The selected attempt failed, but the transport stream is already at a
    /// clean terminal, so suppress the dependency's now-unnecessary abort.
    pub(crate) fn reject_completed_body(&mut self) {
        self.receipt.resolve(AttemptOutcome::AttemptFailure);
        self.mark_inner_completed();
    }

    fn mark_inner_completed(&mut self) {
        match &mut self.inner {
            ProtoStreamInner::Sglang(stream) => stream.mark_completed(),
            ProtoStreamInner::Vllm(stream) => stream.mark_completed(),
        }
    }
}

impl Drop for ProtoStream {
    fn drop(&mut self) {
        // Downstream drop consumes the receipt without publishing an outcome;
        // inner abort-on-drop remains an independent responsibility.
        self.receipt.resolve(AttemptOutcome::Abandoned);
    }
}

/// Unified EmbedRequest that works with both backends
#[derive(Clone)]
pub enum ProtoEmbedRequest {
    Sglang(Box<sglang::EmbedRequest>),
}

impl ProtoEmbedRequest {
    /// Get SGLang variant
    pub fn as_sglang(&self) -> &sglang::EmbedRequest {
        match self {
            Self::Sglang(req) => req,
        }
    }

    /// Get mutable SGLang variant
    pub fn as_sglang_mut(&mut self) -> &mut sglang::EmbedRequest {
        match self {
            Self::Sglang(req) => req,
        }
    }

    /// Check if this is SGLang
    pub fn is_sglang(&self) -> bool {
        matches!(self, Self::Sglang(_))
    }

    /// Clone the inner request (for passing to embed())
    pub fn clone_inner(&self) -> Self {
        self.clone()
    }

    /// Get request ID
    pub fn request_id(&self) -> &str {
        match self {
            Self::Sglang(req) => &req.request_id,
        }
    }
}

/// Unified EmbedResponse
pub enum ProtoEmbedResponse {
    Sglang(sglang::EmbedResponse),
}

impl ProtoEmbedResponse {
    /// Get the response variant (complete or error)
    pub fn into_response(self) -> ProtoEmbedResponseVariant {
        match self {
            Self::Sglang(resp) => match resp.response {
                Some(sglang::embed_response::Response::Complete(complete)) => {
                    ProtoEmbedResponseVariant::Complete(ProtoEmbedComplete::Sglang(complete))
                }
                Some(sglang::embed_response::Response::Error(error)) => {
                    ProtoEmbedResponseVariant::Error(ProtoEmbedError::Sglang(error))
                }
                None => ProtoEmbedResponseVariant::None,
            },
        }
    }
}

/// Response variant extracted from EmbedResponse
pub enum ProtoEmbedResponseVariant {
    Complete(ProtoEmbedComplete),
    Error(ProtoEmbedError),
    None,
}

/// Unified EmbedComplete response
#[derive(Clone)]
pub enum ProtoEmbedComplete {
    Sglang(sglang::EmbedComplete),
}

impl ProtoEmbedComplete {
    /// Get embeddings
    pub fn embedding(&self) -> &[f32] {
        match self {
            Self::Sglang(c) => &c.embedding,
        }
    }

    /// Get prompt tokens
    pub fn prompt_tokens(&self) -> i32 {
        match self {
            Self::Sglang(c) => c.prompt_tokens,
        }
    }

    /// Get cached tokens
    pub fn cached_tokens(&self) -> i32 {
        match self {
            Self::Sglang(c) => c.cached_tokens,
        }
    }

    /// Get embedding dimension
    pub fn embedding_dim(&self) -> i32 {
        match self {
            Self::Sglang(c) => c.embedding_dim,
        }
    }
}

/// Unified EmbedError
#[derive(Clone)]
pub enum ProtoEmbedError {
    Sglang(sglang::EmbedError),
}

impl ProtoEmbedError {
    /// Get error message
    pub fn message(&self) -> &str {
        match self {
            Self::Sglang(e) => &e.message,
        }
    }

    /// Get error code
    pub fn code(&self) -> &str {
        match self {
            Self::Sglang(e) => &e.code,
        }
    }
}

#[cfg(test)]
mod tests;
