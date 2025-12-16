//! Protocol buffer type wrappers for SGLang and vLLM backends
//!
//! This module provides unified enums that wrap proto types from both SGLang and vLLM,
//! allowing the router to work with either backend transparently.

use futures_util::StreamExt;

use crate::grpc_client::{
    sglang_proto::{self as sglang, generate_complete::MatchedStop},
    sglang_scheduler::AbortOnDropStream as SglangStream,
    vllm_engine::AbortOnDropStream as VllmStream,
    vllm_proto as vllm,
};

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
    Sglang(sglang::GenerateResponse),
    Vllm(vllm::GenerateResponse),
}

impl ProtoGenerateResponse {
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
                Some(vllm::generate_response::Response::Error(error)) => {
                    ProtoResponseVariant::Error(ProtoGenerateError::Vllm(error))
                }
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
            Self::Vllm(c) => c.prompt_tokens,
        }
    }

    /// Get completion tokens (cumulative)
    pub fn completion_tokens(&self) -> i32 {
        match self {
            Self::Sglang(c) => c.completion_tokens,
            Self::Vllm(c) => c.completion_tokens,
        }
    }

    /// Get cached tokens (cumulative)
    pub fn cached_tokens(&self) -> i32 {
        match self {
            Self::Sglang(c) => c.cached_tokens,
            Self::Vllm(c) => c.cached_tokens,
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
            Self::Vllm(c) => c.prompt_tokens,
        }
    }

    /// Get completion tokens
    pub fn completion_tokens(&self) -> i32 {
        match self {
            Self::Sglang(c) => c.completion_tokens,
            Self::Vllm(c) => c.completion_tokens,
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
            Self::Vllm(_) => 0, // vLLM doesn't have cached_tokens field
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
#[derive(Clone)]
pub enum ProtoGenerateError {
    Sglang(sglang::GenerateError),
    Vllm(vllm::GenerateError),
}

impl ProtoGenerateError {
    /// Get error message
    pub fn message(&self) -> &str {
        match self {
            Self::Sglang(e) => &e.message,
            Self::Vllm(e) => &e.message,
        }
    }
}

/// Unified stream wrapper
pub enum ProtoStream {
    Sglang(SglangStream),
    Vllm(VllmStream),
}

impl ProtoStream {
    /// Get next item from stream
    pub async fn next(&mut self) -> Option<Result<ProtoGenerateResponse, tonic::Status>> {
        match self {
            Self::Sglang(stream) => stream
                .next()
                .await
                .map(|result| result.map(ProtoGenerateResponse::Sglang)),
            Self::Vllm(stream) => stream
                .next()
                .await
                .map(|result| result.map(ProtoGenerateResponse::Vllm)),
        }
    }

    /// Mark stream as completed (no abort needed)
    pub fn mark_completed(&mut self) {
        match self {
            Self::Sglang(stream) => stream.mark_completed(),
            Self::Vllm(stream) => stream.mark_completed(),
        }
    }
}
