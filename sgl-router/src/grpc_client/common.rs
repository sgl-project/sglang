//! Common utilities shared between SGLang and vLLM gRPC clients

use std::{
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    task::{Context, Poll},
    time::Duration,
};

use tonic::{transport::Channel, Streaming};
use tracing::{debug, warn};

use super::{SglangSchedulerClient, VllmEngineClient};

// Both clients now use the same unified proto from inference.grpc package
use super::sglang_scheduler::proto;

/// Convert grpc:// endpoint to http:// for tonic
pub fn convert_grpc_to_http(endpoint: &str) -> String {
    if let Some(addr) = endpoint.strip_prefix("grpc://") {
        format!("http://{}", addr)
    } else {
        endpoint.to_string()
    }
}

/// Create a gRPC channel with optimized settings for SGLang/vLLM
///
/// This configures the channel with:
/// - HTTP/2 keep-alive (30s interval, 10s timeout)
/// - TCP keep-alive (60s)
/// - TCP nodelay (disable Nagle's algorithm)
/// - HTTP/2 adaptive window sizing
/// - Large stream/connection windows (16MB/32MB)
pub async fn create_grpc_channel(
    endpoint: &str,
) -> Result<Channel, Box<dyn std::error::Error + Send + Sync>> {
    let http_endpoint = convert_grpc_to_http(endpoint);

    let channel = Channel::from_shared(http_endpoint)?
        .http2_keep_alive_interval(Duration::from_secs(30))
        .keep_alive_timeout(Duration::from_secs(10))
        .keep_alive_while_idle(true)
        .tcp_keepalive(Some(Duration::from_secs(60)))
        .tcp_nodelay(true)
        .http2_adaptive_window(true)
        .initial_stream_window_size(Some(16 * 1024 * 1024)) // 16MB
        .initial_connection_window_size(Some(32 * 1024 * 1024)) // 32MB
        .connect()
        .await?;

    Ok(channel)
}

/// Trait for gRPC clients that support abort operations
pub trait AbortableClient: Clone + Send + Sync {
    /// Abort a request by ID
    fn abort_request(
        &self,
        request_id: String,
        reason: String,
    ) -> impl std::future::Future<Output = Result<(), Box<dyn std::error::Error + Send + Sync>>> + Send;
}

/// A smart wrapper around Streaming<R> that automatically sends abort when dropped.
///
/// This leverages Rust's RAII pattern to ensure cleanup happens automatically,
/// regardless of how the stream is dropped (panic, early return, client disconnect, etc.).
///
/// Generic over:
/// - `C`: Client type (must implement AbortableClient)
/// - `R`: Response type from the stream
pub struct AbortOnDropStream<C, R>
where
    C: AbortableClient,
{
    inner: Streaming<R>,
    request_id: String,
    client: C,
    aborted: Arc<AtomicBool>,
}

impl<C, R> AbortOnDropStream<C, R>
where
    C: AbortableClient,
{
    /// Create a new auto-aborting stream wrapper
    pub fn new(stream: Streaming<R>, request_id: String, client: C) -> Self {
        debug!("Created AbortOnDropStream for request {}", request_id);
        Self {
            inner: stream,
            request_id,
            client,
            aborted: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Manually mark the request as completed to prevent abort on drop.
    /// Call this when the request completes successfully to avoid unnecessary abort RPC.
    pub fn mark_completed(&self) {
        // Use Release ordering to ensure that this write is visible to other threads
        // that use Acquire on the same atomic variable
        self.aborted.store(true, Ordering::Release);
        debug!("Request {} marked as completed", self.request_id);
    }
}

impl<C, R> Drop for AbortOnDropStream<C, R>
where
    C: AbortableClient + 'static,
{
    fn drop(&mut self) {
        // Atomically check and set the aborted flag using compare_exchange.
        // If compare_exchange fails, it means the flag was already true (from mark_completed),
        // so we don't need to send abort. AcqRel is used for success to synchronize with
        // mark_completed's Release, and Acquire for failure to see writes from mark_completed.
        if self
            .aborted
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }

        let client = self.client.clone();
        let request_id = self.request_id.clone();

        // Spawn a background task to send abort (since Drop is sync but abort_request is async)
        tokio::spawn(async move {
            debug!(
                "Stream dropped without completion for request {}, sending abort",
                request_id
            );
            // Clone request_id for the error message since abort_request takes ownership
            let request_id_for_log = request_id.clone();
            if let Err(e) = client
                .abort_request(request_id, "Stream dropped".to_string())
                .await
            {
                warn!(
                    "Failed to abort request {} on stream drop: {}",
                    request_id_for_log, e
                );
            }
        });
    }
}

// Implement Stream trait to allow polling
impl<C, R> futures::Stream for AbortOnDropStream<C, R>
where
    C: AbortableClient,
    R: Unpin,
{
    type Item = Result<R, tonic::Status>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // SAFETY: We're not moving out of self, just getting a mutable reference to inner
        let inner = unsafe { &mut self.get_unchecked_mut().inner };
        Pin::new(inner).poll_next(cx)
    }
}

// ============================================================================
// UNIFIED RESPONSE TYPES FOR ENUM DISPATCH
// ============================================================================

/// Response type discriminator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseType {
    Chunk,
    Complete,
    Error,
    None,
}

/// Matched stop information (from SGLang proto)
#[derive(Debug, Clone)]
pub enum MatchedStop {
    TokenId(u32),
    StopStr(String),
}

/// Unified chunk data with ALL SGLang fields
/// For vLLM: SGLang-specific fields are set to None/defaults
#[derive(Debug, Clone)]
pub struct ChunkData {
    // Common fields (both SGLang and vLLM)
    pub token_ids: Vec<u32>,
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub cached_tokens: i32,
    pub index: u32, // vLLM: defaults to 0

    // SGLang-specific fields (vLLM: None/empty)
    pub output_logprobs: Option<proto::OutputLogProbs>,
    pub hidden_states: Vec<f32>,
    pub input_logprobs: Option<proto::InputLogProbs>,
}

/// Unified complete data with ALL SGLang fields
/// For vLLM: SGLang-specific fields are set to None/defaults
#[derive(Debug, Clone)]
pub struct CompleteData {
    // Common fields (both SGLang and vLLM)
    pub output_ids: Vec<u32>,
    pub finish_reason: String,
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub cached_tokens: i32,
    pub index: u32, // vLLM: defaults to 0

    // SGLang-specific fields (vLLM: None/empty)
    pub output_logprobs: Option<proto::OutputLogProbs>,
    pub all_hidden_states: Vec<proto::HiddenStates>,
    pub matched_stop: Option<MatchedStop>,
    pub input_logprobs: Option<proto::InputLogProbs>,
}

/// Unified GenerateResponse that can hold either SGLang or vLLM proto
#[derive(Clone)]
pub enum GenerateResponse {
    Sglang(proto::GenerateResponse),
    Vllm(proto::GenerateResponse),
}

impl GenerateResponse {
    /// Get request ID (common to both)
    pub fn request_id(&self) -> &str {
        match self {
            Self::Sglang(resp) => &resp.request_id,
            Self::Vllm(resp) => &resp.request_id,
        }
    }

    /// Get the response type (chunk/complete/error/none)
    pub fn response_type(&self) -> ResponseType {
        use proto::generate_response::Response as SglangResp;
        use proto::generate_response::Response as VllmResp;

        match self {
            Self::Sglang(resp) => match &resp.response {
                Some(SglangResp::Chunk(_)) => ResponseType::Chunk,
                Some(SglangResp::Complete(_)) => ResponseType::Complete,
                Some(SglangResp::Error(_)) => ResponseType::Error,
                None => ResponseType::None,
            },
            Self::Vllm(resp) => match &resp.response {
                Some(VllmResp::Chunk(_)) => ResponseType::Chunk,
                Some(VllmResp::Complete(_)) => ResponseType::Complete,
                Some(VllmResp::Error(_)) => ResponseType::Error,
                None => ResponseType::None,
            },
        }
    }

    /// Extract chunk data with ALL fields (SGLang-specific fields are None/empty for vLLM)
    pub fn as_chunk(&self) -> Option<ChunkData> {
        use proto::generate_response::Response as SglangResp;
        use proto::generate_response::Response as VllmResp;

        match self {
            Self::Sglang(resp) => {
                if let Some(SglangResp::Chunk(chunk)) = &resp.response {
                    Some(ChunkData {
                        // Common fields
                        token_ids: chunk.token_ids.clone(),
                        prompt_tokens: chunk.prompt_tokens,
                        completion_tokens: chunk.completion_tokens,
                        cached_tokens: chunk.cached_tokens,
                        index: chunk.index,
                        // SGLang-specific fields
                        output_logprobs: chunk.output_logprobs.clone(),
                        hidden_states: chunk.hidden_states.clone(),
                        input_logprobs: chunk.input_logprobs.clone(),
                    })
                } else {
                    None
                }
            }
            Self::Vllm(resp) => {
                if let Some(VllmResp::Chunk(chunk)) = &resp.response {
                    Some(ChunkData {
                        // Common fields
                        token_ids: chunk.token_ids.clone(),
                        prompt_tokens: chunk.prompt_tokens,
                        completion_tokens: chunk.completion_tokens,
                        cached_tokens: chunk.cached_tokens,
                        index: 0, // vLLM doesn't have index field - default to 0
                        // SGLang-specific fields - None/empty for vLLM
                        output_logprobs: None,
                        hidden_states: Vec::new(),
                        input_logprobs: None,
                    })
                } else {
                    None
                }
            }
        }
    }

    /// Extract complete data with ALL fields (SGLang-specific fields are None/empty for vLLM)
    pub fn as_complete(&self) -> Option<CompleteData> {
        use proto::generate_response::Response as SglangResp;
        use proto::generate_response::Response as VllmResp;

        match self {
            Self::Sglang(resp) => {
                if let Some(SglangResp::Complete(complete)) = &resp.response {
                    // Convert matched_stop from proto oneof to our enum
                    use proto::generate_complete::MatchedStop as ProtoMatched;
                    let matched_stop = complete.matched_stop.as_ref().map(|m| match m {
                        ProtoMatched::MatchedTokenId(id) => MatchedStop::TokenId(*id),
                        ProtoMatched::MatchedStopStr(s) => MatchedStop::StopStr(s.clone()),
                    });

                    Some(CompleteData {
                        // Common fields
                        output_ids: complete.output_ids.clone(),
                        finish_reason: complete.finish_reason.clone(),
                        prompt_tokens: complete.prompt_tokens,
                        completion_tokens: complete.completion_tokens,
                        cached_tokens: complete.cached_tokens,
                        index: complete.index,
                        // SGLang-specific fields
                        output_logprobs: complete.output_logprobs.clone(),
                        all_hidden_states: complete.all_hidden_states.clone(),
                        matched_stop,
                        input_logprobs: complete.input_logprobs.clone(),
                    })
                } else {
                    None
                }
            }
            Self::Vllm(resp) => {
                if let Some(VllmResp::Complete(complete)) = &resp.response {
                    Some(CompleteData {
                        // Common fields
                        output_ids: complete.output_ids.clone(),
                        finish_reason: complete.finish_reason.clone(),
                        prompt_tokens: complete.prompt_tokens,
                        completion_tokens: complete.completion_tokens,
                        cached_tokens: complete.cached_tokens,
                        index: 0, // vLLM doesn't have index field - default to 0
                        // SGLang-specific fields - None/empty for vLLM
                        output_logprobs: None,
                        all_hidden_states: Vec::new(),
                        matched_stop: None,
                        input_logprobs: None,
                    })
                } else {
                    None
                }
            }
        }
    }

    /// Get error message (if this is an error response)
    pub fn as_error(&self) -> Option<&str> {
        use proto::generate_response::Response as SglangResp;
        use proto::generate_response::Response as VllmResp;

        match self {
            Self::Sglang(resp) => {
                if let Some(SglangResp::Error(err)) = &resp.response {
                    Some(&err.message)
                } else {
                    None
                }
            }
            Self::Vllm(resp) => {
                if let Some(VllmResp::Error(err)) = &resp.response {
                    Some(&err.message)
                } else {
                    None
                }
            }
        }
    }

    /// Get the underlying SGLang response (if this is SGLang variant)
    pub fn as_sglang(&self) -> Option<&proto::GenerateResponse> {
        match self {
            Self::Sglang(resp) => Some(resp),
            _ => None,
        }
    }

    /// Get the underlying vLLM response (if this is vLLM variant)
    pub fn as_vllm(&self) -> Option<&proto::GenerateResponse> {
        match self {
            Self::Vllm(resp) => Some(resp),
            _ => None,
        }
    }

    /// Get the underlying SGLang chunk (for proto-specific fields like logprobs)
    pub fn get_sglang_chunk(&self) -> Option<&proto::GenerateStreamChunk> {
        use proto::generate_response::Response;
        match self {
            Self::Sglang(resp) => match &resp.response {
                Some(Response::Chunk(chunk)) => Some(chunk),
                _ => None,
            },
            _ => None,
        }
    }

    /// Get the underlying vLLM chunk
    pub fn get_vllm_chunk(&self) -> Option<&proto::GenerateStreamChunk> {
        use proto::generate_response::Response;
        match self {
            Self::Vllm(resp) => match &resp.response {
                Some(Response::Chunk(chunk)) => Some(chunk),
                _ => None,
            },
            _ => None,
        }
    }

    /// Get the underlying SGLang complete (for proto-specific fields)
    pub fn get_sglang_complete(&self) -> Option<&proto::GenerateComplete> {
        use proto::generate_response::Response;
        match self {
            Self::Sglang(resp) => match &resp.response {
                Some(Response::Complete(complete)) => Some(complete),
                _ => None,
            },
            _ => None,
        }
    }

    /// Get the underlying vLLM complete
    pub fn get_vllm_complete(&self) -> Option<&proto::GenerateComplete> {
        use proto::generate_response::Response;
        match self {
            Self::Vllm(resp) => match &resp.response {
                Some(Response::Complete(complete)) => Some(complete),
                _ => None,
            },
            _ => None,
        }
    }
}

/// Unified stream that can yield either SGLang or vLLM responses
pub enum GrpcStream {
    Sglang(AbortOnDropStream<SglangSchedulerClient, proto::GenerateResponse>),
    Vllm(AbortOnDropStream<VllmEngineClient, proto::GenerateResponse>),
}

impl GrpcStream {
    /// Mark the stream as completed (prevents abort on drop)
    pub fn mark_completed(&self) {
        match self {
            Self::Sglang(stream) => stream.mark_completed(),
            Self::Vllm(stream) => stream.mark_completed(),
        }
    }
}

impl futures::Stream for GrpcStream {
    type Item = Result<GenerateResponse, tonic::Status>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.as_mut().get_mut() {
            Self::Sglang(stream) => Pin::new(stream)
                .poll_next(cx)
                .map(|opt| opt.map(|res| res.map(GenerateResponse::Sglang))),
            Self::Vllm(stream) => Pin::new(stream)
                .poll_next(cx)
                .map(|opt| opt.map(|res| res.map(GenerateResponse::Vllm))),
        }
    }
}

/// Unified client enum for runtime dispatch
#[derive(Clone)]
pub enum GrpcClient {
    Sglang(SglangSchedulerClient),
    Vllm(VllmEngineClient),
}

impl GrpcClient {
    /// Get the runtime type of this client
    pub fn runtime_type(&self) -> crate::core::RuntimeType {
        match self {
            Self::Sglang(_) => crate::core::RuntimeType::Sglang,
            Self::Vllm(_) => crate::core::RuntimeType::Vllm,
        }
    }

    /// Generate with the appropriate client based on runtime type
    pub async fn generate(
        &self,
        request: proto::GenerateRequest,
    ) -> Result<GrpcStream, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            Self::Sglang(client) => {
                let stream = client.generate(request).await?;
                Ok(GrpcStream::Sglang(stream))
            }
            Self::Vllm(client) => {
                // Both use the same proto now, no conversion needed
                let stream = client.generate(request).await?;
                Ok(GrpcStream::Vllm(stream))
            }
        }
    }

    /// Abort a request
    pub async fn abort_request(
        &self,
        request_id: String,
        reason: String,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        match self {
            Self::Sglang(client) => client.abort_request(request_id, reason).await,
            Self::Vllm(client) => client.abort_request(request_id, reason).await,
        }
    }
}

/// Convert SGLang GenerateRequest to vLLM GenerateRequest
// ============================================================================
// Unified type exports for router code
// ============================================================================
// Router code should use these types instead of proto::* directly.
// Internally, SGLang proto is used as the canonical format, with automatic
// conversion to vLLM proto when dispatching to vLLM workers.

/// Unified GenerateRequest type (canonically SGLang format)
pub type Request = proto::GenerateRequest;

/// Unified GenerateComplete type (use CompleteData for runtime-agnostic access)
pub type Complete = proto::GenerateComplete;

/// Multimodal inputs (images, etc.)
pub type MultimodalInputs = proto::MultimodalInputs;

/// Output log probabilities
pub type OutputLogProbs = proto::OutputLogProbs;

/// Input log probabilities  
pub type InputLogProbs = proto::InputLogProbs;

/// Hidden states
pub type HiddenStates = proto::HiddenStates;
