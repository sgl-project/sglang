use std::{
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    task::{Context, Poll},
    time::Duration,
};

use tonic::{transport::Channel, Request, Streaming};
use tracing::{debug, warn};

use crate::{
    observability::otel_trace::inject_trace_context_grpc,
    protocols::{
        chat::ChatCompletionRequest,
        common::{ResponseFormat, StringOrArray, ToolChoice, ToolChoiceValue},
        generate::GenerateRequest,
        responses::ResponsesRequest,
        sampling_params::SamplingParams as GenerateSamplingParams,
    },
};

// Include the generated protobuf code
#[allow(clippy::all)]
pub mod proto {
    #![allow(clippy::all, unused_qualifications)]
    tonic::include_proto!("vllm.grpc.engine");
}

// The generated module structure depends on the package name in the .proto file
// package vllm.grpc.engine; generates a nested module structure

/// A smart wrapper around Streaming<GenerateResponse> that automatically
/// sends abort when dropped (e.g., due to client disconnection or early termination).
///
/// This leverages Rust's RAII pattern to ensure cleanup happens automatically,
/// regardless of how the stream is dropped (panic, early return, client disconnect, etc.).
pub struct AbortOnDropStream {
    inner: Streaming<proto::GenerateResponse>,
    request_id: String,
    client: VllmEngineClient,
    aborted: Arc<AtomicBool>,
}

impl AbortOnDropStream {
    /// Create a new auto-aborting stream wrapper
    pub fn new(
        stream: Streaming<proto::GenerateResponse>,
        request_id: String,
        client: VllmEngineClient,
    ) -> Self {
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

impl Drop for AbortOnDropStream {
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
                    "Failed to send abort on drop for request {}: {}",
                    request_id_for_log, e
                );
            }
        });
    }
}

// Implement Stream trait to make AbortOnDropStream work like the original Streaming
impl futures::Stream for AbortOnDropStream {
    type Item = Result<proto::GenerateResponse, tonic::Status>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Delegate to the inner stream
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

/// gRPC client for vLLM scheduler
#[derive(Clone)]
pub struct VllmEngineClient {
    client: proto::vllm_engine_client::VllmEngineClient<Channel>,
}

impl VllmEngineClient {
    /// Create a new client and connect to the vLLM server
    pub async fn connect(endpoint: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        debug!("Connecting to vLLM gRPC server at {}", endpoint);

        // Convert grpc:// to http:// for tonic
        let http_endpoint = if let Some(addr) = endpoint.strip_prefix("grpc://") {
            format!("http://{}", addr)
        } else {
            endpoint.to_string()
        };

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

        let client = proto::vllm_engine_client::VllmEngineClient::new(channel);

        Ok(Self { client })
    }

    /// Submit a generation request (returns auto-aborting streaming response)
    ///
    /// The returned stream automatically sends an abort request when dropped,
    /// ensuring proper cleanup even if the HTTP client disconnects or an error occurs.
    /// Call `mark_completed()` on the stream after successful completion to prevent
    /// unnecessary abort RPCs.
    pub async fn generate(
        &self,
        req: proto::GenerateRequest,
    ) -> Result<AbortOnDropStream, Box<dyn std::error::Error + Send + Sync>> {
        let request_id = req.request_id.clone();
        let mut client = self.client.clone();
        let mut request = Request::new(req);

        // Inject W3C trace context into gRPC metadata for distributed tracing
        inject_trace_context_grpc(request.metadata_mut());

        let response = client.generate(request).await?;

        Ok(AbortOnDropStream::new(
            response.into_inner(),
            request_id,
            self.clone(),
        ))
    }

    /// Perform health check
    pub async fn health_check(
        &self,
    ) -> Result<proto::HealthCheckResponse, Box<dyn std::error::Error + Send + Sync>> {
        debug!("Sending health check request");
        // HealthCheckRequest is now empty - server generates its own health check internally
        let request = Request::new(proto::HealthCheckRequest {});

        let mut client = self.client.clone();
        let response = client.health_check(request).await?;
        debug!("Health check response received");
        Ok(response.into_inner())
    }

    /// Abort a request
    pub async fn abort_request(
        &self,
        request_id: String,
        _reason: String,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        debug!("Sending abort request for {}", request_id);
        let request = Request::new(proto::AbortRequest {
            request_ids: vec![request_id.clone()],
        });

        let mut client = self.client.clone();
        let _response = client.abort(request).await?;
        debug!("Abort response received for {}", request_id);
        Ok(())
    }

    /// Get model information
    pub async fn get_model_info(
        &self,
    ) -> Result<proto::GetModelInfoResponse, Box<dyn std::error::Error + Send + Sync>> {
        debug!("Requesting model info");
        let request = Request::new(proto::GetModelInfoRequest {});

        let mut client = self.client.clone();
        let response = client.get_model_info(request).await?;
        debug!("Model info response received");
        Ok(response.into_inner())
    }

    /// Get server information
    pub async fn get_server_info(
        &self,
    ) -> Result<proto::GetServerInfoResponse, Box<dyn std::error::Error + Send + Sync>> {
        debug!("Requesting server info");
        let request = Request::new(proto::GetServerInfoRequest {});

        let mut client = self.client.clone();
        let response = client.get_server_info(request).await?;
        debug!("Server info response received");
        Ok(response.into_inner())
    }

    /// Build a single vLLM GenerateRequest from OpenAI ChatCompletionRequest
    pub fn build_generate_request_from_chat(
        &self,
        request_id: String,
        body: &ChatCompletionRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        tool_call_constraint: Option<(String, String)>, // (constraint_type, constraint_value)
    ) -> Result<proto::GenerateRequest, String> {
        // Build sampling params
        let sampling_params =
            self.build_grpc_sampling_params_from_chat(body, tool_call_constraint)?;

        let grpc_request = proto::GenerateRequest {
            request_id,
            input: Some(proto::generate_request::Input::Tokenized(
                proto::TokenizedInput {
                    original_text: processed_text,
                    input_ids: token_ids,
                },
            )),
            sampling_params: Some(sampling_params),
            stream: body.stream,
        };

        Ok(grpc_request)
    }

    /// Build a basic GenerateRequest from the vLLM spec GenerateRequest
    pub fn build_plain_generate_request(
        &self,
        request_id: String,
        body: &GenerateRequest,
        original_text: Option<String>,
        token_ids: Vec<u32>,
    ) -> Result<proto::GenerateRequest, String> {
        let sampling_params =
            Self::build_sampling_params_from_plain(body.sampling_params.as_ref())?;

        let grpc_request = proto::GenerateRequest {
            request_id,
            input: Some(proto::generate_request::Input::Tokenized(
                proto::TokenizedInput {
                    original_text: original_text.unwrap_or_default(),
                    input_ids: token_ids,
                },
            )),
            sampling_params: Some(sampling_params),
            stream: body.stream,
        };

        Ok(grpc_request)
    }

    /// Build a GenerateRequest from ResponsesRequest (OpenAI Responses API)
    ///
    /// NOTE: This is used by the Harmony router only. The Regular router uses
    /// responses_to_chat() conversion and goes through the chat pipeline.
    pub fn build_generate_request_from_responses(
        &self,
        request_id: String,
        body: &ResponsesRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        harmony_stop_ids: Option<Vec<u32>>,
        constraint: Option<(String, String)>,
    ) -> Result<proto::GenerateRequest, String> {
        // Build sampling params from ResponsesRequest
        let mut sampling_params =
            self.build_grpc_sampling_params_from_responses(body, constraint)?;

        // Inject Harmony stop token IDs if provided
        if let Some(stop_ids) = harmony_stop_ids {
            sampling_params.stop_token_ids = stop_ids;
        }

        let grpc_request = proto::GenerateRequest {
            request_id,
            input: Some(proto::generate_request::Input::Tokenized(
                proto::TokenizedInput {
                    original_text: processed_text,
                    input_ids: token_ids,
                },
            )),
            sampling_params: Some(sampling_params),
            stream: body.stream.unwrap_or(false),
        };

        Ok(grpc_request)
    }

    /// Build gRPC SamplingParams from ChatCompletionRequest
    fn build_grpc_sampling_params_from_chat(
        &self,
        request: &ChatCompletionRequest,
        tool_call_constraint: Option<(String, String)>,
    ) -> Result<proto::SamplingParams, String> {
        let stop_sequences = self.extract_stop_strings(request);

        let max_tokens = request.max_completion_tokens;

        // Handle skip_special_tokens: set to false if tools are present and tool_choice is not "none"
        let skip_special_tokens = if request.tools.is_some() {
            match &request.tool_choice {
                Some(ToolChoice::Value(ToolChoiceValue::None)) => request.skip_special_tokens,
                Some(_) => false, // tool_choice is not "none"
                None => false, // TODO: this assumes tool_choice defaults to "auto" when tools present
            }
        } else {
            request.skip_special_tokens
        };

        Ok(proto::SamplingParams {
            temperature: request.temperature,
            top_p: request.top_p.unwrap_or(1.0),
            top_k: request.top_k.map(|v| v.max(0) as u32).unwrap_or(0), // 0 means disabled in vLLM
            min_p: request.min_p.unwrap_or(0.0),
            frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
            presence_penalty: request.presence_penalty.unwrap_or(0.0),
            repetition_penalty: request.repetition_penalty.unwrap_or(1.0),
            max_tokens,
            stop: stop_sequences,
            stop_token_ids: request.stop_token_ids.clone().unwrap_or_default(),
            skip_special_tokens,
            spaces_between_special_tokens: true, // Default from Python SamplingParams
            ignore_eos: request.ignore_eos,
            n: request.n.unwrap_or(1),
            constraint: self.build_constraint_for_chat(request, tool_call_constraint)?,
            ..Default::default()
        })
    }

    /// Extract stop strings from request
    fn extract_stop_strings(&self, request: &ChatCompletionRequest) -> Vec<String> {
        match &request.stop {
            Some(StringOrArray::String(s)) => vec![s.clone()],
            Some(StringOrArray::Array(arr)) => arr.clone(),
            None => vec![],
        }
    }

    /// Build constraint for structured generation
    fn build_constraint_for_chat(
        &self,
        request: &ChatCompletionRequest,
        tool_call_constraint: Option<(String, String)>,
    ) -> Result<Option<proto::sampling_params::Constraint>, String> {
        let mut constraints = Vec::new();

        // Handle response_format constraints
        match &request.response_format {
            Some(ResponseFormat::JsonObject) => {
                // json_object mode - constrain to valid JSON object
                let schema = serde_json::json!({"type": "object"});
                let schema_str = serde_json::to_string(&schema)
                    .map_err(|e| format!("Failed to serialize JSON schema: {}", e))?;
                constraints.push(proto::sampling_params::Constraint::JsonSchema(schema_str));
            }
            Some(ResponseFormat::JsonSchema { json_schema }) => {
                let schema_str = serde_json::to_string(&json_schema.schema)
                    .map_err(|e| format!("Failed to serialize JSON schema: {}", e))?;
                constraints.push(proto::sampling_params::Constraint::JsonSchema(schema_str));
            }
            Some(ResponseFormat::Text) | None => {
                // No constraint for text format
            }
        }

        // vLLM supports: json_schema, regex, grammar, structural_tag, json_object, choice
        if let Some(ebnf) = &request.ebnf {
            constraints.push(proto::sampling_params::Constraint::Grammar(ebnf.clone()));
        }

        if let Some(regex) = &request.regex {
            constraints.push(proto::sampling_params::Constraint::Regex(regex.clone()));
        }

        // Handle tool call constraint from preparation stage
        if let Some((constraint_type, constraint_value)) = tool_call_constraint {
            if !constraints.is_empty() {
                return Err("Constrained decoding is not compatible with tool calls.".to_string());
            }
            let tool_constraint = match constraint_type.as_str() {
                "structural_tag" => {
                    proto::sampling_params::Constraint::StructuralTag(constraint_value)
                }
                "json_schema" => proto::sampling_params::Constraint::JsonSchema(constraint_value),
                "grammar" | "ebnf" => proto::sampling_params::Constraint::Grammar(constraint_value),
                "regex" => proto::sampling_params::Constraint::Regex(constraint_value),
                _ => return Err(format!("Unknown constraint type: {}", constraint_type)),
            };
            constraints.push(tool_constraint);
        }

        match constraints.len() {
            0 => Ok(None),
            1 => Ok(constraints.pop()),
            _ => Err("Multiple constraints are not allowed.".to_string()),
        }
    }

    /// Build gRPC SamplingParams from ResponsesRequest
    fn build_grpc_sampling_params_from_responses(
        &self,
        request: &ResponsesRequest,
        constraint: Option<(String, String)>,
    ) -> Result<proto::SamplingParams, String> {
        // Used by Harmony models only. Regular models use Chat API path.
        // Constraints come from Harmony preparation stage (structural_tag) or tool handling.

        let max_tokens = request.max_output_tokens;

        Ok(proto::SamplingParams {
            temperature: request.temperature,
            top_p: request.top_p.unwrap_or(1.0),
            top_k: 0,   // ResponsesRequest doesn't expose top_k (0 means disabled)
            min_p: 0.0, // ResponsesRequest doesn't expose min_p
            frequency_penalty: 0.0, // ResponsesRequest doesn't expose frequency_penalty
            presence_penalty: 0.0, // ResponsesRequest doesn't expose presence_penalty
            repetition_penalty: 1.0, // ResponsesRequest doesn't expose repetition_penalty
            max_tokens,
            stop: vec![],               // No stop sequences in Responses API
            stop_token_ids: vec![],     // Handled by Harmony stop tokens
            skip_special_tokens: false, // Keep special tokens for Harmony
            spaces_between_special_tokens: true,
            ignore_eos: false,
            n: 1, // Responses API doesn't support n>1
            constraint: self.build_constraint_for_responses(constraint)?,
            ..Default::default()
        })
    }

    /// Build constraint for Responses API
    ///
    /// Handles constraints from Harmony preparation stage (structural_tag for Harmony models,
    /// structured output via text field, or tool call constraints).
    ///
    /// Note: Regular gRPC models use Chat API path with response_format, not this function.
    fn build_constraint_for_responses(
        &self,
        constraint: Option<(String, String)>,
    ) -> Result<Option<proto::sampling_params::Constraint>, String> {
        if let Some((constraint_type, constraint_value)) = constraint {
            let parsed_constraint = match constraint_type.as_str() {
                "structural_tag" => {
                    proto::sampling_params::Constraint::StructuralTag(constraint_value)
                }
                "json_schema" => proto::sampling_params::Constraint::JsonSchema(constraint_value),
                "grammar" | "ebnf" => proto::sampling_params::Constraint::Grammar(constraint_value),
                "regex" => proto::sampling_params::Constraint::Regex(constraint_value),
                _ => return Err(format!("Unknown constraint type: {}", constraint_type)),
            };
            Ok(Some(parsed_constraint))
        } else {
            Ok(None)
        }
    }

    fn build_single_constraint_from_plain(
        params: &GenerateSamplingParams,
    ) -> Result<Option<proto::sampling_params::Constraint>, String> {
        let mut constraints = Vec::new();
        if let Some(json_schema) = &params.json_schema {
            constraints.push(proto::sampling_params::Constraint::JsonSchema(
                json_schema.clone(),
            ));
        }
        if let Some(regex) = &params.regex {
            constraints.push(proto::sampling_params::Constraint::Regex(regex.clone()));
        }
        if let Some(ebnf) = &params.ebnf {
            constraints.push(proto::sampling_params::Constraint::Grammar(ebnf.clone()));
        }

        match constraints.len() {
            0 => Ok(None),
            1 => Ok(constraints.pop()),
            _ => Err("Multiple structured constraints are not allowed".to_string()),
        }
    }

    fn build_sampling_params_from_plain(
        params: Option<&GenerateSamplingParams>,
    ) -> Result<proto::SamplingParams, String> {
        let mut sampling = proto::SamplingParams {
            temperature: Some(1.0),
            top_p: 1.0,
            top_k: 0, // 0 means disabled in vLLM
            repetition_penalty: 1.0,
            n: 1,
            skip_special_tokens: true,
            spaces_between_special_tokens: true,
            ..Default::default()
        };

        let Some(p) = params else {
            return Ok(sampling);
        };

        // Handle temperature (now optional)
        if let Some(val) = p.temperature {
            sampling.temperature = Some(val);
        }

        // Simple field mappings
        if let Some(val) = p.top_p {
            sampling.top_p = val;
        }
        if let Some(val) = p.top_k {
            sampling.top_k = val.max(0) as u32; // Clamp negative values to 0 (disabled)
        }
        if let Some(val) = p.frequency_penalty {
            sampling.frequency_penalty = val;
        }
        if let Some(val) = p.presence_penalty {
            sampling.presence_penalty = val;
        }
        if let Some(val) = p.repetition_penalty {
            sampling.repetition_penalty = val;
        }
        if let Some(val) = p.min_p {
            sampling.min_p = val;
        }
        if let Some(val) = p.ignore_eos {
            sampling.ignore_eos = val;
        }
        if let Some(val) = p.skip_special_tokens {
            sampling.skip_special_tokens = val;
        }
        // Note: no_stop_trim not supported in vLLM

        // Handle stop sequences
        if let Some(stop) = &p.stop {
            match stop {
                StringOrArray::String(s) => sampling.stop.push(s.clone()),
                StringOrArray::Array(arr) => sampling.stop.extend(arr.clone()),
            }
        }

        // Handle stop token IDs
        if let Some(stop_token_ids) = &p.stop_token_ids {
            sampling.stop_token_ids = stop_token_ids.clone();
        }

        // Handle max_tokens (read from internal max_new_tokens)
        if let Some(max_new_tokens) = p.max_new_tokens {
            sampling.max_tokens = Some(max_new_tokens);
        }

        // Handle min_tokens (read from internal min_new_tokens)
        if let Some(min_new_tokens) = p.min_new_tokens {
            sampling.min_tokens = min_new_tokens;
        }

        // Handle n
        if let Some(n) = p.n {
            sampling.n = n;
        }

        // Handle constraints (exactly one allowed)
        sampling.constraint = Self::build_single_constraint_from_plain(p)?;

        Ok(sampling)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proto_types_compilation() {
        let _health_req = proto::HealthCheckRequest {};
        // HealthCheckRequest is now empty - no fields to test
    }

    #[test]
    fn test_generate_request_construction() {
        let sampling_params = proto::SamplingParams {
            temperature: Some(0.7),
            max_tokens: Some(128),
            top_p: 0.9,
            top_k: 50,
            stop: vec!["</s>".to_string()],
            ..Default::default()
        };

        let gen_req = proto::GenerateRequest {
            request_id: "test-req-123".to_string(),
            input: Some(proto::generate_request::Input::Tokenized(
                proto::TokenizedInput {
                    original_text: "Hello world".to_string(),
                    input_ids: vec![9906, 1917], // Mock token IDs for "Hello world"
                },
            )),
            sampling_params: Some(sampling_params),
            stream: false,
        };

        assert_eq!(gen_req.request_id, "test-req-123");
        if let Some(proto::generate_request::Input::Tokenized(ref tokenized)) = gen_req.input {
            assert_eq!(tokenized.original_text, "Hello world");
        }
        // vLLM: logprobs are in SamplingParams, not GenerateRequest

        let params = gen_req.sampling_params.unwrap();
        assert_eq!(params.temperature, Some(0.7));
        assert_eq!(params.max_tokens, Some(128));
        assert_eq!(params.stop, vec!["</s>"]);
    }

    #[test]
    fn test_health_check_request() {
        let _health_req = proto::HealthCheckRequest {};
        // HealthCheckRequest is now empty - server generates its own test internally
    }

    #[test]
    fn test_abort_request_construction() {
        let abort_req = proto::AbortRequest {
            request_ids: vec!["req-456".to_string(), "req-789".to_string()],
        };
        assert_eq!(abort_req.request_ids, vec!["req-456", "req-789"]);
    }

    #[test]
    fn test_sampling_params_defaults() {
        let params = proto::SamplingParams::default();
        // Optional float field defaults to None
        assert_eq!(params.temperature, None);
        // Non-optional numeric fields have proto defaults (0)
        assert_eq!(params.top_p, 0.0);
        assert_eq!(params.top_k, 0);
        assert_eq!(params.repetition_penalty, 0.0);
        assert_eq!(params.n, 0);
        // Bool fields have proto defaults (false)
        assert!(!params.skip_special_tokens);
        assert!(!params.spaces_between_special_tokens);
        assert!(!params.ignore_eos);
        assert!(!params.include_stop_str_in_output);
        // Optional fields should be None
        assert_eq!(params.max_tokens, None);
        assert_eq!(params.logprobs, None);
        // Other non-optional fields
        assert_eq!(params.min_p, 0.0);
        assert_eq!(params.frequency_penalty, 0.0);
        assert_eq!(params.presence_penalty, 0.0);
        assert!(params.stop.is_empty());
    }

    // TODO: MultimodalInputs not in vLLM proto - skip test
    // vLLM handles multimodal inputs differently than SGLang

    // TODO: SessionParams not in current proto - skip test

    #[test]
    fn test_embed_request() {
        let embed_req = proto::EmbedRequest {
            request_id: "embed-req-202".to_string(),
            tokenized: Some(proto::TokenizedInput {
                original_text: "This is a test sentence for embedding".to_string(),
                input_ids: vec![2028, 374, 264, 1296, 11914, 369, 28537], // Mock token IDs
            }),
        };

        assert_eq!(embed_req.request_id, "embed-req-202");
        if let Some(ref tokenized) = &embed_req.tokenized {
            assert_eq!(
                tokenized.original_text,
                "This is a test sentence for embedding"
            );
        }
        // vLLM: no data_parallel_rank or log_metrics in EmbedRequest
    }

    #[tokio::test]
    async fn test_client_connect_invalid_endpoint() {
        let result = VllmEngineClient::connect("invalid://endpoint").await;
        assert!(result.is_err());
    }

    #[test]
    fn test_tokenized_input() {
        let tokenized = proto::TokenizedInput {
            original_text: "Hello world".to_string(),
            input_ids: vec![1, 15043, 1917, 2],
        };

        assert_eq!(tokenized.original_text, "Hello world");
        assert_eq!(tokenized.input_ids, vec![1, 15043, 1917, 2]);
    }

    #[test]
    fn test_generate_stream_chunk() {
        let chunk = proto::GenerateStreamChunk {
            token_ids: vec![1234, 5678],
            prompt_tokens: 5,
            completion_tokens: 2,
            cached_tokens: 3,
        };

        assert_eq!(chunk.token_ids, vec![1234, 5678]);
        assert_eq!(chunk.prompt_tokens, 5);
        assert_eq!(chunk.completion_tokens, 2);
        assert_eq!(chunk.cached_tokens, 3);
    }

    // TODO: ModelInfo not in current proto - skip test
}
