use std::{convert::TryFrom, time::Duration};

use tonic::{transport::Channel, Request};
use tracing::debug;

use super::common::{create_grpc_channel, AbortOnDropStream, AbortableClient};
use crate::protocols::{
    chat::ChatCompletionRequest,
    common::{ResponseFormat, StringOrArray, ToolChoice, ToolChoiceValue},
    generate::GenerateRequest,
    responses::ResponsesRequest,
    sampling_params::SamplingParams as GenerateSamplingParams,
};

// Include the generated protobuf code
#[allow(clippy::all)]
pub mod proto {
    #![allow(clippy::all, unused_qualifications)]
    tonic::include_proto!("inference.grpc");
}

/// gRPC client for vLLM engine
#[derive(Clone)]
pub struct VllmEngineClient {
    client: proto::vllm_engine_client::VllmEngineClient<Channel>,
}

impl VllmEngineClient {
    /// Create a new client and connect to the engine
    pub async fn connect(endpoint: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        debug!("Connecting to vLLM engine at {}", endpoint);

        let channel = create_grpc_channel(endpoint).await?;
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
    ) -> Result<
        AbortOnDropStream<VllmEngineClient, proto::GenerateResponse>,
        Box<dyn std::error::Error + Send + Sync>,
    > {
        let request_id = req.request_id.clone();
        let mut client = self.client.clone();
        let request = Request::new(req);
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
        reason: String,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        debug!(
            "Sending abort request for {} (reason: {})",
            request_id, reason
        );
        let request = Request::new(proto::AbortRequest {
            request_id: request_id.clone(),
            reason,
        });

        let mut client = self.client.clone();
        let response = client.abort(request).await?;
        debug!(
            "Abort response for {}: success={}, message={}",
            request_id,
            response.get_ref().success,
            response.get_ref().message
        );
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
            tokenized: Some(proto::TokenizedInput {
                original_text: processed_text,
                input_ids: token_ids,
            }),
            sampling_params: Some(sampling_params),
            stream: body.stream,
            ..Default::default()
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
            tokenized: Some(proto::TokenizedInput {
                original_text: original_text.unwrap_or_default(),
                input_ids: token_ids,
            }),
            sampling_params: Some(sampling_params),
            stream: body.stream,
            ..Default::default()
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
            tokenized: Some(proto::TokenizedInput {
                original_text: processed_text,
                input_ids: token_ids,
            }),
            sampling_params: Some(sampling_params),
            stream: body.stream.unwrap_or(false),
            ..Default::default()
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

        let max_new_tokens = request.max_completion_tokens.map(|v| v as i32);

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
            temperature: request.temperature.unwrap_or(1.0),
            top_p: request.top_p.unwrap_or(1.0),
            top_k: request.top_k.unwrap_or(-1),
            min_p: request.min_p.unwrap_or(0.0),
            frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
            presence_penalty: request.presence_penalty.unwrap_or(0.0),
            repetition_penalty: request.repetition_penalty.unwrap_or(1.0),
            max_new_tokens,
            stop: stop_sequences,
            stop_token_ids: request.stop_token_ids.clone().unwrap_or_default(),
            skip_special_tokens,
            spaces_between_special_tokens: true, // Default from Python SamplingParams
            ignore_eos: request.ignore_eos,
            n: request.n.unwrap_or(1) as i32,
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
            constraints.push(proto::sampling_params::Constraint::EbnfGrammar(ebnf.clone()));
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
                "grammar" | "ebnf" => proto::sampling_params::Constraint::EbnfGrammar(constraint_value),
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

        let max_new_tokens = request.max_output_tokens.map(|v| v as i32);

        Ok(proto::SamplingParams {
            temperature: request.temperature.unwrap_or(1.0),
            top_p: request.top_p.unwrap_or(1.0),
            top_k: -1,               // ResponsesRequest doesn't expose top_k
            min_p: 0.0,              // ResponsesRequest doesn't expose min_p
            frequency_penalty: 0.0,  // ResponsesRequest doesn't expose frequency_penalty
            presence_penalty: 0.0,   // ResponsesRequest doesn't expose presence_penalty
            repetition_penalty: 1.0, // ResponsesRequest doesn't expose repetition_penalty
            max_new_tokens,
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
                "grammar" | "ebnf" => proto::sampling_params::Constraint::EbnfGrammar(constraint_value),
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
            constraints.push(proto::sampling_params::Constraint::EbnfGrammar(ebnf.clone()));
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
            temperature: 1.0,
            top_p: 1.0,
            top_k: -1,
            repetition_penalty: 1.0,
            n: 1,
            skip_special_tokens: true,
            spaces_between_special_tokens: true,
            ..Default::default()
        };

        let Some(p) = params else {
            return Ok(sampling);
        };

        // Simple field mappings using a macro
        macro_rules! map_field {
            ($field:ident) => {
                if let Some(val) = p.$field {
                    sampling.$field = val;
                }
            };
        }

        map_field!(temperature);
        map_field!(top_p);
        map_field!(top_k);
        map_field!(frequency_penalty);
        map_field!(presence_penalty);
        map_field!(repetition_penalty);
        map_field!(min_p);
        map_field!(ignore_eos);
        map_field!(skip_special_tokens);
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

        // Handle max_new_tokens with conversion (read from internal max_new_tokens)
        if let Some(max_new_tokens) = p.max_new_tokens {
            sampling.max_new_tokens = Some(
                i32::try_from(max_new_tokens)
                    .map_err(|_| "max_new_tokens must fit into a 32-bit signed integer".to_string())?,
            );
        }

        // Handle min_new_tokens with conversion (read from internal min_new_tokens)
        if let Some(min_new_tokens) = p.min_new_tokens {
            sampling.min_new_tokens = i32::try_from(min_new_tokens)
                .map_err(|_| "min_new_tokens must fit into a 32-bit signed integer".to_string())?;
        }

        // Handle n with conversion
        if let Some(n) = p.n {
            sampling.n = i32::try_from(n)
                .map_err(|_| "n must fit into a 32-bit signed integer".to_string())?;
        }

        // Handle constraints (exactly one allowed)
        sampling.constraint = Self::build_single_constraint_from_plain(p)?;

        Ok(sampling)
    }
}

// Implement AbortableClient trait for use with generic AbortOnDropStream
impl AbortableClient for VllmEngineClient {
    async fn abort_request(
        &self,
        request_id: String,
        reason: String,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Self::abort_request(self, request_id, reason).await
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
            temperature: 0.7,
            max_new_tokens: Some(128),
            top_p: 0.9,
            top_k: 50,
            stop: vec!["</s>".to_string()],
            ..Default::default()
        };

        let gen_req = proto::GenerateRequest {
            request_id: "test-req-123".to_string(),
            tokenized: Some(proto::TokenizedInput {
                original_text: "Hello world".to_string(),
                input_ids: vec![9906, 1917], // Mock token IDs for "Hello world"
            }),
            sampling_params: Some(sampling_params),
            stream: false,
        };

        assert_eq!(gen_req.request_id, "test-req-123");
        if let Some(ref tokenized) = &gen_req.tokenized {
            assert_eq!(tokenized.original_text, "Hello world");
        }
        // vLLM: logprobs are in SamplingParams, not GenerateRequest

        let params = gen_req.sampling_params.unwrap();
        assert_eq!(params.temperature, 0.7);
        assert_eq!(params.max_new_tokens, Some(128));
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
            request_id: "req-456".to_string(),
            reason: "User canceled".to_string(),
        };
        assert_eq!(abort_req.request_id, "req-456");
        assert_eq!(abort_req.reason, "User canceled");
    }

    #[test]
    fn test_sampling_params_defaults() {
        let params = proto::SamplingParams::default();
        // Numeric fields have proto defaults (0)
        assert_eq!(params.temperature, 0.0);
        assert_eq!(params.top_p, 0.0);
        assert_eq!(params.top_k, 0);
        assert_eq!(params.repetition_penalty, 0.0);
        assert_eq!(params.n, 0);
        // Bool fields have proto defaults (false)
        assert!(!params.skip_special_tokens);
        assert!(!params.spaces_between_special_tokens);
        assert!(!params.ignore_eos);
        assert!(!params.include_stop_str_in_output);
        // Optional int fields should be None
        assert_eq!(params.max_new_tokens, None);
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
