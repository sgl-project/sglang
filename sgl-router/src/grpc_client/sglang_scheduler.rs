use std::time::Duration;
use tonic::{transport::Channel, Request};
use tracing::debug;

use crate::protocols::spec::{ChatCompletionRequest, ResponseFormat};

// Include the generated protobuf code
pub mod proto {
    tonic::include_proto!("sglang.grpc.scheduler");
}

// The generated module structure depends on the package name in the .proto file
// package sglang.grpc.scheduler; generates a nested module structure

/// gRPC client for SGLang scheduler
#[derive(Clone)]
pub struct SglangSchedulerClient {
    client: proto::sglang_scheduler_client::SglangSchedulerClient<Channel>,
}

impl SglangSchedulerClient {
    /// Create a new client and connect to the scheduler
    pub async fn connect(endpoint: &str) -> Result<Self, Box<dyn std::error::Error>> {
        debug!("Connecting to SGLang scheduler at {}", endpoint);

        // Convert grpc:// to http:// for tonic
        let http_endpoint = if endpoint.starts_with("grpc://") {
            endpoint.replace("grpc://", "http://")
        } else {
            endpoint.to_string()
        };

        let channel = Channel::from_shared(http_endpoint)?
            .timeout(Duration::from_secs(30))
            .connect()
            .await?;

        let client = proto::sglang_scheduler_client::SglangSchedulerClient::new(channel);

        Ok(Self { client })
    }

    /// Submit a generation request (returns streaming response)
    pub async fn generate_stream(
        &mut self,
        req: proto::GenerateRequest,
    ) -> Result<tonic::Streaming<proto::GenerateResponse>, Box<dyn std::error::Error>> {
        let request = Request::new(req);
        let response = self.client.generate(request).await?;
        Ok(response.into_inner())
    }

    /// Perform health check
    pub async fn health_check(
        &mut self,
    ) -> Result<proto::HealthCheckResponse, Box<dyn std::error::Error>> {
        debug!("Sending health check request");
        let request = Request::new(proto::HealthCheckRequest {
            tokenized: Some(proto::TokenizedInput {
                original_text: "Hello".to_string(),
                input_ids: vec![9906], // Mock token ID for "Hello"
            }),
        });

        let response = self.client.health_check(request).await?;
        debug!("Health check response received");
        Ok(response.into_inner())
    }

    /// Abort a request
    pub async fn abort_request(
        &mut self,
        request_id: String,
        reason: String,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let request = Request::new(proto::AbortRequest { request_id, reason });

        self.client.abort(request).await?;
        Ok(())
    }

    /// Build a single SGLang GenerateRequest from OpenAI ChatCompletionRequest
    pub fn build_generate_request(
        &self,
        request_id: String,
        body: &ChatCompletionRequest,
        processed_text: String,
        token_ids: Vec<i32>,
        multimodal_inputs: Option<proto::MultimodalInputs>,
        tool_call_constraint: Option<(String, String)>, // (constraint_type, constraint_value)
    ) -> Result<proto::GenerateRequest, String> {
        // Build sampling params
        let sampling_params = self.build_grpc_sampling_params(body, tool_call_constraint)?;

        let grpc_request = proto::GenerateRequest {
            request_id,
            tokenized: Some(proto::TokenizedInput {
                original_text: processed_text,
                input_ids: token_ids,
            }),
            mm_inputs: multimodal_inputs,
            sampling_params: Some(sampling_params),
            return_logprob: body.logprobs,
            logprob_start_len: -1,
            top_logprobs_num: body.top_logprobs.unwrap_or(0) as i32,
            return_hidden_states: body.return_hidden_states,
            stream: body.stream,
            ..Default::default()
        };

        Ok(grpc_request)
    }

    /// Build gRPC SamplingParams from OpenAI request
    fn build_grpc_sampling_params(
        &self,
        request: &ChatCompletionRequest,
        tool_call_constraint: Option<(String, String)>,
    ) -> Result<proto::SamplingParams, String> {
        let stop_sequences = self.extract_stop_strings(request);

        // Handle max tokens: prefer max_completion_tokens (new) over max_tokens (deprecated)
        // If neither is specified, use None to let the backend decide the default
        #[allow(deprecated)]
        let max_new_tokens = request
            .max_completion_tokens
            .or(request.max_tokens)
            .map(|v| v as i32);

        // Handle skip_special_tokens: set to false if tools are present and tool_choice is not "none"
        let skip_special_tokens = if request.tools.is_some() {
            match &request.tool_choice {
                Some(crate::protocols::spec::ToolChoice::Value(
                    crate::protocols::spec::ToolChoiceValue::None,
                )) => request.skip_special_tokens,
                Some(_) => false, // tool_choice is not "none"
                None => false, // TODO: this assumes tool_choice defaults to "auto" when tools present
            }
        } else {
            request.skip_special_tokens
        };

        #[allow(deprecated)]
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
            n: request.n.unwrap_or(1) as i32,
            constraint: self.build_constraint(request, tool_call_constraint)?,
            ..Default::default()
        })
    }

    /// Extract stop strings from request
    fn extract_stop_strings(&self, request: &ChatCompletionRequest) -> Vec<String> {
        match &request.stop {
            Some(crate::protocols::spec::StringOrArray::String(s)) => vec![s.clone()],
            Some(crate::protocols::spec::StringOrArray::Array(arr)) => arr.clone(),
            None => vec![],
        }
    }

    /// Build constraint for structured generation
    fn build_constraint(
        &self,
        request: &ChatCompletionRequest,
        tool_call_constraint: Option<(String, String)>,
    ) -> Result<Option<proto::sampling_params::Constraint>, String> {
        let mut constraints = Vec::new();

        if let Some(ResponseFormat::JsonSchema { json_schema }) = &request.response_format {
            let schema_str = serde_json::to_string(&json_schema.schema)
                .map_err(|e| format!("Failed to serialize JSON schema: {}", e))?;
            constraints.push(proto::sampling_params::Constraint::JsonSchema(schema_str));
        }

        if let Some(ebnf) = &request.ebnf {
            constraints.push(proto::sampling_params::Constraint::EbnfGrammar(
                ebnf.clone(),
            ));
        }

        if let Some(regex) = &request.regex {
            constraints.push(proto::sampling_params::Constraint::Regex(regex.clone()));
        }

        // Handle tool call constraint
        if let Some((constraint_type, constraint_value)) = tool_call_constraint {
            if !constraints.is_empty() {
                return Err("Constrained decoding is not compatible with tool calls.".to_string());
            }
            let tool_constraint = match constraint_type.as_str() {
                "structural_tag" => {
                    proto::sampling_params::Constraint::StructuralTag(constraint_value)
                }
                "json_schema" => proto::sampling_params::Constraint::JsonSchema(constraint_value),
                "ebnf" => proto::sampling_params::Constraint::EbnfGrammar(constraint_value),
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proto_types_compilation() {
        let health_req = proto::HealthCheckRequest {
            tokenized: Some(proto::TokenizedInput {
                original_text: "test".to_string(),
                input_ids: vec![1296],
            }),
        };
        assert!(health_req.tokenized.is_some());
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
            return_logprob: true,
            logprob_start_len: 0,
            top_logprobs_num: 5,
            ..Default::default()
        };

        assert_eq!(gen_req.request_id, "test-req-123");
        if let Some(ref tokenized) = &gen_req.tokenized {
            assert_eq!(tokenized.original_text, "Hello world");
        }
        assert!(gen_req.return_logprob);
        assert_eq!(gen_req.top_logprobs_num, 5);

        let params = gen_req.sampling_params.unwrap();
        assert_eq!(params.temperature, 0.7);
        assert_eq!(params.max_new_tokens, Some(128));
        assert_eq!(params.stop, vec!["</s>"]);
    }

    #[test]
    fn test_health_check_request() {
        let health_req = proto::HealthCheckRequest {
            tokenized: Some(proto::TokenizedInput {
                original_text: "test".to_string(),
                input_ids: vec![1296], // Mock token ID for "test"
            }),
        };
        assert!(health_req.tokenized.is_some());
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
        assert_eq!(params.temperature, 0.0);
        assert_eq!(params.max_new_tokens, None);
        assert_eq!(params.top_p, 0.0);
        assert_eq!(params.top_k, 0);
        assert!(params.stop.is_empty());
    }

    #[test]
    fn test_multimodal_inputs() {
        let mm_inputs = proto::MultimodalInputs {
            image_urls: vec!["http://example.com/image.jpg".to_string()],
            video_urls: vec![],
            audio_urls: vec![],
            image_data: vec![],
            video_data: vec![],
            audio_data: vec![],
            modalities: vec!["image".to_string()],
            ..Default::default()
        };

        assert_eq!(mm_inputs.image_urls.len(), 1);
        assert_eq!(mm_inputs.image_urls[0], "http://example.com/image.jpg");
        assert_eq!(mm_inputs.modalities[0], "image");
    }

    // TODO: SessionParams not in current proto - skip test

    #[test]
    fn test_embed_request() {
        let embed_req = proto::EmbedRequest {
            request_id: "embed-req-202".to_string(),
            tokenized: Some(proto::TokenizedInput {
                original_text: "This is a test sentence for embedding".to_string(),
                input_ids: vec![2028, 374, 264, 1296, 11914, 369, 28537], // Mock token IDs
            }),
            log_metrics: true,
            data_parallel_rank: 0,
            ..Default::default()
        };

        assert_eq!(embed_req.request_id, "embed-req-202");
        if let Some(ref tokenized) = &embed_req.tokenized {
            assert_eq!(
                tokenized.original_text,
                "This is a test sentence for embedding"
            );
        }
        assert!(embed_req.log_metrics);
        assert_eq!(embed_req.data_parallel_rank, 0);
    }

    #[tokio::test]
    async fn test_client_connect_invalid_endpoint() {
        let result = SglangSchedulerClient::connect("invalid://endpoint").await;
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
            ..Default::default()
        };

        assert_eq!(chunk.token_ids, vec![1234, 5678]);
        assert_eq!(chunk.prompt_tokens, 5);
        assert_eq!(chunk.completion_tokens, 2);
        assert_eq!(chunk.cached_tokens, 3);
    }

    // TODO: ModelInfo not in current proto - skip test
}
