use std::time::Duration;
use tonic::{transport::Channel, Request};
use tracing::debug;

// Include the generated protobuf code
pub mod proto {
    tonic::include_proto!("sglang.grpc.scheduler");
}

// The generated module structure depends on the package name in the .proto file
// package sglang.grpc.scheduler; generates a nested module structure

/// gRPC client for SGLang scheduler
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

    /// Initialize the connection
    pub async fn initialize(
        &mut self,
        client_id: String,
    ) -> Result<proto::InitializeResponse, Box<dyn std::error::Error>> {
        let request = Request::new(proto::InitializeRequest {
            client_id,
            client_version: "0.1.0".to_string(),
            mode: proto::initialize_request::Mode::Regular as i32,
        });

        let response = self.client.initialize(request).await?;
        Ok(response.into_inner())
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
            include_detailed_metrics: false,
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

    /// Flush cache
    pub async fn flush_cache(
        &mut self,
        flush_all: bool,
        session_ids: &[String],
    ) -> Result<proto::FlushCacheResponse, Box<dyn std::error::Error>> {
        let request = Request::new(proto::FlushCacheRequest {
            flush_all,
            session_ids: session_ids.to_vec(),
        });

        let response = self.client.flush_cache(request).await?;
        Ok(response.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proto_types_compilation() {
        // Test that protobuf types can be constructed
        let init_req = proto::InitializeRequest {
            client_id: "test-client".to_string(),
            client_version: "0.1.0".to_string(),
            mode: 0,
        };
        assert_eq!(init_req.client_id, "test-client");
        assert_eq!(init_req.client_version, "0.1.0");
        assert_eq!(init_req.mode, 0);
    }

    #[test]
    fn test_generate_request_construction() {
        let sampling_params = proto::SamplingParams {
            temperature: 0.7,
            max_new_tokens: 128,
            top_p: 0.9,
            top_k: 50,
            stop: vec!["</s>".to_string()],
            ..Default::default()
        };

        let gen_req = proto::GenerateRequest {
            request_id: "test-req-123".to_string(),
            input: Some(proto::generate_request::Input::Text(
                "Hello world".to_string(),
            )),
            sampling_params: Some(sampling_params),
            return_logprob: true,
            logprob_start_len: 0,
            top_logprobs_num: 5,
            ..Default::default()
        };

        assert_eq!(gen_req.request_id, "test-req-123");
        if let Some(proto::generate_request::Input::Text(text)) = &gen_req.input {
            assert_eq!(text, "Hello world");
        }
        assert!(gen_req.return_logprob);
        assert_eq!(gen_req.top_logprobs_num, 5);

        let params = gen_req.sampling_params.unwrap();
        assert_eq!(params.temperature, 0.7);
        assert_eq!(params.max_new_tokens, 128);
        assert_eq!(params.stop, vec!["</s>"]);
    }

    #[test]
    fn test_health_check_request() {
        let health_req = proto::HealthCheckRequest {
            include_detailed_metrics: true,
        };
        assert!(health_req.include_detailed_metrics);
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
    fn test_flush_cache_request() {
        let flush_req = proto::FlushCacheRequest {
            flush_all: true,
            session_ids: vec!["session1".to_string(), "session2".to_string()],
        };
        assert!(flush_req.flush_all);
        assert_eq!(flush_req.session_ids.len(), 2);
        assert_eq!(flush_req.session_ids[0], "session1");
    }

    #[test]
    fn test_sampling_params_defaults() {
        let params = proto::SamplingParams::default();
        assert_eq!(params.temperature, 0.0);
        assert_eq!(params.max_new_tokens, 0);
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

    #[test]
    fn test_session_params() {
        let session_params = proto::SessionParams {
            session_id: "sess-789".to_string(),
            request_id: "req-101".to_string(),
            offset: 100,
            replace: true,
            drop_previous_output: false,
        };

        assert_eq!(session_params.session_id, "sess-789");
        assert_eq!(session_params.request_id, "req-101");
        assert_eq!(session_params.offset, 100);
        assert!(session_params.replace);
        assert!(!session_params.drop_previous_output);
    }

    #[test]
    fn test_embed_request() {
        let embed_req = proto::EmbedRequest {
            request_id: "embed-req-202".to_string(),
            input: Some(proto::embed_request::Input::Text(
                "This is a test sentence for embedding".to_string(),
            )),
            log_metrics: true,
            data_parallel_rank: 0,
            ..Default::default()
        };

        assert_eq!(embed_req.request_id, "embed-req-202");
        if let Some(proto::embed_request::Input::Text(text)) = &embed_req.input {
            assert_eq!(text, "This is a test sentence for embedding");
        }
        assert!(embed_req.log_metrics);
        assert_eq!(embed_req.data_parallel_rank, 0);
    }

    #[tokio::test]
    async fn test_client_connect_invalid_endpoint() {
        // Test connecting to an invalid endpoint should return error
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

    // Test response type construction
    #[test]
    fn test_generate_stream_chunk() {
        let chunk = proto::GenerateStreamChunk {
            token_id: 1234,
            text: " world".to_string(),
            prompt_tokens: 5,
            completion_tokens: 2,
            cached_tokens: 3,
            generation_time: 0.025,
            queue_time: 10,
            ..Default::default()
        };

        assert_eq!(chunk.token_id, 1234);
        assert_eq!(chunk.text, " world");
        assert_eq!(chunk.prompt_tokens, 5);
        assert_eq!(chunk.completion_tokens, 2);
        assert_eq!(chunk.cached_tokens, 3);
        assert_eq!(chunk.generation_time, 0.025);
        assert_eq!(chunk.queue_time, 10);
    }

    #[test]
    fn test_model_info() {
        let model_info = proto::ModelInfo {
            model_name: "Meta-Llama-3-8B-Instruct".to_string(),
            max_context_length: 8192,
            vocab_size: 128256,
            supports_tool_calling: true,
            supports_vision: false,
            special_tokens: vec![
                "<|begin_of_text|>".to_string(),
                "<|end_of_text|>".to_string(),
            ],
            model_type: "llama".to_string(),
            num_layers: 32,
            hidden_size: 4096,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            tokenizer_type: "llama".to_string(),
            eos_token_ids: vec![128001, 128009],
            pad_token_id: 128001,
            bos_token_id: 128000,
        };

        assert_eq!(model_info.model_name, "Meta-Llama-3-8B-Instruct");
        assert_eq!(model_info.max_context_length, 8192);
        assert_eq!(model_info.vocab_size, 128256);
        assert!(model_info.supports_tool_calling);
        assert!(!model_info.supports_vision);
        assert_eq!(model_info.special_tokens.len(), 2);
        assert_eq!(model_info.num_layers, 32);
        assert_eq!(model_info.eos_token_ids, vec![128001, 128009]);
    }
}
