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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proto_types_compilation() {
        // Test that protobuf types can be constructed
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
    // #[test]
    // fn test_session_params() { ... }

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

    // TODO: ModelInfo not in current proto - skip test
    // #[test]
    // fn test_model_info() { ... }
}
