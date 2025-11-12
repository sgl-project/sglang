//! Unified gRPC client wrapper for SGLang and vLLM backends

use crate::{
    grpc_client::{SglangSchedulerClient, VllmEngineClient},
    routers::grpc::proto_wrapper::{ProtoGenerateRequest, ProtoStream},
};

/// Polymorphic gRPC client that wraps either SGLang or vLLM
#[derive(Clone)]
pub enum GrpcClient {
    Sglang(SglangSchedulerClient),
    Vllm(VllmEngineClient),
}

impl GrpcClient {
    /// Get reference to SGLang client (panics if vLLM)
    pub fn as_sglang(&self) -> &SglangSchedulerClient {
        match self {
            Self::Sglang(client) => client,
            Self::Vllm(_) => panic!("Expected SGLang client, got vLLM"),
        }
    }

    /// Get mutable reference to SGLang client (panics if vLLM)
    pub fn as_sglang_mut(&mut self) -> &mut SglangSchedulerClient {
        match self {
            Self::Sglang(client) => client,
            Self::Vllm(_) => panic!("Expected SGLang client, got vLLM"),
        }
    }

    /// Get reference to vLLM client (panics if SGLang)
    pub fn as_vllm(&self) -> &VllmEngineClient {
        match self {
            Self::Vllm(client) => client,
            Self::Sglang(_) => panic!("Expected vLLM client, got SGLang"),
        }
    }

    /// Get mutable reference to vLLM client (panics if SGLang)
    pub fn as_vllm_mut(&mut self) -> &mut VllmEngineClient {
        match self {
            Self::Vllm(client) => client,
            Self::Sglang(_) => panic!("Expected vLLM client, got SGLang"),
        }
    }

    /// Check if this is a SGLang client
    pub fn is_sglang(&self) -> bool {
        matches!(self, Self::Sglang(_))
    }

    /// Check if this is a vLLM client
    pub fn is_vllm(&self) -> bool {
        matches!(self, Self::Vllm(_))
    }

    /// Generate streaming response from request
    ///
    /// Dispatches to the appropriate backend client and wraps the result in ProtoStream
    pub async fn generate(
        &mut self,
        req: ProtoGenerateRequest,
    ) -> Result<ProtoStream, Box<dyn std::error::Error + Send + Sync>> {
        match (self, req) {
            (Self::Sglang(client), ProtoGenerateRequest::Sglang(boxed_req)) => {
                let stream = client.generate(*boxed_req).await?;
                Ok(ProtoStream::Sglang(stream))
            }
            (Self::Vllm(client), ProtoGenerateRequest::Vllm(boxed_req)) => {
                let stream = client.generate(*boxed_req).await?;
                Ok(ProtoStream::Vllm(stream))
            }
            _ => panic!("Mismatched client and request types"),
        }
    }
}
