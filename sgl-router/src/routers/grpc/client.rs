//! Unified gRPC client wrapper for SGLang and vLLM backends

use crate::{
    grpc_client::{SglangSchedulerClient, VllmEngineClient},
    routers::grpc::proto_wrapper::{ProtoGenerateRequest, ProtoStream},
};

/// Health check response (common across backends)
#[derive(Debug, Clone)]
pub struct HealthCheckResponse {
    pub healthy: bool,
    pub message: String,
}

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

    /// Connect to gRPC server (runtime-aware)
    pub async fn connect(
        url: &str,
        runtime_type: &str,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        match runtime_type {
            "sglang" => Ok(Self::Sglang(SglangSchedulerClient::connect(url).await?)),
            "vllm" => Ok(Self::Vllm(VllmEngineClient::connect(url).await?)),
            _ => Err(format!("Unknown runtime type: {}", runtime_type).into()),
        }
    }

    /// Perform health check (dispatches to appropriate backend)
    pub async fn health_check(
        &self,
    ) -> Result<HealthCheckResponse, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            Self::Sglang(client) => {
                let resp = client.health_check().await?;
                Ok(HealthCheckResponse {
                    healthy: resp.healthy,
                    message: resp.message,
                })
            }
            Self::Vllm(client) => {
                let resp = client.health_check().await?;
                Ok(HealthCheckResponse {
                    healthy: resp.healthy,
                    message: resp.message,
                })
            }
        }
    }

    /// Get model info (returns enum wrapping backend-specific response)
    pub async fn get_model_info(
        &self,
    ) -> Result<ModelInfo, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            Self::Sglang(client) => {
                let info = client.get_model_info().await?;
                Ok(ModelInfo::Sglang(info))
            }
            Self::Vllm(client) => {
                let info = client.get_model_info().await?;
                Ok(ModelInfo::Vllm(info))
            }
        }
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

/// Unified ModelInfo wrapper
pub enum ModelInfo {
    Sglang(crate::grpc_client::sglang_proto::GetModelInfoResponse),
    Vllm(crate::grpc_client::vllm_proto::GetModelInfoResponse),
}

impl ModelInfo {
    /// Convert model info to label map for worker metadata
    pub fn to_labels(&self) -> std::collections::HashMap<String, String> {
        let mut labels = std::collections::HashMap::new();

        // Serialize to JSON Value (like pydantic's model_dump)
        let value = match self {
            ModelInfo::Sglang(info) => serde_json::to_value(info).ok(),
            ModelInfo::Vllm(info) => serde_json::to_value(info).ok(),
        };

        // Convert JSON object to HashMap, filtering out empty/zero/false values
        if let Some(serde_json::Value::Object(obj)) = value {
            for (key, val) in obj {
                match val {
                    // Insert non-empty strings
                    serde_json::Value::String(s) if !s.is_empty() => {
                        labels.insert(key, s);
                    }
                    // Insert positive numbers
                    serde_json::Value::Number(n) if n.as_i64().unwrap_or(0) > 0 => {
                        labels.insert(key, n.to_string());
                    }
                    // Insert true booleans
                    serde_json::Value::Bool(true) => {
                        labels.insert(key, "true".to_string());
                    }
                    // Insert non-empty arrays as JSON strings (for architectures, etc.)
                    serde_json::Value::Array(arr) if !arr.is_empty() => {
                        if let Ok(json_str) = serde_json::to_string(&arr) {
                            labels.insert(key, json_str);
                        }
                    }
                    // Skip empty strings, zeros, false, nulls, empty arrays, objects
                    _ => {}
                }
            }
        }

        labels
    }
}
