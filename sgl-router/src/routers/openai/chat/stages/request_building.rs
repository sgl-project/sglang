//! Request Building stage for chat pipeline
//!
//! This stage:
//! - Builds the HTTP request payload
//! - Strips SGLang-specific fields
//! - Applies provider-specific transformations
//! - Serializes to JSON

use std::collections::HashSet;

use async_trait::async_trait;
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};
use once_cell::sync::Lazy;
use serde_json::to_value;

use super::ChatStage;
use crate::routers::openai::chat::{ChatRequestContext, PayloadOutput};

/// Fields specific to SGLang that should be stripped when forwarding to OpenAI-compatible endpoints
static SGLANG_CHAT_FIELDS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    HashSet::from([
        "request_id",
        "priority",
        "top_k",
        "min_p",
        "min_tokens",
        "regex",
        "ebnf",
        "stop_token_ids",
        "no_stop_trim",
        "ignore_eos",
        "continue_final_message",
        "skip_special_tokens",
        "lora_path",
        "session_params",
        "separate_reasoning",
        "stream_reasoning",
        "chat_template_kwargs",
        "return_hidden_states",
        "repetition_penalty",
        "sampling_seed",
    ])
});

/// Request building stage for chat pipeline
pub struct ChatRequestBuildingStage;

#[async_trait]
impl ChatStage for ChatRequestBuildingStage {
    async fn execute(&self, ctx: &mut ChatRequestContext) -> Result<(), Response> {
        // Get discovery output (for validation)
        ctx.state.discovery.as_ref().ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Model discovery stage not completed",
            )
                .into_response()
        })?;

        // Serialize chat request to JSON
        let is_streaming = ctx.request().stream;
        let mut payload = to_value(ctx.request()).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                format!("Failed to serialize request: {}", e),
            )
                .into_response()
        })?;

        // Strip SGLang-specific fields and apply transformations
        if let Some(obj) = payload.as_object_mut() {
            obj.retain(|k, _| !SGLANG_CHAT_FIELDS.contains(&k.as_str()));

            // Remove logprobs if false (Gemini compatibility)
            if obj.get("logprobs").and_then(|v| v.as_bool()) == Some(false) {
                obj.remove("logprobs");
            }
        }

        // Store payload output
        ctx.state.payload = Some(PayloadOutput {
            json_payload: payload,
            is_streaming,
        });

        Ok(())
    }

    fn name(&self) -> &'static str {
        "ChatRequestBuilding"
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Instant};

    use dashmap::DashMap;
    use serde_json::json;

    use super::*;
    use crate::{
        core::CircuitBreaker,
        protocols::chat::{ChatCompletionRequest, ChatMessage, MessageContent},
        routers::openai::chat::{ChatDependencies, DiscoveryOutput, ValidationOutput},
    };

    fn create_test_dependencies() -> Arc<ChatDependencies> {
        let client = reqwest::Client::new();
        let circuit_breaker = Arc::new(CircuitBreaker::new());
        let model_cache = Arc::new(DashMap::new());
        let worker_urls = vec!["http://localhost:8000".to_string()];

        Arc::new(ChatDependencies {
            http_client: client,
            circuit_breaker,
            model_cache,
            worker_urls,
        })
    }

    #[tokio::test]
    async fn test_request_building_stage_success() {
        let request = ChatCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![ChatMessage::User {
                content: MessageContent::Text("Hello".to_string()),
                name: None,
            }],
            stream: false,
            ..Default::default()
        };

        let dependencies = create_test_dependencies();
        let mut ctx = ChatRequestContext::new(
            Arc::new(request),
            None,
            None,
            dependencies,
        );

        // Set prerequisite state
        ctx.state.validation = Some(ValidationOutput {
            auth_header: None,
            validated_at: Instant::now(),
        });
        ctx.state.discovery = Some(DiscoveryOutput {
            endpoint_url: "http://localhost:8000".to_string(),
            model: "gpt-4".to_string(),
        });

        let stage = ChatRequestBuildingStage;
        let result = stage.execute(&mut ctx).await;

        assert!(result.is_ok());
        assert!(ctx.state.payload.is_some());

        let payload = ctx.state.payload.unwrap();
        assert_eq!(payload.is_streaming, false);
        assert!(payload.json_payload.is_object());

        // Check SGLang fields are stripped
        let payload_obj = payload.json_payload.as_object().unwrap();
        assert!(!payload_obj.contains_key("request_id"));
        assert!(!payload_obj.contains_key("priority"));
        assert!(!payload_obj.contains_key("regex"));
    }

    #[tokio::test]
    async fn test_request_building_stage_no_discovery() {
        let request = ChatCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![],
            ..Default::default()
        };

        let dependencies = create_test_dependencies();
        let mut ctx = ChatRequestContext::new(
            Arc::new(request),
            None,
            None,
            dependencies,
        );

        // Set validation but not discovery
        ctx.state.validation = Some(ValidationOutput {
            auth_header: None,
            validated_at: Instant::now(),
        });

        let stage = ChatRequestBuildingStage;
        let result = stage.execute(&mut ctx).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_request_building_stage_strips_sglang_fields() {
        let request = ChatCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![],
            ..Default::default()
        };

        // Add SGLang-specific fields (not normally in ChatCompletionRequest, but testing serialization)
        let dependencies = create_test_dependencies();
        let mut ctx = ChatRequestContext::new(
            Arc::new(request),
            None,
            None,
            dependencies,
        );

        ctx.state.validation = Some(ValidationOutput {
            auth_header: None,
            validated_at: Instant::now(),
        });
        ctx.state.discovery = Some(DiscoveryOutput {
            endpoint_url: "http://localhost:8000".to_string(),
            model: "gpt-4".to_string(),
        });

        let stage = ChatRequestBuildingStage;
        let result = stage.execute(&mut ctx).await;

        assert!(result.is_ok());

        let payload = ctx.state.payload.unwrap();
        let payload_obj = payload.json_payload.as_object().unwrap();

        // Verify SGLang fields are not present
        for field in SGLANG_CHAT_FIELDS.iter() {
            assert!(!payload_obj.contains_key(*field), "Field {} should be stripped", field);
        }
    }

    #[tokio::test]
    async fn test_request_building_stage_removes_false_logprobs() {
        let request = ChatCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![],
            logprobs: false, // This should be removed
            ..Default::default()
        };

        let dependencies = create_test_dependencies();
        let mut ctx = ChatRequestContext::new(
            Arc::new(request),
            None,
            None,
            dependencies,
        );

        ctx.state.validation = Some(ValidationOutput {
            auth_header: None,
            validated_at: Instant::now(),
        });
        ctx.state.discovery = Some(DiscoveryOutput {
            endpoint_url: "http://localhost:8000".to_string(),
            model: "gpt-4".to_string(),
        });

        let stage = ChatRequestBuildingStage;
        let result = stage.execute(&mut ctx).await;

        assert!(result.is_ok());

        let payload = ctx.state.payload.unwrap();
        let payload_obj = payload.json_payload.as_object().unwrap();

        // logprobs=false should be removed (Gemini compatibility)
        assert!(!payload_obj.contains_key("logprobs"));
    }
}
