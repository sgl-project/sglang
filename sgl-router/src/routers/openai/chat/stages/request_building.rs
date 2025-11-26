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
