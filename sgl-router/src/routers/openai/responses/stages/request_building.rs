//! Request Building stage for responses pipeline
//!
//! This stage:
//! - Builds the HTTP request payload
//! - Injects conversation items from context loading
//! - Strips SGLang-specific fields (conversation, previous_response_id, store)
//! - Applies provider-specific transformations (xAI/Grok)
//! - Serializes to JSON

use std::collections::HashSet;

use async_trait::async_trait;
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};
use once_cell::sync::Lazy;
use serde_json::{to_value, Value};

use super::ResponsesStage;
use crate::{
    protocols::responses::{ResponseContentPart, ResponseInput, ResponseInputOutputItem},
    routers::openai::responses::{PayloadOutput, ResponsesRequestContext},
};

/// Fields specific to SGLang that should be stripped when forwarding to OpenAI-compatible endpoints
static SGLANG_RESPONSES_FIELDS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
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

/// Request building stage for responses pipeline
pub struct ResponsesRequestBuildingStage;

#[async_trait]
impl ResponsesStage for ResponsesRequestBuildingStage {
    async fn execute(
        &self,
        ctx: &mut ResponsesRequestContext,
    ) -> Result<Option<Response>, Response> {
        // Get prerequisites
        ctx.state.discovery.as_ref().ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Model discovery stage not completed",
            )
                .into_response()
        })?;

        // Clone request for modification
        let mut request_body = ctx.request().clone();

        // Apply model_id if provided
        if let Some(model) = &ctx.input.model_id {
            request_body.model = model.clone();
        }

        // Inject conversation items from context loading
        if let Some(context) = &ctx.state.context {
            if let Some(conversation_items) = &context.conversation_items {
                let mut items = conversation_items.clone();

                // Append current request input
                match &request_body.input {
                    ResponseInput::Text(text) => {
                        items.push(ResponseInputOutputItem::Message {
                            id: format!(
                                "msg_u_{}",
                                context
                                    .previous_response_id
                                    .as_ref()
                                    .or(context.conversation_id.as_ref())
                                    .unwrap_or(&"new".to_string())
                            ),
                            role: "user".to_string(),
                            content: vec![ResponseContentPart::InputText { text: text.clone() }],
                            status: Some("completed".to_string()),
                        });
                    }
                    ResponseInput::Items(current_items) => {
                        // Process all items, normalizing SimpleInputMessage to Message
                        for item in current_items.iter() {
                            let normalized =
                                crate::protocols::responses::normalize_input_item(item);
                            items.push(normalized);
                        }
                    }
                }

                request_body.input = ResponseInput::Items(items);
            }
        }

        // Always set store=false for upstream (we store internally)
        request_body.store = Some(false);

        // Remove conversation and previous_response_id since router handles them internally
        request_body.conversation = None;
        request_body.previous_response_id = None;

        // Filter out reasoning items from input - they're internal processing details
        if let ResponseInput::Items(ref mut items) = request_body.input {
            items.retain(|item| !matches!(item, ResponseInputOutputItem::Reasoning { .. }));
        }

        let is_streaming = request_body.stream.unwrap_or(false);

        // Convert to JSON
        let mut payload = to_value(&request_body).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                format!("Failed to serialize request: {}", e),
            )
                .into_response()
        })?;

        // Strip SGLang-specific fields
        if let Some(obj) = payload.as_object_mut() {
            obj.retain(|k, _| !SGLANG_RESPONSES_FIELDS.contains(&k.as_str()));

            // XAI (Grok models) special handling
            let is_grok_model = obj
                .get("model")
                .and_then(|v| v.as_str())
                .map(|m| m.starts_with("grok"))
                .unwrap_or(false);

            if is_grok_model {
                // XAI doesn't support the full OpenAI item type input
                // Strip extra fields from input messages (id, status)
                // Normalize content types: output_text -> input_text
                if let Some(input_arr) = obj.get_mut("input").and_then(Value::as_array_mut) {
                    for item_obj in input_arr.iter_mut().filter_map(Value::as_object_mut) {
                        // Remove fields not universally supported
                        item_obj.remove("id");
                        item_obj.remove("status");

                        // Normalize content types to input_text (xAI compatibility)
                        if let Some(content_arr) =
                            item_obj.get_mut("content").and_then(Value::as_array_mut)
                        {
                            for content_obj in
                                content_arr.iter_mut().filter_map(Value::as_object_mut)
                            {
                                // Change output_text to input_text
                                if content_obj.get("type").and_then(Value::as_str)
                                    == Some("output_text")
                                {
                                    content_obj.insert(
                                        "type".to_string(),
                                        Value::String("input_text".to_string()),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        // Store payload output
        ctx.state.payload = Some(PayloadOutput {
            json_payload: payload,
            is_streaming,
        });

        Ok(None)
    }

    fn name(&self) -> &'static str {
        "ResponsesRequestBuilding"
    }
}
