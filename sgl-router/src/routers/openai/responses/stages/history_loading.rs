//! History Loading stage for responses pipeline
//!
//! This stage:
//! - Loads previous response chain (if previous_response_id provided)
//! - Loads conversation history (if conversation ID provided)
//! - Builds conversation history context for the request

use async_trait::async_trait;
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use tracing::warn;

use super::ResponsesStage;
use crate::{
    data_connector::{ConversationId, ListParams, ResponseId, SortOrder},
    protocols::responses::{ResponseContentPart, ResponseInputOutputItem},
    routers::openai::responses::{ContextOutput, ResponsesRequestContext},
};

/// History loading stage for responses pipeline
pub struct ResponsesHistoryLoadingStage;

impl ResponsesHistoryLoadingStage {
    /// Maximum conversation history items to load
    const MAX_CONVERSATION_HISTORY_ITEMS: usize = 1000;
}

#[async_trait]
impl ResponsesStage for ResponsesHistoryLoadingStage {
    async fn execute(
        &self,
        ctx: &mut ResponsesRequestContext,
    ) -> Result<Option<Response>, Response> {
        let mut conversation_items: Vec<ResponseInputOutputItem> = Vec::new();

        // Load previous response chain
        if let Some(prev_id_str) = &ctx.request().previous_response_id {
            let prev_id = ResponseId::from(prev_id_str.as_str());
            match ctx
                .dependencies
                .response_storage
                .get_response_chain(&prev_id, None)
                .await
            {
                Ok(chain) => {
                    for stored in chain.responses.iter() {
                        // Convert input items from stored input (JSON array)
                        if let Some(input_arr) = stored.input.as_array() {
                            for item in input_arr {
                                match serde_json::from_value::<ResponseInputOutputItem>(
                                    item.clone(),
                                ) {
                                    Ok(input_item) => {
                                        conversation_items.push(input_item);
                                    }
                                    Err(e) => {
                                        warn!(
                                            "Failed to deserialize stored input item: {}. Item: {}",
                                            e, item
                                        );
                                    }
                                }
                            }
                        }

                        // Convert output items from stored output (JSON array)
                        if let Some(output_arr) = stored.output.as_array() {
                            for item in output_arr {
                                match serde_json::from_value::<ResponseInputOutputItem>(
                                    item.clone(),
                                ) {
                                    Ok(output_item) => {
                                        conversation_items.push(output_item);
                                    }
                                    Err(e) => {
                                        warn!(
                                            "Failed to deserialize stored output item: {}. Item: {}",
                                            e, item
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!(
                        "Failed to load previous response chain for {}: {}",
                        prev_id_str, e
                    );
                }
            }
        }

        // Load conversation history
        if let Some(conv_id_str) = &ctx.request().conversation {
            let conv_id = ConversationId::from(conv_id_str.as_str());

            // Verify conversation exists
            if let Ok(None) = ctx
                .dependencies
                .conversation_storage
                .get_conversation(&conv_id)
                .await
            {
                return Err((
                    StatusCode::NOT_FOUND,
                    Json(json!({"error": "Conversation not found"})),
                )
                    .into_response());
            }

            // Load conversation history (ascending order for chronological context)
            let params = ListParams {
                limit: Self::MAX_CONVERSATION_HISTORY_ITEMS,
                order: SortOrder::Asc,
                after: None,
            };

            match ctx
                .dependencies
                .conversation_item_storage
                .list_items(&conv_id, params)
                .await
            {
                Ok(stored_items) => {
                    for item in stored_items.into_iter() {
                        // Include messages, function calls, and function call outputs
                        // Skip reasoning items as they're internal processing details
                        match item.item_type.as_str() {
                            "message" => {
                                match serde_json::from_value::<Vec<ResponseContentPart>>(
                                    item.content.clone(),
                                ) {
                                    Ok(content_parts) => {
                                        conversation_items.push(ResponseInputOutputItem::Message {
                                            id: item.id.0.clone(),
                                            role: item
                                                .role
                                                .clone()
                                                .unwrap_or_else(|| "user".to_string()),
                                            content: content_parts,
                                            status: item.status.clone(),
                                        });
                                    }
                                    Err(e) => {
                                        warn!("Failed to deserialize message content: {}", e);
                                    }
                                }
                            }
                            "function_call" => {
                                // The entire function_call item is stored in content field
                                match serde_json::from_value::<ResponseInputOutputItem>(
                                    item.content.clone(),
                                ) {
                                    Ok(func_call) => conversation_items.push(func_call),
                                    Err(e) => {
                                        warn!("Failed to deserialize function_call: {}", e);
                                    }
                                }
                            }
                            "function_call_output" => {
                                // The entire function_call_output item is stored in content field
                                match serde_json::from_value::<ResponseInputOutputItem>(
                                    item.content.clone(),
                                ) {
                                    Ok(func_output) => {
                                        conversation_items.push(func_output);
                                    }
                                    Err(e) => {
                                        warn!("Failed to deserialize function_call_output: {}", e);
                                    }
                                }
                            }
                            "reasoning" => {
                                // Skip reasoning items - they're internal processing details
                            }
                            _ => {
                                // Skip unknown item types
                                warn!("Unknown item type in conversation: {}", item.item_type);
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!(
                        "Failed to load conversation history for {}: {}",
                        conv_id_str, e
                    );
                }
            }
        }

        // Store context output
        ctx.state.context = Some(ContextOutput {
            conversation_items: if conversation_items.is_empty() {
                None
            } else {
                Some(conversation_items)
            },
            conversation_id: ctx.request().conversation.clone(),
            previous_response_id: ctx.request().previous_response_id.clone(),
        });

        Ok(None)
    }

    fn name(&self) -> &'static str {
        "ResponsesHistoryLoading"
    }
}
