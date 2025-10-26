//! Harmony response processor for non-streaming responses

use std::sync::Arc;

use super::HarmonyParserAdapter;
use crate::{
    grpc_client::proto,
    protocols::{
        chat::{ChatChoice, ChatCompletionMessage, ChatCompletionRequest, ChatCompletionResponse},
        common::Usage,
    },
    routers::grpc::{
        context::{DispatchMetadata, ExecutionResult},
        utils,
    },
};

/// Processor for non-streaming Harmony responses
///
/// Collects all output tokens from execution and parses them using
/// HarmonyParserAdapter to extract the complete response.
pub struct HarmonyResponseProcessor;

impl HarmonyResponseProcessor {
    /// Create a new Harmony response processor
    pub fn new() -> Self {
        Self
    }

    /// Collect responses from ExecutionResult (similar to regular processor)
    async fn collect_responses(
        execution_result: ExecutionResult,
    ) -> Result<Vec<proto::GenerateComplete>, axum::response::Response> {
        match execution_result {
            ExecutionResult::Single { mut stream } => {
                let responses = utils::collect_stream_responses(&mut stream, "Single").await?;
                stream.mark_completed();
                Ok(responses)
            }
            ExecutionResult::Dual { prefill, decode } => {
                // For Harmony we currently rely only on decode stream for outputs
                let mut decode_stream = *decode;
                let responses =
                    utils::collect_stream_responses(&mut decode_stream, "Decode").await?;
                prefill.mark_completed();
                decode_stream.mark_completed();
                Ok(responses)
            }
        }
    }

    /// Process a non-streaming Harmony chat response
    pub async fn process_non_streaming_chat_response(
        &self,
        execution_result: ExecutionResult,
        chat_request: Arc<ChatCompletionRequest>,
        dispatch: DispatchMetadata,
    ) -> Result<ChatCompletionResponse, axum::response::Response> {
        // Collect all completed responses (one per choice)
        let all_responses = Self::collect_responses(execution_result).await?;
        if all_responses.is_empty() {
            return Err(utils::internal_error_static("No responses from server"));
        }

        // Build choices by parsing Harmony channels from output_ids
        let mut choices: Vec<ChatChoice> = Vec::new();
        for (index, complete) in all_responses.iter().enumerate() {
            let mut parser = HarmonyParserAdapter::new().map_err(|e| {
                utils::internal_error_message(format!("Failed to create Harmony parser: {}", e))
            })?;

            let parsed = parser.parse_complete(&complete.output_ids).map_err(|e| {
                utils::internal_error_message(format!("Harmony parsing failed: {}", e))
            })?;

            // Build response message (assistant)
            let message = ChatCompletionMessage {
                role: "assistant".to_string(),
                content: (!parsed.final_text.is_empty()).then_some(parsed.final_text),
                tool_calls: parsed.commentary, // TODO: parse tool calls from commentary channel
                reasoning_content: parsed.analysis,
            };

            // Determine finish_reason (tool_calls overrides to OpenAI convention)
            let finish_reason = if message.tool_calls.is_some() {
                Some("tool_calls".to_string())
            } else {
                Some(complete.finish_reason.clone())
            };

            // Matched stop
            let matched_stop = complete.matched_stop.as_ref().map(|m| match m {
                proto::generate_complete::MatchedStop::MatchedTokenId(id) => {
                    serde_json::json!(id)
                }
                proto::generate_complete::MatchedStop::MatchedStopStr(s) => {
                    serde_json::json!(s)
                }
            });

            choices.push(ChatChoice {
                index: index as u32,
                message,
                logprobs: None,
                finish_reason,
                matched_stop,
                hidden_states: None,
            });
        }

        // Build usage from proto fields
        let prompt_tokens: u32 = all_responses.iter().map(|r| r.prompt_tokens as u32).sum();
        let completion_tokens: u32 = all_responses
            .iter()
            .map(|r| r.completion_tokens as u32)
            .sum();
        let usage = Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            completion_tokens_details: None,
        };

        // Final ChatCompletionResponse
        let response = ChatCompletionResponse {
            id: dispatch.request_id.clone(),
            object: "chat.completion".to_string(),
            created: dispatch.created,
            model: chat_request.model.clone(),
            choices,
            usage: Some(usage),
            system_fingerprint: dispatch.weight_version.clone(),
        };

        Ok(response)
    }
}

impl Default for HarmonyResponseProcessor {
    fn default() -> Self {
        Self::new()
    }
}
