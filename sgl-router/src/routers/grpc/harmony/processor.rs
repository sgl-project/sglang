//! Harmony response processor for non-streaming responses

use std::sync::Arc;

use axum::response::Response;
use proto::generate_complete::MatchedStop::{MatchedStopStr, MatchedTokenId};

use super::HarmonyParserAdapter;
use crate::{
    grpc_client::proto,
    protocols::{
        chat::{ChatChoice, ChatCompletionMessage, ChatCompletionRequest, ChatCompletionResponse},
        common::{ToolCall, Usage},
        responses::{
            ResponseContentPart, ResponseOutputItem, ResponseReasoningContent, ResponseStatus,
            ResponseUsage, ResponsesRequest, ResponsesResponse, ResponsesUsage,
        },
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
    ) -> Result<Vec<proto::GenerateComplete>, Response> {
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
    ) -> Result<ChatCompletionResponse, Response> {
        // Collect all completed responses (one per choice)
        let all_responses = Self::collect_responses(execution_result).await?;
        if all_responses.is_empty() {
            return Err(utils::internal_error_static("No responses from server"));
        }

        // Build choices by parsing output with HarmonyParserAdapter
        let mut choices: Vec<ChatChoice> = Vec::new();
        for (index, complete) in all_responses.iter().enumerate() {
            // Convert matched_stop from proto to JSON
            let matched_stop = complete.matched_stop.as_ref().map(|m| match m {
                MatchedTokenId(id) => {
                    serde_json::json!(id)
                }
                MatchedStopStr(s) => {
                    serde_json::json!(s)
                }
            });

            // Parse Harmony channels with HarmonyParserAdapter
            let mut parser = HarmonyParserAdapter::new().map_err(|e| {
                utils::internal_error_message(format!("Failed to create Harmony parser: {}", e))
            })?;

            // Parse Harmony channels with finish_reason and matched_stop
            let parsed = parser
                .parse_complete(
                    &complete.output_ids,
                    complete.finish_reason.clone(),
                    matched_stop.clone(),
                )
                .map_err(|e| {
                    utils::internal_error_message(format!("Harmony parsing failed: {}", e))
                })?;

            // Build response message (assistant)
            let message = ChatCompletionMessage {
                role: "assistant".to_string(),
                content: (!parsed.final_text.is_empty()).then_some(parsed.final_text),
                tool_calls: parsed.commentary,
                reasoning_content: parsed.analysis,
            };

            let finish_reason = parsed.finish_reason;

            choices.push(ChatChoice {
                index: index as u32,
                message,
                logprobs: None,
                finish_reason: Some(finish_reason),
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

/// Result of processing a single Responses API iteration
///
/// Used by the MCP tool loop to determine whether to continue
/// executing tools or return the final response.
pub enum ResponsesIterationResult {
    /// Tool calls found in commentary channel - continue MCP loop
    ToolCallsFound {
        tool_calls: Vec<ToolCall>,
        analysis: Option<String>, // For streaming emission
        partial_text: String,     // For streaming emission
    },
    /// No tool calls - return final ResponsesResponse
    Completed {
        response: Box<ResponsesResponse>,
        usage: Usage,
    },
}

impl HarmonyResponseProcessor {
    /// Process a single Responses API iteration
    ///
    /// Parses Harmony channels and determines if tool calls are present.
    /// If tool calls found, returns ToolCallsFound for MCP loop to execute.
    /// If no tool calls, builds final ResponsesResponse.
    ///
    /// # Arguments
    ///
    /// * `execution_result` - The execution result from the model
    /// * `responses_request` - The original Responses API request
    /// * `dispatch` - Dispatch metadata for request tracking
    ///
    /// # Returns
    ///
    /// ResponsesIterationResult indicating whether to continue loop or return
    pub async fn process_responses_iteration(
        &self,
        execution_result: ExecutionResult,
        responses_request: Arc<ResponsesRequest>,
        dispatch: DispatchMetadata,
    ) -> Result<ResponsesIterationResult, Response> {
        // Collect all completed responses
        let all_responses = Self::collect_responses(execution_result).await?;
        if all_responses.is_empty() {
            return Err(utils::internal_error_static("No responses from server"));
        }

        // For Responses API, we only process the first response (n=1)
        let complete = all_responses
            .first()
            .ok_or_else(|| utils::internal_error_static("No complete response"))?;

        // Parse Harmony channels
        let mut parser = HarmonyParserAdapter::new().map_err(|e| {
            utils::internal_error_message(format!("Failed to create Harmony parser: {}", e))
        })?;

        // Convert matched_stop from proto to JSON
        let matched_stop = complete.matched_stop.as_ref().map(|m| match m {
            MatchedTokenId(id) => {
                serde_json::json!(id)
            }
            MatchedStopStr(s) => {
                serde_json::json!(s)
            }
        });

        let parsed = parser
            .parse_complete(
                &complete.output_ids,
                complete.finish_reason.clone(),
                matched_stop,
            )
            .map_err(|e| utils::internal_error_message(format!("Harmony parsing failed: {}", e)))?;

        // VALIDATION: Check if model incorrectly generated Tool role messages
        // This happens when the model copies the format of tool result messages
        // instead of continuing as assistant. This is a model hallucination bug.
        let messages = parser.get_messages();
        let tool_messages_generated = messages.iter().any(|msg| {
            msg.author.role == openai_harmony::chat::Role::Tool
                && msg.recipient.as_deref() == Some("assistant")
        });

        if tool_messages_generated {
            tracing::warn!(
                "Model generated Tool->Assistant message instead of Assistant message. \
                This is a model hallucination bug where it copies tool result format."
            );
        }

        // Check for tool calls in commentary channel
        if let Some(tool_calls) = parsed.commentary {
            // Tool calls found - return for MCP loop execution
            return Ok(ResponsesIterationResult::ToolCallsFound {
                tool_calls,
                analysis: parsed.analysis,
                partial_text: parsed.final_text,
            });
        }

        // No tool calls - build final ResponsesResponse
        let mut output: Vec<ResponseOutputItem> = Vec::new();

        // Map analysis channel → ResponseOutputItem::Reasoning
        if let Some(analysis) = parsed.analysis {
            let reasoning_item = ResponseOutputItem::Reasoning {
                id: format!("reasoning_{}", dispatch.request_id),
                summary: vec![],
                content: vec![ResponseReasoningContent::ReasoningText { text: analysis }],
                status: Some("completed".to_string()),
            };
            output.push(reasoning_item);
        }

        // Map final channel → ResponseOutputItem::Message
        if !parsed.final_text.is_empty() {
            let message_item = ResponseOutputItem::Message {
                id: format!("msg_{}", dispatch.request_id),
                role: "assistant".to_string(),
                content: vec![ResponseContentPart::OutputText {
                    text: parsed.final_text,
                    annotations: vec![],
                    logprobs: None,
                }],
                status: "completed".to_string(),
            };
            output.push(message_item);
        }

        // Build usage
        let prompt_tokens = complete.prompt_tokens as u32;
        let completion_tokens = complete.completion_tokens as u32;
        let usage = Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            completion_tokens_details: None,
        };

        // Build ResponsesResponse with all required fields
        let response = ResponsesResponse {
            id: dispatch.request_id.clone(),
            object: "response".to_string(),
            created_at: dispatch.created as i64,
            status: ResponseStatus::Completed,
            error: None,
            incomplete_details: None,
            instructions: responses_request.instructions.clone(),
            max_output_tokens: responses_request.max_output_tokens,
            model: responses_request.model.clone(),
            output,
            parallel_tool_calls: responses_request.parallel_tool_calls.unwrap_or(true),
            previous_response_id: responses_request.previous_response_id.clone(),
            reasoning: None, // Set by caller if needed
            store: responses_request.store.unwrap_or(true),
            temperature: responses_request.temperature,
            text: None,
            tool_choice: responses_request
                .tool_choice
                .as_ref()
                .map(|tc| serde_json::to_string(tc).unwrap_or_else(|_| "auto".to_string()))
                .unwrap_or_else(|| "auto".to_string()),
            tools: responses_request.tools.clone().unwrap_or_default(),
            top_p: responses_request.top_p,
            truncation: None,
            usage: Some(ResponsesUsage::Modern(ResponseUsage {
                input_tokens: prompt_tokens,
                output_tokens: completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
                input_tokens_details: None,
                output_tokens_details: None,
            })),
            user: None,
            safety_identifier: responses_request.user.clone(),
            metadata: responses_request.metadata.clone().unwrap_or_default(),
        };

        Ok(ResponsesIterationResult::Completed {
            response: Box::new(response),
            usage,
        })
    }
}
