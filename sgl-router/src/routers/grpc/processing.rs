//! Shared response processing logic for gRPC routers
//!
//! This module contains response processing functions that are shared between
//! the regular router and PD router, eliminating ~1,200 lines of exact duplicates.

use std::{sync::Arc, time::Instant};

use proto::generate_complete::MatchedStop;
use serde_json::Value;
use tracing::error;

use super::{
    context::{DispatchMetadata, ExecutionResult},
    utils,
};
use crate::{
    grpc_client::proto,
    protocols::{
        chat::{ChatChoice, ChatCompletionMessage, ChatCompletionRequest, ChatCompletionResponse},
        common::{FunctionCallResponse, ToolCall, ToolChoice, ToolChoiceValue, Usage},
        generate::{GenerateMetaInfo, GenerateRequest, GenerateResponse},
    },
    reasoning_parser::ParserFactory as ReasoningParserFactory,
    tokenizer::{
        stop::{SequenceDecoderOutput, StopSequenceDecoder},
        traits::Tokenizer,
    },
    tool_parser::ParserFactory as ToolParserFactory,
};

// ============================================================================
// Response Processor - Main Entry Point
// ============================================================================

/// Unified response processor for both routers
#[derive(Clone)]
pub struct ResponseProcessor {
    pub tokenizer: Arc<dyn Tokenizer>,
    pub tool_parser_factory: ToolParserFactory,
    pub reasoning_parser_factory: ReasoningParserFactory,
    pub configured_tool_parser: Option<String>,
    pub configured_reasoning_parser: Option<String>,
}

impl ResponseProcessor {
    pub fn new(
        tokenizer: Arc<dyn Tokenizer>,
        tool_parser_factory: ToolParserFactory,
        reasoning_parser_factory: ReasoningParserFactory,
        configured_tool_parser: Option<String>,
        configured_reasoning_parser: Option<String>,
    ) -> Self {
        Self {
            tokenizer,
            tool_parser_factory,
            reasoning_parser_factory,
            configured_tool_parser,
            configured_reasoning_parser,
        }
    }

    /// Helper to collect responses from execution result and merge logprobs if needed
    async fn collect_and_merge_responses(
        execution_result: ExecutionResult,
        request_logprobs: bool,
    ) -> Result<Vec<proto::GenerateComplete>, axum::response::Response> {
        let all_responses = match execution_result {
            ExecutionResult::Single { mut stream } => {
                let responses = utils::collect_stream_responses(&mut stream, "Single").await?;
                stream.mark_completed();
                responses
            }
            ExecutionResult::Dual {
                mut prefill,
                decode,
            } => {
                // Collect prefill for input_logprobs (don't mark completed yet)
                let prefill_responses =
                    utils::collect_stream_responses(&mut prefill, "Prefill").await?;

                // Collect decode for actual output (don't mark completed yet)
                let mut decode_stream = *decode;
                let mut decode_responses =
                    utils::collect_stream_responses(&mut decode_stream, "Decode").await?;

                // Mark both streams as completed now that both succeeded
                prefill.mark_completed();
                decode_stream.mark_completed();

                // Merge prefill input_logprobs if requested
                if request_logprobs {
                    if let Some(prefill_input_logprobs) = prefill_responses
                        .first()
                        .and_then(|r| r.input_logprobs.clone())
                    {
                        for response in &mut decode_responses {
                            response.input_logprobs = Some(prefill_input_logprobs.clone());
                        }
                    }
                }

                decode_responses
            }
        };

        if all_responses.is_empty() {
            return Err(utils::internal_error_static("No responses from server"));
        }

        Ok(all_responses)
    }

    /// Process a single choice from GenerateComplete response (EXACT COPY from router.rs:1573-1725)
    #[allow(clippy::too_many_arguments)]
    pub async fn process_single_choice(
        &self,
        complete: &proto::GenerateComplete,
        index: usize,
        original_request: &ChatCompletionRequest,
        stop_decoder: &mut StopSequenceDecoder,
        history_tool_calls_count: usize,
        reasoning_parser_available: bool,
        tool_parser_available: bool,
    ) -> Result<ChatChoice, String> {
        stop_decoder.reset();
        // Decode tokens
        let outputs = stop_decoder
            .process_tokens(&complete.output_ids)
            .map_err(|e| format!("Failed to process tokens: {}", e))?;

        // Accumulate text with early breaks
        let mut final_text = String::new();
        for output in outputs {
            match output {
                SequenceDecoderOutput::Text(t) => final_text.push_str(&t),
                SequenceDecoderOutput::StoppedWithText(t) => {
                    final_text.push_str(&t);
                    break;
                }
                SequenceDecoderOutput::Stopped => break,
                SequenceDecoderOutput::Held => {}
            }
        }

        // Flush remaining text
        if let SequenceDecoderOutput::Text(t) = stop_decoder.flush() {
            final_text.push_str(&t);
        }

        // Step 1: Handle reasoning content parsing
        let mut reasoning_text: Option<String> = None;
        let mut processed_text = final_text;

        // Check if reasoning parsing is enabled and parser is available
        if original_request.separate_reasoning && reasoning_parser_available {
            let pooled_parser = utils::get_reasoning_parser(
                &self.reasoning_parser_factory,
                self.configured_reasoning_parser.as_ref(),
                &original_request.model,
            );

            let mut parser = pooled_parser.lock().await;
            match parser.detect_and_parse_reasoning(&processed_text) {
                Ok(result) => {
                    if !result.reasoning_text.is_empty() {
                        reasoning_text = Some(result.reasoning_text);
                    }
                    processed_text = result.normal_text;
                }
                Err(e) => {
                    return Err(format!("Reasoning parsing error: {}", e));
                }
            }
        }

        // Step 2: Handle tool call parsing
        let mut tool_calls: Option<Vec<ToolCall>> = None;
        let tool_choice_enabled = !matches!(
            &original_request.tool_choice,
            Some(ToolChoice::Value(ToolChoiceValue::None))
        );

        if tool_choice_enabled && original_request.tools.is_some() {
            // Check if JSON schema constraint was used (specific function or required mode)
            let used_json_schema = match &original_request.tool_choice {
                Some(ToolChoice::Function { .. }) => true,
                Some(ToolChoice::Value(ToolChoiceValue::Required)) => true,
                Some(ToolChoice::AllowedTools { mode, .. }) => mode == "required",
                _ => false,
            };

            if used_json_schema {
                (tool_calls, processed_text) = utils::parse_json_schema_response(
                    &processed_text,
                    &original_request.tool_choice,
                    &original_request.model,
                    history_tool_calls_count,
                );
            } else if tool_parser_available {
                (tool_calls, processed_text) = self
                    .parse_tool_calls(
                        &processed_text,
                        &original_request.model,
                        history_tool_calls_count,
                    )
                    .await;
            }
        }

        // Step 3: Use finish reason directly from proto (already OpenAI-compatible string)
        let finish_reason_str = &complete.finish_reason;

        // Override finish reason if we have tool calls
        let final_finish_reason_str = if tool_calls.is_some() {
            "tool_calls"
        } else {
            finish_reason_str
        };

        // Extract matched_stop information from proto
        let matched_stop = match &complete.matched_stop {
            Some(MatchedStop::MatchedTokenId(token_id)) => {
                Some(Value::Number(serde_json::Number::from(*token_id)))
            }
            Some(MatchedStop::MatchedStopStr(stop_str)) => Some(Value::String(stop_str.clone())),
            None => None,
        };

        // Step 4: Convert output logprobs if present
        let logprobs = if let Some(proto_logprobs) = &complete.output_logprobs {
            match utils::convert_proto_to_openai_logprobs(proto_logprobs, &self.tokenizer) {
                Ok(logprobs) => Some(logprobs),
                Err(e) => {
                    error!("Failed to convert logprobs: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Step 5: Build ChatCompletionMessage (proper response message type)
        let chat_message = ChatCompletionMessage {
            role: "assistant".to_string(),
            content: if processed_text.is_empty() {
                None
            } else {
                Some(processed_text)
            },
            tool_calls,
            reasoning_content: reasoning_text,
        };

        // Step 6: Build ChatChoice
        let choice = ChatChoice {
            index: index as u32,
            message: chat_message,
            logprobs,
            finish_reason: Some(final_finish_reason_str.to_string()),
            matched_stop,
            hidden_states: None,
        };

        Ok(choice)
    }

    /// Process non-streaming chat response (collects all responses and builds final response)
    pub async fn process_non_streaming_chat_response(
        &self,
        execution_result: ExecutionResult,
        chat_request: Arc<ChatCompletionRequest>,
        dispatch: DispatchMetadata,
        stop_decoder: &mut StopSequenceDecoder,
        request_logprobs: bool,
    ) -> Result<ChatCompletionResponse, axum::response::Response> {
        // Collect all responses from the execution result
        let all_responses =
            Self::collect_and_merge_responses(execution_result, request_logprobs).await?;

        let history_tool_calls_count = utils::get_history_tool_calls_count(&chat_request);

        // Check parser availability once upfront (not per choice)
        let reasoning_parser_available = chat_request.separate_reasoning
            && utils::check_reasoning_parser_availability(
                &self.reasoning_parser_factory,
                self.configured_reasoning_parser.as_ref(),
                &chat_request.model,
            );

        let tool_choice_enabled = !matches!(
            &chat_request.tool_choice,
            Some(ToolChoice::Value(ToolChoiceValue::None))
        );

        let tool_parser_available = tool_choice_enabled
            && chat_request.tools.is_some()
            && utils::check_tool_parser_availability(
                &self.tool_parser_factory,
                self.configured_tool_parser.as_ref(),
                &chat_request.model,
            );

        // Log once per request (not per choice)
        if chat_request.separate_reasoning && !reasoning_parser_available {
            tracing::debug!(
                "No reasoning parser found for model '{}', skipping reasoning parsing",
                chat_request.model
            );
        }

        if chat_request.tools.is_some() && tool_choice_enabled && !tool_parser_available {
            tracing::debug!(
                "No tool parser found for model '{}', skipping tool call parsing",
                chat_request.model
            );
        }

        // Process all choices
        let mut choices = Vec::new();
        for (index, complete) in all_responses.iter().enumerate() {
            match self
                .process_single_choice(
                    complete,
                    index,
                    &chat_request,
                    stop_decoder,
                    history_tool_calls_count,
                    reasoning_parser_available,
                    tool_parser_available,
                )
                .await
            {
                Ok(choice) => choices.push(choice),
                Err(e) => {
                    return Err(utils::internal_error_message(format!(
                        "Failed to process choice {}: {}",
                        index, e
                    )));
                }
            }
        }

        // Build usage
        let total_prompt_tokens: u32 = all_responses.iter().map(|r| r.prompt_tokens as u32).sum();
        let total_completion_tokens: u32 = all_responses
            .iter()
            .map(|r| r.completion_tokens as u32)
            .sum();
        let usage = Usage {
            prompt_tokens: total_prompt_tokens,
            completion_tokens: total_completion_tokens,
            total_tokens: total_prompt_tokens + total_completion_tokens,
            completion_tokens_details: None,
        };

        // Build final ChatCompletionResponse
        let response = ChatCompletionResponse {
            id: dispatch.request_id.clone(),
            object: "chat.completion".to_string(),
            created: dispatch.created,
            model: dispatch.model.clone(),
            choices,
            usage: Some(usage),
            system_fingerprint: dispatch.weight_version.clone(),
        };

        Ok(response)
    }

    /// Parse tool calls using model-specific parser (EXACT COPY from router.rs:296-361)
    pub async fn parse_tool_calls(
        &self,
        processed_text: &str,
        model: &str,
        history_tool_calls_count: usize,
    ) -> (Option<Vec<ToolCall>>, String) {
        // Get pooled parser for this model
        let pooled_parser = utils::get_tool_parser(
            &self.tool_parser_factory,
            self.configured_tool_parser.as_ref(),
            model,
        );

        // Try parsing directly (parser will handle detection internally)
        let result = {
            let parser = pooled_parser.lock().await;
            parser.parse_complete(processed_text).await
            // Lock is dropped here
        };

        match result {
            Ok((normal_text, parsed_tool_calls)) => {
                if parsed_tool_calls.is_empty() {
                    return (None, normal_text);
                }

                let spec_tool_calls = parsed_tool_calls
                    .into_iter()
                    .enumerate()
                    .map(|(index, tc)| {
                        // Generate ID for this tool call
                        let id = utils::generate_tool_call_id(
                            model,
                            &tc.function.name,
                            index,
                            history_tool_calls_count,
                        );
                        ToolCall {
                            id,
                            tool_type: "function".to_string(),
                            function: FunctionCallResponse {
                                name: tc.function.name,
                                arguments: Some(tc.function.arguments),
                            },
                        }
                    })
                    .collect();
                (Some(spec_tool_calls), normal_text)
            }
            Err(e) => {
                error!("Tool call parsing error: {}", e);
                (None, processed_text.to_string())
            }
        }
    }

    /// Process non-streaming generate response (collects all responses and builds final response array)
    pub async fn process_non_streaming_generate_response(
        &self,
        execution_result: ExecutionResult,
        _generate_request: Arc<GenerateRequest>,
        dispatch: DispatchMetadata,
        stop_decoder: &mut StopSequenceDecoder,
        request_logprobs: bool,
        start_time: Instant,
    ) -> Result<Vec<GenerateResponse>, axum::response::Response> {
        // Collect all responses from the execution result
        let all_responses =
            Self::collect_and_merge_responses(execution_result, request_logprobs).await?;

        // Process each completion
        let mut result_array = Vec::new();
        for mut complete in all_responses {
            stop_decoder.reset();

            // Process tokens through stop decoder
            let outputs = match stop_decoder.process_tokens(&complete.output_ids) {
                Ok(outputs) => outputs,
                Err(e) => {
                    return Err(utils::internal_error_message(format!(
                        "Failed to process tokens: {}",
                        e
                    )))
                }
            };

            // Accumulate text with early breaks
            let mut decoded_text = String::new();
            for output in outputs {
                match output {
                    SequenceDecoderOutput::Text(t) => decoded_text.push_str(&t),
                    SequenceDecoderOutput::StoppedWithText(t) => {
                        decoded_text.push_str(&t);
                        break;
                    }
                    SequenceDecoderOutput::Stopped => break,
                    SequenceDecoderOutput::Held => {}
                }
            }

            // Flush remaining text
            if let SequenceDecoderOutput::Text(t) = stop_decoder.flush() {
                decoded_text.push_str(&t);
            }

            let output_ids = std::mem::take(&mut complete.output_ids);
            let finish_reason_str = std::mem::take(&mut complete.finish_reason);

            // Parse finish_reason from string to proper type
            let finish_reason =
                utils::parse_finish_reason(&finish_reason_str, complete.completion_tokens);

            // Handle matched_stop if present
            let matched_stop = complete.matched_stop.take().map(|matched| match matched {
                MatchedStop::MatchedTokenId(id) => serde_json::json!(id),
                MatchedStop::MatchedStopStr(s) => serde_json::json!(s),
            });

            // Extract logprobs if requested (convert proto types to Generate format)
            let input_token_logprobs = if request_logprobs {
                complete
                    .input_logprobs
                    .as_ref()
                    .map(utils::convert_generate_input_logprobs)
            } else {
                None
            };

            let output_token_logprobs = if request_logprobs {
                complete
                    .output_logprobs
                    .as_ref()
                    .map(utils::convert_generate_output_logprobs)
            } else {
                None
            };

            // Build GenerateResponse struct
            let meta_info = GenerateMetaInfo {
                id: dispatch.request_id.clone(),
                finish_reason,
                prompt_tokens: complete.prompt_tokens as u32,
                weight_version: dispatch
                    .weight_version
                    .clone()
                    .unwrap_or_else(|| "default".to_string()),
                input_token_logprobs,
                output_token_logprobs,
                completion_tokens: complete.completion_tokens as u32,
                cached_tokens: complete.cached_tokens as u32,
                e2e_latency: start_time.elapsed().as_secs_f64(),
                matched_stop,
            };

            result_array.push(GenerateResponse {
                text: decoded_text,
                output_ids,
                meta_info,
            });
        }

        Ok(result_array)
    }
}
