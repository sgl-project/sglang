//! Shared response processing logic for gRPC routers
//!
//! This module contains response processing functions that are shared between
//! the regular router and PD router.

use std::{sync::Arc, time::Instant};

use serde_json::Value;
use smg_grpc_client::sglang_proto::generate_complete::MatchedStop;
use tracing::error;

use crate::{
    protocols::{
        chat::{ChatChoice, ChatCompletionMessage, ChatCompletionRequest, ChatCompletionResponse},
        common::{FunctionCallResponse, ToolCall, ToolChoice, ToolChoiceValue},
        generate::{GenerateMetaInfo, GenerateRequest, GenerateResponse},
    },
    reasoning_parser::ParserFactory as ReasoningParserFactory,
    routers::{
        error,
        grpc::{
            common::{response_collection, response_formatting},
            context::{DispatchMetadata, ExecutionResult},
            proto_wrapper::ProtoGenerateComplete,
            utils,
        },
    },
    tokenizer::{
        stop::{SequenceDecoderOutput, StopSequenceDecoder},
        traits::Tokenizer,
    },
    tool_parser::ParserFactory as ToolParserFactory,
};

/// Unified response processor for both routers
#[derive(Clone)]
pub(crate) struct ResponseProcessor {
    pub tool_parser_factory: ToolParserFactory,
    pub reasoning_parser_factory: ReasoningParserFactory,
    pub configured_tool_parser: Option<String>,
    pub configured_reasoning_parser: Option<String>,
}

impl ResponseProcessor {
    pub fn new(
        tool_parser_factory: ToolParserFactory,
        reasoning_parser_factory: ReasoningParserFactory,
        configured_tool_parser: Option<String>,
        configured_reasoning_parser: Option<String>,
    ) -> Self {
        Self {
            tool_parser_factory,
            reasoning_parser_factory,
            configured_tool_parser,
            configured_reasoning_parser,
        }
    }

    /// Process a single choice from GenerateComplete response
    #[allow(clippy::too_many_arguments)]
    pub async fn process_single_choice(
        &self,
        complete: &ProtoGenerateComplete,
        index: usize,
        original_request: &ChatCompletionRequest,
        tokenizer: &Arc<dyn Tokenizer>,
        stop_decoder: &mut StopSequenceDecoder,
        history_tool_calls_count: usize,
        reasoning_parser_available: bool,
        tool_parser_available: bool,
        reasoning_started_in_prefill: bool,
    ) -> Result<ChatChoice, String> {
        stop_decoder.reset();
        // Decode tokens
        let outputs = stop_decoder
            .process_tokens(complete.output_ids())
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

        if original_request.separate_reasoning && reasoning_parser_available {
            let mut parser = utils::create_reasoning_parser(
                &self.reasoning_parser_factory,
                self.configured_reasoning_parser.as_deref(),
                &original_request.model,
                reasoning_started_in_prefill,
            )
            .ok_or_else(|| "Reasoning parser is unavailable".to_string())?;

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
        let finish_reason_str = complete.finish_reason();

        // Override finish reason if we have tool calls
        let final_finish_reason_str = if tool_calls.is_some() {
            "tool_calls"
        } else {
            finish_reason_str
        };

        // Extract matched_stop information from proto
        let matched_stop = match complete.matched_stop() {
            Some(MatchedStop::MatchedTokenId(token_id)) => {
                Some(Value::Number(serde_json::Number::from(*token_id)))
            }
            Some(MatchedStop::MatchedStopStr(stop_str)) => Some(Value::String(stop_str.clone())),
            None => None,
        };

        // Step 4: Convert output logprobs if present
        let logprobs = if let Some(proto_logprobs) = complete.output_logprobs() {
            match utils::convert_proto_to_openai_logprobs(proto_logprobs, tokenizer) {
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
        Ok(ChatChoice {
            index: index as u32,
            message: chat_message,
            logprobs,
            finish_reason: Some(final_finish_reason_str.to_string()),
            matched_stop,
            hidden_states: None,
        })
    }

    /// Process non-streaming chat response (collects all responses and builds final response)
    #[allow(clippy::too_many_arguments)]
    pub async fn process_non_streaming_chat_response(
        &self,
        execution_result: ExecutionResult,
        chat_request: Arc<ChatCompletionRequest>,
        dispatch: DispatchMetadata,
        tokenizer: Arc<dyn Tokenizer>,
        stop_decoder: &mut StopSequenceDecoder,
        request_logprobs: bool,
        reasoning_started_in_prefill: bool,
    ) -> Result<ChatCompletionResponse, axum::response::Response> {
        // Collect all responses from the execution result
        let all_responses =
            response_collection::collect_responses(execution_result, request_logprobs).await?;

        let history_tool_calls_count = utils::get_history_tool_calls_count(&chat_request);

        // Check parser availability once upfront (not per choice)
        let reasoning_parser_available = chat_request.separate_reasoning
            && utils::check_reasoning_parser_availability(
                &self.reasoning_parser_factory,
                self.configured_reasoning_parser.as_deref(),
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
                self.configured_tool_parser.as_deref(),
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
                    &tokenizer,
                    stop_decoder,
                    history_tool_calls_count,
                    reasoning_parser_available,
                    tool_parser_available,
                    reasoning_started_in_prefill,
                )
                .await
            {
                Ok(choice) => choices.push(choice),
                Err(e) => {
                    return Err(error::internal_error(
                        "process_choice_failed",
                        format!("Failed to process choice {}: {}", index, e),
                    ));
                }
            }
        }

        // Build usage
        let usage = response_formatting::build_usage(&all_responses);

        // Build final ChatCompletionResponse
        Ok(
            ChatCompletionResponse::builder(&dispatch.request_id, &dispatch.model)
                .created(dispatch.created)
                .choices(choices)
                .usage(usage)
                .maybe_system_fingerprint(dispatch.weight_version.clone())
                .build(),
        )
    }

    /// Parse tool calls using model-specific parser
    pub async fn parse_tool_calls(
        &self,
        processed_text: &str,
        model: &str,
        history_tool_calls_count: usize,
    ) -> (Option<Vec<ToolCall>>, String) {
        // Get pooled parser for this model
        let pooled_parser = utils::get_tool_parser(
            &self.tool_parser_factory,
            self.configured_tool_parser.as_deref(),
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
            response_collection::collect_responses(execution_result, request_logprobs).await?;

        // Process each completion
        let mut result_array = Vec::new();
        for complete in all_responses {
            stop_decoder.reset();

            // Process tokens through stop decoder
            let outputs = match stop_decoder.process_tokens(complete.output_ids()) {
                Ok(outputs) => outputs,
                Err(e) => {
                    return Err(error::internal_error(
                        "process_tokens_failed",
                        format!("Failed to process tokens: {}", e),
                    ))
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

            let output_ids = complete.output_ids().to_vec();
            let finish_reason_str = complete.finish_reason();

            // Parse finish_reason from string to proper type
            let finish_reason =
                utils::parse_finish_reason(finish_reason_str, complete.completion_tokens());

            // Handle matched_stop if present
            let matched_stop = complete.matched_stop().map(|matched| match matched {
                MatchedStop::MatchedTokenId(id) => serde_json::json!(id),
                MatchedStop::MatchedStopStr(s) => serde_json::json!(s),
            });

            // Extract logprobs if requested (convert proto types to Generate format)
            let input_token_logprobs = if request_logprobs {
                complete
                    .input_logprobs()
                    .map(utils::convert_generate_input_logprobs)
            } else {
                None
            };

            let output_token_logprobs = if request_logprobs {
                complete
                    .output_logprobs()
                    .map(utils::convert_generate_output_logprobs)
            } else {
                None
            };

            // Build GenerateResponse struct
            let meta_info = GenerateMetaInfo {
                id: dispatch.request_id.clone(),
                finish_reason,
                prompt_tokens: complete.prompt_tokens() as u32,
                weight_version: dispatch
                    .weight_version
                    .clone()
                    .unwrap_or_else(|| "default".to_string()),
                input_token_logprobs,
                output_token_logprobs,
                completion_tokens: complete.completion_tokens() as u32,
                cached_tokens: complete.cached_tokens() as u32,
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

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use llm_tokenizer::{
        stop::StopSequenceDecoderBuilder,
        traits::{Decoder, Encoder, Encoding, SpecialTokens, Tokenizer},
    };
    use serde_json::json;
    use smg_grpc_client::sglang_proto::GenerateComplete;

    use super::*;

    struct PrefillReasoningTokenizer {
        special_tokens: SpecialTokens,
    }

    impl PrefillReasoningTokenizer {
        fn new() -> Self {
            Self {
                special_tokens: SpecialTokens::default(),
            }
        }
    }

    impl Encoder for PrefillReasoningTokenizer {
        fn encode(&self, _input: &str, _add_special_tokens: bool) -> Result<Encoding> {
            Ok(Encoding::Plain(Vec::new()))
        }

        fn encode_batch(
            &self,
            inputs: &[&str],
            _add_special_tokens: bool,
        ) -> Result<Vec<Encoding>> {
            Ok(inputs.iter().map(|_| Encoding::Plain(Vec::new())).collect())
        }
    }

    impl Decoder for PrefillReasoningTokenizer {
        fn decode(&self, token_ids: &[u32], _skip_special_tokens: bool) -> Result<String> {
            Ok(token_ids
                .iter()
                .filter_map(|token_id| match token_id {
                    1 => Some("reasoning without an opening tag"),
                    2 => Some("</think>"),
                    3 => Some("final answer"),
                    _ => None,
                })
                .collect())
        }
    }

    impl Tokenizer for PrefillReasoningTokenizer {
        fn vocab_size(&self) -> usize {
            3
        }

        fn get_special_tokens(&self) -> &SpecialTokens {
            &self.special_tokens
        }

        fn token_to_id(&self, _token: &str) -> Option<u32> {
            None
        }

        fn id_to_token(&self, _id: u32) -> Option<String> {
            None
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[tokio::test]
    async fn prefilled_think_token_is_reflected_in_non_streaming_chat_choice() {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(PrefillReasoningTokenizer::new());
        let mut stop_decoder = StopSequenceDecoderBuilder::new(tokenizer.clone()).build();
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "qwen3",
            "messages": [{"role": "user", "content": "Why?"}]
        }))
        .unwrap();
        let complete = ProtoGenerateComplete::Sglang(GenerateComplete {
            output_ids: vec![1, 2, 3],
            finish_reason: "stop".to_string(),
            completion_tokens: 3,
            ..Default::default()
        });
        let processor = ResponseProcessor::new(
            ToolParserFactory::new(),
            ReasoningParserFactory::new(),
            None,
            Some("qwen3".to_string()),
        );

        let choice = processor
            .process_single_choice(
                &complete,
                0,
                &request,
                &tokenizer,
                &mut stop_decoder,
                0,
                true,
                false,
                true,
            )
            .await
            .unwrap();

        assert_eq!(
            choice.message.reasoning_content.as_deref(),
            Some("reasoning without an opening tag")
        );
        assert_eq!(choice.message.content.as_deref(), Some("final answer"));
    }
}
