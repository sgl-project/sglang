//! Shared response processing logic for gRPC routers
//!
//! This module contains response processing functions that are shared between
//! the regular router and PD router, eliminating ~1,200 lines of exact duplicates.

use std::sync::Arc;

use serde_json::Value;
use tracing::{debug, error};

use crate::grpc_client::proto;
use crate::protocols::spec::{
    ChatChoice, ChatCompletionMessage, ChatMessage, FunctionCallResponse, ToolCall,
    ToolChoiceValue,
};
use crate::reasoning_parser::ReasoningParserFactory;
use crate::tokenizer::stop::{SequenceDecoderOutput, StopSequenceDecoder};
use crate::tokenizer::traits::Tokenizer;
use crate::tool_parser::ToolParserFactory;

// ============================================================================
// Response Processor - Main Entry Point
// ============================================================================

/// Unified response processor for both routers
#[derive(Clone)]
pub struct ResponseProcessor {
    pub tokenizer: Arc<dyn Tokenizer>,
    pub tool_parser_factory: ToolParserFactory,
    pub reasoning_parser_factory: ReasoningParserFactory,
    configured_tool_parser: Option<String>,
    configured_reasoning_parser: Option<String>,
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

    /// Process a single choice from GenerateComplete response (EXACT COPY from router.rs:1573-1725)
    pub async fn process_single_choice(
        &self,
        complete: &proto::GenerateComplete,
        index: usize,
        original_request: &crate::protocols::spec::ChatCompletionRequest,
        stop_decoder: &mut StopSequenceDecoder,
        history_tool_calls_count: usize,
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

        // Check if reasoning parsing is enabled and separate_reasoning is requested
        if original_request.separate_reasoning {
            let pooled_parser = super::utils::get_reasoning_parser(
                &self.reasoning_parser_factory,
                self.configured_reasoning_parser.as_ref(),
                &original_request.model,
            );

            let mut parser = pooled_parser
                .lock()
                .map_err(|e| format!("Failed to acquire reasoning parser lock: {}", e))?;
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

        // Check if tool calls should be processed
        let tool_choice_enabled = !matches!(
            &original_request.tool_choice,
            Some(crate::protocols::spec::ToolChoice::Value(
                ToolChoiceValue::None
            ))
        );

        if tool_choice_enabled && original_request.tools.is_some() {
            // Check if JSON schema constraint was used (specific function or required mode)
            let used_json_schema = match &original_request.tool_choice {
                Some(crate::protocols::spec::ToolChoice::Function { .. }) => true,
                Some(crate::protocols::spec::ToolChoice::Value(ToolChoiceValue::Required)) => true,
                Some(crate::protocols::spec::ToolChoice::AllowedTools { mode, .. }) => {
                    mode == "required"
                }
                _ => false,
            };

            if used_json_schema {
                (tool_calls, processed_text) =
                    crate::routers::grpc::utils::parse_json_schema_response(
                        &processed_text,
                        &original_request.tool_choice,
                    );
            } else {
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
            Some(proto::generate_complete::MatchedStop::MatchedTokenId(token_id)) => {
                Some(Value::Number(serde_json::Number::from(*token_id)))
            }
            Some(proto::generate_complete::MatchedStop::MatchedStopStr(stop_str)) => {
                Some(Value::String(stop_str.clone()))
            }
            None => None,
        };

        // Step 4: Convert output logprobs if present
        let logprobs = if let Some(proto_logprobs) = &complete.output_logprobs {
            match super::utils::convert_proto_to_openai_logprobs(proto_logprobs, &self.tokenizer) {
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

    /// Parse tool calls using model-specific parser (EXACT COPY from router.rs:296-361)
    pub async fn parse_tool_calls(
        &self,
        processed_text: &str,
        model: &str,
        history_tool_calls_count: usize,
    ) -> (Option<Vec<ToolCall>>, String) {
        // Get pooled parser for this model
        let pooled_parser = super::utils::get_tool_parser(
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
                        let id = generate_tool_call_id(
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
                                arguments: Some(
                                    serde_json::to_string(&tc.function.arguments)
                                        .unwrap_or_else(|_| "{}".to_string()),
                                ),
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
}

// ============================================================================
// Helper Functions (EXACT COPIES from router.rs)
// ============================================================================

/// Generate a tool call ID based on model format (from router.rs:436-449)
pub fn generate_tool_call_id(
    model: &str,
    tool_name: &str,
    tool_index: usize,
    history_count: usize,
) -> String {
    if model.to_lowercase().contains("kimi") {
        // KimiK2 format: functions.{name}:{global_index}
        format!("functions.{}:{}", tool_name, history_count + tool_index)
    } else {
        // Standard OpenAI format: call_{24-char-uuid}
        use uuid::Uuid;
        format!("call_{}", &Uuid::new_v4().simple().to_string()[..24])
    }
}

/// Count the number of tool calls in the request message history (from router.rs:412-424)
pub fn get_history_tool_calls_count(
    request: &crate::protocols::spec::ChatCompletionRequest,
) -> usize {
    request
        .messages
        .iter()
        .filter_map(|msg| {
            if let ChatMessage::Assistant { tool_calls, .. } = msg {
                tool_calls.as_ref().map(|calls| calls.len())
            } else {
                None
            }
        })
        .sum()
}

/// Process a chunk of tokens through the stop decoder (from router.rs:452-482)
pub fn process_chunk_tokens(
    stop_decoder: &mut StopSequenceDecoder,
    token_ids: &[u32],
) -> (String, bool) {
    let mut chunk_text = String::new();
    let mut should_stop = false;

    for &token_id in token_ids {
        match stop_decoder.process_token(token_id).unwrap_or_else(|e| {
            debug!(
                "Error processing token {}: {}. Treating as Held.",
                token_id, e
            );
            SequenceDecoderOutput::Held
        }) {
            SequenceDecoderOutput::Text(text) => {
                chunk_text.push_str(&text);
            }
            SequenceDecoderOutput::Stopped | SequenceDecoderOutput::StoppedWithText(_) => {
                should_stop = true;
                break;
            }
            SequenceDecoderOutput::Held => {
                // Token is being held, continue
            }
        }
    }

    (chunk_text, should_stop)
}
