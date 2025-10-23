//! Conversion utilities for translating between /v1/responses and /v1/chat/completions formats
//!
//! This module implements the conversion approach where:
//! 1. ResponsesRequest → ChatCompletionRequest (for backend processing)
//! 2. ChatCompletionResponse → ResponsesResponse (for client response)
//!
//! This allows the gRPC router to reuse the existing chat pipeline infrastructure
//! without requiring Python backend changes.

use crate::protocols::{
    chat::{ChatCompletionRequest, ChatCompletionResponse, ChatMessage, UserMessageContent},
    common::{FunctionCallResponse, StreamOptions, ToolCall, UsageInfo},
    responses::{
        ResponseContentPart, ResponseInput, ResponseInputOutputItem, ResponseOutputItem,
        ResponseStatus, ResponsesRequest, ResponsesResponse, ResponsesUsage,
    },
};

/// Convert a ResponsesRequest to ChatCompletionRequest for processing through the chat pipeline
///
/// # Conversion Logic
/// - `input` (text/items) → `messages` (chat messages)
/// - `instructions` → system message (prepended)
/// - `max_output_tokens` → `max_completion_tokens`
/// - Tool-related fields are passed through
/// - Response-specific fields (previous_response_id, conversation) are handled by router
pub fn responses_to_chat(req: &ResponsesRequest) -> Result<ChatCompletionRequest, String> {
    let mut messages = Vec::new();

    // 1. Add system message if instructions provided
    if let Some(instructions) = &req.instructions {
        messages.push(ChatMessage::System {
            content: instructions.clone(),
            name: None,
        });
    }

    // 2. Convert input to chat messages
    match &req.input {
        ResponseInput::Text(text) => {
            // Simple text input → user message
            messages.push(ChatMessage::User {
                content: UserMessageContent::Text(text.clone()),
                name: None,
            });
        }
        ResponseInput::Items(items) => {
            // Structured items → convert each to appropriate chat message
            for item in items {
                match item {
                    ResponseInputOutputItem::Message { role, content, .. } => {
                        // Extract text from content parts
                        let text = extract_text_from_content(content);

                        match role.as_str() {
                            "user" => {
                                messages.push(ChatMessage::User {
                                    content: UserMessageContent::Text(text),
                                    name: None,
                                });
                            }
                            "assistant" => {
                                messages.push(ChatMessage::Assistant {
                                    content: Some(text),
                                    name: None,
                                    tool_calls: None,
                                    reasoning_content: None,
                                });
                            }
                            "system" => {
                                messages.push(ChatMessage::System {
                                    content: text,
                                    name: None,
                                });
                            }
                            _ => {
                                // Unknown role, treat as user message
                                messages.push(ChatMessage::User {
                                    content: UserMessageContent::Text(text),
                                    name: None,
                                });
                            }
                        }
                    }
                    ResponseInputOutputItem::FunctionToolCall {
                        id,
                        name,
                        arguments,
                        output,
                        ..
                    } => {
                        // Tool call from history - add as assistant message with tool call
                        // followed by tool response if output exists

                        // Add assistant message with tool_calls (the LLM's decision)
                        messages.push(ChatMessage::Assistant {
                            content: None,
                            name: None,
                            tool_calls: Some(vec![ToolCall {
                                id: id.clone(),
                                tool_type: "function".to_string(),
                                function: FunctionCallResponse {
                                    name: name.clone(),
                                    arguments: Some(arguments.clone()),
                                },
                            }]),
                            reasoning_content: None,
                        });

                        // Add tool result message if output exists
                        if let Some(output_text) = output {
                            messages.push(ChatMessage::Tool {
                                content: output_text.clone(),
                                tool_call_id: id.clone(),
                            });
                        }
                    }
                    ResponseInputOutputItem::Reasoning { content, .. } => {
                        // Reasoning content - add as assistant message with reasoning_content
                        let reasoning_text = content
                            .iter()
                            .map(|c| match c {
                                crate::protocols::responses::ResponseReasoningContent::ReasoningText { text } => {
                                    text.as_str()
                                }
                            })
                            .collect::<Vec<_>>()
                            .join("\n");

                        messages.push(ChatMessage::Assistant {
                            content: None,
                            name: None,
                            tool_calls: None,
                            reasoning_content: Some(reasoning_text),
                        });
                    }
                }
            }
        }
    }

    // Ensure we have at least one message
    if messages.is_empty() {
        return Err("Request must contain at least one message".to_string());
    }

    // 3. Build ChatCompletionRequest
    let is_streaming = req.stream.unwrap_or(false);

    Ok(ChatCompletionRequest {
        messages,
        model: if req.model.is_empty() {
            "default".to_string()
        } else {
            req.model.clone()
        },
        temperature: req.temperature,
        max_completion_tokens: req.max_output_tokens,
        stream: is_streaming,
        stream_options: if is_streaming {
            Some(StreamOptions {
                include_usage: Some(true),
            })
        } else {
            None
        },
        parallel_tool_calls: req.parallel_tool_calls,
        top_logprobs: req.top_logprobs,
        top_p: req.top_p,
        skip_special_tokens: true, // Always skip special tokens // TODO: except for gpt-oss
        // Note: tools and tool_choice will be handled separately for MCP transformation
        tools: None,       // Will be set by caller if needed
        tool_choice: None, // Will be set by caller if needed
        ..Default::default()
    })
}

/// Extract text content from ResponseContentPart array
fn extract_text_from_content(content: &[ResponseContentPart]) -> String {
    content
        .iter()
        .filter_map(|part| match part {
            ResponseContentPart::InputText { text } => Some(text.as_str()),
            ResponseContentPart::OutputText { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

/// Convert a ChatCompletionResponse to ResponsesResponse
///
/// # Conversion Logic
/// - `id` → `response_id_override` if provided, otherwise `chat_resp.id`
/// - `model` → `model` (pass through)
/// - `choices[0].message` → `output` array (convert to ResponseOutputItem::Message)
/// - `choices[0].finish_reason` → determines `status` (stop/length → Completed)
/// - `created` timestamp → `created_at`
pub fn chat_to_responses(
    chat_resp: &ChatCompletionResponse,
    original_req: &ResponsesRequest,
    response_id_override: Option<String>,
) -> Result<ResponsesResponse, String> {
    // Extract the first choice (responses API doesn't support n>1)
    let choice = chat_resp
        .choices
        .first()
        .ok_or_else(|| "Chat response contains no choices".to_string())?;

    // Convert assistant message to output items
    let mut output: Vec<ResponseOutputItem> = Vec::new();

    // Convert message content to output item
    if let Some(content) = &choice.message.content {
        if !content.is_empty() {
            output.push(ResponseOutputItem::Message {
                id: format!("msg_{}", chat_resp.id),
                role: "assistant".to_string(),
                content: vec![ResponseContentPart::OutputText {
                    text: content.clone(),
                    annotations: vec![],
                    logprobs: choice.logprobs.clone(),
                }],
                status: "completed".to_string(),
            });
        }
    }

    // Convert reasoning content if present (O1-style models)
    if let Some(reasoning) = &choice.message.reasoning_content {
        if !reasoning.is_empty() {
            output.push(ResponseOutputItem::Reasoning {
                id: format!("reasoning_{}", chat_resp.id),
                summary: vec![],
                content: vec![
                    crate::protocols::responses::ResponseReasoningContent::ReasoningText {
                        text: reasoning.clone(),
                    },
                ],
                status: Some("completed".to_string()),
            });
        }
    }

    // Convert tool calls if present
    if let Some(tool_calls) = &choice.message.tool_calls {
        for tool_call in tool_calls {
            output.push(ResponseOutputItem::FunctionToolCall {
                id: tool_call.id.clone(),
                name: tool_call.function.name.clone(),
                arguments: tool_call.function.arguments.clone().unwrap_or_default(),
                output: None, // Tool hasn't been executed yet
                status: "in_progress".to_string(),
            });
        }
    }

    // Determine response status based on finish_reason
    let status = match choice.finish_reason.as_deref() {
        Some("stop") | Some("length") => ResponseStatus::Completed,
        Some("tool_calls") => ResponseStatus::InProgress, // Waiting for tool execution
        Some("failed") | Some("error") => ResponseStatus::Failed,
        _ => ResponseStatus::Completed, // Default to completed
    };

    // Convert usage from Usage to UsageInfo, then wrap in ResponsesUsage
    let usage = chat_resp.usage.as_ref().map(|u| {
        let usage_info = UsageInfo {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
            reasoning_tokens: u
                .completion_tokens_details
                .as_ref()
                .and_then(|d| d.reasoning_tokens),
            prompt_tokens_details: None, // Chat response doesn't have this
        };
        ResponsesUsage::Classic(usage_info)
    });

    // Generate response
    Ok(ResponsesResponse {
        id: response_id_override.unwrap_or_else(|| chat_resp.id.clone()),
        object: "response".to_string(),
        created_at: chat_resp.created as i64,
        status,
        error: None,
        incomplete_details: None,
        instructions: original_req.instructions.clone(),
        max_output_tokens: original_req.max_output_tokens,
        model: chat_resp.model.clone(),
        output,
        parallel_tool_calls: original_req.parallel_tool_calls.unwrap_or(true),
        previous_response_id: original_req.previous_response_id.clone(),
        reasoning: None, // TODO: Map reasoning effort if needed
        store: original_req.store.unwrap_or(true),
        temperature: original_req.temperature,
        text: None,
        tool_choice: "auto".to_string(), // TODO: Map from original request
        tools: original_req.tools.clone().unwrap_or_default(),
        top_p: original_req.top_p,
        truncation: None,
        usage,
        user: None, // No user field in chat response
        metadata: original_req.metadata.clone().unwrap_or_default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_input_conversion() {
        let req = ResponsesRequest {
            input: ResponseInput::Text("Hello, world!".to_string()),
            instructions: Some("You are a helpful assistant.".to_string()),
            model: "gpt-4".to_string(),
            temperature: Some(0.7),
            ..Default::default()
        };

        let chat_req = responses_to_chat(&req).unwrap();
        assert_eq!(chat_req.messages.len(), 2); // system + user
        assert_eq!(chat_req.model, "gpt-4");
        assert_eq!(chat_req.temperature, Some(0.7));
    }

    #[test]
    fn test_items_input_conversion() {
        let req = ResponsesRequest {
            input: ResponseInput::Items(vec![
                ResponseInputOutputItem::Message {
                    id: "msg_1".to_string(),
                    role: "user".to_string(),
                    content: vec![ResponseContentPart::InputText {
                        text: "Hello!".to_string(),
                    }],
                    status: None,
                },
                ResponseInputOutputItem::Message {
                    id: "msg_2".to_string(),
                    role: "assistant".to_string(),
                    content: vec![ResponseContentPart::OutputText {
                        text: "Hi there!".to_string(),
                        annotations: vec![],
                        logprobs: None,
                    }],
                    status: None,
                },
            ]),
            ..Default::default()
        };

        let chat_req = responses_to_chat(&req).unwrap();
        assert_eq!(chat_req.messages.len(), 2); // user + assistant
    }

    #[test]
    fn test_empty_input_error() {
        let req = ResponsesRequest {
            input: ResponseInput::Text("".to_string()),
            ..Default::default()
        };

        // Empty text should still create a user message, so this should succeed
        let result = responses_to_chat(&req);
        assert!(result.is_ok());
    }
}
