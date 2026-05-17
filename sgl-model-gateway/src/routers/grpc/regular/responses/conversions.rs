//! Conversion utilities for translating between /v1/responses and /v1/chat/completions formats
//!
//! This module implements the conversion approach where:
//! 1. ResponsesRequest → ChatCompletionRequest (for backend processing)
//! 2. ChatCompletionResponse → ResponsesResponse (for client response)
//!
//! This allows the gRPC router to reuse the existing chat pipeline infrastructure
//! without requiring Python backend changes.

use crate::{
    protocols::{
        chat::{ChatCompletionRequest, ChatCompletionResponse, ChatMessage, MessageContent},
        common::{
            FunctionCallResponse, JsonSchemaFormat, ResponseFormat, StreamOptions, ToolCall,
            UsageInfo,
        },
        responses::{
            ResponseContentPart, ResponseInput, ResponseInputOutputItem, ResponseOutputItem,
            ResponseReasoningContent::ReasoningText, ResponseStatus, ResponsesRequest,
            ResponsesResponse, ResponsesUsage, StringOrContentParts, TextConfig, TextFormat,
        },
        UNKNOWN_MODEL_ID,
    },
    routers::grpc::common::responses::utils::extract_tools_from_response_tools,
};

/// Convert a ResponsesRequest to ChatCompletionRequest for processing through the chat pipeline
///
/// # Conversion Logic
/// - `input` (text/items) → `messages` (chat messages)
/// - `instructions` → system message (prepended)
/// - `max_output_tokens` → `max_completion_tokens`
/// - `tools` → function tools extracted from ResponseTools
/// - `tool_choice` → passed through from request
/// - Response-specific fields (previous_response_id, conversation) are handled by router
pub(crate) fn responses_to_chat(req: &ResponsesRequest) -> Result<ChatCompletionRequest, String> {
    let mut messages = Vec::new();

    // 1. Add system message if instructions provided
    if let Some(instructions) = &req.instructions {
        messages.push(ChatMessage::System {
            content: MessageContent::Text(instructions.clone()),
            name: None,
        });
    }

    // 2. Convert input to chat messages
    match &req.input {
        ResponseInput::Text(text) => {
            // Simple text input → user message
            messages.push(ChatMessage::User {
                content: MessageContent::Text(text.clone()),
                name: None,
            });
        }
        ResponseInput::Items(items) => {
            // Structured items → convert each to appropriate chat message.
            // Consecutive FunctionToolCall items are grouped into a single
            // assistant message with multiple tool_calls so that backends
            // that require parallel tool calls in one message work correctly.
            let mut pending_tool_calls: Vec<ToolCall> = Vec::new();
            let mut pending_tool_outputs: Vec<(String, String)> = Vec::new();

            /// Flush accumulated parallel tool calls into `messages`.
            fn flush_tool_calls(
                messages: &mut Vec<ChatMessage>,
                pending_tool_calls: &mut Vec<ToolCall>,
                pending_tool_outputs: &mut Vec<(String, String)>,
            ) {
                if pending_tool_calls.is_empty() {
                    return;
                }
                messages.push(ChatMessage::Assistant {
                    content: None,
                    name: None,
                    tool_calls: Some(std::mem::take(pending_tool_calls)),
                    reasoning_content: None,
                });
                for (call_id, output_text) in pending_tool_outputs.drain(..) {
                    messages.push(ChatMessage::Tool {
                        content: MessageContent::Text(output_text),
                        tool_call_id: call_id,
                    });
                }
            }

            for item in items {
                match item {
                    ResponseInputOutputItem::FunctionToolCall {
                        call_id,
                        name,
                        arguments,
                        output,
                        ..
                    } => {
                        // Accumulate — will be flushed when a non-tool-call
                        // item is encountered or at the end.
                        pending_tool_calls.push(ToolCall {
                            id: call_id.clone(),
                            tool_type: "function".to_string(),
                            function: FunctionCallResponse {
                                name: name.clone(),
                                arguments: Some(arguments.clone()),
                            },
                        });
                        if let Some(output_text) = output {
                            pending_tool_outputs.push((call_id.clone(), output_text.clone()));
                        }
                    }
                    other => {
                        // Flush any pending parallel tool calls before
                        // processing a non-tool-call item.
                        flush_tool_calls(
                            &mut messages,
                            &mut pending_tool_calls,
                            &mut pending_tool_outputs,
                        );

                        match other {
                            ResponseInputOutputItem::SimpleInputMessage {
                                content, role, ..
                            } => {
                                let text = match content {
                                    StringOrContentParts::String(s) => s.clone(),
                                    StringOrContentParts::Array(parts) => parts
                                        .iter()
                                        .filter_map(|part| match part {
                                            ResponseContentPart::InputText { text } => {
                                                Some(text.as_str())
                                            }
                                            _ => None,
                                        })
                                        .collect::<Vec<_>>()
                                        .join(" "),
                                };
                                messages.push(role_to_chat_message(role.as_str(), text));
                            }
                            ResponseInputOutputItem::Message { role, content, .. } => {
                                let text = extract_text_from_content(content);
                                messages.push(role_to_chat_message(role.as_str(), text));
                            }
                            ResponseInputOutputItem::Reasoning { content, .. } => {
                                let reasoning_text = content
                                    .iter()
                                    .map(|c| match c {
                                        ReasoningText { text } => text.as_str(),
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
                            ResponseInputOutputItem::FunctionCallOutput {
                                call_id, output, ..
                            } => {
                                messages.push(ChatMessage::Tool {
                                    content: MessageContent::Text(output.clone()),
                                    tool_call_id: call_id.clone(),
                                });
                            }
                            // FunctionToolCall is handled in the outer match arm
                            ResponseInputOutputItem::FunctionToolCall { .. } => unreachable!(),
                        }
                    }
                }
            }

            // Flush any remaining tool calls at end of items
            flush_tool_calls(
                &mut messages,
                &mut pending_tool_calls,
                &mut pending_tool_outputs,
            );
        }
    }

    // Ensure we have at least one message
    if messages.is_empty() {
        return Err("Request must contain at least one message".to_string());
    }

    // 3. Extract function tools from ResponseTools
    // Only function tools are extracted here (include_mcp: false).
    // MCP tools are merged later by the tool loop (see tool_loop.rs:prepare_chat_tools_and_choice)
    // before the chat pipeline, where tool_choice constraints are applied to ALL tools combined.
    let function_tools = extract_tools_from_response_tools(req.tools.as_deref(), false);
    let tools = if function_tools.is_empty() {
        None
    } else {
        Some(function_tools)
    };

    // 4. Build ChatCompletionRequest
    let is_streaming = req.stream.unwrap_or(false);

    Ok(ChatCompletionRequest {
        messages,
        model: if req.model.is_empty() {
            UNKNOWN_MODEL_ID.to_string()
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
        skip_special_tokens: true,
        tools,
        tool_choice: req.tool_choice.clone(),
        response_format: map_text_to_response_format(&req.text),
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

/// Convert role and text to ChatMessage
fn role_to_chat_message(role: &str, text: String) -> ChatMessage {
    match role {
        "user" => ChatMessage::User {
            content: MessageContent::Text(text),
            name: None,
        },
        "assistant" => ChatMessage::Assistant {
            content: Some(MessageContent::Text(text)),
            name: None,
            tool_calls: None,
            reasoning_content: None,
        },
        "system" => ChatMessage::System {
            content: MessageContent::Text(text),
            name: None,
        },
        other => {
            tracing::warn!(
                role = other,
                "unknown message role in responses input, treating as user"
            );
            ChatMessage::User {
                content: MessageContent::Text(text),
                name: None,
            }
        }
    }
}

/// Map TextConfig from Responses API to ResponseFormat for Chat API
///
/// Converts the structured output configuration from the Responses API format
/// to the Chat API format for non-Harmony models.
fn map_text_to_response_format(text: &Option<TextConfig>) -> Option<ResponseFormat> {
    let text_config = text.as_ref()?;
    let format = text_config.format.as_ref()?;

    match format {
        TextFormat::Text => Some(ResponseFormat::Text),
        TextFormat::JsonObject => Some(ResponseFormat::JsonObject),
        TextFormat::JsonSchema {
            name,
            schema,
            description: _,
            strict,
        } => Some(ResponseFormat::JsonSchema {
            json_schema: JsonSchemaFormat {
                name: name.clone(),
                schema: schema.clone(),
                strict: *strict,
            },
        }),
    }
}

/// Convert a ChatCompletionResponse to ResponsesResponse
///
/// # Conversion Logic
/// - `id` → `response_id_override` if provided, otherwise `chat_resp.id`
/// - `model` → `model` (pass through)
/// - `choices[0].message` → `output` array (convert to ResponseOutputItem::Message)
/// - `choices[0].finish_reason` → determines `status` (stop/length → Completed)
/// - `created` timestamp → `created_at`
pub(crate) fn chat_to_responses(
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
                content: vec![ReasoningText {
                    text: reasoning.clone(),
                }],
                status: Some("completed".to_string()),
            });
        }
    }

    // Convert tool calls if present
    if let Some(tool_calls) = &choice.message.tool_calls {
        for tool_call in tool_calls {
            output.push(ResponseOutputItem::FunctionToolCall {
                id: tool_call.id.clone(),
                call_id: tool_call.id.clone(),
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
    let response_id = response_id_override.unwrap_or_else(|| chat_resp.id.clone());
    Ok(ResponsesResponse::builder(&response_id, &chat_resp.model)
        .copy_from_request(original_req)
        .created_at(chat_resp.created as i64)
        .status(status)
        .output(output)
        .maybe_text(original_req.text.clone())
        .maybe_usage(usage)
        .build())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::protocols::{
        common::{Function, ToolChoice, ToolChoiceValue},
        responses::{ResponseTool, ResponseToolType},
    };

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

    #[test]
    fn test_function_tools_serialize_like_chat_completions_tools() {
        let req = ResponsesRequest {
            input: ResponseInput::Text("What is the weather in Berlin?".to_string()),
            model: "Qwen/Qwen2.5-3B-Instruct".to_string(),
            max_output_tokens: Some(128),
            parallel_tool_calls: Some(true),
            tool_choice: Some(ToolChoice::Value(ToolChoiceValue::Required)),
            tools: Some(vec![ResponseTool {
                r#type: ResponseToolType::Function,
                function: Some(Function {
                    name: "get_weather".to_string(),
                    description: Some("Get the weather for a city.".to_string()),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "location": { "type": "string" }
                        },
                        "required": ["location"]
                    }),
                    strict: None,
                }),
                server_url: None,
                authorization: None,
                server_label: None,
                server_description: None,
                require_approval: None,
                allowed_tools: None,
            }]),
            ..Default::default()
        };

        let chat_req = responses_to_chat(&req).unwrap();
        let serialized = serde_json::to_value(&chat_req).unwrap();

        assert_eq!(serialized["tool_choice"], "required");
        assert_eq!(serialized["parallel_tool_calls"], true);
        assert_eq!(serialized["tools"][0]["type"], "function");
        assert_eq!(serialized["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(
            serialized["tools"][0]["function"]["parameters"]["required"][0],
            "location"
        );
    }

    #[test]
    fn test_streaming_request_shape_matches_manual_chat_request() {
        let req = ResponsesRequest {
            input: ResponseInput::Text("Calculate 42 * 17. Use the tool.".to_string()),
            model: "Qwen/Qwen2.5-3B-Instruct".to_string(),
            max_output_tokens: Some(128),
            parallel_tool_calls: Some(true),
            stream: Some(true),
            temperature: Some(0.0),
            tool_choice: Some(ToolChoice::Value(ToolChoiceValue::Required)),
            tools: Some(vec![ResponseTool {
                r#type: ResponseToolType::Function,
                function: Some(Function {
                    name: "calculate".to_string(),
                    description: Some("Perform a mathematical calculation.".to_string()),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "expression": { "type": "string" }
                        },
                        "required": ["expression"]
                    }),
                    strict: None,
                }),
                server_url: None,
                authorization: None,
                server_label: None,
                server_description: None,
                require_approval: None,
                allowed_tools: None,
            }]),
            ..Default::default()
        };

        let converted = responses_to_chat(&req).unwrap();
        let manual = ChatCompletionRequest {
            messages: vec![ChatMessage::User {
                content: MessageContent::Text("Calculate 42 * 17. Use the tool.".to_string()),
                name: None,
            }],
            model: "Qwen/Qwen2.5-3B-Instruct".to_string(),
            max_completion_tokens: Some(128),
            parallel_tool_calls: Some(true),
            stream: true,
            stream_options: Some(StreamOptions {
                include_usage: Some(true),
            }),
            temperature: Some(0.0),
            tool_choice: Some(ToolChoice::Value(ToolChoiceValue::Required)),
            tools: Some(vec![crate::protocols::common::Tool {
                tool_type: "function".to_string(),
                function: Function {
                    name: "calculate".to_string(),
                    description: Some("Perform a mathematical calculation.".to_string()),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "expression": { "type": "string" }
                        },
                        "required": ["expression"]
                    }),
                    strict: None,
                },
            }]),
            skip_special_tokens: true,
            ..Default::default()
        };

        assert_eq!(
            serde_json::to_value(&converted).unwrap(),
            serde_json::to_value(&manual).unwrap(),
        );
    }

    #[test]
    fn test_historical_function_tool_call_uses_call_id_for_chat_resume() {
        let req = ResponsesRequest {
            input: ResponseInput::Items(vec![ResponseInputOutputItem::FunctionToolCall {
                id: "fc_item_123".to_string(),
                call_id: "call_abc".to_string(),
                name: "calculate".to_string(),
                arguments: r#"{"expression":"2+2"}"#.to_string(),
                output: Some(r#"{"result":4}"#.to_string()),
                status: Some("completed".to_string()),
            }]),
            model: "mock-model".to_string(),
            ..Default::default()
        };

        let converted = responses_to_chat(&req).unwrap();
        assert_eq!(converted.messages.len(), 2);

        match &converted.messages[0] {
            ChatMessage::Assistant {
                tool_calls: Some(tool_calls),
                ..
            } => {
                assert_eq!(tool_calls.len(), 1);
                assert_eq!(tool_calls[0].id, "call_abc");
                assert_eq!(tool_calls[0].function.name, "calculate");
            }
            other => panic!("expected assistant tool call message, got {:?}", other),
        }

        match &converted.messages[1] {
            ChatMessage::Tool { tool_call_id, .. } => {
                assert_eq!(tool_call_id, "call_abc");
            }
            other => panic!("expected tool response message, got {:?}", other),
        }
    }
}
