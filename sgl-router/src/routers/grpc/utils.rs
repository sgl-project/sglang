//! Shared utilities for gRPC routers

use super::ProcessedMessages;
use crate::core::Worker;
use crate::grpc_client::{proto, SglangSchedulerClient};
use crate::protocols::spec::{
    ChatCompletionRequest, ChatLogProbs, ChatLogProbsContent, ChatMessage, FunctionCallResponse,
    StringOrArray, Tool, ToolCall, ToolChoice, ToolChoiceValue, TopLogProb,
};
use crate::tokenizer::chat_template::{ChatTemplateContentFormat, ChatTemplateParams};
use crate::tokenizer::traits::Tokenizer;
use crate::tokenizer::HuggingFaceTokenizer;
pub use crate::tokenizer::StopSequenceDecoder;
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use futures::StreamExt;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tonic::codec::Streaming;
use tracing::{debug, error, warn};
use uuid::Uuid;

/// Get gRPC client from worker, returning appropriate error response on failure
pub async fn get_grpc_client_from_worker(
    worker: &Arc<dyn Worker>,
) -> Result<SglangSchedulerClient, Response> {
    let client_arc = worker
        .get_grpc_client()
        .await
        .map_err(|e| internal_error_message(format!("Failed to get gRPC client: {}", e)))?
        .ok_or_else(|| internal_error_static("Selected worker is not configured for gRPC"))?;

    let client = client_arc.lock().await.clone();
    Ok(client)
}

/// Process tool call arguments in messages
/// Per Transformers docs, tool call arguments in assistant messages should be dicts
pub fn process_tool_call_arguments(messages: &mut [Value]) -> Result<(), String> {
    for msg in messages {
        // Early return if not assistant message
        let role = msg.get("role").and_then(|v| v.as_str());
        if role != Some("assistant") {
            continue;
        }

        // Early return if no tool_calls
        let Some(tool_calls) = msg.get_mut("tool_calls").and_then(|tc| tc.as_array_mut()) else {
            continue;
        };

        // Process each tool call's arguments
        for call in tool_calls {
            let Some(function) = call.get_mut("function") else {
                continue;
            };
            let Some(args) = function.get_mut("arguments") else {
                continue;
            };
            let Some(args_str) = args.as_str() else {
                continue;
            };

            // Parse JSON string to object (like Python json.loads)
            match serde_json::from_str::<Value>(args_str) {
                Ok(parsed) => *args = parsed,
                Err(e) => {
                    return Err(format!(
                        "Failed to parse tool call arguments as JSON: '{}'. Error: {}",
                        args_str, e
                    ))
                }
            }
        }
    }
    Ok(())
}

/// Process messages based on content format for ANY message type
pub fn process_content_format(
    messages: &[ChatMessage],
    content_format: ChatTemplateContentFormat,
) -> Result<Vec<Value>, String> {
    messages
        .iter()
        .map(|message| {
            let mut message_json = serde_json::to_value(message)
                .map_err(|e| format!("Failed to serialize message: {}", e))?;

            if let Some(obj) = message_json.as_object_mut() {
                if let Some(content_value) = obj.get_mut("content") {
                    transform_content_field(content_value, content_format);
                }
            }

            Ok(message_json)
        })
        .collect()
}

/// Transform a single content field based on content format
pub fn transform_content_field(
    content_value: &mut Value,
    content_format: ChatTemplateContentFormat,
) {
    let Some(content_array) = content_value.as_array() else {
        return; // Not multimodal, keep as-is
    };

    match content_format {
        ChatTemplateContentFormat::String => {
            // Extract and join text parts only
            let text_parts: Vec<String> = content_array
                .iter()
                .filter_map(|part| {
                    part.as_object()?
                        .get("type")?
                        .as_str()
                        .filter(|&t| t == "text")
                        .and_then(|_| part.as_object()?.get("text")?.as_str())
                        .map(String::from)
                })
                .collect();

            if !text_parts.is_empty() {
                *content_value = Value::String(text_parts.join(" "));
            }
        }
        ChatTemplateContentFormat::OpenAI => {
            // Replace media URLs with simple type placeholders
            let processed_parts: Vec<Value> = content_array
                .iter()
                .map(|part| {
                    part.as_object()
                        .and_then(|obj| obj.get("type")?.as_str())
                        .and_then(|type_str| match type_str {
                            "image_url" => Some(json!({"type": "image"})),
                            "video_url" => Some(json!({"type": "video"})),
                            "audio_url" => Some(json!({"type": "audio"})),
                            _ => None,
                        })
                        .unwrap_or_else(|| part.clone())
                })
                .collect();

            *content_value = Value::Array(processed_parts);
        }
    }
}

/// Generate tool constraints for structured generation
/// Note: tools should already be filtered if needed (by allowed_tools or specific function)
pub fn generate_tool_constraints(
    tools: &[Tool],
    tool_choice: &Option<ToolChoice>,
    _model: &str,
) -> Option<(String, String)> {
    let choice = tool_choice.as_ref()?;

    match choice {
        // Specific function: Return parameters schema directly
        // tools should already be filtered to contain only the specific function
        ToolChoice::Function { .. } => {
            if tools.is_empty() {
                return None;
            }
            let tool = &tools[0];

            // Return the tool's parameters schema directly (not wrapped in array)
            let params_schema = serde_json::to_string(&tool.function.parameters).ok()?;
            Some(("json_schema".to_string(), params_schema))
        }

        // Required: Array of tool calls with minItems: 1
        ToolChoice::Value(ToolChoiceValue::Required) => {
            let schema = build_required_array_schema(tools)?;
            Some(("json_schema".to_string(), schema))
        }

        // AllowedTools with required mode: tools are already filtered
        ToolChoice::AllowedTools { mode, .. } => {
            if mode == "required" {
                if tools.is_empty() {
                    return None;
                }
                let schema = build_required_array_schema(tools)?;
                Some(("json_schema".to_string(), schema))
            } else {
                // "auto" mode - no constraint needed
                None
            }
        }

        // "auto" or "none" - no constraint
        _ => None,
    }
}

/// Build JSON schema for required tool calls (array with minItems: 1)
/// Includes $defs consolidation from all tools (matching Python's behavior)
pub fn build_required_array_schema(tools: &[Tool]) -> Option<String> {
    // Build anyOf schemas for each tool
    let mut any_of_schemas = Vec::new();
    for tool in tools {
        let tool_schema = json!({
            "properties": {
                "name": {
                    "type": "string",
                    "enum": [tool.function.name]
                },
                "parameters": tool.function.parameters
            },
            "required": ["name", "parameters"]
        });
        any_of_schemas.push(tool_schema);
    }

    // Consolidate $defs from all tools (matching Python's _get_tool_schema_defs)
    let mut all_defs: HashMap<String, Value> = HashMap::new();
    for tool in tools {
        if let Value::Object(params) = &tool.function.parameters {
            if let Some(Value::Object(defs)) = params.get("$defs") {
                for (def_name, def_schema) in defs {
                    if let Some(existing) = all_defs.get(def_name) {
                        // Check for conflicts
                        if existing != def_schema {
                            error!(
                                "Tool definition '{}' has multiple schemas, which is not supported",
                                def_name
                            );
                            return None;
                        }
                    } else {
                        all_defs.insert(def_name.clone(), def_schema.clone());
                    }
                }
            }
        }
    }

    // Build the full array schema
    let mut array_schema = json!({
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "object",
            "anyOf": any_of_schemas
        }
    });

    // Add $defs if any were found (matching Python's behavior)
    if !all_defs.is_empty() {
        if let Value::Object(ref mut schema_obj) = array_schema {
            let defs_value = Value::Object(all_defs.into_iter().collect::<Map<String, Value>>());
            schema_obj.insert("$defs".to_string(), defs_value);
        }
    }

    serde_json::to_string(&array_schema).ok()
}

/// Filter tools based on tool_choice (shared by both routers)
/// Returns a reference to the original body if no filtering needed,
/// otherwise returns a cloned and filtered body
pub fn filter_tools_for_request(
    body: &ChatCompletionRequest,
) -> std::borrow::Cow<'_, ChatCompletionRequest> {
    match &body.tool_choice {
        Some(ToolChoice::AllowedTools { tools: allowed, .. }) if body.tools.is_some() => {
            let mut filtered_body = body.clone();
            let all_tools = filtered_body.tools.as_ref().unwrap();
            let allowed_names: std::collections::HashSet<&str> =
                allowed.iter().map(|t| t.name.as_str()).collect();
            let filtered_tools: Vec<Tool> = all_tools
                .iter()
                .filter(|t| allowed_names.contains(t.function.name.as_str()))
                .cloned()
                .collect();
            filtered_body.tools = Some(filtered_tools);
            std::borrow::Cow::Owned(filtered_body)
        }
        Some(ToolChoice::Function { function, .. }) if body.tools.is_some() => {
            let mut filtered_body = body.clone();
            let all_tools = filtered_body.tools.as_ref().unwrap();
            let filtered_tools: Vec<Tool> = all_tools
                .iter()
                .filter(|t| t.function.name == function.name)
                .cloned()
                .collect();
            filtered_body.tools = Some(filtered_tools);
            std::borrow::Cow::Owned(filtered_body)
        }
        _ => std::borrow::Cow::Borrowed(body), // No filtering needed, use original
    }
}

/// Process chat messages and apply template (shared by both routers)
/// Requires HuggingFace tokenizer with chat template support
pub fn process_chat_messages(
    request: &ChatCompletionRequest,
    tokenizer: &dyn Tokenizer,
) -> Result<ProcessedMessages, String> {
    // Use the tokenizer's chat template - we require HuggingFace tokenizer for gRPC
    let formatted_text = if let Some(hf_tokenizer) =
        tokenizer.as_any().downcast_ref::<HuggingFaceTokenizer>()
    {
        // Get content format and transform messages accordingly
        let content_format = hf_tokenizer.chat_template_content_format();
        let mut transformed_messages = process_content_format(&request.messages, content_format)?;

        // Process tool call arguments in assistant messages
        process_tool_call_arguments(&mut transformed_messages)?;

        // Convert tools to JSON values for template processing
        let tools_json: Option<Vec<Value>> = request
            .tools
            .as_ref()
            .map(|tools| {
                tools
                    .iter()
                    .map(serde_json::to_value)
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()
            .map_err(|e| format!("Failed to serialize tools: {}", e))?;

        // Build template kwargs, merging reasoning_effort if present
        let mut combined_template_kwargs = HashMap::new();

        // Add reasoning_effort if present (like Python does)
        if let Some(reasoning_effort) = &request.reasoning_effort {
            combined_template_kwargs.insert(
                "reasoning_effort".to_string(),
                Value::String(reasoning_effort.clone()),
            );
        }

        // Add any additional template kwargs from request
        if let Some(template_kwargs) = &request.chat_template_kwargs {
            for (key, value) in template_kwargs {
                combined_template_kwargs.insert(key.clone(), value.clone());
            }
        }

        let final_template_kwargs = if combined_template_kwargs.is_empty() {
            None
        } else {
            Some(&combined_template_kwargs)
        };

        let params = ChatTemplateParams {
            add_generation_prompt: true,
            continue_final_message: request.continue_final_message,
            tools: tools_json.as_deref(),
            template_kwargs: final_template_kwargs,
            ..Default::default()
        };

        // Handle assistant prefix for continue_final_message
        let assistant_prefix = if request.continue_final_message
            && !transformed_messages.is_empty()
            && transformed_messages
                .last()
                .and_then(|msg| msg.get("role"))
                .and_then(|v| v.as_str())
                == Some("assistant")
        {
            // Pop the last message to handle it separately
            let last_msg = transformed_messages.pop().unwrap();
            last_msg
                .get("content")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        } else {
            None
        };

        // Apply chat template with the (now possibly shorter) list of messages
        let rendered = hf_tokenizer
            .apply_chat_template(&transformed_messages, params)
            .map_err(|e| format!("Failed to apply chat template: {}", e))?;

        // Append assistant prefix if we have one
        if let Some(prefix) = assistant_prefix {
            format!("{}{}", rendered, prefix)
        } else {
            rendered
        }
    } else {
        return Err(
            "gRPC router requires HuggingFace tokenizer with chat template support".to_string(),
        );
    };

    // Placeholder for multimodal inputs
    let multimodal_inputs = None;

    Ok(ProcessedMessages {
        text: formatted_text,
        multimodal_inputs,
        stop_sequences: request.stop.clone(),
    })
}

/// Error response helpers (shared between regular and PD routers)
pub fn internal_error_static(msg: &'static str) -> Response {
    error!("{}", msg);
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({
            "error": {
                "message": msg,
                "type": "internal_error",
                "code": 500
            }
        })),
    )
        .into_response()
}

pub fn internal_error_message(message: String) -> Response {
    error!("{}", message);
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({
            "error": {
                "message": message,
                "type": "internal_error",
                "code": 500
            }
        })),
    )
        .into_response()
}

pub fn bad_request_error(message: String) -> Response {
    error!("{}", message);
    (
        StatusCode::BAD_REQUEST,
        Json(json!({
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": 400
            }
        })),
    )
        .into_response()
}

pub fn service_unavailable_error(message: String) -> Response {
    warn!("{}", message);
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({
            "error": {
                "message": message,
                "type": "service_unavailable",
                "code": 503
            }
        })),
    )
        .into_response()
}

/// Create a StopSequenceDecoder from stop parameters
pub fn create_stop_decoder(
    tokenizer: &Arc<dyn Tokenizer>,
    stop: Option<&StringOrArray>,
    stop_token_ids: Option<&Vec<u32>>,
    skip_special_tokens: bool,
    no_stop_trim: bool,
) -> StopSequenceDecoder {
    use crate::tokenizer::stop::StopSequenceDecoderBuilder;

    // Extract stop sequences
    let stop_sequences: Vec<String> = match stop {
        Some(StringOrArray::String(s)) => vec![s.clone()],
        Some(StringOrArray::Array(arr)) => arr.clone(),
        None => vec![],
    };

    // Build stop sequence decoder
    let mut builder =
        StopSequenceDecoderBuilder::new(tokenizer.clone()).skip_special_tokens(skip_special_tokens);

    // Add stop sequences (visible if no_stop_trim is true, hidden otherwise)
    for seq in stop_sequences {
        builder = if no_stop_trim {
            builder.visible_stop_sequence(seq)
        } else {
            builder.stop_sequence(seq)
        };
    }

    // Add stop token IDs (visible if no_stop_trim is true, hidden otherwise)
    if let Some(token_ids) = stop_token_ids {
        for &token_id in token_ids {
            builder = if no_stop_trim {
                builder.visible_stop_token(token_id)
            } else {
                builder.stop_token(token_id)
            };
        }
    }

    builder.build()
}

/// Parse tool calls from JSON schema constrained response
pub fn parse_json_schema_response(
    processed_text: &str,
    tool_choice: &Option<ToolChoice>,
) -> (Option<Vec<ToolCall>>, String) {
    match tool_choice {
        Some(ToolChoice::Function { function, .. }) => {
            // Specific function: Parse parameters directly
            match serde_json::from_str::<Value>(processed_text) {
                Ok(params) => {
                    let tool_call = ToolCall {
                        id: format!("call_{}", uuid::Uuid::new_v4()),
                        tool_type: "function".to_string(),
                        function: FunctionCallResponse {
                            name: function.name.clone(),
                            arguments: Some(
                                serde_json::to_string(&params).unwrap_or_else(|_| "{}".to_string()),
                            ),
                        },
                    };
                    (Some(vec![tool_call]), String::new())
                }
                Err(e) => {
                    error!("Failed to parse specific function parameters: {}", e);
                    (None, processed_text.to_string())
                }
            }
        }
        Some(ToolChoice::Value(ToolChoiceValue::Required))
        | Some(ToolChoice::AllowedTools { .. }) => {
            // Required mode: Parse array of tool calls
            match serde_json::from_str::<Vec<Value>>(processed_text) {
                Ok(parsed_array) => {
                    let spec_tool_calls: Vec<ToolCall> = parsed_array
                        .into_iter()
                        .enumerate()
                        .filter_map(|(i, item)| {
                            let obj = item.as_object()?;
                            let name = obj.get("name")?.as_str()?.to_string();
                            let parameters = obj.get("parameters")?;

                            Some(ToolCall {
                                id: format!("call_{}_{}", i, uuid::Uuid::new_v4()),
                                tool_type: "function".to_string(),
                                function: FunctionCallResponse {
                                    name,
                                    arguments: Some(
                                        serde_json::to_string(parameters)
                                            .unwrap_or_else(|_| "{}".to_string()),
                                    ),
                                },
                            })
                        })
                        .collect();
                    (Some(spec_tool_calls), String::new())
                }
                Err(e) => {
                    error!("Failed to parse required tool call array: {}", e);
                    (None, processed_text.to_string())
                }
            }
        }
        _ => (None, processed_text.to_string()),
    }
}

/// Collect responses from a gRPC stream
///
/// This helper processes a gRPC GenerateResponse stream and collects all Complete responses.
/// Used by both regular and PD routers for non-streaming requests.
///
/// # Arguments
/// * `stream` - The gRPC response stream to consume
/// * `worker_name` - Name for logging (e.g., "Prefill", "Decode", "Worker")
///
/// # Returns
/// * `Ok(Vec<GenerateComplete>)` - All complete responses collected from the stream
/// * `Err(Response)` - Error response if the stream fails or returns an error
pub async fn collect_stream_responses(
    mut stream: Streaming<proto::GenerateResponse>,
    worker_name: &str,
) -> Result<Vec<proto::GenerateComplete>, Response> {
    use proto::generate_response::Response::*;

    let mut all_responses = Vec::new();

    while let Some(response) = stream.next().await {
        match response {
            Ok(gen_response) => {
                match gen_response.response {
                    Some(Complete(complete)) => {
                        debug!(
                        "{} completed: prompt_tokens={}, completion_tokens={}, finish_reason={}",
                        worker_name, complete.prompt_tokens, complete.completion_tokens, complete.finish_reason
                    );
                        all_responses.push(complete);
                    }
                    Some(Error(err)) => {
                        error!("{} error: {}", worker_name, err.message);
                        return Err(internal_error_message(format!(
                            "{} generation failed: {}",
                            worker_name, err.message
                        )));
                    }
                    Some(Chunk(chunk)) => {
                        debug!("{} chunk: {} tokens", worker_name, chunk.token_ids.len());
                    }
                    None => {
                        debug!("{}: empty response", worker_name);
                    }
                }
            }
            Err(e) => {
                error!("{} stream error: {:?}", worker_name, e);
                return Err(internal_error_message(format!(
                    "{} stream failed: {}",
                    worker_name, e
                )));
            }
        }
    }

    debug!("{} stream closed", worker_name);
    Ok(all_responses)
}

/// Count the number of tool calls in the request message history
/// This is used for KimiK2 format which needs globally unique indices
pub fn get_history_tool_calls_count(request: &ChatCompletionRequest) -> usize {
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

/// Generate a tool call ID based on model format
///
/// # Arguments
/// * `model` - Model name to determine ID format
/// * `tool_name` - Name of the tool being called
/// * `tool_index` - Index of this tool call within the current message
/// * `history_count` - Number of tool calls in previous messages
///
/// # Returns
/// A unique ID string. KimiK2 uses `functions.{name}:{global_index}`, others use `call_{uuid}`
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
        format!("call_{}", &Uuid::new_v4().simple().to_string()[..24])
    }
}

/// Get the appropriate reasoning parser for a model
///
/// If a parser name is explicitly configured, use that parser.
/// Otherwise, auto-detect based on the model name.
pub fn get_reasoning_parser(
    reasoning_parser_factory: &crate::reasoning_parser::ReasoningParserFactory,
    configured_parser: Option<&String>,
    model: &str,
) -> crate::reasoning_parser::PooledParser {
    use tracing::warn;

    if let Some(parser_name) = configured_parser {
        // Use configured parser if specified
        reasoning_parser_factory
            .registry()
            .get_pooled_parser(parser_name)
            .unwrap_or_else(|| {
                warn!(
                    "Configured reasoning parser '{}' not found, falling back to model-based selection",
                    parser_name
                );
                reasoning_parser_factory.get_pooled(model)
            })
    } else {
        // Auto-detect based on model
        reasoning_parser_factory.get_pooled(model)
    }
}

/// Get the appropriate tool parser for a model
///
/// If a parser name is explicitly configured, use that parser.
/// Otherwise, auto-detect based on the model name.
pub fn get_tool_parser(
    tool_parser_factory: &crate::tool_parser::ToolParserFactory,
    configured_parser: Option<&String>,
    model: &str,
) -> crate::tool_parser::PooledToolParser {
    use tracing::warn;

    if let Some(parser_name) = configured_parser {
        // Use configured parser if specified
        tool_parser_factory
            .registry()
            .get_pooled_parser(parser_name)
            .unwrap_or_else(|| {
                warn!(
                    "Configured tool parser '{}' not found, falling back to model-based selection",
                    parser_name
                );
                tool_parser_factory.get_pooled(model)
            })
    } else {
        // Auto-detect based on model
        tool_parser_factory.get_pooled(model)
    }
}

/// Convert proto::OutputLogProbs to OpenAI ChatLogProbs format
///
/// This function decodes token IDs using the tokenizer and builds the logprobs structure
/// expected by the OpenAI API format.
pub fn convert_proto_to_openai_logprobs(
    proto_logprobs: &proto::OutputLogProbs,
    tokenizer: &Arc<dyn Tokenizer>,
) -> Result<ChatLogProbs, String> {
    let mut content_items = Vec::new();

    // Decode token IDs to text (always with skip_special_tokens=false for logprobs)
    let token_texts: Vec<String> = proto_logprobs
        .token_ids
        .iter()
        .map(|&token_id| {
            tokenizer
                .decode(&[token_id as u32], false)
                .unwrap_or_else(|_| format!("<token_{}>", token_id))
        })
        .collect();

    // Build ChatLogProbsContent for each token (consume iterator to avoid clones)
    for (i, (&logprob, token_text)) in proto_logprobs
        .token_logprobs
        .iter()
        .zip(token_texts.into_iter())
        .enumerate()
    {
        let bytes = Some(token_text.as_bytes().to_vec());

        // Build top_logprobs for this position
        let mut top_logprobs = Vec::new();
        if let Some(top_logprobs_entry) = proto_logprobs.top_logprobs.get(i) {
            // Decode top token IDs (always with skip_special_tokens=false)
            let top_token_texts: Vec<String> = top_logprobs_entry
                .token_ids
                .iter()
                .map(|&tid| {
                    tokenizer
                        .decode(&[tid as u32], false)
                        .unwrap_or_else(|_| format!("<token_{}>", tid))
                })
                .collect();

            for (j, (&top_logprob, &_top_token_id)) in top_logprobs_entry
                .values
                .iter()
                .zip(top_logprobs_entry.token_ids.iter())
                .enumerate()
            {
                if let Some(top_token_text) = top_token_texts.get(j) {
                    top_logprobs.push(TopLogProb {
                        token: top_token_text.clone(),
                        logprob: top_logprob,
                        bytes: Some(top_token_text.as_bytes().to_vec()),
                    });
                }
            }
        }

        content_items.push(ChatLogProbsContent {
            token: token_text,
            logprob,
            bytes,
            top_logprobs,
        });
    }

    Ok(ChatLogProbs::Detailed {
        content: (!content_items.is_empty()).then_some(content_items),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::spec::{ChatMessage, ContentPart, ImageUrl, UserMessageContent};
    use crate::tokenizer::chat_template::ChatTemplateContentFormat;
    use serde_json::json;

    #[test]
    fn test_transform_messages_string_format() {
        let messages = vec![ChatMessage::User {
            role: "user".to_string(),
            content: UserMessageContent::Parts(vec![
                ContentPart::Text {
                    text: "Hello".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example.com/image.jpg".to_string(),
                        detail: None,
                    },
                },
                ContentPart::Text {
                    text: "World".to_string(),
                },
            ]),
            name: None,
        }];

        let result = process_content_format(&messages, ChatTemplateContentFormat::String).unwrap();

        assert_eq!(result.len(), 1);
        let transformed_message = &result[0];

        // Should flatten multimodal content to text only
        assert_eq!(
            transformed_message["content"].as_str().unwrap(),
            "Hello World"
        );
        assert_eq!(transformed_message["role"].as_str().unwrap(), "user");
    }

    #[test]
    fn test_transform_messages_openai_format() {
        let messages = vec![ChatMessage::User {
            role: "user".to_string(),
            content: UserMessageContent::Parts(vec![
                ContentPart::Text {
                    text: "Describe this image:".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example.com/image.jpg".to_string(),
                        detail: Some("high".to_string()),
                    },
                },
            ]),
            name: None,
        }];

        let result = process_content_format(&messages, ChatTemplateContentFormat::OpenAI).unwrap();

        assert_eq!(result.len(), 1);
        let transformed_message = &result[0];

        // Should replace media URLs with simple type placeholders
        let content_array = transformed_message["content"].as_array().unwrap();
        assert_eq!(content_array.len(), 2);

        // Text part should remain unchanged
        assert_eq!(content_array[0]["type"], "text");
        assert_eq!(content_array[0]["text"], "Describe this image:");

        // Image part should be replaced with simple type placeholder
        assert_eq!(content_array[1], json!({"type": "image"}));
    }

    #[test]
    fn test_transform_messages_simple_string_content() {
        let messages = vec![ChatMessage::User {
            role: "user".to_string(),
            content: UserMessageContent::Text("Simple text message".to_string()),
            name: None,
        }];

        let result = process_content_format(&messages, ChatTemplateContentFormat::String).unwrap();

        assert_eq!(result.len(), 1);
        let transformed_message = &result[0];

        // Simple string content should remain unchanged
        assert_eq!(
            transformed_message["content"].as_str().unwrap(),
            "Simple text message"
        );
    }

    #[test]
    fn test_transform_messages_multiple_messages() {
        let messages = vec![
            ChatMessage::System {
                role: "system".to_string(),
                content: "System prompt".to_string(),
                name: None,
            },
            ChatMessage::User {
                role: "user".to_string(),
                content: UserMessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "User message".to_string(),
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: "https://example.com/image.jpg".to_string(),
                            detail: None,
                        },
                    },
                ]),
                name: None,
            },
        ];

        let result = process_content_format(&messages, ChatTemplateContentFormat::String).unwrap();

        assert_eq!(result.len(), 2);

        // System message should remain unchanged
        assert_eq!(result[0]["role"].as_str().unwrap(), "system");
        assert_eq!(result[0]["content"].as_str().unwrap(), "System prompt");

        // User message should be flattened to text only
        assert_eq!(result[1]["role"].as_str().unwrap(), "user");
        assert_eq!(result[1]["content"].as_str().unwrap(), "User message");
    }

    #[test]
    fn test_transform_messages_empty_text_parts() {
        let messages = vec![ChatMessage::User {
            role: "user".to_string(),
            content: UserMessageContent::Parts(vec![ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "https://example.com/image.jpg".to_string(),
                    detail: None,
                },
            }]),
            name: None,
        }];

        let result = process_content_format(&messages, ChatTemplateContentFormat::String).unwrap();

        assert_eq!(result.len(), 1);
        let transformed_message = &result[0];

        // Should keep original multimodal content when no text parts exist
        assert!(transformed_message["content"].is_array());
    }

    #[test]
    fn test_transform_messages_mixed_content_types() {
        let messages = vec![
            ChatMessage::User {
                role: "user".to_string(),
                content: UserMessageContent::Text("Plain text".to_string()),
                name: None,
            },
            ChatMessage::User {
                role: "user".to_string(),
                content: UserMessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "With image".to_string(),
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: "https://example.com/image.jpg".to_string(),
                            detail: Some("low".to_string()),
                        },
                    },
                ]),
                name: None,
            },
        ];

        let result_string =
            process_content_format(&messages, ChatTemplateContentFormat::String).unwrap();

        assert_eq!(result_string.len(), 2);
        assert_eq!(result_string[0]["content"].as_str().unwrap(), "Plain text");
        assert_eq!(result_string[1]["content"].as_str().unwrap(), "With image");

        let result_openai =
            process_content_format(&messages, ChatTemplateContentFormat::OpenAI).unwrap();

        assert_eq!(result_openai.len(), 2);
        assert_eq!(result_openai[0]["content"].as_str().unwrap(), "Plain text");

        let content_array = result_openai[1]["content"].as_array().unwrap();
        assert_eq!(content_array.len(), 2);
        assert_eq!(content_array[0]["type"], "text");
        assert_eq!(content_array[1], json!({"type": "image"}));
    }
}
