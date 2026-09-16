//! runtime.v1's JSON-carrying RPCs reuse the same OpenAI operations as HTTP.
//! Only protobuf framing and gRPC support/error policy belong here.

use super::{GrpcService, ResponseStream, convert::ConvertError, response};
use crate::openai::{self, OpenAiResponse};
use futures::StreamExt;
use serde_json::Value;
use sglang_grpc_types::proto;
use tonic::{Response, Status};

impl GrpcService {
    pub(super) async fn openai_rpc(
        &self,
        request: proto::OpenAiRequest,
        chat: bool,
    ) -> Result<Response<ResponseStream<proto::OpenAiStreamChunk>>, Status> {
        let value = validate_request(request, chat)?;
        if chat && self.openai.chat_formatter.is_none() {
            return Err(Status::unimplemented(
                "this model has no usable chat template",
            ));
        }
        if self.openai.server_args.skip_tokenizer_init
            && (chat || completion_needs_tokenizer(&value))
        {
            return Err(Status::unimplemented(
                "the requested operation requires a tokenizer",
            ));
        }
        if chat
            && value
                .get("tools")
                .and_then(Value::as_array)
                .is_some_and(|v| !v.is_empty())
            && value.get("tool_choice").and_then(Value::as_str) != Some("none")
            && self.openai.server_args.tool_call_parser.is_none()
        {
            return Err(Status::unimplemented(
                "tool calls require --tool-call-parser",
            ));
        }
        let state = self.openai.clone();
        let operation = async move {
            if chat {
                let request = serde_json::from_value(value)
                    .map_err(|e| Status::invalid_argument(e.to_string()))?;
                Ok::<_, Status>(openai::chat_completions(&state, request).await)
            } else {
                let request = serde_json::from_value(value)
                    .map_err(|e| Status::invalid_argument(e.to_string()))?;
                Ok(openai::completions(&state, request).await)
            }
        };
        let response = tokio::time::timeout(self.config.response_timeout, operation)
            .await
            .map_err(|_| Status::deadline_exceeded("OpenAI operation timed out"))??;
        let timeout = self.config.response_timeout;
        let stream: ResponseStream<_> = match response {
            OpenAiResponse::Error {
                code,
                message,
                kind,
                ..
            } => {
                return Err(match kind {
                    Some(kind) => response::status_from_kind(kind, message),
                    None => status(code.as_u16(), message),
                });
            }
            OpenAiResponse::Json(bytes) => Box::pin(futures::stream::once(async move {
                Ok(proto::OpenAiStreamChunk {
                    json_chunk: bytes,
                    finished: true,
                })
            })),
            OpenAiResponse::Stream(mut source) => Box::pin(async_stream::stream! {
                let terminal = loop {
                    match tokio::time::timeout(timeout, source.next()).await {
                        Ok(Some(item)) if item.data == "[DONE]" => {
                            break Ok(proto::OpenAiStreamChunk { json_chunk: Vec::new(), finished: true });
                        }
                        Ok(Some(item)) => {
                            let data = item.data;
                            let value: Value = match serde_json::from_str(&data) {
                                Ok(value) => value,
                                Err(error) => {
                                    break Err(Status::internal(format!("invalid OpenAI response: {error}")));
                                }
                            };
                            if let Some(error) = value.get("error") {
                                let code = error.get("code").and_then(Value::as_u64)
                                    .and_then(|v| u16::try_from(v).ok()).unwrap_or(500);
                                let message = error.get("message").and_then(Value::as_str)
                                    .unwrap_or("OpenAI generation failed").to_owned();
                                break Err(match item.error_kind {
                                    Some(kind) => response::status_from_kind(kind, message),
                                    None => status(code, message),
                                });
                            }
                            yield Ok(proto::OpenAiStreamChunk { json_chunk: data.into_bytes(), finished: false });
                        }
                        Ok(None) => {
                            break Err(Status::internal("OpenAI response truncated before completion"));
                        }
                        Err(_) => {
                            break Err(Status::deadline_exceeded("OpenAI response chunk timed out"));
                        }
                    }
                };
                // Release the owned FrontendCalls immediately, including on an
                // error, without requiring the client to poll this stream again.
                drop(source);
                yield terminal;
            }),
        };
        Ok(Response::new(stream))
    }
}

fn status(code: u16, message: String) -> Status {
    match code {
        400 | 422 => Status::invalid_argument(message),
        401 => Status::unauthenticated(message),
        403 => Status::permission_denied(message),
        404 => Status::not_found(message),
        409 | 412 => Status::failed_precondition(message),
        429 => Status::resource_exhausted(message),
        499 => Status::cancelled(message),
        501 => Status::unimplemented(message),
        503 => Status::unavailable(message),
        408 | 504 => Status::deadline_exceeded(message),
        _ => Status::internal(message),
    }
}

fn completion_needs_tokenizer(value: &Value) -> bool {
    value.get("echo").and_then(Value::as_bool) == Some(true)
        || value.get("prompt").is_some_and(|v| {
            v.is_string() || v.as_array().is_some_and(|a| a.iter().any(Value::is_string))
        })
}

/// Serde's OpenAI types accept extra fields. For gRPC, reject unimplemented
/// features before deserialization can discard them. HTTP keeps its current
/// acceptance policy. These lists describe fields actually consumed by Rust.
fn validate_request(request: proto::OpenAiRequest, chat: bool) -> Result<Value, ConvertError> {
    if !request.trace_headers.is_empty() {
        return Err(ConvertError::Unsupported(
            "trace_headers is not supported by the Rust frontend".into(),
        ));
    }
    let value: Value = serde_json::from_slice(&request.json_body)
        .map_err(|e| ConvertError::InvalidArgument(e.to_string()))?;
    let object = value
        .as_object()
        .ok_or_else(|| ConvertError::InvalidArgument("expected a JSON object".into()))?;
    let common = [
        "model",
        "stream",
        "stream_options",
        "n",
        "max_tokens",
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "seed",
        "logit_bias",
        "logprobs",
    ];
    let specific: &[&str] = if chat {
        &[
            "messages",
            "max_completion_tokens",
            "top_logprobs",
            "response_format",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "service_tier",
            "modalities",
        ]
    } else {
        &["prompt", "echo", "best_of"]
    };
    for (key, value) in object {
        if !value.is_null() && !common.contains(&key.as_str()) && !specific.contains(&key.as_str())
        {
            return Err(ConvertError::Unsupported(format!(
                "{key} is not supported by the Rust frontend"
            )));
        }
    }
    if !chat
        && object
            .get("best_of")
            .is_some_and(|v| !v.is_null() && v.as_u64() != Some(1))
    {
        return Err(ConvertError::Unsupported(
            "best_of values other than 1 are not supported".into(),
        ));
    }
    if chat
        && (object.get("messages").is_some_and(openai::contains_media)
            || object
                .get("modalities")
                .is_some_and(|v| !v.is_null() && *v != serde_json::json!(["text"])))
    {
        return Err(ConvertError::Unsupported(
            "multimodal chat is not supported by the Rust frontend".into(),
        ));
    }
    if let Some(options) = object.get("stream_options").and_then(Value::as_object) {
        for (key, value) in options {
            if !value.is_null()
                && key != "include_usage"
                && (chat || key != "continuous_usage_stats")
            {
                return Err(ConvertError::Unsupported(format!(
                    "stream_options.{key} is not supported"
                )));
            }
        }
    }
    if chat {
        validate_chat_features(&value)?;
    }
    Ok(value)
}

fn reject_unknown_fields(
    value: &Value,
    allowed: &[&str],
    context: &str,
) -> Result<(), ConvertError> {
    if let Some(object) = value.as_object() {
        for (key, value) in object {
            if !value.is_null() && !allowed.contains(&key.as_str()) {
                return Err(ConvertError::Unsupported(format!(
                    "{context}.{key} is not supported"
                )));
            }
        }
    }
    Ok(())
}

fn validate_chat_features(value: &Value) -> Result<(), ConvertError> {
    if let Some(messages) = value.get("messages").and_then(Value::as_array) {
        for message in messages {
            // Content is subsequently validated by the existing renderer.
            // Reject extensions that Serde would otherwise silently discard.
            reject_unknown_fields(
                message,
                &[
                    "role",
                    "content",
                    "name",
                    "tool_calls",
                    // Historical assistant calls are supported by compatible
                    // chat templates; this is not the top-level request option.
                    "function_call",
                    "tool_call_id",
                    "reasoning_content",
                    "reasoning",
                    "refusal",
                ],
                "messages[]",
            )?;
        }
    }
    if let Some(format) = value.get("response_format") {
        reject_unknown_fields(format, &["type", "json_schema"], "response_format")?;
        if let Some(kind) = format.get("type").and_then(Value::as_str)
            && !matches!(kind, "text" | "json_object" | "json_schema")
        {
            return Err(ConvertError::Unsupported(format!(
                "response_format type {kind} is not supported"
            )));
        }
    }
    if let Some(tools) = value.get("tools").and_then(Value::as_array) {
        for tool in tools {
            reject_unknown_fields(tool, &["type", "function"], "tools[]")?;
            if let Some(kind) = tool.get("type").and_then(Value::as_str)
                && kind != "function"
            {
                return Err(ConvertError::Unsupported(format!(
                    "tool type {kind} is not supported"
                )));
            }
            if let Some(function) = tool.get("function") {
                reject_unknown_fields(
                    function,
                    &["name", "description", "parameters", "strict"],
                    "tools[].function",
                )?;
            }
        }
    }
    Ok(())
}
