//! OpenAI Responses adapter shared by the HTTP route and the gRPC service,
//! the stateless, non-harmony path of Python's `OpenAIServingResponses`: a
//! Responses request becomes one chat request served by the chat core, and
//! the chat reply or stream is re-shaped into Responses output items and
//! events. The response store (`--enable-response-store`), gpt-oss harmony
//! rendering and the built-in tool servers are not served; the requests
//! that need them get the refusals Python gives without them.

use std::collections::HashSet;
use std::pin::Pin;

use axum::http::StatusCode;
use futures::{Stream, StreamExt};
use serde_json::{Map, Value, json};
use sglang_api_types::api::v1 as api;
use sglang_processor::dynamo_protocols::types::{
    ChatCompletionMessageContent, CreateChatCompletionRequest, CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
};
use sglang_processor::{ReasoningOptions, dynamo_tool_choice};
use tonic::{Code, Status};

use super::core::{CoreError, ReceivedAs};
use super::handlers::{CoreCall, CoreHandle};
use super::http::openai::chat::{
    OutputParsing, SamplingDefaults, chat_sampling, prepare_chat_prompt, starts_in_reasoning,
    unary_chat_response,
};
use super::http::openai::pd_routing::PDRoutingFields;
use super::http::openai::{ChatFormatter, unix_seconds_u32};
use crate::message::config::ServerArgs;
use crate::message::ids::Rid;
use crate::message::request::{GenerateRequest, RenderedChat};

/// One SSE event: its `type` (the SSE event name) and its JSON payload.
pub(crate) type ResponsesEvent = (String, Value);
pub(crate) type Events = Pin<Box<dyn Stream<Item = ResponsesEvent> + Send + 'static>>;

/// The per-listener facts a Responses call needs; both transports build one
/// from their own state.
pub(crate) struct Responses<'a> {
    pub(crate) core: &'a CoreHandle,
    pub(crate) chat_formatter: Option<&'a ChatFormatter>,
    pub(crate) server_args: &'a ServerArgs,
}

/// Python's Responses error: `{"error": {"message", "type", "param",
/// "code"}}` with `code` the HTTP status.
#[derive(Debug)]
pub(crate) struct ResponsesFailure {
    pub(crate) status: StatusCode,
    pub(crate) kind: &'static str,
    pub(crate) param: Option<&'static str>,
    pub(crate) message: String,
}

impl ResponsesFailure {
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            kind: "invalid_request_error",
            param: None,
            message: message.into(),
        }
    }

    fn with_param(message: impl Into<String>, param: &'static str) -> Self {
        Self {
            param: Some(param),
            ..Self::invalid(message)
        }
    }

    /// The framework's validation 400: `type` is the status phrase.
    pub(crate) fn bad_request(message: impl Into<String>) -> Self {
        Self {
            kind: "Bad Request",
            ..Self::invalid(message)
        }
    }

    fn from_core(error: CoreError) -> Self {
        Self {
            status: error.http_status(),
            ..Self::invalid(error.to_string())
        }
    }

    /// Python's `_response_store_disabled_error`.
    pub(crate) fn store_disabled(param: &'static str) -> Self {
        Self::with_param(
            "Response store is disabled. Stateful Responses require --enable-response-store on \
             a standalone server; response storage is unavailable in PD mode.",
            param,
        )
    }

    pub(crate) fn envelope(&self) -> Value {
        json!({
            "error": {
                "message": self.message,
                "type": self.kind,
                "param": self.param,
                "code": self.status.as_u16(),
            }
        })
    }

    pub(crate) fn grpc_status(&self) -> Status {
        let code = match self.status {
            StatusCode::BAD_REQUEST => Code::InvalidArgument,
            StatusCode::NOT_FOUND => Code::NotFound,
            StatusCode::SERVICE_UNAVAILABLE => Code::Unavailable,
            StatusCode::TOO_MANY_REQUESTS => Code::ResourceExhausted,
            StatusCode::GATEWAY_TIMEOUT => Code::DeadlineExceeded,
            _ => Code::Internal,
        };
        Status::new(code, self.message.clone())
    }
}

const INCLUDE_VALUES: [&str; 6] = [
    "code_interpreter_call.outputs",
    "computer_call_output.output.image_url",
    "file_search_call.results",
    "message.input_image.image_url",
    "message.output_text.logprobs",
    "reasoning.encrypted_content",
];

/// `reasoning.effort` values Python echoes back; the others echo `null`.
const ECHOABLE_EFFORTS: [&str; 4] = ["minimal", "low", "medium", "high"];

const ENCRYPTED_REASONING_PREFIX: &str = "sglang-reasoning-v1.";

fn random_hex() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

/// Python's `request_id` default.
pub(crate) fn response_id() -> String {
    format!("resp_{}", random_hex())
}

/// What the request echoes back on every response and lifecycle event.
#[derive(Clone)]
struct Echo {
    id: String,
    created_at: i64,
    model: Option<String>,
    parallel_tool_calls: bool,
    tool_choice: Value,
    tools: Value,
    instructions: Option<String>,
    max_output_tokens: Option<i64>,
    reasoning: Value,
    store: Option<bool>,
    temperature: Option<f64>,
    text: Option<Value>,
    top_p: Option<f64>,
    truncation: Option<String>,
    user: Option<String>,
    metadata: Value,
}

impl Echo {
    /// Python's `ResponsesResponse`, serialized with every field.
    fn response(
        &self,
        output: Vec<Value>,
        status: &str,
        usage: Option<Value>,
        error: Option<Value>,
    ) -> Value {
        let incomplete_details = if status == "incomplete" {
            json!({"reason": "max_output_tokens"})
        } else {
            Value::Null
        };
        // `text` echoes the request's, else `{"format": {"type": "text"}}`
        // for a text-only output.
        let text = match &self.text {
            Some(text) => text.clone(),
            None => {
                let text_only = !output.is_empty()
                    && output.iter().all(|item| {
                        item["type"] == "message"
                            && item["content"].as_array().is_some_and(|parts| {
                                parts.iter().all(|p| p["type"] == "output_text")
                            })
                    });
                if text_only {
                    json!({"format": {"type": "text"}})
                } else {
                    Value::Null
                }
            }
        };
        json!({
            "id": self.id,
            "object": "response",
            "created_at": self.created_at,
            "model": self.model,
            "output": output,
            "status": status,
            "usage": usage,
            "parallel_tool_calls": self.parallel_tool_calls,
            "tool_choice": self.tool_choice,
            "tools": self.tools,
            "error": error,
            "incomplete_details": incomplete_details,
            "instructions": self.instructions,
            "max_output_tokens": self.max_output_tokens,
            "previous_response_id": null,
            "reasoning": self.reasoning,
            "store": self.store,
            "temperature": self.temperature,
            "text": text,
            "top_p": self.top_p,
            "truncation": self.truncation,
            "user": self.user,
            "metadata": self.metadata,
        })
    }
}

/// A function or custom tool the request declared, as the chat core sees it.
struct DeclaredTools {
    chat_tools: Vec<Value>,
    custom: HashSet<String>,
}

/// `tool_choice` reduced as Python's `effective_tool_choice`: a string, or
/// `{"type", "name"}` for a named function/custom tool; any other object
/// is `"auto"`.
fn effective_tool_choice(raw: Option<&str>) -> Result<Value, ResponsesFailure> {
    let Some(raw) = raw else {
        return Ok(json!("auto"));
    };
    let value: Value = serde_json::from_str(raw)
        .map_err(|error| ResponsesFailure::bad_request(format!("tool_choice: {error}")))?;
    match &value {
        Value::String(choice) if ["auto", "required", "none"].contains(&choice.as_str()) => {
            Ok(value)
        }
        Value::String(choice) => Err(ResponsesFailure::bad_request(format!(
            "tool_choice: Input should be 'auto', 'required' or 'none', got {choice:?}"
        ))),
        Value::Object(object) => {
            let name = object
                .get("name")
                .and_then(Value::as_str)
                .or_else(|| object.get("function")?.get("name")?.as_str());
            match (object.get("type").and_then(Value::as_str), name) {
                (Some(kind @ ("function" | "custom")), Some(name)) => {
                    let mut choice = json!({"type": kind, "name": name});
                    if let Some(namespace) = object.get("namespace") {
                        choice["namespace"] = namespace.clone();
                    }
                    Ok(choice)
                }
                _ => Ok(json!("auto")),
            }
        }
        _ => Err(ResponsesFailure::bad_request(
            "tool_choice: Input should be a string or an object",
        )),
    }
}

/// A declared tool as Python echoes it: every `ResponseTool` field, null
/// when unset.
fn echoed_tool(tool: &api::OpenAiResponseTool) -> Value {
    let raw = |text: Option<&str>| {
        text.and_then(|text| serde_json::from_str::<Value>(text).ok())
            .unwrap_or(Value::Null)
    };
    let tools = if tool.tools.is_empty() {
        Value::Null
    } else {
        Value::Array(tool.tools.iter().map(|t| raw(Some(t))).collect())
    };
    json!({
        "type": tool.r#type,
        "name": tool.name,
        "description": tool.description,
        "parameters": raw(tool.parameters.as_deref()),
        "strict": tool.strict,
        "tools": tools,
        "defer_loading": tool.defer_loading,
        "format": raw(tool.format.as_deref()),
    })
}

/// Python's `custom_tool_parameters` / `custom_tool_description`: a custom
/// tool shimmed as a function taking one freeform `input` string.
fn custom_tool_shim(tool: &api::OpenAiResponseTool) -> (Option<String>, Value) {
    let mut parts: Vec<String> = Vec::new();
    if let Some(description) = tool.description.as_ref().filter(|d| !d.is_empty()) {
        parts.push(description.clone());
    }
    if let Some(format) = tool
        .format
        .as_deref()
        .and_then(|text| serde_json::from_str::<Value>(text).ok())
        && format["type"] == "grammar"
    {
        let syntax = format["syntax"].as_str().unwrap_or("lark");
        let definition = format["definition"].as_str().unwrap_or("");
        parts.push(format!(
            "The payload must conform to this {syntax} grammar:\n{definition}"
        ));
    }
    let description = (!parts.is_empty()).then(|| parts.join("\n\n"));
    let parameters = json!({
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "The freeform text payload for this tool."}
        },
        "required": ["input"],
        "additionalProperties": false,
    });
    (description, parameters)
}

/// Python's `_response_tools_to_chat_tools`: function and custom tools;
/// every other type is skipped.
fn declared_tools(tools: &[api::OpenAiResponseTool]) -> Result<DeclaredTools, ResponsesFailure> {
    let mut chat_tools = Vec::new();
    let mut custom = HashSet::new();
    for tool in tools {
        match tool.r#type.as_str() {
            "function" => {
                let Some(name) = tool.name.as_ref().filter(|n| !n.is_empty()) else {
                    return Err(ResponsesFailure::bad_request(
                        "Function tools must include a name.",
                    ));
                };
                let parameters = match tool.parameters.as_deref() {
                    Some(text) => serde_json::from_str::<Value>(text).map_err(|error| {
                        ResponsesFailure::bad_request(format!("tools.parameters: {error}"))
                    })?,
                    None => Value::Null,
                };
                chat_tools.push(json!({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool.description,
                        "parameters": parameters,
                        "strict": tool.strict,
                    },
                }));
            }
            "custom" => {
                let Some(name) = tool.name.as_ref().filter(|n| !n.is_empty()) else {
                    continue;
                };
                let (description, parameters) = custom_tool_shim(tool);
                custom.insert(name.clone());
                chat_tools.push(json!({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": parameters,
                        "strict": false,
                    },
                }));
            }
            _ => {}
        }
    }
    Ok(DeclaredTools { chat_tools, custom })
}

/// Python's `label_developer_content`.
fn label_developer_content(content: Value) -> Value {
    const LABEL: &str = "Developer instructions:";
    match content {
        Value::String(text) => Value::String(format!("{LABEL}\n{text}")),
        Value::Array(mut parts) => {
            if let Some(part) = parts
                .iter_mut()
                .find(|part| part.get("text").is_some_and(Value::is_string))
            {
                let text = part["text"].as_str().unwrap_or_default().to_owned();
                part["text"] = Value::String(format!("{LABEL}\n{text}"));
            } else {
                parts.insert(0, json!({"type": "input_text", "text": LABEL}));
            }
            Value::Array(parts)
        }
        other => other,
    }
}

/// Python's `_normalize_response_content_part_for_chat`.
fn chat_content_part(part: &Value) -> Value {
    let Some(object) = part.as_object() else {
        return part.clone();
    };
    match object.get("type").and_then(Value::as_str) {
        Some("input_text") | Some("output_text") => {
            json!({"type": "text", "text": object.get("text").and_then(Value::as_str).unwrap_or("")})
        }
        Some("input_image") | Some("image_url") => {
            let mut image_url = match object.get("image_url") {
                Some(Value::String(url)) => json!({"url": url}),
                Some(Value::Object(fields)) => Value::Object(fields.clone()),
                _ => json!({}),
            };
            if image_url.get("detail").is_none() {
                image_url["detail"] = object
                    .get("detail")
                    .filter(|d| !d.is_null())
                    .cloned()
                    .unwrap_or_else(|| json!("auto"));
            }
            for key in [
                "min_dynamic_patch",
                "max_dynamic_patch",
                "max_long_side_pixel",
            ] {
                if image_url.get(key).is_none()
                    && let Some(value) = object.get(key)
                {
                    image_url[key] = value.clone();
                }
            }
            let mut converted = Map::new();
            for (key, value) in object {
                if key != "type" && key != "image_url" && key != "detail" {
                    converted.insert(key.clone(), value.clone());
                }
            }
            converted.insert("type".into(), json!("image_url"));
            converted.insert("image_url".into(), image_url);
            Value::Object(converted)
        }
        _ => part.clone(),
    }
}

/// Python's `_flatten_tool_output`.
fn flatten_tool_output(output: Option<&Value>) -> Value {
    match output {
        Some(Value::Array(parts)) => Value::String(
            parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<String>(),
        ),
        Some(other) => other.clone(),
        None => json!(""),
    }
}

/// A function call's arguments as the JSON-object string the chat core expects.
fn tool_call_arguments(raw: Option<&Value>) -> String {
    match raw {
        Some(Value::String(text)) => match serde_json::from_str::<Value>(text) {
            Ok(Value::Object(_)) => text.clone(),
            _ => "{}".into(),
        },
        Some(Value::Object(object)) => Value::Object(object.clone()).to_string(),
        _ => "{}".into(),
    }
}

/// Python's `decode_reasoning_state`.
fn decode_reasoning_state(encrypted: &str) -> Option<String> {
    use base64::Engine;
    use std::io::Read;
    let payload = encrypted.strip_prefix(ENCRYPTED_REASONING_PREFIX)?;
    let compressed = base64::engine::general_purpose::URL_SAFE
        .decode(payload)
        .ok()?;
    let mut text = String::new();
    flate2::read::ZlibDecoder::new(&compressed[..])
        .read_to_string(&mut text)
        .ok()?;
    Some(text)
}

/// Python's `encode_reasoning_state`.
fn encode_reasoning_state(trace: &str) -> String {
    use base64::Engine;
    use std::io::Write;
    let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    let _ = encoder.write_all(trace.as_bytes());
    let compressed = encoder.finish().unwrap_or_default();
    format!(
        "{ENCRYPTED_REASONING_PREFIX}{}",
        base64::engine::general_purpose::URL_SAFE.encode(compressed)
    )
}

/// Python's `_normalize_response_message_for_chat`: one input item as a
/// chat message, or `None` for a reasoning item with nothing to carry.
fn chat_message(item: &Value) -> Result<Option<Value>, ResponsesFailure> {
    let Some(object) = item.as_object() else {
        return Err(ResponsesFailure::invalid(
            "Unsupported Responses API input item: expected an object None",
        ));
    };
    let mut object = object.clone();
    if object.get("role").and_then(Value::as_str) == Some("developer") {
        object.insert("role".into(), json!("system"));
        if let Some(content) = object.remove("content") {
            object.insert("content".into(), label_developer_content(content));
        }
    }
    let kind = object
        .get("type")
        .and_then(Value::as_str)
        .map(str::to_owned);
    match kind.as_deref() {
        Some("function_call") => {
            let id = object
                .get("call_id")
                .or_else(|| object.get("id"))
                .cloned()
                .unwrap_or(Value::Null);
            Ok(Some(json!({
                "role": "assistant",
                "tool_calls": [{
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": object.get("name").cloned().unwrap_or(Value::Null),
                        "arguments": tool_call_arguments(object.get("arguments")),
                    },
                }],
            })))
        }
        Some("custom_tool_call") => {
            let id = object
                .get("call_id")
                .or_else(|| object.get("id"))
                .cloned()
                .unwrap_or(Value::Null);
            let payload = object.get("input").and_then(Value::as_str).unwrap_or("");
            Ok(Some(json!({
                "role": "assistant",
                "tool_calls": [{
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": object.get("name").cloned().unwrap_or(Value::Null),
                        "arguments": json!({"input": payload}).to_string(),
                    },
                }],
            })))
        }
        Some("function_call_output") | Some("custom_tool_call_output") => Ok(Some(json!({
            "role": "tool",
            "tool_call_id": object.get("call_id").cloned().unwrap_or(Value::Null),
            "content": flatten_tool_output(object.get("output")),
        }))),
        Some("reasoning") => {
            let texts = |key: &str| -> Vec<String> {
                object
                    .get(key)
                    .and_then(Value::as_array)
                    .map(|parts| {
                        parts
                            .iter()
                            .filter_map(|part| part.get("text").and_then(Value::as_str))
                            .filter(|text| !text.is_empty())
                            .map(str::to_owned)
                            .collect()
                    })
                    .unwrap_or_default()
            };
            let mut trace = texts("summary");
            if trace.is_empty() {
                trace = texts("content");
            }
            if trace.is_empty()
                && let Some(decoded) = object
                    .get("encrypted_content")
                    .and_then(Value::as_str)
                    .and_then(decode_reasoning_state)
            {
                trace.push(decoded);
            }
            if trace.is_empty() {
                return Ok(None);
            }
            Ok(Some(
                json!({"role": "assistant", "reasoning_content": trace.join("\n")}),
            ))
        }
        None | Some("message") => {
            let mut message = Map::new();
            for (key, value) in object {
                if value.is_null() || ["id", "status", "type"].contains(&key.as_str()) {
                    continue;
                }
                if key == "content" {
                    message.insert(
                        key,
                        match value {
                            Value::Array(parts) => {
                                Value::Array(parts.iter().map(chat_content_part).collect())
                            }
                            other => other,
                        },
                    );
                } else {
                    message.insert(key, value);
                }
            }
            Ok(Some(Value::Object(message)))
        }
        Some(other) => Err(ResponsesFailure::invalid(format!(
            "Unsupported Responses API input item type: {other:?} None"
        ))),
    }
}

/// Python's `_merge_consecutive_assistant_messages`.
fn merge_consecutive_assistant(messages: Vec<Value>) -> Vec<Value> {
    let mut merged: Vec<Value> = Vec::with_capacity(messages.len());
    for message in messages {
        let Some(previous) = merged.last_mut() else {
            merged.push(message);
            continue;
        };
        let both_assistant = previous["role"] == "assistant" && message["role"] == "assistant";
        let previous_is_bare_reasoning = previous
            .as_object()
            .is_some_and(|o| o.len() == 2 && o.contains_key("reasoning_content"));
        if !both_assistant || !(previous["phase"] == message["phase"] || previous_is_bare_reasoning)
        {
            merged.push(message);
            continue;
        }
        let Value::Object(incoming) = message else {
            merged.push(message);
            continue;
        };
        for (key, value) in incoming {
            match key.as_str() {
                "role" => {}
                "content" => match (previous.get("content").cloned(), value) {
                    (Some(Value::String(a)), Value::String(b)) => {
                        previous["content"] = Value::String(format!("{a}\n\n{b}"));
                    }
                    (None, value) => previous["content"] = value,
                    (Some(a), b) => {
                        let mut parts = match a {
                            Value::Array(parts) => parts,
                            Value::String(text) => vec![json!({"type": "text", "text": text})],
                            other => vec![other],
                        };
                        match b {
                            Value::Array(more) => parts.extend(more),
                            Value::String(text) => {
                                parts.push(json!({"type": "text", "text": text}))
                            }
                            other => parts.push(other),
                        }
                        previous["content"] = Value::Array(parts);
                    }
                },
                "tool_calls" => {
                    let mut calls = previous
                        .get("tool_calls")
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default();
                    if let Value::Array(more) = value {
                        calls.extend(more);
                    }
                    previous["tool_calls"] = Value::Array(calls);
                }
                "reasoning_content" => {
                    let joined = match (previous.get("reasoning_content"), &value) {
                        (Some(Value::String(a)), Value::String(b)) => {
                            Value::String(format!("{a}\n{b}"))
                        }
                        _ => value,
                    };
                    previous["reasoning_content"] = joined;
                }
                "phase" => {
                    if !value.is_null() {
                        previous["phase"] = value;
                    }
                }
                other => {
                    previous[other] = value;
                }
            }
        }
    }
    merged
}

/// Python's `_construct_input_messages`, with every system message merged
/// into one leading system message (the chat encoders without inline
/// system support), as Python does for them.
fn chat_messages(
    input: Option<&str>,
    instructions: Option<&str>,
) -> Result<Vec<Value>, ResponsesFailure> {
    let mut messages: Vec<Value> = Vec::new();
    if let Some(instructions) = instructions.filter(|i| !i.is_empty()) {
        messages.push(json!({"role": "system", "content": instructions}));
    }
    let input: Value = match input {
        None => {
            return Err(ResponsesFailure::bad_request("input: Field required"));
        }
        Some(text) => serde_json::from_str(text)
            .map_err(|error| ResponsesFailure::bad_request(format!("input: {error}")))?,
    };
    match input {
        Value::String(text) => messages.push(json!({"role": "user", "content": text})),
        Value::Array(items) => {
            for item in &items {
                if let Some(message) = chat_message(item)? {
                    messages.push(message);
                }
            }
        }
        _ => {
            return Err(ResponsesFailure::bad_request(
                "input: Input should be a valid string or list",
            ));
        }
    }
    let messages = merge_consecutive_assistant(messages);
    let mut system: Vec<String> = Vec::new();
    let mut rest: Vec<Value> = Vec::new();
    for message in messages {
        if message["role"] == "system" {
            match &message["content"] {
                Value::String(text) => system.push(text.clone()),
                Value::Array(parts) => system.extend(
                    parts
                        .iter()
                        .filter_map(|part| part.get("text").and_then(Value::as_str))
                        .map(str::to_owned),
                ),
                _ => {}
            }
        } else {
            rest.push(message);
        }
    }
    let mut out = Vec::with_capacity(rest.len() + 1);
    if !system.is_empty() {
        out.push(json!({"role": "system", "content": system.join("\n\n")}));
    }
    out.extend(rest);
    Ok(out)
}

/// Python's `_json_schema_from_text_format`.
fn json_schema_from_text(
    text: Option<&str>,
) -> Result<(Option<Value>, Option<String>), ResponsesFailure> {
    let Some(text) = text else {
        return Ok((None, None));
    };
    let value: Value = serde_json::from_str(text)
        .map_err(|error| ResponsesFailure::bad_request(format!("text: {error}")))?;
    let schema = match value["format"]["type"].as_str() {
        Some("json_object") => Some(r#"{"type": "object"}"#.to_owned()),
        Some("json_schema") => {
            let format = &value["format"];
            if format.get("name").and_then(Value::as_str).is_none() {
                return Err(ResponsesFailure::bad_request(
                    "text.format.name: Field required",
                ));
            }
            let Some(schema) = format.get("schema") else {
                return Err(ResponsesFailure::bad_request(
                    "text.format.schema: Field required",
                ));
            };
            Some(schema.to_string())
        }
        Some("text") | None => None,
        Some(other) => {
            return Err(ResponsesFailure::bad_request(format!(
                "text.format.type: unexpected value {other:?}"
            )));
        }
    };
    Ok((Some(value), schema))
}

/// A Responses request translated to the chat request that serves it, and
/// what shaping its reply needs: Python's `_make_request` plus the
/// sampling fields `to_sampling_params` reads off the Responses request.
pub(crate) struct Plan {
    /// The chat-completions body, as `/v1/chat/completions` takes it.
    pub(crate) chat_body: Value,
    echo: Echo,
    custom_tools: HashSet<String>,
    tools_callable: bool,
    want_logprobs: bool,
    reasoning_summary: bool,
    encrypted_reasoning: bool,
    json_schema: Option<String>,
    kwargs: Map<String, Value>,
}

/// One submitted chat generation and what its reply shaping needs.
struct Submitted {
    call: CoreCall,
    plan: Plan,
    parsing: OutputParsing,
}

/// The chat request's own fields, for the typed request; the body also
/// carries SGLang extensions the chat route reads from the raw JSON.
const CHAT_REQUEST_KEYS: [&str; 16] = [
    "model",
    "messages",
    "stream",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "stop",
    "temperature",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "max_completion_tokens",
    "logprobs",
    "top_logprobs",
    "response_format",
    "reasoning_effort",
];

impl Responses<'_> {
    /// Python's `create_responses` up to the chat request: validation, the
    /// store and tool refusals, the input items as chat messages, and the
    /// sampling fields.
    pub(crate) fn plan(
        &self,
        request: &api::OpenAiResponsesRequest,
        stream: bool,
    ) -> Result<Plan, ResponsesFailure> {
        for include in &request.include {
            if !INCLUDE_VALUES.contains(&include.as_str()) {
                return Err(ResponsesFailure::bad_request(format!(
                    "include: unexpected value {include:?}"
                )));
            }
        }
        if request.tools.iter().any(|tool| tool.r#type == "shell") {
            return Err(ResponsesFailure::with_param(
                "Shell tools are not supported by this model",
                "tools",
            ));
        }
        if request.previous_response_id.is_some() {
            return Err(ResponsesFailure::store_disabled("previous_response_id"));
        }
        if request.background == Some(true) {
            return Err(ResponsesFailure::store_disabled("background"));
        }
        // Python's `_validate_model`: the served name, or `base:adapter`
        // whose base is it (LoRA adapters are refused at launch).
        if let Some(model) = request.model.as_deref() {
            let served = &self.server_args.served_model_name;
            let base = model.split_once(':').map_or(model, |(base, _)| base);
            if model != served && base != served {
                return Err(ResponsesFailure {
                    status: StatusCode::NOT_FOUND,
                    kind: "invalid_request_error",
                    param: Some("model"),
                    message: format!("The model '{model}' does not exist"),
                });
            }
        }
        let declared = declared_tools(&request.tools)?;
        let tool_choice = effective_tool_choice(request.tool_choice.as_deref())?;
        if tool_choice == "required" && declared.chat_tools.is_empty() {
            return Err(ResponsesFailure::invalid(
                "tool_choice=\"required\" requires at least one tool with type=\"function\" or \
                 type=\"custom\"; other built-in tool types cannot be forced.",
            ));
        }
        if let Some(name) = tool_choice.get("name").and_then(Value::as_str)
            && !declared
                .chat_tools
                .iter()
                .any(|tool| tool["function"]["name"] == name)
        {
            return Err(ResponsesFailure::with_param(
                format!("Tool {name:?} is not declared in tools"),
                "tool_choice",
            ));
        }
        let want_logprobs = request
            .include
            .iter()
            .any(|include| include == "message.output_text.logprobs");
        if stream && want_logprobs {
            return Err(ResponsesFailure::with_param(
                "logprobs are not supported in streaming mode",
                "logprobs",
            ));
        }
        let encrypted_reasoning = request
            .include
            .iter()
            .any(|include| include == "reasoning.encrypted_content");

        let messages = chat_messages(request.input.as_deref(), request.instructions.as_deref())?;
        let (text_echo, json_schema) = json_schema_from_text(request.text.as_deref())?;
        // Python passes `reasoning.effort` only when `reasoning` was sent;
        // its default is then "medium".
        let effort = request
            .reasoning
            .as_ref()
            .map(|r| r.effort.clone().unwrap_or_else(|| "medium".into()));
        let reasoning_summary = request
            .reasoning
            .as_ref()
            .is_some_and(|r| r.summary.is_some());
        let tools_callable = !declared.chat_tools.is_empty() && tool_choice != "none";
        if json_schema.is_some() && tools_callable {
            return Err(ResponsesFailure::invalid(
                "Cannot combine tool calls with constrained decoding (text.format / regex / \
                 ebnf / structural_tag / json_schema). Remove one.",
            ));
        }
        let chat_tool_choice = if declared.chat_tools.is_empty() {
            json!("none")
        } else if let Some(name) = tool_choice.get("name").and_then(Value::as_str) {
            json!({"type": "function", "function": {"name": name}})
        } else {
            tool_choice.clone()
        };
        let max_completion_tokens = match request.max_output_tokens {
            // Python's headroom for BOS/EOS.
            Some(n) if n > 2 => Some(n - 2),
            Some(n) => {
                return Err(ResponsesFailure::invalid(format!(
                    "max_output_tokens must be greater than 2, got {n}"
                )));
            }
            None => None,
        };
        let id = request
            .request_id
            .clone()
            .filter(|id| !id.is_empty())
            .unwrap_or_else(response_id);
        let mut chat_body = json!({
            "model": request.model.clone().unwrap_or_else(|| self.server_args.served_model_name.clone()),
            "messages": messages,
            "stream": stream,
            "parallel_tool_calls": request.parallel_tool_calls.unwrap_or(true),
            "temperature": request.temperature,
            "top_p": request.top_p,
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty,
            "max_completion_tokens": max_completion_tokens,
            "logprobs": want_logprobs,
            "top_logprobs": request.top_logprobs.unwrap_or(0),
            "rid": id,
        });
        if stream {
            chat_body["stream_options"] = json!({"include_usage": true});
        }
        if let Some(effort) = &effort {
            chat_body["reasoning_effort"] = json!(effort);
        }
        if !declared.chat_tools.is_empty() {
            chat_body["tools"] = Value::Array(declared.chat_tools.clone());
            chat_body["tool_choice"] = chat_tool_choice;
        }
        if let Some(stop) = &request.stop {
            chat_body["stop"] = serde_json::to_value(stop).unwrap_or(Value::Null);
        }
        if let Some(top_k) = request.top_k {
            chat_body["top_k"] = json!(top_k);
        }
        if let Some(min_p) = request.min_p {
            chat_body["min_p"] = json!(min_p);
        }
        if let Some(repetition_penalty) = request.repetition_penalty {
            chat_body["repetition_penalty"] = json!(repetition_penalty);
        }
        match request
            .text
            .as_deref()
            .and_then(|text| serde_json::from_str::<Value>(text).ok())
        {
            Some(text) if text["format"]["type"] == "json_object" => {
                chat_body["response_format"] = json!({"type": "json_object"});
            }
            Some(text) if text["format"]["type"] == "json_schema" => {
                chat_body["response_format"] = json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": text["format"]["name"],
                        "schema": text["format"]["schema"],
                        "strict": text["format"]["strict"],
                    },
                });
            }
            _ => {}
        }
        for (key, value) in [
            ("session_id", request.session_id.as_ref().map(|v| json!(v))),
            ("extra_key", request.extra_key.as_ref().map(|v| json!(v))),
            ("cache_salt", request.cache_salt.as_ref().map(|v| json!(v))),
            (
                "bootstrap_host",
                request
                    .bootstrap_host
                    .as_ref()
                    .and_then(|v| serde_json::to_value(v).ok()),
            ),
            (
                "bootstrap_port",
                request
                    .bootstrap_port
                    .as_ref()
                    .and_then(|v| serde_json::to_value(v).ok()),
            ),
            (
                "bootstrap_room",
                request
                    .bootstrap_room
                    .as_ref()
                    .and_then(|v| serde_json::to_value(v).ok()),
            ),
            (
                "routed_dp_rank",
                request
                    .routed_dp_rank
                    .or(request.data_parallel_rank)
                    .map(|v| json!(v)),
            ),
            (
                "disagg_prefill_dp_rank",
                request.disagg_prefill_dp_rank.map(|v| json!(v)),
            ),
        ] {
            if let Some(value) = value {
                chat_body[key] = value;
            }
        }
        let mut kwargs: Map<String, Value> = match request.chat_template_kwargs.as_deref() {
            Some(text) => match serde_json::from_str(text) {
                Ok(Value::Object(object)) => object,
                _ => {
                    return Err(ResponsesFailure::bad_request(
                        "chat_template_kwargs: Input should be a valid dictionary",
                    ));
                }
            },
            None => Map::new(),
        };
        if effort.as_deref() == Some("none") {
            kwargs.entry("thinking").or_insert(json!(false));
            kwargs.entry("enable_thinking").or_insert(json!(false));
        }
        if !kwargs.is_empty() {
            chat_body["chat_template_kwargs"] = Value::Object(kwargs.clone());
        }
        let echo = Echo {
            id,
            created_at: i64::from(unix_seconds_u32()),
            model: request.model.clone(),
            parallel_tool_calls: request.parallel_tool_calls.unwrap_or(true),
            tool_choice,
            tools: Value::Array(request.tools.iter().map(echoed_tool).collect()),
            instructions: request.instructions.clone(),
            max_output_tokens: request.max_output_tokens,
            reasoning: json!({
                "effort": effort
                    .as_deref()
                    .filter(|effort| ECHOABLE_EFFORTS.contains(effort)),
                "summary": null,
            }),
            store: request.store,
            temperature: request.temperature,
            text: text_echo,
            top_p: request.top_p,
            truncation: request.truncation.clone(),
            user: request.user.clone(),
            metadata: request
                .metadata
                .as_deref()
                .and_then(|text| serde_json::from_str(text).ok())
                .unwrap_or_else(|| json!({})),
        };
        Ok(Plan {
            chat_body,
            echo,
            custom_tools: declared.custom,
            tools_callable,
            want_logprobs,
            reasoning_summary,
            encrypted_reasoning,
            json_schema,
            kwargs,
        })
    }

    /// The core path, for gRPC: render the planned chat request with the
    /// served template and submit it. (HTTP sends the plan through the chat
    /// route instead, so a model's token-level chat encoder applies.)
    async fn submit(
        &self,
        request: &api::OpenAiResponsesRequest,
        stream: bool,
    ) -> Result<Submitted, ResponsesFailure> {
        let Some(formatter) = self.chat_formatter else {
            return Err(ResponsesFailure::invalid(
                "this model has no usable chat template",
            ));
        };
        let plan = self.plan(request, stream)?;
        let mut typed = Map::new();
        for key in CHAT_REQUEST_KEYS {
            if let Some(value) = plan.chat_body.get(key).filter(|v| !v.is_null()) {
                typed.insert(key.into(), value.clone());
            }
        }
        let mut chat_request: CreateChatCompletionRequest =
            serde_json::from_value(Value::Object(typed.clone()))
                .map_err(|error| ResponsesFailure::invalid(format!("{error} None")))?;
        let mut body = Value::Object(typed);
        if !plan.kwargs.is_empty() {
            body["chat_template_kwargs"] = Value::Object(plan.kwargs.clone());
        }
        let rendered: RenderedChat = prepare_chat_prompt(
            formatter,
            body,
            &mut chat_request,
            &self.server_args.default_chat_template_kwargs,
        )
        .map_err(|message| ResponsesFailure::invalid(format!("{message} None")))?;

        let dynamo_choice = dynamo_tool_choice(&chat_request.tool_choice);
        let chat_tools = sglang_processor::chat_tool_definitions(
            chat_request.tools.as_deref(),
            &chat_request.messages,
        )
        .map_err(|error| ResponsesFailure::invalid(error.to_string()))?;
        let parser = plan
            .tools_callable
            .then(|| self.server_args.tool_call_parser.clone())
            .flatten();
        if plan.tools_callable && parser.is_none() {
            return Err(ResponsesFailure::invalid(
                "tool calls require --tool-call-parser",
            ));
        }
        let reasoning_parser = self.server_args.reasoning_parser.clone();
        let tools = (!chat_tools.is_empty()).then_some(chat_tools);
        let mut sampling = chat_sampling(
            &chat_request,
            SamplingDefaults::RESPONSES,
            parser.as_deref(),
            &dynamo_choice,
            tools.as_deref().unwrap_or_default(),
            chat_request.parallel_tool_calls,
            self.server_args,
        )
        .map_err(ResponsesFailure::invalid)?;
        if let Some(top_k) = request.top_k {
            sampling.top_k = top_k;
        }
        if let Some(min_p) = request.min_p {
            sampling.min_p = min_p;
        }
        if let Some(repetition_penalty) = request.repetition_penalty {
            sampling.repetition_penalty = repetition_penalty;
        }
        if let Some(schema) = plan.json_schema.clone() {
            sampling.json_schema = Some(schema);
        }
        let starts_in_reasoning =
            starts_in_reasoning(reasoning_parser.as_deref(), rendered.prompt.as_str());
        let routing = PDRoutingFields {
            bootstrap_host: request.bootstrap_host.clone(),
            bootstrap_port: request.bootstrap_port.clone(),
            bootstrap_room: request.bootstrap_room.clone(),
            routed_dp_rank: request.routed_dp_rank.or(request.data_parallel_rank),
            data_parallel_rank: None,
            disagg_prefill_dp_rank: request.disagg_prefill_dp_rank,
        }
        .normalize(1, 1)
        .map_err(|error| ResponsesFailure::invalid(error.to_string()))?;
        let native = GenerateRequest {
            rid: Rid::from_client(&plan.echo.id),
            chat_prompt: Some(Box::new(rendered)),
            sampling_params: sampling,
            stream,
            return_logprob: plan.want_logprobs,
            logprob_start_len: -1,
            top_logprobs_num: if plan.want_logprobs {
                request.top_logprobs.unwrap_or(0)
            } else {
                0
            },
            return_text_in_logprobs: plan.want_logprobs.then_some(true),
            session_id: request.session_id.clone(),
            extra_key: request.extra_key.clone(),
            cache_salt: request.cache_salt.clone(),
            bootstrap_host: routing.bootstrap.bootstrap_hosts[0].clone(),
            bootstrap_port: routing.bootstrap.bootstrap_ports[0],
            bootstrap_room: routing.bootstrap.bootstrap_rooms[0],
            routed_dp_rank: routing.routed_dp_rank,
            disagg_prefill_dp_rank: routing.disagg_prefill_dp_rank,
            ..Default::default()
        };
        let call = self
            .core
            .generate(native, ReceivedAs::without_headers())
            .await
            .map_err(ResponsesFailure::from_core)?;
        Ok(Submitted {
            call,
            plan,
            parsing: OutputParsing::Text {
                reasoning_parser,
                reasoning_options: ReasoningOptions {
                    force_reasoning: starts_in_reasoning.then_some(true),
                    ..Default::default()
                },
                tool_parser: parser,
                tools,
            },
        })
    }

    /// `POST /v1/responses` without `stream`, on the core path.
    pub(crate) async fn create(
        &self,
        request: &api::OpenAiResponsesRequest,
    ) -> Result<Value, ResponsesFailure> {
        let Submitted {
            call,
            plan,
            parsing,
            ..
        } = self.submit(request, false).await?;
        let chat = unary_chat_response(
            vec![(0, call)],
            plan.echo.id.clone(),
            plan.echo.model.clone().unwrap_or_default(),
            0,
            plan.want_logprobs,
            parsing,
            0,
            plan.echo.parallel_tool_calls,
            None,
            true,
        )
        .await
        // Python answers every generation failure here as a 400.
        .map_err(|(_, message)| ResponsesFailure::invalid(message))?;
        Ok(plan.shape_unary(chat))
    }
}

impl Plan {
    /// Python's `_output_items` over the folded chat reply, and the response.
    pub(crate) fn shape_unary(&self, chat: CreateChatCompletionResponse) -> Value {
        shape_unary(
            chat,
            &self.echo,
            &self.custom_tools,
            self.reasoning_summary,
            self.encrypted_reasoning,
        )
    }

    /// Python's Responses event sequence over the chat stream's frames
    /// (each a chunk's JSON, or `[DONE]`).
    pub(crate) fn events(self, chunks: impl Stream<Item = String> + Send + 'static) -> Events {
        let events = responses_events(
            chunks,
            self.echo,
            self.custom_tools,
            self.reasoning_summary,
            self.encrypted_reasoning,
        );
        Box::pin(phase_buffered(events, self.tools_callable))
    }
}

/// Python's `_status_from_finish_reason`.
fn status_from_finish_reason(finish_reason: Option<&Value>) -> &'static str {
    match finish_reason.and_then(|reason| {
        reason
            .as_str()
            .map(str::to_owned)
            .or_else(|| reason.get("type")?.as_str().map(str::to_owned))
    }) {
        Some(kind) if kind == "length" => "incomplete",
        Some(kind) if kind == "abort" || kind == "error" => "failed",
        _ => "completed",
    }
}

fn error_from_finish_reason(finish_reason: Option<&Value>) -> Value {
    let message = finish_reason
        .and_then(|reason| reason.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("Generation aborted");
    json!({"code": "server_error", "message": message})
}

/// Python's `_serialize_usage`.
fn usage_value(
    prompt_tokens: u64,
    completion_tokens: u64,
    cached_tokens: u64,
    reasoning_tokens: u64,
    total_tokens: u64,
) -> Value {
    json!({
        "input_tokens": prompt_tokens,
        "input_tokens_details": {"cached_tokens": cached_tokens, "cache_write_tokens": 0},
        "output_tokens": completion_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        "total_tokens": total_tokens,
    })
}

fn reasoning_item(id: &str, trace: &str, status: &str, summary: bool, encrypted: bool) -> Value {
    json!({
        "id": id,
        "type": "reasoning",
        "summary": if summary { json!([{"type": "summary_text", "text": trace}]) } else { json!([]) },
        "content": [{"type": "reasoning_text", "text": trace}],
        "encrypted_content": if encrypted { json!(encode_reasoning_state(trace)) } else { Value::Null },
        "status": status,
    })
}

fn message_item(id: &str, text: &str, phase: &str, logprobs: Value, status: &str) -> Value {
    json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "status": status,
        "phase": phase,
        "content": [{"type": "output_text", "text": text, "annotations": [], "logprobs": logprobs}],
    })
}

/// Python's `decode_custom_tool_input`.
fn decode_custom_tool_input(arguments: &str) -> String {
    match serde_json::from_str::<Value>(arguments) {
        Ok(Value::Object(object)) => {
            if let Some(Value::String(input)) = object.get("input") {
                return input.clone();
            }
            if object.len() == 1
                && let Some(Value::String(only)) = object.values().next()
            {
                return only.clone();
            }
            arguments.to_owned()
        }
        _ => arguments.to_owned(),
    }
}

fn tool_call_item(
    id: &str,
    call_id: &str,
    name: &str,
    arguments: &str,
    custom: bool,
    status: &str,
) -> Value {
    if custom {
        json!({
            "type": "custom_tool_call",
            "id": id,
            "call_id": call_id,
            "name": name,
            "input": decode_custom_tool_input(arguments),
        })
    } else {
        json!({
            "arguments": arguments,
            "call_id": call_id,
            "name": name,
            "type": "function_call",
            "id": id,
            "status": status,
        })
    }
}

fn new_call_id() -> String {
    format!("call_{}", &random_hex()[..24])
}

fn new_item_id(prefix: &str) -> String {
    format!("{prefix}_{}", &random_hex()[..8])
}

/// Python's `_output_items` over the folded chat reply, and the response.
fn shape_unary(
    chat: CreateChatCompletionResponse,
    echo: &Echo,
    custom_tools: &HashSet<String>,
    reasoning_summary: bool,
    encrypted_reasoning: bool,
) -> Value {
    let mut output = Vec::new();
    let choice = chat.choices.into_iter().next();
    let finish_reason = choice
        .as_ref()
        .and_then(|choice| serde_json::to_value(choice.finish_reason).ok())
        .filter(|value| !value.is_null());
    if let Some(choice) = choice {
        let message = choice.message;
        if let Some(reasoning) = message.reasoning_content.filter(|r| !r.is_empty()) {
            output.push(reasoning_item(
                &format!("rs_{}", random_hex()),
                &reasoning,
                "completed",
                reasoning_summary,
                encrypted_reasoning,
            ));
        }
        let calls = message.tool_calls.unwrap_or_default();
        let content = match message.content {
            Some(ChatCompletionMessageContent::Text(text)) => text,
            Some(ChatCompletionMessageContent::Parts(parts)) => {
                serde_json::to_string(&parts).unwrap_or_default()
            }
            None => String::new(),
        };
        if !content.is_empty() {
            let logprobs = match choice.logprobs.and_then(|l| l.content) {
                Some(entries) => Value::Array(
                    entries
                        .into_iter()
                        .map(|entry| {
                            json!({
                                "token": entry.token,
                                "logprob": entry.logprob,
                                "bytes": entry.bytes,
                                "top_logprobs": entry.top_logprobs.into_iter().map(|top| json!({
                                    "token": top.token, "logprob": top.logprob, "bytes": top.bytes
                                })).collect::<Vec<_>>(),
                            })
                        })
                        .collect(),
                ),
                None => Value::Null,
            };
            output.push(message_item(
                &format!("msg_{}", random_hex()),
                &content,
                if calls.is_empty() {
                    "final_answer"
                } else {
                    "commentary"
                },
                logprobs,
                "completed",
            ));
        }
        for call in calls {
            let custom = custom_tools.contains(&call.function.name);
            output.push(tool_call_item(
                &new_item_id(if custom { "ctc" } else { "fc" }),
                &new_call_id(),
                &call.function.name,
                &call.function.arguments,
                custom,
                "completed",
            ));
        }
    }
    let status = status_from_finish_reason(finish_reason.as_ref());
    let usage = chat.usage.map(|usage| {
        let prompt = u64::from(usage.prompt_tokens);
        let completion = u64::from(usage.completion_tokens);
        usage_value(
            prompt,
            completion,
            usage
                .prompt_tokens_details
                .and_then(|details| details.cached_tokens)
                .map(u64::from)
                .unwrap_or(0),
            usage
                .completion_tokens_details
                .and_then(|details| details.reasoning_tokens)
                .map(u64::from)
                .unwrap_or(0),
            prompt + completion,
        )
    });
    let error = (status == "failed").then(|| error_from_finish_reason(finish_reason.as_ref()));
    echo.response(output, status, usage, error)
}

/// The item open on the stream, by output index.
enum OpenItem {
    Reasoning {
        id: String,
        index: i64,
        text: String,
    },
    Message {
        id: String,
        index: i64,
        text: String,
    },
    ToolCall {
        id: String,
        call_id: String,
        index: i64,
        name: String,
        arguments: String,
        custom: bool,
    },
}

/// The Responses event stream over the chat stream: Python's
/// `_responses_stream_generator_non_harmony`.
struct StreamState {
    echo: Echo,
    custom_tools: HashSet<String>,
    reasoning_summary: bool,
    encrypted_reasoning: bool,
    sequence_number: i64,
    output_index: i64,
    open: Option<OpenItem>,
    /// The tool calls by chat index, open or closed, in order.
    open_call: Option<u32>,
    done: Vec<Value>,
    finish_reason: Option<Value>,
    usage: Option<Value>,
}

impl StreamState {
    fn event(&mut self, kind: &str, mut payload: Value) -> ResponsesEvent {
        payload["type"] = json!(kind);
        payload["sequence_number"] = json!(self.sequence_number);
        self.sequence_number += 1;
        (kind.to_owned(), payload)
    }

    fn lifecycle(
        &mut self,
        kind: &str,
        status: &str,
        output: Vec<Value>,
        usage: Option<Value>,
        error: Option<Value>,
    ) -> ResponsesEvent {
        let response = self.echo.response(output, status, usage, error);
        self.event(kind, json!({"response": response}))
    }

    fn next_index(&mut self) -> i64 {
        self.output_index += 1;
        self.output_index
    }

    /// Close whatever item is open, emitting its done events.
    fn close_open(&mut self) -> Vec<ResponsesEvent> {
        let mut events = Vec::new();
        let Some(open) = self.open.take() else {
            return events;
        };
        match open {
            OpenItem::Reasoning { id, index, text } => {
                if self.reasoning_summary {
                    events.push(self.event(
                        "response.reasoning_summary_text.done",
                        json!({
                            "item_id": id, "output_index": index, "summary_index": 0, "text": text
                        }),
                    ));
                    events.push(self.event(
                        "response.reasoning_summary_part.done",
                        json!({
                            "item_id": id, "output_index": index, "summary_index": 0,
                            "part": {"type": "summary_text", "text": text}
                        }),
                    ));
                } else {
                    events.push(self.event(
                        "response.reasoning_text.done",
                        json!({
                            "item_id": id, "output_index": index, "content_index": 0, "text": text
                        }),
                    ));
                    events.push(self.event(
                        "response.content_part.done",
                        json!({
                            "item_id": id, "output_index": index, "content_index": 0,
                            "part": {"type": "reasoning_text", "text": text}
                        }),
                    ));
                }
                let item = reasoning_item(
                    &id,
                    &text,
                    "completed",
                    self.reasoning_summary,
                    self.encrypted_reasoning,
                );
                events.push(self.event(
                    "response.output_item.done",
                    json!({"output_index": index, "item": item}),
                ));
                self.done.push(item);
            }
            OpenItem::Message { id, index, text } => {
                events.push(self.event("response.output_text.done", json!({
                    "item_id": id, "output_index": index, "content_index": 0, "text": text, "logprobs": []
                })));
                events.push(self.event("response.content_part.done", json!({
                    "item_id": id, "output_index": index, "content_index": 0,
                    "part": {"type": "output_text", "text": text, "annotations": [], "logprobs": null}
                })));
                let item = message_item(&id, &text, "final_answer", Value::Null, "completed");
                events.push(self.event(
                    "response.output_item.done",
                    json!({"output_index": index, "item": item}),
                ));
                self.done.push(item);
            }
            OpenItem::ToolCall {
                id,
                call_id,
                index,
                name,
                arguments,
                custom,
            } => {
                if custom {
                    events.push(self.event("response.custom_tool_call_input.done", json!({
                        "item_id": id, "output_index": index, "input": decode_custom_tool_input(&arguments)
                    })));
                } else {
                    events.push(self.event("response.function_call_arguments.done", json!({
                        "item_id": id, "output_index": index, "arguments": arguments, "name": name
                    })));
                }
                let item = tool_call_item(&id, &call_id, &name, &arguments, custom, "completed");
                events.push(self.event(
                    "response.output_item.done",
                    json!({"output_index": index, "item": item}),
                ));
                self.done.push(item);
                self.open_call = None;
            }
        }
        events
    }

    fn reasoning_delta(&mut self, delta: String) -> Vec<ResponsesEvent> {
        let mut events = Vec::new();
        if !matches!(self.open, Some(OpenItem::Reasoning { .. })) {
            events.extend(self.close_open());
            let index = self.next_index();
            let id = format!("rs_{}", random_hex());
            let item = json!({"id": id, "type": "reasoning", "summary": [], "content": [], "encrypted_content": null, "status": "in_progress"});
            events.push(self.event(
                "response.output_item.added",
                json!({"output_index": index, "item": item}),
            ));
            if self.reasoning_summary {
                events.push(self.event("response.reasoning_summary_part.added", json!({
                    "item_id": id, "output_index": index, "summary_index": 0, "part": {"type": "summary_text", "text": ""}
                })));
            } else {
                events.push(self.event("response.content_part.added", json!({
                    "item_id": id, "output_index": index, "content_index": 0, "part": {"type": "reasoning_text", "text": ""}
                })));
            }
            self.open = Some(OpenItem::Reasoning {
                id,
                index,
                text: String::new(),
            });
        }
        let Some(OpenItem::Reasoning { id, index, text }) = &mut self.open else {
            return events;
        };
        text.push_str(&delta);
        let (id, index) = (id.clone(), *index);
        if self.reasoning_summary {
            events.push(self.event(
                "response.reasoning_summary_text.delta",
                json!({
                    "item_id": id, "output_index": index, "summary_index": 0, "delta": delta
                }),
            ));
        } else {
            events.push(self.event(
                "response.reasoning_text.delta",
                json!({
                    "item_id": id, "output_index": index, "content_index": 0, "delta": delta
                }),
            ));
        }
        events
    }

    fn text_delta(&mut self, delta: String) -> Vec<ResponsesEvent> {
        let mut events = Vec::new();
        // Whitespace between adjacent tool-call blocks is not a message.
        if matches!(self.open, Some(OpenItem::ToolCall { .. })) && delta.trim().is_empty() {
            return events;
        }
        if !matches!(self.open, Some(OpenItem::Message { .. })) {
            events.extend(self.close_open());
            let index = self.next_index();
            let id = format!("msg_{}", random_hex());
            let item = json!({"id": id, "type": "message", "role": "assistant", "content": [], "status": "in_progress", "phase": "final_answer"});
            events.push(self.event(
                "response.output_item.added",
                json!({"output_index": index, "item": item}),
            ));
            events.push(self.event(
                "response.content_part.added",
                json!({
                    "item_id": id, "output_index": index, "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": null}
                }),
            ));
            self.open = Some(OpenItem::Message {
                id,
                index,
                text: String::new(),
            });
        }
        let Some(OpenItem::Message { id, index, text }) = &mut self.open else {
            return events;
        };
        text.push_str(&delta);
        let (id, index) = (id.clone(), *index);
        events.push(self.event("response.output_text.delta", json!({
            "item_id": id, "output_index": index, "content_index": 0, "delta": delta, "logprobs": []
        })));
        events
    }

    fn tool_call_chunk(
        &mut self,
        chat_index: u32,
        name: Option<String>,
        arguments: Option<String>,
    ) -> Vec<ResponsesEvent> {
        let mut events = Vec::new();
        let continuing = matches!(self.open, Some(OpenItem::ToolCall { .. }))
            && self.open_call == Some(chat_index);
        if !continuing {
            let Some(name) = name.filter(|n| !n.is_empty()) else {
                tracing::warn!("dropping tool_call argument delta with no open call");
                return events;
            };
            events.extend(self.close_open());
            let index = self.next_index();
            let custom = self.custom_tools.contains(&name);
            let id = new_item_id(if custom { "ctc" } else { "fc" });
            let call_id = new_call_id();
            let item = tool_call_item(&id, &call_id, &name, "", custom, "in_progress");
            events.push(self.event(
                "response.output_item.added",
                json!({"output_index": index, "item": item}),
            ));
            self.open = Some(OpenItem::ToolCall {
                id,
                call_id,
                index,
                name,
                arguments: String::new(),
                custom,
            });
            self.open_call = Some(chat_index);
        }
        if let Some(delta) = arguments.filter(|a| !a.is_empty())
            && let Some(OpenItem::ToolCall {
                id,
                index,
                arguments,
                custom,
                ..
            }) = &mut self.open
        {
            arguments.push_str(&delta);
            let (id, index, custom) = (id.clone(), *index, *custom);
            if custom {
                events.push(self.event(
                    "response.custom_tool_call_input.delta",
                    json!({
                        "item_id": id, "output_index": index, "delta": delta
                    }),
                ));
            } else {
                events.push(self.event(
                    "response.function_call_arguments.delta",
                    json!({
                        "item_id": id, "output_index": index, "delta": delta
                    }),
                ));
            }
        }
        events
    }

    fn chunk(&mut self, chunk: CreateChatCompletionStreamResponse) -> Vec<ResponsesEvent> {
        let mut events = Vec::new();
        if let Some(usage) = chunk.usage.as_ref() {
            let prompt = u64::from(usage.prompt_tokens);
            let completion = u64::from(usage.completion_tokens);
            self.usage = Some(usage_value(
                prompt,
                completion,
                usage
                    .prompt_tokens_details
                    .as_ref()
                    .and_then(|d| d.cached_tokens)
                    .map(u64::from)
                    .unwrap_or(0),
                usage
                    .completion_tokens_details
                    .as_ref()
                    .and_then(|d| d.reasoning_tokens)
                    .map(u64::from)
                    .unwrap_or(0),
                if usage.total_tokens > 0 {
                    u64::from(usage.total_tokens)
                } else {
                    prompt + completion
                },
            ));
        }
        let Some(choice) = chunk.choices.into_iter().next() else {
            return events;
        };
        if let Some(reason) = &choice.finish_reason
            && let Ok(value) = serde_json::to_value(reason)
        {
            self.finish_reason = Some(value);
        }
        let delta = choice.delta;
        if let Some(reasoning) = delta.reasoning_content.filter(|r| !r.is_empty()) {
            events.extend(self.reasoning_delta(reasoning));
        }
        let text = match delta.content {
            Some(ChatCompletionMessageContent::Text(text)) if !text.is_empty() => Some(text),
            Some(ChatCompletionMessageContent::Parts(parts)) => {
                Some(serde_json::to_string(&parts).unwrap_or_default())
            }
            _ => None,
        };
        // Calls continuing the open item first, then the text, then new calls.
        let calls: Vec<_> = delta.tool_calls.into_iter().flatten().collect();
        let (continuing, opening): (Vec<_>, Vec<_>) = calls.into_iter().partition(|call| {
            self.open_call == Some(call.index)
                && matches!(self.open, Some(OpenItem::ToolCall { .. }))
        });
        for call in continuing {
            let function = call.function.unwrap_or_default();
            events.extend(self.tool_call_chunk(call.index, function.name, function.arguments));
        }
        if let Some(text) = text {
            events.extend(self.text_delta(text));
        }
        for call in opening {
            let function = call.function.unwrap_or_default();
            events.extend(self.tool_call_chunk(call.index, function.name, function.arguments));
        }
        events
    }

    fn finish(&mut self) -> Vec<ResponsesEvent> {
        let mut events = self.close_open();
        let status = status_from_finish_reason(self.finish_reason.as_ref());
        let error =
            (status == "failed").then(|| error_from_finish_reason(self.finish_reason.as_ref()));
        let output = std::mem::take(&mut self.done);
        let usage = self.usage.take();
        events.push(self.lifecycle(&format!("response.{status}"), status, output, usage, error));
        events
    }

    fn fail(&mut self, message: String) -> ResponsesEvent {
        let error = json!({"code": "server_error", "message": message});
        self.lifecycle("response.failed", "failed", Vec::new(), None, Some(error))
    }
}

fn responses_events(
    chunks: impl Stream<Item = String> + Send + 'static,
    echo: Echo,
    custom_tools: HashSet<String>,
    reasoning_summary: bool,
    encrypted_reasoning: bool,
) -> impl Stream<Item = ResponsesEvent> + Send + 'static {
    async_stream::stream! {
        let mut state = StreamState {
            echo,
            custom_tools,
            reasoning_summary,
            encrypted_reasoning,
            sequence_number: 0,
            output_index: -1,
            open: None,
            open_call: None,
            done: Vec::new(),
            finish_reason: None,
            usage: None,
        };
        yield state.lifecycle("response.created", "in_progress", Vec::new(), None, None);
        yield state.lifecycle("response.in_progress", "in_progress", Vec::new(), None, None);
        futures::pin_mut!(chunks);
        while let Some(frame) = chunks.next().await {
            if frame == "[DONE]" {
                for event in state.finish() {
                    yield event;
                }
                return;
            }
            match serde_json::from_str::<CreateChatCompletionStreamResponse>(&frame) {
                Ok(chunk) => {
                    for event in state.chunk(chunk) {
                        yield event;
                    }
                }
                Err(_) => {
                    let payload: Value = serde_json::from_str(&frame).unwrap_or_default();
                    let message = payload["error"]["message"]
                        .as_str()
                        .unwrap_or("Stream processing error")
                        .to_owned();
                    yield state.fail(message);
                    return;
                }
            }
        }
        for event in state.finish() {
            yield event;
        }
    }
}

/// Python's phase-buffering wrapper: with callable tools, a message item's
/// events wait until it is known whether the message precedes a tool call
/// (`commentary`) or is the final answer.
fn phase_buffered(
    events: impl Stream<Item = ResponsesEvent> + Send + 'static,
    tools_callable: bool,
) -> impl Stream<Item = ResponsesEvent> + Send + 'static {
    async_stream::stream! {
        futures::pin_mut!(events);
        let mut buffered: Vec<ResponsesEvent> = Vec::new();
        while let Some((kind, mut payload)) = events.next().await {
            if !tools_callable {
                yield (kind, payload);
                continue;
            }
            let item_type = payload["item"]["type"].as_str().map(str::to_owned);
            let added = kind == "response.output_item.added";
            let terminal = matches!(kind.as_str(), "response.completed" | "response.incomplete" | "response.failed");
            let opens_call = added && matches!(item_type.as_deref(), Some("function_call" | "custom_tool_call" | "shell_call"));
            if opens_call || terminal {
                let phase = if opens_call { "commentary" } else { "final_answer" };
                for (buffered_kind, mut buffered_payload) in buffered.drain(..) {
                    if buffered_payload["item"]["type"] == "message" {
                        buffered_payload["item"]["phase"] = json!(phase);
                    }
                    yield (buffered_kind, buffered_payload);
                }
                if terminal {
                    for item in payload["response"]["output"].as_array_mut().into_iter().flatten() {
                        if item["type"] == "message" {
                            item["phase"] = json!(phase);
                        }
                    }
                }
                yield (kind, payload);
                continue;
            }
            if added && item_type.as_deref() == Some("message") || !buffered.is_empty() {
                buffered.push((kind, payload));
                continue;
            }
            yield (kind, payload);
        }
        for event in buffered.drain(..) {
            yield event;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Python's input-item rules: a string is one user turn; `instructions`
    /// lead as system; developer turns are labeled; function calls and
    /// their outputs become assistant tool calls and tool turns; a reasoning
    /// item carries its trace; consecutive assistant items merge; an unknown
    /// item type is refused with Python's message.
    #[test]
    fn input_items_convert_like_python() {
        let messages = chat_messages(Some(r#""hi""#), Some("be brief")).unwrap();
        assert_eq!(
            messages,
            vec![
                json!({"role": "system", "content": "be brief"}),
                json!({"role": "user", "content": "hi"})
            ]
        );
        let input = json!([
            {"role": "developer", "content": "dev rules"},
            {"type": "message", "role": "user", "id": "m1", "status": "completed",
             "content": [{"type": "input_text", "text": "look"}, {"type": "input_image", "image_url": "http://x/i.png"}]},
            {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "think"}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "calling"}]},
            {"type": "function_call", "call_id": "call_1", "name": "f", "arguments": "{\"x\": 1}"},
            {"type": "function_call_output", "call_id": "call_1", "output": [{"type": "input_text", "text": "42"}]}
        ]);
        let messages = chat_messages(Some(&input.to_string()), None).unwrap();
        assert_eq!(
            messages[0],
            json!({"role": "system", "content": "Developer instructions:\ndev rules"})
        );
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(
            messages[1]["content"][0],
            json!({"type": "text", "text": "look"})
        );
        assert_eq!(
            messages[1]["content"][1]["image_url"],
            json!({"url": "http://x/i.png", "detail": "auto"})
        );
        assert_eq!(messages[2]["role"], "assistant");
        assert_eq!(messages[2]["reasoning_content"], "think");
        assert_eq!(messages[2]["content"][0]["text"], "calling");
        assert_eq!(
            messages[2]["tool_calls"][0]["function"]["arguments"],
            "{\"x\": 1}"
        );
        assert_eq!(
            messages[3],
            json!({"role": "tool", "tool_call_id": "call_1", "content": "42"})
        );
        let error = chat_messages(Some(r#"[{"type": "web_search_call"}]"#), None).unwrap_err();
        assert_eq!(
            error.message,
            "Unsupported Responses API input item type: \"web_search_call\" None"
        );
    }

    /// `tool_choice` reduces as Python's `effective_tool_choice`, and the
    /// text format maps to the sampling constraint.
    #[test]
    fn tool_choice_and_text_format_reduce_like_python() {
        assert_eq!(effective_tool_choice(None).unwrap(), "auto");
        assert_eq!(
            effective_tool_choice(Some(r#""required""#)).unwrap(),
            "required"
        );
        assert_eq!(
            effective_tool_choice(Some(r#"{"type": "function", "function": {"name": "f"}}"#))
                .unwrap(),
            json!({"type": "function", "name": "f"})
        );
        assert_eq!(
            effective_tool_choice(Some(r#"{"type": "web_search"}"#)).unwrap(),
            "auto"
        );
        assert!(effective_tool_choice(Some(r#""maybe""#)).is_err());
        assert_eq!(
            json_schema_from_text(Some(r#"{"format": {"type": "json_object"}}"#))
                .unwrap()
                .1
                .as_deref(),
            Some(r#"{"type": "object"}"#)
        );
        assert_eq!(
            json_schema_from_text(Some(
                r#"{"format": {"type": "json_schema", "name": "n", "schema": {"type": "object"}}}"#
            ))
            .unwrap()
            .1
            .as_deref(),
            Some(r#"{"type":"object"}"#)
        );
        assert_eq!(
            json_schema_from_text(Some(r#"{"format": {"type": "text"}}"#))
                .unwrap()
                .1,
            None
        );
    }

    /// Encrypted reasoning round-trips through Python's scheme.
    #[test]
    fn encrypted_reasoning_round_trips() {
        let encoded = encode_reasoning_state("why not");
        assert!(encoded.starts_with(ENCRYPTED_REASONING_PREFIX));
        assert_eq!(decode_reasoning_state(&encoded).as_deref(), Some("why not"));
        assert_eq!(decode_reasoning_state("nope"), None);
    }
}
