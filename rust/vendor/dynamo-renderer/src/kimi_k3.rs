// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native Kimi K3 XTML prompt rendering.
//!
//! K3 does not ship a Jinja chat template. Its model-side `encoding_k3.py`
//! emits a sequence of segments where protocol markers are encoded with
//! tiktoken special IDs and message/tool data is encoded as ordinary text.
//! Keeping that distinction is required both for model parity and to prevent a
//! literal marker in user content from becoming prompt structure.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value};

use crate::{
    OAIChatLikeRequest, OAIPromptFormatter, PromptRenderError, RenderedPrompt, RenderedSegment,
    thinking_bool_from_args,
};

const OPEN_TOKEN: &str = "<|open|>";
const CLOSE_TOKEN: &str = "<|close|>";
const SEP_TOKEN: &str = "<|sep|>";
const END_OF_MSG_TOKEN: &str = "<|end_of_msg|>";
/// The one token this renderer emits per image.
///
/// This is the canonical frontend contract: exactly one `<|media_pad|>` per
/// image, for every engine. It is a registered special token in the K3
/// tokenizer (`config.json`'s `media_placeholder_token_id`), so it encodes to
/// a single id and stays one id no matter what surrounds it.
///
/// The checkpoint's other spelling, `<|kimi_image_placeholder|>`, is a plain
/// string that is *not* in the vocabulary — it BPE-shatters into several ids
/// whose boundaries depend on neighbouring text. Engines that want that form
/// (vLLM) convert from the pad on the worker side, where a single known id is
/// a reliable thing to substitute; matching a shattered string is not.
///
/// Equivalent to calling the checkpoint's own
/// `encoding_k3.build_chat_segments(image_prompts=["<|media_pad|>"] * n)` —
/// `image_prompts` is the model author's hook for exactly this choice, and
/// `<|kimi_image_placeholder|>` is only its `None` fallback.
const MEDIA_PAD: &str = "<|media_pad|>";
const VALID_THINKING_EFFORTS: &[&str] = &["low", "high", "max"];

#[derive(Debug, Clone)]
pub struct KimiK3Formatter {
    exclude_tools_when_tool_choice_none: bool,
}

impl KimiK3Formatter {
    pub fn new(exclude_tools_when_tool_choice_none: bool) -> Self {
        Self {
            exclude_tools_when_tool_choice_none,
        }
    }

    fn build_segments(&self, req: &dyn OAIChatLikeRequest) -> Result<Vec<RenderedSegment>> {
        let messages = json_value(req.messages()).context("Failed to convert K3 messages")?;
        let messages = messages
            .as_array()
            .context("Kimi K3 messages must be an array")?;
        let messages = normalize_tool_result_messages(messages)?;

        let tool_choice = req.tool_choice().map(json_value).transpose()?;
        let (tool_choice_kind, named_tool) = resolve_tool_choice(tool_choice.as_ref())?;
        let mut tools = req.tools().map(json_value).transpose()?;
        if let Some(named_tool) = named_tool
            && !tools
                .as_ref()
                .is_some_and(|tools| contains_tool(tools, named_tool))
        {
            return Err(PromptRenderError::invalid_request(format!(
                "tool named {named_tool:?} in tool_choice is not present in tools"
            ))
            .into());
        }
        if self.exclude_tools_when_tool_choice_none && tool_choice_kind == Some("none") {
            tools = None;
        }
        let tools = tools.map(deep_sort);

        let args = req.chat_template_args();
        // Moonshot's K3 API defines named tool choice as incompatible with
        // thinking. Make the public function-object form work without requiring
        // clients to know K3-specific chat-template arguments.
        let thinking = named_tool.is_none() && thinking_bool_from_args(args).unwrap_or(true);
        let thinking_effort = resolve_thinking_effort(args);
        if thinking && !VALID_THINKING_EFFORTS.contains(&thinking_effort.as_str()) {
            return Err(PromptRenderError::invalid_request(format!(
                "Unsupported Kimi K3 thinking_effort={thinking_effort:?}; supported values are low, high, and max"
            ))
            .into());
        }

        let response_format = req.response_format().map(json_value).transpose()?;
        build_chat_segments(
            &messages,
            tools.as_ref(),
            tool_choice_kind,
            named_tool,
            response_format.as_ref(),
            req.should_add_generation_prompt(),
            thinking,
            thinking_effort.as_str(),
        )
    }
}

impl OAIPromptFormatter for KimiK3Formatter {
    fn supports_add_generation_prompt(&self) -> bool {
        true
    }

    fn render(&self, req: &dyn OAIChatLikeRequest) -> Result<String> {
        Ok(RenderedPrompt::segmented(self.build_segments(req)?).into_text())
    }

    fn render_prompt(&self, req: &dyn OAIChatLikeRequest) -> Result<RenderedPrompt> {
        Ok(RenderedPrompt::segmented(self.build_segments(req)?))
    }
}

fn json_value(value: minijinja::value::Value) -> Result<Value> {
    serde_json::to_value(&value).context("Failed to convert template value to JSON")
}

fn resolve_tool_choice(tool_choice: Option<&Value>) -> Result<(Option<&str>, Option<&str>)> {
    match tool_choice {
        Some(Value::String(kind)) => Ok((Some(kind.as_str()), None)),
        Some(Value::Object(choice)) => {
            if choice.get("type").and_then(Value::as_str) != Some("function") {
                return Err(PromptRenderError::invalid_request(
                    "Kimi K3 named tool_choice must have type=\"function\"",
                )
                .into());
            }
            // Chat Completions uses function.name. Responses API uses a
            // top-level name and is normalized to the same internal request in
            // Dynamo, but accepting both shapes keeps this renderer reusable.
            let name = choice
                .get("function")
                .and_then(Value::as_object)
                .and_then(|function| function.get("name"))
                .or_else(|| choice.get("name"))
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty())
                .ok_or_else(|| {
                    PromptRenderError::invalid_request(
                        "Kimi K3 named tool_choice requires a non-empty function name",
                    )
                })?;
            Ok((Some("specified"), Some(name)))
        }
        Some(Value::Null) | None => Ok((None, None)),
        Some(other) => Err(anyhow::anyhow!(
            "Unsupported Kimi K3 tool_choice value: {other}"
        )),
    }
}

fn contains_tool(tools: &Value, name: &str) -> bool {
    tools.as_array().is_some_and(|tools| {
        tools.iter().any(|tool| {
            tool.get("function")
                .and_then(Value::as_object)
                .and_then(|function| function.get("name"))
                .or_else(|| tool.get("name"))
                .and_then(Value::as_str)
                == Some(name)
        })
    })
}

fn resolve_thinking_effort(args: Option<&HashMap<String, Value>>) -> String {
    args.and_then(|args| {
        args.get("thinking_effort")
            .or_else(|| args.get("reasoning_effort"))
            .and_then(Value::as_str)
    })
    .unwrap_or("max")
    .to_string()
}

fn push_segment(segments: &mut Vec<RenderedSegment>, text: impl Into<String>, allow_special: bool) {
    let text = text.into();
    if !text.is_empty() {
        segments.push(RenderedSegment {
            text,
            allow_special,
        });
    }
}

fn control(segments: &mut Vec<RenderedSegment>, text: impl Into<String>) {
    push_segment(segments, text, true);
}

fn text(segments: &mut Vec<RenderedSegment>, text: impl Into<String>) {
    push_segment(segments, text, false);
}

fn escape_attr_value(value: impl std::fmt::Display) -> String {
    value
        .to_string()
        .replace('&', "&amp;")
        .replace('"', "&quot;")
}

fn open_tag(
    segments: &mut Vec<RenderedSegment>,
    tag: &str,
    attrs: impl IntoIterator<Item = (String, String)>,
) {
    control(segments, OPEN_TOKEN);
    text(segments, tag);
    for (key, value) in attrs {
        text(segments, format!(" {key}"));
        text(segments, "=\"");
        text(segments, escape_attr_value(value));
        text(segments, "\"");
    }
    control(segments, SEP_TOKEN);
}

fn close_tag(segments: &mut Vec<RenderedSegment>, tag: &str) {
    control(segments, CLOSE_TOKEN);
    text(segments, tag);
    control(segments, SEP_TOKEN);
}

fn end_of_msg(segments: &mut Vec<RenderedSegment>) {
    control(segments, END_OF_MSG_TOKEN);
}

fn internal_system_message(segments: &mut Vec<RenderedSegment>, message_type: &str, body: &str) {
    open_tag(
        segments,
        "message",
        [
            ("role".to_string(), "system".to_string()),
            ("type".to_string(), message_type.to_string()),
        ],
    );
    text(segments, body.trim());
    close_tag(segments, "message");
    end_of_msg(segments);
}

fn deep_sort(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut entries: Vec<_> = map.into_iter().collect();
            entries.sort_by(|(left, _), (right, _)| left.cmp(right));
            Value::Object(
                entries
                    .into_iter()
                    .map(|(key, value)| (key, deep_sort(value)))
                    .collect(),
            )
        }
        Value::Array(items) => Value::Array(items.into_iter().map(deep_sort).collect()),
        other => other,
    }
}

fn compact_json(value: &Value) -> Result<String> {
    serde_json::to_string(value).context("Failed to serialize K3 JSON")
}

fn response_schema(response_format: &Value) -> Option<Value> {
    let json_schema = response_format.get("json_schema")?;
    if let Some(schema) = json_schema.get("schema") {
        return Some(schema.clone());
    }
    if let Some(schema) = json_schema.get("json_schema") {
        return Some(schema.clone());
    }
    Some(json_schema.clone())
}

fn value_as_body_text(value: &Value) -> Result<String> {
    match value {
        Value::String(value) => Ok(value.clone()),
        Value::Array(values) if values.iter().all(Value::is_string) => Ok(values
            .iter()
            .filter_map(Value::as_str)
            .filter(|value| !value.is_empty())
            .collect::<Vec<_>>()
            .join("\n")),
        other => compact_json(other),
    }
}

fn render_content_segments(
    segments: &mut Vec<RenderedSegment>,
    content: Option<&Value>,
) -> Result<()> {
    let Some(content) = content else {
        return Ok(());
    };
    match content {
        Value::Null => {}
        Value::String(value) => text(segments, value),
        Value::Array(parts) => {
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("image" | "image_url") => control(segments, MEDIA_PAD),
                    _ => {
                        if let Some(part_text) = part.get("text") {
                            text(segments, value_as_body_text(part_text)?);
                        }
                    }
                }
            }
        }
        other => text(segments, value_as_body_text(other)?),
    }
    Ok(())
}

fn render_role_message(
    segments: &mut Vec<RenderedSegment>,
    message: &Value,
    role: &str,
) -> Result<()> {
    let mut attrs = vec![("role".to_string(), role.to_string())];
    if let Some(name) = message
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
    {
        attrs.push(("name".to_string(), name.to_string()));
    }
    open_tag(segments, "message", attrs);
    render_content_segments(segments, message.get("content"))?;
    close_tag(segments, "message");
    end_of_msg(segments);
    Ok(())
}

fn render_tool_declare(
    segments: &mut Vec<RenderedSegment>,
    tools: &Value,
    dynamic: bool,
) -> Result<()> {
    let tools = compact_json(tools)?;
    let body = if dynamic {
        format!(
            "## New Tools Available\n\
             The system dynamically extends the toolset via lazy-loading.\n\
             You have access to all existing and extended tools.\n\
             Here are the specs for the extended tools.\n\n\
             ```json\n{tools}\n```"
        )
    } else {
        format!(
            "# Tools\n\
             Here are the available tools, described in JSONSchema.\n\n\
             ```json\n{tools}\n```"
        )
    };
    open_tag(
        segments,
        "message",
        [
            ("role".to_string(), "system".to_string()),
            ("type".to_string(), "tool-declare".to_string()),
        ],
    );
    text(segments, body);
    close_tag(segments, "message");
    end_of_msg(segments);
    Ok(())
}

fn xtml_type(value: &Value) -> &'static str {
    match value {
        Value::Bool(_) => "boolean",
        Value::Null => "null",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Object(_) => "object",
        Value::Array(_) => "array",
    }
}

fn xtml_value(value: &Value) -> Result<String> {
    match value {
        Value::String(value) => Ok(value.clone()),
        // Python's `json.dumps(..., ensure_ascii=False)` uses `", "` and
        // `": "` separators by default. Preserve that byte shape in prompt
        // history; the compact form is used only for schemas/tool declarations.
        other => python_default_json(other),
    }
}

fn python_default_json(value: &Value) -> Result<String> {
    let compact = compact_json(value)?;
    let mut output = String::with_capacity(compact.len());
    let mut in_string = false;
    let mut escaped = false;
    for ch in compact.chars() {
        output.push(ch);
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
        } else if ch == '"' {
            in_string = true;
        } else if matches!(ch, ',' | ':') {
            output.push(' ');
        }
    }
    Ok(output)
}

enum NormalizedArguments {
    Object(Map<String, Value>),
    JsonBlock(String),
}

fn normalize_arguments(arguments: Option<&Value>) -> Result<NormalizedArguments> {
    let Some(arguments) = arguments else {
        return Ok(NormalizedArguments::Object(Map::new()));
    };
    match arguments {
        Value::Null => Ok(NormalizedArguments::Object(Map::new())),
        Value::Object(arguments) => Ok(NormalizedArguments::Object(arguments.clone())),
        Value::String(arguments) if arguments.trim().is_empty() => {
            Ok(NormalizedArguments::Object(Map::new()))
        }
        Value::String(arguments) => match serde_json::from_str::<Value>(arguments) {
            Ok(Value::Object(arguments)) => Ok(NormalizedArguments::Object(arguments)),
            Ok(_) => bail!("Kimi K3 tool call arguments must be a JSON object"),
            Err(_) => Ok(NormalizedArguments::JsonBlock(arguments.clone())),
        },
        _ => bail!("Kimi K3 tool call arguments must be an object or JSON object string"),
    }
}

fn render_assistant_segments(
    segments: &mut Vec<RenderedSegment>,
    message: &Value,
    thinking: bool,
) -> Result<()> {
    // Match encoding_k3.py: `reasoning_content` wins when truthy, otherwise
    // fall back to the Responses-style `reasoning` alias.
    let reasoning = message
        .get("reasoning_content")
        .filter(|value| match value {
            Value::Null => false,
            Value::Bool(value) => *value,
            Value::Number(value) => value.as_f64().is_some_and(|value| value != 0.0),
            Value::String(value) => !value.is_empty(),
            Value::Array(value) => !value.is_empty(),
            Value::Object(value) => !value.is_empty(),
        })
        .or_else(|| message.get("reasoning"))
        .map(value_as_body_text)
        .transpose()?;

    // The think channel is structural in the latest K3 model encoding. Every
    // historical assistant message carries it in thinking mode, even if its
    // body is empty. Non-thinking mode drops both the channel and preserved
    // reasoning content.
    if thinking {
        open_tag(segments, "think", []);
        if let Some(reasoning) = reasoning.filter(|reasoning| !reasoning.trim().is_empty()) {
            text(segments, reasoning);
        }
        close_tag(segments, "think");
    }

    open_tag(segments, "response", []);
    render_content_segments(segments, message.get("content"))?;
    close_tag(segments, "response");

    let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) else {
        return Ok(());
    };
    if tool_calls.is_empty() {
        return Ok(());
    }

    open_tag(segments, "tools", []);
    for (position, tool_call) in tool_calls.iter().enumerate() {
        let function = tool_call.get("function").unwrap_or(tool_call);
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .context("Kimi K3 tool call is missing function.name")?;
        open_tag(
            segments,
            "call",
            [
                ("tool".to_string(), name.to_string()),
                ("index".to_string(), (position + 1).to_string()),
            ],
        );

        match normalize_arguments(function.get("arguments"))? {
            NormalizedArguments::JsonBlock(raw) => {
                open_tag(
                    segments,
                    "json",
                    [("type".to_string(), "object".to_string())],
                );
                text(segments, raw);
                close_tag(segments, "json");
            }
            NormalizedArguments::Object(arguments) => {
                for (key, value) in arguments {
                    open_tag(
                        segments,
                        "argument",
                        [
                            ("key".to_string(), key),
                            ("type".to_string(), xtml_type(&value).to_string()),
                        ],
                    );
                    text(segments, xtml_value(&value)?);
                    close_tag(segments, "argument");
                }
            }
        }
        close_tag(segments, "call");
    }
    close_tag(segments, "tools");
    Ok(())
}

fn tool_call_index(tool_calls: Option<&Value>) -> HashMap<String, (usize, Option<String>)> {
    let mut index = HashMap::new();
    let Some(tool_calls) = tool_calls.and_then(Value::as_array) else {
        return index;
    };
    for (position, tool_call) in tool_calls.iter().enumerate() {
        let Some(id) = tool_call.get("id").and_then(Value::as_str) else {
            continue;
        };
        let function = tool_call.get("function").unwrap_or(tool_call);
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .map(str::to_string);
        index.entry(id.to_string()).or_insert((position + 1, name));
    }
    index
}

fn normalize_tool_result_messages(messages: &[Value]) -> Result<Vec<Value>> {
    let mut output = Vec::with_capacity(messages.len());
    let mut current_index = HashMap::new();
    let mut position = 0;

    while position < messages.len() {
        let message = &messages[position];
        let role = message.get("role").and_then(Value::as_str);
        if role == Some("assistant") {
            current_index = tool_call_index(message.get("tool_calls"));
            output.push(message.clone());
            position += 1;
            continue;
        }
        if role != Some("tool") {
            output.push(message.clone());
            position += 1;
            continue;
        }

        let mut run: Vec<(Option<usize>, usize, Value, Option<String>)> = Vec::new();
        let mut unresolved = false;
        let mut offset = 0;
        while position < messages.len()
            && messages[position].get("role").and_then(Value::as_str) == Some("tool")
        {
            let tool_message = &messages[position];
            let call_id = tool_message
                .get("tool_call_id")
                .or_else(|| tool_message.get("id"))
                .and_then(Value::as_str);
            let matched = call_id.and_then(|id| current_index.get(id));
            if let Some((tool_position, name)) = matched {
                run.push((
                    Some(*tool_position),
                    offset,
                    tool_message.clone(),
                    name.clone(),
                ));
            } else {
                unresolved = true;
                run.push((None, offset, tool_message.clone(), None));
            }
            offset += 1;
            position += 1;
        }

        if unresolved {
            output.extend(run.into_iter().map(|(_, _, message, _)| message));
            continue;
        }
        run.sort_by_key(|(tool_position, offset, _, _)| (*tool_position, *offset));
        for (_, _, mut message, name) in run {
            if let (Some(name), Some(message)) = (name, message.as_object_mut()) {
                message.insert("tool".to_string(), Value::String(name.clone()));
                if message.contains_key("name") {
                    message.insert("name".to_string(), Value::String(name));
                }
            }
            output.push(message);
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn build_chat_segments(
    messages: &[Value],
    tools: Option<&Value>,
    tool_choice: Option<&str>,
    named_tool: Option<&str>,
    response_format: Option<&Value>,
    add_generation_prompt: bool,
    thinking: bool,
    thinking_effort: &str,
) -> Result<Vec<RenderedSegment>> {
    let mut segments = Vec::new();
    let mut previous_tool_calls: Option<&Value> = None;
    let mut tool_index = 0usize;

    if let Some(tools) = tools.filter(|tools| !tools.as_array().is_some_and(Vec::is_empty)) {
        render_tool_declare(&mut segments, tools, false)?;
    }

    if thinking {
        internal_system_message(
            &mut segments,
            "thinking-effort",
            &format!(
                "`thinking_effort` guides on how much to think in your thinking channel \
                 (not including the response channel), supported values include `low`, \
                 `medium`, `high`, and `max`.\nNow the system is invoked with \
                 `thinking_effort={thinking_effort}`."
            ),
        );
    }

    for message in messages {
        let role = message.get("role").and_then(Value::as_str).ok_or_else(|| {
            PromptRenderError::invalid_request("Kimi K3 messages must contain a string role")
        })?;
        match role {
            "system" | "developer"
                if message.get("tools").is_some_and(|tools| {
                    !tools.is_null() && !tools.as_array().is_some_and(Vec::is_empty)
                }) =>
            {
                let dynamic_tools = deep_sort(message["tools"].clone());
                render_tool_declare(&mut segments, &dynamic_tools, true)?;
                if role == "developer"
                    && message
                        .get("content")
                        .is_some_and(|content| !content.is_null())
                {
                    render_role_message(&mut segments, message, "system")?;
                }
            }
            "user" | "system" | "developer" => {
                let rendered_role = if role == "developer" { "system" } else { role };
                render_role_message(&mut segments, message, rendered_role)?;
            }
            "assistant" => {
                previous_tool_calls = message.get("tool_calls");
                tool_index = 0;
                let mut attrs = vec![("role".to_string(), "assistant".to_string())];
                if let Some(name) = message
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|name| !name.is_empty())
                {
                    attrs.push(("name".to_string(), name.to_string()));
                }
                open_tag(&mut segments, "message", attrs);
                render_assistant_segments(&mut segments, message, thinking)?;
                close_tag(&mut segments, "message");
                end_of_msg(&mut segments);
            }
            "tool" => {
                tool_index += 1;
                let fallback_name = previous_tool_calls
                    .and_then(Value::as_array)
                    .and_then(|calls| calls.get(tool_index - 1))
                    .map(|call| call.get("function").unwrap_or(call))
                    .and_then(|function| function.get("name"))
                    .and_then(Value::as_str);
                let tool_name = message
                    .get("tool")
                    .or_else(|| message.get("name"))
                    .and_then(Value::as_str)
                    .or(fallback_name)
                    .context(
                        "Kimi K3 tool messages need a tool/name or a preceding assistant tool call",
                    )?;
                open_tag(
                    &mut segments,
                    "message",
                    [
                        ("role".to_string(), "tool".to_string()),
                        ("tool".to_string(), tool_name.to_string()),
                        ("index".to_string(), tool_index.to_string()),
                    ],
                );
                render_content_segments(&mut segments, message.get("content"))?;
                close_tag(&mut segments, "message");
                end_of_msg(&mut segments);
            }
            unsupported => {
                return Err(PromptRenderError::invalid_request(format!(
                    "Kimi K3 does not support message role {unsupported:?}"
                ))
                .into());
            }
        }
    }

    match tool_choice {
        Some("required") => internal_system_message(
            &mut segments,
            "tool-choice",
            "The system is invoked with `tool_choice=required`.\n\
             You MUST call tools in the next message.",
        ),
        Some("none") => internal_system_message(
            &mut segments,
            "tool-choice",
            "The system is invoked with `tool_choice=none`.\n\
             You MUST NOT call any tools in the next message.",
        ),
        Some("specified") => internal_system_message(
            &mut segments,
            "tool-choice",
            &format!(
                "The system is invoked with `tool_choice=specified`.\n\
                 You MUST call the tool `{}` in the next message.",
                named_tool.expect("specified tool_choice has a function name")
            ),
        ),
        _ => {}
    }

    if let Some(response_format) = response_format {
        match response_format.get("type").and_then(Value::as_str) {
            Some("json_object") => internal_system_message(
                &mut segments,
                "response-format",
                "The system is invoked with `response_format=json_object`.\n\
                 Your response must be raw JSON data without markdown code blocks \
                 (```json) or any additional formatting.",
            ),
            Some("json_schema") => {
                let schema = response_schema(response_format)
                    .map(deep_sort)
                    .unwrap_or(Value::Null);
                internal_system_message(
                    &mut segments,
                    "response-format",
                    &format!(
                        "The system is invoked with `response_format=json_schema`.\n\
                         Your response must be raw JSON data without markdown code blocks \
                         (```json) or any additional formatting.\n\
                         The JSON data must match the following schema:\n\
                         ```json\n{}\n```",
                        compact_json(&schema)?
                    ),
                );
            }
            _ => {}
        }
    }

    if add_generation_prompt {
        open_tag(
            &mut segments,
            "message",
            [("role".to_string(), "assistant".to_string())],
        );
        open_tag(
            &mut segments,
            if thinking { "think" } else { "response" },
            [],
        );
    }

    Ok(segments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use minijinja::value::Value as MiniValue;
    use serde_json::json;

    struct Request {
        messages: Value,
        tools: Option<Value>,
        tool_choice: Option<Value>,
        response_format: Option<Value>,
        args: HashMap<String, Value>,
        add_generation_prompt: bool,
    }

    impl Request {
        fn new(messages: Value) -> Self {
            Self {
                messages,
                tools: None,
                tool_choice: None,
                response_format: None,
                args: HashMap::new(),
                add_generation_prompt: true,
            }
        }
    }

    impl OAIChatLikeRequest for Request {
        fn model(&self) -> String {
            "kimi-k3".to_string()
        }

        fn messages(&self) -> MiniValue {
            MiniValue::from_serialize(&self.messages)
        }

        fn tools(&self) -> Option<MiniValue> {
            self.tools.as_ref().map(MiniValue::from_serialize)
        }

        fn tool_choice(&self) -> Option<MiniValue> {
            self.tool_choice.as_ref().map(MiniValue::from_serialize)
        }

        fn response_format(&self) -> Option<MiniValue> {
            self.response_format.as_ref().map(MiniValue::from_serialize)
        }

        fn should_add_generation_prompt(&self) -> bool {
            self.add_generation_prompt
        }

        fn chat_template_args(&self) -> Option<&HashMap<String, Value>> {
            Some(&self.args)
        }
    }

    /// Default formatter: no worker declaration, so the checkpoint token.
    fn fmt() -> KimiK3Formatter {
        KimiK3Formatter::new(true)
    }

    /// One user message carrying a single image part.
    fn image_request() -> Request {
        let mut request = Request::new(json!([{
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "http://example.com/a.png"}}]
        }]));
        request
            .args
            .insert("thinking".to_string(), Value::Bool(false));
        request
    }

    fn image_segments(formatter: &KimiK3Formatter, request: &Request) -> Vec<RenderedSegment> {
        formatter
            .render_prompt(request)
            .unwrap()
            .segments()
            .expect("K3 always renders segmented prompts")
            .to_vec()
    }

    #[test]
    fn renders_one_media_pad_per_image() {
        let segments = image_segments(&fmt(), &image_request());

        let matches: Vec<_> = segments
            .iter()
            .filter(|segment| segment.text == MEDIA_PAD)
            .collect();
        assert_eq!(matches.len(), 1, "exactly one pad per image");
        // The pad MUST stay special: it is a registered token, and only the
        // special-aware encode path yields its single id.
        assert!(matches[0].allow_special);
        // The checkpoint's non-vocabulary spelling must never be emitted --
        // the vLLM worker converts from the pad instead.
        assert!(
            !segments
                .iter()
                .any(|segment| segment.text.contains("kimi_image_placeholder")),
        );
    }

    #[test]
    fn image_token_cardinality_is_one_per_image() {
        let mut request = Request::new(json!([{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "http://example.com/a.png"}},
                {"type": "text", "text": "and"},
                {"type": "image_url", "image_url": {"url": "http://example.com/b.png"}},
                {"type": "text", "text": "compare them"}
            ]
        }]));
        request
            .args
            .insert("thinking".to_string(), Value::Bool(false));

        let segments = image_segments(&fmt(), &request);

        assert_eq!(
            segments
                .iter()
                .filter(|segment| segment.text == MEDIA_PAD)
                .count(),
            2
        );
        // Interleaved prose must stay ordinary text.
        for body in ["and", "compare them"] {
            assert!(
                segments
                    .iter()
                    .any(|segment| segment.text == body && !segment.allow_special)
            );
        }
    }

    #[test]
    fn user_text_spelling_the_pad_stays_ordinary() {
        let body = "please describe <|media_pad|>";
        let mut request = Request::new(json!([{"role": "user", "content": body}]));
        request
            .args
            .insert("thinking".to_string(), Value::Bool(false));

        let segments = image_segments(&fmt(), &request);

        assert!(
            segments
                .iter()
                .any(|segment| segment.text == body && !segment.allow_special),
            "user content must never be promoted into prompt structure"
        );
    }

    #[test]
    fn renders_off_mode_like_model_encoding() {
        let mut request = Request::new(json!([{"role": "user", "content": "Hello"}]));
        request
            .args
            .insert("thinking".to_string(), Value::Bool(false));
        let rendered = fmt().render(&request).unwrap();
        assert_eq!(
            rendered,
            concat!(
                "<|open|>message role=\"user\"<|sep|>Hello",
                "<|close|>message<|sep|><|end_of_msg|>",
                "<|open|>message role=\"assistant\"<|sep|>",
                "<|open|>response<|sep|>"
            )
        );
    }

    #[test]
    fn renders_developer_messages_as_system() {
        let mut request = Request::new(json!([
            {"role": "developer", "content": "Follow this policy", "name": "policy"},
            {"role": "user", "content": "Hello"}
        ]));
        request
            .args
            .insert("thinking".to_string(), Value::Bool(false));

        let rendered = fmt().render(&request).unwrap();

        assert!(
            rendered.contains(
                "<|open|>message role=\"system\" name=\"policy\"<|sep|>Follow this policy"
            )
        );
        assert!(!rendered.contains("role=\"developer\""));
        assert!(
            rendered.find("Follow this policy").unwrap() < rendered.find("Hello").unwrap(),
            "developer instructions must retain their position"
        );
    }

    #[test]
    fn renders_developer_tools_and_content() {
        let mut request = Request::new(json!([
            {
                "role": "developer",
                "content": "Use the newly available tool",
                "tools": [{
                    "type": "function",
                    "function": {"name": "lookup", "parameters": {"type": "object"}}
                }]
            },
            {"role": "user", "content": "Look this up"}
        ]));
        request
            .args
            .insert("thinking".to_string(), Value::Bool(false));

        let rendered = fmt().render(&request).unwrap();

        assert!(rendered.contains("## New Tools Available"));
        assert!(
            rendered.contains("<|open|>message role=\"system\"<|sep|>Use the newly available tool")
        );
    }

    #[test]
    fn rejects_unsupported_message_roles() {
        for role in ["function", "unknown"] {
            let request = Request::new(json!([{"role": role, "content": "ignored before"}]));

            let error = fmt().render(&request).unwrap_err();

            assert!(matches!(
                error.downcast_ref::<PromptRenderError>(),
                Some(PromptRenderError::InvalidRequest(message))
                    if message == &format!("Kimi K3 does not support message role {role:?}")
            ));
        }
    }

    #[test]
    fn rejects_messages_without_a_string_role() {
        for messages in [json!([{"content": "missing"}]), json!([{"role": 7}])] {
            let request = Request::new(messages);

            let error = fmt().render(&request).unwrap_err();

            assert!(matches!(
                error.downcast_ref::<PromptRenderError>(),
                Some(PromptRenderError::InvalidRequest(message))
                    if message == "Kimi K3 messages must contain a string role"
            ));
        }
    }

    #[test]
    fn rejects_unsupported_thinking_effort_as_invalid_request() {
        let mut request = Request::new(json!([{"role": "user", "content": "Hello"}]));
        request.args.insert(
            "thinking_effort".to_string(),
            Value::String("medium".to_string()),
        );

        let error = fmt().render(&request).unwrap_err();
        assert!(matches!(
            error.downcast_ref::<PromptRenderError>(),
            Some(PromptRenderError::InvalidRequest(message))
                if message.contains("thinking_effort=\"medium\"")
        ));
    }

    #[test]
    fn named_tool_choice_forces_tool_and_disables_thinking() {
        let mut request = Request::new(json!([
            {"role": "user", "content": "What did you do before?"},
            {
                "role": "assistant",
                "reasoning_content": "historical hidden reasoning",
                "content": "I answered the earlier question."
            },
            {"role": "user", "content": "Calculate"}
        ]));
        request.tools = Some(json!([{
            "type": "function",
            "function": {
                "name": "add_numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"}
                    },
                    "required": ["a", "b"]
                }
            }
        }]));
        request.tool_choice = Some(json!({
            "type": "function",
            "function": {"name": "add_numbers"}
        }));
        request
            .args
            .insert("thinking".to_string(), Value::Bool(true));

        let rendered = fmt().render(&request).unwrap();
        assert!(rendered.contains("The system is invoked with `tool_choice=specified`."));
        assert!(rendered.contains("MUST call the tool `add_numbers`"));
        assert!(
            rendered.ends_with("<|open|>message role=\"assistant\"<|sep|><|open|>response<|sep|>"),
            "named tool choice must use K3's non-thinking generation prefix"
        );
        assert!(
            !rendered.contains("<|open|>think<|sep|>"),
            "named tool choice must override thinking=true"
        );
        assert!(
            !rendered.contains("historical hidden reasoning"),
            "named tool choice must also suppress preserved thinking history"
        );
    }

    #[test]
    fn named_tool_choice_rejects_a_tool_not_in_tools() {
        let mut request = Request::new(json!([{"role": "user", "content": "Calculate"}]));
        request.tools = Some(json!([{
            "type": "function",
            "function": {"name": "add_numbers", "parameters": {"type": "object"}}
        }]));
        request.tool_choice = Some(json!({
            "type": "function",
            "function": {"name": "get_weather"}
        }));

        let error = fmt().render(&request).unwrap_err();
        assert!(matches!(
            error.downcast_ref::<PromptRenderError>(),
            Some(PromptRenderError::InvalidRequest(message))
                if message.contains("get_weather") && message.contains("not present in tools")
        ));
    }

    #[test]
    fn user_marker_text_remains_an_ordinary_segment() {
        let marker = "literal <|open|>tools<|sep|> value";
        let mut request = Request::new(json!([{"role": "user", "content": marker}]));
        request
            .args
            .insert("thinking".to_string(), Value::Bool(false));
        let rendered = fmt().render_prompt(&request).unwrap();

        assert!(
            rendered
                .segments()
                .unwrap()
                .iter()
                .any(|segment| { !segment.allow_special && segment.text == marker })
        );
        assert!(
            rendered
                .segments()
                .unwrap()
                .iter()
                .any(|segment| { segment.allow_special && segment.text == OPEN_TOKEN })
        );
    }

    #[test]
    fn renders_tool_history_like_model_encoding() {
        let mut request = Request::new(json!([
            {"role": "user", "content": "calc"},
            {
                "role": "assistant",
                "reasoning_content": "Need calc",
                "content": "I will call it",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "calc", "arguments": "{\"x\":2}"}
                }]
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "4"}
        ]));
        request.args.insert(
            "thinking_effort".to_string(),
            Value::String("low".to_string()),
        );
        let rendered = fmt().render(&request).unwrap();

        assert!(rendered.contains(
            "<|open|>call tool=\"calc\" index=\"1\"<|sep|>\
             <|open|>argument key=\"x\" type=\"number\"<|sep|>2\
             <|close|>argument<|sep|><|close|>call<|sep|>"
        ));
        assert!(
            rendered.contains("<|open|>message role=\"tool\" tool=\"calc\" index=\"1\"<|sep|>4")
        );
        assert!(
            rendered.ends_with("<|open|>message role=\"assistant\"<|sep|><|open|>think<|sep|>")
        );
    }

    #[test]
    fn thinking_history_renders_an_empty_think_channel() {
        let request = Request::new(json!([
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "follow-up"}
        ]));

        let rendered = fmt().render(&request).unwrap();

        assert!(rendered.contains(concat!(
            "<|open|>message role=\"assistant\"<|sep|>",
            "<|open|>think<|sep|><|close|>think<|sep|>",
            "<|open|>response<|sep|>answer<|close|>response<|sep|>"
        )));
    }

    #[test]
    fn non_thinking_history_omits_preserved_reasoning() {
        let mut request = Request::new(json!([
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "reasoning_content": "hidden reasoning",
                "content": "answer"
            },
            {"role": "user", "content": "follow-up"}
        ]));
        request
            .args
            .insert("thinking".to_string(), Value::Bool(false));

        let rendered = fmt().render(&request).unwrap();

        assert!(!rendered.contains("hidden reasoning"));
        assert!(!rendered.contains("<|open|>think<|sep|>"));
        assert!(rendered.contains(concat!(
            "<|open|>message role=\"assistant\"<|sep|>",
            "<|open|>response<|sep|>answer<|close|>response<|sep|>"
        )));
    }

    #[test]
    fn tools_are_deep_sorted_before_declaration() {
        let mut request = Request::new(json!([{"role": "user", "content": "Weather?"}]));
        request
            .args
            .insert("thinking".to_string(), Value::Bool(false));
        request.tools = Some(json!([{
            "type": "function",
            "function": {
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                "name": "weather",
                "description": "Get weather"
            }
        }]));
        let rendered = fmt().render(&request).unwrap();
        assert!(rendered.contains(concat!(
            "[{\"function\":{\"description\":\"Get weather\",",
            "\"name\":\"weather\",\"parameters\":{\"properties\":",
            "{\"city\":{\"type\":\"string\"}},\"type\":\"object\"}},",
            "\"type\":\"function\"}]"
        )));
    }

    #[test]
    fn assistant_history_matches_python_json_spacing_and_reasoning_fallback() {
        let request = Request::new(json!([{
            "role": "assistant",
            "reasoning_content": "",
            "reasoning": "fallback",
            "content": null,
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "run",
                    "arguments": {
                        "opts": {"a": 1, "b": [true, false]}
                    }
                }
            }]
        }]));
        let rendered = fmt().render(&request).unwrap();

        assert!(rendered.contains("<|open|>think<|sep|>fallback<|close|>think<|sep|>"));
        assert!(rendered.contains(concat!(
            "<|open|>argument key=\"opts\" type=\"object\"<|sep|>",
            "{\"a\": 1, \"b\": [true, false]}",
            "<|close|>argument<|sep|>"
        )));
    }
}
