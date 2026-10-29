//! MiniMax-M3 and M3.1 prompts as SGLang's `serving_chat.py` renders them with
//! the checkpoint's Jinja template, on Dynamo's template formatter.

use std::collections::HashMap;

use dynamo_renderer::{OAIChatLikeRequest, PromptFormatter};
use serde_json::{Map, Value, json};

use super::glm53::valid_documents;
use super::pydantic_dump::{continues_final_message, engine_message, split_continuation};
use super::qwen38::{prints_unlike_python, template_tools};
use crate::py::python_float;
use crate::render::RenderError;
use crate::render::reasoning::{requested_effort, requested_thinking};

/// `apply_chat_template` arguments SGLang passes itself: as kwargs they fail the
/// render, or `chat_template` renders another template.
const RESERVED_KWARGS: &[&str] = &[
    "add_generation_prompt",
    "chat_template",
    "continue_final_message",
    "conversation",
    "messages",
    "return_assistant_tokens_mask",
    "return_dict",
    "tokenize",
    "tools",
];

/// `serving_chat._M3_THINKING_TYPES`.
const THINKING_TYPES: &[&str] = &["adaptive", "enabled", "disabled"];

/// `serving_chat._M3_REASONING_EFFORTS`.
const REASONING_EFFORTS: &[&str] = &["low", "medium", "high", "xhigh", "max"];

const MEDIA_PARTS: &[&str] = &["image_url", "input_image", "video_url", "audio_url"];

/// `template_detection._is_minimax_m3`: M3's think tags or its tool-call namespace token.
pub(crate) fn is_minimax_m3(template: &str) -> bool {
    template.contains("<mm:think>") || template.contains("]<]minimax[>[")
}

/// Render an SGLang chat request body. Returns the prompt and the
/// `continue_final_message` prefix, which SGLang tokenizes separately.
pub(crate) fn render(
    formatter: &PromptFormatter,
    mut request: Value,
    defaults: &HashMap<String, Value>,
) -> Result<(String, String), RenderError> {
    let Some(Value::Array(messages)) = request.get_mut("messages").map(Value::take) else {
        return Err("messages must be an array".into());
    };
    let mut messages: Vec<Value> = messages.into_iter().map(engine_message).collect();
    validate(&request, &messages).map_err(RenderError::Failed)?;
    let args = template_args(&request, defaults)?;
    let tools = template_tools(&request)?;
    for message in &mut messages {
        parse_tool_arguments(message)?;
    }
    for arguments in messages.iter_mut().flat_map(tool_arguments) {
        python_floats(arguments)?;
    }
    // The template renders tool results by position, not by `tool_call_id`.
    let mut messages = canonical_tool_order(messages)
        .into_iter()
        .map(template_message)
        .collect::<Result<Vec<_>, _>>()?;
    let prefix = split_continuation(&mut messages, continues_final_message(&request)?);
    // `apply_chat_template` fails on an empty conversation.
    if messages.is_empty() {
        return Err("no messages to render".into());
    }
    for message in &mut messages {
        join_text_parts(message)?;
    }
    if messages
        .iter()
        .chain(tools.iter().flatten())
        .any(prints_unlike_python)
    {
        return Err(RenderError::Declined(
            "numbers serde_json prints unlike Python".into(),
        ));
    }
    let PromptFormatter::OAI(formatter) = formatter;
    let prompt = formatter
        .render(&TemplateRequest {
            messages,
            tools,
            args,
        })
        .map_err(|error| format!("{error:#}"))?;
    Ok((prompt, prefix))
}

/// `_validate_minimax_m3_request`: the thinking type, a lone leading `root`
/// message, and every tool call answered.
fn validate(request: &Value, messages: &[Value]) -> Result<(), String> {
    let thinking = &request["thinking"];
    if !thinking.is_null() {
        let kind = &thinking["type"];
        if !kind
            .as_str()
            .is_some_and(|kind| THINKING_TYPES.contains(&kind))
        {
            return Err(format!(
                "thinking.type must be 'adaptive', 'enabled' or 'disabled', got {kind}."
            ));
        }
    }
    let roots: Vec<usize> = (0..messages.len())
        .filter(|&i| messages[i]["role"] == "root")
        .collect();
    if !matches!(roots.as_slice(), [] | [0]) {
        return Err("A root message must be the first message, and at most one is allowed.".into());
    }
    tool_call_pairing(messages)
}

/// `_tool_call_pairing_error`: every assistant tool call is answered by exactly one
/// following tool message, unless that assistant message is the last one.
fn tool_call_pairing(messages: &[Value]) -> Result<(), String> {
    let mut pending: Option<Vec<&Value>> = None;
    let mut unpaired_ids = false;
    for message in messages {
        if message["role"] == "tool" {
            if unpaired_ids {
                continue;
            }
            let Some(pending) = pending.as_mut() else {
                return Err(
                    "A tool message must follow an assistant message with tool_calls.".into(),
                );
            };
            let id = message.get("tool_call_id").unwrap_or(&Value::Null);
            let Some(index) = pending.iter().position(|call_id| *call_id == id) else {
                return Err(format!(
                    "tool_call_id {id} does not match an unanswered tool call."
                ));
            };
            pending.remove(index);
            continue;
        }
        if pending.as_ref().is_some_and(|pending| !pending.is_empty()) {
            return Err("Tool calls have no tool message answering them.".into());
        }
        pending = None;
        unpaired_ids = false;
        let calls = message["tool_calls"]
            .as_array()
            .filter(|calls| !calls.is_empty());
        if let Some(calls) = calls.filter(|_| message["role"] == "assistant") {
            // A set in Python: a repeated id is answered once.
            let mut ids: Vec<&Value> = Vec::new();
            for id in calls.iter().map(|call| &call["id"]) {
                if !ids.contains(&id) {
                    ids.push(id);
                }
            }
            // Calls without ids cannot be paired; leave their results unchecked.
            unpaired_ids = ids.iter().any(|id| id.is_null());
            pending = (!unpaired_ids).then_some(ids);
        }
    }
    if pending.is_some_and(|pending| !pending.is_empty())
        && messages.last().is_some_and(|m| m["role"] == "tool")
    {
        return Err("Tool calls have no tool message answering them.".into());
    }
    Ok(())
}

/// The kwargs SGLang renders with: `_normalize_minimax_m3_request` on the
/// request's own kwargs, then again once the server's defaults fill them in.
fn template_args(
    request: &Value,
    defaults: &HashMap<String, Value>,
) -> Result<HashMap<String, Value>, RenderError> {
    let mut kwargs: Map<String, Value> = match &request["chat_template_kwargs"] {
        Value::Null => Map::new(),
        Value::Object(kwargs) => kwargs.clone(),
        _ => return Err("chat_template_kwargs must be an object".into()),
    };
    // `normalize_reasoning_inputs`.
    if let Some(thinking) = requested_thinking(request) {
        kwargs.entry("thinking").or_insert(thinking.into());
        kwargs.entry("enable_thinking").or_insert(thinking.into());
    }
    // `_convert_to_internal_request`: the kwargs effort replaces the request's.
    let mut effort = kwargs
        .remove("reasoning_effort")
        .filter(|effort| !effort.is_null())
        .or_else(|| requested_effort(request).cloned());
    let thinking = &request["thinking"];
    normalize(&mut kwargs, &mut effort, thinking);
    if !defaults.is_empty() {
        // `_process_messages`: the defaults fill in the kwargs, and a default effort
        // stands in for a missing one.
        for (key, value) in defaults {
            kwargs.entry(key.clone()).or_insert_with(|| value.clone());
        }
        if effort.is_none() {
            effort = kwargs
                .get("reasoning_effort")
                .filter(|e| !e.is_null())
                .cloned();
        }
        normalize(&mut kwargs, &mut effort, thinking);
    }
    if let Some(key) = RESERVED_KWARGS
        .iter()
        .find(|key| kwargs.contains_key(**key))
    {
        // Python's `apply_chat_template` rejects these rather than rendering a
        // different template, so this is a request error (400), not a decline.
        return Err(RenderError::Failed(format!("chat_template_kwargs.{key}")));
    }
    // `render_jinja_template` raises unless `documents` iterates as dicts.
    if !valid_documents(kwargs.get("documents").unwrap_or(&Value::Null)) {
        return Err("chat_template_kwargs.documents must iterate as objects".into());
    }
    // `_apply_jinja_template`: the effort, under the kwargs.
    let mut args: HashMap<String, Value> = effort
        .map(|effort| ("reasoning_effort".to_owned(), effort))
        .into_iter()
        .collect();
    args.extend(kwargs);
    Ok(args)
}

/// `_normalize_minimax_m3_request`: the thinking config and effort aliases mapped
/// onto the template's `thinking_mode` and `reasoning_effort`.
fn normalize(kwargs: &mut Map<String, Value>, effort: &mut Option<Value>, thinking: &Value) {
    let kwargs_effort = kwargs.remove("reasoning_effort").filter(|e| !e.is_null());
    let mut current = effort.take().or(kwargs_effort);
    if !thinking.is_null() {
        kwargs
            .entry("thinking_mode")
            .or_insert_with(|| thinking["type"].clone());
    }
    if current
        .as_ref()
        .is_some_and(|e| e == "none" || e == "minimal")
    {
        kwargs.entry("thinking_mode").or_insert("disabled".into());
    }
    if kwargs
        .get("thinking_mode")
        .is_some_and(|mode| mode == "disabled")
    {
        // An effort means nothing once thinking is off.
        current = None;
    }
    *effort = current.filter(|e| e.as_str().is_some_and(|e| REASONING_EFFORTS.contains(&e)));
}

/// `normalize_assistant_tool_call_arguments(strict=True)`: string arguments must
/// parse to a JSON object.
fn parse_tool_arguments(message: &mut Value) -> Result<(), String> {
    if message["role"] != "assistant" {
        return Ok(());
    }
    let calls = message.get_mut("tool_calls").and_then(Value::as_array_mut);
    for arguments in calls
        .into_iter()
        .flatten()
        .filter_map(|call| call.get_mut("function")?.get_mut("arguments"))
    {
        if let Some(text) = arguments.as_str() {
            let parsed = serde_json::from_str::<Value>(text)
                .map_err(|_| "Assistant tool call function.arguments must be valid JSON.")?;
            if !parsed.is_object() {
                return Err("Assistant tool call function.arguments must be a JSON object.".into());
            }
            *arguments = parsed;
        }
    }
    Ok(())
}

/// Each assistant tool call's parsed `function.arguments`.
fn tool_arguments(message: &mut Value) -> impl Iterator<Item = &mut Value> {
    let is_assistant = message["role"] == "assistant";
    message
        .get_mut("tool_calls")
        .filter(|_| is_assistant)
        .and_then(Value::as_array_mut)
        .into_iter()
        .flatten()
        .filter_map(|call| call.get_mut("function")?.get_mut("arguments"))
}

/// The template prints argument scalars with `{{ val }}`, so floats (orjson reads
/// integers beyond 64 bits as floats too) reach it as Python's `repr` of them.
/// serde_json reads JSON `-0` as `-0.0` where orjson reads the integer `0`.
fn python_floats(value: &mut Value) -> Result<(), RenderError> {
    match value {
        Value::Number(number) if number.is_f64() => {
            let float = number.as_f64().unwrap_or_default();
            if float == 0.0 && float.is_sign_negative() {
                return Err(RenderError::Declined("-0".into()));
            }
            *value = python_float(float).into();
        }
        Value::Array(items) => items.iter_mut().try_for_each(python_floats)?,
        Value::Object(fields) => fields.values_mut().try_for_each(python_floats)?,
        _ => {}
    }
    Ok(())
}

/// `_canonicalize_tool_message_order(sort_text_runs=True)`: each run of tool results
/// after an assistant's tool calls, sorted into call order when the ids pair one to one.
fn canonical_tool_order(messages: Vec<Value>) -> Vec<Value> {
    let mut canonical = Vec::with_capacity(messages.len());
    let mut messages = messages.into_iter().peekable();
    while let Some(message) = messages.next() {
        let call_ids: Vec<Value> = message["tool_calls"]
            .as_array()
            .filter(|_| message["role"] == "assistant")
            .into_iter()
            .flatten()
            .map(|call| call["id"].clone())
            .collect();
        canonical.push(message);
        if call_ids.is_empty() {
            continue;
        }
        let mut run = Vec::new();
        while let Some(result) =
            messages.next_if(|m| m["role"] == "tool" || m["role"] == "function")
        {
            run.push(result);
        }
        let position = |message: &Value| {
            let id = message.get("tool_call_id").unwrap_or(&Value::Null);
            call_ids.iter().position(|call_id| call_id == id)
        };
        let unique = |ids: &[&Value]| (1..ids.len()).all(|i| !ids[..i].contains(&ids[i]));
        let result_ids: Vec<&Value> = run
            .iter()
            .map(|m| m.get("tool_call_id").unwrap_or(&Value::Null))
            .collect();
        let sortable = run.len() >= 2
            && call_ids.iter().all(|id| !id.is_null())
            && unique(&call_ids.iter().collect::<Vec<_>>())
            && run.iter().all(|m| position(m).is_some())
            && unique(&result_ids);
        if sortable {
            run.sort_by_key(|m| position(m));
        }
        canonical.extend(run);
    }
    canonical
}

/// The message SGLang's Jinja path hands the template: content as
/// `process_content_for_template_format` (the openai format) and
/// `normalize_tool_content` leave it.
fn template_message(mut message: Value) -> Result<Value, RenderError> {
    if let Some(fields) = message.as_object_mut() {
        // Only request tools reach the template; Dynamo rejects message-level ones.
        fields.remove("tools");
    }
    let Some(parts) = message["content"].as_array() else {
        return Ok(message);
    };
    let mut texts = Vec::new();
    let mut references = false;
    for part in parts {
        match part["type"].as_str() {
            Some("text" | "input_text") => texts.push(part["text"].clone()),
            Some(kind) if MEDIA_PARTS.contains(&kind) => {
                return Err(RenderError::Declined("MiniMax-M3 media".into()));
            }
            Some("tool_reference") => references = true,
            _ => {}
        }
    }
    // The template skips a `tool_reference` part, but prints one in a tool result as
    // a Python dict, and a turn of only references still counts as content.
    if references && (message["role"] == "tool" || texts.is_empty()) {
        return Err(RenderError::Declined("tool_reference parts".into()));
    }
    message["content"] = if message["role"] == "tool" {
        let texts: Vec<_> = texts.iter().filter_map(Value::as_str).collect();
        texts.join(" ").into()
    } else {
        texts
            .into_iter()
            .map(|text| json!({"type": "text", "text": text}))
            .collect()
    };
    Ok(message)
}

/// The template concatenates text parts, where Dynamo joins them with newlines.
///
/// A non-empty parts list whose texts join to `""` is truthy in the template's
/// `{%- if message.content -%}` checks — Python hands it the list, so a system
/// or root message renders an empty block rather than the default prompt.
/// Joining it to `""` would be falsy, so decline instead of mis-rendering.
fn join_text_parts(message: &mut Value) -> Result<(), RenderError> {
    if let Some(parts) = message["content"]
        .as_array()
        .filter(|parts| !parts.is_empty())
    {
        let text: String = parts
            .iter()
            .filter_map(|part| part["text"].as_str())
            .collect();
        if text.is_empty() {
            return Err(RenderError::Declined("content parts with no text".into()));
        }
        message["content"] = text.into();
    }
    Ok(())
}

/// The template context SGLang builds, as the request Dynamo renders.
struct TemplateRequest {
    messages: Vec<Value>,
    tools: Option<Vec<Value>>,
    args: HashMap<String, Value>,
}

impl OAIChatLikeRequest for TemplateRequest {
    fn model(&self) -> String {
        String::new()
    }

    fn messages(&self) -> minijinja::Value {
        minijinja::Value::from_serialize(&self.messages)
    }

    fn tools(&self) -> Option<minijinja::Value> {
        self.tools.as_ref().map(minijinja::Value::from_serialize)
    }

    fn should_add_generation_prompt(&self) -> bool {
        true
    }

    fn chat_template_args(&self) -> Option<&HashMap<String, Value>> {
        Some(&self.args)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::render::ChatFormatter;

    /// MiniMax-M3 is known by its think tags or its tool-call namespace token.
    #[test]
    fn template_selects_minimax_m3() {
        let formatter = |template: &str| {
            ChatFormatter::from_tokenizer_config(&json!({"chat_template": template})).unwrap()
        };
        assert!(matches!(
            formatter("{# <mm:think> #}"),
            ChatFormatter::MiniMaxM3(_)
        ));
        assert!(matches!(
            formatter("{# ]<]minimax[>[ #}"),
            ChatFormatter::MiniMaxM3(_)
        ));
        assert!(matches!(
            formatter("{# <think> #}"),
            ChatFormatter::HuggingFace { .. }
        ));
    }
}
