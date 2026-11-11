//! MiniMax-M3 and M3.1 prompts as SGLang's `serving_chat.py` renders them with
//! the checkpoint's Jinja template, on Dynamo's template formatter.

use std::collections::HashMap;

use dynamo_renderer::PromptFormatter;
use serde_json::{Map, Value, json};

use super::glm53::valid_documents;
use super::pydantic_dump::{
    continues_final_message, engine_message, parse_tool_arguments, split_continuation,
};
use super::qwen38::{prints_unlike_python, template_tools};
use super::template_context::PreservedRequest;
use crate::py::{py_truthy, python_float, python_json_compact};
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

/// `apply_chat_template` flags that default to `False`: falsy, a kwarg is that
/// default, and the template never sees it.
const DEFAULT_FALSE_KWARGS: &[&str] = &["continue_final_message", "return_assistant_tokens_mask"];

/// `serving_chat._M3_THINKING_TYPES`.
const THINKING_TYPES: &[&str] = &["adaptive", "enabled", "disabled"];

/// `serving_chat._M3_REASONING_EFFORTS`.
const REASONING_EFFORTS: &[&str] = &["low", "medium", "high", "xhigh", "max"];

const MEDIA_PARTS: &[&str] = &["image_url", "input_image", "video_url", "audio_url"];

/// `template_detection._is_minimax_m3`: M3's think tags or its tool-call namespace token.
pub(crate) fn is_minimax_m3(template: &str) -> bool {
    template.contains("<mm:think>") || template.contains("]<]minimax[>[")
}

/// The checkpoint template prints tool-call argument scalars through its `to_xml`
/// macro; a custom template may print them otherwise, `tojson` among them.
pub(crate) fn prints_arguments_as_xml(template: &str) -> bool {
    template.contains("macro to_xml(val, ns)") && template.contains("to_xml(v, ns_token)")
}

/// Render an SGLang chat request body. Returns the prompt and the
/// `continue_final_message` prefix, which SGLang tokenizes separately.
pub(crate) fn render(
    formatter: &PromptFormatter,
    xml_arguments: bool,
    content_parts: Option<bool>,
    message_context: &str,
    mut request: Value,
    defaults: &HashMap<String, Value>,
) -> Result<(String, String), RenderError> {
    let Some(Value::Array(messages)) = request.get_mut("messages").map(Value::take) else {
        return Err("messages must be an array".into());
    };
    let mut messages: Vec<Value> = messages.into_iter().map(engine_message).collect();
    validate(&request, &messages).map_err(RenderError::Failed)?;
    let mut args = template_args(&request, defaults)?;
    let tools = template_tools(&request)?;
    for message in &mut messages {
        parse_tool_arguments(message)?;
    }
    // Other templates get numbers, and the check below declines the floats
    // minijinja prints unlike Python.
    if xml_arguments {
        for arguments in messages.iter_mut().flat_map(tool_arguments) {
            python_floats(arguments)?;
        }
    }
    // The template renders tool results by position, not by `tool_call_id`.
    let mut messages = canonical_tool_order(messages)
        .into_iter()
        .map(|message| template_message(message, content_parts))
        .collect::<Result<Vec<_>, _>>()?;
    let prefix = split_continuation(&mut messages, continues_final_message(&request)?);
    // `apply_chat_template` fails on an empty conversation.
    if messages.is_empty() {
        return Err("no messages to render".into());
    }
    // Keep the existing fallback for non-empty lists containing only empty
    // text. In particular, the system/root fixtures still differ on whether
    // such a turn renders the default prompt or an empty block.
    if messages.iter().any(|message| {
        message["content"].as_array().is_some_and(|parts| {
            !parts.is_empty()
                && parts
                    .iter()
                    .all(|part| part["text"].as_str().is_none_or(str::is_empty))
        })
    }) {
        return Err(RenderError::Declined("content parts with no text".into()));
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
    args.insert(message_context.into(), Value::Array(messages));
    let prompt = formatter
        .render(&PreservedRequest { tools, args })
        .map_err(|error| format!("{error:#}"))?;
    Ok((prompt, prefix))
}

/// `_validate_minimax_m3_request`: the thinking type, the effort within the
/// enum, a lone leading `root` message, and every tool call answered.
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
    // The request's own effort (`reasoning_effort`, or the kwargs one when that is
    // unset) must be in the enum; a server default outside it is dropped instead.
    let effort = requested_effort(request)
        .or_else(|| request["chat_template_kwargs"].get("reasoning_effort"))
        .filter(|effort| !effort.is_null());
    if let Some(effort) = effort.filter(|effort| {
        !effort
            .as_str()
            .is_some_and(|effort| REASONING_EFFORTS.contains(&effort))
    }) {
        let shown = match effort.as_str() {
            Some(effort) => format!("'{effort}'"),
            None => python_json_compact(effort),
        };
        return Err(format!(
            "reasoning_effort must be one of {}, got {shown}.",
            REASONING_EFFORTS.join(", ")
        ));
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
    for key in DEFAULT_FALSE_KWARGS {
        if kwargs.get(*key).is_some_and(|value| !py_truthy(value)) {
            kwargs.remove(*key);
        }
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

/// `_normalize_minimax_m3_request`: the thinking config mapped onto the template's
/// `thinking_mode`, and the effort kept only while it is in the enum and thinking
/// is on. An effort never changes the thinking mode.
fn normalize(kwargs: &mut Map<String, Value>, effort: &mut Option<Value>, thinking: &Value) {
    let kwargs_effort = kwargs.remove("reasoning_effort").filter(|e| !e.is_null());
    let mut current = effort.take().or(kwargs_effort);
    if !thinking.is_null() {
        kwargs
            .entry("thinking_mode")
            .or_insert_with(|| thinking["type"].clone());
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
/// `process_content_for_template_format` (the selected template's format) and
/// `normalize_tool_content` leave it.
fn template_message(mut message: Value, content_parts: Option<bool>) -> Result<Value, RenderError> {
    if let Some(fields) = message.as_object_mut() {
        // M3's templates read only the request's tools.
        fields.remove("tools");
    }
    let Some(parts) = message["content"].as_array() else {
        return Ok(message);
    };
    let content_parts = content_parts.ok_or_else(|| {
        RenderError::Declined("cannot determine the template's content format".into())
    })?;
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
    if content_parts && references && (message["role"] == "tool" || texts.is_empty()) {
        return Err(RenderError::Declined("tool_reference parts".into()));
    }
    message["content"] = if !content_parts || message["role"] == "tool" {
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use serde_json::{Value, json};

    use crate::render::{ChatFormatter, TemplateError};

    /// MiniMax-M3 is known by its think tags or its tool-call namespace token.
    #[test]
    fn template_selects_minimax_m3() {
        let formatter = |template: &str| {
            ChatFormatter::from_tokenizer_config(&json!({"chat_template": template})).unwrap()
        };
        assert!(matches!(
            formatter("{# <mm:think> #}"),
            ChatFormatter::MiniMaxM3 { .. }
        ));
        assert!(matches!(
            formatter("{# ]<]minimax[>[ #}"),
            ChatFormatter::MiniMaxM3 { .. }
        ));
        assert!(matches!(
            formatter("{# <think> #}"),
            ChatFormatter::HuggingFace { .. }
        ));
    }

    /// A `minimax_m3_vl` checkpoint served with `template`.
    fn checkpoint(template: &str) -> ChatFormatter {
        ChatFormatter::from_tokenizer_config_with_type(
            &json!({"chat_template": template}),
            Some("minimax_m3_vl"),
        )
        .unwrap()
    }

    fn render(formatter: &ChatFormatter, request: Value) -> Result<String, TemplateError> {
        formatter
            .render_request(request, &HashMap::new())
            .map(|(prompt, _)| prompt.into_text())
    }

    /// History arguments `{"x": <x>}`, answered by a tool result.
    fn tool_history(x: &str) -> Value {
        json!({"messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "a",
                "type": "function",
                "function": {"name": "f", "arguments": format!("{{\"x\":{x}}}")},
            }]},
            {"role": "tool", "tool_call_id": "a", "content": "done"},
        ]})
    }

    /// Only the checkpoint templates print argument scalars through `to_xml`.
    #[test]
    fn checkpoint_templates_print_arguments_as_xml() {
        for name in ["minimax-m3", "minimax-m3.1-flash"] {
            let path = format!(
                "{}/tests/fixtures/parity/{name}.json",
                env!("CARGO_MANIFEST_DIR")
            );
            let fixture: Value =
                serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
            let formatter = checkpoint(fixture["chat_template"].as_str().unwrap());
            assert!(
                matches!(
                    formatter,
                    ChatFormatter::MiniMaxM3 {
                        xml_arguments: true,
                        ..
                    }
                ),
                "{name}"
            );
        }
        assert!(matches!(
            checkpoint("{{ messages | tojson }}"),
            ChatFormatter::MiniMaxM3 {
                xml_arguments: false,
                ..
            }
        ));
    }

    /// A custom template that prints arguments with `tojson` keeps their floats
    /// numbers, as Python does, rather than M3's `repr` strings.
    #[test]
    fn custom_template_keeps_float_arguments() {
        let formatter = checkpoint("{{ messages[1].tool_calls[0].function.arguments | tojson }}");
        assert_eq!(
            render(&formatter, tool_history("1.5")).unwrap(),
            r#"{"x": 1.5}"#
        );
        // A float minijinja prints unlike Python is left to the template path.
        assert!(matches!(
            render(&formatter, tool_history("1e20")),
            Err(TemplateError::NeedsRenderFallback { .. })
        ));
    }

    /// A falsy `continue_final_message` or `return_assistant_tokens_mask` kwarg is
    /// `apply_chat_template`'s default and never reaches the template; a truthy
    /// one still fails the request.
    #[test]
    fn falsy_default_false_kwargs_render() {
        let formatter = checkpoint(
            "{{ messages[0].content }}|{{ continue_final_message is defined }}\
             |{{ return_assistant_tokens_mask is defined }}",
        );
        let request = |kwargs: Value| {
            json!({
                "messages": [{"role": "user", "content": "Hi"}],
                "chat_template_kwargs": kwargs,
            })
        };
        for falsy in [json!(false), json!(null), json!(0), json!("")] {
            let kwargs = json!({
                "continue_final_message": falsy,
                "return_assistant_tokens_mask": falsy,
            });
            assert_eq!(
                render(&formatter, request(kwargs)).unwrap(),
                "Hi|False|False",
                "{falsy}"
            );
        }
        for key in ["continue_final_message", "return_assistant_tokens_mask"] {
            assert!(
                matches!(
                    render(&formatter, request(json!({key: true}))),
                    Err(TemplateError::Renderer { .. })
                ),
                "{key}"
            );
        }
    }
}
