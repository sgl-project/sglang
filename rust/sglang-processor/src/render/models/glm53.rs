//! GLM-5.3 prompts as SGLang's `serving_chat.py` renders them with the
//! checkpoint's Jinja template, on Dynamo's template formatter.

use std::collections::HashMap;

use dynamo_renderer::{OAIChatLikeRequest, PromptFormatter};
use serde_json::{Value, json};

use super::deepseek_v4::{engine_message, normalize_tools, split_continuation};
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

/// `template_detection._is_glm53`: GLM-4.5's prompt and tool-call format, with a
/// `Reasoning Effort:` header in place of the `enable_thinking` toggle.
pub(crate) fn is_glm53(template: &str) -> bool {
    [
        "[gMASK]<sop>",
        "Reasoning Effort:",
        "<tool_call>",
        "<arg_key>",
        "<arg_value>",
    ]
    .iter()
    .all(|text| template.contains(text))
        && !template.contains("enable_thinking")
}

/// Render an SGLang chat request body. Returns the prompt and the
/// `continue_final_message` prefix, which SGLang tokenizes separately.
pub(crate) fn render(
    formatter: &PromptFormatter,
    mut request: Value,
    defaults: &HashMap<String, Value>,
) -> Result<(String, String), String> {
    let args = template_args(&request, defaults)?;
    let tools = template_tools(&request)?;
    let Some(Value::Array(messages)) = request.get_mut("messages").map(Value::take) else {
        return Err("messages must be an array".into());
    };
    let mut messages = messages
        .into_iter()
        .map(template_message)
        .collect::<Result<Vec<_>, _>>()?;
    let prefix = split_continuation(&mut messages, &request)?;
    // `apply_chat_template` fails on an empty conversation.
    if messages.is_empty() {
        return Err("no messages to render".into());
    }
    messages.iter_mut().for_each(join_text_parts);
    if messages
        .iter()
        .chain(tools.iter().flatten())
        .any(has_negative_zero)
    {
        return Err("-0 renders on the engine".into());
    }
    // The template strips assistant content, and Python also strips U+001C..U+001F;
    // drop once dynamo-renderer includes ai-dynamo/frontend-crates#395.
    if messages
        .iter()
        .any(|m| m["role"] == "assistant" && has_python_only_space(&m["content"]))
    {
        return Err("U+001C..U+001F in assistant content renders on the engine".into());
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

/// The kwargs SGLang renders with: the request's, over its effort, filled in by
/// the server's defaults.
fn template_args(
    request: &Value,
    defaults: &HashMap<String, Value>,
) -> Result<HashMap<String, Value>, String> {
    let mut kwargs: HashMap<String, Value> = match &request["chat_template_kwargs"] {
        Value::Null => HashMap::new(),
        Value::Object(kwargs) => kwargs.clone().into_iter().collect(),
        _ => return Err("chat_template_kwargs must be an object".into()),
    };
    // `normalize_reasoning_inputs`.
    if let Some(thinking) = requested_thinking(request) {
        kwargs.entry("thinking".into()).or_insert(thinking.into());
        kwargs
            .entry("enable_thinking".into())
            .or_insert(thinking.into());
    }
    // `_convert_to_internal_request`: the kwargs effort replaces the request's.
    let effort = kwargs
        .remove("reasoning_effort")
        .filter(|effort| !effort.is_null())
        .or_else(|| requested_effort(request).cloned());
    // `_process_messages`: the defaults fill in the kwargs, and a default effort
    // stands in for a missing one.
    for (key, value) in defaults {
        kwargs.entry(key.clone()).or_insert_with(|| value.clone());
    }
    if let Some(key) = RESERVED_KWARGS
        .iter()
        .find(|key| kwargs.contains_key(**key))
    {
        return Err(format!("chat_template_kwargs.{key} renders on the engine"));
    }
    if !valid_documents(kwargs.get("documents").unwrap_or(&Value::Null)) {
        return Err("chat_template_kwargs.documents must iterate as objects".into());
    }
    let effort = effort.or_else(|| {
        kwargs
            .get("reasoning_effort")
            .filter(|effort| !effort.is_null())
            .cloned()
    });
    // `_apply_jinja_template`: the kwargs override the effort, a default one included.
    let mut args: HashMap<String, Value> = effort
        .map(|effort| ("reasoning_effort".to_owned(), effort))
        .into_iter()
        .collect();
    args.extend(kwargs);
    Ok(args)
}

/// `_process_messages`: the request tools, pydantic-dumped; only the named tool
/// for a named `tool_choice`, and none for `"none"`.
fn template_tools(request: &Value) -> Result<Option<Vec<Value>>, String> {
    let choice = &request["tool_choice"];
    let Some(tools) = request["tools"]
        .as_array()
        .filter(|tools| !tools.is_empty())
    else {
        return Ok(None);
    };
    if choice == "none" {
        return Ok(None);
    }
    let mut tools = normalize_tools(tools)?;
    if choice.is_object() {
        tools.retain(|tool| tool["function"]["name"] == choice["function"]["name"]);
    }
    Ok((!tools.is_empty()).then_some(tools))
}

/// The message SGLang's Jinja path hands the template: the pydantic dump without
/// null fields, string tool arguments parsed, and the content parts
/// `process_content_for_template_format` keeps.
fn template_message(message: Value) -> Result<Value, String> {
    let mut message = engine_message(message);
    if let Some(fields) = message.as_object_mut() {
        // Only request tools reach the template; Dynamo rejects message-level ones.
        fields.remove("tools");
    }
    if message["role"] == "assistant" {
        parse_tool_arguments(&mut message)?;
    }
    if let Some(parts) = message["content"].as_array() {
        let parts = parts
            .iter()
            .filter(|part| !matches!(part["type"].as_str(), Some("thinking" | "reasoning")))
            .map(|part| match part["type"].as_str() {
                Some("text") => Ok(json!({"type": "text", "text": part["text"]})),
                Some("tool_reference") => {
                    Ok(json!({"type": "tool_reference", "name": part["name"]}))
                }
                _ => Err("GLM-5.3 media renders on the engine"),
            })
            .collect::<Result<Vec<_>, _>>()?;
        // `normalize_tool_content`: a tool result's text parts join with spaces.
        message["content"] =
            if message["role"] == "tool" && parts.iter().all(|p| p["type"] == "text") {
                let texts: Vec<_> = parts
                    .iter()
                    .filter_map(|part| part["text"].as_str())
                    .collect();
                texts.join(" ").into()
            } else {
                parts.into()
            };
    }
    Ok(message)
}

/// `normalize_assistant_tool_call_arguments(strict=True)`: string arguments must
/// parse to a JSON object.
fn parse_tool_arguments(message: &mut Value) -> Result<(), String> {
    let calls = message.get_mut("tool_calls").and_then(Value::as_array_mut);
    for arguments in calls
        .into_iter()
        .flatten()
        .filter_map(|call| call.get_mut("function")?.get_mut("arguments"))
    {
        if let Some(text) = arguments.as_str() {
            *arguments = serde_json::from_str::<Value>(text)
                .ok()
                .filter(Value::is_object)
                .ok_or("assistant tool arguments must be a JSON object")?;
        }
    }
    Ok(())
}

/// GLM-5.3's template concatenates text parts, where Dynamo joins them with newlines.
fn join_text_parts(message: &mut Value) {
    if let Some(parts) = message["content"]
        .as_array()
        .filter(|parts| !parts.is_empty() && parts.iter().all(|part| part["type"] == "text"))
    {
        message["content"] = parts
            .iter()
            .filter_map(|part| part["text"].as_str())
            .collect::<String>()
            .into();
    }
}

/// `render_jinja_template`: Python iterates `documents` and requires each item
/// to be a dict, so only an empty string or object passes besides a list of objects.
pub(super) fn valid_documents(documents: &Value) -> bool {
    match documents {
        Value::Null => true,
        Value::Array(items) => items.iter().all(Value::is_object),
        Value::String(text) => text.is_empty(),
        Value::Object(fields) => fields.is_empty(),
        Value::Bool(_) | Value::Number(_) => false,
    }
}

/// Whether `value` holds a character Python counts as whitespace and Rust does not.
fn has_python_only_space(value: &Value) -> bool {
    match value {
        Value::String(text) => text.chars().any(|c| ('\u{1c}'..='\u{1f}').contains(&c)),
        Value::Array(items) => items.iter().any(has_python_only_space),
        Value::Object(fields) => fields.values().any(has_python_only_space),
        _ => false,
    }
}

/// Whether `value` holds a negative zero: serde_json reads JSON `-0` as `-0.0`,
/// where the engine's orjson reads the integer `0`.
fn has_negative_zero(value: &Value) -> bool {
    match value {
        Value::Number(number) => number
            .as_f64()
            .is_some_and(|f| f == 0.0 && f.is_sign_negative()),
        Value::Array(items) => items.iter().any(has_negative_zero),
        Value::Object(fields) => fields.values().any(has_negative_zero),
        _ => false,
    }
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

    /// GLM-5.3 is known by its template and always thinks; GLM-4.5's toggle is not GLM-5.3.
    #[test]
    fn template_selects_glm53() {
        let template = "[gMASK]<sop>Reasoning Effort: <tool_call><arg_key><arg_value>";
        let formatter = ChatFormatter::from_tokenizer_config(&json!({"chat_template": template}));
        let formatter = formatter.unwrap();
        assert!(matches!(formatter, ChatFormatter::Glm53(_)));
        assert_eq!(
            formatter.resolve_thinking(&mut None, false, false),
            Some(true)
        );
        let glm45 = format!("{template}{{{{ enable_thinking }}}}");
        let formatter = ChatFormatter::from_tokenizer_config(&json!({"chat_template": glm45}));
        assert!(matches!(
            formatter.unwrap(),
            ChatFormatter::HuggingFace { .. }
        ));
    }
}
