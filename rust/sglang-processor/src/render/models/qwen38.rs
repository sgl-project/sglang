//! Qwen3.8 prompts as SGLang's `serving_chat.py` renders them with the
//! checkpoint's Jinja template, on Dynamo's template formatter.

use std::collections::HashMap;

use dynamo_renderer::{OAIChatLikeRequest, PromptFormatter};
use serde_json::{Value, json};

use super::glm53::valid_documents;
use super::pydantic_dump::{engine_message, normalize_tools, pydantic_bool, split_continuation};
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

const MEDIA_PARTS: &[&str] = &["image_url", "video_url", "audio_url", "input_audio"];

/// Qwen3.8's templates (Max, 27B, Flash-Next): Qwen3-Coder's tool-call format,
/// with a `reasoning_effort` instruction and `preserve_thinking`.
pub(crate) fn is_qwen38(template: &str) -> bool {
    [
        "<|im_start|>",
        "<function=",
        "<parameter=",
        "preserve_thinking",
        "Reasoning effort is set to",
    ]
    .iter()
    .all(|text| template.contains(text))
}

/// `detect_jinja_template_content_format`: the vision template (27B, Flash-Next)
/// takes content parts lists; Max's text-only template takes strings.
pub(crate) fn takes_content_parts(template: &str) -> bool {
    ["image", "audio", "video", "vision"]
        .iter()
        .any(|keyword| template.contains(keyword))
}

/// Render an SGLang chat request body. Returns the prompt and the
/// `continue_final_message` prefix, which SGLang tokenizes separately.
pub(crate) fn render(
    formatter: &PromptFormatter,
    content_parts: bool,
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
        .map(|message| template_message(message, content_parts))
        .collect::<Result<Vec<_>, _>>()?;
    let prefix = split_continuation(&mut messages, &request)?;
    // The template raises on a later system turn, which Dynamo would demote to a user turn.
    if messages.iter().skip(1).any(|m| m["role"] == "system") {
        return Err("System message must be at the beginning.".into());
    }
    messages.iter_mut().for_each(join_text_parts);
    if messages.iter().any(trims_unlike_python) {
        return Err("text Python's trim strips unlike minijinja's renders on the engine".into());
    }
    if messages
        .iter()
        .chain(tools.iter().flatten())
        .any(prints_unlike_python)
    {
        return Err("numbers serde_json prints unlike Python render on the engine".into());
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
    // `render_jinja_template` raises unless `documents` iterates as dicts.
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

/// `_process_messages`: the request tools as `Tool.model_dump()` gives them; only
/// the named tool for a named `tool_choice`, and none for `"none"`.
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
    let mut tools = normalize_tools(tools)?
        .into_iter()
        .zip(tools)
        .map(|(mut dump, tool)| {
            let defer = tool.get("defer_loading").filter(|v| !v.is_null());
            dump["defer_loading"] = defer.map(pydantic_bool).transpose()?.into();
            // `Tool.type` is any string, dumped as sent.
            if let Some(kind) = tool.get("type").filter(|kind| kind.is_string()) {
                dump["type"] = kind.clone();
            }
            Ok(dump)
        })
        .collect::<Result<Vec<_>, String>>()?;
    if choice.is_object() {
        tools.retain(|tool| tool["function"]["name"] == choice["function"]["name"]);
    }
    Ok((!tools.is_empty()).then_some(tools))
}

/// The message SGLang's Jinja path hands the template: the pydantic dump without
/// null fields, tool calls dumped, and content as
/// `process_content_for_template_format` and `normalize_tool_content` leave it.
fn template_message(message: Value, content_parts: bool) -> Result<Value, String> {
    let mut message = engine_message(message);
    if let Some(fields) = message.as_object_mut() {
        // Only request tools reach the template; Dynamo rejects message-level ones.
        fields.remove("tools");
    }
    if message["role"] == "assistant" {
        dump_tool_calls(&mut message)?;
    }
    let Some(parts) = message["content"].as_array() else {
        return Ok(message);
    };
    if parts
        .iter()
        .any(|part| MEDIA_PARTS.contains(&part["type"].as_str().unwrap_or_default()))
    {
        return Err("Qwen3.8 media renders on the engine".into());
    }
    // Strings keep only text parts, parts lists also tool references; strings and a
    // tool result's text parts join with spaces.
    let kept: Vec<_> = parts
        .iter()
        .filter_map(|part| match part["type"].as_str() {
            Some("text") => Some(json!({"type": "text", "text": part["text"]})),
            Some("tool_reference") if content_parts => {
                Some(json!({"type": "tool_reference", "name": part["name"]}))
            }
            _ => None,
        })
        .collect();
    let all_text = kept.iter().all(|part| part["type"] == "text");
    message["content"] = if !content_parts || (message["role"] == "tool" && all_text) {
        let texts: Vec<_> = kept
            .iter()
            .filter_map(|part| part["text"].as_str())
            .collect();
        texts.join(" ").into()
    } else {
        kept.into()
    };
    Ok(message)
}

/// `FunctionResponse.model_dump()`, after `normalize_assistant_tool_call_arguments(strict=True)`:
/// each function as its name and arguments, null when unset, and string arguments
/// parsed to a JSON object.
fn dump_tool_calls(message: &mut Value) -> Result<(), String> {
    let calls = message.get_mut("tool_calls").and_then(Value::as_array_mut);
    for function in calls
        .into_iter()
        .flatten()
        .filter_map(|call| call.get_mut("function"))
    {
        let arguments = match &function["arguments"] {
            Value::String(text) => serde_json::from_str::<Value>(text)
                .ok()
                .filter(Value::is_object)
                .ok_or("assistant tool arguments must be a JSON object")?,
            arguments => arguments.clone(),
        };
        *function = json!({"name": function["name"], "arguments": arguments});
    }
    Ok(())
}

/// The template concatenates text parts, where Dynamo joins them with newlines.
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

/// Whether the template's `trim` of the message's text differs: Python's also strips
/// the ASCII separators U+001C..U+001F, which minijinja's keeps.
fn trims_unlike_python(message: &Value) -> bool {
    let separator = |c: char| ('\u{1c}'..='\u{1f}').contains(&c);
    ["content", "reasoning_content"]
        .iter()
        .filter_map(|key| message[*key].as_str())
        .map(str::trim)
        .any(|text| text.starts_with(separator) || text.ends_with(separator))
}

/// Whether `value` holds a number Dynamo's serde_json prints unlike Python's
/// `json.dumps`: floats in exponent notation (`1e-06`, `1e+16`), integers
/// beyond 64 bits, and negative zero, which serde_json also reads from `-0`.
fn prints_unlike_python(value: &Value) -> bool {
    match value {
        Value::Number(number) => number.as_f64().is_some_and(|f| {
            let positive_zero = f == 0.0 && f.is_sign_positive();
            number.is_f64() && !positive_zero && !(1e-4..1e16).contains(&f.abs())
        }),
        Value::Array(items) => items.iter().any(prints_unlike_python),
        Value::Object(fields) => fields.values().any(prints_unlike_python),
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

    /// Qwen3.8 is known by its template, which takes content parts lists only when it
    /// renders media; Qwen3's template is not Qwen3.8.
    #[test]
    fn template_selects_qwen38() {
        let template =
            "<|im_start|><function=<parameter=preserve_thinkingReasoning effort is set to";
        let formatter = |template: String| {
            ChatFormatter::from_tokenizer_config(&json!({"chat_template": template})).unwrap()
        };
        assert!(matches!(
            formatter(template.into()),
            ChatFormatter::Qwen38 {
                content_parts: false,
                ..
            }
        ));
        assert!(matches!(
            formatter(format!("{template}<|vision_start|>")),
            ChatFormatter::Qwen38 {
                content_parts: true,
                ..
            }
        ));
        assert!(matches!(
            formatter(template.replace("preserve_thinking", "")),
            ChatFormatter::HuggingFace { .. }
        ));
    }
}
