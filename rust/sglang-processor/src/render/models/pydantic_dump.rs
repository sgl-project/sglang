//! The pydantic dumps of SGLang's `ChatCompletionRequest` that the model
//! renderers build their templates and encoders from.

use serde_json::{Map, Value, json};

use super::schema_types::normalize_json_schema_types;

/// The request's `continue_final_message`, coerced as pydantic's `bool`.
pub fn continues_final_message(request: &Value) -> Result<bool, String> {
    request
        .get("continue_final_message")
        .map_or(Ok(false), pydantic_bool)
}

/// `_handle_last_assistant_message`: the `continue_final_message` prefix, taken
/// off a final assistant turn with text content; without the flag that turn
/// renders as a user turn.
pub(crate) fn split_continuation(messages: &mut Vec<Value>, continuing: bool) -> String {
    let Some(last) = messages.last_mut().filter(|m| m["role"] == "assistant") else {
        return String::new();
    };
    let Some(content) = last["content"].as_str().map(str::to_owned) else {
        return String::new();
    };
    if continuing {
        messages.pop();
        return content;
    }
    *last = json!({"role": "user", "content": content});
    String::new()
}

/// The pydantic dump SGLang renders: roles lowercased, unknown and null fields
/// dropped, `user` reduced to role and content, null content blanked.
pub(super) fn engine_message(message: Value) -> Value {
    let mut message = match message {
        Value::Object(message) => message,
        _ => Map::new(),
    };
    let role = message
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_lowercase();
    let kept: &[&str] = if role == "user" {
        &["content"]
    } else {
        &[
            "content",
            "tool_call_id",
            "name",
            "reasoning_content",
            "tool_calls",
            "tools",
        ]
    };
    message.retain(|key, value| kept.contains(&key.as_str()) && !value.is_null());
    message.insert("role".into(), role.into());
    message.entry("content").or_insert_with(|| "".into());
    if let Some(Value::Array(calls)) = message.get_mut("tool_calls") {
        calls
            .iter_mut()
            .for_each(|call| *call = tool_call_dump(call));
    }
    message.into()
}

/// `ToolCall.model_dump()`: its declared fields only, unset ones null.
fn tool_call_dump(call: &Value) -> Value {
    let function = &call["function"];
    json!({
        "id": call["id"],
        "index": call["index"],
        "type": "function",
        "function": {"name": function["name"], "arguments": function["arguments"]},
    })
}

/// Each assistant tool call's `function.arguments`, absent ones as `null`.
pub(super) fn tool_arguments(message: &mut Value) -> impl Iterator<Item = &mut Value> {
    let is_assistant = message["role"] == "assistant";
    message
        .get_mut("tool_calls")
        .filter(|_| is_assistant)
        .and_then(Value::as_array_mut)
        .into_iter()
        .flatten()
        .filter_map(|call| call.get_mut("function")?.as_object_mut())
        .map(|function| function.entry("arguments").or_insert(Value::Null))
}

/// `Tool.model_dump()`'s `function` for each request tool, typed `"function"`;
/// the encoder serializes it verbatim.
pub(super) fn normalize_tools(tools: &[Value]) -> Result<Vec<Value>, String> {
    tools
        .iter()
        .map(|tool| Ok(json!({"type": "function", "function": function_dump(tool)?})))
        .collect()
}

/// `protocol.py::Function.model_dump()`, after `Tool._propagate_defer_loading`
/// and `_validate_request`'s schema-type normalization. `Function._serialize`
/// keeps the keys in the order the client sent them, then the rest as declared,
/// and leaves out an unset `strict` or `defer_loading`.
pub(crate) fn function_dump(tool: &Value) -> Result<Map<String, Value>, String> {
    let f = &tool["function"];
    let name = f["name"].as_str().ok_or("tool function requires name")?;
    let flag = |value: Option<&Value>| {
        value
            .filter(|value| !value.is_null())
            .map(pydantic_bool)
            .transpose()
    };
    let defer = match flag(f.get("defer_loading"))? {
        Some(defer) => Some(defer),
        None => flag(tool.get("defer_loading"))?,
    };
    let mut parameters = f["parameters"].clone();
    normalize_json_schema_types(&mut parameters);
    let declared = [
        ("description", Some(f["description"].clone())),
        ("name", Some(name.into())),
        ("parameters", Some(parameters)),
        ("strict", flag(f.get("strict"))?.map(Value::from)),
        ("defer_loading", defer.map(Value::from)),
    ];
    let sent = f
        .as_object()
        .into_iter()
        .flat_map(|function| function.keys().map(String::as_str));
    let mut function = Map::new();
    for key in sent.chain(declared.iter().map(|(key, _)| *key)) {
        if let Some((_, Some(value))) = declared.iter().find(|(name, _)| *name == key) {
            function.entry(key).or_insert_with(|| value.clone());
        }
    }
    Ok(function)
}

/// pydantic's lax `bool`: booleans, 0 and 1, and the yes/no words in any case.
pub(crate) fn pydantic_bool(value: &Value) -> Result<bool, String> {
    let parsed = match value {
        Value::Bool(value) => Some(*value),
        Value::Number(number) => match number.as_f64() {
            Some(0.0) => Some(false),
            Some(1.0) => Some(true),
            _ => None,
        },
        Value::String(text) => match text.to_lowercase().as_str() {
            "0" | "off" | "f" | "false" | "n" | "no" => Some(false),
            "1" | "on" | "t" | "true" | "y" | "yes" => Some(true),
            _ => None,
        },
        _ => None,
    };
    parsed.ok_or_else(|| format!("expected a boolean, got {value}"))
}
