//! The pydantic dumps of SGLang's `ChatCompletionRequest` that the model
//! renderers build their templates and encoders from.

use serde_json::{Map, Value, json};

use super::schema_types::normalize_json_schema_types;

/// The `continue_final_message` prefix, taken off a final assistant turn with
/// text content; without the flag that turn renders as a user turn.
pub(super) fn split_continuation(
    messages: &mut Vec<Value>,
    request: &Value,
) -> Result<String, String> {
    let Some(last) = messages.last_mut().filter(|m| m["role"] == "assistant") else {
        return Ok(String::new());
    };
    let Some(content) = last["content"].as_str().map(str::to_owned) else {
        return Ok(String::new());
    };
    if request
        .get("continue_final_message")
        .map_or(Ok(false), pydantic_bool)?
    {
        messages.pop();
        return Ok(content);
    }
    *last = json!({"role": "user", "content": content});
    Ok(String::new())
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

/// `protocol.py::Function.model_dump()`, in declared field order, after
/// `_validate_request` normalizes the schema types; the encoder serializes it verbatim.
pub(super) fn normalize_tools(tools: &[Value]) -> Result<Vec<Value>, String> {
    tools
        .iter()
        .map(|tool| {
            let f = &tool["function"];
            let name = f["name"].as_str().ok_or("tool function requires name")?;
            let mut function = Map::new();
            function.insert("description".into(), f["description"].clone());
            function.insert("name".into(), name.into());
            let mut parameters = f["parameters"].clone();
            normalize_json_schema_types(&mut parameters);
            function.insert("parameters".into(), parameters);
            let strict = f.get("strict").map_or(Ok(false), pydantic_bool)?;
            function.insert("strict".into(), strict.into());
            if let Some(defer) = [f.get("defer_loading"), tool.get("defer_loading")]
                .into_iter()
                .flatten()
                .find(|v| !v.is_null())
            {
                function.insert("defer_loading".into(), pydantic_bool(defer)?.into());
            }
            Ok(json!({"type": "function", "function": function}))
        })
        .collect()
}

/// pydantic's lax `bool`: booleans, 0 and 1, and the yes/no words in any case.
pub(super) fn pydantic_bool(value: &Value) -> Result<bool, String> {
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
