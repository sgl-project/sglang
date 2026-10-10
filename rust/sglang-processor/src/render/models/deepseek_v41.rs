//! DeepSeek-V4.1 prompts as SGLang's `serving_chat.py` builds them, on Dynamo's V4.1 encoder.

use std::collections::HashMap;

use dynamo_renderer::deepseek::{v4::ThinkingMode, v41};
use serde_json::{Map, Value, json};

use super::deepseek_v4::{
    check_tool_arguments, drop_merged_tasks, engine_message, normalize_messages, pydantic_bool,
    split_continuation, thinking,
};
use crate::render::reasoning::requested_effort;

/// Render an SGLang chat request body. Returns the prompt and the
/// `continue_final_message` prefix, which SGLang tokenizes separately.
pub(crate) fn render(
    mut request: Value,
    defaults: &HashMap<String, Value>,
) -> Result<(String, String), String> {
    let Some(Value::Array(messages)) = request.get_mut("messages").map(Value::take) else {
        return Err("messages must be an array".into());
    };
    let mut messages = messages.into_iter().map(engine_message).collect::<Vec<_>>();
    let request = &request;
    for message in &mut messages {
        // Dynamo renders developer turns as system turns, and media expands
        // into image tokens only on the engine.
        if message["role"] == "developer" {
            return Err("DeepSeek-V4.1 developer messages render on the engine".into());
        }
        if message["content"]
            .as_array()
            .is_some_and(|parts| parts.iter().any(|part| part["type"] != "text"))
        {
            return Err("DeepSeek-V4.1 renders only text content parts".into());
        }
        check_tool_arguments(message)?;
    }
    // A parts-list final assistant turn is neither a prefix nor a user turn.
    let prefix = split_continuation(&mut messages, request)?;
    let first = messages.first().ok_or("messages must not be empty")?;
    // Only request tools get the empty system turn, which V4.1 renders as a token.
    let tools = request["tools"]
        .as_array()
        .filter(|tools| !tools.is_empty());
    if tools.is_some() && first["role"] != "system" {
        messages.insert(0, json!({"role": "system", "content": ""}));
    }
    if let Some(task) = request.get("task").filter(|task| !task.is_null()) {
        // `find_last_user_index`: a mid-conversation system turn counts as a user.
        let message = messages
            .iter_mut()
            .enumerate()
            .rev()
            .find(|(index, m)| m["role"] == "user" || (m["role"] == "system" && *index > 0))
            .ok_or("task requires a user or developer message")?
            .1;
        message["task"] = task.clone();
    }
    drop_merged_tasks(&mut messages);
    normalize_messages(&mut messages)?;
    if let Some(tools) = tools {
        messages[0]["tools"] = tools.iter().map(tool_payload).collect::<Result<_, _>>()?;
    }
    let mode = if thinking(request, defaults) {
        ThinkingMode::Thinking
    } else {
        ThinkingMode::Chat
    };
    let prompt = v41::encode_messages(&messages, mode, true, budget(request, defaults))
        .map_err(|error| error.to_string())?;
    Ok((prompt, prefix))
}

/// `chat_encoding.dsv41_tool_payload`: the function fields the client sent,
/// Pydantic-coerced, in OpenAI order. The encoder renders only `function`.
fn tool_payload(tool: &Value) -> Result<Value, String> {
    let f = &tool["function"];
    f["name"].as_str().ok_or("tool function requires name")?;
    let mut function: Map<String, Value> = ["name", "description", "parameters"]
        .into_iter()
        .filter_map(|key| Some((key.into(), f.get(key).filter(|v| !v.is_null())?.clone())))
        .collect();
    if let Some(strict) = f.get("strict") {
        function.insert("strict".into(), pydantic_bool(strict)?.into());
    }
    // `Tool._propagate_defer_loading` fills the function's flag from the tool's.
    if let Some(defer) = [f.get("defer_loading"), tool.get("defer_loading")]
        .into_iter()
        .flatten()
        .find(|v| !v.is_null())
    {
        function.insert("defer_loading".into(), pydantic_bool(defer)?.into());
    }
    Ok(json!({"type": "function", "function": function}))
}

/// `serving_chat._resolve_dsv41_reasoning_effort` as the encoder's 1-100 budget:
/// kwargs effort replaces the request's verbatim, then the server's default
/// kwargs, then `SGLANG_DSV41_REASONING_EFFORT`, else `high`.
fn budget(request: &Value, defaults: &HashMap<String, Value>) -> u8 {
    let kwargs = request["chat_template_kwargs"]
        .get("reasoning_effort")
        .filter(|effort| !effort.is_null());
    let requested = requested_effort(request).map(pydantic_effort);
    let requested = kwargs.cloned().or(requested).or_else(|| {
        defaults
            .get("reasoning_effort")
            .filter(|effort| !effort.is_null())
            .cloned()
    });
    let env = || {
        let raw = std::env::var("SGLANG_DSV41_REASONING_EFFORT").ok()?;
        let raw = raw.trim();
        let value = match raw.parse::<u64>() {
            Ok(budget) => Value::from(budget),
            Err(_) => Value::from(raw),
        };
        effort_budget(&value)
    };
    requested
        .as_ref()
        .and_then(effort_budget)
        .or_else(env)
        .unwrap_or(75)
}

/// The request field's Pydantic type: a tier name or a float.
fn pydantic_effort(effort: &Value) -> Value {
    let number = match effort {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => text.trim().parse::<f64>().ok(),
        _ => None,
    };
    number.map_or_else(|| effort.clone(), Value::from)
}

/// `chat_encoding.parse_dsv41_reasoning_effort`, as the encoder's budget;
/// integers only arrive verbatim through `chat_template_kwargs`.
fn effort_budget(effort: &Value) -> Option<u8> {
    match effort {
        Value::String(tier) => match tier.as_str() {
            "low" => Some(50),
            "high" | "xhigh" => Some(75),
            "max" => Some(100),
            _ => None,
        },
        Value::Number(number) => match number.as_u64() {
            Some(budget) => (1..=100).contains(&budget).then_some(budget as u8),
            None => number
                .as_f64()
                .filter(|effort| (0.0..=0.99).contains(effort))
                .map(|effort| (effort * 100.0).round_ties_even().max(1.0) as u8),
        },
        _ => None,
    }
}
