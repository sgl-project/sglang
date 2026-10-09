// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! SGLang's native DeepSeek-V4.1 serving semantics around Dynamo's V4.1 encoder.
//! DeepSeek-V4 renders through `sglang_processor`.

use anyhow::{ensure, Context, Result};
use dynamo_renderer::deepseek::{v4::ThinkingMode, v41};
use serde_json::{json, Map, Value};
use sglang_processor::DeepSeekV4Profile;

use super::chat_formatter::ChatTemplateKwargs;

#[derive(Clone, Copy)]
pub(super) enum Encoder {
    V4(DeepSeekV4Profile),
    V41,
}

/// `serving_chat._convert_to_internal_request`: `chat_template_kwargs.reasoning_effort`
/// replaces the validated request effort verbatim (thinking was already derived
/// from the request fields); request-level values are coerced as pydantic does.
pub(super) fn request_effort(request: &Value) -> Option<Value> {
    if let Some(effort) = request["chat_template_kwargs"]
        .get("reasoning_effort")
        .filter(|v| !v.is_null())
    {
        return Some(effort.clone());
    }
    let effort = [
        request["reasoning"].get("effort"),
        request["reasoning"].get("reasoning_effort"),
        request.get("reasoning_effort"),
    ]
    .into_iter()
    .flatten()
    .find(|v| !v.is_null())?;
    Some(match effort {
        Value::String(s) => s
            .parse::<f64>()
            .map_or_else(|_| effort.clone(), Value::from),
        Value::Number(n) => n.as_f64().map_or_else(|| effort.clone(), Value::from),
        _ => effort.clone(),
    })
}

/// Before continuation extraction, SGLang parses assistant arguments as JSON
/// objects. Dynamo expects those arguments serialized, but must not apply its
/// permissive malformed-JSON fallback here. V4.1 keeps parts lists (its encoder
/// joins them), so a parts-list final assistant turn is not a continuation prefix.
pub(super) fn normalize_messages(messages: &mut [Value]) -> Result<()> {
    for message in messages {
        if let Some(tools) = message["tools"].as_array() {
            if tools.is_empty() {
                message.as_object_mut().unwrap().remove("tools");
            } else {
                message["tools"] = normalize_tools(tools)?.into();
            }
        }
        if message["role"] == "assistant" {
            if message["tool_calls"].as_array().is_some_and(Vec::is_empty) {
                message.as_object_mut().unwrap().remove("tool_calls");
            }
            for call in message["tool_calls"].as_array_mut().into_iter().flatten() {
                let arguments = &mut call["function"]["arguments"];
                let parsed = match arguments.as_str() {
                    Some(text) => serde_json::from_str::<Value>(text)
                        .context("assistant tool arguments must be valid JSON")?,
                    None => arguments.clone(),
                };
                ensure!(
                    parsed.is_object(),
                    "assistant tool arguments must be a JSON object"
                );
                *arguments = serde_json::to_string(&parsed)?.into();
            }
        }
    }
    Ok(())
}

/// protocol.py::Function.model_dump(), including declared field order and
/// defaults. The native encoder serializes this dictionary verbatim.
fn normalize_tools(tools: &[Value]) -> Result<Vec<Value>> {
    tools
        .iter()
        .map(|tool| {
            let f = &tool["function"];
            let name = f["name"].as_str().context("tool function requires name")?;
            let mut function = Map::new();
            function.insert("description".into(), f["description"].clone());
            function.insert("name".into(), name.into());
            function.insert("parameters".into(), f["parameters"].clone());
            function.insert(
                "strict".into(),
                f.get("strict").cloned().unwrap_or(false.into()),
            );
            if let Some(defer) = f
                .get("defer_loading")
                .filter(|v| !v.is_null())
                .or_else(|| tool.get("defer_loading").filter(|v| !v.is_null()))
            {
                function.insert("defer_loading".into(), defer.clone());
            }
            Ok(json!({"type":"function", "function":function}))
        })
        .collect()
}

/// Use Dynamo's low-level encoder so SGLang, rather than Dynamo's OpenAI
/// defaults, controls tool selection and the numeric reasoning budget.
pub(super) fn render_v41(
    request: &Value,
    mut messages: Vec<Value>,
    kwargs: &ChatTemplateKwargs,
    effort: Option<Value>,
) -> Result<String> {
    // SGLang drops a later user's task when merging it into an existing
    // user/tool-result turn. Dynamo otherwise preserves that field.
    for index in 1..messages.len() {
        if messages[index]["role"] == "user"
            && matches!(messages[index - 1]["role"].as_str(), Some("user" | "tool"))
        {
            messages[index].as_object_mut().unwrap().remove("task");
        }
    }
    ensure!(
        !messages.is_empty(),
        "DeepSeek requires messages after continuation extraction"
    );
    let thinking = kwargs
        .get("thinking")
        .is_some_and(|v| minijinja::Value::from_serialize(v).is_true());
    // Dynamo maps developer to system, unlike this SGLang encoder. Leave
    // that unsupported shape to the worker instead of forwarding wrong IDs.
    ensure!(
        !messages.iter().any(|m| m["role"] == "developer"),
        "V4.1 developer messages require engine-side rendering"
    );
    // Dynamo renders V4.1 images, but only the worker expands them into media tokens.
    ensure!(
        !messages.iter().any(|m| m["content"]
            .as_array()
            .is_some_and(|parts| parts.iter().any(|p| p["type"] != "text"))),
        "V4.1 media requires engine-side rendering"
    );
    if let Some(tools) = request["tools"].as_array().filter(|t| !t.is_empty()) {
        if messages[0]["role"] != "system" {
            messages.insert(0, json!({"role":"system", "content":""}));
        }
        // dsv41_tool_payload: only supplied fields, in the OpenAI field order.
        messages[0]["tools"] = tools
            .iter()
            .map(|tool| {
                let f = &tool["function"];
                let mut function: Map<String, Value> = [
                    "name",
                    "description",
                    "parameters",
                    "strict",
                    "defer_loading",
                ]
                .into_iter()
                .filter_map(|key| {
                    f.get(key)
                        .filter(|v| !v.is_null())
                        .map(|v| (key.into(), v.clone()))
                })
                .collect();
                if !function.contains_key("defer_loading") {
                    if let Some(v) = tool.get("defer_loading").filter(|v| !v.is_null()) {
                        function.insert("defer_loading".into(), v.clone());
                    }
                }
                json!({"type":"function", "function":function})
            })
            .collect();
    }
    // Unsupported values use SGLANG_DSV41_REASONING_EFFORT, else `high`.
    let env_effort = std::env::var("SGLANG_DSV41_REASONING_EFFORT")
        .ok()
        .map(|v| {
            v.trim()
                .parse::<i64>()
                .map_or_else(|_| Value::from(v.trim()), Value::from)
        });
    let budget = dsv41_budget(effort)
        .or_else(|| dsv41_budget(env_effort))
        .unwrap_or(75);
    v41::encode_messages(
        &messages,
        if thinking {
            ThinkingMode::Thinking
        } else {
            ThinkingMode::Chat
        },
        true,
        budget,
    )
}

/// `chat_encoding.parse_dsv41_reasoning_effort`, as a token budget.
fn dsv41_budget(effort: Option<Value>) -> Option<u8> {
    match effort {
        Some(Value::String(s)) => match s.as_str() {
            "low" => Some(50),
            "high" => Some(75),
            "xhigh" => Some(75),
            "max" => Some(100),
            _ => None,
        },
        Some(Value::Number(n)) => match (n.as_i64(), n.as_f64()) {
            (Some(i), _) => u8::try_from(i).ok().filter(|i| (1..=100).contains(i)),
            (None, Some(f)) => (0.0..=0.99)
                .contains(&f)
                .then(|| (f * 100.0).round_ties_even().max(1.0) as u8),
            _ => None,
        },
        _ => None,
    }
}
