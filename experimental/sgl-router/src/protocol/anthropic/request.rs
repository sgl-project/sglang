// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Messages request → chat request. Mirrors the engine's adapter
//! (`entrypoints/anthropic/serving.py`), but thinking history goes to
//! `reasoning_content`. Unconsumed fields pass through.

use serde_json::{json, Map, Value};

#[derive(Debug, Clone, Default)]
pub struct EchoContext {
    pub model: String,
    pub stop_sequences: Vec<String>,
    /// The request set `thinking.type` to `enabled` or `adaptive`.
    pub thinking_requested: bool,
    /// Drop `thinking` blocks from the reply (set by the route from the profile).
    pub hide_thinking: bool,
}

#[derive(Debug)]
pub struct Converted {
    pub chat: Value,
    pub echo: EchoContext,
    pub stream: bool,
}

/// Fields not forwarded as-is.
const CONSUMED: &[&str] = &[
    "model",
    "messages",
    "system",
    "max_tokens",
    "stop_sequences",
    "stream",
    "thinking",
    "output_config",
    "output_format",
    "tools",
    "tool_choice",
    "metadata",
    "betas",
    "container",
    "mcp_servers",
    "service_tier",
    "context_management",
    "inference_geo",
    // Chat fields that would break the conversion.
    "stream_options",
    "n",
];

/// `count_only`: `count_tokens` shape, no `max_tokens` required.
pub fn to_chat(req: Value, count_only: bool) -> Result<Converted, String> {
    let Value::Object(req) = req else {
        return Err("request body must be a JSON object".into());
    };
    let model = match req.get("model") {
        Some(Value::String(m)) if !m.is_empty() => m.clone(),
        Some(Value::String(_)) | None | Some(Value::Null) => {
            return Err("model: field required".into())
        }
        Some(_) => return Err("model: must be a string".into()),
    };
    let stream = match req.get("stream") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(other) => return Err(format!("stream: must be a boolean, got {other}")),
    };
    let max_tokens = match req.get("max_tokens") {
        None | Some(Value::Null) if count_only => None,
        None | Some(Value::Null) => return Err("max_tokens: field required".into()),
        Some(v) if v.as_u64().is_some_and(|n| n >= 1) => Some(v.clone()),
        Some(v) => return Err(format!("max_tokens: must be a positive integer, got {v}")),
    };

    let mut messages = Vec::new();
    let mut system = system_text(req.get("system"))?;
    let Some(Value::Array(turns)) = req.get("messages") else {
        return Err("messages: field required and must be an array".into());
    };
    if turns.is_empty() {
        return Err("messages: at least one message is required".into());
    }
    for (i, turn) in turns.iter().enumerate() {
        convert_turn(turn, i, &mut messages, &mut system)?;
    }
    if let Some(s) = system {
        messages.insert(0, json!({"role": "system", "content": s}));
    }

    let mut chat = Map::new();
    for (k, v) in &req {
        if !CONSUMED.contains(&k.as_str()) {
            chat.insert(k.clone(), v.clone());
        }
    }
    chat.insert("model".into(), Value::String(model.clone()));
    chat.insert("messages".into(), Value::Array(messages));
    chat.insert("stream".into(), Value::Bool(stream));
    if stream {
        // Usage on every chunk so `message_start` has `input_tokens`.
        chat.insert(
            "stream_options".into(),
            json!({"include_usage": true, "continuous_usage_stats": true}),
        );
    }
    if let Some(mt) = max_tokens {
        chat.insert("max_tokens".into(), mt);
    }

    let stop_sequences = match req.get("stop_sequences") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| "stop_sequences: every entry must be a string".to_string())
            })
            .collect::<Result<_, _>>()?,
        Some(_) => return Err("stop_sequences: must be a string or an array of strings".into()),
    };
    if !stop_sequences.is_empty() {
        chat.insert("stop".into(), json!(stop_sequences));
    }

    let mut thinking_requested = false;
    if let Some(thinking) = req.get("thinking").filter(|v| !v.is_null()) {
        let enabled = match thinking.get("type").and_then(Value::as_str) {
            Some("enabled") | Some("adaptive") => true,
            Some("disabled") => false,
            other => {
                return Err(format!(
                    "thinking.type: expected enabled, disabled or adaptive, got {}",
                    other.unwrap_or("<missing>")
                ))
            }
        };
        thinking_requested = enabled;
        // Templates read different keys; explicit client values win.
        let ctk = chat
            .entry("chat_template_kwargs")
            .or_insert_with(|| json!({}));
        if let Value::Object(ctk) = ctk {
            ctk.entry("thinking").or_insert(Value::Bool(enabled));
            ctk.entry("enable_thinking").or_insert(Value::Bool(enabled));
        }
    }

    let output_config = req.get("output_config").filter(|v| !v.is_null());
    if let Some(effort) = output_config
        .and_then(|c| c.get("effort"))
        .filter(|v| !v.is_null())
    {
        let mapped = match effort.as_str() {
            Some("low") => "low",
            Some("medium") => "medium",
            Some("high") => "high",
            Some("xhigh") | Some("max") => "max",
            _ => {
                return Err(format!(
                    "output_config.effort: expected low, medium, high, xhigh or max, got {effort}"
                ))
            }
        };
        chat.insert("reasoning_effort".into(), mapped.into());
    }
    let format = output_config
        .and_then(|c| c.get("format"))
        .or_else(|| req.get("output_format"))
        .filter(|v| !v.is_null());
    if let Some(format) = format {
        chat.insert("response_format".into(), convert_format(format)?);
    }

    let tools = match req.get("tools").filter(|v| !v.is_null()) {
        Some(t) => convert_tools(t)?,
        None => Vec::new(),
    };
    let has_tools = !tools.is_empty();
    if has_tools {
        chat.insert("tools".into(), Value::Array(tools));
    }
    if let Some(tc) = req.get("tool_choice").filter(|v| !v.is_null()) {
        let (choice, parallel) = convert_tool_choice(tc, has_tools)?;
        if let Some(c) = choice {
            chat.insert("tool_choice".into(), c);
        }
        if parallel == Some(false) {
            chat.insert("parallel_tool_calls".into(), Value::Bool(false));
        }
    }

    Ok(Converted {
        chat: Value::Object(chat),
        echo: EchoContext {
            model,
            stop_sequences,
            thinking_requested,
            hide_thinking: false,
        },
        stream,
    })
}

fn system_text(system: Option<&Value>) -> Result<Option<String>, String> {
    match system {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(Value::Array(blocks)) => {
            let mut parts = Vec::new();
            for b in blocks {
                match b.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        parts.push(b.get("text").and_then(Value::as_str).unwrap_or(""));
                    }
                    other => {
                        return Err(format!(
                            "system: block type `{}` is not supported; only text",
                            other.unwrap_or("<missing>")
                        ))
                    }
                }
            }
            Ok(Some(parts.join("\n")))
        }
        Some(_) => Err("system: must be a string or an array of text blocks".into()),
    }
}

fn convert_turn(
    turn: &Value,
    i: usize,
    messages: &mut Vec<Value>,
    system: &mut Option<String>,
) -> Result<(), String> {
    let role = turn
        .get("role")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("messages.{i}.role: field required"))?;
    let content = turn
        .get("content")
        .filter(|v| !v.is_null())
        .ok_or_else(|| format!("messages.{i}.content: field required"))?;
    match role {
        "user" => convert_user(content, i, messages),
        "assistant" => convert_assistant(content, i, messages),
        // Fold into the top-level system prompt.
        "system" => {
            let text = match content {
                Value::String(s) => s.clone(),
                other => system_text(Some(other))?.unwrap_or_default(),
            };
            match system {
                Some(s) => {
                    s.push('\n');
                    s.push_str(&text);
                }
                None => *system = Some(text),
            }
            Ok(())
        }
        other => Err(format!(
            "messages.{i}.role: expected user or assistant, got `{other}`"
        )),
    }
}

fn blocks(content: &Value, i: usize) -> Result<&[Value], String> {
    match content {
        Value::Array(b) => Ok(b),
        _ => Err(format!(
            "messages.{i}.content: must be a string or an array of content blocks"
        )),
    }
}

fn convert_user(content: &Value, i: usize, messages: &mut Vec<Value>) -> Result<(), String> {
    if let Value::String(s) = content {
        messages.push(json!({"role": "user", "content": s}));
        return Ok(());
    }
    let mut parts: Vec<Value> = Vec::new();
    // Keeps order: user(pre) → tool → user(post).
    let flush = |parts: &mut Vec<Value>, messages: &mut Vec<Value>| {
        if !parts.is_empty() {
            messages.push(json!({"role": "user", "content": collapse(std::mem::take(parts))}));
        }
    };
    for (j, block) in blocks(content, i)?.iter().enumerate() {
        let typ = block.get("type").and_then(Value::as_str).unwrap_or("");
        match typ {
            "tool_result" => {
                flush(&mut parts, messages);
                let id = block
                    .get("tool_use_id")
                    .or_else(|| block.get("id"))
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        format!("messages.{i}.content.{j}.tool_use_id: field required")
                    })?;
                let tool_content = match block.get("content") {
                    None | Some(Value::Null) => Value::String(String::new()),
                    Some(Value::String(s)) => Value::String(s.clone()),
                    Some(Value::Array(inner)) => {
                        let mut out = Vec::new();
                        for (k, b) in inner.iter().enumerate() {
                            if let Some(p) =
                                convert_block(b, &format!("{i}.content.{j}.content.{k}"))?
                            {
                                out.push(p);
                            }
                        }
                        collapse(out)
                    }
                    Some(other) => Value::String(other.to_string()),
                };
                let tool_content = match block.get("is_error").and_then(Value::as_bool) {
                    Some(true) => match tool_content {
                        Value::String(s) => Value::String(format!("Error: {s}")),
                        other => other,
                    },
                    _ => tool_content,
                };
                messages.push(json!({"role": "tool", "tool_call_id": id, "content": tool_content}));
            }
            _ => {
                if let Some(p) = convert_block(block, &format!("{i}.content.{j}"))? {
                    parts.push(p);
                }
            }
        }
    }
    flush(&mut parts, messages);
    Ok(())
}

fn convert_assistant(content: &Value, i: usize, messages: &mut Vec<Value>) -> Result<(), String> {
    if let Value::String(s) = content {
        messages.push(json!({"role": "assistant", "content": s}));
        return Ok(());
    }
    let mut text = String::new();
    let mut reasoning: Vec<&str> = Vec::new();
    let mut tool_calls = Vec::new();
    for (j, block) in blocks(content, i)?.iter().enumerate() {
        match block.get("type").and_then(Value::as_str).unwrap_or("") {
            "text" => text.push_str(block.get("text").and_then(Value::as_str).unwrap_or("")),
            "thinking" => {
                if let Some(t) = block.get("thinking").and_then(Value::as_str) {
                    if !t.is_empty() {
                        reasoning.push(t);
                    }
                }
            }
            "redacted_thinking" => {}
            "tool_use" => {
                let id = block
                    .get("id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!("messages.{i}.content.{j}.id: field required"))?;
                let name = block
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!("messages.{i}.content.{j}.name: field required"))?;
                let input = block.get("input").cloned().unwrap_or_else(|| json!({}));
                tool_calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": {"name": name, "arguments": input.to_string()},
                }));
            }
            other => {
                return Err(format!(
                    "messages.{i}.content.{j}.type: `{other}` is not supported in an assistant turn"
                ))
            }
        }
    }
    let mut m = Map::new();
    m.insert("role".into(), "assistant".into());
    m.insert(
        "content".into(),
        if text.is_empty() && !tool_calls.is_empty() {
            Value::Null
        } else {
            Value::String(text)
        },
    );
    if !reasoning.is_empty() {
        m.insert("reasoning_content".into(), reasoning.join("\n").into());
    }
    if !tool_calls.is_empty() {
        m.insert("tool_calls".into(), Value::Array(tool_calls));
    }
    messages.push(Value::Object(m));
    Ok(())
}

fn convert_block(block: &Value, at: &str) -> Result<Option<Value>, String> {
    let typ = block.get("type").and_then(Value::as_str).unwrap_or("");
    Ok(Some(match typ {
        "text" => {
            json!({"type": "text", "text": block.get("text").and_then(Value::as_str).unwrap_or("")})
        }
        "image" => json!({"type": "image_url", "image_url": {"url": source_url(block, at)?}}),
        "search_result" => json!({"type": "text", "text": search_result_text(block)}),
        "image_url" => block.clone(),
        "thinking" | "redacted_thinking" => return Ok(None),
        "" => return Err(format!("messages.{at}.type: field required")),
        other => {
            return Err(format!(
                "messages.{at}.type: content block type `{other}` is not supported"
            ))
        }
    }))
}

fn source_url(block: &Value, at: &str) -> Result<String, String> {
    let source = block
        .get("source")
        .ok_or_else(|| format!("messages.{at}.source: field required"))?;
    match source.get("type").and_then(Value::as_str) {
        Some("base64") => {
            let media = source
                .get("media_type")
                .and_then(Value::as_str)
                .unwrap_or("image/png");
            let data = source
                .get("data")
                .and_then(Value::as_str)
                .filter(|d| !d.is_empty())
                .ok_or_else(|| format!("messages.{at}.source.data: field required"))?;
            Ok(format!("data:{media};base64,{data}"))
        }
        Some("url") => source
            .get("url")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| format!("messages.{at}.source.url: field required")),
        other => Err(format!(
            "messages.{at}.source.type: expected base64 or url, got {}",
            other.unwrap_or("<missing>")
        )),
    }
}

fn search_result_text(block: &Value) -> String {
    let mut parts = Vec::new();
    if let Some(t) = block.get("title").and_then(Value::as_str) {
        parts.push(format!("Title: {t}"));
    }
    match block.get("source") {
        Some(Value::String(s)) => parts.push(format!("Source: {s}")),
        Some(Value::Object(o)) => {
            if let Some(s) = o
                .get("url")
                .or_else(|| o.get("text"))
                .and_then(Value::as_str)
            {
                parts.push(format!("Source: {s}"));
            }
        }
        _ => {}
    }
    let texts: Vec<&str> = block
        .get("content")
        .and_then(Value::as_array)
        .map(|c| {
            c.iter()
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect()
        })
        .unwrap_or_default();
    if !texts.is_empty() {
        parts.push(format!("Content: {}", texts.join("\n")));
    }
    parts.join("\n")
}

fn collapse(parts: Vec<Value>) -> Value {
    if let [single] = parts.as_slice() {
        if single["type"] == "text" {
            return single["text"].clone();
        }
    }
    Value::Array(parts)
}

/// Unknown format types go to the engine to reject.
fn convert_format(format: &Value) -> Result<Value, String> {
    let Value::Object(f) = format else {
        return Err("output_config.format: must be an object".into());
    };
    match f.get("type").and_then(Value::as_str) {
        Some("json_schema") => Ok(json!({
            "type": "json_schema",
            "json_schema": {
                "name": f.get("name").cloned().unwrap_or_else(|| "output".into()),
                "schema": f.get("schema").cloned().unwrap_or(Value::Null),
                "strict": true,
            },
        })),
        Some(_) => Ok(format.clone()),
        None => Err("output_config.format.type: field required".into()),
    }
}

/// Server tools (`web_search_*`, …) are skipped, as the engine's adapter does.
fn convert_tools(tools: &Value) -> Result<Vec<Value>, String> {
    let Value::Array(tools) = tools else {
        return Err("tools: must be an array".into());
    };
    let mut out = Vec::with_capacity(tools.len());
    for (i, tool) in tools.iter().enumerate() {
        match tool.get("type").and_then(Value::as_str) {
            None | Some("custom") => {}
            Some(other) => {
                tracing::info!(
                    tool_type = other,
                    "messages: skipping Anthropic server tool"
                );
                continue;
            }
        }
        let name = tool
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("tools.{i}.name: field required"))?;
        let schema = tool
            .get("input_schema")
            .filter(|v| !v.is_null())
            .ok_or_else(|| format!("tools.{i}.input_schema: field required"))?;
        let mut function = Map::new();
        function.insert("name".into(), name.into());
        if let Some(d) = tool.get("description").filter(|v| !v.is_null()) {
            function.insert("description".into(), d.clone());
        }
        function.insert("parameters".into(), schema.clone());
        if let Some(s) = tool.get("strict").filter(|v| !v.is_null()) {
            function.insert("strict".into(), s.clone());
        }
        out.push(json!({"type": "function", "function": function}));
    }
    Ok(out)
}

/// Returns (`tool_choice`, `parallel_tool_calls` override).
fn convert_tool_choice(
    tc: &Value,
    has_tools: bool,
) -> Result<(Option<Value>, Option<bool>), String> {
    let typ = tc
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| "tool_choice.type: field required".to_string())?;
    let parallel = tc
        .get("disable_parallel_tool_use")
        .and_then(Value::as_bool)
        .map(|d| !d);
    let choice = match typ {
        "none" => Some(json!("none")),
        "auto" => has_tools.then(|| json!("auto")),
        "any" | "tool" if !has_tools => {
            return Err(format!(
                "tool_choice.type `{typ}` requires at least one custom tool in `tools`"
            ))
        }
        "any" => Some(json!("required")),
        "tool" => {
            let name = tc
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| "tool_choice.name: field required".to_string())?;
            Some(json!({"type": "function", "function": {"name": name}}))
        }
        other => {
            return Err(format!(
                "tool_choice.type: expected auto, any, tool or none, got `{other}`"
            ))
        }
    };
    Ok((choice, parallel))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chat(req: Value) -> Value {
        to_chat(req, false).expect("converts").chat
    }

    #[test]
    fn system_forms_and_content_forms() {
        for system in [
            json!("be brief"),
            json!([{"type": "text", "text": "be brief"}]),
        ] {
            let c = chat(json!({"model": "m", "max_tokens": 8, "system": system,
                                "messages": [{"role": "user", "content": "hi"}]}));
            assert_eq!(
                c["messages"][0],
                json!({"role": "system", "content": "be brief"})
            );
        }
        let c = chat(json!({"model": "m", "max_tokens": 8, "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]}]}));
        assert_eq!(c["messages"], json!([{"role": "user", "content": "hi"}]));
        assert_eq!(c["max_tokens"], 8);
    }

    #[test]
    fn images() {
        let c = chat(
            json!({"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAA"}},
                {"type": "image", "source": {"type": "url", "url": "https://x/a.png"}},
                    {"type": "text", "text": "describe"},
            ]}]}),
        );
        assert_eq!(
            c["messages"][0]["content"],
            json!([
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAA"}},
                {"type": "image_url", "image_url": {"url": "https://x/a.png"}},
                {"type": "text", "text": "describe"},
            ])
        );
    }

    #[test]
    fn tool_round_trip_and_thinking_history() {
        let c = chat(json!({"model": "m", "max_tokens": 8, "messages": [
            {"role": "user", "content": "weather in bj?"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "need tool", "signature": ""},
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                 "input": {"city": "bj"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny"},
                {"type": "text", "text": "thanks"},
            ]},
        ]}));
        let m = c["messages"].as_array().unwrap();
        assert_eq!(m.len(), 4);
        assert_eq!(m[1]["content"], "checking");
        assert_eq!(m[1]["reasoning_content"], "need tool");
        assert_eq!(
            m[1]["tool_calls"][0]["function"]["arguments"],
            "{\"city\":\"bj\"}"
        );
        assert_eq!(
            m[2],
            json!({"role": "tool", "tool_call_id": "toolu_1", "content": "sunny"})
        );
        assert_eq!(m[3], json!({"role": "user", "content": "thanks"}));
    }

    #[test]
    fn sampling_thinking_effort_format_tools() {
        let c = chat(json!({
            "model": "m", "max_tokens": 64, "stream": true,
            "messages": [{"role": "user", "content": "x"}],
            "temperature": 0.5, "top_p": 0.9, "top_k": 0,
            "stop_sequences": "<END>",
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "output_config": {"effort": "xhigh",
                              "format": {"type": "json_schema", "schema": {"type": "object"}}},
            "tools": [{"name": "get_weather", "description": "d",
                       "input_schema": {"type": "object"}},
                      {"type": "web_search_20250305", "name": "web_search"}],
            "tool_choice": {"type": "any", "disable_parallel_tool_use": true},
            "metadata": {"user_id": "u"},
            "foo_bar": 1,
        }));
        assert_eq!(
            c["stream_options"],
            json!({"include_usage": true, "continuous_usage_stats": true})
        );
        assert_eq!(c["top_k"], 0);
        assert_eq!(c["temperature"], 0.5);
        assert_eq!(c["stop"], json!(["<END>"]));
        assert_eq!(
            c["chat_template_kwargs"],
            json!({"thinking": true, "enable_thinking": true})
        );
        assert_eq!(c["reasoning_effort"], "max");
        assert_eq!(
            c["response_format"],
            json!({"type": "json_schema",
                   "json_schema": {"name": "output", "schema": {"type": "object"}, "strict": true}})
        );
        assert_eq!(c["tools"].as_array().unwrap().len(), 1);
        assert_eq!(
            c["tools"][0]["function"]["parameters"],
            json!({"type": "object"})
        );
        assert_eq!(c["tool_choice"], "required");
        assert_eq!(c["parallel_tool_calls"], false);
        assert_eq!(c["foo_bar"], 1);
        for gone in [
            "metadata",
            "thinking",
            "output_config",
            "stop_sequences",
            "system",
        ] {
            assert!(c.get(gone).is_none(), "{gone} must not be forwarded");
        }
    }

    #[test]
    fn response_format_passthrough_for_json_object() {
        let c = chat(
            json!({"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "x"}],
                            "response_format": {"type": "json_object"}}),
        );
        assert_eq!(c["response_format"], json!({"type": "json_object"}));
    }

    #[test]
    fn thinking_requested_only_for_enabled_or_adaptive() {
        let req = |t: Value| {
            let mut r = json!({"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "x"}]});
            if !t.is_null() {
                r["thinking"] = t;
            }
            to_chat(r, false).unwrap().echo.thinking_requested
        };
        assert!(!req(Value::Null));
        assert!(!req(json!({"type": "disabled"})));
        assert!(req(json!({"type": "enabled", "budget_tokens": 1024})));
        assert!(req(json!({"type": "adaptive"})));
    }

    #[test]
    fn rejections() {
        for (req, needle) in [
            (
                json!({"max_tokens": 8, "messages": [{"role": "user", "content": "x"}]}),
                "model",
            ),
            (
                json!({"model": "m", "messages": [{"role": "user", "content": "x"}]}),
                "max_tokens",
            ),
            (
                json!({"model": "m", "max_tokens": 8, "messages": []}),
                "at least one",
            ),
            (
                json!({"model": "m", "max_tokens": 8, "messages": [{"content": "x"}]}),
                "role",
            ),
            (
                json!({"model": "m", "max_tokens": 8, "messages": [{"role": "user"}]}),
                "content",
            ),
            (
                json!({"model": "m", "max_tokens": 0, "messages": [{"role": "user", "content": "x"}]}),
                "positive",
            ),
            (
                json!({"model": "m", "max_tokens": 8, "stream": "yes",
                    "messages": [{"role": "user", "content": "x"}]}),
                "boolean",
            ),
            (
                json!({"model": "m", "max_tokens": 8, "tool_choice": {"type": "any"},
                    "messages": [{"role": "user", "content": "x"}]}),
                "requires",
            ),
            (
                json!({"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": [
                    {"type": "document", "source": {}}]}]}),
                "document",
            ),
            (
                json!({"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": [
                    {"type": "video", "source": {"type": "url", "url": "https://x/v.mp4"}}]}]}),
                "video",
            ),
        ] {
            let err = to_chat(req.clone(), false).expect_err(&req.to_string());
            assert!(err.contains(needle), "{req}: {err}");
        }
        assert!(to_chat(
            json!({"model": "m", "messages": [{"role": "user", "content": "x"}]}),
            true
        )
        .is_ok());
    }
}
