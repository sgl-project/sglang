// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Messages request → chat request.

use serde_json::{json, Map, Value};

#[derive(Debug, Clone)]
pub struct EchoContext {
    pub model: String,
    pub stop_sequences: Vec<String>,
}

#[derive(Debug)]
pub struct Converted {
    pub chat: Value,
    pub echo: EchoContext,
    pub stream: bool,
}

/// Forwarded as-is. Anything else could override the conversion, as
/// `max_completion_tokens` would take precedence over `max_tokens`.
const FORWARDED: &[&str] = &[
    "temperature",
    "top_p",
    "top_k",
    "chat_template_kwargs",
    "continue_final_message",
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
    if let Some(s) = system_text(req.get("system"))? {
        messages.push(json!({"role": "system", "content": s}));
    }
    let Some(Value::Array(turns)) = req.get("messages") else {
        return Err("messages: field required and must be an array".into());
    };
    if turns.is_empty() {
        return Err("messages: at least one message is required".into());
    }
    for turn in merge_turns(turns)? {
        match turn.role {
            "user" => convert_user(&turn.blocks, turn.at, &mut messages)?,
            _ => convert_assistant(&turn.blocks, turn.at, &mut messages)?,
        }
    }
    // A final assistant turn is a prefill the reply continues.
    let prefill = messages.last().is_some_and(|m| m["role"] == "assistant");

    let mut chat = Map::new();
    for (k, v) in &req {
        if FORWARDED.contains(&k.as_str()) {
            chat.insert(k.clone(), v.clone());
        }
    }
    chat.insert("model".into(), Value::String(model.clone()));
    chat.insert("messages".into(), Value::Array(messages));
    if prefill && !req.contains_key("continue_final_message") {
        chat.insert("continue_final_message".into(), Value::Bool(true));
    }
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

    if let Some(thinking) = req.get("thinking").filter(|v| !v.is_null()) {
        let enabled = thinking_enabled(thinking)?;
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
        // The engine's own Messages adapter folds `xhigh` into `max`.
        match effort.as_str() {
            Some("xhigh") => {
                chat.insert("reasoning_effort".into(), "max".into());
            }
            Some(e @ ("minimal" | "low" | "medium" | "high" | "max")) => {
                chat.insert("reasoning_effort".into(), e.into());
            }
            _ => {
                return Err(format!(
                    "output_config.effort: expected minimal, low, medium, high, xhigh or max, \
                     got {effort}"
                ))
            }
        }
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
    if let Some(tc) = req.get("tool_choice").filter(|v| !v.is_null()) {
        let (choice, parallel) = convert_tool_choice(tc, &tools)?;
        if let Some(c) = choice {
            chat.insert("tool_choice".into(), c);
        }
        if parallel == Some(false) {
            chat.insert("parallel_tool_calls".into(), Value::Bool(false));
        }
    }
    if !tools.is_empty() {
        chat.insert("tools".into(), Value::Array(tools));
    }

    Ok(Converted {
        chat: Value::Object(chat),
        echo: EchoContext {
            model,
            stop_sequences,
        },
        stream,
    })
}

fn thinking_enabled(thinking: &Value) -> Result<bool, String> {
    let field = |k: &str| thinking.get(k).filter(|v| !v.is_null());
    let budget = field("budget_tokens");
    let display = field("display");
    if let Some(d) = display {
        if !matches!(d.as_str(), Some("summarized" | "omitted")) {
            return Err(format!(
                "thinking.display: expected summarized or omitted, got {d}"
            ));
        }
    }
    let forbidden = |k: &str, typ: &str| -> Result<bool, String> {
        Err(format!(
            "thinking.{k} is not allowed when thinking.type is '{typ}'"
        ))
    };
    match thinking.get("type").and_then(Value::as_str) {
        Some("enabled") => match budget.map(Value::as_i64) {
            None => {
                Err("thinking.budget_tokens is required when thinking.type is 'enabled'".into())
            }
            Some(Some(n)) if n >= 1024 => Ok(true),
            Some(Some(n)) => Err(format!("thinking.budget_tokens must be >= 1024 (got {n})")),
            Some(None) => Err("thinking.budget_tokens: must be an integer".into()),
        },
        Some("adaptive") if budget.is_some() => forbidden("budget_tokens", "adaptive"),
        Some("adaptive") => Ok(true),
        Some("disabled") if budget.is_some() => forbidden("budget_tokens", "disabled"),
        Some("disabled") if display.is_some() => forbidden("display", "disabled"),
        Some("disabled") => Ok(false),
        other => Err(format!(
            "thinking.type: expected enabled, disabled or adaptive, got {}",
            other.unwrap_or("<missing>")
        )),
    }
}

fn system_text(system: Option<&Value>) -> Result<Option<String>, String> {
    let text = match system {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(blocks)) => {
            let mut parts = Vec::new();
            for b in blocks {
                match b.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        if let Some(t) = b.get("text").and_then(Value::as_str) {
                            if !t.is_empty() {
                                parts.push(t);
                            }
                        }
                    }
                    other => {
                        return Err(format!(
                            "system: block type `{}` is not supported; only text",
                            other.unwrap_or("<missing>")
                        ))
                    }
                }
            }
            parts.join("\n")
        }
        Some(_) => return Err("system: must be a string or an array of text blocks".into()),
    };
    Ok((!text.trim().is_empty()).then_some(text))
}

/// One turn after consecutive same-role turns are merged, as the Messages
/// API does; `at` is the first source index, for error paths.
struct Turn<'a> {
    role: &'a str,
    at: usize,
    blocks: Vec<Value>,
}

fn merge_turns(turns: &[Value]) -> Result<Vec<Turn<'_>>, String> {
    let mut out: Vec<Turn> = Vec::new();
    for (i, turn) in turns.iter().enumerate() {
        let role = match turn.get("role").and_then(Value::as_str) {
            Some(r @ ("user" | "assistant")) => r,
            Some(other) => {
                return Err(format!(
                    "messages.{i}.role: expected user or assistant, got `{other}`"
                ))
            }
            None => return Err(format!("messages.{i}.role: field required")),
        };
        let blocks = match turn.get("content") {
            Some(Value::String(s)) => vec![json!({"type": "text", "text": s})],
            Some(Value::Array(b)) if !b.is_empty() => b.clone(),
            Some(Value::Array(_)) => {
                return Err(format!("messages.{i}.content: must not be empty"))
            }
            None | Some(Value::Null) => {
                return Err(format!("messages.{i}.content: field required"))
            }
            Some(_) => {
                return Err(format!(
                    "messages.{i}.content: must be a string or an array of content blocks"
                ))
            }
        };
        match out.last_mut() {
            Some(last) if last.role == role => last.blocks.extend(blocks),
            _ => out.push(Turn {
                role,
                at: i,
                blocks,
            }),
        }
    }
    Ok(out)
}

fn convert_user(blocks: &[Value], i: usize, messages: &mut Vec<Value>) -> Result<(), String> {
    let mut parts: Vec<Value> = Vec::new();
    // Keeps order: user(pre) → tool → user(post).
    let flush = |parts: &mut Vec<Value>, messages: &mut Vec<Value>| {
        if !parts.is_empty() {
            messages.push(json!({"role": "user", "content": collapse(std::mem::take(parts))}));
        }
    };
    for (j, block) in blocks.iter().enumerate() {
        let typ = block.get("type").and_then(Value::as_str).unwrap_or("");
        match typ {
            "tool_result" => {
                flush(&mut parts, messages);
                let id = block
                    .get("tool_use_id")
                    .and_then(Value::as_str)
                    .or_else(|| block.get("id").and_then(Value::as_str))
                    .ok_or_else(|| {
                        format!("messages.{i}.content.{j}.tool_use_id: field required")
                    })?;
                for content in
                    tool_result_contents(block.get("content"), &format!("{i}.content.{j}.content"))?
                {
                    messages.push(json!({"role": "tool", "tool_call_id": id, "content": content}));
                }
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

/// One tool message per run of `tool_reference` parts and per run of other
/// parts: chat templates expand references only at the start of a tool message.
fn tool_result_contents(content: Option<&Value>, at: &str) -> Result<Vec<Value>, String> {
    let blocks = match content {
        None | Some(Value::Null) => return Ok(vec![Value::String(String::new())]),
        Some(Value::String(s)) => return Ok(vec![Value::String(s.clone())]),
        Some(Value::Array(blocks)) => blocks,
        Some(other) => return Ok(vec![Value::String(other.to_string())]),
    };
    let mut groups: Vec<Vec<Value>> = Vec::new();
    for (k, b) in blocks.iter().enumerate() {
        let part = if b.get("type").and_then(Value::as_str) == Some("tool_reference") {
            let name = b
                .get("tool_name")
                .and_then(Value::as_str)
                .or_else(|| b.get("name").and_then(Value::as_str))
                .ok_or_else(|| format!("messages.{at}.{k}.tool_name: field required"))?;
            // The engine's chat templates match on `name`.
            json!({"type": "tool_reference", "name": name})
        } else {
            match convert_block(b, &format!("{at}.{k}"))? {
                Some(p) => p,
                None => continue,
            }
        };
        let is_reference = part["type"] == "tool_reference";
        match groups.last_mut() {
            Some(g) if (g[0]["type"] == "tool_reference") == is_reference => g.push(part),
            _ => groups.push(vec![part]),
        }
    }
    if groups.is_empty() {
        return Ok(vec![Value::String(String::new())]);
    }
    Ok(groups.into_iter().map(collapse).collect())
}

fn convert_assistant(blocks: &[Value], i: usize, messages: &mut Vec<Value>) -> Result<(), String> {
    let mut texts: Vec<&str> = Vec::new();
    let mut reasoning: Vec<&str> = Vec::new();
    let mut tool_calls = Vec::new();
    for (j, block) in blocks.iter().enumerate() {
        match block.get("type").and_then(Value::as_str).unwrap_or("") {
            "text" => texts.push(block.get("text").and_then(Value::as_str).unwrap_or("")),
            "thinking" => {
                if let Some(t) = block.get("thinking").and_then(Value::as_str) {
                    if !t.is_empty() {
                        reasoning.push(t);
                    }
                }
            }
            "redacted_thinking" => {
                return Err(format!(
                    "messages.{i}.content.{j}: redacted_thinking history is not supported"
                ))
            }
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
    // Chat content is one string; tool calls follow it.
    let text = texts.join("\n");
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
        let mut schema = match tool.get("input_schema") {
            Some(Value::Object(o)) => o.clone(),
            None | Some(Value::Null) => {
                return Err(format!("tools.{i}.input_schema: field required"))
            }
            Some(_) => return Err(format!("tools.{i}.input_schema: must be an object")),
        };
        schema.entry("type").or_insert_with(|| "object".into());
        let mut function = Map::new();
        function.insert("name".into(), name.into());
        if let Some(d) = tool.get("description").filter(|v| !v.is_null()) {
            function.insert("description".into(), d.clone());
        }
        function.insert("parameters".into(), Value::Object(schema));
        if let Some(s) = tool.get("strict").filter(|v| !v.is_null()) {
            function.insert("strict".into(), s.clone());
        }
        let mut converted = json!({"type": "function", "function": function});
        // Deferred tools stay listed; the template renders them once referenced.
        if let Some(d) = tool.get("defer_loading").filter(|v| !v.is_null()) {
            converted["defer_loading"] = d.clone();
        }
        out.push(converted);
    }
    Ok(out)
}

/// Returns (`tool_choice`, `parallel_tool_calls` override).
fn convert_tool_choice(
    tc: &Value,
    tools: &[Value],
) -> Result<(Option<Value>, Option<bool>), String> {
    let has_tools = !tools.is_empty();
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
            if !tools
                .iter()
                .any(|t| t.pointer("/function/name").and_then(Value::as_str) == Some(name))
            {
                return Err(format!(
                    "tool_choice references tool `{name}`, which is not a custom tool in `tools`"
                ));
            }
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
            "max_completion_tokens": 100000,
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
        assert_eq!(c["max_tokens"], 64);
        for gone in [
            "max_completion_tokens",
            "foo_bar",
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
    fn non_schema_formats_pass_through() {
        let c = chat(
            json!({"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "x"}],
                   "output_config": {"format": {"type": "json_object"}}}),
        );
        assert_eq!(c["response_format"], json!({"type": "json_object"}));
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
                json!({"model": "m", "max_tokens": 8,
                    "tools": [{"name": "f", "input_schema": {}},
                              {"type": "web_search_20250305", "name": "web_search"}],
                    "tool_choice": {"type": "tool", "name": "web_search"},
                    "messages": [{"role": "user", "content": "x"}]}),
                "`web_search`",
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

    fn req(messages: Value) -> Value {
        json!({"model": "m", "max_tokens": 8, "messages": messages})
    }

    #[test]
    fn consecutive_same_role_turns_merge_in_order() {
        let c = chat(req(json!([
            {"role": "user", "content": "a"},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "r1"},
                {"type": "text", "text": "b"}]},
            {"role": "assistant", "content": "x"},
            {"role": "assistant", "content": [{"type": "text", "text": "y"}]},
            {"role": "user", "content": "c"},
        ])));
        let roles: Vec<&str> = c["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["role"].as_str().unwrap())
            .collect();
        assert_eq!(roles, ["user", "tool", "user", "assistant", "user"]);
        assert_eq!(c["messages"][3]["content"], "x\ny");
    }

    #[test]
    fn final_assistant_turn_is_a_prefill() {
        let c = chat(req(json!([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "{\"answer\":"},
        ])));
        assert_eq!(c["continue_final_message"], true);
        assert!(chat(req(json!([{"role": "user", "content": "q"}])))
            .get("continue_final_message")
            .is_none());
        let mut r =
            req(json!([{"role": "user", "content": "q"}, {"role": "assistant", "content": "p"}]));
        r["continue_final_message"] = json!(false);
        assert_eq!(chat(r)["continue_final_message"], false);
    }

    #[test]
    fn tool_results() {
        let c = chat(req(json!([{"role": "user", "content": [{
            "type": "tool_result", "id": "t", "is_error": true,
            "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]}])));
        assert_eq!(
            c["messages"][0],
            json!({"role": "tool", "tool_call_id": "t", "content": [
                {"type": "text", "text": "a"}, {"type": "text", "text": "b"}]})
        );

        let c = chat(req(json!([{"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": "t", "content": [
                {"type": "text", "text": "found"},
                {"type": "tool_reference", "tool_name": "f"},
                {"type": "tool_reference", "tool_name": "g"},
                {"type": "text", "text": "done"}]}]}])));
        let contents: Vec<&Value> = c["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| {
                assert_eq!(m["tool_call_id"], "t");
                &m["content"]
            })
            .collect();
        assert_eq!(
            contents,
            [
                &json!("found"),
                &json!([{"type": "tool_reference", "name": "f"},
                        {"type": "tool_reference", "name": "g"}]),
                &json!("done"),
            ]
        );
    }

    #[test]
    fn tool_definitions() {
        let c = chat(json!({"model": "m", "max_tokens": 8,
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"name": "f", "input_schema": {"properties": {}}, "defer_loading": true}],
            "tool_choice": {"type": "tool", "name": "f"}}));
        assert_eq!(
            c["tools"][0],
            json!({"type": "function", "defer_loading": true, "function": {
                "name": "f", "parameters": {"type": "object", "properties": {}}}})
        );
        assert_eq!(c["tool_choice"]["function"]["name"], "f");
    }

    #[test]
    fn thinking_shapes() {
        let with = |thinking: Value| {
            let mut r = req(json!([{"role": "user", "content": "q"}]));
            r["thinking"] = thinking;
            to_chat(r, false).map(|c| c.chat["chat_template_kwargs"]["enable_thinking"].clone())
        };
        assert_eq!(
            with(json!({"type": "adaptive", "display": "omitted"})),
            Ok(json!(true))
        );
        assert_eq!(with(json!({"type": "disabled"})), Ok(json!(false)));
        for (thinking, needle) in [
            (json!({"type": "enabled"}), "required"),
            (json!({"type": "enabled", "budget_tokens": 1023}), ">= 1024"),
            (
                json!({"type": "disabled", "budget_tokens": 2048}),
                "not allowed",
            ),
            (
                json!({"type": "disabled", "display": "summarized"}),
                "not allowed",
            ),
            (
                json!({"type": "adaptive", "budget_tokens": 2048}),
                "not allowed",
            ),
            (json!({"type": "adaptive", "display": "full"}), "display"),
        ] {
            let err = with(thinking.clone()).unwrap_err();
            assert!(err.contains(needle), "{thinking}: {err}");
        }
    }

    #[test]
    fn rejects_inputs_outside_the_spec() {
        for (messages, needle) in [
            (
                json!([{"role": "system", "content": "s"}]),
                "expected user or assistant",
            ),
            (
                json!([{"role": "user", "content": []}]),
                "must not be empty",
            ),
            (
                json!([{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "u"}}]}]),
                "image_url",
            ),
            (
                json!([{"role": "user", "content": [{"type": "tool_result", "content": "r"}]}]),
                "tool_use_id",
            ),
            (
                json!([{"role": "user", "content": "q"}, {"role": "assistant", "content": [
                    {"type": "redacted_thinking", "data": "opaque"}]}]),
                "redacted_thinking",
            ),
        ] {
            let err = to_chat(req(messages.clone()), false).unwrap_err();
            assert!(err.contains(needle), "{messages}: {err}");
        }
        // Clients also send `stop_sequences` as a bare string.
        let mut r = req(json!([{"role": "user", "content": "q"}]));
        r["stop_sequences"] = json!("END");
        assert_eq!(chat(r)["stop"], json!(["END"]));
        let mut r = req(json!([{"role": "user", "content": "q"}]));
        r["output_config"] = json!({"effort": "minimal"});
        assert_eq!(chat(r)["reasoning_effort"], "minimal");
        let mut r = req(json!([{"role": "user", "content": "q"}]));
        r["system"] = json!(" \n ");
        assert_eq!(
            chat(r)["messages"],
            json!([{"role": "user", "content": "q"}])
        );
    }
}
