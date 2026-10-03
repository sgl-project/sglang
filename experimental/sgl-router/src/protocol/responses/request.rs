// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Responses request → chat request. `Value`-based so unconsumed fields
//! (sglang extensions) pass through.

use std::collections::HashMap;

use serde_json::{json, Map, Value};

/// Request fields echoed on every Response object.
#[derive(Debug, Clone)]
pub struct EchoContext {
    pub model: String,
    pub fields: Map<String, Value>,
    pub tools: ToolMap,
}

/// A namespaced or `custom` tool, keyed in [`ToolMap`] by its chat function name.
#[derive(Debug, Clone)]
pub struct ToolRef {
    pub namespace: Option<String>,
    pub name: String,
    pub custom: bool,
}

pub type ToolMap = HashMap<String, ToolRef>;

/// Chat function name for a tool, e.g. `mcp__fs__` + `read` → `mcp__fs__read`.
fn flat_name(namespace: Option<&str>, name: &str) -> String {
    match namespace {
        None => name.to_owned(),
        Some(ns) if ns.ends_with("__") => format!("{ns}{name}"),
        Some(ns) => format!("{ns}__{name}"),
    }
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
    "input",
    "instructions",
    "tools",
    "tool_choice",
    "text",
    "reasoning",
    "max_output_tokens",
    "stream",
    "metadata",
    "store",
    "truncation",
    "include",
    "service_tier",
    "prompt_cache_key",
    "prompt_cache_retention",
    "safety_identifier",
    "previous_response_id",
    "conversation",
    "background",
    "prompt",
    "max_tool_calls",
    "top_logprobs",
    // Chat fields that would break the conversion.
    "messages",
    "n",
];

pub fn to_chat(req: Value) -> Result<Converted, String> {
    let Value::Object(req) = req else {
        return Err("request body must be a JSON object".into());
    };

    let model = match req.get("model") {
        Some(Value::String(m)) if !m.is_empty() => m.clone(),
        Some(Value::String(_)) | None | Some(Value::Null) => {
            return Err("`model` is required".into());
        }
        Some(_) => return Err("`model` must be a string".into()),
    };

    reject_unsupported(&req)?;

    let stream = match req.get("stream") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(other) => return Err(format!("`stream` must be a boolean, got {other}")),
    };

    let mut messages = Vec::new();
    match req.get("instructions") {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) => messages.push(json!({"role": "system", "content": s})),
        Some(_) => return Err("`instructions` must be a string".into()),
    }
    match req.get("input") {
        None | Some(Value::Null) => return Err("`input` is required".into()),
        Some(Value::String(s)) => messages.push(json!({"role": "user", "content": s})),
        Some(Value::Array(items)) => {
            if items.is_empty() {
                return Err("`input` must not be an empty array".into());
            }
            convert_items(items, &mut messages)?;
        }
        Some(_) => return Err("`input` must be a string or an array of input items".into()),
    }
    merge_leading_system(&mut messages);

    let mut chat = Map::new();
    // Passthrough first so the translations below win.
    for (k, v) in &req {
        if !CONSUMED.contains(&k.as_str()) {
            chat.insert(k.clone(), v.clone());
        }
    }
    chat.insert("model".into(), Value::String(model.clone()));
    chat.insert("messages".into(), Value::Array(messages));
    chat.insert("stream".into(), Value::Bool(stream));
    if stream {
        let mut opts = match chat.remove("stream_options") {
            Some(Value::Object(o)) => o,
            _ => Map::new(),
        };
        opts.insert("include_usage".into(), Value::Bool(true));
        chat.insert("stream_options".into(), Value::Object(opts));
    }

    if let Some(v) = req.get("max_output_tokens").filter(|v| !v.is_null()) {
        chat.insert("max_completion_tokens".into(), v.clone());
    }
    let (tools, tool_map) = match req.get("tools").filter(|v| !v.is_null()) {
        Some(tools) => convert_tools(tools)?,
        None => Default::default(),
    };
    if !tools.is_empty() {
        chat.insert("tools".into(), Value::Array(tools));
    }
    if let Some(tc) = req.get("tool_choice").filter(|v| !v.is_null()) {
        chat.insert("tool_choice".into(), convert_tool_choice(tc)?);
    }
    if let Some(text) = req.get("text").filter(|v| !v.is_null()) {
        if let Some(rf) = convert_text_format(text)? {
            chat.insert("response_format".into(), rf);
        }
    }
    if let Some(reasoning) = req.get("reasoning").filter(|v| !v.is_null()) {
        if let Some(effort) = convert_reasoning_effort(reasoning)? {
            chat.insert("reasoning_effort".into(), Value::String(effort.into()));
        }
    }

    Ok(Converted {
        chat: Value::Object(chat),
        echo: EchoContext {
            model,
            fields: echo_fields(&req),
            tools: tool_map,
        },
        stream,
    })
}

/// Refuses what the router cannot serve: stored state and logprobs.
fn reject_unsupported(req: &Map<String, Value>) -> Result<(), String> {
    let set = |k: &str| req.get(k).is_some_and(|v| !v.is_null());
    if set("previous_response_id") {
        return Err(
            "`previous_response_id` is not supported: this endpoint does not store responses. \
             Send the full conversation in `input` instead."
                .into(),
        );
    }
    if set("conversation") {
        return Err(
            "`conversation` is not supported: this endpoint does not store conversations. \
             Send the full conversation in `input` instead."
                .into(),
        );
    }
    if req.get("background").and_then(Value::as_bool) == Some(true) {
        return Err("`background: true` is not supported".into());
    }
    if set("prompt") {
        return Err("`prompt` (stored prompt templates) is not supported".into());
    }
    if req.get("top_logprobs").and_then(Value::as_u64).unwrap_or(0) > 0 {
        return Err("`top_logprobs` is not supported".into());
    }
    Ok(())
}

/// Assistant turn built from consecutive reasoning / message / function_call items.
#[derive(Default)]
struct PendingAssistant {
    reasoning: Option<String>,
    content: Option<String>,
    tool_calls: Vec<Value>,
}

impl PendingAssistant {
    fn is_empty(&self) -> bool {
        self.reasoning.is_none() && self.content.is_none() && self.tool_calls.is_empty()
    }

    fn flush(&mut self, messages: &mut Vec<Value>) {
        if self.is_empty() {
            return;
        }
        let p = std::mem::take(self);
        let mut m = Map::new();
        m.insert("role".into(), "assistant".into());
        m.insert(
            "content".into(),
            match p.content {
                Some(c) => Value::String(c),
                None if !p.tool_calls.is_empty() => Value::Null,
                None => Value::String(String::new()),
            },
        );
        if let Some(r) = p.reasoning {
            m.insert("reasoning_content".into(), Value::String(r));
        }
        if !p.tool_calls.is_empty() {
            m.insert("tool_calls".into(), Value::Array(p.tool_calls));
        }
        messages.push(Value::Object(m));
    }
}

fn convert_items(items: &[Value], messages: &mut Vec<Value>) -> Result<(), String> {
    let mut pending = PendingAssistant::default();
    for (i, item) in items.iter().enumerate() {
        let Value::Object(obj) = item else {
            return Err(format!("input[{i}] must be an object"));
        };
        let typ = match obj.get("type").and_then(Value::as_str) {
            Some(t) => t,
            None if obj.contains_key("role") => "message",
            None => return Err(format!("input[{i}] is missing `type`")),
        };
        match typ {
            "message" => {
                let role = obj
                    .get("role")
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!("input[{i}] is missing `role`"))?;
                let content = obj
                    .get("content")
                    .filter(|v| !v.is_null())
                    .ok_or_else(|| format!("input[{i}] is missing `content`"))?;
                match role {
                    "assistant" => {
                        if !pending.tool_calls.is_empty() {
                            pending.flush(messages);
                        }
                        let text = content_text(content, i)?;
                        match &mut pending.content {
                            Some(c) => c.push_str(&text),
                            None => pending.content = Some(text),
                        }
                    }
                    "user" | "system" | "developer" => {
                        pending.flush(messages);
                        // Chat templates know `system`; some drop `developer` messages.
                        let role = if role == "developer" { "system" } else { role };
                        messages.push(json!({
                            "role": role,
                            "content": convert_content(content, i)?,
                        }));
                    }
                    other => return Err(format!("input[{i}] has unsupported role `{other}`")),
                }
            }
            "function_call" | "custom_tool_call" => {
                let call_id = obj
                    .get("call_id")
                    .or_else(|| obj.get("id"))
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!("input[{i}] ({typ}) is missing `call_id`"))?;
                let name = obj
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!("input[{i}] ({typ}) is missing `name`"))?;
                let name = flat_name(obj.get("namespace").and_then(Value::as_str), name);
                let arguments = match (typ, obj.get("arguments"), obj.get("input")) {
                    ("custom_tool_call", _, Some(Value::String(s))) => {
                        json!({ "input": s }).to_string()
                    }
                    (_, Some(Value::String(s)), _) => s.clone(),
                    (_, None | Some(Value::Null), _) => "{}".into(),
                    (_, Some(other), _) => other.to_string(),
                };
                pending.tool_calls.push(json!({
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }));
            }
            "function_call_output" | "custom_tool_call_output" => {
                pending.flush(messages);
                let call_id = obj
                    .get("call_id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!("input[{i}] ({typ}) is missing `call_id`"))?;
                let output = match obj.get("output") {
                    Some(Value::String(s)) => s.clone(),
                    Some(v @ Value::Array(_)) => content_text(v, i)?,
                    None | Some(Value::Null) => String::new(),
                    Some(other) => other.to_string(),
                };
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": output,
                }));
            }
            "reasoning" => {
                // Reasoning belongs to the following assistant turn.
                if pending.content.is_some() || !pending.tool_calls.is_empty() {
                    pending.flush(messages);
                }
                let text = reasoning_text(obj);
                if !text.is_empty() {
                    match &mut pending.reasoning {
                        Some(r) => r.push_str(&text),
                        None => pending.reasoning = Some(text),
                    }
                }
            }
            "item_reference" => {
                return Err(format!(
                    "input[{i}]: `item_reference` is not supported: this endpoint does not \
                     store items; send the full item instead"
                ));
            }
            other => return Err(format!("input[{i}] has unsupported type `{other}`")),
        }
    }
    pending.flush(messages);
    Ok(())
}

/// Join leading text-only system messages into one: templates treat only the
/// first message as the system prompt.
fn merge_leading_system(messages: &mut Vec<Value>) {
    let n = messages
        .iter()
        .take_while(|m| m["role"] == "system" && m["content"].is_string())
        .count();
    if n < 2 {
        return;
    }
    let text = messages[..n]
        .iter()
        .filter_map(|m| m["content"].as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    messages.splice(..n, [json!({"role": "system", "content": text})]);
}

/// `content[].text`, else `summary[].text`.
fn reasoning_text(obj: &Map<String, Value>) -> String {
    let join = |key: &str| -> String {
        obj.get(key)
            .and_then(Value::as_array)
            .map(|parts| {
                parts
                    .iter()
                    .filter_map(|p| p.get("text").and_then(Value::as_str))
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default()
    };
    let content = join("content");
    if content.is_empty() {
        join("summary")
    } else {
        content
    }
}

/// Content as plain text, for assistant turns and tool outputs.
fn content_text(content: &Value, i: usize) -> Result<String, String> {
    match content {
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => {
            let mut out = String::new();
            for part in parts {
                let typ = part.get("type").and_then(Value::as_str).unwrap_or("");
                match typ {
                    "input_text" | "output_text" | "text" => {
                        out.push_str(part.get("text").and_then(Value::as_str).unwrap_or(""));
                    }
                    "refusal" => {
                        out.push_str(part.get("refusal").and_then(Value::as_str).unwrap_or(""));
                    }
                    other => {
                        return Err(format!(
                            "input[{i}]: content part type `{other}` is not supported here"
                        ))
                    }
                }
            }
            Ok(out)
        }
        _ => Err(format!(
            "input[{i}]: `content` must be a string or an array"
        )),
    }
}

/// User / system / developer content → chat content.
fn convert_content(content: &Value, i: usize) -> Result<Value, String> {
    let parts = match content {
        Value::String(_) => return Ok(content.clone()),
        Value::Array(parts) => parts,
        _ => {
            return Err(format!(
                "input[{i}]: `content` must be a string or an array"
            ))
        }
    };
    let mut out = Vec::with_capacity(parts.len());
    for part in parts {
        let typ = part.get("type").and_then(Value::as_str).unwrap_or("");
        let converted = match typ {
            "input_text" | "output_text" | "text" => json!({
                "type": "text",
                "text": part.get("text").and_then(Value::as_str).unwrap_or(""),
            }),
            "input_image" => {
                if part.get("file_id").is_some_and(|v| !v.is_null()) {
                    return Err(format!(
                        "input[{i}]: `input_image.file_id` is not supported; pass `image_url`"
                    ));
                }
                let url = url_field(part.get("image_url"))
                    .ok_or_else(|| format!("input[{i}]: `input_image` is missing `image_url`"))?;
                let mut image_url = Map::new();
                image_url.insert("url".into(), Value::String(url));
                if let Some(d) = part.get("detail").filter(|v| !v.is_null()) {
                    image_url.insert("detail".into(), d.clone());
                }
                json!({"type": "image_url", "image_url": image_url})
            }
            "image_url" => part.clone(),
            other => {
                return Err(format!(
                    "input[{i}]: content part type `{other}` is not supported"
                ))
            }
        };
        out.push(converted);
    }
    if let [single] = out.as_slice() {
        if single["type"] == "text" {
            return Ok(single["text"].clone());
        }
    }
    Ok(Value::Array(out))
}

fn url_field(v: Option<&Value>) -> Option<String> {
    match v? {
        Value::String(s) => Some(s.clone()),
        Value::Object(o) => o.get("url").and_then(Value::as_str).map(str::to_owned),
        _ => None,
    }
}

/// `namespace` groups are flattened; a `custom` tool becomes a function taking
/// one `input` string.
fn convert_tools(tools: &Value) -> Result<(Vec<Value>, ToolMap), String> {
    let Value::Array(tools) = tools else {
        return Err("`tools` must be an array".into());
    };
    let (mut out, mut map) = (Vec::new(), ToolMap::new());
    for (i, tool) in tools.iter().enumerate() {
        if tool.get("type").and_then(Value::as_str) != Some("namespace") {
            convert_tool(tool, None, &format!("tools[{i}]"), &mut out, &mut map)?;
            continue;
        }
        let ns = tool
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("tools[{i}] (namespace) is missing `name`"))?;
        let children = tool
            .get("tools")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("tools[{i}] (namespace) is missing `tools`"))?;
        for (j, child) in children.iter().enumerate() {
            convert_tool(
                child,
                Some(ns),
                &format!("tools[{i}].tools[{j}]"),
                &mut out,
                &mut map,
            )?;
        }
    }
    let mut seen = std::collections::HashSet::new();
    for name in out
        .iter()
        .filter_map(|t| t.pointer("/function/name")?.as_str())
    {
        if !seen.insert(name) {
            return Err(format!("tools: more than one tool is named `{name}`"));
        }
    }
    Ok((out, map))
}

fn convert_tool(
    tool: &Value,
    ns: Option<&str>,
    at: &str,
    out: &mut Vec<Value>,
    map: &mut ToolMap,
) -> Result<(), String> {
    let typ = tool.get("type").and_then(Value::as_str).unwrap_or("");
    if !matches!(typ, "function" | "custom") {
        return Err(format!(
            "{at}: tool type `{typ}` is not supported; only `function`, `custom` and \
             `namespace` tools are"
        ));
    }
    if ns.is_none() && tool.get("function").is_some_and(Value::is_object) {
        out.push(tool.clone());
        return Ok(());
    }
    let name = tool
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{at} is missing `name`"))?;
    let flat = flat_name(ns, name);
    let mut function = Map::new();
    function.insert("name".into(), Value::String(flat.clone()));
    let custom = typ == "custom";
    if custom {
        let mut desc = tool
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        if let Some(def) = tool.pointer("/format/definition").and_then(Value::as_str) {
            let syntax = tool
                .pointer("/format/syntax")
                .and_then(Value::as_str)
                .unwrap_or("");
            desc.push_str(&format!(
                "\n\n`input` must follow this {syntax} grammar:\n{def}"
            ));
        }
        function.insert("description".into(), Value::String(desc));
        function.insert(
            "parameters".into(),
            json!({"type": "object", "properties": {"input": {"type": "string"}},
                   "required": ["input"]}),
        );
    } else {
        for key in ["description", "parameters", "strict"] {
            if let Some(v) = tool.get(key).filter(|v| !v.is_null()) {
                function.insert(key.into(), v.clone());
            }
        }
        function
            .entry("parameters")
            .or_insert_with(|| json!({"type": "object", "properties": {}}));
    }
    if custom || ns.is_some() {
        let r = ToolRef {
            namespace: ns.map(str::to_owned),
            name: name.to_owned(),
            custom,
        };
        map.insert(flat, r);
    }
    out.push(json!({"type": "function", "function": function}));
    Ok(())
}

fn convert_tool_choice(tc: &Value) -> Result<Value, String> {
    match tc {
        Value::String(s) if matches!(s.as_str(), "auto" | "none" | "required") => Ok(tc.clone()),
        Value::String(s) => Err(format!(
            "`tool_choice` `{s}` is not supported; use auto, none, required or a function"
        )),
        Value::Object(o) => match o.get("type").and_then(Value::as_str) {
            Some(t @ ("function" | "custom")) => {
                if o.get("function").is_some_and(Value::is_object) {
                    return Ok(tc.clone());
                }
                let name = o
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!("`tool_choice` of type {t} is missing `name`"))?;
                let name = flat_name(o.get("namespace").and_then(Value::as_str), name);
                Ok(json!({"type": "function", "function": {"name": name}}))
            }
            // Keep the mode, drop the allow-list.
            Some("allowed_tools") => Ok(o
                .get("mode")
                .cloned()
                .unwrap_or_else(|| Value::String("auto".into()))),
            other => Err(format!(
                "`tool_choice` type `{}` is not supported",
                other.unwrap_or("<missing>")
            )),
        },
        _ => Err("`tool_choice` must be a string or an object".into()),
    }
}

/// `text.format` → `response_format`; unknown types go to the engine to reject.
fn convert_text_format(text: &Value) -> Result<Option<Value>, String> {
    let Some(format) = text.get("format").filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    let Value::Object(f) = format else {
        return Err("`text.format` must be an object".into());
    };
    match f.get("type").and_then(Value::as_str) {
        Some("text") => Ok(None),
        Some("json_schema") => {
            let mut js = Map::new();
            for key in ["name", "schema", "strict", "description"] {
                if let Some(v) = f.get(key) {
                    js.insert(key.into(), v.clone());
                }
            }
            Ok(Some(json!({"type": "json_schema", "json_schema": js})))
        }
        Some(_) => Ok(Some(format.clone())),
        None => Err("`text.format` is missing `type`".into()),
    }
}

/// Reasoning tiers the engine accepts, passed through unchanged.
const EFFORTS: &[&str] = &["none", "minimal", "low", "medium", "high", "xhigh", "max"];

fn convert_reasoning_effort(reasoning: &Value) -> Result<Option<&'static str>, String> {
    let Value::Object(r) = reasoning else {
        return Err("`reasoning` must be an object".into());
    };
    match r.get("effort") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(e)) => EFFORTS
            .iter()
            .find(|t| *t == e)
            .map(|t| Some(*t))
            .ok_or_else(|| {
                format!(
                    "`reasoning.effort` `{e}` is invalid; expected one of {}",
                    EFFORTS.join(", ")
                )
            }),
        Some(_) => Err("`reasoning.effort` must be a string".into()),
    }
}

/// Unset sampling fields echo `null`: the engine's model defaults apply.
fn echo_fields(req: &Map<String, Value>) -> Map<String, Value> {
    let get = |k: &str| req.get(k).filter(|v| !v.is_null()).cloned();
    let mut m = Map::new();
    m.insert(
        "instructions".into(),
        get("instructions").unwrap_or(Value::Null),
    );
    m.insert(
        "max_output_tokens".into(),
        get("max_output_tokens").unwrap_or(Value::Null),
    );
    m.insert(
        "max_tool_calls".into(),
        get("max_tool_calls").unwrap_or(Value::Null),
    );
    m.insert(
        "metadata".into(),
        get("metadata").unwrap_or_else(|| json!({})),
    );
    m.insert(
        "parallel_tool_calls".into(),
        get("parallel_tool_calls").unwrap_or(Value::Bool(true)),
    );
    m.insert("previous_response_id".into(), Value::Null);
    let reasoning = get("reasoning").unwrap_or_else(|| json!({}));
    m.insert(
        "reasoning".into(),
        json!({
            "effort": reasoning.get("effort").cloned().unwrap_or(Value::Null),
            "summary": reasoning.get("summary").cloned().unwrap_or(Value::Null),
        }),
    );
    m.insert(
        "service_tier".into(),
        get("service_tier").unwrap_or_else(|| "default".into()),
    );
    m.insert("store".into(), get("store").unwrap_or(Value::Bool(false)));
    m.insert(
        "temperature".into(),
        get("temperature").unwrap_or(Value::Null),
    );
    m.insert(
        "text".into(),
        get("text").unwrap_or_else(|| json!({"format": {"type": "text"}})),
    );
    m.insert(
        "tool_choice".into(),
        get("tool_choice").unwrap_or_else(|| "auto".into()),
    );
    m.insert("tools".into(), get("tools").unwrap_or_else(|| json!([])));
    m.insert(
        "top_logprobs".into(),
        get("top_logprobs").unwrap_or(Value::Null),
    );
    m.insert("top_p".into(), get("top_p").unwrap_or(Value::Null));
    m.insert(
        "truncation".into(),
        get("truncation").unwrap_or_else(|| "disabled".into()),
    );
    m.insert("user".into(), get("user").unwrap_or(Value::Null));
    m.insert("background".into(), Value::Bool(false));
    m.insert(
        "prompt_cache_key".into(),
        get("prompt_cache_key").unwrap_or(Value::Null),
    );
    m.insert(
        "safety_identifier".into(),
        get("safety_identifier").unwrap_or(Value::Null),
    );
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chat(req: Value) -> Value {
        to_chat(req).expect("converts").chat
    }

    #[test]
    fn text_input_and_instructions() {
        let c = chat(json!({"model": "m", "input": "hi", "instructions": "be brief"}));
        assert_eq!(
            c["messages"],
            json!([{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}])
        );
        assert_eq!(c["stream"], false);
        assert!(c.get("stream_options").is_none());
    }

    #[test]
    fn message_items_with_and_without_type() {
        let c = chat(json!({"model": "m", "input": [
            {"role": "user", "content": "a"},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "b"}]},
        ]}));
        assert_eq!(
            c["messages"],
            json!([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}])
        );
    }

    #[test]
    fn developer_becomes_system_and_leading_system_merges() {
        let c = chat(json!({"model": "m", "instructions": "A", "input": [
            {"role": "developer", "content": "B"},
            {"role": "system", "content": [{"type": "input_text", "text": "C"}]},
            {"role": "user", "content": "hi"},
            {"role": "developer", "content": "D"},
        ]}));
        assert_eq!(
            c["messages"],
            json!([
                {"role": "system", "content": "A\n\nB\n\nC"},
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "D"},
            ])
        );
    }

    #[test]
    fn image_parts_become_image_url() {
        let c = chat(json!({"model": "m", "input": [{"role": "user", "content": [
            {"type": "input_text", "text": "what"},
            {"type": "input_image", "image_url": "https://x/a.png", "detail": "low"},
        ]}]}));
        assert_eq!(
            c["messages"][0]["content"],
            json!([
                {"type": "text", "text": "what"},
                {"type": "image_url", "image_url": {"url": "https://x/a.png", "detail": "low"}},
            ])
        );
    }

    #[test]
    fn reasoning_and_function_calls_merge_into_one_assistant_turn() {
        let c = chat(json!({"model": "m", "input": [
            {"role": "user", "content": "weather?"},
            {"type": "reasoning", "id": "rs_1", "summary": [],
             "content": [{"type": "reasoning_text", "text": "need tool"}]},
            {"type": "function_call", "call_id": "c1", "name": "get_weather",
             "arguments": "{\"city\":\"bj\"}"},
            {"type": "function_call", "call_id": "c2", "name": "get_time", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "sunny"},
            {"type": "function_call_output", "call_id": "c2", "output": "noon"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "sunny at noon"}]},
        ]}));
        let msgs = c["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 5);
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["content"], Value::Null);
        assert_eq!(msgs[1]["reasoning_content"], "need tool");
        assert_eq!(msgs[1]["tool_calls"].as_array().unwrap().len(), 2);
        assert_eq!(msgs[1]["tool_calls"][1]["id"], "c2");
        assert_eq!(
            msgs[2],
            json!({"role": "tool", "tool_call_id": "c1", "content": "sunny"})
        );
        assert_eq!(
            msgs[4],
            json!({"role": "assistant", "content": "sunny at noon"})
        );
    }

    #[test]
    fn tools_tool_choice_text_format_reasoning() {
        let c = chat(json!({
            "model": "m", "input": "x",
            "tools": [{"type": "function", "name": "f", "description": "d",
                       "parameters": {"type": "object"}, "strict": true}],
            "tool_choice": {"type": "function", "name": "f"},
            "text": {"format": {"type": "json_schema", "name": "s",
                                "schema": {"type": "object"}, "strict": true}},
            "reasoning": {"effort": "xhigh"},
            "max_output_tokens": 64,
        }));
        assert_eq!(
            c["tools"],
            json!([{"type": "function", "function": {"name": "f", "description": "d",
                    "parameters": {"type": "object"}, "strict": true}}])
        );
        assert_eq!(
            c["tool_choice"],
            json!({"type": "function", "function": {"name": "f"}})
        );
        assert_eq!(
            c["response_format"],
            json!({"type": "json_schema", "json_schema": {"name": "s",
                   "schema": {"type": "object"}, "strict": true}})
        );
        assert_eq!(c["reasoning_effort"], "xhigh");
        assert_eq!(c["max_completion_tokens"], 64);
        assert!(c.get("max_output_tokens").is_none());
        assert!(c.get("text").is_none());
    }

    #[test]
    fn unknown_format_type_is_forwarded_for_the_engine_to_reject() {
        let c = chat(json!({"model": "m", "input": "x",
                            "text": {"format": {"type": "not_a_format"}}}));
        assert_eq!(c["response_format"], json!({"type": "not_a_format"}));
    }

    #[test]
    fn stream_forces_include_usage_and_passthrough_survives() {
        let c = chat(
            json!({"model": "m", "input": "x", "stream": true, "top_k": 20,
                            "chat_template_kwargs": {"thinking": true},
                            "stream_options": {"continuous_usage_stats": true}}),
        );
        assert_eq!(
            c["stream_options"],
            json!({"continuous_usage_stats": true, "include_usage": true})
        );
        assert_eq!(c["top_k"], 20);
        assert_eq!(c["chat_template_kwargs"], json!({"thinking": true}));
    }

    #[test]
    fn rejects_bad_and_stateful_requests() {
        for (req, needle) in [
            (json!({"input": "x"}), "`model` is required"),
            (json!({"model": "m"}), "`input` is required"),
            (json!({"model": "m", "input": []}), "empty"),
            (
                json!({"model": "m", "input": "x", "stream": "yes"}),
                "boolean",
            ),
            (
                json!({"model": "m", "input": "x", "previous_response_id": "resp_1"}),
                "full conversation",
            ),
            (
                json!({"model": "m", "input": [{"content": "x"}]}),
                "missing `type`",
            ),
            (
                json!({"model": "m", "input": "x", "tools": [{"type": "web_search"}]}),
                "web_search",
            ),
            (
                json!({"model": "m", "input": [{"role": "user", "content": [
                    {"type": "input_video", "video_url": "https://x/v.mp4"}]}]}),
                "input_video",
            ),
            (
                json!({"model": "m", "input": "x", "reasoning": {"effort": "huge"}}),
                "invalid",
            ),
            (
                json!({"model": "m", "input": "x", "top_logprobs": 2}),
                "top_logprobs",
            ),
            (
                json!({"model": "m", "input": "x", "tools": [
                    {"type": "function", "name": "mcp__fs__read"},
                    {"type": "namespace", "name": "mcp__fs__",
                     "tools": [{"type": "function", "name": "read"}]}]}),
                "more than one tool",
            ),
        ] {
            let err = to_chat(req.clone()).expect_err(&req.to_string());
            assert!(err.contains(needle), "{req}: {err}");
        }
    }

    #[test]
    fn echo_defaults() {
        let e = to_chat(json!({"model": "m", "input": "x", "max_output_tokens": 16}))
            .unwrap()
            .echo;
        assert_eq!(e.model, "m");
        assert_eq!(e.fields["max_output_tokens"], 16);
        assert_eq!(e.fields["tool_choice"], "auto");
        assert_eq!(e.fields["tools"], json!([]));
        assert_eq!(e.fields["text"], json!({"format": {"type": "text"}}));
        assert_eq!(
            e.fields["reasoning"],
            json!({"effort": null, "summary": null})
        );
        assert_eq!(e.fields["temperature"], Value::Null);
    }

    #[test]
    fn codex_namespace_and_custom_tools() {
        let c = to_chat(json!({"model": "m", "tools": [
            {"type": "function", "name": "shell", "parameters": {"type": "object"}},
            {"type": "custom", "name": "apply_patch", "description": "Edit files.",
             "format": {"type": "grammar", "syntax": "lark", "definition": "start: patch"}},
            {"type": "namespace", "name": "mcp__fs__", "description": "fs",
             "tools": [{"type": "function", "name": "read", "parameters": {"type": "object"}}]},
        ],
            "input": [
                {"type": "message", "role": "user", "content": "go"},
                {"type": "custom_tool_call", "call_id": "c1", "name": "apply_patch",
                 "input": "*** Begin Patch"},
                {"type": "custom_tool_call_output", "call_id": "c1", "output": "ok"},
                {"type": "function_call", "call_id": "c2", "name": "read",
                 "namespace": "mcp__fs__", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c2", "output": "data"},
            ]}))
        .unwrap();
        let names: Vec<&str> = c.chat["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["function"]["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["shell", "apply_patch", "mcp__fs__read"]);
        let patch = &c.chat["tools"][1]["function"];
        assert_eq!(patch["parameters"]["required"], json!(["input"]));
        assert!(patch["description"]
            .as_str()
            .unwrap()
            .contains("start: patch"));
        assert!(c.echo.tools["apply_patch"].custom);
        assert_eq!(
            c.echo.tools["mcp__fs__read"].namespace.as_deref(),
            Some("mcp__fs__")
        );
        assert!(!c.echo.tools.contains_key("shell"));

        let m = &c.chat["messages"];
        assert_eq!(
            m[1]["tool_calls"][0]["function"],
            json!({"name": "apply_patch", "arguments": "{\"input\":\"*** Begin Patch\"}"})
        );
        assert_eq!(
            m[2],
            json!({"role": "tool", "tool_call_id": "c1", "content": "ok"})
        );
        assert_eq!(m[3]["tool_calls"][0]["function"]["name"], "mcp__fs__read");
    }
}
