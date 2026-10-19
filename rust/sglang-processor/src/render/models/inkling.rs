//! Inkling prompts as SGLang's `serving_inkling.py` renders them through
//! tml-renderers' TMLv0 renderer (`inkling_renderer.render_inkling_messages`).

use std::collections::HashMap;

use dynamo_renderer::{RenderedPrompt, RenderedSegment};
use serde_json::{Map, Number, Value};

use super::deepseek_v4::pydantic_bool;
use super::kimi_k25::deep_sort;
use super::schema_types::normalize_json_schema_types;
use crate::render::reasoning::requested_effort;

const MESSAGE_USER: &str = "<|message_user|>";
const MESSAGE_MODEL: &str = "<|message_model|>";
const MESSAGE_SYSTEM: &str = "<|message_system|>";
const MESSAGE_TOOL: &str = "<|message_tool|>";
const CONTENT_TEXT: &str = "<|content_text|>";
const CONTENT_THINKING: &str = "<|content_thinking|>";
const CONTENT_XML: &str = "<|content_xml|>";
const CONTENT_INVOKE_TOOL_JSON: &str = "<|content_invoke_tool_json|>";
const CONTENT_MODEL_END_SAMPLING: &str = "<|content_model_end_sampling|>";
const END_MESSAGE: &str = "<|end_message|>";
const DEFAULT_EFFORT: f64 = 0.9;
const MAX_EFFORT: f64 = 0.99;
const EFFORT_ENV: &str = "SGLANG_INKLING_DEFAULT_REASONING_EFFORT";

/// Render an SGLang chat request body. The prompt's segments keep text out of
/// special tokens, as tml-renderers encodes text with `encode_ordinary`; the
/// `continue_final_message` prefix is part of the prompt, so the returned
/// prefix is always empty.
pub(crate) fn render(
    request: Value,
    defaults: &HashMap<String, Value>,
) -> Result<(RenderedPrompt, String), String> {
    let mut messages = request["messages"]
        .as_array()
        .ok_or("messages must be an array")?
        .iter()
        .map(dump_message)
        .collect::<Result<Vec<_>, _>>()?;
    validate_tool_choice(&request)?;
    validate_response_format(&request["response_format"])?;
    let effort = effort(&request, defaults)?;
    let prefix = pop_assistant_prefix(&mut messages, &request)?;
    let mut declaration = declared_tools(&request)?
        .map(|tools| tool_declaration(&tools))
        .transpose()?;

    let mut blocks = Vec::new();
    let mut call_names = HashMap::new();
    let leading_systems = messages
        .iter()
        .take_while(|m| m["role"] == "system")
        .count();
    for (index, message) in messages.iter().enumerate() {
        if index == leading_systems
            && let Some(declaration) = declaration.take()
        {
            blocks.push(declaration);
        }
        render_message(message, &mut call_names, &mut blocks)?;
    }
    blocks.extend(declaration);
    // `render_for_completion_with_effort`: before the first non-system message.
    let at = blocks
        .iter()
        .position(|block| !block.is_system())
        .unwrap_or(blocks.len());
    blocks.insert(
        at,
        Block::message(
            MESSAGE_SYSTEM,
            None,
            CONTENT_TEXT,
            format!("Thinking effort level: {}", format_effort(effort)),
        ),
    );

    let mut segments = Segments::default();
    for block in &blocks {
        block.write(&mut segments);
    }
    // `render_inkling_assistant_prefix`: an open model text block.
    if let Some(prefix) = prefix {
        segments.special(MESSAGE_MODEL);
        segments.special(CONTENT_TEXT);
        segments.text(&prefix);
    }
    Ok((RenderedPrompt::segmented(segments.0), String::new()))
}

/// The pydantic dump `_to_oss_messages` reads: roles lowercased and checked,
/// `user` reduced to its content, null fields dropped, `developer` as `system`.
fn dump_message(message: &Value) -> Result<Value, String> {
    let role = message["role"].as_str().unwrap_or_default().to_lowercase();
    let kept: &[&str] = match role.as_str() {
        "user" => &["content"],
        "system" | "developer" | "assistant" | "tool" => &[
            "content",
            "tool_call_id",
            "name",
            "reasoning_content",
            "tool_calls",
        ],
        _ => return Err(format!("Unsupported role: {role}")),
    };
    let mut dumped: Map<String, Value> = kept
        .iter()
        .filter_map(|key| Some((key.to_string(), message.get(*key)?.clone())))
        .filter(|(_, value)| !value.is_null())
        .collect();
    let role = if role == "developer" { "system" } else { &role };
    dumped.insert("role".into(), role.into());
    Ok(dumped.into())
}

/// `_validate_request`: a required choice needs request or system-turn tools,
/// and a named choice must name a request tool.
fn validate_tool_choice(request: &Value) -> Result<(), String> {
    let request_tools = request["tools"].as_array().map_or(&[][..], Vec::as_slice);
    let message_tools = request["messages"]
        .as_array()
        .into_iter()
        .flatten()
        .any(|m| {
            matches!(
                m["role"].as_str().map(str::to_lowercase).as_deref(),
                Some("system" | "developer")
            ) && m["tools"].as_array().is_some_and(|tools| !tools.is_empty())
        });
    match &request["tool_choice"] {
        Value::String(choice) if choice == "required" => {
            if request_tools.is_empty() && !message_tools {
                return Err("Tools cannot be empty if tool choice is set to required.".into());
            }
        }
        Value::Object(choice) => {
            let name = &choice.get("function").unwrap_or(&Value::Null)["name"];
            if !request_tools
                .iter()
                .any(|tool| tool["function"]["name"] == *name)
            {
                return Err(format!("Tool {name} not found in tools list."));
            }
        }
        _ => {}
    }
    Ok(())
}

/// `_validate_request` rejects a `json_schema` format without a schema.
fn validate_response_format(format: &Value) -> Result<(), String> {
    if format["type"] == "json_schema"
        && format["json_schema"]["schema"].is_null()
        && format["schema"].is_null()
    {
        return Err("schema_ is required for json_schema response format request.".into());
    }
    Ok(())
}

/// The effort `InklingSpecAdapter.encode_messages` renders: kwargs effort
/// replaces the request's, then the server's default kwargs and
/// `SGLANG_INKLING_DEFAULT_REASONING_EFFORT` fill in.
fn effort(request: &Value, defaults: &HashMap<String, Value>) -> Result<f64, String> {
    // Pydantic validates the request's effort even when kwargs replace it.
    let requested = requested_effort(request).map(parse_effort).transpose()?;
    let kwargs =
        Some(&request["chat_template_kwargs"]["reasoning_effort"]).filter(|e| !e.is_null());
    if let Some(effort) = kwargs {
        return parse_effort(effort);
    }
    if let Some(effort) = requested {
        return Ok(effort);
    }
    if let Some(effort) = defaults.get("reasoning_effort").filter(|e| !e.is_null()) {
        return parse_effort(effort);
    }
    match std::env::var(EFFORT_ENV).ok().filter(|v| !v.is_empty()) {
        None => Ok(DEFAULT_EFFORT),
        Some(value) => value
            .trim()
            .parse::<f64>()
            .ok()
            .filter(|e| e.is_finite() && (0.0..=MAX_EFFORT).contains(e))
            .ok_or_else(|| format!("{EFFORT_ENV} must be in [0.0, 0.99]")),
    }
}

/// `serving_inkling.parse_reasoning_effort`, with pydantic's numeric strings.
fn parse_effort(value: &Value) -> Result<f64, String> {
    let effort = match value {
        Value::Bool(_) => return Err("Inkling reasoning_effort must not be a boolean".into()),
        Value::Number(number) => number.as_f64(),
        Value::String(tier) => match tier.as_str() {
            "none" => Some(0.0),
            "minimal" => Some(0.1),
            "low" => Some(0.2),
            "medium" => Some(0.7),
            "high" => Some(0.9),
            "xhigh" | "max" => Some(MAX_EFFORT),
            numeric => numeric.trim().parse::<f64>().ok(),
        },
        _ => None,
    }
    .ok_or_else(|| format!("invalid Inkling reasoning_effort: {value}"))?;
    if !effort.is_finite() || !(0.0..=MAX_EFFORT).contains(&effort) {
        return Err("Inkling reasoning_effort must be in [0.0, 0.99]".into());
    }
    Ok(effort)
}

/// TMLv0's `ThinkingEffort(millieffort=round(effort * 1000))` text: two
/// decimals of `millieffort / 1000`, both roundings half-to-even in `f64`.
fn format_effort(effort: f64) -> String {
    let milli = (effort * 1000.0).round_ties_even();
    let centi = (milli / 1000.0 * 100.0).round_ties_even() as u32;
    let text = format!("{}.{:02}", centi / 100, centi % 100);
    text.trim_end_matches('0').trim_end_matches('.').to_owned()
}

/// `_pop_assistant_prefix`: a final assistant turn with text content and no
/// tool calls or reasoning is continued rather than closed.
fn pop_assistant_prefix(
    messages: &mut Vec<Value>,
    request: &Value,
) -> Result<Option<String>, String> {
    if !request
        .get("continue_final_message")
        .map_or(Ok(false), pydantic_bool)?
    {
        return Ok(None);
    }
    let continuable = messages.last().is_some_and(|m| {
        m["role"] == "assistant"
            && m["content"].is_string()
            && m["tool_calls"].as_array().is_none_or(Vec::is_empty)
            && m["reasoning_content"].as_str().is_none_or(str::is_empty)
    });
    Ok(continuable.then(|| {
        messages.pop().unwrap()["content"]
            .as_str()
            .unwrap()
            .to_owned()
    }))
}

/// The request tools `_process_messages` renders: none for `tool_choice`
/// `"none"`, only the named one for a named choice.
fn declared_tools(request: &Value) -> Result<Option<Vec<Value>>, String> {
    let tools = match request["tools"].as_array() {
        Some(tools) if !tools.is_empty() => tools,
        _ => return Ok(None),
    };
    let tools: Vec<Value> = match &request["tool_choice"] {
        Value::String(choice) if choice == "none" => return Ok(None),
        Value::Object(choice) => tools
            .iter()
            .filter(|tool| tool["function"]["name"] == choice["function"]["name"])
            .cloned()
            .collect(),
        _ => tools.clone(),
    };
    Ok(Some(tools))
}

/// TMLv0's `ToolSpecJson`: each function flattened to its description, name,
/// parameters and type, after `_validate_request` normalizes schema types.
fn tool_declaration(tools: &[Value]) -> Result<Block, String> {
    let specs = tools
        .iter()
        .map(|tool| {
            let function = &tool["function"];
            let name = function["name"]
                .as_str()
                .ok_or("tool function requires name")?;
            let mut parameters = match &function["parameters"] {
                Value::Null => Value::Object(Map::new()),
                parameters @ Value::Object(_) => parameters.clone(),
                _ => return Err("tool function parameters must be an object".to_string()),
            };
            normalize_json_schema_types(&mut parameters);
            let mut spec = Map::new();
            let description = match &function["description"] {
                Value::Null => "",
                Value::String(description) => description,
                _ => return Err("tool function description must be a string".to_string()),
            };
            spec.insert("description".into(), description.into());
            spec.insert("name".into(), name.into());
            spec.insert("parameters".into(), parameters);
            spec.insert(
                "type".into(),
                tool["type"].as_str().unwrap_or("function").into(),
            );
            Ok(Value::Object(spec))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Block::message(
        MESSAGE_SYSTEM,
        Some("tool_declare".into()),
        CONTENT_XML,
        canonical_json(Value::Array(specs), NumberStyle::Declaration)?,
    ))
}

/// One OSS message through `OpenAIMessage.from_oss_messages(...).to_messages()`.
fn render_message(
    message: &Value,
    call_names: &mut HashMap<String, String>,
    blocks: &mut Vec<Block>,
) -> Result<(), String> {
    let role = message["role"].as_str().unwrap_or_default();
    let calls = message["tool_calls"]
        .as_array()
        .map_or(&[][..], Vec::as_slice);
    let mut parts = content_parts(role, message, !calls.is_empty())?;
    if role != "assistant" {
        let (header, author) = match role {
            "system" => (MESSAGE_SYSTEM, None),
            "user" => (MESSAGE_USER, None),
            _ => (MESSAGE_TOOL, tool_author(message, call_names)),
        };
        for part in parts {
            if part.thinking {
                return Err("thinking content parts are only valid in assistant messages".into());
            }
            blocks.push(Block::message(
                header,
                author.clone(),
                CONTENT_TEXT,
                part.text,
            ));
        }
        return Ok(());
    }
    // An assistant turn with tool calls renders no empty text.
    if !calls.is_empty() {
        parts.retain(|part| part.thinking || !part.text.is_empty());
    }
    let start = blocks.len();
    for part in parts {
        let kind = if part.thinking {
            CONTENT_THINKING
        } else {
            CONTENT_TEXT
        };
        blocks.push(Block::message(MESSAGE_MODEL, None, kind, part.text));
    }
    for call in calls {
        let function = &call["function"];
        let name = function["name"]
            .as_str()
            .ok_or("tool call function requires name")?;
        if let Some(id) = call["id"].as_str().filter(|id| !id.is_empty()) {
            call_names.insert(id.to_owned(), name.to_owned());
        }
        let payload = format!(
            "{{\"name\":{},\"args\":{}}}",
            Value::from(name),
            canonical_json(
                tool_call_arguments(&function["arguments"])?,
                NumberStyle::Arguments
            )?
        );
        blocks.push(Block::message(
            MESSAGE_MODEL,
            Some(name.into()),
            CONTENT_INVOKE_TOOL_JSON,
            payload,
        ));
    }
    if blocks.len() > start {
        blocks.push(Block::EndSampling);
    }
    Ok(())
}

struct Part {
    thinking: bool,
    text: String,
}

/// `_to_oss_messages`: string content is one text part and null content an
/// empty one unless the turn calls tools; truthy `reasoning_content` leads an
/// assistant turn as a thinking part.
fn content_parts(role: &str, message: &Value, has_calls: bool) -> Result<Vec<Part>, String> {
    let text = |text: &str| Part {
        thinking: false,
        text: text.to_owned(),
    };
    let reasoning = match &message["reasoning_content"] {
        Value::Null => None,
        Value::String(reasoning) => Some(reasoning.as_str()).filter(|r| !r.is_empty()),
        _ => return Err("reasoning_content must be a string".into()),
    }
    .filter(|_| role == "assistant");
    let mut parts = Vec::new();
    if let Some(reasoning) = reasoning {
        parts.push(Part {
            thinking: true,
            text: reasoning.to_owned(),
        });
    }
    match &message["content"] {
        Value::Null if reasoning.is_none() && !(has_calls && role == "assistant") => {
            parts.push(text(""))
        }
        Value::Null => {}
        Value::String(content) if reasoning.is_some() && content.is_empty() => {}
        Value::String(content) => parts.push(text(content)),
        Value::Array(items) => {
            for item in items {
                let part = content_part(item)?;
                if part.thinking && reasoning.is_some() {
                    return Err(
                        "assistant message cannot mix reasoning_content with ordered thinking parts"
                            .into(),
                    );
                }
                parts.push(part);
            }
        }
        _ => return Err("message content must be a string or a list".into()),
    }
    Ok(parts)
}

/// `_to_oss_part`; media expands into placeholders only on the engine.
fn content_part(part: &Value) -> Result<Part, String> {
    let as_text = |value: &Value| match value {
        Value::Null => Ok(String::new()),
        Value::String(text) => Ok(text.clone()),
        _ => Err("content part text must be a string".to_string()),
    };
    match part["type"].as_str() {
        Some("text") => Ok(Part {
            thinking: false,
            text: as_text(&part["text"])?,
        }),
        Some("thinking" | "reasoning") => Ok(Part {
            thinking: true,
            text: as_text(match &part["thinking"] {
                Value::Null => &part["text"],
                thinking => thinking,
            })?,
        }),
        Some("image_url" | "video_url" | "audio_url" | "input_audio") => {
            Err("Inkling media renders on the engine".into())
        }
        kind => Err(format!("unsupported content part type: {kind:?}")),
    }
}

/// A tool result's author: its `name`, else the earlier call with its `tool_call_id`.
fn tool_author(message: &Value, call_names: &HashMap<String, String>) -> Option<String> {
    message["name"]
        .as_str()
        .filter(|name| !name.is_empty())
        .or_else(|| {
            call_names
                .get(message["tool_call_id"].as_str()?)
                .map(String::as_str)
        })
        .map(str::to_owned)
}

/// `normalize_assistant_tool_call_arguments(strict=True)`, then TMLv0's object check.
fn tool_call_arguments(arguments: &Value) -> Result<Value, String> {
    let arguments = match arguments {
        Value::String(text) => serde_json::from_str(text).map_err(|_| {
            "Assistant tool call function.arguments must be valid JSON.".to_string()
        })?,
        arguments => arguments.clone(),
    };
    if !arguments.is_object() {
        return Err("tool_call arguments must decode to a JSON object".into());
    }
    Ok(arguments)
}

#[derive(Clone, Copy)]
enum NumberStyle {
    /// Call arguments: TMLv0 reads Python's `json.dumps` text with serde_json's
    /// default (inexact) float parser.
    Arguments,
    /// Tool declarations go through that parser twice and sign positive exponents.
    Declaration,
}

/// Sorted keys, compact separators and raw non-ASCII, as TMLv0 serializes JSON.
fn canonical_json(mut value: Value, style: NumberStyle) -> Result<String, String> {
    deep_sort(&mut value);
    let mut out = String::new();
    write_json(&value, style, &mut out)?;
    Ok(out)
}

fn write_json(value: &Value, style: NumberStyle, out: &mut String) -> Result<(), String> {
    match value {
        Value::Number(number) => out.push_str(&tml_number(number, style)?),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_json(item, style, out)?;
            }
            out.push(']');
        }
        Value::Object(fields) => {
            out.push('{');
            for (index, (key, item)) in fields.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                out.push_str(&Value::from(key.as_str()).to_string());
                out.push(':');
                write_json(item, style, out)?;
            }
            out.push('}');
        }
        scalar => out.push_str(&scalar.to_string()),
    }
    Ok(())
}

/// A number as TMLv0 prints it. Negative zero (Python prints `-0` as `0` but
/// `-0.0` as itself) and integral floats from 2^63 (possibly integer literals
/// beyond 64 bits, which Python keeps exact) are ambiguous once parsed.
fn tml_number(number: &Number, style: NumberStyle) -> Result<String, String> {
    if number.is_u64() || number.is_i64() {
        return Ok(number.to_string());
    }
    let value = number.as_f64().ok_or("number out of range")?;
    if (value == 0.0 && value.is_sign_negative())
        || (value.fract() == 0.0 && value.abs() >= 9223372036854775808.0)
    {
        return Err("Inkling numbers serde_json reads unlike Python render on the engine".into());
    }
    let mut buffer = ryu::Buffer::new();
    Ok(match style {
        NumberStyle::Arguments => buffer.format_finite(serde_reparse(value)).to_owned(),
        NumberStyle::Declaration => {
            let printed = buffer.format_finite(serde_reparse(serde_reparse(value)));
            match printed.split_once('e') {
                Some((mantissa, exponent)) if !exponent.starts_with('-') => {
                    format!("{mantissa}e+{exponent}")
                }
                _ => printed.to_owned(),
            }
        }
    })
}

/// serde_json's default float parse (`f64_from_parts`) of `value` as ryu and
/// Python's `repr` print it: the decimal significand times an `f64` power of ten,
/// which can round twice.
fn serde_reparse(value: f64) -> f64 {
    let mut buffer = ryu::Buffer::new();
    let text = buffer.format_finite(value.abs());
    let (mantissa, exponent) = text.split_once('e').unwrap_or((text, "0"));
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let mut exponent = exponent.parse::<i32>().unwrap() - fraction.len() as i32;
    let mut parsed = format!("{whole}{fraction}").parse::<u64>().unwrap() as f64;
    loop {
        if exponent.unsigned_abs() <= 308 {
            let power = format!("1e{}", exponent.unsigned_abs())
                .parse::<f64>()
                .unwrap();
            if exponent >= 0 {
                parsed *= power;
            } else {
                parsed /= power;
            }
            break;
        }
        if parsed == 0.0 {
            break;
        }
        parsed /= 1e308;
        exponent += 308;
    }
    parsed.copysign(value)
}

enum Block {
    Message {
        header: &'static str,
        author: Option<String>,
        kind: &'static str,
        text: String,
    },
    EndSampling,
}

impl Block {
    fn message(
        header: &'static str,
        author: Option<String>,
        kind: &'static str,
        text: String,
    ) -> Self {
        Block::Message {
            header,
            author,
            kind,
            text,
        }
    }

    fn is_system(&self) -> bool {
        matches!(self, Block::Message { header, .. } if *header == MESSAGE_SYSTEM)
    }

    fn write(&self, segments: &mut Segments) {
        match self {
            Block::Message {
                header,
                author,
                kind,
                text,
            } => {
                segments.special(header);
                segments.text(author.as_deref().unwrap_or_default());
                segments.special(kind);
                segments.text(text);
                segments.special(END_MESSAGE);
            }
            Block::EndSampling => segments.special(CONTENT_MODEL_END_SAMPLING),
        }
    }
}

/// Prompt segments: control tokens may tokenize as specials, text may not.
#[derive(Default)]
struct Segments(Vec<RenderedSegment>);

impl Segments {
    fn special(&mut self, token: &str) {
        self.push(token, true);
    }

    fn text(&mut self, text: &str) {
        self.push(text, false);
    }

    fn push(&mut self, text: &str, allow_special: bool) {
        if text.is_empty() {
            return;
        }
        match self.0.last_mut() {
            Some(last) if last.allow_special && allow_special => last.text.push_str(text),
            _ => self.0.push(RenderedSegment::new(text, allow_special)),
        }
    }
}
