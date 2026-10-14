//! Kimi K3 prompts as SGLang's `serving_chat.py` builds them, on Dynamo's K3 formatter.

use std::collections::HashMap;

use dynamo_renderer::kimi_k3::KimiK3Formatter;
use dynamo_renderer::{OAIChatLikeRequest, OAIPromptFormatter, RenderedPrompt};
use serde_json::{Map, Value, json};

use super::deepseek_v4::{engine_message, pydantic_bool, tool_arguments};
use super::schema_types::normalize_json_schema_types;
use crate::render::reasoning::{requested_effort, requested_thinking};

const IMAGE_PLACEHOLDER: &str = "<|kimi_image_placeholder|>";
const THINKING_EFFORTS: [&str; 3] = ["low", "high", "max"];
const TOKENIZER_ARGS: [&str; 8] = [
    "conversation",
    "messages",
    "tools",
    "add_generation_prompt",
    "padding",
    "truncation",
    "max_length",
    "return_tensors",
];

/// Render an SGLang chat request body. Returns the prompt, whose segments keep
/// the special-token boundaries, and the `continue_final_message` prefix,
/// which SGLang tokenizes separately.
pub(crate) fn render(
    mut request: Value,
    defaults: &HashMap<String, Value>,
) -> Result<(RenderedPrompt, String), String> {
    let Some(Value::Array(messages)) = request.get_mut("messages").map(Value::take) else {
        return Err("messages must be an array".into());
    };
    let mut messages = messages
        .into_iter()
        .map(|message| prepare_message(engine_message(message)))
        .collect::<Result<Vec<_>, _>>()?;
    let request = &request;
    // `_handle_last_assistant_message`, which K3 runs only to continue the turn.
    let mut prefix = String::new();
    if request
        .get("continue_final_message")
        .map_or(Ok(false), pydantic_bool)?
        && let Some(last) = messages.last().filter(|m| m["role"] == "assistant")
        && let Some(content) = last["content"].as_str()
    {
        prefix = content.to_owned();
        messages.pop();
    }

    // The template kwargs: the request's, with `reasoning_effort` moved to the
    // request effort, then what `reasoning` implies and the server's defaults.
    let mut kwargs = request["chat_template_kwargs"]
        .as_object()
        .cloned()
        .unwrap_or_default();
    let kwargs_effort = kwargs.remove("reasoning_effort").filter(|e| !e.is_null());
    if let Some(thinking) = requested_thinking(request) {
        kwargs.entry("thinking").or_insert(thinking.into());
    }
    for (key, value) in defaults {
        kwargs.entry(key).or_insert_with(|| value.clone());
    }
    // Kimi's `apply_chat_template` and `build_chat_segments` take these as their
    // own arguments: SGLang truncates or pads the ids, or fails on a duplicate argument.
    if TOKENIZER_ARGS.iter().any(|arg| kwargs.contains_key(*arg)) {
        return Err(
            "Kimi K3 tokenizer arguments in chat_template_kwargs render on the engine".into(),
        );
    }
    let thinking = kwargs
        .get("thinking")
        .is_none_or(|thinking| minijinja::Value::from_serialize(thinking).is_true());
    let supported = |effort: &Value| {
        effort
            .as_str()
            .is_some_and(|effort| THINKING_EFFORTS.contains(&effort))
    };
    let effort = kwargs_effort
        .as_ref()
        .or_else(|| requested_effort(request))
        .or_else(|| defaults.get("reasoning_effort").filter(|e| !e.is_null()));
    // The tokenizer's default is `max`.
    let thinking_effort = match kwargs.get("thinking_effort") {
        Some(thinking_effort) => thinking_effort.clone(),
        None => effort
            .filter(|effort| supported(effort))
            .cloned()
            .unwrap_or_else(|| "max".into()),
    };
    // `build_chat_segments` rejects other efforts; Dynamo cannot leave out a null one.
    if thinking && !supported(&thinking_effort) {
        return Err(format!("unsupported thinking_effort {thinking_effort}"));
    }
    // Dynamo drops whitespace-only reasoning, which SGLang renders.
    if thinking
        && messages.iter().any(|m| {
            m["role"] == "assistant"
                && m["reasoning_content"]
                    .as_str()
                    .is_some_and(|r| !r.is_empty() && r.trim().is_empty())
        })
    {
        return Err("whitespace-only Kimi K3 reasoning renders on the engine".into());
    }

    let tools = match request["tools"].as_array() {
        Some(tools) if !tools.is_empty() => {
            Some(tools.iter().map(tool_dump).collect::<Result<Value, _>>()?)
        }
        _ => None,
    };
    // `_effective_tools`: the request's and system turns' declarations.
    let has_tools = tools.is_some() || messages.iter().any(|m| m.get("tools").is_some());
    // `_validate_request` rejects a required choice without tools.
    if !has_tools
        && request["tool_choice"]
            .as_str()
            .is_some_and(|choice| choice.eq_ignore_ascii_case("required"))
    {
        return Err("tool_choice required needs tools".into());
    }
    if has_tools && matches!(request["tool_choice"].as_str(), Some("required" | "none")) {
        kwargs
            .entry("tool_choice")
            .or_insert_with(|| request["tool_choice"].clone());
    }
    let request_format = response_format_dump(&request["response_format"]);
    // `_validate_request` rejects a `json_schema` format without a schema.
    if request_format
        .as_ref()
        .is_some_and(|f| f["type"] == "json_schema" && f["json_schema"]["schema"].is_null())
    {
        return Err("json_schema response format requires a schema".into());
    }
    // The encoder also takes a bare type name.
    let mut response_format = match kwargs.get("response_format") {
        Some(Value::String(kind)) => Some(json!({ "type": kind })),
        Some(format) => Some(format.clone()),
        None => request_format,
    };
    if let (Some(format), Some(schema)) = (
        response_format.as_mut().and_then(Value::as_object_mut),
        kwargs.get("response_schema"),
    ) {
        format.insert("json_schema".into(), json!({ "schema": schema }));
    }

    let encoder_input = KimiRequest {
        messages,
        tools,
        // The encoder renders only these choices; a named one stays a constraint.
        tool_choice: kwargs
            .get("tool_choice")
            .filter(|choice| matches!(choice.as_str(), Some("required" | "none")))
            .cloned(),
        response_format,
        args: HashMap::from([
            ("thinking".into(), thinking.into()),
            ("thinking_effort".into(), thinking_effort),
        ]),
    };
    if (encoder_input.messages.iter())
        .chain(&encoder_input.tools)
        .chain(&encoder_input.response_format)
        .any(prints_unlike_python)
    {
        return Err("Kimi K3 numbers serde_json prints unlike Python render on the engine".into());
    }
    // SGLang renders every declared tool, whatever the tool_choice.
    let prompt = KimiK3Formatter::new(false)
        .render_prompt(&encoder_input)
        .map_err(|error| error.to_string())?;
    Ok((prompt, prefix))
}

/// `_prepare_kimi_k3_messages` on one dumped message, after
/// `normalize_assistant_tool_call_arguments(strict=False)`.
fn prepare_message(mut message: Value) -> Result<Value, String> {
    if let Some(parts) = message["content"].as_array_mut() {
        // Media expands into image tokens only on the engine.
        if parts.iter().any(|part| {
            matches!(
                part["type"].as_str(),
                Some("image_url" | "video_url" | "audio_url" | "input_audio")
            )
        }) {
            return Err("Kimi K3 media renders on the engine".into());
        }
        parts.retain(|part| part["type"] == "text");
    }
    neutralize(&mut message["content"]);
    if let Some(reasoning) = message.get_mut("reasoning_content") {
        neutralize(reasoning);
    }
    for arguments in tool_arguments(&mut message) {
        if let Some(object) = arguments
            .as_str()
            .and_then(|text| serde_json::from_str::<Value>(text).ok())
            .filter(Value::is_object)
        {
            *arguments = object;
        }
        // SGLang's encoder and Dynamo parse other text differently.
        if arguments.as_str().is_some_and(|text| !text.is_empty()) {
            return Err("non-object tool arguments render on the engine".into());
        }
        neutralize(arguments);
    }
    if message["role"] == "developer" {
        message["role"] = "system".into();
    }
    // A system turn with tools renders only its declaration; Dynamo rejects
    // tools on other turns, which SGLang ignores.
    let tools = message.as_object_mut().and_then(|m| m.remove("tools"));
    if let Some(Value::Array(tools)) = tools.filter(|_| message["role"] == "system")
        && !tools.is_empty()
    {
        message["tools"] = tools.iter().map(tool_dump).collect::<Result<_, _>>()?;
        message["content"] = "".into();
    }
    Ok(message)
}

/// `neutralize_kimi_k3_image_placeholder_value`: text cannot spell the image placeholder.
fn neutralize(value: &mut Value) {
    match value {
        Value::String(text) if text.contains(IMAGE_PLACEHOLDER) => {
            *text = text.replace(IMAGE_PLACEHOLDER, "<| kimi_image_placeholder |>");
        }
        Value::Array(items) => items.iter_mut().for_each(neutralize),
        Value::Object(fields) => fields.values_mut().for_each(neutralize),
        _ => {}
    }
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

/// `Tool.model_dump(exclude_unset=True)`: the fields the client set, coerced,
/// after `_validate_request` normalizes the schema types. The encoder sorts the keys.
fn tool_dump(tool: &Value) -> Result<Value, String> {
    let f = &tool["function"];
    f["name"].as_str().ok_or("tool function requires name")?;
    let mut function: Map<String, Value> = ["name", "description", "parameters"]
        .into_iter()
        .filter_map(|key| Some((key.into(), f.get(key)?.clone())))
        .collect();
    if let Some(parameters) = function.get_mut("parameters") {
        normalize_json_schema_types(parameters);
    }
    if let Some(strict) = f.get("strict") {
        function.insert("strict".into(), pydantic_bool(strict)?.into());
    }
    // `Tool._propagate_defer_loading`; `Function` drops an unset flag.
    if let Some(defer) = [f.get("defer_loading"), tool.get("defer_loading")]
        .into_iter()
        .flatten()
        .find(|v| !v.is_null())
    {
        function.insert("defer_loading".into(), pydantic_bool(defer)?.into());
    }
    let mut dump = Map::new();
    if let Some(kind) = tool.get("type") {
        dump.insert("type".into(), kind.clone());
    }
    dump.insert("function".into(), function.into());
    if let Some(defer) = tool.get("defer_loading") {
        let defer = match defer {
            Value::Null => Value::Null,
            defer => pydantic_bool(defer)?.into(),
        };
        dump.insert("defer_loading".into(), defer);
    }
    Ok(dump.into())
}

/// `ResponseFormat.model_dump(exclude_unset=True)`, after `set_json_schema`
/// lifts a legacy top-level `schema` without its `strict` property.
fn response_format_dump(format: &Value) -> Option<Value> {
    let kind = format.get("type")?;
    let json_schema = match (format.get("json_schema"), format.get("schema")) {
        (Some(Value::Object(json_schema)), _) if !json_schema.is_empty() => Some(
            json_schema
                .iter()
                .filter(|(key, _)| {
                    ["name", "description", "schema", "strict"].contains(&key.as_str())
                })
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect::<Map<_, _>>()
                .into(),
        ),
        (_, Some(Value::Object(schema))) if kind == "json_schema" && !schema.is_empty() => {
            let mut schema = schema.clone();
            if let Some(Value::Object(properties)) = schema.get_mut("properties") {
                properties.remove("strict");
            }
            Some(json!({ "schema": schema }))
        }
        _ => None,
    };
    let mut dump = json!({ "type": kind });
    if let Some(json_schema) = json_schema {
        dump["json_schema"] = json_schema;
    }
    Some(dump)
}

/// The request Dynamo's formatter sees, built the way SGLang calls its encoder.
struct KimiRequest {
    messages: Vec<Value>,
    tools: Option<Value>,
    tool_choice: Option<Value>,
    response_format: Option<Value>,
    args: HashMap<String, Value>,
}

impl OAIChatLikeRequest for KimiRequest {
    fn model(&self) -> String {
        String::new()
    }

    fn messages(&self) -> minijinja::Value {
        minijinja::Value::from_serialize(&self.messages)
    }

    fn tools(&self) -> Option<minijinja::Value> {
        self.tools.as_ref().map(minijinja::Value::from_serialize)
    }

    fn tool_choice(&self) -> Option<minijinja::Value> {
        self.tool_choice
            .as_ref()
            .map(minijinja::Value::from_serialize)
    }

    fn response_format(&self) -> Option<minijinja::Value> {
        self.response_format
            .as_ref()
            .map(minijinja::Value::from_serialize)
    }

    fn should_add_generation_prompt(&self) -> bool {
        true
    }

    fn chat_template_args(&self) -> Option<&HashMap<String, Value>> {
        Some(&self.args)
    }
}
