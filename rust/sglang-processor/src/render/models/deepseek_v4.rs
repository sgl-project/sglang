//! DeepSeek-V4 and V4.1 prompts through deepseek-recipe, as SGLang's
//! `serving_deepseek_recipe.py` and `parser/deepseek_recipe_renderer.py` build them.

use std::collections::HashMap;
use std::ops::Range;

use deepseek_recipe::openai::ChatCompletionRequest;
use deepseek_recipe::request::{ConversionOptions, ProtocolRequest};
use deepseek_recipe_core::special_tokens::SpecialTokensMap;
use deepseek_recipe_encoding::PromptEncoding;
use deepseek_recipe_encoding::v4::dsv4::DeepseekV4Encoding;
use deepseek_recipe_encoding::v4::dsv41::DeepseekV41Encoding;
use dynamo_renderer::{RenderedPrompt, RenderedSegment};
use serde_json::{Map, Value, json};
use tokenizers::{AddedVocabulary, Tokenizer};

use super::pydantic_dump::pydantic_bool;
use super::schema_types::normalize_json_schema_types;
use crate::py::python_json_compact;
use crate::render::reasoning::{requested_effort, requested_thinking};

/// The checkpoint tokenizer's image token; recipe's bundled tokenizer spells it `<｜image｜>`.
const IMAGE_TOKEN: &str = "<｜deepseek_image｜>";
const ASSISTANT_TOKEN: &str = "<｜Assistant｜>";
const THINK_START_TOKEN: &str = "<think>";
const THINK_END_TOKEN: &str = "</think>";
/// `deepseek_recipe_renderer.TASK_TOKENS`.
const TASK_TOKENS: [(&str, &str); 6] = [
    ("action", "<｜action｜>"),
    ("query", "<｜query｜>"),
    ("authority", "<｜authority｜>"),
    ("domain", "<｜domain｜>"),
    ("title", "<｜title｜>"),
    ("read_url", "<｜read_url｜>"),
];
/// `action` follows the generation header; the other tasks replace it.
const TASKS_AFTER_HEADER: [&str; 1] = ["action"];
/// V4.1 renders an integer budget as this line; recipe's "high" tier is budget 75.
const BUDGET_TIER: &str = "high";
const BUDGET_TIER_LINE: &str = "Reasoning Effort: 75";

/// The checkpoint tokenizer, encoding a rendered prompt as SGLang does: special
/// segments with the added vocabulary, plain text without it (recipe's
/// `PromptTokenizer`), and a task's continuation prefix with it (`encode_text`).
pub struct DeepSeekTokenizer {
    full: Tokenizer,
    plain: Tokenizer,
}

impl DeepSeekTokenizer {
    /// From the checkpoint's `tokenizer.json` text.
    pub fn from_json(tokenizer_json: &str) -> Result<Self, String> {
        let full = Tokenizer::from_bytes(tokenizer_json.as_bytes())
            .map_err(|error| format!("invalid tokenizer.json: {error}"))?;
        let mut plain = full.clone();
        plain.with_added_vocabulary(AddedVocabulary::new());
        Ok(Self { full, plain })
    }

    pub(crate) fn encode(&self, prompt: &RenderedPrompt) -> Result<Vec<u32>, String> {
        let segments = prompt
            .encode_segments()
            .ok_or("a DeepSeek prompt is segmented")?;
        let mut ids = Vec::new();
        for segment in segments {
            let tokenizer = if segment.allow_special {
                &self.full
            } else {
                &self.plain
            };
            let encoding = tokenizer
                .encode_fast(segment.text, false)
                .map_err(|error| error.to_string())?;
            ids.extend_from_slice(encoding.get_ids());
        }
        Ok(ids)
    }
}

impl std::fmt::Debug for DeepSeekTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DeepSeekTokenizer")
    }
}

/// Render an SGLang chat request body for DeepSeek-V4 (`v41` false) or V4.1.
pub(crate) fn render(
    v41: bool,
    request: Value,
    defaults: &HashMap<String, Value>,
) -> Result<RenderedPrompt, String> {
    let thinking = thinking(&request, defaults);
    let (effort, budget) = effort(v41, &request, defaults)?;
    let task = request
        .get("task")
        .filter(|task| !task.is_null())
        .map(|task| task.as_str().ok_or("task must be a string"))
        .transpose()?;
    let mut continue_final = request
        .get("continue_final_message")
        .map_or(Ok(false), pydantic_bool)?;

    let mut messages = match &request["messages"] {
        Value::Array(messages) => messages.clone(),
        _ => return Err("messages must be an array".into()),
    };
    // Pydantic lowercases message roles.
    for message in &mut messages {
        if let Some(role) = message["role"].as_str() {
            message["role"] = role.to_lowercase().into();
        }
    }
    // SGLang's trailing-assistant rule, as `_handle_last_assistant_message`.
    let mut task_prefix = None;
    if let Some(content) = messages
        .last()
        .filter(|last| last["role"] == "assistant")
        .and_then(|last| last["content"].as_str())
        .map(str::to_owned)
    {
        if !continue_final {
            *messages.last_mut().unwrap() = json!({"role": "user", "content": content});
        } else if task.is_some() {
            // A task renders its own tail, so the prefix follows it.
            messages.pop();
            task_prefix = Some(content);
            continue_final = false;
        }
    }

    let mut body = Map::new();
    body.insert("messages".into(), messages.into());
    if let Some(response_format) = request.get("response_format").filter(|f| !f.is_null()) {
        body.insert("response_format".into(), response_format.clone());
    }
    if let Some(tools) = handler_tools(&request)? {
        body.insert("tools".into(), tools.into());
        if let Some(choice) = request.get("tool_choice").filter(|c| !c.is_null()) {
            body.insert("tool_choice".into(), choice.clone());
        }
    }
    pre_normalize(&mut body, thinking, continue_final)?;
    body.insert(
        "model".into(),
        if v41 {
            "DeepSeek-V4.1-Flash"
        } else {
            "DeepSeek-V4-Pro"
        }
        .into(),
    );
    if budget.is_some() {
        body.insert("reasoning_effort".into(), BUDGET_TIER.into());
    } else if let Some(effort) = effort {
        body.insert("reasoning_effort".into(), effort.into());
    }

    let request: ChatCompletionRequest = serde_json::from_value(Value::Object(body))
        .map_err(|error| format!("invalid request body: {error}"))?;
    let tokens = || SpecialTokensMap {
        image: IMAGE_TOKEN.into(),
        ..SpecialTokensMap::default()
    };
    let converted = request
        .convert(ConversionOptions::default().with_special_tokens(tokens()))
        .map_err(|error| error.to_string())?;
    let rendered = if v41 {
        DeepseekV41Encoding::new()
            .with_special_tokens(tokens())
            .render_conversation(&converted.conversation)
    } else {
        DeepseekV4Encoding::new()
            .with_special_tokens(tokens())
            .render_conversation(&converted.conversation)
    };
    // The processor serves text only; SGLang rejects images on a text-only server.
    if !rendered.image_sources.is_empty() {
        return Err("image input is not supported for this model".into());
    }
    let mut segments = segments(&rendered.prompt, &rendered.special_ranges);
    if let Some(budget) = budget.filter(|_| thinking) {
        replace_effort_budget(&mut segments, budget)?;
    }
    if let Some(task) = task {
        attach_task(&mut segments, task, thinking)?;
    }
    if let Some(prefix) = task_prefix.filter(|prefix| !prefix.is_empty()) {
        // `encode_text`: the checkpoint tokenizer with its special tokens.
        segments.push(RenderedSegment::new(prefix, true));
    }
    Ok(RenderedPrompt::segmented(segments))
}

/// `serving_chat.py`: kwargs `thinking` wins, then what `reasoning` and the
/// effort imply, then the server's default kwargs and `SGLANG_DEFAULT_THINKING`.
pub(crate) fn thinking(request: &Value, defaults: &HashMap<String, Value>) -> bool {
    if let Some(thinking) = request["chat_template_kwargs"].get("thinking") {
        return minijinja::Value::from_serialize(thinking).is_true();
    }
    if let Some(thinking) = requested_thinking(request) {
        return thinking;
    }
    if let Some(thinking) = defaults.get("thinking") {
        return minijinja::Value::from_serialize(thinking).is_true();
    }
    std::env::var("SGLANG_DEFAULT_THINKING")
        .is_ok_and(|v| ["true", "1", "yes", "y"].contains(&v.to_lowercase().as_str()))
}

/// The named effort recipe renders, or V4.1's integer budget. Kwargs effort
/// replaces the request effort and the server's default kwargs fill in
/// (`serving_chat.py`); unset or `"none"` falls back to the env default
/// (`serving_deepseek_recipe._encode`).
fn effort(
    v41: bool,
    request: &Value,
    defaults: &HashMap<String, Value>,
) -> Result<(Option<String>, Option<u32>), String> {
    let requested = Some(&request["chat_template_kwargs"]["reasoning_effort"])
        .filter(|effort| !effort.is_null())
        .cloned()
        // pydantic's `Literal[...] | float` coerces numeric request efforts to floats.
        .or_else(|| requested_effort(request).map(request_effort_value))
        .or_else(|| {
            defaults
                .get("reasoning_effort")
                .filter(|effort| !effort.is_null())
                .cloned()
        })
        .filter(|effort| effort != "none")
        .or_else(|| env_effort(v41));
    match requested {
        None => Ok((None, None)),
        Some(Value::String(effort)) => Ok((Some(effort), None)),
        Some(effort) if !v41 => Err(format!(
            "DeepSeek-V4 reasoning_effort must be a named tier, got {effort}"
        )),
        Some(effort) => parse_effort_budget(&effort).map(|budget| (None, Some(budget))),
    }
}

/// `protocol.py`: request efforts are a tier name or a float.
fn request_effort_value(effort: &Value) -> Value {
    match effort {
        Value::String(text) => text
            .parse::<f64>()
            .ok()
            .filter(|_| !TIERS.contains(&text.as_str()))
            .map_or_else(|| effort.clone(), |value| json!(value)),
        Value::Number(number) => number.as_f64().map_or_else(|| effort.clone(), |v| json!(v)),
        _ => effort.clone(),
    }
}

const TIERS: [&str; 8] = [
    "none", "minimal", "low", "medium", "high", "xhigh", "ultra", "max",
];

/// `SGLANG_DSV4_REASONING_EFFORT` / `SGLANG_DSV41_REASONING_EFFORT`: digits are a budget.
fn env_effort(v41: bool) -> Option<Value> {
    let name = if v41 {
        "SGLANG_DSV41_REASONING_EFFORT"
    } else {
        "SGLANG_DSV4_REASONING_EFFORT"
    };
    let raw = std::env::var(name).ok()?;
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    Some(match raw.parse::<u64>() {
        Ok(budget) if raw.chars().all(|c| c.is_ascii_digit()) => json!(budget),
        _ => raw.into(),
    })
}

/// `deepseek_recipe_renderer.parse_effort_budget`.
fn parse_effort_budget(effort: &Value) -> Result<u32, String> {
    if let Value::Number(number) = effort {
        if let Some(value) = number.as_u64().filter(|_| !number.is_f64())
            && (1..=100).contains(&value)
        {
            return Ok(value as u32);
        }
        if let Some(value) = number.as_f64().filter(|_| number.is_f64())
            && (0.0..=0.99).contains(&value)
        {
            // Python's `round` breaks ties to even.
            return Ok(((value * 100.0).round_ties_even() as u32).max(1));
        }
    }
    Err(format!(
        "reasoning_effort={effort} is outside the integer budget [1, 100] and the float range [0, 0.99]"
    ))
}

/// The handler's tool dumps: none when `tool_choice` is `"none"` or the request
/// has no tools, only the chosen one for a named choice (`serving_chat.py`).
fn handler_tools(request: &Value) -> Result<Option<Vec<Value>>, String> {
    let Some(tools) = request["tools"]
        .as_array()
        .filter(|tools| !tools.is_empty())
    else {
        return Ok(None);
    };
    let tools = match &request["tool_choice"] {
        Value::String(choice) if choice == "none" => return Ok(None),
        Value::Object(_) => {
            let name = &request["tool_choice"]["function"]["name"];
            let chosen: Vec<Value> = tools
                .iter()
                .filter(|tool| &tool["function"]["name"] == name)
                .cloned()
                .collect();
            if chosen.is_empty() {
                return Ok(None);
            }
            chosen
        }
        _ => tools.clone(),
    };
    tools
        .iter()
        .map(tool_dump)
        .collect::<Result<_, _>>()
        .map(Some)
}

/// `Tool.model_dump()` after `_validate_request` normalizes the schema types,
/// with the null fields `serving_deepseek_recipe` strips left out.
fn tool_dump(tool: &Value) -> Result<Value, String> {
    let source = &tool["function"];
    let name = source["name"]
        .as_str()
        .ok_or("tool function requires name")?;
    let mut function = Map::new();
    if let Some(description) = source.get("description").filter(|d| !d.is_null()) {
        function.insert("description".into(), description.clone());
    }
    function.insert("name".into(), name.into());
    if let Some(parameters) = source.get("parameters").filter(|p| !p.is_null()) {
        let mut parameters = parameters.clone();
        normalize_json_schema_types(&mut parameters);
        function.insert("parameters".into(), parameters);
    }
    let strict = source.get("strict").map_or(Ok(false), pydantic_bool)?;
    function.insert("strict".into(), strict.into());
    // `Tool._propagate_defer_loading` copies the tool's flag into the function.
    if let Some(defer) = [source.get("defer_loading"), tool.get("defer_loading")]
        .into_iter()
        .flatten()
        .find(|value| !value.is_null())
    {
        function.insert("defer_loading".into(), pydantic_bool(defer)?.into());
    }
    let mut dump = Map::new();
    dump.insert(
        "type".into(),
        tool.get("type")
            .filter(|t| !t.is_null())
            .cloned()
            .unwrap_or_else(|| "function".into()),
    );
    dump.insert("function".into(), function.into());
    if let Some(defer) = tool.get("defer_loading").filter(|value| !value.is_null()) {
        dump.insert("defer_loading".into(), pydantic_bool(defer)?.into());
    }
    Ok(dump.into())
}

/// `deepseek_recipe_renderer.pre_normalize`: translate SGLang's request for
/// recipe's Chat Completions adapter.
fn pre_normalize(
    body: &mut Map<String, Value>,
    thinking: bool,
    continue_final: bool,
) -> Result<(), String> {
    body.insert(
        "thinking".into(),
        json!({"type": if thinking { "enabled" } else { "disabled" }}),
    );
    // The host enforces structural_tag with its grammar.
    if body
        .get("response_format")
        .is_some_and(|format| format["type"] == "structural_tag")
    {
        body.remove("response_format");
    }
    if body.contains_key("tools") {
        match body.get("tool_choice") {
            Some(Value::String(choice)) if choice == "required" => {
                // SGLang enforces the choice with its grammar, which expects the
                // model to open the calls block itself.
                body.insert("tool_choice".into(), "auto".into());
            }
            Some(Value::Object(_)) => _ = body.insert("tool_choice".into(), "auto".into()),
            _ => {}
        }
    } else {
        body.remove("tool_choice");
    }
    let Some(Value::Array(messages)) = body.get_mut("messages") else {
        return Err("messages must be an array".into());
    };
    for message in messages.iter_mut() {
        if message["role"] == "developer" {
            message["role"] = "system".into();
        }
        let calls = message
            .get_mut("tool_calls")
            .and_then(Value::as_array_mut)
            .into_iter()
            .flatten();
        for function in calls.filter_map(|call| call.get_mut("function")?.as_object_mut()) {
            if let Some(arguments) = function.get_mut("arguments")
                && !arguments.is_string()
                && !arguments.is_null()
            {
                *arguments = python_json_compact(arguments).into();
            }
        }
    }
    if continue_final
        && let Some(last) = messages
            .last_mut()
            .filter(|last| last["role"] == "assistant")
    {
        last["prefix"] = true.into();
    }
    Ok(())
}

/// The prompt split at its special tokens; plain text never parses control tokens.
fn segments(prompt: &str, special_ranges: &[Range<usize>]) -> Vec<RenderedSegment> {
    let mut segments = Vec::with_capacity(special_ranges.len() * 2 + 1);
    let mut start = 0;
    for range in special_ranges {
        if range.start > start {
            segments.push(RenderedSegment::new(&prompt[start..range.start], false));
        }
        segments.push(RenderedSegment::new(&prompt[range.clone()], true));
        start = range.end;
    }
    if start < prompt.len() {
        segments.push(RenderedSegment::new(&prompt[start..], false));
    }
    segments
}

/// `RecipeEncoder._replace_effort_budget`: the first effort line takes the budget.
fn replace_effort_budget(segments: &mut [RenderedSegment], budget: u32) -> Result<(), String> {
    let segment = segments
        .iter_mut()
        .find(|segment| !segment.allow_special && segment.text.contains(BUDGET_TIER_LINE))
        .ok_or("deepseek-recipe rendered no reasoning effort line")?;
    segment.text =
        segment
            .text
            .replacen(BUDGET_TIER_LINE, &format!("Reasoning Effort: {budget}"), 1);
    Ok(())
}

/// `RecipeEncoder._attach_task`: a quick instruction task on the last user turn.
fn attach_task(
    segments: &mut Vec<RenderedSegment>,
    task: &str,
    thinking: bool,
) -> Result<(), String> {
    let marker = if thinking {
        THINK_START_TOKEN
    } else {
        THINK_END_TOKEN
    };
    let header = segments.len() >= 2
        && segments[segments.len() - 2].allow_special
        && segments[segments.len() - 2].text == ASSISTANT_TOKEN
        && segments[segments.len() - 1].allow_special
        && segments[segments.len() - 1].text == marker;
    if !header {
        return Err("task requires the conversation to end with a user message".into());
    }
    let token = TASK_TOKENS
        .iter()
        .find(|(name, _)| *name == task)
        .map(|(_, token)| *token)
        .ok_or_else(|| format!("unknown task {task:?}"))?;
    if !TASKS_AFTER_HEADER.contains(&task) {
        segments.truncate(segments.len() - 2);
    }
    segments.push(RenderedSegment::new(token, true));
    Ok(())
}
