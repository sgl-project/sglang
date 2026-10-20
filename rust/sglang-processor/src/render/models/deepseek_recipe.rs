//! DeepSeek-V4 and V4.1 prompts rendered by `deepseek-recipe`, the encoder behind
//! the official API. `pre_normalize` mirrors `deepseek_recipe_bridge.pre_normalize`
//! (python/sglang/srt/entrypoints/openai/) rule for rule; `tests/recipe_parity.rs`
//! pins both to the same outputs.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use deepseek_recipe::openai::ChatCompletionRequest;
use deepseek_recipe::request::{ConversionOptions, ModelVersion, ProtocolRequest};
use deepseek_recipe_core::special_tokens::SpecialTokensMap;
use deepseek_recipe_encoding::v4::dsv4::DeepseekV4Encoding;
use deepseek_recipe_encoding::v4::dsv41::DeepseekV41Encoding;
use deepseek_recipe_encoding::{PromptEncoding, PromptTokenizer, TokenizerEncoder};
use dynamo_renderer::RenderedSegment;
use serde_json::{Value, json};

use crate::render::reasoning::{requested_effort, requested_thinking};

/// The request `model` recipe sees; SGLang serves whatever name the client sent.
pub const RECIPE_MODEL_ALIAS: &str = "sglang-deepseek";
/// The checkpoint tokenizer's image token; recipe's bundled tokenizer spells it `<｜image｜>`.
pub const IMAGE_TOKEN: &str = "<｜deepseek_image｜>";

/// Which recipe encoding renders the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecipeModel {
    V4,
    V41,
}

impl RecipeModel {
    /// The encoding for a checkpoint, from `config.json`'s `model_type` and first
    /// `architectures` entry: V4.1 checkpoints may keep a `deepseek_v4` model type.
    /// Mirrors `deepseek_recipe_bridge.model_version_for`.
    pub fn detect(model_type: Option<&str>, architecture: Option<&str>) -> Option<Self> {
        let architecture = architecture.unwrap_or_default();
        if model_type == Some("deepseek_v41") || architecture.contains("DeepseekV41") {
            return Some(Self::V41);
        }
        if model_type == Some("deepseek_v4") || architecture.contains("DeepseekV4") {
            return Some(Self::V4);
        }
        None
    }

    fn version(self) -> ModelVersion {
        match self {
            Self::V4 => ModelVersion::DeepseekV4Pro,
            Self::V41 => ModelVersion::DeepseekV41Flash,
        }
    }
}

/// The recipe formatter for one checkpoint: recipe's conversion options and
/// prompt encoding, built once, plus recipe's own tokenization (special tokens
/// by id, the plain text between them as plain BPE) when the checkpoint's
/// `tokenizer.json` is attached.
#[derive(Clone)]
pub struct RecipeFormatter {
    options: ConversionOptions,
    encoding: Arc<dyn PromptEncoding + Send + Sync>,
    tokenizer: Option<Arc<PromptTokenizer>>,
}

impl RecipeFormatter {
    /// Render only; [`RecipeRender::token_ids`] stays `None`.
    pub fn new(model: RecipeModel) -> Self {
        let tokens = || SpecialTokensMap {
            image: IMAGE_TOKEN.to_owned(),
            ..SpecialTokensMap::default()
        };
        let encoding: Arc<dyn PromptEncoding + Send + Sync> = match model {
            RecipeModel::V4 => Arc::new(DeepseekV4Encoding::new().with_special_tokens(tokens())),
            RecipeModel::V41 => Arc::new(DeepseekV41Encoding::new().with_special_tokens(tokens())),
        };
        Self {
            options: ConversionOptions::default()
                .with_model_alias(RECIPE_MODEL_ALIAS, model.version())
                .with_special_tokens(tokens()),
            encoding,
            tokenizer: None,
        }
    }

    /// Render and tokenize with the checkpoint's `tokenizer.json` text.
    pub fn from_tokenizer_json(model: RecipeModel, tokenizer_json: &str) -> Result<Self, String> {
        let tokenizer = tokenizers::Tokenizer::from_bytes(tokenizer_json.as_bytes())
            .map_err(|err| format!("invalid tokenizer.json: {err}"))?;
        Ok(Self::new(model).with_tokenizer(PromptTokenizer::new(&tokenizer)))
    }

    /// Render and tokenize with a `tokenizer.json` file.
    pub fn from_tokenizer_file(model: RecipeModel, path: &str) -> Result<Self, String> {
        let tokenizer = tokenizers::Tokenizer::from_file(path)
            .map_err(|err| format!("cannot load {path}: {err}"))?;
        Ok(Self::new(model).with_tokenizer(PromptTokenizer::new(&tokenizer)))
    }

    /// Attach the checkpoint's tokenizer resolved from `model_path` (a model
    /// directory, a tokenizer file or a cached HF repo id); render only when
    /// none is found.
    pub fn for_checkpoint(model: RecipeModel, model_path: &str, revision: Option<&str>) -> Self {
        let loaded = crate::resolve_model_file(model_path, revision, "tokenizer.json")
            .ok_or_else(|| "no tokenizer.json".to_owned())
            .and_then(|file| Self::from_tokenizer_file(model, &file));
        match loaded {
            Ok(formatter) => formatter,
            Err(err) => {
                tracing::warn!(
                    %model_path,
                    error = %err,
                    "deepseek-recipe renders without a tokenizer; it produces no token ids"
                );
                Self::new(model)
            }
        }
    }

    fn with_tokenizer(mut self, tokenizer: PromptTokenizer) -> Self {
        self.tokenizer = Some(Arc::new(tokenizer));
        self
    }
}

/// A prompt rendered by recipe plus the request settings recipe resolved.
#[derive(Debug, Clone, PartialEq)]
pub struct RecipeRender {
    pub prompt: String,
    /// Byte ranges of `prompt` holding explicit special tokens.
    pub special_ranges: Vec<Range<usize>>,
    /// `prompt` split at the special ranges; plain text never parses control tokens.
    pub segments: Vec<RenderedSegment>,
    /// recipe's token ids when the formatter has a tokenizer: the special ranges
    /// as their ids, the text between them as plain BPE.
    pub token_ids: Option<Vec<u32>>,
    pub thinking_mode: bool,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    /// `OutputConstraint::kind` and `OutputConstraint::schema`, when recipe selects one.
    pub output_constraint: Option<(String, String)>,
    /// The prompt ends inside an opened DSML calls block. Always false today:
    /// `pre_normalize` renders required and named tool choices as `auto`.
    pub starts_in_tool_calls: bool,
}

/// `deepseek_recipe_bridge._truthy`: strings are flags spelled out, anything
/// else follows JSON truthiness.
fn truthy(value: &Value) -> bool {
    match value {
        Value::String(text) => {
            ["1", "true", "yes", "y", "on"].contains(&text.trim().to_lowercase().as_str())
        }
        other => minijinja::Value::from_serialize(other).is_true(),
    }
}

/// Rewrite an SGLang chat request body into what recipe's Chat Completions
/// adapter accepts. Mirrors `deepseek_recipe_bridge.pre_normalize`.
pub(crate) fn pre_normalize(
    body: &mut Value,
    defaults: &HashMap<String, Value>,
) -> Result<(), String> {
    let Some(request) = body.as_object_mut() else {
        return Err("request body must be a JSON object".into());
    };
    if request.get("task").is_some_and(|task| !task.is_null()) {
        return Err("task is not supported by the deepseek-recipe chat encoding backend".into());
    }
    // SGLang serves whatever name the client sent.
    request.insert("model".into(), Value::String(RECIPE_MODEL_ALIAS.into()));

    let kwargs = request
        .get("chat_template_kwargs")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    // What `render::reasoning` reads, without cloning the messages.
    let reasoning_view = json!({
        "reasoning": request.get("reasoning"),
        "reasoning_effort": request.get("reasoning_effort"),
    });
    // An explicit `thinking` object (the API's own control) wins; otherwise the
    // kwargs toggle, then the `reasoning` object, then the server default kwargs.
    if !request
        .get("thinking")
        .is_some_and(|thinking| thinking.is_object())
    {
        let kwargs_toggle = ["thinking", "enable_thinking"]
            .iter()
            .find_map(|key| kwargs.get(*key).filter(|value| !value.is_null()))
            .map(truthy);
        let toggle = kwargs_toggle
            .or_else(|| requested_thinking(&reasoning_view))
            .or_else(|| {
                defaults
                    .get("thinking")
                    .filter(|value| !value.is_null())
                    .map(truthy)
            });
        if let Some(enabled) = toggle {
            let kind = if enabled { "enabled" } else { "disabled" };
            request.insert("thinking".into(), json!({ "type": kind }));
        }
    }

    // Effort: kwargs effort replaces the request's, as serving_chat does; numeric
    // budgets collapse onto recipe's tiers until recipe accepts 1-100 directly.
    let effort = kwargs
        .get("reasoning_effort")
        .filter(|value| !value.is_null())
        .cloned()
        .or_else(|| requested_effort(&reasoning_view).cloned())
        .or_else(|| {
            defaults
                .get("reasoning_effort")
                .filter(|value| !value.is_null())
                .cloned()
        });
    match effort.as_ref().and_then(effort_tier) {
        Some(tier) => {
            request.insert("reasoning_effort".into(), Value::String(tier.into()));
        }
        None => {
            request.remove("reasoning_effort");
        }
    }
    request.remove("reasoning");
    request.remove("chat_template_kwargs");

    if request.get("max_tokens").is_none_or(Value::is_null)
        && let Some(max_completion_tokens) = request
            .get("max_completion_tokens")
            .filter(|value| !value.is_null())
            .cloned()
    {
        request.insert("max_tokens".into(), max_completion_tokens);
    }
    // The host samples `n` completions from one prompt and enforces its own
    // response formats (structural_tag) with its grammar.
    request.remove("n");
    if request.get("response_format").is_some_and(|format| {
        format.is_object()
            && !format["type"].as_str().is_some_and(|kind| {
                matches!(kind, "text" | "json_object" | "json_schema" | "regex")
            })
    }) {
        request.remove("response_format");
    }
    normalize_tool_choice(request);

    let continue_final = request
        .remove("continue_final_message")
        .is_some_and(|value| truthy(&value));
    let Some(Value::Array(messages)) = request.get_mut("messages") else {
        return Err("messages must be an array".into());
    };
    normalize_history(messages, continue_final);
    Ok(())
}

/// `required` and a named tool render as `auto`: recipe would pre-open the DSML
/// calls block in the prompt, but SGLang enforces the choice with its own grammar
/// and its parsers expect the model to emit the block. A named tool keeps only
/// that tool, as `serving_chat._process_messages` does.
fn normalize_tool_choice(request: &mut serde_json::Map<String, Value>) {
    let Some(choice) = request.get("tool_choice") else {
        return;
    };
    if let Some(name) = choice["function"]["name"]
        .as_str()
        .filter(|_| choice.is_object())
    {
        let name = name.to_owned();
        if let Some(Value::Array(tools)) = request.get("tools") {
            let chosen: Vec<Value> = tools
                .iter()
                .filter(|tool| tool["function"]["name"].as_str() == Some(name.as_str()))
                .cloned()
                .collect();
            if !chosen.is_empty() {
                request.insert("tools".into(), Value::Array(chosen));
            }
        }
        request.insert("tool_choice".into(), Value::String("auto".into()));
    } else if choice == "required" {
        request.insert("tool_choice".into(), Value::String("auto".into()));
    }
}

/// In place: developer -> system, ids for historical tool calls and their tool
/// messages, serialized arguments, and the continuation `prefix` mark.
fn normalize_history(messages: &mut [Value], continue_final: bool) {
    let mut pending_call_ids: Vec<String> = Vec::new();
    let mut synthesized = 0usize;
    for message in messages.iter_mut() {
        let Some(object) = message.as_object_mut() else {
            continue;
        };
        if object.get("role").and_then(Value::as_str) == Some("developer") {
            object.insert("role".into(), Value::String("system".into()));
        }
        match object.get("role").and_then(Value::as_str) {
            Some("assistant") => {
                pending_call_ids.clear();
                if let Some(Value::Array(tool_calls)) = object.get_mut("tool_calls") {
                    for tool_call in tool_calls.iter_mut() {
                        let Some(call) = tool_call.as_object_mut() else {
                            continue;
                        };
                        if call.get("id").is_none_or(Value::is_null) {
                            call.insert(
                                "id".into(),
                                Value::String(format!("call_{synthesized:02}_sglang")),
                            );
                            synthesized += 1;
                        }
                        if let Some(id) = call.get("id").and_then(Value::as_str) {
                            pending_call_ids.push(id.to_owned());
                        }
                        if let Some(Value::Object(function)) = call.get_mut("function")
                            && let Some(arguments) = function.get("arguments")
                            && !arguments.is_string()
                        {
                            let text = python_json_dumps(arguments);
                            function.insert("arguments".into(), Value::String(text));
                        }
                    }
                }
            }
            Some("tool") => {
                if object.get("tool_call_id").is_none_or(Value::is_null)
                    && !pending_call_ids.is_empty()
                {
                    let id = pending_call_ids.remove(0);
                    object.insert("tool_call_id".into(), Value::String(id));
                } else if let Some(id) = object.get("tool_call_id").and_then(Value::as_str) {
                    pending_call_ids.retain(|pending| pending != id);
                }
            }
            _ => {}
        }
    }
    if continue_final
        && let Some(last) = messages.last_mut().and_then(Value::as_object_mut)
        && last.get("role").and_then(Value::as_str) == Some("assistant")
    {
        last.insert("prefix".into(), Value::Bool(true));
    }
}

/// `json.dumps(value, ensure_ascii=False, separators=(",", ":"))`: compact JSON
/// whose floats print as Python's `repr` does (`1e-05`, `1e+16`, `100.0`).
fn python_json_dumps(value: &Value) -> String {
    fn write(value: &Value, out: &mut String) {
        match value {
            Value::Null => out.push_str("null"),
            Value::Bool(flag) => out.push_str(if *flag { "true" } else { "false" }),
            Value::Number(number) => match number.as_f64().filter(|_| number.is_f64()) {
                Some(float) => out.push_str(&python_float_repr(float)),
                None => out.push_str(&number.to_string()),
            },
            // Both escape only `"`, `\` and control characters.
            Value::String(text) => out.push_str(&Value::String(text.clone()).to_string()),
            Value::Array(items) => {
                out.push('[');
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    write(item, out);
                }
                out.push(']');
            }
            Value::Object(fields) => {
                out.push('{');
                for (index, (key, item)) in fields.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    out.push_str(&Value::String(key.clone()).to_string());
                    out.push(':');
                    write(item, out);
                }
                out.push('}');
            }
        }
    }
    let mut out = String::new();
    write(value, &mut out);
    out
}

/// Python's `repr(float)`: the shortest round-trip digits, positional for
/// exponents in -4..16 and `d.ddde±XX` otherwise.
fn python_float_repr(float: f64) -> String {
    let scientific = format!("{float:e}");
    let (mantissa, exponent) = scientific
        .split_once('e')
        .expect("Rust's {:e} has an exponent");
    let exponent: i32 = exponent
        .parse()
        .expect("Rust's {:e} exponent is an integer");
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", mantissa),
    };
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    if (-4..16).contains(&exponent) {
        let point = exponent + 1;
        let positional = if point <= 0 {
            format!("0.{}{digits}", "0".repeat(point.unsigned_abs() as usize))
        } else if point as usize >= digits.len() {
            format!("{digits}{}.0", "0".repeat(point as usize - digits.len()))
        } else {
            format!(
                "{}.{}",
                &digits[..point as usize],
                &digits[point as usize..]
            )
        };
        format!("{sign}{positional}")
    } else {
        let mantissa = if digits.len() > 1 {
            format!("{}.{}", &digits[..1], &digits[1..])
        } else {
            digits
        };
        let exponent_sign = if exponent < 0 { '-' } else { '+' };
        format!("{sign}{mantissa}e{exponent_sign}{:02}", exponent.abs())
    }
}

/// recipe's effort tiers (`deepseek_recipe_bridge._EFFORT_TIERS`).
const EFFORT_TIERS: [&str; 8] = [
    "none", "minimal", "low", "medium", "high", "xhigh", "ultra", "max",
];

/// Map an effort value onto recipe's tiers. Integers 1-100 and floats 0-0.99 are
/// SGLang's numeric budgets, collapsed onto low / high / max (the inverse of
/// `encoding_dsv41.REASONING_EFFORT_MAPPINGS`); `None` for an unusable value,
/// which recipe then treats as unset (its default is `high`).
fn effort_tier(effort: &Value) -> Option<&'static str> {
    let budget = match effort {
        Value::String(text) => {
            let text = text.trim();
            if let Some(tier) = EFFORT_TIERS.iter().find(|tier| **tier == text) {
                return Some(tier);
            }
            text.parse::<f64>().ok()?
        }
        Value::Number(number) => number.as_f64()?,
        _ => return None,
    };
    // Python's round() is ties-to-even.
    let budget = if (0.0..1.0).contains(&budget) {
        (budget * 100.0).round_ties_even().max(1.0)
    } else {
        budget
    };
    if !(1.0..=100.0).contains(&budget) {
        return None;
    }
    Some(if budget <= 50.0 {
        "low"
    } else if budget <= 75.0 {
        "high"
    } else {
        "max"
    })
}

/// Render an SGLang chat request body with recipe.
pub(crate) fn render(
    formatter: &RecipeFormatter,
    mut body: Value,
    defaults: &HashMap<String, Value>,
) -> Result<RecipeRender, String> {
    pre_normalize(&mut body, defaults)?;
    let request: ChatCompletionRequest =
        serde_json::from_value(body).map_err(|err| format!("invalid request body: {err}"))?;
    let converted = request
        .convert(formatter.options.clone())
        .map_err(|err| err.to_string())?;
    let rendered = formatter
        .encoding
        .render_conversation(&converted.conversation);
    let segments = segments(&rendered.prompt, &rendered.special_ranges);
    let token_ids = match &formatter.tokenizer {
        Some(tokenizer) => Some(tokenizer.encode_ids(&rendered.prompt, &rendered.special_ranges)?),
        None => None,
    };
    let inference = &converted.inference_options;
    Ok(RecipeRender {
        token_ids,
        output_constraint: inference
            .output_constraint
            .as_ref()
            .map(|constraint| (constraint.kind().to_owned(), constraint.schema())),
        temperature: inference.temperature,
        top_p: inference.top_p,
        max_tokens: inference.max_tokens,
        thinking_mode: converted.conversation.thinking_mode,
        starts_in_tool_calls: converted.parsing_options.starts_in_tool_calls,
        special_ranges: rendered.special_ranges,
        prompt: rendered.prompt,
        segments,
    })
}

/// Split the prompt into plain segments, which encode without the added
/// vocabulary, and the special tokens, which encode as their ids.
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

/// Thinking mode recipe resolves for the request: `thinking.type`, then the
/// kwargs toggle, then `reasoning_effort` / `reasoning`, then the server
/// default kwargs, else on.
pub(crate) fn thinking(request: &Value, defaults: &HashMap<String, Value>) -> bool {
    let mut body = request.clone();
    if pre_normalize(&mut body, defaults).is_err() {
        return true;
    }
    match body["thinking"]["type"].as_str() {
        Some(kind) => kind != "disabled",
        None => body["reasoning_effort"].as_str() != Some("none"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(messages: Value) -> Value {
        json!({"model": "deepseek-flash", "messages": messages})
    }

    #[test]
    fn developer_becomes_system_and_kwargs_toggle_becomes_thinking() {
        let mut request = body(json!([
            {"role": "developer", "content": "Be terse."},
            {"role": "user", "content": "hi"}
        ]));
        request["chat_template_kwargs"] = json!({"thinking": false});
        pre_normalize(&mut request, &HashMap::new()).unwrap();
        assert_eq!(request["messages"][0]["role"], "system");
        assert_eq!(request["thinking"], json!({"type": "disabled"}));
        assert_eq!(request["model"], RECIPE_MODEL_ALIAS);
        assert!(request.get("chat_template_kwargs").is_none());
    }

    #[test]
    fn continuation_and_tool_call_ids() {
        let mut request = body(json!([
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"type": "function", "function": {"name": "get_weather", "arguments": {"location": "Paris"}}}
            ]},
            {"role": "tool", "content": "sunny"},
            {"role": "assistant", "content": "It is"}
        ]));
        request["continue_final_message"] = json!(true);
        request["n"] = json!(2);
        pre_normalize(&mut request, &HashMap::new()).unwrap();
        let call = &request["messages"][1]["tool_calls"][0];
        assert_eq!(call["id"], "call_00_sglang");
        assert_eq!(call["function"]["arguments"], "{\"location\":\"Paris\"}");
        assert_eq!(request["messages"][2]["tool_call_id"], "call_00_sglang");
        assert_eq!(request["messages"][3]["prefix"], true);
        assert!(request.get("n").is_none());
    }

    #[test]
    fn numeric_efforts_collapse_onto_tiers() {
        assert_eq!(effort_tier(&json!(75)), Some("high"));
        assert_eq!(effort_tier(&json!(0.3)), Some("low"));
        assert_eq!(effort_tier(&json!(100)), Some("max"));
        assert_eq!(effort_tier(&json!("minimal")), Some("minimal"));
        assert_eq!(effort_tier(&json!("bogus")), None);
    }

    #[test]
    fn renders_a_v41_prompt_with_special_segments() {
        let request = body(json!([{"role": "user", "content": "<｜User｜>"}]));
        let mut request = request;
        request["thinking"] = json!({"type": "disabled"});
        let render = render(
            &RecipeFormatter::new(RecipeModel::V41),
            request,
            &HashMap::new(),
        )
        .unwrap();
        assert_eq!(
            render.prompt,
            "<｜begin▁of▁sentence｜><｜User｜><｜User｜><｜Assistant｜></think>"
        );
        // Template tokens are special; the user's spelling is plain text.
        assert_eq!(render.special_ranges.len(), 4);
        let plain: Vec<&str> = render
            .segments
            .iter()
            .filter(|segment| !segment.allow_special)
            .map(|segment| segment.text.as_str())
            .collect();
        assert_eq!(plain, ["<｜User｜>"]);
        assert!(!render.thinking_mode);
        assert_eq!(render.temperature, Some(1.0));
    }

    #[test]
    fn required_tool_choice_renders_as_auto() {
        let mut request = body(json!([{"role": "user", "content": "weather in Paris?"}]));
        request["tools"] = json!([{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}}}]);
        request["tool_choice"] = json!("required");
        request["thinking"] = json!({"type": "disabled"});
        let render = render(
            &RecipeFormatter::new(RecipeModel::V41),
            request,
            &HashMap::new(),
        )
        .unwrap();
        // SGLang's grammar enforces the choice; the prompt does not pre-open the block.
        assert!(render.prompt.ends_with("<｜Assistant｜></think>"));
        assert!(!render.starts_in_tool_calls);
    }

    #[test]
    fn named_tool_choice_keeps_only_that_tool() {
        let mut request = body(json!([{"role": "user", "content": "hi"}]));
        request["tools"] = json!([
            {"type": "function", "function": {"name": "a"}},
            {"type": "function", "function": {"name": "b"}}
        ]);
        request["tool_choice"] = json!({"type": "function", "function": {"name": "b"}});
        pre_normalize(&mut request, &HashMap::new()).unwrap();
        assert_eq!(request["tool_choice"], "auto");
        assert_eq!(request["tools"].as_array().unwrap().len(), 1);
        assert_eq!(request["tools"][0]["function"]["name"], "b");
    }

    #[test]
    fn a_null_kwargs_toggle_falls_through_to_the_next_source() {
        let mut request = body(json!([{"role": "user", "content": "hi"}]));
        request["chat_template_kwargs"] = json!({"thinking": null, "enable_thinking": false});
        pre_normalize(&mut request, &HashMap::new()).unwrap();
        assert_eq!(request["thinking"], json!({"type": "disabled"}));
    }

    #[test]
    fn floats_print_like_python() {
        for (value, expected) in [
            (0.00001, "1e-05"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (100.0, "100.0"),
            (0.0001, "0.0001"),
            (1.5, "1.5"),
            (-0.0, "-0.0"),
            (0.30000000000000004, "0.30000000000000004"),
            (2.5e-7, "2.5e-07"),
            (1e22, "1e+22"),
            (-123456.789, "-123456.789"),
        ] {
            assert_eq!(python_float_repr(value), expected, "{value}");
        }
        assert_eq!(
            python_json_dumps(&json!({"a": 1, "b": [1.0, "x\"y"], "c": {"d": null, "e": true}})),
            r#"{"a":1,"b":[1.0,"x\"y"],"c":{"d":null,"e":true}}"#
        );
    }

    #[test]
    fn task_is_rejected_explicitly() {
        let mut request = body(json!([{"role": "user", "content": "hi"}]));
        request["task"] = json!("title");
        assert!(
            render(
                &RecipeFormatter::new(RecipeModel::V41),
                request,
                &HashMap::new()
            )
            .unwrap_err()
            .contains("task is not supported")
        );
    }

    #[test]
    fn detects_the_checkpoint_family_like_the_python_bridge() {
        assert_eq!(
            RecipeModel::detect(Some("deepseek_v41"), None),
            Some(RecipeModel::V41)
        );
        assert_eq!(
            RecipeModel::detect(Some("deepseek_v4"), Some("DeepseekV41ForCausalLM")),
            Some(RecipeModel::V41)
        );
        assert_eq!(
            RecipeModel::detect(Some("deepseek_v4"), Some("DeepseekV4ForCausalLM")),
            Some(RecipeModel::V4)
        );
        assert_eq!(
            RecipeModel::detect(None, Some("DeepseekV4ForCausalLM")),
            Some(RecipeModel::V4)
        );
        assert_eq!(RecipeModel::detect(Some("deepseek_v32"), None), None);
        assert_eq!(RecipeModel::detect(None, None), None);
    }
}
