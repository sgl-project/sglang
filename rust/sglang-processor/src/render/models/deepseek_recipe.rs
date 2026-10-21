//! DeepSeek-V4 and V4.1 prompts rendered by `deepseek-recipe`, the encoder behind
//! the official API. `pre_normalize` mirrors `serving_deepseek_recipe.pre_normalize`
//! (python/sglang/srt/entrypoints/openai/); the `*-recipe.json` fixtures of
//! `tests/parity.rs` pin both to the Python serving path.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use deepseek_recipe::openai::ChatCompletionRequest;
use deepseek_recipe::request::{ConversionOptions, ProtocolRequest};
use deepseek_recipe_core::special_tokens::SpecialTokensMap;
use deepseek_recipe_encoding::v4::dsv4::DeepseekV4Encoding;
use deepseek_recipe_encoding::v4::dsv41::DeepseekV41Encoding;
use deepseek_recipe_encoding::{PromptEncoding, PromptTokenizer, TokenizerEncoder};
use dynamo_renderer::RenderedSegment;
use serde_json::{Value, json};

use crate::render::reasoning::{requested_effort, requested_thinking};

/// The checkpoint tokenizer's image token; recipe's bundled tokenizer spells it `<｜image｜>`.
pub const IMAGE_TOKEN: &str = "<｜deepseek_image｜>";

/// Which recipe encoding renders the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecipeModel {
    V4,
    V41,
}

impl RecipeModel {
    /// recipe's own name for the version. It resolves the version from the request
    /// `model` and rejects any other name; SGLang serves whatever name the client
    /// sent, so this replaces it.
    fn name(self) -> &'static str {
        match self {
            Self::V4 => "DeepSeek-V4-Pro",
            Self::V41 => "DeepSeek-V4.1-Flash",
        }
    }

    /// The encoding for a checkpoint, from `config.json`'s `model_type` and first
    /// `architectures` entry, as `chat_encoding.resolve_chat_encoding_spec` tells
    /// `dsv41` from `dsv4`: V4.1 checkpoints may keep a `deepseek_v4` model type.
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
}

/// The recipe formatter for one checkpoint: recipe's conversion options and
/// prompt encoding, built once, plus recipe's own tokenization (special tokens
/// by id, the plain text between them as plain BPE) when the checkpoint's
/// `tokenizer.json` is attached.
#[derive(Clone)]
pub struct RecipeFormatter {
    model: RecipeModel,
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
            model,
            options: ConversionOptions::default().with_special_tokens(tokens()),
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

    fn with_tokenizer(mut self, tokenizer: PromptTokenizer) -> Self {
        self.tokenizer = Some(Arc::new(tokenizer));
        self
    }
}

/// A prompt rendered by recipe plus the request settings recipe resolved.
#[derive(Debug, Clone, PartialEq)]
pub struct RecipeRender {
    pub prompt: String,
    /// `prompt` split at its special tokens; plain text never parses control tokens.
    pub segments: Vec<RenderedSegment>,
    /// recipe's token ids when the formatter has a tokenizer: the special tokens
    /// as their ids, the text between them as plain BPE.
    pub token_ids: Option<Vec<u32>>,
    pub thinking_mode: bool,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
}

/// Rewrite an SGLang chat request body into what recipe's Chat Completions
/// adapter accepts. Mirrors `serving_deepseek_recipe.pre_normalize`: only SGLang's
/// own request extensions are translated, recipe validates and renders the rest.
/// The hosts pass the wire body, so the `reasoning` fold of
/// `protocol.normalize_reasoning_inputs` and serving_chat's kwargs-effort and
/// default-kwargs steps, which the Python handler runs before its adapter, are
/// repeated here.
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

    // protocol.normalize_reasoning_inputs, via `render::reasoning`: `reasoning`
    // folds into the effort and, with setdefault, the kwargs thinking flag.
    let mut kwargs = match request.remove("chat_template_kwargs") {
        Some(Value::Object(kwargs)) => kwargs,
        _ => serde_json::Map::new(),
    };
    let reasoning_view = json!({
        "reasoning": request.remove("reasoning"),
        "reasoning_effort": request.get("reasoning_effort"),
    });
    if let Some(effort) = requested_effort(&reasoning_view) {
        request.insert("reasoning_effort".into(), effort.clone());
    }
    if let Some(thinking) = requested_thinking(&reasoning_view) {
        kwargs.entry("thinking").or_insert(Value::Bool(thinking));
    }
    // `protocol.ReasoningEffortType` has no `ultra`: the Python host rejects it
    // where recipe would accept it.
    if request.get("reasoning_effort") == Some(&Value::String("ultra".into())) {
        return Err("invalid reasoning_effort: ultra".into());
    }
    // serving_chat._convert_to_internal_request: the kwargs effort replaces the
    // request's; _process_messages: the server defaults fill in.
    if let Some(effort) = kwargs
        .remove("reasoning_effort")
        .filter(|effort| !effort.is_null())
    {
        request.insert("reasoning_effort".into(), effort);
    }
    for (key, value) in defaults {
        kwargs.entry(key.clone()).or_insert_with(|| value.clone());
    }
    if request.get("reasoning_effort").is_none_or(Value::is_null)
        && let Some(effort) = kwargs.get("reasoning_effort")
    {
        request.insert("reasoning_effort".into(), effort.clone());
    }
    // The dsv41 branch reads the kwargs flag with Python truthiness; the API's own
    // `thinking` object wins.
    if request.get("thinking").is_none_or(Value::is_null)
        && let Some(flag) = kwargs.get("thinking")
    {
        let enabled = minijinja::Value::from_serialize(flag).is_true();
        let kind = if enabled { "enabled" } else { "disabled" };
        request.insert("thinking".into(), json!({ "type": kind }));
    }

    // The host enforces structural_tag with its grammar; recipe does not know it.
    if request
        .get("response_format")
        .is_some_and(|format| format["type"] == "structural_tag")
    {
        request.remove("response_format");
    }
    normalize_tool_choice(request);

    let continue_final = request
        .remove("continue_final_message")
        .is_some_and(|value| minijinja::Value::from_serialize(value).is_true());
    let Some(Value::Array(messages)) = request.get_mut("messages") else {
        return Err("messages must be an array".into());
    };
    normalize_history(messages, continue_final);
    Ok(())
}

/// `required` and a named tool render as `auto`: recipe would pre-open the DSML
/// calls block in the prompt, but SGLang enforces the choice with its own grammar
/// and its parsers expect the model to emit the block. A named tool keeps only
/// that tool, and `none` or no tools drops them, as `serving_chat._process_messages`
/// hands them to the encoder.
fn normalize_tool_choice(request: &mut serde_json::Map<String, Value>) {
    // `_process_messages` hands the encoder no tools when the request has none
    // or turns them off.
    if request
        .get("tools")
        .and_then(Value::as_array)
        .is_none_or(Vec::is_empty)
        || request.get("tool_choice") == Some(&Value::String("none".into()))
    {
        request.remove("tools");
        request.remove("tool_choice");
        return;
    }
    let Some(choice) = request.get("tool_choice") else {
        return;
    };
    if let Some(name) = choice["function"]["name"].as_str() {
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

/// In place: `developer` turns become `system` (recipe has no developer role),
/// object-typed tool call arguments become the JSON string recipe takes, and
/// `continue_final_message` marks the last assistant turn as recipe's `prefix`.
/// recipe parses the arguments and prints the values itself, so the string's own
/// formatting (Python's `json.dumps` on the engine) does not reach the prompt.
fn normalize_history(messages: &mut [Value], continue_final: bool) {
    for message in messages.iter_mut() {
        let Some(object) = message.as_object_mut() else {
            continue;
        };
        if object.get("role").and_then(Value::as_str) == Some("developer") {
            object.insert("role".into(), Value::String("system".into()));
        }
        // SGLang's message models have no `prefix`; Pydantic drops it.
        object.remove("prefix");
        if let Some(Value::Array(tool_calls)) = object.get_mut("tool_calls") {
            for call in tool_calls.iter_mut() {
                if let Some(Value::Object(function)) = call.get_mut("function")
                    && let Some(arguments) = function.get("arguments")
                    && !arguments.is_string()
                {
                    function.insert("arguments".into(), Value::String(arguments.to_string()));
                }
            }
        }
    }
    if continue_final
        && let Some(last) = messages.last_mut().and_then(Value::as_object_mut)
        && last.get("role").and_then(Value::as_str) == Some("assistant")
    {
        last.insert("prefix".into(), Value::Bool(true));
    }
}

/// Render an SGLang chat request body with recipe; `encode` also tokenizes it
/// when the formatter has the checkpoint's tokenizer.
pub(crate) fn render(
    formatter: &RecipeFormatter,
    mut body: Value,
    defaults: &HashMap<String, Value>,
    encode: bool,
) -> Result<RecipeRender, String> {
    pre_normalize(&mut body, defaults)?;
    body["model"] = Value::String(formatter.model.name().into());
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
        Some(tokenizer) if encode => {
            Some(tokenizer.encode_ids(&rendered.prompt, &rendered.special_ranges)?)
        }
        _ => None,
    };
    let inference = &converted.inference_options;
    Ok(RecipeRender {
        token_ids,
        temperature: inference.temperature,
        top_p: inference.top_p,
        max_tokens: inference.max_tokens,
        thinking_mode: converted.conversation.thinking_mode,
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
        assert!(request.get("chat_template_kwargs").is_none());
    }

    #[test]
    fn continuation_and_object_arguments() {
        let mut request = body(json!([
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_0", "type": "function", "function": {"name": "get_weather", "arguments": {"location": "Paris"}}}
            ]},
            {"role": "tool", "tool_call_id": "call_0", "content": "sunny"},
            {"role": "assistant", "content": "It is"}
        ]));
        request["continue_final_message"] = json!(true);
        pre_normalize(&mut request, &HashMap::new()).unwrap();
        let call = &request["messages"][1]["tool_calls"][0];
        assert_eq!(call["function"]["arguments"], "{\"location\":\"Paris\"}");
        assert_eq!(request["messages"][3]["prefix"], true);
        assert!(request.get("continue_final_message").is_none());
    }

    #[test]
    fn efforts_pass_through_and_ultra_is_not_an_sglang_tier() {
        let mut request = body(json!([{"role": "user", "content": "hi"}]));
        request["reasoning_effort"] = json!("high");
        request["chat_template_kwargs"] = json!({"reasoning_effort": 30});
        pre_normalize(&mut request, &HashMap::new()).unwrap();
        // The kwargs effort replaces the request's verbatim; recipe rejects the number.
        assert_eq!(request["reasoning_effort"], 30);
        let mut request = body(json!([{"role": "user", "content": "hi"}]));
        request["reasoning_effort"] = json!("ultra");
        assert!(pre_normalize(&mut request, &HashMap::new()).is_err());
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
            false,
        )
        .unwrap();
        assert_eq!(
            render.prompt,
            "<｜begin▁of▁sentence｜><｜User｜><｜User｜><｜Assistant｜></think>"
        );
        // Template tokens are special; the user's spelling is plain text.
        let special = render
            .segments
            .iter()
            .filter(|segment| segment.allow_special);
        assert_eq!(special.count(), 4);
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
            false,
        )
        .unwrap();
        // SGLang's grammar enforces the choice; the prompt does not pre-open the block.
        assert!(render.prompt.ends_with("<｜Assistant｜></think>"));
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
    fn a_null_kwargs_toggle_turns_thinking_off_as_serving_chat_does() {
        let mut request = body(json!([{"role": "user", "content": "hi"}]));
        request["chat_template_kwargs"] = json!({"thinking": null});
        pre_normalize(&mut request, &HashMap::new()).unwrap();
        assert_eq!(request["thinking"], json!({"type": "disabled"}));
    }

    #[test]
    fn task_is_rejected_explicitly() {
        let mut request = body(json!([{"role": "user", "content": "hi"}]));
        request["task"] = json!("title");
        assert!(
            render(
                &RecipeFormatter::new(RecipeModel::V41),
                request,
                &HashMap::new(),
                false
            )
            .unwrap_err()
            .contains("task is not supported")
        );
    }

    #[test]
    fn detects_the_checkpoint_family() {
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
