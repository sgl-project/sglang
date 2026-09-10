//! Native (non-Jinja) chat encoders, for models whose prompt the engine builds
//! in code instead of from a `chat_template`.
//!
//! Python parity: `chat_encoding.resolve_chat_encoding_spec` picks the encoder
//! from `config.json`'s architecture list, so it wins over any `chat_template`
//! (DeepSeek-V4 checkpoints ship none) and over `--chat-template`.
//!
//! The rendering is Dynamo's port of `encoding_dsv4.py`
//! (`dynamo_renderer::deepseek::v4`). What this module adds is what is SGLang's
//! rather than Dynamo's: the default thinking mode (Dynamo defaults V4 to
//! thinking, SGLang to chat), the reasoning-effort profile (Dynamo models only
//! the preview ladder), and the message preprocessing `serving_chat.py` runs
//! before the encoder. Byte parity is pinned by engine-generated fixtures in
//! `template_native_tests`.

use serde_json::{Map, Value};

use dynamo_protocols::types::CreateChatCompletionRequest;

use dynamo_renderer::deepseek::v4::{ThinkingMode, encode_messages_with_options, tokens};

use crate::utils::environ::{env_bool, env_str};

use super::template::TemplateError;

/// Verbatim `encoding_dsv4.REASONING_EFFORT_PROMPTS` entry: the preview
/// profile's `max` text, which is also the official profile's `high`.
const EFFORT_PREAMBLE_MAX: &str = "Reasoning Effort: Absolute maximum with no shortcuts permitted.\nYou MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\nExplicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";

/// Verbatim `encoding_dsv4.REASONING_EFFORT_OFFICIAL_MAX`. Dynamo's encoder
/// has no representation for this level, so this layer emits it.
const EFFORT_PREAMBLE_OFFICIAL_MAX: &str = "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\nYou MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\nDo not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n";

/// Mirrors `chat_encoding._DSV4_REASONING_EFFORT_ENCODER` and
/// `_MAX_DSV4_ENCODER_BYTES`.
const DSV4_CHECKPOINT_ENCODER: &str = "encoding/encoding_dsv4.py";
const MAX_DSV4_ENCODER_BYTES: u64 = 1 << 20;

/// Which reasoning-effort ladder the checkpoint's encoder implements: the
/// official one shifts preview's levels down by one, so the same request maps
/// to a different preamble on each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EffortProfile {
    Preview,
    Official,
}

impl EffortProfile {
    /// Python `render_message`: the preamble for `effort`, or `None` when the
    /// profile does not define that level. `serving_chat.py` drops an
    /// unaccepted effort, and both profile defaults render empty.
    fn preamble(self, effort: &str) -> Option<&'static str> {
        match (self, effort) {
            (Self::Preview, "high") | (Self::Official, "low") => Some(""),
            (Self::Preview, "max") | (Self::Official, "high") => Some(EFFORT_PREAMBLE_MAX),
            (Self::Official, "max") => Some(EFFORT_PREAMBLE_OFFICIAL_MAX),
            _ => None,
        }
    }
}

/// A model whose prompt is built in code. `dsv32`, `kimi_k3` and `inkling` are
/// the other specs Python resolves; Dynamo ships renderers for all three, but
/// with selection rules and defaults that differ from
/// `resolve_chat_encoding_spec`, so each needs its own parity work here.
#[derive(Debug, Clone, Copy)]
pub(crate) enum NativeEncoder {
    DeepSeekV4 {
        /// `SGLANG_DEFAULT_THINKING`, read once at load time; Python reads it
        /// per request, but it cannot change in between.
        default_thinking: bool,
        effort_profile: EffortProfile,
    },
}

impl NativeEncoder {
    /// The encoder for this model, or `None` to fall through to the template
    /// paths. `model_config` is the parsed `config.json`.
    pub(super) fn detect(model_config: &Value, model_path: Option<&str>) -> Option<Self> {
        // Python reads `architectures[0]` and substring-matches it, so an
        // unusual second entry cannot flip the decision.
        let arch = model_config
            .get("architectures")?
            .as_array()?
            .first()?
            .as_str()?;
        if !arch.contains("DeepseekV4") {
            return None;
        }
        let effort_profile = detect_effort_profile(model_path);
        tracing::info!(
            arch,
            ?effort_profile,
            "model ships no chat template; using the built-in DeepSeek-V4 encoder"
        );
        Some(Self::DeepSeekV4 {
            default_thinking: env_bool("SGLANG_DEFAULT_THINKING", false),
            effort_profile,
        })
    }

    /// Render an OpenAI chat request to the prompt string.
    ///
    /// The mode comes from `SGLANG_DEFAULT_THINKING` alone: the request type
    /// carries no `chat_template_kwargs`, which is where Python reads a
    /// per-request `thinking` flag from.
    pub(super) fn render_request(
        &self,
        request: &CreateChatCompletionRequest,
    ) -> Result<String, TemplateError> {
        let messages =
            serde_json::to_value(&request.messages).map_err(|error| TemplateError::Renderer {
                message: format!("serialize messages: {error}"),
            })?;
        let messages = messages.as_array().ok_or_else(|| TemplateError::Renderer {
            message: "messages did not serialize to an array".to_owned(),
        })?;
        let tools = request
            .tools
            .as_ref()
            .and_then(|tools| serde_json::to_value(tools).ok());
        // The typed enum serializes to the same lowercase names Python matches
        // against the profile's levels; a level the profile does not define
        // (`medium`, `xhigh`) is dropped, as `serving_chat.py` does.
        let reasoning_effort = request
            .reasoning_effort
            .as_ref()
            .and_then(|effort| serde_json::to_value(effort).ok())
            .and_then(|effort| effort.as_str().map(str::to_owned));
        self.render(messages, tools.as_ref(), None, reasoning_effort.as_deref())
    }

    /// Render `messages` (an OpenAI `messages` array) to the prompt string.
    pub(super) fn render(
        &self,
        messages: &[Value],
        tools: Option<&Value>,
        thinking: Option<bool>,
        reasoning_effort: Option<&str>,
    ) -> Result<String, TemplateError> {
        let Self::DeepSeekV4 {
            default_thinking,
            effort_profile,
        } = *self;
        let thinking = thinking.unwrap_or(default_thinking);

        // Python order: an unaccepted `reasoning_effort` falls back to
        // `SGLANG_DSV4_REASONING_EFFORT`, then to the profile default.
        // Chat mode emits no preamble at all.
        let preamble = if thinking {
            reasoning_effort
                .and_then(|effort| effort_profile.preamble(effort))
                .or_else(|| effort_profile.preamble(&env_str("SGLANG_DSV4_REASONING_EFFORT", "")))
                .unwrap_or("")
        } else {
            ""
        };

        let body = encode_messages_with_options(
            &prepare_messages(messages, tools),
            if thinking {
                ThinkingMode::Thinking
            } else {
                ThinkingMode::Chat
            },
            // BOS is prepended below, so the effort preamble can sit between
            // it and the first message, where Python puts it.
            false,
            // Python's `drop_thinking` default; the encoder itself disables it
            // when the conversation carries tools.
            true,
            // Effort is applied above, from the resolved profile; Dynamo's own
            // handling knows only the preview ladder.
            None,
        )
        .map_err(|error| TemplateError::Renderer {
            message: format!("{error:#}"),
        })?;

        let mut prompt = String::with_capacity(tokens::BOS.len() + preamble.len() + body.len());
        prompt.push_str(tokens::BOS);
        prompt.push_str(preamble);
        prompt.push_str(&body);
        Ok(prompt)
    }
}

/// Python `_detect_dsv4_reasoning_effort_profile`, matched textually rather
/// than through `ast`. Anything unreadable resolves to `preview`, the same
/// fallback Python uses when the encoder cannot be inspected.
fn detect_effort_profile(model_path: Option<&str>) -> EffortProfile {
    let Some(path) = model_path.and_then(resolve_checkpoint_encoder) else {
        return EffortProfile::Preview;
    };
    match std::fs::metadata(&path) {
        Ok(meta) if meta.len() <= MAX_DSV4_ENCODER_BYTES => {}
        _ => return EffortProfile::Preview,
    }
    let Ok(source) = std::fs::read_to_string(&path) else {
        return EffortProfile::Preview;
    };
    // Python asserts both conditions: default `low`, all three levels defined.
    let default_is_low = source.lines().any(|line| {
        let line = line.trim();
        line.starts_with("DEFAULT_REASONING_EFFORT") && line.contains("\"low\"")
    });
    let full_ladder = ["\"low\":", "\"high\":", "\"max\":"]
        .iter()
        .all(|key| source.contains(key));
    if default_is_low && full_ladder {
        EffortProfile::Official
    } else {
        EffortProfile::Preview
    }
}

/// Locate the checkpoint's shipped encoder: a local model dir, else the HF
/// cache. Python can download it; this side is offline-only, so an unfetched
/// `encoding/` directory falls back to `preview`.
fn resolve_checkpoint_encoder(model_path: &str) -> Option<std::path::PathBuf> {
    let local = std::path::Path::new(model_path).join(DSV4_CHECKPOINT_ENCODER);
    if local.is_file() {
        return Some(local);
    }
    crate::tokenizer_manager::tokenizer::resolve_model_file(
        model_path,
        None,
        DSV4_CHECKPOINT_ENCODER,
    )
    .map(std::path::PathBuf::from)
}

/// The preprocessing `serving_chat.py` applies before calling the encoder.
/// Both steps change the prompt bytes: the empty system message keeps the
/// encoder's index logic aligned, and tools ride on the first message.
fn prepare_messages(messages: &[Value], tools: Option<&Value>) -> Vec<Value> {
    let mut prepared: Vec<Value> = messages
        .iter()
        .map(|message| {
            let mut message = message.clone();
            // Python: `if msg.get("content") is None: msg["content"] = ""`.
            if let Some(object) = message.as_object_mut()
                && object.get("content").is_none_or(Value::is_null)
            {
                object.insert("content".into(), Value::String(String::new()));
            }
            message
        })
        .collect();

    let first_is_system = prepared
        .first()
        .and_then(|message| message.get("role"))
        .and_then(Value::as_str)
        == Some("system");
    if !first_is_system {
        prepared.insert(
            0,
            Value::Object(Map::from_iter([
                ("role".into(), Value::String("system".into())),
                ("content".into(), Value::String(String::new())),
            ])),
        );
    }

    if let Some(tools) = tools.and_then(Value::as_array).filter(|t| !t.is_empty())
        && let Some(first) = prepared.first_mut().and_then(Value::as_object_mut)
    {
        first.insert(
            "tools".into(),
            Value::Array(tools.iter().map(canonical_tool).collect()),
        );
    }
    prepared
}

/// Pydantic `Tool.model_dump()`: fixed field order, declared defaults filled
/// in, extras dropped. `serde_json` is built with `preserve_order` in this
/// graph, so insertion order is what the tool-schema block serializes to.
fn canonical_tool(tool: &Value) -> Value {
    let function = tool.get("function");
    let field = |name: &str| function.and_then(|f| f.get(name)).cloned();

    let mut canonical = Map::new();
    canonical.insert(
        "description".into(),
        field("description").unwrap_or(Value::Null),
    );
    canonical.insert("name".into(), field("name").unwrap_or(Value::Null));
    canonical.insert(
        "parameters".into(),
        field("parameters").unwrap_or(Value::Null),
    );
    canonical.insert(
        "strict".into(),
        field("strict").unwrap_or(Value::Bool(false)),
    );
    // Python pops `defer_loading` when None, so absent must stay absent.
    if let Some(defer_loading) = field("defer_loading").filter(|v| !v.is_null()) {
        canonical.insert("defer_loading".into(), defer_loading);
    }

    Value::Object(Map::from_iter([
        (
            "type".into(),
            tool.get("type")
                .cloned()
                .unwrap_or(Value::String("function".into())),
        ),
        ("function".into(), Value::Object(canonical)),
    ]))
}
