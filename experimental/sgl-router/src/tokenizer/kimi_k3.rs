// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Kimi-K3 XTML prompt encoder for cache-aware routing.
//!
//! Kimi-K3 ships no Jinja chat template; the prompt is built in Python by the
//! model repo's own reference encoder (`encoding_k3.build_chat_segments`, driven
//! by `tokenization_kimi.TikTokenTokenizer.apply_chat_template`). This reproduces
//! that encoder so the router's query tokens match the blocks the engine caches.
//!
//! Only the PROMPT is reproduced here. The vocabulary is [`super::kimi_vocab`]
//! over `baseten/kimi-k3-tokenizer`'s `tokenizer.json`. The model's own repo
//! publishes a raw tiktoken rank file instead, which upstream's tiktoken backend
//! CAN read — but only without the reference's input chunking, so it is not what
//! this encoder tokenizes against.
//!
//! Unlike [`super::dsv4`] — which mirrors an encoder that lives in this repo —
//! the authority here is the model repo. Defaults therefore come from
//! `apply_chat_template`'s own signature (`thinking=True`,
//! `add_generation_prompt=True`, and a `thinking_effort` that defaults to
//! `"max"`), not from an sglang server flag.
//!
//! # Segments, not a string
//!
//! The K3 encoder does not emit one prompt string. It emits a list of
//! `EncodeSegment(text, allow_special)`, and the tokenizer encodes each one
//! separately: structural markers with every special token recognized, and
//! everything that came from the client with specials DISABLED, so a literal
//! `<|open|>` typed by a user BPEs as ordinary bytes instead of becoming the
//! control token. Flattening to one string and tokenizing once would promote
//! those markers and diverge from the engine on exactly the requests where it
//! matters, so [`render_segments`] preserves the split and
//! [`super::TokenizerRegistry::encode_chat`] honors it per segment.
//!
//! # Scope
//!
//! Text content, both thinking and non-thinking mode, including tools, tool
//! calls, tool results, `tool_choice`, and `response_format`. Multimodal turns
//! render the text-only placeholder path (see [`IMAGE_PLACEHOLDER`]) — matching
//! the reference encoder called without `image_prompts`, but NOT what an engine
//! with real images renders, so image-carrying requests degrade to a partial
//! prefix match and must never have their ids forwarded as `input_ids`.

use anyhow::{bail, Context, Result};
use dynamo_tokenizers::{traits::Encoder, EncodeSegment};

use super::pyjson::{compact_json, deep_sort, py_json};
use super::Segment;

/// Opens a tag: `<|open|>` `name` (` attr="value"`)* `<|sep|>`.
const OPEN_TOKEN: &str = "<|open|>";
/// Opens a closing tag: `<|close|>` `name` `<|sep|>`.
const CLOSE_TOKEN: &str = "<|close|>";
/// Terminates a tag's attribute list.
const SEP_TOKEN: &str = "<|sep|>";
/// Terminates a top-level message.
const END_OF_MSG_TOKEN: &str = "<|end_of_msg|>";
/// Stand-in emitted for an image when no rendered image prompt is available.
///
/// The reference encoder substitutes
/// `<|media_begin|>image {w}x{h}<|media_content|><|media_pad|><|media_end|>`
/// per image, built from the image's pre-resize pixel dimensions. The router
/// does not fetch or decode images, so it emits this placeholder exactly as the
/// reference encoder does when called without `image_prompts`. The string is not
/// in the vocabulary, so it BPEs as ordinary text either way.
const IMAGE_PLACEHOLDER: &str = "<|kimi_image_placeholder|>";

/// Every structural marker this encoder emits as a control token.
///
/// These are NOT part of the base vocabulary: they exist only as `added_tokens`
/// in `tokenizer.json` (or `added_tokens_decoder` in a sibling
/// `tokenizer_config.json`, which [`super::kimi_vocab::KimiVocab::from_file`]
/// merges in). A vocabulary missing them may have no token at that id AT ALL —
/// the 256-slot reserved block was a property of tiktoken's loader, and a
/// `tokenizer.json` only has the ids its `added_tokens` list; Baseten's
/// conversion happens to carry `<|reserved_token_N|>` entries, but nothing
/// guarantees that. Either way, emitting `<|open|>` then BPEs it into several
/// ordinary tokens instead of resolving to its control id, silently, for every
/// tag in every prompt. [`markers_resolve`] is the guard; see its caller.
pub const CONTROL_MARKERS: [&str; 4] = [OPEN_TOKEN, CLOSE_TOKEN, SEP_TOKEN, END_OF_MSG_TOKEN];

/// Whether `tokenizer` can resolve every marker in [`CONTROL_MARKERS`].
///
/// A vocabulary that cannot is unusable for this encoder: it would produce a
/// token stream with no structural tokens at all, which the router would then
/// hand the engine as `input_ids`. Callers must decline to attach the encoder
/// rather than emit that.
///
/// Probed BEHAVIOURALLY rather than by name lookup: a registered marker encodes
/// to exactly one token, an unregistered one BPEs into several ordinary ones.
/// That is the property this encoder actually depends on, and unlike a
/// `special_id` lookup it holds for any backend.
///
/// Returns the reason rather than a bare `false`, because the two ways this fails
/// need different advice. "Not a registered added token" means the operator has
/// the wrong vocabulary; an encode ERROR means the vocabulary is broken in some
/// other way (a missing single-byte token, a pathological pre-tokenizer regex)
/// and the backend's own message is the only thing that says which. Collapsing
/// them sent an operator to inspect `added_tokens` that were already correct.
pub fn markers_resolve(tokenizer: &dyn Encoder) -> Result<()> {
    for marker in CONTROL_MARKERS {
        let ids = tokenizer
            .encode_segments(&[EncodeSegment::new(marker, true)])
            .with_context(|| format!("probe control marker {marker}"))?;
        let n = ids.token_ids().len();
        if n != 1 {
            bail!(
                "control marker {marker} encodes to {n} tokens rather than 1; \
                 it is not a registered added token in this vocabulary"
            );
        }
    }
    Ok(())
}

/// How much the model should think, mirroring `encoding_k3`'s `thinking_effort`.
///
/// Only these three values are accepted by the reference encoder (its
/// `_VALID_THINKING_EFFORTS`), even though the prompt text it emits also
/// mentions `medium` — the code is the contract, so `medium` is not a variant
/// here. Rendered only in thinking mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThinkingEffort {
    Low,
    High,
    Max,
}

impl ThinkingEffort {
    fn as_str(self) -> &'static str {
        match self {
            ThinkingEffort::Low => "low",
            ThinkingEffort::High => "high",
            ThinkingEffort::Max => "max",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        match s {
            "low" => Some(ThinkingEffort::Low),
            "high" => Some(ThinkingEffort::High),
            "max" => Some(ThinkingEffort::Max),
            _ => None,
        }
    }
}

/// A resolved `thinking_effort`, keeping "the client asked for something this
/// encoder cannot render" distinct from "no effort preamble".
///
/// Collapsing the two is a silent divergence: the reference encoder raises on an
/// unsupported effort, so rendering no preamble instead would hand the engine a
/// prompt the request never described.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestedEffort {
    /// Render this effort's preamble (thinking mode only).
    Valid(ThinkingEffort),
    /// Render no preamble — the reference encoder's `thinking_effort=None`.
    None,
    /// Present but unsupported; [`render_segments`] refuses to render.
    Invalid,
}

impl From<Option<ThinkingEffort>> for RequestedEffort {
    fn from(v: Option<ThinkingEffort>) -> Self {
        v.map_or(RequestedEffort::None, RequestedEffort::Valid)
    }
}

/// The request's `tool_choice`, for the two values that alter the prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolChoice {
    /// Neither `required` nor `none` — including `auto` and a named-function
    /// object, neither of which the reference encoder renders.
    Unset,
    Required,
    None,
}

/// How to render a K3 chat request.
///
/// `response_format` is kept as the raw request value rather than pre-resolved
/// because the reference encoder reads it twice, in two different ways: once to
/// extract a `json_schema` payload and once for its `type` discriminator.
#[derive(Clone, Debug)]
pub struct RenderOpts {
    pub thinking: bool,
    pub thinking_effort: RequestedEffort,
    pub tool_choice: ToolChoice,
    /// The request's `response_format` object, when present.
    pub response_format: Option<serde_json::Value>,
    pub add_generation_prompt: bool,
}

impl Default for RenderOpts {
    /// `apply_chat_template`'s own defaults: thinking on, effort `max`,
    /// generation prompt appended.
    fn default() -> Self {
        RenderOpts {
            thinking: true,
            thinking_effort: RequestedEffort::Valid(ThinkingEffort::Max),
            tool_choice: ToolChoice::Unset,
            response_format: None,
            add_generation_prompt: true,
        }
    }
}

/// Resolve render options for a request, mirroring what a serving stack passes
/// through to `apply_chat_template`.
///
/// `chat_template_kwargs.thinking` wins over the router default, matching how
/// every sglang chat path threads per-request template kwargs. Below it sits
/// Moonshot's own spelling, the `thinking` object (`{type, keep, effort}`),
/// which the engine maps onto the same template kwargs -- a request carrying
/// `thinking: {"type": "disabled"}` renders the response channel, not the think
/// channel, and rendering it as thinking costs 67 prompt tokens the caller never
/// asked for. Effort follows the matching precedence
/// (`chat_template_kwargs.thinking_effort` > `thinking.effort` > top-level
/// `thinking_effort` > default). `tool_choice` and `response_format` are read
/// from their standard OpenAI top-level positions.
///
/// The router is a separate process from the engine, so a default the request
/// does not carry cannot be observed; `SGLANG_ROUTER_K3_DEFAULT_THINKING` /
/// `SGLANG_ROUTER_K3_THINKING_EFFORT` (read once) MUST be set to match the
/// engine's, or cache-aware routing degrades. Degrades, never breaks: the engine
/// re-tokenizes for correctness, and `input_ids` forwarding is separately gated.
pub fn resolve_render_opts(request: &serde_json::Value) -> RenderOpts {
    let ctk = request.get("chat_template_kwargs");

    // Only an object is Moonshot's thinking control; `thinking: true` is the
    // template-kwarg spelling and is handled by the ctk branch above it.
    let thinking_config = request.get("thinking").filter(|v| v.is_object());

    let thinking = ctk
        .and_then(|k| k.get("thinking"))
        .map(json_truthy)
        .or_else(|| {
            thinking_config
                .and_then(|t| t.get("type"))
                .and_then(|v| v.as_str())
                // "adaptive" is Anthropic's third variant; only "disabled" is off.
                .map(|kind| kind != "disabled")
        })
        .unwrap_or_else(default_thinking);

    // A per-request effort the reference encoder would REJECT must not silently
    // collapse to "no preamble" — that renders a different prompt and still
    // reports it as engine-equivalent. `medium` is the live case: the encoder's
    // own prompt text advertises it while `_VALID_THINKING_EFFORTS` rejects it,
    // so clients will send it. `Invalid` propagates to `render_segments`, which
    // fails the render and drops the request to raw-text routing.
    let requested = ctk
        .and_then(|k| k.get("thinking_effort"))
        .or_else(|| thinking_config.and_then(|t| t.get("effort")))
        .or_else(|| request.get("thinking_effort"))
        .filter(|v| !v.is_null());
    let thinking_effort = match requested {
        None => (*default_thinking_effort()).into(),
        Some(v) => v.as_str().map_or(RequestedEffort::Invalid, |s| {
            ThinkingEffort::parse(s).map_or(RequestedEffort::Invalid, RequestedEffort::Valid)
        }),
    };

    let tool_choice = match request.get("tool_choice").and_then(|v| v.as_str()) {
        Some("required") => ToolChoice::Required,
        Some("none") => ToolChoice::None,
        _ => ToolChoice::Unset,
    };

    RenderOpts {
        thinking,
        thinking_effort,
        tool_choice,
        response_format: request.get("response_format").cloned(),
        add_generation_prompt: true,
    }
}

/// Python-truthiness of a JSON value, matching [`super::dsv4`]'s treatment of
/// the same `chat_template_kwargs.thinking` field.
fn json_truthy(v: &serde_json::Value) -> bool {
    match v {
        serde_json::Value::Bool(b) => *b,
        serde_json::Value::String(s) => !s.is_empty(),
        serde_json::Value::Number(n) => n.as_f64().is_none_or(|f| f != 0.0),
        serde_json::Value::Array(a) => !a.is_empty(),
        serde_json::Value::Object(o) => !o.is_empty(),
        serde_json::Value::Null => false,
    }
}

/// The router's K3 render defaults, resolved once from env.
///
/// Deliberately does NOT log: this is reached on the first request of ANY model
/// (the ingress resolves every encoder's options together), so logging here made
/// a DeepSeek-only router announce Kimi settings. [`log_defaults`] is called
/// from encoder attach instead, where the model really is K3.
///
/// The built-in defaults are the reference encoder's own
/// (`thinking=True`, `thinking_effort="max"`), so an unconfigured router matches
/// an unconfigured engine.
fn router_defaults() -> &'static (bool, Option<ThinkingEffort>) {
    static V: std::sync::OnceLock<(bool, Option<ThinkingEffort>)> = std::sync::OnceLock::new();
    V.get_or_init(|| {
        let thinking = match std::env::var("SGLANG_ROUTER_K3_DEFAULT_THINKING").ok() {
            Some(s) if !s.is_empty() => parse_env_bool(&s).unwrap_or_else(|| {
                tracing::warn!(value = %s,
                    "SGLANG_ROUTER_K3_DEFAULT_THINKING is not a recognized boolean \
                     (true/1/yes/y | false/0/no/n); using the K3 default (true)");
                true
            }),
            _ => true,
        };
        let effort = match std::env::var("SGLANG_ROUTER_K3_THINKING_EFFORT").ok() {
            Some(s) if !s.is_empty() => ThinkingEffort::parse(&s).or_else(|| {
                tracing::warn!(value = %s,
                    "SGLANG_ROUTER_K3_THINKING_EFFORT is not one of low/high/max; \
                     rendering with no thinking-effort preamble");
                None
            }),
            _ => Some(ThinkingEffort::Max),
        };
        (thinking, effort)
    })
}

/// Parse a boolean env value the way the engine's `EnvBool` does.
fn parse_env_bool(value: &str) -> Option<bool> {
    match value.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "y" => Some(true),
        "false" | "0" | "no" | "n" => Some(false),
        _ => None,
    }
}

/// Log the resolved K3 defaults, once, at encoder-attach time.
///
/// An operator has to be able to confirm these match the engine's chat-template
/// defaults — a mismatch silently degrades cache-aware routing — so this is INFO
/// and not DEBUG. Called only when the K3 encoder actually attaches.
pub fn log_defaults() {
    let (thinking, effort) = router_defaults();
    tracing::info!(
        default_thinking = thinking,
        default_thinking_effort = effort.map_or("(none)", ThinkingEffort::as_str),
        "Kimi-K3 router render defaults resolved; must match the engine's chat-template \
         defaults for cache-aware routing to match"
    );
}

fn default_thinking() -> bool {
    router_defaults().0
}

fn default_thinking_effort() -> &'static Option<ThinkingEffort> {
    &router_defaults().1
}

/// Render `messages` (+ the request's top-level `tools`) into K3 XTML segments.
///
/// Mirrors `encoding_k3.build_chat_segments`. `messages` is the request's
/// `messages` array; `tools` is the request's top-level `tools` array (OpenAI
/// format), or `None`.
///
/// Returns `Err` — the caller then falls back to raw prompt-text routing —
/// whenever the reference encoder would itself raise: a tool result whose name
/// cannot be resolved, a tool call with no function name, or `arguments` that
/// are neither an object, a JSON-object string, nor absent. Rendering something
/// plausible instead would produce ids that silently disagree with the engine,
/// which is worse than routing by raw text.
pub fn render_segments(
    messages: &serde_json::Value,
    tools: Option<&serde_json::Value>,
    opts: &RenderOpts,
) -> Result<Vec<Segment>> {
    let raw = messages.as_array().map(Vec::as_slice).unwrap_or(&[]);
    let ordered = sort_tool_results_by_call_order(raw);

    let mut out = Segments::default();

    // An empty `tools` array is falsy in Python, so it renders nothing.
    let tool_list = tools
        .and_then(|t| t.as_array())
        .filter(|arr| !arr.is_empty());
    if let Some(list) = tool_list {
        render_tool_declare(&mut out, &serde_json::Value::Array(list.clone()), false);
    }

    if opts.thinking {
        if opts.thinking_effort == RequestedEffort::Invalid {
            bail!("unsupported thinking_effort (the reference encoder accepts only low/high/max)");
        }
        if let RequestedEffort::Valid(effort) = opts.thinking_effort {
            internal_system_message(
                &mut out,
                "thinking-effort",
                &format!(
                    "`thinking_effort` guides on how much to think in your thinking channel \
                     (not including the response channel), supported values include `low`, \
                     `medium`, `high`, and `max`.\nNow the system is invoked with \
                     `thinking_effort={}`.",
                    effort.as_str()
                ),
            );
        }
    }

    // `tool_calls` of the most recent assistant turn, used to name a tool
    // result that carries no name of its own; `tool_index` is that result's
    // 1-based position within the turn. Both reset on every assistant message.
    let mut last_tool_calls: Option<&serde_json::Value> = None;
    let mut tool_index: usize = 0;

    for message in &ordered {
        let Some(obj) = message.as_object() else {
            continue;
        };
        let role = obj.get("role").and_then(|r| r.as_str()).unwrap_or_default();
        match role {
            "user" => {
                open_tag(&mut out, "message", &named_attrs("user", message));
                render_content(&mut out, message.get("content"))?;
                close_message(&mut out);
            }
            // A system message carrying `tools` is the lazy-loading extension
            // path, rendered as a *dynamic* declaration instead of a turn.
            "system" if truthy_field(message, "tools") => {
                let list = message.get("tools").expect("checked by truthy_field");
                render_tool_declare(&mut out, list, true);
            }
            "system" => {
                open_tag(&mut out, "message", &named_attrs("system", message));
                render_content(&mut out, message.get("content"))?;
                close_message(&mut out);
            }
            "tool" => {
                tool_index += 1;
                let name = resolve_tool_result_name(message, last_tool_calls, tool_index)?;
                open_tag(
                    &mut out,
                    "message",
                    &[
                        ("role".into(), "tool".into()),
                        ("tool".into(), name),
                        ("index".into(), tool_index.to_string()),
                    ],
                );
                render_content(&mut out, message.get("content"))?;
                close_message(&mut out);
            }
            "assistant" => {
                last_tool_calls = message.get("tool_calls");
                tool_index = 0;
                open_tag(&mut out, "message", &named_attrs("assistant", message));
                render_assistant_body(&mut out, message, opts.thinking)?;
                close_message(&mut out);
            }
            // Any other role renders nothing at all, matching the reference
            // encoder's if/elif chain with no else.
            _ => {}
        }
    }

    match opts.tool_choice {
        ToolChoice::Required => internal_system_message(
            &mut out,
            "tool-choice",
            "The system is invoked with `tool_choice=required`.\n\
             You MUST call tools in the next message.",
        ),
        ToolChoice::None => internal_system_message(
            &mut out,
            "tool-choice",
            "The system is invoked with `tool_choice=none`.\n\
             You MUST NOT call any tools in the next message.",
        ),
        ToolChoice::Unset => {}
    }

    render_response_format(&mut out, opts.response_format.as_ref());

    if opts.add_generation_prompt {
        open_tag(&mut out, "message", &[("role".into(), "assistant".into())]);
        open_tag(
            &mut out,
            if opts.thinking { "think" } else { "response" },
            &[],
        );
    }

    Ok(out.into_inner())
}

/// Accumulates segments, coalescing nothing — the split between adjacent
/// same-mode segments is invisible to tiktoken only when they are encoded
/// together, and they are not, so the boundaries are preserved exactly as the
/// reference encoder produces them.
#[derive(Default)]
struct Segments(Vec<Segment>);

impl Segments {
    /// A structural marker: encoded with special tokens recognized.
    ///
    /// `&'static str` (via [`Segment::marker`]) is what makes it impossible to
    /// mark client text as structure — every marker is a `const` in this module.
    fn control(&mut self, text: &'static str) {
        if !text.is_empty() {
            self.0.push(Segment::marker(text));
        }
    }

    /// Client-provided or encoder-formatted text: encoded with special tokens
    /// DISABLED, so nothing in it can become a control token.
    ///
    /// This is also `encoding_k3._append_text`'s no-split branch. That helper
    /// splits image placeholders into their own allow-special segments only when
    /// the caller supplied rendered image prompts; the router never does (it
    /// decodes no images), so the split can never apply here.
    fn text(&mut self, text: impl Into<String>) {
        // Empties are dropped, matching `encoding_k3._segment`.
        let text = text.into();
        if !text.is_empty() {
            self.0.push(Segment::client_text(text));
        }
    }

    fn into_inner(self) -> Vec<Segment> {
        self.0
    }
}

/// `<|open|>` tag ( ` key="value"` )* `<|sep|>`
fn open_tag(out: &mut Segments, tag: &str, attrs: &[(String, String)]) {
    out.control(OPEN_TOKEN);
    out.text(tag.to_string());
    for (key, value) in attrs {
        out.text(format!(" {key}"));
        out.text("=\"");
        out.text(escape_attr_value(value));
        out.text("\"");
    }
    out.control(SEP_TOKEN);
}

/// `<|close|>` tag `<|sep|>`
fn close_tag(out: &mut Segments, tag: &str) {
    out.control(CLOSE_TOKEN);
    out.text(tag.to_string());
    out.control(SEP_TOKEN);
}

fn close_message(out: &mut Segments) {
    close_tag(out, "message");
    out.control(END_OF_MSG_TOKEN);
}

/// `&` then `"` — in that order, and nothing else. `<` and `>` are deliberately
/// NOT escaped (`encoding_k3._escape_attr_value`); adding them would change the
/// bytes for any tool name or key containing them.
fn escape_attr_value(value: &str) -> String {
    value.replace('&', "&amp;").replace('"', "&quot;")
}

/// `[("role", role)]`, plus `("name", …)` when the message carries a truthy one.
fn named_attrs(role: &str, message: &serde_json::Value) -> Vec<(String, String)> {
    let mut attrs = vec![("role".to_string(), role.to_string())];
    if truthy_field(message, "name") {
        if let Some(name) = message.get("name") {
            attrs.push(("name".to_string(), py_str(name)));
        }
    }
    attrs
}

/// Python truthiness of `message[key]` (absent counts as falsy).
fn truthy_field(message: &serde_json::Value, key: &str) -> bool {
    message.get(key).is_some_and(json_truthy)
}

/// A `role="system"` message with a `type`, used for the encoder's own
/// injected instructions. The body is stripped, matching
/// `encoding_k3._internal_system_message`.
fn internal_system_message(out: &mut Segments, message_type: &str, body: &str) {
    open_tag(
        out,
        "message",
        &[
            ("role".into(), "system".into()),
            ("type".into(), message_type.into()),
        ],
    );
    out.text(body.trim().to_string());
    close_message(out);
}

/// The tool declaration block. `dynamic` selects the lazy-loading wording used
/// for tools that arrive mid-conversation on a system message.
///
/// The schema JSON is COMPACT (`separators=(",", ":")`), unlike the
/// default-separator JSON used for argument values — see [`super::pyjson`].
fn render_tool_declare(out: &mut Segments, tools: &serde_json::Value, dynamic: bool) {
    // Sorted HERE rather than by the caller: the canonicalization is a
    // precondition of the rendering, and a signature cannot state it. `deep_sort`
    // is idempotent, so a caller that already sorted loses nothing.
    let schemas = compact_json(&deep_sort(tools));
    let body = if dynamic {
        format!(
            "## New Tools Available\n\
             The system dynamically extends the toolset via lazy-loading.\n\
             You have access to all existing and extended tools.\n\
             Here are the specs for the extended tools.\n\n\
             ```json\n{schemas}\n```"
        )
    } else {
        format!(
            "# Tools\n\
             Here are the available tools, described in JSONSchema.\n\n\
             ```json\n{schemas}\n```"
        )
    };
    open_tag(
        out,
        "message",
        &[
            ("role".into(), "system".into()),
            ("type".into(), "tool-declare".into()),
        ],
    );
    // NOT stripped — unlike `internal_system_message`.
    out.text(body);
    close_message(out);
}

/// The `response_format` instruction block, keyed on the format's `type`.
///
/// A bare string `response_format` is accepted as its own type, mirroring the
/// reference encoder's `_get_value(rf, "type", rf) if isinstance(rf, dict) else rf`.
fn render_response_format(out: &mut Segments, response_format: Option<&serde_json::Value>) {
    let Some(rf) = response_format else {
        return;
    };
    let rf_type = match rf {
        serde_json::Value::Object(map) => map.get("type").and_then(|t| t.as_str()),
        serde_json::Value::String(s) => Some(s.as_str()),
        _ => None,
    };
    match rf_type {
        Some("json_object") => internal_system_message(
            out,
            "response-format",
            "The system is invoked with `response_format=json_object`.\n\
             Your response must be raw JSON data without markdown code blocks (```json) \
             or any additional formatting.",
        ),
        Some("json_schema") => {
            // An absent schema renders the literal `null` the reference encoder
            // would emit from `json.dumps(None)`, rather than skipping the block.
            let schema = extract_response_schema(rf)
                .map(|s| deep_sort(&s))
                .unwrap_or(serde_json::Value::Null);
            internal_system_message(
                out,
                "response-format",
                &format!(
                    "The system is invoked with `response_format=json_schema`.\n\
                     Your response must be raw JSON data without markdown code blocks (```json) \
                     or any additional formatting.\n\
                     The JSON data must match the following schema:\n```json\n{}\n```",
                    compact_json(&schema)
                ),
            );
        }
        _ => {}
    }
}

/// Dig the schema out of a `response_format`, mirroring
/// `encoding_k3.extract_response_schema`: `json_schema.schema`, else
/// `json_schema.json_schema`, else the `json_schema` object itself.
fn extract_response_schema(response_format: &serde_json::Value) -> Option<serde_json::Value> {
    let js = response_format.get("json_schema")?;
    if let Some(map) = js.as_object() {
        return Some(
            map.get("schema")
                .or_else(|| map.get("json_schema"))
                .unwrap_or(js)
                .clone(),
        );
    }
    Some(js.clone())
}

/// A message's `content`: a bare string, or a list of typed parts where image
/// parts become the image placeholder.
fn render_content(out: &mut Segments, content: Option<&serde_json::Value>) -> Result<()> {
    match content {
        None | Some(serde_json::Value::Null) => Ok(()),
        Some(serde_json::Value::String(s)) => {
            out.text(s.clone());
            Ok(())
        }
        Some(serde_json::Value::Array(parts)) => {
            for part in parts {
                let part_type = part
                    .get("type")
                    // A part with NO `type` is malformed: the reference indexes
                    // `part["type"]` and raises `KeyError`, so erroring here keeps
                    // the request on raw-text routing rather than inventing a
                    // render the engine will never produce. A `type` that is
                    // present but not a string is different — Python's `in` test
                    // just fails, and the part falls through to its `text`.
                    .ok_or_else(|| anyhow::anyhow!("content part has no `type` field"))?
                    .as_str()
                    .unwrap_or_default();
                if matches!(part_type, "image" | "image_url") {
                    out.control(IMAGE_PLACEHOLDER);
                } else {
                    let Some(text) = part.get("text") else {
                        bail!("content part of type {part_type:?} has no `text` field");
                    };
                    out.text(py_str(text));
                }
            }
            Ok(())
        }
        // A number/bool/object `content` is malformed: the reference encoder
        // reaches `for part in content` and raises. Erroring here degrades the
        // request to raw-text routing, which is right — rendering some
        // plausible stringification would produce ids the engine never sees.
        Some(other) => bail!(
            "message content must be a string or a list of parts, got {}",
            match other {
                serde_json::Value::Object(_) => "an object",
                serde_json::Value::Bool(_) => "a boolean",
                _ => "a number",
            }
        ),
    }
}

/// The inside of an assistant `<message>`: the think channel (thinking mode
/// only), the response channel, then any tool calls.
fn render_assistant_body(
    out: &mut Segments,
    message: &serde_json::Value,
    thinking: bool,
) -> Result<()> {
    if thinking {
        // Structural: the tags are emitted even with nothing to put in them.
        // `reasoning_content or reasoning` is Python's `or`, so an EMPTY
        // reasoning_content falls through to `reasoning`.
        let reasoning = message
            .get("reasoning_content")
            .filter(|v| json_truthy(v))
            .or_else(|| message.get("reasoning"));
        open_tag(out, "think", &[]);
        if let Some(r) = reasoning {
            let text = py_str(r);
            if !text.trim().is_empty() {
                out.text(text);
            }
        }
        close_tag(out, "think");
    }

    open_tag(out, "response", &[]);
    render_content(out, message.get("content"))?;
    close_tag(out, "response");

    if truthy_field(message, "tool_calls") {
        let calls = message
            .get("tool_calls")
            .and_then(|c| c.as_array())
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        open_tag(out, "tools", &[]);
        for (i, call) in calls.iter().enumerate() {
            render_tool_call(out, call, i + 1)?;
        }
        close_tag(out, "tools");
    }
    Ok(())
}

/// One `<call tool=… index=…>` and its arguments.
fn render_tool_call(out: &mut Segments, call: &serde_json::Value, index: usize) -> Result<()> {
    // `tool_call.get("function", tool_call)` — the OpenAI nesting is optional.
    let func = call.get("function").unwrap_or(call);
    let Some(name) = func.get("name").and_then(|n| n.as_str()) else {
        bail!("tool_call at index {index} has no `function.name`");
    };
    open_tag(
        out,
        "call",
        &[
            ("tool".into(), name.to_string()),
            ("index".into(), index.to_string()),
        ],
    );

    match normalize_tool_arguments(func.get("arguments"))? {
        // A string that did not parse as JSON is passed through verbatim inside
        // a `<json>` block rather than being dropped or guessed at.
        ToolArguments::RawJsonBlock(raw) => {
            open_tag(out, "json", &[("type".into(), "object".into())]);
            out.text(raw);
            close_tag(out, "json");
        }
        ToolArguments::Object(map) => {
            for (key, value) in map {
                open_tag(
                    out,
                    "argument",
                    &[
                        ("key".into(), key),
                        ("type".into(), xtml_type(&value).into()),
                    ],
                );
                out.text(xtml_value(&value));
                close_tag(out, "argument");
            }
        }
    }

    close_tag(out, "call");
    Ok(())
}

/// Normalized `tool_calls[].function.arguments`.
enum ToolArguments {
    Object(Vec<(String, serde_json::Value)>),
    RawJsonBlock(String),
}

/// Mirror `encoding_k3.normalize_tool_arguments`.
///
/// Absent / empty-string arguments become an empty object; an object passes
/// through; a string is parsed as JSON and must yield an OBJECT. A string that
/// fails to parse is preserved verbatim as a `<json>` block — but a string that
/// parses to a non-object (an array, a bare number) is an error, exactly as the
/// reference encoder raises, because there is no correct rendering for it.
fn normalize_tool_arguments(arguments: Option<&serde_json::Value>) -> Result<ToolArguments> {
    let empty = || ToolArguments::Object(Vec::new());
    match arguments {
        None | Some(serde_json::Value::Null) => Ok(empty()),
        Some(serde_json::Value::Object(map)) => Ok(ToolArguments::Object(
            map.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        )),
        Some(serde_json::Value::String(s)) => {
            if s.trim().is_empty() {
                return Ok(empty());
            }
            match serde_json::from_str::<serde_json::Value>(s) {
                Ok(serde_json::Value::Object(map)) => {
                    Ok(ToolArguments::Object(map.into_iter().collect()))
                }
                Ok(_) => bail!("Kimi K3 tool call arguments must be a JSON object"),
                Err(_) => Ok(ToolArguments::RawJsonBlock(s.clone())),
            }
        }
        Some(_) => bail!("Kimi K3 tool call arguments must be an object or a JSON object string"),
    }
}

/// The `type` attribute of an `<argument>` (`encoding_k3._xtml_type`). Booleans
/// are classified BEFORE numbers, as in the Python.
fn xtml_type(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Null => "null",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Object(_) => "object",
        serde_json::Value::Array(_) => "array",
    }
}

/// The body of an `<argument>` (`encoding_k3._xtml_value`): a string goes in
/// raw, everything else as `json.dumps(v, ensure_ascii=False)` — DEFAULT
/// separators, unlike the compact tool-declare JSON.
fn xtml_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        other => py_json(other),
    }
}

/// Python's `str()` for the JSON scalars that reach a text position.
///
/// Containers CAN reach here — a message `name`, a tool `name`, a
/// `tool_call_id`, a call `id`, or a part's `text` is whatever the client sent,
/// and only the `content` path is structurally constrained. For those the
/// fallback is not faithful: Python's `str()` on a dict is a `repr`
/// (`{'a': 1}`), while `py_json` emits JSON (`{"a": 1}`) — measured, not assumed:
/// it is the ONE render divergence left after diffing this encoder against the
/// reference over every scenario in `testdata/gen_kimi_k3_multimodal_cases.py`
/// plus the malformed-input battery. Scalars are faithful, including `True` /
/// `False` / `None` and a LIST (Python's `str([1, 2])` and `json.dumps` with
/// default separators agree). A request that puts a
/// container in one of those positions therefore renders differently here than
/// on the engine and loses its exact cache match — a routing degradation, not a
/// wrong answer, since the engine re-tokenizes for correctness. Left as a known
/// gap rather than reimplementing Python's `repr`, which would need its own
/// parity fixtures for a shape no observed client sends.
fn py_str(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Bool(true) => "True".to_string(),
        serde_json::Value::Bool(false) => "False".to_string(),
        serde_json::Value::Null => "None".to_string(),
        other => py_json(other),
    }
}

/// Resolve the `tool` attribute for a tool-result message: its own
/// `tool`/`name`, else the matching call in the preceding assistant turn.
fn resolve_tool_result_name(
    message: &serde_json::Value,
    last_tool_calls: Option<&serde_json::Value>,
    tool_index: usize,
) -> Result<String> {
    // `message.get("tool", message.get("name"))` — an explicit null `tool` does
    // NOT fall through to `name`, so key presence is what matters here.
    let own = if message.get("tool").is_some() {
        message.get("tool")
    } else {
        message.get("name")
    };
    if let Some(v) = own.filter(|v| !v.is_null()) {
        return Ok(py_str(v));
    }

    if let Some(calls) = last_tool_calls.and_then(|c| c.as_array()) {
        if tool_index <= calls.len() {
            let call = &calls[tool_index - 1];
            let func = call.get("function").unwrap_or(call);
            if let Some(name) = func.get("name").and_then(|n| n.as_str()) {
                return Ok(name.to_string());
            }
        }
    }
    bail!(
        "Kimi K3 tool messages need a resolvable tool name: carry `tool`/`name`, \
         or match a preceding assistant tool_call by order"
    )
}

/// Re-sort each run of consecutive `tool` messages into the order of the
/// preceding assistant turn's `tool_calls`, mirroring
/// `encoding_k3.normalize_xtml_tool_result_messages`.
///
/// Serving stacks usually deliver results already in call order; a direct client
/// may not. Matching is by opaque `tool_call_id` == `tool_calls[].id`. The
/// matched call is authoritative, so a matched message's `tool`/`name` is
/// REWRITTEN to that call's function name — otherwise a stale name would
/// contradict the reordered index. A run that cannot be fully matched is left
/// exactly as it arrived.
///
/// Returns owned values because matched messages are rewritten; the caller's
/// input is never mutated.
fn sort_tool_results_by_call_order(messages: &[serde_json::Value]) -> Vec<serde_json::Value> {
    let mut out: Vec<serde_json::Value> = Vec::with_capacity(messages.len());
    // tool_call_id → (1-based position, function name)
    let mut index: Vec<(String, usize, Option<String>)> = Vec::new();
    let mut i = 0;

    while i < messages.len() {
        let message = &messages[i];
        let role = message.get("role").and_then(|r| r.as_str());

        if message.is_object() && role == Some("assistant") {
            index = tool_call_id_index(message.get("tool_calls"));
            out.push(message.clone());
            i += 1;
            continue;
        }
        if !message.is_object() || role != Some("tool") {
            out.push(message.clone());
            i += 1;
            continue;
        }

        // (position, arrival offset, message, resolved name)
        let mut run: Vec<(Option<usize>, usize, &serde_json::Value, Option<String>)> = Vec::new();
        let mut unresolved = false;
        let mut offset = 0;
        while i < messages.len()
            && messages[i].is_object()
            && messages[i].get("role").and_then(|r| r.as_str()) == Some("tool")
        {
            let m = &messages[i];
            let call_id = m
                .get("tool_call_id")
                .or_else(|| m.get("id"))
                .filter(|v| !v.is_null())
                .map(py_str);
            let matched = call_id
                .as_ref()
                .and_then(|id| index.iter().find(|(k, _, _)| k == id))
                .map(|(_, pos, name)| (*pos, name.clone()));
            match matched {
                Some((pos, name)) => run.push((Some(pos), offset, m, name)),
                None => {
                    unresolved = true;
                    run.push((None, offset, m, None));
                }
            }
            offset += 1;
            i += 1;
        }

        if unresolved {
            out.extend(run.into_iter().map(|(_, _, m, _)| m.clone()));
            continue;
        }
        run.sort_by_key(|(pos, offset, _, _)| (pos.unwrap_or(0), *offset));
        for (_, _, m, name) in run {
            match name {
                None => out.push(m.clone()),
                Some(name) => {
                    let mut resolved = m.clone();
                    if let Some(map) = resolved.as_object_mut() {
                        map.insert("tool".into(), serde_json::Value::String(name.clone()));
                        if map.contains_key("name") {
                            map.insert("name".into(), serde_json::Value::String(name));
                        }
                    }
                    out.push(resolved);
                }
            }
        }
    }
    out
}

/// Map assistant `tool_calls[].id` → (1-based position, function name).
///
/// Every entry advances the position, even a non-object or id-less one, so the
/// position matches the `<call index=…>` the renderer will emit. Duplicate ids
/// keep their first occurrence.
fn tool_call_id_index(
    tool_calls: Option<&serde_json::Value>,
) -> Vec<(String, usize, Option<String>)> {
    let mut index = Vec::new();
    let Some(calls) = tool_calls.and_then(|c| c.as_array()) else {
        return index;
    };
    for (i, call) in calls.iter().enumerate() {
        let position = i + 1;
        let Some(obj) = call.as_object() else {
            continue;
        };
        let Some(id) = obj.get("id").filter(|v| !v.is_null()) else {
            continue;
        };
        let key = py_str(id);
        if index
            .iter()
            .any(|(k, _, _): &(String, usize, Option<String>)| k == &key)
        {
            continue;
        }
        let name = call
            .get("function")
            .and_then(|f| f.get("name"))
            .or_else(|| obj.get("name"))
            .and_then(|n| n.as_str())
            .map(str::to_string);
        index.push((key, position, name));
    }
    index
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::kimi_vocab::KimiVocab;
    use dynamo_tokenizers::BasetenTokenizer;
    use serde_json::json;
    use std::path::{Path, PathBuf};

    /// Every fixture is produced by the MODEL REPO's own encoder
    /// (`encoding_k3.build_chat_segments` / `TikTokenTokenizer`) — see
    /// `testdata/gen_kimi_k3_cases.py`. A mismatch here is exactly a
    /// router↔engine tokenization divergence, the thing that silently collapses
    /// cache-aware routing.
    #[derive(serde::Deserialize)]
    struct Case {
        name: String,
        messages: serde_json::Value,
        tools: Option<serde_json::Value>,
        opts: FixtureOpts,
        segments: Vec<FixtureSegment>,
        #[serde(default)]
        token_ids: Option<Vec<u32>>,
        #[serde(default)]
        full_vocab_token_ids: Option<Vec<u32>>,
        /// Above `MAX_INLINE_IDS` the generator stores a digest instead of the
        /// verbatim list (see `gen_kimi_k3_cases.py`). These MUST be read: a
        /// `Case` that deserializes only `token_ids` turns every digested case
        /// into a silent skip, which is how three tool-path cases went
        /// unchecked.
        #[serde(default)]
        token_ids_len: Option<usize>,
        #[serde(default)]
        token_ids_sha256: Option<String>,
        /// Present only on cases where the REFERENCE raises; the render must fail.
        #[serde(default)]
        raises: Option<String>,
        #[serde(default)]
        full_vocab_token_ids_len: Option<usize>,
        #[serde(default)]
        full_vocab_token_ids_sha256: Option<String>,
    }

    /// The generator's digest: sha256 over each id as 4-byte little-endian.
    fn ids_digest(ids: &[u32]) -> String {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        for id in ids {
            h.update(id.to_le_bytes());
        }
        format!("{:x}", h.finalize())
    }

    /// Assert `got` against whichever form the fixture carries, verbatim or
    /// digest. Returns `false` only when the fixture carries NEITHER, which the
    /// callers turn into a hard failure rather than a skip.
    fn assert_ids(
        got: &[u32],
        verbatim: Option<&Vec<u32>>,
        len: Option<usize>,
        sha256: Option<&String>,
        what: &str,
        case_name: &str,
    ) -> bool {
        if let Some(want) = verbatim {
            assert_eq!(got, want.as_slice(), "{what} mismatch for case {case_name}");
            return true;
        }
        let (Some(want_len), Some(want_sha)) = (len, sha256) else {
            return false;
        };
        assert_eq!(
            got.len(),
            want_len,
            "{what} length mismatch for case {case_name}"
        );
        assert_eq!(
            &ids_digest(got),
            want_sha,
            "{what} digest mismatch for case {case_name}"
        );
        true
    }

    #[derive(serde::Deserialize)]
    struct FixtureOpts {
        thinking: bool,
        thinking_effort: Option<String>,
        /// A `Value`, not a string: `tool_choice` may be a named-function OBJECT,
        /// which is exactly what production sees on the request. Only `"required"`
        /// and `"none"` change the prompt; every other shape is `Unset`.
        tool_choice: Option<serde_json::Value>,
        response_format: Option<serde_json::Value>,
        add_generation_prompt: bool,
    }

    #[derive(serde::Deserialize)]
    struct FixtureSegment {
        text: String,
        allow_special: bool,
    }

    impl FixtureOpts {
        fn to_render_opts(&self) -> RenderOpts {
            RenderOpts {
                thinking: self.thinking,
                // Mirrors `resolve_render_opts`, including the distinction that
                // matters: a present-but-unsupported effort is `Invalid`, NOT
                // `None`. Collapsing the two here let a fixture whose effort the
                // reference REJECTS render happily with the default preamble, so
                // the harness silently disagreed with production on exactly the
                // input the reject case exists to pin.
                thinking_effort: match self.thinking_effort.as_deref() {
                    None => RequestedEffort::None,
                    Some(s) => ThinkingEffort::parse(s)
                        .map_or(RequestedEffort::Invalid, RequestedEffort::Valid),
                },
                tool_choice: match self.tool_choice.as_ref().and_then(|v| v.as_str()) {
                    Some("required") => ToolChoice::Required,
                    Some("none") => ToolChoice::None,
                    _ => ToolChoice::Unset,
                },
                response_format: self.response_format.clone(),
                add_generation_prompt: self.add_generation_prompt,
            }
        }
    }

    fn cases() -> Vec<Case> {
        serde_json::from_str(include_str!("testdata/kimi_k3_cases.json"))
            .expect("parse kimi_k3_cases.json")
    }

    /// Render-only cases (no `token_ids`); `Case`'s id fields are all optional.
    fn render_cases() -> Vec<Case> {
        serde_json::from_str(include_str!("testdata/kimi_k3_render_cases.json"))
            .expect("parse kimi_k3_render_cases.json")
    }

    fn testdata_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/tokenizer/testdata")
    }

    /// The committed ~300-entry vocabulary, as the `tokenizer.json` the Baseten
    /// backend reads. Converted from the tiktoken rank file the PYTHON reference
    /// encoder uses by `testdata/gen_kimi_k3_tiny_tokenizer_json.py`, so the ids
    /// asserted below are the reference's over the same vocabulary.
    fn tiny_vocab() -> KimiVocab {
        KimiVocab::from_file(&testdata_dir().join("kimi_k3_tiny_vocab/tokenizer.json"))
            .expect("load tiny vocabulary")
    }

    /// Deliberately delegates to the production interpreter rather than
    /// reimplementing it: a second copy here is what previously let the
    /// production `allow_special` branch be inverted with every test still
    /// passing.
    fn encode_segments(tk: &KimiVocab, segments: &[Segment]) -> Vec<u32> {
        tk.encode_segments(segments).expect("segmented encode")
    }

    fn render(case: &Case) -> Vec<Segment> {
        render_segments(
            &case.messages,
            case.tools.as_ref(),
            &case.opts.to_render_opts(),
        )
        .unwrap_or_else(|e| panic!("case {} failed to render: {e:#}", case.name))
    }

    /// The whole rendering surface, segment by segment: the exact text AND the
    /// exact `allow_special` split. Comparing only the concatenated text would
    /// pass on a build that put every marker in one ordinary-text segment —
    /// which tokenizes to completely different ids.
    #[test]
    fn segment_parity_with_reference_encoder() {
        for case in cases() {
            let got = render(&case);
            let want: Vec<Segment> = case
                .segments
                .iter()
                .map(|s| Segment::from_fixture(s.text.clone(), s.allow_special))
                .collect();
            assert_eq!(
                got, want,
                "segment mismatch for case {}\n got: {:#?}\nwant: {:#?}",
                case.name, got, want
            );
        }
    }

    /// Multimodal render parity, segment for segment.
    ///
    /// Separate fixture because it needs no tokenizer: `encoding_k3.py` is pure
    /// stdlib, so `gen_kimi_k3_multimodal_cases.py` runs the REFERENCE encoder in
    /// a bare interpreter and these expectations are its actual output. Segments
    /// are the right level: the render is the half that differs on image input,
    /// and the tokenizer half is pinned id-for-id separately by
    /// [`full_vocab_parity_when_available`].
    ///
    /// Covers how an image reference is spelled, where it sits in a message and
    /// across turns, and every interaction with thinking/effort/tools/
    /// response_format/tool_choice.
    #[test]
    fn render_parity_with_reference_encoder() {
        let all = render_cases();
        assert!(all.len() >= 60, "fixture went missing: {} cases", all.len());
        let mut rendered = 0;
        let mut rejected = 0;
        for case in all {
            // Inputs the reference REFUSES must not render here either: inventing a
            // prompt the engine raises on would route a request that is failing
            // anyway on ids the engine can never produce.
            if let Some(exc) = &case.raises {
                let err = render_segments(
                    &case.messages,
                    case.tools.as_ref(),
                    &case.opts.to_render_opts(),
                )
                .expect_err(&format!(
                    "case {} must fail: the reference raises {exc}",
                    case.name
                ));
                let _ = err;
                rejected += 1;
                continue;
            }
            rendered += 1;
            let got = render(&case);
            let want: Vec<Segment> = case
                .segments
                .iter()
                .map(|s| Segment::from_fixture(s.text.clone(), s.allow_special))
                .collect();
            assert_eq!(
                got, want,
                "segment mismatch for render case {}\n got: {:#?}\nwant: {:#?}",
                case.name, got, want
            );
        }
        assert!(
            rendered >= 55 && rejected >= 8,
            "expected both halves of the fixture to be exercised, got {rendered} rendered \
             and {rejected} rejected"
        );
    }

    /// The two properties the multimodal fixtures exist to pin, called out so a
    /// regression names the cause instead of dumping 38 segments.
    ///
    /// 1. An image part becomes the placeholder as a MARKER (`allow_special`),
    ///    because the reference emits it with `allow_special=True`.
    /// 2. The same string typed by a CLIENT stays ordinary text and consumes no
    ///    image slot — `_append_text`'s substitution branch is unreachable while
    ///    `image_prompts is None`, which is the only way the router can call the
    ///    encoder (substituting needs pre-resize pixel dimensions it never sees).
    #[test]
    fn image_parts_are_markers_but_client_placeholder_text_is_not() {
        let by_name = |n: &str| {
            render_cases()
                .into_iter()
                .find(|c| c.name == n)
                .unwrap_or_else(|| panic!("fixture case {n} missing"))
        };

        let real = render(&by_name("mm_image_only"));
        let marker = real
            .iter()
            .find(|s| s.text() == IMAGE_PLACEHOLDER)
            .expect("an image part renders the placeholder as its own segment");
        assert!(
            marker.allows_special(),
            "the reference emits the placeholder with allow_special=True"
        );

        let literal = render(&by_name("mm_literal_placeholder_in_user_text"));
        assert!(
            !literal.iter().any(|s| s.text() == IMAGE_PLACEHOLDER),
            "client text must NOT be split into a placeholder segment"
        );
        let carrying = literal
            .iter()
            .find(|s| s.text().contains(IMAGE_PLACEHOLDER))
            .expect("the literal text survives verbatim");
        assert!(
            !carrying.allows_special(),
            "a client-typed placeholder must tokenize with specials disabled"
        );
    }

    /// The rendered text alone, as a faster-to-read failure when the split is
    /// right but the bytes are not (or vice versa).
    #[test]
    fn rendered_text_parity_with_reference_encoder() {
        for case in cases() {
            let got: String = render(&case).iter().map(Segment::text).collect();
            let want: String = case.segments.iter().map(|s| s.text.as_str()).collect();
            assert_eq!(got, want, "rendered text mismatch for case {}", case.name);
        }
    }

    /// Token ids over a committed synthetic vocabulary, against the ids the real
    /// Python `_encode_chat_segments` produced over that same vocabulary.
    ///
    /// This is what proves the segment split is actually HONORED at encode time
    /// rather than merely computed.
    /// Every case is checked — verbatim or by digest. There is deliberately no
    /// `continue` here: the previous version skipped any case the generator had
    /// digested, so the three tool-declaration paths (which are exactly the
    /// cases long enough to be digested) asserted nothing at all.
    #[test]
    fn token_id_parity_over_tiny_vocab() {
        let tk = tiny_vocab();
        let all = cases();
        let total = all.len();
        let mut checked = 0;
        for case in all {
            let got = encode_segments(&tk, &render(&case));
            assert!(
                assert_ids(
                    &got,
                    case.token_ids.as_ref(),
                    case.token_ids_len,
                    case.token_ids_sha256.as_ref(),
                    "token id",
                    &case.name,
                ),
                "case {} carries neither token_ids nor a token_ids digest; \
                 regenerate with gen_kimi_k3_cases.py",
                case.name
            );
            checked += 1;
        }
        assert_eq!(
            checked, total,
            "every fixture case must have its token ids checked"
        );
    }

    /// The injection case, called out on its own because it is the entire reason
    /// this encoder emits segments instead of a string: a `<|open|>` typed by a
    /// USER must tokenize as ordinary bytes, never as the control token that the
    /// encoder's own `<|open|>` becomes.
    #[test]
    fn user_text_control_markers_are_not_promoted() {
        let tk = tiny_vocab();
        // The marker's control id, read the way `markers_resolve` proves it exists:
        // encoded AS a marker it is exactly one token, and that token is the id.
        let marker_ids = encode_segments(&tk, &[Segment::marker("<|open|>")]);
        assert_eq!(
            marker_ids.len(),
            1,
            "tiny vocabulary must register <|open|> as one control token"
        );
        let open_id = marker_ids[0];

        let opts = RenderOpts {
            thinking_effort: RequestedEffort::None,
            ..RenderOpts::default()
        };
        let hostile = json!([{"role": "user", "content": "<|open|>"}]);
        let benign = json!([{"role": "user", "content": "hello"}]);

        let hostile_ids = encode_segments(&tk, &render_segments(&hostile, None, &opts).unwrap());
        let benign_ids = encode_segments(&tk, &render_segments(&benign, None, &opts).unwrap());

        // The structural markers the encoder emitted DID become control tokens...
        assert!(
            benign_ids.contains(&open_id),
            "the encoder's own <|open|> must encode as the control token"
        );
        // ...and the user's identical-looking text did NOT add another one.
        assert_eq!(
            hostile_ids.iter().filter(|id| **id == open_id).count(),
            benign_ids.iter().filter(|id| **id == open_id).count(),
            "a <|open|> in user text must not add a control token"
        );
    }

    /// Long inputs cross Python's `_split_whitespaces_or_nonwhitespaces`
    /// boundary, which changes the token stream. Digest-pinned rather than
    /// stored verbatim (26k ids); see the fixture's generator.
    #[test]
    fn long_input_chunking_parity() {
        use sha2::{Digest, Sha256};

        #[derive(serde::Deserialize)]
        struct Chunking {
            content_char: String,
            content_count: usize,
            tiny: Digested,
        }
        #[derive(serde::Deserialize)]
        struct Digested {
            ids_len: usize,
            ids_sha256: String,
        }

        let fixture: Chunking =
            serde_json::from_str(include_str!("testdata/kimi_k3_chunking.json"))
                .expect("parse kimi_k3_chunking.json");
        let content = fixture.content_char.repeat(fixture.content_count);
        let messages = json!([{"role": "user", "content": content}]);
        let ids = encode_segments(
            &tiny_vocab(),
            &render_segments(&messages, None, &RenderOpts::default()).unwrap(),
        );

        assert_eq!(ids.len(), fixture.tiny.ids_len, "chunked id count");
        let mut hasher = Sha256::new();
        for id in &ids {
            hasher.update(id.to_le_bytes());
        }
        assert_eq!(
            format!("{:x}", hasher.finalize()),
            fixture.tiny.ids_sha256,
            "chunked id sequence"
        );
    }

    /// THE regression test for [`super::kimi_vocab`]'s reason to exist, at id
    /// level over the real vocabulary.
    ///
    /// A >25,000-char run containing a C0 separator must tokenize the way Python
    /// does — which is NOT split, because `str.isspace()` treats U+001F as
    /// whitespace and so resets the run. The upstream wrapper sees one 40,001-char
    /// run (`char::is_whitespace` excludes U+001F), splits at 25,000, and emits
    /// three extra tokens. Needs the real vocabulary: the tiny one's merges are
    /// too short for a split point to change any ids.
    #[test]
    fn c0_separator_chunking_follows_python_when_available() {
        let Ok(dir) = std::env::var("SGLANG_ROUTER_K3_TEST_VOCAB_DIR") else {
            eprintln!(
                "skipping: set SGLANG_ROUTER_K3_TEST_VOCAB_DIR to a directory holding \
                 `baseten/kimi-k3-tokenizer`'s tokenizer.json"
            );
            return;
        };
        let Some(tk) = real_vocab(&dir) else { return };
        let text = format!("{}\u{1f}{}", "x".repeat(20_000), "x".repeat(20_000));

        let ours = encode_segments(&tk, &[Segment::client_text(text.clone())]);
        let unsplit = tk
            .encode_unsplit_for_test(&text)
            .expect("unsplit reference encode");
        assert_eq!(
            ours, unsplit,
            "Python's runs are 20000/1/20000, none over the cap, so no split happens"
        );

        let json = Path::new(&dir).join("tokenizer.json");
        let upstream = BasetenTokenizer::from_file(json.to_str().unwrap()).unwrap();
        let theirs = Encoder::encode_segments(&upstream, &[EncodeSegment::new(&text, false)])
            .expect("upstream encode")
            .token_ids()
            .to_vec();
        assert_ne!(
            ours, theirs,
            "if these ever agree, upstream fixed its whitespace definition and the \
             chunking can be handed back to it"
        );
    }

    /// The real vocabulary, or `None` with a reason — so a mis-set env var reads
    /// as a skip rather than a panic that looks like a broken tree.
    fn real_vocab(dir: &str) -> Option<KimiVocab> {
        let json = Path::new(dir).join("tokenizer.json");
        if !json.is_file() {
            eprintln!(
                "skipping: {} does not exist. SGLANG_ROUTER_K3_TEST_VOCAB_DIR must hold \
                 `baseten/kimi-k3-tokenizer`'s tokenizer.json (the model repo's own \
                 tiktoken.model is NOT enough for the segment encoder)",
                json.display()
            );
            return None;
        }
        Some(KimiVocab::from_file(&json).expect("load real vocabulary"))
    }

    /// Full-vocabulary parity, against the REAL 163,584-entry vocabulary.
    ///
    /// Opt-in because that `tokenizer.json` is ~10 MB and cannot be committed.
    /// Point `SGLANG_ROUTER_K3_TEST_VOCAB_DIR` at a directory holding
    /// `baseten/kimi-k3-tokenizer`'s `tokenizer.json` to run it:
    ///
    /// ```text
    /// SGLANG_ROUTER_K3_TEST_VOCAB_DIR=/path/to/Kimi-K3 cargo test full_vocab
    /// ```
    ///
    /// The tiny-vocab tests cover the encoder's logic; this one additionally pins
    /// the real 296k-entry merge table and the `&&` character-class intersections
    /// in the vocabulary's own pre-tokenizer pattern, which a ~300-entry synthetic
    /// vocabulary cannot exercise.
    #[test]
    fn full_vocab_parity_when_available() {
        let Ok(dir) = std::env::var("SGLANG_ROUTER_K3_TEST_VOCAB_DIR") else {
            eprintln!(
                "skipping: set SGLANG_ROUTER_K3_TEST_VOCAB_DIR to a directory holding \
                 `baseten/kimi-k3-tokenizer`'s tokenizer.json"
            );
            return;
        };
        let Some(tk) = real_vocab(&dir) else { return };
        let all = cases();
        let total = all.len();
        let mut checked = 0;
        for case in all {
            let got = encode_segments(&tk, &render(&case));
            if assert_ids(
                &got,
                case.full_vocab_token_ids.as_ref(),
                case.full_vocab_token_ids_len,
                case.full_vocab_token_ids_sha256.as_ref(),
                "full-vocab token id",
                &case.name,
            ) {
                checked += 1;
            }
        }
        assert_eq!(
            checked, total,
            "every fixture case must carry full-vocab token ids (verbatim or digest)"
        );
    }

    /// The same 25,000-char whitespace-run split as [`long_input_chunking_parity`],
    /// over the REAL vocabulary.
    ///
    /// Worth its own test rather than folding into the tiny-vocab one: reproducing
    /// the reference's input chunking is the entire reason this encoder runs on the
    /// Baseten backend instead of `dynamo_tokenizers::TikTokenTokenizer`, which
    /// hands each segment to the BPE whole. A silent regression here would change
    /// the ids the router forwards to the engine on exactly the long prompts
    /// cache-aware routing exists for.
    #[test]
    fn full_vocab_chunking_parity_when_available() {
        use sha2::{Digest, Sha256};

        #[derive(serde::Deserialize)]
        struct Chunking {
            content_char: String,
            content_count: usize,
            full: Digested,
        }
        #[derive(serde::Deserialize)]
        struct Digested {
            ids_len: usize,
            ids_sha256: String,
        }

        let Ok(dir) = std::env::var("SGLANG_ROUTER_K3_TEST_VOCAB_DIR") else {
            eprintln!(
                "skipping: set SGLANG_ROUTER_K3_TEST_VOCAB_DIR to a directory holding \
                 `baseten/kimi-k3-tokenizer`'s tokenizer.json"
            );
            return;
        };
        let Some(bt) = real_vocab(&dir) else { return };

        let fixture: Chunking =
            serde_json::from_str(include_str!("testdata/kimi_k3_chunking.json"))
                .expect("parse kimi_k3_chunking.json");
        let content = fixture.content_char.repeat(fixture.content_count);
        let messages = json!([{"role": "user", "content": content}]);
        let ids = encode_segments(
            &bt,
            &render_segments(&messages, None, &RenderOpts::default()).unwrap(),
        );

        let mut hasher = Sha256::new();
        for id in &ids {
            hasher.update(id.to_le_bytes());
        }
        let digest = format!("{:x}", hasher.finalize());
        assert_eq!(
            ids.len(),
            fixture.full.ids_len,
            "chunked id count over the real vocabulary"
        );
        assert_eq!(
            digest, fixture.full.ids_sha256,
            "chunked id digest over the real vocabulary"
        );
    }

    /// A tool result whose name cannot be resolved makes the render FAIL rather
    /// than guess — the caller then routes by raw text instead of by ids that
    /// disagree with the engine.
    #[test]
    fn unresolvable_tool_name_is_an_error() {
        let messages = json!([
            {"role": "user", "content": "go"},
            {"role": "tool", "content": "orphan result"},
        ]);
        assert!(render_segments(&messages, None, &RenderOpts::default()).is_err());
    }

    /// Arguments that parse to a non-object have no rendering, so they error;
    /// arguments that do not parse at all are preserved verbatim in a `<json>`
    /// block. The two look similar and behave oppositely.
    #[test]
    fn tool_argument_string_handling() {
        let with_args = |args: serde_json::Value| {
            json!([
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": "",
                 "tool_calls": [{"id": "c", "function": {"name": "f", "arguments": args}}]},
            ])
        };
        let opts = RenderOpts::default();

        assert!(render_segments(&with_args(json!("[1,2]")), None, &opts).is_err());
        assert!(render_segments(&with_args(json!("7")), None, &opts).is_err());

        let text: String = render_segments(&with_args(json!("{oops")), None, &opts)
            .expect("unparsable arguments render as a json block")
            .iter()
            .map(|s| s.text.as_str())
            .collect();
        assert!(text.contains("json"), "expected a <json> block: {text}");
        assert!(
            text.contains("{oops"),
            "raw arguments must be preserved: {text}"
        );
    }

    /// A vocabulary whose control markers do not resolve must be rejected, not
    /// used. Without the guard the encoder emits `<|open|>` as ~7 ordinary
    /// tokens instead of one control id — for every tag in every prompt — and
    /// still reports success, so the ids get forwarded to the engine.
    #[test]
    fn markers_must_resolve_for_the_encoder_to_be_usable() {
        let good = tiny_vocab();
        assert!(markers_resolve(&good).is_ok());
        // The ids a GOOD vocabulary gives the markers, so the negative case below
        // can be checked against an axis `markers_resolve` does not itself define.
        let control_ids: Vec<u32> = CONTROL_MARKERS
            .iter()
            .map(|m| encode_segments(&good, &[Segment::marker(m)])[0])
            .collect();

        // The same base vocabulary with its `added_tokens` stripped: the ordinary
        // ranks still load, but no marker has a control id any more. Written
        // WITHOUT a sibling `tokenizer_config.json`, because `from_file` merges
        // specials back in from one when present — which is the very thing under
        // test here.
        let dir = std::env::temp_dir().join("sgl_router_k3_no_specials");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut json: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(testdata_dir().join("kimi_k3_tiny_vocab/tokenizer.json"))
                .unwrap(),
        )
        .unwrap();
        json["added_tokens"] = json!([]);
        let path = dir.join("tokenizer.json");
        std::fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();

        let nameless = KimiVocab::from_file(&path).expect("loads; raw text still works");
        let err = markers_resolve(&nameless)
            .expect_err("a vocabulary with no added tokens must not pass the marker guard");
        assert!(
            format!("{err:#}").contains("not a registered added token"),
            "the guard must say WHY, so the operator knows to change vocabulary: {err:#}"
        );

        // The consequence, not the predicate: a whole rendered prompt over this
        // vocabulary contains none of the control ids. This is what the guard
        // exists to prevent being forwarded, and it is independent of how
        // `markers_resolve` is implemented.
        let messages = json!([{"role": "user", "content": "hi"}]);
        let ids = encode_segments(
            &nameless,
            &render_segments(&messages, None, &RenderOpts::default()).unwrap(),
        );
        for id in &control_ids {
            assert!(
                !ids.contains(id),
                "control id {id} must not appear when the markers do not resolve"
            );
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// An unsupported `thinking_effort` must fail the render, not silently drop
    /// the preamble. `medium` is the live case — the encoder's own prompt text
    /// advertises it while the reference encoder rejects it.
    #[test]
    fn unsupported_thinking_effort_fails_the_render() {
        let messages = json!([{"role": "user", "content": "hi"}]);

        let opts = resolve_render_opts(&json!({"thinking_effort": "medium"}));
        assert_eq!(opts.thinking_effort, RequestedEffort::Invalid);
        assert!(render_segments(&messages, None, &opts).is_err());

        // A non-string value is equally unrenderable.
        let opts = resolve_render_opts(&json!({"thinking_effort": 3}));
        assert_eq!(opts.thinking_effort, RequestedEffort::Invalid);

        // Explicit null means "absent" -> the default still applies.
        let opts = resolve_render_opts(&json!({"thinking_effort": null}));
        assert_eq!(
            opts.thinking_effort,
            RequestedEffort::Valid(ThinkingEffort::Max)
        );

        // Non-thinking mode renders no preamble at all, so an unsupported
        // effort is moot and must NOT fail the request.
        let opts = RenderOpts {
            thinking: false,
            thinking_effort: RequestedEffort::Invalid,
            ..RenderOpts::default()
        };
        assert!(render_segments(&messages, None, &opts).is_ok());
    }

    /// Moonshot's own spelling of the reasoning controls. The router renders the
    /// prompt the engine will be billed for, so ignoring `thinking` here means a
    /// `type: "disabled"` request is routed (and cache-keyed) as a thinking
    /// prompt while the engine renders the response channel.
    #[test]
    fn resolve_render_opts_reads_the_thinking_object() {
        let disabled = resolve_render_opts(&json!({"thinking": {"type": "disabled"}}));
        assert!(!disabled.thinking);

        for kind in ["enabled", "adaptive"] {
            let opts = resolve_render_opts(&json!({"thinking": {"type": kind}}));
            assert!(opts.thinking, "{kind} must leave thinking on");
        }

        // No type at all still means on, and `keep` is not the router's business.
        let opts = resolve_render_opts(&json!({"thinking": {"keep": "all"}}));
        assert!(opts.thinking);

        // thinking.effort feeds the same slot as the template kwarg.
        let opts = resolve_render_opts(&json!({"thinking": {"effort": "low"}}));
        assert_eq!(
            opts.thinking_effort,
            RequestedEffort::Valid(ThinkingEffort::Low)
        );

        // chat_template_kwargs stays the top override, matching the engine.
        let opts = resolve_render_opts(&json!({
            "thinking": {"type": "disabled", "effort": "low"},
            "chat_template_kwargs": {"thinking": true, "thinking_effort": "max"},
        }));
        assert!(opts.thinking);
        assert_eq!(
            opts.thinking_effort,
            RequestedEffort::Valid(ThinkingEffort::Max)
        );

        // thinking.effort outranks the top-level field.
        let opts = resolve_render_opts(&json!({
            "thinking": {"effort": "low"},
            "thinking_effort": "max",
        }));
        assert_eq!(
            opts.thinking_effort,
            RequestedEffort::Valid(ThinkingEffort::Low)
        );

        // A bare `thinking: true` is the template-kwarg spelling, not the object;
        // it must not be read as a config and must not disable thinking.
        let opts = resolve_render_opts(&json!({"thinking": true}));
        assert!(opts.thinking);
    }

    /// The whole point: a disabled-thinking request renders the response channel
    /// and no effort preamble, which is what makes it 67 tokens shorter.
    #[test]
    fn thinking_object_disabled_renders_the_response_channel() {
        let messages = json!([{"role": "user", "content": "hi"}]);
        let opts =
            resolve_render_opts(&json!({"thinking": {"type": "disabled", "effort": "high"}}));
        let segments = render_segments(&messages, None, &opts).unwrap();
        let text: String = segments.iter().map(|s| s.text.as_str()).collect();

        assert!(text.ends_with("<|open|>response<|sep|>"), "tail: {text:?}");
        assert!(!text.contains("<|open|>think<|sep|>"));
        assert!(!text.contains("thinking_effort"));
    }

    /// `chat_template_kwargs` overrides beat the defaults, and the defaults are
    /// the reference encoder's own (thinking on, effort max) rather than off.
    #[test]
    fn resolve_render_opts_reads_the_request() {
        let opts = resolve_render_opts(&json!({}));
        assert!(opts.thinking, "K3 defaults to thinking mode");
        assert_eq!(
            opts.thinking_effort,
            RequestedEffort::Valid(ThinkingEffort::Max)
        );

        let opts = resolve_render_opts(&json!({"chat_template_kwargs": {"thinking": false}}));
        assert!(!opts.thinking);

        let opts = resolve_render_opts(&json!({
            "chat_template_kwargs": {"thinking_effort": "low"},
            "thinking_effort": "high",
        }));
        assert_eq!(
            opts.thinking_effort,
            RequestedEffort::Valid(ThinkingEffort::Low),
            "chat_template_kwargs wins over the top-level field"
        );

        let opts = resolve_render_opts(&json!({"tool_choice": "required"}));
        assert_eq!(opts.tool_choice, ToolChoice::Required);
        // A named-function tool_choice object renders nothing, like `auto`.
        let opts = resolve_render_opts(&json!({"tool_choice": {"type": "function"}}));
        assert_eq!(opts.tool_choice, ToolChoice::Unset);
    }
}
