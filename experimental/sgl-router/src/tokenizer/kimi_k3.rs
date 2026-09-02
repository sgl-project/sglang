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
//! # Which revision this mirrors
//!
//! `moonshotai/Kimi-K3` at `a590ce09`. That matters because the model repo moves
//! under this file and the encoder's BEHAVIOUR moves with it — the same repo has
//! shipped two different tool-argument renderings. The fixture generators pin
//! the same revision (`KIMI_K3_REVISION` in `testdata/gen_kimi_k3_cases.py`), so
//! a bump lands as one commit: new revision, regenerated fixtures, and whatever
//! change to this file they force.
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
/// Returns [`RenderErr`] — the caller then falls back to raw prompt-text
/// routing — whenever the reference encoder would itself raise. That set is the
/// variants of `RenderErr`, which is the whole error type here precisely so this
/// list cannot drift: a non-list `messages`, a message that is not a mapping, an
/// unknown or missing role, an unsupported `thinking_effort`, a malformed
/// content part, a tool call with no `function.name`, `arguments` of the wrong
/// TYPE, and an unresolvable tool-result name.
///
/// Note what is NOT an error: a string `arguments` that is not a well-formed
/// JSON object renders as a raw `<json>` block, matching the reference.
///
/// Rendering something plausible instead would produce ids that silently
/// disagree with the engine, which is worse than routing by raw text.
pub fn render_segments(
    messages: &serde_json::Value,
    tools: Option<&serde_json::Value>,
    opts: &RenderOpts,
) -> RenderResult<Vec<Segment>> {
    // A non-list `messages` used to render an empty prompt and report success;
    // the reference `enumerate`s it and refuses at the first element.
    let Some(raw) = messages.as_array().map(Vec::as_slice) else {
        return Err(RenderErr::MessagesNotAList {
            found: py_type_name(messages),
        });
    };
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
            return Err(RenderErr::UnsupportedThinkingEffort);
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

    for (message_index, message) in ordered.iter().enumerate() {
        // A message that is not a mapping used to be SKIPPED. That did NOT
        // render an empty prompt — a batched call still emitted the effort
        // preamble and the generation prompt, ~10 segments with the whole
        // conversation missing — which is why `mod.rs`'s zero-token guard could
        // never catch it, and why the ids were stamped engine-equivalent and
        // forwardable. The reference refuses it now.
        let Some(obj) = message.as_object() else {
            return Err(RenderErr::MessageNotAMapping {
                index: message_index,
                found: py_type_name(message),
            });
        };
        // `message.get("role")`, so a MISSING role reaches the same refusal as
        // an unrecognised one rather than raising a distinct error earlier.
        //
        // Lowercased for the same reason `super::dsv4` lowercases: the engine's
        // generic-role model case-normalizes (`_normalize_role`), so `"System"`
        // reaches the reference encoder as `system` and renders. Comparing
        // case-sensitively here would refuse a request the engine serves. The
        // user-role model is a bare `Literal["user"]`, so `"User"` is a 422 at
        // the engine and lowercasing it here is harmless alignment.
        let role = obj
            .get("role")
            .and_then(|r| r.as_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        match role.as_str() {
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
            // An unrecognised role used to render nothing FOR THAT TURN while
            // the rest of the prompt rendered normally and reported success — so
            // a typo'd role silently dropped a message and the router then
            // routed on, and forwarded, a prefix the engine never produces.
            other => {
                return Err(RenderErr::UnknownRole {
                    index: message_index,
                    role: other.to_string(),
                })
            }
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
fn render_content(out: &mut Segments, content: Option<&serde_json::Value>) -> RenderResult<()> {
    match content {
        None | Some(serde_json::Value::Null) => Ok(()),
        Some(serde_json::Value::String(s)) => {
            out.text(s.clone());
            Ok(())
        }
        Some(serde_json::Value::Array(parts)) => {
            for (part_index, part) in parts.iter().enumerate() {
                // `part["type"]` on a non-mapping is a TypeError in the
                // reference, not a KeyError — so the shape is checked before the
                // key. The fixture battery asserts that split.
                if !part.is_object() {
                    return Err(RenderErr::ContentPartNotAMapping {
                        index: part_index,
                        found: py_type_name(part),
                    });
                }
                let part_type = part
                    .get("type")
                    // A part with NO `type` is malformed: the reference indexes
                    // `part["type"]` and raises `KeyError`, so erroring here keeps
                    // the request on raw-text routing rather than inventing a
                    // render the engine will never produce. A `type` that is
                    // present but not a string is different — Python's `in` test
                    // just fails, and the part falls through to its `text`.
                    .ok_or(RenderErr::ContentPartMissingType { index: part_index })?
                    .as_str()
                    .unwrap_or_default();
                if matches!(part_type, "image" | "image_url") {
                    out.control(IMAGE_PLACEHOLDER);
                } else {
                    let Some(text) = part.get("text") else {
                        return Err(RenderErr::ContentPartMissingText {
                            index: part_index,
                            part_type: part_type.to_string(),
                        });
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
        Some(other) => Err(RenderErr::ContentNotStringOrParts {
            found: py_type_name(other),
        }),
    }
}

/// The inside of an assistant `<message>`: the think channel (thinking mode
/// only), the response channel, then any tool calls.
fn render_assistant_body(
    out: &mut Segments,
    message: &serde_json::Value,
    thinking: bool,
) -> RenderResult<()> {
    if thinking {
        // Structural: the tags are emitted even with nothing to put in them.
        // `reasoning_content or reasoning` is Python's `or`, so an EMPTY
        // reasoning_content falls through to `reasoning`.
        let reasoning = message
            .get("reasoning_content")
            .filter(|v| json_truthy(v))
            .or_else(|| message.get("reasoning"))
            // The reference's condition is a CONJUNCTION: `reasoning_content or
            // reasoning` then `is not None and str(...) != ""`. The null guard
            // must apply to the RESULT of the `or`, so a present `"reasoning":
            // null` renders nothing rather than the four characters `None`.
            // Deliberately `is_null`, NOT `json_truthy`: the reference tests
            // `is not None`, so `false` / `0` / `[]` DO render.
            .filter(|v| !v.is_null());
        open_tag(out, "think", &[]);
        if let Some(r) = reasoning {
            // Only an EMPTY string counts as no reasoning, so whitespace-only
            // reasoning is still rendered into the think channel.
            let text = py_str(r);
            if !text.is_empty() {
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
fn render_tool_call(
    out: &mut Segments,
    call: &serde_json::Value,
    index: usize,
) -> RenderResult<()> {
    // `tool_call.get("function", tool_call)` — the OpenAI nesting is optional.
    let func = call.get("function").unwrap_or(call);
    let Some(name) = func.get("name").and_then(|n| n.as_str()) else {
        return Err(RenderErr::ToolCallWithoutName { index });
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
        // A string that is not a well-formed JSON OBJECT is passed through
        // verbatim inside a `<json>` block rather than being dropped, guessed
        // at, or turned into an error.
        ToolArguments::RawJsonBlock(raw) => {
            open_tag(out, "json", &[("type".into(), "object".into())]);
            out.text(raw);
            close_tag(out, "json");
        }
        ToolArguments::Arguments(args) => {
            for arg in args {
                open_tag(
                    out,
                    "argument",
                    &[
                        ("key".into(), arg.key),
                        ("type".into(), arg.arg_type.into()),
                    ],
                );
                out.text(arg.text);
                close_tag(out, "argument");
            }
        }
    }

    close_tag(out, "call");
    Ok(())
}

/// Errors from rendering. EVERY variant mirrors an exception the REFERENCE
/// encoder itself raises, so the engine rejects the same input and a caller
/// treating this as a client error rather than encoder breakage reproduces the
/// engine's own outcome.
///
/// This is the entire error type of [`render_segments`], deliberately.
/// `super::dsv4::RenderErr` is the same shape for the same reason: with a typed
/// error the request-vs-breakage split is exhaustive **by construction**, where
/// a runtime downcast over `anyhow` silently lets each new `bail!` escape the
/// classification. That is not hypothetical — the two-variant version of this
/// type covered 2 of the 10 refusals the fixture battery already recorded.
///
/// A future variant that is a ROUTER limitation rather than an engine rejection
/// must NOT be added here. Two consumers key on this type: `super::mod` keeps a
/// client error from consuming the model's one-shot broken-encoder WARN latch,
/// and `crate::server::routes::chat` keeps it out of the broken-offload metric
/// `sgl_router_ingress_tokenize_errors_total`.
///
/// Every variant is pinned by a `raises` case in
/// `testdata/kimi_k3_render_cases.json`, and the fixture harness asserts the
/// CLASS — each reject case must surface as a `RenderErr` — so a new reference
/// refusal cannot be added without being classified.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderErr {
    /// `messages` is not a list at all. The reference `enumerate`s whatever it
    /// is and hits the per-message mapping check (`ValueError`).
    MessagesNotAList { found: &'static str },
    /// A message that is not a mapping. A batched call — a list OF
    /// conversations rather than of messages — is the shape that lands here
    /// (`ValueError`).
    MessageNotAMapping { index: usize, found: &'static str },
    /// A role the reference does not define, including a missing or non-string
    /// one (`ValueError`). Roles are compared after ASCII-lowercasing, matching
    /// the engine's own `_normalize_role`.
    UnknownRole { index: usize, role: String },
    /// `thinking_effort` outside `_VALID_THINKING_EFFORTS` (`AssertionError`).
    /// Cheap for a client to trigger: the reference's own rendered preamble
    /// advertises `medium` as supported and then refuses it.
    UnsupportedThinkingEffort,
    /// A content part that is not a mapping at all. The reference indexes
    /// `part["type"]`, and subscripting a str/int/list with a string key is a
    /// `TypeError` — a DIFFERENT exception from a mapping that merely lacks the
    /// key, which is why these are two variants and not one.
    ContentPartNotAMapping { index: usize, found: &'static str },
    /// A mapping content part with no `type` key; the reference indexes
    /// `part["type"]` and gets a `KeyError`.
    ContentPartMissingType { index: usize },
    /// A non-image content part with no `text` (`KeyError`).
    ContentPartMissingText { index: usize, part_type: String },
    /// `content` is neither a string nor a list of parts, so the reference
    /// reaches `for part in content` and raises (`TypeError`).
    ContentNotStringOrParts { found: &'static str },
    /// A `tool_calls` entry with no `function.name`; the reference indexes
    /// `fn["name"]` (`KeyError`).
    ToolCallWithoutName { index: usize },
    /// `arguments` is neither a mapping, a string, nor absent (`TypeError`).
    ToolArgumentsWrongType { found: &'static str },
    /// A `tool` message whose name resolves neither from its own `tool`/`name`
    /// nor from the preceding assistant turn's `tool_calls` (`ValueError`).
    UnresolvableToolName { index: usize },
}

impl std::fmt::Display for RenderErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RenderErr::MessagesNotAList { found } => {
                write!(f, "Kimi K3 messages must be a list, got {found}")
            }
            RenderErr::MessageNotAMapping { index, found } => write!(
                f,
                "Kimi K3 messages must be dicts, got {found} at index {index}"
            ),
            RenderErr::UnknownRole { index, role } => {
                write!(f, "unknown message role {role:?} at index {index}")
            }
            RenderErr::UnsupportedThinkingEffort => write!(
                f,
                "unsupported thinking_effort (the reference accepts only low/high/max)"
            ),
            RenderErr::ContentPartNotAMapping { index, found } => {
                write!(f, "content part {index} must be a mapping, got {found}")
            }
            RenderErr::ContentPartMissingType { index } => {
                write!(f, "content part {index} has no `type` field")
            }
            RenderErr::ContentPartMissingText { index, part_type } => write!(
                f,
                "content part {index} of type {part_type:?} has no `text` field"
            ),
            RenderErr::ContentNotStringOrParts { found } => write!(
                f,
                "message content must be a string or a list of parts, got {found}"
            ),
            RenderErr::ToolCallWithoutName { index } => {
                write!(f, "tool_call at index {index} has no `function.name`")
            }
            RenderErr::ToolArgumentsWrongType { found } => write!(
                f,
                "Kimi K3 tool call arguments must be a dict or a JSON object string, got {found}"
            ),
            RenderErr::UnresolvableToolName { index } => write!(
                f,
                "tool message {index} needs a resolvable tool name: carry `tool`/`name`, \
                 or match a preceding assistant tool_call by order"
            ),
        }
    }
}

impl std::error::Error for RenderErr {}

/// Every render path returns [`RenderErr`]; see that type for why it is typed
/// rather than `anyhow`.
type RenderResult<T> = std::result::Result<T, RenderErr>;

/// Python's `type(x).__name__` for a JSON value, so a refusal names the shape
/// the way the reference's own message does — `list`, not the XTML vocabulary's
/// `array`.
fn py_type_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "NoneType",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(n) if n.is_f64() => "float",
        serde_json::Value::Number(_) => "int",
        serde_json::Value::String(_) => "str",
        serde_json::Value::Array(_) => "list",
        serde_json::Value::Object(_) => "dict",
    }
}

/// One rendered `<argument>`: its key, its XTML type, and the EXACT text that
/// becomes the tag body. Mirrors `encoding_k3.XtmlArgument`.
///
/// The text is carried instead of the parsed value because the reference keeps
/// each non-string value's ORIGINAL JSON literal — `1e2` stays `1e2`, `[1,2]`
/// keeps its exact bytes — and re-serializing a `serde_json::Value` cannot
/// reproduce that.
struct XtmlArgument {
    key: String,
    arg_type: &'static str,
    text: String,
}

/// Normalized `tool_calls[].function.arguments`.
enum ToolArguments {
    Arguments(Vec<XtmlArgument>),
    RawJsonBlock(String),
}

/// Mirror `encoding_k3.normalize_tool_arguments`.
///
/// Absent or empty-string arguments render nothing; a MAPPING is rendered
/// value-by-value through [`xtml_value`], which re-serializes with Python's
/// default separators; a STRING is parsed one level deep by
/// [`parse_arguments_object`], which keeps each value's original literal text.
/// The same logical arguments therefore render differently depending on which
/// of the two shapes the client sent — `[1, 2]` for a mapping, `[1,2]` for a
/// string — and that asymmetry is the reference's, so it is pinned by fixture.
///
/// Any string that is not a well-formed JSON object — unparsable,
/// whitespace-only, or valid non-object JSON like `[1,2]` — falls back to the
/// raw `<json>` block. It is NOT an error: the previous revision raised on
/// valid non-object JSON, which failed the whole request over a tool call the
/// engine would have rendered.
///
/// The empty test is `is_empty`, NOT `trim().is_empty()`: a whitespace-only
/// string is a fallback here, not an empty argument list.
fn normalize_tool_arguments(arguments: Option<&serde_json::Value>) -> RenderResult<ToolArguments> {
    match arguments {
        None | Some(serde_json::Value::Null) => Ok(ToolArguments::Arguments(Vec::new())),
        Some(serde_json::Value::Object(map)) => Ok(ToolArguments::Arguments(
            map.iter()
                .map(|(key, value)| XtmlArgument {
                    key: key.clone(),
                    arg_type: xtml_type(value),
                    text: xtml_value(value),
                })
                .collect(),
        )),
        Some(serde_json::Value::String(s)) => {
            if s.is_empty() {
                return Ok(ToolArguments::Arguments(Vec::new()));
            }
            Ok(match parse_arguments_object(s) {
                Ok(args) => ToolArguments::Arguments(args),
                Err(why) => {
                    // The render SUCCEEDS from here, so this degradation is
                    // invisible to every other signal: no fallback log, and the
                    // ids are still stamped engine-equivalent. It is also the one
                    // place where a faithful fallback and a DIVERGENT one look
                    // identical — for the four families listed on
                    // `parse_arguments_object` the reference parses what we
                    // refuse, so the raw block silently disagrees with the
                    // engine's prompt. Carry the cause so the two are
                    // distinguishable. Never log the argument text itself: it is
                    // client tool-call payload.
                    tracing::debug!(
                        error = %why, len = s.len(),
                        "kimi-k3 tool arguments are not a well-formed JSON object; \
                         rendering the raw <json> block"
                    );
                    ToolArguments::RawJsonBlock(s.clone())
                }
            })
        }
        Some(other) => Err(RenderErr::ToolArgumentsWrongType {
            found: py_type_name(other),
        }),
    }
}

/// `encoding_k3._skip_whitespaces`: exactly space, tab, LF and CR.
///
/// Measured: serde_json's own whitespace set is these same four, so the point is
/// NOT that this is narrower than serde's. It is that serde is never invoked at
/// the six structural positions this skips (before `{`, after `{`, either side
/// of `:`, after a value, after a separator), so this function alone decides
/// them — and Rust's `char::is_whitespace` would wrongly accept form feed,
/// NBSP and U+2028 there, where the reference falls back to the raw block.
fn skip_json_ws(bytes: &[u8], from: usize) -> usize {
    let mut idx = from;
    while matches!(bytes.get(idx), Some(b' ' | b'\t' | b'\n' | b'\r')) {
        idx += 1;
    }
    idx
}

/// `json.JSONDecoder.raw_decode`: parse ONE value starting at `idx` and report
/// the offset just past it, leaving whatever follows unexamined.
///
/// `serde_json::from_str` cannot serve here: it calls `Deserializer::end`, so it
/// requires the string to hold nothing but the one value. A bare `Deserializer`
/// would parse-and-stop fine — measured — but does not expose `byte_offset`,
/// which is the "where did this value end" half of `raw_decode` and the only
/// reason a `StreamDeserializer` is used.
fn raw_decode(s: &str, idx: usize) -> Result<(serde_json::Value, usize)> {
    // A StreamDeserializer, because `byte_offset` — the "where did this value
    // end" half of `raw_decode` — is only exposed there, and a plain
    // `Deserializer` would additionally demand the input hold nothing else.
    let mut stream = serde_json::Deserializer::from_str(&s[idx..]).into_iter::<serde_json::Value>();
    let Some(value) = stream.next() else {
        bail!("expected a JSON value at offset {idx}");
    };
    let value = value?;
    Ok((value, idx + stream.byte_offset()))
}

/// Mirror `encoding_k3._parse_arguments_object`: parse a JSON object string one
/// level deep, one `<argument>` per top-level pair.
///
/// Each value's text is its ORIGINAL slice of `s` unless the value is a string,
/// in which case it is the decoded (unescaped) string. Nested values are never
/// re-serialized, so inner spacing and number spelling survive verbatim.
/// Anything after the closing `}` is ignored, and DUPLICATE KEYS are all kept —
/// collecting into a `serde_json::Map` would silently drop all but the last.
///
/// FOUR divergences from the reference, all measured by differential fuzz over
/// 134,520 inputs (0 panics; 98.5% byte-exact render agreement). Every one fails
/// the SAME safe way — Python parses, serde_json refuses, so we emit the raw
/// `<json>` block — and there was no input where both parsed and rendered
/// differently, nor any where we accepted what the reference refused. The engine
/// still re-tokenizes correctly; only the exact prefix-cache match is lost.
///
/// 1. Literal control characters inside strings (712 inputs). The reference
///    decodes with `JSONDecoder(strict=False)`; serde_json rejects them. This is
///    also what the PREVIOUS revision did with such input, so it is a narrower
///    improvement rather than a regression.
/// 2. Non-finite and out-of-f64-range numbers — `NaN`, `Infinity`, `1e309`
///    (753 inputs). The most reachable of the four: `json.dumps` emits bare
///    `NaN`/`Infinity` BY DEFAULT, so any Python client serializing a NaN tool
///    argument produces one, and `json.JSONDecoder` accepts them.
/// 3. Lone-surrogate `\u` escapes (599 inputs) — arguably unfixable rather than
///    a defect: a lone surrogate has no Rust `String` encoding at all.
/// 4. Value nesting at or beyond serde_json's `RECURSION_LIMIT` of 128 (48
///    inputs); measured fine at 127. The reference tolerated 100,000 levels.
///
/// Each family is pinned by a fixture asserting the raw-block fallback, so
/// "fixing" one is a red test rather than a silent change in rendered bytes.
///
/// Indices are BYTE offsets throughout, where the reference's are character
/// offsets. That is safe because the two are only ever used to CUT the same
/// string, never compared to each other, and the cuts land identically.
///
/// The no-panic argument, stated exactly, because the obvious version of it is
/// wrong: most indices come from `byte_offset` or from a skip over bytes in
/// `{space, tab, LF, CR, ',', ':', '{', '}'}`, all < 0x80 and so never a UTF-8
/// lead or continuation byte. But there IS a third origin — the `idx += 1` past
/// an unvalidated separator byte, mirroring the reference's `_next_char()` —
/// which CAN land mid-character. It is safe only because that path reaches
/// `bail!` before any slice: a continuation byte can never compare equal to
/// `b','` or `b'}'`, so the `continue`/`return` arms are unreachable from a
/// mid-character index. A future refactor that recovers from a bad separator
/// instead of bailing would introduce a slicing panic here.
fn parse_arguments_object(s: &str) -> Result<Vec<XtmlArgument>> {
    let bytes = s.as_bytes();
    let mut idx = skip_json_ws(bytes, 0);
    if bytes.get(idx) != Some(&b'{') {
        bail!("JSON arguments must be an object");
    }
    idx = skip_json_ws(bytes, idx + 1);

    let mut parsed = Vec::new();
    match bytes.get(idx) {
        None => bail!("Unexpected end of JSON object"),
        // An empty object returns WITHOUT consuming the `}`, as the reference
        // does; either way the trailing bytes are ignored.
        Some(&b'}') => return Ok(parsed),
        Some(_) => {}
    }

    loop {
        let (key, after_key) = raw_decode(s, idx)?;
        idx = after_key;
        let key = match key {
            serde_json::Value::String(key) => key,
            other => bail!("JSON object key must be a string, got {other}"),
        };

        idx = skip_json_ws(bytes, idx);
        if bytes.get(idx) != Some(&b':') {
            bail!("Expects ':' after {key}");
        }
        idx = skip_json_ws(bytes, idx + 1);

        let value_start = idx;
        let (value, after_value) = raw_decode(s, idx)?;
        idx = after_value;
        let text = match &value {
            serde_json::Value::String(decoded) => decoded.clone(),
            _ => s[value_start..idx].to_string(),
        };
        parsed.push(XtmlArgument {
            key,
            arg_type: xtml_type(&value),
            text,
        });

        // The reference's `_next_char()` reads one CHARACTER and advances; this
        // advances one BYTE. They differ only for a non-ASCII separator, which
        // is malformed either way and bails below before anything is sliced —
        // see the no-panic argument on this function. At the end of the string
        // both yield nothing and advance nothing.
        idx = skip_json_ws(bytes, idx);
        let separator = bytes.get(idx).copied();
        if separator.is_some() {
            idx += 1;
        }
        idx = skip_json_ws(bytes, idx);

        match separator {
            Some(b'}') => return Ok(parsed),
            Some(b',') => continue,
            Some(other) => bail!("Expect '}}' or ',', got {:?}", other as char),
            None => bail!("Expect '}}' or ',', got end of input"),
        }
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
) -> RenderResult<String> {
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
    Err(RenderErr::UnresolvableToolName { index: tool_index })
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
        let total = all.len();
        let names: std::collections::HashSet<String> = all.iter().map(|c| c.name.clone()).collect();
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
                // `render_segments`' error type IS `RenderErr`, so "this is a
                // request error, not encoder breakage" is now compiler-enforced
                // and needs no assertion. What is still worth binding is the
                // MAPPING: each variant claims in its doc comment to mirror a
                // specific reference exception, and the fixture records which
                // one the reference actually raised. Asserting the two agree
                // pins the taxonomy against the reference rather than against
                // itself, and it binds every case the generator emits — not
                // just the ones someone remembered to hand-write.
                let want_exc = reference_exception_for(&err);
                assert_eq!(
                    want_exc, exc,
                    "case {} — the reference raises {exc}, but {err:?} claims to \
                     mirror {want_exc}; one of the two is misclassified",
                    case.name
                );
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
            rendered + rejected == total,
            "every case must be either rendered or rejected, got {rendered} + {rejected} \
             of {total}"
        );
        // A count records the example; this records the POLICY — the cases whose
        // absence would silently un-test a behaviour. A deleted or renamed case
        // then shows up as a diff on this list instead of as a number that still
        // clears a floor, which is how `unknown_role_renders_nothing` (the only
        // index>0 refusal in the tree) went missing without a red test.
        for required in REQUIRED_RENDER_CASES {
            assert!(
                names.contains(*required),
                "required fixture case {required} is missing — if it was renamed, \
                 update REQUIRED_RENDER_CASES; if it was deleted, say why in the diff"
            );
        }
    }

    /// Which reference exception each `RenderErr` variant claims to mirror.
    ///
    /// This is the Rust half of the parity contract that the fixture's `raises`
    /// field is the Python half of. Adding a variant forces an arm here, and the
    /// battery then checks it against what the reference actually raised.
    fn reference_exception_for(err: &RenderErr) -> &'static str {
        match err {
            RenderErr::MessagesNotAList { .. }
            | RenderErr::MessageNotAMapping { .. }
            | RenderErr::UnknownRole { .. }
            | RenderErr::UnresolvableToolName { .. } => "ValueError",
            RenderErr::ContentPartMissingType { .. }
            | RenderErr::ContentPartMissingText { .. }
            | RenderErr::ToolCallWithoutName { .. } => "KeyError",
            RenderErr::ContentNotStringOrParts { .. }
            | RenderErr::ContentPartNotAMapping { .. }
            | RenderErr::ToolArgumentsWrongType { .. } => "TypeError",
            RenderErr::UnsupportedThinkingEffort => "AssertionError",
        }
    }

    /// Fixture cases whose absence would un-test a behaviour, named rather than
    /// counted. Keep the reason with the name.
    const REQUIRED_RENDER_CASES: &[&str] = &[
        // Refusals at index > 0 — the only shape real traffic takes. Without
        // these, refusing only at index 0 is a green mutation.
        "misc_unknown_role_at_index_1",
        "misc_batched_conversation_at_index_1",
        // `messages` not a list at all.
        "misc_messages_is_a_string",
        // The reference's `is not None` conjunct on reasoning.
        "as_reasoning_null",
        "as_reasoning_empty_then_null_alias",
        // Whitespace-only reasoning DOES render (the `!= ""` half).
        "as_reasoning_whitespace_only",
        // Literal argument text, and the mapping path's contrasting spacing.
        "tc_args_str_literal_text_preserved",
        "tc_args_mapping_reserialized",
        // A truncated tool call — what a model emits at `max_tokens`.
        "tc_args_str_truncated",
        // `byte_offset` must be the value's end, not the whitespace after it.
        "tc_args_str_spaced_separators",
        // A sliced value containing non-ASCII: the slice-panic path.
        "tc_args_str_multibyte_literal",
        // Duplicate keys are all kept.
        "tc_args_str_duplicate_keys",
        // Wrong-TYPE arguments raise, unlike a non-object argument STRING.
        "tc_arguments_wrong_type",
    ];

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

    /// One assistant turn whose single tool call carries `arguments` verbatim,
    /// rendered to its joined text.
    fn render_tool_arguments(args: serde_json::Value) -> Result<String> {
        let messages = json!([
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c", "function": {"name": "f", "arguments": args}}]},
        ]);
        Ok(render_segments(&messages, None, &RenderOpts::default())?
            .iter()
            .map(|s| s.text.as_str())
            .collect())
    }

    /// A JSON-STRING `arguments` is parsed one level deep and every value keeps
    /// its ORIGINAL literal text. Re-serializing a parsed value instead agrees
    /// with the reference on canonical `json.dumps` spacing and ONLY on that —
    /// which is why a battery written in canonical spacing cannot tell the two
    /// spellings apart. Every case here is one re-serializing gets wrong.
    #[test]
    fn tool_argument_string_is_parsed_one_level_deep() {
        let render = |args: &str| render_tool_arguments(json!(args)).expect("renders");

        for (args, want) in [
            (
                r#"{"exp":1e2}"#,
                r#"key="exp" type="number"<|sep|>1e2<|close|>"#,
            ),
            (
                r#"{"arr":[1,2]}"#,
                r#"key="arr" type="array"<|sep|>[1,2]<|close|>"#,
            ),
            (
                r#"{"t":1.50}"#,
                r#"key="t" type="number"<|sep|>1.50<|close|>"#,
            ),
            (
                r#"{"a": { "b" : 1 }}"#,
                r#"key="a" type="object"<|sep|>{ "b" : 1 }<|close|>"#,
            ),
            // Bytes after the closing brace are discarded rather than making the
            // whole string unparsable.
            (
                r#"{"a":1} junk"#,
                r#"key="a" type="number"<|sep|>1<|close|>"#,
            ),
            // A string VALUE is the one that IS decoded, not passed through.
            (
                r#"{"s":"a\nb"}"#,
                "key=\"s\" type=\"string\"<|sep|>a\nb<|close|>",
            ),
        ] {
            let text = render(args);
            assert!(
                text.contains(want),
                "for {args}: wanted {want:?} in {text:?}"
            );
        }

        // Duplicate keys are ALL kept; collecting into a map drops the first.
        let dup = render(r#"{"a":1,"a":2}"#);
        assert_eq!(
            dup.matches(r#"argument key="a""#).count(),
            2,
            "both duplicate keys must render: {dup}"
        );

        // The MAPPING path is UNCHANGED, so the same logical arguments render
        // with Python's DEFAULT separators. The asymmetry is the reference's.
        let mapping = render_tool_arguments(json!({"arr": [1, 2]})).expect("renders");
        assert!(
            mapping.contains(r#"key="arr" type="array"<|sep|>[1, 2]<|close|>"#),
            "a mapping is re-serialized with a space after the comma: {mapping}"
        );
    }

    /// Every string that is not a well-formed JSON object falls back to the raw
    /// `<json>` block. Valid non-object JSON (`[1,2]`, `7`) used to be a hard
    /// ERROR, which failed the whole request over a tool call the engine renders
    /// fine — and whitespace-only used to render as NO arguments, silently
    /// dropping the block.
    #[test]
    fn tool_argument_string_falls_back_instead_of_erroring() {
        for args in ["[1,2]", "7", "{oops", r#"{"a":1,}"#, "   "] {
            let text = render_tool_arguments(json!(args)).expect("never errors");
            assert!(
                text.contains(r#"json type="object""#),
                "{args:?} must land in a <json> block: {text:?}"
            );
            assert!(
                text.contains(args),
                "{args:?} must be preserved verbatim: {text:?}"
            );
        }

        // Only the EMPTY string means "no arguments" — and it emits no block.
        let empty = render_tool_arguments(json!("")).expect("renders");
        assert!(!empty.contains("argument key="), "no arguments: {empty:?}");
        assert!(
            !empty.contains(r#"json type="object""#),
            "and no json block: {empty:?}"
        );
    }

    /// A message that is not a mapping, and a role the encoder does not know,
    /// are REFUSED. Both used to render nothing for that turn and report
    /// success, so a batched call or a typo'd role silently produced a prompt
    /// the engine never emits — and the router then routed on it.
    ///
    /// The INDEX is asserted, not just the variant. Every refusal fixture used
    /// to sit at index 0, which left `if index == 0 { refuse } else { continue }`
    /// green — and that mutation restores the original bug for every real
    /// conversation, since a stray turn is never the first one.
    #[test]
    fn malformed_messages_and_unknown_roles_are_refused() {
        let opts = RenderOpts::default();
        let cases: Vec<(serde_json::Value, RenderErr)> = vec![
            (
                json!([{"role": "user", "content": "a"}, {"role": "developer", "content": "x"}]),
                RenderErr::UnknownRole {
                    index: 1,
                    role: "developer".into(),
                },
            ),
            (
                json!([{"role": "user", "content": "a"}, [{"role": "user", "content": "b"}]]),
                RenderErr::MessageNotAMapping {
                    index: 1,
                    found: "list",
                },
            ),
            (
                json!([{"content": "aside"}]),
                RenderErr::UnknownRole {
                    index: 0,
                    role: String::new(),
                },
            ),
            (
                json!([{"role": 123, "content": "aside"}]),
                RenderErr::UnknownRole {
                    index: 0,
                    role: String::new(),
                },
            ),
            (json!("hi"), RenderErr::MessagesNotAList { found: "str" }),
            (
                json!({"role": "user", "content": "q"}),
                RenderErr::MessagesNotAList { found: "dict" },
            ),
        ];
        for (messages, want) in &cases {
            let err = render_segments(messages, None, &opts)
                .expect_err("the reference refuses this, so the router must too");
            assert_eq!(&err, want, "wrong refusal for {messages}");
        }
    }

    /// Roles are compared after lowercasing, because the ENGINE lowercases before
    /// its encoder ever sees them (`_normalize_role`). The reference cannot pin
    /// this — hand it `"System"` directly and it raises — so the oracle is an
    /// equivalence: a case-variant role must render byte-identically to its
    /// lowercase spelling. Without it the router refuses `{"role":"Assistant"}`,
    /// which sglang serves normally.
    #[test]
    fn case_variant_roles_render_as_their_lowercase_spelling() {
        let opts = RenderOpts::default();
        let convo = |role: &str| {
            json!([
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c", "function": {"name": "f", "arguments": {}}}]},
                {"role": role, "content": "r", "tool": "f"},
            ])
        };
        for (variant, canonical) in [("Tool", "tool"), ("TOOL", "tool")] {
            assert_eq!(
                render_segments(&convo(variant), None, &opts).expect("renders"),
                render_segments(&convo(canonical), None, &opts).expect("renders"),
                "role {variant:?} must render as {canonical:?}"
            );
        }
        for (variant, canonical) in [("System", "system"), ("Assistant", "assistant")] {
            let mixed =
                json!([{"role": variant, "content": "s"}, {"role": "user", "content": "q"}]);
            let lower =
                json!([{"role": canonical, "content": "s"}, {"role": "user", "content": "q"}]);
            assert_eq!(
                render_segments(&mixed, None, &opts).expect("renders"),
                render_segments(&lower, None, &opts).expect("renders"),
                "role {variant:?} must render as {canonical:?}"
            );
        }
    }

    /// The four MEASURED divergences from the reference, pinned as our own
    /// behaviour so a future "fix" is a red test rather than a silent change in
    /// rendered bytes. The reference PARSES all of these into `<argument>` tags;
    /// serde_json refuses them, so we emit the raw block. Counts and reasoning
    /// are on `parse_arguments_object`.
    #[test]
    fn known_divergences_fall_back_to_the_raw_json_block() {
        let deep = format!("{{\"a\":{}1{}}}", "[".repeat(130), "]".repeat(130));
        let control = "{\"a\":\"x\u{1}y\"}".to_string();
        for args in [
            control.as_str(),
            // `json.dumps` emits these by default, so a Python client reaches them.
            "{\"a\":NaN}",
            "{\"a\":Infinity}",
            "{\"a\":1e309}",
            // A lone surrogate has no Rust `String` encoding at all.
            "{\"a\":\"\\ud800\"}",
            deep.as_str(),
        ] {
            let text = render_tool_arguments(json!(args)).expect("never errors");
            assert!(
                text.contains(r#"json type="object""#),
                "{args:?} must fall back to the raw block: {text:?}"
            );
        }
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
        let opts = resolve_render_opts(&json!({ "thinking_effort": null }));
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
