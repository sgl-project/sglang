// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! DeepSeek-V4 prompt encoder for cache-aware routing.
//!
//! DeepSeek-V4 ships no Jinja chat template; the engine builds the prompt in
//! code (`python/sglang/srt/entrypoints/openai/encoding_dsv4.py`, selected for
//! the `DeepseekV4` architecture). So to make the router's query tokens match
//! the engine's cached blocks, this reproduces that encoder's output for the
//! routing-relevant subset.
//!
//! # Scope
//!
//! Text content, both chat (non-thinking) and thinking mode — the engine picks
//! per request from `chat_template_kwargs.thinking`, falling back to its
//! `SGLANG_DEFAULT_THINKING`; [`resolve_render_opts`] mirrors that so the router
//! matches whichever mode the engine runs. Chat mode emits a user turn as
//! `BOS <｜User｜> content <｜Assistant｜> </think>`; thinking mode opens the
//! assistant with `<think>` and renders a kept prior assistant turn's
//! `reasoning_content` as `<think>…</think>` (subject to the drop-thinking rules —
//! prior reasoning is dropped without tools, kept with). Preprocessing mirrors
//! `serving_chat.py`/`encode_messages`: an empty system message is inserted when
//! the first message isn't a system message, and `merge_tool_messages` folds
//! `tool` messages into the following user turn (as `<tool_result>` blocks) and
//! coalesces consecutive user turns. Tools, tool calls, and tool results all
//! render the engine's way — the request's `tools` after the system content (see
//! [`render_tools`]), an assistant turn's `tool_calls` as a `DSML` block (see
//! [`render_tool_calls`]), and multiple results ordered by their originating
//! call (see [`sort_tool_results_by_call_order`]). Byte-exactness here is what
//! lets a tool-carrying request's block hashes match the engine's cached blocks
//! instead of diverging from the first block (tools) or the first assistant turn
//! (thinking) and routing by min-load. `tasks` stay out of scope — they drive
//! internal task-classification rendering and never appear on router traffic.
//!
//! Tokenization does not auto-prepend special tokens (the `dynamo_tokenizers`
//! HF wrapper hardcodes `add_special_tokens = false`;
//! [`super::adapter::encode`] adds none of its own), so the literal marker text
//! below is what maps to the special token ids. Pinned byte-exact against the
//! live engine's `/tokenize` (DeepSeek-V4-Flash, snapshot `6976c7ff`):
//! `[{user:"ABCD"}]` → `[0, 128803, 51453, 128804, 128822]`.

use serde::Serialize;

/// Beginning-of-sequence marker (token id 0).
const BOS: &str = "<｜begin▁of▁sentence｜>";
/// End-of-sequence marker, closing each prior assistant turn (token id 1).
const EOS: &str = "<｜end▁of▁sentence｜>";
/// User-turn marker (token id 128803).
const USER: &str = "<｜User｜>";
/// Assistant-turn marker, opening the generation prompt (token id 128804).
const ASSISTANT: &str = "<｜Assistant｜>";
/// Thinking-start marker (`encoding_dsv4.thinking_start_token`); a thinking-mode
/// generation prompt / historical assistant turn opens with it.
const THINK_START: &str = "<think>";
/// Thinking-end marker; the chat-mode generation prompt ends with it (128822).
const THINK_END: &str = "</think>";
/// DSML block token wrapping tool-call / tools markup (`encoding_dsv4.dsml_token`).
const DSML: &str = "｜DSML｜";

/// Reasoning-effort level, mirroring `encoding_dsv4`'s `reasoning_effort`
/// (`"max"` / `"high"` / `None`). Only `Max` alters the prompt — it prepends the
/// [`REASONING_EFFORT_MAX`] preamble at the very front in thinking mode. `High` is
/// kept distinct (not collapsed into `None`) intentionally: it is a lossless 1:1
/// mirror of the engine's own tri-state, so a future engine build that gives
/// `high` its own rendering only touches [`render_one`], not this type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReasoningEffort {
    None,
    High,
    Max,
}

/// How to render, mirroring the engine's per-request `thinking_mode` +
/// `reasoning_effort`. The engine derives these from the request's
/// `chat_template_kwargs.thinking` / `reasoning_effort` (falling back to its
/// `SGLANG_DEFAULT_THINKING` / `SGLANG_DSV4_REASONING_EFFORT` defaults); the
/// router mirrors that resolution in [`resolve_render_opts`] so its routing
/// tokens match the engine's cached blocks even when the engine runs a
/// non-default (e.g. thinking-on) mode. `RenderOpts::chat()` reproduces the
/// chat-mode encoder output byte-for-byte.
#[derive(Clone, Copy, Debug)]
pub struct RenderOpts {
    /// `true` = thinking mode (engine `thinking_mode == "thinking"`).
    pub thinking: bool,
    pub reasoning_effort: ReasoningEffort,
}

impl RenderOpts {
    /// Chat / non-thinking mode with no effort preamble — the engine default,
    /// byte-identical to the chat-mode encoder output.
    pub const fn chat() -> Self {
        RenderOpts {
            thinking: false,
            reasoning_effort: ReasoningEffort::None,
        }
    }
}

/// Resolve the render options for a request, mirroring the engine's dsv4 request
/// normalization. Reasoning effort follows the engine precedence
/// (`serving_chat._convert_to_internal_request` pops `chat_template_kwargs.reasoning_effort`
/// onto the top-level field before the encoder reads it):
/// `chat_template_kwargs.reasoning_effort` > top-level `reasoning_effort` > env
/// default; only `max`/`high` alter the prompt. Thinking follows the engine's
/// `chat_template_kwargs.thinking` truthiness (`serving_chat.py`), else the router's
/// env default.
///
/// The router is a separate process from the engine, so it cannot observe an
/// engine-side default (`SGLANG_DEFAULT_THINKING` / `--default-chat-template-kwargs`)
/// a request doesn't carry; the env defaults (`SGLANG_ROUTER_DSV4_DEFAULT_THINKING`
/// / `SGLANG_ROUTER_DSV4_REASONING_EFFORT`, read once) MUST be set to match the
/// engine's, or cache-aware routing degrades (best-effort, never incorrect — the
/// engine re-tokenizes for correctness, see `input_ids_safe_to_forward`). Deeper
/// `protocol.py` normalizations (the `reasoning` object's `effort` alias,
/// `reasoning_effort="none"` force-disabling thinking, `enable_thinking`) are NOT
/// mirrored — they too only degrade routing. `enable_thinking` is intentionally
/// not read: the dsv4 engine path keys solely on `chat_template_kwargs.thinking`.
pub fn resolve_render_opts(request: &serde_json::Value) -> RenderOpts {
    let ctk = request.get("chat_template_kwargs");

    let effort = ctk
        .and_then(|k| k.get("reasoning_effort"))
        .and_then(|v| v.as_str())
        .or_else(|| request.get("reasoning_effort").and_then(|v| v.as_str()))
        .map(str::to_owned)
        .or_else(|| default_reasoning_effort().clone());
    let reasoning_effort = match effort.as_deref() {
        Some("max") => ReasoningEffort::Max,
        Some("high") => ReasoningEffort::High,
        _ => ReasoningEffort::None,
    };

    let thinking = ctk
        .and_then(|k| k.get("thinking"))
        .map(json_truthy)
        .unwrap_or_else(default_thinking);

    RenderOpts {
        thinking,
        reasoning_effort,
    }
}

/// Python-truthiness of a JSON value, mirroring the engine's `if thinking_requested`
/// test on the raw `chat_template_kwargs.thinking` value (`serving_chat.py`): a bool
/// as-is, a string/array/object truthy iff non-empty, a number iff non-zero, null
/// falsy. A conformant client sends a bool; this only matters for odd payloads, and
/// keeps routing matching the engine on them.
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

/// Parse a boolean env value the way the engine's `EnvBool` does (`environ.py`):
/// case-insensitive `true`/`1`/`yes`/`y` → `Some(true)`, `false`/`0`/`no`/`n` →
/// `Some(false)`, anything else `None`. Matching the engine's exact token set is
/// load-bearing: `y` is engine-true, so a set that omitted it would silently
/// disagree with a `SGLANG_DEFAULT_THINKING=y` engine.
fn parse_env_bool(value: &str) -> Option<bool> {
    match value.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "y" => Some(true),
        "false" | "0" | "no" | "n" => Some(false),
        _ => None,
    }
}

/// Resolve the thinking default from the env value (pure; the env read + one-time
/// caching + logging live in [`router_defaults`]). Mirrors the engine's
/// `EnvBool` + `EnvField.get`: a recognized token → that bool; a non-empty
/// unrecognized value → WARN + `false` — the engine behaves identically (its
/// `EnvField.get` catches `EnvBool.parse`'s `ValueError`, warns, and returns the
/// `False` default; it does NOT surface a hard error). Unset/empty → `false`.
fn resolve_default_thinking(env: Option<&str>) -> bool {
    match env {
        Some(s) if !s.is_empty() => parse_env_bool(s).unwrap_or_else(|| {
            tracing::warn!(
                value = %s,
                "SGLANG_ROUTER_DSV4_DEFAULT_THINKING is not a recognized boolean \
                 (true/1/yes/y | false/0/no/n); using false (the engine warns + defaults \
                 false the same way) — dsv4 routing renders chat mode, mismatching a thinking engine"
            );
            false
        }),
        _ => false,
    }
}

/// Resolve the reasoning-effort default from the env value (pure). Only `max`/`high`
/// affect rendering; anything else collapses to `None` silently — matching the
/// engine, which keeps only `max`/`high` (`serving_chat.py`) and does not warn.
fn resolve_default_effort(env: Option<&str>) -> Option<String> {
    match env {
        Some("max") => Some("max".to_owned()),
        Some("high") => Some("high".to_owned()),
        _ => None,
    }
}

/// The router's render defaults, resolved once from env
/// (`SGLANG_ROUTER_DSV4_DEFAULT_THINKING` / `SGLANG_ROUTER_DSV4_REASONING_EFFORT`)
/// and logged once at INFO so an operator can confirm they match the engine's
/// `SGLANG_DEFAULT_THINKING` / `SGLANG_DSV4_REASONING_EFFORT`. Both are logged
/// together because an effort mismatch is a LARGER divergence than a thinking one
/// (the `max` preamble sits at block 0, so a mismatch shifts every block hash, not
/// just the trailing token) — it must be equally visible.
fn router_defaults() -> &'static (bool, Option<String>) {
    static V: std::sync::OnceLock<(bool, Option<String>)> = std::sync::OnceLock::new();
    V.get_or_init(|| {
        let thinking = resolve_default_thinking(
            std::env::var("SGLANG_ROUTER_DSV4_DEFAULT_THINKING")
                .ok()
                .as_deref(),
        );
        let effort = resolve_default_effort(
            std::env::var("SGLANG_ROUTER_DSV4_REASONING_EFFORT")
                .ok()
                .as_deref(),
        );
        tracing::info!(
            default_thinking = thinking,
            default_reasoning_effort = effort.as_deref().unwrap_or("(none)"),
            "dsv4 router render defaults resolved; must match the engine's SGLANG_DEFAULT_THINKING \
             / SGLANG_DSV4_REASONING_EFFORT (i.e. --default-chat-template-kwargs) for cache-aware \
             routing to match"
        );
        (thinking, effort)
    })
}

fn default_thinking() -> bool {
    router_defaults().0
}

fn default_reasoning_effort() -> &'static Option<String> {
    &router_defaults().1
}

/// The `encoding_dsv4.REASONING_EFFORT_MAX` preamble, emitted at the very front
/// of the prompt (after BOS, before the system content) only in thinking mode
/// with `reasoning_effort == Max`. Kept byte-identical to the engine.
const REASONING_EFFORT_MAX: &str = "Reasoning Effort: Absolute maximum with no shortcuts permitted.\nYou MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\nExplicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";

/// Render `messages` (+ the request's top-level `tools`) into the DeepSeek-V4
/// chat prompt for routing.
///
/// Mirrors `encoding_dsv4.encode_messages` for the routing subset (chat and
/// thinking mode, text content), including tool calls, tool results, and per-turn
/// reasoning. `messages` is the request's `messages` array; `tools` is the
/// request's top-level `tools` array (OpenAI format), or `None`; `opts` carries
/// the resolved thinking mode / reasoning effort. Non-array `messages` renders to
/// just the BOS marker (the caller then tokenizes it and, finding no useful
/// prefix, degrades to min-load like any short prompt).
///
/// Byte-exactness with the engine is what lets a request's block hashes match
/// the engine's cached blocks. These line up with `encoding_dsv4`: tools
/// render right after the system content (where `serving_chat` attaches
/// `request.tools`); a `tool` message folds into the following user turn as a
/// `<tool_result>` block (`merge_tool_messages`); an assistant turn's
/// `tool_calls` render as a `DSML` block; multiple tool results in one turn are
/// ordered by their originating call (`sort_tool_results_by_call_order`); and in
/// thinking mode the `<think>` transitions and prior-turn `reasoning_content`
/// render per the engine's `drop_thinking` rules (see [`render_one`],
/// [`drop_thinking_messages`]). `tasks` stay out of scope — they never appear on
/// router traffic.
pub fn render_messages(
    messages: &serde_json::Value,
    tools: Option<&serde_json::Value>,
    opts: RenderOpts,
) -> String {
    let raw = messages.as_array().map(Vec::as_slice).unwrap_or(&[]);
    let mut msgs = merge_tool_messages(raw);

    // The engine inserts an empty system message when the first message isn't a
    // system message; it renders to nothing but keeps the index logic aligned
    // (and is where tools attach). Doing it after the merge is equivalent — a
    // system turn never joins a user run.
    if msgs.first().map(|m| m.role != "system").unwrap_or(true) {
        msgs.insert(0, MergedMsg::plain("system"));
    }

    sort_tool_results_by_call_order(&mut msgs);

    // The engine attaches `request.tools` to `messages[0]` (the system message,
    // always present after the insertion above) and renders them immediately
    // after the system content. An empty `tools` array is falsy engine-side, so
    // treat it as no tools.
    let tool_list = tools
        .and_then(|t| t.as_array())
        .filter(|arr| !arr.is_empty());

    // Mirror `encode_messages`' drop resolution: the engine calls it with the
    // default `drop_thinking = true`, then forces it OFF when any message carries
    // tools (`effective_drop_thinking`). So: no tools → drop earlier turns'
    // reasoning; tools present → keep it (DeepSeek requires prior reasoning in the
    // context of a tool-calling multi-turn conversation).
    let effective_drop_thinking = tool_list.is_none();
    if opts.thinking && effective_drop_thinking {
        msgs = drop_thinking_messages(msgs);
    }
    let last_user_idx = last_user_index(&msgs);

    let mut out = String::from(BOS);
    // Reasoning-effort preamble sits at the very front (after BOS, before the
    // system content), thinking mode + `max` only — `encoding_dsv4.render_message`
    // index 0.
    if opts.thinking && opts.reasoning_effort == ReasoningEffort::Max {
        out.push_str(REASONING_EFFORT_MAX);
    }
    for i in 0..msgs.len() {
        render_one(
            i,
            &msgs,
            &mut out,
            opts,
            last_user_idx,
            effective_drop_thinking,
        );
        if i == 0 {
            if let Some(list) = tool_list {
                out.push_str("\n\n");
                out.push_str(&render_tools(list));
            }
        }
    }
    out
}

/// Index of the last `user`/`developer` message (the engine's
/// `find_last_user_index`), or `-1` when there is none. `i64` mirrors the
/// engine's `-1` sentinel so the `index >= last_user_idx` transition comparisons
/// are exact.
fn last_user_index(msgs: &[MergedMsg]) -> i64 {
    for i in (0..msgs.len()).rev() {
        if matches!(msgs[i].role.as_str(), "user" | "developer") {
            return i as i64;
        }
    }
    -1
}

/// Mirror `encoding_dsv4._drop_thinking_messages` (applied only in thinking mode
/// when dropping is in effect, i.e. no tools): messages at/after the last user
/// turn are kept verbatim; before it, `user`/`system`/`tool`/`latest_reminder`/
/// `direct_search_results` pass through, an assistant turn keeps everything but its
/// `reasoning_content`, and a `developer` (or any other) turn is dropped entirely.
fn drop_thinking_messages(msgs: Vec<MergedMsg>) -> Vec<MergedMsg> {
    let last_user = last_user_index(&msgs);
    let mut out = Vec::with_capacity(msgs.len());
    for (idx, mut m) in msgs.into_iter().enumerate() {
        let keep = matches!(
            m.role.as_str(),
            "user" | "system" | "tool" | "latest_reminder" | "direct_search_results"
        );
        if keep || idx as i64 >= last_user {
            out.push(m);
        } else if m.role == "assistant" {
            m.reasoning_content.clear();
            out.push(m);
        }
        // developer / other roles before the last user turn are dropped.
    }
    out
}

/// A tool call on an assistant turn, in the fields DSML rendering needs.
struct ToolCall {
    name: String,
    /// The OpenAI `arguments` — spec'd as a JSON string, but the type also
    /// permits an inlined object — kept raw and decoded at render time by
    /// [`encode_arguments_to_dsml`] (which mirrors the engine's `json.loads` +
    /// wrap-on-failure).
    arguments: serde_json::Value,
    /// OpenAI `id` (falling back to `function.id`); orders tool results.
    id: String,
}

/// One piece of a merged user turn: literal text, or a folded-in tool result.
enum Block {
    Text(String),
    ToolResult {
        content: String,
        tool_use_id: String,
    },
}

/// A message after `merge_tool_messages`. `blocks` is `Some` only for user turns
/// (their text + folded-in tool results); `tool_calls` is non-empty only for
/// assistant turns; `reasoning_content` is a prior assistant turn's thinking
/// block, rendered only in thinking mode (see [`render_one`]).
struct MergedMsg {
    role: String,
    content: String,
    tool_calls: Vec<ToolCall>,
    blocks: Option<Vec<Block>>,
    reasoning_content: String,
}

impl MergedMsg {
    fn plain(role: &str) -> Self {
        MergedMsg {
            role: role.to_string(),
            content: String::new(),
            tool_calls: Vec::new(),
            blocks: None,
            reasoning_content: String::new(),
        }
    }
}

/// Index of the trailing message iff it is a user turn already carrying blocks —
/// i.e. one a following tool/user message should fold into.
fn open_user_idx(merged: &[MergedMsg]) -> Option<usize> {
    match merged.last() {
        Some(last) if last.role == "user" && last.blocks.is_some() => Some(merged.len() - 1),
        _ => None,
    }
}

/// Mirror `encoding_dsv4.merge_tool_messages`: DeepSeek-V4 has no standalone
/// `tool` role, so a tool message folds into the preceding (or a fresh) user
/// turn as a `tool_result` block, and consecutive user turns coalesce into one
/// with a text block each. Other roles pass through unchanged. (The engine's
/// `task is None` guard on user-merge is irrelevant here — tasks are out of
/// scope, so a user run always coalesces.)
fn merge_tool_messages(raw: &[serde_json::Value]) -> Vec<MergedMsg> {
    let mut merged: Vec<MergedMsg> = Vec::with_capacity(raw.len());
    for m in raw {
        let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("");
        match role {
            "tool" => {
                let block = Block::ToolResult {
                    content: tool_result_content(m.get("content")),
                    tool_use_id: str_field(m, "tool_call_id"),
                };
                match open_user_idx(&merged) {
                    Some(idx) => merged[idx].blocks.as_mut().unwrap().push(block),
                    None => {
                        let mut um = MergedMsg::plain("user");
                        um.blocks = Some(vec![block]);
                        merged.push(um);
                    }
                }
            }
            "user" => {
                let text = content_to_string(m.get("content"));
                match open_user_idx(&merged) {
                    Some(idx) => merged[idx].blocks.as_mut().unwrap().push(Block::Text(text)),
                    None => {
                        let mut um = MergedMsg::plain("user");
                        um.content = text.clone();
                        um.blocks = Some(vec![Block::Text(text)]);
                        merged.push(um);
                    }
                }
            }
            "assistant" => {
                let mut am = MergedMsg::plain("assistant");
                am.content = content_to_string(m.get("content"));
                am.tool_calls = parse_tool_calls(m.get("tool_calls"));
                // Prior-turn thinking block. The engine renders it verbatim as
                // `reasoning_content or ""` (a plain string), so pass it through
                // as-is; a missing/non-string value is treated as empty.
                am.reasoning_content = str_field(m, "reasoning_content");
                merged.push(am);
            }
            other => {
                let mut om = MergedMsg::plain(other);
                om.content = content_to_string(m.get("content"));
                merged.push(om);
            }
        }
    }
    merged
}

/// Extract an assistant turn's `tool_calls` into [`ToolCall`]s. Missing fields
/// degrade to empty rather than dropping the call — a malformed call still
/// contributes stable bytes to the hash. `id` falls back to `function.id` when
/// absent at the top level, matching the engine's sort-key extraction
/// (`tc.get("id") or tc.get("function",{}).get("id")`).
fn parse_tool_calls(v: Option<&serde_json::Value>) -> Vec<ToolCall> {
    let Some(arr) = v.and_then(|t| t.as_array()) else {
        return Vec::new();
    };
    arr.iter()
        .map(|tc| {
            let func = tc.get("function");
            let id = {
                let top = str_field(tc, "id");
                if top.is_empty() {
                    func.map(|f| str_field(f, "id")).unwrap_or_default()
                } else {
                    top
                }
            };
            ToolCall {
                name: func.map(|f| str_field(f, "name")).unwrap_or_default(),
                // Keep the raw value (string or inlined object);
                // `encode_arguments_to_dsml` reproduces the engine's handling of
                // both.
                arguments: func
                    .and_then(|f| f.get("arguments"))
                    .cloned()
                    .unwrap_or(serde_json::Value::Null),
                id,
            }
        })
        .collect()
}

/// Mirror `encoding_dsv4.sort_tool_results_by_call_order`: when a user turn
/// carries more than one tool result, order them by the position of their
/// originating call in the most recent assistant turn's `tool_calls`. Text
/// blocks keep their slots; only the tool-result slots are reordered among
/// themselves. A single tool result (or no preceding calls) is left untouched.
fn sort_tool_results_by_call_order(merged: &mut [MergedMsg]) {
    let mut order: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for m in merged.iter_mut() {
        if m.role == "assistant" && !m.tool_calls.is_empty() {
            order.clear();
            for (idx, tc) in m.tool_calls.iter().enumerate() {
                if !tc.id.is_empty() {
                    order.insert(tc.id.clone(), idx);
                }
            }
            continue;
        }
        let Some(blocks) = m.blocks.as_mut() else {
            continue;
        };
        // Capture the tool-result slot positions FIRST: the extraction below
        // swaps a `Text` placeholder into each, so a later `matches!(ToolResult)`
        // would no longer find them.
        let slots: Vec<usize> = blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| matches!(b, Block::ToolResult { .. }))
            .map(|(idx, _)| idx)
            .collect();
        if slots.len() <= 1 || order.is_empty() {
            continue;
        }
        // Pull the tool-result blocks out, stable-sort by call order (unknown
        // ids sort to 0, matching the engine's `.get(id, 0)`), drop them back
        // into the same slots — text blocks keep their positions.
        let mut results: Vec<Block> = slots
            .iter()
            .map(|&idx| std::mem::replace(&mut blocks[idx], Block::Text(String::new())))
            .collect();
        results.sort_by_key(|b| match b {
            Block::ToolResult { tool_use_id, .. } => *order.get(tool_use_id).unwrap_or(&0),
            Block::Text(_) => 0,
        });
        for (slot, b) in slots.into_iter().zip(results) {
            blocks[slot] = b;
        }
    }
}

/// Append merged message `i`'s encoded form to `out`.
fn render_one(
    i: usize,
    msgs: &[MergedMsg],
    out: &mut String,
    opts: RenderOpts,
    last_user_idx: i64,
    effective_drop_thinking: bool,
) {
    let m = &msgs[i];
    match m.role.as_str() {
        "system" => out.push_str(&m.content),
        "user" | "developer" => {
            out.push_str(USER);
            match &m.blocks {
                // A merged user turn renders its blocks joined by `\n\n`.
                Some(blocks) => {
                    let parts: Vec<String> = blocks.iter().map(render_block).collect();
                    out.push_str(&parts.join("\n\n"));
                }
                // `developer` (and any user that never went through the merge)
                // renders its bare content.
                None => out.push_str(&m.content),
            }
        }
        "assistant" => {
            // Thinking mode renders the turn's `reasoning_content` followed by
            // `</think>` before its content; the opening `<think>` came from the
            // preceding user turn's transition below. Kept when reasoning isn't
            // being dropped (tools present) OR this turn is strictly after the last
            // user turn — matching `encoding_dsv4.render_message` (`prev_has_task`
            // is always false here; tasks are out of scope). Chat mode emits none.
            if opts.thinking && (!effective_drop_thinking || i as i64 > last_user_idx) {
                out.push_str(&m.reasoning_content);
                out.push_str(THINK_END);
            }
            out.push_str(&m.content);
            if !m.tool_calls.is_empty() {
                out.push_str("\n\n");
                out.push_str(&render_tool_calls(&m.tool_calls));
            }
            out.push_str(EOS);
        }
        // Unknown roles aren't part of routing traffic; emit the content so a
        // stray role still contributes something rather than vanishing.
        _ => out.push_str(&m.content),
    }

    // Generation-prompt transition. The engine appends it only when this is the
    // last message OR the next message is an assistant/reminder turn, and only
    // for user/developer messages.
    let next_takes_transition = match msgs.get(i + 1) {
        Some(next) => next.role == "assistant" || next.role == "latest_reminder",
        None => true,
    };
    if next_takes_transition && (m.role == "user" || m.role == "developer") {
        out.push_str(ASSISTANT);
        // Chat mode always closes with `</think>`. Thinking mode opens the next
        // assistant turn with `<think>`: always when reasoning is kept (tools
        // present), else only at/after the last user turn (the current
        // generation) — mirroring `encoding_dsv4.render_message`.
        let token = if opts.thinking && (!effective_drop_thinking || i as i64 >= last_user_idx) {
            THINK_START
        } else {
            THINK_END
        };
        out.push_str(token);
    }
}

/// Render one merged-user block: text verbatim, a tool result wrapped in the
/// engine's `<tool_result>…</tool_result>` (`tool_output_template`).
fn render_block(b: &Block) -> String {
    match b {
        Block::Text(t) => t.clone(),
        Block::ToolResult { content, .. } => format!("<tool_result>{content}</tool_result>"),
    }
}

/// Render an assistant turn's tool calls as the engine's DSML block
/// (`tool_calls_template` wrapping one `tool_call_template` per call).
fn render_tool_calls(tool_calls: &[ToolCall]) -> String {
    let invokes = tool_calls
        .iter()
        .map(|tc| {
            format!(
                "<{DSML}invoke name=\"{}\">\n{}\n</{DSML}invoke>",
                tc.name,
                encode_arguments_to_dsml(&tc.arguments)
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!("<{DSML}tool_calls>\n{invokes}\n</{DSML}tool_calls>")
}

/// Encode a tool call's `arguments` into DSML `<parameter>` lines, mirroring
/// `encoding_dsv4.encode_arguments_to_dsml` for the case that matters: a valid
/// JSON-object string, whose keys each become a param — a string value raw with
/// `string="true"`, anything else the Python-`json.dumps` form with
/// `string="false"`. The other shapes are degenerate — an inlined object, an
/// unparsable string, or a parsed non-object — which a conformant client never
/// sends and the engine rejects (its exact error path varies by version), so the
/// router handles them defensively rather than reproducing that error: a
/// non-string value or unparsable string is wrapped as a single
/// `{"arguments": …}` param, and a parsed non-object yields none.
fn encode_arguments_to_dsml(arguments: &serde_json::Value) -> String {
    let parsed = match arguments {
        serde_json::Value::String(s) => serde_json::from_str::<serde_json::Value>(s)
            .unwrap_or_else(|_| serde_json::json!({ "arguments": s })),
        other => serde_json::json!({ "arguments": other }),
    };
    let Some(obj) = parsed.as_object() else {
        return String::new();
    };
    obj.iter()
        .map(|(k, v)| {
            let (is_str, value) = match v {
                serde_json::Value::String(s) => ("true", s.clone()),
                _ => ("false", py_json(v)),
            };
            format!("<{DSML}parameter name=\"{k}\" string=\"{is_str}\">{value}</{DSML}parameter>")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Flatten a `tool` message's `content` for a `<tool_result>` body: a string as
/// is; a parts array to its `text` parts joined with `\n\n` (non-text parts
/// become `[Unsupported <type>]`), mirroring the engine's list handling.
fn tool_result_content(content: Option<&serde_json::Value>) -> String {
    match content {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(parts)) => parts
            .iter()
            .map(|p| match p.get("text").and_then(|t| t.as_str()) {
                Some(t) => t.to_string(),
                None => format!(
                    "[Unsupported {}]",
                    p.get("type").and_then(|t| t.as_str()).unwrap_or("")
                ),
            })
            .collect::<Vec<_>>()
            .join("\n\n"),
        _ => String::new(),
    }
}

/// A message field as an owned string, empty when absent/non-string.
fn str_field(m: &serde_json::Value, key: &str) -> String {
    m.get(key)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

/// Flatten a message `content` field to a string: a plain string as-is; an
/// OpenAI parts array to its `type == "text"` parts joined with a single space
/// (mirroring `process_content_for_template_format(_, "string")`, which the
/// engine applies to dsv4 before encoding and which ignores non-text parts);
/// anything else to empty.
fn content_to_string(content: Option<&serde_json::Value>) -> String {
    match content {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(parts)) => parts
            .iter()
            .filter(|p| p.get("type").and_then(|t| t.as_str()) == Some("text"))
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(" "),
        _ => String::new(),
    }
}

/// Fixed tools-section text from the engine's `encoding_dsv4.TOOLS_TEMPLATE`
/// with the constant tokens (`dsml_token`, thinking start/end) already
/// substituted; the tool schemas are the only variable part and slot between
/// `TOOLS_PREFIX` and `TOOLS_SUFFIX`. Kept byte-identical to the engine so the
/// router's block hashes match its cached blocks.
const TOOLS_PREFIX: &str = "## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"$TOOL_NAME\">\n<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n...\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"$TOOL_NAME2\">\n...\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>\n\nString parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\nIf thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\nOtherwise, output directly after </think> with tool calls or final response.\n\n### Available Tool Schemas\n\n";
const TOOLS_SUFFIX: &str =
    "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n";

/// Render the request's top-level OpenAI `tools` array into the DeepSeek-V4
/// tools section, mirroring `encoding_dsv4.render_tools`: each tool's canonical
/// `function` (see [`canonical_function`]) is serialized Python-style and the
/// schemas are joined with `\n`, between the fixed preamble and trailer.
fn render_tools(tools: &[serde_json::Value]) -> String {
    let schemas = tools
        .iter()
        .map(|t| py_json(&canonical_function(t)))
        .collect::<Vec<_>>()
        .join("\n");
    format!("{TOOLS_PREFIX}{schemas}{TOOLS_SUFFIX}")
}

/// Reproduce the engine's `Function.model_dump()` (`serving_chat`) for one
/// OpenAI tool: take its `function` object and re-emit exactly
/// `description, name, parameters, strict` (plus `defer_loading` only when set),
/// injecting the pydantic defaults for the optional fields (`description` and
/// `parameters` → null, `strict` → false) and dropping unknown fields. `name`
/// has no engine-side default (a missing one is a 422), so an absent name emits
/// `null` here rather than being "defaulted". A tool-level `defer_loading` (a
/// GLM extension the engine propagates into the function) is out of scope — only
/// `function.defer_loading` is read. This canonical shape — not the raw client
/// object — is what the engine serializes, so the router must key on the same
/// bytes.
fn canonical_function(tool: &serde_json::Value) -> serde_json::Value {
    let func = tool.get("function");
    let field = |k: &str| func.and_then(|f| f.get(k)).cloned();
    let mut m = serde_json::Map::new();
    m.insert(
        "description".to_string(),
        field("description").unwrap_or(serde_json::Value::Null),
    );
    m.insert(
        "name".to_string(),
        field("name").unwrap_or(serde_json::Value::Null),
    );
    m.insert(
        "parameters".to_string(),
        field("parameters").unwrap_or(serde_json::Value::Null),
    );
    m.insert(
        "strict".to_string(),
        field("strict").unwrap_or(serde_json::Value::Bool(false)),
    );
    if let Some(dl) = field("defer_loading") {
        if !dl.is_null() {
            m.insert("defer_loading".to_string(), dl);
        }
    }
    serde_json::Value::Object(m)
}

/// Serialize `value` the way Python's `json.dumps(v, ensure_ascii=False)` does:
/// `", "` between elements/members and `": "` after each key (serde's compact
/// form omits those spaces, changing the bytes the engine's block hashes key
/// on). Key order and non-ASCII pass through unchanged (`ensure_ascii=False` +
/// the crate's `preserve_order` feature).
///
/// CAVEAT: number formatting matches Python for the ints/floats/bools that
/// appear in real tool schemas, but not universally — scientific-notation floats
/// (`1e-5` renders `0.00001` here vs Python `1e-05`) and integers beyond f64's
/// exact range diverge from Python's `repr`. Such a value in a tool schema would
/// miss the exact cache match and degrade to min-load; none appear in observed
/// (OpenCode) tool schemas, so this is left as a known edge rather than
/// reimplementing Python float formatting.
fn py_json(value: &serde_json::Value) -> String {
    let mut buf = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut buf, PyJsonFormatter);
    value
        .serialize(&mut serializer)
        .expect("serializing a serde_json::Value into a Vec is infallible");
    String::from_utf8(buf).expect("serde_json emits valid UTF-8")
}

/// serde_json formatter emitting Python `json.dumps` default separators
/// (`", "` / `": "`) instead of serde's compact `,` / `:`.
struct PyJsonFormatter;

impl serde_json::ser::Formatter for PyJsonFormatter {
    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        writer.write_all(b": ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn effort(s: Option<&str>) -> ReasoningEffort {
        match s {
            Some("max") => ReasoningEffort::Max,
            Some("high") => ReasoningEffort::High,
            _ => ReasoningEffort::None,
        }
    }

    /// Byte-exact parity against the engine's `encoding_dsv4.encode_messages` in
    /// BOTH chat and thinking mode, across fixtures generated from the engine
    /// encoder itself (transition token `<think>`/`</think>`, prior reasoning
    /// kept-with-tools vs dropped-without, the empty-reasoning `<think></think>`
    /// block, and the `reasoning_effort=max` front preamble). A mismatch here is
    /// exactly a router↔engine routing-tokenization divergence — the thing that
    /// collapses cache-aware routing on a thinking-mode engine.
    #[test]
    fn thinking_and_chat_parity_fixtures() {
        let raw = include_str!("testdata/dsv4_thinking_cases.json");
        let cases: Vec<serde_json::Value> = serde_json::from_str(raw).expect("fixture json parses");
        assert!(!cases.is_empty(), "fixtures present");
        for c in &cases {
            let name = c["name"].as_str().unwrap();
            let tools = c.get("tools").filter(|t| !t.is_null());
            let opts = RenderOpts {
                thinking: c["thinking"].as_bool().unwrap(),
                reasoning_effort: effort(c["reasoning_effort"].as_str()),
            };
            let got = render_messages(&c["messages"], tools, opts);
            assert_eq!(got, c["expected"].as_str().unwrap(), "case `{name}`");
        }
    }

    /// Byte-exact against the engine's `/tokenize`: a single user turn renders
    /// `BOS <｜User｜> content <｜Assistant｜> </think>`.
    #[test]
    fn single_user_turn() {
        let out = render_messages(
            &json!([{"role":"user","content":"ABCD"}]),
            None,
            RenderOpts::chat(),
        );
        assert_eq!(
            out,
            "<｜begin▁of▁sentence｜><｜User｜>ABCD<｜Assistant｜></think>"
        );
    }

    /// A leading system message renders as bare content (no marker), before the
    /// user turn.
    #[test]
    fn system_then_user() {
        let out = render_messages(
            &json!([
                {"role":"system","content":"SYS"},
                {"role":"user","content":"ABCD"}
            ]),
            None,
            RenderOpts::chat(),
        );
        assert_eq!(
            out,
            "<｜begin▁of▁sentence｜>SYS<｜User｜>ABCD<｜Assistant｜></think>"
        );
    }

    /// Multi-turn: each prior user turn gets the generation prompt, the prior
    /// assistant turn is closed by EOS. Matches the engine token stream
    /// `[0,128803,55,19,128804,128822,35,19,1,128803,55,20,128804,128822]`.
    #[test]
    fn multi_turn() {
        let out = render_messages(
            &json!([
                {"role":"user","content":"U1"},
                {"role":"assistant","content":"A1"},
                {"role":"user","content":"U2"}
            ]),
            None,
            RenderOpts::chat(),
        );
        assert_eq!(
            out,
            "<｜begin▁of▁sentence｜><｜User｜>U1<｜Assistant｜></think>A1<｜end▁of▁sentence｜><｜User｜>U2<｜Assistant｜></think>"
        );
    }

    /// An empty leading system message (already present) is not duplicated and
    /// renders to nothing — same result as a bare user turn.
    #[test]
    fn explicit_empty_system_is_not_duplicated() {
        let out = render_messages(
            &json!([
                {"role":"system","content":""},
                {"role":"user","content":"ABCD"}
            ]),
            None,
            RenderOpts::chat(),
        );
        assert_eq!(
            out,
            "<｜begin▁of▁sentence｜><｜User｜>ABCD<｜Assistant｜></think>"
        );
    }

    /// Multi-part text content flattens to its `type == "text"` parts joined
    /// with a single space (mirroring the engine's
    /// `process_content_for_template_format(_, "string")`), NOT concatenated.
    #[test]
    fn array_content_flattens_text_parts() {
        let out = render_messages(
            &json!([
                {"role":"user","content":[{"type":"text","text":"AB"},{"type":"text","text":"CD"}]}
            ]),
            None,
            RenderOpts::chat(),
        );
        assert_eq!(
            out,
            "<｜begin▁of▁sentence｜><｜User｜>AB CD<｜Assistant｜></think>"
        );
    }

    /// Consecutive user turns merge into one `<｜User｜>` turn joined with `\n\n`
    /// (the engine's `merge_tool_messages`), so only one user marker and one
    /// generation prompt are emitted — not a marker per message.
    #[test]
    fn consecutive_user_turns_merge() {
        let out = render_messages(
            &json!([
                {"role":"user","content":"U1"},
                {"role":"user","content":"U2"}
            ]),
            None,
            RenderOpts::chat(),
        );
        assert_eq!(
            out,
            "<｜begin▁of▁sentence｜><｜User｜>U1\n\nU2<｜Assistant｜></think>"
        );
    }

    /// A run of user turns split by an assistant turn does NOT merge across the
    /// assistant: each side is its own user turn.
    #[test]
    fn user_runs_do_not_merge_across_assistant() {
        let out = render_messages(
            &json!([
                {"role":"user","content":"U1"},
                {"role":"user","content":"U2"},
                {"role":"assistant","content":"A1"},
                {"role":"user","content":"U3"}
            ]),
            None,
            RenderOpts::chat(),
        );
        assert_eq!(
            out,
            "<｜begin▁of▁sentence｜><｜User｜>U1\n\nU2<｜Assistant｜></think>A1<｜end▁of▁sentence｜><｜User｜>U3<｜Assistant｜></think>"
        );
    }

    /// A `developer` turn renders identically to a user turn for text content
    /// (the engine nests the same `<｜User｜>` marker) and takes the generation
    /// prompt. Developer turns are not merged (only `user` runs merge), so two
    /// developers emit two markers.
    #[test]
    fn developer_role_renders_like_user_without_merging() {
        assert_eq!(
            render_messages(
                &json!([{"role":"developer","content":"D1"}]),
                None,
                RenderOpts::chat()
            ),
            "<｜begin▁of▁sentence｜><｜User｜>D1<｜Assistant｜></think>"
        );
        assert_eq!(
            render_messages(
                &json!([
                    {"role":"developer","content":"D1"},
                    {"role":"developer","content":"D2"}
                ]),
                None,
                RenderOpts::chat()
            ),
            "<｜begin▁of▁sentence｜><｜User｜>D1<｜User｜>D2<｜Assistant｜></think>"
        );
    }

    /// An empty messages list renders to just the BOS marker — the documented
    /// degrade path (the caller then routes by min-load on the empty prefix).
    #[test]
    fn empty_messages_renders_bos_only() {
        assert_eq!(
            render_messages(&json!([]), None, RenderOpts::chat()),
            "<｜begin▁of▁sentence｜>"
        );
    }

    /// `py_json` reproduces Python `json.dumps(v, ensure_ascii=False)` default
    /// separators (`", "` / `": "`) and preserves key order — the exact bytes the
    /// engine's `to_json` produces (serde's compact form would drop the spaces).
    #[test]
    fn py_json_uses_python_separators_and_preserves_order() {
        let v = json!({"b": 1, "a": [1, 2], "c": {"x": true}});
        assert_eq!(py_json(&v), r#"{"b": 1, "a": [1, 2], "c": {"x": true}}"#);
    }

    /// A tool's `function` is re-emitted as the engine's `Function.model_dump`:
    /// fixed order (description, name, parameters, strict), pydantic defaults
    /// injected for omitted fields, unknown fields dropped, then Python-serialized.
    #[test]
    fn canonical_function_reorders_and_injects_defaults() {
        // Client sends keys out of order, omits description/strict, adds an extra.
        let tool = json!({
            "type": "function",
            "function": {"name": "ping", "parameters": {"type": "object"}, "x-extra": 1}
        });
        assert_eq!(
            py_json(&canonical_function(&tool)),
            r#"{"description": null, "name": "ping", "parameters": {"type": "object"}, "strict": false}"#
        );
    }

    /// Byte-exact against the engine (`encode_messages`, chat mode): tools render
    /// right after the system content, each `function` canonicalized + serialized
    /// Python-style. Pinned against `encoding_dsv4.encode_messages` output.
    #[test]
    fn renders_tools_after_system_content() {
        let messages = json!([
            {"role":"system","content":"SYS"},
            {"role":"user","content":"hi"}
        ]);
        let tools = json!([
            {"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}
        ]);
        let expected = "<｜begin▁of▁sentence｜>SYS\n\n## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"$TOOL_NAME\">\n<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n...\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"$TOOL_NAME2\">\n...\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>\n\nString parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\nIf thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\nOtherwise, output directly after </think> with tool calls or final response.\n\n### Available Tool Schemas\n\n{\"description\": \"Get weather\", \"name\": \"get_weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}}, \"strict\": false}\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n<｜User｜>hi<｜Assistant｜></think>";
        assert_eq!(
            render_messages(&messages, Some(&tools), RenderOpts::chat()),
            expected
        );
    }

    /// An empty `tools` array is falsy engine-side — no tools block is rendered.
    #[test]
    fn empty_tools_array_renders_no_tools_block() {
        let messages = json!([{"role":"user","content":"hi"}]);
        assert_eq!(
            render_messages(&messages, Some(&json!([])), RenderOpts::chat()),
            "<｜begin▁of▁sentence｜><｜User｜>hi<｜Assistant｜></think>"
        );
    }

    /// With no system message the engine inserts an empty one and still renders
    /// the tools block (right after the empty system content, i.e. after BOS).
    #[test]
    fn renders_tools_when_no_system_message_present() {
        let messages = json!([{"role":"user","content":"hi"}]);
        let tools = json!([{"type":"function","function":{"name":"ping","description":"p"}}]);
        let out = render_messages(&messages, Some(&tools), RenderOpts::chat());
        assert!(
            out.starts_with("<｜begin▁of▁sentence｜>\n\n## Tools\n\n"),
            "tools block should follow the inserted empty system; got: {out}"
        );
        assert!(out.contains(
            r#"{"description": "p", "name": "ping", "parameters": null, "strict": false}"#
        ));
        assert!(out.ends_with("<｜User｜>hi<｜Assistant｜></think>"));
    }

    /// Byte-exact against the engine (`encode_messages`, chat mode): an assistant
    /// `tool_calls` turn renders the DSML block — string args raw with
    /// `string="true"`, others Python-serialized with `string="false"` — and a
    /// following `tool` message folds into the next user turn as a
    /// `<tool_result>` block.
    #[test]
    fn renders_assistant_tool_calls_and_tool_result() {
        let messages = json!([
            {"role":"user","content":"u1"},
            {"role":"assistant","content":"reading","tool_calls":[
                {"id":"c1","type":"function","function":{"name":"read","arguments":"{\"filePath\": \"/x\", \"limit\": 10, \"nested\": {\"a\": 1}}"}}
            ]},
            {"role":"tool","tool_call_id":"c1","content":"FILE"},
            {"role":"user","content":"u2"}
        ]);
        let expected = "<｜begin▁of▁sentence｜><｜User｜>u1<｜Assistant｜></think>reading\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"read\">\n<｜DSML｜parameter name=\"filePath\" string=\"true\">/x</｜DSML｜parameter>\n<｜DSML｜parameter name=\"limit\" string=\"false\">10</｜DSML｜parameter>\n<｜DSML｜parameter name=\"nested\" string=\"false\">{\"a\": 1}</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls><｜end▁of▁sentence｜><｜User｜><tool_result>FILE</tool_result>\n\nu2<｜Assistant｜></think>";
        assert_eq!(
            render_messages(&messages, None, RenderOpts::chat()),
            expected
        );
    }

    /// Byte-exact against the engine: multiple tool results in one user turn are
    /// reordered to the preceding assistant's `tool_calls` order. Results arrive
    /// b,a; the calls were a,b → rendered a,b.
    #[test]
    fn tool_results_sorted_by_call_order() {
        let messages = json!([
            {"role":"assistant","content":"","tool_calls":[
                {"id":"a","type":"function","function":{"name":"t1","arguments":"{}"}},
                {"id":"b","type":"function","function":{"name":"t2","arguments":"{}"}}
            ]},
            {"role":"tool","tool_call_id":"b","content":"RB"},
            {"role":"tool","tool_call_id":"a","content":"RA"}
        ]);
        let expected = "<｜begin▁of▁sentence｜>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"t1\">\n\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"t2\">\n\n</｜DSML｜invoke>\n</｜DSML｜tool_calls><｜end▁of▁sentence｜><｜User｜><tool_result>RA</tool_result>\n\n<tool_result>RB</tool_result><｜Assistant｜></think>";
        assert_eq!(
            render_messages(&messages, None, RenderOpts::chat()),
            expected
        );
    }

    /// Byte-exact against the engine: multiple tools render one canonical schema
    /// per line (`\n`-joined). Tool `b` omits everything but `name` (defaults
    /// injected: description/parameters → null) and sets `strict: true`.
    #[test]
    fn renders_multiple_tools() {
        let messages = json!([
            {"role":"system","content":"S"},
            {"role":"user","content":"hi"}
        ]);
        let tools = json!([
            {"type":"function","function":{"name":"a","description":"da","parameters":{"type":"object"}}},
            {"type":"function","function":{"name":"b","strict":true}}
        ]);
        let expected = "<｜begin▁of▁sentence｜>S\n\n## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"$TOOL_NAME\">\n<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n...\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"$TOOL_NAME2\">\n...\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>\n\nString parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\nIf thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\nOtherwise, output directly after </think> with tool calls or final response.\n\n### Available Tool Schemas\n\n{\"description\": \"da\", \"name\": \"a\", \"parameters\": {\"type\": \"object\"}, \"strict\": false}\n{\"description\": null, \"name\": \"b\", \"parameters\": null, \"strict\": true}\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n<｜User｜>hi<｜Assistant｜></think>";
        assert_eq!(
            render_messages(&messages, Some(&tools), RenderOpts::chat()),
            expected
        );
    }

    /// Byte-exact against the engine: an inlined-object `arguments` (permitted by
    /// the OpenAI type) wraps into a single `arguments` param, not expanded per
    /// key — mirroring the engine's `json.loads(<dict>)` TypeError → wrap.
    #[test]
    fn inlined_object_arguments_wrapped_like_engine() {
        let messages = json!([
            {"role":"assistant","content":"","tool_calls":[
                {"id":"c1","type":"function","function":{"name":"f","arguments":{"x":1,"y":"z"}}}
            ]},
            {"role":"tool","tool_call_id":"c1","content":"R"}
        ]);
        let expected = "<｜begin▁of▁sentence｜>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n<｜DSML｜parameter name=\"arguments\" string=\"false\">{\"x\": 1, \"y\": \"z\"}</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls><｜end▁of▁sentence｜><｜User｜><tool_result>R</tool_result><｜Assistant｜></think>";
        assert_eq!(
            render_messages(&messages, None, RenderOpts::chat()),
            expected
        );
    }

    /// Byte-exact against the engine: an unparsable `arguments` string wraps
    /// into a single `arguments` param carrying the raw string (`string="true"`),
    /// mirroring the engine's `json.loads` except-branch.
    #[test]
    fn invalid_json_arguments_wrapped_like_engine() {
        let messages = json!([
            {"role":"assistant","content":"","tool_calls":[
                {"id":"c1","type":"function","function":{"name":"f","arguments":"not json"}}
            ]},
            {"role":"tool","tool_call_id":"c1","content":"R"}
        ]);
        let expected = "<｜begin▁of▁sentence｜>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n<｜DSML｜parameter name=\"arguments\" string=\"true\">not json</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls><｜end▁of▁sentence｜><｜User｜><tool_result>R</tool_result><｜Assistant｜></think>";
        assert_eq!(
            render_messages(&messages, None, RenderOpts::chat()),
            expected
        );
    }

    /// `parse_env_bool` matches the engine's `EnvBool` token set exactly
    /// (`true/1/yes/y` | `false/0/no/n`, case-insensitive), and returns `None`
    /// for anything else — including `on`, which a hand-rolled set might wrongly
    /// accept and thereby diverge from the engine.
    #[test]
    fn parse_env_bool_matches_engine_token_set() {
        for t in ["true", "1", "yes", "y", "TRUE", "Yes", "Y"] {
            assert_eq!(parse_env_bool(t), Some(true), "{t}");
        }
        for f in ["false", "0", "no", "n", "FALSE", "No"] {
            assert_eq!(parse_env_bool(f), Some(false), "{f}");
        }
        for u in ["on", "off", "enabled", "t", "", "2"] {
            assert_eq!(parse_env_bool(u), None, "{u}");
        }
    }

    /// `json_truthy` mirrors Python truthiness on the raw `thinking` value.
    #[test]
    fn json_truthy_mirrors_python() {
        assert!(json_truthy(&json!(true)));
        assert!(!json_truthy(&json!(false)));
        assert!(json_truthy(&json!("true"))); // non-empty string is truthy
        assert!(json_truthy(&json!("false"))); // even "false" — non-empty
        assert!(!json_truthy(&json!("")));
        assert!(json_truthy(&json!(1)));
        assert!(!json_truthy(&json!(0)));
        assert!(!json_truthy(&serde_json::Value::Null));
        assert!(!json_truthy(&json!([])));
        assert!(json_truthy(&json!([1])));
        assert!(!json_truthy(&json!({})));
        assert!(json_truthy(&json!({ "k": 1 })));
    }

    /// `resolve_render_opts` honors per-request overrides (the paths that don't
    /// depend on env): `chat_template_kwargs.thinking` truthiness, and the effort
    /// precedence `chat_template_kwargs.reasoning_effort` > top-level > (env).
    #[test]
    fn resolve_render_opts_request_overrides() {
        let opts = resolve_render_opts(&json!({"chat_template_kwargs": {"thinking": true}}));
        assert!(opts.thinking);
        assert_eq!(opts.reasoning_effort, ReasoningEffort::None);

        assert!(
            !resolve_render_opts(&json!({"chat_template_kwargs": {"thinking": false}})).thinking
        );
        // non-bool thinking follows truthiness (matches the engine), not `.as_bool()`.
        assert!(
            resolve_render_opts(&json!({"chat_template_kwargs": {"thinking": "true"}})).thinking
        );
        assert!(
            !resolve_render_opts(&json!({"chat_template_kwargs": {"thinking": null}})).thinking
        );

        // chat_template_kwargs.reasoning_effort wins over the top-level field.
        assert_eq!(
            resolve_render_opts(&json!({
                "reasoning_effort": "high",
                "chat_template_kwargs": {"reasoning_effort": "max"}
            }))
            .reasoning_effort,
            ReasoningEffort::Max
        );
        // top-level reasoning_effort is honored when chat_template_kwargs lacks it.
        assert_eq!(
            resolve_render_opts(&json!({"reasoning_effort": "high"})).reasoning_effort,
            ReasoningEffort::High
        );
        // unknown effort collapses to None (matches the engine's max/high-only gate).
        assert_eq!(
            resolve_render_opts(&json!({"reasoning_effort": "medium"})).reasoning_effort,
            ReasoningEffort::None
        );
    }

    /// The env-default resolvers (the deploy-critical `SGLANG_ROUTER_DSV4_*` knobs):
    /// pure over the env value so the empty / unrecognized / valid branches are
    /// pinned without the process-global `OnceLock`.
    #[test]
    fn resolve_default_thinking_branches() {
        assert!(!resolve_default_thinking(None)); // unset → false (engine default)
        assert!(!resolve_default_thinking(Some(""))); // set-empty → false
        assert!(resolve_default_thinking(Some("true")));
        assert!(resolve_default_thinking(Some("y"))); // engine EnvBool token
        assert!(!resolve_default_thinking(Some("false")));
        assert!(!resolve_default_thinking(Some("enabled"))); // unrecognized → false (WARNs)
    }

    #[test]
    fn resolve_default_effort_branches() {
        assert_eq!(resolve_default_effort(None), None);
        assert_eq!(resolve_default_effort(Some("max")), Some("max".to_owned()));
        assert_eq!(
            resolve_default_effort(Some("high")),
            Some("high".to_owned())
        );
        assert_eq!(resolve_default_effort(Some("medium")), None); // engine keeps only max/high
        assert_eq!(resolve_default_effort(Some("")), None);
    }
}
