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
//! Text content, chat (non-thinking) mode — the engine default
//! (`SGLANG_DEFAULT_THINKING=false`). For a user turn the engine emits
//! `BOS <｜User｜> content <｜Assistant｜> </think>`. Two preprocessing steps mirror
//! `serving_chat.py`/`encode_messages`: an empty system message is inserted when
//! the first message isn't a system message, and consecutive user turns are
//! merged into one (joined with `\n\n`). Tools, tasks, and per-turn reasoning
//! content are out of scope: the engine renders tools immediately after the
//! system content at the front of the prompt, so a tools-carrying request
//! diverges from the first block and routes by min-load; tasks alter only the
//! trailing turn transition; reasoning content is never emitted in chat mode,
//! so it causes no divergence.
//!
//! Tokenization does not auto-prepend special tokens (the `dynamo_tokenizers`
//! HF wrapper hardcodes `add_special_tokens = false`;
//! [`super::adapter::encode`] adds none of its own), so the literal marker text
//! below is what maps to the special token ids. Pinned byte-exact against the
//! live engine's `/tokenize` (DeepSeek-V4-Flash, snapshot `6976c7ff`):
//! `[{user:"ABCD"}]` → `[0, 128803, 51453, 128804, 128822]`.

/// Beginning-of-sequence marker (token id 0).
const BOS: &str = "<｜begin▁of▁sentence｜>";
/// End-of-sequence marker, closing each prior assistant turn (token id 1).
const EOS: &str = "<｜end▁of▁sentence｜>";
/// User-turn marker (token id 128803).
const USER: &str = "<｜User｜>";
/// Assistant-turn marker, opening the generation prompt (token id 128804).
const ASSISTANT: &str = "<｜Assistant｜>";
/// Thinking-end marker; the chat-mode generation prompt ends with it (128822).
const THINK_END: &str = "</think>";

/// Render `messages` into the DeepSeek-V4 chat prompt for routing.
///
/// Mirrors `encoding_dsv4.encode_messages` for the routing subset (chat mode,
/// text content, no tools/tasks). `messages` is the request's `messages` array;
/// non-array input renders to just the BOS marker (the caller then tokenizes it
/// and, finding no useful prefix, degrades to min-load like any short prompt).
pub fn render_messages(messages: &serde_json::Value) -> String {
    let mut msgs: Vec<(String, String)> = messages
        .as_array()
        .map(|arr| {
            arr.iter()
                .map(|m| {
                    let role = m
                        .get("role")
                        .and_then(|r| r.as_str())
                        .unwrap_or("")
                        .to_string();
                    (role, content_to_string(m.get("content")))
                })
                .collect()
        })
        .unwrap_or_default();

    // The engine inserts an empty system message when the first message isn't a
    // system message; it renders to nothing but keeps the index logic aligned.
    if msgs.first().map(|(r, _)| r != "system").unwrap_or(true) {
        msgs.insert(0, ("system".to_string(), String::new()));
    }

    merge_consecutive_user_turns(&mut msgs);

    let mut out = String::from(BOS);
    for i in 0..msgs.len() {
        render_one(i, &msgs, &mut out);
    }
    out
}

/// Collapse runs of consecutive `user` turns into one, joining their content
/// with `\n\n` — the engine merges them (`merge_tool_messages`) before encoding,
/// so back-to-back user messages must hash like the single turn it builds.
/// `developer` and other roles break a run and are left as-is.
fn merge_consecutive_user_turns(msgs: &mut Vec<(String, String)>) {
    let mut merged: Vec<(String, String)> = Vec::with_capacity(msgs.len());
    for (role, content) in msgs.drain(..) {
        match merged.last_mut() {
            Some((last_role, last_content)) if last_role == "user" && role == "user" => {
                last_content.push_str("\n\n");
                last_content.push_str(&content);
            }
            _ => merged.push((role, content)),
        }
    }
    *msgs = merged;
}

/// Append message `i`'s encoded form to `out`.
fn render_one(i: usize, msgs: &[(String, String)], out: &mut String) {
    let (role, content) = &msgs[i];
    match role.as_str() {
        "system" => out.push_str(content),
        "user" | "developer" => {
            out.push_str(USER);
            out.push_str(content);
        }
        "assistant" => {
            // Chat mode emits no reasoning block, so a prior assistant turn is
            // just its content closed by EOS.
            out.push_str(content);
            out.push_str(EOS);
        }
        // Unknown roles aren't part of routing traffic; emit the content so a
        // stray role still contributes something rather than vanishing.
        _ => out.push_str(content),
    }

    // Generation-prompt transition. The engine appends it only when this is the
    // last message OR the next message is an assistant/reminder turn, and only
    // for user/developer messages.
    let next_takes_transition = match msgs.get(i + 1) {
        Some((next_role, _)) => next_role == "assistant" || next_role == "latest_reminder",
        None => true,
    };
    if next_takes_transition && (role == "user" || role == "developer") {
        out.push_str(ASSISTANT);
        out.push_str(THINK_END);
    }
}

/// Flatten a message `content` field to a string: a plain string as-is, an
/// OpenAI parts array to its concatenated `text` parts, anything else to empty.
fn content_to_string(content: Option<&serde_json::Value>) -> String {
    match content {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect(),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Byte-exact against the engine's `/tokenize`: a single user turn renders
    /// `BOS <｜User｜> content <｜Assistant｜> </think>`.
    #[test]
    fn single_user_turn() {
        let out = render_messages(&json!([{"role":"user","content":"ABCD"}]));
        assert_eq!(
            out,
            "<｜begin▁of▁sentence｜><｜User｜>ABCD<｜Assistant｜></think>"
        );
    }

    /// A leading system message renders as bare content (no marker), before the
    /// user turn.
    #[test]
    fn system_then_user() {
        let out = render_messages(&json!([
            {"role":"system","content":"SYS"},
            {"role":"user","content":"ABCD"}
        ]));
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
        let out = render_messages(&json!([
            {"role":"user","content":"U1"},
            {"role":"assistant","content":"A1"},
            {"role":"user","content":"U2"}
        ]));
        assert_eq!(
            out,
            "<｜begin▁of▁sentence｜><｜User｜>U1<｜Assistant｜></think>A1<｜end▁of▁sentence｜><｜User｜>U2<｜Assistant｜></think>"
        );
    }

    /// An empty leading system message (already present) is not duplicated and
    /// renders to nothing — same result as a bare user turn.
    #[test]
    fn explicit_empty_system_is_not_duplicated() {
        let out = render_messages(&json!([
            {"role":"system","content":""},
            {"role":"user","content":"ABCD"}
        ]));
        assert_eq!(
            out,
            "<｜begin▁of▁sentence｜><｜User｜>ABCD<｜Assistant｜></think>"
        );
    }

    /// Array (multimodal) content flattens to its text parts.
    #[test]
    fn array_content_flattens_text_parts() {
        let out = render_messages(&json!([
            {"role":"user","content":[{"type":"text","text":"AB"},{"type":"text","text":"CD"}]}
        ]));
        assert_eq!(
            out,
            "<｜begin▁of▁sentence｜><｜User｜>ABCD<｜Assistant｜></think>"
        );
    }

    /// Consecutive user turns merge into one `<｜User｜>` turn joined with `\n\n`
    /// (the engine's `merge_tool_messages`), so only one user marker and one
    /// generation prompt are emitted — not a marker per message.
    #[test]
    fn consecutive_user_turns_merge() {
        let out = render_messages(&json!([
            {"role":"user","content":"U1"},
            {"role":"user","content":"U2"}
        ]));
        assert_eq!(
            out,
            "<｜begin▁of▁sentence｜><｜User｜>U1\n\nU2<｜Assistant｜></think>"
        );
    }

    /// A run of user turns split by an assistant turn does NOT merge across the
    /// assistant: each side is its own user turn.
    #[test]
    fn user_runs_do_not_merge_across_assistant() {
        let out = render_messages(&json!([
            {"role":"user","content":"U1"},
            {"role":"user","content":"U2"},
            {"role":"assistant","content":"A1"},
            {"role":"user","content":"U3"}
        ]));
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
            render_messages(&json!([{"role":"developer","content":"D1"}])),
            "<｜begin▁of▁sentence｜><｜User｜>D1<｜Assistant｜></think>"
        );
        assert_eq!(
            render_messages(&json!([
                {"role":"developer","content":"D1"},
                {"role":"developer","content":"D2"}
            ])),
            "<｜begin▁of▁sentence｜><｜User｜>D1<｜User｜>D2<｜Assistant｜></think>"
        );
    }

    /// An empty messages list renders to just the BOS marker — the documented
    /// degrade path (the caller then routes by min-load on the empty prefix).
    #[test]
    fn empty_messages_renders_bos_only() {
        assert_eq!(render_messages(&json!([])), "<｜begin▁of▁sentence｜>");
    }
}
