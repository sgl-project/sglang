// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::{adapter, TokenizerRegistry};
use crate::discovery::ModelId;

/// Tokens produced at ingress for routing and optional engine forwarding.
pub struct RequestTokens {
    /// The prompt token ids.
    pub ids: Vec<u32>,
    /// Whether the IDs came from rendered chat messages.
    /// Forwarding also requires the request safety guard.
    pub rendered_from_chat: bool,
}

/// Whether the caller pre-tokenized the prompt (`input_ids` present and not
/// null). Such a request is never re-rendered: its ids drive routing and the
/// body is forwarded untouched, malformed values included, for the engine to
/// validate.
pub fn has_caller_input_ids(value: &serde_json::Value) -> bool {
    value.get("input_ids").is_some_and(|v| !v.is_null())
}

/// Tokenizes a request for routing. Caller `input_ids` win; chat-rendered
/// tokens may also be forwarded to the engine (the chat route decides); raw
/// prompt tokens are routing-only.
pub fn request_tokens_for(
    tokenizers: &TokenizerRegistry,
    model_id: &ModelId,
    value: &serde_json::Value,
) -> Option<RequestTokens> {
    if has_caller_input_ids(value) {
        // A flat u32 array (empty included) supplies routing tokens; anything
        // else yields none, leaving validation to the engine.
        let ids = serde::Deserialize::deserialize(&value["input_ids"]).ok()?;
        return Some(RequestTokens {
            ids,
            rendered_from_chat: false,
        });
    }
    if tokenizers.has_chat_formatter(&model_id.0)
        && value.get("messages").is_some_and(|m| m.is_array())
    {
        if let Some(ids) = tokenizers.encode_chat(&model_id.0, value) {
            return Some(RequestTokens {
                ids,
                rendered_from_chat: true,
            });
        }
    }
    let text = extract_prompt_text_from_value(value)?;
    let ids = tokenize_text(tokenizers, model_id, &text)?;
    Some(RequestTokens {
        ids,
        rendered_from_chat: false,
    })
}

/// Tokenizes text with the model tokenizer.
fn tokenize_text(
    tokenizers: &TokenizerRegistry,
    model_id: &ModelId,
    text: &str,
) -> Option<Vec<u32>> {
    let tokenizer = tokenizers.get(&model_id.0)?;
    match adapter::encode(&tokenizer, text) {
        Ok(ids) if !ids.is_empty() => Some(ids),
        Ok(_) => None,
        Err(e) => {
            tracing::warn!(
                model = %model_id,
                error = %e,
                "ingress tokenize failed; routing/forwarding skips this prompt",
            );
            None
        }
    }
}

/// Extracts raw prompt text from a parsed request body.
///
/// Supported shapes (in priority order):
///   1. `"prompt": "..."` — `/v1/completions`-style.
///   2. `"prompt": ["...", "..."]` — `/v1/completions` array form;
///      concatenated with `"\n"`.
///   3. `"messages": [{"content": "..."}]` — `/v1/chat/completions`
///      with string content; concatenated with `"\n"`.
///   4. `"messages": [{"content": [{"text": "..."}]}]` — chat with
///      multimodal content blocks; text-only blocks concatenated.
///   5. `"text": "..."` — SGLang `/generate` native form.
///
pub(crate) fn extract_prompt_text_from_value(v: &serde_json::Value) -> Option<String> {
    if let Some(s) = v.get("prompt").and_then(|p| p.as_str()) {
        return Some(s.to_string());
    }
    if let Some(arr) = v.get("prompt").and_then(|p| p.as_array()) {
        let parts: Vec<&str> = arr.iter().filter_map(|x| x.as_str()).collect();
        if !parts.is_empty() {
            return Some(parts.join("\n"));
        }
    }
    if let Some(msgs) = v.get("messages").and_then(|m| m.as_array()) {
        let mut buf = String::new();
        for m in msgs {
            match m.get("content") {
                Some(serde_json::Value::String(s)) => {
                    if !buf.is_empty() {
                        buf.push('\n');
                    }
                    buf.push_str(s);
                }
                Some(serde_json::Value::Array(parts)) => {
                    for part in parts {
                        if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                            if !buf.is_empty() {
                                buf.push('\n');
                            }
                            buf.push_str(t);
                        }
                    }
                }
                _ => {}
            }
        }
        if !buf.is_empty() {
            return Some(buf);
        }
    }
    if let Some(s) = v.get("text").and_then(|t| t.as_str()) {
        return Some(s.to_string());
    }
    None
}
