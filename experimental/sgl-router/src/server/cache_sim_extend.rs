//! Response-completion tee to the theoretical cache-sim's insert-only
//! `POST /extend_ids` path.
//!
//! The ingress tee ([`crate::server::cache_sim_tee`]) sends each request's
//! prompt `input_ids` — but a real engine's KV cache also holds the tokens it
//! GENERATED, and the next round of a multi-turn conversation re-sends those
//! as an assistant turn. With only the ingress tee, that re-sent output scores
//! as a miss in the sim and the oracle hit rate under-reads exactly on the
//! agentic/multi-turn traffic it matters most for.
//!
//! So, when a response completes cleanly, this module reconstructs the
//! assistant reply (from the buffered JSON body, or from the captured SSE
//! stream) and produces the token ids of `messages + [assistant reply]` —
//! exactly what the next round's probe re-derives, so the chain-hashed
//! blocks match byte-for-byte.
//!
//! Two ways to produce those ids, tried in order:
//!
//! 1. **Incremental** (`TokenizerRegistry::encode_chat_extension`): reuse the
//!    ingress-computed `prompt_ids` and encode only the reply's rendered turn
//!    suffix — O(generated output) tokenize CPU per response. Guarded by a
//!    one-time per-model self-check proving the concatenation is
//!    byte-identical to a full re-encode (true for DSV4: rendering is
//!    concatenative and the prompt render ends in the added token
//!    `</think>`, which BPE merges cannot cross).
//! 2. **Full re-encode** (fallback): append the reply to the original
//!    request's `messages` and re-run the very same [`request_tokens_for`]
//!    the ingress uses — O(whole conversation), always correct.
//!
//! Everything here is off the serving path: the handler only clones two
//! `Bytes` handles (plus the ingress ids) and spawns; parsing and tokenizing
//! run on a background task, and delivery inherits the tee's fire-and-forget
//! contract.

use std::sync::Arc;

use bytes::Bytes;
use serde_json::Value;

use crate::discovery::ModelId;
use crate::policies::request_tokens_for;
use crate::server::app_context::AppContext;
use crate::tokenizer::dsv4::RenderOpts;

/// Cap on a captured/buffered response the extend tee will process. Bounds
/// both the SSE capture buffer (per in-flight stream) and the non-streaming
/// re-read. Far above any real generation (a 32k-token streamed response with
/// per-token SSE envelopes is ~2 MiB); past it the capture is discarded, never
/// truncated.
pub(crate) const MAX_EXTEND_CAPTURE_BYTES: usize = 16 << 20;

/// Where the completed response's bytes came from.
pub(crate) enum ReplySource {
    /// The full (non-streaming) OpenAI response JSON.
    Json(Bytes),
    /// The raw SSE byte stream of a completed streaming response.
    Sse(Vec<u8>),
}

/// The ingress tokenization the incremental extension appends to: the
/// chat-encoder (engine-equivalent) ids plus the `RenderOpts` they were
/// rendered under — the suffix must render under the SAME opts, since
/// thinking mode changes both the boundary token and whether the reply's
/// `reasoning_content` renders.
#[derive(Clone)]
pub(crate) struct IngressPrompt {
    pub ids: Vec<u32>,
    pub opts: RenderOpts,
}

/// Whether a completed response's extension can ever be MATCHED by the next
/// round's re-rendered prompt — the arm/skip gate for the whole tee.
///
/// In DSV4 thinking mode WITHOUT tools, it cannot: the next round's history
/// rendering drops the turn's reasoning and flips the generation-prompt
/// transition from `<think>` to `</think>`, so the re-derived tokens diverge
/// from the generated sequence right at the transition and chain-hashing kills
/// every block after it. That mirrors the real engine (its KV holds the
/// thinking tokens and misses the same way), so skipping loses no measurement
/// — while inserting would add O(output) permanently-unmatchable blocks per
/// response, which at production rates is real pressure on the sim's TTL shed
/// valve (~2.8M dead blocks/hour at 100 resp/s × 500-token outputs vs the 8M
/// default cap).
///
/// Thinking WITH tools stays armed: DSV4's tool-calling contract has clients
/// echo `reasoning_content` back and the history rendering keeps it, so the
/// extension matches. Known loss: a conversation that adds tools only on the
/// NEXT round (rendering would then keep this turn's reasoning) skips an
/// extension that could have matched — accepted as exotic.
///
/// The tools test mirrors `dsv4::render_messages` (an empty `tools` array is
/// falsy). The thinking flag comes from [`RenderOpts`], whose resolution is
/// DSV4-specific — for Jinja-encoder models the renderer ignores thinking
/// mode, so keep the DSV4-named env defaults unset there or this gate skips
/// extensions that would have been fine.
pub(crate) fn extension_can_match(request: &Value) -> bool {
    let opts = crate::tokenizer::dsv4::resolve_render_opts(request);
    if !opts.thinking {
        return true;
    }
    request
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|a| !a.is_empty())
}

/// Spawn the background task that reconstructs the assistant reply(ies) from
/// `source`, produces the full prompt+reply token sequence, and offers it to
/// the cache-sim extend tee. A no-op unless the tee is configured. Never
/// fails the caller: every parse/tokenize shortfall just drops the extension
/// (debug-logged), mirroring the tee's observational contract.
///
/// `prompt` — this request's ingress chat-encoder tokenization
/// (`RequestTokens.ids` with `engine_equivalent = true`, plus the resolved
/// `RenderOpts`), or `None` for the raw-prompt fallback. When present, the
/// extension is computed **incrementally**: `prompt.ids ++ encode(rendered
/// assistant turn)` via
/// [`crate::tokenizer::TokenizerRegistry::encode_chat_extension`] — O(generated output) instead
/// of O(whole conversation), which at production rates is the difference
/// between the tee roughly doubling the router's tokenize CPU and it being
/// nearly free. The full `messages + [reply]` re-encode remains as the
/// fallback whenever the incremental path declines (no encoder, failed
/// per-model concat self-check, suffix render failure).
pub(crate) fn spawn_extend_tee(
    ctx: Arc<AppContext>,
    model: String,
    request_body: Bytes,
    prompt: Option<IngressPrompt>,
    source: ReplySource,
) {
    if ctx.cache_sim_tee.is_none() {
        return;
    }
    tokio::spawn(async move {
        let replies = match &source {
            ReplySource::Json(body) => assistant_messages_from_response_json(body),
            ReplySource::Sse(bytes) => assistant_messages_from_sse(bytes),
        };
        if replies.is_empty() {
            tracing::debug!(model = %model, "cache-sim extend: no assistant reply reconstructed; skipping");
            return;
        }
        let model_id = ModelId(model.clone());
        // The fallback's parsed request body, materialized only if some reply
        // actually needs the full re-encode — when the incremental path serves
        // every choice, the request JSON is never even parsed here.
        let mut fallback_request: Option<Value> = None;
        // One extension per choice (`n > 1` yields alternative continuations —
        // whichever the client continues with should count as cached).
        for reply in replies {
            let ids = prompt
                .as_ref()
                .and_then(|p| {
                    ctx.tokenizers
                        .encode_chat_extension(&model, &p.ids, &reply, p.opts)
                })
                .or_else(|| {
                    full_reencode_extension(
                        &ctx,
                        &model_id,
                        &request_body,
                        &mut fallback_request,
                        reply,
                    )
                });
            if let (Some(tee), Some(ids)) = (ctx.cache_sim_tee.as_ref(), ids) {
                tee.offer_extend(&model, &ids);
            } else {
                tracing::debug!(model = %model, "cache-sim extend: tokenize failed; skipping");
            }
        }
    });
}

/// Fallback: append `reply` to the original request's `messages` and re-run
/// the ingress tokenization over the whole conversation. Parses
/// `request_body` into `parsed` on first use; the reply is pushed, encoded,
/// then popped so the parsed value can serve the next choice.
fn full_reencode_extension(
    ctx: &AppContext,
    model_id: &ModelId,
    request_body: &Bytes,
    parsed: &mut Option<Value>,
    reply: Value,
) -> Option<Vec<u32>> {
    if parsed.is_none() {
        let request = serde_json::from_slice::<Value>(request_body).ok()?;
        // Chat-route bodies always carry `messages` by the time the handler
        // tees; anything else has no history to extend.
        request.get("messages").filter(|m| m.is_array())?;
        *parsed = Some(request);
    }
    let request = parsed.as_mut()?;
    request
        .get_mut("messages")
        .and_then(Value::as_array_mut)?
        .push(reply);
    let tokens = request_tokens_for(&ctx.tokenizers, model_id, request);
    if let Some(messages) = request.get_mut("messages").and_then(Value::as_array_mut) {
        messages.pop();
    }
    tokens.map(|t| t.ids)
}

/// Pull the assistant message(s) out of a buffered (non-streaming) chat
/// completion response: `choices[*].message`, verbatim — the exact object a
/// client would echo back as history next round (the chat encoders ignore
/// fields they don't render, e.g. `reasoning_content`).
pub(crate) fn assistant_messages_from_response_json(body: &[u8]) -> Vec<Value> {
    let Ok(v) = serde_json::from_slice::<Value>(body) else {
        return Vec::new();
    };
    let Some(choices) = v.get("choices").and_then(Value::as_array) else {
        return Vec::new();
    };
    choices
        .iter()
        .filter_map(|c| c.get("message"))
        .filter(|m| m.is_object())
        .cloned()
        .collect()
}

/// Per-choice accumulator for streamed deltas.
#[derive(Default)]
struct ChoiceAcc {
    content: String,
    /// Thinking-mode reasoning deltas. Carried on the rebuilt message so the
    /// chat encoder can render it where the next round's history rendering
    /// would (DSV4 keeps prior-turn reasoning in tool-carrying thinking
    /// conversations — the agentic case); chat-mode rendering ignores it.
    reasoning_content: String,
    /// keyed by the delta's tool-call `index`; ordered map so the rebuilt
    /// `tool_calls` array preserves the engine's emission order.
    tool_calls: std::collections::BTreeMap<u64, ToolCallAcc>,
}

#[derive(Default)]
struct ToolCallAcc {
    id: Option<String>,
    name: String,
    arguments: String,
}

/// Reconstruct the assistant message(s) from a captured SSE stream by
/// replaying the OpenAI chunk deltas: `choices[*].delta.content` and
/// `choices[*].delta.reasoning_content` concatenate (into separate fields —
/// the chat encoder decides whether reasoning renders into history);
/// `choices[*].delta.tool_calls[*]` merge by tool-call `index` (`id`/`name`
/// arrive once, `arguments` fragments concatenate). Returns one message per
/// choice index, in index order.
///
/// The input is the FULL byte stream (the pump only delivers cleanly-completed
/// captures), so SSE events split across network chunks reassemble here via
/// plain line splitting. Unparseable `data:` payloads are skipped — one
/// malformed event costs its delta, not the reconstruction.
pub(crate) fn assistant_messages_from_sse(bytes: &[u8]) -> Vec<Value> {
    let mut choices: std::collections::BTreeMap<u64, ChoiceAcc> = std::collections::BTreeMap::new();
    let text = String::from_utf8_lossy(bytes);
    for line in text.split('\n') {
        let line = line.strip_suffix('\r').unwrap_or(line);
        let Some(payload) = line.strip_prefix("data:").map(str::trim_start) else {
            continue; // event:/comment/blank separator lines
        };
        if payload == "[DONE]" || payload.is_empty() {
            continue;
        }
        let Ok(chunk) = serde_json::from_str::<Value>(payload) else {
            continue;
        };
        let Some(chunk_choices) = chunk.get("choices").and_then(Value::as_array) else {
            continue;
        };
        for c in chunk_choices {
            let idx = c.get("index").and_then(Value::as_u64).unwrap_or(0);
            let acc = choices.entry(idx).or_default();
            let Some(delta) = c.get("delta") else {
                continue;
            };
            if let Some(s) = delta.get("content").and_then(Value::as_str) {
                acc.content.push_str(s);
            }
            if let Some(s) = delta.get("reasoning_content").and_then(Value::as_str) {
                acc.reasoning_content.push_str(s);
            }
            let Some(tcs) = delta.get("tool_calls").and_then(Value::as_array) else {
                continue;
            };
            for tc in tcs {
                let tc_idx = tc.get("index").and_then(Value::as_u64).unwrap_or(0);
                let tacc = acc.tool_calls.entry(tc_idx).or_default();
                if let Some(id) = tc.get("id").and_then(Value::as_str) {
                    tacc.id = Some(id.to_string());
                }
                if let Some(f) = tc.get("function") {
                    if let Some(name) = f.get("name").and_then(Value::as_str) {
                        tacc.name.push_str(name);
                    }
                    if let Some(args) = f.get("arguments").and_then(Value::as_str) {
                        tacc.arguments.push_str(args);
                    }
                }
            }
        }
    }

    choices
        .into_values()
        .filter_map(|acc| {
            if acc.content.is_empty()
                && acc.reasoning_content.is_empty()
                && acc.tool_calls.is_empty()
            {
                return None; // nothing generated (e.g. an aborted empty stream)
            }
            let mut msg = serde_json::json!({
                "role": "assistant",
                "content": acc.content,
            });
            if !acc.reasoning_content.is_empty() {
                msg["reasoning_content"] = Value::String(acc.reasoning_content);
            }
            if !acc.tool_calls.is_empty() {
                let tool_calls: Vec<Value> = acc
                    .tool_calls
                    .into_values()
                    .map(|t| {
                        serde_json::json!({
                            "id": t.id.unwrap_or_default(),
                            "type": "function",
                            "function": {"name": t.name, "arguments": t.arguments},
                        })
                    })
                    .collect();
                msg["tool_calls"] = Value::Array(tool_calls);
            }
            Some(msg)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extension_can_match_gates_dsv4_thinking_without_tools() {
        // Chat mode (no thinking): always matchable.
        let chat = serde_json::json!({"messages": []});
        assert!(extension_can_match(&chat));
        // Thinking without tools: next-round rendering diverges at the
        // generation-prompt transition — never matchable, skip.
        let thinking = serde_json::json!({
            "messages": [], "chat_template_kwargs": {"thinking": true},
        });
        assert!(!extension_can_match(&thinking));
        // Thinking with tools: history keeps reasoning, extension matches.
        let thinking_tools = serde_json::json!({
            "messages": [], "chat_template_kwargs": {"thinking": true},
            "tools": [{"type": "function", "function": {"name": "f"}}],
        });
        assert!(extension_can_match(&thinking_tools));
        // An empty tools array is falsy, mirroring dsv4::render_messages.
        let empty_tools = serde_json::json!({
            "messages": [], "chat_template_kwargs": {"thinking": true}, "tools": [],
        });
        assert!(!extension_can_match(&empty_tools));
    }

    #[test]
    fn json_response_yields_choice_messages_verbatim() {
        let body = serde_json::json!({
            "id": "x", "object": "chat.completion",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "hi", "reasoning_content": "hmm"}, "finish_reason": "stop"},
                {"index": 1, "message": {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                ]}, "finish_reason": "tool_calls"}
            ]
        });
        let msgs = assistant_messages_from_response_json(&serde_json::to_vec(&body).unwrap());
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["content"], "hi");
        // Verbatim: extra fields (reasoning_content) ride along; encoders
        // ignore what they don't render.
        assert_eq!(msgs[0]["reasoning_content"], "hmm");
        assert_eq!(msgs[1]["tool_calls"][0]["function"]["name"], "f");
    }

    #[test]
    fn json_response_without_choices_yields_nothing() {
        assert!(assistant_messages_from_response_json(b"{\"error\":{}}").is_empty());
        assert!(assistant_messages_from_response_json(b"not json").is_empty());
    }

    fn sse(events: &[&str]) -> Vec<u8> {
        let mut out = String::new();
        for e in events {
            out.push_str("data: ");
            out.push_str(e);
            out.push_str("\n\n");
        }
        out.into_bytes()
    }

    #[test]
    fn sse_content_deltas_concatenate() {
        let bytes = sse(&[
            r#"{"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"content":"Hel"}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"content":"lo!"}}]}"#,
            r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
            "[DONE]",
        ]);
        let msgs = assistant_messages_from_sse(&bytes);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "assistant");
        assert_eq!(msgs[0]["content"], "Hello!");
        assert!(msgs[0].get("tool_calls").is_none());
    }

    #[test]
    fn sse_reasoning_content_accumulates_separately_from_content() {
        // Thinking deltas stream as `reasoning_content`; they belong on the
        // rebuilt message's own field (the chat encoder decides whether they
        // render into next-round history) and must never bleed into `content`.
        let bytes = sse(&[
            r#"{"choices":[{"index":0,"delta":{"reasoning_content":"let me "}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"reasoning_content":"think"}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"content":"42"}}]}"#,
            "[DONE]",
        ]);
        let msgs = assistant_messages_from_sse(&bytes);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["content"], "42");
        assert_eq!(msgs[0]["reasoning_content"], "let me think");
    }

    #[test]
    fn sse_tool_call_deltas_merge_by_index() {
        let bytes = sse(&[
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"SF\"}"}}]}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_b","type":"function","function":{"name":"f2","arguments":"{}"}}]}}]}"#,
            "[DONE]",
        ]);
        let msgs = assistant_messages_from_sse(&bytes);
        assert_eq!(msgs.len(), 1);
        let tcs = msgs[0]["tool_calls"].as_array().unwrap();
        assert_eq!(tcs.len(), 2);
        assert_eq!(tcs[0]["id"], "call_a");
        assert_eq!(tcs[0]["function"]["name"], "get_weather");
        assert_eq!(tcs[0]["function"]["arguments"], r#"{"city":"SF"}"#);
        assert_eq!(tcs[1]["id"], "call_b");
    }

    #[test]
    fn sse_multiple_choices_yield_one_message_each_in_index_order() {
        let bytes = sse(&[
            r#"{"choices":[{"index":1,"delta":{"content":"B"}}]}"#,
            r#"{"choices":[{"index":0,"delta":{"content":"A"}}]}"#,
            "[DONE]",
        ]);
        let msgs = assistant_messages_from_sse(&bytes);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["content"], "A");
        assert_eq!(msgs[1]["content"], "B");
    }

    #[test]
    fn sse_empty_or_garbage_yields_nothing() {
        assert!(assistant_messages_from_sse(b"").is_empty());
        // A malformed event is skipped, an empty generation is dropped.
        let bytes = sse(&[r#"{"choices":[{"index":0,"delta":{}}]}"#, "not json"]);
        assert!(assistant_messages_from_sse(&bytes).is_empty());
    }

    #[test]
    fn sse_crlf_lines_parse() {
        let bytes = b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"}}]}\r\n\r\ndata: [DONE]\r\n\r\n".to_vec();
        let msgs = assistant_messages_from_sse(&bytes);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["content"], "ok");
    }
}
