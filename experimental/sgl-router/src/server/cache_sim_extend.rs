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
///
/// Surgery-shaped traffic is additionally disarmed via
/// [`request_has_extension_unsafe_shape`].
pub(crate) fn extension_can_match(request: &Value) -> bool {
    if request_has_extension_unsafe_shape(request) {
        return false;
    }
    let opts = crate::tokenizer::dsv4::resolve_render_opts(request);
    if !opts.thinking {
        return true;
    }
    request
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|a| !a.is_empty())
}

/// Request/request-message shapes the incremental extension is NOT proven
/// against, so the tee stays disarmed: `continue_final_message` (pydantic
/// coerced) or a trailing assistant message (the engine's trailing-assistant
/// surgery changes where the prompt ends, so prompt-ids ++ reply-suffix no
/// longer models the next round); request-level `task` or any message-level
/// `task` (the task transition changes the tail, and the reply's
/// `prev_has_task` suppression breaks the suffix-vs-full-render comparison);
/// and the engine-internal message keys the encoder doesn't model (`wo_eos`,
/// `mask`, `content_blocks`). The next round's full re-encode still measures
/// these — only the O(output) incremental path is skipped.
fn request_has_extension_unsafe_shape(request: &Value) -> bool {
    if request
        .get("continue_final_message")
        .filter(|v| !v.is_null())
        .is_some_and(|v| crate::tokenizer::openai_bool(v) != Some(false))
    {
        return true;
    }
    if request.get("task").is_some_and(|v| !v.is_null()) {
        return true;
    }
    let Some(msgs) = request.get("messages").and_then(Value::as_array) else {
        return false;
    };
    if msgs
        .last()
        .is_some_and(|m| m.get("role").and_then(Value::as_str) == Some("assistant"))
    {
        return true;
    }
    msgs.iter().any(|m| {
        ["task", "wo_eos", "mask", "content_blocks"]
            .iter()
            .any(|k| m.get(k).is_some_and(|v| !v.is_null()))
    })
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
///
/// `request_id` is the join key shared with this request's ingress tee, so the
/// oracle can pair the two records and report the response's output tokens
/// against the prompt they extend.
pub(crate) fn spawn_extend_tee(
    ctx: Arc<AppContext>,
    model: String,
    request_id: String,
    request_body: Bytes,
    prompt: Option<IngressPrompt>,
    source: ReplySource,
) {
    if ctx.cache_sim_tee.is_none() && ctx.s3_export_sink.is_none() {
        return;
    }
    tokio::spawn(async move {
        let replies = match &source {
            ReplySource::Json(body) => assistant_messages_from_response_json(body),
            ReplySource::Sse(bytes) => assistant_messages_from_sse(bytes),
        };
        // The engine's own count, when it reported one. Preferred over
        // `len(ids) - prompt_len`, which measures the RE-RENDERED history turn
        // and so drifts from what was generated — see `IngestIdsBody`.
        let output_tokens = match &source {
            ReplySource::Json(body) => completion_tokens_from_response_json(body),
            ReplySource::Sse(bytes) => completion_tokens_from_sse(bytes),
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
        // Fan-out width comes from the REQUEST's `n`, not from how many
        // replies reconstructed. A choice the engine emitted but that produced
        // no content is dropped during reconstruction, so `replies.len()` would
        // report 1 for a genuine 2-choice response — the suppression below
        // would not fire and the response-wide usage total would be stamped on
        // a single choice. That is the N-times inflation the discriminator
        // exists to prevent, reached through the other door.
        let requested_n = serde_json::from_slice::<Value>(&request_body)
            .ok()
            .and_then(|v| v.get("n").and_then(Value::as_u64))
            .unwrap_or(1)
            .max(1) as usize;
        let choice_count = requested_n.max(replies.len());
        for (choice_index, reply) in replies {
            // Which path produced the ids decides whether a prompt/output
            // boundary exists at all, so the two are kept distinguishable
            // rather than collapsed with `or_else`. Incremental output is
            // `prompt.ids ++ suffix`, so the boundary is exactly the ingress
            // length; the fallback re-encodes the whole conversation, where no
            // prefix is guaranteed to be the prompt.
            let incremental = prompt.as_ref().and_then(|p| {
                ctx.tokenizers
                    .encode_chat_extension(&model, &p.ids, &reply, p.opts)
                    .map(|ids| (ids, Some(p.ids.len())))
            });
            let produced = match incremental {
                Some(v) => Some(v),
                None => full_reencode_extension(
                    &ctx,
                    &model_id,
                    &request_body,
                    &mut fallback_request,
                    reply,
                )
                .map(|ids| (ids, None)),
            };
            if let Some((ids, prompt_len)) = produced {
                // Only tag a genuine fan-out. A single-choice response is the
                // common case and needs no discriminator; N>1 must have one or
                // the N records collapse onto one join key.
                let choice = (choice_count > 1).then_some(crate::server::cache_sim_tee::Choice {
                    index: choice_index as usize,
                    count: choice_count,
                });
                // The engine reports usage for the response as a whole, so it
                // describes a fan-out's total, not any one alternative.
                // Attributing it to every choice would multiply it by N.
                let per_choice_output = if choice_count > 1 {
                    None
                } else {
                    output_tokens
                };
                if let Some(tee) = ctx.cache_sim_tee.as_ref() {
                    tee.offer_extend(
                        &model,
                        &ids,
                        &request_id,
                        prompt_len,
                        choice,
                        per_choice_output,
                    );
                }
                if let Some(sink) = ctx.s3_export_sink.as_ref() {
                    sink.offer_extend(
                        &model,
                        &ids,
                        &request_id,
                        prompt_len,
                        per_choice_output,
                        choice.as_ref().map(|c| c.index),
                        choice.as_ref().map(|c| c.count),
                        None, // extend 路径当前不持有 attribution（见 spec §11）
                    );
                }
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
    // Reconstruct the ids WITHOUT request-level surgery (`encode_chat_plain`):
    // the appended reply is the NEXT round's history — an ordinary closed
    // assistant turn — not this request's trailing continuation. The raw
    // fallback (`request_tokens_for`) stays for models without a chat encoder.
    let messages = request.get("messages").unwrap().clone();
    let opts = crate::tokenizer::dsv4::resolve_render_opts(request);
    let ids = ctx
        .tokenizers
        .encode_chat_plain(&model_id.0, &messages, request.get("tools"), opts);
    let tokens = match ids {
        Some(ids) => Some(ids),
        None => request_tokens_for(&ctx.tokenizers, model_id, request).map(|t| t.ids),
    };
    if let Some(messages) = request.get_mut("messages").and_then(Value::as_array_mut) {
        messages.pop();
    }
    tokens
}

/// `usage.completion_tokens` from a buffered chat response, when present.
pub(crate) fn completion_tokens_from_response_json(body: &[u8]) -> Option<u64> {
    serde_json::from_slice::<Value>(body)
        .ok()?
        .get("usage")?
        .get("completion_tokens")?
        .as_u64()
}

/// `usage.completion_tokens` from a captured SSE stream — the RESPONSE-WIDE
/// total, or `None`.
///
/// Accepts a chunk only when its `choices` is absent or empty, which is what
/// distinguishes the aggregate usage event from a per-choice one. Two client
/// options put `usage` on the wire and they mean different things:
///
/// - `stream_options.include_usage` emits one final aggregate chunk with
///   `choices: []` — the response total, which is what we want.
/// - `stream_options.continuous_usage_stats` stamps `usage` on EVERY content
///   chunk, scoped to that chunk's single choice, and can be set WITHOUT
///   `include_usage` (so no aggregate ever arrives).
///
/// Taking the last `usage` seen would therefore return one choice's partial
/// count on a continuous-stats stream and present it as the response total —
/// a number that looks authoritative and is silently short. Absent is the
/// honest answer there.
pub(crate) fn completion_tokens_from_sse(bytes: &[u8]) -> Option<u64> {
    let text = String::from_utf8_lossy(bytes);
    for line in text.split('\n').rev() {
        let line = line.strip_suffix('\r').unwrap_or(line);
        let Some(payload) = line.strip_prefix("data:").map(str::trim_start) else {
            continue;
        };
        if payload == "[DONE]" {
            continue;
        }
        let Ok(v) = serde_json::from_str::<Value>(payload) else {
            continue;
        };
        // Per-choice usage rides alongside content; only the aggregate event
        // carries no choices.
        let is_aggregate = v
            .get("choices")
            .is_none_or(|c| c.as_array().is_some_and(|a| a.is_empty()));
        if !is_aggregate {
            continue;
        }
        if let Some(n) = v
            .get("usage")
            .and_then(|u| u.get("completion_tokens"))
            .and_then(Value::as_u64)
        {
            return Some(n);
        }
    }
    None
}

/// Pull the assistant message(s) out of a buffered (non-streaming) chat
/// completion response: `choices[*].message`, verbatim — the exact object a
/// client would echo back as history next round (the chat encoders ignore
/// fields they don't render, e.g. `reasoning_content`).
pub(crate) fn assistant_messages_from_response_json(body: &[u8]) -> Vec<(u64, Value)> {
    let Ok(v) = serde_json::from_slice::<Value>(body) else {
        return Vec::new();
    };
    let Some(choices) = v.get("choices").and_then(Value::as_array) else {
        return Vec::new();
    };
    choices
        .iter()
        .filter_map(|c| {
            let msg = c.get("message").filter(|m| m.is_object())?.clone();
            // The ENGINE's ordinal, not this vector's position. A choice the
            // engine emitted but we could not reconstruct is skipped, so a
            // positional index would silently renumber every later choice and
            // point accounting at the wrong alternative.
            let idx = c.get("index").and_then(Value::as_u64).unwrap_or(0);
            Some((idx, msg))
        })
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
/// plain line splitting. Unparsable `data:` payloads are skipped — one
/// malformed event costs its delta, not the reconstruction.
pub(crate) fn assistant_messages_from_sse(bytes: &[u8]) -> Vec<(u64, Value)> {
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
        .into_iter()
        .filter_map(|(idx, acc)| {
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
            Some((idx, msg))
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

    /// Shapes the incremental extension can't model: the prompt tail touches
    /// engine behaviors (trailing-assistant surgery / task transitions /
    /// engine-internal message keys) that break the ids ++ reply-suffix
    /// correspondence. These must disarm even under shapes that would
    /// otherwise arm (chat mode, or thinking-with-tools).
    #[test]
    fn extension_can_match_disarms_surgery_shapes() {
        let thinking_tools = serde_json::json!({
            "messages": [{"role":"user","content":"q"}],
            "chat_template_kwargs": {"thinking": true},
            "tools": [{"type":"function","function":{"name":"f"}}],
        });
        // Baseline arms.
        assert!(extension_can_match(&thinking_tools));

        for body in [
            serde_json::json!({"messages":[{"role":"user","content":"q"}],
                               "continue_final_message":true}),
            serde_json::json!({"messages":[{"role":"user","content":"q"}],
                               "continue_final_message":"true"}), // pydantic-coerced too
            serde_json::json!({"messages":[{"role":"user","content":"q"}],"task":"query"}),
            serde_json::json!({"messages":[{"role":"user","content":"q","task":"query"}]}),
            serde_json::json!({"messages":[{"role":"user","content":"q","wo_eos":true}]}),
            serde_json::json!({"messages":[{"role":"user","content":"q"},
                                           {"role":"assistant","content":"partial"}],
                               "continue_final_message":false}),
        ] {
            assert!(
                !extension_can_match(&body),
                "extension must stay disarmed for: {body}"
            );
        }
        // And null/absent forms stay armed.
        let cleared = serde_json::json!({"messages":[{"role":"user","content":"q"}],
                                         "continue_final_message":false, "task":null});
        assert!(extension_can_match(&cleared));
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
        assert_eq!(msgs[0].1["content"], "hi");
        // Verbatim: extra fields (reasoning_content) ride along; encoders
        // ignore what they don't render.
        assert_eq!(msgs[0].1["reasoning_content"], "hmm");
        assert_eq!(msgs[1].1["tool_calls"][0]["function"]["name"], "f");
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
        assert_eq!(msgs[0].1["role"], "assistant");
        assert_eq!(msgs[0].1["content"], "Hello!");
        assert!(msgs[0].1.get("tool_calls").is_none());
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
        assert_eq!(msgs[0].1["content"], "42");
        assert_eq!(msgs[0].1["reasoning_content"], "let me think");
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
        let tcs = msgs[0].1["tool_calls"].as_array().unwrap();
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
        assert_eq!(msgs[0].1["content"], "A");
        assert_eq!(msgs[1].1["content"], "B");
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
        assert_eq!(msgs[0].1["content"], "ok");
    }
}

#[cfg(test)]
mod usage_tests {
    use super::*;

    #[test]
    fn completion_tokens_read_from_a_buffered_response() {
        let body = br#"{"choices":[{"message":{"content":"hi"}}],
                        "usage":{"prompt_tokens":10,"completion_tokens":7}}"#;
        assert_eq!(completion_tokens_from_response_json(body), Some(7));
        // No usage block: absent, NOT zero. Zero would read downstream as a
        // real measurement of "generated nothing".
        assert_eq!(
            completion_tokens_from_response_json(br#"{"choices":[]}"#),
            None
        );
        assert_eq!(completion_tokens_from_response_json(b"not json"), None);
    }

    /// Per-choice usage must NOT be mistaken for the response total.
    ///
    /// `continuous_usage_stats` stamps `usage` on every content chunk, scoped
    /// to that chunk's single choice, and can be set WITHOUT `include_usage`
    /// so no aggregate ever arrives. Taking the last `usage` seen would return
    /// one choice's partial count and present it as the whole response's — a
    /// number that looks authoritative and is silently short.
    #[test]
    fn per_choice_usage_is_not_read_as_the_response_total() {
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}],\"usage\":{\"completion_tokens\":3}}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"b\"}}],\"usage\":{\"completion_tokens\":5}}\n\n",
            "data: [DONE]\n\n"
        );
        assert_eq!(
            completion_tokens_from_sse(sse.as_bytes()),
            None,
            "a continuous-usage stream has no aggregate chunk; absent is the honest answer"
        );

        // With the aggregate present, it wins over the per-chunk numbers.
        let with_agg = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}],\"usage\":{\"completion_tokens\":3}}\n\n",
            "data: {\"choices\":[],\"usage\":{\"completion_tokens\":9}}\n\n",
            "data: [DONE]\n\n"
        );
        assert_eq!(completion_tokens_from_sse(with_agg.as_bytes()), Some(9));
    }

    /// The engine's ordinal must survive reconstruction. A choice that emitted
    /// nothing is dropped, so a positional index would renumber every later
    /// choice and point accounting at the wrong alternative.
    #[test]
    fn reconstruction_carries_the_engines_choice_index() {
        let sse = concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"first\"}}]}\n\n",
            "data: {\"choices\":[{\"index\":2,\"delta\":{\"content\":\"third\"}}]}\n\n",
            "data: [DONE]\n\n"
        );
        let got = assistant_messages_from_sse(sse.as_bytes());
        let idxs: Vec<u64> = got.iter().map(|(i, _)| *i).collect();
        assert_eq!(idxs, vec![0, 2], "engine indices, not 0/1");

        let json = br#"{"choices":[{"index":1,"message":{"role":"assistant","content":"only"}}]}"#;
        let got = assistant_messages_from_response_json(json);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, 1, "engine ordinal, not vector position");
    }

    #[test]
    fn completion_tokens_read_from_the_streaming_usage_chunk() {
        // The usage chunk is the last data event before [DONE], and it carries
        // a null `choices` — so the scan must not stop at the content chunks.
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"completion_tokens\":11}}\n\n",
            "data: [DONE]\n\n"
        );
        assert_eq!(completion_tokens_from_sse(sse.as_bytes()), Some(11));
    }

    #[test]
    fn no_usage_chunk_means_absent_not_zero() {
        // The common case: the client did not set stream_options.include_usage.
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
            "data: [DONE]\n\n"
        );
        assert_eq!(completion_tokens_from_sse(sse.as_bytes()), None);
    }
}

#[cfg(test)]
mod spawn_tests {
    use super::*;
    use crate::server::app_context::AppContext;
    use crate::server::cache_sim_tee::CacheSimTee;
    use crate::server::metrics::MetricsRegistry;
    use crate::tokenizer::{ChatEncoder, TokenizerRegistry};
    use axum::{extract::State, routing::post, Router};
    use std::sync::Mutex;

    type Captured = Arc<Mutex<Vec<Value>>>;

    /// Mock cache-sim that records every `/extend_ids` body it is POSTed.
    async fn mock_cache_sim() -> (String, Captured) {
        let cap: Captured = Arc::new(Mutex::new(Vec::new()));
        let app = Router::new()
            .route(
                "/extend_ids",
                post(|State(c): State<Captured>, body: Bytes| async move {
                    c.lock()
                        .unwrap()
                        .push(serde_json::from_slice(&body).unwrap());
                    axum::http::StatusCode::NO_CONTENT
                }),
            )
            .with_state(Arc::clone(&cap));
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
        (format!("http://{addr}"), cap)
    }

    fn cfg_for(model_id: &str) -> crate::config::Config {
        crate::config::Config {
            server: crate::config::ServerConfig {
                host: "x".into(),
                port: 0,
                ..Default::default()
            },
            observability: Default::default(),
            model: crate::config::ModelConfig {
                id: model_id.into(),
                tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
                tokenizer_shards: 1,
                tokenizer_backend: Default::default(),
                tokenizer_l1_cache_mb: 0,
                policy: crate::config::PolicyKind::RoundRobin,
                circuit_breaker: None,
                cache_aware: None,
                sticky: None,
                max_output_tokens: None,
                forward_input_ids: true,
            },
            discovery: crate::config::DiscoveryBackend::StaticUrls(
                crate::config::StaticUrlsDiscoveryConfig {
                    urls: vec!["http://placeholder:0".into()],
                },
            ),
            proxy: crate::config::ProxyConfig::default(),
            active_load: crate::config::ActiveLoadConfig::default(),
            admission: crate::config::AdmissionConfig::default(),
            retry: crate::config::RetryConfig::default(),
        }
    }

    /// `ctx` wired to a live mock cache-sim, with `model` loaded on the tiny
    /// fixture tokenizer. `with_encoder` attaches the DSV4 chat encoder — the
    /// difference between the incremental and full-re-encode paths.
    async fn ctx_with(model: &str, with_encoder: bool) -> (Arc<AppContext>, Captured) {
        let (url, cap) = mock_cache_sim().await;
        let reg = TokenizerRegistry::load_from_config(&cfg_for(model)).unwrap();
        if with_encoder {
            reg.attach_chat_encoder_for_test(model, ChatEncoder::DeepSeekV4);
        }
        let mut ctx = AppContext::stub();
        ctx.tokenizers = Arc::new(reg);
        ctx.cache_sim_tee = Some(CacheSimTee::spawn(url, MetricsRegistry::new(), 64));
        (Arc::new(ctx), cap)
    }

    async fn wait_for(cap: &Captured, n: usize) -> Vec<Value> {
        for _ in 0..200 {
            {
                let g = cap.lock().unwrap();
                if g.len() >= n {
                    return g.clone();
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        panic!(
            "timed out waiting for {n} extend posts; got {:?}",
            cap.lock().unwrap()
        );
    }

    fn messages() -> Value {
        serde_json::json!([{"role": "user", "content": "and 3+3?"}])
    }

    fn body_with(n_choices: usize, completion_tokens: u64) -> Bytes {
        let choices: Vec<Value> = (0..n_choices)
            .map(|i| {
                serde_json::json!({
                    "index": i,
                    "message": {"role": "assistant", "content": format!("answer {i}")},
                })
            })
            .collect();
        Bytes::from(
            serde_json::to_vec(&serde_json::json!({
                "choices": choices,
                "usage": {"completion_tokens": completion_tokens},
            }))
            .unwrap(),
        )
    }

    fn request_body() -> Bytes {
        Bytes::from(
            serde_json::to_vec(&serde_json::json!({
                "model": "m", "messages": messages(),
            }))
            .unwrap(),
        )
    }

    /// PATH SELECTION, arm 1: an ingress prompt + a working chat encoder takes
    /// the INCREMENTAL path, whose output is exactly `prompt.ids ++ suffix` —
    /// so the wire must carry the exact boundary `prompt.ids.len()`.
    #[tokio::test]
    async fn incremental_path_reports_the_ingress_prompt_length_as_the_boundary() {
        let (ctx, cap) = ctx_with("dsv4", true).await;
        let opts = RenderOpts::chat();
        let prompt_ids = ctx
            .tokenizers
            .encode_chat(
                "dsv4",
                &messages(),
                None,
                opts,
                crate::tokenizer::dsv4::RequestParts::default(),
            )
            .expect("chat encode");
        let n = prompt_ids.len();

        spawn_extend_tee(
            Arc::clone(&ctx),
            "dsv4".into(),
            "rid-inc".into(),
            request_body(),
            Some(IngressPrompt {
                ids: prompt_ids.clone(),
                opts,
            }),
            ReplySource::Json(body_with(1, 9)),
        );

        let got = wait_for(&cap, 1).await;
        assert_eq!(got.len(), 1);
        let v = &got[0];
        assert_eq!(v["request_id"], "rid-inc");
        assert_eq!(
            v["prompt_len"], n,
            "the incremental path's boundary IS the ingress prompt length: {v}"
        );
        let ids: Vec<u64> = v["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap())
            .collect();
        assert!(
            ids.len() > n,
            "the extension must append a non-empty suffix"
        );
        assert_eq!(
            ids[..n],
            prompt_ids.iter().map(|&i| i as u64).collect::<Vec<_>>()[..],
            "prompt_len must actually index the prompt/output boundary"
        );
        assert_eq!(v["output_tokens"], 9);
    }

    /// PATH SELECTION, arm 2: no ingress prompt (raw-prompt fallback at
    /// ingress) forces the FULL RE-ENCODE, where no prefix is guaranteed to be
    /// the prompt — so NO boundary may be claimed.
    #[tokio::test]
    async fn fallback_path_claims_no_boundary() {
        let (ctx, cap) = ctx_with("tiny", false).await;
        spawn_extend_tee(
            Arc::clone(&ctx),
            "tiny".into(),
            "rid-fb".into(),
            request_body(),
            None,
            ReplySource::Json(body_with(1, 5)),
        );
        let got = wait_for(&cap, 1).await;
        let v = &got[0];
        assert_eq!(v["request_id"], "rid-fb");
        assert!(
            v.get("prompt_len").is_none(),
            "the full-re-encode fallback must claim no boundary: {v}"
        );
    }

    /// PATH SELECTION, arm 3: an ingress prompt whose model has NO usable chat
    /// encoder cannot extend incrementally, so it must FALL BACK — and the
    /// fallback's no-boundary rule still holds. Guards against the prompt's
    /// mere presence being taken as proof the incremental path ran.
    #[tokio::test]
    async fn prompt_without_an_encoder_falls_back_and_claims_no_boundary() {
        let (ctx, cap) = ctx_with("tiny", false).await;
        let opts = RenderOpts::chat();
        spawn_extend_tee(
            Arc::clone(&ctx),
            "tiny".into(),
            "rid-noenc".into(),
            request_body(),
            Some(IngressPrompt {
                ids: vec![1, 2, 3],
                opts,
            }),
            ReplySource::Json(body_with(1, 5)),
        );
        let got = wait_for(&cap, 1).await;
        assert!(got[0].get("prompt_len").is_none(), "{:?}", got[0]);
    }

    /// `n > 1`: N choices produce N extends that SHARE the request_id and are
    /// distinguished only by `choice_index`; and the engine's whole-response
    /// `completion_tokens` is suppressed, because attributing it to each of N
    /// records would multiply the oracle's output-token total by N.
    #[tokio::test]
    async fn fanout_yields_one_extend_per_choice_sharing_the_id() {
        let (ctx, cap) = ctx_with("dsv4", true).await;
        let opts = RenderOpts::chat();
        let prompt_ids = ctx
            .tokenizers
            .encode_chat(
                "dsv4",
                &messages(),
                None,
                opts,
                crate::tokenizer::dsv4::RequestParts::default(),
            )
            .unwrap();

        spawn_extend_tee(
            Arc::clone(&ctx),
            "dsv4".into(),
            "rid-fan".into(),
            request_body(),
            Some(IngressPrompt {
                ids: prompt_ids,
                opts,
            }),
            ReplySource::Json(body_with(3, 30)),
        );

        let got = wait_for(&cap, 3).await;
        assert_eq!(got.len(), 3, "one extend per choice");
        let mut indices: Vec<u64> = got
            .iter()
            .map(|v| {
                assert_eq!(v["request_id"], "rid-fan", "all N share the join key");
                assert_eq!(v["choice_count"], 3);
                assert!(
                    v.get("output_tokens").is_none(),
                    "whole-response usage must NOT be attributed per choice: {v}"
                );
                v["choice_index"].as_u64().expect("fan-out must be tagged")
            })
            .collect();
        indices.sort_unstable();
        assert_eq!(indices, vec![0, 1, 2], "indices must be distinct");
    }

    /// A LOSSY fan-out must still be tagged.
    ///
    /// Reconstruction drops a choice that generated nothing, so deriving the
    /// width from `replies.len()` would report 1 for a genuine n=2 response —
    /// the suppression would not fire and the whole-response usage total would
    /// be stamped on a single choice, which is the N-times inflation the
    /// discriminator exists to prevent, reached through the other door. Width
    /// therefore comes from the REQUEST's `n`.
    #[tokio::test]
    async fn a_fanout_that_reconstructs_one_reply_is_still_tagged_as_a_fanout() {
        let (ctx, cap) = ctx_with("dsv4", true).await;
        let opts = RenderOpts::chat();
        let prompt_ids = ctx
            .tokenizers
            .encode_chat(
                "dsv4",
                &messages(),
                None,
                opts,
                crate::tokenizer::dsv4::RequestParts::default(),
            )
            .unwrap();
        // The request asked for 2; only one choice carries a usable message.
        let req = Bytes::from(
            serde_json::to_vec(&serde_json::json!({
                "model": "m", "n": 2, "messages": messages(),
            }))
            .unwrap(),
        );
        let resp = Bytes::from(
            serde_json::to_vec(&serde_json::json!({
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "only one"}},
                    {"index": 1, "finish_reason": "stop"},
                ],
                "usage": {"completion_tokens": 40},
            }))
            .unwrap(),
        );
        spawn_extend_tee(
            Arc::clone(&ctx),
            "dsv4".into(),
            "rid-lossy".into(),
            req,
            Some(IngressPrompt {
                ids: prompt_ids,
                opts,
            }),
            ReplySource::Json(resp),
        );
        let got = wait_for(&cap, 1).await;
        let v = &got[0];
        assert_eq!(
            v["choice_count"], 2,
            "width must come from the request's n, not from what reconstructed: {v}"
        );
        assert!(
            v.get("output_tokens").is_none(),
            "a 2-choice response's total must not be stamped on the one surviving choice: {v}"
        );
    }

    /// `choice_index` is the ENGINE's ordinal, not this vector's position.
    ///
    /// With a middle choice dropped, a positional index would renumber the
    /// survivors and point accounting at the wrong alternative — while looking
    /// perfectly well-formed.
    #[tokio::test]
    async fn choice_index_is_the_engines_ordinal_not_a_vector_position() {
        let (ctx, cap) = ctx_with("dsv4", true).await;
        let opts = RenderOpts::chat();
        let prompt_ids = ctx
            .tokenizers
            .encode_chat(
                "dsv4",
                &messages(),
                None,
                opts,
                crate::tokenizer::dsv4::RequestParts::default(),
            )
            .unwrap();
        let req = Bytes::from(
            serde_json::to_vec(&serde_json::json!({
                "model": "m", "n": 3, "messages": messages(),
            }))
            .unwrap(),
        );
        // Engine emitted 0 and 2; choice 1 produced nothing.
        let resp = Bytes::from(
            serde_json::to_vec(&serde_json::json!({
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "first"}},
                    {"index": 2, "message": {"role": "assistant", "content": "third"}},
                ],
                "usage": {"completion_tokens": 30},
            }))
            .unwrap(),
        );
        spawn_extend_tee(
            Arc::clone(&ctx),
            "dsv4".into(),
            "rid-idx".into(),
            req,
            Some(IngressPrompt {
                ids: prompt_ids,
                opts,
            }),
            ReplySource::Json(resp),
        );
        let got = wait_for(&cap, 2).await;
        let mut idx: Vec<u64> = got
            .iter()
            .map(|v| v["choice_index"].as_u64().expect("tagged"))
            .collect();
        idx.sort_unstable();
        assert_eq!(
            idx,
            vec![0, 2],
            "engine ordinals must survive; a positional index would give [0,1]"
        );
    }

    /// The single-choice common case carries no discriminator — absent means
    /// "not a fan-out", and the engine's usage IS attributable.
    #[tokio::test]
    async fn a_single_choice_is_untagged_and_keeps_its_output_tokens() {
        let (ctx, cap) = ctx_with("dsv4", true).await;
        let opts = RenderOpts::chat();
        let prompt_ids = ctx
            .tokenizers
            .encode_chat(
                "dsv4",
                &messages(),
                None,
                opts,
                crate::tokenizer::dsv4::RequestParts::default(),
            )
            .unwrap();
        spawn_extend_tee(
            Arc::clone(&ctx),
            "dsv4".into(),
            "rid-one".into(),
            request_body(),
            Some(IngressPrompt {
                ids: prompt_ids,
                opts,
            }),
            ReplySource::Json(body_with(1, 12)),
        );
        let got = wait_for(&cap, 1).await;
        let v = &got[0];
        assert!(v.get("choice_index").is_none(), "{v}");
        assert!(v.get("choice_count").is_none(), "{v}");
        assert_eq!(v["output_tokens"], 12);
    }
}
