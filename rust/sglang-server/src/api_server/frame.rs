//! Frame shaping for the native `/generate` protocol: the cumulative
//! [`OutputAccumulator`] plus the functions that render [`ChunkEvent`]s /
//! accumulated state into wire JSON (`meta_info`, logprob tuples, error and
//! abort frames). No HTTP here — the sibling `native_api` module owns the handlers
//! and streams; it calls these per frame.

use crate::message::{ChunkEvent, ChunkExtras};

/// The text slot of a `[logprob, token_id, text]` tuple: the decoded token when
/// `return_text_in_logprobs` supplied a text buffer, else `null`.
fn text_slot(texts: Option<&[String]>, j: usize) -> serde_json::Value {
    texts
        .and_then(|t| t.get(j))
        .map(|s| serde_json::json!(s))
        .unwrap_or(serde_json::Value::Null)
}

/// A decoded-text column becomes the tuples' text source only when populated
/// (`return_text_in_logprobs`); empty → `None` → null text slots.
fn opt_texts(t: &[String]) -> Option<&[String]> {
    (!t.is_empty()).then_some(t)
}

/// The logprob slot of a tuple: a finite value, or `null` for the `NaN` sentinel.
fn lp_value(v: f32) -> serde_json::Value {
    if v.is_nan() {
        serde_json::Value::Null
    } else {
        serde_json::json!(v)
    }
}

/// SGLang logprob shape: a list of `[logprob, token_id, text]` tuples. `texts`
/// (parallel to `idxs`) fills the text slot when set, else `null`.
fn logprob_tuples(vals: &[f32], idxs: &[i32], texts: Option<&[String]>) -> serde_json::Value {
    let tuples: Vec<serde_json::Value> = vals
        .iter()
        .zip(idxs.iter())
        .enumerate()
        .map(|(j, (&v, &tid))| serde_json::json!([lp_value(v), tid, text_slot(texts, j)]))
        .collect();
    serde_json::Value::Array(tuples)
}

/// Ragged top-k / token-ids shape: one entry per position — a list of
/// `[logprob, token_id, text]` tuples, or `null` when `lens[p] == 0` (mirrors
/// `detokenize_top_logprobs_tokens`). `texts` is parallel to `vals`/`idxs`.
fn ragged_logprob_tuples(
    vals: &[f32],
    idxs: &[i32],
    lens: &[u32],
    texts: Option<&[String]>,
) -> serde_json::Value {
    let mut positions = Vec::with_capacity(lens.len());
    let mut off = 0usize;
    for &l in lens {
        let l = l as usize;
        if l == 0 {
            positions.push(serde_json::Value::Null);
        } else {
            let tuples: Vec<serde_json::Value> = (off..off + l)
                .map(|j| serde_json::json!([lp_value(vals[j]), idxs[j], text_slot(texts, j)]))
                .collect();
            positions.push(serde_json::Value::Array(tuples));
        }
        off += l;
    }
    serde_json::Value::Array(positions)
}

/// Reshape flat hidden-state f32s + per-row lengths into `meta_info`'s nested
/// `list[list[float]]` (one row per output position).
fn hidden_states_rows(vals: &[f32], lens: &[u32]) -> serde_json::Value {
    let mut rows = Vec::with_capacity(lens.len());
    let mut off = 0usize;
    for &l in lens {
        let l = l as usize;
        rows.push(serde_json::json!(&vals[off..(off + l).min(vals.len())]));
        off += l;
    }
    serde_json::Value::Array(rows)
}

/// Classify a terminal finish reason: `Some((code, message))` when it's an `abort`
/// carrying a `status_code` (a scheduler request error, e.g. over-context → 400).
/// Both unary + streaming paths inspect this instead of treating `Done` as normal.
pub(super) fn abort_status(finish_reason: &Option<serde_json::Value>) -> Option<(u16, String)> {
    let fr = finish_reason.as_ref()?;
    if fr.get("type").and_then(|t| t.as_str()) != Some("abort") {
        return None;
    }
    let code = fr.get("status_code").and_then(|s| s.as_u64())? as u16;
    let message = fr
        .get("message")
        .and_then(|m| m.as_str())
        .unwrap_or("request aborted")
        .to_string();
    Some((code, message))
}

/// The `{ "error": { message, code } }` object every error path emits (an SSE
/// event's data, a unary body, or one entry of a batch array).
pub(super) fn error_value(code: u16, message: &str) -> serde_json::Value {
    serde_json::json!({ "error": { "message": message, "code": code } })
}

/// Format a decoded [`ChunkEvent`] as one SGLang `/generate` frame's JSON. `rid`
/// (response `meta_info.id`) is passed as a string; the event's numeric `rid` is
/// just the shard routing key.
pub(super) fn sglang_frame_value(out: &ChunkEvent, rid: &str) -> serde_json::Value {
    let mut v = serde_json::json!({
        "text": out.text,
        "meta_info": {
            "id": rid,
            "prompt_tokens": out.prompt_tokens,
            "completion_tokens": out.completion_tokens,
            // Full dict (type + matched + message + status_code + …), or null.
            "finish_reason": out.finish_reason,
        },
    });
    if !out.token_ids.is_empty() {
        v["output_ids"] = serde_json::json!(out.token_ids);
    }
    // Logprobs + hidden states ride behind the boxed extras (absent for a plain
    // token/text frame). `[logprob, token_id, text|null]` tuples; text
    // (`return_text_in_logprobs`) was decoded on the detok shard into `*_txt`.
    let Some(ex) = out.extras.as_deref() else {
        return v;
    };
    if !ex.out_lp_val.is_empty() {
        v["meta_info"]["output_token_logprobs"] =
            logprob_tuples(&ex.out_lp_val, &ex.out_lp_idx, opt_texts(&ex.out_lp_txt));
    }
    if !ex.in_lp_val.is_empty() {
        v["meta_info"]["input_token_logprobs"] =
            logprob_tuples(&ex.in_lp_val, &ex.in_lp_idx, opt_texts(&ex.in_lp_txt));
    }
    if !ex.out_top_lens.is_empty() {
        v["meta_info"]["output_top_logprobs"] = ragged_logprob_tuples(
            &ex.out_top_val,
            &ex.out_top_idx,
            &ex.out_top_lens,
            opt_texts(&ex.out_top_txt),
        );
    }
    if !ex.in_top_lens.is_empty() {
        v["meta_info"]["input_top_logprobs"] = ragged_logprob_tuples(
            &ex.in_top_val,
            &ex.in_top_idx,
            &ex.in_top_lens,
            opt_texts(&ex.in_top_txt),
        );
    }
    if !ex.out_tid_lens.is_empty() {
        v["meta_info"]["output_token_ids_logprobs"] = ragged_logprob_tuples(
            &ex.out_tid_val,
            &ex.out_tid_idx,
            &ex.out_tid_lens,
            opt_texts(&ex.out_tid_txt),
        );
    }
    if !ex.in_tid_lens.is_empty() {
        v["meta_info"]["input_token_ids_logprobs"] = ragged_logprob_tuples(
            &ex.in_tid_val,
            &ex.in_tid_idx,
            &ex.in_tid_lens,
            opt_texts(&ex.in_tid_txt),
        );
    }
    if !ex.hidden_lens.is_empty() {
        v["meta_info"]["hidden_states"] = hidden_states_rows(&ex.hidden_val, &ex.hidden_lens);
    }
    v
}

/// Cumulative frame JSON from the accumulator's memoized ids/text — O(T), not O(T²).
/// Byte-identical to `sglang_frame_value(..).to_string()` (a `BTreeMap` keeps keys
/// alphabetical); pinned by `cumulative_frame_json_matches_serde`. `None` on extras.
pub(super) fn cumulative_frame_json(
    acc: &OutputAccumulator,
    rid: &str,
    index: Option<usize>,
) -> Option<String> {
    use std::fmt::Write;

    let o = acc.snapshot();
    if o.extras.is_some() {
        return None;
    }
    // Fixed size regardless of T, so this stays O(1) per frame.
    let meta = serde_json::json!({
        "id": rid,
        "prompt_tokens": o.prompt_tokens,
        "completion_tokens": o.completion_tokens,
        "finish_reason": o.finish_reason,
    })
    .to_string();

    let mut s = String::with_capacity(acc.text_json.len() + acc.ids_json.len() + meta.len() + 40);
    s.push('{');
    if let Some(i) = index {
        let _ = write!(s, "\"index\":{i},");
    }
    s.push_str("\"meta_info\":");
    s.push_str(&meta);
    if !acc.ids_json.is_empty() {
        s.push_str(",\"output_ids\":[");
        s.push_str(&acc.ids_json);
        s.push(']');
    }
    s.push_str(",\"text\":\"");
    s.push_str(&acc.text_json);
    s.push_str("\"}");
    Some(s)
}

/// Attach the batch `index` (batch streams only) and render to the SSE `data` text.
pub(super) fn tag_value(mut v: serde_json::Value, index: Option<usize>) -> String {
    if let Some(i) = index {
        v["index"] = serde_json::json!(i);
    }
    v.to_string()
}

/// One streaming frame's JSON: cumulative ignores `delta`, incremental ships it.
pub(super) fn stream_frame_string(
    delta: ChunkEvent,
    acc: &OutputAccumulator,
    incremental: bool,
    rid_str: &str,
    index: Option<usize>,
) -> String {
    if !incremental {
        return cumulative_frame_string(acc, rid_str, index);
    }
    tag_value(stream_frame_value(delta, acc, true, rid_str), index)
}

/// A cumulative frame's JSON, built purely from the accumulator (which is why a
/// backlog can coalesce to its last); falls back to the `Value` builder on extras.
pub(super) fn cumulative_frame_string(
    acc: &OutputAccumulator,
    rid_str: &str,
    index: Option<usize>,
) -> String {
    cumulative_frame_json(acc, rid_str, index)
        .unwrap_or_else(|| tag_value(sglang_frame_value(acc.snapshot(), rid_str), index))
}

/// Format one streaming frame: the accumulator's cumulative view (default), or this
/// step's delta with the cumulative token count in `meta_info` (matching Python).
pub(super) fn stream_frame_value(
    delta: ChunkEvent,
    acc: &OutputAccumulator,
    incremental: bool,
    rid_str: &str,
) -> serde_json::Value {
    if incremental {
        let mut d = delta;
        d.completion_tokens = acc.snapshot().completion_tokens;
        sglang_frame_value(&d, rid_str)
    } else {
        sglang_frame_value(acc.snapshot(), rid_str)
    }
}

/// Folds per-chunk [`ChunkEvent`] deltas into a cumulative view — used by the drain
/// loops needing cumulative output (every unary response + the cumulative SGLang
/// stream; OpenAI streaming forwards deltas and skips this). Holds a single
/// [`ChunkEvent`] so `snapshot` hands back a **borrow** per frame — no per-frame
/// clone of the growing buffers (that added O(T²) atop the wire's inherent O(T²)).
/// Shared with the [`openai`] submodule.
#[derive(Default)]
pub(super) struct OutputAccumulator {
    out: ChunkEvent,
    /// Serialized cumulative `output_ids` body (`"1,2,3"`, no brackets), appended per
    /// delta so a frame memcpy's it instead of rebuilding the array — O(T), not O(T²).
    ids_json: String,
    /// JSON-escaped cumulative text, without the surrounding quotes. Escaping is
    /// per-character, so `escape(a + b) == escape(a) + escape(b)` and deltas append.
    text_json: String,
}

/// Append `s` JSON-escaped (no surrounding quotes) — `serde_json` quotes it, and the
/// quotes are the first and last bytes of a string encoding.
fn push_escaped(dst: &mut String, s: &str) {
    if s.is_empty() {
        return;
    }
    let quoted = serde_json::to_string(s).expect("str-to-json should never fail");
    dst.push_str(&quoted[1..quoted.len() - 1]);
}

impl OutputAccumulator {
    /// Fold one delta frame in. Output families concatenate; input families and
    /// hidden states are set-once / last-writer-wins (they ride the prefill/final
    /// chunk), matching the Python `meta_info` assignment.
    pub(super) fn fold(&mut self, d: &ChunkEvent) {
        use std::fmt::Write;

        // Grow the memoized serializations alongside the raw cumulative buffers.
        push_escaped(&mut self.text_json, &d.text);
        for &id in &d.token_ids {
            if !self.ids_json.is_empty() {
                self.ids_json.push(',');
            }
            let _ = write!(self.ids_json, "{id}");
        }

        let o = &mut self.out;
        o.rid_hash = d.rid_hash; // constant across the request; keeps the accumulated view coherent
        o.text.push_str(&d.text);
        o.token_ids.extend_from_slice(&d.token_ids); // token_ids doubles as output_ids
        o.completion_tokens += d.completion_tokens;
        o.prompt_tokens = d.prompt_tokens; // constant across the request
        if d.finish_reason.is_some() {
            o.finish_reason = d.finish_reason.clone();
        }
        // Logprobs/hidden ride behind the boxed extras — most frames have none, so
        // only allocate the accumulator's box once a delta actually carries some.
        let Some(de) = d.extras.as_deref() else {
            return;
        };
        let oe = o
            .extras
            .get_or_insert_with(|| Box::new(ChunkExtras::default()));
        oe.out_lp_val.extend_from_slice(&de.out_lp_val);
        oe.out_lp_idx.extend_from_slice(&de.out_lp_idx);
        oe.out_top_val.extend_from_slice(&de.out_top_val);
        oe.out_top_idx.extend_from_slice(&de.out_top_idx);
        oe.out_top_lens.extend_from_slice(&de.out_top_lens);
        oe.out_tid_val.extend_from_slice(&de.out_tid_val);
        oe.out_tid_idx.extend_from_slice(&de.out_tid_idx);
        oe.out_tid_lens.extend_from_slice(&de.out_tid_lens);
        oe.out_lp_txt.extend_from_slice(&de.out_lp_txt);
        oe.out_top_txt.extend_from_slice(&de.out_top_txt);
        oe.out_tid_txt.extend_from_slice(&de.out_tid_txt);
        if !de.in_lp_val.is_empty() {
            oe.in_lp_val = de.in_lp_val.clone();
            oe.in_lp_idx = de.in_lp_idx.clone();
            oe.in_lp_txt = de.in_lp_txt.clone();
        }
        // Input families ride once (prefill); `lens` non-empty marks their arrival.
        if !de.in_top_lens.is_empty() {
            oe.in_top_val = de.in_top_val.clone();
            oe.in_top_idx = de.in_top_idx.clone();
            oe.in_top_lens = de.in_top_lens.clone();
            oe.in_top_txt = de.in_top_txt.clone();
        }
        if !de.in_tid_lens.is_empty() {
            oe.in_tid_val = de.in_tid_val.clone();
            oe.in_tid_idx = de.in_tid_idx.clone();
            oe.in_tid_lens = de.in_tid_lens.clone();
            oe.in_tid_txt = de.in_tid_txt.clone();
        }
        // Hidden states are non-cumulative: the latest non-empty set wins.
        if !de.hidden_lens.is_empty() {
            oe.hidden_val = de.hidden_val.clone();
            oe.hidden_lens = de.hidden_lens.clone();
        }
    }

    /// Borrow the cumulative output for an intermediate streaming frame.
    pub(super) fn snapshot(&self) -> &ChunkEvent {
        &self.out
    }

    /// Consume into the final cumulative output.
    pub(super) fn into_output(self) -> ChunkEvent {
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_logprob_tuples_shape() {
        let v = logprob_tuples(&[-0.5, -1.5], &[10, 20], None);
        assert_eq!(
            v,
            serde_json::json!([
                [-0.5f32, 10, serde_json::Value::Null],
                [-1.5f32, 20, serde_json::Value::Null]
            ])
        );
    }

    /// With a text buffer, the tuple's third slot carries the decoded token.
    #[test]
    fn flat_logprob_tuples_with_text() {
        let texts = vec!["a".to_string(), "b".to_string()];
        let v = logprob_tuples(&[-0.5, -1.5], &[10, 20], Some(&texts));
        assert_eq!(
            v,
            serde_json::json!([[-0.5f32, 10, "a"], [-1.5f32, 20, "b"]])
        );
    }

    /// Ragged reshape restores null positions (len 0) — mirrors
    /// detokenize_top_logprobs_tokens emitting None for empty positions.
    #[test]
    fn ragged_logprob_tuples_restores_null_positions() {
        // 2 positions: first null (len 0), second k=1.
        let v = ragged_logprob_tuples(&[-0.3], &[9], &[0, 1], None);
        assert_eq!(
            v,
            serde_json::json!([
                serde_json::Value::Null,
                [[-0.3f32, 9, serde_json::Value::Null]]
            ])
        );
    }

    /// The `NaN` sentinel (the Python `None` logprob for the first prompt token)
    /// becomes a JSON `null` logprob, while its token id in the parallel `idx`
    /// column is preserved. Guards the scheduler-killing prompt-logprob crash.
    #[test]
    fn nan_sentinel_becomes_null_logprob() {
        // Flat (input/output logprobs): first value absent, second present.
        let flat = logprob_tuples(&[f32::NAN, -0.5], &[10, 20], None);
        assert_eq!(
            flat,
            serde_json::json!([
                [serde_json::Value::Null, 10, serde_json::Value::Null],
                [-0.5f32, 20, serde_json::Value::Null],
            ])
        );
        // Ragged (top-k / token-ids logprobs): a NaN inside a position → null.
        let ragged = ragged_logprob_tuples(&[f32::NAN], &[7], &[1], None);
        assert_eq!(
            ragged,
            serde_json::json!([[[serde_json::Value::Null, 7, serde_json::Value::Null]]])
        );
    }

    /// End-to-end: a `ChunkEvent` carrying a prompt-logprob request (first input
    /// logprob is the `NaN` sentinel) formats without panicking and emits
    /// `input_token_logprobs` with a leading `[null, token_id, text]`.
    #[test]
    fn prompt_logprob_frame_emits_null_first() {
        let out = ChunkEvent {
            extras: Some(Box::new(ChunkExtras {
                in_lp_val: vec![f32::NAN, -0.5],
                in_lp_idx: vec![10, 20],
                in_lp_txt: vec!["<s>".into(), "hi".into()],
                ..Default::default()
            })),
            ..Default::default()
        };
        let frame = sglang_frame_value(&out, "1");
        assert_eq!(
            frame["meta_info"]["input_token_logprobs"],
            serde_json::json!([[serde_json::Value::Null, 10, "<s>"], [-0.5f32, 20, "hi"]])
        );
    }

    /// The accumulator folds deltas cumulatively and `snapshot` borrows the
    /// running state (no per-frame clone); `into_output` moves the same state.
    #[test]
    fn accumulator_snapshot_is_cumulative() {
        let mut acc = OutputAccumulator::default();
        acc.fold(&ChunkEvent {
            text: "he".into(),
            token_ids: vec![1, 2],
            completion_tokens: 2,
            ..Default::default()
        });
        {
            let s = acc.snapshot();
            assert_eq!(s.text, "he");
            assert_eq!(s.token_ids, vec![1, 2]);
        }
        acc.fold(&ChunkEvent {
            text: "llo".into(),
            token_ids: vec![3],
            completion_tokens: 1,
            ..Default::default()
        });
        {
            let s = acc.snapshot();
            assert_eq!(s.text, "hello"); // cumulative
            assert_eq!(s.token_ids, vec![1, 2, 3]);
            assert_eq!(s.completion_tokens, 3);
        }
        let out = acc.into_output();
        assert_eq!(out.text, "hello");
    }

    /// A populated text column (decoded on the detok shard) → `Some`; empty
    /// (`return_text_in_logprobs` off) → `None` → null text slots.
    #[test]
    fn opt_texts_gates_on_population() {
        assert!(opt_texts(&[]).is_none());
        let t = vec!["x".to_string()];
        assert_eq!(opt_texts(&t), Some(t.as_slice()));
    }

    /// The shared classifier both paths use: a validation abort yields its
    /// `(code, message)` (the streaming path turns this into an SSE error event
    /// instead of a normal `Done` frame); anything else yields `None`.
    #[test]
    fn abort_status_extracts_code_and_message() {
        let (code, msg) = abort_status(&Some(serde_json::json!({
            "type": "abort", "message": "over the limit", "status_code": 400
        })))
        .expect("validation abort → (code, message)");
        assert_eq!(code, 400);
        assert_eq!(msg, "over the limit");
        // Normal finish, bare abort (no status), and no finish → not an error.
        assert!(abort_status(&Some(serde_json::json!({"type": "stop"}))).is_none());
        assert!(abort_status(&Some(serde_json::json!({"type": "abort"}))).is_none());
        assert!(abort_status(&None).is_none());
    }

    /// A normal finish, a bare abort (no status), and no finish are not errors
    /// (the unary path returns them as a 200 result frame).
    #[test]
    fn non_error_finishes_stay_ok() {
        assert!(abort_status(&Some(serde_json::json!({"type": "stop", "matched": 5}))).is_none());
        assert!(abort_status(&Some(serde_json::json!({"type": "length", "length": 8}))).is_none());
        assert!(
            abort_status(&Some(
                serde_json::json!({"type": "abort", "message": "Aborted"})
            ))
            .is_none()
        );
        assert!(abort_status(&None).is_none());
    }

    /// The memoized cumulative fast path must emit **byte-identical** JSON to the
    /// `serde_json::Value` builder it replaces — same keys, same alphabetical order
    /// (`Map` is a `BTreeMap`; no `preserve_order`), same escaping. Covers unicode
    /// and control chars, an empty-ids first frame, a finish_reason, and the batch
    /// `index`. Guards the O(T) rewrite of the O(T²) `output_ids` serialization.
    #[test]
    fn cumulative_frame_json_matches_serde() {
        let deltas = [
            ChunkEvent {
                rid_hash: 7,
                text: String::new(),
                token_ids: vec![],
                completion_tokens: 0,
                prompt_tokens: 128,
                ..Default::default()
            },
            ChunkEvent {
                rid_hash: 7,
                text: "He\"llo\n\t".into(),
                token_ids: vec![1000],
                completion_tokens: 1,
                prompt_tokens: 128,
                ..Default::default()
            },
            ChunkEvent {
                rid_hash: 7,
                text: " 世界 🌍 \\".into(),
                token_ids: vec![-2, 3],
                completion_tokens: 2,
                prompt_tokens: 128,
                ..Default::default()
            },
            ChunkEvent {
                rid_hash: 7,
                text: "!".into(),
                token_ids: vec![9],
                completion_tokens: 1,
                prompt_tokens: 128,
                finish_reason: Some(serde_json::json!({"type": "stop", "matched": 9})),
                ..Default::default()
            },
        ];

        for index in [None, Some(3usize)] {
            let mut acc = OutputAccumulator::default();
            for d in &deltas {
                acc.fold(d);
                let fast = cumulative_frame_json(&acc, "7", index).expect("no extras → fast path");
                let slow = tag_value(sglang_frame_value(acc.snapshot(), "7"), index);
                assert_eq!(fast, slow, "index={index:?} text={:?}", acc.snapshot().text);
            }
        }
    }

    /// A frame carrying logprobs falls back to the `Value` builder (the fast path
    /// only knows the plain text/ids shape).
    #[test]
    fn cumulative_frame_json_defers_on_extras() {
        let mut acc = OutputAccumulator::default();
        acc.fold(&ChunkEvent {
            rid_hash: 1,
            token_ids: vec![5],
            text: "x".into(),
            extras: Some(Box::new(ChunkExtras {
                out_lp_val: vec![-0.5],
                out_lp_idx: vec![5],
                ..Default::default()
            })),
            ..Default::default()
        });
        assert!(cumulative_frame_json(&acc, "1", None).is_none());
    }
}
