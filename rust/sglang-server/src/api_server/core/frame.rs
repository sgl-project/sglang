//! Frame shaping for the native `/generate` protocol: the cumulative
//! [`OutputAccumulator`] plus the functions that shape [`ChunkEvent`]s /
//! accumulated state into the generated `GenerateResponse` (`meta_info`,
//! logprob tuples, error and abort frames) and serialize it for the HTTP wire.
//! No HTTP here — the sibling `native_api` module owns the handlers and
//! streams; it calls these per frame.

use sglang_api_types::api::v1 as genapi;

use crate::message::response::{ChunkEvent, ChunkExtras};

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

/// Append a flat family's `[logprob, token_id, text]` tuples to `dst`, comma
/// separated and WITHOUT the enclosing brackets, so a cumulative frame can
/// concatenate each delta instead of re-rendering every accumulated position.
///
/// Byte-identical to the serialization of [`typed_logprob_entries`]: `serde_json`
/// writes an array as `[`, elements joined by `,`, `]` with no spaces, floats by
/// the same ryu encoding.
fn push_logprob_tuples(dst: &mut String, vals: &[f32], idxs: &[i32], texts: Option<&[String]>) {
    use std::fmt::Write;
    for (j, (&v, &tid)) in vals.iter().zip(idxs.iter()).enumerate() {
        if !dst.is_empty() {
            dst.push(',');
        }
        let _ = write!(dst, "[{},{tid},{}]", lp_value(v), text_slot(texts, j));
    }
}

/// Ragged counterpart of [`push_logprob_tuples`] — one entry per position, `null`
/// where `lens[p] == 0`. Mirrors [`typed_ragged_rows`] including its
/// bounds-checked skip, so a header whose `lens` run past the value buffer yields
/// the same (shortened) row rather than panicking.
fn push_ragged_tuples(
    dst: &mut String,
    vals: &[f32],
    idxs: &[i32],
    lens: &[u32],
    texts: Option<&[String]>,
) {
    use std::fmt::Write;
    let mut off = 0usize;
    for &l in lens {
        let l = l as usize;
        if !dst.is_empty() {
            dst.push(',');
        }
        if l == 0 {
            dst.push_str("null");
        } else {
            dst.push('[');
            let mut first = true;
            for j in off..off + l {
                let (Some(&v), Some(&tid)) = (vals.get(j), idxs.get(j)) else {
                    continue;
                };
                if !first {
                    dst.push(',');
                }
                first = false;
                let _ = write!(dst, "[{},{tid},{}]", lp_value(v), text_slot(texts, j));
            }
            dst.push(']');
        }
        off += l;
    }
}

/// Render one set-once family (they ride the prefill or the final chunk) into a
/// standalone JSON array, so it is serialized when it arrives rather than on every
/// subsequent frame.
fn ragged_array_json(vals: &[f32], idxs: &[i32], lens: &[u32], texts: Option<&[String]>) -> String {
    let mut body = String::new();
    push_ragged_tuples(&mut body, vals, idxs, lens, texts);
    format!("[{body}]")
}

/// The native error body — the same `{"error": {...}}` object every native
/// path emits, not bare text, which a client parsing JSON chokes on.
pub(crate) fn error_value(code: u16, message: &str) -> serde_json::Value {
    serde_json::json!({ "error": { "message": message, "code": code } })
}

/// Attach the batch `index` (batch streams only) and serialize a typed frame to
/// its wire JSON — the HTTP rendering of every non-memoized frame.
pub(crate) fn typed_frame_string(
    mut frame: genapi::GenerateResponse,
    index: Option<usize>,
) -> String {
    frame.index = index.map(|i| u32::try_from(i).unwrap_or(u32::MAX));
    serde_json::to_string(&frame).expect("a generated frame always serializes")
}

/// Cumulative frame JSON from the accumulator's memoized parts — O(1) in the
/// accumulated length, where re-rendering the typed frame is O(T) per frame and
/// so O(T²) per request.
///
/// Encodes the same bytes as `typed_frame_string(frame_typed(..))` — the
/// generated serializer's field-number key order, same values, same escaping —
/// pinned for both the plain and the logprob shapes by
/// `cumulative_frame_json_matches_typed*`.
pub(crate) fn cumulative_frame_json(
    acc: &OutputAccumulator,
    rid: &str,
    index: Option<usize>,
) -> Option<String> {
    use std::fmt::Write;

    if acc.extras_memo_broken {
        return None;
    }
    let o = acc.snapshot();
    // `to_string` on the typed reason: the same encoder the generated
    // `GenerateMetaInfo` serializer runs it through, so any representation
    // quirk is reproduced instead of re-derived.
    let finish = serde_json::to_string(&o.finish_reason).ok()?;

    let mut m = String::new();
    let _ = write!(m, "{{\"id\":{}", serde_json::Value::String(rid.to_string()));
    let _ = write!(m, ",\"prompt_tokens\":{}", o.prompt_tokens);
    let _ = write!(m, ",\"completion_tokens\":{}", o.completion_tokens);
    let _ = write!(m, ",\"finish_reason\":{finish}");
    // Input+output token logprobs are emitted as a PAIR whenever either side has
    // data (empty list included), matching `frame_typed` byte for byte — see the
    // PD-router rationale there.
    let lp_pair = acc.in_lp_json.is_some() || !acc.out_lp_json.is_empty();
    if lp_pair {
        let _ = write!(m, ",\"output_token_logprobs\":[{}]", acc.out_lp_json);
        let v = acc.in_lp_json.as_deref().unwrap_or("[]");
        let _ = write!(m, ",\"input_token_logprobs\":{v}");
    }
    // The typed path keys these off the source columns being non-empty; an empty
    // source renders to an empty body, so the two guards coincide.
    if !acc.out_top_json.is_empty() {
        let _ = write!(m, ",\"output_top_logprobs\":[{}]", acc.out_top_json);
    }
    if let Some(v) = &acc.in_top_json {
        let _ = write!(m, ",\"input_top_logprobs\":{v}");
    }
    if !acc.out_tid_json.is_empty() {
        let _ = write!(m, ",\"output_token_ids_logprobs\":[{}]", acc.out_tid_json);
    }
    if let Some(v) = &acc.in_tid_json {
        let _ = write!(m, ",\"input_token_ids_logprobs\":{v}");
    }
    if let Some(h) = &acc.hidden_json {
        let _ = write!(m, ",\"hidden_states\":{h}");
    }
    m.push('}');

    let mut s = String::with_capacity(acc.text_json.len() + acc.ids_json.len() + m.len() + 40);
    s.push_str("{\"text\":\"");
    s.push_str(&acc.text_json);
    s.push_str("\",\"meta_info\":");
    s.push_str(&m);
    if !acc.ids_json.is_empty() {
        s.push_str(",\"output_ids\":[");
        s.push_str(&acc.ids_json);
        s.push(']');
    }
    if let Some(i) = index {
        let _ = write!(s, ",\"index\":{i}");
    }
    s.push('}');
    Some(s)
}

/// Attach the batch `index` (batch streams only) and render to the SSE `data` text.
pub(crate) fn tag_value(mut v: serde_json::Value, index: Option<usize>) -> String {
    if let Some(i) = index {
        v["index"] = serde_json::json!(i);
    }
    v.to_string()
}

/// A cumulative frame's JSON, built purely from the accumulator (which is why a
/// backlog can coalesce to its last); falls back to serializing the typed frame
/// when the extras memo is broken.
pub(crate) fn cumulative_frame_string(
    acc: &OutputAccumulator,
    rid_str: &str,
    index: Option<usize>,
) -> String {
    cumulative_frame_json(acc, rid_str, index)
        .unwrap_or_else(|| typed_frame_string(frame_typed(acc.snapshot(), rid_str), index))
}

/// Typed twin of [`lp_value`]: the `NaN` sentinel becomes the absent (JSON
/// null) logprob slot.
fn typed_lp_slot(v: f32) -> Option<f64> {
    (!v.is_nan()).then(|| f64::from(v))
}

/// Typed twin of one `[logprob, token_id, text|null]` tuple.
fn typed_entry(v: f32, tid: i32, texts: Option<&[String]>, j: usize) -> genapi::LogprobEntry {
    genapi::LogprobEntry {
        logprob: typed_lp_slot(v),
        token_id: i64::from(tid),
        text: texts.and_then(|t| t.get(j)).cloned(),
    }
}

/// SGLang logprob shape: a list of `[logprob, token_id, text]` tuples. `texts`
/// (parallel to `idxs`) fills the text slot when set, else `null`.
fn typed_logprob_entries(
    vals: &[f32],
    idxs: &[i32],
    texts: Option<&[String]>,
) -> genapi::LogprobEntries {
    genapi::LogprobEntries {
        entries: vals
            .iter()
            .zip(idxs.iter())
            .enumerate()
            .map(|(j, (&v, &tid))| typed_entry(v, tid, texts, j))
            .collect(),
    }
}

/// Ragged top-k / token-ids shape: one entry per position — a row of tuples, or
/// null when `lens[p] == 0` (mirrors `detokenize_top_logprobs_tokens`). Entries
/// are bounds-checked: a header whose `lens` run past the value buffer yields a
/// shortened row rather than panicking the api thread.
fn typed_ragged_rows(
    vals: &[f32],
    idxs: &[i32],
    lens: &[u32],
    texts: Option<&[String]>,
) -> Vec<genapi::NullableTopLogprobs> {
    let mut positions = Vec::with_capacity(lens.len());
    let mut off = 0usize;
    for &l in lens {
        let l = l as usize;
        let row = (l != 0).then(|| genapi::TopLogprobRow {
            entries: (off..off + l)
                .filter_map(|j| Some(typed_entry(*vals.get(j)?, *idxs.get(j)?, texts, j)))
                .collect(),
        });
        positions.push(genapi::NullableTopLogprobs { row });
        off += l;
    }
    positions
}

/// Reshape flat hidden-state f32s + per-row lengths into `meta_info`'s nested
/// `list[list[float]]` (one row per output position).
fn typed_hidden_rows(vals: &[f32], lens: &[u32]) -> Vec<genapi::HiddenStateRow> {
    let mut rows = Vec::with_capacity(lens.len());
    let mut off = 0usize;
    for &l in lens {
        let l = l as usize;
        // `get`, not a clamped index: clamping only the END leaves `off` past
        // `vals.len()` after one over-long row, making the next range reversed
        // — a panic on the api thread rather than an empty row.
        rows.push(genapi::HiddenStateRow {
            values: vals
                .get(off..off + l)
                .unwrap_or(&[])
                .iter()
                .map(|&v| f64::from(v))
                .collect(),
        });
        off += l;
    }
    rows
}

/// Shape a decoded [`ChunkEvent`] as one SGLang `/generate` frame — the single
/// shaping source for both transports: gRPC ships the struct, HTTP serializes it
/// ([`typed_frame_string`]), and the memoized cumulative path reproduces its
/// bytes ([`cumulative_frame_json`]). `rid` (response `meta_info.id`) is passed
/// as a string; the event's numeric `rid` is just the shard routing key.
pub(crate) fn frame_typed(out: &ChunkEvent, rid: &str) -> genapi::GenerateResponse {
    let mut meta = genapi::GenerateMetaInfo {
        id: rid.to_string(),
        prompt_tokens: out.prompt_tokens,
        completion_tokens: out.completion_tokens,
        finish_reason: out.finish_reason.clone(),
        ..Default::default()
    };
    if let Some(ex) = out.extras.as_deref() {
        // Python (`add_logprob_to_meta_info`) always sets input+output token
        // logprobs together, empty lists included. A PD decode node never receives
        // input logprobs (they belong to prefill), yet its response must still
        // carry the key — the PD router keys its merge of prefill's
        // `input_token_logprobs` on its presence.
        if !ex.out_lp_val.is_empty() || !ex.in_lp_val.is_empty() {
            meta.output_token_logprobs = Some(typed_logprob_entries(
                &ex.out_lp_val,
                &ex.out_lp_idx,
                opt_texts(&ex.out_lp_txt),
            ));
            meta.input_token_logprobs = Some(typed_logprob_entries(
                &ex.in_lp_val,
                &ex.in_lp_idx,
                opt_texts(&ex.in_lp_txt),
            ));
        }
        // Empty `lens` yield the empty (JSON-absent) default, so no guards needed.
        meta.output_top_logprobs = typed_ragged_rows(
            &ex.out_top_val,
            &ex.out_top_idx,
            &ex.out_top_lens,
            opt_texts(&ex.out_top_txt),
        );
        meta.input_top_logprobs = typed_ragged_rows(
            &ex.in_top_val,
            &ex.in_top_idx,
            &ex.in_top_lens,
            opt_texts(&ex.in_top_txt),
        );
        meta.output_token_ids_logprobs = typed_ragged_rows(
            &ex.out_tid_val,
            &ex.out_tid_idx,
            &ex.out_tid_lens,
            opt_texts(&ex.out_tid_txt),
        );
        meta.input_token_ids_logprobs = typed_ragged_rows(
            &ex.in_tid_val,
            &ex.in_tid_idx,
            &ex.in_tid_lens,
            opt_texts(&ex.in_tid_txt),
        );
        meta.hidden_states = typed_hidden_rows(&ex.hidden_val, &ex.hidden_lens);
    }
    genapi::GenerateResponse {
        text: out.text.clone(),
        meta_info: Some(meta),
        output_ids: (!out.token_ids.is_empty()).then(|| genapi::TokenIds {
            ids: out.token_ids.clone(),
        }),
        index: None,
    }
}

/// Shape one streaming frame: the accumulator's cumulative view (default), or
/// this step's delta with the cumulative token count in `meta_info` (matching
/// Python).
pub(crate) fn stream_frame_typed(
    delta: ChunkEvent,
    acc: &OutputAccumulator,
    incremental: bool,
    rid_str: &str,
) -> genapi::GenerateResponse {
    if incremental {
        let mut d = delta;
        d.completion_tokens = acc.snapshot().completion_tokens;
        frame_typed(&d, rid_str)
    } else {
        frame_typed(acc.snapshot(), rid_str)
    }
}

/// Folds per-chunk [`ChunkEvent`] deltas into a cumulative view — used by the drain
/// loops needing cumulative output (every unary response + the cumulative SGLang
/// stream; OpenAI streaming forwards deltas and skips this). Holds a single
/// [`ChunkEvent`] so `snapshot` hands back a **borrow** per frame — no per-frame
/// clone of the growing buffers (that added O(T²) atop the wire's inherent O(T²)).
/// Shared with the [`openai`] submodule.
#[derive(Default)]
pub(crate) struct OutputAccumulator {
    out: ChunkEvent,
    /// Serialized cumulative `output_ids` body (`"1,2,3"`, no brackets), appended per
    /// delta so a frame memcpy's it instead of rebuilding the array — O(T), not O(T²).
    ids_json: String,
    /// JSON-escaped cumulative text, without the surrounding quotes. Escaping is
    /// per-character, so `escape(a + b) == escape(a) + escape(b)` and deltas append.
    text_json: String,
    /// Memoized bodies (no enclosing brackets) of the three CUMULATIVE logprob
    /// families, appended per delta — the same O(T) trick `ids_json` uses, extended
    /// to the families that made a cumulative stream with logprobs O(T²). Cumulative
    /// is SGLang's default, so that path re-rendered every accumulated position on
    /// every frame: measured 117 ms for one 500-token top-5 request, versus 1.2 ms
    /// incremental.
    out_lp_json: String,
    out_top_json: String,
    out_tid_json: String,
    /// Set-once families — they ride the prefill or the final chunk, so they are
    /// rendered when they arrive rather than on every frame after.
    in_lp_json: Option<String>,
    in_top_json: Option<String>,
    in_tid_json: Option<String>,
    hidden_json: Option<String>,
    /// Set once a family's text column falls out of lockstep with its values, at
    /// which point the memo is abandoned for the typed serialization.
    ///
    /// Appending per delta assumes `text_slot(accumulated, global_j)` equals
    /// `text_slot(delta, local_j)`, which holds only while every delta supplies
    /// either a text per value or none at all. That is what a real request does —
    /// `return_text_in_logprobs` is per-request, so the detok shard fills `*_txt`
    /// for all deltas or none — but a mixed sequence would silently diverge from the
    /// typed serialization, so it is detected rather than assumed.
    extras_memo_broken: bool,
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
    pub(crate) fn fold(&mut self, d: &ChunkEvent) {
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
        o.rid.clone_from(&d.rid); // constant across the request; keeps the accumulated view coherent
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
        // Append THIS delta's tuples, indexed within the delta — equivalent to
        // indexing the accumulated arrays only while texts stay in lockstep, which
        // the guard below verifies.
        push_logprob_tuples(
            &mut self.out_lp_json,
            &de.out_lp_val,
            &de.out_lp_idx,
            opt_texts(&de.out_lp_txt),
        );
        push_ragged_tuples(
            &mut self.out_top_json,
            &de.out_top_val,
            &de.out_top_idx,
            &de.out_top_lens,
            opt_texts(&de.out_top_txt),
        );
        push_ragged_tuples(
            &mut self.out_tid_json,
            &de.out_tid_val,
            &de.out_tid_idx,
            &de.out_tid_lens,
            opt_texts(&de.out_tid_txt),
        );
        let lockstep = |txt: &Vec<String>, val: &Vec<f32>| txt.is_empty() || txt.len() == val.len();
        if !lockstep(&oe.out_lp_txt, &oe.out_lp_val)
            || !lockstep(&oe.out_top_txt, &oe.out_top_val)
            || !lockstep(&oe.out_tid_txt, &oe.out_tid_val)
        {
            self.extras_memo_broken = true;
        }
        if !de.in_lp_val.is_empty() {
            oe.in_lp_val = de.in_lp_val.clone();
            oe.in_lp_idx = de.in_lp_idx.clone();
            oe.in_lp_txt = de.in_lp_txt.clone();
            let mut body = String::new();
            push_logprob_tuples(
                &mut body,
                &oe.in_lp_val,
                &oe.in_lp_idx,
                opt_texts(&oe.in_lp_txt),
            );
            self.in_lp_json = Some(format!("[{body}]"));
        }
        // Input families ride once (prefill); `lens` non-empty marks their arrival.
        if !de.in_top_lens.is_empty() {
            oe.in_top_val = de.in_top_val.clone();
            oe.in_top_idx = de.in_top_idx.clone();
            oe.in_top_lens = de.in_top_lens.clone();
            oe.in_top_txt = de.in_top_txt.clone();
            self.in_top_json = Some(ragged_array_json(
                &oe.in_top_val,
                &oe.in_top_idx,
                &oe.in_top_lens,
                opt_texts(&oe.in_top_txt),
            ));
        }
        if !de.in_tid_lens.is_empty() {
            oe.in_tid_val = de.in_tid_val.clone();
            oe.in_tid_idx = de.in_tid_idx.clone();
            oe.in_tid_lens = de.in_tid_lens.clone();
            oe.in_tid_txt = de.in_tid_txt.clone();
            self.in_tid_json = Some(ragged_array_json(
                &oe.in_tid_val,
                &oe.in_tid_idx,
                &oe.in_tid_lens,
                opt_texts(&oe.in_tid_txt),
            ));
        }
        // Hidden states are non-cumulative: the latest non-empty set wins.
        if !de.hidden_lens.is_empty() {
            oe.hidden_val = de.hidden_val.clone();
            oe.hidden_lens = de.hidden_lens.clone();
            self.hidden_json = Some(
                serde_json::to_string(&typed_hidden_rows(&oe.hidden_val, &oe.hidden_lens))
                    .expect("hidden rows always serialize"),
            );
        }
    }

    /// Borrow the cumulative output for an intermediate streaming frame.
    pub(crate) fn snapshot(&self) -> &ChunkEvent {
        &self.out
    }

    /// Consume into the final cumulative output.
    pub(crate) fn into_output(self) -> ChunkEvent {
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::finish_reason::FinishReason;

    fn fr(v: serde_json::Value) -> Option<FinishReason> {
        Some(serde_json::from_value(v).expect("finish reason must parse"))
    }

    /// Frames spanning every `meta_info` family, the NaN logprob sentinel, null
    /// ragged rows, lens overrunning their buffers, decoded tuple texts, the
    /// input/output logprob pairing rule, and each finish-reason shape
    /// (unknown-type passthrough included).
    fn frame_corpus() -> Vec<ChunkEvent> {
        vec![
            ChunkEvent {
                rid: "1".into(),
                text: "héllo\n".into(),
                prompt_tokens: 3,
                completion_tokens: 1,
                ..Default::default()
            },
            ChunkEvent {
                rid: "1".into(),
                text: "done".into(),
                token_ids: vec![5, 6],
                prompt_tokens: 2,
                completion_tokens: 2,
                finish_reason: fr(serde_json::json!({"type": "stop", "matched": 5})),
                ..Default::default()
            },
            ChunkEvent {
                rid: "1".into(),
                finish_reason: fr(serde_json::json!({
                    "type": "abort", "message": "over the limit",
                    "status_code": 400, "err_type": "BadRequestError"
                })),
                ..Default::default()
            },
            ChunkEvent {
                rid: "1".into(),
                finish_reason: fr(serde_json::json!({"type": "tool_calls", "name": "search"})),
                ..Default::default()
            },
            // Output logprobs alone still emit the input/output pair; NaN and a
            // decimal a f32 cannot represent exactly probe the null sentinel and
            // the f32→f64 widening.
            ChunkEvent {
                rid: "1".into(),
                text: "t".into(),
                prompt_tokens: 1,
                completion_tokens: 3,
                finish_reason: fr(serde_json::json!({"type": "length", "length": 3})),
                extras: Some(Box::new(ChunkExtras {
                    out_lp_val: vec![f32::NAN, -0.5, 0.1],
                    out_lp_idx: vec![1, 2, 3],
                    ..Default::default()
                })),
                ..Default::default()
            },
            // Input logprobs alone (the PD prefill shape) pair the same way.
            ChunkEvent {
                rid: "1".into(),
                extras: Some(Box::new(ChunkExtras {
                    in_lp_val: vec![0.25],
                    in_lp_idx: vec![9],
                    ..Default::default()
                })),
                ..Default::default()
            },
            // Every family at once, with decoded texts and a zero-length (null)
            // top-logprob row.
            ChunkEvent {
                rid: "1".into(),
                text: "full".into(),
                token_ids: vec![1],
                prompt_tokens: 4,
                completion_tokens: 5,
                extras: Some(Box::new(ChunkExtras {
                    out_lp_val: vec![-0.1, -0.2],
                    out_lp_idx: vec![1, 2],
                    out_lp_txt: vec!["a".into(), "b".into()],
                    in_lp_val: vec![f32::NAN, -1.1],
                    in_lp_idx: vec![7, 8],
                    in_lp_txt: vec!["p".into(), "q".into()],
                    out_top_val: vec![-0.3, -0.4, -0.5],
                    out_top_idx: vec![3, 4, 5],
                    out_top_lens: vec![2, 0, 1],
                    out_top_txt: vec!["x".into(), "y".into(), "z".into()],
                    in_top_val: vec![-0.6],
                    in_top_idx: vec![6],
                    in_top_lens: vec![1],
                    out_tid_val: vec![-0.7, -0.8],
                    out_tid_idx: vec![10, 11],
                    out_tid_lens: vec![2],
                    in_tid_val: vec![-0.9],
                    in_tid_idx: vec![12],
                    in_tid_lens: vec![0, 1],
                    hidden_val: vec![0.1, 0.2, 0.3, 0.4, 0.5],
                    hidden_lens: vec![2, 3],
                    ..Default::default()
                })),
                ..Default::default()
            },
            // Corrupt lens overrunning their buffers: the bounds-checked skip and
            // the empty hidden row, not a panic.
            ChunkEvent {
                rid: "1".into(),
                extras: Some(Box::new(ChunkExtras {
                    out_top_val: vec![-0.1, -0.2],
                    out_top_idx: vec![1, 2],
                    out_top_lens: vec![5],
                    hidden_val: vec![1.0],
                    hidden_lens: vec![3, 1],
                    ..Default::default()
                })),
                ..Default::default()
            },
        ]
    }

    /// The typed frame must survive the JSON round trip: parsing
    /// `typed_frame_string`'s output back through the generated deserializer
    /// reproduces the struct, so the wire loses nothing — presence rules
    /// (the logprob pair, empty families, `output_ids`) encode faithfully.
    #[test]
    fn typed_frame_round_trips_through_its_wire_json() {
        for out in frame_corpus() {
            for index in [None, Some(3usize)] {
                let mut typed = frame_typed(&out, "rid-x");
                typed.index = index.map(|i| u32::try_from(i).unwrap());
                let wire = typed_frame_string(typed.clone(), index);
                let parsed: genapi::GenerateResponse =
                    serde_json::from_str(&wire).expect("a typed frame's wire JSON parses back");
                assert_eq!(parsed, typed, "round trip diverged on {out:?}");
            }
        }
    }

    /// For every corpus shape the memoized cumulative path must emit the same
    /// BYTES as serializing the typed frame — the memo is an encoding of
    /// `frame_typed`, never a second shaping.
    #[test]
    fn cumulative_memo_matches_typed_serialization_across_corpus() {
        for delta in frame_corpus() {
            for index in [None, Some(3usize)] {
                let mut acc = OutputAccumulator::default();
                acc.fold(&delta);
                let fast = cumulative_frame_json(&acc, "rid-x", index)
                    .expect("each corpus event alone keeps the memo valid");
                let slow = typed_frame_string(frame_typed(acc.snapshot(), "rid-x"), index);
                assert_eq!(fast, slow, "memo diverged on {delta:?}");
            }
        }
    }

    /// An incremental frame ships this step's delta text but the CUMULATIVE
    /// completion count (matching Python); cumulative ships the snapshot.
    #[test]
    fn stream_frame_typed_incremental_overrides_token_count() {
        let mut acc = OutputAccumulator::default();
        let d1 = ChunkEvent {
            rid: "1".into(),
            text: "he".into(),
            completion_tokens: 2,
            ..Default::default()
        };
        let d2 = ChunkEvent {
            rid: "1".into(),
            text: "llo".into(),
            completion_tokens: 1,
            ..Default::default()
        };
        acc.fold(&d1);
        acc.fold(&d2);
        let inc = stream_frame_typed(d2.clone(), &acc, true, "rid");
        assert_eq!(inc.text, "llo");
        assert_eq!(inc.meta_info.unwrap().completion_tokens, 3);
        let cum = stream_frame_typed(d2, &acc, false, "rid");
        assert_eq!(cum.text, "hello");
    }

    fn to_json<T: serde::Serialize>(v: &T) -> serde_json::Value {
        serde_json::to_value(v).expect("shaping output serializes")
    }

    #[test]
    fn flat_logprob_tuples_shape() {
        let v = to_json(&typed_logprob_entries(&[-0.5, -1.5], &[10, 20], None));
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
        let v = to_json(&typed_logprob_entries(
            &[-0.5, -1.5],
            &[10, 20],
            Some(&texts),
        ));
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
        let v = to_json(&typed_ragged_rows(&[-0.3], &[9], &[0, 1], None));
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
        let flat = to_json(&typed_logprob_entries(&[f32::NAN, -0.5], &[10, 20], None));
        assert_eq!(
            flat,
            serde_json::json!([
                [serde_json::Value::Null, 10, serde_json::Value::Null],
                [-0.5f32, 20, serde_json::Value::Null],
            ])
        );
        // Ragged (top-k / token-ids logprobs): a NaN inside a position → null.
        let ragged = to_json(&typed_ragged_rows(&[f32::NAN], &[7], &[1], None));
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
        let frame = to_json(&frame_typed(&out, "1"));
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

    /// The memoized cumulative fast path must emit the **same bytes** as
    /// serializing the typed frame it replaces — same key order, same values,
    /// same escaping. Covers unicode and control chars, an empty-ids first
    /// frame, a finish_reason, and the batch `index`. Guards the O(T) rewrite
    /// of the O(T²) `output_ids` serialization.
    #[test]
    fn cumulative_frame_json_matches_typed() {
        let deltas = [
            ChunkEvent {
                rid: "7".into(),
                text: String::new(),
                token_ids: vec![],
                completion_tokens: 0,
                prompt_tokens: 128,
                ..Default::default()
            },
            ChunkEvent {
                rid: "7".into(),
                text: "He\"llo\n\t".into(),
                token_ids: vec![1000],
                completion_tokens: 1,
                prompt_tokens: 128,
                ..Default::default()
            },
            ChunkEvent {
                rid: "7".into(),
                text: " 世界 🌍 \\".into(),
                token_ids: vec![-2, 3],
                completion_tokens: 2,
                prompt_tokens: 128,
                ..Default::default()
            },
            ChunkEvent {
                rid: "7".into(),
                text: "!".into(),
                token_ids: vec![9],
                completion_tokens: 1,
                prompt_tokens: 128,
                finish_reason: serde_json::from_value(
                    serde_json::json!({"type": "stop", "matched": 9}),
                )
                .expect("finish reason must parse"),
                ..Default::default()
            },
        ];

        for index in [None, Some(3usize)] {
            let mut acc = OutputAccumulator::default();
            for d in &deltas {
                acc.fold(d);
                let fast = cumulative_frame_json(&acc, "7", index).expect("no extras → fast path");
                let slow = typed_frame_string(frame_typed(acc.snapshot(), "7"), index);
                assert_eq!(fast, slow, "index={index:?} text={:?}", acc.snapshot().text);
            }
        }
    }

    /// The same equivalence, for the shape that made cumulative streaming O(T²):
    /// every logprob family at once, across several deltas, with and without
    /// `return_text_in_logprobs` texts and with a null ragged position.
    #[test]
    fn cumulative_frame_json_matches_typed_with_logprobs() {
        for with_texts in [false, true] {
            let txt = |v: &[&str]| -> Vec<String> {
                if with_texts {
                    v.iter().map(|s| (*s).to_string()).collect()
                } else {
                    Vec::new()
                }
            };
            let deltas = [
                // Prefill: the set-once input families and a null top-k position.
                ChunkEvent {
                    rid: "9".into(),
                    prompt_tokens: 4,
                    extras: Some(Box::new(ChunkExtras {
                        in_lp_val: vec![f32::NAN, -1.5],
                        in_lp_idx: vec![10, 11],
                        in_lp_txt: txt(&["a", "b"]),
                        in_top_val: vec![-0.25],
                        in_top_idx: vec![12],
                        in_top_lens: vec![0, 1],
                        in_top_txt: txt(&["c"]),
                        in_tid_val: vec![-2.0],
                        in_tid_idx: vec![13],
                        in_tid_lens: vec![1],
                        in_tid_txt: txt(&["d"]),
                        ..Default::default()
                    })),
                    ..Default::default()
                },
                ChunkEvent {
                    rid: "9".into(),
                    text: "He\"llo".into(),
                    token_ids: vec![100],
                    completion_tokens: 1,
                    prompt_tokens: 4,
                    extras: Some(Box::new(ChunkExtras {
                        out_lp_val: vec![-0.5],
                        out_lp_idx: vec![100],
                        out_lp_txt: txt(&["He\"llo"]),
                        out_top_val: vec![-0.5, -3.0],
                        out_top_idx: vec![100, 7],
                        out_top_lens: vec![2],
                        out_top_txt: txt(&["He\"llo", "x"]),
                        out_tid_val: vec![-0.5],
                        out_tid_idx: vec![100],
                        out_tid_lens: vec![1],
                        out_tid_txt: txt(&["He\"llo"]),
                        ..Default::default()
                    })),
                    ..Default::default()
                },
                ChunkEvent {
                    rid: "9".into(),
                    text: " 世界".into(),
                    token_ids: vec![-2, 3],
                    completion_tokens: 2,
                    prompt_tokens: 4,
                    finish_reason: serde_json::from_value(
                        serde_json::json!({"type": "stop", "matched": 3}),
                    )
                    .expect("finish reason must parse"),
                    extras: Some(Box::new(ChunkExtras {
                        out_lp_val: vec![f32::NAN, -0.125],
                        out_lp_idx: vec![-2, 3],
                        out_lp_txt: txt(&[" 世", "界"]),
                        // A zero-length position must render as `null`, not `[]`.
                        out_top_val: vec![-0.125],
                        out_top_idx: vec![3],
                        out_top_lens: vec![0, 1],
                        out_top_txt: txt(&["界"]),
                        out_tid_val: vec![],
                        out_tid_idx: vec![],
                        out_tid_lens: vec![0, 0],
                        out_tid_txt: txt(&[]),
                        hidden_val: vec![0.5, -0.25, 1.0],
                        hidden_lens: vec![2, 1],
                        ..Default::default()
                    })),
                },
            ];

            for index in [None, Some(2usize)] {
                let mut acc = OutputAccumulator::default();
                for d in &deltas {
                    acc.fold(d);
                    let fast = cumulative_frame_json(&acc, "9", index)
                        .expect("the extras memo must stay valid for a well-formed request");
                    let slow = typed_frame_string(frame_typed(acc.snapshot(), "9"), index);
                    assert_eq!(fast, slow, "with_texts={with_texts} index={index:?}");
                }
            }
        }
    }

    /// A delta sequence that supplies texts for some values and not others breaks
    /// the append-equivalence the memo rests on (`text_slot` is indexed globally,
    /// so a gap shifts every later text). The accumulator must notice and defer to
    /// the typed serialization rather than emit a frame that disagrees with it.
    #[test]
    fn mismatched_logprob_texts_fall_back_to_the_typed_path() {
        let mut acc = OutputAccumulator::default();
        acc.fold(&ChunkEvent {
            rid: "1".into(),
            extras: Some(Box::new(ChunkExtras {
                out_lp_val: vec![-0.5],
                out_lp_idx: vec![5],
                ..Default::default() // no texts
            })),
            ..Default::default()
        });
        assert!(cumulative_frame_json(&acc, "1", None).is_some());
        acc.fold(&ChunkEvent {
            rid: "1".into(),
            extras: Some(Box::new(ChunkExtras {
                out_lp_val: vec![-0.25],
                out_lp_idx: vec![6],
                out_lp_txt: vec!["b".into()], // …now texts: out of lockstep
                ..Default::default()
            })),
            ..Default::default()
        });
        assert!(
            cumulative_frame_json(&acc, "1", None).is_none(),
            "a text column out of lockstep must invalidate the memo"
        );
    }
}
