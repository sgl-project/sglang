//! Frame shaping for the native `/generate` protocol: the cumulative
//! [`OutputAccumulator`] plus the functions that render [`ChunkEvent`]s /
//! accumulated state into wire JSON (`meta_info`, logprob tuples, error and
//! abort frames). No HTTP here — the sibling `native_api` module owns the handlers
//! and streams; it calls these per frame.

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
            // Bounds-checked like `hidden_states_rows`: a header whose `lens` run
            // past the value buffer would otherwise panic the api thread on an
            // out-of-range index.
            let tuples: Vec<serde_json::Value> = (off..off + l)
                .filter_map(|j| {
                    Some(serde_json::json!([
                        lp_value(*vals.get(j)?),
                        *idxs.get(j)?,
                        text_slot(texts, j)
                    ]))
                })
                .collect();
            positions.push(serde_json::Value::Array(tuples));
        }
        off += l;
    }
    serde_json::Value::Array(positions)
}

/// Append a flat family's `[logprob, token_id, text]` tuples to `dst`, comma
/// separated and WITHOUT the enclosing brackets, so a cumulative frame can
/// concatenate each delta instead of re-rendering every accumulated position.
///
/// Byte-identical to [`logprob_tuples`]'s serialization: `serde_json` writes an
/// array as `[`, elements joined by `,`, `]` with no spaces, and each element here
/// is rendered by the same `Value` Display.
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
/// where `lens[p] == 0`. Mirrors [`ragged_logprob_tuples`] including its
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

/// Reshape flat hidden-state f32s + per-row lengths into `meta_info`'s nested
/// `list[list[float]]` (one row per output position).
fn hidden_states_rows(vals: &[f32], lens: &[u32]) -> serde_json::Value {
    let mut rows = Vec::with_capacity(lens.len());
    let mut off = 0usize;
    for &l in lens {
        let l = l as usize;
        // `get`, not a clamped index: clamping only the END leaves `off` past
        // `vals.len()` after one over-long row, making the next range reversed
        // (`start > end`) — which panics on the api thread rather than yielding
        // an empty row. Same reasoning as the decoder's `take_f32`.
        rows.push(serde_json::json!(vals.get(off..off + l).unwrap_or(&[])));
        off += l;
    }
    serde_json::Value::Array(rows)
}

pub(super) fn hidden_states_value(extras: &ChunkExtras) -> serde_json::Value {
    match &extras.hidden_shape {
        Some(shape) => shape
            .reshape(&extras.hidden_val)
            .unwrap_or(serde_json::Value::Null),
        None => hidden_states_rows(&extras.hidden_val, &extras.hidden_lens),
    }
}

/// Format a decoded [`ChunkEvent`] as one SGLang `/generate` frame's JSON. `rid`
/// (response `meta_info.id`) is passed as a string; the event's numeric `rid` is
/// just the shard routing key.
pub(super) fn frame_value(out: &ChunkEvent, rid: &str) -> serde_json::Value {
    frame_value_with_visible_tokens(out, rid, out.token_ids.len())
}

fn frame_value_with_visible_tokens(
    out: &ChunkEvent,
    rid: &str,
    visible_tokens: usize,
) -> serde_json::Value {
    let mut v = serde_json::json!({
        "text": out.text,
        "meta_info": core_metadata(out, rid, visible_tokens),
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
    for (key, values) in &ex.metadata.customized_info {
        v["meta_info"][key] = serde_json::json!(values);
    }
    if let Some(length) = sampling_mask_length(out) {
        v["meta_info"]["output_token_sampling_mask_length"] = serde_json::json!(length);
    }
    insert_prompt_metadata(&mut v, ex);
    // Python (`add_logprob_to_meta_info`) always sets input+output token
    // logprobs together, empty lists included. A PD decode node never receives
    // input logprobs (they belong to prefill), yet its response must still
    // carry the key — the PD router keys its merge of prefill's
    // `input_token_logprobs` on its presence.
    if ex.logprobs.is_some() || !ex.out_lp_val.is_empty() || !ex.in_lp_val.is_empty() {
        v["meta_info"]["output_token_logprobs"] =
            logprob_tuples(&ex.out_lp_val, &ex.out_lp_idx, opt_texts(&ex.out_lp_txt));
        v["meta_info"]["output_token_logprobs_length"] = serde_json::json!(ex.out_lp_val.len());
    }
    if ex.logprobs.is_some_and(|options| options.top_k > 0) || !ex.out_top_lens.is_empty() {
        v["meta_info"]["output_top_logprobs"] = ragged_logprob_tuples(
            &ex.out_top_val,
            &ex.out_top_idx,
            &ex.out_top_lens,
            opt_texts(&ex.out_top_txt),
        );
    }
    if ex.logprobs.is_some_and(|options| options.token_ids) || !ex.out_tid_lens.is_empty() {
        v["meta_info"]["output_token_ids_logprobs"] = ragged_logprob_tuples(
            &ex.out_tid_val,
            &ex.out_tid_idx,
            &ex.out_tid_lens,
            opt_texts(&ex.out_tid_txt),
        );
    }
    if ex.hidden_shape.is_some() || !ex.hidden_lens.is_empty() {
        v["meta_info"]["hidden_states"] = hidden_states_value(ex);
    }
    if let Some(beams) = &ex.beam_output {
        let results: Vec<_> = beams
            .sequences
            .iter()
            .enumerate()
            .map(|(index, beam)| {
                let mut metadata = if index == 0 {
                    let mut metadata = core_metadata(out, rid, visible_tokens);
                    for (key, values) in &beams.scheduler_metadata.customized_info {
                        metadata[key] = serde_json::json!(values);
                    }
                    for (key, value) in &beams.scheduler_metadata.fields {
                        metadata[key] = value.clone();
                    }
                    metadata
                } else {
                    serde_json::json!({})
                };
                metadata["finish_reason"] = serde_json::json!(beam.finish_reason);
                metadata["sequence_score"] = serde_json::json!(beam.sequence_score);
                let mut value =
                    serde_json::json!({"output_ids": beam.token_ids, "meta_info": metadata});
                if let Some(text) = &beam.text {
                    value["text"] = serde_json::json!(text);
                }
                value
            })
            .collect();
        if let Some(first) = results.first() {
            v["output_ids"] = first["output_ids"].clone();
            v["meta_info"]["sequence_score"] = first["meta_info"]["sequence_score"].clone();
            v["meta_info"]["beam_results"] = serde_json::Value::Array(results);
        }
    }
    v
}

fn core_metadata(out: &ChunkEvent, rid: &str, visible_tokens: usize) -> serde_json::Value {
    // Python projects beam weight-version spans onto the whole group's output
    // count, before replacing the leader's placeholder IDs with the best beam.
    let visible_tokens = if out
        .extras
        .as_ref()
        .is_some_and(|extras| extras.beam_output.is_some())
    {
        usize::try_from(out.completion_tokens).unwrap_or(usize::MAX)
    } else {
        visible_tokens
    };
    let mut metadata = serde_json::to_value(out.counts.metadata(visible_tokens))
        .expect("token counts are JSON values");
    metadata["id"] = serde_json::json!(rid);
    metadata["prompt_tokens"] = serde_json::json!(out.prompt_tokens);
    metadata["completion_tokens"] = serde_json::json!(out.completion_tokens);
    metadata["finish_reason"] = serde_json::json!(out.finish_reason);
    metadata
}

/// Prompt columns and request annotations are snapshots, including in the
/// incremental protocol; only output-token columns are sliced into deltas.
fn insert_prompt_metadata(value: &mut serde_json::Value, extras: &ChunkExtras) {
    if let Some(ids) = &extras.prompt_token_ids {
        value["prompt_token_ids"] = serde_json::json!(ids.as_ref());
    }
    for (key, field) in &extras.metadata.fields {
        value["meta_info"][key] = field.clone();
    }
    if extras.logprobs.is_some() || !extras.out_lp_val.is_empty() || !extras.in_lp_val.is_empty() {
        value["meta_info"]["input_token_logprobs"] = logprob_tuples(
            &extras.in_lp_val,
            &extras.in_lp_idx,
            opt_texts(&extras.in_lp_txt),
        );
    }
    if extras
        .logprobs
        .is_some_and(|options| options.top_k > 0 && !options.flat)
        || !extras.in_top_lens.is_empty()
    {
        value["meta_info"]["input_top_logprobs"] = ragged_logprob_tuples(
            &extras.in_top_val,
            &extras.in_top_idx,
            &extras.in_top_lens,
            opt_texts(&extras.in_top_txt),
        );
    }
    if extras.logprobs.is_some_and(|options| options.token_ids) || !extras.in_tid_lens.is_empty() {
        value["meta_info"]["input_token_ids_logprobs"] = ragged_logprob_tuples(
            &extras.in_tid_val,
            &extras.in_tid_idx,
            &extras.in_tid_lens,
            opt_texts(&extras.in_tid_txt),
        );
    }
}

/// Cumulative frame JSON from the accumulator's memoized parts — O(1) in the
/// accumulated length, where rebuilding the `Value` is O(T) per frame and so O(T²)
/// per request.
///
/// Encodes the same JSON document as `frame_value(..).to_string()` — same keys,
/// same values, same escaping — pinned for both the plain and the logprob shapes
/// by `cumulative_frame_json_matches_serde`.
pub(super) fn cumulative_frame_json(
    acc: &OutputAccumulator,
    rid: &str,
    index: Option<usize>,
) -> Option<String> {
    use std::fmt::Write;

    if acc.extras_memo_broken {
        return None;
    }
    let o = acc.snapshot();
    // Through `Value` rather than `to_string` on the struct: it is the same
    // encoder the slow path runs the finish reason through, so any representation
    // quirk is reproduced instead of re-derived.
    let finish = serde_json::to_value(&o.finish_reason).ok()?.to_string();

    // Scheduler counts are bounded metadata, independent of generated length.
    let mut m = serde_json::to_string(&o.counts.metadata(o.token_ids.len())).ok()?;
    m.pop(); // Append the remaining metadata inside the same JSON object.
    let _ = write!(m, ",\"completion_tokens\":{}", o.completion_tokens);
    let _ = write!(m, ",\"finish_reason\":{finish}");
    if let Some(h) = &acc.hidden_json {
        let _ = write!(m, ",\"hidden_states\":{h}");
    }
    let _ = write!(m, ",\"id\":{}", serde_json::Value::String(rid.to_string()));
    let logprobs = o.extras.as_deref().and_then(|extras| extras.logprobs);
    if logprobs.is_some_and(|options| options.token_ids) || acc.in_tid_json.is_some() {
        let v = acc.in_tid_json.as_deref().unwrap_or("[]");
        let _ = write!(m, ",\"input_token_ids_logprobs\":{v}");
    }
    // Input+output token logprobs are emitted as a PAIR whenever either side has
    // data (empty list included), matching `frame_value` byte for byte — see the
    // PD-router rationale there.
    let lp_pair = logprobs.is_some() || acc.in_lp_json.is_some() || !acc.out_lp_json.is_empty();
    if lp_pair {
        let v = acc.in_lp_json.as_deref().unwrap_or("[]");
        let _ = write!(m, ",\"input_token_logprobs\":{v}");
    }
    if logprobs.is_some_and(|options| options.top_k > 0 && !options.flat)
        || acc.in_top_json.is_some()
    {
        let v = acc.in_top_json.as_deref().unwrap_or("[]");
        let _ = write!(m, ",\"input_top_logprobs\":{v}");
    }
    // Requested families remain present before they have produced any values.
    if logprobs.is_some_and(|options| options.token_ids) || !acc.out_tid_json.is_empty() {
        let _ = write!(m, ",\"output_token_ids_logprobs\":[{}]", acc.out_tid_json);
    }
    if lp_pair {
        let _ = write!(m, ",\"output_token_logprobs\":[{}]", acc.out_lp_json);
        let _ = write!(
            m,
            ",\"output_token_logprobs_length\":{}",
            o.extras.as_ref()?.out_lp_val.len()
        );
    }
    if logprobs.is_some_and(|options| options.top_k > 0) || !acc.out_top_json.is_empty() {
        let _ = write!(m, ",\"output_top_logprobs\":[{}]", acc.out_top_json);
    }
    for (key, values) in &acc.customized_json {
        let _ = write!(m, ",{}:[{values}]", serde_json::json!(key));
    }
    if let Some(length) = sampling_mask_length(o) {
        let _ = write!(m, ",\"output_token_sampling_mask_length\":{length}");
    }
    if let Some(extras) = &o.extras {
        for (key, value) in &extras.metadata.fields {
            let _ = write!(m, ",{}:{value}", serde_json::json!(key));
        }
    }
    let _ = write!(m, ",\"prompt_tokens\":{}}}", o.prompt_tokens);

    let mut s = String::with_capacity(acc.text_json.len() + acc.ids_json.len() + m.len() + 40);
    s.push('{');
    if let Some(i) = index {
        let _ = write!(s, "\"index\":{i},");
    }
    s.push_str("\"meta_info\":");
    s.push_str(&m);
    if !acc.ids_json.is_empty() {
        s.push_str(",\"output_ids\":[");
        s.push_str(&acc.ids_json);
        s.push(']');
    }
    if let Some(ids) = &acc.prompt_ids_json {
        let _ = write!(s, ",\"prompt_token_ids\":{ids}");
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
        .unwrap_or_else(|| tag_value(frame_value(acc.snapshot(), rid_str), index))
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
        let mut value =
            frame_value_with_visible_tokens(&d, rid_str, acc.snapshot().token_ids.len());
        if let Some(length) = sampling_mask_length(acc.snapshot()) {
            value["meta_info"]["output_token_sampling_mask_length"] = serde_json::json!(length);
        }
        if let Some(extras) = acc.snapshot().extras.as_deref() {
            insert_prompt_metadata(&mut value, extras);
            if extras.logprobs.is_some()
                || !extras.out_lp_val.is_empty()
                || !extras.in_lp_val.is_empty()
            {
                value["meta_info"]["output_token_logprobs_length"] =
                    serde_json::json!(extras.out_lp_val.len());
            }
        }
        value
    } else {
        frame_value(acc.snapshot(), rid_str)
    }
}

fn sampling_mask_length(output: &ChunkEvent) -> Option<usize> {
    output
        .extras
        .as_ref()?
        .metadata
        .customized_info
        .get("output_token_sampling_mask")
        .map(Vec::len)
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
    prompt_ids_json: Option<String>,
    hidden_json: Option<String>,
    customized_json: std::collections::BTreeMap<String, String>,
    /// Set once a family's text column falls out of lockstep with its values, at
    /// which point the memo is abandoned for the `Value` path.
    ///
    /// Appending per delta assumes `text_slot(accumulated, global_j)` equals
    /// `text_slot(delta, local_j)`, which holds only while every delta supplies
    /// either a text per value or none at all. That is what a real request does —
    /// `return_text_in_logprobs` is per-request, so the detok shard fills `*_txt`
    /// for all deltas or none — but a mixed sequence would silently diverge from
    /// `frame_value`, so it is detected rather than assumed.
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
        o.rid.clone_from(&d.rid); // constant across the request; keeps the accumulated view coherent
        o.text.push_str(&d.text);
        o.token_ids.extend_from_slice(&d.token_ids); // token_ids doubles as output_ids
        o.completion_tokens += d.completion_tokens;
        o.prompt_tokens = d.prompt_tokens; // constant across the request
        o.counts.clone_from(&d.counts);
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
        if let Some(ids) = &de.prompt_token_ids {
            if oe
                .prompt_token_ids
                .as_ref()
                .is_none_or(|previous| !std::sync::Arc::ptr_eq(previous, ids))
            {
                self.prompt_ids_json = Some(serde_json::json!(ids.as_ref()).to_string());
            }
            oe.prompt_token_ids = Some(ids.clone());
        }
        if let Some(beams) = &de.beam_output {
            oe.beam_output = Some(beams.clone());
            // Beams arrive only at completion; use the full candidate renderer.
            self.extras_memo_broken = true;
        }
        for (key, values) in &de.metadata.customized_info {
            oe.metadata
                .customized_info
                .entry(key.clone())
                .or_default()
                .extend_from_slice(values);
            let memo = self.customized_json.entry(key.clone()).or_default();
            for value in values {
                if !memo.is_empty() {
                    memo.push(',');
                }
                let _ = write!(memo, "{value}");
            }
        }
        oe.metadata.fields.extend(de.metadata.fields.clone());
        oe.logprobs = de.logprobs.or(oe.logprobs);
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
        // Hidden states are scheduler snapshots, including an explicit empty list.
        if de.hidden_shape.is_some() || !de.hidden_lens.is_empty() {
            oe.hidden_val = de.hidden_val.clone();
            oe.hidden_lens = de.hidden_lens.clone();
            oe.hidden_shape = de.hidden_shape.clone();
            self.hidden_json = Some(hidden_states_value(oe).to_string());
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
    fn output_switches_match_python_llm_wire_and_response_contract() {
        use crate::message::request::GenerateBody;

        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../testdata/output_flags_python.json")).unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let (requests, _) = serde_json::from_value::<GenerateBody>(case["body"].clone())
                .unwrap()
                .into_requests()
                .unwrap();
            assert_eq!(requests.len(), case["expected"].as_array().unwrap().len());
            for (request, expected) in requests
                .into_iter()
                .zip(case["expected"].as_array().unwrap())
            {
                let body = serde_json::to_value(&request).unwrap();
                for field in ["return_bytes", "return_entropy"] {
                    assert_eq!(body[field], case["body"][field]);
                }
                let (forwarded, _) = serde_json::from_value::<GenerateBody>(body)
                    .unwrap()
                    .into_requests()
                    .unwrap();
                for request in [request, forwarded.into_iter().next().unwrap()] {
                    let header: serde_json::Value =
                        rmp_serde::from_slice(&request.encode_header().unwrap()).unwrap();
                    for index in fixture["wire_indices"].as_object().unwrap().values() {
                        assert_eq!(header[index.as_u64().unwrap() as usize], false);
                    }
                }
                let output = ChunkEvent {
                    text: "hello".into(),
                    token_ids: vec![5, 6],
                    prompt_tokens: 3,
                    completion_tokens: 2,
                    finish_reason: serde_json::from_value(
                        serde_json::json!({"type": "length", "length": 2}),
                    )
                    .unwrap(),
                    ..Default::default()
                };
                let mut accumulator = OutputAccumulator::default();
                accumulator.fold(&output);
                for mut value in [
                    frame_value(&output, "flags"),
                    stream_frame_value(output.clone(), &accumulator, true, "flags"),
                    serde_json::from_str(
                        &cumulative_frame_json(&accumulator, "flags", None).unwrap(),
                    )
                    .unwrap(),
                ] {
                    value["meta_info"].as_object_mut().unwrap().remove("id");
                    assert_eq!(&value, expected);
                }
            }
        }
        for body in fixture["rejected"].as_array().unwrap() {
            assert!(serde_json::from_value::<GenerateBody>(body.clone()).is_err());
        }
    }

    #[test]
    fn delimiter_scoring_matches_python_request_wire_and_sparse_logprobs() {
        use crate::message::request::GenerateBody;
        use crate::message::response::{LogprobOptions, for_each_chunk, frame_decode_batch_cols};
        use base64::Engine;

        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../testdata/multi_item_scoring_python.json"
        ))
        .unwrap();
        let field = fixture["field_index"].as_u64().unwrap() as usize;
        for case in fixture["requests"].as_array().unwrap() {
            let body: GenerateBody = serde_json::from_value(case["body"].clone()).unwrap();
            let (requests, _) = body.into_requests().unwrap();
            assert_eq!(requests.len(), case["expected"].as_array().unwrap().len());
            for (index, request) in requests.into_iter().enumerate() {
                let forwarded: GenerateBody =
                    serde_json::from_value(serde_json::to_value(&request).unwrap()).unwrap();
                let (forwarded, _) = forwarded.into_requests().unwrap();
                for request in [request, forwarded.into_iter().next().unwrap()] {
                    let header: serde_json::Value =
                        rmp_serde::from_slice(&request.encode_header().unwrap()).unwrap();
                    assert_eq!(header[field], case["expected"][index]);
                }
            }
        }
        let data = base64::engine::general_purpose::STANDARD
            .decode(fixture["data_b64"].as_str().unwrap())
            .unwrap();
        let frame =
            frame_decode_batch_cols(&rmp_serde::to_vec(&fixture["header"]).unwrap(), &[&data]);
        let mut outputs = Vec::new();
        assert!(for_each_chunk(&frame[1..], |event| outputs.push(event)).ok);
        assert_eq!(outputs.len(), 1);
        let mut output = outputs.pop().unwrap();
        output.extras.as_mut().unwrap().logprobs = Some(LogprobOptions {
            top_k: 2,
            token_ids: true,
            flat: false,
            base64: false,
        });
        assert_eq!(output.prompt_tokens, 6);
        assert_eq!(output.completion_tokens, 0);
        let mut accumulator = OutputAccumulator::default();
        accumulator.fold(&output);
        for value in [
            frame_value(&output, "delimiter-scores"),
            stream_frame_value(output.clone(), &accumulator, true, "delimiter-scores"),
            serde_json::from_str(
                &cumulative_frame_json(&accumulator, "delimiter-scores", None).unwrap(),
            )
            .unwrap(),
        ] {
            for (key, expected) in fixture["expected"].as_object().unwrap() {
                assert_eq!(&value["meta_info"][key], expected, "{key}");
            }
        }
    }

    #[test]
    fn beam_wire_decode_and_frames_match_python_candidates() {
        use crate::message::detok::DetokMsg;
        use crate::message::response::{
            OutputMetadata, ResponseItem, ResponseSink, for_each_chunk, frame_decode_batch_cols,
        };
        use crate::tokenizer_manager::detokenizer::{DetokenizerBackend, DetokenizerWorker};
        use crate::utils::runtime::Runnable;
        use base64::Engine;
        use std::sync::Arc;

        #[derive(Debug)]
        struct FrontendAnnotation;
        impl crate::ResponseProcessor for FrontendAnnotation {
            fn process(
                &mut self,
                metadata: &mut OutputMetadata,
                _finished: bool,
            ) -> Result<(), String> {
                metadata
                    .fields
                    .insert("frontend_annotation".into(), true.into());
                Ok(())
            }
        }
        impl crate::HttpExtension for FrontendAnnotation {
            fn apply(&self, router: axum::Router) -> axum::Router {
                router
            }
            fn new_response_processor(&self) -> Option<Box<dyn crate::ResponseProcessor>> {
                Some(Box::new(Self))
            }
        }
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../testdata/beam_outputs_python.json")).unwrap();
        let decoders: serde_json::Value =
            serde_json::from_str(include_str!("../../testdata/decoder_python.json")).unwrap();
        let path = std::env::temp_dir().join(format!("sglang-beam-{}.json", uuid::Uuid::new_v4()));
        std::fs::write(
            &path,
            serde_json::to_vec(
                &decoders[fixture["tokenizer_fixture"].as_u64().unwrap() as usize]["tokenizer"],
            )
            .unwrap(),
        )
        .unwrap();
        let tokenizer = dynamo_tokenizers::Tokenizer::from_file(path.to_str().unwrap()).unwrap();
        std::fs::remove_file(path).unwrap();
        let data = base64::engine::general_purpose::STANDARD
            .decode(fixture["data_b64"].as_str().unwrap())
            .unwrap();
        let frame =
            frame_decode_batch_cols(&rmp_serde::to_vec(&fixture["header"]).unwrap(), &[&data]);
        for case in fixture["cases"].as_array().unwrap() {
            let backend = if case["skip_tokenizer"].as_bool().unwrap() {
                DetokenizerBackend::Skip
            } else {
                DetokenizerBackend::Dynamo {
                    tokenizer: tokenizer.clone(),
                    vocab_size: fixture["vocab_size"].as_u64(),
                }
            };
            let (send, receive) = flume::unbounded();
            let (lifecycle, _) = flume::unbounded();
            let worker = DetokenizerWorker::new(0, receive, backend, lifecycle)
                .with_http_extension(Some(Arc::new(FrontendAnnotation)));
            let thread = std::thread::spawn(move || worker.run());
            let mut receivers = Vec::new();
            for rid in fixture["header"][0].as_array().unwrap() {
                let (sink, receive) = tokio::sync::mpsc::channel(1);
                send.send(DetokMsg::Register {
                    rid: rid.as_str().unwrap().into(),
                    sink: ResponseSink::Local(sink),
                    decode_logprob_text: false,
                    skip_special_tokens: case["skip_special_tokens"].as_bool().unwrap(),
                    no_stop_trim: case["no_stop_trim"].as_bool().unwrap(),
                    stop_texts: Vec::new(),
                    control_replies: 0,
                    metrics: None,
                    logprobs: None,
                })
                .unwrap();
                receivers.push(receive);
            }
            let mut outputs = Vec::new();
            assert!(for_each_chunk(&frame[1..], |event| outputs.push(event)).ok);
            send.send(DetokMsg::Chunks(outputs)).unwrap();
            drop(send);
            thread.join().unwrap();
            for (index, mut receive) in receivers.into_iter().enumerate() {
                let ResponseItem::Done(output) = receive.try_recv().unwrap() else {
                    panic!("beam output must finish")
                };
                let rid = fixture["header"][0][index].as_str().unwrap();
                let mut accumulator = OutputAccumulator::default();
                accumulator.fold(&output);
                let mut expected = case["expected"][index].clone();
                if expected.is_null() {
                    assert_eq!(output.token_ids, [3]);
                    assert_eq!(output.completion_tokens, 1);
                    assert!(
                        frame_value(&output, rid)["meta_info"]
                            .get("beam_results")
                            .is_none()
                    );
                    continue;
                }
                expected["meta_info"]["frontend_annotation"] = true.into();
                assert_eq!(frame_value(&output, rid), expected, "{case}");
                for incremental in [false, true] {
                    let frame =
                        stream_frame_string(output.clone(), &accumulator, incremental, rid, None);
                    assert_eq!(
                        serde_json::from_str::<serde_json::Value>(&frame).unwrap(),
                        expected,
                        "incremental={incremental} {case}"
                    );
                }
            }
        }
    }

    #[test]
    fn incremental_logprobs_retain_prompt_snapshots_and_cumulative_lengths() {
        let fixtures: serde_json::Value =
            serde_json::from_str(include_str!("../../testdata/flat_logprobs_python.json")).unwrap();
        let mut empty = OutputAccumulator::default();
        for _ in 0..2 {
            let event = ChunkEvent {
                extras: Some(Box::new(ChunkExtras {
                    logprobs: Some(crate::message::response::LogprobOptions {
                        top_k: 2,
                        token_ids: true,
                        flat: false,
                        base64: false,
                    }),
                    ..Default::default()
                })),
                ..Default::default()
            };
            empty.fold(&event);
            for frame in [
                stream_frame_value(event, &empty, true, "empty"),
                serde_json::from_str(&cumulative_frame_json(&empty, "empty", None).unwrap())
                    .unwrap(),
            ] {
                for (key, expected) in fixtures["empty_logprobs"].as_object().unwrap() {
                    assert_eq!(&frame["meta_info"][key], expected);
                }
            }
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(
                    &cumulative_frame_json(&empty, "empty", None).unwrap()
                )
                .unwrap(),
                frame_value(empty.snapshot(), "empty")
            );
        }
        for case in fixtures["cases"].as_array().unwrap() {
            for format in ["json", "base64"] {
                let mut accumulator = OutputAccumulator::default();
                let mut first = ChunkEvent {
                    token_ids: vec![11],
                    completion_tokens: 1,
                    extras: Some(Box::new(ChunkExtras {
                        out_lp_val: vec![-0.25],
                        out_lp_idx: vec![11],
                        in_lp_val: vec![f32::NAN, -0.5],
                        in_lp_idx: vec![1, 2],
                        ..Default::default()
                    })),
                    ..Default::default()
                };
                first.extras.as_mut().unwrap().metadata.fields =
                    case[format].as_object().unwrap().clone();
                let second = ChunkEvent {
                    token_ids: vec![12],
                    completion_tokens: 1,
                    extras: Some(Box::new(ChunkExtras {
                        out_lp_val: vec![-0.125],
                        out_lp_idx: vec![12],
                        ..Default::default()
                    })),
                    ..Default::default()
                };
                for (i, event) in [first, second].into_iter().enumerate() {
                    accumulator.fold(&event);
                    let incremental = stream_frame_value(event, &accumulator, true, "flat");
                    let cumulative = frame_value(accumulator.snapshot(), "flat");
                    assert_eq!(
                        serde_json::from_str::<serde_json::Value>(
                            &cumulative_frame_json(&accumulator, "flat", None).unwrap()
                        )
                        .unwrap(),
                        cumulative
                    );
                    for output in [&incremental, &cumulative] {
                        let meta = &output["meta_info"];
                        for (key, value) in case[format].as_object().unwrap() {
                            assert_eq!(meta[key], *value);
                        }
                        assert_eq!(
                            meta["input_token_logprobs"],
                            serde_json::json!([[null, 1, null], [-0.5, 2, null]])
                        );
                        assert_eq!(meta["output_token_logprobs_length"], i + 1);
                    }
                    assert_eq!(
                        incremental["meta_info"]["output_token_logprobs"]
                            .as_array()
                            .unwrap()
                            .len(),
                        1
                    );
                }
            }
        }
    }

    #[test]
    fn sampling_masks_match_python_columns_and_streaming_accumulation() {
        use crate::message::response::{BatchHeader, for_each_chunk, frame_decode_batch_cols};

        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../testdata/sampling_masks_python.json"))
                .unwrap();
        let mut accumulators: Vec<_> = (0..3).map(|_| OutputAccumulator::default()).collect();
        let mut expected_masks = vec![Vec::<serde_json::Value>::new(); 3];
        let mut expected_logprobs = expected_masks.clone();
        for step in fixture["steps"].as_array().unwrap() {
            let tokens: Vec<Vec<i32>> = serde_json::from_value(step["tokens"].clone()).unwrap();
            let mut header = BatchHeader {
                rids: (0..3).map(|i| i.to_string()).collect(),
                finish_reasons: vec![None; 3],
                prompt_tokens: vec![2; 3],
                tok_lens: tokens.iter().map(|ids| ids.len() as u32).collect(),
                sampling_mask_shapes: serde_json::from_value(step["shapes"].clone()).unwrap(),
                ..Default::default()
            };
            let mut data: Vec<u8> = tokens
                .into_iter()
                .flatten()
                .flat_map(i32::to_le_bytes)
                .collect();
            let mask_ids: Vec<i32> = serde_json::from_value(step["ids"].clone()).unwrap();
            data.extend(mask_ids.into_iter().flat_map(i32::to_le_bytes));
            let values: Vec<Option<f32>> = serde_json::from_value(step["values"].clone()).unwrap();
            data.extend(
                values
                    .into_iter()
                    .flat_map(|v| v.unwrap_or(f32::NAN).to_le_bytes()),
            );
            let frame = frame_decode_batch_cols(&rmp_serde::to_vec(&header).unwrap(), &[&data]);
            let mut events = Vec::new();
            assert!(for_each_chunk(&frame[1..], |event| events.push(event)).ok);
            for (i, mut event) in events.into_iter().enumerate() {
                event.completion_tokens = event.token_ids.len() as u64;
                let acc = &mut accumulators[i];
                acc.fold(&event);
                let incremental = stream_frame_value(event, acc, true, "mask");
                let cumulative: serde_json::Value =
                    serde_json::from_str(&cumulative_frame_json(acc, "mask", None).unwrap())
                        .unwrap();
                assert_eq!(cumulative, frame_value(acc.snapshot(), "mask"));
                if step["masks"][i].is_null() {
                    assert!(
                        cumulative["meta_info"]
                            .get("output_token_sampling_mask")
                            .is_none()
                    );
                    continue;
                }
                expected_masks[i].extend(step["masks"][i].as_array().unwrap().clone());
                expected_logprobs[i].extend(step["logprobs"][i].as_array().unwrap().clone());
                assert_eq!(
                    incremental["meta_info"]["output_token_sampling_mask"],
                    step["masks"][i]
                );
                assert_eq!(
                    incremental["meta_info"]["output_token_sampling_logprobs"],
                    step["logprobs"][i]
                );
                assert_eq!(
                    cumulative["meta_info"]["output_token_sampling_mask"],
                    serde_json::json!(expected_masks[i])
                );
                assert_eq!(
                    cumulative["meta_info"]["output_token_sampling_logprobs"],
                    serde_json::json!(expected_logprobs[i])
                );
                for value in [&incremental, &cumulative] {
                    assert_eq!(
                        value["meta_info"]["output_token_sampling_mask_length"],
                        expected_masks[i].len()
                    );
                }
            }
            header.sampling_mask_shapes.pop();
            let frame = frame_decode_batch_cols(&rmp_serde::to_vec(&header).unwrap(), &[&data]);
            assert!(
                !for_each_chunk(&frame[1..], |_| panic!(
                    "partial sampling shape column was routed"
                ))
                .ok
            );
        }
    }

    #[test]
    fn hidden_state_shapes_match_python_and_cumulative_frames() {
        use crate::message::response::{
            BatchHeader, HiddenStateShape, for_each_chunk, frame_decode_batch_cols,
        };

        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../testdata/hidden_states_python.json")).unwrap();
        let cases = fixture["outputs"].as_array().unwrap();
        let lengths: Vec<Vec<u32>> = cases
            .iter()
            .map(|case| serde_json::from_value(case["lengths"].clone()).unwrap())
            .collect();
        let mut header = BatchHeader {
            rids: (0..cases.len()).map(|i| i.to_string()).collect(),
            finish_reasons: vec![None; cases.len()],
            prompt_tokens: vec![2; cases.len()],
            tok_lens: vec![1; cases.len()],
            hidden_reqlens: lengths.iter().map(|lens| lens.len() as u32).collect(),
            hidden_poslens: lengths.into_iter().flatten().collect(),
            hidden_shapes: cases
                .iter()
                .map(|case| serde_json::from_value(case["shape"].clone()).unwrap())
                .collect(),
            ..Default::default()
        };
        let mut data: Vec<u8> = vec![3i32; cases.len()]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect();
        for case in cases {
            let values: Vec<f32> = serde_json::from_value(case["values"].clone()).unwrap();
            data.extend(values.into_iter().flat_map(f32::to_le_bytes));
        }
        let frame = frame_decode_batch_cols(&rmp_serde::to_vec(&header).unwrap(), &[&data]);
        let mut events = Vec::new();
        assert!(for_each_chunk(&frame[1..], |event| events.push(event)).ok);
        for (event, expected) in events.iter().zip(cases) {
            let value = frame_value(event, "hidden");
            assert_eq!(value["meta_info"]["hidden_states"], expected["output"]);
            assert_eq!(
                value["meta_info"].get("hidden_states").is_some(),
                !expected["output"].is_null()
            );
            let mut acc = OutputAccumulator::default();
            acc.fold(event);
            acc.fold(&ChunkEvent::default());
            let value: serde_json::Value =
                serde_json::from_str(&cumulative_frame_json(&acc, "hidden", None).unwrap())
                    .unwrap();
            assert_eq!(value["meta_info"]["hidden_states"], expected["output"]);
            assert_eq!(value, frame_value(acc.snapshot(), "hidden"));
        }
        header.hidden_shapes[0] = Some(HiddenStateShape::Vector(3));
        let frame = frame_decode_batch_cols(&rmp_serde::to_vec(&header).unwrap(), &[&data]);
        assert!(!for_each_chunk(&frame[1..], |_| panic!("invalid hidden shape was routed")).ok);
        header.hidden_shapes.pop();
        let frame = frame_decode_batch_cols(&rmp_serde::to_vec(&header).unwrap(), &[&data]);
        assert!(!for_each_chunk(&frame[1..], |_| panic!("partial shape column was routed")).ok);
    }

    #[test]
    fn custom_metadata_agrees_in_unary_cumulative_and_incremental_frames() {
        use serde_json::json;
        let mut accumulator = OutputAccumulator::default();
        for (i, values) in [vec![serde_json::Value::Null, json!(2)], vec![json!(4)]]
            .into_iter()
            .enumerate()
        {
            let delta = ChunkEvent {
                token_ids: vec![11; values.len()],
                completion_tokens: values.len() as u64,
                extras: Some(Box::new(ChunkExtras {
                    metadata: crate::OutputMetadata {
                        customized_info: [("aux".into(), values.clone())].into(),
                        fields: serde_json::Map::from_iter([(
                            "model_result".into(),
                            json!({"score": i + 1}),
                        )]),
                    },
                    ..Default::default()
                })),
                ..Default::default()
            };
            accumulator.fold(&delta);
            let incremental = stream_frame_value(delta, &accumulator, true, "request");
            assert_eq!(incremental["meta_info"]["aux"], json!(values));
            assert_eq!(incremental["meta_info"]["model_result"]["score"], i + 1);
            let fast: serde_json::Value = serde_json::from_str(
                &cumulative_frame_json(&accumulator, "request", None).unwrap(),
            )
            .unwrap();
            assert_eq!(fast, frame_value(accumulator.snapshot(), "request"));
        }
        let unary = frame_value(&accumulator.into_output(), "request");
        assert_eq!(unary["meta_info"]["aux"], json!([null, 2, 4]));
        assert_eq!(unary["meta_info"]["model_result"]["score"], 2);
    }

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
        let frame = frame_value(&out, "1");
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

    /// Parse a frame so the fast and slow paths can be compared as documents.
    fn as_json(frame: &str) -> serde_json::Value {
        serde_json::from_str(frame).expect("a frame must be valid JSON")
    }

    /// The memoized cumulative fast path must emit the **same JSON document** as the
    /// `serde_json::Value` builder it replaces — same keys, same values, same
    /// escaping. Covers unicode and control chars, an empty-ids first frame, a
    /// finish_reason, and the batch `index`. Guards the O(T) rewrite of the O(T²)
    /// `output_ids` serialization.
    #[test]
    fn cumulative_frame_json_matches_serde() {
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
            for (step, mut d) in deltas.iter().cloned().enumerate() {
                d.counts = std::sync::Arc::new(crate::message::response::TokenCounts {
                    reasoning_tokens: step as u64,
                    cached_tokens: 96,
                    cached_tokens_details: Some(Some(
                        crate::message::response::CachedTokensDetails {
                            device: 64,
                            host: 16,
                            storage: Some(16),
                            storage_backend: Some("FileSystem".into()),
                        },
                    )),
                    dp_rank: Some(Some(2)),
                    image_tokens: Some(8),
                    weight_version: Some("current".into()),
                    weight_versions: Some(vec![
                        crate::message::response::WeightVersionSpan {
                            version: "old".into(),
                            start: 0,
                            end: 2,
                        },
                        crate::message::response::WeightVersionSpan {
                            version: "current".into(),
                            start: 2,
                            end: 100,
                        },
                    ]),
                    ..Default::default()
                });
                acc.fold(&d);
                let fast = cumulative_frame_json(&acc, "7", index).expect("no extras → fast path");
                let slow = tag_value(frame_value(acc.snapshot(), "7"), index);
                println!("fast={fast:?}");
                println!("slow={slow:?}");
                assert_eq!(
                    as_json(&fast),
                    as_json(&slow),
                    "index={index:?} text={:?}",
                    acc.snapshot().text
                );
                let value = as_json(&fast);
                assert_eq!(value["meta_info"]["cached_tokens"], 96);
                assert_eq!(value["meta_info"]["reasoning_tokens"], step);
                assert_eq!(value["meta_info"]["cached_tokens_details"]["storage"], 16);
                assert_eq!(value["meta_info"]["dp_rank"], 2);
                assert_eq!(value["meta_info"]["image_tokens"], 8);
                let visible = acc.snapshot().token_ids.len();
                assert_eq!(
                    value["meta_info"]["weight_version"],
                    if visible <= 2 { "old" } else { "current" }
                );
                let spans = value["meta_info"]["weight_versions"].as_array().unwrap();
                assert_eq!(spans.last().unwrap()["end"], visible);
                let incremental = stream_frame_value(d, &acc, true, "7");
                assert_eq!(incremental["meta_info"], value["meta_info"]);
            }
        }
    }

    /// The same equivalence, for the shape that made cumulative streaming O(T²):
    /// every logprob family at once, across several deltas, with and without
    /// `return_text_in_logprobs` texts and with a null ragged position.
    #[test]
    fn cumulative_frame_json_matches_serde_with_logprobs() {
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
                    counts: std::sync::Arc::new(crate::message::response::TokenCounts {
                        reasoning_tokens: 1,
                        cached_tokens: 2,
                        ..Default::default()
                    }),
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
                    let slow = tag_value(frame_value(acc.snapshot(), "9"), index);
                    assert_eq!(
                        as_json(&fast),
                        as_json(&slow),
                        "with_texts={with_texts} index={index:?}"
                    );
                }
            }
        }
    }

    /// A delta sequence that supplies texts for some values and not others breaks
    /// the append-equivalence the memo rests on (`text_slot` is indexed globally,
    /// so a gap shifts every later text). The accumulator must notice and defer to
    /// the `Value` builder rather than emit a frame that disagrees with it.
    #[test]
    fn mismatched_logprob_texts_fall_back_to_the_value_path() {
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

    #[test]
    fn prompt_ids_are_top_level_snapshots_in_every_response_mode() {
        use std::sync::Arc;
        for ids in [None, Some(Arc::from([])), Some(Arc::from([1, 2000, 3]))] {
            let mut accumulator = OutputAccumulator::default();
            for token in [4, 5] {
                let event = ChunkEvent {
                    token_ids: vec![token],
                    completion_tokens: 1,
                    extras: Some(Box::new(ChunkExtras {
                        prompt_token_ids: ids.clone(),
                        ..Default::default()
                    })),
                    ..Default::default()
                };
                accumulator.fold(&event);
                let cumulative: serde_json::Value = serde_json::from_str(
                    &cumulative_frame_json(&accumulator, "request", None).unwrap(),
                )
                .unwrap();
                for value in [
                    frame_value(accumulator.snapshot(), "request"),
                    stream_frame_value(event, &accumulator, true, "request"),
                    cumulative,
                ] {
                    assert!(value["meta_info"].get("prompt_token_ids").is_none());
                    assert_eq!(
                        value.get("prompt_token_ids"),
                        ids.as_ref()
                            .map(|ids| serde_json::json!(ids.as_ref()))
                            .as_ref()
                    );
                }
            }
        }
    }
}
