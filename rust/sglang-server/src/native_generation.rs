//! Shared wire semantics for SGLang-native generation responses.
//!
//! Native HTTP places these values in a JSON `meta_info` object, while
//! `runtime.v1` gRPC JSON-encodes each value into `map<string, string>`. Keeping
//! the logical shape here prevents the sibling adapters from drifting without
//! making either adapter depend on the other.

use crate::frontend::FrontendOutput;

/// The text slot of a `[logprob, token_id, text]` tuple: the decoded token when
/// `return_text_in_logprobs` supplied a text buffer, else `null`.
pub(crate) fn text_slot(texts: Option<&[String]>, j: usize) -> serde_json::Value {
    texts
        .and_then(|texts| texts.get(j))
        .map(|text| serde_json::json!(text))
        .unwrap_or(serde_json::Value::Null)
}

/// A decoded-text column becomes the tuples' text source only when populated.
pub(crate) fn opt_texts(texts: &[String]) -> Option<&[String]> {
    (!texts.is_empty()).then_some(texts)
}

/// The logprob slot of a tuple: a finite value, or `null` for the `NaN` sentinel.
pub(crate) fn lp_value(value: f32) -> serde_json::Value {
    if value.is_nan() {
        serde_json::Value::Null
    } else {
        serde_json::json!(value)
    }
}

/// SGLang logprob shape: a list of `[logprob, token_id, text]` tuples.
pub(crate) fn logprob_tuples(
    values: &[f32],
    token_ids: &[i32],
    texts: Option<&[String]>,
) -> serde_json::Value {
    let tuples = values
        .iter()
        .zip(token_ids.iter())
        .enumerate()
        .map(|(index, (&value, &token_id))| {
            serde_json::json!([lp_value(value), token_id, text_slot(texts, index)])
        })
        .collect();
    serde_json::Value::Array(tuples)
}

/// Ragged top-k / requested-token shape: one entry per position, or `null`
/// where that position's length is zero.
pub(crate) fn ragged_logprob_tuples(
    values: &[f32],
    token_ids: &[i32],
    lengths: &[u32],
    texts: Option<&[String]>,
) -> serde_json::Value {
    let mut positions = Vec::with_capacity(lengths.len());
    let mut offset = 0usize;
    for &length in lengths {
        let length = length as usize;
        if length == 0 {
            positions.push(serde_json::Value::Null);
        } else {
            // Bounds checks turn malformed parallel columns into a shortened
            // row instead of panicking a transport worker.
            let tuples = (offset..offset + length)
                .filter_map(|index| {
                    Some(serde_json::json!([
                        lp_value(*values.get(index)?),
                        *token_ids.get(index)?,
                        text_slot(texts, index)
                    ]))
                })
                .collect();
            positions.push(serde_json::Value::Array(tuples));
        }
        offset += length;
    }
    serde_json::Value::Array(positions)
}

/// Reshape flat hidden-state values and per-row lengths into nested rows.
pub(crate) fn hidden_states_rows(values: &[f32], lengths: &[u32]) -> serde_json::Value {
    let mut rows = Vec::with_capacity(lengths.len());
    let mut offset = 0usize;
    for &length in lengths {
        let length = length as usize;
        rows.push(serde_json::json!(
            values.get(offset..offset + length).unwrap_or(&[])
        ));
        offset += length;
    }
    serde_json::Value::Array(rows)
}

/// Build the heterogeneous native `meta_info` object before either adapter
/// chooses its wire encoding.
pub(crate) fn meta_info_value(out: &FrontendOutput, rid: &str) -> serde_json::Value {
    let mut meta = serde_json::json!({
        "id": rid,
        "prompt_tokens": out.prompt_tokens,
        "completion_tokens": out.completion_tokens,
        "finish_reason": out.finish_reason,
    });
    let Some(extras) = out.extras.as_deref() else {
        return meta;
    };

    // Python always sets input and output token logprobs together, including
    // the empty input list emitted by a disaggregated decode node.
    if !extras.out_lp_val.is_empty() || !extras.in_lp_val.is_empty() {
        meta["output_token_logprobs"] = logprob_tuples(
            &extras.out_lp_val,
            &extras.out_lp_idx,
            opt_texts(&extras.out_lp_txt),
        );
        meta["input_token_logprobs"] = logprob_tuples(
            &extras.in_lp_val,
            &extras.in_lp_idx,
            opt_texts(&extras.in_lp_txt),
        );
    }
    if !extras.out_top_lens.is_empty() {
        meta["output_top_logprobs"] = ragged_logprob_tuples(
            &extras.out_top_val,
            &extras.out_top_idx,
            &extras.out_top_lens,
            opt_texts(&extras.out_top_txt),
        );
    }
    if !extras.in_top_lens.is_empty() {
        meta["input_top_logprobs"] = ragged_logprob_tuples(
            &extras.in_top_val,
            &extras.in_top_idx,
            &extras.in_top_lens,
            opt_texts(&extras.in_top_txt),
        );
    }
    if !extras.out_tid_lens.is_empty() {
        meta["output_token_ids_logprobs"] = ragged_logprob_tuples(
            &extras.out_tid_val,
            &extras.out_tid_idx,
            &extras.out_tid_lens,
            opt_texts(&extras.out_tid_txt),
        );
    }
    if !extras.in_tid_lens.is_empty() {
        meta["input_token_ids_logprobs"] = ragged_logprob_tuples(
            &extras.in_tid_val,
            &extras.in_tid_idx,
            &extras.in_tid_lens,
            opt_texts(&extras.in_tid_txt),
        );
    }
    if !extras.hidden_lens.is_empty() {
        meta["hidden_states"] = hidden_states_rows(&extras.hidden_val, &extras.hidden_lens);
    }
    meta
}
