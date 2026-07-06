//! Sampling-params normalization — the Rust port of `SamplingParams`'s
//! `__post_init__` + `normalize` + `verify`
//! (python/sglang/srt/sampling/sampling_params.py).
//!
//! The embedded Rust server replaces the Python `TokenizerManager`, which is the
//! only place those three run on the normal (zmq) path. Running them here, in the
//! ingress `Normalizing` FSM step, keeps the per-request CPU (notably the
//! stop-string work) off the scheduler's latency-critical loop. We set
//! `is_normalized=true` on the wire so the scheduler's `__post_init__` and
//! `normalize` early-return; its `verify` is likewise skipped (we did it here).
//!
//! KEEP IN SYNC with `sampling_params.py`: the constants and ranges below mirror
//! that file. `stop_str_max_len` is the stop string's **UTF-8 byte length** — a
//! provably safe over-estimate of its token length (a token spans ≥ 1 byte, so
//! `bytes ≥ tokens`; `chars` is *not* a bound — one char can be several tokens,
//! e.g. `𓀀` → 3). The scheduler uses this only as a match-window *size* (capped at
//! the output length), so any over-estimate matches the same stops — only an
//! under-estimate misses. Python encodes each stop with the tokenizer for the
//! exact token count; the byte-length bound avoids needing the tokenizer here.
//!
//! Not ported (the Rust OpenAI/`/generate` handlers don't populate them): the
//! vocab-bounded `logit_bias` range check, and `stop_token_ids` filtering. Add
//! them here if those fields start flowing through.

use rmpv::Value;

use crate::error::Error;

/// `_SAMPLING_EPS` — temperatures in `[0, eps)` mean greedy decoding.
const SAMPLING_EPS: f64 = 1e-6;
/// `TOP_K_ALL = 1 << 30` — `top_k` sentinel for "consider the whole vocabulary".
const TOP_K_ALL: i64 = 1 << 30;
/// Conservative `stop_regex_max_len`: Python's `get_max_seq_length` returns
/// `MAX_LEN = 1 << 30` for any unbounded quantifier (the common case). The
/// scheduler caps the match tail at the output length, so this just means "scan
/// the whole output", matching Python's worst case.
const STOP_REGEX_MAX_LEN: i64 = 1 << 30;

/// Normalize and verify the request's sampling params in place, then mark them
/// `is_normalized=true` so the scheduler skips its own pass. `None`/absent map →
/// an all-defaults map. Returns `Error::Validation` (HTTP 400) for any param the
/// Python `verify` would reject.
///
/// One pass over the map: the fields we normalize are read into typed locals and
/// everything else (max_new_tokens, grammars, unknowns) is carried through
/// untouched. See `TM_CORES` (`runtime::threads`) for the ingress-scaling ceiling
/// this constant feeds into.
pub fn normalize_sampling_params(sp: &mut Option<Value>) -> Result<(), Error> {
    let map = match sp.take() {
        Some(Value::Map(m)) => m,
        None => Vec::new(),
        Some(_) => return Err(Error::Validation("sampling_params must be a map".into())),
    };

    // Defaults match the SamplingParams field defaults; a present-but-null value
    // reads as absent (keeps the default), matching Python's `x if x is not None`.
    let mut temperature_in = 1.0;
    let mut top_k_in = -1i64;
    let mut top_p = 1.0;
    let mut min_p = 0.0;
    let mut frequency_penalty = 0.0;
    let mut presence_penalty = 0.0;
    let mut repetition_penalty = 1.0;
    let mut min_new_tokens = 0i64;
    let mut max_new_tokens = 128i64; // field default is 128, not None
    let mut stop_strs: Vec<String> = Vec::new();
    let mut stop_regex_strs: Vec<String> = Vec::new();
    let mut grammars = 0usize;

    // Single pass: pull the normalized fields into locals, keep everything else as
    // passthrough. Fields we re-emit below are dropped here to avoid duplicate keys.
    let mut out: Vec<(Value, Value)> = Vec::with_capacity(map.len() + 8);
    for (k, v) in map {
        match k.as_str() {
            Some("temperature") => temperature_in = as_f64(&v).unwrap_or(temperature_in),
            Some("top_k") => top_k_in = as_i64(&v).unwrap_or(top_k_in),
            Some("top_p") => top_p = as_f64(&v).unwrap_or(top_p),
            Some("min_p") => min_p = as_f64(&v).unwrap_or(min_p),
            Some("frequency_penalty") => {
                frequency_penalty = as_f64(&v).unwrap_or(frequency_penalty)
            }
            Some("presence_penalty") => presence_penalty = as_f64(&v).unwrap_or(presence_penalty),
            Some("repetition_penalty") => {
                repetition_penalty = as_f64(&v).unwrap_or(repetition_penalty)
            }
            Some("min_new_tokens") => min_new_tokens = as_i64(&v).unwrap_or(min_new_tokens),
            // Read for verify but kept on the wire — the scheduler still needs it.
            Some("max_new_tokens") => {
                max_new_tokens = as_i64(&v).unwrap_or(max_new_tokens);
                out.push((k, v));
            }
            // stop / stop_regex → string lists + match-window lengths (below). Kept
            // on the wire too; the scheduler reads stop_strs, the alias is harmless.
            Some("stop") => {
                stop_strs = to_string_list(Some(&v));
                out.push((k, v));
            }
            Some("stop_regex") => {
                stop_regex_strs = to_string_list(Some(&v));
                out.push((k, v));
            }
            // Grammars are mutually exclusive: count the non-null ones, keep them.
            Some("regex") | Some("json_schema") | Some("ebnf") => {
                if !v.is_nil() {
                    grammars += 1;
                }
                out.push((k, v));
            }
            // Drop any pre-existing copies of the fields we re-emit below.
            Some("stop_strs")
            | Some("stop_str_max_len")
            | Some("stop_regex_strs")
            | Some("stop_regex_max_len")
            | Some("is_normalized") => {}
            _ => out.push((k, v)),
        }
    }

    // --- __post_init__: greedy / top_k special cases ---
    let (temperature, top_k) = if (0.0..SAMPLING_EPS).contains(&temperature_in) {
        // Greedy: temperature ~0 → temperature=1.0, top_k=1.
        (1.0, 1)
    } else if top_k_in == -1 {
        (temperature_in, TOP_K_ALL) // -1 disables top_k → whole vocabulary
    } else {
        (temperature_in, top_k_in)
    };
    // Match window: UTF-8 byte length is a safe upper bound on the token count.
    let stop_str_max_len = stop_strs.iter().map(|s| s.len() as i64).max().unwrap_or(0);
    let stop_regex_max_len = if stop_regex_strs.is_empty() {
        0
    } else {
        STOP_REGEX_MAX_LEN
    };

    // --- verify: same ranges as SamplingParams.verify (range-only subset) ---
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(bad(format!(
            "temperature must be a non-negative finite number, got {temperature}"
        )));
    }
    if !(top_p > 0.0 && top_p <= 1.0) {
        return Err(bad(format!("top_p must be in (0, 1], got {top_p}")));
    }
    if !(0.0..=1.0).contains(&min_p) {
        return Err(bad(format!("min_p must be in [0, 1], got {min_p}")));
    }
    if top_k < 1 {
        return Err(bad(format!(
            "top_k must be -1 (disable) or at least 1, got {top_k_in}"
        )));
    }
    if !(-2.0..=2.0).contains(&frequency_penalty) {
        return Err(bad(format!(
            "frequency_penalty must be in [-2, 2], got {frequency_penalty}"
        )));
    }
    if !(-2.0..=2.0).contains(&presence_penalty) {
        return Err(bad(format!(
            "presence_penalty must be in [-2, 2], got {presence_penalty}"
        )));
    }
    if !(repetition_penalty > 0.0 && repetition_penalty <= 2.0) {
        return Err(bad(format!(
            "repetition_penalty must be in (0, 2], got {repetition_penalty}"
        )));
    }
    if min_new_tokens < 0 {
        return Err(bad(format!(
            "min_new_tokens must be non-negative, got {min_new_tokens}"
        )));
    }
    if max_new_tokens < 0 {
        return Err(bad(format!(
            "max_new_tokens must be at least 0, got {max_new_tokens}"
        )));
    }
    if min_new_tokens > max_new_tokens {
        return Err(bad(format!(
            "min_new_tokens must be in [0, max_new_tokens({max_new_tokens})], got {min_new_tokens}"
        )));
    }
    // Grammars are mutually exclusive (count taken during the pass above).
    if grammars > 1 {
        return Err(bad(
            "Only one of regex, json_schema, or ebnf can be set".into()
        ));
    }

    // --- emit the normalized fields; is_normalized=true tells the scheduler its
    // __post_init__/normalize/verify are already done ---
    out.push((Value::from("temperature"), Value::F64(temperature)));
    out.push((Value::from("top_k"), Value::from(top_k)));
    out.push((Value::from("top_p"), Value::F64(top_p)));
    out.push((Value::from("min_p"), Value::F64(min_p)));
    out.push((
        Value::from("frequency_penalty"),
        Value::F64(frequency_penalty),
    ));
    out.push((
        Value::from("presence_penalty"),
        Value::F64(presence_penalty),
    ));
    out.push((
        Value::from("repetition_penalty"),
        Value::F64(repetition_penalty),
    ));
    out.push((Value::from("min_new_tokens"), Value::from(min_new_tokens)));
    out.push((Value::from("stop_strs"), string_array(&stop_strs)));
    out.push((
        Value::from("stop_str_max_len"),
        Value::from(stop_str_max_len),
    ));
    out.push((
        Value::from("stop_regex_strs"),
        string_array(&stop_regex_strs),
    ));
    out.push((
        Value::from("stop_regex_max_len"),
        Value::from(stop_regex_max_len),
    ));
    out.push((Value::from("is_normalized"), Value::Boolean(true)));

    *sp = Some(Value::Map(out));
    Ok(())
}

fn bad(msg: String) -> Error {
    Error::Validation(msg)
}

/// Coerce a value to f64 (int or float); non-numeric (incl. null) → None, so the
/// caller keeps its default — matching Python's `x if x is not None else default`.
fn as_f64(v: &Value) -> Option<f64> {
    match v {
        Value::F64(f) => Some(*f),
        Value::F32(f) => Some(*f as f64),
        Value::Integer(i) => i.as_f64(),
        _ => None,
    }
}

fn as_i64(v: &Value) -> Option<i64> {
    match v {
        Value::Integer(i) => i.as_i64(),
        Value::F64(f) => Some(*f as i64),
        Value::F32(f) => Some(*f as i64),
        _ => None,
    }
}

/// `stop` / `stop_regex` accept a single string or a list of strings; null/absent
/// or any other shape → empty list.
fn to_string_list(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::String(s)) => s.as_str().map(|s| vec![s.to_string()]).unwrap_or_default(),
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(|x| x.as_str().map(String::from))
            .collect(),
        _ => Vec::new(),
    }
}

fn string_array(strs: &[String]) -> Value {
    Value::Array(strs.iter().map(|s| Value::from(s.as_str())).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norm(pairs: &[(&str, Value)]) -> Vec<(Value, Value)> {
        let mut sp = Some(Value::Map(
            pairs
                .iter()
                .map(|(k, v)| (Value::from(*k), v.clone()))
                .collect(),
        ));
        normalize_sampling_params(&mut sp).unwrap();
        match sp.unwrap() {
            Value::Map(m) => m,
            _ => unreachable!(),
        }
    }

    fn get<'a>(m: &'a [(Value, Value)], k: &str) -> &'a Value {
        m.iter()
            .find(|(kk, _)| kk.as_str() == Some(k))
            .map(|(_, v)| v)
            .unwrap()
    }

    #[test]
    fn greedy_sets_temp_one_topk_one() {
        let m = norm(&[("temperature", Value::F64(0.0))]);
        assert_eq!(get(&m, "temperature"), &Value::F64(1.0));
        assert_eq!(get(&m, "top_k").as_i64(), Some(1));
        assert_eq!(get(&m, "is_normalized"), &Value::Boolean(true));
    }

    #[test]
    fn topk_minus_one_becomes_all() {
        let m = norm(&[("temperature", Value::F64(0.7))]);
        assert_eq!(get(&m, "top_k").as_i64(), Some(TOP_K_ALL));
    }

    #[test]
    fn stop_list_and_max_len_by_bytes() {
        let m = norm(&[(
            "stop",
            Value::Array(vec![Value::from("Question:"), Value::from("\n\n")]),
        )]);
        assert_eq!(get(&m, "stop_str_max_len").as_i64(), Some(9)); // "Question:" (ASCII)
        let Value::Array(a) = get(&m, "stop_strs") else {
            panic!("stop_strs not array")
        };
        assert_eq!(a.len(), 2);
    }

    /// A multi-byte stop char must use its byte length as the window bound: `𓀀`
    /// is 1 char but 4 UTF-8 bytes (and 3 tokens on Qwen3). Char count (1) would
    /// under-size the tail and miss the stop; byte count (4) ≥ the token span.
    #[test]
    fn stop_str_max_len_uses_bytes_not_chars() {
        let m = norm(&[("stop", Value::from("𓀀"))]);
        assert_eq!("𓀀".chars().count(), 1);
        assert_eq!("𓀀".len(), 4);
        assert_eq!(get(&m, "stop_str_max_len").as_i64(), Some(4));
    }

    #[test]
    fn no_stop_yields_empty_list_zero_len() {
        let m = norm(&[("temperature", Value::F64(0.0))]);
        assert_eq!(get(&m, "stop_str_max_len").as_i64(), Some(0));
        assert_eq!(get(&m, "stop_strs"), &Value::Array(vec![]));
    }

    /// The single pass must carry unknown fields through untouched and must not
    /// duplicate a field it normalizes (client-sent temperature is replaced, not
    /// appended alongside the original).
    #[test]
    fn passthrough_preserved_and_normalized_fields_not_duplicated() {
        let m = norm(&[
            ("temperature", Value::F64(0.7)),
            ("max_new_tokens", Value::from(64i64)),
            ("ignore_eos", Value::Boolean(true)),
        ]);
        // Unknown / read-only fields survive verbatim.
        assert_eq!(get(&m, "max_new_tokens").as_i64(), Some(64));
        assert_eq!(get(&m, "ignore_eos"), &Value::Boolean(true));
        // Exactly one temperature entry, holding the normalized value.
        let temps: Vec<_> = m
            .iter()
            .filter(|(k, _)| k.as_str() == Some("temperature"))
            .collect();
        assert_eq!(temps.len(), 1);
        assert_eq!(temps[0].1, Value::F64(0.7));
    }

    #[test]
    fn verify_rejects_bad_top_p() {
        let mut sp = Some(Value::Map(vec![(Value::from("top_p"), Value::F64(2.0))]));
        assert!(normalize_sampling_params(&mut sp).is_err());
    }

    #[test]
    fn verify_rejects_topk_zero() {
        let mut sp = Some(Value::Map(vec![
            (Value::from("temperature"), Value::F64(0.7)),
            (Value::from("top_k"), Value::from(0i64)),
        ]));
        assert!(normalize_sampling_params(&mut sp).is_err());
    }
}
