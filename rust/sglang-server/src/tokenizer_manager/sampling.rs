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
pub fn normalize_sampling_params(sp: &mut Option<Value>) -> Result<(), Error> {
    let mut map = match sp.take() {
        Some(Value::Map(m)) => m,
        None => Vec::new(),
        Some(_) => return Err(Error::Validation("sampling_params must be a map".into())),
    };

    // --- __post_init__: fill defaults, then the greedy / top_k special cases ---
    let temperature_in = get_f64(&map, "temperature").unwrap_or(1.0);
    let top_k_in = get_i64(&map, "top_k").unwrap_or(-1);
    let (temperature, top_k) = if (0.0..SAMPLING_EPS).contains(&temperature_in) {
        // Greedy: temperature ~0 → temperature=1.0, top_k=1.
        (1.0, 1)
    } else if top_k_in == -1 {
        (temperature_in, TOP_K_ALL) // -1 disables top_k → whole vocabulary
    } else {
        (temperature_in, top_k_in)
    };
    let top_p = get_f64(&map, "top_p").unwrap_or(1.0);
    let min_p = get_f64(&map, "min_p").unwrap_or(0.0);
    let frequency_penalty = get_f64(&map, "frequency_penalty").unwrap_or(0.0);
    let presence_penalty = get_f64(&map, "presence_penalty").unwrap_or(0.0);
    let repetition_penalty = get_f64(&map, "repetition_penalty").unwrap_or(1.0);
    let min_new_tokens = get_i64(&map, "min_new_tokens").unwrap_or(0);
    // Field default is 128 (not None), so an absent value verifies against 128.
    let max_new_tokens = get_i64(&map, "max_new_tokens").unwrap_or(128);

    // stop / stop_regex → string lists + match-window lengths. Use the UTF-8 byte
    // length as a safe upper bound on the token count.
    let stop_strs = to_string_list(find(&map, "stop"));
    let stop_str_max_len = stop_strs.iter().map(|s| s.len() as i64).max().unwrap_or(0);
    let stop_regex_strs = to_string_list(find(&map, "stop_regex"));
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
    // Grammars are mutually exclusive.
    let grammars = ["regex", "json_schema", "ebnf"]
        .iter()
        .filter(|k| present_non_null(&map, k))
        .count();
    if grammars > 1 {
        return Err(bad(
            "Only one of regex, json_schema, or ebnf can be set".into()
        ));
    }

    // --- write the normalized fields back; is_normalized=true tells the
    // scheduler its __post_init__/normalize/verify are already done ---
    set(&mut map, "temperature", Value::F64(temperature));
    set(&mut map, "top_k", Value::from(top_k));
    set(&mut map, "top_p", Value::F64(top_p));
    set(&mut map, "min_p", Value::F64(min_p));
    set(&mut map, "frequency_penalty", Value::F64(frequency_penalty));
    set(&mut map, "presence_penalty", Value::F64(presence_penalty));
    set(
        &mut map,
        "repetition_penalty",
        Value::F64(repetition_penalty),
    );
    set(&mut map, "min_new_tokens", Value::from(min_new_tokens));
    set(&mut map, "stop_strs", string_array(&stop_strs));
    set(&mut map, "stop_str_max_len", Value::from(stop_str_max_len));
    set(&mut map, "stop_regex_strs", string_array(&stop_regex_strs));
    set(
        &mut map,
        "stop_regex_max_len",
        Value::from(stop_regex_max_len),
    );
    set(&mut map, "is_normalized", Value::Boolean(true));

    *sp = Some(Value::Map(map));
    Ok(())
}

fn bad(msg: String) -> Error {
    Error::Validation(msg)
}

fn find<'a>(map: &'a [(Value, Value)], key: &str) -> Option<&'a Value> {
    map.iter()
        .find(|(k, _)| k.as_str() == Some(key))
        .map(|(_, v)| v)
}

/// A present-but-null value reads as absent (→ default), matching Python's
/// `x if x is not None else default` coercion.
fn get_f64(map: &[(Value, Value)], key: &str) -> Option<f64> {
    match find(map, key)? {
        Value::F64(f) => Some(*f),
        Value::F32(f) => Some(*f as f64),
        Value::Integer(i) => i.as_f64(),
        _ => None,
    }
}

fn get_i64(map: &[(Value, Value)], key: &str) -> Option<i64> {
    match find(map, key)? {
        Value::Integer(i) => i.as_i64(),
        Value::F64(f) => Some(*f as i64),
        Value::F32(f) => Some(*f as i64),
        _ => None,
    }
}

fn present_non_null(map: &[(Value, Value)], key: &str) -> bool {
    matches!(find(map, key), Some(v) if !v.is_nil())
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

fn set(map: &mut Vec<(Value, Value)>, key: &str, val: Value) {
    if let Some(slot) = map.iter_mut().find(|(k, _)| k.as_str() == Some(key)) {
        slot.1 = val;
    } else {
        map.push((Value::from(key), val));
    }
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
        find(m, k).unwrap()
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
