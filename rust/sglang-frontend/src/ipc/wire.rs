// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! Build msgpack-encoded IPC messages matching Python `msgspec.Struct, tag=True` format.
//!
//! Wire format: every struct is a msgpack fixmap where the first entry is
//! `"type": "<ClassName>"` followed by field name → value pairs.
//! Nested structs (e.g. SamplingParamsIPC) use the same encoding.

use crate::detokenizer::msgpack_encode;
use rmpv::Value;
use tokenizers::Tokenizer;

const SAMPLING_EPS: f64 = 1e-6;
const TOP_K_ALL: i32 = 1 << 30;

// ─────────────────────────────── helpers ────────────────────────────────────

fn vs(s: &str) -> Value {
    Value::String(rmpv::Utf8String::from(s.to_string()))
}

fn vi(n: i64) -> Value {
    Value::Integer(n.into())
}

fn vf(f: f64) -> Value {
    Value::F64(f)
}

fn vb(b: bool) -> Value {
    Value::Boolean(b)
}

fn vnil() -> Value {
    Value::Nil
}

// ─────────────────────────── SamplingParams ─────────────────────────────────

/// Rust-side sampling parameters mirroring Python's `SamplingParams` /
/// `SamplingParamsIPC`.  Call [`SamplingParams::normalize`] before sending to
/// the scheduler — it mirrors `SamplingParams.normalize()` in sampling_params.py.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub max_new_tokens: u32,
    pub temperature: f64,
    pub top_p: f64,
    /// -1 means "whole vocabulary" (TOP_K_ALL); normalize() converts it.
    pub top_k: i32,
    pub min_p: f64,
    pub frequency_penalty: f64,
    pub presence_penalty: f64,
    pub repetition_penalty: f64,
    pub min_new_tokens: u32,
    pub n: u32,
    pub stop: Option<Vec<String>>,
    pub stop_token_ids: Option<Vec<i32>>,
    pub stop_regex: Option<Vec<String>>,
    pub ignore_eos: bool,
    pub skip_special_tokens: bool,
    pub spaces_between_special_tokens: bool,
    pub no_stop_trim: bool,
    pub seed: Option<i64>,
    pub json_schema: Option<String>,
    /// Populated by normalize(): max token span of any stop string.
    pub stop_str_max_len: i64,
    /// Populated by normalize(): max character span of any stop regex pattern.
    pub stop_regex_max_len: i64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams {
            max_new_tokens: 128,
            temperature: 1.0,
            top_p: 1.0,
            top_k: -1,
            min_p: 0.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            min_new_tokens: 0,
            n: 1,
            stop: None,
            stop_token_ids: None,
            stop_regex: None,
            ignore_eos: false,
            skip_special_tokens: true,
            spaces_between_special_tokens: true,
            no_stop_trim: false,
            seed: None,
            json_schema: None,
            stop_str_max_len: 0,
            stop_regex_max_len: 0,
        }
    }
}

impl SamplingParams {
    /// Mirror Python's `SamplingParams.__init__` normalization and
    /// `SamplingParams.normalize(tokenizer)`.
    ///
    /// Must be called once before the params are encoded and sent to the
    /// scheduler.  Requires an optional tokenizer reference to compute
    /// `stop_str_max_len` accurately; falls back to character count if None.
    pub fn normalize(&mut self, tokenizer: Option<&Tokenizer>) {
        // Near-zero temperature → greedy (top_k = 1)
        if (0.0..SAMPLING_EPS).contains(&self.temperature) {
            self.temperature = 1.0;
            self.top_k = 1;
        }
        // top_k = -1 means "use whole vocabulary"
        if self.top_k == -1 {
            self.top_k = TOP_K_ALL;
        }

        // Normalize None → [] for stop lists (Python normalize() does the same;
        // the scheduler calls len() on these fields and requires a list, not None).
        let stops = self.stop.get_or_insert_with(Vec::new);
        self.stop_str_max_len = stops
            .iter()
            .map(|s| {
                tokenizer
                    .and_then(|tok| tok.encode(s.as_str(), false).ok())
                    .map(|enc| enc.get_ids().len())
                    .unwrap_or(s.len()) as i64
            })
            .max()
            .unwrap_or(0);

        let regexes = self.stop_regex.get_or_insert_with(Vec::new);
        self.stop_regex_max_len = regexes.iter().map(|r| regex_max_len(r)).max().unwrap_or(0);
    }
}

/// Upper-bound the number of characters (≈ tokens) a regex pattern can match.
/// Mirrors Python's `get_max_seq_length` / `_max_length_from_subpattern`.
///
/// Returns 2^30 for patterns that contain unbounded quantifiers (`*`, `+`,
/// or `{n,}` with no upper bound), and the pattern's byte length otherwise
/// (conservative: each character ≈ one token in the worst case).
fn regex_max_len(pattern: &str) -> i64 {
    const MAX_LEN: i64 = 1 << 30;
    let mut chars = pattern.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            chars.next(); // skip escaped character — not a quantifier
            continue;
        }
        match c {
            '*' | '+' => return MAX_LEN,
            '{' => {
                // Collect up to closing '}'
                let inner: String = chars.by_ref().take_while(|&x| x != '}').collect();
                if let Some(after_comma) = inner.splitn(2, ',').nth(1) {
                    if after_comma.trim().is_empty() {
                        // {n,} — unbounded upper bound
                        return MAX_LEN;
                    }
                }
            }
            _ => {}
        }
    }
    pattern.len() as i64
}

// ─────────────────────── msgpack value builders ──────────────────────────────

fn sampling_params_to_value(sp: &SamplingParams) -> Value {
    // After normalize(), stop/stop_regex are Some(vec![]) for the empty case.
    // Serialize Some(vec) as an array (even if empty) so the scheduler receives []
    // instead of null — Python scheduler calls len() on these fields without None checks.
    let stop_val = match &sp.stop {
        Some(stops) => Value::Array(stops.iter().map(|s| vs(s)).collect()),
        None => vnil(),
    };

    let stop_token_ids_val = match &sp.stop_token_ids {
        Some(ids) => Value::Array(ids.iter().map(|&x| vi(x as i64)).collect()),
        None => vnil(),
    };

    let stop_regex_val = match &sp.stop_regex {
        Some(regexes) => Value::Array(regexes.iter().map(|r| vs(r)).collect()),
        None => vnil(),
    };

    let json_schema_val = sp.json_schema.as_deref().map(vs).unwrap_or_else(vnil);
    let seed_val = sp.seed.map(vi).unwrap_or_else(vnil);

    Value::Map(vec![
        (vs("type"), vs("SamplingParamsIPC")),
        (vs("max_new_tokens"), vi(sp.max_new_tokens as i64)),
        (vs("stop"), stop_val),
        (vs("stop_token_ids"), stop_token_ids_val),
        (vs("stop_regex"), stop_regex_val),
        (vs("temperature"), vf(sp.temperature)),
        (vs("top_p"), vf(sp.top_p)),
        (vs("top_k"), vi(sp.top_k as i64)),
        (vs("min_p"), vf(sp.min_p)),
        (vs("frequency_penalty"), vf(sp.frequency_penalty)),
        (vs("presence_penalty"), vf(sp.presence_penalty)),
        (vs("repetition_penalty"), vf(sp.repetition_penalty)),
        (vs("min_new_tokens"), vi(sp.min_new_tokens as i64)),
        (vs("n"), vi(sp.n as i64)),
        (vs("json_schema"), json_schema_val),
        (vs("regex"), vnil()),
        (vs("ebnf"), vnil()),
        (vs("structural_tag"), vnil()),
        (vs("ignore_eos"), vb(sp.ignore_eos)),
        (vs("skip_special_tokens"), vb(sp.skip_special_tokens)),
        (vs("spaces_between_special_tokens"), vb(sp.spaces_between_special_tokens)),
        (vs("no_stop_trim"), vb(sp.no_stop_trim)),
        (vs("custom_params"), vnil()),
        (vs("stream_interval"), vnil()),
        (vs("logit_bias"), vnil()),
        (vs("sampling_seed"), seed_val),
        (vs("stop_str_max_len"), vi(sp.stop_str_max_len)),
        (vs("stop_regex_max_len"), vi(sp.stop_regex_max_len)),
    ])
}

/// Build and encode a `TokenizedGenerateReqInput` msgpack message ready to
/// send to the scheduler via ZMQ PUSH.
///
/// `http_worker_ipc` is the ZMQ address the detokenizer should send responses
/// back to (the tokenizer manager's PULL socket).
pub fn build_generate_req(
    rid: &str,
    input_ids: &[u32],
    sp: &SamplingParams,
    stream: bool,
    http_worker_ipc: &str,
) -> Vec<u8> {
    let ids_val = Value::Array(input_ids.iter().map(|&x| vi(x as i64)).collect());
    let sp_val = sampling_params_to_value(sp);

    let fields: Vec<(Value, Value)> = vec![
        (vs("type"), vs("TokenizedGenerateReqInput")),
        (vs("input_ids"), ids_val),
        (vs("sampling_params"), sp_val),
        (vs("return_logprob"), vb(false)),
        (vs("logprob_start_len"), vi(0)),
        (vs("top_logprobs_num"), vi(0)),
        (vs("stream"), vb(stream)),
        // Required BaseReq fields
        (vs("rid"), vs(rid)),
        (vs("http_worker_ipc"), vs(http_worker_ipc)),
        // Optional fields – all nil for the core text path
        (vs("input_text"), vnil()),
        (vs("token_ids_logprob"), vnil()),
        (vs("mm_inputs"), vnil()),
        (vs("return_hidden_states"), vb(false)),
        (vs("return_routed_experts"), vb(false)),
        (vs("routed_experts_start_len"), vi(0)),
        (vs("input_embeds"), vnil()),
        (vs("positional_embed_overrides"), vnil()),
        (vs("session_params"), vnil()),
        (vs("lora_id"), vnil()),
        (vs("custom_logit_processor"), vnil()),
        (vs("bootstrap_host"), vnil()),
        (vs("bootstrap_port"), vnil()),
        (vs("bootstrap_room"), vnil()),
        (vs("bootstrap_pair_key"), vnil()),
        (vs("decode_tp_size"), vnil()),
        (vs("require_reasoning"), vb(false)),
        (vs("routed_dp_rank"), vnil()),
        (vs("disagg_prefill_dp_rank"), vnil()),
        (vs("priority"), vnil()),
        (vs("extra_key"), vnil()),
        (vs("routing_key"), vnil()),
        (vs("no_logs"), vb(false)),
        (vs("return_bytes"), vb(false)),
        (vs("return_entropy"), vb(false)),
        (vs("token_type_ids"), vnil()),
        (vs("need_wait_for_mm_inputs"), vnil()),
        (vs("num_items_assigned"), vnil()),
        (vs("multi_item_delimiter_indices"), vnil()),
        (vs("time_stats"), vnil()),
    ];

    msgpack_encode(&Value::Map(fields))
}

/// Build and encode an `AbortReq` msgpack message to cancel a pending request.
pub fn build_abort_req(rid: &str) -> Vec<u8> {
    let fields: Vec<(Value, Value)> = vec![
        (vs("type"), vs("AbortReq")),
        (vs("abort_all"), vb(false)),
        (vs("finished_reason"), vnil()),
        (vs("abort_message"), vnil()),
        (vs("rid"), vs(rid)),
        (vs("http_worker_ipc"), vnil()),
    ];
    msgpack_encode(&Value::Map(fields))
}
