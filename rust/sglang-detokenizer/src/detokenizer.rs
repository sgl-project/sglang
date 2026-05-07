use base64::{engine::general_purpose::STANDARD as B64, Engine};
use indexmap::IndexMap;
use rmpv::Value;
use tokenizers::Tokenizer;

// ─────────────────────────────── state ──────────────────────────────────────

pub struct DecodeStatus {
    pub decoded_text: String,
    pub decode_ids: Vec<i64>,
    pub surr_offset: usize,
    pub read_offset: usize,
    pub sent_offset: usize,
}

/// OrderedDict with a maximum capacity; oldest entry is evicted when full.
pub struct LimitedCapacityDict {
    map: IndexMap<String, DecodeStatus>,
    capacity: usize,
}

impl LimitedCapacityDict {
    pub fn new(capacity: usize) -> Self {
        Self {
            map: IndexMap::new(),
            capacity,
        }
    }

    pub fn get(&self, key: &str) -> Option<&DecodeStatus> {
        self.map.get(key)
    }

    pub fn get_mut(&mut self, key: &str) -> Option<&mut DecodeStatus> {
        self.map.get_mut(key)
    }

    pub fn insert(&mut self, key: String, value: DecodeStatus) {
        if self.map.len() >= self.capacity && !self.map.contains_key(&key) {
            self.map.shift_remove_index(0);
        }
        self.map.insert(key, value);
    }

    pub fn remove(&mut self, key: &str) {
        self.map.shift_remove(key);
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// ─────────────────────────── rmpv helpers ───────────────────────────────────

/// Find the value for `key` in a msgpack map (Vec of (k,v) pairs).
pub fn map_get<'a>(map: &'a [(Value, Value)], key: &str) -> Option<&'a Value> {
    map.iter()
        .find(|(k, _)| k.as_str() == Some(key))
        .map(|(_, v)| v)
}

fn v_str(v: &Value) -> Option<String> {
    v.as_str().map(|s| s.to_string())
}

fn v_bool(v: &Value) -> bool {
    match v {
        Value::Boolean(b) => *b,
        _ => false,
    }
}

fn v_i64(v: &Value) -> Option<i64> {
    match v {
        Value::Integer(i) => i.as_i64(),
        _ => None,
    }
}

/// Convert a msgpack value to a Vec<i64>. Handles both Array([ints]) and a bare Integer.
fn v_i64_array(v: &Value) -> Vec<i64> {
    match v {
        Value::Array(arr) => arr.iter().filter_map(v_i64).collect(),
        Value::Integer(i) => i.as_i64().map(|x| vec![x]).unwrap_or_default(),
        Value::Nil => vec![],
        _ => vec![],
    }
}

fn v_bool_array(v: &Value) -> Vec<bool> {
    match v {
        Value::Array(arr) => arr.iter().map(v_bool).collect(),
        _ => vec![],
    }
}

fn v_string_array(v: &Value) -> Vec<String> {
    match v {
        Value::Array(arr) => arr.iter().filter_map(v_str).collect(),
        Value::Nil => vec![],
        _ => vec![],
    }
}

/// Extract the string tag from a msgpack map (the "type" discriminant).
pub fn read_type_tag(map: &[(Value, Value)]) -> Option<String> {
    map_get(map, "type").and_then(|v| v.as_str().map(|s| s.to_string()))
}

// ──────────────────────────── Unicode helpers ────────────────────────────────

fn is_chinese_char(cp: char) -> bool {
    let c = cp as u32;
    matches!(
        c,
        0x4E00..=0x9FFF
            | 0x3400..=0x4DBF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B820..=0x2CEAF
            | 0xF900..=0xFAFF
            | 0x2F800..=0x2FA1F
    )
}

/// Returns the longest prefix of `text` that is safe to emit in a streaming
/// context (no incomplete multi-byte sequences, no partially-decoded words).
fn find_printable_text(text: &str) -> &str {
    if text.ends_with('\n') {
        return text;
    }
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return text;
    }
    if is_chinese_char(*chars.last().unwrap()) {
        return text;
    }
    if chars.len() > 1 && is_chinese_char(chars[chars.len() - 2]) {
        let byte_len: usize = chars[..chars.len() - 1].iter().map(|c| c.len_utf8()).sum();
        return &text[..byte_len];
    }
    match text.rfind(' ') {
        Some(pos) => &text[..pos + 1],
        None => "",
    }
}

// ─────────────────────── stop-sequence trimming ──────────────────────────────

fn trim_matched_stop_ids(
    ids: &[i64],
    finished_reason: Option<&Value>,
    no_stop_trim: bool,
    is_gpt_oss: bool,
) -> Vec<i64> {
    if no_stop_trim || finished_reason.is_none() {
        return ids.to_vec();
    }
    let fr = finished_reason.unwrap();
    let matched = match fr {
        Value::Map(m) => map_get(m, "matched"),
        _ => None,
    };
    match matched {
        Some(Value::Integer(i)) => {
            let token_id = i.as_i64().unwrap_or(-1);
            // 200012 is <|call|> for gpt-oss model – keep it even though it's a stop token
            if is_gpt_oss && token_id == 200012 {
                return ids.to_vec();
            }
            if ids.is_empty() {
                ids.to_vec()
            } else {
                ids[..ids.len() - 1].to_vec()
            }
        }
        _ => ids.to_vec(),
    }
}

fn trim_matched_stop_str(
    text: &str,
    finished_reason: Option<&Value>,
    no_stop_trim: bool,
) -> String {
    if no_stop_trim || finished_reason.is_none() {
        return text.to_string();
    }
    let fr = finished_reason.unwrap();
    let matched = match fr {
        Value::Map(m) => map_get(m, "matched"),
        _ => None,
    };
    match matched {
        Some(Value::String(s)) => {
            if let Some(s) = s.as_str() {
                match text.find(s) {
                    Some(pos) => text[..pos].to_string(),
                    None => text.to_string(),
                }
            } else {
                text.to_string()
            }
        }
        _ => text.to_string(),
    }
}

// ───────────────────────── tokenizer decode ──────────────────────────────────

fn grouped_batch_decode(
    tokenizer: &Tokenizer,
    ids_list: &[Vec<i64>],
    skip_list: &[bool],
    _space_list: &[bool],
    disable_batch: bool,
) -> Vec<String> {
    let bs = ids_list.len();
    if bs == 0 {
        return vec![];
    }

    if disable_batch {
        return ids_list
            .iter()
            .zip(skip_list.iter())
            .map(|(ids, &skip)| {
                let u32_ids: Vec<u32> = ids.iter().map(|&x| x as u32).collect();
                tokenizer.decode(&u32_ids, skip).unwrap_or_default()
            })
            .collect();
    }

    let first_skip = skip_list.first().copied().unwrap_or(true);
    if skip_list.iter().all(|&s| s == first_skip) {
        let sequences: Vec<Vec<u32>> = ids_list
            .iter()
            .map(|ids| ids.iter().map(|&x| x as u32).collect())
            .collect();
        let refs: Vec<&[u32]> = sequences.iter().map(|v| v.as_slice()).collect();
        return tokenizer
            .decode_batch(&refs, first_skip)
            .unwrap_or_else(|_| vec![String::new(); bs]);
    }

    let mut results = vec![String::new(); bs];
    let mut false_indices: Vec<usize> = vec![];
    let mut true_indices: Vec<usize> = vec![];
    for (i, &skip) in skip_list.iter().enumerate() {
        if skip {
            true_indices.push(i);
        } else {
            false_indices.push(i);
        }
    }

    for (skip, indices) in [(true, &true_indices), (false, &false_indices)] {
        if indices.is_empty() {
            continue;
        }
        let sequences: Vec<Vec<u32>> = indices
            .iter()
            .map(|&i| ids_list[i].iter().map(|&x| x as u32).collect())
            .collect();
        let refs: Vec<&[u32]> = sequences.iter().map(|v| v.as_slice()).collect();
        let decoded = tokenizer
            .decode_batch(&refs, skip)
            .unwrap_or_else(|_| vec![String::new(); indices.len()]);
        for (idx, text) in indices.iter().zip(decoded) {
            results[*idx] = text;
        }
    }
    results
}

// ─────────────────── three-phase incremental-decoding API ───────────────────

#[derive(Clone)]
pub struct Config {
    pub disable_batch_decode: bool,
    pub is_gpt_oss: bool,
    pub max_states: usize,
}

/// Intermediate state produced by Phase 1; passed through Phase 2 and consumed by Phase 3.
/// All fields are `Send + 'static` so this can be moved into `spawn_blocking`.
pub struct DecodeWork {
    pub rids: Vec<String>,
    pub finished_reasons: Vec<Option<Value>>,
    pub no_stop_trim: Vec<bool>,
    pub surr_ids: Vec<Vec<i64>>,
    pub read_ids: Vec<Vec<i64>>,
    pub skip_special: Vec<bool>,
    pub spaces_between: Vec<bool>,
}

/// Results of Phase 2 (CPU-bound tokenizer decode).
pub struct DecodeResults {
    pub surr_texts: Vec<String>,
    pub read_texts: Vec<String>,
}

/// Phase 1: update per-request state and collect token-ID slices for batch decode.
/// Fast; touches `state` but performs no tokenizer I/O.
pub fn prepare_decode_work(
    rids: &[String],
    finished_reasons: Vec<Option<Value>>,
    decoded_texts: &[String],
    new_decode_ids: &[Vec<i64>],
    read_offsets: &[usize],
    skip_special: Vec<bool>,
    spaces_between: Vec<bool>,
    no_stop_trim: Vec<bool>,
    state: &mut LimitedCapacityDict,
    config: &Config,
) -> DecodeWork {
    let bs = rids.len();
    let mut read_ids: Vec<Vec<i64>> = Vec::with_capacity(bs);
    let mut surr_ids: Vec<Vec<i64>> = Vec::with_capacity(bs);

    for i in 0..bs {
        let rid = &rids[i];
        if !state.contains_key(rid) {
            state.insert(
                rid.clone(),
                DecodeStatus {
                    decoded_text: decoded_texts[i].clone(),
                    decode_ids: new_decode_ids[i].clone(),
                    surr_offset: 0,
                    read_offset: read_offsets[i],
                    sent_offset: 0,
                },
            );
        } else {
            let s = state.get_mut(rid).unwrap();
            s.decode_ids.extend_from_slice(&new_decode_ids[i]);
        }

        let s = state.get(rid).expect("state just inserted");
        let fr = finished_reasons[i].as_ref();
        let raw_read = &s.decode_ids[s.surr_offset..];
        let trimmed = trim_matched_stop_ids(raw_read, fr, no_stop_trim[i], config.is_gpt_oss);
        read_ids.push(trimmed);
        surr_ids.push(s.decode_ids[s.surr_offset..s.read_offset].to_vec());
    }

    DecodeWork {
        rids: rids.to_vec(),
        finished_reasons,
        no_stop_trim,
        surr_ids,
        read_ids,
        skip_special,
        spaces_between,
    }
}

/// Phase 2: pure CPU-bound tokenizer decode. No state access; safe for `spawn_blocking`.
pub fn execute_decode(work: &DecodeWork, tokenizer: &Tokenizer, config: &Config) -> DecodeResults {
    let surr_texts = grouped_batch_decode(
        tokenizer,
        &work.surr_ids,
        &work.skip_special,
        &work.spaces_between,
        config.disable_batch_decode,
    );
    let read_texts = grouped_batch_decode(
        tokenizer,
        &work.read_ids,
        &work.skip_special,
        &work.spaces_between,
        config.disable_batch_decode,
    );
    DecodeResults { surr_texts, read_texts }
}

/// Phase 3: compute incremental output strings and advance per-request state.
/// Fast; touches `state` but performs no tokenizer I/O.
pub fn finalize_decode(
    work: &DecodeWork,
    results: &DecodeResults,
    state: &mut LimitedCapacityDict,
) -> Vec<String> {
    let bs = work.rids.len();
    let capacity = state.capacity();
    let mut output_strs = Vec::with_capacity(bs);

    for i in 0..bs {
        let rid = &work.rids[i];
        let s = state.get_mut(rid).unwrap_or_else(|| {
            panic!(
                "Decode status not found for request {rid}. \
                 Increase SGLANG_DETOKENIZER_MAX_STATES (current: {capacity})."
            )
        });

        let new_text = &results.read_texts[i][results.surr_texts[i].len()..];
        let fr = work.finished_reasons[i].as_ref();

        let new_text: String = if fr.is_none() {
            if !new_text.is_empty() && !new_text.ends_with('\u{FFFD}') {
                s.decoded_text.push_str(new_text);
                s.surr_offset = s.read_offset;
                s.read_offset = s.decode_ids.len();
                String::new()
            } else {
                find_printable_text(new_text).to_string()
            }
        } else {
            new_text.to_string()
        };

        let full = format!("{}{}", s.decoded_text, new_text);
        let output_str = trim_matched_stop_str(&full, fr, work.no_stop_trim[i]);

        let incremental = output_str[s.sent_offset..].to_string();
        s.sent_offset = output_str.len();
        output_strs.push(incremental);

        if fr.is_some() {
            state.remove(rid);
        }
    }

    output_strs
}

// ─────────────────── batch-field extraction + output building ───────────────

/// Fields extracted from a `BatchTokenIDOutput` message map.
pub struct BatchInputFields {
    pub rids: Vec<String>,
    pub finished_reasons: Vec<Option<Value>>,
    pub decoded_texts: Vec<String>,
    pub new_decode_ids: Vec<Vec<i64>>,
    pub read_offsets: Vec<usize>,
    pub skip_special_tokens: Vec<bool>,
    pub spaces_between_special_tokens: Vec<bool>,
    pub no_stop_trim: Vec<bool>,
}

/// Borrow-extract the detokenization fields from a `BatchTokenIDOutput` map.
pub fn extract_batch_input(map: &[(Value, Value)]) -> BatchInputFields {
    let rids = map_get(map, "rids").map(v_string_array).unwrap_or_default();
    let bs = rids.len();

    let finished_reasons = map_get(map, "finished_reasons")
        .map(|v| match v {
            Value::Array(arr) => arr
                .iter()
                .map(|item| match item {
                    Value::Nil => None,
                    other => Some(other.clone()),
                })
                .collect(),
            _ => vec![None; bs],
        })
        .unwrap_or_else(|| vec![None; bs]);

    let decoded_texts = map_get(map, "decoded_texts")
        .map(v_string_array)
        .unwrap_or_else(|| vec![String::new(); bs]);

    let new_decode_ids = map_get(map, "decode_ids")
        .map(|v| match v {
            Value::Array(arr) => arr.iter().map(|item| v_i64_array(item)).collect(),
            _ => vec![vec![]; bs],
        })
        .unwrap_or_else(|| vec![vec![]; bs]);

    let read_offsets = map_get(map, "read_offsets")
        .map(|v| v_i64_array(v).into_iter().map(|x| x as usize).collect())
        .unwrap_or_else(|| vec![0usize; bs]);

    let skip_special_tokens = map_get(map, "skip_special_tokens")
        .map(v_bool_array)
        .unwrap_or_else(|| vec![true; bs]);

    let spaces_between_special_tokens = map_get(map, "spaces_between_special_tokens")
        .map(v_bool_array)
        .unwrap_or_else(|| vec![true; bs]);

    let no_stop_trim = map_get(map, "no_stop_trim")
        .map(v_bool_array)
        .unwrap_or_else(|| vec![false; bs]);

    BatchInputFields {
        rids,
        finished_reasons,
        decoded_texts,
        new_decode_ids,
        read_offsets,
        skip_special_tokens,
        spaces_between_special_tokens,
        no_stop_trim,
    }
}

/// Build the `BatchStrOutput` map from the original message map and computed output strings.
/// Consumes `map`; applies type-tag rename, field removal, routed_experts encoding, etc.
pub fn build_batch_str_output(map: Vec<(Value, Value)>, output_strs: Vec<String>) -> Value {
    const SKIP: &[&str] = &[
        "type",
        "decoded_texts",
        "decode_ids",
        "read_offsets",
        "skip_special_tokens",
        "spaces_between_special_tokens",
        "no_stop_trim",
    ];

    let mut out: Vec<(Value, Value)> = Vec::with_capacity(map.len() + 2);
    out.push((
        Value::String("type".into()),
        Value::String("BatchStrOutput".into()),
    ));

    for (k, v) in map {
        let key_str = k.as_str().unwrap_or("");
        if SKIP.contains(&key_str) {
            continue;
        }
        if key_str == "routed_experts" {
            out.push((k, encode_routed_experts(v)));
            continue;
        }
        if key_str == "placeholder_tokens_idx" || key_str == "placeholder_tokens_val" {
            out.push((k, Value::Nil));
            continue;
        }
        out.push((k, v));
    }

    let strs_val = Value::Array(
        output_strs
            .into_iter()
            .map(|s| Value::String(rmpv::Utf8String::from(s)))
            .collect(),
    );
    out.push((Value::String("output_strs".into()), strs_val));

    Value::Map(out)
}

// ─────────────────────── routed_experts encoding ────────────────────────────

pub fn encode_routed_experts(v: Value) -> Value {
    match v {
        Value::Nil => Value::Nil,
        Value::Array(items) => Value::Array(
            items
                .into_iter()
                .map(|item| match item {
                    Value::Nil => Value::Nil,
                    Value::Array(ref triple) if triple.len() == 3 => match &triple[2] {
                        Value::Binary(bytes) => {
                            let encoded = B64.encode(bytes);
                            Value::String(rmpv::Utf8String::from(encoded))
                        }
                        _ => Value::Nil,
                    },
                    _ => Value::Nil,
                })
                .collect(),
        ),
        _ => Value::Nil,
    }
}

// ─────────────────────── msgpack encode / decode ────────────────────────────

pub fn msgpack_decode(data: &[u8]) -> Result<Value, rmpv::decode::Error> {
    let mut cursor = std::io::Cursor::new(data);
    rmpv::decode::read_value(&mut cursor)
}

pub fn msgpack_encode(val: &Value) -> Vec<u8> {
    let mut buf = Vec::new();
    rmpv::encode::write_value(&mut buf, val).expect("Failed to encode msgpack");
    buf
}
