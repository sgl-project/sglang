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
fn map_get<'a>(map: &'a [(Value, Value)], key: &str) -> Option<&'a Value> {
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
        // all but the last character
        let byte_len: usize = chars[..chars.len() - 1].iter().map(|c| c.len_utf8()).sum();
        return &text[..byte_len];
    }
    // Up to and including the last space
    match text.rfind(' ') {
        Some(pos) => &text[..pos + 1],
        None => "",
    }
}

// ─────────────────────── stop-sequence trimming ──────────────────────────────

/// Trim a matched stop *token* from the end of an id list (used before decode).
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

/// Trim a matched stop *string* from the end of a decoded string.
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

/// Batch-decode a list of id sequences, grouping by (skip_special, spaces_between)
/// to minimise tokenizer calls.
fn grouped_batch_decode(
    tokenizer: &Tokenizer,
    ids_list: &[Vec<i64>],
    skip_list: &[bool],
    _space_list: &[bool], // spaces_between_special_tokens – not supported by the Rust tokenizers crate
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

    // Fast path: all requests share the same settings
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

    // Group indices by skip_special_tokens value
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

// ─────────────────────── main processing function ───────────────────────────

pub struct Config {
    pub disable_batch_decode: bool,
    pub is_gpt_oss: bool,
    pub max_states: usize,
}

/// Core incremental-decoding logic. Returns the `output_strs` vector.
pub fn compute_output_strs(
    rids: &[String],
    finished_reasons: &[Option<Value>],
    decoded_texts: &[String],
    new_decode_ids: &[Vec<i64>],
    read_offsets: &[usize],
    skip_special_tokens: &[bool],
    spaces_between: &[bool],
    no_stop_trim: &[bool],
    state: &mut LimitedCapacityDict,
    tokenizer: &Tokenizer,
    config: &Config,
) -> Vec<String> {
    let bs = rids.len();

    // ── Phase 1: update / initialise per-request decode status ──────────────
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
        let trimmed_read = trim_matched_stop_ids(raw_read, fr, no_stop_trim[i], config.is_gpt_oss);
        read_ids.push(trimmed_read);
        surr_ids.push(s.decode_ids[s.surr_offset..s.read_offset].to_vec());
    }

    // ── Phase 2: batch decode ────────────────────────────────────────────────
    let surr_texts = grouped_batch_decode(
        tokenizer,
        &surr_ids,
        skip_special_tokens,
        spaces_between,
        config.disable_batch_decode,
    );
    let read_texts = grouped_batch_decode(
        tokenizer,
        &read_ids,
        skip_special_tokens,
        spaces_between,
        config.disable_batch_decode,
    );

    // ── Phase 3: incremental update + final output ───────────────────────────
    let mut output_strs = Vec::with_capacity(bs);
    let capacity = state.capacity();
    for i in 0..bs {
        let rid = &rids[i];
        let s = state.get_mut(rid).unwrap_or_else(|| {
            panic!(
                "Decode status not found for request {rid}. \
                 Increase SGLANG_DETOKENIZER_MAX_STATES (current: {capacity})."
            )
        });

        let new_text = &read_texts[i][surr_texts[i].len()..];
        let fr = finished_reasons[i].as_ref();

        let new_text: String = if fr.is_none() {
            // Streaming chunk
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

        // Apply stop-string trim to the full accumulated text
        let full = format!("{}{}", s.decoded_text, new_text);
        let output_str = trim_matched_stop_str(&full, fr, no_stop_trim[i]);

        let incremental = output_str[s.sent_offset..].to_string();
        s.sent_offset = output_str.len();
        output_strs.push(incremental);

        // Clean up finished requests
        if fr.is_some() {
            state.remove(rid);
        }
    }

    output_strs
}

// ─────────────────────── routed_experts encoding ────────────────────────────

/// Base64-encode each per-request tensor (encoded by Python as [shape, dtype, bytes]).
/// Returns the same structure but with binary data replaced by a base64 string.
pub fn encode_routed_experts(v: Value) -> Value {
    match v {
        Value::Nil => Value::Nil,
        Value::Array(items) => Value::Array(
            items
                .into_iter()
                .map(|item| match item {
                    Value::Nil => Value::Nil,
                    Value::Array(ref triple) if triple.len() == 3 => {
                        // triple is [shape, dtype_str, binary_bytes]
                        match &triple[2] {
                            Value::Binary(bytes) => {
                                let encoded = B64.encode(bytes);
                                Value::String(rmpv::Utf8String::from(encoded))
                            }
                            _ => Value::Nil,
                        }
                    }
                    _ => Value::Nil,
                })
                .collect(),
        ),
        _ => Value::Nil,
    }
}

// ─────────────────────── message transformation ──────────────────────────────

/// Transform a `BatchTokenIDOutput` msgpack Value into a `BatchStrOutput` Value.
///
/// Fields consumed (not forwarded): decoded_texts, decode_ids, read_offsets,
/// skip_special_tokens, spaces_between_special_tokens, no_stop_trim.
/// Fields added: output_strs.
/// Fields transformed: type tag → "BatchStrOutput", routed_experts → base64.
/// Fields zeroed: placeholder_tokens_idx, placeholder_tokens_val.
pub fn transform_batch_token_id(
    val: Value,
    state: &mut LimitedCapacityDict,
    tokenizer: Option<&Tokenizer>,
    config: &Config,
) -> Value {
    let map: Vec<(Value, Value)> = match val {
        Value::Map(m) => m,
        other => return other, // shouldn't happen
    };

    // ── Extract fields needed for detokenization ─────────────────────────────
    let rids: Vec<String> = map_get(&map, "rids")
        .map(v_string_array)
        .unwrap_or_default();
    let bs = rids.len();

    let finished_reasons: Vec<Option<Value>> = map_get(&map, "finished_reasons")
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

    let decoded_texts: Vec<String> = map_get(&map, "decoded_texts")
        .map(v_string_array)
        .unwrap_or_else(|| vec![String::new(); bs]);

    let new_decode_ids: Vec<Vec<i64>> = map_get(&map, "decode_ids")
        .map(|v| match v {
            Value::Array(arr) => arr.iter().map(|item| v_i64_array(item)).collect(),
            _ => vec![vec![]; bs],
        })
        .unwrap_or_else(|| vec![vec![]; bs]);

    let read_offsets: Vec<usize> = map_get(&map, "read_offsets")
        .map(|v| v_i64_array(v).into_iter().map(|x| x as usize).collect())
        .unwrap_or_else(|| vec![0usize; bs]);

    let skip_special: Vec<bool> = map_get(&map, "skip_special_tokens")
        .map(v_bool_array)
        .unwrap_or_else(|| vec![true; bs]);

    let spaces_between: Vec<bool> = map_get(&map, "spaces_between_special_tokens")
        .map(v_bool_array)
        .unwrap_or_else(|| vec![true; bs]);

    let no_stop_trim: Vec<bool> = map_get(&map, "no_stop_trim")
        .map(v_bool_array)
        .unwrap_or_else(|| vec![false; bs]);

    // ── Compute output strings ────────────────────────────────────────────────
    let output_strs: Vec<Value> = if bs == 0 || tokenizer.is_none() {
        vec![Value::String(rmpv::Utf8String::from("")); bs]
    } else {
        let tok = tokenizer.unwrap();
        compute_output_strs(
            &rids,
            &finished_reasons,
            &decoded_texts,
            &new_decode_ids,
            &read_offsets,
            &skip_special,
            &spaces_between,
            &no_stop_trim,
            state,
            tok,
            config,
        )
        .into_iter()
        .map(|s| Value::String(rmpv::Utf8String::from(s)))
        .collect()
    };

    // ── Build output map ──────────────────────────────────────────────────────
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

    out.push((
        Value::String("output_strs".into()),
        Value::Array(output_strs),
    ));

    Value::Map(out)
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
