//! [`SamplingParams`] — the typed Rust port of Python `SamplingParams`
//! (python/sglang/srt/sampling/sampling_params.py): every field, plus its
//! `__post_init__` → `normalize` → `verify` pipeline (run in that order, as
//! `TokenizerManager._create_tokenized_object` does).
//!
//! The embedded Rust server replaces the Python `TokenizerManager`, which is the
//! only place those three run on the normal (zmq) path. Running them here, in the
//! ingress `Normalizing` FSM step, keeps the per-request CPU (notably the
//! stop-string work) off the scheduler's latency-critical loop. We set
//! `is_normalized=true` on the wire so the scheduler's `__post_init__` and
//! `normalize` early-return; its `verify` is likewise skipped (we did it here).
//!
//! KEEP IN SYNC with `sampling_params.py`: the field list, defaults and ranges
//! below mirror that file, and the struct is serialized by field name into the
//! `TokenizedGenerateReqInput` header, so a renamed/added Python field must be
//! mirrored here (an unknown key would be silently dropped by msgspec).
//!
//! Two deliberate deviations, both safe over-estimates or stricter:
//!   * `stop_str_max_len` is the stop string's **UTF-8 byte length** — a provably
//!     safe over-estimate of its token length (a token spans ≥ 1 byte, so
//!     `bytes ≥ tokens`; `chars` is *not* a bound — one char can be several
//!     tokens, e.g. `𓀀` → 3). The scheduler uses it only as a match-window
//!     *size* (capped at the output length), so an over-estimate matches the same
//!     stops — only an under-estimate misses. Python encodes each stop with the
//!     tokenizer for the exact count; the byte bound avoids needing it here.
//!   * `n > 1` (parallel sampling) is rejected — the rust egress maps one rid to
//!     one response, so every sample past the first would be dropped.

use std::collections::BTreeMap;
use std::fmt;

use serde::de::value::{MapAccessDeserializer, SeqAccessDeserializer};
use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};

use super::OneOrMany;
use crate::error::Error;

/// `_SAMPLING_EPS` — temperatures in `[0, eps)` mean greedy decoding.
const SAMPLING_EPS: f64 = 1e-6;
/// `TOP_K_ALL = 1 << 30` — `top_k` sentinel for "consider the whole vocabulary".
const TOP_K_ALL: i64 = 1 << 30;
/// `MAX_LEN` from Python's `get_max_seq_length`: the bound for an *unbounded* stop
/// regex (`\d+`, `.*`, …) or one we can't statically size — the scheduler then
/// scans the whole output tail. A *bounded* regex gets its finite length instead
/// (see [`regex_max_seq_length`]); assigning this to every regex made the scheduler
/// re-scan the full accumulated output every token (O(T²)).
const STOP_REGEX_MAX_LEN: i64 = 1 << 30;

/// One module per field default, each exposing the two hooks serde needs under
/// one name: `default` (key absent) and `deserialize` (key present — including
/// an explicit `null`, which Python's `__post_init__` maps back to the default:
/// "callers can pass null without crashing verify"). They cannot be one function
/// — serde calls `default()` with no arguments and `deserialize(deserializer)` —
/// but `deserialize` defers to `default()`, so the value is written once.
macro_rules! defaulted {
    ($($name:ident: $ty:ty = $value:expr;)*) => {$(
        mod $name {
            use serde::{Deserialize, Deserializer};

            pub(super) fn default() -> $ty { $value }

            pub(super) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<$ty, D::Error> {
                Ok(Option::<$ty>::deserialize(d)?.unwrap_or_else(default))
            }
        }
    )*};
}

defaulted! {
    f64_one: f64 = 1.0;
    f64_zero: f64 = 0.0;
    i64_top_k_all: i64 = super::TOP_K_ALL;
    i64_zero: i64 = 0;
    i64_one: i64 = 1;
    bool_false: bool = false;
    bool_true: bool = true;
}

/// `max_new_tokens` is `Optional[int] = 128`: an *absent* key means 128, but an
/// explicit `null` means None (no limit) — so it keeps its `Option` rather than
/// going through [`defaulted`].
fn max_new_tokens_default() -> Option<i64> {
    Some(128)
}

/// The sampling parameters of one `/generate` request. Deserialized from the
/// client's `sampling_params` object (unknown keys are a 400, mirroring Python's
/// `SamplingParams(**kwargs)` TypeError) and serialized by field name into the
/// scheduler header once [`normalize`](Self::normalize) has run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingParams {
    // --- API parameters (set by callers) ---
    #[serde(default = "max_new_tokens_default")]
    pub max_new_tokens: Option<i64>,
    /// API input alias, copied to `stop_strs` then cleared by `normalize`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<OneOrMany<String>>,
    /// Python `Optional[Set[int]]`. A `null` *element* is a 400 here where Python
    /// filters it out — a typed list can't hold one, and it is malformed input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_token_ids: Option<Vec<i64>>,
    /// API input alias, copied to `stop_regex_strs` then cleared by `normalize`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_regex: Option<OneOrMany<String>>,
    #[serde(
        default = "f64_one::default",
        deserialize_with = "f64_one::deserialize"
    )]
    pub temperature: f64,
    #[serde(
        default = "f64_one::default",
        deserialize_with = "f64_one::deserialize"
    )]
    pub top_p: f64,
    #[serde(
        default = "i64_top_k_all::default",
        deserialize_with = "i64_top_k_all::deserialize"
    )]
    pub top_k: i64,
    #[serde(
        default = "f64_zero::default",
        deserialize_with = "f64_zero::deserialize"
    )]
    pub min_p: f64,
    #[serde(
        default = "f64_zero::default",
        deserialize_with = "f64_zero::deserialize"
    )]
    pub frequency_penalty: f64,
    #[serde(
        default = "f64_zero::default",
        deserialize_with = "f64_zero::deserialize"
    )]
    pub presence_penalty: f64,
    #[serde(
        default = "f64_one::default",
        deserialize_with = "f64_one::deserialize"
    )]
    pub repetition_penalty: f64,
    #[serde(
        default = "i64_zero::default",
        deserialize_with = "i64_zero::deserialize"
    )]
    pub min_new_tokens: i64,
    #[serde(
        default = "i64_one::default",
        deserialize_with = "i64_one::deserialize"
    )]
    pub n: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regex: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ebnf: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structural_tag: Option<String>,
    #[serde(
        default = "bool_false::default",
        deserialize_with = "bool_false::deserialize"
    )]
    pub ignore_eos: bool,
    #[serde(
        default = "bool_true::default",
        deserialize_with = "bool_true::deserialize"
    )]
    pub skip_special_tokens: bool,
    #[serde(
        default = "bool_true::default",
        deserialize_with = "bool_true::deserialize"
    )]
    pub spaces_between_special_tokens: bool,
    #[serde(
        default = "bool_false::default",
        deserialize_with = "bool_false::deserialize"
    )]
    pub no_stop_trim: bool,
    /// Opaque JSON object forwarded to a custom logit processor. Python types it
    /// as `Dict[str, JsonScalar | list | dict]`; it is never inspected here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom_params: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_interval: Option<i64>,
    /// Token id (as a string key, matching Python) → bias. Keys are vocab-bounded
    /// by [`verify`](Self::verify).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<BTreeMap<String, f64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_seed: Option<i64>,

    // --- Internal fields (populated by the pipeline below, not API-facing) ---
    //
    // All `skip_deserializing`: they are outputs of `normalize`, and a client that
    // could set them would be setting the pipeline's own state. `is_normalized` is
    // the dangerous one — `{"is_normalized": true, "temperature": 0.0}` makes
    // `post_init` early-return, so the greedy mapping never runs and temperature 0
    // reaches the scheduler's `logits.div_()`; `stop` would likewise be dropped
    // without ever reaching `stop_strs`. They still SERIALIZE: the scheduler needs
    // them on the wire.
    /// From `stop`; a list after `normalize` (Python widens str → [str] there).
    #[serde(skip_deserializing)]
    pub stop_strs: Vec<String>,
    /// From `stop_regex`.
    #[serde(skip_deserializing)]
    pub stop_regex_strs: Vec<String>,
    #[serde(skip_deserializing)]
    pub stop_str_max_len: i64,
    #[serde(skip_deserializing)]
    pub stop_regex_max_len: i64,
    /// Set by `normalize`; tells the scheduler its own pass can early-return.
    #[serde(skip_deserializing)]
    pub is_normalized: bool,
}

/// The `/generate` body's `sampling_params`: one object (broadcast to every
/// prompt) or a list of them (one per prompt), fanned out by `GenerateBody::into_requests`.
///
/// Hand-written `Deserialize` rather than `#[serde(untagged)]`: untagged buffers
/// the input and, on failure, reports only "data did not match any variant" —
/// losing the field-level message ("unknown field `temperature`, expected one of
/// …") that makes a typo actionable. Object-vs-list is unambiguous here, so a
/// single `deserialize_any` dispatch keeps the inner error verbatim.
#[derive(Debug, Clone, PartialEq)]
pub enum SamplingParamsInput {
    /// Boxed: `SamplingParams` is ~440 bytes, so an inline variant would make
    /// every `GenerateBody` that big regardless of which form arrived.
    One(Box<SamplingParams>),
    Many(Vec<SamplingParams>),
}

impl<'de> Deserialize<'de> for SamplingParamsInput {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct InputVisitor;

        impl<'de> Visitor<'de> for InputVisitor {
            type Value = SamplingParamsInput;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a sampling_params object, or a list of them (one per prompt)")
            }

            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<Self::Value, A::Error> {
                SamplingParams::deserialize(MapAccessDeserializer::new(map))
                    .map(|p| SamplingParamsInput::One(Box::new(p)))
            }

            fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<Self::Value, A::Error> {
                Vec::deserialize(SeqAccessDeserializer::new(seq)).map(SamplingParamsInput::Many)
            }
        }

        deserializer.deserialize_any(InputVisitor)
    }
}

impl Default for SamplingParams {
    fn default() -> Self {
        // Each field reads the same `default()` the serde attribute above names,
        // so the values still live in one place — without re-parsing `{}` on
        // every call (once per prompt of every params-less `/generate`). The
        // struct literal makes completeness a compile error.
        Self {
            max_new_tokens: max_new_tokens_default(),
            stop: None,
            stop_token_ids: None,
            stop_regex: None,
            temperature: f64_one::default(),
            top_p: f64_one::default(),
            top_k: i64_top_k_all::default(),
            min_p: f64_zero::default(),
            frequency_penalty: f64_zero::default(),
            presence_penalty: f64_zero::default(),
            repetition_penalty: f64_one::default(),
            min_new_tokens: i64_zero::default(),
            n: i64_one::default(),
            json_schema: None,
            regex: None,
            ebnf: None,
            structural_tag: None,
            ignore_eos: bool_false::default(),
            skip_special_tokens: bool_true::default(),
            spaces_between_special_tokens: bool_true::default(),
            no_stop_trim: bool_false::default(),
            custom_params: None,
            stream_interval: None,
            logit_bias: None,
            sampling_seed: None,
            stop_strs: Vec::new(),
            stop_regex_strs: Vec::new(),
            stop_str_max_len: 0,
            stop_regex_max_len: 0,
            is_normalized: false,
        }
    }
}

impl SamplingParams {
    /// `__post_init__` → `normalize` → `verify`, the order
    /// `TokenizerManager._create_tokenized_object` runs them in. `Err` is a
    /// request-local 400. `skip_tokenizer_init` stands in for Python's
    /// `tokenizer is None`; `vocab_size` bounds `logit_bias` keys.
    pub fn normalize(
        &mut self,
        skip_tokenizer_init: bool,
        vocab_size: Option<u64>,
    ) -> Result<(), Error> {
        self.post_init();
        self.normalize_stops(skip_tokenizer_init)?;
        self.verify(vocab_size)
    }

    /// Python `__post_init__` (minus the null-to-default coercions, which the
    /// null-tolerant deserializers above already did): copy the API aliases into
    /// the internal fields and apply the greedy / `top_k` special cases.
    fn post_init(&mut self) {
        // Python's `__post_init__` guard. Without it a second `normalize` reads
        // the aliases `normalize_stops` already cleared and silently wipes
        // `stop_strs`/`stop_regex_strs` to empty — the request would stop
        // matching its stop strings.
        if self.is_normalized {
            return;
        }
        // Moved out, not cloned: `normalize_stops` clears both aliases anyway.
        self.stop_strs = take_one_or_many(self.stop.take());
        self.stop_regex_strs = take_one_or_many(self.stop_regex.take());
        // Python drops null entries and maps an empty set to None.
        if self.stop_token_ids.as_ref().is_some_and(|v| v.is_empty()) {
            self.stop_token_ids = None;
        }
        if (0.0..SAMPLING_EPS).contains(&self.temperature) {
            // Greedy: temperature ~0 → temperature=1.0, top_k=1.
            self.temperature = 1.0;
            self.top_k = 1;
        }
        if self.top_k == -1 {
            self.top_k = TOP_K_ALL; // -1 disables top_k → whole vocabulary
        }
    }

    /// Python `normalize(tokenizer)`: size the stop match windows, reject
    /// tokenizer-dependent features when there is no tokenizer, and clear the API
    /// aliases so they don't ride the wire twice.
    fn normalize_stops(&mut self, skip_tokenizer_init: bool) -> Result<(), Error> {
        // Match window: UTF-8 byte length is a safe upper bound on the token count.
        self.stop_str_max_len = self
            .stop_strs
            .iter()
            .map(|s| s.len() as i64)
            .max()
            .unwrap_or(0);
        // The full-scan sentinel for any stop_regex, with no attempt to derive a
        // tighter bound. `recompute_stop_regex_max_len` (managers/utils.py) re-derives
        // it with `sre_parse` + `re.compile` on the way in, so anything computed here
        // is overwritten before the scheduler sees it — and the scheduler runs these
        // through Python's `re`, so only Python's parser can judge them. Matching
        // dialects from here bought a bound nobody reads and cost two rounds of
        // scheduler-killing mismatches; the sentinel is always a safe over-estimate.
        let stop_regex_max_len = if self.stop_regex_strs.is_empty() {
            0
        } else {
            STOP_REGEX_MAX_LEN
        };
        self.stop_regex_max_len = stop_regex_max_len;

        // Python `raise_if_tokenizer_required`: these need `tokenizer.decode` /
        // `eos_token_id`, which `skip_tokenizer_init` does not have.
        if skip_tokenizer_init {
            if !self.stop_strs.is_empty() {
                return Err(bad(
                    "stop is unavailable when skip_tokenizer_init=True (requires a \
                     tokenizer to decode tokens to text for matching)"
                        .into(),
                ));
            }
            if !self.stop_regex_strs.is_empty() {
                return Err(bad(
                    "stop_regex is unavailable when skip_tokenizer_init=True (requires a \
                     tokenizer to decode tokens to text for matching)"
                        .into(),
                ));
            }
            if self.min_new_tokens > 0 {
                return Err(bad(format!(
                    "min_new_tokens={} is unavailable when skip_tokenizer_init=True \
                     (requires a tokenizer for eos_token_id)",
                    self.min_new_tokens
                )));
            }
        }

        self.stop = None;
        self.stop_regex = None;
        self.is_normalized = true;
        Ok(())
    }

    /// Python `verify(vocab_size)` — the same ranges, messages and mutual
    /// exclusions, plus the rust-server `n == 1` restriction.
    fn verify(&self, vocab_size: Option<u64>) -> Result<(), Error> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(bad(format!(
                "temperature must be a non-negative finite number, got {}",
                self.temperature
            )));
        }
        if !(self.top_p > 0.0 && self.top_p <= 1.0) {
            return Err(bad(format!("top_p must be in (0, 1], got {}", self.top_p)));
        }
        if !(0.0..=1.0).contains(&self.min_p) {
            return Err(bad(format!("min_p must be in [0, 1], got {}", self.min_p)));
        }
        if self.top_k < 1 {
            return Err(bad(format!(
                "top_k must be -1 (disable) or at least 1, got {}",
                self.top_k
            )));
        }
        if !(-2.0..=2.0).contains(&self.frequency_penalty) {
            return Err(bad(format!(
                "frequency_penalty must be in [-2, 2], got {}",
                self.frequency_penalty
            )));
        }
        if !(-2.0..=2.0).contains(&self.presence_penalty) {
            return Err(bad(format!(
                "presence_penalty must be in [-2, 2], got {}",
                self.presence_penalty
            )));
        }
        if !(self.repetition_penalty > 0.0 && self.repetition_penalty <= 2.0) {
            return Err(bad(format!(
                "repetition_penalty must be in (0, 2], got {}",
                self.repetition_penalty
            )));
        }
        if self.min_new_tokens < 0 {
            return Err(bad(format!(
                "min_new_tokens must be non-negative, got {}",
                self.min_new_tokens
            )));
        }
        // `None` = no limit, so the max_new_tokens checks only apply when set.
        if let Some(max_new_tokens) = self.max_new_tokens {
            if max_new_tokens < 0 {
                return Err(bad(format!(
                    "max_new_tokens must be at least 0, got {max_new_tokens}"
                )));
            }
            if self.min_new_tokens > max_new_tokens {
                return Err(bad(format!(
                    "min_new_tokens must be in [0, max_new_tokens({max_new_tokens})], got {}",
                    self.min_new_tokens
                )));
            }
        }
        // A non-numeric bias key raises in the scheduler's `int(key)`, and an
        // out-of-vocabulary one would index past the logits row, so both are
        // rejected here (Python `verify` does the same, in that order). Only the
        // *range* check needs the vocab size (`None` = unknown, skip it); the key
        // format is checked either way, since `int(key)` runs regardless.
        if let Some(logit_bias) = &self.logit_bias {
            for key in logit_bias.keys() {
                let token_id: u64 = key
                    .parse()
                    .map_err(|_| bad(format!("logit_bias keys must be token ids, got {key:?}")))?;
                if let Some(vocab_size) = vocab_size
                    && token_id >= vocab_size
                {
                    return Err(bad(format!(
                        "logit_bias must have keys in [0, {}], got {token_id}",
                        vocab_size - 1
                    )));
                }
            }
        }
        // Grammars are mutually exclusive.
        let grammars = [&self.json_schema, &self.regex, &self.ebnf]
            .iter()
            .filter(|g| g.is_some())
            .count();
        if grammars > 1 {
            return Err(bad(
                "Only one of regex, json_schema, or ebnf can be set".into()
            ));
        }
        // Not a Python restriction: the rust egress maps one rid to one response,
        // so parallel sampling would drop all but the first sample. This is the
        // only place it is rejected — `n` lives in `sampling_params`, where
        // Python reads it, and the `/generate` body has no `n` of its own.
        if self.n != 1 {
            return Err(bad(format!(
                "n must be 1 (parallel sampling is not supported), got {}",
                self.n
            )));
        }
        Ok(())
    }
}

fn bad(msg: String) -> Error {
    Error::Validation(msg)
}

/// Widen a `str | [str]` API alias into the list form the internal field holds.
fn take_one_or_many(v: Option<OneOrMany<String>>) -> Vec<String> {
    match v {
        None => Vec::new(),
        Some(OneOrMany::One(s)) => vec![s],
        Some(OneOrMany::Many(v)) => v,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse client JSON exactly as `/generate` does, then run the full pipeline.
    fn norm(json: &str) -> SamplingParams {
        let mut sp: SamplingParams = serde_json::from_str(json).expect("parses");
        sp.normalize(false, None).expect("normalizes");
        sp
    }

    fn norm_err(json: &str) -> Error {
        let mut sp: SamplingParams = serde_json::from_str(json).expect("parses");
        sp.normalize(false, None).expect_err("must reject")
    }

    /// The wire shape the scheduler decodes: a map of field names → values — the
    /// same `serde_json::Value` the header encoder hands to msgpack.
    fn wire(sp: &SamplingParams) -> serde_json::Value {
        serde_json::to_value(sp).expect("serializes")
    }

    fn get(v: &serde_json::Value, key: &str) -> Option<serde_json::Value> {
        v.get(key).cloned()
    }

    #[test]
    fn greedy_sets_temp_one_topk_one() {
        let sp = norm(r#"{"temperature": 0.0}"#);
        assert_eq!(sp.temperature, 1.0);
        assert_eq!(sp.top_k, 1);
        assert!(sp.is_normalized);
    }

    #[test]
    fn topk_minus_one_becomes_all() {
        assert_eq!(norm(r#"{"temperature": 0.7}"#).top_k, TOP_K_ALL);
        assert_eq!(
            norm(r#"{"top_k": -1, "temperature": 0.7}"#).top_k,
            TOP_K_ALL
        );
    }

    #[test]
    fn stop_list_and_max_len_by_bytes() {
        let sp = norm(r#"{"stop": ["Question:", "\n\n"]}"#);
        assert_eq!(sp.stop_strs.len(), 2);
        assert_eq!(sp.stop_str_max_len, 9); // "Question:" (ASCII)
        // The API alias is cleared, so it never rides the wire twice.
        assert!(sp.stop.is_none());
    }

    /// A multi-byte stop char must use its byte length as the window bound: `𓀀`
    /// is 1 char but 4 UTF-8 bytes (and 3 tokens on Qwen3). Char count (1) would
    /// under-size the tail and miss the stop; byte count (4) ≥ the token span.
    #[test]
    fn stop_str_max_len_uses_bytes_not_chars() {
        let sp = norm(r#"{"stop": "𓀀"}"#);
        assert_eq!("𓀀".chars().count(), 1);
        assert_eq!("𓀀".len(), 4);
        assert_eq!(sp.stop_strs, vec!["𓀀".to_string()]); // scalar widened to a list
        assert_eq!(sp.stop_str_max_len, 4);
    }

    #[test]
    fn no_stop_yields_empty_list_zero_len() {
        let sp = norm(r#"{"temperature": 0.0}"#);
        assert!(sp.stop_strs.is_empty());
        assert_eq!(sp.stop_str_max_len, 0);
    }

    /// The wire map is what the scheduler's msgspec decoder reads by field name:
    /// the normalized values must be present under the Python names, and the
    /// `is_normalized` flag must be set so its own pass early-returns.
    #[test]
    fn wire_map_carries_python_field_names() {
        let sp = norm(r#"{"temperature": 0.7, "max_new_tokens": 64, "ignore_eos": true}"#);
        let w = wire(&sp);
        assert_eq!(get(&w, "temperature").unwrap().as_f64(), Some(0.7));
        assert_eq!(get(&w, "max_new_tokens").unwrap().as_i64(), Some(64));
        assert_eq!(get(&w, "ignore_eos").unwrap().as_bool(), Some(true));
        assert_eq!(get(&w, "top_k").unwrap().as_i64(), Some(TOP_K_ALL));
        assert_eq!(get(&w, "is_normalized").unwrap().as_bool(), Some(true));
        assert_eq!(get(&w, "stop_str_max_len").unwrap().as_i64(), Some(0));
        // Unset optionals are omitted, so msgspec applies the Python defaults.
        assert_eq!(get(&w, "regex"), None);
        assert_eq!(get(&w, "stop"), None);
    }

    /// `max_new_tokens` is the one field where absent and null differ: absent =
    /// 128 (the Python field default), explicit null = None (no limit).
    #[test]
    fn max_new_tokens_null_is_unlimited_absent_is_default() {
        assert_eq!(norm("{}").max_new_tokens, Some(128));
        assert_eq!(norm(r#"{"max_new_tokens": null}"#).max_new_tokens, None);
        // None = no limit, so a large min_new_tokens is not a range error.
        let sp = norm(r#"{"max_new_tokens": null, "min_new_tokens": 4096}"#);
        assert_eq!(sp.min_new_tokens, 4096);
    }

    /// The scheduler decodes this map with msgspec, which **silently drops**
    /// unknown keys — so a field renamed on one side only is invisible at runtime:
    /// the request succeeds with that sampling param quietly not applied. That is
    /// the drift this file's header warns about, so pin the key vocabulary; a
    /// rename or addition then has to be a deliberate edit here too.
    ///
    /// The two sets are separate because the `Option` fields are
    /// `skip_serializing_if`: absent from a default request, present once set.
    /// Both must stay in sync with `sampling_params.py` (30 fields there).
    #[test]
    fn serialized_key_set_is_pinned() {
        /// Always on the wire: every non-`Option` field.
        const ALWAYS: &[&str] = &[
            "frequency_penalty",
            "ignore_eos",
            "is_normalized",
            "max_new_tokens",
            "min_new_tokens",
            "min_p",
            "n",
            "no_stop_trim",
            "presence_penalty",
            "repetition_penalty",
            "skip_special_tokens",
            "spaces_between_special_tokens",
            "stop_regex_max_len",
            "stop_regex_strs",
            "stop_str_max_len",
            "stop_strs",
            "temperature",
            "top_k",
            "top_p",
        ];
        /// Emitted only when set (`skip_serializing_if = "Option::is_none"`).
        const OPTIONAL: &[&str] = &[
            "custom_params",
            "ebnf",
            "json_schema",
            "logit_bias",
            "regex",
            "sampling_seed",
            "stop",
            "stop_regex",
            "stop_token_ids",
            "stream_interval",
            "structural_tag",
        ];

        let keys = |sp: &SamplingParams| -> Vec<String> {
            let v = wire(sp);
            let mut k: Vec<String> = v.as_object().expect("a map").keys().cloned().collect();
            k.sort();
            k
        };

        assert_eq!(
            keys(&SamplingParams::default()),
            ALWAYS,
            "unset key set drifted"
        );

        // Every optional populated. Parsed, not normalized: `normalize` moves the
        // `stop`/`stop_regex` aliases into `stop_strs`/`stop_regex_strs` and clears
        // them, so a normalized request never carries all 30 at once.
        let full: SamplingParams = serde_json::from_str(
            r#"{
                "stop": "x", "stop_token_ids": [1], "stop_regex": "y",
                "json_schema": "{}", "regex": "a", "ebnf": "b", "structural_tag": "t",
                "custom_params": {"k": 1}, "stream_interval": 2,
                "logit_bias": {"3": 1.0}, "sampling_seed": 4
            }"#,
        )
        .expect("parses");
        let mut expected: Vec<&str> = ALWAYS.iter().chain(OPTIONAL).copied().collect();
        expected.sort();
        assert_eq!(keys(&full), expected, "populated key set drifted");
        assert_eq!(expected.len(), 30, "sampling_params.py declares 30 fields");
    }

    /// Each range check rejects, and rejects for the *stated* reason — the message
    /// must name the offending field, so a mis-ordered or copy-pasted check (e.g.
    /// `presence_penalty` guarded by the `frequency_penalty` bound) can't pass by
    /// merely erroring.
    #[test]
    fn verify_rejects_out_of_range() {
        for (json, want) in [
            (r#"{"top_p": 2.0}"#, "top_p"),
            (r#"{"top_k": 0, "temperature": 0.7}"#, "top_k"),
            (r#"{"min_p": 1.5}"#, "min_p"),
            (r#"{"frequency_penalty": 3.0}"#, "frequency_penalty"),
            (r#"{"presence_penalty": -3.0}"#, "presence_penalty"),
            (r#"{"repetition_penalty": 0.0}"#, "repetition_penalty"),
            (
                r#"{"max_new_tokens": 8, "min_new_tokens": 9}"#,
                "min_new_tokens",
            ),
            (r#"{"temperature": -0.1}"#, "temperature"),
            (r#"{"max_new_tokens": -1}"#, "max_new_tokens"),
            (r#"{"regex": "a", "ebnf": "b"}"#, "Only one of"),
            (r#"{"n": 2}"#, "n must be 1"),
        ] {
            let err = norm_err(json).to_string();
            assert!(
                err.contains(want),
                "{json} must be rejected for {want}: {err}"
            );
        }
    }

    /// The inclusive bounds must ACCEPT their endpoints. Only the rejecting side
    /// was covered, and far from the edge (`frequency_penalty: 3.0`), so flipping
    /// any `..=` to `..` — or `>= 1` to `> 1` — would 400 legitimate requests
    /// without failing a single test.
    #[test]
    fn verify_accepts_inclusive_boundaries() {
        for json in [
            r#"{"top_p": 1.0, "temperature": 0.7}"#,
            r#"{"min_p": 0.0, "temperature": 0.7}"#,
            r#"{"min_p": 1.0, "temperature": 0.7}"#,
            r#"{"top_k": 1, "temperature": 0.7}"#,
            r#"{"frequency_penalty": 2.0}"#,
            r#"{"frequency_penalty": -2.0}"#,
            r#"{"presence_penalty": 2.0}"#,
            r#"{"presence_penalty": -2.0}"#,
            r#"{"repetition_penalty": 2.0}"#,
            r#"{"max_new_tokens": 0}"#,
            r#"{"min_new_tokens": 0}"#,
            // min == max is in range: `[0, max_new_tokens]` is inclusive.
            r#"{"max_new_tokens": 8, "min_new_tokens": 8}"#,
            // Greedy: temperature 0 is the documented sentinel, not an under-run.
            r#"{"temperature": 0.0}"#,
            r#"{"n": 1}"#,
        ] {
            let mut sp: SamplingParams = serde_json::from_str(json).expect("parses");
            sp.normalize(false, None)
                .unwrap_or_else(|e| panic!("{json} is in range but was rejected: {e}"));
        }
    }

    /// And the first value past each endpoint is still rejected — the pair of
    /// tests brackets the boundary instead of testing one side of it.
    #[test]
    fn verify_rejects_just_past_the_boundaries() {
        for json in [
            r#"{"top_p": 0.0, "temperature": 0.7}"#, // exclusive lower bound
            r#"{"repetition_penalty": 0.0}"#,        // exclusive lower bound
            r#"{"top_k": 0, "temperature": 0.7}"#,
            r#"{"min_p": 1.0000001, "temperature": 0.7}"#,
            r#"{"frequency_penalty": 2.0000001}"#,
            r#"{"presence_penalty": -2.0000001}"#,
            r#"{"repetition_penalty": 2.0000001}"#,
            r#"{"max_new_tokens": -1}"#,
            r#"{"min_new_tokens": -1}"#,
            r#"{"max_new_tokens": 8, "min_new_tokens": 9}"#,
        ] {
            let mut sp: SamplingParams = serde_json::from_str(json).expect("parses");
            assert!(
                sp.normalize(false, None).is_err(),
                "{json} is out of range but was accepted"
            );
        }
    }

    /// A wrong JSON type for a numeric field is rejected at parse time — it must
    /// NOT silently fall back to the default (`temperature: "bad"` has different
    /// semantics than an unset temperature).
    #[test]
    fn wrong_typed_field_is_rejected() {
        for json in [
            r#"{"temperature": "bad"}"#,
            r#"{"top_k": "bad"}"#,
            r#"{"max_new_tokens": "bad"}"#,
            r#"{"stop": 3}"#,
        ] {
            assert!(
                serde_json::from_str::<SamplingParams>(json).is_err(),
                "{json} must not parse"
            );
        }
    }

    /// An unknown key is a 400, mirroring Python's `SamplingParams(**kwargs)`
    /// TypeError — a typo must not be silently ignored. (The bogus key is
    /// deliberately not a near-miss of a real field: an editor spell-checker
    /// kept "correcting" a misspelling here into a valid name, which silently
    /// turned this assertion into a tautology.)
    #[test]
    fn unknown_field_is_rejected() {
        assert!(serde_json::from_str::<SamplingParams>(r#"{"zzz_not_a_field": 1}"#).is_err());
        // ...while every declared field still parses.
        assert!(serde_json::from_str::<SamplingParams>(r#"{"temperature": 0.7}"#).is_ok());
    }

    /// A present-but-null non-optional field keeps the default (Python's
    /// `x if x is not None`) — null is absent, not a wrong type.
    #[test]
    fn null_field_keeps_default() {
        let sp = norm(r#"{"temperature": null, "top_k": null, "skip_special_tokens": null}"#);
        assert_eq!(sp.temperature, 1.0);
        assert_eq!(sp.top_k, TOP_K_ALL);
        assert!(sp.skip_special_tokens);
    }

    /// `normalize` must be idempotent: `post_init` reads the API aliases, which
    /// `normalize_stops` clears, so without Python's `if self.is_normalized:
    /// return` guard a second call wipes `stop_strs` and drops the stop bound to
    /// zero — silently, leaving a request that never stops.
    #[test]
    fn normalize_is_idempotent() {
        let mut once = norm(r#"{"stop": ["END", "STOP"], "stop_regex": "\\d{3}"}"#);
        let twice = {
            let mut p = once.clone();
            p.normalize(false, None).expect("second normalize");
            p
        };
        assert_eq!(once, twice, "a second normalize must change nothing");
        assert_eq!(twice.stop_strs, vec!["END".to_string(), "STOP".to_string()]);
        assert_eq!(twice.stop_str_max_len, 4);
        assert_eq!(twice.stop_regex_max_len, STOP_REGEX_MAX_LEN);

        // Greedy handling must not re-fire either: temperature is 1.0 after the
        // first pass, which is not in the greedy window.
        once.normalize(false, None).unwrap();
        assert_eq!(once.top_k, twice.top_k);
    }

    /// `skip_tokenizer_init` has no tokenizer, so the text-matching stop features
    /// and `min_new_tokens` (needs eos_token_id) are 400s, not silent no-ops.
    /// Mirrors Python `raise_if_tokenizer_required`.
    #[test]
    fn tokenizer_dependent_features_rejected_without_tokenizer() {
        for json in [
            r#"{"stop": "END"}"#,
            r#"{"stop_regex": "\\d+"}"#,
            r#"{"min_new_tokens": 1}"#,
        ] {
            let mut sp: SamplingParams = serde_json::from_str(json).expect("parses");
            assert!(
                sp.normalize(true, None).is_err(),
                "{json} must be rejected under skip_tokenizer_init"
            );
        }
        // The same params are fine when a tokenizer is present.
        let mut sp: SamplingParams = serde_json::from_str(r#"{"stop": "END"}"#).unwrap();
        assert!(sp.normalize(false, None).is_ok());
    }

    /// `logit_bias` keys index the logits row, so an out-of-vocab id is a 400
    /// (Python `verify`'s vocab bound). Skipped when the vocab size is unknown.
    #[test]
    fn logit_bias_keys_are_vocab_bounded() {
        let mut sp: SamplingParams =
            serde_json::from_str(r#"{"logit_bias": {"1000": 1.0}}"#).unwrap();
        assert!(sp.clone().normalize(false, Some(1000)).is_err());
        assert!(sp.normalize(false, Some(1001)).is_ok());

        let mut sp: SamplingParams =
            serde_json::from_str(r#"{"logit_bias": {"999": -1.0}}"#).unwrap();
        assert!(sp.normalize(false, Some(1000)).is_ok());
    }

    /// The key *format* check must not hang off the vocab bound: the scheduler
    /// does `logit_bias[i, int(key)]` unconditionally, so a non-numeric (or
    /// negative, which would index from the end of the row) key has to be a 400
    /// even when the vocab size is unknown.
    #[test]
    fn logit_bias_keys_are_token_ids_without_vocab_size() {
        for json in [
            r#"{"logit_bias": {"abc": 1.0}}"#,
            r#"{"logit_bias": {"-1": 1.0}}"#,
            r#"{"logit_bias": {"1.5": 1.0}}"#,
            r#"{"logit_bias": {"": 1.0}}"#,
        ] {
            let _ = norm_err(json); // norm_err passes vocab_size = None
        }
        assert!(norm(r#"{"logit_bias": {"7": 1.0}}"#).logit_bias.is_some());
    }

    /// Rust no longer judges `stop_regex` at all: it stamps the full-scan sentinel
    /// and leaves both the bound and the verdict to Python, whose `re` is the engine
    /// that actually runs the pattern (`recompute_stop_regex_max_len` overwrites this
    /// value on the way in, and 400s what it cannot compile).
    ///
    /// So patterns `regex-syntax` used to reject — including ones Python accepts,
    /// like `\Z` — must pass through here, and even a malformed one is Python's call.
    #[test]
    fn stop_regex_gets_the_sentinel_and_no_verdict() {
        assert_eq!(
            norm(r#"{"stop_regex": "\\d{6}"}"#).stop_regex_max_len,
            STOP_REGEX_MAX_LEN
        );
        assert_eq!(
            norm(r#"{"stop_regex": "\\d+"}"#).stop_regex_max_len,
            STOP_REGEX_MAX_LEN
        );
        // No stop_regex → 0, so the scheduler skips the regex path entirely.
        assert_eq!(norm(r#"{"temperature": 0.7}"#).stop_regex_max_len, 0);

        for pattern in [
            r"(a)\1",    // backreference — regex-syntax rejects, `re` accepts
            r"(?=x)y",   // look-around — same
            r"\Z",       // python-only anchor spelling — same
            r"(?<=a*)b", // variable-width look-behind: parses, fails to compile
            "(",         // malformed even for `re` — still Python's 400 to give
        ] {
            let json = format!(
                r#"{{"stop_regex": {}}}"#,
                serde_json::to_string(pattern).unwrap()
            );
            let sp = norm(&json);
            assert_eq!(
                sp.stop_regex_max_len, STOP_REGEX_MAX_LEN,
                "{pattern} must pass through with the sentinel"
            );
        }
    }
}
