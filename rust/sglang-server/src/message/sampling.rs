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
        // Validate + bound every stop_regex here, before it can reach the
        // scheduler's `re.search` (see `stop_regex_bound`). A rejected pattern is a
        // 400 for this request; an accepted one carries a bound the scheduler uses
        // to size its match window.
        if self.stop_strs.len() > MAX_STOP_COUNT {
            return Err(bad(format!(
                "at most {MAX_STOP_COUNT} stop strings are allowed, got {}",
                self.stop_strs.len()
            )));
        }
        if self.stop_regex_strs.len() > MAX_STOP_REGEX_COUNT {
            return Err(bad(format!(
                "at most {MAX_STOP_REGEX_COUNT} stop_regex patterns are allowed, got {}",
                self.stop_regex_strs.len()
            )));
        }
        let mut stop_regex_max_len = 0;
        for pattern in &self.stop_regex_strs {
            if pattern.len() > MAX_STOP_REGEX_LEN {
                return Err(bad(format!(
                    "stop_regex is {} bytes, over the {MAX_STOP_REGEX_LEN}-byte limit",
                    pattern.len()
                )));
            }
            stop_regex_max_len = stop_regex_max_len.max(stop_regex_bound(pattern)?);
        }
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

/// Escapes that mean the same thing to `regex-syntax` and to Python's `re`.
///
/// An allowlist, not a blocklist. The blocklist version of this function is what
/// shipped `\p{L}` and `(?<n>a)` to a scheduler that could not compile them: every
/// escape either side adds lands in the gap by default. Here the default is
/// "reject", so a new escape is a 400 until someone checks both dialects.
/// Inline flags both dialects understand. Rust also has `R`/`U`, Python `a`/`L`;
/// each errors on the other's.
const PORTABLE_FLAGS: &[char] = &['i', 'm', 's', 'x', 'u', '-'];

/// Repetition counts at or above this are refused — as a PRODUCT down the nesting,
/// not per node: `(?:(?:a*){65535}){65535}` is 22 bytes, compiles fine, and costs
/// several GiB inside `re.search` on the decode hot path (`MemoryError`, which the
/// seatbelt does not catch). Well below that, `(?:){1048575}x` measured 428 ms per
/// decode step in GIL-holding C. A stop-string window needs nothing like this. `regex-syntax` parses counts as
/// `u32` and so accepts everything up to `u32::MAX`, while CPython's `MAXREPEAT`
/// sentinel IS `u32::MAX` (`a{4294967295}` → `OverflowError`) and a large count on a
/// nested group exhausts memory at compile time (`(?:a*){4294967294}` → several GB,
/// then `MemoryError`). Neither is an `re.error`, so neither is caught downstream.
/// A stop-string window this long is meaningless anyway.
const MAX_REPEAT_COUNT: u64 = 256;

/// Longest `stop_regex` accepted. A 1 MB literal pattern takes ~677 ms just to
/// compile, and that cost lands on the scheduler.
const MAX_STOP_REGEX_LEN: usize = 4096;

/// Most stop STRINGS accepted per request. The scheduler scans the decoded text
/// once per stop per decode step, so this is a per-step multiplier: 50k stops
/// measured 20.4 ms/step from a 586 KB body.
const MAX_STOP_COUNT: usize = 256;

/// Most `stop_regex` patterns accepted per request. Python's `re` cache holds 512
/// (`re._MAXCACHE`), so past that every pattern recompiles on every decode step.
const MAX_STOP_REGEX_COUNT: usize = 64;

const SHARED_ESCAPES: &[char] = &[
    'A', 'b', 'B', 'd', 'D', 's', 'S', 'w', 'W', 'a', 'f', 'n', 'r', 't', 'v',
    // `\xHH` (exactly two hex digits) is shared; only the braced `\x{…}` form is
    // Rust-only, and `check_escape` rejects that separately. Omitting `x` here
    // contradicted this list's own doc comment and 400ed `\x41`.
    'x',
];

/// Reject the constructs `regex-syntax` accepts but Python's `re` cannot compile.
///
/// Everything else in this module rests on one property: **anything Rust admits,
/// Python can compile.** The reverse is allowed to fail — rejecting a pattern
/// Python would have accepted (`\Z`, backreferences, look-around) costs a client
/// a 400, while admitting one it cannot compile costs the whole scheduler, since
/// `re.search` runs on the decode hot path where nothing catches it.
fn reject_python_incompatible(pattern: &str) -> Result<(), Error> {
    let reject = |what: String| {
        Err(bad(format!(
            "stop_regex {pattern:?} uses {what}, which Python's `re` cannot compile"
        )))
    };
    // ASCII-only comparisons, so scanning bytes is safe: a UTF-8 continuation byte
    // is >= 0x80 and matches no arm.
    let b = pattern.as_bytes();
    let mut i = 0;
    // Still inside the run of leading `(?flags)` groups.
    let mut leading = true;
    while i < b.len() {
        match b[i] {
            b'\\' => {
                if let Err(what) = check_escape(b, i) {
                    return reject(what);
                }
                leading = false;
                i += 2; // skip the escaped character, so `\(` is not a group open
            }
            // `(?<name>…)` is a named group to Rust; Python spells it `(?P<name>…)`
            // and errors on this one. `(?<=` / `(?<!` are look-behind, which
            // `regex-syntax` rejects on its own.
            b'(' if b[i..].starts_with(b"(?<")
                && !b[i..].starts_with(b"(?<=")
                && !b[i..].starts_with(b"(?<!") =>
            {
                return reject("a `(?<name>…)` group (Python spells it `(?P<name>…)`)".into());
            }
            // A flag-setting group. Python 3.11+ reads these as GLOBAL flags: they
            // must sit at position 0, and the clearing form (`(?-i)`) is invalid on
            // its own — it wants `(?-i:…)`. The flag letters also differ: Rust adds
            // `R`/`U`, Python adds `a`/`L`, so only their intersection is portable.
            b'(' if flag_group_bytes(&b[i..]).is_some() => {
                let flags = flag_group_bytes(&b[i..]).expect("just matched");
                // `(?flags:…)` is scoped: legal anywhere, and its clearing form is
                // legal too. Only the GLOBAL form is position- and sign-restricted.
                let scoped = b[i..].get(2 + flags.len()).is_some_and(|&c| c == b':');
                // Python allows global flags only at the start, but allows SEVERAL
                // (`(?i)(?m)a`); `leading` stays true while we are still in that run.
                if !scoped && !leading {
                    return reject("inline flags after the start of the pattern".into());
                }
                if !scoped && flags.contains(&b'-') {
                    return reject(
                        "a clearing `(?-flags)` group (Python wants `(?-flags:…)`)".into(),
                    );
                }
                if let Some(&f) = flags
                    .iter()
                    .find(|f| !PORTABLE_FLAGS.contains(&(**f as char)))
                {
                    return reject(format!("the inline flag `{}`", f as char));
                }
                // Advance past the WHOLE group, not one byte: scanning its inner
                // `?`/letters/`)` through the default arm would clear `leading` and
                // make the next `(?m)` look like a mid-pattern flag change.
                if scoped {
                    leading = false;
                    i += 1;
                } else {
                    i += 2 + flags.len() + 1; // `(?` + flags + `)`
                }
                continue; // still in the leading flag run
            }
            // A `[` inside a character class. Rust reads it as a literal (or a POSIX
            // class); Python's parser terminates the class differently and can end up
            // parsing the remainder as a group.
            b'[' => {
                let mut j = i + 1;
                if b.get(j) == Some(&b'^') {
                    j += 1;
                }
                if b.get(j) == Some(&b']') {
                    j += 1; // a leading `]` is a literal in both dialects
                }
                while j < b.len() && b[j] != b']' {
                    match b[j] {
                        // Escapes inside a class follow the same rules as outside.
                        b'\\' => {
                            if let Err(what) = check_escape(b, j) {
                                return reject(what);
                            }
                            j += 2;
                        }
                        b'[' => return reject("a `[` nested inside a character class".into()),
                        // `[a--b]` is a class-difference operator in Rust and a bad
                        // character range in Python.
                        b'-' if b.get(j + 1) == Some(&b'-') => {
                            return reject("a `--` class-difference operator".into());
                        }
                        _ => j += 1,
                    }
                }
                i = j.max(i + 1);
            }
            _ => {
                leading = false;
                i += 1;
            }
        }
    }
    Ok(())
}

/// The flag bytes of a flag-setting group (`(?i)`, `(?-i)`, `(?imsx)`), or `None`
/// if `b` does not open one. A `(?i:…)` scoped group is not one of these.
fn flag_group_bytes(b: &[u8]) -> Option<&[u8]> {
    let rest = b.strip_prefix(b"(?")?;
    // Stop at `)` OR `:` — the scoped form `(?i:…)` carries the same flag letters
    // and was falling through unvalidated, so `(?R:a)` reached the scheduler.
    let end = rest.iter().position(|&c| c == b')' || c == b':')?;
    let flags = &rest[..end];
    (!flags.is_empty() && flags.iter().all(|&c| c.is_ascii_alphabetic() || c == b'-'))
        .then_some(flags)
}

/// Check the escape starting at `b[i]` (a backslash). `Err` names why Python's
/// `re` would refuse it. Used for escapes both inside and outside character
/// classes — the class scanner used to skip escapes entirely, which is how
/// `[\p{L}]` slipped past the very check written for `\p{L}`.
fn check_escape(b: &[u8], i: usize) -> Result<(), String> {
    let Some(&e) = b.get(i + 1) else {
        return Err("a trailing backslash".into());
    };
    // `\xHH` is shared; Rust's braced `\x{10FFFF}` is not.
    if e == b'x' && b.get(i + 2) == Some(&b'{') {
        return Err("a braced `\\x{…}` escape".into());
    }
    // `\b{start}` is one zero-width assertion to Rust, but `\b` followed by the
    // literal "{start}" to Python — 7 characters this side would score as 0, so
    // the scheduler sizes a 1-token window and the stop silently never fires.
    if e == b'b' && b.get(i + 2) == Some(&b'{') {
        return Err("a `\\b{…}` assertion".into());
    }
    if e.is_ascii_alphanumeric() && !SHARED_ESCAPES.contains(&(e as char)) {
        return Err(format!("the escape `\\{}`", e as char));
    }
    // `\<` / `\>` are GNU word-boundary ASSERTIONS to `regex-syntax` (width 0) but
    // escaped LITERALS to Python (`\<END\>` needs 5 characters of tail). Scoring
    // them 0 sizes the match window too small, so the stop silently never fires and
    // the request runs to `max_new_tokens` — the one failure mode this module exists
    // to prevent, and `\<WORD\>` is idiomatic from grep/vim.
    if e == b'<' || e == b'>' {
        return Err(format!("the escape `\\{}`", e as char));
    }
    Ok(())
}

/// Validate a `stop_regex` and return the strict upper bound on the characters it
/// can match — the Rust port of Python's `get_max_seq_length`.
///
/// Bounded expressions get their real length; unbounded quantifiers get
/// [`STOP_REGEX_MAX_LEN`] (an over-estimate, so the scheduler never under-buffers
/// and misses a stop). **Anything `regex-syntax` cannot parse is a 400** — the
/// admit-on-unknown fallback this used to have is precisely how `\p{L}` and a
/// variable-width look-behind reached the scheduler. The nest limit is pinned well
/// under the ~495 levels Python's parser survives, so deep nesting is rejected here
/// rather than raising `RecursionError` there.
fn stop_regex_bound(pattern: &str) -> Result<i64, Error> {
    reject_python_incompatible(pattern)?;
    let hir = regex_syntax::ParserBuilder::new()
        .nest_limit(100)
        .build()
        .parse(pattern)
        .map_err(|e| {
            bad(format!(
                "stop_regex {pattern:?} is not a valid regular expression: {e}"
            ))
        })?;
    let ast = regex_syntax::ast::parse::ParserBuilder::new()
        .nest_limit(100)
        .build()
        .parse(pattern)
        .map_err(|e| {
            bad(format!(
                "stop_regex {pattern:?} is not a valid regular expression: {e}"
            ))
        })?;
    if repetition_cost_too_large(&ast, 1, false) {
        return Err(bad(format!(
            "stop_regex {pattern:?} repeats too many times or nests unbounded \
             repetitions; matching it would dominate every decode step"
        )));
    }
    if repeats_an_assertion(&ast) {
        return Err(bad(format!(
            "stop_regex {pattern:?} quantifies a zero-width assertion, which Python's \
             `re` rejects, or a repetition count Python cannot honour"
        )));
    }
    Ok(hir_max_len(&hir))
}

/// Reject repetitions whose cost compounds down the nesting.
///
/// `outer` is the product of the counted repeats enclosing `ast`. Two families die
/// here: a counted product over [`MAX_REPEAT_COUNT`] (memory), and an unbounded
/// repeat nested inside another (`(?:a+)+b` — catastrophic backtracking, measured
/// 2.3 s on a 26-character tail, and since its bound is the full-scan sentinel the
/// tail grows every step, so the loop is dead within ~30 tokens).
fn repetition_cost_too_large(ast: &regex_syntax::ast::Ast, outer: u64, unbounded: bool) -> bool {
    use regex_syntax::ast::{Ast, RepetitionKind, RepetitionRange};
    match ast {
        Ast::Repetition(rep) => {
            let (factor, is_unbounded) = match &rep.op.kind {
                RepetitionKind::Range(RepetitionRange::Exactly(n)) => (*n as u64, false),
                RepetitionKind::Range(RepetitionRange::Bounded(_, hi)) => (*hi as u64, false),
                RepetitionKind::Range(RepetitionRange::AtLeast(n)) => (*n as u64, true), // codespell:ignore atleast
                _ => (1, true), // `*`, `+`, `?`
            };
            let total = outer.saturating_mul(factor.max(1));
            total >= MAX_REPEAT_COUNT
                || (is_unbounded && unbounded)
                || repetition_cost_too_large(&rep.ast, total, unbounded || is_unbounded)
        }
        Ast::Group(g) => repetition_cost_too_large(&g.ast, outer, unbounded),
        Ast::Concat(c) => c
            .asts
            .iter()
            .any(|a| repetition_cost_too_large(a, outer, unbounded)),
        Ast::Alternation(a) => a
            .asts
            .iter()
            .any(|a| repetition_cost_too_large(a, outer, unbounded)),
        _ => false,
    }
}

/// Whether any repetition in `ast` applies to a zero-width assertion — `$*`,
/// `\b{2}`, `^+`. `regex-syntax` accepts them; Python's `re` raises "nothing to
/// repeat". Found by fuzzing the two parsers against each other, not by reading
/// either one's docs.
///
/// Checked on the AST, not the HIR: the HIR translator folds `$+` down to a bare
/// `Look`, so by then the shape Python objects to is gone.
fn repeats_an_assertion(ast: &regex_syntax::ast::Ast) -> bool {
    use regex_syntax::ast::Ast;
    match ast {
        // A quantified assertion (`$*`) or a quantified quantifier (`a?*`, which
        // Python calls "multiple repeat"). Both parse fine in Rust.
        Ast::Repetition(rep) => {
            matches!(&*rep.ast, Ast::Assertion(_) | Ast::Repetition(_))
                || repeats_an_assertion(&rep.ast)
        }
        Ast::Group(g) => repeats_an_assertion(&g.ast),
        Ast::Concat(c) => c.asts.iter().any(repeats_an_assertion),
        Ast::Alternation(a) => a.asts.iter().any(repeats_an_assertion),
        _ => false,
    }
}

/// Strict upper bound on the characters `hir` can match; `None` (unbounded) maps to
/// the full-scan sentinel. Saturating throughout: a nested `{65535}` repeat would
/// otherwise overflow into a small — and therefore unsafe — bound.
fn hir_max_len(hir: &regex_syntax::hir::Hir) -> i64 {
    use regex_syntax::hir::HirKind;
    match hir.kind() {
        HirKind::Empty | HirKind::Look(_) => 0,
        HirKind::Literal(lit) => lit.0.len() as i64,
        HirKind::Class(_) => 1,
        HirKind::Repetition(rep) => match rep.max {
            None => STOP_REGEX_MAX_LEN,
            Some(max) => (max as i64)
                .saturating_mul(hir_max_len(&rep.sub))
                .min(STOP_REGEX_MAX_LEN),
        },
        HirKind::Capture(cap) => hir_max_len(&cap.sub),
        HirKind::Concat(subs) => subs
            .iter()
            .map(hir_max_len)
            .fold(0i64, i64::saturating_add)
            .min(STOP_REGEX_MAX_LEN),
        HirKind::Alternation(subs) => subs.iter().map(hir_max_len).max().unwrap_or(0),
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
        assert_eq!(twice.stop_regex_max_len, 3);

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

    /// The property this whole design rests on: **anything Rust admits, Python can
    /// compile.** The reverse may fail — rejecting a pattern Python would accept
    /// costs one client a 400, while admitting one it cannot compile costs the
    /// scheduler, because `re.search` runs on the decode hot path where nothing
    /// Budget for one `re.search` on the scheduler's decode thread. Every safe
    /// pattern below measures under 0.1 ms; the cheapest unsafe one is 650 ms.
    const SEARCH_BUDGET_MS: f64 = 5.0;

    /// What the admission policy must do with a pattern.
    #[derive(Debug, PartialEq)]
    enum Policy {
        /// Admitting it kills the scheduler or silently misses the stop.
        MustReject,
        /// A pattern real clients send. A 400 here is a client-visible break —
        /// several of these ship in SGLang's own CI kits.
        MustAdmit,
        /// Python compiles it, but `regex-syntax` is the stricter dialect. A 400
        /// is safe (the client loses a feature, nothing crashes), so either
        /// verdict passes.
        MayReject,
    }

    /// One corpus row. Every column except `policy` is a MEASURED fact, recorded
    /// so a future edit cannot re-derive it by guessing:
    ///   * `py_max_len`  — CPython `get_max_seq_length`, or `None` if `re.compile`
    ///     rejects the pattern.
    ///   * `worst_ms`    — worst `re.search` over a growing tail (16→88 chars of
    ///     prose, or a matching run where the pattern needs one). `INFINITY` means
    ///     it did not return inside 8 s under a 2 GiB cap.
    struct Case {
        pattern: String,
        policy: Policy,
        /// Expected bound when admitted. Pins `hir_max_len` against silent drift.
        rust_bound: i64,
        py_max_len: Option<i64>,
        worst_ms: f64,
    }

    fn case(pattern: &str, policy: Policy, rust_bound: i64, py: Option<i64>, ms: f64) -> Case {
        Case {
            pattern: pattern.to_string(),
            policy,
            rust_bound,
            py_max_len: py,
            worst_ms: ms,
        }
    }

    /// The single source of truth for `stop_regex` admission.
    ///
    /// This table exists because eight review rounds each found a NEW spelling of
    /// an already-fixed hazard, and the previous corpus could not catch any of
    /// them: its assertion was `!admitted || python_compiles`, which any row with
    /// `python_compiles = true` satisfies vacuously — including four rows whose own
    /// comments called them scheduler-fatal. It also could not fail on a spurious
    /// 400, so a round that rejected `(?i)[a-z]+` and `colou?r` shipped green.
    ///
    /// KEEP IN SYNC: adding a row means MEASURING `py_max_len` and `worst_ms`, not
    /// guessing them. `corpus_rows_are_self_consistent` refuses a row that records
    /// a fatal measurement and then claims the pattern is safe to admit.
    fn corpus() -> Vec<Case> {
        const UNBOUNDED: i64 = STOP_REGEX_MAX_LEN;
        const INF: f64 = f64::INFINITY;
        let mut c = vec![
            // ---- Direction A: CPython cannot compile these. Admitting one puts a
            // `re.error` in `_check_str_based_finish`, on the decode path, uncaught.
            case(r"\p{L}", Policy::MustReject, 0, None, INF), // round 1
            case(r"\P{L}", Policy::MustReject, 0, None, INF),
            case(r"\pL", Policy::MustReject, 0, None, INF),
            case("(?<n>a)", Policy::MustReject, 0, None, INF),
            case(r"\x{1F600}", Policy::MustReject, 0, None, INF),
            case(r"\u{41}", Policy::MustReject, 0, None, INF),
            case("(?<=a*)b", Policy::MustReject, 0, None, INF), // round 2: variable-width lookbehind
            case("(", Policy::MustReject, 0, None, INF),
            case("[z-a]", Policy::MustReject, 0, None, INF),
            case("a{2,1}", Policy::MustReject, 0, None, INF),
            case("$*", Policy::MustReject, 0, None, INF),
            case(r"\b{2}", Policy::MustReject, 0, None, INF),
            case("^+", Policy::MustReject, 0, None, INF),
            case("a?*", Policy::MustReject, 0, None, INF),
            case("a{2,5}?*", Policy::MustReject, 0, None, INF),
            case("a(?i)b", Policy::MustReject, 0, None, INF),
            case("(?-i)a", Policy::MustReject, 0, None, INF),
            case("[a[:alpha:](?=-]", Policy::MustReject, 0, None, INF),
            // Round 4: the escape check skipped character-class bodies entirely,
            // so round 1's hole reopened one bracket pair away.
            case(r"[\p{L}]", Policy::MustReject, 0, None, INF),
            case(r"[\pL]", Policy::MustReject, 0, None, INF),
            case(r"[\P{L}]", Policy::MustReject, 0, None, INF),
            case(r"[\x{41}]", Policy::MustReject, 0, None, INF),
            case("[a--b]", Policy::MustReject, 0, None, INF),
            case("(?R)a", Policy::MustReject, 0, None, INF), // round 4: Rust-only flag
            case("(?U)a", Policy::MustReject, 0, None, INF),
            case("(?R:a)", Policy::MustReject, 0, None, INF), // round 5: the scoped spelling
            case("(?U:a)", Policy::MustReject, 0, None, INF),
            // `regex-syntax` parses counts as u32 and accepts up to u32::MAX;
            // CPython's MAXREPEAT *is* u32::MAX and raises OverflowError, which is
            // neither `re.error` nor `RecursionError` and so escapes every guard.
            case("a{4294967295}", Policy::MustReject, 0, None, INF), // round 4
            case("a{5000000000}", Policy::MustReject, 0, None, INF), // round 3
            // ---- Bound UNDER-estimates. Both compile and run fast, so only the
            // `rust_bound >= py_max_len` column catches them: `regex-syntax` reads
            // a zero-width word boundary where CPython reads escaped literals, so
            // the scheduler sizes too small a window and the stop never fires.
            case(r"\<END\>", Policy::MustReject, 3, Some(5), 0.02), // round 5
            case(r"\b{start}xyz", Policy::MustReject, 3, Some(10), 0.04), // round 4
            // ---- Compounding repeat cost. Both compile in CPython; both are fatal
            // there. `repetition_cost_too_large` covers these.
            case(
                "(?:(?:a*){65535}){65535}",
                Policy::MustReject,
                0,
                Some(4611545282012774400),
                INF,
            ),
            case("(?:){1048575}x", Policy::MustReject, 0, Some(1), INF),
            // ---- AMBIGUITY (rounds 6-8). Every one compiles cleanly on both sides
            // and raises nothing, so the `except (re.error, RecursionError)` seatbelt
            // in `_check_str_based_finish` is irrelevant: the match simply never
            // returns, inside GIL-holding CPython C that no watchdog can preempt.
            //
            // Two distinct kill modes, both represented:
            //   * unbounded bound -> `_stop_match_tail_len` hands `re.search` the
            //     WHOLE accumulated output, so cost grows every decode step;
            //   * finite bound -> a fixed but ruinous cost paid EVERY step forever,
            //     and `MAX_STOP_REGEX_COUNT` allows 64 patterns per request.
            // Timings on a matching subject; see the module docs for the method.
            case(
                "(?:.|.)*Z",
                Policy::MustReject,
                UNBOUNDED,
                Some(1073741825),
                INF,
            ),
            case(
                "(a|a)*b",
                Policy::MustReject,
                UNBOUNDED,
                Some(1073741825),
                INF,
            ),
            case("(?:a+)+b", Policy::MustReject, 0, Some(1073741825), INF),
            case(
                "(?:a*){10}b",
                Policy::MustReject,
                UNBOUNDED,
                Some(10737418241),
                INF,
            ),
            case(
                "a*a*a*a*a*a*a*a*b",
                Policy::MustReject,
                UNBOUNDED,
                Some(8589934593),
                636.05,
            ),
            case(
                "(?:.*){20}Z",
                Policy::MustReject,
                UNBOUNDED,
                Some(21474836481),
                INF,
            ),
            case(
                ".*.*.*.*.*.*.*.*Z",
                Policy::MustReject,
                UNBOUNDED,
                Some(8589934593),
                INF,
            ),
            case("(?:.?){30}Z", Policy::MustReject, 31, Some(31), INF), // round 7
            case("(?:.?){255}Z", Policy::MustReject, 256, Some(256), INF),
            case(
                "(?:.{0,1}.{0,1}.{0,1}){8}Z",
                Policy::MustReject,
                25,
                Some(25),
                INF,
            ),
            case(
                "(?:(?:.?){15}){15}Z",
                Policy::MustReject,
                226,
                Some(226),
                INF,
            ),
            case("(?:.?.?.?.?){60}Z", Policy::MustReject, 241, Some(241), INF),
            // ---- MustAdmit: ordinary patterns. Three of these ship in SGLang's own
            // CI (`python/sglang/test/kits/matched_stop_kit.py`), consumed by five
            // registered suites; rejecting them 400s the project's own fixtures.
            case(
                r"[.!?]\s*$",
                Policy::MustAdmit,
                UNBOUNDED,
                Some(1073741825),
                0.03,
            ),
            case("and|or", Policy::MustAdmit, 3, Some(3), 0.03),
            case(r"\d+", Policy::MustAdmit, UNBOUNDED, Some(1073741824), 0.03),
            case(
                r"\s+$",
                Policy::MustAdmit,
                UNBOUNDED,
                Some(1073741824),
                0.04,
            ),
            case(
                "Answer: .*",
                Policy::MustAdmit,
                UNBOUNDED,
                Some(1073741832),
                0.03,
            ),
            case(".*", Policy::MustAdmit, UNBOUNDED, Some(1073741824), 0.04),
            case(
                "a{3,}",
                Policy::MustAdmit,
                UNBOUNDED,
                Some(1073741824),
                0.03,
            ),
            // Round 8 regressed every `?`/`*`/`+` to a 400 by routing them into an
            // "unbounded" catch-all that returned u64::MAX.
            case("colou?r", Policy::MustAdmit, 6, Some(6), 0.03),
            case("https?://", Policy::MustAdmit, 8, Some(8), 0.02),
            case("END(ING)?", Policy::MustAdmit, 6, Some(6), 0.02),
            // Round 7 regressed these by scanning the whole pattern for `-` instead
            // of just the flag bytes.
            case(
                "(?i)[a-z]+",
                Policy::MustAdmit,
                UNBOUNDED,
                Some(1073741824),
                0.04,
            ),
            case(r"(?i)\d{4}-\d{2}", Policy::MustAdmit, 7, Some(7), 0.06),
            case("(?imsx)a-b", Policy::MustAdmit, 3, Some(3), 0.04),
            case("(?-i:abc)", Policy::MustAdmit, 3, Some(3), 0.03),
            case("(?i-s:a)", Policy::MustAdmit, 1, Some(1), 0.04),
            case("(?i)(?m)a", Policy::MustAdmit, 1, Some(1), 0.03),
            case(r"\x41", Policy::MustAdmit, 1, Some(1), 0.02),
            case(r"\d{6}", Policy::MustAdmit, 6, Some(6), 0.03),
            case("abc", Policy::MustAdmit, 3, Some(3), 0.03),
            case("(?P<n>a)", Policy::MustAdmit, 1, Some(1), 0.03),
            case(r"a\.b", Policy::MustAdmit, 3, Some(3), 0.03),
            case(r"\bword\b", Policy::MustAdmit, 4, Some(4), 0.03),
            case(r"[\d\s]{2}", Policy::MustAdmit, 2, Some(2), 0.03),
            // ---- MayReject: CPython accepts, `regex-syntax` is stricter. A 400
            // costs the client a feature; admitting costs nothing either. Listed so
            // the set of deliberate over-rejections is visible rather than folklore.
            case(r"a\Z", Policy::MayReject, 1, Some(1), 0.03),
            case(r"(a)\1", Policy::MayReject, 0, Some(1073741825), 0.04),
            case("(?=x)y", Policy::MayReject, 0, Some(1073741825), 0.03),
            case("a{,5}", Policy::MayReject, 5, Some(5), 0.04),
            case(r"\N{SNOWMAN}", Policy::MayReject, 1, Some(1), 0.03),
            case(r"\0", Policy::MayReject, 1, Some(1), 0.03),
        ];
        // Flat concatenations of optional atoms — the round-8 escape. Built rather
        // than written out because they are 73-221 bytes of repetition.
        c.push(case(
            &format!("{}Z", ".{0,1}".repeat(20)),
            Policy::MustReject,
            21,
            Some(21),
            650.62,
        ));
        c.push(case(
            &format!("{}Z", ".{0,4}".repeat(12)),
            Policy::MustReject,
            49,
            Some(49),
            INF,
        ));
        c
    }

    /// A row may not record a fatal measurement and then claim the pattern is safe
    /// to admit. Without this, the table can be made green by editing a verdict
    /// instead of fixing the code — which is exactly how round 8's `(?i)[a-z]+`
    /// regression survived (the corpus row was left alone and a *different* test
    /// was edited from `(?i)[a-z]+` to `(?i)[a-z]{1,8}` to keep it passing).
    #[test]
    fn corpus_rows_are_self_consistent() {
        for c in corpus() {
            if c.py_max_len.is_none() || c.worst_ms > SEARCH_BUDGET_MS {
                assert_eq!(
                    c.policy,
                    Policy::MustReject,
                    "{:?} does not compile in Python, or costs {} ms per decode step \
                     (budget {SEARCH_BUDGET_MS} ms) — it cannot be admitted",
                    c.pattern,
                    c.worst_ms
                );
            }
        }
    }

    /// The corpus, asserted in BOTH directions plus the bound.
    ///
    /// Three independent invariants, each of which caught a real bug that the
    /// others missed:
    ///   1. `MustReject` really is rejected — Direction A (scheduler death) and the
    ///      ambiguity family (scheduler wedge).
    ///   2. `MustAdmit` really is admitted — a spurious 400 breaks working clients
    ///      and, twice now, SGLang's own registered suites.
    ///   3. an admitted pattern's bound is >= CPython's, so the scheduler's match
    ///      window is never too small. This is the only mechanical check for the
    ///      `\b{start}` / `\<` class, which nobody found by reading.
    #[test]
    fn stop_regex_corpus_holds_in_both_directions() {
        let mut failures: Vec<String> = Vec::new();
        for c in corpus() {
            let got = stop_regex_bound(&c.pattern);
            match (&c.policy, &got) {
                (Policy::MustReject, Ok(bound)) => failures.push(format!(
                    "ADMITTED but must be rejected: {:?} (bound {bound}, \
                     worst re.search {} ms)",
                    c.pattern, c.worst_ms
                )),
                (Policy::MustAdmit, Err(e)) => failures.push(format!(
                    "REJECTED but must be admitted: {:?} — {e}",
                    c.pattern
                )),
                _ => {}
            }
            if let Ok(bound) = got {
                if c.policy != Policy::MustReject && bound != c.rust_bound {
                    failures.push(format!(
                        "bound drift: {:?} expected {} got {bound}",
                        c.pattern, c.rust_bound
                    ));
                }
                // Only meaningful when CPython's own bound is finite: for unbounded
                // patterns both sides emit an absurd sentinel that the scheduler
                // caps at the output length anyway.
                if let Some(py) = c.py_max_len
                    && py < STOP_REGEX_MAX_LEN
                    && bound < py
                {
                    {
                        failures.push(format!(
                            "UNDER-estimate: {:?} rust bound {bound} < python {py} — \
                             the scheduler's window is too small and the stop never fires",
                            c.pattern
                        ));
                    }
                }
            }
        }
        assert!(
            failures.is_empty(),
            "{} corpus row(s) failed:\n  {}",
            failures.len(),
            failures.join("\n  ")
        );
    }

    /// The leading-flag check must look at the FLAG BYTES, not the rest of the
    /// pattern: scanning the whole tail for `-` made `(?i)[a-z]+` — about as
    /// ordinary as a stop_regex gets — a 400.
    #[test]
    fn leading_inline_flags_are_accepted() {
        for pattern in [
            "(?i)[a-z]{1,8}",
            r"(?i)\d{4}-\d{2}",
            "(?imsx)a-b",
            "(?i)abc",
        ] {
            assert!(
                stop_regex_bound(pattern).is_ok(),
                "{pattern} is valid Python and must not be rejected"
            );
        }
        // …but only leading, only set-flags, and only portable letters.
        for pattern in ["a(?i)b", "(?-i)a", "(?R)a", "(?U)a"] {
            assert!(
                stop_regex_bound(pattern).is_err(),
                "{pattern} must be rejected"
            );
        }
    }

    /// Patterns Python compiles fine that this validator used to 400. A false
    /// rejection is safe but it is still a bug: `(?i-s:a)` alone was 267 hits in
    /// the review corpus, and `\x41` was rejected by the very list whose doc
    /// comment calls `\xHH` shared.
    #[test]
    fn ordinary_python_patterns_are_not_spuriously_rejected() {
        for pattern in [
            "(?-i:abc)", // scoped clearing group: legal anywhere
            "(?i-s:a)",  // mixed set/clear inside a scoped group
            r"\x41",     // two-hex escape — shared with Python
            r"a\x41b",
            "(?i)(?m)a", // several LEADING global flag groups
            "(?i)abc",
        ] {
            assert!(
                stop_regex_bound(pattern).is_ok(),
                "{pattern} is valid Python and must not be rejected"
            );
        }
        // The genuinely Rust-only forms still reject.
        for pattern in [r"\x{41}", "a(?i)b", "(?R)a"] {
            assert!(
                stop_regex_bound(pattern).is_err(),
                "{pattern} must be rejected"
            );
        }
    }

    /// Both `stop_regex` caps, neither of which had a test: deleting either `if`
    /// left the suite green. The count cap bounds per-step recompilation (Python's
    /// `re` cache is 512 entries); the length cap bounds compile time (a 1 MB
    /// literal pattern measured ~677 ms).
    #[test]
    fn stop_regex_count_and_length_are_capped() {
        let over: Vec<String> = (0..MAX_STOP_REGEX_COUNT + 1)
            .map(|i| format!("a{i}"))
            .collect();
        let json = serde_json::json!({ "stop_regex": over }).to_string();
        assert!(norm_err(&json).to_string().contains("at most"));

        let at_cap: Vec<String> = (0..MAX_STOP_REGEX_COUNT).map(|i| format!("a{i}")).collect();
        let json = serde_json::json!({ "stop_regex": at_cap }).to_string();
        assert!(
            serde_json::from_str::<SamplingParams>(&json)
                .unwrap()
                .normalize(false, None)
                .is_ok(),
            "the cap itself must be accepted"
        );

        let long = "a".repeat(MAX_STOP_REGEX_LEN + 1);
        let json = serde_json::json!({ "stop_regex": long }).to_string();
        let err = norm_err(&json).to_string();
        assert!(err.contains("over the"), "{err}");
    }

    /// The commoner field had no limit at all: the scheduler scans the decoded text
    /// once per stop per decode step.
    #[test]
    fn stop_string_count_is_capped() {
        let stops: Vec<String> = (0..MAX_STOP_COUNT + 1).map(|i| i.to_string()).collect();
        let json = serde_json::json!({ "stop": stops }).to_string();
        let err = norm_err(&json).to_string();
        assert!(err.contains("at most"), "{err}");
    }

    /// A repetition count Python cannot honour: `u32::MAX` is its `MAXREPEAT`
    /// sentinel (`OverflowError`), and a large count on a group exhausts memory at
    /// compile time (`MemoryError`). Neither is an `re.error`, so the decode-loop
    /// seatbelt would not catch either.
    #[test]
    fn oversized_repeat_counts_are_rejected() {
        for pattern in [
            "a{4294967295}",
            "a{4294967294}",
            "(?:a*){4294967294}",
            "a{1048576}",
            "a{0,4294967295}",
            "a{1048576,}",
        ] {
            assert!(
                stop_regex_bound(pattern).is_err(),
                "{pattern} must be rejected"
            );
        }
        // An ordinary count still works, and still yields a finite bound.
        assert_eq!(stop_regex_bound("a{200}").unwrap(), 200);
    }

    /// `\b{start}` is one zero-width assertion to Rust (bound 0) but `\b` plus the
    /// literal `{start}` to Python (7 characters). Scoring it 0 would size a
    /// 1-token match window where 7 characters are needed, and the stop would
    /// silently never fire — an UNDER-estimate, the one failure mode the sentinel
    /// design exists to prevent.
    #[test]
    fn b_brace_assertion_is_rejected_not_under_estimated() {
        assert!(stop_regex_bound(r"\b{start}xyz").is_err());
        assert!(stop_regex_bound(r"\b{end}").is_err());
        assert_eq!(
            stop_regex_bound(r"\bword").unwrap(),
            4,
            "plain \\b still works"
        );
    }

    /// Round 5's under-estimate: `regex-syntax` reads `\<`/`\>` as GNU word-boundary
    /// assertions (width 0), CPython as escaped literals. Scoring `\<END\>` as 3
    /// instead of 5 sizes the scheduler's match window too small, so the stop never
    /// fires and the request burns GPU to `max_new_tokens`.
    #[test]
    fn gnu_word_boundary_escapes_are_rejected() {
        for pattern in [r"\<END\>", r"\<word", r"end\>"] {
            assert!(
                stop_regex_bound(pattern).is_err(),
                "{pattern} must be rejected"
            );
        }
        // A plain `<` is a literal in both and still bounds correctly.
        assert_eq!(stop_regex_bound("<END>").unwrap(), 5);
    }

    /// Repetition cost compounds down the nesting, so a per-node cap misses
    /// `(?:(?:a*){65535}){65535}` — 22 bytes, compiles fine in Python, then eats
    /// GiB inside `re.search` on the decode hot path (`MemoryError`, which the
    /// seatbelt does not catch). Nested UNBOUNDED repeats are the backtracking
    /// family, fatal in wall-clock rather than memory.
    #[test]
    fn compounding_repetition_cost_is_rejected() {
        for pattern in [
            "(?:(?:a*){65535}){65535}",
            "(?:){1048575}x",
            "(?:a{100}){100}",
            "(?:a+)+b",
            "(a*)*b",
        ] {
            assert!(
                stop_regex_bound(pattern).is_err(),
                "{pattern} must be rejected"
            );
        }
        // Ordinary nesting still works.
        assert_eq!(stop_regex_bound("(?:ab){3}").unwrap(), 6);
        assert_eq!(stop_regex_bound(r"\d{6}").unwrap(), 6);
    }

    /// Deep nesting is rejected here rather than blowing Python's parser stack:
    /// CPython compiles up to ~495 levels and raises `RecursionError` past that, so
    /// the parser's nest limit is pinned well below it.
    #[test]
    fn deep_nesting_is_rejected_below_pythons_limit() {
        let nest = |n: usize| format!("{}a{}", "(".repeat(n), ")".repeat(n));
        assert!(
            stop_regex_bound(&nest(10)).is_ok(),
            "ordinary nesting is fine"
        );
        assert!(
            stop_regex_bound(&nest(400)).is_err(),
            "must be rejected here — Python raises RecursionError, not re.error"
        );
        assert!(stop_regex_bound(&nest(2000)).is_err());
    }

    /// Bounded patterns get their real length; unbounded ones the full-scan
    /// sentinel, so the scheduler never under-buffers and misses a stop.
    #[test]
    fn stop_regex_bound_is_finite_when_bounded() {
        let len = |p: &str| stop_regex_bound(p).expect("valid pattern");
        assert_eq!(len(r"\d{6}"), 6);
        assert_eq!(len("abc"), 3);
        assert_eq!(len(r"^abc$"), 3); // anchors are zero-width
        assert_eq!(len("a|bbb"), 3); // alternation → max branch
        assert_eq!(len(r"(ab){3}"), 6);
        assert_eq!(len(r"a\d{2,5}"), 6);
        assert_eq!(len(r"\d+"), STOP_REGEX_MAX_LEN);
        assert_eq!(len(".*"), STOP_REGEX_MAX_LEN);
        assert_eq!(len(r"a{3,}"), STOP_REGEX_MAX_LEN);
        // End to end, through the path `/generate` takes.
        assert_eq!(norm(r#"{"stop_regex": "\\d{6}"}"#).stop_regex_max_len, 6);
        assert_eq!(norm(r#"{"temperature": 0.7}"#).stop_regex_max_len, 0);
        let _ = norm_err(r#"{"stop_regex": "("}"#);
        let _ = norm_err(r#"{"stop_regex": ["\\d{6}", "("]}"#);
    }
}
