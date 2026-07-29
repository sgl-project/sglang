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

use serde::{Deserialize, Serialize};

use crate::error::Error;

/// The sampling parameters of one `/generate` request. Deserialized from the
/// client's `sampling_params` object (unknown keys are a 400, mirroring Python's
/// `SamplingParams(**kwargs)` TypeError) and serialized by field name into the
/// scheduler header once [`normalize`](Self::normalize) has run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct SamplingParams {
    pub max_new_tokens: Option<i64>,
    pub temperature: f64,
}

impl SamplingParams {
    /// Normalize the fields, applying defaults and coercing types. Mirrors
    /// Python's `SamplingParams.__post_init__` → `normalize`.
    pub fn normalize(
        &mut self,
        _skip_tokenizer_init: bool,
        _vocab_size: Option<u64>,
    ) -> Result<(), Error> {
        todo!()
    }

    /// Verify the normalized fields are in range. Mirrors Python's
    /// `SamplingParams.verify`.
    pub fn verify(&self) -> Result<(), String> {
        todo!()
    }

    pub fn max_tokens_len(&self) -> usize {
        todo!()
    }
}

/// The `/generate` body's `sampling_params`: one object (broadcast to every
/// prompt) or a list of them (one per prompt), fanned out by `GenerateBody::into_requests`.
///
/// Hand-written `Deserialize` rather than `#[serde(untagged)]`: untagged buffers
/// the input and, on failure, reports only "data did not match any variant" —
/// losing the field-level message ("unknown field `temperature`, expected one of
/// …") that makes a typo actionable. Object-vs-list is unambiguous here, so a
/// single `deserialize_any` dispatch keeps the inner error verbatim.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SamplingParamsInput {
    /// Boxed: `SamplingParams` is ~440 bytes, so an inline variant would make
    /// every `GenerateBody` that big regardless of which form arrived.
    One(Box<SamplingParams>),
    Many(Vec<SamplingParams>),
}
