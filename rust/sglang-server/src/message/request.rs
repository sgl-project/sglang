//! The `/generate` request path: the HTTP body and its per-request fan-out
//! ([`GenerateBody`] → [`GenerateRequest`]s), the variant bodies, and the
//! scheduler ingress encodings (`TokenizedGenerateReqInput` header,
//! control/abort, `IngressMsg`).

use std::collections::HashSet;

use bytes::Bytes;
use itertools::izip;
use serde::Deserialize;

use super::io_struct::{ControlRequest, TokenizedGenerateReqInput};
use super::{OneOrMany, OneOrManyItem, SamplingParams, SamplingParamsInput, TokenIds};
use crate::error::Error;

/// Hard cap on prompts per `/generate` body. Every column below is allocated per
/// item, so this bounds the work a single request can ask for.
const MAX_BATCH_SIZE: usize = 4096;

/// Hard cap on the total bytes a broadcast value may clone into the batch (see
/// the `One` arms of the fan-out).
const MAX_BROADCAST_CLONE_BYTES: usize = 64 << 20;

/// Live heap per byte of serialized JSON. Measured across shapes at 1.0–7.0×
/// (`serde_json::Value` pays for enum tags, `String` headers and map nodes that
/// the wire form does not); 8 is the ceiling of that range, not a worst case.
const JSON_TO_HEAP_FACTOR: usize = 8;

/// The `/generate` wire body before batch splitting: `text`/`input_ids`/`sampling_params`
/// each scalar-or-list, fanned into per-request [`GenerateRequest`]s by
/// [`into_requests`](GenerateBody::into_requests). `deny_unknown_fields` rejects (4xx) unknowns.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerateBody {
    /// Optional client-supplied request id(s): a single string (a batch fans it
    /// out as `{rid}_{i}`, mirroring Python `_normalize_batch`) or one per item.
    #[serde(default)]
    pub rid: Option<OneOrMany<String>>,
    #[serde(default)]
    pub text: Option<OneOrMany<String>>,
    #[serde(default)]
    pub input_ids: Option<OneOrMany<TokenIds>>,
    #[serde(default)]
    pub stream: bool,
    /// One params object (broadcast) or a list of them (per item); see
    /// [`SamplingParamsInput`].
    #[serde(default)]
    pub sampling_params: Option<SamplingParamsInput>,
    /// Logprob / hidden-state options: a scalar broadcasts to every prompt, a
    /// list is per-prompt (Python `_normalize_logprob_params`).
    #[serde(default)]
    pub return_logprob: Option<OneOrMany<bool>>,
    #[serde(default)]
    pub logprob_start_len: Option<OneOrMany<i64>>,
    #[serde(default)]
    pub top_logprobs_num: Option<OneOrMany<i64>>,
    /// Token ids to report logprobs for: one list (broadcast to every prompt) or
    /// one list per prompt, mirroring Python's
    /// `Union[List[int], List[List[int]]]` fan-out in `_normalize_batch`.
    #[serde(default)]
    pub token_ids_logprob: Option<OneOrMany<TokenIds>>,
    #[serde(default)]
    pub return_hidden_states: Option<OneOrMany<bool>>,
    /// Scalar-only in Python too (`return_text_in_logprobs: bool`).
    #[serde(default)]
    pub return_text_in_logprobs: Option<bool>,
    // Multimodal inputs. Permissive `Value` types: any JSON shape the Python
    // `GenerateReqInput` accepts (URL / base64 str / list / list-of-lists)
    // parses; `into_requests` fans them out per item mirroring the Python
    // `_normalize_{image,video,audio}_data` batch rules.
    #[serde(default)]
    pub image_data: Option<rmpv::Value>,
    #[serde(default)]
    pub video_data: Option<rmpv::Value>,
    #[serde(default)]
    pub audio_data: Option<rmpv::Value>,

    // Accepted for wire-compat with the native `bench_serving` payload (a full
    // `GenerateReqInput`) but NOT yet wired into the scheduler — parsed and
    // dropped in `into_requests`. Declaring them keeps `deny_unknown_fields` (typos still
    // 400) while letting a benchmark request through. Permissive `Value` types so
    // any valid shape (str / list / list-of-lists) parses. `dead_code`-allowed:
    // deserialized then intentionally ignored.
    #[serde(default)]
    #[allow(dead_code)]
    pub lora_path: Option<rmpv::Value>,
    #[serde(default)]
    #[allow(dead_code)]
    pub return_routed_experts: Option<bool>,
}

impl GenerateBody {
    /// Validate, normalize and fan the body into one [`GenerateRequest`] per
    /// prompt + `is_batch` (list form — a 1-element list is still a batch → JSON
    /// array response). The Rust counterpart of Python
    /// `GenerateReqInput.normalize_batch_and_arguments`; an invalid/inconsistent
    /// batch is [`Error::Validation`], which the handler surfaces with the
    /// variant's own status (400).
    pub fn into_requests(self) -> Result<(Vec<GenerateRequest>, bool), Error> {
        let GenerateBody {
            rid,
            text,
            input_ids,
            stream,
            sampling_params,
            return_logprob,
            logprob_start_len,
            top_logprobs_num,
            token_ids_logprob,
            return_hidden_states,
            return_text_in_logprobs,
            image_data,
            video_data,
            audio_data,
            // Accepted for bench_serving compat, not wired through — see the struct.
            ..
        } = self;

        // Cap the batch BEFORE the columns below allocate anything. Reading the
        // declared length off the input costs nothing; the previous placement (after
        // the match) had already allocated ~1.7 GiB for a 114 MiB body, most of it
        // the `vec![None; n]` twin column.
        let declared_n = match (&text, &input_ids) {
            (Some(OneOrMany::Many(v)), None) => v.len(),
            (None, Some(OneOrMany::Many(v))) => v.len(),
            _ => 1,
        };
        if declared_n > MAX_BATCH_SIZE {
            return Err(Error::Validation(format!(
                "batch size {declared_n} exceeds the maximum of {MAX_BATCH_SIZE}"
            )));
        }

        // Per-item (text, input_ids) columns + whether the input used list form.
        type Columns = (Vec<Option<String>>, Vec<Option<TokenIds>>, bool);
        // Exactly one of text / input_ids (Python `_validate_inputs`), and no
        // empty id list (Python `_determine_batch_size`).
        let (texts, id_lists, is_batch): Columns = match (text, input_ids) {
            (Some(_), Some(_)) => {
                return Err(Error::Validation(
                    "provide either `text` or `input_ids`, not both".into(),
                ));
            }
            (None, None) => {
                return Err(Error::Validation(
                    "either `text` or `input_ids` must be provided".into(),
                ));
            }
            (Some(OneOrMany::One(s)), None) => (vec![Some(s)], vec![None], false),
            (Some(OneOrMany::Many(v)), None) => {
                let n = v.len();
                (v.into_iter().map(Some).collect(), vec![None; n], true)
            }
            // `[]` parses as `One(vec![])` (one prompt with no ids), so the
            // `n == 0` guard below never sees it — reject it here, as Python's
            // `_determine_batch_size` does.
            (None, Some(OneOrMany::One(x))) => {
                if x.is_empty() {
                    return Err(Error::Validation("input_ids cannot be empty".into()));
                }
                (vec![None], vec![Some(x)], false)
            }
            (None, Some(OneOrMany::Many(vv))) => {
                if vv.iter().any(|ids| ids.is_empty()) {
                    return Err(Error::Validation(
                        "input_ids cannot be empty for any prompt in the batch".into(),
                    ));
                }
                let n = vv.len();
                (vec![None; n], vv.into_iter().map(Some).collect(), true)
            }
        };
        let n = texts.len();
        if n == 0 {
            return Err(Error::Validation(
                "batch must contain at least one item".into(),
            ));
        }

        // A list is per-item; a single object broadcasts to every item.
        let sps: Vec<SamplingParams> = match sampling_params {
            None => vec![SamplingParams::default(); n],
            Some(SamplingParamsInput::Many(v)) => {
                if v.len() != n {
                    return Err(Error::Validation(format!(
                        "sampling_params list length {} does not match batch size {n}",
                        v.len()
                    )));
                }
                v
            }
            Some(SamplingParamsInput::One(sp)) => {
                // Broadcasting deep-clones the client's params once per prompt,
                // heap and all — `stop`, `logit_bias` and `custom_params` (arbitrary
                // JSON) are still unnormalized client data here. The blow-up is
                // quadratic in the body: ~1 MB of `custom_params` broadcast to 200k
                // prompts is ~200 GB of clones, and a Rust allocation failure calls
                // `abort()`, which is uncatchable and takes the scheduler process
                // with it. Bound the product, not just `n`.
                // `n == 1` is not a broadcast, so skip the sizing entirely: measuring
                // it means serializing the client's whole `custom_params` to a
                // throwaway `String` on every single request. The callee's own
                // `n > 1` guard cannot prevent that — the cost is in the argument.
                if n > 1 {
                    // Serialized bytes are NOT the clone cost: measured, 63.7 MiB of
                    // JSON became ~1008 MiB of live heap once parsed into `Value`
                    // nodes, `String`s and map entries. Scale by that measured factor
                    // so the budget bounds memory rather than wire size.
                    let per_clone = serde_json::to_string(&*sp)
                        .map_or(0, |s| s.len())
                        .saturating_mul(JSON_TO_HEAP_FACTOR);
                    check_broadcast_budget(per_clone, n, "sampling_params")?;
                }
                vec![*sp; n]
            }
        };

        // rid: absent → mint one uuid per item here, so every request carries its
        // final rid from this point on; a single string fans out as `{rid}_{i}`
        // for a batch (Python `_normalize_batch`); a list is per-item.
        let rids: Vec<String> = match rid {
            None => (0..n).map(|_| crate::ids::new_rid()).collect(),
            Some(OneOrMany::One(r)) if !is_batch => vec![r],
            Some(OneOrMany::One(r)) => {
                check_broadcast_budget(r.len(), n, "rid")?;
                (0..n).map(|i| format!("{r}_{i}")).collect()
            }
            Some(OneOrMany::Many(v)) => {
                if !is_batch || v.len() != n {
                    return Err(Error::Validation(format!(
                        "rid list length {} does not match batch size {n}",
                        v.len()
                    )));
                }
                // Python `_validate_rid_uniqueness`: two items sharing an rid would
                // share a `RidHash` slot, so the first is orphaned before it starts.
                {
                    let mut seen = HashSet::with_capacity(v.len());
                    let duplicates: Vec<&String> = v.iter().filter(|r| !seen.insert(*r)).collect();
                    if !duplicates.is_empty() {
                        return Err(Error::Validation(format!(
                            "duplicate request IDs detected within the request: {duplicates:?}"
                        )));
                    }
                }
                v
            }
        };

        // Fans out exactly like the scalar options: one list broadcasts, a list of
        // lists is per item (Python `_normalize_batch`'s nested branch). Empties
        // are collapsed per item below, not here.
        let tid_logprobs = fan_out(token_ids_logprob, n, "token_ids_logprob")?;

        // Each logprob/hidden opt: absent → None for every item, a scalar
        // broadcasts, a list is per-item (Python `normalize_param`, plus a length
        // check Python lacks — it would `IndexError` later instead).
        let return_logprobs = fan_out(return_logprob, n, "return_logprob")?;
        let logprob_start_lens = fan_out(logprob_start_len, n, "logprob_start_len")?;
        let top_logprobs_nums = fan_out(top_logprobs_num, n, "top_logprobs_num")?;
        let return_hidden = fan_out(return_hidden_states, n, "return_hidden_states")?;

        // Multimodal columns, mirroring the Python batch-normalize rules
        // (`_normalize_image_data` / `_normalize_video_data` / `_normalize_audio_data`).
        let images = split_mm_column(image_data, n, is_batch, MmBroadcast::WrapInList)
            .map_err(|e| Error::Validation(format!("image_data: {e}")))?;
        let videos = split_mm_column(video_data, n, is_batch, MmBroadcast::AsIs)
            .map_err(|e| Error::Validation(format!("video_data: {e}")))?;
        let audios = split_mm_column(audio_data, n, is_batch, MmBroadcast::AsIs)
            .map_err(|e| Error::Validation(format!("audio_data: {e}")))?;

        // Every column above is exactly `n` long, so zip them by value: each
        // request takes ownership of its cell, with no indexing or bounds checks.
        let requests = izip!(
            rids,
            texts,
            id_lists,
            sps,
            return_logprobs,
            logprob_start_lens,
            top_logprobs_nums,
            tid_logprobs,
            return_hidden,
            images,
            videos,
            audios,
        )
        .map(
            |(
                rid,
                text,
                input_ids,
                sampling_params,
                return_logprob,
                logprob_start_len,
                top_logprobs_num,
                token_ids_logprob,
                return_hidden_states,
                image,
                video,
                audio,
            )| GenerateRequest {
                rid,
                text,
                input_ids,
                sampling_params,
                stream,
                return_logprob,
                logprob_start_len,
                top_logprobs_num,
                // `Some` here means "these ids were requested", so an empty list
                // collapses to None.
                token_ids_logprob: token_ids_logprob.filter(|ids| !ids.is_empty()),
                return_hidden_states,
                return_text_in_logprobs,
                mm: pack_mm(image, video, audio),
            },
        )
        .collect();
        Ok((requests, is_batch))
    }
}

/// Box the per-item mm values, or `None` when the item has none (the common
/// text-only case keeps `GenerateRequest` slim).
fn pack_mm(
    image_data: Option<rmpv::Value>,
    video_data: Option<rmpv::Value>,
    audio_data: Option<rmpv::Value>,
) -> Option<Box<MmData>> {
    if image_data.is_none() && video_data.is_none() && audio_data.is_none() {
        return None;
    }
    Some(Box::new(MmData {
        image_data,
        video_data,
        audio_data,
    }))
}

/// How a scalar (non-list) mm value broadcasts across a batch: images become a
/// single-image list per item (`[[img]] * num` in Python `_normalize_image_data`),
/// video/audio broadcast the bare value (`[v] * num` in `_normalize_video_data`).
#[derive(Clone, Copy)]
enum MmBroadcast {
    WrapInList,
    AsIs,
}

/// Fan one mm field (`image_data` / `video_data` / `audio_data`) into per-item
/// values, mirroring the Python batch-normalize semantics:
///   * `None` / empty list → `None` for every item;
///   * single request → the raw value passes through (the processor-side
///     normalize wraps a non-list into a one-element list);
///   * batch + non-list → broadcast to every item (per `MmBroadcast`);
///   * batch + list → per item; length must equal the batch size.
fn split_mm_column(
    v: Option<rmpv::Value>,
    n: usize,
    is_batch: bool,
    broadcast: MmBroadcast,
) -> Result<Vec<Option<rmpv::Value>>, String> {
    let Some(v) = v else {
        return Ok(vec![None; n]);
    };
    if v.is_nil() {
        return Ok(vec![None; n]);
    }
    if !is_batch {
        return Ok(vec![Some(v)]);
    }
    match v {
        rmpv::Value::Array(items) if items.is_empty() => Ok(vec![None; n]),
        rmpv::Value::Array(items) => {
            if items.len() != n {
                return Err(format!(
                    "list length {} does not match batch size {n}",
                    items.len()
                ));
            }
            Ok(items.into_iter().map(Some).collect())
        }
        scalar => Ok(match broadcast {
            MmBroadcast::WrapInList => vec![Some(rmpv::Value::Array(vec![scalar])); n],
            MmBroadcast::AsIs => vec![Some(scalar); n],
        }),
    }
}

/// One request handed to the MM worker pool: the rid to correlate the result
/// plus the msgpack payload from [`GenerateRequest::to_mm_payload_msgpack`].
#[derive(Debug)]
pub struct MmRequest {
    pub rid: String,
    pub payload: Bytes,
}

/// Rust mirror of Python `has_valid_data` for an opaque mm field: `null` and
/// (recursively) empty / all-null lists don't count as multimodal input.
/// Delegates to the same `value_present` the MM worker's payload parser uses,
/// so routing and parsing can never disagree on what counts as present.
fn mm_value_present(v: &Option<rmpv::Value>) -> bool {
    v.as_ref().is_some_and(super::mm_payload::value_present)
}

/// Request variant — selects the ingress branch, scheduler wire message, and
/// egress shape. Each owns its body, so generate/control fields stay type-separate.
#[derive(Debug)]
pub enum RequestKind {
    /// `/generate`: tokenize (if needed) then push a `TokenizedGenerateReqInput`.
    Generate(Box<GenerateRequest>),
    /// A control endpoint (e.g. `/server_info`, `/health`): no tokenization, and
    /// the egress is a single non-streamed JSON result.
    Control(Box<ControlRequest>),
}

/// A single in-flight `/generate` request (per-item from
/// [`GenerateBody::into_requests`]),
/// serialized to the scheduler wire once tokenized (see `to_header_msgpack`). Not a
/// wire type — built by `into_requests`/handlers, never (de)serialized; `input_ids` is
/// client-supplied or filled by the Tokenizer stage.
#[derive(Debug, Default)]
pub struct GenerateRequest {
    /// This item's final rid: the client's (normalized per item by `into_requests`) or a
    /// uuid minted there when none was sent.
    ///
    /// Duplicates *within* one request are rejected by `into_requests` (Python
    /// `_validate_rid_uniqueness`). A collision with a *concurrent* request's rid
    /// is rejected too, by the in-flight registry `api_server::submit` consults —
    /// as Python's `TokenizerManager` does ("Duplicate request ID detected"). It
    /// used to overwrite the earlier request's `RidHash` slot, and this comment
    /// used to call that parity with Python; both were wrong.
    pub rid: String,
    pub text: Option<String>,
    /// Client-supplied token ids, or filled by the Tokenizer stage.
    pub input_ids: Option<TokenIds>,
    /// Sampling params (defaults when the client sent none, as in Python);
    /// normalized + verified at ingress, then serialized into the header.
    pub sampling_params: SamplingParams,
    /// Whether the client asked for SSE streaming.
    pub stream: bool,
    /// Logprob / hidden-state options. This path bypasses the Python
    /// `TokenizerManager`, so the ingress replicates its scalar normalization
    /// (defaults applied in `to_header_msgpack`) before the scheduler sees them.
    pub return_logprob: Option<bool>,
    pub logprob_start_len: Option<i64>,
    pub top_logprobs_num: Option<i64>,
    /// This request's `token_ids_logprob` ids, fanned out by `into_requests` and
    /// collapsed to `None` when empty (the scheduler branches on `is not None`).
    pub token_ids_logprob: Option<TokenIds>,
    pub return_hidden_states: Option<bool>,
    /// Decode logprob token ids to text in each `[logprob, token_id, text]` tuple
    /// (default leaves the text slot null). Deliberately NOT in the scheduler
    /// header — Python's `TokenizedGenerateReqInput` has no such field either;
    /// it is consumed on the way out, by `register_detok` → `DetokMsg::Register`
    /// → the shard's `decode_logprob_texts`.
    pub return_text_in_logprobs: Option<bool>,
    /// Multimodal inputs, carried opaquely (URL / base64 / path / nested lists —
    /// any JSON shape the Python `GenerateReqInput` accepts). Consumed by the
    /// Encoding stage, which ships them to the MM worker pool; never read by
    /// the tokenizer or serialized onto the scheduler header. Boxed: absent on
    /// the common text-only request, so it shouldn't grow every `Request` moved
    /// between stages.
    pub mm: Option<Box<MmData>>,
}

/// The three opaque multimodal fields of one request (see [`GenerateRequest::mm`]).
#[derive(Debug, Default)]
pub struct MmData {
    pub image_data: Option<rmpv::Value>,
    pub video_data: Option<rmpv::Value>,
    pub audio_data: Option<rmpv::Value>,
}

impl GenerateRequest {
    /// True when the client already supplied token ids → skip tokenization.
    pub fn already_tokenized(&self) -> bool {
        self.input_ids.as_ref().is_some_and(|v| !v.is_empty())
    }

    /// True when the request carries any usable multimodal payload — the Rust
    /// mirror of Python `GenerateReqInput.contains_mm_input()` /
    /// `has_valid_data`: `null` and (recursively) empty lists don't count.
    pub fn has_multimodal(&self) -> bool {
        self.mm.as_ref().is_some_and(|mm| {
            mm_value_present(&mm.image_data)
                || mm_value_present(&mm.video_data)
                || mm_value_present(&mm.audio_data)
        })
    }

    /// Serialize the fields the MM worker pool needs for this request: a
    /// msgpack array `[text, input_ids, image_data, video_data, audio_data]`
    /// (decoded by [`super::mm_payload::parse`] in the native pipeline).
    pub fn to_mm_payload_msgpack(&self) -> Result<Bytes, Error> {
        use rmpv::Value;
        let text_val = match &self.text {
            Some(t) => Value::from(t.as_str()),
            None => Value::Nil,
        };
        let input_ids_val = match &self.input_ids {
            Some(ids) => Value::Array(ids.iter().map(|&i| Value::from(i)).collect()),
            None => Value::Nil,
        };
        let mm_field = |f: fn(&MmData) -> &Option<Value>| -> Value {
            self.mm
                .as_deref()
                .and_then(|m| f(m).clone())
                .unwrap_or(Value::Nil)
        };
        let arr = Value::Array(vec![
            text_val,
            input_ids_val,
            mm_field(|m| &m.image_data),
            mm_field(|m| &m.video_data),
            mm_field(|m| &m.audio_data),
        ]);
        let mut buf = Vec::new();
        rmpv::encode::write_value(&mut buf, &arr).map_err(|e| Error::Codec(e.to_string()))?;
        Ok(Bytes::from(buf))
    }

    pub fn encode_header(&self) -> Result<Bytes, Error> {
        TokenizedGenerateReqInput::from(self).encode()
    }

    /// `input_ids` widened to raw little-endian int64 bytes (the scheduler's
    /// `array("q")` columnar cell — rides the ingress ring outside msgpack). Empty
    /// when not tokenized.
    pub fn encode_data_buf(&self) -> Bytes {
        let ids = self.input_ids.as_deref().unwrap_or(&[]);
        let mut buf = Vec::with_capacity(ids.len() * 8);
        for &id in ids {
            buf.extend_from_slice(&(id as i64).to_le_bytes());
        }
        Bytes::from(buf)
    }
}

/// Fan one scalar-or-list option out to `n` per-item values: absent → `None`
/// each, a scalar broadcasts, a list must match the batch size.
/// Bytes a broadcast value costs per clone. Only the heap matters — the inline
/// part is bounded by the type.
trait HeapBytes {
    fn heap_bytes(&self) -> usize;
}
impl HeapBytes for bool {
    fn heap_bytes(&self) -> usize {
        0
    }
}
impl HeapBytes for i64 {
    fn heap_bytes(&self) -> usize {
        0
    }
}
impl HeapBytes for String {
    fn heap_bytes(&self) -> usize {
        self.len()
    }
}
impl HeapBytes for TokenIds {
    fn heap_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<i32>()
    }
}

/// Reject a broadcast whose clones would exceed [`MAX_BROADCAST_CLONE_BYTES`].
fn check_broadcast_budget(per_clone: usize, n: usize, name: &str) -> Result<(), Error> {
    // `n == 1` is not a broadcast — there is one value and one prompt, so nothing
    // is duplicated. Charging it here rejected ordinary single requests with a
    // message about a batch they never sent.
    if n > 1 && per_clone.saturating_mul(n) > MAX_BROADCAST_CLONE_BYTES {
        return Err(Error::Validation(format!(
            "{name} ({per_clone} bytes) broadcast to {n} prompts would allocate more \
             than the {MAX_BROADCAST_CLONE_BYTES}-byte limit; send a shorter {name} \
             or a smaller batch"
        )));
    }
    Ok(())
}

fn fan_out<T: OneOrManyItem + Clone + HeapBytes>(
    value: Option<OneOrMany<T>>,
    n: usize,
    name: &str,
) -> Result<Vec<Option<T>>, Error> {
    match value {
        None => Ok(vec![None; n]),
        Some(OneOrMany::One(v)) => {
            // Same budget as the `sampling_params` broadcast: `vec![Some(v); n]`
            // deep-clones client data once per prompt, so a 16 MiB
            // `token_ids_logprob` fanned to 4096 prompts is ~64 GiB — an
            // allocation failure, which `abort()`s the scheduler process.
            check_broadcast_budget(v.heap_bytes(), n, name)?;
            Ok(vec![Some(v); n])
        }
        Some(OneOrMany::Many(v)) => {
            if v.len() != n {
                return Err(Error::Validation(format!(
                    "{name} list length {} does not match batch size {n}",
                    v.len()
                )));
            }
            Ok(v.into_iter().map(Some).collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn requests(body: &str) -> Result<(Vec<GenerateRequest>, bool), Error> {
        serde_json::from_str::<GenerateBody>(body)
            .unwrap()
            .into_requests()
    }

    /// Scalar `text` → one item, not a batch (response stays a single object).
    #[test]
    fn scalar_text_is_single() {
        let (ps, is_batch) = requests(r#"{"text": "hi"}"#).unwrap();
        assert!(!is_batch);
        assert_eq!(ps.len(), 1);
        assert_eq!(ps[0].text.as_deref(), Some("hi"));
    }

    /// List `text` → batch (even length 1); each prompt becomes its own payload.
    #[test]
    fn list_text_is_batch() {
        let (ps, is_batch) = requests(r#"{"text": ["a", "b"]}"#).unwrap();
        assert!(is_batch);
        assert_eq!(ps.len(), 2);
        assert_eq!(ps[0].text.as_deref(), Some("a"));
        assert_eq!(ps[1].text.as_deref(), Some("b"));

        let (ps, is_batch) = requests(r#"{"text": ["only"]}"#).unwrap();
        assert!(is_batch, "single-element list is still a batch");
        assert_eq!(ps.len(), 1);
    }

    /// Scalar `sampling_params` broadcasts to every item; a list maps per item.
    #[test]
    fn sampling_params_broadcast_and_per_item() {
        let (ps, _) =
            requests(r#"{"text": ["a", "b"], "sampling_params": {"temperature": 0.5}}"#).unwrap();
        assert_eq!(ps[0].sampling_params, ps[1].sampling_params);
        assert_eq!(ps[0].sampling_params.temperature, 0.5);

        let (ps, _) = requests(
            r#"{"text": ["a", "b"], "sampling_params": [{"temperature": 0.1}, {"temperature": 0.9}]}"#,
        )
        .unwrap();
        assert_ne!(ps[0].sampling_params, ps[1].sampling_params);
    }

    /// A per-item `sampling_params` list whose length ≠ batch size is a 400.
    #[test]
    fn sampling_params_length_mismatch_errors() {
        let err = requests(r#"{"text": ["a", "b"], "sampling_params": [{}]}"#).unwrap_err();
        assert!(err.to_string().contains("length"), "{err}");
    }

    /// `input_ids` batch (list of lists) fans out; scalar (list of ints) is single.
    #[test]
    fn input_ids_scalar_vs_batch() {
        let (ps, is_batch) = requests(r#"{"input_ids": [1, 2, 3]}"#).unwrap();
        assert!(!is_batch);
        assert_eq!(ps[0].input_ids, Some(vec![1, 2, 3]));

        let (ps, is_batch) = requests(r#"{"input_ids": [[1, 2], [3]]}"#).unwrap();
        assert!(is_batch);
        assert_eq!(ps.len(), 2);
        assert_eq!(ps[1].input_ids, Some(vec![3]));
    }

    /// Both / neither of text+input_ids is a 400; the wire still rejects unknowns.
    #[test]
    fn split_validates_inputs() {
        assert!(requests(r#"{"text": "a", "input_ids": [1]}"#).is_err());
        assert!(requests(r#"{"stream": true}"#).is_err());
        assert!(
            serde_json::from_str::<GenerateBody>(r#"{"text": "hi", "bogus": 1}"#).is_err(),
            "GenerateBody must deny unknown fields"
        );
        // Python's `GenerateReqInput` has no top-level `n` either (parallel
        // sampling reads `sampling_params.n`) — but FastAPI builds it from a
        // pydantic dataclass, which *ignores* the extra key, where
        // `deny_unknown_fields` makes it a 400. Deliberately stricter: a
        // top-level `n` is a client bug that Python swallows silently.
        assert!(serde_json::from_str::<GenerateBody>(r#"{"text": "a", "n": 1}"#).is_err());
        // Parallel sampling is rejected where Python reads it — in the params,
        // at normalization (the ingress step), not here.
        let (mut ps, _) = requests(r#"{"text": "a", "sampling_params": {"n": 2}}"#).unwrap();
        assert!(ps[0].sampling_params.normalize(false, None).is_err());
    }

    /// Client-supplied rid semantics mirror Python's `_normalize_batch`: a
    /// single string passes through for a single request, fans out as
    /// `{rid}_{i}` for a batch, and a list must match the batch length. An
    /// absent rid is minted here, one uuid per item.
    #[test]
    fn split_rid_matches_python_normalize() {
        let (ps, _) = requests(r#"{"text": "a", "rid": "r"}"#).unwrap();
        assert_eq!(ps[0].rid, "r");

        let (ps, _) = requests(r#"{"text": ["a", "b"], "rid": "base"}"#).unwrap();
        assert_eq!(ps[0].rid, "base_0");
        assert_eq!(ps[1].rid, "base_1");

        let (ps, _) = requests(r#"{"text": ["a", "b"], "rid": ["x", "y"]}"#).unwrap();
        assert_eq!(ps[0].rid, "x");
        assert_eq!(ps[1].rid, "y");

        let (ps, _) = requests(r#"{"text": ["a", "b"]}"#).unwrap();
        // Absent → `into_requests` mints one uuid per item, all distinct.
        assert_eq!(ps[0].rid.len(), 32);
        assert_ne!(ps[0].rid, ps[1].rid);

        assert!(
            requests(r#"{"text": ["a", "b"], "rid": ["x"]}"#).is_err(),
            "rid list length must match batch size"
        );
        assert!(
            requests(r#"{"text": "a", "rid": ["x"]}"#).is_err(),
            "rid list with a single (non-batch) prompt is rejected"
        );
    }

    /// The native `bench_serving` payload (a `GenerateReqInput` superset) parses:
    /// its `lora_path`/`return_routed_experts` are accepted-but-ignored and a
    /// `null` `image_data` means "no multimodal input", so `split` succeeds
    /// while the real fields survive.
    #[test]
    fn accepts_bench_serving_payload() {
        let (ps, is_batch) = requests(
            r#"{"text": "hi", "sampling_params": {"max_new_tokens": 8},
                "stream": true, "lora_path": null, "return_logprob": false,
                "return_routed_experts": false, "logprob_start_len": -1,
                "image_data": null}"#,
        )
        .unwrap();
        assert!(!is_batch);
        assert_eq!(ps.len(), 1);
        assert_eq!(ps[0].text.as_deref(), Some("hi"));
        assert!(ps[0].stream);
        assert!(!ps[0].has_multimodal());
    }

    /// Mm columns fan out per the Python `_normalize_{image,video}_data` rules: a
    /// single request passes the raw value through; a batch broadcasts a scalar
    /// image as `[img]` per item (`[[img]]*n`), maps a list per item, requires
    /// matching lengths, and treats `null`/`[]` as absent.
    #[test]
    fn split_mm_fanout_matches_python_normalize() {
        let image_of = |p: &GenerateRequest| p.mm.as_ref().unwrap().image_data.clone().unwrap();

        // Single request: raw value passes through untouched.
        let (ps, _) = requests(r#"{"text": "a", "image_data": "http://x/i.jpg"}"#).unwrap();
        assert_eq!(image_of(&ps[0]).as_str(), Some("http://x/i.jpg"));
        assert!(ps[0].has_multimodal());

        // Batch + scalar image: broadcast, wrapped as a one-image list per item.
        let (ps, _) = requests(r#"{"text": ["a", "b"], "image_data": "u"}"#).unwrap();
        for p in &ps {
            assert_eq!(image_of(p).as_array().unwrap().len(), 1);
            assert!(p.has_multimodal());
        }

        // Batch + per-item list: element i goes to item i.
        let (ps, _) = requests(r#"{"text": ["a", "b"], "image_data": ["u1", "u2"]}"#).unwrap();
        assert_eq!(image_of(&ps[0]).as_str(), Some("u1"));
        assert_eq!(image_of(&ps[1]).as_str(), Some("u2"));

        // Batch + wrong-length list is a 400.
        assert!(requests(r#"{"text": ["a", "b"], "image_data": ["u1"]}"#).is_err());

        // null / [] mean "no multimodal input".
        let (ps, _) = requests(r#"{"text": "a", "image_data": null}"#).unwrap();
        assert!(!ps[0].has_multimodal());
        let (ps, _) = requests(r#"{"text": "a", "image_data": []}"#).unwrap();
        assert!(!ps[0].has_multimodal());

        // Batch + scalar video: broadcast bare (not wrapped), per Python
        // `_normalize_video_data`.
        let (ps, _) = requests(r#"{"text": ["a", "b"], "video_data": "v"}"#).unwrap();
        let video = ps[1].mm.as_ref().unwrap().video_data.clone().unwrap();
        assert_eq!(video.as_str(), Some("v"));
        assert!(ps[1].has_multimodal());
    }

    /// The mm payload for the MM worker pool is a positional msgpack array
    /// `[text, input_ids, image_data, video_data, audio_data]`.
    #[test]
    fn mm_payload_shape() {
        let (ps, _) =
            requests(r#"{"text": "hi", "image_data": ["u1", "u2"], "audio_data": "a"}"#).unwrap();
        let payload = ps[0].to_mm_payload_msgpack().unwrap();
        let val = rmpv::decode::read_value(&mut &payload[..]).unwrap();
        let arr = val.as_array().unwrap();
        assert_eq!(arr.len(), 5);
        assert_eq!(arr[0].as_str(), Some("hi"));
        assert!(arr[1].is_nil());
        assert_eq!(arr[2].as_array().unwrap().len(), 2);
        assert!(arr[3].is_nil());
        assert_eq!(arr[4].as_str(), Some("a"));
    }

    /// The body limit is disabled, so an unbounded batch turns a small body into an
    /// unbounded allocation. Worse, broadcasting `sampling_params` deep-clones the
    /// client's `custom_params`/`logit_bias`/`stop` once per prompt, so the blow-up
    /// is quadratic in the body — and a Rust allocation failure `abort()`s the
    /// scheduler process rather than raising. Both the count and the product are
    /// capped before any column is built.
    #[test]
    fn oversized_batches_are_rejected_before_allocating() {
        let texts: Vec<String> = (0..MAX_BATCH_SIZE + 1).map(|i| i.to_string()).collect();
        let body = serde_json::json!({ "text": texts }).to_string();
        let err = requests(&body).unwrap_err().to_string();
        assert!(err.contains("exceeds the maximum"), "{err}");

        // At the cap it is accepted.
        let texts: Vec<String> = (0..MAX_BATCH_SIZE).map(|i| i.to_string()).collect();
        let (reqs, _) = requests(&serde_json::json!({ "text": texts }).to_string()).unwrap();
        assert_eq!(reqs.len(), MAX_BATCH_SIZE);

        // A small batch with a huge broadcast `custom_params` is the quadratic case:
        // few items, but each clone carries the whole blob.
        let blob = "x".repeat(1 << 20); // 1 MiB
        let body = serde_json::json!({
            "text": vec!["hi"; 200],
            "sampling_params": { "custom_params": { "k": blob } },
        })
        .to_string();
        let err = requests(&body).unwrap_err().to_string();
        assert!(err.contains("would allocate more than"), "{err}");
    }

    /// `token_ids_logprob` mirrors Python `_normalize_batch`'s nested-structure
    /// branch: a flat list broadcasts to every prompt, a list of lists is
    /// per-prompt. Regression — the whole value used to be cloned to every item.
    #[test]
    fn token_ids_logprob_broadcasts_flat_and_splits_nested() {
        let (ps, _) = requests(r#"{"text": ["a", "b"], "token_ids_logprob": [1, 2]}"#).unwrap();
        assert_eq!(ps[0].token_ids_logprob, Some(vec![1, 2]));
        assert_eq!(ps[1].token_ids_logprob, Some(vec![1, 2]));

        let (ps, _) =
            requests(r#"{"text": ["a", "b"], "token_ids_logprob": [[1], [2, 3]]}"#).unwrap();
        assert_eq!(ps[0].token_ids_logprob, Some(vec![1]));
        assert_eq!(ps[1].token_ids_logprob, Some(vec![2, 3]));

        let err = requests(r#"{"text": ["a", "b"], "token_ids_logprob": [[1]]}"#).unwrap_err();
        assert!(
            err.to_string().contains("does not match batch size"),
            "{err}"
        );

        let (ps, _) = requests(r#"{"text": ["a", "b"]}"#).unwrap();
        assert_eq!(ps[0].token_ids_logprob, None);
    }

    /// An empty `token_ids_logprob` means "none requested" and must reach the
    /// scheduler as None, whose guards are `x is not None` — `Some([])` enters the
    /// token-ids-logprob path and computes nothing. The collapse is per item, so it
    /// covers every shape: Python only collapses the outer value
    /// (`if not self.token_ids_logprob`, io_struct.py:439,612) and passes inner
    /// empties through its nested branch verbatim.
    #[test]
    fn empty_token_ids_logprob_collapses_to_none() {
        let (ps, _) = requests(r#"{"text": "a", "token_ids_logprob": []}"#).unwrap();
        assert_eq!(ps[0].token_ids_logprob, None);

        let (ps, _) = requests(r#"{"text": ["a", "b"], "token_ids_logprob": []}"#).unwrap();
        assert!(ps.iter().all(|p| p.token_ids_logprob.is_none()));

        // Nested, every item empty — Python would ship four `[]`s here.
        let (ps, _) =
            requests(r#"{"text": ["a", "b", "c", "d"], "token_ids_logprob": [[], [], [], []]}"#)
                .unwrap();
        assert!(ps.iter().all(|p| p.token_ids_logprob.is_none()));

        // Nested and mixed: only the empty cell collapses.
        let (ps, _) = requests(r#"{"text": ["a", "b"], "token_ids_logprob": [[], [7]]}"#).unwrap();
        assert_eq!(ps[0].token_ids_logprob, None);
        assert_eq!(ps[1].token_ids_logprob, Some(vec![7]));

        // A non-empty list is untouched.
        let (ps, _) = requests(r#"{"text": "a", "token_ids_logprob": [7]}"#).unwrap();
        assert_eq!(ps[0].token_ids_logprob, Some(vec![7]));
    }

    /// The logprob/hidden options take Python's batch form too
    /// (`Union[List[T], T]`): a scalar broadcasts, a list is per-prompt.
    #[test]
    fn logprob_options_broadcast_scalar_and_split_list() {
        let (ps, _) =
            requests(r#"{"text": ["a", "b"], "return_logprob": true, "top_logprobs_num": 3}"#)
                .unwrap();
        assert_eq!(ps[0].return_logprob, Some(true));
        assert_eq!(ps[1].top_logprobs_num, Some(3));

        let (ps, _) = requests(
            r#"{"text": ["a", "b"], "return_logprob": [true, false],
                "logprob_start_len": [0, 2], "return_hidden_states": [false, true]}"#,
        )
        .unwrap();
        assert_eq!(ps[0].return_logprob, Some(true));
        assert_eq!(ps[1].return_logprob, Some(false));
        assert_eq!(ps[0].logprob_start_len, Some(0));
        assert_eq!(ps[1].logprob_start_len, Some(2));
        assert_eq!(ps[1].return_hidden_states, Some(true));

        let err = requests(r#"{"text": ["a", "b"], "return_logprob": [true]}"#).unwrap_err();
        assert!(
            err.to_string().contains("does not match batch size"),
            "{err}"
        );
    }

    /// `{"input_ids": []}` parses as one prompt with no ids, so the batch-size
    /// guard misses it; Python's `_determine_batch_size` raises "input_ids cannot
    /// be empty." Regression — it used to reach the tokenizer with no text.
    #[test]
    fn empty_input_ids_is_rejected() {
        let err = requests(r#"{"input_ids": []}"#).unwrap_err();
        assert!(
            err.to_string().contains("input_ids cannot be empty"),
            "{err}"
        );

        let err = requests(r#"{"input_ids": [[1, 2], []]}"#).unwrap_err();
        assert!(err.to_string().contains("cannot be empty"), "{err}");

        assert!(requests(r#"{"input_ids": [1, 2]}"#).is_ok());
        assert!(requests(r#"{"input_ids": [[1], [2]]}"#).is_ok());
    }

    /// Two items in one request cannot share an rid: they would map to the same
    /// `RidHash` slot. Mirrors Python `_validate_rid_uniqueness`.
    #[test]
    fn duplicate_rids_within_one_request_are_rejected() {
        let err = requests(r#"{"text": ["a", "b"], "rid": ["x", "x"]}"#).unwrap_err();
        assert!(err.to_string().contains("duplicate request IDs"), "{err}");

        assert!(requests(r#"{"text": ["a", "b"], "rid": ["x", "y"]}"#).is_ok());
        let (ps, _) = requests(r#"{"text": ["a", "b"], "rid": "x"}"#).unwrap();
        assert_eq!(ps[0].rid, "x_0");
        assert_eq!(ps[1].rid, "x_1");
    }
}
