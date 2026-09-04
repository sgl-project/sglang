//! The `/generate` request path: the HTTP body and its per-request fan-out
//! ([`GenerateBody`] → [`GenerateRequest`]s).

use std::collections::HashSet;
use std::sync::LazyLock;

use bytes::Bytes;
use itertools::izip;
use serde::Deserialize;

use super::io_struct::{ControlRequest, TokenizedGenerateReqInput};
use super::response::ResponseSink;
use super::sampling::{SamplingParams, SamplingParamsInput};
use super::types::{OneOrMany, OneOrManyItem, TokenIds};
use crate::message::ids::{Rid, UNIQ_SUFFIX_LEN};
use crate::utils::fsm::RequestState;
use crate::utils::{environ::env_i64, error::Error};

/// Hard cap on how many scheduler requests one `/generate` HTTP call may expand
/// into. Every column below is allocated per item before anything is dispatched,
/// so this bounds the work — and the resident memory — a single call can ask for.
///
/// NOT a concurrency limit: it is a pure function of the body being parsed, so
/// separate HTTP calls never interact with it.
///
/// Read once from `SGLANG_MAX_BATCH_REQS_PER_HTTP_REQ` (registered in
/// `python/sglang/srt/environ.py`, which owns the default). Memoized because the
/// value is process-static — Python sets it before launching this server — and a
/// per-request `env::var` would take a lock on the hot path for a constant.
static MAX_BATCH_REQS_PER_HTTP_REQ: LazyLock<i64> =
    LazyLock::new(|| env_i64("SGLANG_MAX_BATCH_REQS_PER_HTTP_REQ", 4096));

fn batch_size_exceeds_limit(batch_size: usize, limit: i64) -> bool {
    limit >= 0 && batch_size as u128 > limit as u128
}

/// Hard cap on the total bytes a broadcast value may clone into the batch (see
/// the `One` arms of the fan-out).
const MAX_BROADCAST_CLONE_BYTES: usize = 64 << 20;

/// Live heap per byte of serialized JSON. Measured across shapes at 1.0–7.0×
/// (`serde_json::Value` pays for enum tags, `String` headers and map nodes that
/// the wire form does not); 8 is the ceiling of that range, not a worst case.
const JSON_TO_HEAP_FACTOR: usize = 8;

/// The `/generate` wire body before batch splitting: `text`/`input_ids`/`sampling_params`
/// each scalar-or-list, fanned into per-request [`GenerateRequest`]s by
/// [`into_requests`](GenerateBody::into_requests).
///
/// Unknown keys are IGNORED, matching Python: FastAPI builds `GenerateReqInput`
/// as a pydantic dataclass, which drops extras. `deny_unknown_fields` here turned
/// every `GenerateReqInput` field this server has not ported — `priority`,
/// `extra_key`, `session_id`, `session_params`, `return_sampling_mask`,
/// `custom_logit_processor`, and ~40 more — into a 400, so a client that worked
/// against the Python server broke against this one. The cost of dropping it is
/// that a typo (`temperature`) is silently ignored rather than reported; that is
/// the same trade Python already makes.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct GenerateBody {
    /// Optional client-supplied request id(s): a single string (a batch fans it
    /// out as `{rid}_{i}`, mirroring Python `_normalize_batch`) or one per item.
    pub rid: Option<OneOrMany<String>>,
    pub text: Option<OneOrMany<String>>,
    pub input_ids: Option<OneOrMany<TokenIds>>,
    #[serde(default)]
    pub stream: bool,
    /// One params object (broadcast) or a list of them (per item); see
    /// [`SamplingParamsInput`].
    pub sampling_params: Option<SamplingParamsInput>,
    /// Logprob / hidden-state options: a scalar broadcasts to every prompt, a
    /// list is per-prompt (Python `_normalize_logprob_params`).
    pub return_logprob: Option<OneOrMany<bool>>,
    pub logprob_start_len: Option<OneOrMany<i64>>,
    pub top_logprobs_num: Option<OneOrMany<i64>>,
    /// Token ids to report logprobs for: one list (broadcast to every prompt) or
    /// one list per prompt, mirroring Python's
    /// `Union[List[int], List[List[int]]]` fan-out in `_normalize_batch`.
    pub token_ids_logprob: Option<OneOrMany<TokenIds>>,
    pub return_hidden_states: Option<OneOrMany<bool>>,
    /// Scalar-only in Python too (`return_text_in_logprobs: bool`).
    pub return_text_in_logprobs: Option<bool>,
    // PD-disaggregation routing, injected per request by the PD router
    // (mini_lb / sgl-model-gateway): a scalar for a single prompt, one-per-item
    // lists for a batch. Elements are nullable (`List[Optional[...]]` in
    // Python) — the router sends `bootstrap_port: [null, …]` when deferring to
    // the scheduler's `--disaggregation-bootstrap-port` default.
    pub bootstrap_host: Option<OneOrMany<Option<String>>>,
    pub bootstrap_port: Option<OneOrMany<Option<i64>>>,
    /// `bootstrap_room` fits in i64: the PD routers draw it from `[0, 2^63)`.
    pub bootstrap_room: Option<OneOrMany<Option<i64>>>,
    pub bootstrap_pair_key: Option<OneOrMany<Option<String>>>,
    pub decode_tp_size: Option<OneOrMany<Option<i64>>>,
    /// DP routing hints — per-request scalars even for batches, as in Python.
    pub routed_dp_rank: Option<i64>,
    pub disagg_prefill_dp_rank: Option<i64>,
    // Multimodal inputs, permissive `Value` so any shape Python's
    // `GenerateReqInput` accepts (URL / base64 / list / list-of-lists) parses.
    // `into_requests` fans them out per the Python
    // `_normalize_{image,video,audio}_data` batch rules.
    pub image_data: Option<rmpv::Value>,
    /// Caller-supplied per-item content hashes (hex) overriding the computed
    /// ones, so an external router's keys align with the prefix cache. Single
    /// requests only: Python declares the batched shapes but `__getitem__` never
    /// forwards them, so a batch is rejected here rather than answered with
    /// hashes it did not ask for.
    pub mm_hashes: Option<rmpv::Value>,
    pub video_data: Option<rmpv::Value>,
    pub audio_data: Option<rmpv::Value>,
}

impl GenerateBody {
    /// Merge operator-provided sampling defaults beneath request values,
    /// matching Python TokenizerManager's preferred/request precedence.
    pub fn apply_preferred_sampling(&mut self, preferred: &serde_json::Value) -> Result<(), Error> {
        match &mut self.sampling_params {
            Some(params) => params.apply_preferred(preferred),
            None => SamplingParamsInput::from_preferred(preferred).map(|params| {
                self.sampling_params = Some(params);
            }),
        }
        .map_err(|e| Error::Validation(format!("invalid preferred_sampling_params: {e}")))
    }

    /// Validate, normalize and fan the body into one [`GenerateRequest`] per
    /// prompt + `is_batch` (list form — a 1-element list is still a batch → JSON
    /// array response). The Rust counterpart of Python
    /// `GenerateReqInput.normalize_batch_and_arguments`; an invalid/inconsistent
    /// batch is [`Error::Validation`], which the handler surfaces with the
    /// variant's own status (400).
    pub fn into_requests(self) -> Result<(Vec<GenerateRequest>, bool, usize), Error> {
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
            bootstrap_host,
            bootstrap_port,
            bootstrap_room,
            bootstrap_pair_key,
            decode_tp_size,
            routed_dp_rank,
            disagg_prefill_dp_rank,
            image_data,
            video_data,
            audio_data,
            mm_hashes,
            // Unported `GenerateReqInput` fields land here and are dropped, as they
            // are on the Python path.
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
        if batch_size_exceeds_limit(declared_n, *MAX_BATCH_REQS_PER_HTTP_REQ) {
            return Err(Error::Validation(format!(
                "batch size {declared_n} exceeds the maximum of {}",
                *MAX_BATCH_REQS_PER_HTTP_REQ
            )));
        }

        // Parallel sampling (`sampling_params.n`). Read and bounded HERE, before a
        // single column is allocated, for the same reason `declared_n` is: the
        // columns below are `vec![_; n]`, so a body that will be rejected must be
        // rejected first.
        //
        // `n` is `i64` on the wire, so it can arrive negative or astronomically
        // large. `as usize` would turn `-1` into `usize::MAX` and slip past every
        // bound below, and a plain `declared_n * num_samples` would wrap in release
        // builds — hence the explicit `try_from` + `checked_mul`. Same shape as
        // `api_server::openai::completions`, which already caps `prompts * n`.
        let common_n: i64 = match &sampling_params {
            None => 1,
            Some(SamplingParamsInput::One(sp)) => sp.n,
            Some(SamplingParamsInput::Many(v)) => {
                // Python `_handle_parallel_sampling` requires one `n` for the batch.
                let first = v.first().map_or(1, |sp| sp.n);
                if v.iter().any(|sp| sp.n != first) {
                    return Err(Error::Validation(
                        "the same n is required for all entries of sampling_params".into(),
                    ));
                }
                first
            }
        };
        if common_n < 1 {
            return Err(Error::Validation(format!(
                "n must be at least 1, got {common_n}"
            )));
        }
        let num_samples = usize::try_from(common_n)
            .map_err(|_| Error::Validation(format!("n is too large: {common_n}")))?;
        match declared_n.checked_mul(num_samples) {
            Some(total) if total <= *MAX_BATCH_REQS_PER_HTTP_REQ => {}
            _ => {
                return Err(Error::Validation(format!(
                    "prompt count times n exceeds the maximum of {}",
                    *MAX_BATCH_REQS_PER_HTTP_REQ
                )));
            }
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
        let mut sps: Vec<SamplingParams> = match sampling_params {
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
        // The base requests carry `n = 1`: parallel sampling is a frontend fan-out
        // (`expand_parallel_samples`), and the scheduler never reads `n` — Python's
        // does not either. `SamplingParams::verify` keeps rejecting anything else as
        // an internal invariant, so a request that skipped the fan-out cannot reach
        // the ring silently claiming n samples and get one.
        for sp in &mut sps {
            sp.n = 1;
        }

        // rid: absent → mint one uuid per item here, so every request carries its
        // final rid from this point on; a single string fans out as `{rid}_{i}`
        // for a batch (Python `_normalize_batch`); a list is per-item.
        //
        // Every CLIENT-supplied rid goes through `Rid::from_client`, which appends a
        // uniquifier so two concurrent requests sharing an rid cannot collide on the
        // detok table. `client_facing` strips it back off for `meta_info.id`, so the
        // client sees exactly what it sent. Minted rids (`Rid::default`) are already
        // unique and are left bare.
        let rids: Vec<Rid> = match rid {
            None => (0..n).map(|_| Rid::default()).collect(),
            Some(OneOrMany::One(r)) if !is_batch => vec![Rid::from_client(&r)],
            Some(OneOrMany::One(r)) => {
                check_broadcast_budget(r.len(), n, "rid")?;
                // Uniquify AFTER the `_{i}` split, so the split index stays part of
                // the rid the client gets back.
                (0..n)
                    .map(|i| Rid::from_client(&format!("{r}_{i}")))
                    .collect()
            }
            Some(OneOrMany::Many(v)) => {
                if !is_batch || v.len() != n {
                    return Err(Error::Validation(format!(
                        "rid list length {} does not match batch size {n}",
                        v.len()
                    )));
                }
                // Python `_validate_rid_uniqueness`. `from_client` below would make
                // even these unique, so this is parity rather than safety: Python
                // 400s a request that names one id twice, and echoing the same
                // `meta_info.id` on two entries of one batch response is useless to
                // the client regardless. Checked on the RAW strings, before the
                // uniquifier hides the duplication.
                {
                    let mut seen = HashSet::with_capacity(v.len());
                    let duplicates: Vec<&String> = v.iter().filter(|r| !seen.insert(*r)).collect();
                    if !duplicates.is_empty() {
                        return Err(Error::Validation(format!(
                            "duplicate request IDs detected within the request: {duplicates:?}"
                        )));
                    }
                }
                v.iter().map(|r| Rid::from_client(r)).collect()
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

        // PD fields fan out like Python `_normalize_bootstrap_params`: scalars
        // broadcast — except a scalar `bootstrap_room`, which becomes `room + i`
        // (each item needs a distinct room; rooms are the P↔D pairing key).
        // `fan_out` yields `Option<Option<T>>` for these nullable elements
        // (outer: absent, inner: an explicit `null` element) — flatten, both
        // mean "not set" downstream.
        let bootstrap_hosts = flatten_column(fan_out(bootstrap_host, n, "bootstrap_host")?);
        let bootstrap_ports = flatten_column(fan_out(bootstrap_port, n, "bootstrap_port")?);
        let bootstrap_rooms = match bootstrap_room {
            // `wrapping_add`, not `checked_`: rooms are drawn from `[0, 2^63)`,
            // so a batch can only overflow by starting within `n` of `i64::MAX`
            // — and distinct-but-wrapped still pairs P↔D, where saturating
            // would collide every item onto one room.
            Some(OneOrMany::One(Some(room))) => {
                (0..n).map(|i| Some(room.wrapping_add(i as i64))).collect()
            }
            other => flatten_column(fan_out(other, n, "bootstrap_room")?),
        };
        let bootstrap_pair_keys =
            flatten_column(fan_out(bootstrap_pair_key, n, "bootstrap_pair_key")?);
        let decode_tp_sizes = flatten_column(fan_out(decode_tp_size, n, "decode_tp_size")?);
        // `mm_hashes` has no batch form: honoring it only here would give the two
        // servers different prefix-cache keys for the same body. Reject instead of
        // dropping it silently as Python does — the field exists to align a
        // caller's keys, so ignoring it returns subtly wrong ones.
        if is_batch && mm_value_present(&mm_hashes) {
            return Err(Error::Validation(
                "mm_hashes is not supported for batch requests; send one request per prompt".into(),
            ));
        }
        // Multimodal columns; see `split_mm_column` for the Python parity rules.
        let images = split_mm_column(image_data, n, is_batch, MmBroadcast::WrapInList)
            .map_err(|e| Error::Validation(format!("image_data: {e}")))?;
        let videos = split_mm_column(video_data, n, is_batch, MmBroadcast::AsIs)
            .map_err(|e| Error::Validation(format!("video_data: {e}")))?;
        let audios = split_mm_column(audio_data, n, is_batch, MmBroadcast::AsIs)
            .map_err(|e| Error::Validation(format!("audio_data: {e}")))?;

        // Every column above is exactly `n` long, so zip them by value: each
        // request takes ownership of its cell, with no indexing or bounds checks.
        let mut requests: Vec<GenerateRequest> = izip!(
            rids,
            texts,
            id_lists,
            sps,
            return_logprobs,
            logprob_start_lens,
            top_logprobs_nums,
            tid_logprobs,
            return_hidden,
            bootstrap_hosts,
            bootstrap_ports,
            bootstrap_rooms,
            bootstrap_pair_keys,
            decode_tp_sizes,
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
                bootstrap_host,
                bootstrap_port,
                bootstrap_room,
                bootstrap_pair_key,
                decode_tp_size,
                image_data,
                video_data,
                audio_data,
            )| GenerateRequest {
                rid,
                text,
                input_ids,
                // Native text prompts keep the post-processor specials; the
                // chat flow sets this explicitly.
                skip_special_tokens: false,
                sampling_params,
                stream,
                // Python `GenerateReqInput` defaults.
                return_logprob: return_logprob.unwrap_or(false),
                logprob_start_len: logprob_start_len.unwrap_or(-1),
                top_logprobs_num: top_logprobs_num.unwrap_or(0),
                // `Some` here means "these ids were requested", so an empty list
                // collapses to None.
                token_ids_logprob: token_ids_logprob.filter(|ids| !ids.is_empty()),
                return_sampling_mask: false, // TODO: port Python's `return_sampling_mask`
                return_hidden_states: return_hidden_states.unwrap_or(false),
                return_text_in_logprobs,
                bootstrap_host,
                bootstrap_port,
                bootstrap_room,
                bootstrap_pair_key,
                decode_tp_size,
                routed_dp_rank,
                disagg_prefill_dp_rank,
                mm: pack_mm(image_data, video_data, audio_data),
            },
        )
        .collect();
        // Single requests only (batches rejected above). Malformed entries are
        // dropped here and warned about in `mm::apply_caller_hashes`, never a 400.
        if !is_batch
            && let (Some(rmpv::Value::Array(vals)), Some(req)) = (mm_hashes, requests.first_mut())
            && let Some(mm) = req.mm.as_deref_mut()
        {
            mm.mm_hashes = vals
                .iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect();
        }
        Ok((requests, is_batch, num_samples))
    }
}

/// Box the per-item mm values, `None` when the item has none — the common
/// text-only case keeps `GenerateRequest` slim.
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
        ..Default::default()
    }))
}

/// How a scalar mm value broadcasts across a batch: images become a one-image
/// list per item (`[[img]] * num` in Python `_normalize_image_data`),
/// video/audio broadcast bare (`[v] * num` in `_normalize_video_data`).
#[derive(Clone, Copy)]
enum MmBroadcast {
    WrapInList,
    AsIs,
}

/// Fan one mm field into per-item values, mirroring Python's
/// `_normalize_{image,video,audio}_data`:
///   * `None` / empty list → `None` for every item;
///   * single request → the raw value passes through (the processor wraps a
///     non-list into a one-element list);
///   * batch + scalar → broadcast to every item, per `MmBroadcast`;
///   * batch + list → per item, length must equal the batch size.
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
        scalar => {
            // A broadcast deep-clones once per prompt — same blow-up as
            // sampling_params above, so bound the product before any clone.
            check_broadcast_budget(scalar.heap_bytes(), n, "value").map_err(|e| e.to_string())?;
            Ok(match broadcast {
                MmBroadcast::WrapInList => vec![Some(rmpv::Value::Array(vec![scalar])); n],
                MmBroadcast::AsIs => vec![Some(scalar); n],
            })
        }
    }
}

/// One request handed to the MM worker pool: the rid to correlate the result,
/// plus the owned inputs from [`GenerateRequest::take_mm_work`].
#[derive(Debug)]
pub struct MmRequest {
    pub rid: Rid,
    pub work: MmWorkItem,
}

/// The parked request's fields the MM worker owns; converted to the driver input
/// by [`crate::multi_modality::payload::to_mm_input`].
#[derive(Debug, Default)]
pub struct MmWorkItem {
    pub text: Option<String>,
    pub input_ids: Option<Vec<i32>>,
    pub image_data: Option<rmpv::Value>,
    pub video_data: Option<rmpv::Value>,
    pub audio_data: Option<rmpv::Value>,
    /// See [`MmData::prefetched`].
    pub prefetched: Vec<Bytes>,
    /// See [`GenerateBody::mm_hashes`].
    pub mm_hashes: Vec<String>,
}

/// Whether an optional mm field counts as multimodal input, via the same
/// `value_present` the MM worker's payload parser uses.
fn mm_value_present(v: &Option<rmpv::Value>) -> bool {
    v.as_ref()
        .is_some_and(crate::multi_modality::payload::value_present)
}

/// The owned request as it travels request stages (single owner, so `state` is
/// mutated lock-free). Common fields here; variant data in [`RequestKind`].
#[derive(Debug)]
pub struct Request {
    /// Client-visible request id (uuid hex) — what the scheduler wire and
    /// `meta_info.id` carry.
    pub rid: Rid,
    pub state: RequestState,
    /// Back-channel to the client connection for response frames.
    pub sink: ResponseSink,
    /// Discriminant + variant body (generate vs control).
    pub kind: RequestKind,
}

/// One to_scheduler channel entry, split columnar: the scalar `header` (msgpack, `input_ids`
/// omitted) + the raw int64 `ids` cell, so the big tensor never goes through msgpack.
#[derive(Debug)]
pub struct SchedulerRequest {
    pub header: Bytes,
    pub ids: Bytes,
}

/// Request variant — selects the request branch, scheduler wire message, and
/// response shape. Each owns its body, so generate/control fields stay type-separate.
#[derive(Debug)]
pub enum RequestKind {
    /// `/generate`: tokenize (if needed) then push a `TokenizedGenerateReqInput`.
    Generate(Box<GenerateRequest>),
    /// A control endpoint (e.g. `/server_info`, `/health`): no tokenization, and
    /// the response is a single non-streamed JSON result.
    Control(Box<ControlRequest>),
    /// Internal service call: decode a complete token-id sequence to text. Walks
    /// the same FSM as every request (validate → register → Queued), but the
    /// stage that answers it is the detok shard itself, never the scheduler
    /// ring; the result arrives on the registered sink as one `Data` payload
    /// (the raw UTF-8 text). First caller: `/v1/completions` `echo` for
    /// token-id prompts; a future `/detokenize` parity endpoint maps 1:1.
    Detokenize { token_ids: TokenIds },
}

/// A single in-flight `/generate` request (per-item from
/// [`GenerateBody::into_requests`]),
/// serialized to the scheduler wire once tokenized (see `to_header_msgpack`). Not a
/// wire type — built by `into_requests`/handlers, never (de)serialized; `input_ids` is
/// client-supplied or filled by the Tokenizer stage.
#[derive(Debug, Default)]
pub struct GenerateRequest {
    /// This item's final rid: the client's (normalized per item by `into_requests`) or a
    /// uuid minted there when none was sent. A [`Rid`], not a `String`: the wire
    /// forms stay textual (`GenerateBody` on the way in, `TokenizedGenerateReqInput`
    /// on the way out) but every in-process carrier names the type.
    ///
    /// Duplicates *within* one request are rejected by `into_requests` (Python
    /// `_validate_rid_uniqueness`). A collision with a *concurrent* request's rid
    /// cannot arise: [`Rid::from_client`] appends a uniquifier to every
    /// client-supplied rid, so this value is unique for the process's lifetime and
    /// only [`client_facing`](Rid::client_facing) is ever shown back.
    ///
    /// This diverges from Python, which 400s the second request ("Duplicate request
    /// ID detected"). Serving both is the friendlier answer and strictly safer —
    /// what the rejection protected against was one request evicting the other's
    /// detok sink, which is now unrepresentable.
    pub rid: Rid,
    pub text: Option<String>,
    /// Client-supplied token ids, or filled by the Tokenizer stage.
    pub input_ids: Option<TokenIds>,
    /// Template-rendered prompts (chat) already contain their role/special
    /// tokens, so the tokenizer pool strips the auto-added BOS/EOS prefix —
    /// the Rust analogue of Python's `add_special_tokens=False` at the
    /// chat-template encode site (`serving_chat._encode_messages`). Consumed
    /// by the pool before the header is built; never reaches the scheduler wire.
    pub skip_special_tokens: bool,
    /// Sampling params (defaults when the client sent none, as in Python);
    /// normalized + verified, then serialized into the header.
    pub sampling_params: SamplingParams,
    /// Whether the client asked for SSE streaming.
    pub stream: bool,
    /// Logprob / hidden-state options. This path bypasses the Python
    /// `TokenizerManager`, so `into_requests` replicates its scalar
    /// normalization. Resolved to concrete values THERE rather than at the wire
    /// boundary: an `Option` surviving past construction invites two call sites
    /// to disagree about what absent means, and only the wire knew the answer.
    /// The defaults are `GenerateReqInput`'s own.
    pub return_logprob: bool,
    pub logprob_start_len: i64,
    pub top_logprobs_num: i64,
    /// This request's `token_ids_logprob` ids, fanned out by `into_requests` and
    /// collapsed to `None` when empty (the scheduler branches on `is not None`).
    pub token_ids_logprob: Option<TokenIds>,
    pub return_sampling_mask: bool,
    pub return_hidden_states: bool,
    /// Decode logprob token ids to text in each `[logprob, token_id, text]` tuple
    /// (default leaves the text slot null). Deliberately NOT in the scheduler
    /// header — Python's `TokenizedGenerateReqInput` has no such field either;
    /// it is consumed on the way out, by `register_detok` → `DetokMsg::Register`
    /// → the shard's `decode_logprob_texts`.
    pub return_text_in_logprobs: Option<bool>,
    /// PD-disaggregation routing, forwarded verbatim to the scheduler (which
    /// fills a `None` port from `--disaggregation-bootstrap-port` and 400-aborts
    /// a room-less request in PD mode).
    pub bootstrap_host: Option<String>,
    pub bootstrap_port: Option<i64>,
    pub bootstrap_room: Option<i64>,
    pub bootstrap_pair_key: Option<String>,
    pub decode_tp_size: Option<i64>,
    /// DP routing hints. The embedded server is rank-0-only (no DP controller),
    /// so these are pure passthrough for the scheduler/LB protocol.
    pub routed_dp_rank: Option<i64>,
    pub disagg_prefill_dp_rank: Option<i64>,
    /// Multimodal inputs, carried opaquely. Consumed by the Encoding stage,
    /// which ships them to the MM worker pool; never read by the tokenizer or
    /// serialized onto the scheduler header. Boxed so the common text-only
    /// request doesn't grow every `Request` moved between stages.
    pub mm: Option<Box<MmData>>,
}

/// The opaque multimodal fields of one request (see [`GenerateRequest::mm`]).
///
/// Constructed directly only by tests: `api_server::prefetch` fills its
/// `prefetched` field, everything else gets it packed inside a `GenerateRequest`.
///
/// `Clone` is a DEEP copy, and parallel sampling depends on that: `take_mm_work`
/// MOVES these fields out of the request, so two siblings sharing one `MmData`
/// would leave whichever reached `Encoding` second with nothing. `prefetched`'s
/// `Bytes` are refcounted handles to immutable buffers, so sharing those is safe
/// and the media is not duplicated — only the handle array is.
#[derive(Debug, Default, Clone)]
pub struct MmData {
    pub image_data: Option<rmpv::Value>,
    pub video_data: Option<rmpv::Value>,
    pub audio_data: Option<rmpv::Value>,
    /// Bytes of `image_data`'s I/O-backed sources, resolved by
    /// `api_server::prefetch` in `payload::io_sources` order so MM workers
    /// never block on I/O. Out-of-band: the values above stay as the client
    /// sent them.
    pub prefetched: Vec<bytes::Bytes>,
    /// See [`GenerateBody::mm_hashes`]; applied by the MM worker.
    pub mm_hashes: Vec<String>,
}

impl GenerateRequest {
    /// One parallel-sampling sibling of `self`: same prompt, same params, its own
    /// `rid`.
    ///
    /// Written out rather than `#[derive(Clone)]` on `GenerateRequest`: every
    /// stage below MOVES its request (`take_mm_work` empties the mm fields, the
    /// `izip!` fan-out consumes each column by value), so the one deep copy in the
    /// pipeline should be visible at the call site instead of available anywhere.
    ///
    /// `sampling_params.n` is already 1 — `into_requests` set it on the base — so
    /// this does not touch it. `bootstrap_room` is copied VERBATIM, deliberately:
    /// see `expand_parallel_samples`.
    fn fork(&self, rid: Rid) -> Self {
        Self {
            rid,
            text: self.text.clone(),
            input_ids: self.input_ids.clone(),
            skip_special_tokens: self.skip_special_tokens,
            sampling_params: self.sampling_params.clone(),
            stream: self.stream,
            return_logprob: self.return_logprob,
            logprob_start_len: self.logprob_start_len,
            top_logprobs_num: self.top_logprobs_num,
            token_ids_logprob: self.token_ids_logprob.clone(),
            return_sampling_mask: self.return_sampling_mask,
            return_hidden_states: self.return_hidden_states,
            return_text_in_logprobs: self.return_text_in_logprobs,
            bootstrap_host: self.bootstrap_host.clone(),
            bootstrap_port: self.bootstrap_port,
            bootstrap_room: self.bootstrap_room,
            bootstrap_pair_key: self.bootstrap_pair_key.clone(),
            decode_tp_size: self.decode_tp_size,
            routed_dp_rank: self.routed_dp_rank,
            disagg_prefill_dp_rank: self.disagg_prefill_dp_rank,
            mm: self.mm.clone(),
        }
    }

    /// True when the client already supplied token ids → skip tokenization.
    pub fn already_tokenized(&self) -> bool {
        self.input_ids.as_ref().is_some_and(|v| !v.is_empty())
    }

    /// True when the request carries a usable multimodal payload — the mirror of
    /// Python `GenerateReqInput.contains_mm_input()`.
    pub fn has_multimodal(&self) -> bool {
        self.mm.as_ref().is_some_and(|mm| {
            mm_value_present(&mm.image_data)
                || mm_value_present(&mm.video_data)
                || mm_value_present(&mm.audio_data)
        })
    }

    /// Carve out the MM worker's inputs: `text` is cloned (the scheduler header
    /// still needs it), `input_ids` is taken (the expanded ids replace it), and
    /// the mm values move wholesale.
    pub fn take_mm_work(&mut self) -> MmWorkItem {
        let mut work = MmWorkItem {
            text: self.text.clone(),
            input_ids: self.input_ids.take(),
            ..Default::default()
        };
        if let Some(m) = self.mm.as_deref_mut() {
            work.image_data = m.image_data.take();
            work.video_data = m.video_data.take();
            work.audio_data = m.audio_data.take();
            work.prefetched = std::mem::take(&mut m.prefetched);
            work.mm_hashes = std::mem::take(&mut m.mm_hashes);
        }
        work
    }

    pub fn encode_header(&self) -> Result<Bytes, Error> {
        TokenizedGenerateReqInput::from(self).encode()
    }

    /// `input_ids` widened to raw little-endian int64 bytes (the scheduler's
    /// `array("q")` columnar cell — rides the to-scheduler channel outside
    /// msgpack). Empty when not tokenized.
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
impl<T: HeapBytes> HeapBytes for Option<T> {
    fn heap_bytes(&self) -> usize {
        self.as_ref().map_or(0, HeapBytes::heap_bytes)
    }
}
impl HeapBytes for rmpv::Value {
    fn heap_bytes(&self) -> usize {
        use rmpv::Value;
        const NODE: usize = std::mem::size_of::<rmpv::Value>();
        match self {
            Value::String(s) => s.as_bytes().len(),
            Value::Binary(b) => b.len(),
            Value::Ext(_, b) => b.len(),
            Value::Array(items) => items.iter().map(|v| NODE + v.heap_bytes()).sum(),
            Value::Map(entries) => entries
                .iter()
                .map(|(k, v)| 2 * NODE + k.heap_bytes() + v.heap_bytes())
                .sum(),
            _ => 0,
        }
    }
}

/// Collapse `fan_out`'s nullable-element output: outer `None` (field absent /
/// scalar broadcast of nothing) and inner `None` (an explicit `null` list
/// element) both mean "not set".
fn flatten_column<T>(column: Vec<Option<Option<T>>>) -> Vec<Option<T>> {
    column.into_iter().map(Option::flatten).collect()
}

/// Expand each base request into `n` parallel-sampling siblings, prompt-major
/// (`p0s0, p0s1, …, p1s0, …`) so the response array matches Python's
/// `_handle_batch_request` ordering and the OpenAI adapters' `prompt_index * n +
/// sample_index`.
///
/// Called from the HTTP handler AFTER `prefetch_all`, never from
/// [`GenerateBody::into_requests`]: prefetch resolves media per request, so
/// expanding first would download the same URL `n` times.
///
/// `bootstrap_room` is carried over UNCHANGED. It looks like each sibling needs a
/// distinct room (it is the P↔D pairing key), but Python gives all `n` samples of
/// one prompt the same room — `_normalize_bootstrap_params` computes
/// `batch_size * n` rooms and then `_handle_batch_request` only ever reads the
/// first `batch_size` of them. Diverging here would break drop-in parity;
/// copying it is only sound because the caller rejects `n > 1` under PD.
///
/// Only [`expand_parallel_samples`] is exported, not [`GenerateRequest::fork`]:
/// a bare `fork` invites forgetting the fresh rid, and two live requests sharing
/// an rid would collide on the detok table.
pub(crate) fn expand_parallel_samples(
    base: Vec<GenerateRequest>,
    n: usize,
) -> Vec<GenerateRequest> {
    if n <= 1 {
        return base;
    }
    let mut out = Vec::with_capacity(base.len().saturating_mul(n));
    for req in base {
        // The last sibling consumes `req` instead of cloning it, so an `n`-way
        // expansion performs `n - 1` deep copies rather than `n`.
        for s in 0..n - 1 {
            let rid = Rid::from_client(&format!("{}_{s}", req.rid.client_facing()));
            out.push(req.fork(rid));
        }
        let rid = Rid::from_client(&format!("{}_{}", req.rid.client_facing(), n - 1));
        out.push(GenerateRequest { rid, ..req });
    }
    out
}

/// Bytes one request would cost to clone, counting every variable-length field
/// [`GenerateRequest::fork`] copies or regenerates.
///
/// Missing a field here is a hole, not an inaccuracy: a tiny prompt with a huge
/// `token_ids_logprob` (or a very long rid) sails past both the request-count cap
/// and any prompt-size intuition.
///
/// Container overhead counts. `impl HeapBytes for String` returns only `len()`,
/// so summing it over a `Vec<String>` scores a million empty strings as zero —
/// while the handle array alone is ~24 MB. This mirrors what
/// `impl HeapBytes for rmpv::Value` already does for arrays (`NODE + ..` per
/// element).
fn clone_bytes(req: &GenerateRequest, sample_index_digits: usize) -> usize {
    // Each sibling mints `{client_rid}_{i}` and `Rid::from_client` appends a
    // fixed-width uniquifier, so the rid is rebuilt per sibling, not shared.
    let rid = req
        .rid
        .client_facing()
        .len()
        .saturating_add(1 + sample_index_digits + UNIQ_SUFFIX_LEN);
    // Serialized-JSON × measured heap factor is how the broadcast path already
    // sizes `SamplingParams`; it is the only estimator that reaches
    // `custom_params` (a `serde_json::Value`, which `HeapBytes` does not cover).
    let sampling = serde_json::to_string(&req.sampling_params)
        .map_or(0, |s| s.len())
        .saturating_mul(JSON_TO_HEAP_FACTOR);
    let mm = req.mm.as_deref().map_or(0, |m| {
        m.image_data
            .heap_bytes()
            .saturating_add(m.video_data.heap_bytes())
            .saturating_add(m.audio_data.heap_bytes())
            // Handles only: the `Bytes` payloads are refcounted, so the media
            // itself is shared rather than duplicated.
            .saturating_add(m.prefetched.len().saturating_mul(size_of::<Bytes>()))
            .saturating_add(vec_string_bytes(&m.mm_hashes))
    });
    rid.saturating_add(req.text.heap_bytes())
        .saturating_add(req.input_ids.heap_bytes())
        .saturating_add(req.token_ids_logprob.heap_bytes())
        .saturating_add(sampling)
        .saturating_add(req.bootstrap_host.heap_bytes())
        .saturating_add(req.bootstrap_pair_key.heap_bytes())
        .saturating_add(mm)
}

/// `Vec<String>`: the handle array plus the contents (see [`clone_bytes`]).
fn vec_string_bytes(v: &[String]) -> usize {
    v.len()
        .saturating_mul(size_of::<String>())
        .saturating_add(v.iter().map(String::len).sum::<usize>())
}

/// Reject a parallel-sampling expansion whose clones would exceed
/// [`MAX_BROADCAST_CLONE_BYTES`]. The request-count cap in `into_requests` does
/// NOT imply this one: 4096 copies of a 10 MB prompt is ~40 GB, and a failed Rust
/// allocation calls `abort()`, which is uncatchable and takes the scheduler
/// process down with the frontend.
pub(crate) fn check_parallel_sample_budget(
    payloads: &[GenerateRequest],
    num_samples: usize,
) -> Result<(), Error> {
    if num_samples <= 1 {
        return Ok(());
    }
    let digits = (num_samples - 1).to_string().len();
    let per_clone = payloads
        .iter()
        .map(|req| clone_bytes(req, digits))
        .fold(0usize, usize::saturating_add);
    check_broadcast_budget(per_clone, num_samples, "parallel samples")
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

    /// Vocab size for tests that aren't about the vocab bound (see
    /// `sampling::tests::TEST_VOCAB`).
    const TEST_VOCAB: u64 = 1000;

    /// The base requests + `is_batch`, dropping `num_samples` — most tests here
    /// predate parallel sampling and only care about the fan-out columns.
    fn requests(body: &str) -> Result<(Vec<GenerateRequest>, bool), Error> {
        requests_n(body).map(|(reqs, is_batch, _)| (reqs, is_batch))
    }

    /// Full `into_requests` output, for the parallel-sampling tests.
    fn requests_n(body: &str) -> Result<(Vec<GenerateRequest>, bool, usize), Error> {
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

    /// Both / neither of text+input_ids is a 400.
    #[test]
    fn split_validates_inputs() {
        assert!(requests(r#"{"text": "a", "input_ids": [1]}"#).is_err());
        assert!(requests(r#"{"stream": true}"#).is_err());
    }

    /// `into_requests` reads `n` but does NOT expand: it returns the BASE requests
    /// plus the sample count, and the handler expands after `prefetch_all`.
    ///
    /// Asserting an expanded length here would be the wrong shape and would push
    /// the expansion back into this function — which is exactly what must not
    /// happen, since prefetch resolves media per request and would then fetch the
    /// same URL `n` times.
    #[test]
    fn into_requests_reads_n_without_expanding() {
        let (mut base, is_batch, num_samples) =
            requests_n(r#"{"text": "a", "sampling_params": {"n": 2}}"#).unwrap();
        assert!(!is_batch, "one prompt stays a single request");
        assert_eq!(base.len(), 1, "into_requests must not expand");
        assert_eq!(num_samples, 2);
        // The base carries n=1, so the invariant in `verify` passes downstream.
        assert_eq!(base[0].sampling_params.n, 1);
        assert!(base[0].sampling_params.normalize(false, TEST_VOCAB).is_ok());

        let expanded = expand_parallel_samples(base, num_samples);
        assert_eq!(expanded.len(), 2, "expansion happens here");
    }

    /// `n` is read from `sampling_params`; a TOP-LEVEL `n` stays ignored (Python's
    /// `GenerateReqInput` has no such field either — see
    /// `unported_generate_req_input_fields_are_ignored`).
    #[test]
    fn top_level_n_is_not_parallel_sampling() {
        let (_, _, num_samples) = requests_n(r#"{"text": "a", "n": 5}"#).unwrap();
        assert_eq!(num_samples, 1, "top-level n must not drive the fan-out");
    }

    /// `n` is `i64` on the wire, so out-of-range values must be rejected rather
    /// than cast: `as usize` turns -1 into `usize::MAX`, and a plain multiply
    /// wraps in release builds — either one slips past every cap below.
    #[test]
    fn out_of_range_n_is_rejected_not_cast() {
        for body in [
            r#"{"text": "a", "sampling_params": {"n": 0}}"#,
            r#"{"text": "a", "sampling_params": {"n": -1}}"#,
            r#"{"text": "a", "sampling_params": {"n": 9223372036854775807}}"#,
        ] {
            assert!(
                requests_n(body).is_err(),
                "{body} must be rejected, not wrapped or cast"
            );
        }
    }

    /// `prompts * n` is capped by the same knob as the batch itself, and the
    /// product is computed with `checked_mul` so it cannot wrap into range.
    #[test]
    fn prompt_count_times_n_is_capped() {
        let cap = *MAX_BATCH_REQS_PER_HTTP_REQ;
        let body = format!(
            r#"{{"text": ["a", "b"], "sampling_params": {{"n": {}}}}}"#,
            cap / 2 + 1
        );
        let err = requests_n(&body).expect_err("2 * (cap/2 + 1) exceeds the cap");
        assert!(err.to_string().contains("exceeds the maximum"), "{err}");
    }

    /// A `sampling_params` LIST must agree on `n` (Python
    /// `_handle_parallel_sampling` raises for the same reason).
    #[test]
    fn sampling_params_list_must_agree_on_n() {
        let body = r#"{"text": ["a", "b"], "sampling_params": [{"n": 2}, {"n": 3}]}"#;
        let err = requests_n(body).expect_err("mismatched n must be rejected");
        assert!(err.to_string().contains("same n"), "{err}");
    }

    // ----- check_parallel_sample_budget -----

    /// A request whose only large field is `f`, ready to weigh.
    fn budget_req(f: impl FnOnce(&mut GenerateRequest)) -> GenerateRequest {
        let mut req = GenerateRequest {
            rid: Rid::from("r".to_string()),
            text: Some("hi".into()),
            ..Default::default()
        };
        f(&mut req);
        req
    }

    fn over_budget(req: GenerateRequest, n: usize) -> bool {
        check_parallel_sample_budget(std::slice::from_ref(&req), n).is_err()
    }

    /// A small prompt fanned out wide is fine — the cap must not reject ordinary
    /// parallel sampling.
    #[test]
    fn budget_allows_small_prompt_with_large_n() {
        assert!(!over_budget(budget_req(|_| {}), 1024));
    }

    /// `n == 1` is not an expansion, so nothing is weighed at all.
    #[test]
    fn budget_is_skipped_for_one_sample() {
        let huge = budget_req(|r| r.text = Some("x".repeat(MAX_BROADCAST_CLONE_BYTES)));
        assert!(!over_budget(huge, 1));
    }

    /// Every variable-length field `fork` copies is weighed. Each of these is a
    /// standalone bypass: the prompt stays tiny, so neither the request-count cap
    /// nor a prompt-size heuristic would catch it.
    #[test]
    fn budget_covers_every_cloned_field() {
        let big = MAX_BROADCAST_CLONE_BYTES / 8;
        let cases: Vec<(&str, GenerateRequest)> = vec![
            ("text", budget_req(|r| r.text = Some("x".repeat(big)))),
            ("rid", budget_req(|r| r.rid = Rid::from("x".repeat(big)))),
            (
                "token_ids_logprob",
                budget_req(|r| r.token_ids_logprob = Some(vec![7; big])),
            ),
            (
                "bootstrap_host",
                budget_req(|r| r.bootstrap_host = Some("x".repeat(big))),
            ),
            (
                "bootstrap_pair_key",
                budget_req(|r| r.bootstrap_pair_key = Some("x".repeat(big))),
            ),
            (
                "custom_params",
                budget_req(|r| {
                    r.sampling_params.custom_params =
                        Some(serde_json::json!({ "k": "x".repeat(big) }));
                }),
            ),
            (
                "image_data",
                budget_req(|r| {
                    r.mm = Some(Box::new(MmData {
                        image_data: Some(rmpv::Value::from("x".repeat(big))),
                        ..Default::default()
                    }));
                }),
            ),
        ];
        for (field, req) in cases {
            assert!(
                over_budget(req, 64),
                "{field} must count toward the clone budget"
            );
        }
    }

    /// `Vec<String>` is weighed by its HANDLE ARRAY plus contents, not contents
    /// alone. A million EMPTY strings sums to zero bytes of content while the
    /// handles alone are ~24 MB — summing `len()` would wave this straight
    /// through. A single long string would pass either way, which is why the
    /// empties are the case that actually locks the hole.
    #[test]
    fn budget_counts_container_handles_not_just_contents() {
        let empties = MAX_BROADCAST_CLONE_BYTES / size_of::<String>();
        let req = budget_req(|r| {
            r.mm = Some(Box::new(MmData {
                mm_hashes: vec![String::new(); empties],
                ..Default::default()
            }));
        });
        assert!(
            over_budget(req, 4),
            "a Vec of empty Strings still costs its handle array"
        );
    }

    /// The reverse hazard: `prefetched` holds refcounted `Bytes`, so cloning a
    /// sibling shares the media instead of duplicating it. Counting the payload
    /// would 400 legitimate multimodal requests — only the handle array is real.
    #[test]
    fn budget_ignores_shared_media_payloads() {
        let media = Bytes::from(vec![0u8; MAX_BROADCAST_CLONE_BYTES]);
        let req = budget_req(|r| {
            r.mm = Some(Box::new(MmData {
                prefetched: vec![media],
                ..Default::default()
            }));
        });
        assert!(
            !over_budget(req, 64),
            "refcounted media must not be charged per sibling"
        );
    }

    // ----- expand_parallel_samples -----

    /// One prompt, `n = 3`: three siblings with distinct `{rid}_{i}` ids, each
    /// carrying `n = 1` (the scheduler never fans out).
    #[test]
    fn expansion_mints_one_sibling_per_sample() {
        let (base, _, num_samples) =
            requests_n(r#"{"text": "a", "rid": "r", "sampling_params": {"n": 3}}"#).unwrap();
        let out = expand_parallel_samples(base, num_samples);

        assert_eq!(out.len(), 3);
        let client_rids: Vec<&str> = out.iter().map(|r| r.rid.client_facing()).collect();
        assert_eq!(client_rids, ["r_0", "r_1", "r_2"]);
        // Internally unique too — the detok table is keyed by the full rid, and two
        // live requests sharing a key would evict each other's sink.
        let uniq: HashSet<&str> = out.iter().map(|r| r.rid.as_str()).collect();
        assert_eq!(uniq.len(), 3, "sibling rids must be internally distinct");
        assert!(out.iter().all(|r| r.sampling_params.n == 1));
    }

    /// Two prompts × `n = 3` come back PROMPT-MAJOR (`p0s0 p0s1 p0s2 p1s0 …`),
    /// matching Python's `_handle_batch_request` order and the OpenAI adapters'
    /// `prompt_index * n + sample_index`.
    #[test]
    fn expansion_is_prompt_major() {
        let (base, _, num_samples) =
            requests_n(r#"{"text": ["a", "b"], "sampling_params": {"n": 3}}"#).unwrap();
        let out = expand_parallel_samples(base, num_samples);

        assert_eq!(out.len(), 6);
        let texts: Vec<&str> = out.iter().map(|r| r.text.as_deref().unwrap()).collect();
        assert_eq!(texts, ["a", "a", "a", "b", "b", "b"]);
    }

    /// `bootstrap_room` is carried over UNCHANGED.
    ///
    /// The tempting "fix" is to make it unique across the flattened index, since
    /// the room is the P↔D pairing key. Python does not: it computes
    /// `batch_size * n` rooms and then only ever reads the first `batch_size`, so
    /// all `n` samples of a prompt share one. Offsetting here would break drop-in
    /// parity; the handler refuses `n > 1` under PD instead.
    #[test]
    fn expansion_leaves_bootstrap_room_untouched() {
        let (base, _, num_samples) = requests_n(
            r#"{"text": ["a", "b"], "bootstrap_room": 100, "sampling_params": {"n": 2}}"#,
        )
        .unwrap();
        // The per-prompt offset the batch fan-out already applied.
        assert_eq!(
            base.iter().map(|r| r.bootstrap_room).collect::<Vec<_>>(),
            [Some(100), Some(101)]
        );
        let out = expand_parallel_samples(base, num_samples);
        assert_eq!(
            out.iter().map(|r| r.bootstrap_room).collect::<Vec<_>>(),
            [Some(100), Some(100), Some(101), Some(101)],
            "siblings share their prompt's room, as in Python"
        );
    }

    /// `n == 1` is a no-op: the base requests pass through untouched, so a plain
    /// `/generate` never pays for the expansion path.
    #[test]
    fn expansion_is_a_noop_for_one_sample() {
        let (base, _, num_samples) = requests_n(r#"{"text": "a", "rid": "r"}"#).unwrap();
        assert_eq!(num_samples, 1);
        let rid_before = base[0].rid.as_str().to_owned();
        let out = expand_parallel_samples(base, num_samples);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].rid.as_str(), rid_before, "rid must not be re-minted");
    }

    /// Multimodal payloads are copied to every sibling — `take_mm_work` MOVES them
    /// out of whichever request reaches `Encoding`, so siblings must not share.
    #[test]
    fn expansion_gives_every_sibling_its_own_mm() {
        let (base, _, num_samples) =
            requests_n(r#"{"text": "a", "image_data": "u", "sampling_params": {"n": 3}}"#).unwrap();
        let mut out = expand_parallel_samples(base, num_samples);
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(GenerateRequest::has_multimodal));

        // Draining one leaves the others intact.
        let _ = out[0].take_mm_work();
        assert!(!out[0].has_multimodal());
        assert!(
            out[1].has_multimodal() && out[2].has_multimodal(),
            "siblings must own independent MmData"
        );
    }

    /// Unported `GenerateReqInput` fields are IGNORED, not rejected.
    ///
    /// These are all real fields on Python's `GenerateReqInput` that this server
    /// has not ported. `deny_unknown_fields` turned every one of them into a 400,
    /// so a client that worked against the Python server broke here — and the
    /// wire-compat fields (`lora_path`, `image_data`, `return_routed_experts`) had
    /// to be declared and dropped by hand just to let `bench_serving` through.
    /// FastAPI's pydantic dataclass drops extras, so ignoring them is the parity
    /// behavior; a typo being silently ignored is the same trade Python makes.
    #[test]
    fn unported_generate_req_input_fields_are_ignored() {
        for field in [
            r#""priority": 3"#,
            r#""extra_key": "k""#,
            r#""session_id": "s""#,
            r#""session_params": {"a": 1}"#,
            r#""return_sampling_mask": true"#,
            r#""custom_logit_processor": "cls""#,
            r#""lora_path": "adapter""#,
            r#""image_data": "base64""#,
            r#""return_routed_experts": true"#,
            r#""bootstrap_host": "h""#,
            // Python has no top-level `n` either, and ignores it just the same.
            r#""n": 1"#,
            r#""totally_made_up": 1"#,
        ] {
            let body = format!(r#"{{"text": "hi", {field}}}"#);
            let (ps, _) = requests(&body)
                .unwrap_or_else(|e| panic!("{field} must be ignored, not rejected: {e}"));
            assert_eq!(ps.len(), 1, "{field}");
            assert_eq!(ps[0].text.as_deref(), Some("hi"), "{field}");
        }
    }

    /// Client-supplied rid semantics mirror Python's `_normalize_batch`: a
    /// single string passes through for a single request, fans out as
    /// `{rid}_{i}` for a batch, and a list must match the batch length. An
    /// absent rid is minted here, one uuid per item.
    ///
    /// Asserted on `client_facing()`, which is what `meta_info.id` echoes: the
    /// internal rid additionally carries the `from_client` uniquifier, and that
    /// suffix must never be visible in the parity-defined shape.
    #[test]
    fn split_rid_matches_python_normalize() {
        let (ps, _) = requests(r#"{"text": "a", "rid": "r"}"#).unwrap();
        assert_eq!(ps[0].rid.client_facing(), "r");

        let (ps, _) = requests(r#"{"text": ["a", "b"], "rid": "base"}"#).unwrap();
        assert_eq!(ps[0].rid.client_facing(), "base_0");
        assert_eq!(ps[1].rid.client_facing(), "base_1");

        let (ps, _) = requests(r#"{"text": ["a", "b"], "rid": ["x", "y"]}"#).unwrap();
        assert_eq!(ps[0].rid.client_facing(), "x");
        assert_eq!(ps[1].rid.client_facing(), "y");

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

    /// Mm columns fan out per Python `_normalize_{image,video}_data`: a single
    /// request passes the raw value through; a batch broadcasts a scalar image as
    /// `[img]` per item, maps a list per item with matching lengths, and treats
    /// `null`/`[]` as absent.
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

    /// A scalar broadcast is budget-checked before the deep clones (16 MiB ×
    /// 4096 prompts would be 64 GiB and an abort); per-item lists clone nothing
    /// and are never charged.
    #[test]
    fn oversized_mm_broadcast_rejected() {
        let big = rmpv::Value::from("x".repeat(MAX_BROADCAST_CLONE_BYTES / 2 + 1));
        let err = split_mm_column(Some(big.clone()), 2, true, MmBroadcast::WrapInList)
            .err()
            .unwrap();
        assert!(err.contains("broadcast"), "{err}");
        // A per-item list of the same total size moves, not clones: accepted.
        let list = rmpv::Value::Array(vec![big, rmpv::Value::from("y")]);
        assert!(split_mm_column(Some(list), 2, true, MmBroadcast::WrapInList).is_ok());
        // Small scalars broadcast fine.
        let small = rmpv::Value::from("u1");
        assert!(split_mm_column(Some(small), 2, true, MmBroadcast::AsIs).is_ok());
    }

    /// `mm_hashes` rides only on single requests (Python `__getitem__`
    /// parity: batches drop it) and moves into the work item.
    #[test]
    fn mm_hashes_single_only() {
        let (mut ps, _) =
            requests(r#"{"text": "a", "image_data": "u", "mm_hashes": ["a1b2", "0xff"]}"#).unwrap();
        assert_eq!(ps[0].mm.as_ref().unwrap().mm_hashes, vec!["a1b2", "0xff"]);
        assert_eq!(ps[0].take_mm_work().mm_hashes, vec!["a1b2", "0xff"]);
        assert!(ps[0].mm.as_ref().unwrap().mm_hashes.is_empty());

        // A batch cannot carry hashes (Python drops them), so it is rejected...
        for body in [
            r#"{"text": ["a", "b"], "image_data": ["u", "v"], "mm_hashes": [["x"], ["y"]]}"#,
            r#"{"text": ["a", "b"], "image_data": ["u", "v"], "mm_hashes": ["x", "y"]}"#,
        ] {
            let err = requests(body).err().unwrap();
            assert!(matches!(err, Error::Validation(_)), "{body}: {err:?}");
        }
        // ...while an absent or empty field is not a payload and must still pass.
        for body in [
            r#"{"text": ["a", "b"], "image_data": ["u", "v"], "mm_hashes": null}"#,
            r#"{"text": ["a", "b"], "image_data": ["u", "v"], "mm_hashes": []}"#,
        ] {
            assert!(requests(body).is_ok(), "{body}");
        }
    }

    /// `take_mm_work` clones `text` (the scheduler header still needs it) and
    /// moves everything the worker owns out of the request.
    #[test]
    fn mm_work_item_takes_owned_fields() {
        let (mut ps, _) =
            requests(r#"{"text": "hi", "image_data": ["u1", "u2"], "audio_data": "a"}"#).unwrap();
        let work = ps[0].take_mm_work();
        assert_eq!(work.text.as_deref(), Some("hi"));
        assert!(work.input_ids.is_none());
        assert_eq!(work.image_data.unwrap().as_array().unwrap().len(), 2);
        assert!(work.video_data.is_none());
        assert_eq!(work.audio_data.unwrap().as_str(), Some("a"));
        // Moved out, not cloned; `text` survives for the header.
        assert!(ps[0].mm.as_ref().unwrap().image_data.is_none());
        assert_eq!(ps[0].text.as_deref(), Some("hi"));
    }

    /// The body limit is disabled, so an unbounded batch turns a small body into an
    /// unbounded allocation. Worse, broadcasting `sampling_params` deep-clones the
    /// client's `custom_params`/`logit_bias`/`stop` once per prompt, so the blow-up
    /// is quadratic in the body — and a Rust allocation failure `abort()`s the
    /// scheduler process rather than raising. Both the count and the product are
    /// capped before any column is built.
    #[test]
    fn oversized_batches_are_rejected_before_allocating() {
        let cap = usize::try_from(*MAX_BATCH_REQS_PER_HTTP_REQ).unwrap();
        let texts: Vec<String> = (0..cap + 1).map(|i| i.to_string()).collect();
        let body = serde_json::json!({ "text": texts }).to_string();
        let err = requests(&body).unwrap_err().to_string();
        assert!(err.contains("exceeds the maximum"), "{err}");

        // At the cap it is accepted.
        let texts: Vec<String> = (0..cap).map(|i| i.to_string()).collect();
        let (reqs, _) = requests(&serde_json::json!({ "text": texts }).to_string()).unwrap();
        assert_eq!(reqs.len(), cap);

        // A small batch with a huge broadcast `custom_params` is the quadratic case:
        // few items, but each clone carries the whole blob. The item count is a
        // literal because this half asserts the BYTE budget, not the item cap —
        // it therefore assumes the default `SGLANG_MAX_BATCH_REQS_PER_HTTP_REQ`, since
        // a cap below 200 would trip the item check first and report that instead.
        let blob = "x".repeat(1 << 20); // 1 MiB
        let body = serde_json::json!({
            "text": vec!["hi"; 200],
            "sampling_params": { "custom_params": { "k": blob } },
        })
        .to_string();
        let err = requests(&body).unwrap_err().to_string();
        assert!(err.contains("would allocate more than"), "{err}");
    }

    #[test]
    fn negative_batch_limit_disables_the_item_cap() {
        assert!(!batch_size_exceeds_limit(usize::MAX, -1));
        assert!(batch_size_exceeds_limit(11, 10));
        assert!(!batch_size_exceeds_limit(10, 10));
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
        assert!(ps[0].return_logprob);
        assert_eq!(ps[1].top_logprobs_num, 3);

        let (ps, _) = requests(
            r#"{"text": ["a", "b"], "return_logprob": [true, false],
                "logprob_start_len": [0, 2], "return_hidden_states": [false, true]}"#,
        )
        .unwrap();
        assert!(ps[0].return_logprob);
        assert!(!ps[1].return_logprob);
        assert_eq!(ps[0].logprob_start_len, 0);
        assert_eq!(ps[1].logprob_start_len, 2);
        assert!(ps[1].return_hidden_states);

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

    /// Two items in one request cannot share an rid. Mirrors Python
    /// `_validate_rid_uniqueness` — and it must be checked on the RAW strings,
    /// because `Rid::from_client` would otherwise make the duplicates distinct and
    /// the client would get two response entries carrying the same `meta_info.id`.
    #[test]
    fn duplicate_rids_within_one_request_are_rejected() {
        let err = requests(r#"{"text": ["a", "b"], "rid": ["x", "x"]}"#).unwrap_err();
        assert!(err.to_string().contains("duplicate request IDs"), "{err}");

        assert!(requests(r#"{"text": ["a", "b"], "rid": ["x", "y"]}"#).is_ok());
        let (ps, _) = requests(r#"{"text": ["a", "b"], "rid": "x"}"#).unwrap();
        assert_eq!(ps[0].rid.client_facing(), "x_0");
        assert_eq!(ps[1].rid.client_facing(), "x_1");
    }

    /// The collision this whole scheme exists to prevent: two CONCURRENT requests
    /// naming the same rid. They must end up with different internal `Rid`s — the
    /// detok table is keyed on it, and `Register` is an insert-overwrite, so equal
    /// rids would evict the first client's sink and deliver its remaining chunks to
    /// the second's connection. Both still see their own rid echoed back.
    #[test]
    fn concurrent_requests_sharing_an_rid_get_distinct_internal_rids() {
        let (a, _) = requests(r#"{"text": "a", "rid": "same"}"#).unwrap();
        let (b, _) = requests(r#"{"text": "b", "rid": "same"}"#).unwrap();
        assert_ne!(
            a[0].rid, b[0].rid,
            "a shared client rid must not become a shared internal rid"
        );
        assert_eq!(a[0].rid.client_facing(), "same");
        assert_eq!(b[0].rid.client_facing(), "same");
    }

    /// PD bootstrap fields fan out like Python `_normalize_bootstrap_params`:
    /// scalars broadcast, except a scalar `bootstrap_room` which becomes
    /// `room + i` (each batch item needs a distinct room — rooms are the P↔D
    /// pairing key); lists are per-item and must match the batch length.
    #[test]
    fn bootstrap_fields_fan_out() {
        let (ps, _) = requests(
            r#"{"text": ["a", "b"], "bootstrap_host": "h", "bootstrap_port": 8998,
                "bootstrap_room": 7, "routed_dp_rank": 1}"#,
        )
        .unwrap();
        for (i, p) in ps.iter().enumerate() {
            assert_eq!(p.bootstrap_host.as_deref(), Some("h"));
            assert_eq!(p.bootstrap_port, Some(8998));
            assert_eq!(p.bootstrap_room, Some(7 + i as i64));
            assert_eq!(p.routed_dp_rank, Some(1));
        }

        let (ps, _) = requests(
            r#"{"text": ["a", "b"], "bootstrap_host": ["h1", "h2"],
                "bootstrap_room": [10, 20]}"#,
        )
        .unwrap();
        assert_eq!(ps[0].bootstrap_host.as_deref(), Some("h1"));
        assert_eq!(ps[1].bootstrap_host.as_deref(), Some("h2"));
        assert_eq!(ps[0].bootstrap_room, Some(10));
        assert_eq!(ps[1].bootstrap_room, Some(20));

        let err = requests(r#"{"text": ["a", "b"], "bootstrap_room": [1, 2, 3]}"#).unwrap_err();
        assert!(err.to_string().contains("bootstrap_room"), "{err}");
    }

    /// The PD router (mini_lb) and PD-warmup payload shapes must parse. The
    /// router sends `bootstrap_port: [null, …]` when no port was configured
    /// (the scheduler fills its default) — null list elements must parse.
    #[test]
    fn accepts_pd_router_and_warmup_payloads() {
        let (ps, _) = requests(
            r#"{"text": ["a", "b"], "bootstrap_host": ["h", "h"],
                "bootstrap_port": [null, null],
                "bootstrap_room": [123456789, 987654321]}"#,
        )
        .unwrap();
        assert_eq!(ps[0].bootstrap_host.as_deref(), Some("h"));
        assert_eq!(ps[0].bootstrap_port, None);
        assert_eq!(ps[1].bootstrap_room, Some(987654321));

        let (ps, is_batch) = requests(
            r#"{"sampling_params": {"temperature": 0.0, "max_new_tokens": 8,
                                    "ignore_eos": true},
                "bootstrap_host": "2.2.2.2", "bootstrap_room": 0,
                "input_ids": [10, 11, 12, 13], "routed_dp_rank": 0}"#,
        )
        .unwrap();
        assert!(!is_batch);
        assert_eq!(ps[0].bootstrap_host.as_deref(), Some("2.2.2.2"));
        assert_eq!(ps[0].bootstrap_room, Some(0));
        assert_eq!(ps[0].routed_dp_rank, Some(0));
    }
}
