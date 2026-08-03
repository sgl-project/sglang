//! The `/generate` request path: the HTTP body and its per-request fan-out
//! ([`GenerateBody`] → [`GenerateRequest`]s), the variant bodies, and the
//! scheduler ingress encodings (`TokenizedGenerateReqInput` header,
//! control/abort, `IngressMsg`).

use std::collections::HashSet;
use std::sync::LazyLock;

use bytes::Bytes;
use itertools::izip;
use serde::Deserialize;

use super::io_struct::{ControlRequest, TokenizedGenerateReqInput};
use super::{OneOrMany, OneOrManyItem, SamplingParams, SamplingParamsInput, TokenIds};
use crate::environ::env_u64;
use crate::error::Error;
use crate::ids::Rid;

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
static MAX_BATCH_REQS_PER_HTTP_REQ: LazyLock<usize> =
    LazyLock::new(|| env_u64("SGLANG_MAX_BATCH_REQS_PER_HTTP_REQ", 4096) as usize);

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
    // PD-disaggregation routing, injected per request by the PD router
    // (mini_lb / sgl-model-gateway): a scalar for a single prompt, one-per-item
    // lists for a batch. Elements are nullable (`List[Optional[...]]` in
    // Python) — the router sends `bootstrap_port: [null, …]` when deferring to
    // the scheduler's `--disaggregation-bootstrap-port` default.
    #[serde(default)]
    pub bootstrap_host: Option<OneOrMany<Option<String>>>,
    #[serde(default)]
    pub bootstrap_port: Option<OneOrMany<Option<i64>>>,
    /// `bootstrap_room` fits in i64: the PD routers draw it from `[0, 2^63)`.
    #[serde(default)]
    pub bootstrap_room: Option<OneOrMany<Option<i64>>>,
    #[serde(default)]
    pub bootstrap_pair_key: Option<OneOrMany<Option<String>>>,
    #[serde(default)]
    pub decode_tp_size: Option<OneOrMany<Option<i64>>>,
    /// DP routing hints — per-request scalars even for batches, as in Python.
    #[serde(default)]
    pub routed_dp_rank: Option<i64>,
    #[serde(default)]
    pub disagg_prefill_dp_rank: Option<i64>,
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
        if declared_n > *MAX_BATCH_REQS_PER_HTTP_REQ {
            return Err(Error::Validation(format!(
                "batch size {declared_n} exceeds the maximum of {}",
                *MAX_BATCH_REQS_PER_HTTP_REQ
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
        prefetched: Vec::new(),
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
        scalar => {
            // A scalar broadcast deep-clones the value once per prompt — the
            // same abort-on-allocation-failure blow-up as sampling_params
            // above. Bound the product before any clone.
            check_broadcast_budget(scalar.heap_bytes(), n, "value")
                .map_err(|e| e.to_string())?;
            Ok(match broadcast {
                MmBroadcast::WrapInList => vec![Some(rmpv::Value::Array(vec![scalar])); n],
                MmBroadcast::AsIs => vec![Some(scalar); n],
            })
        }
    }
}

/// One request handed to the MM worker pool: the rid to correlate the result
/// plus the typed, owned inputs from [`GenerateRequest::take_mm_work`] — no
/// serialization within the process.
#[derive(Debug)]
pub struct MmRequest {
    pub rid: crate::ids::Rid,
    pub work: MmWorkItem,
}

/// The parked request's fields the MM worker owns, converted to the driver
/// input by [`super::mm_payload::to_mm_input`].
#[derive(Debug, Default)]
pub struct MmWorkItem {
    pub text: Option<String>,
    pub input_ids: Option<Vec<i32>>,
    pub image_data: Option<rmpv::Value>,
    pub video_data: Option<rmpv::Value>,
    pub audio_data: Option<rmpv::Value>,
    /// See [`MmData::prefetched`].
    pub prefetched: Vec<Bytes>,
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
    /// normalized + verified at ingress, then serialized into the header.
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
    /// Bytes of `image_data`'s I/O-backed sources (URLs and file paths),
    /// resolved by `api_server::prefetch` (in `mm_payload::io_sources` order)
    /// so MM workers never block on I/O. Out-of-band: the opaque values above
    /// stay exactly as the client sent them.
    pub prefetched: Vec<bytes::Bytes>,
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

    /// Carve out the MM worker's inputs. `text` is cloned — the scheduler
    /// header still needs it; `input_ids` is taken — the expanded ids replace
    /// it when the worker finishes; the mm values move wholesale.
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
        }
        work
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

    /// Both / neither of text+input_ids is a 400.
    #[test]
    fn split_validates_inputs() {
        assert!(requests(r#"{"text": "a", "input_ids": [1]}"#).is_err());
        assert!(requests(r#"{"stream": true}"#).is_err());
        // Parallel sampling is rejected where Python reads it — in the params,
        // at normalization (the ingress step), not here.
        let (mut ps, _) = requests(r#"{"text": "a", "sampling_params": {"n": 2}}"#).unwrap();
        assert!(ps[0].sampling_params.normalize(false, TEST_VOCAB).is_err());
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

    /// A scalar mm value broadcast to a batch is budget-checked before the
    /// deep clones (16 MiB × 4096 prompts would be 64 GiB and an abort);
    /// per-item lists clone nothing and are never charged.
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
        let texts: Vec<String> = (0..*MAX_BATCH_REQS_PER_HTTP_REQ + 1)
            .map(|i| i.to_string())
            .collect();
        let body = serde_json::json!({ "text": texts }).to_string();
        let err = requests(&body).unwrap_err().to_string();
        assert!(err.contains("exceeds the maximum"), "{err}");

        // At the cap it is accepted.
        let texts: Vec<String> = (0..*MAX_BATCH_REQS_PER_HTTP_REQ)
            .map(|i| i.to_string())
            .collect();
        let (reqs, _) = requests(&serde_json::json!({ "text": texts }).to_string()).unwrap();
        assert_eq!(reqs.len(), *MAX_BATCH_REQS_PER_HTTP_REQ);

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
