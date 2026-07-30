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
use super::sampling::TOP_K_ALL;
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PdBootstrap {
    pub host: String,
    pub port: u16,
    pub room: u64,
    pub attempt_id: String,
    pub batch_index: u32,
}

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
    #[allow(dead_code)]
    pub return_sampling_mask: Option<serde_json::Value>,
    #[serde(default)]
    pub return_text_in_logprobs: Option<bool>,
    /// Top-level compatibility field. Upstream ignores it in favor of
    /// `sampling_params.n`; PD validation retains it so an unsupported request
    /// cannot be silently accepted.
    #[serde(default)]
    pub n: Option<serde_json::Value>,
    /// Gateway-owned PD identity. These four fields are all-or-none. Scalar
    /// requests require scalar values; batch requests require exact-length
    /// arrays so no item can inherit another item's Room identity.
    #[serde(default)]
    pub bootstrap_host: Option<OneOrMany<String>>,
    #[serde(default)]
    pub bootstrap_port: Option<OneOrMany<u16>>,
    #[serde(default)]
    pub bootstrap_room: Option<OneOrMany<u64>>,
    #[serde(default)]
    pub bootstrap_attempt_id: Option<OneOrMany<String>>,

    // These upstream-compatible fields remain ignored on the ordinary path.
    // Retaining their arbitrary JSON shape lets PD fail closed without changing
    // the non-PD parser's permissive unknown-field behavior.
    #[serde(default)]
    #[allow(dead_code)]
    pub lora_path: Option<serde_json::Value>,
    #[serde(default)]
    #[allow(dead_code)]
    pub return_routed_experts: Option<serde_json::Value>,
    #[serde(default)]
    #[allow(dead_code)]
    pub image_data: Option<serde_json::Value>,
    #[serde(default)]
    #[allow(dead_code)]
    pub custom_logit_processor: Option<serde_json::Value>,
}

impl GenerateBody {
    /// Fail-closed request-level portion of the frozen PD support matrix.
    ///
    /// Type/range violations are request-invalid; valid capabilities outside
    /// the MVP (random sampling, logprobs, grammar, LoRA, multimodal, etc.) are
    /// unsupported. Startup-level model/topology checks live in `ServerArgs`.
    pub fn validate_pd_support(&self) -> Result<(), crate::pd::room::PdReason> {
        use crate::pd::room::PdReason;

        if one_or_many_bool(&self.return_logprob)
            || one_or_many_nonzero(&self.top_logprobs_num)
            || self.token_ids_logprob.is_some()
            || one_or_many_bool(&self.return_hidden_states)
            || pd_json_flag(&self.return_sampling_mask)?
            || pd_json_flag(&self.return_routed_experts)?
            || pd_json_present(&self.lora_path)
            || pd_json_present(&self.image_data)
            || pd_json_present(&self.custom_logit_processor)
        {
            return Err(PdReason::Unsupported);
        }
        if let Some(value) = self.n.as_ref().filter(|value| !value.is_null()) {
            match value.as_i64() {
                Some(1) => {}
                Some(_) => return Err(PdReason::Unsupported),
                None => return Err(PdReason::RequestInvalid),
            }
        }

        let sampling = self.sampling_params.as_ref().ok_or(PdReason::Unsupported)?;
        let params: Vec<&SamplingParams> = match sampling {
            SamplingParamsInput::One(params) => vec![params],
            SamplingParamsInput::Many(params) => params.iter().collect(),
        };
        for params in params {
            validate_pd_sampling_params(params)?;
        }
        Ok(())
    }

    /// Validate, normalize and fan the body into one [`GenerateRequest`] per
    /// prompt + `is_batch` (list form — a 1-element list is still a batch → JSON
    /// array response). The Rust counterpart of Python
    /// `GenerateReqInput.normalize_batch_and_arguments`; an invalid/inconsistent
    /// batch is [`Error::Validation`], which the handler surfaces with the
    /// variant's own status (400).
    pub fn into_requests(self) -> Result<(Vec<GenerateRequest>, bool), Error> {
        self.into_requests_inner(false)
    }

    /// PD variant of [`Self::into_requests`]: itemizes and validates the
    /// Gateway-owned bootstrap columns. Ordinary mode keeps ignoring those
    /// extension fields, preserving upstream request compatibility.
    pub fn into_pd_requests(self) -> Result<(Vec<GenerateRequest>, bool), Error> {
        self.into_requests_inner(true)
    }

    fn into_requests_inner(self, pd_enabled: bool) -> Result<(Vec<GenerateRequest>, bool), Error> {
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
            bootstrap_attempt_id,
            // Unported `GenerateReqInput` fields land here and are dropped, as they
            // are on the ordinary Python-compatible path.
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

        let pd_bootstrap = if pd_enabled {
            split_pd_bootstrap(
                bootstrap_host,
                bootstrap_port,
                bootstrap_room,
                bootstrap_attempt_id,
                n,
                is_batch,
            )?
        } else {
            vec![None; n]
        };

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
            pd_bootstrap,
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
                pd_bootstrap,
            )| GenerateRequest {
                rid,
                text,
                input_ids,
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
                pd_bootstrap,
                pd_sidecar: None,
            },
        )
        .collect();
        Ok((requests, is_batch))
    }
}

fn one_or_many_bool(value: &Option<OneOrMany<bool>>) -> bool {
    match value {
        Some(OneOrMany::One(value)) => *value,
        Some(OneOrMany::Many(values)) => values.iter().any(|value| *value),
        None => false,
    }
}

fn one_or_many_nonzero(value: &Option<OneOrMany<i64>>) -> bool {
    match value {
        Some(OneOrMany::One(value)) => *value != 0,
        Some(OneOrMany::Many(values)) => values.iter().any(|value| *value != 0),
        None => false,
    }
}

fn pd_json_present(value: &Option<serde_json::Value>) -> bool {
    value.as_ref().is_some_and(|value| !value.is_null())
}

fn pd_json_flag(value: &Option<serde_json::Value>) -> Result<bool, crate::pd::room::PdReason> {
    match value {
        None | Some(serde_json::Value::Null) => Ok(false),
        Some(serde_json::Value::Bool(value)) => Ok(*value),
        Some(_) => Err(crate::pd::room::PdReason::RequestInvalid),
    }
}

fn validate_pd_sampling_params(params: &SamplingParams) -> Result<(), crate::pd::room::PdReason> {
    use crate::pd::room::PdReason;

    if params.n != 1
        || params.regex.is_some()
        || params.json_schema.is_some()
        || params.ebnf.is_some()
        || params.structural_tag.is_some()
    {
        return Err(PdReason::Unsupported);
    }

    if params.temperature != 0.0 {
        return Err(PdReason::Unsupported);
    }
    // Before normalization, an omitted `top_k` uses SamplingParams' whole-vocab
    // sentinel; greedy post-init rewrites it to 1. Accept both representations,
    // but fail closed on an explicitly non-greedy value such as 2.
    if !matches!(params.top_k, 1 | TOP_K_ALL) || params.top_p != 1.0 || params.min_p != 0.0 {
        return Err(PdReason::Unsupported);
    }
    let max_new_tokens = params.max_new_tokens.unwrap_or(128);
    if !(0..=256).contains(&max_new_tokens) {
        return Err(PdReason::RequestInvalid);
    }
    Ok(())
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
    /// Validated Gateway identity, before tokenization/request-digest binding.
    pub pd_bootstrap: Option<PdBootstrap>,
    /// Versioned map appended to the Scheduler request only after tokenization
    /// and frozen sampling normalization have produced the request digest.
    pub pd_sidecar: Option<rmpv::Value>,
}

fn split_pd_bootstrap(
    host: Option<OneOrMany<String>>,
    port: Option<OneOrMany<u16>>,
    room: Option<OneOrMany<u64>>,
    attempt_id: Option<OneOrMany<String>>,
    count: usize,
    is_batch: bool,
) -> Result<Vec<Option<PdBootstrap>>, Error> {
    let present = [
        host.is_some(),
        port.is_some(),
        room.is_some(),
        attempt_id.is_some(),
    ];
    if present.iter().all(|value| !value) {
        return Ok(vec![None; count]);
    }
    if !present.iter().all(|value| *value) {
        return Err(Error::Validation(
            "PD bootstrap fields must be provided together".into(),
        ));
    }
    if count > 8 {
        return Err(Error::Validation(
            "PD batch must contain at most 8 items".into(),
        ));
    }

    let hosts = exact_column(host.unwrap(), count, is_batch, "bootstrap_host")?;
    let ports = exact_column(port.unwrap(), count, is_batch, "bootstrap_port")?;
    let rooms = exact_column(room.unwrap(), count, is_batch, "bootstrap_room")?;
    let attempts = exact_column(attempt_id.unwrap(), count, is_batch, "bootstrap_attempt_id")?;

    hosts
        .into_iter()
        .zip(ports)
        .zip(rooms)
        .zip(attempts)
        .enumerate()
        .map(|(batch_index, (((host, port), room), attempt_id))| {
            if host.is_empty() || port == 0 || room > i64::MAX as u64 {
                return Err(Error::Validation("PD bootstrap identity is invalid".into()));
            }
            let parsed = uuid::Uuid::parse_str(&attempt_id).map_err(|_| {
                Error::Validation("bootstrap_attempt_id must be a canonical UUIDv4".into())
            })?;
            if parsed.get_version() != Some(uuid::Version::Random)
                || parsed.get_variant() != uuid::Variant::RFC4122
                || parsed.to_string() != attempt_id
            {
                return Err(Error::Validation(
                    "bootstrap_attempt_id must be a canonical UUIDv4".into(),
                ));
            }
            Ok(Some(PdBootstrap {
                host,
                port,
                room,
                attempt_id,
                batch_index: u32::try_from(batch_index)
                    .map_err(|_| Error::Validation("PD batch index is invalid".into()))?,
            }))
        })
        .collect()
}

fn exact_column<T: OneOrManyItem>(
    value: OneOrMany<T>,
    count: usize,
    is_batch: bool,
    field: &str,
) -> Result<Vec<T>, Error> {
    match (is_batch, value) {
        (false, OneOrMany::One(value)) => Ok(vec![value]),
        (true, OneOrMany::Many(values)) if values.len() == count => Ok(values),
        (false, OneOrMany::Many(_)) => Err(Error::Validation(format!("{field} must be scalar"))),
        (true, OneOrMany::One(_)) => Err(Error::Validation(format!(
            "{field} must be a per-item array"
        ))),
        (true, OneOrMany::Many(values)) => Err(Error::Validation(format!(
            "{field} list length {} does not match batch size {count}",
            values.len()
        ))),
    }
}

impl GenerateRequest {
    /// True when the client already supplied token ids → skip tokenization.
    pub fn already_tokenized(&self) -> bool {
        self.input_ids.as_ref().is_some_and(|v| !v.is_empty())
    }

    /// Bind the tokenized, normalized request to the Rust transport snapshot.
    /// The emitted map is the only PD object crossing into Python.
    pub fn bind_pd_sidecar(
        &mut self,
        readiness: Option<&crate::pd::transport::PdReadinessHandle>,
    ) -> Result<(), crate::pd::room::PdReason> {
        use crate::pd::protocol::Role;
        use crate::pd::request::{
            AuxSchema, RequestContractDigest, RequestContractInput, RequestSampling,
        };
        use crate::pd::room::PdReason;
        use crate::pd::runtime::RuntimeLifecycle;

        if self.pd_bootstrap.is_none() && readiness.is_none() {
            return Ok(());
        }
        if self.pd_sidecar.is_some() {
            return Ok(());
        }
        let bootstrap = self.pd_bootstrap.as_ref().ok_or(PdReason::RequestInvalid)?;
        let snapshot = readiness.ok_or(PdReason::PeerUnavailable)?.snapshot();
        if snapshot.runtime.lifecycle != RuntimeLifecycle::PairReady || !snapshot.runtime.pair_ready
        {
            return Err(PdReason::PeerUnavailable);
        }
        if bootstrap.host != snapshot.expected_bootstrap_host
            || !snapshot.allowed_bootstrap_ports.contains(&bootstrap.port)
        {
            return Err(PdReason::RequestInvalid);
        }

        let input_ids = self
            .input_ids
            .as_ref()
            .filter(|ids| (1..=4096).contains(&ids.len()))
            .ok_or(PdReason::RequestInvalid)?
            .iter()
            .map(|token| u32::try_from(*token).map_err(|_| PdReason::RequestInvalid))
            .collect::<Result<Vec<_>, _>>()?;
        let sampling = RequestSampling::from_normalized(&self.sampling_params)
            .map_err(|_| PdReason::RequestInvalid)?;
        let digest = RequestContractDigest::new(RequestContractInput {
            model_manifest_digest: snapshot.model_manifest_digest,
            tokenizer_manifest_digest: snapshot.tokenizer_manifest_digest,
            layout_fingerprint: snapshot.layout_fingerprint,
            profile_digest: snapshot.runtime.profile_digest,
            batch_index: bootstrap.batch_index,
            normalized_input_ids: input_ids,
            sampling,
            aux_schema: AuxSchema {
                version: 1,
                bytes: 64,
            },
        })
        .map_err(|_| PdReason::RequestInvalid)?;
        let decode_process_epoch = match snapshot.runtime.role {
            Role::Decode => snapshot.runtime.process_epoch.as_bytes(),
            Role::Prefill => snapshot
                .runtime
                .peer_process_epoch
                .ok_or(PdReason::PeerUnavailable)?
                .into_array(),
        };
        self.pd_sidecar = Some(rmpv::Value::Map(vec![
            (rmpv::Value::from("version"), rmpv::Value::from(1)),
            (
                rmpv::Value::from("bootstrap_host"),
                rmpv::Value::from(bootstrap.host.as_str()),
            ),
            (
                rmpv::Value::from("bootstrap_port"),
                rmpv::Value::from(bootstrap.port),
            ),
            (
                rmpv::Value::from("bootstrap_room"),
                rmpv::Value::from(bootstrap.room),
            ),
            (
                rmpv::Value::from("attempt_id"),
                rmpv::Value::from(bootstrap.attempt_id.as_str()),
            ),
            (
                rmpv::Value::from("batch_index"),
                rmpv::Value::from(bootstrap.batch_index),
            ),
            (
                rmpv::Value::from("request_digest"),
                rmpv::Value::from(digest.to_hex()),
            ),
            (
                rmpv::Value::from("decode_process_epoch"),
                rmpv::Value::from(uuid::Uuid::from_bytes(decode_process_epoch).to_string()),
            ),
        ]));
        Ok(())
    }

    /// Multimodal detection hook. Deferred (Encoder stubbed): always false until mm
    /// fields are wired in.
    #[allow(dead_code)]
    pub fn has_multimodal(&self) -> bool {
        false
    }

    pub fn encode_header(&self) -> Result<Bytes, Error> {
        let header = TokenizedGenerateReqInput::from(self);
        match self.pd_bootstrap.as_ref() {
            Some(bootstrap) => header.encode_with_pd_bootstrap(bootstrap),
            None => header.encode(),
        }
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

    /// Vocab size for tests that aren't about the vocab bound (see
    /// `sampling::tests::TEST_VOCAB`).
    const TEST_VOCAB: u64 = 1000;

    fn requests(body: &str) -> Result<(Vec<GenerateRequest>, bool), Error> {
        serde_json::from_str::<GenerateBody>(body)
            .unwrap()
            .into_requests()
    }

    fn pd_requests(body: &str) -> Result<(Vec<GenerateRequest>, bool), Error> {
        serde_json::from_str::<GenerateBody>(body)
            .unwrap()
            .into_pd_requests()
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
    /// its `lora_path`/`return_routed_experts`/`image_data` are accepted-but-ignored,
    /// so `split` succeeds and drops them while the real fields survive.
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

    #[test]
    fn pd_bootstrap_scalar_and_batch_are_itemized() {
        let (scalar_requests, is_batch) = pd_requests(
            r#"{"input_ids":[1],"bootstrap_host":"prefill.internal",
                "bootstrap_port":8998,"bootstrap_room":0,
                "bootstrap_attempt_id":"44444444-4444-4444-8444-444444444444"}"#,
        )
        .unwrap();
        assert!(!is_batch);
        let scalar = scalar_requests[0].pd_bootstrap.as_ref().unwrap();
        assert_eq!(scalar.host, "prefill.internal");
        assert_eq!(scalar.port, 8998);
        assert_eq!(scalar.room, 0);
        assert_eq!(scalar.batch_index, 0);

        let (batch_requests, is_batch) = pd_requests(
            r#"{"input_ids":[[1],[2]],"bootstrap_host":["prefill.internal","prefill.internal"],
                "bootstrap_port":[8998,8998],"bootstrap_room":[0,9223372036854775807],
                "bootstrap_attempt_id":["44444444-4444-4444-8444-444444444444",
                                        "55555555-5555-4555-8555-555555555555"]}"#,
        )
        .unwrap();
        assert!(is_batch);
        assert_eq!(
            batch_requests[0].pd_bootstrap.as_ref().unwrap().batch_index,
            0
        );
        assert_eq!(
            batch_requests[1].pd_bootstrap.as_ref().unwrap().room,
            i64::MAX as u64
        );
        assert_eq!(
            batch_requests[1].pd_bootstrap.as_ref().unwrap().batch_index,
            1
        );
    }

    #[test]
    fn pd_bootstrap_is_all_or_none_with_exact_cardinality() {
        assert!(
            pd_requests(
                r#"{"input_ids":[1],"bootstrap_host":"prefill.internal",
                    "bootstrap_port":8998,"bootstrap_room":0}"#
            )
            .is_err()
        );
        assert!(
            pd_requests(
                r#"{"input_ids":[[1],[2]],"bootstrap_host":"prefill.internal",
                    "bootstrap_port":[8998,8998],"bootstrap_room":[0,1],
                    "bootstrap_attempt_id":["44444444-4444-4444-8444-444444444444",
                                            "55555555-5555-4555-8555-555555555555"]}"#
            )
            .is_err()
        );
        assert!(
            pd_requests(
                r#"{"input_ids":[1],"bootstrap_host":"prefill.internal",
                    "bootstrap_port":8998,"bootstrap_room":9223372036854775808,
                    "bootstrap_attempt_id":"44444444-4444-4444-8444-444444444444"}"#
            )
            .is_err()
        );
        assert!(
            pd_requests(
                r#"{"input_ids":[1],"bootstrap_host":"prefill.internal",
                    "bootstrap_port":8998,"bootstrap_room":0,
                    "bootstrap_attempt_id":"44444444-4444-1444-8444-444444444445"}"#
            )
            .is_err()
        );
    }

    #[test]
    fn pd_support_matrix_distinguishes_invalid_and_unsupported() {
        let greedy: GenerateBody = serde_json::from_str(
            r#"{"input_ids":[1],"sampling_params":{"temperature":0,"max_new_tokens":256}}"#,
        )
        .unwrap();
        assert_eq!(greedy.validate_pd_support(), Ok(()));

        let random: GenerateBody =
            serde_json::from_str(r#"{"input_ids":[1],"sampling_params":{"temperature":0.5}}"#)
                .unwrap();
        assert_eq!(
            random.validate_pd_support(),
            Err(crate::pd::room::PdReason::Unsupported)
        );

        let non_greedy_top_k: GenerateBody = serde_json::from_str(
            r#"{"input_ids":[1],"sampling_params":{"temperature":0,"top_k":2}}"#,
        )
        .unwrap();
        assert_eq!(
            non_greedy_top_k.validate_pd_support(),
            Err(crate::pd::room::PdReason::Unsupported)
        );

        let too_long: GenerateBody = serde_json::from_str(
            r#"{"input_ids":[1],"sampling_params":{"temperature":0,"max_new_tokens":257}}"#,
        )
        .unwrap();
        assert_eq!(
            too_long.validate_pd_support(),
            Err(crate::pd::room::PdReason::RequestInvalid)
        );

        let logprobs: GenerateBody = serde_json::from_str(
            r#"{"input_ids":[1],"sampling_params":{"temperature":0},"return_logprob":true}"#,
        )
        .unwrap();
        assert_eq!(
            logprobs.validate_pd_support(),
            Err(crate::pd::room::PdReason::Unsupported)
        );
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
}
