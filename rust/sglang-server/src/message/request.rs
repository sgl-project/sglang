//! The `/generate` request path: the HTTP body and its per-request fan-out
//! ([`GenerateBody`] → [`GenerateRequest`]s).

use std::collections::{BTreeMap, HashSet};
use std::sync::LazyLock;
use std::time::{Duration, Instant};

use bytes::Bytes;
use itertools::izip;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use super::embeddings::PositionalEmbeds;
use super::io_struct::{ControlRequest, TokenizedGenerateReqInput};
use super::multimodal::{self, MmDataInput, MmItem};
use super::response::ResponseSink;
use super::sampling::{CustomParamValue, SamplingParams, SamplingParamsInput};
use super::types::{HiddenStatesMode, InputEmbeddings, OneOrMany, OneOrManyItem, TokenIds};
use crate::message::ids::Rid;
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

/// Top-level fields in this namespace belong to the selected multimodal
/// processor. Everything else unknown to [`GenerateBody`] keeps Python's
/// accepted-but-ignored behavior.
const PROCESSOR_EXTENSION_PREFIX: &str = "multimodal_";

/// Model-owned request fields. The shared server preserves and batches their
/// MessagePack value representation; the selected processor deserializes that
/// map into its own concrete schema.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(transparent)]
pub struct ProcessorExtensions(BTreeMap<String, rmpv::Value>);

impl ProcessorExtensions {
    /// Deserialize the model-agnostic value tree directly into the selected
    /// processor's schema. This does not encode or decode MessagePack bytes.
    pub fn deserialize<T: DeserializeOwned>(self) -> Result<T, String> {
        let fields = self
            .0
            .into_iter()
            .map(|(name, value)| (rmpv::Value::from(name), value))
            .collect();
        rmpv::ext::from_value(rmpv::Value::Map(fields))
            .map_err(|error| format!("invalid processor extensions: {error}"))
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn values(&self) -> impl Iterator<Item = &rmpv::Value> {
        self.0.values()
    }
}

impl FromIterator<(String, rmpv::Value)> for ProcessorExtensions {
    fn from_iter<T: IntoIterator<Item = (String, rmpv::Value)>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

/// The `/generate` wire body before batch splitting: `text`/`input_ids`/`sampling_params`
/// each scalar-or-list, fanned into per-request [`GenerateRequest`]s by
/// [`into_requests`](GenerateBody::into_requests).
///
/// Unknown keys are ignored, matching Python, except `multimodal_*` fields. Those
/// are opaque processor extensions: this layer only fans them out with the
/// request batch and passes them to the selected multimodal processor.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct GenerateBody {
    /// Optional client-supplied request id(s): a single string (a batch fans it
    /// out as `{rid}_{i}`, mirroring Python `_normalize_batch`) or one per item.
    pub rid: Option<OneOrMany<String>>,
    pub text: Option<OneOrMany<String>>,
    pub input_ids: Option<OneOrMany<TokenIds>>,
    pub input_embeds: Option<OneOrMany<InputEmbeddings>>,
    pub positional_embed_overrides: Option<OneOrMany<Option<PositionalEmbeds>>>,
    #[serde(default)]
    pub stream: bool,
    /// One params object (broadcast) or a list of them (per item); see
    /// [`SamplingParamsInput`].
    pub sampling_params: Option<SamplingParamsInput>,
    /// Logprob / hidden-state options: a scalar broadcasts to every prompt, a
    /// list is per-prompt (Python `_normalize_logprob_params`).
    pub return_logprob: Option<OneOrMany<bool>>,
    pub return_sampling_mask: Option<OneOrMany<bool>>,
    #[serde(default)]
    pub return_flat_raw_top_logprobs: bool,
    #[serde(default)]
    pub return_flat_raw_top_logprobs_b64: bool,
    #[serde(default)]
    pub return_routed_experts: bool,
    #[serde(default)]
    pub routed_experts_start_len: i64,
    #[serde(default)]
    pub return_indexer_topk: bool,
    #[serde(default)]
    pub return_prompt_token_ids: bool,
    /// Python accepts these switches, but its LLM tokenizer/scheduler path
    /// does not produce image bytes or token entropy. Preserve them for DP
    /// transport and validate their types without inventing response fields.
    #[serde(default)]
    pub return_bytes: bool,
    #[serde(default)]
    pub return_entropy: bool,
    pub logprob_start_len: Option<OneOrMany<i64>>,
    pub top_logprobs_num: Option<OneOrMany<i64>>,
    /// Token ids to report logprobs for: one list (broadcast to every prompt) or
    /// one list per prompt, mirroring Python's
    /// `Union[List[int], List[List[int]]]` fan-out in `_normalize_batch`.
    pub token_ids_logprob: Option<OneOrMany<TokenIds>>,
    pub multi_item_delimiter_indices: Option<OneOrMany<TokenIds>>,
    #[serde(default)]
    pub return_hidden_states: OneOrMany<HiddenStatesMode>,
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
    pub data_parallel_rank: Option<i64>,
    pub disagg_prefill_dp_rank: Option<i64>,
    pub disagg_prefill_serve_addr: Option<OneOrMany<Option<String>>>,
    pub conversation_id: Option<String>,
    pub routing_key: Option<String>,
    pub extra_key: Option<OneOrMany<String>>,
    pub cache_salt: Option<OneOrMany<String>>,
    pub custom_logit_processor: Option<OneOrMany<Option<String>>>,
    pub require_reasoning: Option<bool>,
    pub max_thinking_tokens: Option<i64>,
    pub priority: Option<i64>,
    pub log_metrics: Option<bool>,
    pub custom_labels: Option<BTreeMap<String, String>>,
    pub received_time: Option<f64>,
    // Multimodal inputs (Python `MultimodalDataInputFormat`), fanned out per
    // request by `multimodal::fan_out`.
    pub image_data: Option<MmDataInput>,
    /// Caller-supplied per-item feature hashes (hex) overriding the computed
    /// ones, so an external router's keys align with the prefix cache.
    pub mm_hashes: Option<Vec<OneOrMany<String>>>,
    /// Original-media identities, separate from the processor-feature hashes.
    pub mm_content_hashes: Option<Vec<OneOrMany<Option<String>>>>,
    pub video_data: Option<MmDataInput>,
    pub audio_data: Option<MmDataInput>,
    #[serde(flatten)]
    pub processor_options: multimodal::MmProcessorOptions,
    /// Model-specific multimodal fields, retained without teaching the shared
    /// request schema their contents. Other unknown fields remain ignored.
    #[serde(flatten)]
    processor_extensions: ProcessorExtensions,
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
    /// prompt/sample + `is_batch` (list form or n > 1 → JSON array response).
    /// The Rust counterpart of Python
    /// `GenerateReqInput.normalize_batch_and_arguments`; an invalid/inconsistent
    /// batch is [`Error::Validation`], which the handler surfaces with the
    /// variant's own status (400).
    pub fn into_requests(self) -> Result<(Vec<GenerateRequest>, bool), Error> {
        reject_unsupported_fields(&self.processor_extensions)?;
        if self.return_flat_raw_top_logprobs_b64 && !self.return_flat_raw_top_logprobs {
            return Err(Error::Validation(
                "return_flat_raw_top_logprobs_b64 requires return_flat_raw_top_logprobs.".into(),
            ));
        }
        if self.return_flat_raw_top_logprobs && self.multi_item_delimiter_indices.is_some() {
            return Err(Error::Validation(
                "return_flat_raw_top_logprobs does not support multi-item scoring: delimiter-sparse top logprob rows have no contiguous position mapping.".into(),
            ));
        }
        let GenerateBody {
            rid,
            text,
            input_ids,
            input_embeds,
            positional_embed_overrides,
            stream,
            sampling_params,
            return_logprob,
            return_sampling_mask,
            return_flat_raw_top_logprobs,
            return_flat_raw_top_logprobs_b64,
            return_routed_experts,
            routed_experts_start_len,
            return_indexer_topk,
            return_prompt_token_ids,
            return_bytes,
            return_entropy,
            logprob_start_len,
            top_logprobs_num,
            token_ids_logprob,
            multi_item_delimiter_indices,
            return_hidden_states,
            return_text_in_logprobs,
            bootstrap_host,
            bootstrap_port,
            bootstrap_room,
            bootstrap_pair_key,
            decode_tp_size,
            routed_dp_rank,
            data_parallel_rank,
            disagg_prefill_dp_rank,
            disagg_prefill_serve_addr,
            conversation_id,
            routing_key,
            extra_key,
            cache_salt,
            custom_logit_processor,
            require_reasoning,
            max_thinking_tokens,
            priority,
            log_metrics,
            custom_labels,
            received_time,
            image_data,
            video_data,
            audio_data,
            mm_hashes,
            mm_content_hashes,
            processor_options,
            processor_extensions,
        } = self;
        let routed_dp_rank = routed_dp_rank.or(data_parallel_rank);
        let samples = parallel_sample_count(sampling_params.as_ref())?;

        // Cap the batch BEFORE the columns below allocate anything. Reading the
        // declared length off the input costs nothing; the previous placement (after
        // the match) had already allocated ~1.7 GiB for a 114 MiB body, most of it
        // the `vec![None; n]` twin column.
        let declared_n = match (&text, &input_ids, &input_embeds) {
            (Some(OneOrMany::Many(v)), None, None) => v.len(),
            (None, Some(OneOrMany::Many(v)), None) => v.len(),
            (None, None, Some(OneOrMany::Many(v))) => v.len(),
            _ => 1,
        };
        let expanded_n = declared_n.checked_mul(samples).ok_or_else(|| {
            Error::Validation("prompt count times n overflows the request limit".into())
        })?;
        if batch_size_exceeds_limit(expanded_n, *MAX_BATCH_REQS_PER_HTTP_REQ) {
            return Err(Error::Validation(format!(
                "batch size {expanded_n} exceeds the maximum of {}",
                *MAX_BATCH_REQS_PER_HTTP_REQ
            )));
        }
        if samples > 1 {
            for (per_prompt, name) in [
                (
                    matches!(return_logprob, Some(OneOrMany::Many(_))),
                    "return_logprob",
                ),
                (
                    matches!(return_sampling_mask, Some(OneOrMany::Many(_))),
                    "return_sampling_mask",
                ),
                (
                    matches!(logprob_start_len, Some(OneOrMany::Many(_))),
                    "logprob_start_len",
                ),
                (
                    matches!(top_logprobs_num, Some(OneOrMany::Many(_))),
                    "top_logprobs_num",
                ),
                (
                    matches!(token_ids_logprob, Some(OneOrMany::Many(_))),
                    "token_ids_logprob",
                ),
                (
                    matches!(custom_logit_processor, Some(OneOrMany::Many(_))),
                    "custom_logit_processor",
                ),
            ] {
                if per_prompt {
                    return Err(Error::Validation(format!(
                        "Cannot use list {name} with parallel_sample_num > 1"
                    )));
                }
            }
        }

        type Columns = (
            Vec<Option<String>>,
            Vec<Option<TokenIds>>,
            Vec<Option<InputEmbeddings>>,
            bool,
        );
        let (texts, id_lists, embeddings, is_batch): Columns = match (text, input_ids, input_embeds)
        {
            (Some(OneOrMany::One(s)), None, None) => (vec![Some(s)], vec![None], vec![None], false),
            (Some(OneOrMany::Many(v)), None, None) => {
                let n = v.len();
                (
                    v.into_iter().map(Some).collect(),
                    vec![None; n],
                    vec![None; n],
                    true,
                )
            }
            // `[]` parses as `One(vec![])` (one prompt with no ids), so the
            // `n == 0` guard below never sees it — reject it here, as Python's
            // `_determine_batch_size` does.
            (None, Some(OneOrMany::One(x)), None) => {
                if x.is_empty() {
                    return Err(Error::Validation("input_ids cannot be empty".into()));
                }
                (vec![None], vec![Some(x)], vec![None], false)
            }
            (None, Some(OneOrMany::Many(vv)), None) => {
                if vv.iter().any(|ids| ids.is_empty()) {
                    return Err(Error::Validation(
                        "input_ids cannot be empty for any prompt in the batch".into(),
                    ));
                }
                let n = vv.len();
                (
                    vec![None; n],
                    vv.into_iter().map(Some).collect(),
                    vec![None; n],
                    true,
                )
            }
            (None, None, Some(OneOrMany::One(embeds))) => {
                (vec![None], vec![None], vec![Some(embeds)], false)
            }
            (None, None, Some(OneOrMany::Many(embeds))) => {
                let n = embeds.len();
                (
                    vec![None; n],
                    vec![None; n],
                    embeds.into_iter().map(Some).collect(),
                    true,
                )
            }
            _ => {
                return Err(Error::Validation(
                    "provide exactly one of `text`, `input_ids`, or `input_embeds`".into(),
                ));
            }
        };
        let n = texts.len();
        if n == 0 {
            return Err(Error::Validation(
                "batch must contain at least one item".into(),
            ));
        }
        if let Some(key) = &routing_key {
            check_broadcast_budget(key.len(), n, "routing_key")?;
        }
        let extra_keys = cache_key_column(extra_key, n, is_batch || samples > 1, "extra_key")?;
        let cache_salts = cache_key_column(cache_salt, n, is_batch || samples > 1, "cache_salt")?;
        if n > 1
            && let Some(labels) = &custom_labels
        {
            let bytes = labels.iter().fold(0usize, |size, (name, value)| {
                size.saturating_add(name.len())
                    .saturating_add(value.len())
                    .saturating_add(96)
            });
            check_broadcast_budget(bytes, n, "custom_labels")?;
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
                if (!is_batch && samples == 1) || v.len() != n {
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
        if is_batch && matches!(multi_item_delimiter_indices, Some(OneOrMany::One(_))) {
            return Err(Error::Validation(
                "multi_item_delimiter_indices must contain one list per request".into(),
            ));
        }
        let delimiter_indices = fan_out(
            multi_item_delimiter_indices,
            n,
            "multi_item_delimiter_indices",
        )?;
        let positional_embeds = flatten_column(fan_out(
            positional_embed_overrides,
            n,
            "positional_embed_overrides",
        )?);

        // Each logprob/hidden opt: absent → None for every item, a scalar
        // broadcasts, a list is per-item (Python `normalize_param`, plus a length
        // check Python lacks — it would `IndexError` later instead).
        let return_logprobs = fan_out(return_logprob, n, "return_logprob")?;
        let return_sampling_masks = fan_out(return_sampling_mask, n, "return_sampling_mask")?;
        let logprob_start_lens = fan_out(logprob_start_len, n, "logprob_start_len")?;
        let top_logprobs_nums = fan_out(top_logprobs_num, n, "top_logprobs_num")?;
        let return_hidden = fan_out(Some(return_hidden_states), n, "return_hidden_states")?;
        let custom_logit_processors = flatten_column(fan_out(
            custom_logit_processor,
            n,
            "custom_logit_processor",
        )?);

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
        // Multimodal columns; see `multimodal::fan_out` for the Python parity rules.
        let images = multimodal::fan_out(image_data, n, is_batch, "image_data")?;
        let videos = multimodal::fan_out(video_data, n, is_batch, "video_data")?;
        let audios = multimodal::fan_out(audio_data, n, is_batch, "audio_data")?;
        let mm_hashes = multimodal::fan_out_hashes(mm_hashes, &images, is_batch, "mm_hashes")?;
        let mm_content_hashes =
            multimodal::fan_out_hashes(mm_content_hashes, &images, is_batch, "mm_content_hashes")?;
        if n > 1 && !processor_options.is_default() {
            let bytes = serde_json::to_vec(&processor_options)
                .map_err(|error| Error::Validation(error.to_string()))?
                .len()
                .saturating_mul(JSON_TO_HEAP_FACTOR);
            check_broadcast_budget(bytes, n, "multimodal processor options")?;
        }
        let processor_options = vec![processor_options; n];
        let prefill_serve_addrs = flatten_column(fan_out(
            disagg_prefill_serve_addr,
            n,
            "disagg_prefill_serve_addr",
        )?);
        let processor_extensions = split_extension_columns(processor_extensions, n, is_batch)?;

        // Every column above is exactly `n` long, so zip them by value: each
        // request takes ownership of its cell, with no indexing or bounds checks.
        let mut requests: Vec<GenerateRequest> = izip!(
            rids,
            texts,
            id_lists,
            embeddings,
            positional_embeds,
            sps,
            return_logprobs,
            return_sampling_masks,
            logprob_start_lens,
            top_logprobs_nums,
            tid_logprobs,
            delimiter_indices,
            return_hidden,
            bootstrap_hosts,
            bootstrap_ports,
            bootstrap_rooms,
            bootstrap_pair_keys,
            decode_tp_sizes,
            images,
            videos,
            audios,
            mm_hashes,
            mm_content_hashes,
            processor_extensions,
            processor_options,
            prefill_serve_addrs,
            extra_keys,
            cache_salts,
            custom_logit_processors,
        )
        .map(
            |(
                rid,
                text,
                input_ids,
                input_embeds,
                positional_embed_overrides,
                sampling_params,
                return_logprob,
                return_sampling_mask,
                logprob_start_len,
                top_logprobs_num,
                token_ids_logprob,
                multi_item_delimiter_indices,
                return_hidden_states,
                bootstrap_host,
                bootstrap_port,
                bootstrap_room,
                bootstrap_pair_key,
                decode_tp_size,
                image_data,
                video_data,
                audio_data,
                mm_hashes,
                mm_content_hashes,
                processor_extensions,
                processor_options,
                disagg_prefill_serve_addr,
                extra_key,
                cache_salt,
                custom_logit_processor,
            )| GenerateRequest {
                rid,
                text,
                input_ids,
                input_embeds,
                positional_embed_overrides,
                // Plain text prompts keep the post-processor specials; the
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
                multi_item_delimiter_indices,
                return_sampling_mask: return_sampling_mask.unwrap_or(false),
                return_flat_raw_top_logprobs,
                return_flat_raw_top_logprobs_b64,
                return_routed_experts,
                routed_experts_start_len,
                return_indexer_topk,
                return_prompt_token_ids,
                return_bytes,
                return_entropy,
                return_hidden_states: return_hidden_states.unwrap_or_default(),
                return_text_in_logprobs,
                bootstrap_host,
                bootstrap_port,
                bootstrap_room,
                bootstrap_pair_key,
                decode_tp_size,
                routed_dp_rank,
                disagg_prefill_dp_rank,
                disagg_prefill_serve_addr,
                conversation_id: conversation_id.clone(),
                routing_key: routing_key.clone(),
                extra_key,
                cache_salt,
                custom_logit_processor,
                require_reasoning: require_reasoning.unwrap_or(false),
                max_thinking_tokens,
                priority,
                log_metrics,
                custom_labels: custom_labels.clone(),
                received_time,
                started: None,
                metric_state: None,
                response_metadata: None,
                mm_pad_spans: Vec::new(),
                mm: pack_mm(
                    image_data,
                    video_data,
                    audio_data,
                    mm_hashes,
                    mm_content_hashes,
                    processor_extensions,
                    processor_options,
                ),
            },
        )
        .collect();
        for request in &mut requests {
            if let Some(mm) = request.mm.as_deref_mut() {
                mm.mm_content_hashes = multimodal::normalize_content_hashes(
                    &mm.image_data,
                    mm.mm_content_hashes.take(),
                )
                .map_err(Error::Validation)?;
            }
            if let Some(budget) = request.max_thinking_tokens {
                request
                    .sampling_params
                    .custom_params
                    .get_or_insert_default()
                    .insert("thinking_budget".into(), CustomParamValue::Signed(budget));
            }
        }
        Ok((
            expand_parallel_samples(requests, samples)?,
            is_batch || samples > 1,
        ))
    }
}

fn reject_unsupported_fields(fields: &ProcessorExtensions) -> Result<(), Error> {
    for name in [
        "session_id",
        "session_params",
        "lora_path",
        "lora_id",
        "background",
        "no_logs",
        "external_trace_header",
        "http_worker_ipc",
        "need_wait_for_mm_inputs",
        "num_items_assigned",
        "encoder_urls",
    ] {
        if fields.0.get(name).is_some_and(feature_requested) {
            return Err(Error::Validation(format!(
                "The Rust frontend does not support `{name}`"
            )));
        }
    }
    Ok(())
}

fn feature_requested(value: &rmpv::Value) -> bool {
    match value {
        rmpv::Value::Nil | rmpv::Value::Boolean(false) => false,
        rmpv::Value::String(value) => !value.as_str().is_some_and(str::is_empty),
        rmpv::Value::Array(values) => values.iter().any(feature_requested),
        rmpv::Value::Map(values) => !values.is_empty(),
        _ => true,
    }
}

fn parallel_sample_count(params: Option<&SamplingParamsInput>) -> Result<usize, Error> {
    let values = match params {
        None => return Ok(1),
        Some(SamplingParamsInput::One(params)) => std::slice::from_ref(params.as_ref()),
        Some(SamplingParamsInput::Many(params)) => params.as_slice(),
    };
    let Some(first) = values.first() else {
        return Ok(1);
    };
    let count = usize::try_from(first.n)
        .ok()
        .filter(|count| *count > 0)
        .ok_or_else(|| Error::Validation("n must be at least 1".into()))?;
    if values.iter().any(|params| params.n != first.n) {
        return Err(Error::Validation(
            "The parallel_sample_num should be the same for all samples in sample params.".into(),
        ));
    }
    // The scheduler expands beam rows jointly; n selects returned beams there.
    Ok(if first.beam_width.is_some_and(|width| width > 1) {
        1
    } else {
        count
    })
}

fn expand_parallel_samples(
    requests: Vec<GenerateRequest>,
    samples: usize,
) -> Result<Vec<GenerateRequest>, Error> {
    if samples == 1 {
        return Ok(requests);
    }
    // Validate the aggregate clone cost before allocating children. This uses
    // the same memory budget as scalar-to-batch field expansion above.
    let mut bytes = 0usize;
    for request in &requests {
        let size = serde_json::to_vec(request)
            .map_err(|error| Error::Validation(format!("cannot expand parallel samples: {error}")))?
            .len();
        bytes = bytes.saturating_add(size.saturating_mul(JSON_TO_HEAP_FACTOR));
    }
    check_broadcast_budget(bytes, samples, "parallel samples")?;
    let mut expanded = Vec::with_capacity(requests.len() * samples);
    for mut request in requests {
        // A forwarded scalar child must not expand again on its DP worker.
        // Python copies the seed unchanged for every sample as well.
        request.sampling_params.n = 1;
        let parent = request.rid.client_facing().to_owned();
        let room = request.bootstrap_room;
        for sample in 0..samples {
            let mut child = request.clone();
            child.rid = Rid::from_client(&format!("{parent}_{sample}"));
            // Keep the P/D pairing deterministic while giving concurrent samples
            // distinct transfer rooms, even when the caller supplies room lists.
            child.bootstrap_room = room.map(|room| {
                room.wrapping_mul(samples as i64)
                    .wrapping_add(sample as i64)
            });
            expanded.push(child);
        }
    }
    Ok(expanded)
}

/// Box the per-item mm values, `None` when the item has none — the common
/// text-only case keeps `GenerateRequest` slim.
fn pack_mm(
    image_data: Vec<MmItem>,
    video_data: Vec<MmItem>,
    audio_data: Vec<MmItem>,
    mm_hashes: Option<Vec<String>>,
    mm_content_hashes: Option<Vec<Option<String>>>,
    processor_extensions: ProcessorExtensions,
    processor_options: multimodal::MmProcessorOptions,
) -> Option<Box<MmData>> {
    if image_data.is_empty()
        && video_data.is_empty()
        && audio_data.is_empty()
        && processor_extensions.is_empty()
        && processor_options.is_default()
    {
        return None;
    }
    Some(Box::new(MmData {
        image_data,
        video_data,
        audio_data,
        mm_hashes: mm_hashes.unwrap_or_default(),
        mm_content_hashes,
        processor_extensions,
        processor_options,
        ..Default::default()
    }))
}

fn split_extension_columns(
    fields: ProcessorExtensions,
    n: usize,
    is_batch: bool,
) -> Result<Vec<ProcessorExtensions>, Error> {
    let mut requests = vec![ProcessorExtensions::default(); n];
    for (name, value) in fields.0 {
        if !name.starts_with(PROCESSOR_EXTENSION_PREFIX) || value.is_nil() {
            continue;
        }
        if !is_batch {
            requests[0].0.insert(name, value);
            continue;
        }
        let rmpv::Value::Array(values) = value else {
            return Err(Error::Validation(format!(
                "{name} must be a list for batch processing"
            )));
        };
        if values.is_empty() {
            for request in &mut requests {
                request
                    .0
                    .insert(name.clone(), rmpv::Value::Array(Vec::new()));
            }
            continue;
        }
        if values.len() != n {
            return Err(Error::Validation(format!(
                "{name} list length {} does not match batch size {n}",
                values.len()
            )));
        }
        for (request, value) in requests.iter_mut().zip(values) {
            request.0.insert(name.clone(), value);
        }
    }
    Ok(requests)
}

fn extension_value_present(value: &rmpv::Value) -> bool {
    match value {
        rmpv::Value::Nil => false,
        rmpv::Value::Array(values) => values.iter().any(extension_value_present),
        _ => true,
    }
}

/// One request handed to the MM worker pool: the rid to correlate the result,
/// plus the owned inputs from [`GenerateRequest::take_mm_work`].
#[derive(Debug)]
pub struct MmRequest {
    pub rid: Rid,
    pub work: MmWorkItem,
}

#[derive(Debug, Clone)]
pub struct MmFetchTiming {
    pub started: Instant,
    pub elapsed: Duration,
    pub bytes: usize,
}

/// Local measurements, kept out of client JSON and DP forwarding. Downloads
/// retain request order even when their I/O completes out of order.
#[derive(Debug, Clone, Default)]
pub struct MmPrefetchStats {
    pub load_wall: Duration,
    pub downloads: Vec<MmFetchTiming>,
}

/// The parked request's fields the MM worker owns; converted to the driver input
/// by [`crate::multi_modality::payload::to_mm_input`].
#[derive(Debug, Default)]
pub struct MmWorkItem {
    pub queued_at: Option<Instant>,
    pub prefetch_stats: MmPrefetchStats,
    /// PD pairing identity, retained through preprocessing for metadata handoff.
    pub bootstrap_room: Option<i64>,
    pub text: Option<String>,
    pub input_ids: Option<Vec<i32>>,
    pub image_data: Vec<MmItem>,
    pub video_data: Vec<MmItem>,
    pub audio_data: Vec<MmItem>,
    pub processor_extensions: ProcessorExtensions,
    pub processor_options: multimodal::MmProcessorOptions,
    /// See [`MmData::prefetched`].
    pub prefetched: Vec<Bytes>,
    /// See [`GenerateBody::mm_hashes`].
    pub mm_hashes: Vec<String>,
    /// See [`GenerateBody::mm_content_hashes`].
    pub mm_content_hashes: Option<Vec<Option<String>>>,
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
    /// Encode text on a tokenizer worker without submitting scheduler work.
    Tokenize {
        text: String,
        add_special_tokens: bool,
    },
    /// Internal service call: decode a complete token-id sequence to text. Walks
    /// the same FSM as every request (validate → register → Queued), but the
    /// stage that answers it is the detok shard itself, never the scheduler
    /// ring; the result arrives on the registered sink as one `Data` payload
    /// (the raw UTF-8 text). Used by `/detokenize` and completion prompt echo.
    Detokenize {
        token_ids: TokenIds,
        skip_special_tokens: bool,
    },
}

/// A single in-flight `/generate` request (per-item from
/// [`GenerateBody::into_requests`]),
/// serialized to the scheduler wire once tokenized (see `to_header_msgpack`).
/// The DP ingress can serialize its client fields back to a scalar HTTP body;
/// internal identity suffixes, decoded media and trusted metadata never leave it.
#[derive(Debug, Clone, Default, Serialize)]
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
    #[serde(serialize_with = "serialize_client_rid")]
    pub rid: Rid,
    pub text: Option<String>,
    /// Client-supplied token ids, or filled by the Tokenizer stage.
    pub input_ids: Option<TokenIds>,
    /// Owned through batch/DP transport; scheduler intake creates the same
    /// placeholder ids as Python's `Scheduler.handle_generate_request`.
    pub input_embeds: Option<InputEmbeddings>,
    pub positional_embed_overrides: Option<PositionalEmbeds>,
    /// Template-rendered prompts (chat) already contain their role/special
    /// tokens, so the tokenizer pool strips the auto-added BOS/EOS prefix —
    /// the Rust analogue of Python's `add_special_tokens=False` at the
    /// chat-template encode site (`serving_chat._encode_messages`). Consumed
    /// by the pool before the header is built; never reaches the scheduler wire.
    #[serde(skip)]
    pub skip_special_tokens: bool,
    /// Sampling params (defaults when the client sent none, as in Python);
    /// normalized + verified, then serialized into the header.
    #[serde(serialize_with = "SamplingParams::serialize_client")]
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
    /// Scoring boundaries in the finalized prompt; the scheduler's MIS path
    /// uses them to select both attention segments and logprob positions.
    pub multi_item_delimiter_indices: Option<TokenIds>,
    pub return_sampling_mask: bool,
    pub return_flat_raw_top_logprobs: bool,
    pub return_flat_raw_top_logprobs_b64: bool,
    pub return_routed_experts: bool,
    pub routed_experts_start_len: i64,
    pub return_indexer_topk: bool,
    pub return_prompt_token_ids: bool,
    pub return_bytes: bool,
    pub return_entropy: bool,
    pub return_hidden_states: HiddenStatesMode,
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
    /// DP routing hints, resolved before worker tokenization and media I/O.
    pub routed_dp_rank: Option<i64>,
    pub disagg_prefill_dp_rank: Option<i64>,
    pub disagg_prefill_serve_addr: Option<String>,
    pub conversation_id: Option<String>,
    pub routing_key: Option<String>,
    pub extra_key: Option<String>,
    pub cache_salt: Option<String>,
    pub custom_logit_processor: Option<String>,
    pub require_reasoning: bool,
    pub max_thinking_tokens: Option<i64>,
    pub priority: Option<i64>,
    pub log_metrics: Option<bool>,
    pub custom_labels: Option<BTreeMap<String, String>>,
    pub received_time: Option<f64>,
    #[serde(skip)]
    pub started: Option<std::time::Instant>,
    #[serde(skip)]
    pub metric_state: Option<Box<crate::metrics::RequestMetrics>>,
    /// Model-owned metadata attached only to the terminal native response.
    /// Never accepted from the HTTP body or passed to the scheduler.
    #[serde(skip)]
    pub response_metadata: Option<serde_json::Map<String, serde_json::Value>>,
    /// Only `set_canonical_mm_input` can install validated pad spans. They
    /// permit canonical media pads without weakening text-token validation.
    #[serde(skip)]
    pub(crate) mm_pad_spans: Vec<MmPadSpan>,
    /// Multimodal inputs. Consumed by the Encoding stage, which ships them to
    /// the MM worker pool; never read by the tokenizer or serialized onto the
    /// scheduler header. Boxed so the common text-only request doesn't grow
    /// every `Request` moved between stages.
    #[serde(flatten)]
    pub mm: Option<Box<MmData>>,
}

fn serialize_client_rid<S: serde::Serializer>(rid: &Rid, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(rid.client_facing())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MmPadSpan {
    pub start: usize,
    pub end: usize,
    pub pad_value: i32,
}

/// The multimodal fields of one request (see [`GenerateRequest::mm`]), each
/// modality already fanned out to this request's own item list.
///
/// Constructed directly only by tests: `api_server::prefetch` fills its
/// `prefetched` field, everything else gets it packed inside a `GenerateRequest`.
#[derive(Debug, Clone, Default, Serialize)]
pub struct MmData {
    pub image_data: Vec<MmItem>,
    pub video_data: Vec<MmItem>,
    pub audio_data: Vec<MmItem>,
    #[serde(flatten)]
    pub processor_extensions: ProcessorExtensions,
    #[serde(flatten)]
    pub processor_options: multimodal::MmProcessorOptions,
    /// Bytes of `image_data`'s I/O-backed sources, resolved by
    /// `api_server::prefetch` in `payload::io_sources` order so MM workers
    /// never block on I/O. Out-of-band: the values above stay as the client
    /// sent them.
    #[serde(skip)]
    pub prefetched: Vec<bytes::Bytes>,
    #[serde(skip)]
    pub prefetch_stats: MmPrefetchStats,
    /// See [`GenerateBody::mm_hashes`]; applied by the MM worker.
    pub mm_hashes: Vec<String>,
    /// See [`GenerateBody::mm_content_hashes`]; normalized before processing.
    pub mm_content_hashes: Option<Vec<Option<String>>>,
}

impl GenerateRequest {
    /// Install canonical input from a paired prefill. Every declared media span
    /// must contain exactly its pad value; ordinary ids still undergo vocabulary
    /// validation before admission. No media features are needed on decode.
    pub fn set_canonical_mm_input(
        &mut self,
        input_ids: Vec<i32>,
        mut spans: Vec<MmPadSpan>,
    ) -> Result<(), String> {
        if input_ids.is_empty() {
            return Err("input metadata token ids cannot be empty".into());
        }
        spans.sort_unstable_by_key(|span| span.start);
        let mut normalized: Vec<MmPadSpan> = Vec::with_capacity(spans.len());
        for span in spans {
            if span.start > span.end || span.end >= input_ids.len() || span.pad_value < 0 {
                return Err("invalid input metadata pad span".into());
            }
            if let Some(previous) = normalized.last_mut()
                && span.start <= previous.end
            {
                if previous.pad_value != span.pad_value {
                    return Err("input metadata has conflicting overlapping spans".into());
                }
                previous.end = previous.end.max(span.end);
            } else {
                normalized.push(span);
            }
        }
        for span in &normalized {
            if input_ids[span.start..=span.end]
                .iter()
                .any(|&id| id != span.pad_value)
            {
                return Err("input metadata span does not contain the declared pad value".into());
            }
        }
        self.input_ids = Some(input_ids);
        self.text = None;
        self.mm_pad_spans = normalized;
        self.mm = None;
        Ok(())
    }

    /// True when the client already supplied token ids → skip tokenization.
    pub fn already_tokenized(&self) -> bool {
        self.input_ids.as_ref().is_some_and(|v| !v.is_empty())
    }

    /// True when the request carries a usable multimodal payload — the mirror of
    /// Python `GenerateReqInput.contains_mm_input()`.
    pub fn has_multimodal(&self) -> bool {
        self.mm.as_ref().is_some_and(|mm| {
            !mm.image_data.is_empty()
                || !mm.video_data.is_empty()
                || !mm.audio_data.is_empty()
                || mm
                    .processor_extensions
                    .values()
                    .any(extension_value_present)
        })
    }

    /// Carve out the MM worker's inputs: `text` is cloned (the scheduler header
    /// still needs it), `input_ids` is taken (the expanded ids replace it), and
    /// the mm values move wholesale.
    pub fn take_mm_work(&mut self) -> MmWorkItem {
        let mut work = MmWorkItem {
            queued_at: Some(Instant::now()),
            bootstrap_room: self.bootstrap_room,
            text: self.text.clone(),
            input_ids: self.input_ids.take(),
            ..Default::default()
        };
        if let Some(m) = self.mm.as_deref_mut() {
            work.image_data = std::mem::take(&mut m.image_data);
            work.video_data = std::mem::take(&mut m.video_data);
            work.audio_data = std::mem::take(&mut m.audio_data);
            work.processor_extensions = std::mem::take(&mut m.processor_extensions);
            work.processor_options = std::mem::take(&mut m.processor_options);
            work.prefetched = std::mem::take(&mut m.prefetched);
            work.prefetch_stats = std::mem::take(&mut m.prefetch_stats);
            work.mm_hashes = std::mem::take(&mut m.mm_hashes);
            work.mm_content_hashes = m.mm_content_hashes.take();
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
pub(super) trait HeapBytes {
    fn heap_bytes(&self) -> usize;
}
impl HeapBytes for bool {
    fn heap_bytes(&self) -> usize {
        0
    }
}
impl HeapBytes for HiddenStatesMode {
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
impl HeapBytes for PositionalEmbeds {
    fn heap_bytes(&self) -> usize {
        self.embeds.iter().fold(
            self.positions
                .len()
                .saturating_mul(std::mem::size_of::<i64>()),
            |size, row| {
                size.saturating_add(std::mem::size_of::<Vec<f32>>())
                    .saturating_add(row.len().saturating_mul(std::mem::size_of::<f32>()))
            },
        )
    }
}
impl<T: HeapBytes> HeapBytes for Option<T> {
    fn heap_bytes(&self) -> usize {
        self.as_ref().map_or(0, HeapBytes::heap_bytes)
    }
}

/// Collapse `fan_out`'s nullable-element output: outer `None` (field absent /
/// scalar broadcast of nothing) and inner `None` (an explicit `null` list
/// element) both mean "not set".
fn flatten_column<T>(column: Vec<Option<Option<T>>>) -> Vec<Option<T>> {
    column.into_iter().map(Option::flatten).collect()
}

fn cache_key_column(
    value: Option<OneOrMany<String>>,
    n: usize,
    is_batch: bool,
    name: &str,
) -> Result<Vec<Option<String>>, Error> {
    if !is_batch && matches!(value, Some(OneOrMany::Many(_))) {
        return Err(Error::Validation(format!(
            "{name} should be a string for a single request"
        )));
    }
    Ok(fan_out(value, n, name)?
        .into_iter()
        .map(|key| key.filter(|key| !key.is_empty()))
        .collect())
}

/// Reject a broadcast whose clones would exceed [`MAX_BROADCAST_CLONE_BYTES`].
pub(super) fn check_broadcast_budget(per_clone: usize, n: usize, name: &str) -> Result<(), Error> {
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

    #[derive(Debug, Deserialize, PartialEq)]
    struct TestProcessorExtensions {
        multimodal_custom: TestProcessorExtension,
    }

    #[derive(Debug, Deserialize, PartialEq)]
    #[serde(deny_unknown_fields)]
    struct TestProcessorExtension {
        value: i64,
    }

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
        assert!(requests(r#"{"text": "a", "sampling_params": {"n": 0}}"#).is_err());
    }

    #[test]
    fn parallel_samples_preserve_prompt_order_seed_and_pairing_without_reexpansion() {
        let (samples, batch) = requests(
            r#"{
            "text":["first","second"], "rid":["a","b"],
            "bootstrap_room":[10,11], "return_logprob":true,
            "sampling_params":[
                {"n":3,"sampling_seed":17,"regex":"[a-z]+"},
                {"n":3,"sampling_seed":19,"regex":"[0-9]+"}
            ]
        }"#,
        )
        .unwrap();
        assert!(batch);
        assert_eq!(samples.len(), 6);
        for (index, sample) in samples.iter().enumerate() {
            let prompt = index / 3;
            assert_eq!(sample.text.as_deref(), Some(["first", "second"][prompt]));
            assert_eq!(sample.sampling_params.sampling_seed, Some([17, 19][prompt]));
            assert_eq!(
                sample.sampling_params.regex.as_deref(),
                Some(["[a-z]+", "[0-9]+"][prompt])
            );
            assert_eq!(
                sample.rid.client_facing(),
                format!("{}_{}", ["a", "b"][prompt], index % 3)
            );
            assert_eq!(sample.bootstrap_room, Some(30 + index as i64));
            assert!(sample.return_logprob);
            let forwarded: GenerateBody =
                serde_json::from_slice(&serde_json::to_vec(sample).unwrap()).unwrap();
            let (forwarded, batch) = forwarded.into_requests().unwrap();
            assert!(!batch);
            assert_eq!(forwarded.len(), 1);
            assert_eq!(forwarded[0].bootstrap_room, sample.bootstrap_room);
            assert_eq!(forwarded[0].rid.client_facing(), sample.rid.client_facing());
        }
        let (samples, batch) = requests(r#"{"text":"only","sampling_params":{"n":5}}"#).unwrap();
        assert!(batch);
        assert_eq!(samples.len(), 5);
        assert_eq!(
            samples
                .iter()
                .map(|sample| sample.rid.as_str())
                .collect::<HashSet<_>>()
                .len(),
            5
        );
        for (body, expected) in [
            (
                r#"{"text":["a","b"],"sampling_params":[{"n":2},{"n":3}]}"#,
                "same for all",
            ),
            (
                r#"{"text":"a","sampling_params":{"n":2},"return_logprob":[true]}"#,
                "Cannot use list",
            ),
            (
                r#"{"text":"a","sampling_params":{"n":2},"custom_logit_processor":[null]}"#,
                "Cannot use list custom_logit_processor",
            ),
            (
                r#"{"text":["a","b"],"sampling_params":{"n":9223372036854775807}}"#,
                "maximum",
            ),
        ] {
            assert!(requests(body).unwrap_err().to_string().contains(expected));
        }
    }

    #[test]
    fn unknown_generate_fields_are_ignored() {
        for field in [
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

    #[test]
    fn beam_requests_preserve_group_width_and_return_count() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../testdata/beam_outputs_python.json")).unwrap();
        for case in fixture["requests"].as_array().unwrap() {
            let (mut normalized, _) = requests(&case["body"].to_string()).unwrap();
            let expected = case["expected"].as_array().unwrap();
            assert_eq!(normalized.len(), expected.len(), "{case}");
            for (request, expected) in normalized.iter_mut().zip(expected) {
                request
                    .sampling_params
                    .normalize(true, fixture["vocab_size"].as_u64().unwrap())
                    .unwrap();
                assert_eq!(serde_json::json!(request.input_ids), expected["input_ids"]);
                assert_eq!(
                    serde_json::json!(request.sampling_params.beam_width),
                    expected["beam_width"]
                );
                // Ordinary n-way sampling is already split into independent
                // requests; a beam group's return count must reach the scheduler.
                assert_eq!(
                    serde_json::json!(request.sampling_params.n),
                    if expected["beam_width"].as_u64().unwrap() > 1 {
                        expected["n"].clone()
                    } else {
                        serde_json::json!(1)
                    }
                );
            }
        }
    }

    #[test]
    fn deferred_features_reject_nondefaults_but_accept_default_serialization() {
        for (name, value) in [
            ("session_id", serde_json::json!("session")),
            ("session_params", serde_json::json!({"id":"session"})),
            ("lora_path", serde_json::json!([null, "adapter"])),
            ("lora_id", serde_json::json!("adapter")),
            ("background", serde_json::json!(true)),
            ("no_logs", serde_json::json!(true)),
            (
                "external_trace_header",
                serde_json::json!({"trace":"value"}),
            ),
            ("http_worker_ipc", serde_json::json!("ipc:///worker")),
            ("need_wait_for_mm_inputs", serde_json::json!(true)),
            ("num_items_assigned", serde_json::json!({"image":[0]})),
            ("encoder_urls", serde_json::json!(["http://encoder"])),
        ] {
            let mut body = serde_json::json!({"text":"prompt"});
            body[name] = value;
            let error = requests(&body.to_string()).unwrap_err();
            assert_eq!(error.http_status(), 400);
            assert!(error.to_string().contains(name));
            for empty in [
                serde_json::Value::Null,
                serde_json::json!(false),
                serde_json::json!([]),
                serde_json::json!([null, null]),
                serde_json::json!({}),
                serde_json::json!(""),
            ] {
                body[name] = empty;
                assert!(requests(&body.to_string()).is_ok(), "{body}");
            }
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

    /// Full benchmark payloads include default values for optional features.
    /// Null media/adapters and disabled auxiliary outputs preserve text serving.
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
    /// request keeps its items; a batch broadcasts a scalar to every item, maps
    /// a list per item with matching lengths, and treats `null`/`[]` as absent.
    #[test]
    fn split_mm_fanout_matches_python_normalize() {
        let src = |s: &str| MmItem::Source(s.to_owned());
        let images_of = |p: &GenerateRequest| p.mm.as_ref().unwrap().image_data.clone();

        // Single request: one item, or a flat list, kept as sent.
        let (ps, _) = requests(r#"{"text": "a", "image_data": "http://x/i.jpg"}"#).unwrap();
        assert_eq!(images_of(&ps[0]), vec![src("http://x/i.jpg")]);
        assert!(ps[0].has_multimodal());
        let (ps, _) = requests(r#"{"text": "a", "image_data": ["u1", {"url": "u2"}]}"#).unwrap();
        assert_eq!(
            images_of(&ps[0]),
            vec![
                src("u1"),
                MmItem::Ref {
                    url: "u2".into(),
                    content_hash: None
                }
            ]
        );

        // Batch + scalar image: broadcast, one image per item.
        let (ps, _) = requests(r#"{"text": ["a", "b"], "image_data": "u"}"#).unwrap();
        for p in &ps {
            assert_eq!(images_of(p), vec![src("u")]);
            assert!(p.has_multimodal());
        }

        // Batch + per-item list: element i goes to item i; nested lists are
        // per-item lists.
        let (ps, _) = requests(r#"{"text": ["a", "b"], "image_data": ["u1", "u2"]}"#).unwrap();
        assert_eq!(images_of(&ps[0]), vec![src("u1")]);
        assert_eq!(images_of(&ps[1]), vec![src("u2")]);
        let (ps, _) =
            requests(r#"{"text": ["a", "b"], "image_data": [["u1", "u2"], null]}"#).unwrap();
        assert_eq!(images_of(&ps[0]), vec![src("u1"), src("u2")]);
        assert!(!ps[1].has_multimodal());

        // Batch + wrong-length list is a 400, as is the batch shape on a single.
        assert!(requests(r#"{"text": ["a", "b"], "image_data": ["u1"]}"#).is_err());
        assert!(requests(r#"{"text": "a", "image_data": [["u1"]]}"#).is_err());

        // null / [] mean "no multimodal input".
        let (ps, _) = requests(r#"{"text": "a", "image_data": null}"#).unwrap();
        assert!(!ps[0].has_multimodal());
        let (ps, _) = requests(r#"{"text": "a", "image_data": []}"#).unwrap();
        assert!(!ps[0].has_multimodal());

        // Batch + scalar video broadcasts too (Python leaves it unwrapped, but
        // every request's input is an item list here).
        let (ps, _) = requests(r#"{"text": ["a", "b"], "video_data": "v"}"#).unwrap();
        assert_eq!(ps[1].mm.as_ref().unwrap().video_data, vec![src("v")]);
        assert!(ps[1].has_multimodal());
    }

    #[test]
    fn multimodal_extensions_follow_request_batch_shape() {
        let single = r#"{"input_ids":[9],"image_data":"u","multimodal_placeholders":[{"type":"image","token_index":0,"item_index":0}]}"#;
        let (reqs, is_batch) = requests(single).unwrap();
        assert!(!is_batch);
        let value = reqs[0]
            .mm
            .as_ref()
            .unwrap()
            .processor_extensions
            .0
            .get("multimodal_placeholders")
            .unwrap();
        assert_eq!(value.as_array().unwrap().len(), 1);

        let batched = r#"{"input_ids":[[9],[8]],"image_data":["u","v"],"multimodal_placeholders":[[{"type":"image","token_index":0,"item_index":0}],[{"type":"image","token_index":0,"item_index":0}]]}"#;
        let (reqs, is_batch) = requests(batched).unwrap();
        assert!(is_batch);
        assert_eq!(reqs.len(), 2);
        assert!(reqs.iter().all(GenerateRequest::has_multimodal));
        assert!(reqs.iter().all(|request| {
            request
                .mm
                .as_ref()
                .and_then(|mm| mm.processor_extensions.0.get("multimodal_placeholders"))
                .and_then(rmpv::Value::as_array)
                .is_some_and(|placeholders| placeholders.len() == 1)
        }));

        let invalid = r#"{"input_ids":[[9],[8]],"image_data":["u","v"],"multimodal_placeholders":[{"type":"image","token_index":0,"item_index":0}]}"#;
        assert!(requests(invalid).is_err());

        let generic = r#"{"input_ids":[[9],[8]],"image_data":["u","v"],"multimodal_custom":[{"value":1},{"value":2}]}"#;
        let (reqs, _) = requests(generic).unwrap();
        assert_eq!(
            reqs[1]
                .mm
                .as_ref()
                .unwrap()
                .processor_extensions
                .0
                .get("multimodal_custom")
                .unwrap()
                .as_map()
                .unwrap()[0]
                .1
                .as_i64(),
            Some(2)
        );

        let extensions: TestProcessorExtensions =
            requests(r#"{"input_ids":[9],"multimodal_custom":{"value":3}}"#)
                .unwrap()
                .0
                .pop()
                .unwrap()
                .mm
                .unwrap()
                .processor_extensions
                .deserialize()
                .unwrap();
        assert_eq!(extensions.multimodal_custom.value, 3);

        for fields in [
            r#"{"multimodal_custom":{"value":true}}"#,
            r#"{"multimodal_custom":{"value":"3"}}"#,
            r#"{"multimodal_custom":{"value":3,"unknown":0}}"#,
            r#"{"multimodal_custom":{}}"#,
        ] {
            let extensions: ProcessorExtensions = serde_json::from_str(fields).unwrap();
            assert!(
                extensions.deserialize::<TestProcessorExtensions>().is_err(),
                "{fields}"
            );
        }

        let (reqs, _) = requests(r#"{"text":"hi","totally_made_up":1}"#).unwrap();
        assert!(reqs[0].mm.is_none());

        let (reqs, _) = requests(r#"{"input_ids":[9],"multimodal_custom":null}"#).unwrap();
        assert!(!reqs[0].has_multimodal());
    }

    /// A scalar broadcast is budget-checked before the deep clones (16 MiB ×
    /// 4096 prompts would be 64 GiB and an abort); per-item lists clone nothing
    /// and are never charged.
    #[test]
    fn oversized_mm_broadcast_rejected() {
        let big = MmItem::Source("x".repeat(MAX_BROADCAST_CLONE_BYTES / 2 + 1));
        let err = multimodal::fan_out(Some(MmDataInput::One(big.clone())), 2, true, "image_data")
            .err()
            .unwrap();
        assert!(err.to_string().contains("broadcast"), "{err}");
        // A per-item list of the same total size moves, not clones: accepted.
        let list = MmDataInput::Many(vec![Some(big), Some(MmItem::Source("y".into()))]);
        assert!(multimodal::fan_out(Some(list), 2, true, "image_data").is_ok());
        // Small scalars broadcast fine.
        let small = MmDataInput::One(MmItem::Source("u1".into()));
        assert!(multimodal::fan_out(Some(small), 2, true, "audio_data").is_ok());
    }

    /// Hashes follow their image lists through batching, forwarding, and the
    /// worker handoff. Python generates the expected normalization and errors.
    #[test]
    fn mm_hashes_match_python_request_contract() {
        let cases: serde_json::Value =
            serde_json::from_str(include_str!("../../testdata/mm_hashes_python.json")).unwrap();
        for case in cases.as_array().unwrap() {
            let normalized = serde_json::from_value::<GenerateBody>(case["body"].clone())
                .unwrap()
                .into_requests();
            if let Err(error) = &normalized {
                assert_eq!(error.http_status(), 400, "{case}");
            }
            let result = (|| {
                let (requests, _) = normalized.map_err(|error| error.to_string())?;
                let mut values = Vec::new();
                for mut request in requests {
                    // DP forwarding serializes the normalized child back through
                    // GenerateBody before running its processor.
                    let forwarded = serde_json::to_value(&request).unwrap();
                    let (mut children, _) = serde_json::from_value::<GenerateBody>(forwarded)
                        .unwrap()
                        .into_requests()
                        .unwrap();
                    assert_eq!(children.len(), 1);
                    let direct = request.take_mm_work();
                    let work = children[0].take_mm_work();
                    assert_eq!(direct.mm_hashes, work.mm_hashes);
                    assert_eq!(direct.mm_content_hashes, work.mm_content_hashes);
                    assert!(request.mm.as_ref().unwrap().mm_hashes.is_empty());
                    let content = multimodal::normalize_content_hashes(
                        &work.image_data,
                        work.mm_content_hashes,
                    )?;
                    values.push(serde_json::json!({
                        "mm_hashes": work.mm_hashes, "mm_content_hashes": content,
                    }));
                }
                Ok::<_, String>(values)
            })();
            match result {
                Ok(values) => assert_eq!(serde_json::json!(values), case["expected"], "{case}"),
                Err(error) => assert!(
                    error.contains(case["error"].as_str().unwrap()),
                    "{error}: {case}"
                ),
            }
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
        assert_eq!(work.image_data.len(), 2);
        assert!(work.video_data.is_empty());
        assert_eq!(work.audio_data, vec![MmItem::Source("a".into())]);
        // Moved out, not cloned; `text` survives for the header.
        assert!(ps[0].mm.as_ref().unwrap().image_data.is_empty());
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
        assert_eq!(ps[1].return_hidden_states, HiddenStatesMode::Full);

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

    #[test]
    fn prompt_ids_option_survives_batches_parallel_samples_and_dp_forwarding() {
        let body: GenerateBody = serde_json::from_value(serde_json::json!({
            "input_ids": [[1, 2], [3]],
            "return_prompt_token_ids": true,
            "sampling_params": {"n": 2},
        }))
        .unwrap();
        let (requests, is_batch) = body.into_requests().unwrap();
        assert!(is_batch);
        assert_eq!(requests.len(), 4);
        for request in requests {
            assert!(request.return_prompt_token_ids);
            let forwarded: GenerateBody =
                serde_json::from_value(serde_json::to_value(request).unwrap()).unwrap();
            assert!(forwarded.into_requests().unwrap().0[0].return_prompt_token_ids);
        }
    }
}
