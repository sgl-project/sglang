//! The response direction: the per-request back-channel the API handler
//! drains ([`ResponseSink`] / [`ResponseItem`]), the response frame encodings
//! (batch / control result / error), and the columnar batch decode into
//! per-request [`ChunkEvent`]s.

mod flat_logprobs;
mod speculative;

pub use flat_logprobs::{FlatTopLogprobShape, FlatTopLogprobs, LogprobOptions};

use std::collections::BTreeMap;
use std::sync::Arc;

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use super::finish_reason::FinishReason;
use super::types::TokenIds;
use crate::message::ids::Rid;
use crate::utils::error::Error;
use speculative::SpeculativeStats;

/// Python's `INIT_INCREMENTAL_DETOKENIZATION_OFFSET`: the unpadded prompt tail
/// needed to distinguish a generated prefix from the tokenizer's prompt text.
pub(crate) const PROMPT_CONTEXT_TOKENS: usize = 5;

/// Per-request back-channel the detok shard writes decode frames to and the API
/// handler drains for SSE; bounded, and receiver-drop (disconnect) = stream end.
#[derive(Clone, Debug)]
pub enum ResponseSink {
    Local(mpsc::Sender<ResponseItem>),
}

/// Why an [`ResponseSink::try_send`] failed: `Full` = client backpressure, `Closed`
/// = client gone. Both terminal for a stream; the caller distinguishes for logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SinkError {
    Full,
    Closed,
}

impl ResponseSink {
    pub fn is_closed(&self) -> bool {
        match self {
            ResponseSink::Local(tx) => tx.is_closed(),
        }
    }
    /// Non-blocking send. `Err(Full)` = backpressure, `Err(Closed)` = client gone.
    pub fn try_send(&self, item: ResponseItem) -> Result<(), SinkError> {
        match self {
            ResponseSink::Local(tx) => tx.try_send(item).map_err(|e| match e {
                mpsc::error::TrySendError::Full(_) => SinkError::Full,
                mpsc::error::TrySendError::Closed(_) => SinkError::Closed,
            }),
        }
    }
}

#[allow(dead_code)] // the receiver half is created inline in api_server::submit.
pub type ResponseSource = mpsc::Receiver<ResponseItem>;

/// What the connection handler receives on the decode stream: a detok-decoded
/// [`ChunkEvent`] (handler formats it), a verbatim control payload, or an error.
#[derive(Debug)]
pub enum ResponseItem {
    Tokenized(crate::message::types::TokenIds),
    /// An intermediate streamed generation step (only sent for streaming reqs).
    Frame(ChunkEvent),
    /// The final generation step.
    Done(ChunkEvent),
    /// A control-request result: one verbatim payload (e.g. `/server_info`),
    /// delivered as-is with no per-protocol formatting.
    Control(Vec<Bytes>),
    /// Reply to an internal service request (`RequestKind::Detokenize`): raw
    /// bytes for the SUBMITTER to consume (e.g. the decoded prompt text), not
    /// client-bound JSON like `Control` and not a generation frame. Generation
    /// and control drains never see it.
    Data(Bytes),
    /// Terminal failure: handler emits an error frame (stream) or status (unary).
    Error(Error),
}

/// Response frame tag (first byte, prepended Rust-side; Python wire unchanged):
/// a single control-request result payload.
pub const DISPATCH_TAG_RESULT: u8 = 1;
/// A whole decode batch: msgpack columnar header + one concatenated raw buffer;
/// from-scheduler decodes it into per-request [`ChunkEvent`]s (no per-request FFI).
pub const DISPATCH_TAG_BATCH: u8 = 2;
/// A per-request failure `[rid, message]`: the Python drain couldn't decode a
/// header, so it routes a 400 back to that request instead of crashing the loop.
pub const DISPATCH_TAG_ERROR: u8 = 3;
/// A scheduler abort echo, preserving the generation state already delivered.
pub const DISPATCH_TAG_ABORT: u8 = 4;
pub const DISPATCH_TAG_RESULT_PART: u8 = 5;

/// Read `n` little-endian f32s from `data` at `*off`, advancing `*off`. `None` when
/// the range runs past the buffer (a malformed / positional-ABI-drifted frame): the
/// caller rejects the whole frame. Bounds-checked via `data.get` — clamping only the
/// end is unsafe because a prior bad length can push `*off` past `len`, making the
/// range reversed (`start > end`) and the slice panic.
fn take_f32(data: &[u8], off: &mut usize, n: usize) -> Option<Vec<f32>> {
    let start = *off;
    let end = start.checked_add(n.checked_mul(4)?)?;
    let out = data
        .get(start..end)?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    *off = end;
    Some(out)
}

/// Read `n` little-endian i32s from `data` at `*off`, advancing `*off`. `None` past
/// the buffer end (see [`take_f32`]).
fn take_i32(data: &[u8], off: &mut usize, n: usize) -> Option<Vec<i32>> {
    let start = *off;
    let end = start.checked_add(n.checked_mul(4)?)?;
    let out = data
        .get(start..end)?
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    *off = end;
    Some(out)
}

/// Frame a decode batch: `[BATCH tag][u32 header len][header][data cols…]`. The
/// caller's `data_cols` are concatenated straight into the frame (one copy, no
/// `b"".join`); `header` is the msgpack [`BatchHeader`].
pub fn frame_decode_batch_cols(header: &[u8], data_cols: &[&[u8]]) -> Bytes {
    let data_len: usize = data_cols.iter().map(|c| c.len()).sum();
    let mut buf = Vec::with_capacity(1 + 4 + header.len() + data_len);
    buf.push(DISPATCH_TAG_BATCH);
    buf.extend_from_slice(&(header.len() as u32).to_le_bytes());
    buf.extend_from_slice(header);
    for col in data_cols {
        buf.extend_from_slice(col);
    }
    Bytes::from(buf)
}

/// Columnar scalar header for a whole decode batch. The first four fields are
/// required; later fields default empty for legacy frames. Field order is the
/// wire ABI and must match
/// `RustServer.push_generation`'s `header_cols` in
/// `python/sglang/srt/rust_server/server.py`.
///
/// Field names follow `direction_family_shape`:
/// - direction: `out` = decode output, `in` = prefill input;
/// - family: `lp` = token logprobs, `top` = top-k logprobs, `tokids_lp` =
///   requested-token logprobs, and `hidden` = hidden states;
/// - shape: `lens` counts elements per request, `reqlens` counts positions or
///   rows per request, and `poslens` counts elements per position or row.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct BatchHeader {
    /// Request ids, as the same strings Python holds (`Req.rid`, uuid hex) —
    /// hashed back to the internal routing key in `decode_one`
    /// (`Rid::shard`), mirroring the control path. The wire has no
    /// rid-shape coupling; any string is a valid rid.
    pub rids: Vec<String>,
    pub finish_reasons: Vec<Option<FinishReason>>,
    pub prompt_tokens: Vec<u32>,
    pub tok_lens: Vec<u32>,
    #[serde(default)]
    pub out_lp_lens: Vec<u32>,
    #[serde(default)]
    pub in_lp_lens: Vec<u32>,
    #[serde(default)]
    pub out_top_reqlens: Vec<u32>,
    #[serde(default)]
    pub out_top_poslens: Vec<u32>,
    #[serde(default)]
    pub in_top_reqlens: Vec<u32>,
    #[serde(default)]
    pub in_top_poslens: Vec<u32>,
    #[serde(default)]
    pub out_tokids_lp_reqlens: Vec<u32>,
    #[serde(default)]
    pub out_tokids_lp_poslens: Vec<u32>,
    #[serde(default)]
    pub in_tokids_lp_reqlens: Vec<u32>,
    #[serde(default)]
    pub in_tokids_lp_poslens: Vec<u32>,
    #[serde(default)]
    pub hidden_reqlens: Vec<u32>,
    #[serde(default)]
    pub hidden_poslens: Vec<u32>,
    #[serde(default)]
    pub counts: Option<BatchTokenCounts>,
    #[serde(default)]
    pub customized_info: BTreeMap<String, Vec<Vec<serde_json::Value>>>,
    #[serde(default)]
    pub prompt_contexts: Vec<Vec<i32>>,
    #[serde(default)]
    pub hidden_shapes: Vec<Option<HiddenStateShape>>,
    /// Per request: absent, or one nullable support length per emitted token.
    /// Raw support IDs and selected-token logprobs follow the hidden-state data.
    #[serde(default)]
    pub sampling_mask_shapes: Vec<Option<Vec<Option<u32>>>>,
    #[serde(default)]
    pub flat_top_logprob_shapes: Vec<Option<FlatTopLogprobShape>>,
    /// Byte lengths, with null for an absent request and zero for an empty tensor.
    #[serde(default)]
    pub routed_experts_bytes: Vec<Option<u32>>,
    #[serde(default)]
    pub indexer_topk_bytes: Vec<Option<u32>>,
    /// Ranked candidates per request; generated token IDs follow all tensor data.
    #[serde(default)]
    pub beam_headers: Vec<Option<Vec<BeamSequenceHeader>>>,
    /// Scheduler-owned timestamps and queue durations, when metrics are enabled.
    #[serde(default)]
    pub time_metadata: Vec<serde_json::Map<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeamSequenceHeader {
    pub token_count: u32,
    pub finish_reason: Option<FinishReason>,
    pub sequence_score: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct BeamSequence {
    pub token_ids: Vec<i32>,
    pub finish_reason: Option<FinishReason>,
    pub sequence_score: Option<f64>,
    pub text: Option<String>,
}

#[derive(Debug, Clone)]
pub struct BeamOutput {
    pub sequences: Vec<BeamSequence>,
    // Python copies scheduler metadata onto the first candidate before
    // frontend response processors annotate the top-level result.
    pub scheduler_metadata: OutputMetadata,
}

/// Shape of Python's nested hidden-state lists. Numeric data remains in the
/// shared f32 column; a vector contributes only its length to the header.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum HiddenStateShape {
    Vector(u32),
    Nested(Vec<HiddenStateShape>),
}

impl HiddenStateShape {
    fn elements(&self) -> Option<usize> {
        match self {
            Self::Vector(length) => Some(*length as usize),
            Self::Nested(children) => children
                .iter()
                .try_fold(0usize, |total, child| total.checked_add(child.elements()?)),
        }
    }

    pub fn reshape(&self, values: &[f32]) -> Option<serde_json::Value> {
        fn read(shape: &HiddenStateShape, values: &mut &[f32]) -> Option<serde_json::Value> {
            match shape {
                HiddenStateShape::Vector(length) => {
                    let (head, tail) = values.split_at_checked(*length as usize)?;
                    *values = tail;
                    Some(serde_json::json!(head))
                }
                HiddenStateShape::Nested(children) => children
                    .iter()
                    .map(|child| read(child, values))
                    .collect::<Option<Vec<_>>>()
                    .map(serde_json::Value::Array),
            }
        }
        let mut remaining = values;
        let result = read(self, &mut remaining)?;
        remaining.is_empty().then_some(result)
    }
}

/// Scheduler-owned accounting stays columnar across the Python/Rust boundary.
/// Unlike emitted token deltas, these counts are cumulative request snapshots.
#[derive(Debug, Serialize, Deserialize)]
pub struct BatchTokenCounts {
    pub reasoning_tokens: Vec<u64>,
    pub cached_tokens: Vec<u64>,
    #[serde(default)]
    pub retraction_counts: Vec<u64>,
    #[serde(default)]
    pub cached_tokens_details: Vec<Option<CachedTokensDetails>>,
    #[serde(default)]
    pub dp_ranks: Vec<Option<u32>>,
    #[serde(default)]
    pub image_tokens: Vec<u64>,
    #[serde(default)]
    pub audio_tokens: Vec<u64>,
    #[serde(default)]
    pub video_tokens: Vec<u64>,
    #[serde(default)]
    pub weight_version: Option<String>,
    #[serde(default)]
    pub weight_versions: Vec<Option<Vec<WeightVersionSpan>>>,
    #[serde(default)]
    pub generation_tokens: Vec<u64>,
    #[serde(default)]
    pub spec_verify_ct: Vec<u64>,
    #[serde(default)]
    pub spec_num_correct_drafts: Vec<u64>,
    #[serde(default)]
    pub spec_num_cap_tokens: Vec<u64>,
    #[serde(default)]
    pub spec_num_block_accept_tokens: Vec<u64>,
    #[serde(default)]
    pub spec_correct_drafts_histogram: Vec<Vec<u64>>,
    #[serde(default)]
    pub spec_cap_lens_histogram: Vec<Vec<u64>>,
    #[serde(default)]
    pub spec_num_draft_tokens: u64,
    #[serde(default)]
    pub spec_ragged_verify_cap_accept: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightVersionSpan {
    pub version: String,
    pub start: u64,
    pub end: u64,
}

#[derive(Debug, Deserialize)]
pub struct SchedulerAbort {
    pub finished_reason: FinishReason,
    pub weight_version: Option<String>,
    pub weight_versions: Option<Vec<WeightVersionSpan>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CachedTokensDetails {
    pub device: u64,
    pub host: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub storage: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub storage_backend: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct TokenCounts {
    #[serde(skip)]
    pub generation_tokens: Option<u64>,
    #[serde(skip)]
    pub spec_verify_ct: Option<u64>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub speculative: Option<Box<SpeculativeStats>>,
    pub reasoning_tokens: u64,
    pub cached_tokens: u64,
    pub num_retractions: u64,
    // The outer option preserves an absent batch column versus a null entry.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens_details: Option<Option<CachedTokensDetails>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dp_rank: Option<Option<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub video_tokens: Option<u64>,
    #[serde(skip)]
    pub weight_version: Option<String>,
    /// Spans are projected onto visible output ids by `metadata`, including
    /// when the final stop token has been trimmed or the request was aborted.
    #[serde(skip)]
    pub weight_versions: Option<Vec<WeightVersionSpan>>,
}

impl TokenCounts {
    pub fn metadata(&self, visible_tokens: usize) -> impl Serialize + '_ {
        TokenMetadata {
            counts: self,
            weight_version: self
                .weight_versions
                .as_ref()
                .and_then(|spans| {
                    spans
                        .iter()
                        .rfind(|span| span.start == 0 || span.start < visible_tokens as u64)
                })
                .map(|span| span.version.as_str())
                .or(self.weight_version.as_deref()),
            weight_versions: self
                .weight_versions
                .as_ref()
                .map(|spans| VisibleWeightVersions {
                    spans,
                    visible_tokens: visible_tokens as u64,
                }),
        }
    }
}

#[derive(Serialize)]
struct TokenMetadata<'a> {
    #[serde(flatten)]
    counts: &'a TokenCounts,
    #[serde(skip_serializing_if = "Option::is_none")]
    weight_version: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    weight_versions: Option<VisibleWeightVersions<'a>>,
}

struct VisibleWeightVersions<'a> {
    spans: &'a [WeightVersionSpan],
    visible_tokens: u64,
}

impl Serialize for VisibleWeightVersions<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct VisibleSpan<'a> {
            version: &'a str,
            start: u64,
            end: u64,
        }

        serializer.collect_seq(
            self.spans
                .iter()
                .filter(|span| span.start == 0 || span.start < self.visible_tokens)
                .map(|span| VisibleSpan {
                    version: &span.version,
                    start: span.start,
                    end: span.end.min(self.visible_tokens),
                }),
        )
    }
}

impl BatchTokenCounts {
    fn valid_batch_size(&self, n: usize) -> bool {
        self.spec_verify_ct.iter().all(|count| {
            count
                .checked_mul(self.spec_num_draft_tokens.saturating_sub(1))
                .is_some()
        }) && self.reasoning_tokens.len() == n
            && self.cached_tokens.len() == n
            && [
                self.cached_tokens_details.len(),
                self.retraction_counts.len(),
                self.dp_ranks.len(),
                self.image_tokens.len(),
                self.audio_tokens.len(),
                self.video_tokens.len(),
                self.weight_versions.len(),
                self.generation_tokens.len(),
                self.spec_verify_ct.len(),
            ]
            .into_iter()
            .all(|len| len == 0 || len == n)
            && [
                self.spec_num_correct_drafts.len(),
                self.spec_num_cap_tokens.len(),
                self.spec_num_block_accept_tokens.len(),
                self.spec_correct_drafts_histogram.len(),
                self.spec_cap_lens_histogram.len(),
            ]
            .into_iter()
            .all(|len| len <= n)
    }

    fn take_request(&mut self, i: usize) -> TokenCounts {
        let nonzero = |column: &[u64]| column.get(i).copied().filter(|&count| count != 0);
        TokenCounts {
            speculative: SpeculativeStats::take(self, i).map(Box::new),
            generation_tokens: self.generation_tokens.get(i).copied(),
            spec_verify_ct: self.spec_verify_ct.get(i).copied(),
            reasoning_tokens: self.reasoning_tokens[i],
            cached_tokens: self.cached_tokens[i],
            num_retractions: self.retraction_counts.get(i).copied().unwrap_or(0),
            cached_tokens_details: self.cached_tokens_details.get_mut(i).map(Option::take),
            dp_rank: self.dp_ranks.get(i).copied(),
            image_tokens: nonzero(&self.image_tokens),
            audio_tokens: nonzero(&self.audio_tokens),
            video_tokens: nonzero(&self.video_tokens),
            weight_version: self.weight_version.clone(),
            weight_versions: self.weight_versions.get_mut(i).and_then(Option::take),
        }
    }
}

/// Read a request's flat logprob column (`l` val/idx pairs) from `data` at cursors
/// `cv`/`ci`, advancing them. Only this request is copied — no whole-column buffer.
/// `None` if either read runs past the buffer (see [`take_f32`]).
fn take_flat(
    data: &[u8],
    cv: &mut usize,
    ci: &mut usize,
    l: usize,
) -> Option<(Vec<f32>, Vec<i32>)> {
    Some((take_f32(data, cv, l)?, take_i32(data, ci, l)?))
}

/// Read `np` per-position lengths from `poslens` at `*pcur`, advancing it. `None`
/// when the range runs past the column — the header's `reqlens` promised more
/// positions than `poslens` carries, so this request's lengths are unknowable.
/// Clamping instead would return a short `lens`, and since `lens` also drives how
/// far the val/idx cursors advance, every later request in that column would read
/// from the wrong offset: silently wrong logprobs rather than a rejected frame.
fn take_poslens(poslens: &[u32], pcur: &mut usize, np: usize) -> Option<Vec<u32>> {
    let start = *pcur;
    let end = start.checked_add(np)?;
    let lens = poslens.get(start..end)?.to_vec();
    *pcur = end;
    Some(lens)
}

/// Read a request's ragged logprob column (`np` positions): its per-position `lens`
/// from `poslens` (advancing `pcur`), then that many val/idx from `cv`/`ci`. `None`
/// if either read runs past its buffer (see [`take_poslens`], [`take_f32`]).
fn take_ragged(
    data: &[u8],
    cv: &mut usize,
    ci: &mut usize,
    poslens: &[u32],
    pcur: &mut usize,
    np: usize,
) -> Option<(Vec<f32>, Vec<i32>, Vec<u32>)> {
    let lens = take_poslens(poslens, pcur, np)?;
    let nv: usize = lens.iter().map(|&x| x as usize).sum();
    Some((take_f32(data, cv, nv)?, take_i32(data, ci, nv)?, lens))
}

/// Like [`take_ragged`] but for hidden states — a val column + row `poslens`, no
/// idx column. `None` if either read runs past its buffer (see [`take_poslens`],
/// [`take_f32`]).
fn take_hidden(
    data: &[u8],
    cv: &mut usize,
    poslens: &[u32],
    pcur: &mut usize,
    nr: usize,
) -> Option<(Vec<f32>, Vec<u32>)> {
    let lens = take_poslens(poslens, pcur, nr)?;
    let nv: usize = lens.iter().map(|&x| x as usize).sum();
    Some((take_f32(data, cv, nv)?, lens))
}

/// Decode a batch frame (tag stripped), calling `route` with each request's
/// [`ChunkEvent`] as it's decoded — one pass, no intermediate `Vec`, peak memory
/// one request. Column order matches `push_generation`.
///
/// `ok == false` means the frame was rejected. The caller discards everything it
/// routed and fails the frame's requests instead of forwarding a partial fan-out
/// (see `tokenizer_manager::from_scheduler`), so a rejected frame delivers nothing —
/// `rids` exists precisely so those requests can be failed rather than left
/// waiting for a `Done` that no longer exists.
pub fn for_each_chunk(body: &[u8], mut route: impl FnMut(ChunkEvent)) -> Decoded {
    let mut decoded = Decoded::default();
    if body.len() < 4 {
        return decoded;
    }
    let hlen = u32::from_le_bytes([body[0], body[1], body[2], body[3]]) as usize;
    let Some(header) = body.get(4..4 + hlen) else {
        return decoded;
    };
    // Every rejection past the header slice names its requests by re-reading the
    // rid column, rather than the whole frame paying a clone for the error path.
    macro_rules! reject {
        () => {{
            decoded.rids = recover_rids(header);
            return decoded;
        }};
    }

    let data = &body[4 + hlen..];
    let mut h = match rmp_serde::from_slice::<BatchHeader>(header) {
        Ok(h) => h,
        Err(_) => reject!(),
    };
    let n = h.rids.len();
    // Every column must agree with `rids`. A SHORT column contributes nothing to
    // `base`, so the `base != data.len()` check below cannot see it: a short
    // `tok_lens` delivered a 200 with an empty completion, and a short
    // `finish_reasons` dropped the terminal marker so the request never completed
    // and its unary drain pended forever. The producer already asserts this for the
    // extras columns; the four core ones were unchecked.
    if h.finish_reasons.len() != n || h.prompt_tokens.len() != n || h.tok_lens.len() != n {
        reject!()
    }
    if h.counts
        .as_ref()
        .is_some_and(|counts| !counts.valid_batch_size(n))
    {
        reject!()
    }
    if h.customized_info.values().any(|column| column.len() != n) {
        reject!()
    }
    if !h.hidden_shapes.is_empty() && h.hidden_shapes.len() != n {
        reject!()
    }
    if !h.sampling_mask_shapes.is_empty() && h.sampling_mask_shapes.len() != n {
        reject!()
    }
    if !h.flat_top_logprob_shapes.is_empty() && h.flat_top_logprob_shapes.len() != n {
        reject!()
    }
    for lengths in [&h.routed_experts_bytes, &h.indexer_topk_bytes] {
        if !lengths.is_empty() && lengths.len() != n {
            reject!()
        }
    }
    if !h.beam_headers.is_empty() && h.beam_headers.len() != n {
        reject!()
    }
    if !h.time_metadata.is_empty() && h.time_metadata.len() != n {
        reject!()
    }
    if (!h.prompt_contexts.is_empty() && h.prompt_contexts.len() != n)
        || h.prompt_contexts
            .iter()
            .any(|ids| ids.len() > PROMPT_CONTEXT_TOKENS)
    {
        reject!()
    }
    // The per-request extras columns are either absent (no request asked) or one
    // entry per request — never partial.
    let per_req_ok = |c: &[u32]| c.is_empty() || c.len() == n;
    if !per_req_ok(&h.out_lp_lens)
        || !per_req_ok(&h.in_lp_lens)
        || !per_req_ok(&h.out_top_reqlens)
        || !per_req_ok(&h.in_top_reqlens)
        || !per_req_ok(&h.out_tokids_lp_reqlens)
        || !per_req_ok(&h.in_tokids_lp_reqlens)
        || !per_req_ok(&h.hidden_reqlens)
    {
        reject!()
    }
    let sum = |v: &[u32]| v.iter().map(|&x| x as usize).sum::<usize>();
    // Each ragged family's per-request counts must consume its position column
    // EXACTLY. Only the deficit was caught (`take_poslens` runs off the end); a
    // surplus left positions unread and delivered a truncated row with a 200.
    if sum(&h.out_top_reqlens) != h.out_top_poslens.len()
        || sum(&h.in_top_reqlens) != h.in_top_poslens.len()
        || sum(&h.out_tokids_lp_reqlens) != h.out_tokids_lp_poslens.len()
        || sum(&h.in_tokids_lp_reqlens) != h.in_tokids_lp_poslens.len()
        || sum(&h.hidden_reqlens) != h.hidden_poslens.len()
    {
        reject!()
    }

    // Per-column byte cursors, advanced per request — no whole-column read. Columns
    // are concatenated in exactly this order, every element 4 bytes.
    let mut base = 0usize;
    let mut col = |count: usize| -> usize {
        let start = base;
        base = base.saturating_add(count.saturating_mul(4));
        start
    };
    // Each val/idx column pair shares one element count — sum it once.
    let n_ids = sum(&h.tok_lens);
    let n_olp = sum(&h.out_lp_lens);
    let n_ilp = sum(&h.in_lp_lens);
    let n_ot = sum(&h.out_top_poslens);
    let n_it = sum(&h.in_top_poslens);
    let n_od = sum(&h.out_tokids_lp_poslens);
    let n_id = sum(&h.in_tokids_lp_poslens);
    let n_h = sum(&h.hidden_poslens);
    let mut c_ids = col(n_ids);
    let mut c_olp_v = col(n_olp);
    let mut c_olp_i = col(n_olp);
    let mut c_ilp_v = col(n_ilp);
    let mut c_ilp_i = col(n_ilp);
    let mut c_ot_v = col(n_ot);
    let mut c_ot_i = col(n_ot);
    let mut c_it_v = col(n_it);
    let mut c_it_i = col(n_it);
    let mut c_od_v = col(n_od);
    let mut c_od_i = col(n_od);
    let mut c_id_v = col(n_id);
    let mut c_id_i = col(n_id);
    let mut c_h_v = col(n_h);
    let n_mask_ids = h
        .sampling_mask_shapes
        .iter()
        .flatten()
        .flatten()
        .flatten()
        .map(|&length| length as usize)
        .sum();
    let n_mask_logprobs = h.sampling_mask_shapes.iter().flatten().map(Vec::len).sum();
    let mut c_mask_ids = col(n_mask_ids);
    let mut c_mask_logprobs = col(n_mask_logprobs);
    let Some(n_flat_top) = h
        .flat_top_logprob_shapes
        .iter()
        .flatten()
        .try_fold(0usize, |total, shape| total.checked_add(shape.elements()?))
    else {
        reject!()
    };
    let mut c_flat_top_values = col(n_flat_top);
    let mut c_flat_top_indices = col(n_flat_top);
    let mut c_routed_experts = base;
    for &length in h.routed_experts_bytes.iter().flatten() {
        base = base.saturating_add(length as usize);
    }
    let mut c_indexer_topk = base;
    for &length in h.indexer_topk_bytes.iter().flatten() {
        base = base.saturating_add(length as usize);
    }
    let mut c_beam_tokens = base;
    for sequence in h.beam_headers.iter().flatten().flatten() {
        base = base.saturating_add((sequence.token_count as usize).saturating_mul(4));
    }

    // `col` summed every column's span into `base`, so a truncated frame is caught
    // here — the one rejection that is genuinely whole-frame, since it precedes the
    // routing loop. Past this point a failure can only be partial.
    //
    // The check is EQUALITY, not `>`: every column length is header-determined, so
    // a data buffer longer than `base` means the header and the buffer disagree.
    // Accepting the surplus let a producer-side val/idx mismatch through with a
    // 200, every later column reading off by the difference.
    if base != data.len() {
        reject!()
    }

    // Mirror of Python's `has_extra` guard: checking once per frame lets the
    // per-request loop skip the extras machinery entirely on a plain decode frame.
    let has_extras = !(h.out_lp_lens.is_empty()
        && h.in_lp_lens.is_empty()
        && h.out_top_reqlens.is_empty()
        && h.in_top_reqlens.is_empty()
        && h.out_tokids_lp_reqlens.is_empty()
        && h.in_tokids_lp_reqlens.is_empty()
        && h.hidden_reqlens.is_empty());

    // Position cursors into the header's per-request `poslens` (ragged + hidden).
    let (mut p_ot, mut p_it, mut p_od, mut p_id, mut p_h) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    let lens_i = |v: &[u32], i: usize| v.get(i).copied().unwrap_or(0) as usize;

    // Decode one request's slice of every column, advancing the cursors. `None` if a
    // read overruns `data` (belt-and-suspenders past the upfront check, which already
    // covers every column's span — so this is unreachable today). It stops the loop
    // before slicing out of bounds, but requests decoded earlier in the frame are
    // already routed; making that abort atomic would mean buffering the whole frame,
    // which is exactly what the streaming decode avoids.
    let mut decode_one = |i: usize| -> Option<ChunkEvent> {
        let token_ids = take_i32(data, &mut c_ids, lens_i(&h.tok_lens, i))?;

        // Plain decode frame (no request in the batch asked for logprobs/hidden):
        // the extras columns are all zero-width, so skip reading them entirely.
        let mut extras = if !has_extras {
            None
        } else {
            let (out_lp_val, out_lp_idx) =
                take_flat(data, &mut c_olp_v, &mut c_olp_i, lens_i(&h.out_lp_lens, i))?;
            let (in_lp_val, in_lp_idx) =
                take_flat(data, &mut c_ilp_v, &mut c_ilp_i, lens_i(&h.in_lp_lens, i))?;
            let (out_top_val, out_top_idx, out_top_lens) = take_ragged(
                data,
                &mut c_ot_v,
                &mut c_ot_i,
                &h.out_top_poslens,
                &mut p_ot,
                lens_i(&h.out_top_reqlens, i),
            )?;
            let (in_top_val, in_top_idx, in_top_lens) = take_ragged(
                data,
                &mut c_it_v,
                &mut c_it_i,
                &h.in_top_poslens,
                &mut p_it,
                lens_i(&h.in_top_reqlens, i),
            )?;
            let (out_tid_val, out_tid_idx, out_tid_lens) = take_ragged(
                data,
                &mut c_od_v,
                &mut c_od_i,
                &h.out_tokids_lp_poslens,
                &mut p_od,
                lens_i(&h.out_tokids_lp_reqlens, i),
            )?;
            let (in_tid_val, in_tid_idx, in_tid_lens) = take_ragged(
                data,
                &mut c_id_v,
                &mut c_id_i,
                &h.in_tokids_lp_poslens,
                &mut p_id,
                lens_i(&h.in_tokids_lp_reqlens, i),
            )?;
            let (hidden_val, hidden_lens) = take_hidden(
                data,
                &mut c_h_v,
                &h.hidden_poslens,
                &mut p_h,
                lens_i(&h.hidden_reqlens, i),
            )?;

            // Even in an extras batch, most requests carry none — box only if this
            // one actually does, so its `ChunkEvent` stays the small common frame.
            let ex = ChunkExtras {
                out_lp_val,
                out_lp_idx,
                in_lp_val,
                in_lp_idx,
                out_top_val,
                out_top_idx,
                out_top_lens,
                in_top_val,
                in_top_idx,
                in_top_lens,
                out_tid_val,
                out_tid_idx,
                out_tid_lens,
                in_tid_val,
                in_tid_idx,
                in_tid_lens,
                hidden_val,
                hidden_lens,
                hidden_shape: None,
                // Explicit, NOT `..Default::default()` — same reason as `ChunkEvent`
                // below: a new column must fail to compile here until it is decoded.
                out_lp_txt: Vec::new(),
                in_lp_txt: Vec::new(),
                out_top_txt: Vec::new(),
                in_top_txt: Vec::new(),
                out_tid_txt: Vec::new(),
                in_tid_txt: Vec::new(),
                metadata: OutputMetadata::default(),
                prompt_context: Vec::new(),
                flat_input_top_logprobs: None,
                logprobs: None,
                routed_experts: None,
                indexer_topk: None,
                beam_output: None,
                prompt_token_ids: None,
            };
            (!ex.is_empty()).then(|| Box::new(ex))
        };
        if let Some(shape) = h.hidden_shapes.get_mut(i).and_then(Option::take) {
            let ex = extras.get_or_insert_with(Default::default);
            if shape.elements() != Some(ex.hidden_val.len()) {
                return None;
            }
            ex.hidden_shape = Some(shape);
        }
        if !h.customized_info.is_empty() {
            let metadata = &mut extras.get_or_insert_with(Default::default).metadata;
            for (key, values) in &mut h.customized_info {
                metadata
                    .customized_info
                    .insert(key.clone(), std::mem::take(&mut values[i]));
            }
        }
        if let Some(fields) = h
            .time_metadata
            .get_mut(i)
            .filter(|fields| !fields.is_empty())
        {
            extras
                .get_or_insert_with(Default::default)
                .metadata
                .fields
                .extend(std::mem::take(fields));
        }
        if let Some(context) = h.prompt_contexts.get_mut(i).filter(|ids| !ids.is_empty()) {
            extras.get_or_insert_with(Default::default).prompt_context = std::mem::take(context);
        }
        if let Some(shape) = h.sampling_mask_shapes.get(i).and_then(Option::as_ref) {
            let masks = shape
                .iter()
                .map(|length| match length {
                    Some(length) => take_i32(data, &mut c_mask_ids, *length as usize)
                        .map(|ids| serde_json::json!(ids)),
                    None => Some(serde_json::Value::Null),
                })
                .collect::<Option<Vec<_>>>()?;
            let logprobs = take_f32(data, &mut c_mask_logprobs, shape.len())?
                .into_iter()
                .map(|value| serde_json::json!(value))
                .collect();
            let metadata = &mut extras.get_or_insert_with(Default::default).metadata;
            metadata
                .customized_info
                .insert("output_token_sampling_mask".into(), masks);
            metadata
                .customized_info
                .insert("output_token_sampling_logprobs".into(), logprobs);
        }
        if let Some(shape) = h.flat_top_logprob_shapes.get(i).copied().flatten() {
            let bytes = shape.elements()?.checked_mul(4)?;
            let read = |offset: &mut usize| {
                let end = offset.checked_add(bytes)?;
                let value = Bytes::copy_from_slice(data.get(*offset..end)?);
                *offset = end;
                Some(value)
            };
            extras
                .get_or_insert_with(Default::default)
                .flat_input_top_logprobs = Some(FlatTopLogprobs {
                shape,
                values: read(&mut c_flat_top_values)?,
                indices: read(&mut c_flat_top_indices)?,
            });
        }
        for (lengths, offset, routed) in [
            (&h.routed_experts_bytes, &mut c_routed_experts, true),
            (&h.indexer_topk_bytes, &mut c_indexer_topk, false),
        ] {
            if let Some(length) = lengths.get(i).copied().flatten() {
                let end = offset.checked_add(length as usize)?;
                let bytes = Bytes::copy_from_slice(data.get(*offset..end)?);
                *offset = end;
                let extras = extras.get_or_insert_with(Default::default);
                if routed {
                    extras.routed_experts = Some(bytes);
                } else {
                    extras.indexer_topk = Some(bytes);
                }
            }
        }
        if let Some(headers) = h.beam_headers.get_mut(i).and_then(Option::take) {
            let sequences = headers
                .into_iter()
                .map(|header| {
                    Some(BeamSequence {
                        token_ids: take_i32(data, &mut c_beam_tokens, header.token_count as usize)?,
                        finish_reason: header.finish_reason,
                        sequence_score: header.sequence_score,
                        text: None,
                    })
                })
                .collect::<Option<Vec<_>>>()?;
            if !sequences.is_empty() {
                extras.get_or_insert_with(Default::default).beam_output = Some(BeamOutput {
                    sequences,
                    scheduler_metadata: OutputMetadata::default(),
                });
            }
        }

        Some(ChunkEvent {
            // Any string is a valid rid; hash to the routing key. An unknown
            // rid routes to a shard whose table has no entry → dropped there.
            rid: std::mem::take(&mut h.rids[i]).into(),
            token_ids,
            finish_reason: h.finish_reasons.get(i).cloned().flatten(),
            prompt_tokens: h.prompt_tokens.get(i).copied().unwrap_or(0),
            counts: Arc::new(
                h.counts
                    .as_mut()
                    .map(|counts| counts.take_request(i))
                    .unwrap_or_default(),
            ),
            extras,
            // Listed explicitly, NOT `..Default::default()`: a new column added to
            // `ChunkEvent` and wired into the response must fail to compile here
            // until it is actually decoded. With the struct-update syntax it
            // compiled clean and silently shipped zeros.
            text: String::new(),
            completion_tokens: 0,
        })
    };

    for i in 0..n {
        let Some(ev) = decode_one(i) else { reject!() };
        route(ev);
    }
    decoded.ok = true;
    decoded
}

/// Recover the rid column straight from the header bytes.
///
/// Read through `rmpv` rather than a serde tuple: `(Vec<String>,)` decodes ONLY a
/// 1-element array, while real headers carry 4 or 16 columns, so it returned
/// `Err(LengthMismatch(1))` every time and named nobody. Arity independence is the
/// whole point — the trigger for the typed-decode-failure path is Python appending
/// a column an older Rust build does not know.
///
/// Called only when a frame is rejected, which is why the accepted path no longer
/// clones the column up front. That clone was discarded unread on every good frame
/// and cost ~27% of the whole decode at batch 4096, plus a third of the crate's
/// steady-state allocations. It also has to be a re-read rather than a snapshot:
/// `decode_one` moves each rid out of the header as it goes, so by the time the
/// routing loop can fail, the decoded header no longer holds the earlier ones.
fn recover_rids(header: &[u8]) -> Vec<Rid> {
    rmpv::decode::read_value(&mut &header[..])
        .ok()
        .and_then(|v| match v {
            rmpv::Value::Array(cols) => cols.into_iter().next(),
            _ => None,
        })
        .and_then(|c| match c {
            rmpv::Value::Array(rids) => Some(
                rids.iter()
                    .filter_map(|r| r.as_str().map(Rid::from))
                    .collect(),
            ),
            _ => None,
        })
        .unwrap_or_default()
}

/// Outcome of [`for_each_chunk`]: whether the frame was accepted, plus the rids it
/// named. `rids` is populated as soon as the header parses, so a caller can fail
/// every request in a rejected frame — including ones whose chunk never decoded,
/// which would otherwise wait forever for a `Done` that no longer exists.
#[derive(Debug, Default)]
pub struct Decoded {
    pub ok: bool,
    pub rids: Vec<Rid>,
}

/// Frame a control result `[rid, payload]` for the response ring (tag prepended).
pub fn frame_control_result(rid: &str, payload: &[u8]) -> Bytes {
    frame_payload(DISPATCH_TAG_RESULT, rid, payload)
}

pub fn frame_control_result_part(rid: &str, dp_rank: u32, payload: &[u8]) -> Bytes {
    use rmpv::Value;
    let arr = Value::Array(vec![
        Value::from(rid),
        Value::from(dp_rank),
        Value::Binary(payload.to_vec()),
    ]);
    let mut buf = Vec::with_capacity(1 + payload.len() + rid.len() + 16);
    buf.push(DISPATCH_TAG_RESULT_PART);
    let _ = rmpv::encode::write_value(&mut buf, &arr);
    Bytes::from(buf)
}

pub fn frame_abort_result(rid: &str, payload: &[u8]) -> Bytes {
    frame_payload(DISPATCH_TAG_ABORT, rid, payload)
}

fn frame_payload(tag: u8, rid: &str, payload: &[u8]) -> Bytes {
    use rmpv::Value;
    let arr = Value::Array(vec![Value::from(rid), Value::Binary(payload.to_vec())]);
    let mut buf = Vec::with_capacity(1 + payload.len() + rid.len() + 8);
    buf.push(tag);
    let _ = rmpv::encode::write_value(&mut buf, &arr);
    Bytes::from(buf)
}

/// Frame a per-request failure `[rid, message]` for the response — routes a
/// terminal error back to the owning request (→ HTTP 400) instead of crashing.
pub fn frame_error(rid: &str, message: &str) -> Bytes {
    use rmpv::Value;
    let arr = Value::Array(vec![Value::from(rid), Value::from(message)]);
    let mut buf = Vec::with_capacity(1 + rid.len() + message.len() + 8);
    buf.push(DISPATCH_TAG_ERROR);
    let _ = rmpv::encode::write_value(&mut buf, &arr);
    Bytes::from(buf)
}

/// One scheduler output increment — the common, always-present frame. `token_ids`
/// / `prompt_tokens` / `finish_reason` arrive from Python (pre-decode); the detok
/// shard fills `text` in place. Deltas — fold with `OutputAccumulator` for
/// cumulative. Logprobs + hidden states (rare, and large) live behind the boxed
/// [`ChunkExtras`] (`None` unless requested) so this frame stays small even when
/// the decoder builds an inline array at up to batch 4096 per step.
///
/// Not a wire type — built by `for_each_chunk` from the columnar [`BatchHeader`]
/// frame and moved between stages in-process (never serialized), so no serde.
#[derive(Debug, Clone, Default)]
pub struct ChunkEvent {
    /// Client-visible rid — the request's IDENTITY. Moved out of the frame header
    /// (which owns it and drops it), so carrying it costs no allocation. The shard
    /// is still chosen by `Rid::shard`, but a hash collision there now only
    /// co-locates two requests instead of merging them.
    pub rid: Rid,
    /// New token ids for this step. Empty allowed (e.g. metadata-only frames).
    pub token_ids: TokenIds,
    /// `None` while streaming, the [`FinishReason`] on the final chunk.
    pub finish_reason: Option<FinishReason>,
    /// Prompt token count for this request (constant across its chunks).
    pub prompt_tokens: u32,
    /// Decoded text **delta** for this chunk (empty in skip mode / on partial UTF-8),
    /// filled by the detok shard. `token_ids` doubles as `output_ids`;
    /// `completion_tokens` is this chunk's count.
    pub text: String,
    pub completion_tokens: u64,
    /// Immutable scheduler accounting, shared with cumulative response state so
    /// each stream frame stays small and avoids cloning cache-source strings.
    pub counts: Arc<TokenCounts>,
    /// Logprob + hidden-state columns — `None` unless the request asked for them.
    /// Boxed to keep the common token/text/finish frame small at large decode
    /// batches (the decoder allocates it only when a column is non-empty).
    pub extras: Option<Box<ChunkExtras>>,
}

/// Logprob + hidden-state columns for a [`ChunkEvent`], allocated only when the
/// request enabled logprobs / hidden states. Columnar `val`/`idx` (+ ragged `lens`)
/// buffers arrive pre-decode; the detok shard fills the parallel `*_txt` columns
/// when `return_text_in_logprobs` is set. In-process only — no serde (see
/// [`ChunkEvent`]).
#[derive(Debug, Clone, Default)]
pub struct ChunkExtras {
    /// Final preprocessed prompt, shared across the request's output chunks.
    pub prompt_token_ids: Option<Arc<[i32]>>,
    /// Frontend request options also describe requested-but-empty columns.
    pub logprobs: Option<LogprobOptions>,
    /// Output-token logprobs (parallel `val`/`idx`, one entry per new output token).
    pub out_lp_val: Vec<f32>,
    pub out_lp_idx: Vec<i32>,
    /// Input (prefill) token logprobs, sent once on the first chunk.
    pub in_lp_val: Vec<f32>,
    pub in_lp_idx: Vec<i32>,
    /// Top-k logprobs (2-level ragged): flat `val`/`idx` + per-position `lens` (0 =
    /// null). Output = per-step delta, input = once on the first chunk.
    pub out_top_val: Vec<f32>,
    pub out_top_idx: Vec<i32>,
    pub out_top_lens: Vec<u32>,
    pub in_top_val: Vec<f32>,
    pub in_top_idx: Vec<i32>,
    pub in_top_lens: Vec<u32>,
    pub flat_input_top_logprobs: Option<FlatTopLogprobs>,
    /// CPU tensor bytes; encoded to base64 on the detokenizer shard.
    pub routed_experts: Option<Bytes>,
    pub indexer_topk: Option<Bytes>,
    pub beam_output: Option<BeamOutput>,
    /// Token-ids logprobs (same ragged layout); set only when `token_ids_logprob` was.
    pub out_tid_val: Vec<f32>,
    pub out_tid_idx: Vec<i32>,
    pub out_tid_lens: Vec<u32>,
    pub in_tid_val: Vec<f32>,
    pub in_tid_idx: Vec<i32>,
    pub in_tid_lens: Vec<u32>,
    /// Hidden states (dense f32): flat buffer + per-row lengths. Last-writer-wins
    /// across chunks (the final message has the full set).
    pub hidden_val: Vec<f32>,
    pub hidden_lens: Vec<u32>,
    pub hidden_shape: Option<HiddenStateShape>,
    /// Decoded logprob token text (`return_text_in_logprobs`), parallel to the
    /// `*_idx` buffers; empty when not requested (the tuple's text slot stays null).
    pub out_lp_txt: Vec<String>,
    pub in_lp_txt: Vec<String>,
    pub out_top_txt: Vec<String>,
    pub in_top_txt: Vec<String>,
    pub out_tid_txt: Vec<String>,
    pub in_tid_txt: Vec<String>,
    pub metadata: OutputMetadata,
    /// Consumed once by the native decoder; never exposed as response metadata.
    pub prompt_context: Vec<i32>,
}

/// Token-aligned scheduler output and model-owned response annotations. A
/// processor may consume a raw column and replace it with reduced metadata.
#[derive(Debug, Clone, Default)]
pub struct OutputMetadata {
    pub customized_info: BTreeMap<String, Vec<serde_json::Value>>,
    pub fields: serde_json::Map<String, serde_json::Value>,
}

impl ChunkExtras {
    /// True when no logprob / hidden column carries data — lets the decoder skip the
    /// box allocation for the common (extras-free) frame.
    fn is_empty(&self) -> bool {
        self.prompt_token_ids.is_none()
            && self.logprobs.is_none()
            && self.out_lp_val.is_empty()
            && self.in_lp_val.is_empty()
            && self.out_top_lens.is_empty()
            && self.in_top_lens.is_empty()
            && self.flat_input_top_logprobs.is_none()
            && self.routed_experts.is_none()
            && self.indexer_topk.is_none()
            && self.beam_output.is_none()
            && self.out_tid_lens.is_empty()
            && self.in_tid_lens.is_empty()
            && self.hidden_lens.is_empty()
            && self.hidden_shape.is_none()
            && self.metadata.customized_info.is_empty()
            && self.metadata.fields.is_empty()
            && self.prompt_context.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::finish_reason::{FinishKind, Matched};

    #[test]
    fn beam_columns_reject_partial_batches_and_misaligned_tokens() {
        use base64::Engine;
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../testdata/beam_outputs_python.json")).unwrap();
        let original = rmp_serde::to_vec(&fixture["header"]).unwrap();
        let data = base64::engine::general_purpose::STANDARD
            .decode(fixture["data_b64"].as_str().unwrap())
            .unwrap();
        for failure in 0..5 {
            let mut header: BatchHeader = rmp_serde::from_slice(&original).unwrap();
            let mut data = data.clone();
            match failure {
                0 => {
                    header.beam_headers.pop();
                }
                1 => header.beam_headers.push(None),
                2 => header.beam_headers[1].as_mut().unwrap()[0].token_count = u32::MAX,
                3 => {
                    data.pop();
                }
                _ => data.push(0),
            }
            let frame = frame_decode_batch_cols(&rmp_serde::to_vec(&header).unwrap(), &[&data]);
            let mut delivered = 0;
            let result = for_each_chunk(&frame[1..], |_| delivered += 1);
            assert!(!result.ok, "case {failure}");
            assert_eq!(
                delivered, 0,
                "a malformed beam must reject the complete batch"
            );
            assert_eq!(
                result
                    .rids
                    .iter()
                    .map(|rid| rid.as_str())
                    .collect::<Vec<_>>(),
                header.rids.iter().map(String::as_str).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn customized_output_preserves_sparse_rows_and_rejects_partial_batches() {
        let mut header = BatchHeader {
            rids: vec!["sparse".into(), "empty".into()],
            finish_reasons: vec![None, None],
            prompt_tokens: vec![3, 4],
            tok_lens: vec![2, 0],
            prompt_contexts: vec![vec![-101, 3], vec![]],
            customized_info: [(
                "probe".into(),
                vec![
                    vec![serde_json::Value::Null, serde_json::json!(0.25)],
                    vec![],
                ],
            )]
            .into(),
            ..Default::default()
        };
        let data: Vec<u8> = [11i32, 12].into_iter().flat_map(i32::to_le_bytes).collect();
        let frame = frame_decode_batch_cols(&rmp_serde::to_vec(&header).unwrap(), &[&data]);
        let mut events = Vec::new();
        assert!(for_each_chunk(&frame[1..], |event| events.push(event)).ok);
        assert_eq!(events[0].rid, Rid::from("sparse"));
        assert_eq!(events[0].token_ids, vec![11, 12]);
        assert_eq!(
            events[0].extras.as_ref().unwrap().prompt_context,
            vec![-101, 3]
        );
        assert_eq!(
            events[0].extras.as_ref().unwrap().metadata.customized_info["probe"],
            header.customized_info["probe"][0]
        );
        assert_eq!(events[1].rid, Rid::from("empty"));
        assert!(events[1].extras.as_ref().unwrap().prompt_context.is_empty());
        assert!(events[1].extras.as_ref().unwrap().metadata.customized_info["probe"].is_empty());
        for invalid_contexts in [
            vec![vec![]],
            vec![vec![3; PROMPT_CONTEXT_TOKENS + 1], vec![]],
        ] {
            header.prompt_contexts = invalid_contexts;
            let frame = frame_decode_batch_cols(&rmp_serde::to_vec(&header).unwrap(), &[&data]);
            let invalid =
                for_each_chunk(&frame[1..], |_| panic!("invalid context batch was routed"));
            assert!(!invalid.ok);
            assert_eq!(invalid.rids, vec![Rid::from("sparse"), Rid::from("empty")]);
        }
        header.prompt_contexts.clear();
        header.customized_info.get_mut("probe").unwrap().pop();
        let frame = frame_decode_batch_cols(&rmp_serde::to_vec(&header).unwrap(), &[&data]);
        let invalid = for_each_chunk(&frame[1..], |_| panic!("partial batch was routed"));
        assert!(!invalid.ok);
        assert_eq!(invalid.rids, vec![Rid::from("sparse"), Rid::from("empty")]);
    }

    #[test]
    fn scheduler_token_counts_keep_request_identity_and_reject_partial_columns() {
        use serde_json::json;

        // Literal Python header layout: four base columns, twelve optional
        // numeric shapes, then one named map of scheduler accounting columns.
        let mut columns = vec![
            json!(["cached", "cold"]),
            json!([null, null]),
            json!([512, 64]),
            json!([1, 1]),
        ];
        columns.extend(std::iter::repeat_n(json!([]), 12));
        columns.push(json!({
            "reasoning_tokens": [3, 0],
            "retraction_counts": [2, 0],
            "cached_tokens": [192, 0],
            "cached_tokens_details": [{"device": 128, "host": 32, "storage": 32,
                                       "storage_backend": "FileSystem"}, null],
            "dp_ranks": [2, null],
            "image_tokens": [8, 0],
            "audio_tokens": [0, 0],
            "video_tokens": []
        }));
        let data: Vec<u8> = [11i32, 12].into_iter().flat_map(i32::to_le_bytes).collect();
        let frame = frame_decode_batch_cols(&rmp_serde::to_vec(&columns).unwrap(), &[&data]);
        let mut events = Vec::new();
        assert!(for_each_chunk(&frame[1..], |event| events.push(event)).ok);
        assert_eq!(events[0].rid, Rid::from("cached"));
        assert_eq!(events[0].counts.cached_tokens, 192);
        assert_eq!(events[0].counts.reasoning_tokens, 3);
        assert_eq!(events[0].counts.num_retractions, 2);
        let details = events[0]
            .counts
            .cached_tokens_details
            .as_ref()
            .unwrap()
            .as_ref()
            .unwrap();
        assert_eq!(
            (details.device, details.host, details.storage),
            (128, 32, Some(32))
        );
        assert_eq!(details.storage_backend.as_deref(), Some("FileSystem"));
        assert_eq!(events[0].counts.dp_rank, Some(Some(2)));
        assert_eq!(events[0].counts.image_tokens, Some(8));
        assert_eq!(events[1].rid, Rid::from("cold"));
        assert_eq!(events[1].counts.cached_tokens, 0);
        let cold = serde_json::to_value(events[1].counts.as_ref()).unwrap();
        assert_eq!(
            cold,
            json!({"reasoning_tokens": 0, "cached_tokens": 0, "num_retractions": 0,
                               "cached_tokens_details": null, "dp_rank": null})
        );

        for (field, bad) in [
            ("cached_tokens", json!([192])),
            ("reasoning_tokens", json!([])),
            ("retraction_counts", json!([2])),
            ("cached_tokens_details", json!([null])),
            ("dp_ranks", json!([0, 1, 2])),
            ("image_tokens", json!([8])),
            ("cached_tokens", json!([-1, 0])),
        ] {
            let mut invalid = columns.clone();
            invalid[16][field] = bad;
            let frame = frame_decode_batch_cols(&rmp_serde::to_vec(&invalid).unwrap(), &[&data]);
            let decoded = for_each_chunk(&frame[1..], |_| panic!("partial {field} was routed"));
            assert!(!decoded.ok, "{field}");
            assert_eq!(decoded.rids, vec![Rid::from("cached"), Rid::from("cold")]);
        }
    }
    #[test]
    fn batch_cols_match_single_joined_buffer() {
        let header = [1u8, 2, 3];
        let a = [10u8, 11];
        let b = [12u8, 13, 14];
        let multi = frame_decode_batch_cols(&header, &[&a[..], &b[..]]);
        let joined: Vec<u8> = a.iter().chain(&b).copied().collect();
        let single = frame_decode_batch_cols(&header, &[joined.as_slice()]);
        assert_eq!(multi, single);
        assert_eq!(multi[0], DISPATCH_TAG_BATCH);
        assert_eq!(
            u32::from_le_bytes([multi[1], multi[2], multi[3], multi[4]]),
            3
        );
        assert_eq!(&multi[5..8], &header); // header
        assert_eq!(&multi[8..], &[10, 11, 12, 13, 14]); // columns end-to-end
    }

    /// A batch frame (the fast path) decodes into per-request ChunkEvents, with
    /// token ids sliced from the single concatenated buffer by `tok_lens`. The
    /// header is a msgspec-style positional array (what Python emits).
    #[test]
    fn decodes_batch_frame() {
        use rmpv::Value;
        // 3 requests: rids "1","2","3"; finish [nil, {type:stop,matched:5}, nil];
        // prompt_tokens [4,5,6]; tok_lens [2,0,1] -> ids [10,11 | (none) | 12].
        let stop = Value::Map(vec![
            (Value::from("type"), Value::from("stop")),
            (Value::from("matched"), Value::from(5)),
        ]);
        let header_arr = Value::Array(vec![
            Value::Array(vec![Value::from("1"), Value::from("2"), Value::from("3")]),
            Value::Array(vec![Value::Nil, stop, Value::Nil]),
            Value::Array(vec![
                Value::from(4u32),
                Value::from(5u32),
                Value::from(6u32),
            ]),
            Value::Array(vec![
                Value::from(2u32),
                Value::from(0u32),
                Value::from(1u32),
            ]),
        ]);
        let mut header = Vec::new();
        rmpv::encode::write_value(&mut header, &header_arr).unwrap();
        let data: Vec<u8> = [10i32, 11, 12]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();

        let framed = frame_decode_batch_cols(&header, &[&data]);
        assert_eq!(framed[0], DISPATCH_TAG_BATCH);
        let mut events = Vec::new();
        assert!(for_each_chunk(&framed[1..], |ev| events.push(ev)).ok);
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].rid, Rid::from("1"));
        assert_eq!(events[0].token_ids, vec![10, 11]);
        assert_eq!(events[0].prompt_tokens, 4);
        assert!(events[0].finish_reason.is_none());
        assert_eq!(events[1].rid, Rid::from("2"));
        assert!(events[1].token_ids.is_empty());
        // The whole reason survives msgpack (type + matched), not just the type.
        assert_eq!(
            events[1].finish_reason,
            Some(
                FinishKind::Stop {
                    matched: Some(Matched::Token(5))
                }
                .into()
            )
        );
        assert_eq!(events[2].rid, Rid::from("3"));
        assert_eq!(events[2].token_ids, vec![12]);
        assert_eq!(events[2].prompt_tokens, 6);
        // A plain decode frame carries no extras columns at all, so the per-frame
        // `has_extras` guard must skip the extras machinery entirely for every
        // request (this is the from-scheduler hot path — see `for_each_chunk`).
        assert!(events.iter().all(|e| e.extras.is_none()));
    }

    /// A header whose column lengths exceed the data buffer (a Python/Rust
    /// positional-ABI drift, or a truncated frame) is rejected: `for_each_chunk`
    /// returns false and routes nothing — it must NOT panic the sole from_scheduler thread
    /// on an out-of-bounds slice. Built the way Python emits (positional msgpack
    /// header + concatenated data columns).
    #[test]
    fn rejects_frame_with_lengths_past_data() {
        use rmpv::Value;
        // 1 request: tok_lens[0]=10 claims 40 bytes and out_lp_lens[0]=1 puts the
        // logprob column's base past the 4-byte data buffer. The old clamp-only-`end`
        // code advanced the cursor past `len`, then sliced `data[40..4]` (start > end)
        // and panicked.
        let header_arr = Value::Array(vec![
            Value::Array(vec![Value::from("1")]),   // rids
            Value::Array(vec![Value::Nil]),         // finish_reasons
            Value::Array(vec![Value::from(0u32)]),  // prompt_tokens
            Value::Array(vec![Value::from(10u32)]), // tok_lens (claims 40 bytes)
            Value::Array(vec![Value::from(1u32)]),  // out_lp_lens (base now past data)
        ]);
        let mut header = Vec::new();
        rmpv::encode::write_value(&mut header, &header_arr).unwrap();
        let data: Vec<u8> = [0i32].iter().flat_map(|x| x.to_le_bytes()).collect(); // 4 bytes

        let framed = frame_decode_batch_cols(&header, &[&data]);
        let mut routed = 0usize;
        let decoded = for_each_chunk(&framed[1..], |_| routed += 1);
        assert!(!decoded.ok, "malformed frame must be rejected, not decoded");
        assert_eq!(routed, 0, "no request may be routed from a rejected frame");
        // The rid is still reported, so the caller can fail the request that was
        // waiting on this frame instead of letting it hang.
        assert_eq!(decoded.rids, vec![Rid::from("1")]);
    }

    /// A header whose `reqlens` claim more positions than `poslens` carries is
    /// rejected, not truncated. This drift passes the upfront `base > data.len()`
    /// check — the data columns are exactly as long as `poslens` says — so only the
    /// per-column bound catches it. Clamping (the old behavior) handed req0 a short
    /// `lens`, which also under-advanced the val/idx cursors, so req1 read from the
    /// wrong offset: a frame that decodes "successfully" into wrong logprobs.
    #[test]
    fn rejects_poslens_shorter_than_reqlens_claims() {
        use rmpv::Value;
        let f = |xs: &[f32]| -> Vec<u8> { xs.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let i = |xs: &[i32]| -> Vec<u8> { xs.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let arr_u = |xs: &[u32]| Value::Array(xs.iter().map(|&x| Value::from(x)).collect());
        let rids = Value::Array(vec![Value::from("1"), Value::from("2")]);
        let finish = Value::Array(vec![Value::Nil, Value::Nil]);

        // Ragged column: reqs claim 2 + 1 = 3 positions, `out_top_poslens` has 2.
        let header_arr = Value::Array(vec![
            rids.clone(),
            finish.clone(),
            arr_u(&[0, 0]), // prompt
            arr_u(&[1, 1]), // tok_lens
            arr_u(&[0, 0]), // out_lp_lens
            arr_u(&[0, 0]), // in_lp_lens
            arr_u(&[2, 1]), // out_top_reqlens — 3 positions claimed
            arr_u(&[2, 2]), // out_top_poslens — only 2 supplied
        ]);
        let mut header = Vec::new();
        rmpv::encode::write_value(&mut header, &header_arr).unwrap();
        let mut data = Vec::new();
        data.extend(i(&[10, 20])); // token_ids
        data.extend(f(&[-0.1, -0.2, -0.3, -0.4])); // out_top_val (sum of poslens = 4)
        data.extend(i(&[1, 2, 3, 4])); // out_top_idx
        let framed = frame_decode_batch_cols(&header, &[&data]);
        let mut routed = 0usize;
        assert!(
            !for_each_chunk(&framed[1..], |_| routed += 1).ok,
            "ragged poslens drift must be rejected, not truncated"
        );

        // Same drift in the hidden column, which has no idx pair: 2 rows claimed, 1 supplied.
        let header_arr = Value::Array(vec![
            rids,
            finish,
            arr_u(&[0, 0]), // prompt
            arr_u(&[1, 1]), // tok_lens
            arr_u(&[0, 0]), // out_lp_lens
            arr_u(&[0, 0]), // in_lp_lens
            arr_u(&[0, 0]), // out_top_reqlens
            arr_u(&[]),     // out_top_poslens
            arr_u(&[0, 0]), // in_top_reqlens
            arr_u(&[]),     // in_top_poslens
            arr_u(&[0, 0]), // out_tokids_lp_reqlens
            arr_u(&[]),     // out_tokids_lp_poslens
            arr_u(&[0, 0]), // in_tokids_lp_reqlens
            arr_u(&[]),     // in_tokids_lp_poslens
            arr_u(&[2, 0]), // hidden_reqlens — 2 rows claimed
            arr_u(&[3]),    // hidden_poslens — only 1 supplied
        ]);
        let mut header = Vec::new();
        rmpv::encode::write_value(&mut header, &header_arr).unwrap();
        let mut data = Vec::new();
        data.extend(i(&[10, 20])); // token_ids
        data.extend(f(&[0.1, 0.2, 0.3])); // hidden_val (sum of poslens = 3)
        let framed = frame_decode_batch_cols(&header, &[&data]);
        let mut routed = 0usize;
        assert!(
            !for_each_chunk(&framed[1..], |_| routed += 1).ok,
            "hidden poslens drift must be rejected, not truncated"
        );
    }

    /// A frame that fails at request 0 buckets nothing, so bucket-driven cleanup
    /// would leave every request in it hanging. The rids come from the header,
    /// which is fully parsed before the decode loop.
    #[test]
    fn rejected_frame_still_reports_all_its_rids() {
        use rmpv::Value;
        let header_arr = Value::Array(vec![
            Value::Array(vec![Value::from("r0"), Value::from("r1")]), // rids
            Value::Array(vec![Value::Nil, Value::Nil]),
            Value::Array(vec![Value::from(0u32), Value::from(0u32)]),
            Value::Array(vec![Value::from(9u32), Value::from(9u32)]), // tok_lens past data
        ]);
        let mut header = Vec::new();
        rmpv::encode::write_value(&mut header, &header_arr).unwrap();
        let framed = frame_decode_batch_cols(&header, &[&[0u8; 4][..]]);
        let mut routed = 0usize;
        let decoded = for_each_chunk(&framed[1..], |_| routed += 1);
        assert!(!decoded.ok);
        assert_eq!(routed, 0, "fails at request 0 — nothing bucketed");
        assert_eq!(
            decoded.rids,
            vec![Rid::from("r0"), Rid::from("r1")],
            "both requests must be nameable so the caller can fail them"
        );
    }

    /// A data buffer LONGER than the header's columns means the two disagree — a
    /// producer-side val/idx mismatch. Accepting the surplus delivered another
    /// column's bytes as logprobs with a 200.
    #[test]
    fn frame_longer_than_its_columns_is_rejected() {
        use rmpv::Value;
        let header_arr = Value::Array(vec![
            Value::Array(vec![Value::from("1")]),
            Value::Array(vec![Value::Nil]),
            Value::Array(vec![Value::from(0u32)]),
            Value::Array(vec![Value::from(1u32)]), // tok_lens: 1 id = 4 bytes
        ]);
        let mut header = Vec::new();
        rmpv::encode::write_value(&mut header, &header_arr).unwrap();
        let data: Vec<u8> = vec![0u8; 8]; // 4 bytes too many
        let framed = frame_decode_batch_cols(&header, &[&data]);
        let decoded = for_each_chunk(&framed[1..], |_| {});
        assert!(!decoded.ok, "header and data must agree exactly");
    }

    /// Request/response rid agreement: the rid decoded from the frame must be the
    /// one Python sent, AND both sides must derive the same shard from it. The
    /// partition key is memoized inside `Rid`, so a per-conversion hasher seed
    /// would send a request's chunks to a shard that never registered it.
    #[test]
    fn uuid_rid_decodes_to_the_same_rid_and_shard() {
        use rmpv::Value;
        let rid = Rid::from("9f86d081884c7d659a2feaa0c55ad015");
        let header_arr = Value::Array(vec![
            Value::Array(vec![Value::from(rid.to_string())]), // rids
            Value::Array(vec![Value::Nil]),                   // finish_reasons
            Value::Array(vec![Value::from(1u32)]),            // prompt_tokens
            Value::Array(vec![Value::from(1u32)]),            // tok_lens
        ]);
        let mut header = Vec::new();
        rmpv::encode::write_value(&mut header, &header_arr).unwrap();
        let data: Vec<u8> = [0i32].iter().flat_map(|x| x.to_le_bytes()).collect();

        let framed = frame_decode_batch_cols(&header, &[&data]);
        let mut events = Vec::new();
        assert!(for_each_chunk(&framed[1..], |ev| events.push(ev)).ok);
        assert_eq!(events.len(), 1);
        // The equality below reuses `from_rid`, so on its own it would hold for
        // any decoder that hashed *something*. These pin the two failure modes it
        // is meant to catch: a decoder that parsed the rid as an integer (a uuid
        // is not numeric, so it would fall back to 0), and one that keyed off the
        // request's position in the batch rather than its rid.
        assert_eq!(
            events[0].rid, rid,
            "the rid IS the identity — carried, not hashed"
        );
        assert_ne!(events[0].rid, Rid::from("0"));
        // Same string, independently constructed → same shard, on any shard count.
        for shards in [1usize, 3, 8, 64] {
            assert_eq!(
                events[0].rid.shard(shards),
                Rid::from("9f86d081884c7d659a2feaa0c55ad015").shard(shards),
                "the partition key must not depend on WHERE the Rid was built"
            );
        }
        assert_ne!(
            events[0].rid,
            Rid::from("another-rid"),
            "distinct rids stay distinct — identity is the string, not a digest"
        );
    }

    /// A batch frame carrying the numeric columns (extras path): 2 requests,
    /// req0 with output logprobs + top-k + hidden, req1 empty. Verifies the
    /// column-major data split by the header's reqlens/poslens.
    #[test]
    fn decodes_batch_frame_with_extras() {
        use rmpv::Value;
        let f = |xs: &[f32]| -> Vec<u8> { xs.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let i = |xs: &[i32]| -> Vec<u8> { xs.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let arr_u = |xs: &[u32]| Value::Array(xs.iter().map(|&x| Value::from(x)).collect());
        // header: rids, finish, prompt, tok_lens, out_lp_lens, in_lp_lens,
        //   out_top_reqlens, out_top_poslens, in_top_*, out_tokids_lp_*,
        //   in_tokids_lp_*,
        //   hidden_reqlens, hidden_poslens
        let header_arr = Value::Array(vec![
            Value::Array(vec![Value::from("1"), Value::from("2")]), // rids
            Value::Array(vec![Value::Nil, Value::Nil]),             // finish
            arr_u(&[3, 4]),                                         // prompt
            arr_u(&[1, 1]),                                         // tok_lens
            arr_u(&[2, 0]),                                         // out_lp_lens
            arr_u(&[0, 0]),                                         // in_lp_lens
            arr_u(&[1, 0]),                                         // out_top_reqlens (req0: 1 pos)
            arr_u(&[2]),    // out_top_poslens (that pos: k=2)
            arr_u(&[0, 0]), // in_top_reqlens
            arr_u(&[]),     // in_top_poslens
            arr_u(&[0, 0]), // out_tokids_lp_reqlens
            arr_u(&[]),     // out_tokids_lp_poslens
            arr_u(&[0, 0]), // in_tokids_lp_reqlens
            arr_u(&[]),     // in_tokids_lp_poslens
            arr_u(&[1, 0]), // hidden_reqlens (req0: 1 row)
            arr_u(&[3]),    // hidden_poslens (dim 3)
        ]);
        let mut header = Vec::new();
        rmpv::encode::write_value(&mut header, &header_arr).unwrap();
        let mut data = Vec::new();
        data.extend(i(&[10, 20])); // token_ids: req0=[10], req1=[20]
        data.extend(f(&[-0.5, -0.6])); // out_lp_val (req0, 2)
        data.extend(i(&[10, 99])); // out_lp_idx
        data.extend(f(&[-0.1, -0.2])); // out_top_val (1 pos, k=2)
        data.extend(i(&[10, 11])); // out_top_idx
        data.extend(f(&[0.1, 0.2, 0.3])); // hidden_val (1 row, dim 3)

        let framed = frame_decode_batch_cols(&header, &[&data]);
        let mut events = Vec::new();
        assert!(for_each_chunk(&framed[1..], |ev| events.push(ev)).ok);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].token_ids, vec![10]);
        let ex0 = events[0]
            .extras
            .as_deref()
            .expect("req0 has logprob/hidden extras");
        assert_eq!(ex0.out_lp_val, vec![-0.5, -0.6]);
        assert_eq!(ex0.out_lp_idx, vec![10, 99]);
        assert_eq!(ex0.out_top_val, vec![-0.1, -0.2]);
        assert_eq!(ex0.out_top_lens, vec![2]);
        assert_eq!(ex0.hidden_val, vec![0.1, 0.2, 0.3]);
        assert_eq!(ex0.hidden_lens, vec![3]);
        // req1 has a token id but no numeric columns → no extras box allocated.
        assert_eq!(events[1].token_ids, vec![20]);
        assert!(events[1].extras.is_none());
    }

    /// An inactive extras family may arrive as EMPTY columns or as `batch_size`
    /// zeros, and both must decode to exactly the same events.
    ///
    /// This is the contract the Python producer's per-family gating relies on. It
    /// used to run all seven families whenever any one of them was active, so an
    /// inactive family shipped 4096 zeros per column; it now skips `accept`, which
    /// leaves those columns empty. `per_req_ok` admits either and `lens_i` reads 0
    /// past the end — but nothing pinned that, so a later tightening of the column
    /// validation (say, requiring every column to be `batch_size` long) would
    /// silently start rejecting whole frames, hanging every request in them.
    #[test]
    fn empty_and_zero_filled_inactive_families_decode_alike() {
        use rmpv::Value;
        let f = |xs: &[f32]| -> Vec<u8> { xs.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let i = |xs: &[i32]| -> Vec<u8> { xs.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let arr_u = |xs: &[u32]| Value::Array(xs.iter().map(|&x| Value::from(x)).collect());

        // Only `out_lp` is active; the other six families are inactive. `zeros`
        // picks how they are spelled: the old producer's per-request zeros, or the
        // gated producer's empty column.
        let build = |zeros: bool| {
            // Inactive: the old producer ran `accept` anyway and appended a 0 per
            // request; the gated producer skips it and leaves the column empty.
            let reqlens = if zeros { arr_u(&[0, 0]) } else { arr_u(&[]) };
            let header_arr = Value::Array(vec![
                Value::Array(vec![Value::from("1"), Value::from("2")]), // rids
                Value::Array(vec![Value::Nil, Value::Nil]),             // finish
                arr_u(&[3, 4]),                                         // prompt
                arr_u(&[1, 1]),                                         // tok_lens
                arr_u(&[2, 0]),                                         // out_lp_lens (ACTIVE)
                reqlens.clone(),                                        // in_lp_lens
                reqlens.clone(),                                        // out_top_reqlens
                arr_u(&[]),                                             // out_top_poslens
                reqlens.clone(),                                        // in_top_reqlens
                arr_u(&[]),                                             // in_top_poslens
                reqlens.clone(),                                        // out_tokids_lp_reqlens
                arr_u(&[]),                                             // out_tokids_lp_poslens
                reqlens.clone(),                                        // in_tokids_lp_reqlens
                arr_u(&[]),                                             // in_tokids_lp_poslens
                reqlens,                                                // hidden_reqlens
                arr_u(&[]),                                             // hidden_poslens
            ]);
            let mut header = Vec::new();
            rmpv::encode::write_value(&mut header, &header_arr).unwrap();
            let mut data = Vec::new();
            data.extend(i(&[10, 20])); // token_ids
            data.extend(f(&[-0.5, -0.6])); // out_lp_val (req0)
            data.extend(i(&[10, 99])); // out_lp_idx
            let framed = frame_decode_batch_cols(&header, &[&data]);
            let mut events = Vec::new();
            assert!(
                for_each_chunk(&framed[1..], |ev| events.push(ev)).ok,
                "frame rejected (zeros={zeros})"
            );
            events
        };

        let old = build(true);
        let new = build(false);
        assert_eq!(old.len(), 2);
        assert_eq!(old.len(), new.len());
        for (o, n) in old.iter().zip(&new) {
            assert_eq!(o.rid, n.rid);
            assert_eq!(o.token_ids, n.token_ids);
            match (o.extras.as_deref(), n.extras.as_deref()) {
                (Some(a), Some(b)) => {
                    assert_eq!(a.out_lp_val, b.out_lp_val);
                    assert_eq!(a.out_lp_idx, b.out_lp_idx);
                    // Every inactive family stays empty in BOTH spellings.
                    assert!(a.in_lp_val.is_empty() && b.in_lp_val.is_empty());
                    assert!(a.out_top_lens.is_empty() && b.out_top_lens.is_empty());
                    assert!(a.in_top_lens.is_empty() && b.in_top_lens.is_empty());
                    assert!(a.out_tid_lens.is_empty() && b.out_tid_lens.is_empty());
                    assert!(a.in_tid_lens.is_empty() && b.in_tid_lens.is_empty());
                    assert!(a.hidden_lens.is_empty() && b.hidden_lens.is_empty());
                }
                (None, None) => {}
                _ => panic!("extras presence differs between the two spellings"),
            }
        }
        // req0 really did carry its active family through both spellings.
        assert_eq!(
            new[0]
                .extras
                .as_deref()
                .expect("req0 has out_lp")
                .out_lp_val,
            vec![-0.5, -0.6]
        );
    }

    /// All SEVEN extras families in one frame, each with a distinct length AND
    /// distinct values. The existing extras test exercises only `out_lp` /
    /// `out_top` / `hidden`, so transposing a header pair — `in_top_*` with
    /// `out_tokids_lp_*`, say, leaves every assertion passing while the client
    /// receives another request's logprobs under the wrong key. Lengths differ per
    /// family (2/1/2/1/2/1/3 elements) so a swap misaligns the cursors too.
    #[test]
    fn decodes_all_extras_families_without_transposition() {
        use rmpv::Value;
        let f = |xs: &[f32]| -> Vec<u8> { xs.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let i = |xs: &[i32]| -> Vec<u8> { xs.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let arr_u = |xs: &[u32]| Value::Array(xs.iter().map(|&x| Value::from(x)).collect());

        let header_arr = Value::Array(vec![
            Value::Array(vec![Value::from("1")]), // rids
            Value::Array(vec![Value::Nil]),       // finish
            arr_u(&[9]),                          // prompt
            arr_u(&[1]),                          // tok_lens
            arr_u(&[2]),                          // out_lp_lens      (2 flat)
            arr_u(&[1]),                          // in_lp_lens       (1 flat)
            arr_u(&[1]),                          // out_top_reqlens  (1 position…
            arr_u(&[2]),                          // out_top_poslens  …k=2)
            arr_u(&[1]),                          // in_top_reqlens   (1 position…
            arr_u(&[1]),                          // in_top_poslens   …k=1)
            arr_u(&[1]),                          // out_tokids_lp_reqlens (1 position...
            arr_u(&[2]),                          // out_tokids_lp_poslens ...2 ids)
            arr_u(&[1]),                          // in_tokids_lp_reqlens  (1 position...
            arr_u(&[1]),                          // in_tokids_lp_poslens  ...1 id)
            arr_u(&[1]),                          // hidden_reqlens   (1 row…
            arr_u(&[3]),                          // hidden_poslens   …dim 3)
        ]);
        let mut header = Vec::new();
        rmpv::encode::write_value(&mut header, &header_arr).unwrap();

        // Concatenated in `for_each_chunk`'s column order.
        let mut data = Vec::new();
        data.extend(i(&[100])); // token_ids
        data.extend(f(&[-1.1, -1.2])); // out_lp_val
        data.extend(i(&[11, 12])); // out_lp_idx
        data.extend(f(&[-2.1])); // in_lp_val
        data.extend(i(&[21])); // in_lp_idx
        data.extend(f(&[-3.1, -3.2])); // out_top_val
        data.extend(i(&[31, 32])); // out_top_idx
        data.extend(f(&[-4.1])); // in_top_val
        data.extend(i(&[41])); // in_top_idx
        data.extend(f(&[-5.1, -5.2])); // out_tid_val
        data.extend(i(&[51, 52])); // out_tid_idx
        data.extend(f(&[-6.1])); // in_tid_val
        data.extend(i(&[61])); // in_tid_idx
        data.extend(f(&[7.1, 7.2, 7.3])); // hidden_val

        let framed = frame_decode_batch_cols(&header, &[&data]);
        let mut events = Vec::new();
        assert!(for_each_chunk(&framed[1..], |ev| events.push(ev)).ok);
        assert_eq!(events.len(), 1);
        let ex = events[0].extras.as_deref().expect("extras present");

        assert_eq!(ex.out_lp_val, vec![-1.1, -1.2]);
        assert_eq!(ex.out_lp_idx, vec![11, 12]);
        assert_eq!(ex.in_lp_val, vec![-2.1]);
        assert_eq!(ex.in_lp_idx, vec![21]);
        assert_eq!(ex.out_top_val, vec![-3.1, -3.2]);
        assert_eq!(ex.out_top_idx, vec![31, 32]);
        assert_eq!(ex.out_top_lens, vec![2]);
        assert_eq!(ex.in_top_val, vec![-4.1]);
        assert_eq!(ex.in_top_idx, vec![41]);
        assert_eq!(ex.in_top_lens, vec![1]);
        assert_eq!(ex.out_tid_val, vec![-5.1, -5.2]);
        assert_eq!(ex.out_tid_idx, vec![51, 52]);
        assert_eq!(ex.out_tid_lens, vec![2]);
        assert_eq!(ex.in_tid_val, vec![-6.1]);
        assert_eq!(ex.in_tid_idx, vec![61]);
        assert_eq!(ex.in_tid_lens, vec![1]);
        assert_eq!(ex.hidden_val, vec![7.1, 7.2, 7.3]);
        assert_eq!(ex.hidden_lens, vec![3]);
        // Every byte of the data buffer was consumed by exactly one family.
        assert_eq!(events[0].token_ids, vec![100]);
    }

    /// Two DISTINCT rids that hash to the same shard must stay separate requests.
    /// Identity is the rid string now, so a shard-hash collision can only co-locate
    /// them — it can no longer merge their `DetokState`, which used to evict one
    /// client's sink and deliver their tokens to the other's connection.
    #[test]
    fn colliding_rids_stay_distinct_requests() {
        use rmpv::Value;
        let header_arr = Value::Array(vec![
            Value::Array(vec![Value::from("alice"), Value::from("bob")]),
            Value::Array(vec![Value::Nil, Value::Nil]),
            Value::Array(vec![Value::from(0u32), Value::from(0u32)]),
            Value::Array(vec![Value::from(1u32), Value::from(1u32)]),
        ]);
        let mut header = Vec::new();
        rmpv::encode::write_value(&mut header, &header_arr).unwrap();
        let data: Vec<u8> = [7i32, 8].iter().flat_map(|x| x.to_le_bytes()).collect();
        let framed = frame_decode_batch_cols(&header, &[&data]);
        let mut events = Vec::new();
        assert!(for_each_chunk(&framed[1..], |ev| events.push(ev)).ok);
        // Each chunk carries its OWN rid — the value a shard keys its table on.
        assert_eq!(events[0].rid, Rid::from("alice"));
        assert_eq!(events[1].rid, Rid::from("bob"));
        assert_ne!(events[0].rid, events[1].rid);
    }

    /// The common frame must stay small: logprob/hidden columns are boxed behind
    /// `ChunkExtras`.
    #[test]
    fn chunk_event_frame_stays_small() {
        let sz = std::mem::size_of::<ChunkEvent>();
        assert!(
            sz <= 144,
            "ChunkEvent grew to {sz} bytes; keep rare columns behind ChunkExtras"
        );
    }
}

#[cfg(test)]
mod rid_recovery_tests {
    use super::*;

    /// A header this build cannot type-decode (Python appended a column) must
    /// still name its requests, or every one of them hangs. The previous
    /// `(Vec<String>,)` tuple decoded ONLY a 1-element array, so it failed on every
    /// real header — arity independence is the entire point of this path.
    #[test]
    fn rid_recovery_works_at_every_header_arity() {
        use rmpv::Value;
        for extra_cols in [0usize, 3, 15, 16] {
            let mut cols = vec![Value::Array(vec![Value::from("a"), Value::from("b")])];
            // Columns of a type this build would reject (strings where u32 is
            // expected) — the "Python widened a column" case.
            cols.extend((0..extra_cols).map(|_| Value::from("unexpected")));
            let mut header = Vec::new();
            rmpv::encode::write_value(&mut header, &Value::Array(cols)).unwrap();
            let framed = frame_decode_batch_cols(&header, &[]);
            let decoded = for_each_chunk(&framed[1..], |_| {});
            assert!(!decoded.ok, "arity {extra_cols}: must reject");
            assert_eq!(
                decoded.rids,
                vec![Rid::from("a"), Rid::from("b")],
                "arity {extra_cols}: rids must survive so the caller can fail them"
            );
        }
    }
}
