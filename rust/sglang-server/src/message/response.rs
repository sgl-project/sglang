//! The runtime response direction: per-request channels into the core
//! boundary ([`ResponseSink`] / [`ResponseItem`]), response frame encodings,
//! and columnar batch decoding into per-request [`ChunkEvent`]s.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use super::finish_reason::FinishReason;
use crate::message::ids::Rid;
use crate::message::types::TokenIds;
use crate::utils::error::Error;

/// Per-request back-channel the detok shard writes runtime outputs to and the
/// core boundary drains; bounded, and receiver-drop cancels the producer.
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

#[allow(dead_code)] // the receiver half is owned inside the core boundary.
pub type ResponseSource = mpsc::Receiver<ResponseItem>;

/// Runtime-facing output delivered to the core boundary: a detok-decoded
/// [`ChunkEvent`], a serialized internal-operation result, or a stage error.
/// [`crate::api_server::handlers::CoreHandle`] translates these variants before a
/// transport adapter sees them.
#[derive(Debug)]
pub enum ResponseItem {
    /// An intermediate generation step: every streamed step of a streaming
    /// request, and every `SGLANG_FORCE_STREAM_INTERVAL` tokens of a
    /// non-streaming one.
    Frame(ChunkEvent),
    /// The final generation step.
    Done(ChunkEvent),
    /// A control-request result encoded by the Python scheduler.
    Control(Bytes),
    /// Reply to an internal service request (`RequestKind::Detokenize`): raw
    /// bytes for the core to decode (e.g. detokenized text), not a
    /// client-facing wire payload and not a generation frame.
    Data(Bytes),
    /// Reply to `RequestKind::Tokenize`: the encoded token ids.
    TokenIds(TokenIds),
    /// Terminal runtime failure; the core maps it to a semantic error.
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
/// A bodiless acknowledgement of the health probes a busy scheduler skipped,
/// sent after its next step.
pub const DISPATCH_TAG_HEALTH_ACK: u8 = 4;

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

/// Read `n` little-endian i64 token ids from `data` at `*off`, advancing `*off`.
/// The scheduler stores output ids as `array("q")` and emits their bytes as-is,
/// so this is the crate's [`TokenIds`] width with no conversion. `None` past
/// the buffer end.
fn take_i64(data: &[u8], off: &mut usize, n: usize) -> Option<TokenIds> {
    let start = *off;
    let end = start.checked_add(n.checked_mul(8)?)?;
    let out = data
        .get(start..end)?
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().expect("8-byte chunk")))
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

/// Columnar scalar header for a whole decode batch. The first six fields are
/// required; every field after `reasoning_tokens` defaults empty, so the hot
/// path emits a six-element header. Field order is the wire ABI and must match
/// `RustTokenizerManager.push_generation`'s `header_cols` in
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
    /// Python's `cached_tokens` / `reasoning_tokens`: running totals, not deltas.
    pub cached_tokens: Vec<u32>,
    pub reasoning_tokens: Vec<u64>,
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
    /// The scheduler's `weight_versions`: a finished request's spans, `None`
    /// for one still running; empty when no request in the batch finished.
    #[serde(default)]
    pub weight_versions: Vec<Option<Vec<WeightVersionSpan>>>,
    /// Media prompt tokens per modality; empty on a text-only model.
    #[serde(default)]
    pub image_tokens: Vec<u32>,
    #[serde(default)]
    pub audio_tokens: Vec<u32>,
    #[serde(default)]
    pub video_tokens: Vec<u32>,
    /// A finished non-streaming request's decode latency in seconds, measured
    /// by the scheduler (`None` for every other row).
    #[serde(default)]
    pub decode_latencies: Vec<Option<f64>>,
    /// Rows the scheduler ended outside the output stream
    /// (`RustServer.push_abort`), which Python's TokenizerManager answers in
    /// `_handle_abort_req` without recording request metrics; empty for an
    /// output-stream batch.
    #[serde(default)]
    pub abort_outputs: Vec<bool>,
    /// The scheduler's `customized_info` as JSON text (see
    /// [`CustomizedInfoJson`]); empty when no request in the batch carries
    /// any.
    #[serde(default)]
    pub customized_info: CustomizedInfoJson,
    /// Each row's `cached_tokens` by cache tier, sent by a scheduler whose
    /// cache has a host or storage tier, or by a PD decode server, which
    /// takes the split from prefill. Empty otherwise: every cached token is
    /// on device.
    #[serde(default)]
    pub cached_tokens_split: Vec<CachedTokensSplit>,
    /// The attached storage tier's backend, on a frame that carries
    /// `cached_tokens_split`: one value for the frame, not a column.
    #[serde(default)]
    pub storage_backend: Option<String>,
    /// A finished request's [`RequestStats`], `None` for one still running;
    /// empty when no request in the batch finished.
    #[serde(default)]
    pub request_stats: Vec<Option<RequestStats>>,
}

/// The `meta_info` keys Python's TokenizerManager derives from scheduler
/// state for a finished request (`SchedulerOutputStreamer.get_request_stats`):
/// the retraction count, the scheduler's timings when metrics are on, and the
/// speculative-decoding stats. Built by the scheduler with Python's own
/// helpers, so a native `/generate` terminal frame reports them as they are,
/// keys and values both, without per-key code here.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RequestStats(pub serde_json::Map<String, serde_json::Value>);

/// The completion tokens `[start, end)` one weight version produced, as the
/// scheduler's `WeightVersionSpan` encodes it: `[version, start, end]`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightVersionSpan {
    pub version: String,
    pub start: u64,
    pub end: u64,
}

/// The batch's `customized_info` as the JSON text Python's responses carry
/// it in (orjson, numpy values as lists, a value it has no form for as its
/// `str`): per key, one list of values per row, a row whose request lacks the
/// key padded with nulls, one per new output token. Text, so no value a model
/// puts there can fail the header's decode.
#[derive(Debug, Default)]
pub struct CustomizedInfoJson(Vec<u8>);

impl Serialize for CustomizedInfoJson {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(&self.0)
    }
}

impl<'de> Deserialize<'de> for CustomizedInfoJson {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        /// msgpack bin, as Python's `bytes` encode.
        struct Text;

        impl serde::de::Visitor<'_> for Text {
            type Value = CustomizedInfoJson;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("customized_info JSON bytes")
            }

            fn visit_bytes<E: serde::de::Error>(self, bytes: &[u8]) -> Result<Self::Value, E> {
                Ok(CustomizedInfoJson(bytes.to_vec()))
            }

            fn visit_byte_buf<E: serde::de::Error>(self, bytes: Vec<u8>) -> Result<Self::Value, E> {
                Ok(CustomizedInfoJson(bytes))
            }
        }

        deserializer.deserialize_bytes(Text)
    }
}

impl CustomizedInfoJson {
    /// Per key, in the scheduler's order, one list of values per row of a
    /// `rows`-request batch. A text that is not that shape costs the batch its
    /// `customized_info` (logged), never the frame.
    fn columns(&self, rows: usize) -> Vec<(Arc<str>, Vec<Vec<serde_json::Value>>)> {
        if self.0.is_empty() {
            return Vec::new();
        }
        let columns = serde_json::from_slice::<KeyedColumns>(&self.0)
            .map_err(|error| error.to_string())
            .and_then(|KeyedColumns(columns)| {
                if columns.iter().all(|(_, column)| column.len() == rows) {
                    Ok(columns)
                } else {
                    Err(format!("a column without one entry per request ({rows})"))
                }
            });
        match columns {
            Ok(columns) => columns
                .into_iter()
                .map(|(key, column)| (Arc::from(key), column))
                .collect(),
            Err(error) => {
                static DROPPED: AtomicU64 = AtomicU64::new(0);
                let dropped = DROPPED.fetch_add(1, Ordering::Relaxed) + 1;
                if dropped.is_power_of_two() {
                    tracing::warn!(dropped, %error, "dropped a batch's unreadable customized_info");
                }
                Vec::new()
            }
        }
    }
}

/// A JSON object of columns, read entry by entry so the key order survives.
struct KeyedColumns(Vec<(String, Vec<Vec<serde_json::Value>>)>);

impl<'de> Deserialize<'de> for KeyedColumns {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Columns;

        impl<'de> serde::de::Visitor<'de> for Columns {
            type Value = KeyedColumns;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a map of customized_info columns")
            }

            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<Self::Value, A::Error> {
                let mut columns = Vec::new();
                while let Some(column) = map.next_entry()? {
                    columns.push(column);
                }
                Ok(KeyedColumns(columns))
            }
        }

        deserializer.deserialize_map(Columns)
    }
}

/// One request's `customized_info` in one frame: for each key its batch
/// carried, the values for the request's new output tokens.
#[derive(Debug, Clone, Default)]
pub struct CustomizedInfo(pub(crate) Vec<(Arc<str>, Vec<serde_json::Value>)>);

impl CustomizedInfo {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl IntoIterator for CustomizedInfo {
    type Item = (Arc<str>, Vec<serde_json::Value>);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

/// One row of [`BatchHeader::cached_tokens_split`], encoded
/// `[device, host, storage]`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CachedTokensSplit {
    pub device: u32,
    pub host: u32,
    pub storage: u32,
}

/// A request's cached prompt tokens by the cache tier that served them, as
/// `meta_info.cached_tokens_details` reports them. A chunk carries one only
/// when it says more than `cached_tokens` does (see [`Self::from_split`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedTokensDetails {
    pub device: u32,
    pub host: u32,
    /// Reported when nonzero or while a storage tier is attached.
    pub storage: Option<u32>,
    /// The attached storage tier's backend.
    pub storage_backend: Option<Arc<str>>,
}

impl CachedTokensDetails {
    /// The details a row's split reports, or `None` when they are what
    /// [`Self::on_device`] makes of `cached_tokens` alone: a split without
    /// hits falls back to the total, and one with every hit on device and no
    /// storage tier attached reports just the device count.
    pub fn from_split(
        split: CachedTokensSplit,
        cached_tokens: u32,
        storage_backend: Option<&Arc<str>>,
    ) -> Option<Self> {
        let CachedTokensSplit {
            device,
            host,
            storage,
        } = split;
        let device_only = host == 0 && storage == 0 && storage_backend.is_none();
        if (device == 0 && host == 0 && storage == 0) || (device_only && device == cached_tokens) {
            return None;
        }
        Some(Self {
            device,
            host,
            storage: (storage > 0 || storage_backend.is_some()).then_some(storage),
            storage_backend: storage_backend.cloned(),
        })
    }

    /// The details of a chunk that carries none: every cached token on
    /// device, or `None` without cached tokens.
    pub fn on_device(cached_tokens: u32) -> Option<Self> {
        (cached_tokens > 0).then_some(Self {
            device: cached_tokens,
            host: 0,
            storage: None,
            storage_backend: None,
        })
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
    // extras columns; the core ones were unchecked.
    if h.finish_reasons.len() != n
        || h.prompt_tokens.len() != n
        || h.tok_lens.len() != n
        || h.cached_tokens.len() != n
        || h.reasoning_tokens.len() != n
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
        || !(h.weight_versions.is_empty() || h.weight_versions.len() == n)
        || !per_req_ok(&h.image_tokens)
        || !per_req_ok(&h.audio_tokens)
        || !per_req_ok(&h.video_tokens)
        || !(h.decode_latencies.is_empty() || h.decode_latencies.len() == n)
        || !(h.abort_outputs.is_empty() || h.abort_outputs.len() == n)
        || !(h.cached_tokens_split.is_empty() || h.cached_tokens_split.len() == n)
        || !(h.request_stats.is_empty() || h.request_stats.len() == n)
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
    // Token ids are 8-byte (`array("q")` bytes as the scheduler holds them);
    // every logprob / hidden column is 4-byte (f32 values, i32 indices).
    let mut col = |count: usize, width: usize| -> usize {
        let start = base;
        base += count * width;
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
    let mut c_ids = col(n_ids, 8);
    let mut c_olp_v = col(n_olp, 4);
    let mut c_olp_i = col(n_olp, 4);
    let mut c_ilp_v = col(n_ilp, 4);
    let mut c_ilp_i = col(n_ilp, 4);
    let mut c_ot_v = col(n_ot, 4);
    let mut c_ot_i = col(n_ot, 4);
    let mut c_it_v = col(n_it, 4);
    let mut c_it_i = col(n_it, 4);
    let mut c_od_v = col(n_od, 4);
    let mut c_od_i = col(n_od, 4);
    let mut c_id_v = col(n_id, 4);
    let mut c_id_i = col(n_id, 4);
    let mut c_h_v = col(n_h, 4);

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
    // Each key once per frame, shared by the rows.
    let mut customized_columns = h.customized_info.columns(n);
    // Shared by every row whose cached-token details name it.
    let storage_backend: Option<Arc<str>> = h.storage_backend.take().map(Arc::from);

    // Decode one request's slice of every column, advancing the cursors. `None` if a
    // read overruns `data` (belt-and-suspenders past the upfront check, which already
    // covers every column's span — so this is unreachable today). It stops the loop
    // before slicing out of bounds, but requests decoded earlier in the frame are
    // already routed; making that abort atomic would mean buffering the whole frame,
    // which is exactly what the streaming decode avoids.
    let mut decode_one = |i: usize| -> Option<ChunkEvent> {
        let token_ids = take_i64(data, &mut c_ids, lens_i(&h.tok_lens, i))?;
        // The metadata columns after the extras: per-row values, present when a
        // request in the frame finished or the model is multimodal.
        let weight_versions = h
            .weight_versions
            .get_mut(i)
            .and_then(Option::take)
            .unwrap_or_default();
        let image_tokens = h.image_tokens.get(i).copied().unwrap_or(0);
        let audio_tokens = h.audio_tokens.get(i).copied().unwrap_or(0);
        let video_tokens = h.video_tokens.get(i).copied().unwrap_or(0);
        let decode_latency = h.decode_latencies.get(i).copied().flatten();
        let abort_output = h.abort_outputs.get(i).copied().unwrap_or(false);
        let customized_info = CustomizedInfo(
            customized_columns
                .iter_mut()
                .map(|(key, rows)| (key.clone(), std::mem::take(&mut rows[i])))
                .collect(),
        );
        let cached_tokens = h.cached_tokens.get(i).copied().unwrap_or(0);
        let cached_tokens_details = h.cached_tokens_split.get(i).and_then(|&split| {
            CachedTokensDetails::from_split(split, cached_tokens, storage_backend.as_ref())
        });
        let request_stats = h.request_stats.get_mut(i).and_then(Option::take);

        // Plain decode frame (no request in the batch asked for logprobs/hidden):
        // the extras columns are all zero-width, so skip reading them entirely,
        // and box the metadata only for a row that carries any.
        let extras = if !has_extras {
            let has_metadata = !weight_versions.is_empty()
                || image_tokens != 0
                || audio_tokens != 0
                || video_tokens != 0
                || decode_latency.is_some()
                || abort_output
                || !customized_info.is_empty()
                || cached_tokens_details.is_some()
                || request_stats.is_some();
            has_metadata.then(|| {
                Box::new(ChunkExtras {
                    weight_versions,
                    image_tokens,
                    audio_tokens,
                    video_tokens,
                    decode_latency,
                    abort_output,
                    customized_info,
                    cached_tokens_details,
                    request_stats,
                    ..Default::default()
                })
            })
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
                weight_versions,
                image_tokens,
                audio_tokens,
                video_tokens,
                decode_latency,
                abort_output,
                customized_info,
                cached_tokens_details,
                request_stats,
                // Explicit, NOT `..Default::default()` — same reason as `ChunkEvent`
                // below: a new column must fail to compile here until it is decoded.
                out_lp_txt: Vec::new(),
                in_lp_txt: Vec::new(),
                out_top_txt: Vec::new(),
                in_top_txt: Vec::new(),
                out_tid_txt: Vec::new(),
                in_tid_txt: Vec::new(),
            };
            (!ex.is_empty()).then(|| Box::new(ex))
        };

        Some(ChunkEvent {
            // Any string is a valid rid; hash to the routing key. An unknown
            // rid routes to a shard whose table has no entry → dropped there.
            rid: std::mem::take(&mut h.rids[i]).into(),
            token_ids,
            finish_reason: h.finish_reasons.get(i).cloned().flatten(),
            prompt_tokens: h.prompt_tokens.get(i).copied().unwrap_or(0),
            cached_tokens,
            reasoning_tokens: h.reasoning_tokens.get(i).copied().unwrap_or(0),
            extras,
            // Listed explicitly, NOT `..Default::default()`: a new column added to
            // `ChunkEvent` and wired into the response must fail to compile here
            // until it is actually decoded. With the struct-update syntax it
            // compiled clean and silently shipped zeros.
            text: String::new(),
            completion_tokens: 0,
            stop_token_trimmed: false,
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
    use rmpv::Value;
    let arr = Value::Array(vec![Value::from(rid), Value::Binary(payload.to_vec())]);
    let mut buf = Vec::with_capacity(1 + payload.len() + rid.len() + 8);
    buf.push(DISPATCH_TAG_RESULT);
    let _ = rmpv::encode::write_value(&mut buf, &arr);
    Bytes::from(buf)
}

/// Frame a per-request validation failure `[rid, message]` for the response —
/// routes a terminal error back to the owning request instead of crashing.
pub fn frame_error(rid: &str, message: &str) -> Bytes {
    use rmpv::Value;
    let arr = Value::Array(vec![Value::from(rid), Value::from(message)]);
    let mut buf = Vec::with_capacity(1 + rid.len() + message.len() + 8);
    buf.push(DISPATCH_TAG_ERROR);
    let _ = rmpv::encode::write_value(&mut buf, &arr);
    Bytes::from(buf)
}

/// Frame a health-probe acknowledgement: the tag alone.
pub fn frame_health_ack() -> Bytes {
    Bytes::from_static(&[DISPATCH_TAG_HEALTH_ACK])
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
    /// Runtime correlation ID. Moved out of the frame header (which owns it and
    /// drops it), so carrying it costs no allocation. The core strips this
    /// field before exposing semantic output; the shard routes it via `Rid::shard`.
    pub rid: Rid,
    /// New token ids for this step, widened from the scheduler's int32 wire
    /// width at parse time. Empty allowed (e.g. metadata-only frames).
    pub token_ids: TokenIds,
    /// `None` while streaming, the [`FinishReason`] on the final chunk.
    pub finish_reason: Option<FinishReason>,
    /// Prompt token count for this request (constant across its chunks).
    pub prompt_tokens: u32,
    /// Prefix-cache hits and reasoning tokens so far: running totals, so the
    /// latest chunk's value is the request's.
    pub cached_tokens: u32,
    pub reasoning_tokens: u64,
    /// Decoded text **delta** for this chunk (empty in skip mode / on partial UTF-8),
    /// filled by the detok shard. `token_ids` doubles as `output_ids`;
    /// `completion_tokens` is this chunk's count.
    pub text: String,
    pub completion_tokens: u64,
    /// Set by the detok shard when the last of `token_ids` is a matched stop
    /// token kept in `output_ids` but left out of `text`.
    pub stop_token_trimmed: bool,
    /// Logprob + hidden-state columns — `None` unless the request asked for them —
    /// and the rarer per-request metadata. Boxed to keep the common
    /// token/text/finish frame small at large decode batches (the decoder
    /// allocates it only when a column is non-empty).
    pub extras: Option<Box<ChunkExtras>>,
}

/// Logprob + hidden-state columns for a [`ChunkEvent`], allocated only when the
/// request enabled logprobs / hidden states, finished (its weight-version spans
/// and request stats), has media prompt tokens, its batch carried
/// `customized_info` or it reports its cached tokens by tier. Columnar
/// `val`/`idx` (+ ragged `lens`) buffers arrive pre-decode; the detok shard
/// fills the parallel `*_txt` columns when `return_text_in_logprobs` is set.
/// In-process only — no serde (see [`ChunkEvent`]).
#[derive(Debug, Clone, Default)]
pub struct ChunkExtras {
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
    /// Which weight version produced which output tokens; on a finished
    /// request's chunk only.
    pub weight_versions: Vec<WeightVersionSpan>,
    /// Media prompt tokens per modality; zero for a text-only request.
    pub image_tokens: u32,
    pub audio_tokens: u32,
    pub video_tokens: u32,
    /// A finished non-streaming request's decode latency in seconds, from the
    /// scheduler.
    pub decode_latency: Option<f64>,
    /// The scheduler ended this request outside the output stream (see
    /// [`BatchHeader::abort_outputs`]).
    pub abort_output: bool,
    /// The scheduler's `customized_info` for this frame's new output tokens.
    pub customized_info: CustomizedInfo,
    /// This request's cached tokens by cache tier, when they say more than
    /// `cached_tokens` does.
    pub cached_tokens_details: Option<CachedTokensDetails>,
    /// A finished request's stats (see [`BatchHeader::request_stats`]).
    pub request_stats: Option<RequestStats>,
    /// Decoded logprob token text (`return_text_in_logprobs`), parallel to the
    /// `*_idx` buffers; empty when not requested (the tuple's text slot stays null).
    pub out_lp_txt: Vec<String>,
    pub in_lp_txt: Vec<String>,
    pub out_top_txt: Vec<String>,
    pub in_top_txt: Vec<String>,
    pub out_tid_txt: Vec<String>,
    pub in_tid_txt: Vec<String>,
}

impl ChunkExtras {
    /// True when no column carries data — lets the decoder skip the box
    /// allocation for the common (extras-free) frame.
    fn is_empty(&self) -> bool {
        self.out_lp_val.is_empty()
            && self.in_lp_val.is_empty()
            && self.out_top_lens.is_empty()
            && self.in_top_lens.is_empty()
            && self.out_tid_lens.is_empty()
            && self.in_tid_lens.is_empty()
            && self.hidden_lens.is_empty()
            && self.weight_versions.is_empty()
            && self.image_tokens == 0
            && self.audio_tokens == 0
            && self.video_tokens == 0
            && self.decode_latency.is_none()
            && !self.abort_output
            && self.customized_info.is_empty()
            && self.cached_tokens_details.is_none()
            && self.request_stats.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::finish_reason::{FinishKind, Matched};

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
        // prompt_tokens [4,5,6]; tok_lens [2,0,1] -> ids [10,11 | (none) | 12];
        // cached_tokens [3,0,1]; reasoning_tokens [0,0,7].
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
            Value::Array(vec![
                Value::from(3u32),
                Value::from(0u32),
                Value::from(1u32),
            ]),
            Value::Array(vec![
                Value::from(0u64),
                Value::from(0u64),
                Value::from(7u64),
            ]),
        ]);
        let mut header = Vec::new();
        rmpv::encode::write_value(&mut header, &header_arr).unwrap();
        let data: Vec<u8> = [10i64, 11, 12]
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
        assert_eq!(events[0].cached_tokens, 3);
        assert_eq!(events[2].reasoning_tokens, 7);
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
        // 1 request: tok_lens[0]=10 claims 80 bytes and out_lp_lens[0]=1 puts the
        // logprob column's base past the 4-byte data buffer. The old clamp-only-`end`
        // code advanced the cursor past `len`, then sliced `data[40..4]` (start > end)
        // and panicked.
        let header_arr = Value::Array(vec![
            Value::Array(vec![Value::from("1")]),   // rids
            Value::Array(vec![Value::Nil]),         // finish_reasons
            Value::Array(vec![Value::from(0u32)]),  // prompt_tokens
            Value::Array(vec![Value::from(10u32)]), // tok_lens (claims 80 bytes)
            Value::Array(vec![Value::from(0u32)]),  // cached_tokens
            Value::Array(vec![Value::from(0u32)]),  // reasoning_tokens
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
        let q = |xs: &[i64]| -> Vec<u8> { xs.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let arr_u = |xs: &[u32]| Value::Array(xs.iter().map(|&x| Value::from(x)).collect());
        let rids = Value::Array(vec![Value::from("1"), Value::from("2")]);
        let finish = Value::Array(vec![Value::Nil, Value::Nil]);

        // Ragged column: reqs claim 2 + 1 = 3 positions, `out_top_poslens` has 2.
        let header_arr = Value::Array(vec![
            rids.clone(),
            finish.clone(),
            arr_u(&[0, 0]), // prompt
            arr_u(&[1, 1]), // tok_lens
            arr_u(&[0, 0]), // cached_tokens
            arr_u(&[0, 0]), // reasoning_tokens
            arr_u(&[0, 0]), // out_lp_lens
            arr_u(&[0, 0]), // in_lp_lens
            arr_u(&[2, 1]), // out_top_reqlens — 3 positions claimed
            arr_u(&[2, 2]), // out_top_poslens — only 2 supplied
        ]);
        let mut header = Vec::new();
        rmpv::encode::write_value(&mut header, &header_arr).unwrap();
        let mut data = Vec::new();
        data.extend(q(&[10, 20])); // token_ids
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
            arr_u(&[0, 0]), // cached_tokens
            arr_u(&[0, 0]), // reasoning_tokens
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
        data.extend(q(&[10, 20])); // token_ids
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
            Value::Array(vec![Value::from(0u32), Value::from(0u32)]), // cached_tokens
            Value::Array(vec![Value::from(0u32), Value::from(0u32)]), // reasoning_tokens
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
            Value::Array(vec![Value::from(1u32)]), // tok_lens: 1 id = 8 bytes
            Value::Array(vec![Value::from(0u32)]), // cached_tokens
            Value::Array(vec![Value::from(0u32)]), // reasoning_tokens
        ]);
        let mut header = Vec::new();
        rmpv::encode::write_value(&mut header, &header_arr).unwrap();
        let data: Vec<u8> = vec![0u8; 16]; // 8 bytes too many
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
            Value::Array(vec![Value::from(0u32)]),            // cached_tokens
            Value::Array(vec![Value::from(0u32)]),            // reasoning_tokens
        ]);
        let mut header = Vec::new();
        rmpv::encode::write_value(&mut header, &header_arr).unwrap();
        let data: Vec<u8> = [0i64].iter().flat_map(|x| x.to_le_bytes()).collect();

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
        let q = |xs: &[i64]| -> Vec<u8> { xs.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let arr_u = |xs: &[u32]| Value::Array(xs.iter().map(|&x| Value::from(x)).collect());
        // header: rids, finish, prompt, tok_lens, cached_tokens, reasoning_tokens,
        //   out_lp_lens, in_lp_lens,
        //   out_top_reqlens, out_top_poslens, in_top_*, out_tokids_lp_*,
        //   in_tokids_lp_*,
        //   hidden_reqlens, hidden_poslens
        let header_arr = Value::Array(vec![
            Value::Array(vec![Value::from("1"), Value::from("2")]), // rids
            Value::Array(vec![Value::Nil, Value::Nil]),             // finish
            arr_u(&[3, 4]),                                         // prompt
            arr_u(&[1, 1]),                                         // tok_lens
            arr_u(&[0, 0]),                                         // cached_tokens
            arr_u(&[0, 0]),                                         // reasoning_tokens
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
        data.extend(q(&[10, 20])); // token_ids: req0=[10], req1=[20]
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
        let q = |xs: &[i64]| -> Vec<u8> { xs.iter().flat_map(|x| x.to_le_bytes()).collect() };
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
                arr_u(&[0, 0]),                                         // cached_tokens
                arr_u(&[0, 0]),                                         // reasoning_tokens
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
            data.extend(q(&[10, 20])); // token_ids
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
        let q = |xs: &[i64]| -> Vec<u8> { xs.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let arr_u = |xs: &[u32]| Value::Array(xs.iter().map(|&x| Value::from(x)).collect());

        let header_arr = Value::Array(vec![
            Value::Array(vec![Value::from("1")]), // rids
            Value::Array(vec![Value::Nil]),       // finish
            arr_u(&[9]),                          // prompt
            arr_u(&[1]),                          // tok_lens
            arr_u(&[0]),                          // cached_tokens
            arr_u(&[0]),                          // reasoning_tokens
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
        data.extend(q(&[100])); // token_ids
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
            Value::Array(vec![Value::from(0u32), Value::from(0u32)]),
            Value::Array(vec![Value::from(0u32), Value::from(0u32)]),
        ]);
        let mut header = Vec::new();
        rmpv::encode::write_value(&mut header, &header_arr).unwrap();
        let data: Vec<u8> = [7i64, 8].iter().flat_map(|x| x.to_le_bytes()).collect();
        let framed = frame_decode_batch_cols(&header, &[&data]);
        let mut events = Vec::new();
        assert!(for_each_chunk(&framed[1..], |ev| events.push(ev)).ok);
        // Each chunk carries its OWN rid — the value a shard keys its table on.
        assert_eq!(events[0].rid, Rid::from("alice"));
        assert_eq!(events[1].rid, Rid::from("bob"));
        assert_ne!(events[0].rid, events[1].rid);
    }

    /// A finished request's spans ride the frame without any logprob columns;
    /// only rows that carry metadata get a `ChunkExtras` box.
    #[test]
    fn metadata_columns_box_only_the_rows_that_carry_them() {
        let span = WeightVersionSpan {
            version: "v".into(),
            start: 0,
            end: 1,
        };
        let header = BatchHeader {
            rids: vec!["a".into(), "b".into(), "c".into()],
            finish_reasons: vec![None, None, None],
            prompt_tokens: vec![1, 1, 1],
            tok_lens: vec![1, 1, 1],
            cached_tokens: vec![0, 0, 0],
            reasoning_tokens: vec![0, 0, 0],
            weight_versions: vec![None, Some(vec![span.clone()]), None],
            ..Default::default()
        };
        let header = rmp_serde::to_vec(&header).unwrap();
        let data: Vec<u8> = [1i64, 2, 3].iter().flat_map(|x| x.to_le_bytes()).collect();
        let framed = frame_decode_batch_cols(&header, &[&data]);
        let mut events = Vec::new();
        assert!(for_each_chunk(&framed[1..], |ev| events.push(ev)).ok);
        assert!(events[0].extras.is_none() && events[2].extras.is_none());
        let extras = events[1].extras.as_deref().unwrap();
        assert_eq!(extras.weight_versions, [span]);
        assert!(extras.out_lp_val.is_empty() && extras.image_tokens == 0);
    }

    /// Whatever text the `customized_info` column holds, the frame decodes:
    /// a well-formed one splits by row, keeping the key order and any JSON
    /// value; malformed JSON, a wrong shape or a short column costs the batch
    /// only its `customized_info`.
    #[test]
    fn customized_info_text_never_rejects_the_frame() {
        let decode = |text: &str| {
            let header = BatchHeader {
                rids: vec!["a".into(), "b".into()],
                finish_reasons: vec![None, None],
                prompt_tokens: vec![1, 1],
                tok_lens: vec![1, 0],
                cached_tokens: vec![0, 0],
                reasoning_tokens: vec![0, 0],
                customized_info: CustomizedInfoJson(text.as_bytes().to_vec()),
                ..Default::default()
            };
            let header = rmp_serde::to_vec(&header).unwrap();
            let framed = frame_decode_batch_cols(&header, &[&7i64.to_le_bytes()]);
            let mut events = Vec::new();
            assert!(
                for_each_chunk(&framed[1..], |ev| events.push(ev)).ok,
                "{text}"
            );
            events
                .into_iter()
                .map(|ev| {
                    ev.extras
                        .map(|extras| extras.customized_info.0)
                        .unwrap_or_default()
                })
                .collect::<Vec<_>>()
        };
        let rows = decode(r#"{"z": [[{"k": 1}], []], "a": [[null], ["b'x'"]]}"#);
        let keys: Vec<&str> = rows[0].iter().map(|(key, _)| &**key).collect();
        assert_eq!(keys, ["z", "a"]);
        assert_eq!(rows[0][0].1, [serde_json::json!({"k": 1})]);
        assert_eq!(rows[1][1].1, [serde_json::json!("b'x'")]);
        for odd in [
            "{not json",
            r#"["a", "list"]"#,
            r#"{"k": [1, 2]}"#,
            r#"{"k": [[1]]}"#,
        ] {
            assert_eq!(decode(odd), [vec![], vec![]], "{odd}");
        }
    }

    /// The common frame must stay small: logprob/hidden columns are boxed behind
    /// `ChunkExtras`. 152 since the matched stop token rides in the frame (#42637).
    #[test]
    fn chunk_event_frame_stays_small() {
        let sz = std::mem::size_of::<ChunkEvent>();
        assert!(
            sz <= 152,
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
