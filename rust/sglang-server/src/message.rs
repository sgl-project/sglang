//! Messages moved between stages. All payloads are *moved* through `flume`
//! channels (zero copy); variable-length buffers are `bytes::Bytes` so the
//! egress fan-out to detokenizer shards is a refcount bump, never a copy.

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use tokio::sync::mpsc;

use crate::error::Error;
use crate::fsm::RequestState;
use crate::ids::RequestId;

/// Where the egress side (detok shard) writes a request's generation/control
/// frames — the in-process per-request `Local` channel the API handler drains
/// for SSE. One per request, bounded for backpressure; dropping the receiver
/// (client disconnect) is observed as stream end.
#[derive(Clone, Debug)]
pub enum EgressSink {
    Local(mpsc::Sender<EgressItem>),
}

/// Why an [`EgressSink::try_send`] failed. `Full` = the client isn't reading fast
/// enough (backpressure); `Closed` = the client is gone. Both are terminal for a
/// streaming request (the shard can't buffer unboundedly), but the caller
/// distinguishes them for logging and reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SinkError {
    Full,
    Closed,
}

impl EgressSink {
    /// Non-blocking send. `Err(Full)` = backpressure, `Err(Closed)` = client gone.
    pub fn try_send(&self, item: EgressItem) -> Result<(), SinkError> {
        match self {
            EgressSink::Local(tx) => tx.try_send(item).map_err(|e| match e {
                mpsc::error::TrySendError::Full(_) => SinkError::Full,
                mpsc::error::TrySendError::Closed(_) => SinkError::Closed,
            }),
        }
    }
}

#[allow(dead_code)] // the receiver half is created inline in api_server::submit.
pub type EgressSource = mpsc::Receiver<EgressItem>;

/// Protocol-neutral output for one generation step — a **per-chunk delta**. The
/// detok shard emits one per decode step (it keeps no cumulative buffer); a
/// consumer that needs the cumulative view folds them with `OutputAccumulator`
/// (every unary response and the cumulative SGLang `/generate` stream). The
/// `/generate` handler formats the result into the SGLang wire shape.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerationOutput {
    /// Decoded text **delta** for this chunk (empty in `skip_tokenizer_init`
    /// mode, or when the step only produced a partial multi-byte sequence).
    pub text: String,
    /// Output token ids for this chunk. Surfaced as the `/generate` response's
    /// `output_ids` (returned by default, like the Python server) — populated in
    /// both normal and `skip_tokenizer_init` mode.
    pub output_ids: Vec<i32>,
    /// Prompt token count (from the scheduler; constant across the request).
    pub prompt_tokens: u32,
    /// Output token count for this chunk (the accumulator sums them).
    pub completion_tokens: u64,
    /// Full finish-reason dict on the final step; `None` while streaming.
    pub finish_reason: Option<serde_json::Value>,
    /// Output-token logprobs (delta for this chunk; the accumulator concatenates
    /// them). Parallel `val`/`idx` buffers. Empty unless `return_logprob`.
    #[serde(default)]
    pub out_lp_val: Vec<f32>,
    #[serde(default)]
    pub out_lp_idx: Vec<i32>,
    /// Input (prefill) token logprobs — set once, on the first chunk.
    #[serde(default)]
    pub in_lp_val: Vec<f32>,
    #[serde(default)]
    pub in_lp_idx: Vec<i32>,
    /// Top-k logprobs (2-level ragged): flat `val`/`idx` + per-position `lens`.
    #[serde(default)]
    pub out_top_val: Vec<f32>,
    #[serde(default)]
    pub out_top_idx: Vec<i32>,
    #[serde(default)]
    pub out_top_lens: Vec<u32>,
    #[serde(default)]
    pub in_top_val: Vec<f32>,
    #[serde(default)]
    pub in_top_idx: Vec<i32>,
    #[serde(default)]
    pub in_top_lens: Vec<u32>,
    /// Token-ids logprobs (same 2-level ragged layout).
    #[serde(default)]
    pub out_tid_val: Vec<f32>,
    #[serde(default)]
    pub out_tid_idx: Vec<i32>,
    #[serde(default)]
    pub out_tid_lens: Vec<u32>,
    #[serde(default)]
    pub in_tid_val: Vec<f32>,
    #[serde(default)]
    pub in_tid_idx: Vec<i32>,
    #[serde(default)]
    pub in_tid_lens: Vec<u32>,
    /// Hidden states (dense f32): flat buffer + per-row lengths. Last-writer-wins
    /// across chunks (mirrors Python's non-cumulative `meta_info` assignment).
    #[serde(default)]
    pub hidden_val: Vec<f32>,
    #[serde(default)]
    pub hidden_lens: Vec<u32>,
    /// Decoded logprob token text (`return_text_in_logprobs`), flat and parallel
    /// to the matching `*_idx` buffers. Decoded on the detok shard; empty when
    /// the request didn't ask for text (the tuple's text slot stays null).
    #[serde(default)]
    pub out_lp_txt: Vec<String>,
    #[serde(default)]
    pub in_lp_txt: Vec<String>,
    #[serde(default)]
    pub out_top_txt: Vec<String>,
    #[serde(default)]
    pub in_top_txt: Vec<String>,
    #[serde(default)]
    pub out_tid_txt: Vec<String>,
    #[serde(default)]
    pub in_tid_txt: Vec<String>,
}

/// What the connection handler receives on the egress stream. Generation output
/// is protocol-neutral (the handler formats it); control results are a single
/// verbatim payload.
#[derive(Debug)]
pub enum EgressItem {
    /// An intermediate streamed generation step (only sent for streaming reqs).
    Frame(GenerationOutput),
    /// The final generation step.
    Done(GenerationOutput),
    /// A control-request result: one verbatim payload (e.g. `/server_info`),
    /// delivered as-is with no per-protocol formatting.
    Control(Bytes),
    /// Terminal failure: handler emits an error frame (stream) or status (unary).
    Error(Error),
}

/// What kind of request this is — selects the ingress branch, the wire message
/// pushed to the scheduler, and the egress shape. Each variant owns its own
/// body, so the type system keeps generate fields off control requests (and
/// vice versa); a control endpoint migrated with parameters grows
/// `ControlRequest` rather than abusing the generate payload.
#[derive(Debug)]
pub enum RequestKind {
    /// `/generate`: tokenize (if needed) then push a `TokenizedGenerateReqInput`.
    Generate(GenerateRequest),
    /// A control endpoint (e.g. `/server_info`, `/health`): no tokenization, and
    /// the egress is a single non-streamed JSON result.
    Control(ControlRequest),
}

/// Body of a `/generate` request.
#[derive(Debug, Default)]
pub struct GenerateRequest {
    /// Decoded HTTP body (the `GenerateReqInput` view we need for tokenization).
    pub payload: GeneratePayload,
    /// Token ids, populated by the Tokenizer stage (or already present from the
    /// client).
    pub input_ids: Option<Vec<i32>>,
    /// Whether the client asked for SSE streaming.
    pub stream: bool,
    /// Internal `/health_generate` probe: the scheduler skips it when busy so it
    /// never occupies a waiting-queue slot. Never set from the client wire.
    pub is_health_check: bool,
}

/// Body of a control request. `tag` is the scheduler request-struct name
/// (msgspec class, e.g. `"GetInternalStateReq"`) pushed as a bare
/// `[tag, rid, nil]`. Typed params for migrated control endpoints land here.
#[derive(Debug)]
pub struct ControlRequest {
    pub tag: &'static str,
}

/// The owned request as it travels ingress stages. Single owner at all times,
/// so the embedded `state` FSM is mutated without any lock. Fields common to
/// every request live here; variant-specific data lives in [`RequestKind`].
#[derive(Debug)]
pub struct Request {
    pub id: RequestId,
    pub state: RequestState,
    /// Back-channel to the client connection for egress frames.
    pub sink: EgressSink,
    /// Discriminant + variant body (generate vs control).
    pub kind: RequestKind,
}

/// Encode a bare `BaseReq` control message (just `rid` + `http_worker_ipc`) as
/// the msgspec tagged array `[tag, rid, nil]`. Used for control requests like
/// `GetInternalStateReq` that carry no extra fields.
pub fn control_req_msgpack(tag: &str, rid: &str) -> Result<Bytes, Error> {
    use rmpv::Value;
    let arr = Value::Array(vec![
        Value::from(tag), // struct tag
        Value::from(rid), // rid
        Value::Nil,       // http_worker_ipc
    ]);
    let mut buf = Vec::new();
    rmpv::encode::write_value(&mut buf, &arr).map_err(|e| Error::Codec(e.to_string()))?;
    Ok(Bytes::from(buf))
}

/// Encode an `AbortReq(rid)` as its msgspec tagged array. `AbortReq` extends
/// `BaseReq` (`rid`, `http_worker_ipc`) with `abort_all`, `finished_reason`,
/// `abort_message`, so the array is `["AbortReq", rid, nil, false, nil, nil]`.
/// The scheduler decodes it off the ingress ring and dispatches to its
/// `abort_request`, stopping generation for `rid`.
pub fn abort_req_msgpack(rid: &str) -> Result<Bytes, Error> {
    use rmpv::Value;
    let arr = Value::Array(vec![
        Value::from("AbortReq"), // struct tag
        Value::from(rid),        // rid
        Value::Nil,              // http_worker_ipc
        Value::from(false),      // abort_all
        Value::Nil,              // finished_reason
        Value::Nil,              // abort_message
    ]);
    let mut buf = Vec::new();
    rmpv::encode::write_value(&mut buf, &arr).map_err(|e| Error::Codec(e.to_string()))?;
    Ok(Bytes::from(buf))
}

/// Minimal decoded view of an incoming `/generate` body. Core fields are typed;
/// everything else round-trips through `extra` so we stay faithful to the full
/// Python schema (and the in-flight msgpack-migration) without enumerating it.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GeneratePayload {
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub input_ids: Option<Vec<i32>>,
    #[serde(default)]
    pub stream: bool,
    /// Opaque sampling params, carried through to the scheduler untouched.
    #[serde(default)]
    pub sampling_params: Option<rmpv::Value>,
    /// Logprob / hidden-state request options. This path bypasses the Python
    /// `TokenizerManager`, so the ingress replicates its scalar normalization
    /// (see [`TokenizedReqPayload`]) before handing them to the scheduler.
    #[serde(default)]
    pub return_logprob: Option<bool>,
    #[serde(default)]
    pub logprob_start_len: Option<i64>,
    #[serde(default)]
    pub top_logprobs_num: Option<i64>,
    #[serde(default)]
    pub token_ids_logprob: Option<rmpv::Value>,
    #[serde(default)]
    pub return_hidden_states: Option<bool>,
    /// Decode logprob token ids to text in each `[logprob, token_id, text]`
    /// tuple (the api-server does this at frame time; default leaves text null).
    #[serde(default)]
    pub return_text_in_logprobs: Option<bool>,
    /// Any other fields on the request body, preserved for re-serialization.
    #[serde(flatten)]
    pub extra: BTreeMap<String, rmpv::Value>,
}

impl GeneratePayload {
    /// True when the client already supplied token ids → skip tokenization.
    pub fn already_tokenized(&self) -> bool {
        self.input_ids.as_ref().is_some_and(|v| !v.is_empty())
    }

    /// Multimodal detection hook. Deferred this iteration (Encoder stubbed):
    /// always false until mm fields are wired in. Unused for now.
    #[allow(dead_code)]
    pub fn has_multimodal(&self) -> bool {
        false
    }
}

/// One ingress-ring entry, split columnar: the scalar `header` (msgpack, with
/// `input_ids` omitted) plus the request's raw `ids` cell (little-endian int64,
/// empty for control requests). `recv_requests` concatenates the `ids` cells of
/// a drained batch into one buffer so the large tensor never goes through
/// msgpack; the scalar headers stay tiny.
#[derive(Debug)]
pub struct IngressMsg {
    pub header: Bytes,
    pub ids: Bytes,
}

/// Wire form of `TokenizedGenerateReqInput`.
///
/// The scheduler decodes this with msgspec, whose IPC structs are
/// `array_like=True, tag=True` — so the wire format is a **tagged msgpack
/// array** `[tag, ...fields in declaration order]`, NOT a named map. msgspec
/// fills trailing fields (all of which carry defaults) from a short array, so
/// we emit only through `stream` (index 11) and let it default the rest.
#[derive(Debug)]
pub struct TokenizedReqPayload {
    pub rid: String,
    pub input_text: Option<String>,
    pub input_ids: Vec<i32>,
    pub sampling_params: Option<rmpv::Value>,
    /// Scalar-normalized logprob options (defaults already applied at ingress).
    pub return_logprob: bool,
    pub logprob_start_len: i64,
    pub top_logprobs_num: i64,
    pub token_ids_logprob: Option<rmpv::Value>,
    pub return_hidden_states: bool,
    /// Health-check probe marker (scheduler skips it when busy).
    pub is_health_check: bool,
    pub stream: bool,
}

impl TokenizedReqPayload {
    /// Widen `input_ids` (i32) to the raw little-endian **int64** bytes the
    /// scheduler's `array("q")` expects. This is the *columnar tensor cell*: it
    /// travels the ingress ring as raw bytes (no msgpack) and is concatenated
    /// with the other requests' cells in `recv_requests`.
    pub fn input_ids_i64_le(&self) -> Bytes {
        let mut buf = Vec::with_capacity(self.input_ids.len() * 8);
        for &id in &self.input_ids {
            buf.extend_from_slice(&(id as i64).to_le_bytes());
        }
        Bytes::from(buf)
    }

    /// Serialize the *scalar header* to the msgspec-compatible tagged array,
    /// with `input_ids` left as `Nil` — the ids ride alongside as a raw columnar
    /// buffer (see [`input_ids_i64_le`](Self::input_ids_i64_le)) and are set on
    /// the decoded struct by the Python `drain`.
    pub fn to_header_msgpack(&self) -> Result<Bytes, Error> {
        use rmpv::Value;

        // input_ids omitted from the header; delivered as a columnar buffer.
        let input_ids_val = Value::Nil;

        let input_text_val = match &self.input_text {
            Some(t) => Value::from(t.as_str()),
            None => Value::Nil,
        };

        // `sampling_params: SamplingParams` is required (not Optional) and
        // map-encoded; default to an empty map (all-defaults) when absent. Send
        // only what the client set: the scheduler normalizes these and turns an
        // absent `stop` / `stop_regex` into an empty list. Injecting `""` here
        // instead would make `normalize` expand it to `[""]`, which matches at
        // every position and ends generation on the first token.
        let sampling_params_val = match self.sampling_params.clone() {
            Some(v @ Value::Map(_)) => v,
            _ => Value::Map(Vec::new()),
        };

        let token_ids_logprob_val = self.token_ids_logprob.clone().unwrap_or(Value::Nil);

        // Tagged array in TokenizedGenerateReqInput declaration order (BaseReq
        // fields first), truncated at `is_health_check`. msgspec requires the
        // array to be at least as long as the last non-defaulted field (`stream`,
        // index 13), so `input_embeds` and `token_type_ids` must be present even
        // though we always send them Nil.
        let arr = Value::Array(vec![
            Value::from("TokenizedGenerateReqInput"), // 0  tag
            Value::from(self.rid.as_str()),           // 1  rid
            Value::Nil,                               // 2  http_worker_ipc
            input_text_val,                           // 3  input_text
            input_ids_val,                            // 4  input_ids (columnar)
            Value::Nil,                               // 5  input_embeds
            Value::Nil,                               // 6  mm_inputs
            Value::Nil,                               // 7  token_type_ids
            sampling_params_val,                      // 8  sampling_params
            Value::from(self.return_logprob),         // 9  return_logprob
            Value::from(self.logprob_start_len),      // 10 logprob_start_len
            Value::from(self.top_logprobs_num),       // 11 top_logprobs_num
            token_ids_logprob_val,                    // 12 token_ids_logprob
            Value::from(self.stream),                 // 13 stream
            Value::from(self.return_hidden_states),   // 14 return_hidden_states
            Value::from(self.is_health_check),        // 15 is_health_check
        ]);

        let mut buf = Vec::new();
        rmpv::encode::write_value(&mut buf, &arr).map_err(|e| Error::Codec(e.to_string()))?;
        Ok(Bytes::from(buf))
    }
}

/// Egress-ring frame discriminator (first byte). Internal to the Rust egress
/// ring: Python pushes raw payloads via `push_batch` / `push_result` and the
/// tag is prepended on the Rust side, so the Python wire format is unchanged.
pub const EGRESS_TAG_RESULT: u8 = 1;
/// A whole decode batch in one frame: columnar scalars + numeric-column length
/// metadata (msgpack header) plus one concatenated raw buffer (token ids +
/// optional logprob/hidden columns). The tm-egress dispatcher decodes it into
/// per-request [`ChunkEvent`]s and routes each by rid — no per-request FFI /
/// msgpack from Python.
pub const EGRESS_TAG_BATCH: u8 = 2;
/// A per-request failure `[rid, message]`: the Python drain couldn't decode a
/// request's header (e.g. a malformed field), so instead of letting the error
/// escape the scheduler loop it routes a 400 back to the owning request.
pub const EGRESS_TAG_ERROR: u8 = 3;

/// Read `n` little-endian f32s from `data` at `*off`, advancing `*off`.
fn take_f32(data: &[u8], off: &mut usize, n: usize) -> Vec<f32> {
    let end = (*off + n * 4).min(data.len());
    let out = data[*off..end]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    *off = end;
    out
}

/// Read `n` little-endian i32s from `data` at `*off`, advancing `*off`.
fn take_i32(data: &[u8], off: &mut usize, n: usize) -> Vec<i32> {
    let end = (*off + n * 4).min(data.len());
    let out = data[*off..end]
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    *off = end;
    out
}

/// Frame a whole decode batch: `[EGRESS_TAG_BATCH][u32 header_len][header][data]`.
/// `header` is the msgpack [`BatchHeader`] (columnar scalars + lens); `data` is
/// the concatenated raw little-endian numeric buffer.
/// Frame a decode batch: `[BATCH tag][u32 header len][header][data...]`, where
/// `data` is the data *columns* (the caller's `data_cols`) concatenated directly
/// into the frame — one copy, instead of the caller first `b"".join`-ing them
/// into a single `bytes`. Run off the GIL (the big memcpy of the whole decode
/// batch doesn't need it).
pub fn frame_egress_batch_cols(header: &[u8], data_cols: &[&[u8]]) -> Bytes {
    let data_len: usize = data_cols.iter().map(|c| c.len()).sum();
    let mut buf = Vec::with_capacity(1 + 4 + header.len() + data_len);
    buf.push(EGRESS_TAG_BATCH);
    buf.extend_from_slice(&(header.len() as u32).to_le_bytes());
    buf.extend_from_slice(header);
    for col in data_cols {
        buf.extend_from_slice(col);
    }
    Bytes::from(buf)
}

/// Columnar scalar header for a whole decode batch. . All numeric fields are
/// `#[serde(default)]`, the hot path (no extras) emits just the first four.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct BatchHeader {
    /// Request ids as raw `u64` (a numeric column like `prompt_tokens`) — the
    /// rids originate from `RequestIdGen` (a u64 counter), so carrying them
    /// numerically avoids a per-request string encode/decode + parse.
    pub rids: Vec<u64>,
    pub finish_reasons: Vec<Option<serde_json::Value>>,
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
    pub out_tid_reqlens: Vec<u32>,
    #[serde(default)]
    pub out_tid_poslens: Vec<u32>,
    #[serde(default)]
    pub in_tid_reqlens: Vec<u32>,
    #[serde(default)]
    pub in_tid_poslens: Vec<u32>,
    #[serde(default)]
    pub hidden_reqlens: Vec<u32>,
    #[serde(default)]
    pub hidden_poslens: Vec<u32>,
}

/// Read request `i`'s flat logprob column (`l` parallel val/idx elements)
/// directly from the frame `data` at the per-column byte cursors `cv`/`ci`,
/// advancing them. No whole-column intermediate — only this request is copied.
fn take_flat(data: &[u8], cv: &mut usize, ci: &mut usize, l: usize) -> (Vec<f32>, Vec<i32>) {
    (take_f32(data, cv, l), take_i32(data, ci, l))
}

/// Read a request's ragged logprob column (`np` positions) from `data`: pull its
/// per-position `lens` from the header's `poslens` (advancing `pcur`), then the
/// summed val/idx count from the val/idx byte cursors `cv`/`ci`.
fn take_ragged(
    data: &[u8],
    cv: &mut usize,
    ci: &mut usize,
    poslens: &[u32],
    pcur: &mut usize,
    np: usize,
) -> (Vec<f32>, Vec<i32>, Vec<u32>) {
    let pe = (*pcur + np).min(poslens.len());
    let lens = poslens[(*pcur).min(pe)..pe].to_vec();
    *pcur = pe;
    let nv: usize = lens.iter().map(|&x| x as usize).sum();
    (take_f32(data, cv, nv), take_i32(data, ci, nv), lens)
}

/// Like [`take_ragged`] but for hidden states — a val column + row `poslens`, no
/// idx column.
fn take_hidden(
    data: &[u8],
    cv: &mut usize,
    poslens: &[u32],
    pcur: &mut usize,
    nr: usize,
) -> (Vec<f32>, Vec<u32>) {
    let pe = (*pcur + nr).min(poslens.len());
    let lens = poslens[(*pcur).min(pe)..pe].to_vec();
    *pcur = pe;
    let nv: usize = lens.iter().map(|&x| x as usize).sum();
    (take_f32(data, cv, nv), lens)
}

/// Decode a batch egress frame (tag stripped), invoking `route` with each
/// request's [`ChunkEvent`] **as it is decoded** — one pass, no intermediate
/// `Vec<ChunkEvent>` and no whole-column buffers. Each request reads its own
/// slice straight from `data` via per-column byte cursors, so peak extra memory
/// is one request (routing overlaps decode). Returns `false` on a malformed
/// frame (nothing routed). Column order must match Python's `push_generation`.
pub fn for_each_chunk(body: &[u8], mut route: impl FnMut(ChunkEvent)) -> bool {
    if body.len() < 4 {
        return false;
    }
    let hlen = u32::from_le_bytes([body[0], body[1], body[2], body[3]]) as usize;
    let Some(header) = body.get(4..4 + hlen) else {
        return false;
    };
    let data = &body[4 + hlen..];
    let Ok(h) = rmp_serde::from_slice::<BatchHeader>(header) else {
        return false;
    };

    let n = h.rids.len();
    let sum = |v: &[u32]| v.iter().map(|&x| x as usize).sum::<usize>();

    // Per-column byte cursors: each starts at that column's base offset in `data`
    // (columns are concatenated in this exact order, every element 4 bytes) and
    // advances per request. No whole-column read.
    let mut base = 0usize;
    let mut col = |count: usize| -> usize {
        let start = base;
        base += count * 4;
        start
    };
    let mut c_ids = col(sum(&h.tok_lens));
    let mut c_olp_v = col(sum(&h.out_lp_lens));
    let mut c_olp_i = col(sum(&h.out_lp_lens));
    let mut c_ilp_v = col(sum(&h.in_lp_lens));
    let mut c_ilp_i = col(sum(&h.in_lp_lens));
    let mut c_ot_v = col(sum(&h.out_top_poslens));
    let mut c_ot_i = col(sum(&h.out_top_poslens));
    let mut c_it_v = col(sum(&h.in_top_poslens));
    let mut c_it_i = col(sum(&h.in_top_poslens));
    let mut c_od_v = col(sum(&h.out_tid_poslens));
    let mut c_od_i = col(sum(&h.out_tid_poslens));
    let mut c_id_v = col(sum(&h.in_tid_poslens));
    let mut c_id_i = col(sum(&h.in_tid_poslens));
    let mut c_h_v = col(sum(&h.hidden_poslens));

    // Position cursors into the header's per-request `poslens` (ragged + hidden).
    let (mut p_ot, mut p_it, mut p_od, mut p_id, mut p_h) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    let lens_i = |v: &[u32], i: usize| v.get(i).copied().unwrap_or(0) as usize;

    for i in 0..n {
        let token_ids = take_i32(data, &mut c_ids, lens_i(&h.tok_lens, i));
        let (out_lp_val, out_lp_idx) =
            take_flat(data, &mut c_olp_v, &mut c_olp_i, lens_i(&h.out_lp_lens, i));
        let (in_lp_val, in_lp_idx) =
            take_flat(data, &mut c_ilp_v, &mut c_ilp_i, lens_i(&h.in_lp_lens, i));
        let (out_top_val, out_top_idx, out_top_lens) = take_ragged(
            data,
            &mut c_ot_v,
            &mut c_ot_i,
            &h.out_top_poslens,
            &mut p_ot,
            lens_i(&h.out_top_reqlens, i),
        );
        let (in_top_val, in_top_idx, in_top_lens) = take_ragged(
            data,
            &mut c_it_v,
            &mut c_it_i,
            &h.in_top_poslens,
            &mut p_it,
            lens_i(&h.in_top_reqlens, i),
        );
        let (out_tid_val, out_tid_idx, out_tid_lens) = take_ragged(
            data,
            &mut c_od_v,
            &mut c_od_i,
            &h.out_tid_poslens,
            &mut p_od,
            lens_i(&h.out_tid_reqlens, i),
        );
        let (in_tid_val, in_tid_idx, in_tid_lens) = take_ragged(
            data,
            &mut c_id_v,
            &mut c_id_i,
            &h.in_tid_poslens,
            &mut p_id,
            lens_i(&h.in_tid_reqlens, i),
        );
        let (hidden_val, hidden_lens) = take_hidden(
            data,
            &mut c_h_v,
            &h.hidden_poslens,
            &mut p_h,
            lens_i(&h.hidden_reqlens, i),
        );

        route(ChunkEvent {
            rid: h.rids[i],
            token_ids,
            finish_reason: h.finish_reasons.get(i).cloned().flatten(),
            prompt_tokens: h.prompt_tokens.get(i).copied().unwrap_or(0),
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
            ..Default::default()
        });
    }
    true
}

/// Frame a control result `[rid, payload]` for the egress ring (tag prepended).
pub fn frame_egress_result(rid: &str, payload: &[u8]) -> Bytes {
    use rmpv::Value;
    let arr = Value::Array(vec![Value::from(rid), Value::Binary(payload.to_vec())]);
    let mut buf = Vec::with_capacity(1 + payload.len() + rid.len() + 8);
    buf.push(EGRESS_TAG_RESULT);
    let _ = rmpv::encode::write_value(&mut buf, &arr);
    Bytes::from(buf)
}

/// Frame a per-request failure `[rid, message]` for the egress ring — routes a
/// terminal error back to the owning request (→ HTTP 400) instead of crashing.
pub fn frame_egress_error(rid: &str, message: &str) -> Bytes {
    use rmpv::Value;
    let arr = Value::Array(vec![Value::from(rid), Value::from(message)]);
    let mut buf = Vec::with_capacity(1 + rid.len() + message.len() + 8);
    buf.push(EGRESS_TAG_ERROR);
    let _ = rmpv::encode::write_value(&mut buf, &arr);
    Bytes::from(buf)
}

/// One scheduler output increment for a request, pushed from Python via
/// `push_chunk` into the egress ring. Decoded on a Rust detok shard.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ChunkEvent {
    /// Request id as raw `u64` — the shard routing key; no parse, no clone.
    pub rid: u64,
    pub seq: u64,
    /// New token ids for this step. Empty allowed (e.g. metadata-only frames).
    pub token_ids: Vec<i32>,
    /// `None` while streaming, the full finish-reason dict on the final chunk.
    pub finish_reason: Option<serde_json::Value>,
    /// Prompt token count for this request (constant across its chunks).
    /// `#[serde(default)]` keeps the wire backward-compatible with 4-field frames.
    #[serde(default)]
    pub prompt_tokens: u32,
    /// Output-token logprobs for this step (columnar: parallel `val`/`idx`
    /// buffers, one entry per new output token). Empty unless `return_logprob`.
    #[serde(default)]
    pub out_lp_val: Vec<f32>,
    #[serde(default)]
    pub out_lp_idx: Vec<i32>,
    /// Input (prefill) token logprobs, sent once on the first chunk. Empty
    /// otherwise.
    #[serde(default)]
    pub in_lp_val: Vec<f32>,
    #[serde(default)]
    pub in_lp_idx: Vec<i32>,
    /// Top-k logprobs (2-level ragged): flat `val`/`idx` buffers plus a
    /// per-position `lens` (top-k count at each position; 0 = null position).
    /// Output = per-step delta; input = once on the first chunk. Empty unless
    /// `top_logprobs_num > 0`.
    #[serde(default)]
    pub out_top_val: Vec<f32>,
    #[serde(default)]
    pub out_top_idx: Vec<i32>,
    #[serde(default)]
    pub out_top_lens: Vec<u32>,
    #[serde(default)]
    pub in_top_val: Vec<f32>,
    #[serde(default)]
    pub in_top_idx: Vec<i32>,
    #[serde(default)]
    pub in_top_lens: Vec<u32>,
    /// Token-ids logprobs (same 2-level ragged layout). Empty unless
    /// `token_ids_logprob` was set on the request.
    #[serde(default)]
    pub out_tid_val: Vec<f32>,
    #[serde(default)]
    pub out_tid_idx: Vec<i32>,
    #[serde(default)]
    pub out_tid_lens: Vec<u32>,
    #[serde(default)]
    pub in_tid_val: Vec<f32>,
    #[serde(default)]
    pub in_tid_idx: Vec<i32>,
    #[serde(default)]
    pub in_tid_lens: Vec<u32>,
    /// Hidden states (dense f32): flat buffer + per-row lengths (one row per
    /// output position). Last-writer-wins across chunks (the final message
    /// carries the full set). Empty unless `return_hidden_states`.
    #[serde(default)]
    pub hidden_val: Vec<f32>,
    #[serde(default)]
    pub hidden_lens: Vec<u32>,
}

#[cfg(test)]
mod chunk_event_tests {
    use super::*;

    /// Concatenating N data columns produces the exact same frame as one joined
    /// buffer (the `b"".join` the Python side used to do), with the layout
    /// `[tag][u32 header len][header][col0 col1 …]`.
    #[test]
    fn batch_cols_match_single_joined_buffer() {
        let header = [1u8, 2, 3];
        let a = [10u8, 11];
        let b = [12u8, 13, 14];
        let multi = frame_egress_batch_cols(&header, &[&a[..], &b[..]]);
        let joined: Vec<u8> = a.iter().chain(&b).copied().collect();
        let single = frame_egress_batch_cols(&header, &[joined.as_slice()]);
        assert_eq!(multi, single);
        assert_eq!(multi[0], EGRESS_TAG_BATCH);
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
            Value::Array(vec![
                Value::from(1u64),
                Value::from(2u64),
                Value::from(3u64),
            ]),
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

        let framed = frame_egress_batch_cols(&header, &[&data]);
        assert_eq!(framed[0], EGRESS_TAG_BATCH);
        let mut events = Vec::new();
        assert!(for_each_chunk(&framed[1..], |ev| events.push(ev)));
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].rid, 1);
        assert_eq!(events[0].token_ids, vec![10, 11]);
        assert_eq!(events[0].prompt_tokens, 4);
        assert!(events[0].finish_reason.is_none());
        assert_eq!(events[1].rid, 2);
        assert!(events[1].token_ids.is_empty());
        // The whole dict survives (type + matched), not just the type.
        assert_eq!(
            events[1].finish_reason,
            Some(serde_json::json!({ "type": "stop", "matched": 5 }))
        );
        assert_eq!(events[2].rid, 3);
        assert_eq!(events[2].token_ids, vec![12]);
        assert_eq!(events[2].prompt_tokens, 6);
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
        //   out_top_reqlens, out_top_poslens, in_top_*, out_tid_*, in_tid_*,
        //   hidden_reqlens, hidden_poslens
        let header_arr = Value::Array(vec![
            Value::Array(vec![Value::from(1u64), Value::from(2u64)]), // rids
            Value::Array(vec![Value::Nil, Value::Nil]),               // finish
            arr_u(&[3, 4]),                                           // prompt
            arr_u(&[1, 1]),                                           // tok_lens
            arr_u(&[2, 0]),                                           // out_lp_lens
            arr_u(&[0, 0]),                                           // in_lp_lens
            arr_u(&[1, 0]), // out_top_reqlens (req0: 1 pos)
            arr_u(&[2]),    // out_top_poslens (that pos: k=2)
            arr_u(&[0, 0]), // in_top_reqlens
            arr_u(&[]),     // in_top_poslens
            arr_u(&[0, 0]), // out_tid_reqlens
            arr_u(&[]),     // out_tid_poslens
            arr_u(&[0, 0]), // in_tid_reqlens
            arr_u(&[]),     // in_tid_poslens
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

        let framed = frame_egress_batch_cols(&header, &[&data]);
        let mut events = Vec::new();
        assert!(for_each_chunk(&framed[1..], |ev| events.push(ev)));
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].token_ids, vec![10]);
        assert_eq!(events[0].out_lp_val, vec![-0.5, -0.6]);
        assert_eq!(events[0].out_lp_idx, vec![10, 99]);
        assert_eq!(events[0].out_top_val, vec![-0.1, -0.2]);
        assert_eq!(events[0].out_top_lens, vec![2]);
        assert_eq!(events[0].hidden_val, vec![0.1, 0.2, 0.3]);
        assert_eq!(events[0].hidden_lens, vec![3]);
        // req1 has token id but no numeric columns.
        assert_eq!(events[1].token_ids, vec![20]);
        assert!(events[1].out_lp_val.is_empty() && events[1].hidden_val.is_empty());
    }
}

#[cfg(test)]
mod abort_tests {
    use super::*;

    #[test]
    fn abort_req_msgpack_shape() {
        let b = abort_req_msgpack("12345").unwrap();
        let val = rmpv::decode::read_value(&mut &b[..]).unwrap();
        let arr = val.as_array().expect("array");
        assert_eq!(
            arr.len(),
            6,
            "AbortReq = [tag, rid, http_ipc, abort_all, finished_reason, abort_message]"
        );
        assert_eq!(arr[0].as_str(), Some("AbortReq"));
        assert_eq!(arr[1].as_str(), Some("12345"));
        assert!(arr[2].is_nil());
        assert_eq!(arr[3].as_bool(), Some(false));
        assert!(arr[4].is_nil());
        assert!(arr[5].is_nil());
    }
}

#[cfg(test)]
mod ingress_header_tests {
    use super::*;

    /// The ingress header must carry ALL fields the scheduler's msgspec struct
    /// `TokenizedGenerateReqInput` requires positionally through `stream` — i.e.
    /// `input_embeds` (idx 5) and `token_type_ids` (idx 7) must be present, so
    /// `sampling_params` lands at idx 8. Omitting them makes the array too short
    /// (msgspec: "Expected array of at least length 14") and misaligns
    /// `sampling_params`. Regression guard for that exact decode failure.
    #[test]
    fn to_header_msgpack_is_positionally_aligned() {
        let payload = TokenizedReqPayload {
            rid: "r1".into(),
            input_text: Some("hi".into()),
            input_ids: vec![1, 2, 3],
            sampling_params: Some(rmpv::Value::Map(vec![(
                rmpv::Value::from("max_new_tokens"),
                rmpv::Value::from(5),
            )])),
            return_logprob: true,
            logprob_start_len: -1,
            top_logprobs_num: 3,
            token_ids_logprob: None,
            return_hidden_states: false,
            is_health_check: true,
            stream: true,
        };
        let bytes = payload.to_header_msgpack().unwrap();
        let val = rmpv::decode::read_value(&mut &bytes[..]).unwrap();
        let arr = val.as_array().expect("array");
        // msgspec requires >= 14 (through `stream`); we emit 16.
        assert!(
            arr.len() >= 14,
            "header must have >=14 elements, got {}",
            arr.len()
        );
        assert_eq!(arr[0].as_str(), Some("TokenizedGenerateReqInput"));
        assert_eq!(arr[1].as_str(), Some("r1"));
        assert!(arr[5].is_nil(), "idx 5 must be input_embeds (nil)");
        assert!(arr[7].is_nil(), "idx 7 must be token_type_ids (nil)");
        assert!(arr[8].is_map(), "sampling_params must land at idx 8");
        assert_eq!(arr[9].as_bool(), Some(true), "return_logprob at idx 9");
        assert_eq!(arr[11].as_u64(), Some(3), "top_logprobs_num at idx 11");
        assert_eq!(arr[13].as_bool(), Some(true), "stream at idx 13");
        assert_eq!(arr[15].as_bool(), Some(true), "is_health_check at idx 15");
    }
}
