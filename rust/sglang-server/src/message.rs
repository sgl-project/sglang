//! Messages moved between stages via `flume` (zero-copy moves); variable-length
//! buffers are `bytes::Bytes`, so egress fan-out to detok shards is a refcount bump.

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::error::Error;
use crate::fsm::RequestState;
use crate::ids::RidHash;

/// Per-request back-channel the detok shard writes egress frames to and the API
/// handler drains for SSE; bounded, and receiver-drop (disconnect) = stream end.
#[derive(Clone, Debug)]
pub enum EgressSink {
    Local(mpsc::Sender<EgressItem>),
}

/// Why an [`EgressSink::try_send`] failed: `Full` = client backpressure, `Closed`
/// = client gone. Both terminal for a stream; the caller distinguishes for logging.
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

/// What the connection handler receives on the egress stream: a detok-decoded
/// [`ChunkEvent`] (handler formats it), a verbatim control payload, or an error.
#[derive(Debug)]
pub enum EgressItem {
    /// An intermediate streamed generation step (only sent for streaming reqs).
    Frame(ChunkEvent),
    /// The final generation step.
    Done(ChunkEvent),
    /// A control-request result: one verbatim payload (e.g. `/server_info`),
    /// delivered as-is with no per-protocol formatting.
    Control(Bytes),
    /// Terminal failure: handler emits an error frame (stream) or status (unary).
    Error(Error),
}

/// Request variant — selects the ingress branch, scheduler wire message, and
/// egress shape. Each owns its body, so generate/control fields stay type-separate.
#[derive(Debug)]
pub enum RequestKind {
    /// `/generate`: tokenize (if needed) then push a `TokenizedGenerateReqInput`.
    Generate(GenerateRequest),
    /// A control endpoint (e.g. `/server_info`, `/health`): no tokenization, and
    /// the egress is a single non-streamed JSON result.
    Control(ControlRequest),
}

/// A single in-flight `/generate` request (per-item from [`GenerateBody::split`]),
/// serialized to the scheduler wire once tokenized (see `to_header_msgpack`). Not a
/// wire type — built by `split`/handlers, never (de)serialized; `input_ids` is
/// client-supplied or filled by the Tokenizer stage.
#[derive(Debug, Default)]
pub struct GenerateRequest {
    /// Client-requested rid for this item (`None` → the server mints a uuid).
    /// Duplicate in-flight rids collide on the same `RidHash` slot, orphaning
    /// the earlier request — same garbage-in behavior as the Python server's
    /// `rid_to_state` overwrite.
    pub rid: Option<String>,
    pub text: Option<String>,
    /// Client-supplied token ids, or filled by the Tokenizer stage.
    pub input_ids: Option<Vec<i32>>,
    /// Opaque sampling params, normalized in place at ingress then carried through.
    pub sampling_params: Option<rmpv::Value>,
    /// Whether the client asked for SSE streaming.
    pub stream: bool,
    /// Internal `/health_generate` probe: the scheduler skips it when busy so it
    /// never occupies a waiting-queue slot. Never set from the client wire.
    pub is_health_check: bool,
    /// Logprob / hidden-state options. This path bypasses the Python
    /// `TokenizerManager`, so the ingress replicates its scalar normalization
    /// (defaults applied in `to_header_msgpack`) before the scheduler sees them.
    pub return_logprob: Option<bool>,
    pub logprob_start_len: Option<i64>,
    pub top_logprobs_num: Option<i64>,
    pub token_ids_logprob: Option<rmpv::Value>,
    pub return_hidden_states: Option<bool>,
    /// Decode logprob token ids to text in each `[logprob, token_id, text]` tuple
    /// (the api-server does this at frame time; default leaves text null).
    pub return_text_in_logprobs: Option<bool>,
}

impl GenerateRequest {
    /// True when the client already supplied token ids → skip tokenization.
    pub fn already_tokenized(&self) -> bool {
        self.input_ids.as_ref().is_some_and(|v| !v.is_empty())
    }

    /// Multimodal detection hook. Deferred (Encoder stubbed): always false until mm
    /// fields are wired in.
    #[allow(dead_code)]
    pub fn has_multimodal(&self) -> bool {
        false
    }

    /// `input_ids` widened to raw little-endian int64 bytes (the scheduler's
    /// `array("q")` columnar cell — rides the ingress ring outside msgpack). Empty
    /// when not tokenized.
    pub fn input_ids_i64_le(&self) -> Bytes {
        let ids = self.input_ids.as_deref().unwrap_or(&[]);
        let mut buf = Vec::with_capacity(ids.len() * 8);
        for &id in ids {
            buf.extend_from_slice(&(id as i64).to_le_bytes());
        }
        Bytes::from(buf)
    }

    /// Serialize the scalar header as the scheduler's `TokenizedGenerateReqInput`
    /// positional tagged msgpack array, resolving `Option` scalars to wire defaults.
    /// `input_ids` is `Nil` (rides columnar via `input_ids_i64_le`); idx 5/7 stay
    /// `Nil` so the array reaches the last non-defaulted field (`stream`, idx 13).
    pub fn to_header_msgpack(&self, rid: &str) -> Result<Bytes, Error> {
        use rmpv::Value;

        let input_text_val = match &self.text {
            Some(t) => Value::from(t.as_str()),
            None => Value::Nil,
        };
        // `sampling_params` is required + map-encoded; empty map when absent (send
        // only what the client set — injecting `""` would make the scheduler's
        // normalize expand it to `[""]`, stopping on the first token).
        let sampling_params_val = match &self.sampling_params {
            Some(v @ Value::Map(_)) => v.clone(),
            _ => Value::Map(Vec::new()),
        };
        let token_ids_logprob_val = self.token_ids_logprob.clone().unwrap_or(Value::Nil);

        let arr = Value::Array(vec![
            Value::from("TokenizedGenerateReqInput"),          // 0  tag
            Value::from(rid),                                  // 1  rid
            Value::Nil,                                        // 2  http_worker_ipc
            input_text_val,                                    // 3  input_text
            Value::Nil,                                        // 4  input_ids (columnar)
            Value::Nil,                                        // 5  input_embeds
            Value::Nil,                                        // 6  mm_inputs
            Value::Nil,                                        // 7  token_type_ids
            sampling_params_val,                               // 8  sampling_params
            Value::from(self.return_logprob.unwrap_or(false)), // 9  return_logprob
            Value::from(self.logprob_start_len.unwrap_or(-1)), // 10 logprob_start_len
            Value::from(self.top_logprobs_num.unwrap_or(0)),   // 11 top_logprobs_num
            token_ids_logprob_val,                             // 12 token_ids_logprob
            Value::from(self.stream),                          // 13 stream
            Value::from(self.return_hidden_states.unwrap_or(false)), // 14 return_hidden_states
            Value::from(self.is_health_check),                 // 15 is_health_check
        ]);

        let mut buf = Vec::new();
        rmpv::encode::write_value(&mut buf, &arr).map_err(|e| Error::Codec(e.to_string()))?;
        Ok(Bytes::from(buf))
    }
}

/// Body of a control request. `tag` = the scheduler request-struct name (e.g.
/// `"GetInternalStateReq"`), pushed as a bare `[tag, rid, nil]`.
#[derive(Debug)]
pub struct ControlRequest {
    pub tag: &'static str,
}

/// The owned request as it travels ingress stages (single owner, so `state` is
/// mutated lock-free). Common fields here; variant data in [`RequestKind`].
#[derive(Debug)]
pub struct Request {
    /// Routing key: `RidHash::from_rid(&rid)`.
    pub rid_hash: RidHash,
    /// Client-visible request id (uuid hex) — what the scheduler wire and
    /// `meta_info.id` carry.
    pub rid: String,
    pub state: RequestState,
    /// Back-channel to the client connection for egress frames.
    pub sink: EgressSink,
    /// Discriminant + variant body (generate vs control).
    pub kind: RequestKind,
}

/// Encode a bare `BaseReq` control message as the msgspec tagged array
/// `[tag, rid, nil]` (e.g. `GetInternalStateReq`; no extra fields).
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

/// Encode `AbortReq(rid)` as its msgspec tagged array
/// `["AbortReq", rid, nil, false, nil, nil]`; the scheduler stops generation for `rid`.
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

/// A field accepting a scalar or a list: deserializes a bare `T` **or** `[T,…]` —
/// so `/generate` takes `text: "hi"` or `text: ["a","b"]` through one body type.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum OneOrMany<T> {
    One(T),
    Many(Vec<T>),
}

/// The `/generate` wire body before batch splitting: `text`/`input_ids`/`sampling_params`
/// each scalar-or-list, fanned into per-request [`GenerateRequest`]s by
/// [`split`](GenerateBody::split). `deny_unknown_fields` rejects (4xx) unknowns.
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
    pub input_ids: Option<OneOrMany<Vec<i32>>>,
    #[serde(default)]
    pub stream: bool,
    /// A single params map (broadcast) or a list of maps (per item). Raw `Value`,
    /// not `OneOrMany` — `rmpv::Value` matches a JSON array, so `split` decides
    /// map-vs-array at fan-out time.
    #[serde(default)]
    pub sampling_params: Option<rmpv::Value>,
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
    #[serde(default)]
    pub return_text_in_logprobs: Option<bool>,

    // Accepted for wire-compat with the native `bench_serving` payload (a full
    // `GenerateReqInput`) but NOT yet wired into the scheduler — parsed and
    // dropped in `split`. Declaring them keeps `deny_unknown_fields` (typos still
    // 400) while letting a benchmark request through. Permissive `Value` types so
    // any valid shape (str / list / list-of-lists) parses. `dead_code`-allowed:
    // deserialized then intentionally ignored.
    #[serde(default)]
    #[allow(dead_code)]
    pub lora_path: Option<rmpv::Value>,
    #[serde(default)]
    #[allow(dead_code)]
    pub return_routed_experts: Option<bool>,
    #[serde(default)]
    #[allow(dead_code)]
    pub image_data: Option<rmpv::Value>,
}

impl GenerateBody {
    /// Fan the body into one [`GenerateRequest`] per prompt + `is_batch` (list form
    /// — a 1-element list is still a batch → JSON array response). `Err` (→ 400) on
    /// an invalid/inconsistent batch.
    pub fn split(self) -> Result<(Vec<GenerateRequest>, bool), String> {
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
            // Accepted for bench_serving compat, not wired through — see the struct.
            ..
        } = self;

        // Per-item (text, input_ids) columns + whether the input used list form.
        type Columns = (Vec<Option<String>>, Vec<Option<Vec<i32>>>, bool);
        // Exactly one of text / input_ids, like the Python `_validate_inputs`.
        let (mut texts, mut id_lists, is_batch): Columns = match (text, input_ids) {
            (Some(_), Some(_)) => {
                return Err("provide either `text` or `input_ids`, not both".into());
            }
            (None, None) => return Err("either `text` or `input_ids` must be provided".into()),
            (Some(OneOrMany::One(s)), None) => (vec![Some(s)], vec![None], false),
            (Some(OneOrMany::Many(v)), None) => {
                let n = v.len();
                (v.into_iter().map(Some).collect(), vec![None; n], true)
            }
            (None, Some(OneOrMany::One(x))) => (vec![None], vec![Some(x)], false),
            (None, Some(OneOrMany::Many(vv))) => {
                let n = vv.len();
                (vec![None; n], vv.into_iter().map(Some).collect(), true)
            }
        };
        let n = texts.len();
        if n == 0 {
            return Err("batch must contain at least one item".into());
        }

        // sampling_params: an array is per-item; anything else (a map) broadcasts.
        let mut sps: Vec<Option<rmpv::Value>> = match sampling_params {
            None => vec![None; n],
            Some(rmpv::Value::Array(v)) => {
                if v.len() != n {
                    return Err(format!(
                        "sampling_params list length {} does not match batch size {n}",
                        v.len()
                    ));
                }
                v.into_iter().map(Some).collect()
            }
            Some(sp) => vec![Some(sp); n],
        };

        // rid: absent → mint per item at submit; a single string fans out as
        // `{rid}_{i}` for a batch (Python `_normalize_batch`); a list is per-item.
        let mut rids: Vec<Option<String>> = match rid {
            None => vec![None; n],
            Some(OneOrMany::One(r)) if !is_batch => vec![Some(r)],
            Some(OneOrMany::One(r)) => (0..n).map(|i| Some(format!("{r}_{i}"))).collect(),
            Some(OneOrMany::Many(v)) => {
                if !is_batch || v.len() != n {
                    return Err(format!(
                        "rid list length {} does not match batch size {n}",
                        v.len()
                    ));
                }
                v.into_iter().map(Some).collect()
            }
        };

        // The scalar logprob/hidden opts broadcast to every item. `is_health_check`
        // is never client-set (only the internal `/health_generate` probe sets it).
        let requests = (0..n)
            .map(|i| GenerateRequest {
                rid: rids[i].take(),
                text: texts[i].take(),
                input_ids: id_lists[i].take(),
                sampling_params: sps[i].take(),
                stream,
                is_health_check: false,
                return_logprob,
                logprob_start_len,
                top_logprobs_num,
                token_ids_logprob: token_ids_logprob.clone(),
                return_hidden_states,
                return_text_in_logprobs,
            })
            .collect();
        Ok((requests, is_batch))
    }
}

/// One ingress-ring entry, split columnar: the scalar `header` (msgpack, `input_ids`
/// omitted) + the raw int64 `ids` cell, so the big tensor never goes through msgpack.
#[derive(Debug)]
pub struct IngressMsg {
    pub header: Bytes,
    pub ids: Bytes,
}

/// Egress-ring frame tag (first byte, prepended Rust-side; Python wire unchanged):
/// a single control-request result payload.
pub const EGRESS_TAG_RESULT: u8 = 1;
/// A whole decode batch: msgpack columnar header + one concatenated raw buffer;
/// tm-egress decodes it into per-request [`ChunkEvent`]s (no per-request FFI).
pub const EGRESS_TAG_BATCH: u8 = 2;
/// A per-request failure `[rid, message]`: the Python drain couldn't decode a
/// header, so it routes a 400 back to that request instead of crashing the loop.
pub const EGRESS_TAG_ERROR: u8 = 3;

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
/// `b"".join`); `header` is the msgpack [`BatchHeader`]. Runs off the GIL.
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

/// Columnar scalar header for a whole decode batch. All numeric fields are
/// `#[serde(default)]`; the hot path (no extras) emits just the first four.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct BatchHeader {
    /// Request ids, as the same strings Python holds (`Req.rid`, uuid hex) —
    /// hashed back to the internal routing key in `decode_one`
    /// (`RidHash::from_rid`), mirroring the control path. The wire has no
    /// rid-shape coupling; any string is a valid rid.
    pub rids: Vec<String>,
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

/// Read a request's ragged logprob column (`np` positions): its per-position `lens`
/// from `poslens` (advancing `pcur`), then that many val/idx from `cv`/`ci`. `None`
/// if the val/idx read runs past the buffer (see [`take_f32`]).
fn take_ragged(
    data: &[u8],
    cv: &mut usize,
    ci: &mut usize,
    poslens: &[u32],
    pcur: &mut usize,
    np: usize,
) -> Option<(Vec<f32>, Vec<i32>, Vec<u32>)> {
    let pe = (*pcur + np).min(poslens.len());
    let lens = poslens[(*pcur).min(pe)..pe].to_vec();
    *pcur = pe;
    let nv: usize = lens.iter().map(|&x| x as usize).sum();
    Some((take_f32(data, cv, nv)?, take_i32(data, ci, nv)?, lens))
}

/// Like [`take_ragged`] but for hidden states — a val column + row `poslens`, no
/// idx column. `None` if the val read runs past the buffer (see [`take_f32`]).
fn take_hidden(
    data: &[u8],
    cv: &mut usize,
    poslens: &[u32],
    pcur: &mut usize,
    nr: usize,
) -> Option<(Vec<f32>, Vec<u32>)> {
    let pe = (*pcur + nr).min(poslens.len());
    let lens = poslens[(*pcur).min(pe)..pe].to_vec();
    *pcur = pe;
    let nv: usize = lens.iter().map(|&x| x as usize).sum();
    Some((take_f32(data, cv, nv)?, lens))
}

/// Decode a batch egress frame (tag stripped), calling `route` with each request's
/// [`ChunkEvent`] as it's decoded — one pass, no intermediate `Vec`, peak memory
/// one request. `false` on a malformed frame. Column order matches `push_generation`.
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

    // Per-column byte cursors, advanced per request — no whole-column read. Columns
    // are concatenated in exactly this order, every element 4 bytes.
    let mut base = 0usize;
    let mut col = |count: usize| -> usize {
        let start = base;
        base += count * 4;
        start
    };
    // Each val/idx column pair shares one element count — sum it once.
    let n_ids = sum(&h.tok_lens);
    let n_olp = sum(&h.out_lp_lens);
    let n_ilp = sum(&h.in_lp_lens);
    let n_ot = sum(&h.out_top_poslens);
    let n_it = sum(&h.in_top_poslens);
    let n_od = sum(&h.out_tid_poslens);
    let n_id = sum(&h.in_tid_poslens);
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

    // `col` summed every column's span into `base`. Reject a malformed frame whole,
    // *before* routing any request: a partial fan-out would deliver garbage.
    if base > data.len() {
        return false;
    }

    // Mirror of Python's `has_extra` guard: checking once per frame lets the
    // per-request loop skip the extras machinery entirely on a plain decode frame.
    let has_extras = !(h.out_lp_lens.is_empty()
        && h.in_lp_lens.is_empty()
        && h.out_top_reqlens.is_empty()
        && h.in_top_reqlens.is_empty()
        && h.out_tid_reqlens.is_empty()
        && h.in_tid_reqlens.is_empty()
        && h.hidden_reqlens.is_empty());

    // Position cursors into the header's per-request `poslens` (ragged + hidden).
    let (mut p_ot, mut p_it, mut p_od, mut p_id, mut p_h) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    let lens_i = |v: &[u32], i: usize| v.get(i).copied().unwrap_or(0) as usize;

    // Decode one request's slice of every column, advancing the cursors. `None` if a
    // read overruns `data` (belt-and-suspenders past the upfront check) — the caller
    // then rejects the frame instead of slicing out of bounds.
    let mut decode_one = |i: usize| -> Option<ChunkEvent> {
        let token_ids = take_i32(data, &mut c_ids, lens_i(&h.tok_lens, i))?;

        // Plain decode frame (no request in the batch asked for logprobs/hidden):
        // the extras columns are all zero-width, so skip reading them entirely.
        let extras = if !has_extras {
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
                &h.out_tid_poslens,
                &mut p_od,
                lens_i(&h.out_tid_reqlens, i),
            )?;
            let (in_tid_val, in_tid_idx, in_tid_lens) = take_ragged(
                data,
                &mut c_id_v,
                &mut c_id_i,
                &h.in_tid_poslens,
                &mut p_id,
                lens_i(&h.in_tid_reqlens, i),
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
                ..Default::default()
            };
            (!ex.is_empty()).then(|| Box::new(ex))
        };

        Some(ChunkEvent {
            // Any string is a valid rid; hash to the routing key. An unknown
            // rid routes to a shard whose table has no entry → dropped there.
            rid_hash: RidHash::from_rid(&h.rids[i]).0,
            token_ids,
            finish_reason: h.finish_reasons.get(i).cloned().flatten(),
            prompt_tokens: h.prompt_tokens.get(i).copied().unwrap_or(0),
            extras,
            ..Default::default()
        })
    };

    for i in 0..n {
        let Some(ev) = decode_one(i) else {
            return false;
        };
        route(ev);
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
    /// `RidHash` digest of the rid — the shard routing key; `Copy`, no clone.
    pub rid_hash: u64,
    /// New token ids for this step. Empty allowed (e.g. metadata-only frames).
    pub token_ids: Vec<i32>,
    /// `None` while streaming, the full finish-reason dict on the final chunk.
    pub finish_reason: Option<serde_json::Value>,
    /// Prompt token count for this request (constant across its chunks).
    pub prompt_tokens: u32,
    /// Decoded text **delta** for this chunk (empty in skip mode / on partial UTF-8),
    /// filled by the detok shard. `token_ids` doubles as `output_ids`;
    /// `completion_tokens` is this chunk's count.
    pub text: String,
    pub completion_tokens: u64,
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
    /// True when no logprob / hidden column carries data — lets the decoder skip the
    /// box allocation for the common (extras-free) frame.
    fn is_empty(&self) -> bool {
        self.out_lp_val.is_empty()
            && self.in_lp_val.is_empty()
            && self.out_top_lens.is_empty()
            && self.in_top_lens.is_empty()
            && self.out_tid_lens.is_empty()
            && self.in_tid_lens.is_empty()
            && self.hidden_lens.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(body: &str) -> Result<(Vec<GenerateRequest>, bool), String> {
        serde_json::from_str::<GenerateBody>(body).unwrap().split()
    }

    /// Scalar `text` → one item, not a batch (response stays a single object).
    #[test]
    fn scalar_text_is_single() {
        let (ps, is_batch) = split(r#"{"text": "hi"}"#).unwrap();
        assert!(!is_batch);
        assert_eq!(ps.len(), 1);
        assert_eq!(ps[0].text.as_deref(), Some("hi"));
    }

    /// List `text` → batch (even length 1); each prompt becomes its own payload.
    #[test]
    fn list_text_is_batch() {
        let (ps, is_batch) = split(r#"{"text": ["a", "b"]}"#).unwrap();
        assert!(is_batch);
        assert_eq!(ps.len(), 2);
        assert_eq!(ps[0].text.as_deref(), Some("a"));
        assert_eq!(ps[1].text.as_deref(), Some("b"));

        let (ps, is_batch) = split(r#"{"text": ["only"]}"#).unwrap();
        assert!(is_batch, "single-element list is still a batch");
        assert_eq!(ps.len(), 1);
    }

    /// Scalar `sampling_params` broadcasts to every item; a list maps per item.
    #[test]
    fn sampling_params_broadcast_and_per_item() {
        let (ps, _) =
            split(r#"{"text": ["a", "b"], "sampling_params": {"temperature": 0.5}}"#).unwrap();
        assert_eq!(ps[0].sampling_params, ps[1].sampling_params);
        assert!(ps[0].sampling_params.is_some());

        let (ps, _) = split(
            r#"{"text": ["a", "b"], "sampling_params": [{"temperature": 0.1}, {"temperature": 0.9}]}"#,
        )
        .unwrap();
        assert_ne!(ps[0].sampling_params, ps[1].sampling_params);
    }

    /// A per-item `sampling_params` list whose length ≠ batch size is a 400.
    #[test]
    fn sampling_params_length_mismatch_errors() {
        let err = split(r#"{"text": ["a", "b"], "sampling_params": [{}]}"#).unwrap_err();
        assert!(err.contains("length"), "{err}");
    }

    /// `input_ids` batch (list of lists) fans out; scalar (list of ints) is single.
    #[test]
    fn input_ids_scalar_vs_batch() {
        let (ps, is_batch) = split(r#"{"input_ids": [1, 2, 3]}"#).unwrap();
        assert!(!is_batch);
        assert_eq!(ps[0].input_ids, Some(vec![1, 2, 3]));

        let (ps, is_batch) = split(r#"{"input_ids": [[1, 2], [3]]}"#).unwrap();
        assert!(is_batch);
        assert_eq!(ps.len(), 2);
        assert_eq!(ps[1].input_ids, Some(vec![3]));
    }

    /// Both / neither of text+input_ids is a 400; the wire still rejects unknowns.
    #[test]
    fn split_validates_inputs() {
        assert!(split(r#"{"text": "a", "input_ids": [1]}"#).is_err());
        assert!(split(r#"{"stream": true}"#).is_err());
        assert!(
            serde_json::from_str::<GenerateBody>(r#"{"text": "hi", "bogus": 1}"#).is_err(),
            "GenerateBody must deny unknown fields"
        );
    }

    /// Client-supplied rid semantics mirror Python's `_normalize_batch`: a
    /// single string passes through for a single request, fans out as
    /// `{rid}_{i}` for a batch, a list must match the batch length, and absent
    /// rid leaves every slot `None` (server mints uuids at submit).
    #[test]
    fn split_rid_matches_python_normalize() {
        let (ps, _) = split(r#"{"text": "a", "rid": "r"}"#).unwrap();
        assert_eq!(ps[0].rid.as_deref(), Some("r"));

        let (ps, _) = split(r#"{"text": ["a", "b"], "rid": "base"}"#).unwrap();
        assert_eq!(ps[0].rid.as_deref(), Some("base_0"));
        assert_eq!(ps[1].rid.as_deref(), Some("base_1"));

        let (ps, _) = split(r#"{"text": ["a", "b"], "rid": ["x", "y"]}"#).unwrap();
        assert_eq!(ps[0].rid.as_deref(), Some("x"));
        assert_eq!(ps[1].rid.as_deref(), Some("y"));

        let (ps, _) = split(r#"{"text": ["a", "b"]}"#).unwrap();
        assert!(ps[0].rid.is_none() && ps[1].rid.is_none());

        assert!(
            split(r#"{"text": ["a", "b"], "rid": ["x"]}"#).is_err(),
            "rid list length must match batch size"
        );
        assert!(
            split(r#"{"text": "a", "rid": ["x"]}"#).is_err(),
            "rid list with a single (non-batch) prompt is rejected"
        );
    }

    /// The native `bench_serving` payload (a `GenerateReqInput` superset) parses:
    /// its `lora_path`/`return_routed_experts`/`image_data` are accepted-but-ignored,
    /// so `split` succeeds and drops them while the real fields survive.
    #[test]
    fn accepts_bench_serving_payload() {
        let (ps, is_batch) = split(
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

        let framed = frame_egress_batch_cols(&header, &[&data]);
        assert_eq!(framed[0], EGRESS_TAG_BATCH);
        let mut events = Vec::new();
        assert!(for_each_chunk(&framed[1..], |ev| events.push(ev)));
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].rid_hash, RidHash::from_rid("1").0);
        assert_eq!(events[0].token_ids, vec![10, 11]);
        assert_eq!(events[0].prompt_tokens, 4);
        assert!(events[0].finish_reason.is_none());
        assert_eq!(events[1].rid_hash, RidHash::from_rid("2").0);
        assert!(events[1].token_ids.is_empty());
        // The whole dict survives (type + matched), not just the type.
        assert_eq!(
            events[1].finish_reason,
            Some(serde_json::json!({ "type": "stop", "matched": 5 }))
        );
        assert_eq!(events[2].rid_hash, RidHash::from_rid("3").0);
        assert_eq!(events[2].token_ids, vec![12]);
        assert_eq!(events[2].prompt_tokens, 6);
        // A plain decode frame carries no extras columns at all, so the per-frame
        // `has_extras` guard must skip the extras machinery entirely for every
        // request (this is the tm-egress hot path — see `for_each_chunk`).
        assert!(events.iter().all(|e| e.extras.is_none()));
    }

    /// A header whose column lengths exceed the data buffer (a Python/Rust
    /// positional-ABI drift, or a truncated frame) is rejected: `for_each_chunk`
    /// returns false and routes nothing — it must NOT panic the sole egress thread
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

        let framed = frame_egress_batch_cols(&header, &[&data]);
        let mut routed = 0usize;
        let ok = for_each_chunk(&framed[1..], |_| routed += 1);
        assert!(!ok, "malformed frame must be rejected, not decoded");
        assert_eq!(routed, 0, "no request may be routed from a rejected frame");
    }

    /// Ingress/egress rid agreement: the routing key decoded from a uuid-rid
    /// batch frame must equal `RidHash::from_rid` of the same string — the
    /// invariant that lets shard routing work without any shared map. Guards a
    /// rewrite that hashes differently on the two sides (e.g. a keyed hasher).
    #[test]
    fn uuid_rid_decodes_to_from_rid_key() {
        use rmpv::Value;
        let rid = "9f86d081884c7d659a2feaa0c55ad015";
        let header_arr = Value::Array(vec![
            Value::Array(vec![Value::from(rid)]),  // rids
            Value::Array(vec![Value::Nil]),        // finish_reasons
            Value::Array(vec![Value::from(1u32)]), // prompt_tokens
            Value::Array(vec![Value::from(1u32)]), // tok_lens
        ]);
        let mut header = Vec::new();
        rmpv::encode::write_value(&mut header, &header_arr).unwrap();
        let data: Vec<u8> = [0i32].iter().flat_map(|x| x.to_le_bytes()).collect();

        let framed = frame_egress_batch_cols(&header, &[&data]);
        let mut events = Vec::new();
        assert!(for_each_chunk(&framed[1..], |ev| events.push(ev)));
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].rid_hash, RidHash::from_rid(rid).0);
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

    /// The common frame must stay small: logprob/hidden columns are boxed behind
    /// `ChunkExtras`, so the inline decode array is a few KiB — not MiB — even at
    /// batch 4096. A regression that inlines a rare column would blow this up.
    #[test]
    fn chunk_event_frame_stays_small() {
        let sz = std::mem::size_of::<ChunkEvent>();
        assert!(
            sz <= 128,
            "ChunkEvent grew to {sz} bytes; keep rare columns behind ChunkExtras"
        );
    }

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

    /// The header must be positionally aligned: `input_embeds` (idx 5) /
    /// `token_type_ids` (idx 7) present as nil so `sampling_params` lands at idx 8 and
    /// the array reaches msgspec's min length. Regression guard for that decode failure.
    #[test]
    fn to_header_msgpack_is_positionally_aligned() {
        let req = GenerateRequest {
            text: Some("hi".into()),
            input_ids: Some(vec![1, 2, 3]),
            sampling_params: Some(rmpv::Value::Map(vec![(
                rmpv::Value::from("max_new_tokens"),
                rmpv::Value::from(5),
            )])),
            return_logprob: Some(true),
            logprob_start_len: Some(-1),
            top_logprobs_num: Some(3),
            is_health_check: true,
            stream: true,
            ..Default::default()
        };
        let bytes = req.to_header_msgpack("r1").unwrap();
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
