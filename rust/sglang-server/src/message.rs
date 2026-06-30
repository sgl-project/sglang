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
/// frames. The detok calls [`EgressSink::try_send`] without caring which kind it
/// is:
///   * `Local` — the in-process per-request channel the API handler drains for
///     SSE. One per request, bounded for backpressure; dropping the receiver
///     (client disconnect) is observed as stream end.
///   * `Net` — headless mode: each item is tagged with its `rid` and funneled to
///     the shared TCP egress writer that multiplexes one connection per DP rank.
#[derive(Clone, Debug)]
pub enum EgressSink {
    Local(mpsc::Sender<EgressItem>),
    #[allow(dead_code)] // wired by the headless TCP transport (next step).
    Net {
        rid: RequestId,
        tx: mpsc::Sender<(RequestId, EgressItem)>,
    },
}

impl EgressSink {
    /// Non-blocking send. `Err(())` means the consumer is gone (client
    /// disconnected / writer closed); callers treat that as an abort signal.
    pub fn try_send(&self, item: EgressItem) -> Result<(), ()> {
        match self {
            EgressSink::Local(tx) => tx.try_send(item).map_err(|_| ()),
            EgressSink::Net { rid, tx } => tx.try_send((*rid, item)).map_err(|_| ()),
        }
    }
}

#[allow(dead_code)] // the receiver half is created inline in api_server::submit.
pub type EgressSource = mpsc::Receiver<EgressItem>;

/// Protocol-neutral output for one generation step — a **per-chunk delta**. The
/// detok shard emits one per decode step (it keeps no cumulative buffer); a
/// consumer that needs the cumulative view folds them with `OutputAccumulator`
/// (every unary response and the cumulative SGLang `/generate` stream). OpenAI
/// streaming forwards the deltas directly. Each API handler formats the result
/// into its own wire shape — SGLang `/generate`, OpenAI `/v1/completions`,
/// `/v1/chat/completions`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerationOutput {
    pub rid: String,
    /// Decoded text **delta** for this chunk (empty in `skip_tokenizer_init`
    /// mode, or when the step only produced a partial multi-byte sequence).
    pub text: String,
    /// Output token ids for this chunk (`skip_tokenizer_init` mode; empty
    /// otherwise).
    pub output_ids: Vec<i32>,
    /// Prompt token count (from the scheduler; constant across the request).
    pub prompt_tokens: u32,
    /// Output token count for this chunk (the accumulator sums them).
    pub completion_tokens: u64,
    /// `Some(reason)` on the final step, `None` while streaming.
    pub finish_reason: Option<String>,
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
#[derive(Debug)]
pub struct GenerateRequest {
    /// Decoded HTTP body (the `GenerateReqInput` view we need for tokenization).
    pub payload: GeneratePayload,
    /// Token ids, populated by the Tokenizer stage (or already present from the
    /// client).
    pub input_ids: Option<Vec<i32>>,
    /// Whether the client asked for SSE streaming.
    pub stream: bool,
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
    /// always false until mm fields are wired in.
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

        // Tagged array in TokenizedGenerateReqInput declaration order (BaseReq
        // fields first), truncated at `stream`.
        let arr = Value::Array(vec![
            Value::from("TokenizedGenerateReqInput"), // tag
            Value::from(self.rid.as_str()),           // rid
            Value::Nil,                               // http_worker_ipc
            input_text_val,                           // input_text
            input_ids_val,                            // input_ids
            Value::Nil,                               // mm_inputs
            sampling_params_val,                      // sampling_params
            Value::from(false),                       // return_logprob
            Value::from(-1i64),                       // logprob_start_len
            Value::from(0i64),                        // top_logprobs_num
            Value::Nil,                               // token_ids_logprob
            Value::from(self.stream),                 // stream
        ]);

        let mut buf = Vec::new();
        rmpv::encode::write_value(&mut buf, &arr).map_err(|e| Error::Codec(e.to_string()))?;
        Ok(Bytes::from(buf))
    }
}

/// Egress-ring frame discriminator (first byte). Internal to the Rust egress
/// ring: Python pushes raw payloads via `push_chunk` / `push_result` and the
/// tag is prepended on the Rust side, so the Python wire format is unchanged.
pub const EGRESS_TAG_CHUNK: u8 = 0;
pub const EGRESS_TAG_RESULT: u8 = 1;

/// Frame a generation chunk for the egress ring (msgpack already built by
/// Python's `push_chunk`; just prepend the tag).
pub fn frame_egress_chunk(chunk: &[u8]) -> Bytes {
    let mut buf = Vec::with_capacity(1 + chunk.len());
    buf.push(EGRESS_TAG_CHUNK);
    buf.extend_from_slice(chunk);
    Bytes::from(buf)
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

/// One scheduler output increment for a request, pushed from Python via
/// `push_chunk` into the egress ring. Decoded on a Rust detok shard.
#[derive(Debug, Serialize, Deserialize)]
pub struct ChunkEvent {
    pub rid: String,
    pub seq: u64,
    /// New token ids for this step. Empty allowed (e.g. metadata-only frames).
    pub token_ids: Vec<i32>,
    /// `None` while streaming, `Some(reason)` on the final chunk.
    pub finish_reason: Option<String>,
    /// Prompt token count for this request (constant across its chunks).
    /// `#[serde(default)]` keeps the wire backward-compatible with 4-field frames.
    #[serde(default)]
    pub prompt_tokens: u32,
}
