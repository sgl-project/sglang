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

/// Sink the API-server connection handler reads from to emit SSE frames.
/// One per request, bounded for backpressure. The FSM owner holds the sender;
/// dropping it (disconnect) is observed by the handler as stream end.
pub type EgressSink = mpsc::Sender<EgressItem>;
#[allow(dead_code)] // the receiver half is created inline in api_server::submit.
pub type EgressSource = mpsc::Receiver<EgressItem>;

/// What the connection handler receives on the egress stream.
#[derive(Debug)]
pub enum EgressItem {
    /// A streamed delta, already serialized to the JSON bytes the client expects
    /// (built by the detok shard / TM egress so the handler stays trivial).
    Frame(Bytes),
    /// Terminal success: handler emits the final frame then `data: [DONE]`.
    Done(Bytes),
    /// Terminal failure: handler emits an error frame (stream) or status (unary).
    Error(Error),
}

/// What kind of request this is — selects the ingress branch and the wire
/// message pushed to the scheduler. Control requests reuse the same ingress
/// FSM as generate (validate → queue → ring) but skip tokenization, and their
/// egress is a single non-streamed JSON result rather than detokenized chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestKind {
    /// `/generate`: tokenize then push a `TokenizedGenerateReqInput`.
    Generate,
    /// A control endpoint (e.g. `/server_info`, `/health`): no tokenization.
    /// The payload is the scheduler request-struct tag (msgspec class name,
    /// e.g. `"GetInternalStateReq"`) pushed as a bare `[tag, rid, nil]`; the
    /// scheduler replies with a single JSON result via the egress ring.
    Control(&'static str),
}

/// The owned request as it travels ingress stages. Single owner at all times,
/// so the embedded `state` FSM is mutated without any lock.
#[derive(Debug)]
pub struct Request {
    pub id: RequestId,
    pub kind: RequestKind,
    pub state: RequestState,
    /// Decoded HTTP body (the `GenerateReqInput` view we need for tokenization).
    pub payload: GeneratePayload,
    /// Token ids, populated by the Tokenizer stage (or already present from the
    /// client). `Bytes` so handing it to the ring is copy-free.
    pub input_ids: Option<Vec<i32>>,
    /// Back-channel to the client connection for egress frames.
    pub sink: EgressSink,
    /// Whether the client asked for SSE streaming.
    pub stream: bool,
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
    /// Serialize to the msgspec-compatible tagged array (one allocation; the
    /// ring takes ownership of the resulting `Bytes`).
    pub fn to_msgpack(&self) -> Result<Bytes, Error> {
        use rmpv::Value;

        // `input_ids: Optional[array]` is a Python `array("q", ...)` (int64);
        // msgspec's enc_hook encodes it as the 2-tuple `(typecode, le_bytes)`.
        let mut id_bytes = Vec::with_capacity(self.input_ids.len() * 8);
        for &id in &self.input_ids {
            id_bytes.extend_from_slice(&(id as i64).to_le_bytes());
        }
        let input_ids_val = Value::Array(vec![Value::from("q"), Value::Binary(id_bytes)]);

        let input_text_val = match &self.input_text {
            Some(t) => Value::from(t.as_str()),
            None => Value::Nil,
        };

        // `sampling_params: SamplingParams` is required (not Optional) and
        // map-encoded; default to an empty map (all-defaults) when absent, and
        // make sure `stop` / `stop_regex` are present so the scheduler doesn't
        // panic on the `None` they'd otherwise carry into `stop_strs` /
        // `stop_regex_strs` (see SamplingParams.__post_init__).
        let sampling_params_val = with_stop_defaults(self.sampling_params.clone());

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

/// Return the sampling-params map with `stop` and `stop_regex` guaranteed
/// present (defaulting to `""`). `SamplingParams.__post_init__` assigns
/// `stop_strs = stop` / `stop_regex_strs = stop_regex`; a `None` there can panic
/// downstream Python, and `""` is falsy so it behaves as "no stop". A
/// client-provided value for either key is left untouched.
fn with_stop_defaults(sp: Option<rmpv::Value>) -> rmpv::Value {
    use rmpv::Value;
    let mut map = match sp {
        Some(Value::Map(m)) => m,
        _ => Vec::new(),
    };
    for key in ["stop", "stop_regex"] {
        if !map.iter().any(|(k, _)| k.as_str() == Some(key)) {
            map.push((Value::from(key), Value::from("")));
        }
    }
    Value::Map(map)
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
}
