//! The in-flight request: variant bodies, the egress back-channel, and the
//! scheduler-wire encodings (`TokenizedGenerateReqInput` header, control/abort).

use bytes::Bytes;
use tokio::sync::mpsc;

use super::chunk::ChunkEvent;
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
    /// Internal `/health_generate` probe. Not a wire field — the probe is
    /// recognized (and skipped when busy) by its `HEALTH_CHECK_` rid prefix,
    /// mirroring Python `constants.HEALTH_CHECK_RID_PREFIX`; here it only
    /// drives that rid minting. Never set from the client wire.
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
            Value::from(false),                                // 14 return_sampling_mask
            Value::from(self.return_hidden_states.unwrap_or(false)), // 15 return_hidden_states
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

/// One ingress-ring entry, split columnar: the scalar `header` (msgpack, `input_ids`
/// omitted) + the raw int64 `ids` cell, so the big tensor never goes through msgpack.
#[derive(Debug)]
pub struct IngressMsg {
    pub header: Bytes,
    pub ids: Bytes,
}

#[cfg(test)]
mod tests {
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
            return_hidden_states: Some(true),
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
        // idx 14 is `return_sampling_mask` (never client-set); a shift here would
        // silently flip the wrong scheduler field.
        assert_eq!(
            arr[14].as_bool(),
            Some(false),
            "return_sampling_mask at idx 14"
        );
        assert_eq!(
            arr[15].as_bool(),
            Some(true),
            "return_hidden_states at idx 15"
        );
    }
}
