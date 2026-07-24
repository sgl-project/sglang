//! The `/generate` request path: the HTTP body and its per-request fan-out
//! ([`GenerateBody`] → [`GenerateRequest`]s), the variant bodies, and the
//! scheduler ingress encodings (`TokenizedGenerateReqInput` header,
//! control/abort, `IngressMsg`).

use bytes::Bytes;
use serde::Deserialize;

use crate::error::Error;

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
    /// Parallel sampling factor (`GenerateReqInput.n`). Only `n == 1` is
    /// supported; a larger value is a 400 (not a silent single sample).
    #[serde(default)]
    pub n: Option<i64>,

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
            n: parallel_n,
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

        if parallel_n.unwrap_or(1) != 1 {
            return Err("parallel sampling (n > 1) is not supported".into());
        }

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
        // `n` is accepted for wire-compat but only n == 1 is supported.
        assert!(split(r#"{"text": "a", "n": 1}"#).is_ok());
        assert!(split(r#"{"text": "a", "n": 2}"#).is_err());
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
