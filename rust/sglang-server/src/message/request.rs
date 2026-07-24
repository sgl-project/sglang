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

    // Multimodal inputs. Permissive `Value` types: any JSON shape the Python
    // `GenerateReqInput` accepts (URL / base64 str / list / list-of-lists)
    // parses; `split` fans them out per item mirroring the Python
    // `_normalize_{image,video,audio}_data` batch rules.
    #[serde(default)]
    pub image_data: Option<rmpv::Value>,
    #[serde(default)]
    pub video_data: Option<rmpv::Value>,
    #[serde(default)]
    pub audio_data: Option<rmpv::Value>,

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
            image_data,
            video_data,
            audio_data,
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

        // Multimodal columns, mirroring the Python batch-normalize rules
        // (`_normalize_image_data` / `_normalize_video_data` / `_normalize_audio_data`).
        let mut images = split_mm_column(image_data, n, is_batch, MmBroadcast::WrapInList)
            .map_err(|e| format!("image_data: {e}"))?;
        let mut videos = split_mm_column(video_data, n, is_batch, MmBroadcast::AsIs)
            .map_err(|e| format!("video_data: {e}"))?;
        let mut audios = split_mm_column(audio_data, n, is_batch, MmBroadcast::AsIs)
            .map_err(|e| format!("audio_data: {e}"))?;

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
                mm: pack_mm(images[i].take(), videos[i].take(), audios[i].take()),
            })
            .collect();
        Ok((requests, is_batch))
    }
}

/// Box the per-item mm values, or `None` when the item has none (the common
/// text-only case keeps `GenerateRequest` slim).
fn pack_mm(
    image_data: Option<rmpv::Value>,
    video_data: Option<rmpv::Value>,
    audio_data: Option<rmpv::Value>,
) -> Option<Box<MmData>> {
    if image_data.is_none() && video_data.is_none() && audio_data.is_none() {
        return None;
    }
    Some(Box::new(MmData {
        image_data,
        video_data,
        audio_data,
    }))
}

/// How a scalar (non-list) mm value broadcasts across a batch: images become a
/// single-image list per item (`[[img]] * num` in Python `_normalize_image_data`),
/// video/audio broadcast the bare value (`[v] * num` in `_normalize_video_data`).
#[derive(Clone, Copy)]
enum MmBroadcast {
    WrapInList,
    AsIs,
}

/// Fan one mm field (`image_data` / `video_data` / `audio_data`) into per-item
/// values, mirroring the Python batch-normalize semantics:
///   * `None` / empty list → `None` for every item;
///   * single request → the raw value passes through (the processor-side
///     normalize wraps a non-list into a one-element list);
///   * batch + non-list → broadcast to every item (per `MmBroadcast`);
///   * batch + list → per item; length must equal the batch size.
fn split_mm_column(
    v: Option<rmpv::Value>,
    n: usize,
    is_batch: bool,
    broadcast: MmBroadcast,
) -> Result<Vec<Option<rmpv::Value>>, String> {
    let Some(v) = v else {
        return Ok(vec![None; n]);
    };
    if v.is_nil() {
        return Ok(vec![None; n]);
    }
    if !is_batch {
        return Ok(vec![Some(v)]);
    }
    match v {
        rmpv::Value::Array(items) if items.is_empty() => Ok(vec![None; n]),
        rmpv::Value::Array(items) => {
            if items.len() != n {
                return Err(format!(
                    "list length {} does not match batch size {n}",
                    items.len()
                ));
            }
            Ok(items.into_iter().map(Some).collect())
        }
        scalar => Ok(match broadcast {
            MmBroadcast::WrapInList => vec![Some(rmpv::Value::Array(vec![scalar])); n],
            MmBroadcast::AsIs => vec![Some(scalar); n],
        }),
    }
}

/// One request handed to the MM worker pool: the rid to correlate the result
/// plus the msgpack payload from [`GenerateRequest::to_mm_payload_msgpack`].
#[derive(Debug)]
pub struct MmRequest {
    pub rid: String,
    pub payload: Bytes,
}

/// Rust mirror of Python `has_valid_data` for an opaque mm field: `null` and
/// (recursively) empty / all-null lists don't count as multimodal input.
fn mm_value_present(v: &Option<rmpv::Value>) -> bool {
    fn valid(v: &rmpv::Value) -> bool {
        match v {
            rmpv::Value::Nil => false,
            rmpv::Value::Array(items) => items.iter().any(valid),
            _ => true,
        }
    }
    v.as_ref().is_some_and(valid)
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
    /// Multimodal inputs, carried opaquely (URL / base64 / path / nested lists —
    /// any JSON shape the Python `GenerateReqInput` accepts). Consumed by the
    /// Encoding stage, which ships them to the MM worker pool; never read by
    /// the tokenizer or serialized onto the scheduler header. Boxed: absent on
    /// the common text-only request, so it shouldn't grow every `Request` moved
    /// between stages.
    pub mm: Option<Box<MmData>>,
}

/// The three opaque multimodal fields of one request (see [`GenerateRequest::mm`]).
#[derive(Debug, Default)]
pub struct MmData {
    pub image_data: Option<rmpv::Value>,
    pub video_data: Option<rmpv::Value>,
    pub audio_data: Option<rmpv::Value>,
}

impl GenerateRequest {
    /// True when the client already supplied token ids → skip tokenization.
    pub fn already_tokenized(&self) -> bool {
        self.input_ids.as_ref().is_some_and(|v| !v.is_empty())
    }

    /// True when the request carries any usable multimodal payload — the Rust
    /// mirror of Python `GenerateReqInput.contains_mm_input()` /
    /// `has_valid_data`: `null` and (recursively) empty lists don't count.
    pub fn has_multimodal(&self) -> bool {
        self.mm.as_ref().is_some_and(|mm| {
            mm_value_present(&mm.image_data)
                || mm_value_present(&mm.video_data)
                || mm_value_present(&mm.audio_data)
        })
    }

    /// Serialize the fields the MM worker pool needs for this request: a
    /// msgpack array `[text, input_ids, image_data, video_data, audio_data]`
    /// (decoded by `sglang-mm`'s `payload::parse` in the native pipeline).
    pub fn to_mm_payload_msgpack(&self) -> Result<Bytes, Error> {
        use rmpv::Value;
        let text_val = match &self.text {
            Some(t) => Value::from(t.as_str()),
            None => Value::Nil,
        };
        let input_ids_val = match &self.input_ids {
            Some(ids) => Value::Array(ids.iter().map(|&i| Value::from(i)).collect()),
            None => Value::Nil,
        };
        let mm_field = |f: fn(&MmData) -> &Option<Value>| -> Value {
            self.mm
                .as_deref()
                .and_then(|m| f(m).clone())
                .unwrap_or(Value::Nil)
        };
        let arr = Value::Array(vec![
            text_val,
            input_ids_val,
            mm_field(|m| &m.image_data),
            mm_field(|m| &m.video_data),
            mm_field(|m| &m.audio_data),
        ]);
        let mut buf = Vec::new();
        rmpv::encode::write_value(&mut buf, &arr).map_err(|e| Error::Codec(e.to_string()))?;
        Ok(Bytes::from(buf))
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
    /// its `lora_path`/`return_routed_experts` are accepted-but-ignored and a
    /// `null` `image_data` means "no multimodal input", so `split` succeeds
    /// while the real fields survive.
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
        assert!(!ps[0].has_multimodal());
    }

    /// Mm columns fan out per the Python `_normalize_{image,video}_data` rules: a
    /// single request passes the raw value through; a batch broadcasts a scalar
    /// image as `[img]` per item (`[[img]]*n`), maps a list per item, requires
    /// matching lengths, and treats `null`/`[]` as absent.
    #[test]
    fn split_mm_fanout_matches_python_normalize() {
        let image_of = |p: &GenerateRequest| p.mm.as_ref().unwrap().image_data.clone().unwrap();

        // Single request: raw value passes through untouched.
        let (ps, _) = split(r#"{"text": "a", "image_data": "http://x/i.jpg"}"#).unwrap();
        assert_eq!(image_of(&ps[0]).as_str(), Some("http://x/i.jpg"));
        assert!(ps[0].has_multimodal());

        // Batch + scalar image: broadcast, wrapped as a one-image list per item.
        let (ps, _) = split(r#"{"text": ["a", "b"], "image_data": "u"}"#).unwrap();
        for p in &ps {
            assert_eq!(image_of(p).as_array().unwrap().len(), 1);
            assert!(p.has_multimodal());
        }

        // Batch + per-item list: element i goes to item i.
        let (ps, _) = split(r#"{"text": ["a", "b"], "image_data": ["u1", "u2"]}"#).unwrap();
        assert_eq!(image_of(&ps[0]).as_str(), Some("u1"));
        assert_eq!(image_of(&ps[1]).as_str(), Some("u2"));

        // Batch + wrong-length list is a 400.
        assert!(split(r#"{"text": ["a", "b"], "image_data": ["u1"]}"#).is_err());

        // null / [] mean "no multimodal input".
        let (ps, _) = split(r#"{"text": "a", "image_data": null}"#).unwrap();
        assert!(!ps[0].has_multimodal());
        let (ps, _) = split(r#"{"text": "a", "image_data": []}"#).unwrap();
        assert!(!ps[0].has_multimodal());

        // Batch + scalar video: broadcast bare (not wrapped), per Python
        // `_normalize_video_data`.
        let (ps, _) = split(r#"{"text": ["a", "b"], "video_data": "v"}"#).unwrap();
        let video = ps[1].mm.as_ref().unwrap().video_data.clone().unwrap();
        assert_eq!(video.as_str(), Some("v"));
        assert!(ps[1].has_multimodal());
    }

    /// The mm payload for the MM worker pool is a positional msgpack array
    /// `[text, input_ids, image_data, video_data, audio_data]`.
    #[test]
    fn mm_payload_shape() {
        let (ps, _) =
            split(r#"{"text": "hi", "image_data": ["u1", "u2"], "audio_data": "a"}"#).unwrap();
        let payload = ps[0].to_mm_payload_msgpack().unwrap();
        let val = rmpv::decode::read_value(&mut &payload[..]).unwrap();
        let arr = val.as_array().unwrap();
        assert_eq!(arr.len(), 5);
        assert_eq!(arr[0].as_str(), Some("hi"));
        assert!(arr[1].is_nil());
        assert_eq!(arr[2].as_array().unwrap().len(), 2);
        assert!(arr[3].is_nil());
        assert_eq!(arr[4].as_str(), Some("a"));
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
