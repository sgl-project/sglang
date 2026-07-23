//! The `/generate` HTTP body: the scalar-or-list wire shape and its fan-out
//! into per-request [`GenerateRequest`]s.

use serde::Deserialize;

use super::request::GenerateRequest;

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
}
