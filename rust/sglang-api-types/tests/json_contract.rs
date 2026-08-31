//! Golden tests for the schema-driven JSON contract: each sglang.json.v1
//! option template, pinned against the behavior of the hand-written types it
//! replaces (message/sampling.rs, message/request.rs, message/finish_reason.rs,
//! api_server/core/frame.rs). These fixtures are the contract's only gate,
//! so they keep the generator honest on their own.

use sglang_api_types::api::v1::*;

fn sp(json: &str) -> Result<SamplingParams, serde_json::Error> {
    serde_json::from_str(json)
}

/// Absent keys yield the schema defaults (sampling.rs `defaulted!` values).
#[test]
fn sampling_defaults_match_hand_written() {
    let p = sp("{}").unwrap();
    assert_eq!(p.max_new_tokens, Some(128));
    assert_eq!(p.temperature, Some(1.0));
    assert_eq!(p.top_p, Some(1.0));
    assert_eq!(p.top_k, Some(1 << 30));
    assert_eq!(p.min_p, Some(0.0));
    assert_eq!(p.repetition_penalty, Some(1.0));
    assert_eq!(p.min_new_tokens, Some(0));
    assert_eq!(p.n, Some(1));
    assert_eq!(p.ignore_eos, Some(false));
    assert_eq!(p.skip_special_tokens, Some(true));
    assert_eq!(p.spaces_between_special_tokens, Some(true));
    assert_eq!(p.no_stop_trim, Some(false));
}

/// Protobuf arrival: a prost-decoded (all-unset) SamplingParams reads its
/// schema defaults through the accessors — the top_p=0 bug the first live
/// gRPC call found.
#[test]
fn proto_unset_reads_schema_defaults() {
    let p = SamplingParams::default();
    assert_eq!(p.temperature_or_default(), 1.0);
    assert_eq!(p.top_p_or_default(), 1.0);
    assert_eq!(p.top_k_or_default(), 1 << 30);
    assert_eq!(p.n_or_default(), 1);
    assert!(p.skip_special_tokens_or_default());
    let mut p = SamplingParams::default();
    p.apply_absent_defaults();
    assert_eq!(p.max_new_tokens, Some(128));
    assert_eq!(p.top_p, Some(1.0));
}

/// null_resets_default: an explicit null is the default, not an error
/// (Python `__post_init__` parity).
#[test]
fn sampling_null_resets_default() {
    let p = sp(r#"{"temperature": null, "top_k": null, "ignore_eos": null}"#).unwrap();
    assert_eq!(p.temperature, Some(1.0));
    assert_eq!(p.top_k, Some(1 << 30));
    assert_eq!(p.ignore_eos, Some(false));
}

/// max_new_tokens is the null_is_none field: absent = 128, null = unbounded.
#[test]
fn max_new_tokens_absent_vs_null() {
    assert_eq!(sp("{}").unwrap().max_new_tokens, Some(128));
    assert_eq!(
        sp(r#"{"max_new_tokens": null}"#).unwrap().max_new_tokens,
        None
    );
    assert_eq!(
        sp(r#"{"max_new_tokens": 7}"#).unwrap().max_new_tokens,
        Some(7)
    );
}

/// SamplingParams DENIES unknown keys with serde's own error text (parity
/// with the hand-written `deny_unknown_fields`).
#[test]
fn sampling_denies_unknown_keys() {
    let err = sp(r#"{"temperatur": 0.5}"#).unwrap_err().to_string();
    assert!(
        err.starts_with("unknown field `temperatur`, expected one of"),
        "{err}"
    );
}

/// GenerateRequest IGNORES unknown keys (unported Python fields must not 400;
/// pinned in message/request.rs).
#[test]
fn generate_request_ignores_unknown_keys() {
    let req: GenerateRequest = serde_json::from_str(
        r#"{"text": "hi", "priority": 3, "session_id": "s", "custom_logit_processor": "x"}"#,
    )
    .unwrap();
    assert!(matches!(
        req.text,
        Some(StringOrList {
            value: Some(string_or_list::Value::One(ref s))
        }) if s == "hi"
    ));
}

/// one_or_many: a scalar and a list both parse; serialization flattens back.
#[test]
fn one_or_many_flattens() {
    let one: StringOrList = serde_json::from_str(r#""a""#).unwrap();
    assert_eq!(serde_json::to_string(&one).unwrap(), r#""a""#);
    let many: StringOrList = serde_json::from_str(r#"["a", "b"]"#).unwrap();
    assert_eq!(serde_json::to_string(&many).unwrap(), r#"["a","b"]"#);
}

/// The TokenIds ambiguity: [1,2] is one id list, [[1],[2]] is a batch, [] is
/// one empty list (the hand-written untagged first-arm rule).
#[test]
fn token_ids_or_list_shape_dispatch() {
    let one: TokenIdsOrList = serde_json::from_str("[1, 2]").unwrap();
    assert!(matches!(
        one.value,
        Some(token_ids_or_list::Value::One(ref t)) if t.ids == vec![1, 2]
    ));
    let many: TokenIdsOrList = serde_json::from_str("[[1], [2, 3]]").unwrap();
    assert!(matches!(
        many.value,
        Some(token_ids_or_list::Value::Many(ref l)) if l.items.len() == 2
    ));
    let empty: TokenIdsOrList = serde_json::from_str("[]").unwrap();
    assert!(matches!(
        empty.value,
        Some(token_ids_or_list::Value::One(ref t)) if t.ids.is_empty()
    ));
}

/// Nullable list elements (PD bootstrap columns): [null, "h"] keeps per-item
/// presence; a bare scalar is the one-arm.
#[test]
fn optional_carriers_keep_element_nulls() {
    let v: OptionalStringOrList = serde_json::from_str(r#"[null, "h"]"#).unwrap();
    assert_eq!(serde_json::to_string(&v).unwrap(), r#"[null,"h"]"#);
    match v.value {
        Some(optional_string_or_list::Value::Many(l)) => {
            assert_eq!(l.items[0].value, None);
            assert_eq!(l.items[1].value.as_deref(), Some("h"));
        }
        other => panic!("expected many, got {other:?}"),
    }
    let one: OptionalInt64OrList = serde_json::from_str("17000").unwrap();
    assert!(matches!(
        one.value,
        Some(optional_int64_or_list::Value::One(OptionalInt64 {
            value: Some(17000)
        }))
    ));
}

/// SamplingParams one-or-many keeps serde's field-level error texts (the
/// hand-written SamplingParamsInput contract: never "data did not match any
/// variant").
#[test]
fn sampling_or_list_keeps_field_errors() {
    let err = serde_json::from_str::<SamplingParamsOrList>(r#"{"temperatur": 1}"#)
        .unwrap_err()
        .to_string();
    assert!(
        err.starts_with("unknown field `temperatur`"),
        "field-level text lost: {err}"
    );
}

/// finish_reason round-trips key-for-key, tagged by "type"; matched keeps its
/// wire shape (str / int / int list).
#[test]
fn finish_reason_round_trips() {
    for json in [
        r#"{"type":"stop","matched":"</s>"}"#,
        r#"{"type":"stop","matched":7}"#,
        r#"{"type":"stop","matched":[1,2]}"#,
        r#"{"type":"length","length":2}"#,
    ] {
        let parsed: FinishReason = serde_json::from_str(json).unwrap();
        let back: serde_json::Value = serde_json::to_value(&parsed).unwrap();
        let want: serde_json::Value = serde_json::from_str(json).unwrap();
        assert_eq!(back, want, "round trip of {json}");
    }
}

/// An abort carries all three keys, nulls included (Python emits them).
#[test]
fn finish_abort_emits_null_keys() {
    let parsed: FinishReason = serde_json::from_str(r#"{"type":"abort","message":"m"}"#).unwrap();
    let back = serde_json::to_string(&parsed).unwrap();
    let v: serde_json::Value = serde_json::from_str(&back).unwrap();
    assert_eq!(v["type"], "abort");
    assert_eq!(v["message"], "m");
    assert!(v.as_object().unwrap().contains_key("status_code"));
    assert!(v["status_code"].is_null());
    assert!(v.as_object().unwrap().contains_key("err_type"));
}

/// An unknown finish type round-trips unchanged (the forward-compat escape
/// hatch: a Python-side addition must not fail frame decode).
#[test]
fn unknown_finish_reason_passes_through() {
    let json = r#"{"type":"paused","step":3}"#;
    let parsed: FinishReason = serde_json::from_str(json).unwrap();
    assert!(matches!(parsed.kind, Some(finish_reason::Kind::Unknown(_))));
    let back: serde_json::Value = serde_json::to_value(&parsed).unwrap();
    let want: serde_json::Value = serde_json::from_str(json).unwrap();
    assert_eq!(back, want);
}

/// A known tag with a malformed payload also falls back to passthrough (the
/// hand-written FinishReason's untagged-outer behavior).
#[test]
fn malformed_known_finish_reason_passes_through() {
    let json = r#"{"type":"stop","matched":{"bad":"shape"}}"#;
    let parsed: FinishReason = serde_json::from_str(json).unwrap();
    assert!(matches!(parsed.kind, Some(finish_reason::Kind::Unknown(_))));
}

/// Logprob entries are [logprob, token_id, text|null] tuples; the logprob
/// slot is nullable (Python emits null for the first prefill position).
#[test]
fn logprob_tuple_shape() {
    let e = LogprobEntry {
        logprob: Some(-0.25),
        token_id: 7,
        text: None,
    };
    assert_eq!(serde_json::to_string(&e).unwrap(), "[-0.25,7,null]");
    let parsed: LogprobEntry = serde_json::from_str(r#"[-0.5, 9, "x"]"#).unwrap();
    assert_eq!(parsed.token_id, 9);
    assert_eq!(parsed.text.as_deref(), Some("x"));
    let first_prefill: LogprobEntry = serde_json::from_str(r#"[null, 10, null]"#).unwrap();
    assert_eq!(first_prefill.logprob, None);
    assert_eq!(
        serde_json::to_string(&first_prefill).unwrap(),
        "[null,10,null]"
    );
}

/// The /generate frame shape: key order (text, meta_info, output_ids),
/// conditional keys, finish_reason always present (null pre-terminal) --
/// matching api_core/frame.rs's frame_value.
#[test]
fn generate_response_matches_frame_value_shape() {
    let frame = GenerateResponse {
        text: "ok".into(),
        output_ids: Some(TokenIds { ids: vec![7, 8] }),
        meta_info: Some(GenerateMetaInfo {
            id: "client-rid".into(),
            prompt_tokens: 5,
            completion_tokens: 2,
            finish_reason: None,
            ..Default::default()
        }),
        index: None,
    };
    assert_eq!(
        serde_json::to_string(&frame).unwrap(),
        r#"{"text":"ok","meta_info":{"id":"client-rid","prompt_tokens":5,"completion_tokens":2,"finish_reason":null},"output_ids":[7,8]}"#
    );
}

/// raw_json passthrough: image_data et al. cross untouched (stored as JSON
/// text; integers keep their integer-ness).
#[test]
fn raw_json_passthrough() {
    let req: GenerateRequest = serde_json::from_str(
        r#"{"image_data": [["data:image/png;base64,xx"], null], "custom_params": {"k": 3}}"#,
    )
    .unwrap();
    let stored = req.image_data.expect("image_data present");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&stored).unwrap(),
        serde_json::json!([["data:image/png;base64,xx"], null])
    );
    let sp = req.sampling_params;
    let _ = sp; // custom_params rides SamplingParams, checked below
    let p: SamplingParams = serde_json::from_str(r#"{"custom_params": {"k": 3}}"#).unwrap();
    assert_eq!(p.custom_params.as_deref(), Some(r#"{"k":3}"#));
}
