use serde::de::DeserializeOwned;
use sglang_api_types::api::v1::*;

const I64_VALID: &[&str] = &["-9223372036854775808", "0", "9223372036854775807"];
const I64_INVALID: &[&str] = &["-9223372036854775809", "9223372036854775808"];
const U32_VALID: &[&str] = &["0", "4294967295"];
const U32_INVALID: &[&str] = &["-1", "4294967296"];
const U64_VALID: &[&str] = &["0", "9223372036854775808", "18446744073709551615"];
const U64_INVALID: &[&str] = &["-1", "18446744073709551616"];

fn check_range<T: DeserializeOwned>(
    input: impl Fn(&str) -> String,
    valid: &[&str],
    invalid: &[&str],
) {
    for value in valid {
        let json = input(value);
        assert!(
            serde_json::from_str::<T>(&json).is_ok(),
            "valid integer rejected: {json}"
        );
    }
    for value in invalid.iter().chain(&[
        "-1267650600228229401496703205376",
        "1267650600228229401496703205376",
        "true",
        "false",
        "1.0",
    ]) {
        let json = input(value);
        assert!(
            serde_json::from_str::<T>(&json).is_err(),
            "invalid integer accepted: {json}"
        );
    }
}

#[test]
fn integer_field_bounds() {
    for field in ["max_new_tokens", "top_k", "stream_interval"] {
        check_range::<SamplingParams>(
            |value| format!(r#"{{"{field}":{value}}}"#),
            I64_VALID,
            I64_INVALID,
        );
    }
    check_range::<GenerateRequest>(
        |value| format!(r#"{{"routed_experts_start_len":{value}}}"#),
        I64_VALID,
        I64_INVALID,
    );
    for field in ["prompt_tokens", "image_tokens"] {
        check_range::<GenerateMetaInfo>(
            |value| format!(r#"{{"{field}":{value}}}"#),
            U32_VALID,
            U32_INVALID,
        );
    }
    for field in ["completion_tokens", "output_token_logprobs_length"] {
        check_range::<GenerateMetaInfo>(
            |value| format!(r#"{{"{field}":{value}}}"#),
            U64_VALID,
            U64_INVALID,
        );
    }
}

#[test]
fn repeated_integer_bounds() {
    check_range::<DetokenizeRequest>(
        |value| format!(r#"{{"tokens":[0,{value}]}}"#),
        I64_VALID,
        I64_INVALID,
    );
}

#[test]
fn tuple_integer_bounds() {
    check_range::<LogprobEntry>(
        |value| format!("[null,{value},null]"),
        I64_VALID,
        I64_INVALID,
    );
}

#[test]
fn transparent_integer_bounds() {
    check_range::<OptionalInt64>(str::to_owned, I64_VALID, I64_INVALID);
}

#[test]
fn integer_union_bounds() {
    for input in [str::to_owned, |value: &str| format!("[0,{value}]")] {
        check_range::<Int64OrList>(input, I64_VALID, I64_INVALID);
        check_range::<OptionalInt64OrList>(input, I64_VALID, I64_INVALID);
        check_range::<Matched>(input, I64_VALID, I64_INVALID);
    }
}

#[test]
fn matched_preserves_strings() {
    let parsed: Matched = serde_json::from_str(r#""1""#).unwrap();
    assert_eq!(serde_json::to_string(&parsed).unwrap(), r#""1""#);
    assert!(serde_json::from_str::<Matched>(r#"[0,"1"]"#).is_err());
}

#[test]
fn nested_integer_list_bounds() {
    check_range::<TokenIdsOrList>(|value| format!("[[0],[{value}]]"), I64_VALID, I64_INVALID);
}

#[test]
fn tagged_integer_overflow_passes_through() {
    for json in [
        r#"{"type":"stop","matched":-9223372036854775809}"#,
        r#"{"type":"stop","matched":9223372036854775808}"#,
        r#"{"type":"stop","matched":[0,9223372036854775808]}"#,
        r#"{"type":"length","length":-1}"#,
        r#"{"type":"length","length":18446744073709551616}"#,
        r#"{"type":"abort","status_code":-1}"#,
        r#"{"type":"abort","status_code":4294967296}"#,
    ] {
        let parsed: FinishReason = serde_json::from_str(json).unwrap();
        assert!(matches!(parsed.kind, Some(finish_reason::Kind::Unknown(_))));
        assert_eq!(
            serde_json::to_value(parsed).unwrap(),
            serde_json::from_str::<serde_json::Value>(json).unwrap(),
            "{json}"
        );
    }
}
