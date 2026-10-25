//! Cross-language parity: the fixtures in `testdata/json_parity.json` are
//! checked here against the generated Rust types and, case for case, against
//! the generated Python types by `test/registered/unit/api/v1/test_parity.py`.
//! Both emitters must decode each `input` to the same `canonical` bytes (field
//! set, key order, presence and null rules) and keep `canonical` fixed under
//! a second round trip, or reject it with the fixture's `error`.

use serde::{Serialize, de::DeserializeOwned};
use sglang_api_types::api::v1 as api;

const FIXTURES: &str = include_str!("../testdata/json_parity.json");

fn check<T: DeserializeOwned + Serialize>(name: &str, case: &serde_json::Value) {
    let input = &case["input"];
    let decoded = serde_json::from_value::<T>(input.clone());
    let bytes = serde_json::to_vec(input).unwrap();
    let from_bytes = serde_json::from_slice::<T>(&bytes);
    let from_reader = serde_json::from_reader::<_, T>(std::io::Cursor::new(&bytes));
    if let Some(expected) = case.get("error").and_then(|v| v.as_str()) {
        let error = decoded
            .err()
            .unwrap_or_else(|| panic!("{name}: input unexpectedly decoded"));
        // Deferred raw JSON may add its own source position, including when
        // the outer input came from a Value rather than directly from bytes.
        assert!(error.to_string().starts_with(expected), "{name}: {error}");
        for result in [from_bytes, from_reader] {
            let error = result
                .err()
                .unwrap_or_else(|| panic!("{name}: input unexpectedly decoded"));
            assert!(error.to_string().starts_with(expected), "{name}: {error}");
        }
        return;
    }
    let canonical = case["canonical"].as_str().expect("canonical");
    for result in [decoded, from_bytes, from_reader] {
        let decoded =
            result.unwrap_or_else(|error| panic!("{name}: input does not decode: {error}"));
        assert_eq!(
            serde_json::to_string(&decoded).unwrap(),
            canonical,
            "{name}: input -> canonical"
        );
    }
    let again: T = serde_json::from_str(canonical)
        .unwrap_or_else(|error| panic!("{name}: canonical does not decode: {error}"));
    assert_eq!(
        serde_json::to_string(&again).unwrap(),
        canonical,
        "{name}: canonical is not a fixed point"
    );
}

#[test]
fn fixtures_decode_to_their_canonical_bytes() {
    let fixtures: serde_json::Value = serde_json::from_str(FIXTURES).unwrap();
    let cases = fixtures["cases"].as_array().expect("cases array");
    assert!(!cases.is_empty());
    for case in cases {
        let ty = case["type"].as_str().expect("type");
        let name = format!("{ty}: {}", case["name"].as_str().expect("name"));
        match ty {
            "GenerateRequest" => check::<api::GenerateRequest>(&name, case),
            "SamplingParams" => check::<api::SamplingParams>(&name, case),
            "GenerateResponse" => check::<api::GenerateResponse>(&name, case),
            "GenerateStreamItem" => check::<api::GenerateStreamItem>(&name, case),
            "GenerateResponseOrList" => check::<api::GenerateResponseOrList>(&name, case),
            "GenerateStreamError" => check::<api::GenerateStreamError>(&name, case),
            "GetModelInfoResponse" => check::<api::GetModelInfoResponse>(&name, case),
            "GetServerInfoResponse" => check::<api::GetServerInfoResponse>(&name, case),
            "AttachHiCacheStorageRequest" => check::<api::AttachHiCacheStorageRequest>(&name, case),
            "HiCacheStatus" => check::<api::HiCacheStatus>(&name, case),
            "HiCacheStorageResponse" => check::<api::HiCacheStorageResponse>(&name, case),
            "ClearHiCacheStorageRequest" => check::<api::ClearHiCacheStorageRequest>(&name, case),
            "DetachHiCacheStorageRequest" => check::<api::DetachHiCacheStorageRequest>(&name, case),
            "GetHiCacheStorageRequest" => check::<api::GetHiCacheStorageRequest>(&name, case),
            "ListModelsResponse" => check::<api::ListModelsResponse>(&name, case),
            "DetokenizeRequest" => check::<api::DetokenizeRequest>(&name, case),
            "DetokenizeResponse" => check::<api::DetokenizeResponse>(&name, case),
            "AbortRequest" => check::<api::AbortRequest>(&name, case),
            "AbortResponse" => check::<api::AbortResponse>(&name, case),
            other => panic!("fixture type {other} has no Rust mapping; add it here"),
        }
    }
}

/// Buffering must preserve duplicate keys for the selected payload's decoder.
/// A JSON object map would silently collapse them before validation.
#[test]
fn stream_item_validates_duplicates_only_in_the_selected_payload() {
    use api::generate_stream_item::Item;

    for (input, field) in [
        (r#"{"text":"first","text":"second"}"#, "text"),
        (r#"{"error":{},"error":{}}"#, "error"),
    ] {
        let error = serde_json::from_str::<api::GenerateStreamItem>(input).unwrap_err();
        assert!(
            error
                .to_string()
                .starts_with(&format!("duplicate field `{field}`")),
            "{error}"
        );
    }

    // The escaped selecting key follows invalid and repeated success fields.
    let item: api::GenerateStreamItem = serde_json::from_str(
        r#"{"text":7,"text":false,"err\u006fr":{"message":"failed","code":500}}"#,
    )
    .unwrap();
    let Some(Item::Error(error)) = item.item else {
        panic!("expected the error outcome");
    };
    assert_eq!(error.error.unwrap().message, "failed");
}
