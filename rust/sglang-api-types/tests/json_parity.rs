//! Cross-language parity: the fixtures in `testdata/json_parity.json` are
//! checked here against the generated Rust types and, case for case, against
//! the generated Python types by `test/registered/unit/api/v1/test_parity.py`.
//! Both emitters must decode each `input` to the same `canonical` bytes (field
//! set, key order, presence and null rules) and keep `canonical` fixed under
//! a second round trip.

use serde::{Serialize, de::DeserializeOwned};
use sglang_api_types::api::v1 as api;

const FIXTURES: &str = include_str!("../testdata/json_parity.json");

fn check<T: DeserializeOwned + Serialize>(name: &str, input: &serde_json::Value, canonical: &str) {
    let decoded: T = serde_json::from_value(input.clone())
        .unwrap_or_else(|error| panic!("{name}: input does not decode: {error}"));
    assert_eq!(
        serde_json::to_string(&decoded).unwrap(),
        canonical,
        "{name}: input -> canonical"
    );
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
        let input = &case["input"];
        let canonical = case["canonical"].as_str().expect("canonical");
        match ty {
            "GenerateRequest" => check::<api::GenerateRequest>(&name, input, canonical),
            "SamplingParams" => check::<api::SamplingParams>(&name, input, canonical),
            "GenerateResponse" => check::<api::GenerateResponse>(&name, input, canonical),
            "GenerateStreamError" => check::<api::GenerateStreamError>(&name, input, canonical),
            "GetModelInfoResponse" => check::<api::GetModelInfoResponse>(&name, input, canonical),
            "GetServerInfoResponse" => check::<api::GetServerInfoResponse>(&name, input, canonical),
            other => panic!("fixture type {other} has no Rust mapping; add it here"),
        }
    }
}
