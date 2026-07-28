use serde_json::{Value, json};
use sglang_server::pd::config::{
    EXPECTED_PROFILE_CANONICAL_BYTES, EXPECTED_PROFILE_DIGEST_HEX, PdProfileV1,
};

const PROFILE: &[u8] = include_bytes!("../contracts/profile-v1.json");

fn profile_value() -> Value {
    serde_json::from_slice(PROFILE).expect("checked-in profile must be JSON")
}

fn encoded(value: &Value) -> Vec<u8> {
    serde_json::to_vec(value).expect("test mutation must be JSON")
}

#[test]
fn frozen_profile_has_exact_canonical_bytes_and_digest() {
    let profile = PdProfileV1::from_slice(PROFILE).expect("frozen profile must load");

    assert_eq!(
        profile.canonical_body().expect("profile must encode").len(),
        EXPECTED_PROFILE_CANONICAL_BYTES
    );
    assert_eq!(
        profile.digest_hex().expect("profile must hash"),
        EXPECTED_PROFILE_DIGEST_HEX
    );
}

#[test]
fn object_order_does_not_change_profile_digest() {
    let compact_reordered = serde_json::to_vec(&profile_value()).expect("profile must serialize");
    let profile = PdProfileV1::from_slice(&compact_reordered).expect("reordered profile must load");

    assert_eq!(
        profile.digest_hex().expect("profile must hash"),
        EXPECTED_PROFILE_DIGEST_HEX
    );
}

#[test]
fn strict_profile_rejects_structural_and_canonicalization_errors() {
    let mut missing = profile_value();
    missing
        .as_object_mut()
        .expect("root object")
        .remove("profile_version");

    let mut unknown = profile_value();
    unknown
        .as_object_mut()
        .expect("root object")
        .insert("future".into(), json!(1));

    let mut wrong_type = profile_value();
    wrong_type["profile_version"] = json!("1");

    let mut overflow = profile_value();
    overflow["control"]["max_payload_bytes"] = json!(4_294_967_296_u64);

    let mut uppercase_hex = profile_value();
    uppercase_hex["transport"]["commit"] = json!("E9C61075720039BCFC5FFFD19F847608402BE3D0");

    let mut duplicate_list_item = profile_value();
    duplicate_list_item["request"]["input_kinds"] = json!(["input_ids", "input_ids", "text"]);

    let mut out_of_order_list = profile_value();
    out_of_order_list["request"]["input_kinds"] = json!(["text", "input_ids"]);

    let mut wrong_float = profile_value();
    wrong_float["request"]["temperature"] = json!(0.5);

    for (name, bytes) in [
        ("missing", encoded(&missing)),
        ("unknown", encoded(&unknown)),
        ("wrong_type", encoded(&wrong_type)),
        ("overflow", encoded(&overflow)),
        ("uppercase_hex", encoded(&uppercase_hex)),
        ("duplicate_list_item", encoded(&duplicate_list_item)),
        ("out_of_order_list", encoded(&out_of_order_list)),
        ("wrong_float", encoded(&wrong_float)),
        ("invalid_utf8", vec![0xff]),
    ] {
        assert!(
            PdProfileV1::from_slice(&bytes).is_err(),
            "{name} mutation unexpectedly loaded"
        );
    }
}

#[test]
fn strict_profile_rejects_duplicate_keys() {
    let duplicate = String::from_utf8(PROFILE.to_vec())
        .expect("profile is UTF-8")
        .replacen(
            "\"profile_version\": 1,",
            "\"profile_version\": 1, \"profile_version\": 1,",
            1,
        );

    assert!(PdProfileV1::from_slice(duplicate.as_bytes()).is_err());
}

#[test]
fn negative_zero_is_normalized_before_digesting() {
    let mut profile = profile_value();
    profile["request"]["temperature"] = json!(-0.0);
    profile["request"]["min_p"] = json!(-0.0);

    let profile = PdProfileV1::from_slice(&encoded(&profile))
        .expect("negative zero is the canonical positive-zero value");
    assert_eq!(
        profile.digest_hex().expect("profile must hash"),
        EXPECTED_PROFILE_DIGEST_HEX
    );
}
