use serde_json::json;
use smg::extended_chat::ExtendedChatCompletionRequest;

/// Every typed SGLang extra must survive a JSON → wrapper → JSON roundtrip and
/// be accessible via direct field access (no HashMap lookup).
#[test]
fn test_typed_fields_roundtrip() {
    let input = json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.7,
        // SGLang chat extras we explicitly support.
        "return_routed_experts": true,
        "return_cached_tokens_details": true,
        "return_prompt_token_ids": true,
        "return_meta_info": true,
        "input_ids": [1, 2, 3, 4],
    });

    let extended: ExtendedChatCompletionRequest =
        serde_json::from_value(input).expect("deserialization should succeed");

    // Inner openai-protocol fields still populated correctly.
    assert_eq!(extended.inner.model, "test-model");
    assert_eq!(extended.inner.temperature, Some(0.7));

    // All five extras populated as typed fields.
    assert!(extended.return_routed_experts);
    assert!(extended.return_cached_tokens_details);
    assert!(extended.return_prompt_token_ids);
    assert!(extended.return_meta_info);
    assert_eq!(extended.input_ids.as_deref(), Some(&[1, 2, 3, 4][..]));

    // Re-serialize: every extra appears at the top level, ready for downstream
    // Python (serving_chat._convert_to_internal_format) to pick up.
    let output = serde_json::to_value(&extended).expect("serialization should succeed");
    let obj = output.as_object().expect("should be object");
    assert_eq!(obj.get("model").and_then(|v| v.as_str()), Some("test-model"));
    assert_eq!(
        obj.get("return_routed_experts").and_then(|v| v.as_bool()),
        Some(true)
    );
    assert_eq!(
        obj.get("return_cached_tokens_details")
            .and_then(|v| v.as_bool()),
        Some(true)
    );
    assert_eq!(
        obj.get("return_prompt_token_ids").and_then(|v| v.as_bool()),
        Some(true)
    );
    assert_eq!(
        obj.get("return_meta_info").and_then(|v| v.as_bool()),
        Some(true)
    );
    assert_eq!(obj.get("input_ids"), Some(&json!([1, 2, 3, 4])));
}

/// `skip_serializing_if` must keep default values out of the forwarded JSON so
/// we don't lie to downstreams about what the client asked for.
#[test]
fn test_defaults_when_absent() {
    let input = json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hello"}],
    });

    let extended: ExtendedChatCompletionRequest = serde_json::from_value(input).unwrap();

    assert!(!extended.return_routed_experts);
    assert!(!extended.return_cached_tokens_details);
    assert!(!extended.return_prompt_token_ids);
    assert!(!extended.return_meta_info);
    assert!(extended.input_ids.is_none());

    let output = serde_json::to_value(&extended).expect("serialize");
    let obj = output.as_object().expect("object");
    assert!(!obj.contains_key("return_routed_experts"));
    assert!(!obj.contains_key("return_cached_tokens_details"));
    assert!(!obj.contains_key("return_prompt_token_ids"));
    assert!(!obj.contains_key("return_meta_info"));
    assert!(!obj.contains_key("input_ids"));
}

/// Unknown JSON fields (fields we haven't typed) must be silently dropped.
/// This guards against regression back to the old HashMap-flatten design,
/// which accepted arbitrary fields and forwarded them to backends with no
/// router-side awareness of router-owned collisions.
#[test]
fn test_unknown_field_silently_dropped() {
    let input = json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hello"}],
        "priority": 5,
        "cache_salt": "salt-abc",
        "rid": "req-001",
    });

    let extended: ExtendedChatCompletionRequest = serde_json::from_value(input).unwrap();

    let output = serde_json::to_value(&extended).expect("serialize");
    let obj = output.as_object().expect("object");
    assert!(!obj.contains_key("priority"));
    assert!(!obj.contains_key("cache_salt"));
    assert!(!obj.contains_key("rid"));
}

/// Deref gives downstream code transparent access to inner openai-protocol
/// fields without requiring `.inner.` plumbing everywhere.
#[test]
fn test_deref_access() {
    let input = json!({
        "model": "my-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true,
        "return_meta_info": true,
    });

    let extended: ExtendedChatCompletionRequest = serde_json::from_value(input).unwrap();
    assert_eq!(extended.model, "my-model");
    assert!(extended.stream);
    // Typed extras are accessed directly on the wrapper.
    assert!(extended.return_meta_info);
}

/// Validation must delegate to `inner.validate()` so the crate's schema-level
/// checks (e.g. top_p ∈ [0, 1]) still fire on the wrapped request.
#[test]
fn test_validation_delegates_to_inner() {
    use validator::Validate;

    let bad = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "top_p": 2.0, // out of range per ChatCompletionRequest validation
    });
    let extended: ExtendedChatCompletionRequest = serde_json::from_value(bad).unwrap();
    assert!(extended.validate().is_err(), "top_p=2.0 should fail");

    let good = json!({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.5,
        "return_meta_info": true,
        "input_ids": [1, 2, 3],
    });
    let extended: ExtendedChatCompletionRequest = serde_json::from_value(good).unwrap();
    assert!(extended.validate().is_ok(), "legal request should pass");
}
