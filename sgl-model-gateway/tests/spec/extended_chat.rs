use serde_json::json;
use smg::extended_chat::ExtendedChatCompletionRequest;

/// Verify that unknown/extra fields survive a JSON → ExtendedChatCompletionRequest → JSON roundtrip.
#[test]
fn test_extra_fields_roundtrip() {
    let input = json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.7,
        // --- sglang-specific fields NOT modeled in openai-protocol crate ---
        "rid": "req-001",
        "priority": 5,
        "input_ids": [1, 2, 3, 4],
        "data_parallel_rank": 2,
        "stop_regex": "\\n\\n",
        "cache_salt": "salt-abc",
        "custom_field_xyz": "should_survive"
    });

    // Deserialize
    let extended: ExtendedChatCompletionRequest =
        serde_json::from_value(input.clone()).expect("deserialization should succeed");

    // Known fields land in inner
    assert_eq!(extended.inner.model, "test-model");
    assert_eq!(extended.inner.temperature, Some(0.7));

    // Extra fields are captured
    assert_eq!(extended.extra.get("rid").and_then(|v| v.as_str()), Some("req-001"));
    assert_eq!(extended.extra.get("priority").and_then(|v| v.as_i64()), Some(5));
    assert_eq!(
        extended.extra.get("input_ids"),
        Some(&json!([1, 2, 3, 4]))
    );
    assert_eq!(
        extended.extra.get("data_parallel_rank").and_then(|v| v.as_i64()),
        Some(2)
    );
    assert_eq!(
        extended.extra.get("stop_regex").and_then(|v| v.as_str()),
        Some("\\n\\n")
    );
    assert_eq!(
        extended.extra.get("cache_salt").and_then(|v| v.as_str()),
        Some("salt-abc")
    );
    assert_eq!(
        extended.extra.get("custom_field_xyz").and_then(|v| v.as_str()),
        Some("should_survive")
    );

    // Re-serialize and verify all fields are present
    let output = serde_json::to_value(&extended).expect("serialization should succeed");
    let obj = output.as_object().expect("should be object");

    assert_eq!(obj.get("model").and_then(|v| v.as_str()), Some("test-model"));
    // temperature is f32 in the struct, so precision may differ slightly
    let temp = obj.get("temperature").and_then(|v| v.as_f64()).unwrap();
    assert!((temp - 0.7).abs() < 1e-6, "temperature should be ~0.7, got {}", temp);
    assert_eq!(obj.get("rid").and_then(|v| v.as_str()), Some("req-001"));
    assert_eq!(obj.get("priority").and_then(|v| v.as_i64()), Some(5));
    assert_eq!(obj.get("input_ids"), Some(&json!([1, 2, 3, 4])));
    assert_eq!(obj.get("data_parallel_rank").and_then(|v| v.as_i64()), Some(2));
    assert_eq!(obj.get("stop_regex").and_then(|v| v.as_str()), Some("\\n\\n"));
    assert_eq!(obj.get("cache_salt").and_then(|v| v.as_str()), Some("salt-abc"));
    assert_eq!(
        obj.get("custom_field_xyz").and_then(|v| v.as_str()),
        Some("should_survive")
    );
}

/// Verify Deref gives transparent access to inner fields.
#[test]
fn test_deref_access() {
    let input = json!({
        "model": "my-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true,
        "rid": "r1"
    });

    let extended: ExtendedChatCompletionRequest =
        serde_json::from_value(input).unwrap();

    // Access via Deref (no .inner needed)
    assert_eq!(extended.model, "my-model");
    assert!(extended.stream);
}

/// Verify that a request with no extra fields works fine (extra map is empty).
#[test]
fn test_no_extra_fields() {
    let input = json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hello"}]
    });

    let extended: ExtendedChatCompletionRequest =
        serde_json::from_value(input).unwrap();

    assert!(extended.extra.is_empty());
    assert_eq!(extended.model, "test");
}
