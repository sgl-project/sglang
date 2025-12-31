//! Partial JSON Parser Tests
//!
//! Tests for the partial JSON parser with allow_partial_strings flag behavior

use sgl_model_gateway::tool_parser::partial_json::PartialJson;

#[test]
fn test_partial_string_flag_disallows_incomplete_strings() {
    // Test case from the bug report: {"name": "
    // With allow_partial_strings=false, should return {} (stop before incomplete string)
    let parser = PartialJson::new(32, true);
    let input = r#"{"name": ""#;

    let result = parser.parse_value(input, false);
    assert!(result.is_ok());

    let (obj, consumed) = result.unwrap();

    // Should parse just the opening brace and stop at the incomplete string
    assert!(obj.is_object());
    let obj_map = obj.as_object().unwrap();

    // Should have empty object (stopped before parsing incomplete "name" key)
    assert!(
        obj_map.is_empty() || !obj_map.contains_key("name"),
        "Should not parse incomplete string key, got: {:?}",
        obj_map
    );

    // Should consume characters up to the incomplete string
    assert!(consumed <= input.len());
}

#[test]
fn test_partial_string_flag_allows_incomplete_strings() {
    // Test case: {"name": "
    // With allow_partial_strings=true, should parse the incomplete string
    let parser = PartialJson::new(32, true);
    let input = r#"{"name": ""#;

    let result = parser.parse_value(input, true);
    assert!(result.is_ok());

    let (obj, consumed) = result.unwrap();

    // Should parse the object with incomplete string value
    assert!(obj.is_object());
    let obj_map = obj.as_object().unwrap();

    // With allow_partial_strings=true, should parse "name" key with empty string value
    assert!(
        obj_map.contains_key("name"),
        "Should parse incomplete string with allow_partial_strings=true"
    );

    assert_eq!(consumed, input.len());
}

#[test]
fn test_partial_string_flag_complete_json() {
    // Test case: {"name": "test"}
    // Both flags should parse complete JSON the same way
    let input = r#"{"name": "test"}"#;

    let parser = PartialJson::new(32, true);
    let result1 = parser.parse_value(input, false);
    assert!(result1.is_ok());
    let (obj1, consumed1) = result1.unwrap();

    let result2 = parser.parse_value(input, true);
    assert!(result2.is_ok());
    let (obj2, consumed2) = result2.unwrap();

    // Both should parse the same complete JSON
    assert_eq!(obj1, obj2);
    assert_eq!(consumed1, consumed2);
    assert_eq!(consumed1, input.len());

    // Check the parsed value
    assert!(obj1.is_object());
    let obj_map = obj1.as_object().unwrap();
    assert_eq!(obj_map.get("name").and_then(|v| v.as_str()), Some("test"));
}

#[test]
fn test_backward_compatibility_default() {
    // Test that default PartialJson still allows partial strings (backward compatible)
    let parser = PartialJson::default();
    let input = r#"{"name": ""#;

    let result = parser.parse_value(input, true);
    assert!(result.is_ok());

    let (obj, _) = result.unwrap();
    assert!(obj.is_object());

    // Default behavior should allow partial strings
    let obj_map = obj.as_object().unwrap();
    assert!(
        obj_map.contains_key("name"),
        "Default should allow partial strings for backward compatibility"
    );
}

#[test]
fn test_partial_string_in_nested_object() {
    // Test case: {"tool": {"name": "
    let parser = PartialJson::new(32, true);
    let input = r#"{"tool": {"name": ""#;

    let result = parser.parse_value(input, false);
    assert!(result.is_ok());

    let (obj, _) = result.unwrap();
    assert!(obj.is_object());

    // With allow_partial_strings=false, should stop before incomplete nested string
    let obj_map = obj.as_object().unwrap();
    if let Some(tool) = obj_map.get("tool") {
        if let Some(tool_map) = tool.as_object() {
            assert!(
                !tool_map.contains_key("name")
                    || tool_map.get("name").and_then(|v| v.as_str()).is_none(),
                "Should not parse incomplete nested string"
            );
        }
    }
}

#[test]
fn test_bug_fix_exact_scenario() {
    // This test verifies the exact bug scenario from the issue:
    // buffer = "{\"name\": \""
    // flags = Allow.ALL & ~Allow.STR
    // Python returns: Parsed object: {}, consumed length: 10

    let parser = PartialJson::new(32, true);
    let input = r#"{"name": ""#;

    let result = parser.parse_value(input, false);
    assert!(result.is_ok());

    let (obj, consumed) = result.unwrap();

    // Should return empty object (not {"name": null} or {"name": ""})
    assert!(obj.is_object());
    let obj_map = obj.as_object().unwrap();
    assert!(
        obj_map.is_empty(),
        "Expected empty object, got: {:?}. This matches Python behavior with Allow.ALL & ~Allow.STR",
        obj_map
    );

    // Should consume all characters (10 bytes)
    assert_eq!(consumed, 10, "Should consume all 10 characters");
}
