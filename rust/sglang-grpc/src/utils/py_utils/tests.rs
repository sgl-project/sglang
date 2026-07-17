#[test]
fn json_values_are_preserved() {
    let value = serde_json::json!({"nested": [1, true, "value"]});
    assert_eq!(value["nested"][2], "value");
}
