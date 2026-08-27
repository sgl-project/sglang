use super::*;

#[test]
fn fallback_string_is_json_encoded() {
    let encoded = json_encode_string("<Foo at 0x123>");
    let decoded: String = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, "<Foo at 0x123>");
}
