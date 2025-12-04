use serde_json::{from_str, json, to_string};
use sgl_model_gateway::protocols::{common::GenerationRequest, embedding::EmbeddingRequest};

#[test]
fn test_embedding_request_serialization_string_input() {
    let req = EmbeddingRequest {
        model: "test-emb".to_string(),
        input: json!("hello"),
        encoding_format: Some("float".to_string()),
        user: Some("user-1".to_string()),
        dimensions: Some(128),
        rid: Some("rid-123".to_string()),
    };

    let serialized = to_string(&req).unwrap();
    let deserialized: EmbeddingRequest = from_str(&serialized).unwrap();

    assert_eq!(deserialized.model, req.model);
    assert_eq!(deserialized.input, req.input);
    assert_eq!(deserialized.encoding_format, req.encoding_format);
    assert_eq!(deserialized.user, req.user);
    assert_eq!(deserialized.dimensions, req.dimensions);
    assert_eq!(deserialized.rid, req.rid);
}

#[test]
fn test_embedding_request_serialization_array_input() {
    let req = EmbeddingRequest {
        model: "test-emb".to_string(),
        input: json!(["a", "b", "c"]),
        encoding_format: None,
        user: None,
        dimensions: None,
        rid: None,
    };

    let serialized = to_string(&req).unwrap();
    let de: EmbeddingRequest = from_str(&serialized).unwrap();
    assert_eq!(de.model, req.model);
    assert_eq!(de.input, req.input);
}

#[test]
fn test_embedding_generation_request_trait_string() {
    let req = EmbeddingRequest {
        model: "emb-model".to_string(),
        input: json!("hello"),
        encoding_format: None,
        user: None,
        dimensions: None,
        rid: None,
    };
    assert!(!req.is_stream());
    assert_eq!(req.get_model(), Some("emb-model"));
    assert_eq!(req.extract_text_for_routing(), "hello");
}

#[test]
fn test_embedding_generation_request_trait_array() {
    let req = EmbeddingRequest {
        model: "emb-model".to_string(),
        input: json!(["hello", "world"]),
        encoding_format: None,
        user: None,
        dimensions: None,
        rid: None,
    };
    assert_eq!(req.extract_text_for_routing(), "hello world");
}

#[test]
fn test_embedding_generation_request_trait_non_text() {
    let req = EmbeddingRequest {
        model: "emb-model".to_string(),
        input: json!({"tokens": [1, 2, 3]}),
        encoding_format: None,
        user: None,
        dimensions: None,
        rid: None,
    };
    assert_eq!(req.extract_text_for_routing(), "");
}

#[test]
fn test_embedding_generation_request_trait_mixed_array_ignores_nested() {
    let req = EmbeddingRequest {
        model: "emb-model".to_string(),
        input: json!(["a", ["b", "c"], 123, {"k": "v"}]),
        encoding_format: None,
        user: None,
        dimensions: None,
        rid: None,
    };
    // Only top-level string elements are extracted
    assert_eq!(req.extract_text_for_routing(), "a");
}
