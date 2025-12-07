use std::collections::HashMap;

use serde_json::{from_str, to_string, Number, Value};
use sgl_model_gateway::protocols::{
    common::{GenerationRequest, StringOrArray, UsageInfo},
    rerank::{RerankRequest, RerankResponse, RerankResult, V1RerankReqInput},
};
use validator::Validate;

#[test]
fn test_rerank_request_serialization() {
    let request = RerankRequest {
        query: "test query".to_string(),
        documents: vec!["doc1".to_string(), "doc2".to_string()],
        model: "test-model".to_string(),
        top_k: Some(5),
        return_documents: true,
        rid: Some(StringOrArray::String("req-123".to_string())),
        user: Some("user-456".to_string()),
    };

    let serialized = to_string(&request).unwrap();
    let deserialized: RerankRequest = from_str(&serialized).unwrap();

    assert_eq!(deserialized.query, request.query);
    assert_eq!(deserialized.documents, request.documents);
    assert_eq!(deserialized.model, request.model);
    assert_eq!(deserialized.top_k, request.top_k);
    assert_eq!(deserialized.return_documents, request.return_documents);
    assert_eq!(deserialized.rid, request.rid);
    assert_eq!(deserialized.user, request.user);
}

#[test]
fn test_rerank_request_deserialization_with_defaults() {
    let json = r#"{
        "query": "test query",
        "documents": ["doc1", "doc2"]
    }"#;

    let request: RerankRequest = from_str(json).unwrap();

    assert_eq!(request.query, "test query");
    assert_eq!(request.documents, vec!["doc1", "doc2"]);
    assert_eq!(request.model, "unknown");
    assert_eq!(request.top_k, None);
    assert!(request.return_documents);
    assert_eq!(request.rid, None);
    assert_eq!(request.user, None);
}

#[test]
fn test_rerank_request_validation_success() {
    let request = RerankRequest {
        query: "valid query".to_string(),
        documents: vec!["doc1".to_string(), "doc2".to_string()],
        model: "test-model".to_string(),
        top_k: Some(2),
        return_documents: true,
        rid: None,
        user: None,
    };

    assert!(request.validate().is_ok());
}

#[test]
fn test_rerank_request_validation_empty_query() {
    let request = RerankRequest {
        query: "".to_string(),
        documents: vec!["doc1".to_string()],
        model: "test-model".to_string(),
        top_k: None,
        return_documents: true,
        rid: None,
        user: None,
    };

    let result = request.validate();
    assert!(result.is_err(), "Should reject empty query");
}

#[test]
fn test_rerank_request_validation_whitespace_query() {
    let request = RerankRequest {
        query: "   ".to_string(),
        documents: vec!["doc1".to_string()],
        model: "test-model".to_string(),
        top_k: None,
        return_documents: true,
        rid: None,
        user: None,
    };

    let result = request.validate();
    assert!(result.is_err(), "Should reject whitespace-only query");
}

#[test]
fn test_rerank_request_validation_empty_documents() {
    let request = RerankRequest {
        query: "test query".to_string(),
        documents: vec![],
        model: "test-model".to_string(),
        top_k: None,
        return_documents: true,
        rid: None,
        user: None,
    };

    let result = request.validate();
    assert!(result.is_err(), "Should reject empty documents list");
}

#[test]
fn test_rerank_request_validation_top_k_zero() {
    let request = RerankRequest {
        query: "test query".to_string(),
        documents: vec!["doc1".to_string(), "doc2".to_string()],
        model: "test-model".to_string(),
        top_k: Some(0),
        return_documents: true,
        rid: None,
        user: None,
    };

    let result = request.validate();
    assert!(result.is_err(), "Should reject top_k of zero");
}

#[test]
fn test_rerank_request_validation_top_k_greater_than_docs() {
    let request = RerankRequest {
        query: "test query".to_string(),
        documents: vec!["doc1".to_string(), "doc2".to_string()],
        model: "test-model".to_string(),
        top_k: Some(5),
        return_documents: true,
        rid: None,
        user: None,
    };

    // This should pass but log a warning
    assert!(request.validate().is_ok());
}

#[test]
fn test_rerank_request_effective_top_k() {
    let request = RerankRequest {
        query: "test query".to_string(),
        documents: vec!["doc1".to_string(), "doc2".to_string(), "doc3".to_string()],
        model: "test-model".to_string(),
        top_k: Some(2),
        return_documents: true,
        rid: None,
        user: None,
    };

    assert_eq!(request.effective_top_k(), 2);
}

#[test]
fn test_rerank_request_effective_top_k_none() {
    let request = RerankRequest {
        query: "test query".to_string(),
        documents: vec!["doc1".to_string(), "doc2".to_string(), "doc3".to_string()],
        model: "test-model".to_string(),
        top_k: None,
        return_documents: true,
        rid: None,
        user: None,
    };

    assert_eq!(request.effective_top_k(), 3);
}

#[test]
fn test_rerank_response_creation() {
    let results = vec![
        RerankResult {
            score: 0.8,
            document: Some("doc1".to_string()),
            index: 0,
            meta_info: None,
        },
        RerankResult {
            score: 0.6,
            document: Some("doc2".to_string()),
            index: 1,
            meta_info: None,
        },
    ];

    let response = RerankResponse::new(
        results.clone(),
        "test-model".to_string(),
        Some(StringOrArray::String("req-123".to_string())),
    );

    assert_eq!(response.results.len(), 2);
    assert_eq!(response.model, "test-model");
    assert_eq!(
        response.id,
        Some(StringOrArray::String("req-123".to_string()))
    );
    assert_eq!(response.object, "rerank");
    assert!(response.created > 0);
}

#[test]
fn test_rerank_response_serialization() {
    let results = vec![RerankResult {
        score: 0.8,
        document: Some("doc1".to_string()),
        index: 0,
        meta_info: None,
    }];

    let response = RerankResponse::new(
        results,
        "test-model".to_string(),
        Some(StringOrArray::String("req-123".to_string())),
    );

    let serialized = to_string(&response).unwrap();
    let deserialized: RerankResponse = from_str(&serialized).unwrap();

    assert_eq!(deserialized.results.len(), response.results.len());
    assert_eq!(deserialized.model, response.model);
    assert_eq!(deserialized.id, response.id);
    assert_eq!(deserialized.object, response.object);
}

#[test]
fn test_rerank_response_apply_top_k() {
    let results = vec![
        RerankResult {
            score: 0.8,
            document: Some("doc1".to_string()),
            index: 0,
            meta_info: None,
        },
        RerankResult {
            score: 0.6,
            document: Some("doc2".to_string()),
            index: 1,
            meta_info: None,
        },
        RerankResult {
            score: 0.4,
            document: Some("doc3".to_string()),
            index: 2,
            meta_info: None,
        },
    ];

    let mut response = RerankResponse::new(
        results,
        "test-model".to_string(),
        Some(StringOrArray::String("req-123".to_string())),
    );

    response.apply_top_k(2);

    assert_eq!(response.results.len(), 2);
    assert_eq!(response.results[0].score, 0.8);
    assert_eq!(response.results[1].score, 0.6);
}

#[test]
fn test_rerank_response_apply_top_k_larger_than_results() {
    let results = vec![RerankResult {
        score: 0.8,
        document: Some("doc1".to_string()),
        index: 0,
        meta_info: None,
    }];

    let mut response = RerankResponse::new(
        results,
        "test-model".to_string(),
        Some(StringOrArray::String("req-123".to_string())),
    );

    response.apply_top_k(5);

    assert_eq!(response.results.len(), 1);
}

#[test]
fn test_rerank_response_drop_documents() {
    let results = vec![RerankResult {
        score: 0.8,
        document: Some("doc1".to_string()),
        index: 0,
        meta_info: None,
    }];
    let mut response = RerankResponse::new(
        results,
        "test-model".to_string(),
        Some(StringOrArray::String("req-123".to_string())),
    );

    response.drop_documents();

    assert_eq!(response.results[0].document, None);
}

#[test]
fn test_rerank_result_serialization() {
    let result = RerankResult {
        score: 0.85,
        document: Some("test document".to_string()),
        index: 42,
        meta_info: Some(HashMap::from([
            ("confidence".to_string(), Value::String("high".to_string())),
            (
                "processing_time".to_string(),
                Value::Number(Number::from(150)),
            ),
        ])),
    };

    let serialized = to_string(&result).unwrap();
    let deserialized: RerankResult = from_str(&serialized).unwrap();

    assert_eq!(deserialized.score, result.score);
    assert_eq!(deserialized.document, result.document);
    assert_eq!(deserialized.index, result.index);
    assert_eq!(deserialized.meta_info, result.meta_info);
}

#[test]
fn test_rerank_result_serialization_without_document() {
    let result = RerankResult {
        score: 0.85,
        document: None,
        index: 42,
        meta_info: None,
    };

    let serialized = to_string(&result).unwrap();
    let deserialized: RerankResult = from_str(&serialized).unwrap();

    assert_eq!(deserialized.score, result.score);
    assert_eq!(deserialized.document, result.document);
    assert_eq!(deserialized.index, result.index);
    assert_eq!(deserialized.meta_info, result.meta_info);
}

#[test]
fn test_v1_rerank_req_input_serialization() {
    let v1_input = V1RerankReqInput {
        query: "test query".to_string(),
        documents: vec!["doc1".to_string(), "doc2".to_string()],
    };

    let serialized = to_string(&v1_input).unwrap();
    let deserialized: V1RerankReqInput = from_str(&serialized).unwrap();

    assert_eq!(deserialized.query, v1_input.query);
    assert_eq!(deserialized.documents, v1_input.documents);
}

#[test]
fn test_v1_to_rerank_request_conversion() {
    let v1_input = V1RerankReqInput {
        query: "test query".to_string(),
        documents: vec!["doc1".to_string(), "doc2".to_string()],
    };

    let request: RerankRequest = v1_input.into();

    assert_eq!(request.query, "test query");
    assert_eq!(request.documents, vec!["doc1", "doc2"]);
    assert_eq!(request.model, "unknown");
    assert_eq!(request.top_k, None);
    assert!(request.return_documents);
    assert_eq!(request.rid, None);
    assert_eq!(request.user, None);
}

#[test]
fn test_rerank_request_generation_request_trait() {
    let request = RerankRequest {
        query: "test query".to_string(),
        documents: vec!["doc1".to_string()],
        model: "test-model".to_string(),
        top_k: None,
        return_documents: true,
        rid: None,
        user: None,
    };

    assert_eq!(request.get_model(), Some("test-model"));
    assert!(!request.is_stream());
    assert_eq!(request.extract_text_for_routing(), "test query");
}

#[test]
fn test_rerank_request_very_long_query() {
    let long_query = "a".repeat(100000);
    let request = RerankRequest {
        query: long_query,
        documents: vec!["doc1".to_string()],
        model: "test-model".to_string(),
        top_k: None,
        return_documents: true,
        rid: None,
        user: None,
    };

    assert!(request.validate().is_ok());
}

#[test]
fn test_rerank_request_many_documents() {
    let documents: Vec<String> = (0..1000).map(|i| format!("doc{}", i)).collect();
    let request = RerankRequest {
        query: "test query".to_string(),
        documents,
        model: "test-model".to_string(),
        top_k: Some(100),
        return_documents: true,
        rid: None,
        user: None,
    };

    assert!(request.validate().is_ok());
    assert_eq!(request.effective_top_k(), 100);
}

#[test]
fn test_rerank_request_special_characters() {
    let request = RerankRequest {
        query: "query with émojis 🚀 and unicode: 测试".to_string(),
        documents: vec![
            "doc with émojis 🎉".to_string(),
            "doc with unicode: 测试".to_string(),
        ],
        model: "test-model".to_string(),
        top_k: None,
        return_documents: true,
        rid: Some(StringOrArray::String("req-🚀-123".to_string())),
        user: Some("user-🎉-456".to_string()),
    };

    assert!(request.validate().is_ok());
}

#[test]
fn test_rerank_request_rid_array() {
    let request = RerankRequest {
        query: "test query".to_string(),
        documents: vec!["doc1".to_string()],
        model: "test-model".to_string(),
        top_k: None,
        return_documents: true,
        rid: Some(StringOrArray::Array(vec![
            "req1".to_string(),
            "req2".to_string(),
        ])),
        user: None,
    };

    assert!(request.validate().is_ok());
}

#[test]
fn test_rerank_response_with_usage_info() {
    let results = vec![RerankResult {
        score: 0.8,
        document: Some("doc1".to_string()),
        index: 0,
        meta_info: None,
    }];

    let mut response = RerankResponse::new(
        results,
        "test-model".to_string(),
        Some(StringOrArray::String("req-123".to_string())),
    );

    response.usage = Some(UsageInfo {
        prompt_tokens: 100,
        completion_tokens: 50,
        total_tokens: 150,
        reasoning_tokens: None,
        prompt_tokens_details: None,
    });

    let serialized = to_string(&response).unwrap();
    let deserialized: RerankResponse = from_str(&serialized).unwrap();

    assert!(deserialized.usage.is_some());
    let usage = deserialized.usage.unwrap();
    assert_eq!(usage.prompt_tokens, 100);
    assert_eq!(usage.completion_tokens, 50);
    assert_eq!(usage.total_tokens, 150);
}

#[test]
fn test_full_rerank_workflow() {
    // Create request
    let request = RerankRequest {
        query: "machine learning".to_string(),
        documents: vec![
            "Introduction to machine learning algorithms".to_string(),
            "Deep learning for computer vision".to_string(),
            "Natural language processing basics".to_string(),
            "Statistics and probability theory".to_string(),
        ],
        model: "rerank-model".to_string(),
        top_k: Some(2),
        return_documents: true,
        rid: Some(StringOrArray::String("req-123".to_string())),
        user: Some("user-456".to_string()),
    };

    // Validate request
    assert!(request.validate().is_ok());

    // Simulate reranking results (in real scenario, this would come from the model)
    let results = vec![
        RerankResult {
            score: 0.95,
            document: Some("Introduction to machine learning algorithms".to_string()),
            index: 0,
            meta_info: None,
        },
        RerankResult {
            score: 0.87,
            document: Some("Deep learning for computer vision".to_string()),
            index: 1,
            meta_info: None,
        },
        RerankResult {
            score: 0.72,
            document: Some("Natural language processing basics".to_string()),
            index: 2,
            meta_info: None,
        },
        RerankResult {
            score: 0.45,
            document: Some("Statistics and probability theory".to_string()),
            index: 3,
            meta_info: None,
        },
    ];

    // Create response
    let mut response = RerankResponse::new(results, request.model.clone(), request.rid.clone());

    // Apply top_k
    response.apply_top_k(request.effective_top_k());

    assert_eq!(response.results.len(), 2);
    assert_eq!(response.results[0].score, 0.95);
    assert_eq!(response.results[0].index, 0);
    assert_eq!(response.results[1].score, 0.87);
    assert_eq!(response.results[1].index, 1);
    assert_eq!(response.model, "rerank-model");

    // Serialize and deserialize
    let serialized = to_string(&response).unwrap();
    let deserialized: RerankResponse = from_str(&serialized).unwrap();
    assert_eq!(deserialized.results.len(), 2);
    assert_eq!(deserialized.model, response.model);
}
