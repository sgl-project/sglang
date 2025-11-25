//! Integration tests for OpenAI router pipeline
//!
//! Phase 1: Basic infrastructure tests

use std::sync::Arc;

use dashmap::DashMap;

use super::{
    context::{RequestType, SharedComponents},
    pipeline::RequestPipeline,
};
use crate::{
    core::CircuitBreaker,
    data_connector::{
        MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
    },
    mcp::{config::McpConfig, McpManager},
    protocols::chat::ChatCompletionRequest,
};

/// Helper to create test components
async fn create_test_components() -> Arc<SharedComponents> {
    let client = reqwest::Client::new();
    let circuit_breaker = Arc::new(CircuitBreaker::new());
    let model_cache = Arc::new(DashMap::new());

    // Create MCP manager with empty config (no servers)
    let mcp_config = McpConfig {
        servers: vec![],
        pool: Default::default(),
        proxy: None,
        warmup: vec![],
        inventory: Default::default(),
    };
    let mcp_manager = Arc::new(
        McpManager::new(mcp_config, 10)
            .await
            .expect("Failed to create MCP manager for tests"),
    );

    let response_storage = Arc::new(MemoryResponseStorage::new());
    let conversation_storage = Arc::new(MemoryConversationStorage::new());
    let conversation_item_storage = Arc::new(MemoryConversationItemStorage::new());

    Arc::new(SharedComponents {
        http_client: client,
        circuit_breaker,
        model_cache,
        mcp_manager,
        response_storage,
        conversation_storage,
        conversation_item_storage,
        worker_urls: vec!["http://localhost:8000".to_string()],
    })
}

#[tokio::test]
async fn test_pipeline_creation() {
    let pipeline = RequestPipeline::new(vec!["http://localhost:8000".to_string()]);
    // Pipeline should have 8 stages
    assert_eq!(pipeline.stage_count(), 8);
}

#[tokio::test]
async fn test_pipeline_stage_names() {
    let pipeline = RequestPipeline::new(vec!["http://localhost:8000".to_string()]);

    let expected_names = [
        "Validation",
        "ModelDiscovery",
        "ContextLoading",
        "RequestBuilding",
        "McpPreparation",
        "RequestExecution",
        "ResponseProcessing",
        "Persistence",
    ];

    let stage_names = pipeline.stage_names();
    assert_eq!(stage_names.len(), expected_names.len());
    for (idx, name) in stage_names.iter().enumerate() {
        assert_eq!(*name, expected_names[idx]);
    }
}

#[tokio::test]
async fn test_pipeline_executes_without_panic() {
    // Create a simple chat request
    use crate::protocols::chat::{ChatMessage, MessageContent};
    let request = ChatCompletionRequest {
        model: "gpt-4".to_string(),
        messages: vec![ChatMessage::User {
            content: MessageContent::Text("Hello".to_string()),
            name: None,
        }],
        stream: false,
        temperature: Some(0.7),
        max_completion_tokens: Some(100),
        top_p: None,
        n: None,
        stop: None,
        presence_penalty: None,
        frequency_penalty: None,
        logit_bias: None,
        tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
        logprobs: false,
        top_logprobs: None,
        response_format: None,
        ..Default::default()
    };

    let components = create_test_components().await;
    let pipeline = RequestPipeline::new(components.worker_urls.clone());

    // Execute pipeline (will fail at some stage since we don't have a real server,
    // but should not panic)
    let response = pipeline
        .execute(RequestType::Chat(Arc::new(request)), None, None, components)
        .await;

    // Response should be an error (since we're hitting localhost with no server)
    // but we just want to verify it doesn't panic
    assert!(response.status().is_client_error() || response.status().is_server_error());
}

#[tokio::test]
async fn test_shared_components_creation() {
    let components = create_test_components().await;
    assert_eq!(components.worker_urls.len(), 1);
    assert_eq!(components.worker_urls[0], "http://localhost:8000");
}
