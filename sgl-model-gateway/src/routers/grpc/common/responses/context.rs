//! Shared context for /v1/responses endpoint handlers
//!
//! This context is used by both regular and harmony response implementations.

use std::sync::{Arc, RwLock as StdRwLock};

use crate::{
    data_connector::{ConversationItemStorage, ConversationStorage, ResponseStorage},
    mcp::McpManager,
    routers::grpc::{context::SharedComponents, pipeline::RequestPipeline},
};

/// Context for /v1/responses endpoint
///
/// Used by both regular and harmony implementations.
/// All fields are Arc/shared references, so cloning this context is cheap.
#[derive(Clone)]
pub(crate) struct ResponsesContext {
    /// Chat pipeline for executing requests
    pub pipeline: Arc<RequestPipeline>,

    /// Shared components (tokenizer, parsers)
    pub components: Arc<SharedComponents>,

    /// Response storage backend
    pub response_storage: Arc<dyn ResponseStorage>,

    /// Conversation storage backend
    pub conversation_storage: Arc<dyn ConversationStorage>,

    /// Conversation item storage backend
    pub conversation_item_storage: Arc<dyn ConversationItemStorage>,

    /// MCP manager for tool support
    pub mcp_manager: Arc<McpManager>,

    /// Server keys for MCP tools requested in this context
    pub requested_servers: Arc<StdRwLock<Vec<String>>>,
}

impl ResponsesContext {
    /// Create a new responses context
    pub fn new(
        pipeline: Arc<RequestPipeline>,
        components: Arc<SharedComponents>,
        response_storage: Arc<dyn ResponseStorage>,
        conversation_storage: Arc<dyn ConversationStorage>,
        conversation_item_storage: Arc<dyn ConversationItemStorage>,
        mcp_manager: Arc<McpManager>,
    ) -> Self {
        Self {
            pipeline,
            components,
            response_storage,
            conversation_storage,
            conversation_item_storage,
            mcp_manager,
            requested_servers: Arc::new(StdRwLock::new(Vec::new())),
        }
    }
}
