//! Context for /v1/responses endpoint handlers
//!
//! Bundles all dependencies needed by responses handlers to avoid passing
//! 10+ parameters to every function.

use std::{collections::HashMap, sync::Arc};

use tokio::sync::RwLock;

use super::types::BackgroundTaskInfo;
use crate::{
    core::WorkerRegistry,
    data_connector::{ConversationItemStorage, ConversationStorage, ResponseStorage},
    mcp::McpManager,
    routers::grpc::{context::SharedComponents, pipeline::RequestPipeline},
};

/// Context for /v1/responses endpoint
///
/// All fields are Arc/shared references, so cloning this context is cheap.
#[derive(Clone)]
pub struct ResponsesContext {
    /// Chat pipeline for executing requests
    pub pipeline: Arc<RequestPipeline>,

    /// Shared components (tokenizer, parsers, worker_registry)
    pub components: Arc<SharedComponents>,

    /// Worker registry for validation
    pub worker_registry: Arc<WorkerRegistry>,

    /// Response storage backend
    pub response_storage: Arc<dyn ResponseStorage>,

    /// Conversation storage backend
    pub conversation_storage: Arc<dyn ConversationStorage>,

    /// Conversation item storage backend
    pub conversation_item_storage: Arc<dyn ConversationItemStorage>,

    /// MCP manager for tool support
    pub mcp_manager: Arc<McpManager>,

    /// Background task handles for cancellation support
    pub background_tasks: Arc<RwLock<HashMap<String, BackgroundTaskInfo>>>,
}

impl ResponsesContext {
    /// Create a new responses context
    pub fn new(
        pipeline: Arc<RequestPipeline>,
        components: Arc<SharedComponents>,
        worker_registry: Arc<WorkerRegistry>,
        response_storage: Arc<dyn ResponseStorage>,
        conversation_storage: Arc<dyn ConversationStorage>,
        conversation_item_storage: Arc<dyn ConversationItemStorage>,
        mcp_manager: Arc<McpManager>,
    ) -> Self {
        Self {
            pipeline,
            components,
            worker_registry,
            response_storage,
            conversation_storage,
            conversation_item_storage,
            mcp_manager,
            background_tasks: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}
