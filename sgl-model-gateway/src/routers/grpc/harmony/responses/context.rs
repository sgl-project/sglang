//! Context for Harmony Responses execution

use std::sync::Arc;

use crate::{
    data_connector::{ConversationItemStorage, ConversationStorage, ResponseStorage},
    mcp::McpManager,
    routers::grpc::{context::SharedComponents, pipeline::RequestPipeline},
};

/// Context for Harmony Responses execution with MCP tool support
///
/// Contains all dependencies needed for multi-turn Responses API execution.
/// Cheap to clone (all Arc references).
#[derive(Clone)]
pub(crate) struct HarmonyResponsesContext {
    /// Pipeline for executing Harmony requests
    pub pipeline: Arc<RequestPipeline>,

    /// Shared components (tokenizer, parsers)
    pub components: Arc<SharedComponents>,

    /// MCP manager for tool execution
    pub mcp_manager: Arc<McpManager>,

    /// Response storage for loading conversation history
    pub response_storage: Arc<dyn ResponseStorage>,

    /// Conversation storage for persisting conversations
    pub conversation_storage: Arc<dyn ConversationStorage>,

    /// Conversation item storage for persisting conversation items
    pub conversation_item_storage: Arc<dyn ConversationItemStorage>,
}

impl HarmonyResponsesContext {
    /// Create a new Harmony Responses context
    pub fn new(
        pipeline: Arc<RequestPipeline>,
        components: Arc<SharedComponents>,
        mcp_manager: Arc<McpManager>,
        response_storage: Arc<dyn ResponseStorage>,
        conversation_storage: Arc<dyn ConversationStorage>,
        conversation_item_storage: Arc<dyn ConversationItemStorage>,
    ) -> Self {
        Self {
            pipeline,
            components,
            mcp_manager,
            response_storage,
            conversation_storage,
            conversation_item_storage,
        }
    }
}
