//! Typed workflow engines collection
//!
//! This module provides a collection of typed workflow engines for different workflow types.
//! Each workflow type has its own engine with compile-time type safety.

use std::sync::Arc;

use super::{
    create_external_worker_workflow, create_local_worker_workflow,
    create_mcp_registration_workflow, create_tokenizer_registration_workflow,
    create_wasm_module_registration_workflow, create_wasm_module_removal_workflow,
    create_worker_removal_workflow, create_worker_update_workflow, ExternalWorkerWorkflowData,
    LocalWorkerWorkflowData, McpWorkflowData, TokenizerWorkflowData, WasmRegistrationWorkflowData,
    WasmRemovalWorkflowData, WorkerRemovalWorkflowData, WorkerUpdateWorkflowData,
};
use crate::{
    config::RouterConfig,
    workflow::{EventSubscriber, InMemoryStore, WorkflowEngine},
};

/// Type alias for local worker workflow engine
pub type LocalWorkerEngine =
    WorkflowEngine<LocalWorkerWorkflowData, InMemoryStore<LocalWorkerWorkflowData>>;

/// Type alias for external worker workflow engine
pub type ExternalWorkerEngine =
    WorkflowEngine<ExternalWorkerWorkflowData, InMemoryStore<ExternalWorkerWorkflowData>>;

/// Type alias for worker removal workflow engine
pub type WorkerRemovalEngine =
    WorkflowEngine<WorkerRemovalWorkflowData, InMemoryStore<WorkerRemovalWorkflowData>>;

/// Type alias for worker update workflow engine
pub type WorkerUpdateEngine =
    WorkflowEngine<WorkerUpdateWorkflowData, InMemoryStore<WorkerUpdateWorkflowData>>;

/// Type alias for MCP registration workflow engine
pub type McpEngine = WorkflowEngine<McpWorkflowData, InMemoryStore<McpWorkflowData>>;

/// Type alias for tokenizer registration workflow engine
pub type TokenizerEngine =
    WorkflowEngine<TokenizerWorkflowData, InMemoryStore<TokenizerWorkflowData>>;

/// Type alias for WASM registration workflow engine
pub type WasmRegistrationEngine =
    WorkflowEngine<WasmRegistrationWorkflowData, InMemoryStore<WasmRegistrationWorkflowData>>;

/// Type alias for WASM removal workflow engine
pub type WasmRemovalEngine =
    WorkflowEngine<WasmRemovalWorkflowData, InMemoryStore<WasmRemovalWorkflowData>>;

/// Collection of typed workflow engines
///
/// Each workflow type has its own engine with compile-time type safety.
/// This replaces the old `WorkflowEngine<AnyWorkflowData, ...>` approach.
#[derive(Clone, Debug)]
pub struct WorkflowEngines {
    /// Engine for local worker registration workflows
    pub local_worker: Arc<LocalWorkerEngine>,
    /// Engine for external worker registration workflows
    pub external_worker: Arc<ExternalWorkerEngine>,
    /// Engine for worker removal workflows
    pub worker_removal: Arc<WorkerRemovalEngine>,
    /// Engine for worker update workflows
    pub worker_update: Arc<WorkerUpdateEngine>,
    /// Engine for MCP server registration workflows
    pub mcp: Arc<McpEngine>,
    /// Engine for tokenizer registration workflows
    pub tokenizer: Arc<TokenizerEngine>,
    /// Engine for WASM module registration workflows
    pub wasm_registration: Arc<WasmRegistrationEngine>,
    /// Engine for WASM module removal workflows
    pub wasm_removal: Arc<WasmRemovalEngine>,
}

impl WorkflowEngines {
    /// Create and initialize all workflow engines with their workflow definitions
    pub fn new(router_config: &RouterConfig) -> Self {
        // Create local worker engine
        let local_worker = WorkflowEngine::new();
        local_worker
            .register_workflow(create_local_worker_workflow(router_config))
            .expect("local_worker_registration workflow should be valid");

        // Create external worker engine
        let external_worker = WorkflowEngine::new();
        external_worker
            .register_workflow(create_external_worker_workflow())
            .expect("external_worker_registration workflow should be valid");

        // Create worker removal engine
        let worker_removal = WorkflowEngine::new();
        worker_removal
            .register_workflow(create_worker_removal_workflow())
            .expect("worker_removal workflow should be valid");

        // Create worker update engine
        let worker_update = WorkflowEngine::new();
        worker_update
            .register_workflow(create_worker_update_workflow())
            .expect("worker_update workflow should be valid");

        // Create MCP engine
        let mcp = WorkflowEngine::new();
        mcp.register_workflow(create_mcp_registration_workflow())
            .expect("mcp_registration workflow should be valid");

        // Create tokenizer engine
        let tokenizer = WorkflowEngine::new();
        tokenizer
            .register_workflow(create_tokenizer_registration_workflow())
            .expect("tokenizer_registration workflow should be valid");

        // Create WASM registration engine
        let wasm_registration = WorkflowEngine::new();
        wasm_registration
            .register_workflow(create_wasm_module_registration_workflow())
            .expect("wasm_module_registration workflow should be valid");

        // Create WASM removal engine
        let wasm_removal = WorkflowEngine::new();
        wasm_removal
            .register_workflow(create_wasm_module_removal_workflow())
            .expect("wasm_module_removal workflow should be valid");

        Self {
            local_worker: Arc::new(local_worker),
            external_worker: Arc::new(external_worker),
            worker_removal: Arc::new(worker_removal),
            worker_update: Arc::new(worker_update),
            mcp: Arc::new(mcp),
            tokenizer: Arc::new(tokenizer),
            wasm_registration: Arc::new(wasm_registration),
            wasm_removal: Arc::new(wasm_removal),
        }
    }

    /// Subscribe an event subscriber to all workflow engines
    pub async fn subscribe_all<S: EventSubscriber + 'static>(&self, subscriber: Arc<S>) {
        self.local_worker
            .event_bus()
            .subscribe(subscriber.clone())
            .await;
        self.external_worker
            .event_bus()
            .subscribe(subscriber.clone())
            .await;
        self.worker_removal
            .event_bus()
            .subscribe(subscriber.clone())
            .await;
        self.worker_update
            .event_bus()
            .subscribe(subscriber.clone())
            .await;
        self.mcp.event_bus().subscribe(subscriber.clone()).await;
        self.tokenizer
            .event_bus()
            .subscribe(subscriber.clone())
            .await;
        self.wasm_registration
            .event_bus()
            .subscribe(subscriber.clone())
            .await;
        self.wasm_removal.event_bus().subscribe(subscriber).await;
    }
}
