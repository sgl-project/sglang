//! MCP server registration workflow steps
//!
//! Each step is atomic and performs a single operation in the MCP server registration process.
//! Updated for flat manager architecture - single McpManager manages all clients directly.
//!
//! Workflow order:
//! 1. ConnectMcpServer - Establish connection to MCP server using McpManager::connect_server()
//! 2. DiscoverMcpInventory - Discover and cache inventory using McpManager::load_server_inventory()
//! 3. RegisterMcpServer - Register McpClient in McpManager's client map

use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use rmcp::{service::RunningService, RoleClient};
use tracing::{debug, error, info, warn};

use crate::{
    app_context::AppContext,
    core::workflow::*,
    mcp::{config::McpServerConfig, manager::McpManager},
};

/// MCP server connection configuration
#[derive(Debug, Clone)]
pub struct McpServerConfigRequest {
    /// Server name (unique identifier)
    pub name: String,
    /// Server configuration (transport, proxy, etc.)
    pub config: McpServerConfig,
}

impl McpServerConfigRequest {
    /// Check if this server is required for router startup
    pub fn is_required(&self) -> bool {
        self.config.required
    }
}

/// Step 1: Connect to MCP server
///
/// This step establishes a connection to the MCP server using the flat manager architecture.
/// The connection is retried aggressively (100 attempts) with a long timeout (2 hours)
/// to handle slow-starting servers or network issues.
pub struct ConnectMcpServerStep;

#[async_trait]
impl StepExecutor for ConnectMcpServerStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config_request: Arc<McpServerConfigRequest> = context
            .get("mcp_server_config")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("mcp_server_config".to_string()))?;
        let app_context: Arc<AppContext> = context
            .get("app_context")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;

        debug!("Connecting to MCP server: {}", config_request.name);

        // Get proxy config from router_config if available, otherwise fall back to env
        let proxy_config = app_context
            .router_config
            .mcp_config
            .as_ref()
            .and_then(|cfg| cfg.proxy.as_ref());

        // Connect to MCP server
        let client = McpManager::connect_server(&config_request.config, proxy_config)
            .await
            .map_err(|e| WorkflowError::StepFailed {
                step_id: StepId::new("connect_mcp_server"),
                message: format!(
                    "Failed to connect to MCP server {}: {}",
                    config_request.name, e
                ),
            })?;

        info!(
            "Successfully connected to MCP server: {}",
            config_request.name
        );

        // Store client in context (context.set() will wrap in Arc)
        context.set("mcp_client", client);

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true // Connection failures are retryable
    }
}

/// Step 2: Discover MCP inventory (tools, prompts, resources)
///
/// This step queries the MCP server for its capabilities using McpManager::load_server_inventory().
/// - Tools: Available function calls
/// - Prompts: Reusable prompt templates
/// - Resources: Accessible files/data
pub struct DiscoverMcpInventoryStep;

#[async_trait]
impl StepExecutor for DiscoverMcpInventoryStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        use rmcp::{service::RunningService, RoleClient};

        let config_request: Arc<McpServerConfigRequest> = context
            .get("mcp_server_config")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("mcp_server_config".to_string()))?;
        let app_context: Arc<AppContext> = context
            .get("app_context")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let mcp_client: Arc<RunningService<RoleClient, ()>> = context
            .get("mcp_client")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("mcp_client".to_string()))?;

        debug!(
            "Discovering inventory for MCP server: {}",
            config_request.name
        );

        // Get shared ToolInventory from McpManager
        let mcp_manager =
            app_context
                .mcp_manager
                .get()
                .ok_or_else(|| WorkflowError::StepFailed {
                    step_id: StepId::new("discover_mcp_inventory"),
                    message: "MCP manager not initialized".to_string(),
                })?;

        let inventory = mcp_manager.inventory();

        // Use the public load_server_inventory method
        McpManager::load_server_inventory(&inventory, &config_request.name, &mcp_client).await;

        info!("Completed inventory discovery for {}", config_request.name);

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true // Discovery failures are retryable
    }
}

/// Step 3: Register MCP server in manager
///
/// This step adds the MCP client to the McpManager's client map so it can be
/// used for tool calls and inventory management.
pub struct RegisterMcpServerStep;

#[async_trait]
impl StepExecutor for RegisterMcpServerStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        use rmcp::{service::RunningService, RoleClient};

        let config_request: Arc<McpServerConfigRequest> = context
            .get("mcp_server_config")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("mcp_server_config".to_string()))?;
        let app_context: Arc<AppContext> = context
            .get("app_context")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let mcp_client: Arc<RunningService<RoleClient, ()>> = context
            .get("mcp_client")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("mcp_client".to_string()))?;

        debug!("Registering MCP server: {}", config_request.name);

        // Get MCP manager from app context
        let mcp_manager =
            app_context
                .mcp_manager
                .get()
                .ok_or_else(|| WorkflowError::StepFailed {
                    step_id: StepId::new("register_mcp_server"),
                    message: "MCP manager not initialized".to_string(),
                })?;

        // Register the client in the manager's client map
        mcp_manager.register_static_server(config_request.name.clone(), mcp_client);

        info!("Registered MCP server: {}", config_request.name);

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false // Registration is a simple operation, not retryable
    }
}

/// Step 4: Validate registration based on required flag
///
/// This step checks if the server is marked as required. If the server is required
/// but wasn't successfully registered (client not in context), this step fails the workflow.
/// For optional servers, this step always succeeds, allowing the workflow to complete
/// even if earlier steps failed.
pub struct ValidateRegistrationStep;

#[async_trait]
impl StepExecutor for ValidateRegistrationStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config_request: Arc<McpServerConfigRequest> = context
            .get("mcp_server_config")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("mcp_server_config".to_string()))?;

        let client_registered = context
            .get::<RunningService<RoleClient, ()>>("mcp_client")
            .is_some();

        if client_registered {
            info!(
                "MCP server '{}' registered successfully",
                config_request.name
            );
            return Ok(StepResult::Success);
        }

        if config_request.is_required() {
            error!(
                "Required MCP server '{}' failed to register",
                config_request.name
            );
            Err(WorkflowError::StepFailed {
                step_id: StepId::new("validate_registration"),
                message: format!(
                    "Required MCP server '{}' failed to register",
                    config_request.name
                ),
            })
        } else {
            warn!(
                "Optional MCP server '{}' failed to register, continuing workflow",
                config_request.name
            );
            Ok(StepResult::Success)
        }
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}

/// Create MCP server registration workflow
///
/// This workflow adapts its failure behavior based on the `required` field in the server config:
/// - If `required == true`: Uses FailWorkflow - router startup fails if server cannot be reached
/// - If `required == false` (default): Uses ContinueNextStep - logs warning but continues
///
/// Workflow configuration:
/// - ConnectMcpServer: 100 retries, 2hr timeout (aggressive retry for slow servers)
/// - DiscoverMcpInventory: 3 retries, 10s timeout (discovery + caching)
/// - RegisterMcpServer: No retry, 5s timeout (fast registration)
/// - ValidateRegistration: Final validation step
pub fn create_mcp_registration_workflow() -> WorkflowDefinition {
    WorkflowDefinition::new("mcp_registration", "MCP Server Registration")
        .add_step(
            StepDefinition::new(
                "connect_mcp_server",
                "Connect to MCP Server",
                Arc::new(ConnectMcpServerStep),
            )
            .with_retry(RetryPolicy {
                max_attempts: 100,
                backoff: BackoffStrategy::Linear {
                    increment: Duration::from_secs(1),
                    max: Duration::from_secs(5),
                },
            })
            .with_timeout(Duration::from_secs(7200)) // 2 hours
            .with_failure_action(FailureAction::ContinueNextStep),
        )
        .add_step(
            StepDefinition::new(
                "discover_mcp_inventory",
                "Discover and Cache MCP Inventory",
                Arc::new(DiscoverMcpInventoryStep),
            )
            .with_retry(RetryPolicy {
                max_attempts: 3,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(1)),
            })
            .with_timeout(Duration::from_secs(10))
            .with_failure_action(FailureAction::ContinueNextStep),
        )
        .add_step(
            StepDefinition::new(
                "register_mcp_server",
                "Register MCP Server",
                Arc::new(RegisterMcpServerStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::ContinueNextStep),
        )
        .add_step(
            StepDefinition::new(
                "validate_registration",
                "Validate MCP Registration",
                Arc::new(ValidateRegistrationStep),
            )
            .with_timeout(Duration::from_secs(1))
            .with_failure_action(FailureAction::FailWorkflow),
        )
}
