use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use tracing::{debug, error, info, warn};

use super::workflow_data::McpWorkflowData;
use crate::{
    app_context::AppContext,
    mcp::{config::McpServerConfig, manager::McpManager},
    observability::metrics::Metrics,
    workflow::{
        BackoffStrategy, FailureAction, RetryPolicy, StepDefinition, StepExecutor, StepId,
        StepResult, WorkflowContext, WorkflowDefinition, WorkflowError, WorkflowResult,
    },
};

/// MCP server connection configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
impl StepExecutor<McpWorkflowData> for ConnectMcpServerStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<McpWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let config_request = &context.data.config;
        let app_context = context
            .data
            .app_context
            .as_ref()
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

        // Store client in typed data
        context.data.mcp_client = Some(Arc::new(client));

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
impl StepExecutor<McpWorkflowData> for DiscoverMcpInventoryStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<McpWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let config_request = &context.data.config;
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let mcp_client = context
            .data
            .mcp_client
            .as_ref()
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
        let server_key = McpManager::server_key(&config_request.config);

        // Discover and load inventory
        McpManager::load_server_inventory(&inventory, &server_key, mcp_client).await;

        info!(
            "Completed inventory discovery for {} (key: {})",
            config_request.name, server_key
        );

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
impl StepExecutor<McpWorkflowData> for RegisterMcpServerStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<McpWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let config_request = &context.data.config;
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let mcp_client = context
            .data
            .mcp_client
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("mcp_client".to_string()))?
            .clone();

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

        let server_key = McpManager::server_key(&config_request.config);

        // Register the client in the manager's client map
        mcp_manager.register_static_server(server_key.clone(), mcp_client);

        // Update active MCP servers metric
        Metrics::set_mcp_servers_active(mcp_manager.list_servers().len());

        info!(
            "Registered MCP server: {} (key: {})",
            config_request.name, server_key
        );

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
impl StepExecutor<McpWorkflowData> for ValidateRegistrationStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<McpWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let config_request = &context.data.config;
        let client_registered = context.data.mcp_client.is_some();

        if client_registered {
            info!(
                "MCP server '{}' registered successfully",
                config_request.name
            );

            // Mark as validated
            context.data.validated = true;

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
pub fn create_mcp_registration_workflow() -> WorkflowDefinition<McpWorkflowData> {
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
            .with_failure_action(FailureAction::ContinueNextStep)
            .depends_on(&["connect_mcp_server"]),
        )
        .add_step(
            StepDefinition::new(
                "register_mcp_server",
                "Register MCP Server",
                Arc::new(RegisterMcpServerStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::ContinueNextStep)
            .depends_on(&["discover_mcp_inventory"]),
        )
        .add_step(
            StepDefinition::new(
                "validate_registration",
                "Validate MCP Registration",
                Arc::new(ValidateRegistrationStep),
            )
            .with_timeout(Duration::from_secs(1))
            .with_failure_action(FailureAction::FailWorkflow)
            .depends_on(&["register_mcp_server"]),
        )
}

/// Helper to create initial workflow data for MCP registration
pub fn create_mcp_workflow_data(
    config: McpServerConfigRequest,
    app_context: Arc<AppContext>,
) -> McpWorkflowData {
    McpWorkflowData {
        config,
        validated: false,
        app_context: Some(app_context),
        mcp_client: None,
    }
}
