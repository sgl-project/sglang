//! MCP server registration workflow steps
//!
//! Each step is atomic and performs a single operation in the MCP server registration process.
//!
//! Workflow order:
//! 1. ConnectMcpServer - Establish connection to MCP server with retry logic
//! 2. DiscoverMcpInventory - Discover tools, prompts, and resources from server
//! 3. CacheMcpTools - Cache discovered items in ToolInventory
//! 4. RegisterMcpServer - Register server in MCP manager's static server map

use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use tracing::{debug, info};

use crate::{
    app_context::AppContext,
    core::workflow::*,
    mcp::{
        manager::McpClientManager, config::McpServerConfig, PromptInfo, ResourceInfo,
        ToolInfo,
    },
};

/// MCP server connection configuration
#[derive(Debug, Clone)]
pub struct McpServerConfigRequest {
    /// Server name (unique identifier)
    pub name: String,
    /// Server configuration (transport, proxy, etc.)
    pub config: McpServerConfig,
}

/// Step 1: Connect to MCP server
///
/// This step establishes a connection to the MCP server and initializes the client manager.
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

        // Get shared ToolInventory from McpManager
        let mcp_manager =
            app_context
                .mcp_manager
                .get()
                .ok_or_else(|| WorkflowError::StepFailed {
                    step_id: StepId::new("connect_mcp_server"),
                    message: "MCP manager not initialized".to_string(),
                })?;

        let inventory = mcp_manager.inventory();

        // Get proxy config from router_config if available, otherwise fall back to env
        let proxy_config = app_context
            .router_config
            .mcp_config
            .as_ref()
            .and_then(|cfg| cfg.proxy.clone())
            .or_else(crate::mcp::McpProxyConfig::from_env);

        // Create MCP config with single server
        let mcp_config = crate::mcp::McpConfig {
            servers: vec![config_request.config.clone()],
            pool: Default::default(),
            proxy: proxy_config,
            warmup: Vec::new(),
            inventory: Default::default(),
        };

        // Connect to MCP server with shared inventory
        let client_manager = McpClientManager::new(mcp_config, inventory)
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

        // Store client manager in context (without wrapping in Arc - context.set() will do it)
        context.set("mcp_client_manager", client_manager);

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true // Connection failures are retryable
    }
}

/// Step 2: Discover MCP inventory (tools, prompts, resources)
///
/// This step queries the MCP server for its capabilities:
/// - Tools: Available function calls
/// - Prompts: Reusable prompt templates
/// - Resources: Accessible files/data
pub struct DiscoverMcpInventoryStep;

#[async_trait]
impl StepExecutor for DiscoverMcpInventoryStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config_request: Arc<McpServerConfigRequest> = context
            .get("mcp_server_config")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("mcp_server_config".to_string()))?;
        let mcp_client_manager: Arc<McpClientManager> = context
            .get("mcp_client_manager")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("mcp_client_manager".to_string()))?;

        debug!(
            "Discovering inventory for MCP server: {}",
            config_request.name
        );

        let manager = mcp_client_manager.as_ref();

        // Get tools
        let tools = manager.list_tools();
        debug!(
            "Discovered {} tools from {}",
            tools.len(),
            config_request.name
        );

        // Get prompts
        let prompts = manager.list_prompts();
        debug!(
            "Discovered {} prompts from {}",
            prompts.len(),
            config_request.name
        );

        // Get resources
        let resources = manager.list_resources();
        debug!(
            "Discovered {} resources from {}",
            resources.len(),
            config_request.name
        );

        // Store inventory in context
        context.set("discovered_tools", tools);
        context.set("discovered_prompts", prompts);
        context.set("discovered_resources", resources);

        info!(
            "Discovered inventory for {}: {} tools, {} prompts, {} resources",
            config_request.name,
            context
                .get::<Vec<ToolInfo>>("discovered_tools")
                .unwrap()
                .len(),
            context
                .get::<Vec<PromptInfo>>("discovered_prompts")
                .unwrap()
                .len(),
            context
                .get::<Vec<ResourceInfo>>("discovered_resources")
                .unwrap()
                .len()
        );

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true // Discovery failures are retryable
    }
}

/// Step 3: Cache MCP tools in inventory
///
/// This step stores the discovered tools, prompts, and resources in the shared
/// ToolInventory from McpManager for fast lookup and TTL-based expiration.
pub struct CacheMcpToolsStep;

#[async_trait]
impl StepExecutor for CacheMcpToolsStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config_request: Arc<McpServerConfigRequest> = context
            .get("mcp_server_config")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("mcp_server_config".to_string()))?;
        let app_context: Arc<AppContext> = context
            .get("app_context")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let tools: Arc<Vec<ToolInfo>> = context
            .get("discovered_tools")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("discovered_tools".to_string()))?;
        let prompts: Arc<Vec<PromptInfo>> = context
            .get("discovered_prompts")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("discovered_prompts".to_string()))?;
        let resources: Arc<Vec<ResourceInfo>> =
            context.get("discovered_resources").ok_or_else(|| {
                WorkflowError::ContextValueNotFound("discovered_resources".to_string())
            })?;

        debug!("Caching inventory for MCP server: {}", config_request.name);

        // Get shared ToolInventory from McpManager
        let mcp_manager =
            app_context
                .mcp_manager
                .get()
                .ok_or_else(|| WorkflowError::StepFailed {
                    step_id: StepId::new("cache_mcp_tools"),
                    message: "MCP manager not initialized".to_string(),
                })?;

        let inventory = mcp_manager.inventory();

        // Cache tools
        for tool in tools.iter() {
            inventory.insert_tool(tool.name.clone(), config_request.name.clone(), tool.clone());
        }

        // Cache prompts
        for prompt in prompts.iter() {
            inventory.insert_prompt(
                prompt.name.clone(),
                config_request.name.clone(),
                prompt.clone(),
            );
        }

        // Cache resources
        for resource in resources.iter() {
            inventory.insert_resource(
                resource.uri.clone(),
                config_request.name.clone(),
                resource.clone(),
            );
        }

        // Mark server as refreshed
        inventory.mark_refreshed(&config_request.name);

        info!(
            "Cached inventory for {}: {} tools, {} prompts, {} resources",
            config_request.name,
            tools.len(),
            prompts.len(),
            resources.len()
        );

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false // Caching is an in-memory operation, not retryable
    }
}

/// Step 4: Register MCP server in manager
///
/// This step adds the MCP server to the McpManager's static server map so it can be
/// used for tool calls and inventory management.
pub struct RegisterMcpServerStep;

#[async_trait]
impl StepExecutor for RegisterMcpServerStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config_request: Arc<McpServerConfigRequest> = context
            .get("mcp_server_config")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("mcp_server_config".to_string()))?;
        let app_context: Arc<AppContext> = context
            .get("app_context")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let mcp_client_manager: Arc<McpClientManager> = context
            .get("mcp_client_manager")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("mcp_client_manager".to_string()))?;

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

        // Register the server in the static server map
        mcp_manager
            .register_static_server(config_request.name.clone(), Arc::clone(&mcp_client_manager));

        info!("Registered MCP server: {}", config_request.name);

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false // Registration is a simple operation, not retryable
    }
}

/// Create MCP server registration workflow definition
///
/// Workflow configuration:
/// - ConnectMcpServer: 100 retries, 2hr timeout (aggressive retry for slow servers)
/// - DiscoverMcpInventory: 3 retries, 10s timeout (quick discovery)
/// - CacheMcpTools: No retry, 5s timeout (fast in-memory operation)
/// - RegisterMcpServer: No retry, 5s timeout (fast registration)
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
            .with_failure_action(FailureAction::FailWorkflow),
        )
        .add_step(
            StepDefinition::new(
                "discover_mcp_inventory",
                "Discover MCP Inventory",
                Arc::new(DiscoverMcpInventoryStep),
            )
            .with_retry(RetryPolicy {
                max_attempts: 3,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(1)),
            })
            .with_timeout(Duration::from_secs(10))
            .with_failure_action(FailureAction::FailWorkflow),
        )
        .add_step(
            StepDefinition::new(
                "cache_mcp_tools",
                "Cache MCP Tools",
                Arc::new(CacheMcpToolsStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::FailWorkflow),
        )
        .add_step(
            StepDefinition::new(
                "register_mcp_server",
                "Register MCP Server",
                Arc::new(RegisterMcpServerStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::ContinueNextStep), // Allow workflow to succeed even if registration fails
        )
}
