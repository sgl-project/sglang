//! MCP Tool Loop Abstraction
//!
//! This module provides a trait-based abstraction for the MCP tool calling loop
//! that is shared across different router implementations (OpenAI, gRPC Regular, gRPC Harmony).
//!
//! ## Architecture
//!
//! The MCP tool loop follows a common algorithm across all backends:
//!
//! ```text
//! 1. Initialize state (iteration=0, total_calls=0)
//! 2. Merge MCP tools with request tools
//! 3. Loop:
//!    a. Execute request
//!    b. Parse response for tool calls
//!    c. If no tool calls → return final response
//!    d. Separate MCP tools from function tools
//!    e. If function tools present → execute MCP, return to caller
//!    f. Check limits → return incomplete if exceeded
//!    g. Execute all MCP tools
//!    h. Build resume request with history
//!    i. Continue loop
//! ```
//!
//! ## Usage
//!
//! Each router implements [`McpLoopBackend`] with its specific types,
//! then calls [`execute_mcp_loop`] with the backend instance.

use std::{collections::HashSet, sync::Arc, time::Instant};

use async_trait::async_trait;
use tracing::{debug, trace, warn};

use crate::{
    mcp::McpManager,
    observability::metrics::{metrics_labels, Metrics},
    protocols::responses::{ResponseToolType, ResponsesRequest},
};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for MCP tool loop execution
#[derive(Debug, Clone)]
pub struct McpLoopConfig {
    /// Maximum iterations as safety limit (default: 10)
    pub max_iterations: usize,
    /// User-specified maximum tool calls (from request)
    pub max_tool_calls: Option<usize>,
}

impl Default for McpLoopConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            max_tool_calls: None,
        }
    }
}

impl McpLoopConfig {
    /// Create config from request parameters
    pub fn from_request(max_tool_calls: Option<u32>) -> Self {
        Self {
            max_iterations: 10,
            max_tool_calls: max_tool_calls.map(|n| n as usize),
        }
    }

    /// Get effective limit (min of user limit and safety limit)
    pub fn effective_limit(&self) -> usize {
        match self.max_tool_calls {
            Some(user_max) => user_max.min(self.max_iterations),
            None => self.max_iterations,
        }
    }
}

// ============================================================================
// Core Types
// ============================================================================

/// Represents a parsed tool call from a model response
#[derive(Debug, Clone)]
pub struct ParsedToolCall {
    /// Unique identifier for this tool call
    pub call_id: String,
    /// Name of the tool to execute
    pub name: String,
    /// JSON-encoded arguments
    pub arguments: String,
}

/// Result of executing an MCP tool
#[derive(Debug, Clone)]
pub struct ToolExecutionResult {
    /// Call ID (echoed back)
    pub call_id: String,
    /// Tool name (echoed back)
    pub name: String,
    /// Original arguments
    pub arguments: String,
    /// JSON-encoded output
    pub output: String,
    /// Whether execution succeeded
    pub success: bool,
    /// Error message if execution failed
    pub error: Option<String>,
}

/// Result of a single loop iteration
pub enum IterationOutcome<R> {
    /// No tool calls found - this is the final response
    Completed(R),
    /// Tool calls found that need processing
    ToolCallsFound(Vec<ParsedToolCall>),
}

/// Reason for exiting the MCP loop early
#[derive(Debug, Clone)]
pub enum LoopExitReason {
    /// Function tools detected - caller needs to handle them
    FunctionToolsDetected,
    /// Max tool calls limit reached
    MaxToolCallsExceeded,
    /// Max iterations exceeded (safety limit)
    MaxIterationsExceeded,
}

// ============================================================================
// Backend Trait
// ============================================================================

/// Trait for backend-specific MCP loop operations
///
/// Each router (OpenAI, gRPC Regular, gRPC Harmony) implements this trait
/// with its specific request/response types and execution mechanisms.
#[async_trait]
pub trait McpLoopBackend: Send + Sync {
    /// The request type for this backend
    type Request: Clone + Send + Sync;

    /// The final response type
    type Response: Send + Sync;

    /// Error type for backend operations
    type Error: Send;

    /// Get the model name from the current request (for metrics)
    fn model_name(&self) -> &str;

    /// Get the MCP server label (for metadata)
    fn server_label(&self) -> &str;

    /// Prepare the request with MCP tools merged in
    ///
    /// Called once at the start to set up tools and tool_choice
    fn prepare_request_with_mcp_tools(
        &mut self,
        request: &mut Self::Request,
        mcp_tools: &[crate::mcp::Tool],
        iteration: usize,
    );

    /// Execute a single request and get the iteration result
    ///
    /// Returns either a completed response or detected tool calls.
    /// Backends may store additional iteration state (e.g., analysis, partial_text)
    /// for use in `build_resume_request`.
    async fn execute_iteration(
        &mut self,
        request: &Self::Request,
    ) -> Result<IterationOutcome<Self::Response>, Self::Error>;

    /// Record an executed tool call in backend state
    ///
    /// Called after each MCP tool execution to update conversation history
    fn record_tool_execution(&mut self, result: &ToolExecutionResult);

    /// Build the resume request for the next iteration
    ///
    /// Incorporates conversation history from recorded tool executions
    fn build_resume_request(
        &self,
        base_request: &Self::Request,
    ) -> Result<Self::Request, Self::Error>;

    /// Inject MCP metadata into the final response
    ///
    /// Adds mcp_list_tools and mcp_call items to response output
    fn inject_mcp_metadata(&self, response: &mut Self::Response, mcp_manager: &McpManager);

    /// Mark response as incomplete due to limit exceeded
    fn mark_incomplete(&self, response: &mut Self::Response);
}

// ============================================================================
// Loop Executor
// ============================================================================

/// Execute the MCP tool calling loop with the given backend
///
/// This function implements the common MCP loop algorithm:
/// 1. Merge MCP tools with request
/// 2. Execute iterations until no more tool calls
/// 3. Handle limits and function tool detection
/// 4. Inject MCP metadata into final response
///
/// # Arguments
///
/// * `backend` - The backend implementing [`McpLoopBackend`]
/// * `initial_request` - The starting request
/// * `mcp_manager` - MCP manager for tool execution
/// * `config` - Loop configuration (limits, etc.)
///
/// # Returns
///
/// The final response after all MCP tools have been executed,
/// or an early exit with function tools / limit exceeded
pub async fn execute_mcp_loop<B: McpLoopBackend>(
    backend: &mut B,
    mut current_request: B::Request,
    mcp_manager: &Arc<McpManager>,
    config: McpLoopConfig,
) -> Result<B::Response, B::Error> {
    let mut iteration = 0;
    let mut total_calls = 0;
    let effective_limit = config.effective_limit();

    // Get MCP tools and their names for partitioning
    let mcp_tools = mcp_manager.list_tools();
    let mcp_tool_names: HashSet<String> = mcp_tools.iter().map(|t| t.name.to_string()).collect();

    let model_name = backend.model_name().to_string();
    let server_label = backend.server_label().to_string();

    trace!(
        model = %model_name,
        server_label = %server_label,
        mcp_tool_count = mcp_tools.len(),
        effective_limit = effective_limit,
        "Starting MCP tool loop"
    );

    loop {
        iteration += 1;

        // Record iteration metric
        Metrics::record_mcp_tool_iteration(&model_name);

        // Safety check: prevent infinite loops
        if iteration > config.max_iterations {
            warn!(
                iteration = iteration,
                max = config.max_iterations,
                "MCP loop exceeded maximum iterations"
            );
            // Execute one more time to get a response, then mark incomplete
            let result = backend.execute_iteration(&current_request).await?;
            if let IterationOutcome::Completed(mut response) = result {
                backend.mark_incomplete(&mut response);
                return Ok(response);
            }
            // If we still got tool calls, just return an error
            return Err(backend
                .execute_iteration(&current_request)
                .await
                .err()
                .unwrap());
        }

        trace!(iteration = iteration, "Executing MCP loop iteration");

        // Prepare request with MCP tools (first iteration only merges, subsequent iterations already have them)
        if iteration == 1 {
            backend.prepare_request_with_mcp_tools(&mut current_request, &mcp_tools, iteration);
        }

        // Execute iteration
        let result = backend.execute_iteration(&current_request).await?;

        match result {
            IterationOutcome::Completed(mut response) => {
                // No tool calls - inject metadata and return
                debug!(
                    iteration = iteration,
                    total_calls = total_calls,
                    "MCP loop completed - no more tool calls"
                );

                if total_calls > 0 {
                    backend.inject_mcp_metadata(&mut response, mcp_manager);
                }

                return Ok(response);
            }
            IterationOutcome::ToolCallsFound(tool_calls) => {
                trace!(
                    iteration = iteration,
                    tool_call_count = tool_calls.len(),
                    "Tool calls found"
                );

                // Partition into MCP and function tools
                let (mcp_calls, function_calls): (Vec<_>, Vec<_>) = tool_calls
                    .into_iter()
                    .partition(|tc| mcp_tool_names.contains(&tc.name));

                debug!(
                    mcp_calls = mcp_calls.len(),
                    function_calls = function_calls.len(),
                    "Partitioned tool calls"
                );

                // Check if limit would be exceeded
                if total_calls + mcp_calls.len() > effective_limit {
                    warn!(
                        current = total_calls,
                        new = mcp_calls.len(),
                        limit = effective_limit,
                        "Tool call limit exceeded"
                    );

                    // Get a response and mark as incomplete
                    let result = backend.execute_iteration(&current_request).await?;
                    if let IterationOutcome::Completed(mut response) = result {
                        backend.mark_incomplete(&mut response);
                        if total_calls > 0 {
                            backend.inject_mcp_metadata(&mut response, mcp_manager);
                        }
                        return Ok(response);
                    }
                }

                // Execute MCP tools
                for call in &mcp_calls {
                    let result = execute_single_tool(mcp_manager, call, &model_name).await;
                    backend.record_tool_execution(&result);
                    total_calls += 1;
                }

                // If function tools present, return to caller
                if !function_calls.is_empty() {
                    debug!(
                        function_calls = function_calls.len(),
                        "Function tools detected - returning to caller"
                    );

                    // Get a base response to build upon
                    let result = backend.execute_iteration(&current_request).await?;
                    if let IterationOutcome::Completed(mut response) = result {
                        if total_calls > 0 {
                            backend.inject_mcp_metadata(&mut response, mcp_manager);
                        }
                        return Ok(response);
                    }
                }

                // Build resume request for next iteration
                current_request = backend.build_resume_request(&current_request)?;

                // Update tool_choice for subsequent iterations
                backend.prepare_request_with_mcp_tools(
                    &mut current_request,
                    &mcp_tools,
                    iteration + 1,
                );
            }
        }
    }
}

/// Execute a single MCP tool and return the result
async fn execute_single_tool(
    mcp_manager: &Arc<McpManager>,
    call: &ParsedToolCall,
    model_name: &str,
) -> ToolExecutionResult {
    trace!(
        tool = %call.name,
        call_id = %call.call_id,
        "Executing MCP tool"
    );

    let start = Instant::now();
    let result = mcp_manager
        .call_tool(&call.name, call.arguments.as_str())
        .await;
    let duration = start.elapsed();

    // Record metrics
    Metrics::record_mcp_tool_duration(model_name, &call.name, duration);

    let (output, success, error) = match result {
        Ok(value) => match serde_json::to_string(&value) {
            Ok(output) => {
                Metrics::record_mcp_tool_call(
                    model_name,
                    &call.name,
                    metrics_labels::RESULT_SUCCESS,
                );
                (output, true, None)
            }
            Err(e) => {
                let err = format!("Failed to serialize tool result: {}", e);
                warn!("{}", err);
                Metrics::record_mcp_tool_call(model_name, &call.name, metrics_labels::RESULT_ERROR);
                (
                    serde_json::json!({ "error": &err }).to_string(),
                    false,
                    Some(err),
                )
            }
        },
        Err(e) => {
            let err = format!("Tool call failed: {}", e);
            warn!(tool = %call.name, error = %err, "MCP tool execution failed");
            Metrics::record_mcp_tool_call(model_name, &call.name, metrics_labels::RESULT_ERROR);
            (
                serde_json::json!({ "error": &err }).to_string(),
                false,
                Some(err),
            )
        }
    };

    ToolExecutionResult {
        call_id: call.call_id.clone(),
        name: call.name.clone(),
        arguments: call.arguments.clone(),
        output,
        success,
        error,
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Default server label for MCP metadata
pub const DEFAULT_SERVER_LABEL: &str = "mcp";

/// Extract server label from request tools
///
/// Finds the first MCP tool in the request and extracts its server_label.
/// Falls back to [`DEFAULT_SERVER_LABEL`] if not found.
pub fn extract_server_label(request: &ResponsesRequest) -> String {
    request
        .tools
        .as_ref()
        .and_then(|tools| {
            tools
                .iter()
                .find(|t| matches!(t.r#type, ResponseToolType::Mcp))
                .and_then(|t| t.server_label.clone())
        })
        .unwrap_or_else(|| DEFAULT_SERVER_LABEL.to_string())
}
