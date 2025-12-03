//! Utility types and constants for OpenAI router

use std::collections::HashMap;

use axum::http::{HeaderMap, HeaderValue};

// ============================================================================
// SSE Event Type Constants
// ============================================================================

/// SSE event type constants - single source of truth for event type strings
pub(crate) mod event_types {
    // Response lifecycle events
    pub const RESPONSE_CREATED: &str = "response.created";
    pub const RESPONSE_IN_PROGRESS: &str = "response.in_progress";
    pub const RESPONSE_COMPLETED: &str = "response.completed";

    // Output item events
    pub const OUTPUT_ITEM_ADDED: &str = "response.output_item.added";
    pub const OUTPUT_ITEM_DONE: &str = "response.output_item.done";
    pub const OUTPUT_ITEM_DELTA: &str = "response.output_item.delta";

    // Function call events
    pub const FUNCTION_CALL_ARGUMENTS_DELTA: &str = "response.function_call_arguments.delta";
    pub const FUNCTION_CALL_ARGUMENTS_DONE: &str = "response.function_call_arguments.done";

    // MCP call events
    pub const MCP_CALL_ARGUMENTS_DELTA: &str = "response.mcp_call_arguments.delta";
    pub const MCP_CALL_ARGUMENTS_DONE: &str = "response.mcp_call_arguments.done";
    pub const MCP_CALL_IN_PROGRESS: &str = "response.mcp_call.in_progress";
    pub const MCP_CALL_COMPLETED: &str = "response.mcp_call.completed";
    pub const MCP_LIST_TOOLS_IN_PROGRESS: &str = "response.mcp_list_tools.in_progress";
    pub const MCP_LIST_TOOLS_COMPLETED: &str = "response.mcp_list_tools.completed";

    // Item types
    pub const ITEM_TYPE_FUNCTION_CALL: &str = "function_call";
    pub const ITEM_TYPE_FUNCTION_TOOL_CALL: &str = "function_tool_call";
    pub const ITEM_TYPE_MCP_CALL: &str = "mcp_call";
    pub const ITEM_TYPE_FUNCTION: &str = "function";
    pub const ITEM_TYPE_MCP_LIST_TOOLS: &str = "mcp_list_tools";
}

// ============================================================================
// Stream Action Enum
// ============================================================================

/// Action to take based on streaming event processing
#[derive(Debug)]
pub(crate) enum StreamAction {
    Forward,      // Pass event to client
    Buffer,       // Accumulate for tool execution
    ExecuteTools, // Function call complete, execute now
}

// ============================================================================
// Output Index Mapper
// ============================================================================

/// Maps upstream output indices to sequential downstream indices
#[derive(Debug, Default)]
pub(crate) struct OutputIndexMapper {
    next_index: usize,
    // Map upstream output_index -> remapped output_index
    assigned: HashMap<usize, usize>,
}

impl OutputIndexMapper {
    pub fn with_start(next_index: usize) -> Self {
        Self {
            next_index,
            assigned: HashMap::new(),
        }
    }

    pub fn ensure_mapping(&mut self, upstream_index: usize) -> usize {
        *self.assigned.entry(upstream_index).or_insert_with(|| {
            let assigned = self.next_index;
            self.next_index += 1;
            assigned
        })
    }

    pub fn lookup(&self, upstream_index: usize) -> Option<usize> {
        self.assigned.get(&upstream_index).copied()
    }

    pub fn allocate_synthetic(&mut self) -> usize {
        let assigned = self.next_index;
        self.next_index += 1;
        assigned
    }

    pub fn next_index(&self) -> usize {
        self.next_index
    }
}

// ============================================================================
// Provider Detection and Header Handling
// ============================================================================

/// Extract authorization header from request headers
/// Checks both "authorization" and "Authorization" (case variations)
pub fn extract_auth_header(headers: Option<&HeaderMap>) -> Option<&str> {
    headers.and_then(|h| {
        h.get("authorization")
            .or_else(|| h.get("Authorization"))
            .and_then(|v| v.to_str().ok())
    })
}

/// API provider types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiProvider {
    Anthropic,
    Xai,
    OpenAi,
    Gemini,
    Generic,
}

impl ApiProvider {
    /// Detect provider type from URL
    pub fn from_url(url: &str) -> Self {
        if url.contains("anthropic") {
            ApiProvider::Anthropic
        } else if url.contains("x.ai") {
            ApiProvider::Xai
        } else if url.contains("openai.com") {
            ApiProvider::OpenAi
        } else if url.contains("googleapis.com") {
            ApiProvider::Gemini
        } else {
            ApiProvider::Generic
        }
    }
}

/// Apply provider-specific headers to request
pub fn apply_provider_headers(
    mut req: reqwest::RequestBuilder,
    url: &str,
    auth_header: Option<&HeaderValue>,
) -> reqwest::RequestBuilder {
    let provider = ApiProvider::from_url(url);

    match provider {
        ApiProvider::Anthropic => {
            // Anthropic requires x-api-key instead of Authorization
            // Extract Bearer token and use as x-api-key
            if let Some(auth) = auth_header {
                if let Ok(auth_str) = auth.to_str() {
                    let api_key = auth_str.strip_prefix("Bearer ").unwrap_or(auth_str);
                    req = req
                        .header("x-api-key", api_key)
                        .header("anthropic-version", "2023-06-01");
                }
            }
        }
        ApiProvider::Gemini | ApiProvider::Xai | ApiProvider::OpenAi | ApiProvider::Generic => {
            // Standard OpenAI-compatible: use Authorization header as-is
            if let Some(auth) = auth_header {
                req = req.header("Authorization", auth);
            }
        }
    }

    req
}

/// Probe a single endpoint to check if it has the model
/// Returns Ok(url) if model found, Err(()) otherwise
pub async fn probe_endpoint_for_model(
    client: reqwest::Client,
    url: String,
    model: String,
    auth: Option<String>,
) -> Result<String, ()> {
    use tracing::debug;

    let probe_url = format!("{}/v1/models/{}", url, model);
    let req = client
        .get(&probe_url)
        .timeout(std::time::Duration::from_secs(5));

    // Apply provider-specific headers (handles Anthropic, xAI, OpenAI, etc.)
    let auth_header_value = auth.as_ref().and_then(|a| HeaderValue::from_str(a).ok());
    let req = apply_provider_headers(req, &url, auth_header_value.as_ref());

    match req.send().await {
        Ok(resp) => {
            let status = resp.status();
            if status.is_success() {
                debug!(
                    url = %url,
                    model = %model,
                    status = %status,
                    "Model found on endpoint"
                );
                Ok(url)
            } else {
                debug!(
                    url = %url,
                    model = %model,
                    status = %status,
                    "Model not found on endpoint (unsuccessful status)"
                );
                Err(())
            }
        }
        Err(e) => {
            debug!(
                url = %url,
                model = %model,
                error = %e,
                "Probe request to endpoint failed"
            );
            Err(())
        }
    }
}

// ============================================================================
// Re-export FunctionCallInProgress from mcp module
// ============================================================================
pub(crate) use super::mcp::FunctionCallInProgress;
