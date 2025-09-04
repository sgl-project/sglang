// tool_server.rs - Main MCP implementation (matching Python's tool_server.py)
use crate::mcp::types::*;
use serde_json::{json, Value};
use std::collections::HashMap;

/// Main MCP Tool Server
pub struct MCPToolServer {
    /// Tool descriptions by server
    tool_descriptions: HashMap<String, Value>,
    /// Server URLs
    urls: HashMap<String, String>,
}

impl Default for MCPToolServer {
    fn default() -> Self {
        Self::new()
    }
}

impl MCPToolServer {
    /// Create new MCPToolServer
    pub fn new() -> Self {
        Self {
            tool_descriptions: HashMap::new(),
            urls: HashMap::new(),
        }
    }

    /// Clears all existing tool servers and adds new ones from the provided URL(s).
    /// URLs can be a single string or multiple comma-separated strings.
    pub async fn add_tool_server(&mut self, server_url: String) -> MCPResult<()> {
        let tool_urls: Vec<&str> = server_url.split(",").collect();
        let mut successful_connections = 0;
        let mut errors = Vec::new();

        // Clear existing
        self.tool_descriptions = HashMap::new();
        self.urls = HashMap::new();

        for url_str in tool_urls {
            let url_str = url_str.trim();

            // Format URL for MCP-compliant connection
            let formatted_url = if url_str.starts_with("http://") || url_str.starts_with("https://")
            {
                url_str.to_string()
            } else {
                // Default to MCP endpoint if no protocol specified
                format!("http://{}", url_str)
            };

            // Server connection with retry and error recovery
            match self.connect_to_server(&formatted_url).await {
                Ok((_init_response, tools_response)) => {
                    // Process tools with validation
                    let tools_obj = post_process_tools_description(tools_response);

                    // Tool storage with conflict detection
                    for tool in &tools_obj.tools {
                        let tool_name = &tool.name;

                        // Check for duplicate tools
                        if self.tool_descriptions.contains_key(tool_name) {
                            tracing::warn!(
                                "Tool {} already exists. Ignoring duplicate tool from server {}",
                                tool_name,
                                formatted_url
                            );
                            continue;
                        }

                        // Store individual tool descriptions
                        let tool_json = json!(tool);
                        self.tool_descriptions
                            .insert(tool_name.clone(), tool_json.clone());
                        self.urls.insert(tool_name.clone(), formatted_url.clone());
                    }

                    successful_connections += 1;
                }
                Err(e) => {
                    errors.push(format!("Failed to connect to {}: {}", formatted_url, e));
                    tracing::warn!("Failed to connect to MCP server {}: {}", formatted_url, e);
                }
            }
        }

        // Error handling - succeed if at least one server connects
        if successful_connections == 0 {
            let combined_error = errors.join("; ");
            return Err(MCPError::ConnectionError(format!(
                "Failed to connect to any MCP servers: {}",
                combined_error
            )));
        }

        if !errors.is_empty() {
            tracing::warn!("Some MCP servers failed to connect: {}", errors.join("; "));
        }

        tracing::info!(
            "Successfully connected to {} MCP server(s), discovered {} tool(s)",
            successful_connections,
            self.tool_descriptions.len()
        );

        Ok(())
    }

    /// Server connection with retries (internal helper)
    async fn connect_to_server(
        &self,
        url: &str,
    ) -> MCPResult<(InitializeResponse, ListToolsResponse)> {
        const MAX_RETRIES: u32 = 3;
        const RETRY_DELAY_MS: u64 = 1000;

        let mut last_error = None;

        for attempt in 1..=MAX_RETRIES {
            match list_server_and_tools(url).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    last_error = Some(e);
                    if attempt < MAX_RETRIES {
                        tracing::debug!(
                            "MCP server connection attempt {}/{} failed for {}: {}. Retrying...",
                            attempt,
                            MAX_RETRIES,
                            url,
                            last_error.as_ref().unwrap()
                        );
                        tokio::time::sleep(tokio::time::Duration::from_millis(
                            RETRY_DELAY_MS * attempt as u64,
                        ))
                        .await;
                    }
                }
            }
        }

        Err(last_error.unwrap())
    }

    /// Check if tool exists (matching Python's has_tool)
    pub fn has_tool(&self, tool_name: &str) -> bool {
        self.tool_descriptions.contains_key(tool_name)
    }

    /// Get tool description (matching Python's get_tool_description)
    pub fn get_tool_description(&self, tool_name: &str) -> Option<&Value> {
        self.tool_descriptions.get(tool_name)
    }

    /// Get tool session (matching Python's get_tool_session)
    pub async fn get_tool_session(&self, tool_name: &str) -> MCPResult<ToolSession> {
        let url = self
            .urls
            .get(tool_name)
            .ok_or_else(|| MCPError::ToolNotFound(tool_name.to_string()))?;

        // Create session
        ToolSession::new(url.clone()).await
    }

    /// Create multi-tool session manager
    pub async fn create_multi_tool_session(
        &self,
        tool_names: Vec<String>,
    ) -> MCPResult<MultiToolSessionManager> {
        let mut session_manager = MultiToolSessionManager::new();

        // Group tools by server URL for efficient session creation
        let mut server_tools: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();

        for tool_name in tool_names {
            if let Some(url) = self.urls.get(&tool_name) {
                server_tools.entry(url.clone()).or_default().push(tool_name);
            } else {
                return Err(MCPError::ToolNotFound(format!(
                    "Tool not found: {}",
                    tool_name
                )));
            }
        }

        // Create sessions for each server
        for (server_url, tools) in server_tools {
            session_manager
                .add_tools_from_server(server_url, tools)
                .await?;
        }

        Ok(session_manager)
    }

    /// List all available tools
    pub fn list_tools(&self) -> Vec<String> {
        self.tool_descriptions.keys().cloned().collect()
    }

    /// Get tool statistics
    pub fn get_tool_stats(&self) -> ToolStats {
        ToolStats {
            total_tools: self.tool_descriptions.len(),
            total_servers: self
                .urls
                .values()
                .collect::<std::collections::HashSet<_>>()
                .len(),
        }
    }

    /// List all connected servers
    pub fn list_servers(&self) -> Vec<String> {
        self.urls
            .values()
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect()
    }

    /// Check if a specific server is connected
    pub fn has_server(&self, server_url: &str) -> bool {
        self.urls.values().any(|url| url == server_url)
    }

    /// Execute a tool directly (convenience method for simple usage)
    pub async fn call_tool(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> MCPResult<serde_json::Value> {
        let session = self.get_tool_session(tool_name).await?;
        session.call_tool(tool_name, arguments).await
    }

    /// Create a tool session from server URL (convenience method)
    pub async fn create_session_from_url(&self, server_url: &str) -> MCPResult<ToolSession> {
        ToolSession::new(server_url.to_string()).await
    }
}

/// Tool statistics for monitoring
#[derive(Debug, Clone)]
pub struct ToolStats {
    pub total_tools: usize,
    pub total_servers: usize,
}

/// MCP-compliant server connection using JSON-RPC over SSE
async fn list_server_and_tools(
    server_url: &str,
) -> MCPResult<(InitializeResponse, ListToolsResponse)> {
    // MCP specification:
    // 1. Connect to MCP endpoint with GET (SSE) or POST (JSON-RPC)
    // 2. Send initialize request
    // 3. Send tools/list request
    // 4. Parse JSON-RPC responses

    let client = reqwest::Client::new();

    // Step 1: Send initialize request
    let init_request = MCPRequest {
        jsonrpc: "2.0".to_string(),
        id: "1".to_string(),
        method: "initialize".to_string(),
        params: Some(json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {}
        })),
    };

    let init_response = send_mcp_request(&client, server_url, init_request).await?;
    let init_result: InitializeResponse = serde_json::from_value(init_response).map_err(|e| {
        MCPError::SerializationError(format!("Failed to parse initialize response: {}", e))
    })?;

    // Step 2: Send tools/list request
    let tools_request = MCPRequest {
        jsonrpc: "2.0".to_string(),
        id: "2".to_string(),
        method: "tools/list".to_string(),
        params: Some(json!({})),
    };

    let tools_response = send_mcp_request(&client, server_url, tools_request).await?;
    let tools_result: ListToolsResponse = serde_json::from_value(tools_response).map_err(|e| {
        MCPError::SerializationError(format!("Failed to parse tools/list response: {}", e))
    })?;

    Ok((init_result, tools_result))
}

/// Send MCP JSON-RPC request (supports both HTTP POST and SSE)
async fn send_mcp_request(
    client: &reqwest::Client,
    url: &str,
    request: MCPRequest,
) -> MCPResult<Value> {
    // Use HTTP POST for JSON-RPC requests
    let response = client
        .post(url)
        .header("Content-Type", "application/json")
        .header("Accept", "application/json")
        .json(&request)
        .send()
        .await
        .map_err(|e| MCPError::ConnectionError(format!("MCP request failed: {}", e)))?;

    if !response.status().is_success() {
        return Err(MCPError::ProtocolError(format!(
            "HTTP {}",
            response.status()
        )));
    }

    let mcp_response: MCPResponse = response.json().await.map_err(|e| {
        MCPError::SerializationError(format!("Failed to parse MCP response: {}", e))
    })?;

    if let Some(error) = mcp_response.error {
        return Err(MCPError::ProtocolError(format!(
            "MCP error: {}",
            error.message
        )));
    }

    mcp_response
        .result
        .ok_or_else(|| MCPError::ProtocolError("No result in MCP response".to_string()))
}

// Removed old send_http_request - now using send_mcp_request with proper MCP protocol

/// Parse SSE event format (MCP-compliant JSON-RPC only)
pub fn parse_sse_event(event: &str) -> MCPResult<Option<Value>> {
    let mut data_lines = Vec::new();

    for line in event.lines() {
        if let Some(stripped) = line.strip_prefix("data: ") {
            data_lines.push(stripped);
        }
    }

    if data_lines.is_empty() {
        return Ok(None);
    }

    let json_data = data_lines.join("\n");
    if json_data.trim().is_empty() {
        return Ok(None);
    }

    // Parse as MCP JSON-RPC response only (no custom events)
    let mcp_response: MCPResponse = serde_json::from_str(&json_data).map_err(|e| {
        MCPError::SerializationError(format!(
            "Failed to parse JSON-RPC response: {} - Data: {}",
            e, json_data
        ))
    })?;

    if let Some(error) = mcp_response.error {
        return Err(MCPError::ProtocolError(error.message));
    }

    Ok(mcp_response.result)
}

/// Schema adaptation matching Python's trim_schema()
fn trim_schema(schema: &mut Value) {
    if let Some(obj) = schema.as_object_mut() {
        // Remove title and null defaults
        obj.remove("title");
        if obj.get("default") == Some(&Value::Null) {
            obj.remove("default");
        }

        // Convert anyOf to type arrays
        if let Some(any_of) = obj.remove("anyOf") {
            if let Some(array) = any_of.as_array() {
                let types: Vec<String> = array
                    .iter()
                    .filter_map(|item| {
                        item.get("type")
                            .and_then(|t| t.as_str())
                            .filter(|t| *t != "null")
                            .map(|t| t.to_string())
                    })
                    .collect();

                // Handle single type vs array of types
                match types.len() {
                    0 => {} // No valid types found
                    1 => {
                        obj.insert("type".to_string(), json!(types[0]));
                    }
                    _ => {
                        obj.insert("type".to_string(), json!(types));
                    }
                }
            }
        }

        // Handle oneOf similar to anyOf
        if let Some(one_of) = obj.remove("oneOf") {
            if let Some(array) = one_of.as_array() {
                let types: Vec<String> = array
                    .iter()
                    .filter_map(|item| {
                        item.get("type")
                            .and_then(|t| t.as_str())
                            .filter(|t| *t != "null")
                            .map(|t| t.to_string())
                    })
                    .collect();

                if !types.is_empty() {
                    obj.insert("type".to_string(), json!(types));
                }
            }
        }

        // Recursive processing for properties
        if let Some(properties) = obj.get_mut("properties") {
            if let Some(props_obj) = properties.as_object_mut() {
                for (_, value) in props_obj.iter_mut() {
                    trim_schema(value);
                }
            }
        }

        // Handle nested schemas in items (for arrays)
        if let Some(items) = obj.get_mut("items") {
            trim_schema(items);
        }

        // Handle nested schemas in additionalProperties
        if let Some(additional_props) = obj.get_mut("additionalProperties") {
            if additional_props.is_object() {
                trim_schema(additional_props);
            }
        }

        // Handle patternProperties (for dynamic property names)
        if let Some(pattern_props) = obj.get_mut("patternProperties") {
            if let Some(pattern_obj) = pattern_props.as_object_mut() {
                for (_, value) in pattern_obj.iter_mut() {
                    trim_schema(value);
                }
            }
        }

        // Handle allOf in nested contexts
        if let Some(all_of) = obj.get_mut("allOf") {
            if let Some(array) = all_of.as_array_mut() {
                for item in array.iter_mut() {
                    trim_schema(item);
                }
            }
        }
    }
}

/// Tool processing with filtering
fn post_process_tools_description(mut tools_response: ListToolsResponse) -> ListToolsResponse {
    // Adapt schemas for Harmony
    for tool in &mut tools_response.tools {
        trim_schema(&mut tool.input_schema);
    }

    // Tool filtering based on annotations
    let initial_count = tools_response.tools.len();

    tools_response.tools.retain(|tool| {
        // Check include_in_prompt annotation (Python behavior)
        let include_in_prompt = tool
            .annotations
            .as_ref()
            .and_then(|a| a.get("include_in_prompt"))
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        if !include_in_prompt {
            tracing::debug!(
                "Filtering out tool '{}' due to include_in_prompt=false",
                tool.name
            );
            return false;
        }

        // Check if tool is explicitly disabled
        let disabled = tool
            .annotations
            .as_ref()
            .and_then(|a| a.get("disabled"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        if disabled {
            tracing::debug!("Filtering out disabled tool '{}'", tool.name);
            return false;
        }

        // Validate tool has required fields
        if tool.name.trim().is_empty() {
            tracing::warn!("Filtering out tool with empty name");
            return false;
        }

        // Check for valid input schema
        if tool.input_schema.is_null() {
            tracing::warn!("Tool '{}' has null input schema, but keeping it", tool.name);
        }

        true
    });

    let filtered_count = tools_response.tools.len();
    if filtered_count != initial_count {
        tracing::info!(
            "Filtered tools: {} -> {} ({} removed)",
            initial_count,
            filtered_count,
            initial_count - filtered_count
        );
    }

    tools_response
}

// Tests moved to tests/mcp_comprehensive_test.rs for better organization
