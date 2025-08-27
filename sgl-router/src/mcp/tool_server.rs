// tool_server.rs - Main MCP implementation (matching Python's tool_server.py)
use crate::mcp::types::*;
use std::collections::HashMap;
use serde_json::{json, Value};

/// Main MCP Tool Server (enhanced with caching and indexing)
pub struct MCPToolServer {
    /// Tool descriptions by server (matching Python's harmony_tool_descriptions)
    tool_descriptions: HashMap<String, Value>,
    /// Server URLs (matching Python's urls dict)
    urls: HashMap<String, String>,
    /// Tool cache with metadata (enhancement)
    tool_cache: ToolCache,
}

/// Tool caching system for improved performance
#[derive(Debug, Clone)]
pub struct ToolCache {
    /// Index of tools by category/tag
    pub category_index: HashMap<String, Vec<String>>,
    /// Index of tools by input parameters
    pub parameter_index: HashMap<String, Vec<String>>,
    /// Tool metadata cache
    pub metadata_cache: HashMap<String, ToolMetadata>,
    /// Cache timestamp for invalidation
    pub last_updated: Option<std::time::Instant>,
}

/// Metadata for cached tools
#[derive(Debug, Clone)]
pub struct ToolMetadata {
    pub name: String,
    pub description: String,
    pub categories: Vec<String>,
    pub parameter_count: usize,
    pub has_required_params: bool,
    pub server_url: String,
}

impl ToolCache {
    pub fn new() -> Self {
        Self {
            category_index: HashMap::new(),
            parameter_index: HashMap::new(),
            metadata_cache: HashMap::new(),
            last_updated: None,
        }
    }
    
    /// Update cache with tool information
    pub fn update_tool(&mut self, tool_name: &str, tool_info: &Value, server_url: &str) {
        // Extract tool metadata
        let metadata = ToolMetadata {
            name: tool_name.to_string(),
            description: tool_info.get("description")
                .and_then(|d| d.as_str())
                .unwrap_or("").to_string(),
            categories: extract_categories(tool_info),
            parameter_count: count_parameters(tool_info),
            has_required_params: has_required_parameters(tool_info),
            server_url: server_url.to_string(),
        };
        
        // Update metadata cache
        self.metadata_cache.insert(tool_name.to_string(), metadata.clone());
        
        // Update category index
        for category in &metadata.categories {
            self.category_index
                .entry(category.clone())
                .or_insert_with(Vec::new)
                .push(tool_name.to_string());
        }
        
        // Update parameter index
        if let Some(schema) = tool_info.get("input_schema").or_else(|| tool_info.get("inputSchema")) {
            if let Some(properties) = schema.get("properties") {
                if let Some(props_obj) = properties.as_object() {
                    for param_name in props_obj.keys() {
                        self.parameter_index
                            .entry(param_name.clone())
                            .or_insert_with(Vec::new)
                            .push(tool_name.to_string());
                    }
                }
            }
        }
        
        self.last_updated = Some(std::time::Instant::now());
    }
    
    /// Get tools by category
    pub fn get_tools_by_category(&self, category: &str) -> Vec<String> {
        self.category_index
            .get(category)
            .cloned()
            .unwrap_or_default()
    }
    
    /// Get tools that have a specific parameter
    pub fn get_tools_by_parameter(&self, parameter: &str) -> Vec<String> {
        self.parameter_index
            .get(parameter)
            .cloned()
            .unwrap_or_default()
    }
    
    /// Clear all cache data
    pub fn clear(&mut self) {
        self.category_index.clear();
        self.parameter_index.clear();
        self.metadata_cache.clear();
        self.last_updated = None;
    }
}

impl MCPToolServer {
    /// Create new MCPToolServer (enhanced with caching)
    pub fn new() -> Self {
        Self {
            tool_descriptions: HashMap::new(),
            urls: HashMap::new(),
            tool_cache: ToolCache::new(),
        }
    }

    /// Add tool server (enhanced with better server detection and error recovery)
    pub async fn add_tool_server(&mut self, server_url: String) -> MCPResult<()> {
        let tool_urls: Vec<&str> = server_url.split(",").collect();
        let mut successful_connections = 0;
        let mut errors = Vec::new();
        
        // Clear existing (like Python does)
        self.tool_descriptions = HashMap::new();
        self.urls = HashMap::new();
        self.tool_cache.clear();
        
        for url_str in tool_urls {
            let url_str = url_str.trim();
            
            // Determine connection type and format URL
            let formatted_url = if url_str.starts_with("http://") || url_str.starts_with("https://") {
                url_str.to_string()
            } else {
                format!("http://{}/sse", url_str)
            };
            
            // Enhanced server connection with retry and error recovery
            match self.connect_to_server(&formatted_url).await {
                Ok((_init_response, tools_response)) => {
                    // Process tools with enhanced validation
                    let processed_tools = post_process_tools_description(tools_response);
                    
                    // Enhanced tool storage with conflict detection
                    if let Ok(tools_obj) = serde_json::from_value::<ListToolsResponse>(processed_tools.clone()) {
                        for tool in &tools_obj.tools {
                            let tool_name = &tool.name;
                            
                            // Check for duplicate tools (like Python's warning)
                            if self.tool_descriptions.contains_key(tool_name) {
                                tracing::warn!(
                                    "Tool {} already exists. Ignoring duplicate tool from server {}",
                                    tool_name, 
                                    formatted_url
                                );
                                continue;
                            }
                            
                            // Store individual tool descriptions (matching Python's approach)
                            let tool_json = json!(tool);
                            self.tool_descriptions.insert(tool_name.clone(), tool_json.clone());
                            self.urls.insert(tool_name.clone(), formatted_url.clone());
                            
                            // Update cache with tool metadata
                            self.tool_cache.update_tool(tool_name, &tool_json, &formatted_url);
                        }
                    }
                    
                    successful_connections += 1;
                },
                Err(e) => {
                    errors.push(format!("Failed to connect to {}: {}", formatted_url, e));
                    tracing::warn!("Failed to connect to MCP server {}: {}", formatted_url, e);
                }
            }
        }
        
        // Enhanced error handling - succeed if at least one server connects
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
    
    /// Enhanced server connection with retries (internal helper)
    async fn connect_to_server(&self, url: &str) -> MCPResult<(InitializeResponse, ListToolsResponse)> {
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
                            attempt, MAX_RETRIES, url, last_error.as_ref().unwrap()
                        );
                        tokio::time::sleep(tokio::time::Duration::from_millis(RETRY_DELAY_MS * attempt as u64)).await;
                    }
                }
            }
        }
        
        Err(last_error.unwrap())
    }
    
    /// Add stdio-based MCP server (for local development)
    pub async fn add_stdio_server(&mut self, _command: String) -> MCPResult<()> {
        // This would start a local process and communicate via stdin/stdout
        // For now, placeholder implementation
        Err(MCPError::ProtocolError("Stdio servers not yet implemented".to_string()))
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
        let url = self.urls.get(tool_name).ok_or_else(|| {
            MCPError::ToolNotFound(tool_name.to_string())
        })?;
        
        // Create session (like Python's async context manager)
        ToolSession::new(url.clone()).await
    }
    
    /// Get tools by category (enhanced caching feature)
    pub fn get_tools_by_category(&self, category: &str) -> Vec<String> {
        self.tool_cache.get_tools_by_category(category)
    }
    
    /// Get tools by parameter (enhanced caching feature)
    pub fn get_tools_by_parameter(&self, parameter: &str) -> Vec<String> {
        self.tool_cache.get_tools_by_parameter(parameter)
    }
    
    /// Get tool metadata (enhanced caching feature)
    pub fn get_tool_metadata(&self, tool_name: &str) -> Option<&ToolMetadata> {
        self.tool_cache.metadata_cache.get(tool_name)
    }
    
    /// List all available tools (enhanced version)
    pub fn list_tools(&self) -> Vec<String> {
        self.tool_descriptions.keys().cloned().collect()
    }
    
    /// Get tool statistics (enhanced feature)
    pub fn get_tool_stats(&self) -> ToolStats {
        ToolStats {
            total_tools: self.tool_descriptions.len(),
            total_servers: self.urls.values().collect::<std::collections::HashSet<_>>().len(),
            categories: self.tool_cache.category_index.len(),
            parameters: self.tool_cache.parameter_index.len(),
            last_updated: self.tool_cache.last_updated,
        }
    }
    
    /// List all connected servers (Phase 4 enhancement)
    pub fn list_servers(&self) -> Vec<String> {
        self.urls.values().cloned().collect::<std::collections::HashSet<_>>().into_iter().collect()
    }
    
    /// Check if a specific server is connected
    pub fn has_server(&self, server_url: &str) -> bool {
        self.urls.values().any(|url| url == server_url)
    }
}

/// Tool statistics for monitoring
#[derive(Debug, Clone)]
pub struct ToolStats {
    pub total_tools: usize,
    pub total_servers: usize,
    pub categories: usize,
    pub parameters: usize,
    pub last_updated: Option<std::time::Instant>,
}

/// Match Python's list_server_and_tools() function exactly
async fn list_server_and_tools(server_url: &str) -> MCPResult<(InitializeResponse, ListToolsResponse)> {
    // Python: async with sse_client(url=server_url) as streams, ClientSession(*streams) as session:
    let client = reqwest::Client::new();
    
    // Initialize session (matching Python's session.initialize())
    let init_request = MCPRequest {
        jsonrpc: "2.0".to_string(),
        id: "1".to_string(),
        method: "initialize".to_string(),
        params: Some(json!({
            "protocolVersion": "1.0",
            "capabilities": {}
        })),
    };
    
    let init_response = send_sse_request(&client, server_url, init_request).await?;
    let init_result: InitializeResponse = serde_json::from_value(init_response)
        .map_err(|e| MCPError::SerializationError(e.to_string()))?;
    
    // List tools (matching Python's session.list_tools())
    let tools_request = MCPRequest {
        jsonrpc: "2.0".to_string(),
        id: "2".to_string(),
        method: "list_tools".to_string(),
        params: Some(json!({})),
    };
    
    let tools_response = send_sse_request(&client, server_url, tools_request).await?;
    let tools_result: ListToolsResponse = serde_json::from_value(tools_response)
        .map_err(|e| MCPError::SerializationError(e.to_string()))?;
    
    Ok((init_result, tools_result))
}

/// Helper function for SSE requests (matching Python's SSE client approach)
async fn send_sse_request(
    client: &reqwest::Client,
    url: &str,
    request: MCPRequest,
) -> MCPResult<Value> {
    
    // Try SSE first (like Python's sse_client), fallback to HTTP POST
    if let Ok(response) = send_sse_stream_request(client, url, &request).await {
        return Ok(response);
    }
    
    // Fallback to simple HTTP POST for MCP servers that don't support SSE
    let response = client
        .post(url)
        .json(&request)
        .send()
        .await
        .map_err(|e| MCPError::ConnectionError(e.to_string()))?;
    
    let mcp_response: MCPResponse = response
        .json()
        .await
        .map_err(|e| MCPError::ConnectionError(e.to_string()))?;
    
    if let Some(error) = mcp_response.error {
        return Err(MCPError::ProtocolError(error.message));
    }
    
    mcp_response.result.ok_or_else(|| MCPError::ProtocolError("No result".to_string()))
}

/// SSE stream request implementation (matching Python's SSE approach)
async fn send_sse_stream_request(
    client: &reqwest::Client,
    url: &str,
    request: &MCPRequest,
) -> MCPResult<Value> {
    use futures_util::StreamExt;
    
    // Send SSE request with proper headers
    let response = client
        .post(url)
        .header("Accept", "text/event-stream")
        .header("Cache-Control", "no-cache")
        .json(request)
        .send()
        .await
        .map_err(|e| MCPError::ConnectionError(e.to_string()))?;
    
    // Check if response is SSE
    let content_type = response.headers()
        .get("content-type")
        .and_then(|ct| ct.to_str().ok())
        .unwrap_or("");
    
    if !content_type.contains("text/event-stream") {
        return Err(MCPError::ProtocolError("Not an SSE stream".to_string()));
    }
    
    // Process SSE stream
    let mut stream = response.bytes_stream();
    let mut buffer = String::new();
    
    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|e| MCPError::ConnectionError(e.to_string()))?;
        let chunk_str = std::str::from_utf8(&chunk)
            .map_err(|e| MCPError::ProtocolError(format!("Invalid UTF-8: {}", e)))?;
        
        buffer.push_str(chunk_str);
        
        // Process complete SSE events (ending with \n\n)
        while let Some(event_end) = buffer.find("\n\n") {
            let event = buffer[..event_end].to_string();
            buffer = buffer[event_end + 2..].to_string();
            
            if let Some(response) = parse_sse_event(&event)? {
                return Ok(response);
            }
        }
    }
    
    Err(MCPError::ProtocolError("SSE stream ended without response".to_string()))
}

/// Parse SSE event format (matching Python's SSE event parsing)
pub fn parse_sse_event(event: &str) -> MCPResult<Option<Value>> {
    let mut data_lines = Vec::new();
    
    for line in event.lines() {
        if line.starts_with("data: ") {
            data_lines.push(&line[6..]);
        }
    }
    
    if data_lines.is_empty() {
        return Ok(None);
    }
    
    let json_data = data_lines.join("\n");
    if json_data.trim().is_empty() {
        return Ok(None);
    }
    
    let mcp_response: MCPResponse = serde_json::from_str(&json_data)
        .map_err(|e| MCPError::SerializationError(e.to_string()))?;
    
    if let Some(error) = mcp_response.error {
        return Err(MCPError::ProtocolError(error.message));
    }
    
    Ok(mcp_response.result)
}

/// Enhanced schema adaptation matching Python's trim_schema() with additional features
fn trim_schema(mut schema: Value) -> Value {
    if let Some(obj) = schema.as_object_mut() {
        // Remove title and null defaults (exactly like Python)
        obj.remove("title");
        if obj.get("default") == Some(&Value::Null) {
            obj.remove("default");
        }
        
        // Convert anyOf to type arrays (enhanced version)
        if let Some(any_of) = obj.remove("anyOf") {
            if let Some(array) = any_of.as_array() {
                let types: Vec<String> = array.iter()
                    .filter_map(|item| {
                        item.get("type")
                            .and_then(|t| t.as_str())
                            .filter(|t| *t != "null")
                            .map(|t| t.to_string())
                    })
                    .collect();
                
                // Handle single type vs array of types
                match types.len() {
                    0 => {}, // No valid types found
                    1 => {
                        obj.insert("type".to_string(), json!(types[0]));
                    },
                    _ => {
                        obj.insert("type".to_string(), json!(types));
                    }
                }
            }
        }
        
        // Handle oneOf similar to anyOf (additional enhancement)
        if let Some(one_of) = obj.remove("oneOf") {
            if let Some(array) = one_of.as_array() {
                let types: Vec<String> = array.iter()
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
        
        // Recursive processing for properties (exactly like Python)
        if let Some(properties) = obj.get_mut("properties") {
            if let Some(props_obj) = properties.as_object_mut() {
                for (_, value) in props_obj.iter_mut() {
                    *value = trim_schema(value.clone());
                }
            }
        }
        
        // Handle nested schemas in items (for arrays)
        if let Some(items) = obj.get_mut("items") {
            *items = trim_schema(items.clone());
        }
        
        // Handle nested schemas in additionalProperties (for objects)
        if let Some(additional_props) = obj.get_mut("additionalProperties") {
            if additional_props.is_object() {
                *additional_props = trim_schema(additional_props.clone());
            }
        }
        
        // Handle patternProperties (for dynamic property names)
        if let Some(pattern_props) = obj.get_mut("patternProperties") {
            if let Some(pattern_obj) = pattern_props.as_object_mut() {
                for (_, value) in pattern_obj.iter_mut() {
                    *value = trim_schema(value.clone());
                }
            }
        }
        
        // Handle allOf in nested contexts
        if let Some(all_of) = obj.get_mut("allOf") {
            if let Some(array) = all_of.as_array_mut() {
                for item in array.iter_mut() {
                    *item = trim_schema(item.clone());
                }
            }
        }
    }
    
    schema
}

/// Enhanced tool processing with advanced filtering (matching Python plus improvements)
fn post_process_tools_description(mut tools_response: ListToolsResponse) -> Value {
    // Adapt schemas for Harmony (exactly like Python)
    for tool in &mut tools_response.tools {
        tool.input_schema = trim_schema(tool.input_schema.clone());
    }
    
    // Enhanced tool filtering based on annotations
    let initial_count = tools_response.tools.len();
    
    tools_response.tools.retain(|tool| {
        // Check include_in_prompt annotation (Python behavior)
        let include_in_prompt = tool.annotations
            .as_ref()
            .and_then(|a| a.get("include_in_prompt"))
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        
        if !include_in_prompt {
            tracing::debug!("Filtering out tool '{}' due to include_in_prompt=false", tool.name);
            return false;
        }
        
        // Additional filtering: Check if tool is explicitly disabled
        let disabled = tool.annotations
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
    
    serde_json::to_value(tools_response).unwrap_or(json!({}))
}

/// Extract categories from tool annotations or description
fn extract_categories(tool_info: &Value) -> Vec<String> {
    let mut categories = Vec::new();
    
    // Check annotations for categories
    if let Some(annotations) = tool_info.get("annotations") {
        if let Some(cats) = annotations.get("categories") {
            if let Some(array) = cats.as_array() {
                for cat in array {
                    if let Some(cat_str) = cat.as_str() {
                        categories.push(cat_str.to_string());
                    }
                }
            }
        }
        
        if let Some(tags) = annotations.get("tags") {
            if let Some(array) = tags.as_array() {
                for tag in array {
                    if let Some(tag_str) = tag.as_str() {
                        categories.push(tag_str.to_string());
                    }
                }
            }
        }
    }
    
    // Fallback: try to infer category from tool name
    if categories.is_empty() {
        if let Some(name) = tool_info.get("name").and_then(|n| n.as_str()) {
            if name.contains("browser") || name.contains("web") {
                categories.push("web".to_string());
            } else if name.contains("python") || name.contains("code") {
                categories.push("code".to_string());
            } else if name.contains("file") || name.contains("fs") {
                categories.push("filesystem".to_string());
            } else {
                categories.push("general".to_string());
            }
        }
    }
    
    categories
}

/// Count parameters in tool input schema
fn count_parameters(tool_info: &Value) -> usize {
    if let Some(schema) = tool_info.get("input_schema").or_else(|| tool_info.get("inputSchema")) {
        if let Some(properties) = schema.get("properties") {
            if let Some(props_obj) = properties.as_object() {
                return props_obj.len();
            }
        }
    }
    0
}

/// Check if tool has required parameters
fn has_required_parameters(tool_info: &Value) -> bool {
    if let Some(schema) = tool_info.get("input_schema").or_else(|| tool_info.get("inputSchema")) {
        if let Some(required) = schema.get("required") {
            if let Some(required_array) = required.as_array() {
                return !required_array.is_empty();
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcp_tool_server_new() {
        let server = MCPToolServer::new();
        assert_eq!(server.tool_descriptions.len(), 0);
        assert_eq!(server.urls.len(), 0);
    }

    #[test]
    fn test_trim_schema() {
        let schema = json!({
            "title": "Test Schema",
            "type": "object",
            "default": null,
            "anyOf": [
                {"type": "string"},
                {"type": "null"}
            ]
        });

        let result = trim_schema(schema);
        
        // Should remove title and null default
        assert!(result.get("title").is_none());
        assert!(result.get("default").is_none());
        
        // Should convert anyOf to single type when only one non-null type
        assert_eq!(result.get("type").unwrap(), &json!("string"));
    }
    
    #[test]
    fn test_trim_schema_multiple_types() {
        let schema = json!({
            "title": "Multi Type Schema",
            "anyOf": [
                {"type": "string"},
                {"type": "number"},
                {"type": "null"}
            ]
        });

        let result = trim_schema(schema);
        
        // Should convert anyOf to type array for multiple types (without null)
        assert_eq!(result.get("type").unwrap(), &json!(["string", "number"]));
        assert!(result.get("title").is_none());
    }
    
    #[test]
    fn test_trim_schema_nested() {
        let schema = json!({
            "type": "object",
            "properties": {
                "nested": {
                    "title": "Nested Field",
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"}
                    ]
                }
            }
        });

        let result = trim_schema(schema);
        
        // Should recursively process nested properties
        let nested = result.get("properties")
            .unwrap()
            .get("nested")
            .unwrap();
        assert!(nested.get("title").is_none());
        assert_eq!(nested.get("type").unwrap(), &json!("string"));
    }

    #[test]
    fn test_has_tool() {
        let mut server = MCPToolServer::new();
        assert!(!server.has_tool("test_tool"));
        
        server.tool_descriptions.insert("test_tool".to_string(), json!({}));
        assert!(server.has_tool("test_tool"));
    }
}