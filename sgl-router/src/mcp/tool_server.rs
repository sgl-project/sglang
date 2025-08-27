// tool_server.rs - Main MCP implementation (matching Python's tool_server.py)
use crate::mcp::types::*;
use std::collections::HashMap;
use serde_json::{json, Value};

/// Main MCP Tool Server (matching Python's MCPToolServer class)
pub struct MCPToolServer {
    /// Tool descriptions by server (matching Python's harmony_tool_descriptions)
    tool_descriptions: HashMap<String, Value>,
    /// Server URLs (matching Python's urls dict)
    urls: HashMap<String, String>,
}

impl MCPToolServer {
    /// Create new MCPToolServer (matching Python's __init__)
    pub fn new() -> Self {
        Self {
            tool_descriptions: HashMap::new(),
            urls: HashMap::new(),
        }
    }

    /// Add tool server (matching Python's add_tool_server exactly)
    pub async fn add_tool_server(&mut self, server_url: String) -> MCPResult<()> {
        let tool_urls: Vec<&str> = server_url.split(",").collect();
        
        // Clear existing (like Python does)
        self.tool_descriptions = HashMap::new();
        self.urls = HashMap::new();
        
        for url in tool_urls {
            let url = format!("http://{}/sse", url.trim());
            
            // Call helper function (like Python's list_server_and_tools)
            let (init_response, tools_response) = list_server_and_tools(&url).await?;
            
            // Process tools (like Python's post_process_tools_description)
            let processed_tools = post_process_tools_description(tools_response);
            
            // Store tools by server name
            let server_name = &init_response.server_info.name;
            self.tool_descriptions.insert(server_name.clone(), processed_tools);
            if !self.urls.contains_key(server_name) {
                self.urls.insert(server_name.clone(), url);
            }
        }
        
        Ok(())
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

/// Helper function for SSE requests (simple, like Python)
async fn send_sse_request(
    client: &reqwest::Client,
    url: &str,
    request: MCPRequest,
) -> MCPResult<Value> {
    // For now, implement as simple HTTP POST (will enhance to SSE later)
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

/// Match Python's trim_schema() function exactly
fn trim_schema(mut schema: Value) -> Value {
    if let Some(obj) = schema.as_object_mut() {
        // Remove title and null defaults (exactly like Python)
        obj.remove("title");
        if obj.get("default") == Some(&Value::Null) {
            obj.remove("default");
        }
        
        // Convert anyOf to type arrays (exactly like Python)
        if let Some(any_of) = obj.remove("anyOf") {
            if let Some(array) = any_of.as_array() {
                let types: Vec<String> = array.iter()
                    .filter_map(|item| {
                        item.get("type")?
                            .as_str()
                            .filter(|t| *t != "null")
                            .map(|t| t.to_string())
                    })
                    .collect();
                if !types.is_empty() {
                    obj.insert("type".to_string(), json!(types));
                }
            }
        }
        
        // Recursive processing (exactly like Python)
        if let Some(properties) = obj.get_mut("properties") {
            if let Some(props_obj) = properties.as_object_mut() {
                for (_, value) in props_obj.iter_mut() {
                    *value = trim_schema(value.clone());
                }
            }
        }
    }
    
    schema
}

/// Match Python's post_process_tools_description() function
fn post_process_tools_description(mut tools_response: ListToolsResponse) -> Value {
    // Adapt schemas for Harmony (exactly like Python)
    for tool in &mut tools_response.tools {
        tool.input_schema = trim_schema(tool.input_schema.clone());
    }
    
    // Filter tools based on annotations (exactly like Python)
    tools_response.tools.retain(|tool| {
        tool.annotations
            .as_ref()
            .and_then(|a| a.get("include_in_prompt"))
            .and_then(|v| v.as_bool())
            .unwrap_or(true)
    });
    
    serde_json::to_value(tools_response).unwrap_or(json!({}))
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
        
        // Should convert anyOf to type array without null
        assert_eq!(result.get("type").unwrap(), &json!(["string"]));
    }

    #[test]
    fn test_has_tool() {
        let mut server = MCPToolServer::new();
        assert!(!server.has_tool("test_tool"));
        
        server.tool_descriptions.insert("test_tool".to_string(), json!({}));
        assert!(server.has_tool("test_tool"));
    }
}