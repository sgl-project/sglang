use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::config::types::McpProxyConfig;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpConfig {
    pub servers: Vec<McpServerConfig>,
    /// Global proxy configuration for all MCP servers
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proxy: Option<McpProxyConfig>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpServerConfig {
    pub name: String,
    #[serde(flatten)]
    pub transport: McpTransport,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "protocol", rename_all = "lowercase")]
pub enum McpTransport {
    Stdio {
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        envs: HashMap<String, String>,
    },
    Sse {
        url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        token: Option<String>,
    },
    Streamable {
        url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        token: Option<String>,
    },
}

impl McpConfig {
    /// Load configuration from a YAML file
    pub async fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = tokio::fs::read_to_string(path).await?;
        let config: Self = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    /// Load configuration from environment variables (optional)
    pub fn from_env() -> Option<Self> {
        // This could be expanded to read from env vars
        // For now, return None to indicate env config not implemented
        None
    }
}
