//! MCP configuration types and utilities.
//!
//! Defines configuration structures for MCP servers, transports, proxies, and inventory.

use std::{collections::HashMap, fmt};

pub use rmcp::model::{Prompt, RawResource, Tool};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpConfig {
    /// Static MCP servers (loaded at startup)
    pub servers: Vec<McpServerConfig>,

    /// Connection pool settings
    #[serde(default)]
    pub pool: McpPoolConfig,

    /// Global MCP proxy configuration (default for all servers)
    /// Can be overridden per-server
    #[serde(default)]
    pub proxy: Option<McpProxyConfig>,

    /// Pre-warm these connections at startup
    #[serde(default)]
    pub warmup: Vec<WarmupServer>,

    /// Tool inventory refresh settings
    #[serde(default)]
    pub inventory: InventoryConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpServerConfig {
    pub name: String,
    #[serde(flatten)]
    pub transport: McpTransport,

    /// Per-server proxy override (overrides global proxy)
    /// Set to `null` in YAML to force direct connection (no proxy)
    #[serde(default)]
    pub proxy: Option<McpProxyConfig>,

    /// Whether this server is required for router startup
    /// - true: Router startup fails if this server cannot be reached
    /// - false: Log warning but continue (default)
    #[serde(default)]
    pub required: bool,
}

#[derive(Clone, Deserialize, Serialize)]
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

impl fmt::Debug for McpTransport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            McpTransport::Stdio {
                command,
                args,
                envs,
            } => f
                .debug_struct("Stdio")
                .field("command", command)
                .field("args", args)
                .field("envs", envs)
                .finish(),
            McpTransport::Sse { url, token } => f
                .debug_struct("Sse")
                .field("url", url)
                .field("token", &token.as_ref().map(|_| "****"))
                .finish(),
            McpTransport::Streamable { url, token } => f
                .debug_struct("Streamable")
                .field("url", url)
                .field("token", &token.as_ref().map(|_| "****"))
                .finish(),
        }
    }
}

/// MCP-specific proxy configuration (does NOT affect LLM API traffic)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpProxyConfig {
    /// HTTP proxy URL (e.g., "http://proxy.internal:8080")
    pub http: Option<String>,

    /// HTTPS proxy URL
    pub https: Option<String>,

    /// Comma-separated hosts to exclude from proxying
    /// Example: "localhost,127.0.0.1,*.internal,10.*"
    pub no_proxy: Option<String>,

    /// Custom proxy authentication (if needed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub username: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub password: Option<String>,
}

/// Connection pool configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpPoolConfig {
    /// Maximum cached connections per server URL
    #[serde(default = "default_max_connections")]
    pub max_connections: usize,

    /// Idle timeout before closing connection (seconds)
    #[serde(default = "default_idle_timeout")]
    pub idle_timeout: u64,
}

/// Tool inventory refresh configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InventoryConfig {
    /// Enable automatic tool inventory refresh
    #[serde(default = "default_true")]
    pub enable_refresh: bool,

    /// Tool cache TTL (seconds) - how long tools are considered fresh
    #[serde(default = "default_tool_ttl")]
    pub tool_ttl: u64,

    /// Background refresh interval (seconds) - proactive refresh
    #[serde(default = "default_refresh_interval")]
    pub refresh_interval: u64,

    /// Refresh on tool call failure (try refreshing if tool not found)
    #[serde(default = "default_true")]
    pub refresh_on_error: bool,
}

/// Pre-warm server connections at startup
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WarmupServer {
    /// Server URL
    pub url: String,

    /// Server label/name
    pub label: String,

    /// Optional authentication token
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
}

// Default value functions
fn default_max_connections() -> usize {
    100
}

fn default_idle_timeout() -> u64 {
    300 // 5 minutes
}

fn default_true() -> bool {
    true
}

fn default_tool_ttl() -> u64 {
    300 // 5 minutes
}

fn default_refresh_interval() -> u64 {
    60 // 1 minute
}

// Default implementations
impl Default for McpPoolConfig {
    fn default() -> Self {
        Self {
            max_connections: default_max_connections(),
            idle_timeout: default_idle_timeout(),
        }
    }
}

impl Default for InventoryConfig {
    fn default() -> Self {
        Self {
            enable_refresh: true,
            tool_ttl: default_tool_ttl(),
            refresh_interval: default_refresh_interval(),
            refresh_on_error: true,
        }
    }
}

impl McpProxyConfig {
    /// Load proxy config from standard environment variables
    pub fn from_env() -> Option<Self> {
        let http = std::env::var("MCP_HTTP_PROXY")
            .ok()
            .or_else(|| std::env::var("HTTP_PROXY").ok());

        let https = std::env::var("MCP_HTTPS_PROXY")
            .ok()
            .or_else(|| std::env::var("HTTPS_PROXY").ok());

        let no_proxy = std::env::var("MCP_NO_PROXY")
            .ok()
            .or_else(|| std::env::var("NO_PROXY").ok());

        if http.is_some() || https.is_some() {
            Some(Self {
                http,
                https,
                no_proxy,
                username: None,
                password: None,
            })
        } else {
            None
        }
    }
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

    /// Merge with environment-based proxy config
    pub fn with_env_proxy(mut self) -> Self {
        if self.proxy.is_none() {
            self.proxy = McpProxyConfig::from_env();
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_pool_config() {
        let config = McpPoolConfig::default();
        assert_eq!(config.max_connections, 100);
        assert_eq!(config.idle_timeout, 300);
    }

    #[test]
    fn test_default_inventory_config() {
        let config = InventoryConfig::default();
        assert!(config.enable_refresh);
        assert_eq!(config.tool_ttl, 300);
        assert_eq!(config.refresh_interval, 60);
        assert!(config.refresh_on_error);
    }

    #[test]
    fn test_proxy_from_env_empty() {
        // Ensure no proxy env vars are set for this test
        std::env::remove_var("MCP_HTTP_PROXY");
        std::env::remove_var("MCP_HTTPS_PROXY");
        std::env::remove_var("HTTP_PROXY");
        std::env::remove_var("HTTPS_PROXY");

        let proxy = McpProxyConfig::from_env();
        assert!(proxy.is_none(), "Should return None when no env vars set");
    }

    #[test]
    fn test_proxy_from_env_with_vars() {
        std::env::set_var("MCP_HTTP_PROXY", "http://test-proxy:8080");
        std::env::set_var("MCP_NO_PROXY", "localhost,127.0.0.1");

        let proxy = McpProxyConfig::from_env();
        assert!(proxy.is_some(), "Should return Some when env vars set");

        let proxy = proxy.unwrap();
        assert_eq!(proxy.http.as_ref().unwrap(), "http://test-proxy:8080");
        assert_eq!(proxy.no_proxy.as_ref().unwrap(), "localhost,127.0.0.1");

        // Cleanup
        std::env::remove_var("MCP_HTTP_PROXY");
        std::env::remove_var("MCP_NO_PROXY");
    }

    #[tokio::test]
    async fn test_yaml_minimal_config() {
        let yaml = r#"
servers:
  - name: "test-server"
    protocol: sse
    url: "http://localhost:3000/sse"
"#;

        let config: McpConfig = serde_yaml::from_str(yaml).expect("Failed to parse YAML");
        assert_eq!(config.servers.len(), 1);
        assert_eq!(config.servers[0].name, "test-server");
        assert!(!config.servers[0].required); // Should default to false
        assert!(config.servers[0].proxy.is_none()); // Should default to None
        assert_eq!(config.pool.max_connections, 100); // Should use default
        assert_eq!(config.inventory.tool_ttl, 300); // Should use default
    }

    #[tokio::test]
    async fn test_yaml_full_config() {
        let yaml = r#"
# Global proxy configuration
proxy:
  http: "http://global-proxy:8080"
  https: "http://global-proxy:8080"
  no_proxy: "localhost,127.0.0.1,*.internal"

# Connection pool settings
pool:
  max_connections: 50
  idle_timeout: 600

# Tool inventory settings
inventory:
  enable_refresh: true
  tool_ttl: 600
  refresh_interval: 120
  refresh_on_error: true

# Static servers
servers:
  - name: "required-server"
    protocol: sse
    url: "https://api.example.com/sse"
    token: "secret-token"
    required: true

  - name: "optional-server"
    protocol: stdio
    command: "mcp-server"
    args: ["--port", "3000"]
    required: false
    proxy:
      http: "http://server-specific-proxy:9090"

# Pre-warm connections
warmup:
  - url: "http://localhost:3000/sse"
    label: "local-dev"
"#;

        let config: McpConfig = serde_yaml::from_str(yaml).expect("Failed to parse YAML");

        // Check global proxy
        assert!(config.proxy.is_some());
        let global_proxy = config.proxy.as_ref().unwrap();
        assert_eq!(
            global_proxy.http.as_ref().unwrap(),
            "http://global-proxy:8080"
        );

        // Check pool config
        assert_eq!(config.pool.max_connections, 50);
        assert_eq!(config.pool.idle_timeout, 600);

        // Check inventory config
        assert_eq!(config.inventory.tool_ttl, 600);
        assert_eq!(config.inventory.refresh_interval, 120);

        // Check servers
        assert_eq!(config.servers.len(), 2);

        // Required server
        assert_eq!(config.servers[0].name, "required-server");
        assert!(config.servers[0].required);
        assert!(config.servers[0].proxy.is_none()); // Inherits global proxy

        // Optional server with custom proxy
        assert_eq!(config.servers[1].name, "optional-server");
        assert!(!config.servers[1].required);
        assert!(config.servers[1].proxy.is_some());
        assert_eq!(
            config.servers[1]
                .proxy
                .as_ref()
                .unwrap()
                .http
                .as_ref()
                .unwrap(),
            "http://server-specific-proxy:9090"
        );

        // Check warmup
        assert_eq!(config.warmup.len(), 1);
        assert_eq!(config.warmup[0].label, "local-dev");
    }

    #[tokio::test]
    async fn test_yaml_backward_compatibility() {
        // Old config format should still work
        let yaml = r#"
servers:
  - name: "legacy-server"
    protocol: sse
    url: "http://localhost:3000/sse"
"#;

        let config: McpConfig = serde_yaml::from_str(yaml).expect("Failed to parse old format");
        assert_eq!(config.servers.len(), 1);
        assert_eq!(config.servers[0].name, "legacy-server");
        assert!(!config.servers[0].required); // New field defaults to false
        assert!(config.servers[0].proxy.is_none()); // New field defaults to None
        assert!(config.proxy.is_none()); // New field defaults to None
        assert!(config.warmup.is_empty()); // New field defaults to empty
    }

    #[tokio::test]
    async fn test_yaml_null_proxy_override() {
        // Test that explicit null in YAML sets proxy to None
        let yaml = r#"
proxy:
  http: "http://global-proxy:8080"

servers:
  - name: "direct-connection"
    protocol: sse
    url: "http://localhost:3000/sse"
    proxy: null
"#;

        let config: McpConfig = serde_yaml::from_str(yaml).expect("Failed to parse YAML");
        assert!(config.proxy.is_some()); // Global proxy set
        assert_eq!(config.servers.len(), 1);
        assert!(config.servers[0].proxy.is_none()); // Explicitly set to null
    }

    #[test]
    fn test_transport_stdio() {
        let yaml = r#"
name: "test"
protocol: stdio
command: "mcp-server"
args: ["--port", "3000"]
envs:
  VAR1: "value1"
  VAR2: "value2"
"#;

        let config: McpServerConfig = serde_yaml::from_str(yaml).expect("Failed to parse stdio");
        assert_eq!(config.name, "test");

        match config.transport {
            McpTransport::Stdio {
                command,
                args,
                envs,
            } => {
                assert_eq!(command, "mcp-server");
                assert_eq!(args.len(), 2);
                assert_eq!(args[0], "--port");
                assert_eq!(envs.get("VAR1").unwrap(), "value1");
            }
            _ => panic!("Expected Stdio transport"),
        }
    }

    #[test]
    fn test_transport_sse() {
        let yaml = r#"
name: "test"
protocol: sse
url: "http://localhost:3000/sse"
token: "secret"
"#;

        let config: McpServerConfig = serde_yaml::from_str(yaml).expect("Failed to parse sse");
        assert_eq!(config.name, "test");

        match config.transport {
            McpTransport::Sse { url, token } => {
                assert_eq!(url, "http://localhost:3000/sse");
                assert_eq!(token.unwrap(), "secret");
            }
            _ => panic!("Expected Sse transport"),
        }
    }

    #[test]
    fn test_transport_streamable() {
        let yaml = r#"
name: "test"
protocol: streamable
url: "http://localhost:3000"
"#;

        let config: McpServerConfig =
            serde_yaml::from_str(yaml).expect("Failed to parse streamable");
        assert_eq!(config.name, "test");

        match config.transport {
            McpTransport::Streamable { url, token } => {
                assert_eq!(url, "http://localhost:3000");
                assert!(token.is_none());
            }
            _ => panic!("Expected Streamable transport"),
        }
    }
}
