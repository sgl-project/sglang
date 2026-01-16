//! HTTP proxy configuration for MCP connections.
//!
//! Resolves proxy settings and creates HTTP clients for MCP server connections.

use std::time::Duration;

use crate::mcp::{
    config::{McpProxyConfig, McpServerConfig},
    error::{McpError, McpResult},
};

/// Resolve proxy configuration for a server
/// Priority: server.proxy > global.proxy > None
///
/// # Arguments
/// * `server_config` - Server-specific configuration
/// * `global_proxy` - Global proxy configuration from McpConfig
///
/// # Returns
/// The resolved proxy configuration, or None for direct connection
pub(crate) fn resolve_proxy_config<'a>(
    server_config: &'a McpServerConfig,
    global_proxy: Option<&'a McpProxyConfig>,
) -> Option<&'a McpProxyConfig> {
    // Priority 1: Check if server has explicit proxy config
    // Note: server.proxy = Some(config) uses that config
    //       server.proxy = None (set explicitly in YAML as null) forces direct connection
    //       server.proxy not set (field missing) falls back to global
    if server_config.proxy.is_some() {
        server_config.proxy.as_ref()
    } else {
        // Priority 2: Fall back to global proxy
        global_proxy
    }
}

/// Apply proxy configuration to a ClientBuilder
///
/// This is a reusable helper that applies proxy settings without building the client,
/// allowing additional configuration (like auth headers) to be added afterward.
///
/// # Arguments
/// * `builder` - The reqwest::ClientBuilder to configure
/// * `proxy_config` - The proxy configuration to apply
///
/// # Returns
/// The configured builder or error
pub(super) fn apply_proxy_to_builder(
    mut builder: reqwest::ClientBuilder,
    proxy_cfg: &McpProxyConfig,
) -> McpResult<reqwest::ClientBuilder> {
    // Configure HTTP proxy
    if let Some(ref http_proxy) = proxy_cfg.http {
        let mut proxy = reqwest::Proxy::http(http_proxy)
            .map_err(|e| McpError::Config(format!("Invalid HTTP proxy: {}", e)))?;

        // Apply no_proxy exclusions
        if let Some(ref no_proxy) = proxy_cfg.no_proxy {
            proxy = proxy.no_proxy(reqwest::NoProxy::from_string(no_proxy));
        }

        // Apply authentication if configured
        if let (Some(ref username), Some(ref password)) = (&proxy_cfg.username, &proxy_cfg.password)
        {
            proxy = proxy.basic_auth(username, password);
        }

        builder = builder.proxy(proxy);
    }

    // Configure HTTPS proxy
    if let Some(ref https_proxy) = proxy_cfg.https {
        let mut proxy = reqwest::Proxy::https(https_proxy)
            .map_err(|e| McpError::Config(format!("Invalid HTTPS proxy: {}", e)))?;

        // Apply no_proxy exclusions
        if let Some(ref no_proxy) = proxy_cfg.no_proxy {
            proxy = proxy.no_proxy(reqwest::NoProxy::from_string(no_proxy));
        }

        // Apply authentication if configured
        if let (Some(ref username), Some(ref password)) = (&proxy_cfg.username, &proxy_cfg.password)
        {
            proxy = proxy.basic_auth(username, password);
        }

        builder = builder.proxy(proxy);
    }

    Ok(builder)
}

/// Create HTTP client with MCP-specific proxy configuration
///
/// # Arguments
/// * `proxy_config` - Optional proxy configuration to apply
///
/// # Returns
/// A configured reqwest::Client or error
pub(crate) fn create_http_client(
    proxy_config: Option<&McpProxyConfig>,
) -> McpResult<reqwest::Client> {
    let mut builder = reqwest::Client::builder().connect_timeout(Duration::from_secs(10));

    // Apply MCP-specific proxy if configured
    if let Some(proxy_cfg) = proxy_config {
        builder = apply_proxy_to_builder(builder, proxy_cfg)?;
    }

    builder
        .build()
        .map_err(|e| McpError::Transport(format!("Failed to build HTTP client: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::config::McpTransport;

    #[test]
    fn test_resolve_proxy_no_config() {
        let server = McpServerConfig {
            name: "test".to_string(),
            transport: McpTransport::Sse {
                url: "http://localhost:3000/sse".to_string(),
                token: None,
                headers: None,
            },
            proxy: None,
            required: false,
        };

        let result = resolve_proxy_config(&server, None);
        assert!(
            result.is_none(),
            "Should return None when no proxy configured"
        );
    }

    #[test]
    fn test_resolve_proxy_global_only() {
        let server = McpServerConfig {
            name: "test".to_string(),
            transport: McpTransport::Sse {
                url: "http://localhost:3000/sse".to_string(),
                token: None,
                headers: None,
            },
            proxy: None,
            required: false,
        };

        let global = McpProxyConfig {
            http: Some("http://global-proxy:8080".to_string()),
            https: None,
            no_proxy: None,
            username: None,
            password: None,
        };

        let result = resolve_proxy_config(&server, Some(&global));
        assert!(result.is_some(), "Should use global proxy");
        assert_eq!(
            result.unwrap().http.as_ref().unwrap(),
            "http://global-proxy:8080"
        );
    }

    #[test]
    fn test_resolve_proxy_server_override() {
        let server_proxy = McpProxyConfig {
            http: Some("http://server-proxy:9090".to_string()),
            https: None,
            no_proxy: None,
            username: None,
            password: None,
        };

        let server = McpServerConfig {
            name: "test".to_string(),
            transport: McpTransport::Sse {
                url: "http://localhost:3000/sse".to_string(),
                token: None,
                headers: None,
            },
            proxy: Some(server_proxy),
            required: false,
        };

        let global = McpProxyConfig {
            http: Some("http://global-proxy:8080".to_string()),
            https: None,
            no_proxy: None,
            username: None,
            password: None,
        };

        let result = resolve_proxy_config(&server, Some(&global));
        assert!(result.is_some(), "Should use server-specific proxy");
        assert_eq!(
            result.unwrap().http.as_ref().unwrap(),
            "http://server-proxy:9090",
            "Server proxy should override global"
        );
    }

    #[test]
    fn test_create_http_client_no_proxy() {
        let client = create_http_client(None);
        assert!(client.is_ok(), "Should create client without proxy");
    }

    #[test]
    fn test_create_http_client_with_proxy() {
        let proxy = McpProxyConfig {
            http: Some("http://proxy.example.com:8080".to_string()),
            https: None,
            no_proxy: Some("localhost,127.0.0.1".to_string()),
            username: None,
            password: None,
        };

        let client = create_http_client(Some(&proxy));
        assert!(client.is_ok(), "Should create client with proxy");
    }

    #[test]
    fn test_create_http_client_with_auth() {
        let proxy = McpProxyConfig {
            http: Some("http://proxy.example.com:8080".to_string()),
            https: None,
            no_proxy: None,
            username: Some("user".to_string()),
            password: Some("pass".to_string()),
        };

        let client = create_http_client(Some(&proxy));
        assert!(
            client.is_ok(),
            "Should create client with proxy authentication"
        );
    }

    #[test]
    fn test_create_http_client_invalid_proxy() {
        let proxy = McpProxyConfig {
            http: Some("://invalid".to_string()), // Invalid URL format
            https: None,
            no_proxy: None,
            username: None,
            password: None,
        };

        let client = create_http_client(Some(&proxy));
        assert!(client.is_err(), "Should fail with invalid proxy URL");
    }
}
