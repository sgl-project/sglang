//! Global HTTP client management for MCP connections
//!
//! This module provides a globally shared HTTP client for all MCP connections,
//! initialized once during server startup with optional proxy configuration.

use std::sync::OnceLock;

use reqwest::header::HeaderMap;

use crate::config::types::McpProxyConfig;

/// Globally shared HTTP client for all MCP connections
static SHARED_MCP_HTTP_CLIENT: OnceLock<std::sync::Arc<reqwest::Client>> = OnceLock::new();

/// Stored proxy configuration for building derived clients
static STORED_PROXY_CONFIG: OnceLock<Option<McpProxyConfig>> = OnceLock::new();

/// Initialize the global MCP HTTP client with optional proxy configuration.
///
/// This should be called once during server startup. If called multiple times,
/// subsequent calls will be ignored with a warning.
///
/// # Arguments
/// * `proxy_config` - Optional proxy configuration from RouterConfig
///
/// # Returns
/// * `Ok(true)` - Successfully initialized
/// * `Ok(false)` - Already initialized (ignored with warning)
/// * `Err(String)` - Failed to build HTTP client
pub fn init(proxy_config: Option<McpProxyConfig>) -> Result<bool, String> {
    // Check if already initialized
    if SHARED_MCP_HTTP_CLIENT.get().is_some() {
        tracing::warn!(
            "MCP HTTP client already initialized, ignoring new proxy config"
        );
        return Ok(false);
    }

    let mut builder = reqwest::Client::builder();

    // Store proxy config for later use in build_with_headers
    let proxy_cfg_ref = if let Some(ref cfg) = proxy_config {
        builder = apply_proxy_config(builder, cfg)?;

        let no_proxy_entry_count = cfg
            .no_proxy
            .as_ref()
            .map(|raw| raw.split(',').filter(|s| !s.trim().is_empty()).count())
            .unwrap_or(0);

        tracing::info!(
            http_proxy = cfg.http_proxy.as_deref(),
            https_proxy = cfg.https_proxy.as_deref(),
            no_proxy_entries = no_proxy_entry_count,
            "Initialized global MCP HTTP client with proxy configuration"
        );
        Some(cfg)
    } else {
        tracing::info!("Initialized global MCP HTTP client without proxy");
        None
    };

    let client = builder
        .build()
        .map_err(|e| format!("Failed to build MCP HTTP client: {}", e))?;

    // Store the proxy config
    let _ = STORED_PROXY_CONFIG.set(proxy_cfg_ref.cloned());

    // Store the client
    SHARED_MCP_HTTP_CLIENT
        .set(std::sync::Arc::new(client))
        .map_err(|_| "MCP HTTP client already initialized".to_string())?;

    Ok(true)
}

/// Get the global shared MCP HTTP client.
///
/// If not yet initialized via `init()`, this will lazily initialize with default
/// configuration (no proxy). In production, `init()` should be called during
/// server startup to configure proxy settings.
pub fn client() -> std::sync::Arc<reqwest::Client> {
    SHARED_MCP_HTTP_CLIENT
        .get_or_init(|| {
            // Lazy initialization with default config (no proxy)
            let client = reqwest::Client::builder()
                .build()
                .expect("Failed to build default MCP HTTP client");

            tracing::debug!("Lazily initialized MCP HTTP client with default configuration (no proxy)");

            // Store empty proxy config
            let _ = STORED_PROXY_CONFIG.set(None);

            std::sync::Arc::new(client)
        })
        .clone()
}

/// Get the global shared MCP HTTP client if initialized.
///
/// Returns `None` if not yet initialized. Unlike `client()`, this will NOT
/// perform lazy initialization.
pub fn try_client() -> Option<std::sync::Arc<reqwest::Client>> {
    SHARED_MCP_HTTP_CLIENT.get().cloned()
}

/// Build a new HTTP client with custom headers, inheriting global proxy configuration.
///
/// This is used for SSE connections that require authentication headers.
/// The returned client will have the same proxy configuration as the global client,
/// but with additional default headers.
///
/// # Arguments
/// * `headers` - Default headers to include in all requests
///
/// # Returns
/// A new `reqwest::Client` with the specified headers and global proxy config
pub fn build_with_headers(headers: HeaderMap) -> Result<reqwest::Client, String> {
    let mut builder = reqwest::Client::builder().default_headers(headers);

    // Apply the same proxy config as the global client
    if let Some(Some(proxy_cfg)) = STORED_PROXY_CONFIG.get() {
        builder = apply_proxy_config(builder, proxy_cfg)?;
    }

    builder
        .build()
        .map_err(|e| format!("Failed to build MCP client with headers: {}", e))
}

/// Apply proxy configuration to a reqwest::ClientBuilder
fn apply_proxy_config(
    builder: reqwest::ClientBuilder,
    proxy_config: &McpProxyConfig,
) -> Result<reqwest::ClientBuilder, String> {
    let mut builder = builder;

    // Parse no_proxy configuration
    let no_proxy = proxy_config.no_proxy.as_ref().and_then(|raw| {
        let result = reqwest::NoProxy::from_string(raw);
        if result.is_none() {
            tracing::warn!("Invalid MCP no_proxy value '{}'. Ignoring no_proxy.", raw);
        }
        result
    });

    // Apply HTTP proxy
    if let Some(http_proxy) = &proxy_config.http_proxy {
        let proxy = build_proxy(http_proxy, "http", no_proxy.as_ref())?;
        builder = builder.proxy(proxy);
    }

    // Apply HTTPS proxy
    if let Some(https_proxy) = &proxy_config.https_proxy {
        let proxy = build_proxy(https_proxy, "https", no_proxy.as_ref())?;
        builder = builder.proxy(proxy);
    }

    Ok(builder)
}

/// Helper function to build a proxy with optional no_proxy configuration
fn build_proxy(
    proxy_url: &str,
    proxy_type: &str,
    no_proxy: Option<&reqwest::NoProxy>,
) -> Result<reqwest::Proxy, String> {
    let proxy = match proxy_type {
        "http" => reqwest::Proxy::http(proxy_url),
        "https" => reqwest::Proxy::https(proxy_url),
        _ => unreachable!("Invalid proxy type"),
    }
    .map_err(|e| {
        format!(
            "invalid {}_proxy '{}': {}",
            proxy_type, proxy_url, e
        )
    })?;

    let proxy = if let Some(no_proxy_cfg) = no_proxy {
        proxy.no_proxy(Some(no_proxy_cfg.clone()))
    } else {
        proxy
    };

    Ok(proxy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_without_proxy() {
        // Note: This test may fail if run after other tests that initialize the client
        // In a real scenario, we'd need a way to reset the static for testing
        let result = init(None);
        assert!(result.is_ok());
    }

    #[test]
    fn apply_proxy_config_accepts_http_proxy() {
        let cfg = McpProxyConfig {
            http_proxy: Some("http://127.0.0.1:18080".into()),
            https_proxy: None,
            no_proxy: None,
        };
        let builder = reqwest::Client::builder();
        let result = apply_proxy_config(builder, &cfg);
        assert!(result.is_ok());
    }

    #[test]
    fn apply_proxy_config_accepts_https_proxy() {
        let cfg = McpProxyConfig {
            http_proxy: None,
            https_proxy: Some("https://127.0.0.1:18443".into()),
            no_proxy: None,
        };
        let builder = reqwest::Client::builder();
        let result = apply_proxy_config(builder, &cfg);
        assert!(result.is_ok());
    }

    #[test]
    fn apply_proxy_config_supports_no_proxy_list() {
        let cfg = McpProxyConfig {
            http_proxy: Some("http://127.0.0.1:18080".into()),
            https_proxy: Some("https://127.0.0.1:18443".into()),
            no_proxy: Some("localhost,example.com".into()),
        };
        let builder = reqwest::Client::builder();
        let result = apply_proxy_config(builder, &cfg);
        assert!(result.is_ok());
    }

    #[test]
    fn apply_proxy_config_rejects_invalid_http_proxy() {
        let cfg = McpProxyConfig {
            http_proxy: Some("http://[::1".into()),
            https_proxy: None,
            no_proxy: None,
        };
        let builder = reqwest::Client::builder();
        let result = apply_proxy_config(builder, &cfg);
        assert!(result.is_err());
        if let Err(msg) = result {
            assert!(msg.contains("invalid http_proxy"));
        }
    }
}
