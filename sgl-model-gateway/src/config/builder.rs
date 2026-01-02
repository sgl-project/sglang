use super::{
    CircuitBreakerConfig, ConfigError, ConfigResult, DiscoveryConfig, HealthCheckConfig,
    HistoryBackend, MetricsConfig, OracleConfig, PolicyConfig, PostgresConfig, RetryConfig,
    RouterConfig, RoutingMode, TokenizerCacheConfig, TraceConfig,
};
use crate::{core::ConnectionMode, mcp::McpConfig};

/// Builder for RouterConfig that wraps the config itself
/// This eliminates field duplication and stays in sync automatically
#[derive(Debug, Clone, Default)]
pub struct RouterConfigBuilder {
    config: RouterConfig,
    // Temporary fields for certificate paths (read during build)
    client_cert_path: Option<String>,
    client_key_path: Option<String>,
    ca_cert_paths: Vec<String>,
    mcp_config_path: Option<String>,
}

impl RouterConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Takes ownership
    pub fn from_config(config: RouterConfig) -> Self {
        Self {
            config,
            client_cert_path: None,
            client_key_path: None,
            ca_cert_paths: Vec::new(),
            mcp_config_path: None,
        }
    }

    pub fn from_config_ref(config: &RouterConfig) -> Self {
        Self::from_config(config.clone())
    }

    // ==================== Routing Mode ====================

    pub fn regular_mode(mut self, worker_urls: Vec<String>) -> Self {
        self.config.mode = RoutingMode::Regular { worker_urls };
        self
    }

    pub fn prefill_decode_mode(
        mut self,
        prefill_urls: Vec<(String, Option<u16>)>,
        decode_urls: Vec<String>,
    ) -> Self {
        self.config.mode = RoutingMode::PrefillDecode {
            prefill_urls,
            decode_urls,
            prefill_policy: None,
            decode_policy: None,
        };
        self
    }

    /// With separate policies
    pub fn prefill_decode_mode_with_policies(
        mut self,
        prefill_urls: Vec<(String, Option<u16>)>,
        decode_urls: Vec<String>,
        prefill_policy: Option<PolicyConfig>,
        decode_policy: Option<PolicyConfig>,
    ) -> Self {
        self.config.mode = RoutingMode::PrefillDecode {
            prefill_urls,
            decode_urls,
            prefill_policy,
            decode_policy,
        };
        self
    }

    pub fn openai_mode(mut self, worker_urls: Vec<String>) -> Self {
        self.config.mode = RoutingMode::OpenAI { worker_urls };
        self
    }

    pub fn mode(mut self, mode: RoutingMode) -> Self {
        self.config.mode = mode;
        self
    }

    // ==================== Policy ====================

    pub fn policy(mut self, policy: PolicyConfig) -> Self {
        self.config.policy = policy;
        self
    }

    pub fn random_policy(mut self) -> Self {
        self.config.policy = PolicyConfig::Random;
        self
    }

    pub fn round_robin_policy(mut self) -> Self {
        self.config.policy = PolicyConfig::RoundRobin;
        self
    }

    pub fn cache_aware_policy(
        mut self,
        cache_threshold: f32,
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
        eviction_interval_secs: u64,
        max_tree_size: usize,
    ) -> Self {
        self.config.policy = PolicyConfig::CacheAware {
            cache_threshold,
            balance_abs_threshold,
            balance_rel_threshold,
            eviction_interval_secs,
            max_tree_size,
        };
        self
    }

    pub fn power_of_two_policy(mut self, load_check_interval_secs: u64) -> Self {
        self.config.policy = PolicyConfig::PowerOfTwo {
            load_check_interval_secs,
        };
        self
    }

    // ==================== Connection ====================

    pub fn connection_mode(mut self, mode: ConnectionMode) -> Self {
        self.config.connection_mode = mode;
        self
    }

    pub fn http_connection(mut self) -> Self {
        self.config.connection_mode = ConnectionMode::Http;
        self
    }

    pub fn grpc_connection(mut self, port: Option<u16>) -> Self {
        self.config.connection_mode = ConnectionMode::Grpc { port };
        self
    }

    pub fn grpc_connection_default(mut self) -> Self {
        self.config.connection_mode = ConnectionMode::Grpc { port: None };
        self
    }

    pub fn host<S: Into<String>>(mut self, host: S) -> Self {
        self.config.host = host.into();
        self
    }

    pub fn port(mut self, port: u16) -> Self {
        self.config.port = port;
        self
    }

    // ==================== Request ====================

    pub fn max_payload_size(mut self, size: usize) -> Self {
        self.config.max_payload_size = size;
        self
    }

    pub fn request_timeout_secs(mut self, timeout: u64) -> Self {
        self.config.request_timeout_secs = timeout;
        self
    }

    pub fn worker_startup_timeout_secs(mut self, timeout: u64) -> Self {
        self.config.worker_startup_timeout_secs = timeout;
        self
    }

    pub fn worker_startup_check_interval_secs(mut self, interval: u64) -> Self {
        self.config.worker_startup_check_interval_secs = interval;
        self
    }

    // ==================== Rate Limiting ====================

    pub fn max_concurrent_requests(mut self, max: i32) -> Self {
        self.config.max_concurrent_requests = max;
        self
    }

    pub fn disable_rate_limiting(mut self) -> Self {
        self.config.max_concurrent_requests = -1;
        self
    }

    pub fn queue_size(mut self, size: usize) -> Self {
        self.config.queue_size = size;
        self
    }

    pub fn queue_timeout_secs(mut self, timeout: u64) -> Self {
        self.config.queue_timeout_secs = timeout;
        self
    }

    pub fn rate_limit_tokens_per_second(mut self, tokens: i32) -> Self {
        self.config.rate_limit_tokens_per_second = Some(tokens);
        self
    }

    // ==================== Security & CORS ====================

    pub fn api_key<S: Into<String>>(mut self, key: S) -> Self {
        self.config.api_key = Some(key.into());
        self
    }

    pub fn cors_allowed_origins(mut self, origins: Vec<String>) -> Self {
        self.config.cors_allowed_origins = origins;
        self
    }

    pub fn add_cors_origin<S: Into<String>>(mut self, origin: S) -> Self {
        self.config.cors_allowed_origins.push(origin.into());
        self
    }

    // ==================== Retry ====================

    pub fn retry_config(mut self, retry: RetryConfig) -> Self {
        self.config.retry = retry;
        self
    }

    pub fn disable_retries(mut self) -> Self {
        self.config.disable_retries = true;
        self
    }

    pub fn enable_retries(mut self) -> Self {
        self.config.disable_retries = false;
        self
    }

    // ==================== Circuit Breaker ====================

    pub fn circuit_breaker_config(mut self, circuit_breaker: CircuitBreakerConfig) -> Self {
        self.config.circuit_breaker = circuit_breaker;
        self
    }

    pub fn disable_circuit_breaker(mut self) -> Self {
        self.config.disable_circuit_breaker = true;
        self
    }

    pub fn enable_circuit_breaker(mut self) -> Self {
        self.config.disable_circuit_breaker = false;
        self
    }

    // ==================== Health Check ====================

    pub fn health_check_config(mut self, health_check: HealthCheckConfig) -> Self {
        self.config.health_check = health_check;
        self
    }

    // ==================== Discovery ====================

    pub fn discovery_config(mut self, discovery: DiscoveryConfig) -> Self {
        self.config.discovery = Some(discovery);
        self
    }

    /// With default settings
    pub fn enable_discovery(mut self) -> Self {
        self.config.discovery = Some(DiscoveryConfig {
            enabled: true,
            ..Default::default()
        });
        self
    }

    // ==================== Metrics ====================

    pub fn metrics_config(mut self, metrics: MetricsConfig) -> Self {
        self.config.metrics = Some(metrics);
        self
    }

    pub fn enable_metrics<S: Into<String>>(mut self, host: S, port: u16) -> Self {
        self.config.metrics = Some(MetricsConfig {
            host: host.into(),
            port,
        });
        self
    }

    // ===================== Otel Trace ====================

    pub fn enable_trace<S: Into<String>>(mut self, endpoint: S) -> Self {
        self.config.trace_config = Some(TraceConfig {
            enable_trace: true,
            otlp_traces_endpoint: endpoint.into(),
        });
        self
    }

    pub fn disable_trace(mut self) -> Self {
        self.config.trace_config = Some(TraceConfig {
            enable_trace: false,
            otlp_traces_endpoint: "".to_string(),
        });
        self
    }

    // ==================== Logging ====================

    pub fn log_dir<S: Into<String>>(mut self, dir: S) -> Self {
        self.config.log_dir = Some(dir.into());
        self
    }

    pub fn log_level<S: Into<String>>(mut self, level: S) -> Self {
        self.config.log_level = Some(level.into());
        self
    }

    pub fn request_id_headers(mut self, headers: Vec<String>) -> Self {
        self.config.request_id_headers = Some(headers);
        self
    }

    // ==================== IGW Mode ====================

    pub fn enable_igw(mut self) -> Self {
        self.config.enable_igw = true;
        self
    }

    /// Use proxy mode
    pub fn disable_igw(mut self) -> Self {
        self.config.enable_igw = false;
        self
    }

    // ==================== WASM ====================

    pub fn enable_wasm(mut self, enable: bool) -> Self {
        self.config.enable_wasm = enable;
        self
    }

    pub fn model_path<S: Into<String>>(mut self, path: S) -> Self {
        self.config.model_path = Some(path.into());
        self
    }

    /// Overrides model_path tokenizer
    pub fn tokenizer_path<S: Into<String>>(mut self, path: S) -> Self {
        self.config.tokenizer_path = Some(path.into());
        self
    }

    pub fn chat_template<S: Into<String>>(mut self, path: S) -> Self {
        self.config.chat_template = Some(path.into());
        self
    }

    // ==================== History Backend ====================

    pub fn history_backend(mut self, backend: HistoryBackend) -> Self {
        self.config.history_backend = backend;
        self
    }

    pub fn memory_history(mut self) -> Self {
        self.config.history_backend = HistoryBackend::Memory;
        self
    }

    pub fn no_history(mut self) -> Self {
        self.config.history_backend = HistoryBackend::None;
        self
    }

    pub fn oracle_history(mut self, oracle_config: OracleConfig) -> Self {
        self.config.history_backend = HistoryBackend::Oracle;
        self.config.oracle = Some(oracle_config);
        self
    }

    // ==================== Parsers ====================

    pub fn reasoning_parser<S: Into<String>>(mut self, parser: S) -> Self {
        self.config.reasoning_parser = Some(parser.into());
        self
    }

    pub fn tool_call_parser<S: Into<String>>(mut self, parser: S) -> Self {
        self.config.tool_call_parser = Some(parser.into());
        self
    }

    // ==================== Tokenizer Cache ====================

    pub fn tokenizer_cache(mut self, cache: TokenizerCacheConfig) -> Self {
        self.config.tokenizer_cache = cache;
        self
    }

    pub fn enable_l0_cache(mut self, max_entries: usize) -> Self {
        self.config.tokenizer_cache.enable_l0 = true;
        self.config.tokenizer_cache.l0_max_entries = max_entries;
        self
    }

    pub fn enable_l1_cache(mut self, max_memory: usize) -> Self {
        self.config.tokenizer_cache.enable_l1 = true;
        self.config.tokenizer_cache.l1_max_memory = max_memory;
        self
    }

    // ==================== Data Parallelism ====================

    pub fn enable_dp_aware(mut self) -> Self {
        self.config.dp_aware = true;
        self
    }

    pub fn disable_dp_aware(mut self) -> Self {
        self.config.dp_aware = false;
        self
    }

    // ==================== Boolean Setters ====================
    // Accept bool parameters to conditionally set flags without if statements

    pub fn dp_aware(mut self, enable: bool) -> Self {
        self.config.dp_aware = enable;
        self
    }

    /// Inverse of disable_retries field
    pub fn retries(mut self, enable: bool) -> Self {
        self.config.disable_retries = !enable;
        self
    }

    /// Inverse of disable_circuit_breaker field
    pub fn circuit_breaker(mut self, enable: bool) -> Self {
        self.config.disable_circuit_breaker = !enable;
        self
    }

    pub fn igw(mut self, enable: bool) -> Self {
        self.config.enable_igw = enable;
        self
    }

    // ==================== Option Setters ====================
    // Accept Option<T> and only set if Some

    pub fn maybe_api_key(mut self, key: Option<impl Into<String>>) -> Self {
        if let Some(k) = key {
            self.config.api_key = Some(k.into());
        }
        self
    }

    pub fn maybe_discovery(mut self, discovery: Option<DiscoveryConfig>) -> Self {
        self.config.discovery = discovery;
        self
    }

    pub fn maybe_metrics(mut self, metrics: Option<MetricsConfig>) -> Self {
        self.config.metrics = metrics;
        self
    }

    pub fn maybe_trace(mut self, trace_config: Option<TraceConfig>) -> Self {
        self.config.trace_config = trace_config;
        self
    }

    pub fn maybe_log_dir(mut self, dir: Option<impl Into<String>>) -> Self {
        self.config.log_dir = dir.map(|d| d.into());
        self
    }

    pub fn maybe_log_level(mut self, level: Option<impl Into<String>>) -> Self {
        self.config.log_level = level.map(|l| l.into());
        self
    }

    pub fn maybe_request_id_headers(mut self, headers: Option<Vec<String>>) -> Self {
        self.config.request_id_headers = headers;
        self
    }

    pub fn maybe_rate_limit_tokens_per_second(mut self, tokens: Option<i32>) -> Self {
        self.config.rate_limit_tokens_per_second = tokens;
        self
    }

    pub fn maybe_model_path(mut self, path: Option<impl Into<String>>) -> Self {
        self.config.model_path = path.map(|p| p.into());
        self
    }

    pub fn maybe_tokenizer_path(mut self, path: Option<impl Into<String>>) -> Self {
        self.config.tokenizer_path = path.map(|p| p.into());
        self
    }

    pub fn maybe_chat_template(mut self, template: Option<impl Into<String>>) -> Self {
        self.config.chat_template = template.map(|t| t.into());
        self
    }

    pub fn maybe_oracle(mut self, oracle: Option<OracleConfig>) -> Self {
        if let Some(cfg) = oracle {
            self.config.history_backend = HistoryBackend::Oracle;
            self.config.oracle = Some(cfg);
        }
        self
    }

    pub fn maybe_postgres(mut self, postgres: Option<PostgresConfig>) -> Self {
        if let Some(cfg) = postgres {
            self.config.history_backend = HistoryBackend::Postgres;
            self.config.postgres = Some(cfg);
        }
        self
    }

    pub fn maybe_reasoning_parser(mut self, parser: Option<impl Into<String>>) -> Self {
        self.config.reasoning_parser = parser.map(|p| p.into());
        self
    }

    pub fn maybe_tool_call_parser(mut self, parser: Option<impl Into<String>>) -> Self {
        self.config.tool_call_parser = parser.map(|p| p.into());
        self
    }

    // ==================== mTLS ====================

    /// Both paths must be provided together. Files read during build()
    pub fn client_cert_and_key<S1: Into<String>, S2: Into<String>>(
        mut self,
        cert_path: S1,
        key_path: S2,
    ) -> Self {
        self.client_cert_path = Some(cert_path.into());
        self.client_key_path = Some(key_path.into());
        self
    }

    /// Files read during build()
    pub fn maybe_client_cert_and_key(
        mut self,
        cert_path: Option<impl Into<String>>,
        key_path: Option<impl Into<String>>,
    ) -> Self {
        self.client_cert_path = cert_path.map(|p| p.into());
        self.client_key_path = key_path.map(|p| p.into());
        self
    }

    /// File read during build()
    pub fn add_ca_certificate<S: Into<String>>(mut self, ca_cert_path: S) -> Self {
        self.ca_cert_paths.push(ca_cert_path.into());
        self
    }

    /// Files read during build()
    pub fn add_ca_certificates<S: Into<String>>(mut self, ca_cert_paths: Vec<S>) -> Self {
        self.ca_cert_paths
            .extend(ca_cert_paths.into_iter().map(|p| p.into()));
        self
    }

    // ==================== MCP ====================

    /// Config file loaded during build()
    pub fn mcp_config_path<S: Into<String>>(mut self, path: S) -> Self {
        self.mcp_config_path = Some(path.into());
        self
    }

    /// Config file loaded during build()
    pub fn maybe_mcp_config_path(mut self, path: Option<impl Into<String>>) -> Self {
        self.mcp_config_path = path.map(|p| p.into());
        self
    }

    // ==================== Build ====================

    pub fn build(self) -> ConfigResult<RouterConfig> {
        self.build_with_validation(true)
    }

    pub fn build_unchecked(self) -> RouterConfig {
        self.into()
    }

    pub fn build_with_validation(mut self, validate: bool) -> ConfigResult<RouterConfig> {
        // Read mTLS certificates from paths if provided
        self = self.read_mtls_certificates()?;

        // Read MCP config from path if provided
        self = self.read_mcp_config()?;

        let config: RouterConfig = self.into();
        if validate {
            config.validate()?;
        }
        Ok(config)
    }

    /// Internal method to read mTLS certificates from paths
    fn read_mtls_certificates(mut self) -> ConfigResult<Self> {
        // Read client certificate and key
        match (&self.client_cert_path, &self.client_key_path) {
            (Some(cert_path), Some(key_path)) => {
                let cert = std::fs::read(cert_path).map_err(|e| ConfigError::ValidationFailed {
                    reason: format!(
                        "Failed to read client certificate from {}: {}",
                        cert_path, e
                    ),
                })?;
                let key = std::fs::read(key_path).map_err(|e| ConfigError::ValidationFailed {
                    reason: format!("Failed to read client key from {}: {}", key_path, e),
                })?;

                // Combine cert and key into single PEM for reqwest::Identity
                // When using rustls, certificate must come first, then key
                // Ensure proper PEM formatting with newlines
                let mut combined = cert;
                if !combined.ends_with(b"\n") {
                    combined.push(b'\n');
                }
                combined.extend_from_slice(&key);
                if !combined.ends_with(b"\n") {
                    combined.push(b'\n');
                }

                self.config.client_identity = Some(combined);
            }
            (None, None) => {
                // No client cert configured, that's fine
            }
            _ => {
                return Err(ConfigError::ValidationFailed {
                    reason:
                        "Both --client-cert-path and --client-key-path must be specified together"
                            .to_string(),
                });
            }
        }

        // Read CA certificates
        for path in &self.ca_cert_paths {
            let cert = std::fs::read(path).map_err(|e| ConfigError::ValidationFailed {
                reason: format!("Failed to read CA certificate from {}: {}", path, e),
            })?;
            self.config.ca_certificates.push(cert);
        }

        Ok(self)
    }

    /// Internal method to read MCP config from path
    fn read_mcp_config(mut self) -> ConfigResult<Self> {
        if let Some(mcp_config_path) = &self.mcp_config_path {
            let contents = std::fs::read_to_string(mcp_config_path).map_err(|e| {
                ConfigError::ValidationFailed {
                    reason: format!("Failed to read MCP config from {}: {}", mcp_config_path, e),
                }
            })?;
            let mcp_config: McpConfig =
                serde_yaml::from_str(&contents).map_err(|e| ConfigError::ValidationFailed {
                    reason: format!("Failed to parse MCP config from {}: {}", mcp_config_path, e),
                })?;
            self.config.mcp_config = Some(mcp_config);
        }

        Ok(self)
    }
}

impl From<RouterConfigBuilder> for RouterConfig {
    fn from(builder: RouterConfigBuilder) -> Self {
        builder.config
    }
}

impl RouterConfig {
    /// Create a builder for RouterConfig
    pub fn builder() -> RouterConfigBuilder {
        RouterConfigBuilder::new()
    }

    /// Create a builder from this configuration
    pub fn to_builder(&self) -> RouterConfigBuilder {
        RouterConfigBuilder::from_config_ref(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that .to_builder() round-trip conversion works correctly
    #[test]
    fn test_builder_from_existing_config() {
        let original = RouterConfigBuilder::new()
            .regular_mode(vec!["http://worker1:8000".to_string()])
            .port(3000)
            .build()
            .unwrap();

        let modified = original
            .to_builder()
            .port(4000)
            .enable_metrics("0.0.0.0", 29000)
            .enable_trace("localhost:4317")
            .build()
            .unwrap();

        assert_eq!(modified.port, 4000);
        assert!(modified.metrics.is_some());
        assert!(modified.trace_config.is_some());
    }

    /// Test complex routing mode helper method
    #[test]
    fn test_builder_prefill_decode_mode() {
        let config = RouterConfigBuilder::new()
            .prefill_decode_mode(
                vec![("http://prefill:8000".to_string(), Some(8001))],
                vec!["http://decode:8000".to_string()],
            )
            .power_of_two_policy(60)
            .build()
            .unwrap();

        assert!(config.mode.is_pd_mode());
        assert_eq!(config.mode.worker_count(), 2);
    }

    /// Test complex policy helper method with multiple parameters
    #[test]
    fn test_builder_cache_aware_policy() {
        let config = RouterConfigBuilder::new()
            .regular_mode(vec!["http://worker1:8000".to_string()])
            .cache_aware_policy(0.8, 10, 1.5, 300, 1000)
            .build()
            .unwrap();

        match config.policy {
            PolicyConfig::CacheAware {
                cache_threshold, ..
            } => {
                assert!((cache_threshold - 0.8).abs() < 0.0001);
            }
            _ => panic!("Expected CacheAware policy"),
        }
    }
}
