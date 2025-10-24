use super::{
    CircuitBreakerConfig, ConfigResult, DiscoveryConfig, HealthCheckConfig, HistoryBackend,
    MetricsConfig, OracleConfig, PolicyConfig, RetryConfig, RouterConfig, RoutingMode,
    TokenizerCacheConfig,
};
use crate::core::ConnectionMode;

/// Builder for RouterConfig that wraps the config itself
/// This eliminates field duplication and stays in sync automatically
#[derive(Debug, Clone, Default)]
pub struct RouterConfigBuilder {
    config: RouterConfig,
}

impl RouterConfigBuilder {
    /// Create a new builder with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a builder from an existing configuration (takes ownership)
    pub fn from_config(config: RouterConfig) -> Self {
        Self { config }
    }

    /// Create a builder from a reference to an existing configuration
    pub fn from_config_ref(config: &RouterConfig) -> Self {
        Self::from_config(config.clone())
    }

    // ==================== Routing Mode Setters ====================

    /// Set regular routing mode with worker URLs
    pub fn regular_mode(mut self, worker_urls: Vec<String>) -> Self {
        self.config.mode = RoutingMode::Regular { worker_urls };
        self
    }

    /// Set prefill-decode routing mode
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

    /// Set prefill-decode mode with separate policies
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

    /// Set OpenAI routing mode
    pub fn openai_mode(mut self, worker_urls: Vec<String>) -> Self {
        self.config.mode = RoutingMode::OpenAI { worker_urls };
        self
    }

    /// Set the routing mode directly
    pub fn mode(mut self, mode: RoutingMode) -> Self {
        self.config.mode = mode;
        self
    }

    // ==================== Policy Setters ====================

    /// Set the routing policy
    pub fn policy(mut self, policy: PolicyConfig) -> Self {
        self.config.policy = policy;
        self
    }

    /// Set random policy
    pub fn random_policy(mut self) -> Self {
        self.config.policy = PolicyConfig::Random;
        self
    }

    /// Set round-robin policy
    pub fn round_robin_policy(mut self) -> Self {
        self.config.policy = PolicyConfig::RoundRobin;
        self
    }

    /// Set cache-aware policy with parameters
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

    /// Set power-of-two policy
    pub fn power_of_two_policy(mut self, load_check_interval_secs: u64) -> Self {
        self.config.policy = PolicyConfig::PowerOfTwo {
            load_check_interval_secs,
        };
        self
    }

    // ==================== Connection Settings ====================

    /// Set connection mode
    pub fn connection_mode(mut self, mode: ConnectionMode) -> Self {
        self.config.connection_mode = mode;
        self
    }

    /// Set HTTP connection mode
    pub fn http_connection(mut self) -> Self {
        self.config.connection_mode = ConnectionMode::Http;
        self
    }

    /// Set gRPC connection mode with optional port
    pub fn grpc_connection(mut self, port: Option<u16>) -> Self {
        self.config.connection_mode = ConnectionMode::Grpc { port };
        self
    }

    /// Set gRPC connection mode without specifying a port
    pub fn grpc_connection_default(mut self) -> Self {
        self.config.connection_mode = ConnectionMode::Grpc { port: None };
        self
    }

    /// Set host address
    pub fn host<S: Into<String>>(mut self, host: S) -> Self {
        self.config.host = host.into();
        self
    }

    /// Set port number
    pub fn port(mut self, port: u16) -> Self {
        self.config.port = port;
        self
    }

    // ==================== Request Settings ====================

    /// Set maximum payload size in bytes
    pub fn max_payload_size(mut self, size: usize) -> Self {
        self.config.max_payload_size = size;
        self
    }

    /// Set request timeout in seconds
    pub fn request_timeout_secs(mut self, timeout: u64) -> Self {
        self.config.request_timeout_secs = timeout;
        self
    }

    /// Set worker startup timeout in seconds
    pub fn worker_startup_timeout_secs(mut self, timeout: u64) -> Self {
        self.config.worker_startup_timeout_secs = timeout;
        self
    }

    /// Set worker startup check interval in seconds
    pub fn worker_startup_check_interval_secs(mut self, interval: u64) -> Self {
        self.config.worker_startup_check_interval_secs = interval;
        self
    }

    // ==================== Rate Limiting ====================

    /// Set maximum concurrent requests
    pub fn max_concurrent_requests(mut self, max: i32) -> Self {
        self.config.max_concurrent_requests = max;
        self
    }

    /// Disable rate limiting
    pub fn disable_rate_limiting(mut self) -> Self {
        self.config.max_concurrent_requests = -1;
        self
    }

    /// Set queue size for pending requests
    pub fn queue_size(mut self, size: usize) -> Self {
        self.config.queue_size = size;
        self
    }

    /// Set queue timeout in seconds
    pub fn queue_timeout_secs(mut self, timeout: u64) -> Self {
        self.config.queue_timeout_secs = timeout;
        self
    }

    /// Set rate limit tokens per second
    pub fn rate_limit_tokens_per_second(mut self, tokens: i32) -> Self {
        self.config.rate_limit_tokens_per_second = Some(tokens);
        self
    }

    // ==================== Security & CORS ====================

    /// Set API key for worker authorization
    pub fn api_key<S: Into<String>>(mut self, key: S) -> Self {
        self.config.api_key = Some(key.into());
        self
    }

    /// Set CORS allowed origins
    pub fn cors_allowed_origins(mut self, origins: Vec<String>) -> Self {
        self.config.cors_allowed_origins = origins;
        self
    }

    /// Add a single CORS origin
    pub fn add_cors_origin<S: Into<String>>(mut self, origin: S) -> Self {
        self.config.cors_allowed_origins.push(origin.into());
        self
    }

    // ==================== Retry Configuration ====================

    /// Set retry configuration
    pub fn retry_config(mut self, retry: RetryConfig) -> Self {
        self.config.retry = retry;
        self
    }

    /// Disable retries
    pub fn disable_retries(mut self) -> Self {
        self.config.disable_retries = true;
        self
    }

    /// Enable retries
    pub fn enable_retries(mut self) -> Self {
        self.config.disable_retries = false;
        self
    }

    // ==================== Circuit Breaker Configuration ====================

    /// Set circuit breaker configuration
    pub fn circuit_breaker_config(mut self, circuit_breaker: CircuitBreakerConfig) -> Self {
        self.config.circuit_breaker = circuit_breaker;
        self
    }

    /// Disable circuit breaker
    pub fn disable_circuit_breaker(mut self) -> Self {
        self.config.disable_circuit_breaker = true;
        self
    }

    /// Enable circuit breaker
    pub fn enable_circuit_breaker(mut self) -> Self {
        self.config.disable_circuit_breaker = false;
        self
    }

    // ==================== Health Check Configuration ====================

    /// Set health check configuration
    pub fn health_check_config(mut self, health_check: HealthCheckConfig) -> Self {
        self.config.health_check = health_check;
        self
    }

    // ==================== Discovery Configuration ====================

    /// Set service discovery configuration
    pub fn discovery_config(mut self, discovery: DiscoveryConfig) -> Self {
        self.config.discovery = Some(discovery);
        self
    }

    /// Enable service discovery with default settings
    pub fn enable_discovery(mut self) -> Self {
        self.config.discovery = Some(DiscoveryConfig {
            enabled: true,
            ..Default::default()
        });
        self
    }

    // ==================== Metrics Configuration ====================

    /// Set metrics configuration
    pub fn metrics_config(mut self, metrics: MetricsConfig) -> Self {
        self.config.metrics = Some(metrics);
        self
    }

    /// Enable metrics with host and port
    pub fn enable_metrics<S: Into<String>>(mut self, host: S, port: u16) -> Self {
        self.config.metrics = Some(MetricsConfig {
            host: host.into(),
            port,
        });
        self
    }

    // ==================== Logging Configuration ====================

    /// Set log directory
    pub fn log_dir<S: Into<String>>(mut self, dir: S) -> Self {
        self.config.log_dir = Some(dir.into());
        self
    }

    /// Set log level
    pub fn log_level<S: Into<String>>(mut self, level: S) -> Self {
        self.config.log_level = Some(level.into());
        self
    }

    /// Set custom request ID headers
    pub fn request_id_headers(mut self, headers: Vec<String>) -> Self {
        self.config.request_id_headers = Some(headers);
        self
    }

    // ==================== IGW Mode Configuration ====================

    /// Enable Inference Gateway mode
    pub fn enable_igw(mut self) -> Self {
        self.config.enable_igw = true;
        self
    }

    /// Disable Inference Gateway mode (use proxy mode)
    pub fn disable_igw(mut self) -> Self {
        self.config.enable_igw = false;
        self
    }

    /// Set model path for tokenizer
    pub fn model_path<S: Into<String>>(mut self, path: S) -> Self {
        self.config.model_path = Some(path.into());
        self
    }

    /// Set tokenizer path (overrides model_path tokenizer)
    pub fn tokenizer_path<S: Into<String>>(mut self, path: S) -> Self {
        self.config.tokenizer_path = Some(path.into());
        self
    }

    /// Set chat template path
    pub fn chat_template<S: Into<String>>(mut self, path: S) -> Self {
        self.config.chat_template = Some(path.into());
        self
    }

    // ==================== History Backend Configuration ====================

    /// Set history backend
    pub fn history_backend(mut self, backend: HistoryBackend) -> Self {
        self.config.history_backend = backend;
        self
    }

    /// Use memory history backend
    pub fn memory_history(mut self) -> Self {
        self.config.history_backend = HistoryBackend::Memory;
        self
    }

    /// Disable history storage
    pub fn no_history(mut self) -> Self {
        self.config.history_backend = HistoryBackend::None;
        self
    }

    /// Use Oracle history backend
    pub fn oracle_history(mut self, oracle_config: OracleConfig) -> Self {
        self.config.history_backend = HistoryBackend::Oracle;
        self.config.oracle = Some(oracle_config);
        self
    }

    // ==================== Parsers Configuration ====================

    /// Set reasoning parser
    pub fn reasoning_parser<S: Into<String>>(mut self, parser: S) -> Self {
        self.config.reasoning_parser = Some(parser.into());
        self
    }

    /// Set tool call parser
    pub fn tool_call_parser<S: Into<String>>(mut self, parser: S) -> Self {
        self.config.tool_call_parser = Some(parser.into());
        self
    }

    // ==================== Tokenizer Cache Configuration ====================

    /// Set tokenizer cache configuration
    pub fn tokenizer_cache(mut self, cache: TokenizerCacheConfig) -> Self {
        self.config.tokenizer_cache = cache;
        self
    }

    /// Enable L0 cache with entry limit
    pub fn enable_l0_cache(mut self, max_entries: usize) -> Self {
        self.config.tokenizer_cache.enable_l0 = true;
        self.config.tokenizer_cache.l0_max_entries = max_entries;
        self
    }

    /// Enable L1 cache with memory limit
    pub fn enable_l1_cache(mut self, max_memory: usize) -> Self {
        self.config.tokenizer_cache.enable_l1 = true;
        self.config.tokenizer_cache.l1_max_memory = max_memory;
        self
    }

    // ==================== Data Parallelism ====================

    /// Enable data parallelism aware scheduling
    pub fn enable_dp_aware(mut self) -> Self {
        self.config.dp_aware = true;
        self
    }

    /// Disable data parallelism aware scheduling
    pub fn disable_dp_aware(mut self) -> Self {
        self.config.dp_aware = false;
        self
    }

    // ==================== Conditional Boolean Setters ====================
    // These methods accept bool parameters to conditionally set flags,
    // eliminating the need for if statements in calling code

    /// Set dp_aware flag conditionally
    pub fn dp_aware(mut self, enable: bool) -> Self {
        self.config.dp_aware = enable;
        self
    }

    /// Enable or disable retries (inverse of disable_retries field)
    pub fn retries(mut self, enable: bool) -> Self {
        self.config.disable_retries = !enable;
        self
    }

    /// Enable or disable circuit breaker (inverse of disable_circuit_breaker field)
    pub fn circuit_breaker(mut self, enable: bool) -> Self {
        self.config.disable_circuit_breaker = !enable;
        self
    }

    /// Set enable_igw flag conditionally
    pub fn igw(mut self, enable: bool) -> Self {
        self.config.enable_igw = enable;
        self
    }

    // ==================== Option-Aware Setters ====================
    // These methods accept Option<T> and only set if Some, making it easier
    // to conditionally set values without if-let chains

    /// Set API key if Some
    pub fn maybe_api_key(mut self, key: Option<impl Into<String>>) -> Self {
        if let Some(k) = key {
            self.config.api_key = Some(k.into());
        }
        self
    }

    /// Set discovery config if Some
    pub fn maybe_discovery(mut self, discovery: Option<DiscoveryConfig>) -> Self {
        self.config.discovery = discovery;
        self
    }

    /// Set metrics config if Some
    pub fn maybe_metrics(mut self, metrics: Option<MetricsConfig>) -> Self {
        self.config.metrics = metrics;
        self
    }

    /// Set log directory if Some
    pub fn maybe_log_dir(mut self, dir: Option<impl Into<String>>) -> Self {
        self.config.log_dir = dir.map(|d| d.into());
        self
    }

    /// Set log level if Some
    pub fn maybe_log_level(mut self, level: Option<impl Into<String>>) -> Self {
        self.config.log_level = level.map(|l| l.into());
        self
    }

    /// Set request ID headers if Some
    pub fn maybe_request_id_headers(mut self, headers: Option<Vec<String>>) -> Self {
        self.config.request_id_headers = headers;
        self
    }

    /// Set rate limit tokens per second if Some
    pub fn maybe_rate_limit_tokens_per_second(mut self, tokens: Option<i32>) -> Self {
        self.config.rate_limit_tokens_per_second = tokens;
        self
    }

    /// Set model path if Some
    pub fn maybe_model_path(mut self, path: Option<impl Into<String>>) -> Self {
        self.config.model_path = path.map(|p| p.into());
        self
    }

    /// Set tokenizer path if Some
    pub fn maybe_tokenizer_path(mut self, path: Option<impl Into<String>>) -> Self {
        self.config.tokenizer_path = path.map(|p| p.into());
        self
    }

    /// Set chat template if Some
    pub fn maybe_chat_template(mut self, template: Option<impl Into<String>>) -> Self {
        self.config.chat_template = template.map(|t| t.into());
        self
    }

    /// Set oracle config if Some
    pub fn maybe_oracle(mut self, oracle: Option<OracleConfig>) -> Self {
        if let Some(cfg) = oracle {
            self.config.history_backend = HistoryBackend::Oracle;
            self.config.oracle = Some(cfg);
        }
        self
    }

    /// Set reasoning parser if Some
    pub fn maybe_reasoning_parser(mut self, parser: Option<impl Into<String>>) -> Self {
        self.config.reasoning_parser = parser.map(|p| p.into());
        self
    }

    /// Set tool call parser if Some
    pub fn maybe_tool_call_parser(mut self, parser: Option<impl Into<String>>) -> Self {
        self.config.tool_call_parser = parser.map(|p| p.into());
        self
    }

    // ==================== Builder Methods ====================

    /// Build the RouterConfig, validating if requested
    pub fn build(self) -> ConfigResult<RouterConfig> {
        self.build_with_validation(true)
    }

    /// Build the RouterConfig without validation
    pub fn build_unchecked(self) -> RouterConfig {
        self.into()
    }

    /// Build with optional validation
    pub fn build_with_validation(self, validate: bool) -> ConfigResult<RouterConfig> {
        let config: RouterConfig = self.into();
        if validate {
            config.validate()?;
        }
        Ok(config)
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
            .build()
            .unwrap();

        assert_eq!(modified.port, 4000);
        assert!(modified.metrics.is_some());
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
