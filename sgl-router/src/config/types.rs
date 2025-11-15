use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use url::Url;

use super::ConfigResult;
use crate::core::ConnectionMode;

/// Main router configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    pub mode: RoutingMode,
    #[serde(default)]
    pub connection_mode: ConnectionMode,
    pub policy: PolicyConfig,
    pub host: String,
    pub port: u16,
    pub max_payload_size: usize,
    pub request_timeout_secs: u64,
    pub worker_startup_timeout_secs: u64,
    pub worker_startup_check_interval_secs: u64,
    pub worker_load_check_interval_secs: u64,
    pub dp_aware: bool,
    pub dp_minimum_tokens_scheduler: bool,
    pub api_key: Option<String>,
    pub discovery: Option<DiscoveryConfig>,
    pub metrics: Option<MetricsConfig>,
    pub log_dir: Option<String>,
    pub log_level: Option<String>,
    pub request_id_headers: Option<Vec<String>>,
    /// Set to -1 to disable rate limiting
    pub max_concurrent_requests: i32,
    pub queue_size: usize,
    pub queue_timeout_secs: u64,
    /// If not set, defaults to max_concurrent_requests
    pub rate_limit_tokens_per_second: Option<i32>,
    pub cors_allowed_origins: Vec<String>,
    pub retry: RetryConfig,
    pub circuit_breaker: CircuitBreakerConfig,
    /// When true, overrides retry.max_retries to 1
    #[serde(default)]
    pub disable_retries: bool,
    /// When true, overrides circuit_breaker.failure_threshold to u32::MAX
    #[serde(default)]
    pub disable_circuit_breaker: bool,
    pub health_check: HealthCheckConfig,
    #[serde(default)]
    pub enable_igw: bool,
    /// Can be a HuggingFace model ID or local path
    pub model_path: Option<String>,
    /// Overrides model_path tokenizer if provided
    pub tokenizer_path: Option<String>,
    pub chat_template: Option<String>,
    #[serde(default = "default_history_backend")]
    pub history_backend: HistoryBackend,
    /// Required when history_backend = "oracle"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oracle: Option<OracleConfig>,
    /// Required when history_backend = "postgres"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub postgres: Option<PostgresConfig>,
    /// For reasoning models (e.g., deepseek-r1, qwen3)
    pub reasoning_parser: Option<String>,
    /// For tool-call interactions
    pub tool_call_parser: Option<String>,
    #[serde(default)]
    pub tokenizer_cache: TokenizerCacheConfig,
    /// Combined certificate + key in PEM format, loaded from client_cert_path and client_key_path during config creation
    #[serde(skip)]
    pub client_identity: Option<Vec<u8>>,
    /// PEM format, loaded from ca_cert_paths during config creation
    #[serde(default)]
    pub ca_certificates: Vec<Vec<u8>>,
    /// Loaded from mcp_config_path during config creation
    #[serde(skip)]
    pub mcp_config: Option<crate::mcp::McpConfig>,
}

/// Tokenizer cache configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TokenizerCacheConfig {
    /// Whole-string exact match cache
    #[serde(default = "default_enable_l0")]
    pub enable_l0: bool,
    #[serde(default = "default_l0_max_entries")]
    pub l0_max_entries: usize,
    /// Prefix matching at fixed boundaries
    #[serde(default = "default_enable_l1")]
    pub enable_l1: bool,
    #[serde(default = "default_l1_max_memory")]
    pub l1_max_memory: usize,
}

fn default_enable_l0() -> bool {
    false
}

fn default_l0_max_entries() -> usize {
    10_000
}

fn default_enable_l1() -> bool {
    false
}

fn default_l1_max_memory() -> usize {
    50 * 1024 * 1024 // 50MB
}

impl Default for TokenizerCacheConfig {
    fn default() -> Self {
        Self {
            enable_l0: default_enable_l0(),
            l0_max_entries: default_l0_max_entries(),
            enable_l1: default_enable_l1(),
            l1_max_memory: default_l1_max_memory(),
        }
    }
}

fn default_history_backend() -> HistoryBackend {
    HistoryBackend::Memory
}

/// History backend configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum HistoryBackend {
    Memory,
    None,
    Oracle,
    Postgres,
}

/// Oracle history backend configuration
#[derive(Clone, Serialize, Deserialize, PartialEq)]
pub struct OracleConfig {
    /// ATP wallet or TLS config files directory
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wallet_path: Option<String>,
    /// DSN (e.g. `tcps://host:port/service`)
    pub connect_descriptor: String,
    pub username: String,
    pub password: String,
    #[serde(default = "default_pool_min")]
    pub pool_min: usize,
    #[serde(default = "default_pool_max")]
    pub pool_max: usize,
    #[serde(default = "default_pool_timeout_secs")]
    pub pool_timeout_secs: u64,
}

impl OracleConfig {
    pub fn default_pool_min() -> usize {
        default_pool_min()
    }

    pub fn default_pool_max() -> usize {
        default_pool_max()
    }

    pub fn default_pool_timeout_secs() -> u64 {
        default_pool_timeout_secs()
    }
}

fn default_pool_min() -> usize {
    1
}

fn default_pool_max() -> usize {
    16
}

fn default_pool_timeout_secs() -> u64 {
    30
}

impl std::fmt::Debug for OracleConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OracleConfig")
            .field("wallet_path", &self.wallet_path)
            .field("connect_descriptor", &self.connect_descriptor)
            .field("username", &self.username)
            .field("pool_min", &self.pool_min)
            .field("pool_max", &self.pool_max)
            .field("pool_timeout_secs", &self.pool_timeout_secs)
            .finish()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PostgresConfig {
    // Database connection URL,
    // postgres://[user[:password]@][netloc][:port][/dbname][?param1=value1&...]
    pub db_url: String,
    // Database pool max size
    pub pool_max: usize,
}

impl PostgresConfig {
    pub fn default_pool_max() -> usize {
        16
    }

    pub fn validate(&self) -> Result<(), String> {
        let s = self.db_url.trim();
        if s.is_empty() {
            return Err("is it db-url should be not empty".to_string());
        }

        let url = Url::parse(s).map_err(|e| format!("invalid db_url: {}", e))?;

        let scheme = url.scheme();
        if scheme != "postgres" && scheme != "postgresql" {
            return Err(format!("don't support URL scheme: {}", scheme));
        }

        if url.host().is_none() {
            return Err("db_url must need host".to_string());
        }

        let path = url.path();
        let dbname = path
            .strip_prefix('/')
            .filter(|p| !p.is_empty())
            .map(|s| s.to_string());
        if dbname.is_none() {
            return Err("db_url must need database name".to_string());
        }

        if self.pool_max == 0 {
            return Err("pool_max must be greater 1, default is 16".to_string());
        }

        Ok(())
    }
}

/// Routing mode configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RoutingMode {
    #[serde(rename = "regular")]
    Regular { worker_urls: Vec<String> },
    #[serde(rename = "prefill_decode")]
    PrefillDecode {
        /// With optional bootstrap ports
        prefill_urls: Vec<(String, Option<u16>)>,
        decode_urls: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        prefill_policy: Option<PolicyConfig>,
        #[serde(skip_serializing_if = "Option::is_none")]
        decode_policy: Option<PolicyConfig>,
    },
    #[serde(rename = "openai")]
    OpenAI { worker_urls: Vec<String> },
}

impl RoutingMode {
    pub fn is_pd_mode(&self) -> bool {
        matches!(self, RoutingMode::PrefillDecode { .. })
    }

    pub fn worker_count(&self) -> usize {
        match self {
            RoutingMode::Regular { worker_urls } => worker_urls.len(),
            RoutingMode::PrefillDecode {
                prefill_urls,
                decode_urls,
                ..
            } => prefill_urls.len() + decode_urls.len(),
            RoutingMode::OpenAI { .. } => 1,
        }
    }

    /// Get the effective prefill policy for PD mode
    /// Falls back to the main policy if no specific prefill policy is set
    pub fn get_prefill_policy<'a>(&'a self, main_policy: &'a PolicyConfig) -> &'a PolicyConfig {
        match self {
            RoutingMode::PrefillDecode { prefill_policy, .. } => {
                prefill_policy.as_ref().unwrap_or(main_policy)
            }
            _ => main_policy,
        }
    }

    /// Get the effective decode policy for PD mode
    /// Falls back to the main policy if no specific decode policy is set
    pub fn get_decode_policy<'a>(&'a self, main_policy: &'a PolicyConfig) -> &'a PolicyConfig {
        match self {
            RoutingMode::PrefillDecode { decode_policy, .. } => {
                decode_policy.as_ref().unwrap_or(main_policy)
            }
            _ => main_policy,
        }
    }
}

/// Policy configuration for routing
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum PolicyConfig {
    #[serde(rename = "random")]
    Random,

    #[serde(rename = "round_robin")]
    RoundRobin,

    #[serde(rename = "cache_aware")]
    CacheAware {
        cache_threshold: f32,
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
        eviction_interval_secs: u64,
        max_tree_size: usize,
    },

    #[serde(rename = "power_of_two")]
    PowerOfTwo { load_check_interval_secs: u64 },

    #[serde(rename = "bucket")]
    Bucket {
        /// Absolute load difference threshold for load balancing
        balance_abs_threshold: usize,
        /// Relative load ratio threshold for load balancing
        balance_rel_threshold: f32,
        /// Interval between bucket boundary adjustment cycles (seconds)
        bucket_adjust_interval_secs: usize,
    },
}

impl PolicyConfig {
    pub fn name(&self) -> &'static str {
        match self {
            PolicyConfig::Random => "random",
            PolicyConfig::RoundRobin => "round_robin",
            PolicyConfig::CacheAware { .. } => "cache_aware",
            PolicyConfig::PowerOfTwo { .. } => "power_of_two",
            PolicyConfig::Bucket { .. } => "bucket",
        }
    }
}

/// Service discovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    pub enabled: bool,
    /// None = all namespaces
    pub namespace: Option<String>,
    pub port: u16,
    pub check_interval_secs: u64,
    /// Regular mode
    pub selector: HashMap<String, String>,
    /// PD mode prefill
    pub prefill_selector: HashMap<String, String>,
    /// PD mode decode
    pub decode_selector: HashMap<String, String>,
    pub bootstrap_port_annotation: String,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            namespace: None,
            port: 8000,
            check_interval_secs: 120,
            selector: HashMap::new(),
            prefill_selector: HashMap::new(),
            decode_selector: HashMap::new(),
            bootstrap_port_annotation: "sglang.ai/bootstrap-port".to_string(),
        }
    }
}

/// Retry configuration for request handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub initial_backoff_ms: u64,
    pub max_backoff_ms: u64,
    pub backoff_multiplier: f32,
    /// D' = D * (1 + U[-j, +j]) where j is jitter factor
    #[serde(default = "default_retry_jitter_factor")]
    pub jitter_factor: f32,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 5,
            initial_backoff_ms: 50,
            max_backoff_ms: 30000,
            backoff_multiplier: 1.5,
            jitter_factor: 0.2,
        }
    }
}

fn default_retry_jitter_factor() -> f32 {
    0.2
}

/// Health check configuration for worker monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    pub failure_threshold: u32,
    pub success_threshold: u32,
    pub timeout_secs: u64,
    pub check_interval_secs: u64,
    pub endpoint: String,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 3,
            success_threshold: 2,
            timeout_secs: 5,
            check_interval_secs: 60,
            endpoint: "/health".to_string(),
        }
    }
}

/// Circuit breaker configuration for worker reliability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: u32,
    pub success_threshold: u32,
    pub timeout_duration_secs: u64,
    pub window_duration_secs: u64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 10,
            success_threshold: 3,
            timeout_duration_secs: 60,
            window_duration_secs: 120,
        }
    }
}

/// Metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    pub port: u16,
    pub host: String,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            port: 29000,
            host: "0.0.0.0".to_string(),
        }
    }
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            mode: RoutingMode::Regular {
                worker_urls: vec![],
            },
            policy: PolicyConfig::Random,
            host: "0.0.0.0".to_string(),
            port: 3001,
            max_payload_size: 536_870_912, // 512MB
            request_timeout_secs: 1800,    // 30 minutes
            worker_startup_timeout_secs: 600,
            worker_startup_check_interval_secs: 30,
            worker_load_check_interval_secs: 10,
            dp_aware: false,
            dp_minimum_tokens_scheduler: false,
            api_key: None,
            discovery: None,
            metrics: None,
            log_dir: None,
            log_level: None,
            request_id_headers: None,
            max_concurrent_requests: -1,
            queue_size: 100,
            queue_timeout_secs: 60,
            rate_limit_tokens_per_second: None,
            cors_allowed_origins: vec![],
            retry: RetryConfig::default(),
            circuit_breaker: CircuitBreakerConfig::default(),
            disable_retries: false,
            disable_circuit_breaker: false,
            health_check: HealthCheckConfig::default(),
            enable_igw: false,
            connection_mode: ConnectionMode::Http,
            model_path: None,
            tokenizer_path: None,
            chat_template: None,
            history_backend: default_history_backend(),
            oracle: None,
            postgres: None,
            reasoning_parser: None,
            tool_call_parser: None,
            tokenizer_cache: TokenizerCacheConfig::default(),
            client_identity: None,
            ca_certificates: vec![],
            mcp_config: None,
        }
    }
}

impl RouterConfig {
    /// Create a new configuration with mode and policy
    pub fn new(mode: RoutingMode, policy: PolicyConfig) -> Self {
        Self {
            mode,
            policy,
            ..Default::default()
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> ConfigResult<()> {
        crate::config::validation::ConfigValidator::validate(self)
    }

    /// Get the routing mode type as a string
    pub fn mode_type(&self) -> &'static str {
        match self.mode {
            RoutingMode::Regular { .. } => "regular",
            RoutingMode::PrefillDecode { .. } => "prefill_decode",
            RoutingMode::OpenAI { .. } => "openai",
        }
    }

    /// Check if service discovery is enabled
    pub fn has_service_discovery(&self) -> bool {
        self.discovery.as_ref().is_some_and(|d| d.enabled)
    }

    /// Check if metrics are enabled
    pub fn has_metrics(&self) -> bool {
        self.metrics.is_some()
    }

    /// Compute the effective retry config considering disable flag
    pub fn effective_retry_config(&self) -> RetryConfig {
        let mut cfg = self.retry.clone();
        if self.disable_retries {
            cfg.max_retries = 1;
        }
        cfg
    }

    /// Compute the effective circuit breaker config considering disable flag
    pub fn effective_circuit_breaker_config(&self) -> CircuitBreakerConfig {
        let mut cfg = self.circuit_breaker.clone();
        if self.disable_circuit_breaker {
            cfg.failure_threshold = u32::MAX;
        }
        cfg
    }

    /// Check if running in IGW (Inference Gateway) mode
    pub fn is_igw_mode(&self) -> bool {
        self.enable_igw
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_config_default() {
        let config = RouterConfig::default();

        assert!(
            matches!(config.mode, RoutingMode::Regular { worker_urls } if worker_urls.is_empty())
        );
        assert!(matches!(config.policy, PolicyConfig::Random));
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 3001);
        assert_eq!(config.max_payload_size, 536_870_912);
        assert_eq!(config.request_timeout_secs, 1800);
        assert_eq!(config.worker_startup_timeout_secs, 600);
        assert_eq!(config.worker_startup_check_interval_secs, 30);
        assert!(config.discovery.is_none());
        assert!(config.metrics.is_none());
        assert!(config.log_dir.is_none());
        assert!(config.log_level.is_none());
    }

    #[test]
    fn test_router_config_new() {
        let mode = RoutingMode::Regular {
            worker_urls: vec!["http://worker1".to_string(), "http://worker2".to_string()],
        };
        let policy = PolicyConfig::RoundRobin;

        let config = RouterConfig::new(mode, policy);

        match config.mode {
            RoutingMode::Regular { worker_urls } => {
                assert_eq!(worker_urls.len(), 2);
                assert_eq!(worker_urls[0], "http://worker1");
                assert_eq!(worker_urls[1], "http://worker2");
            }
            _ => panic!("Expected Regular mode"),
        }

        assert!(matches!(config.policy, PolicyConfig::RoundRobin));
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 3001);
    }

    #[test]
    fn test_router_config_serialization() {
        let config = RouterConfig::builder()
            .regular_mode(vec!["http://worker1".to_string()])
            .random_policy()
            .host("0.0.0.0")
            .port(8080)
            .log_dir("/var/log")
            .log_level("debug")
            .build_unchecked();

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: RouterConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.host, deserialized.host);
        assert_eq!(config.port, deserialized.port);
        assert_eq!(config.max_payload_size, deserialized.max_payload_size);
        assert_eq!(config.log_dir, deserialized.log_dir);
        assert_eq!(config.log_level, deserialized.log_level);
        assert!(deserialized.discovery.is_none());
        assert!(deserialized.metrics.is_none());
    }

    #[test]
    fn test_routing_mode_is_pd_mode() {
        let regular = RoutingMode::Regular {
            worker_urls: vec!["http://worker1".to_string()],
        };
        assert!(!regular.is_pd_mode());

        let pd = RoutingMode::PrefillDecode {
            prefill_urls: vec![("http://prefill1".to_string(), Some(8001))],
            decode_urls: vec!["http://decode1".to_string()],
            prefill_policy: None,
            decode_policy: None,
        };
        assert!(pd.is_pd_mode());
    }

    #[test]
    fn test_routing_mode_worker_count() {
        let regular = RoutingMode::Regular {
            worker_urls: vec![
                "http://worker1".to_string(),
                "http://worker2".to_string(),
                "http://worker3".to_string(),
            ],
        };
        assert_eq!(regular.worker_count(), 3);

        let pd = RoutingMode::PrefillDecode {
            prefill_urls: vec![
                ("http://prefill1".to_string(), Some(8001)),
                ("http://prefill2".to_string(), None),
            ],
            decode_urls: vec![
                "http://decode1".to_string(),
                "http://decode2".to_string(),
                "http://decode3".to_string(),
            ],
            prefill_policy: None,
            decode_policy: None,
        };
        assert_eq!(pd.worker_count(), 5);

        let empty_regular = RoutingMode::Regular {
            worker_urls: vec![],
        };
        assert_eq!(empty_regular.worker_count(), 0);
    }

    #[test]
    fn test_routing_mode_serialization() {
        let regular = RoutingMode::Regular {
            worker_urls: vec!["http://worker1".to_string()],
        };
        let json = serde_json::to_string(&regular).unwrap();
        assert!(json.contains("\"type\":\"regular\""));
        assert!(json.contains("\"worker_urls\""));

        let pd = RoutingMode::PrefillDecode {
            prefill_urls: vec![("http://prefill1".to_string(), Some(8001))],
            decode_urls: vec!["http://decode1".to_string()],
            prefill_policy: None,
            decode_policy: None,
        };
        let json = serde_json::to_string(&pd).unwrap();
        assert!(json.contains("\"type\":\"prefill_decode\""));
        assert!(json.contains("\"prefill_urls\""));
        assert!(json.contains("\"decode_urls\""));
    }

    #[test]
    fn test_policy_config_name() {
        assert_eq!(PolicyConfig::Random.name(), "random");
        assert_eq!(PolicyConfig::RoundRobin.name(), "round_robin");

        let cache_aware = PolicyConfig::CacheAware {
            cache_threshold: 0.8,
            balance_abs_threshold: 10,
            balance_rel_threshold: 1.5,
            eviction_interval_secs: 300,
            max_tree_size: 1000,
        };
        assert_eq!(cache_aware.name(), "cache_aware");

        let power_of_two = PolicyConfig::PowerOfTwo {
            load_check_interval_secs: 60,
        };
        assert_eq!(power_of_two.name(), "power_of_two");
    }

    #[test]
    fn test_policy_config_serialization() {
        let random = PolicyConfig::Random;
        let json = serde_json::to_string(&random).unwrap();
        assert_eq!(json, r#"{"type":"random"}"#);

        let cache_aware = PolicyConfig::CacheAware {
            cache_threshold: 0.8,
            balance_abs_threshold: 10,
            balance_rel_threshold: 1.5,
            eviction_interval_secs: 300,
            max_tree_size: 1000,
        };
        let json = serde_json::to_string(&cache_aware).unwrap();
        assert!(json.contains("\"type\":\"cache_aware\""));
        assert!(json.contains("\"cache_threshold\":0.8"));
        assert!(json.contains("\"balance_abs_threshold\":10"));

        let power_of_two = PolicyConfig::PowerOfTwo {
            load_check_interval_secs: 60,
        };
        let json = serde_json::to_string(&power_of_two).unwrap();
        assert!(json.contains("\"type\":\"power_of_two\""));
        assert!(json.contains("\"load_check_interval_secs\":60"));
    }

    #[test]
    fn test_cache_aware_parameters() {
        let cache_aware = PolicyConfig::CacheAware {
            cache_threshold: 0.75,
            balance_abs_threshold: 20,
            balance_rel_threshold: 2.0,
            eviction_interval_secs: 600,
            max_tree_size: 5000,
        };

        match cache_aware {
            PolicyConfig::CacheAware {
                cache_threshold,
                balance_abs_threshold,
                balance_rel_threshold,
                eviction_interval_secs,
                max_tree_size,
            } => {
                assert!((cache_threshold - 0.75).abs() < 0.0001);
                assert_eq!(balance_abs_threshold, 20);
                assert!((balance_rel_threshold - 2.0).abs() < 0.0001);
                assert_eq!(eviction_interval_secs, 600);
                assert_eq!(max_tree_size, 5000);
            }
            _ => panic!("Expected CacheAware"),
        }
    }

    #[test]
    fn test_power_of_two_parameters() {
        let power_of_two = PolicyConfig::PowerOfTwo {
            load_check_interval_secs: 120,
        };

        match power_of_two {
            PolicyConfig::PowerOfTwo {
                load_check_interval_secs,
            } => {
                assert_eq!(load_check_interval_secs, 120);
            }
            _ => panic!("Expected PowerOfTwo"),
        }
    }

    #[test]
    fn test_bucket_parameters() {
        let bucket = PolicyConfig::Bucket {
            balance_abs_threshold: 20,
            balance_rel_threshold: 2.0,
            bucket_adjust_interval_secs: 5,
        };

        match bucket {
            PolicyConfig::Bucket {
                balance_abs_threshold,
                balance_rel_threshold,
                bucket_adjust_interval_secs,
            } => {
                assert_eq!(balance_abs_threshold, 20);
                assert!((balance_rel_threshold - 2.0).abs() < 0.0001);
                assert_eq!(bucket_adjust_interval_secs, 5);
            }
            _ => panic!("Expected Bucket"),
        }
    }

    #[test]
    fn test_discovery_config_default() {
        let config = DiscoveryConfig::default();

        assert!(!config.enabled);
        assert!(config.namespace.is_none());
        assert_eq!(config.port, 8000);
        assert_eq!(config.check_interval_secs, 120);
        assert!(config.selector.is_empty());
        assert!(config.prefill_selector.is_empty());
        assert!(config.decode_selector.is_empty());
        assert_eq!(config.bootstrap_port_annotation, "sglang.ai/bootstrap-port");
    }

    #[test]
    fn test_discovery_config_with_selectors() {
        let mut selector = HashMap::new();
        selector.insert("app".to_string(), "sglang".to_string());
        selector.insert("role".to_string(), "worker".to_string());

        let config = DiscoveryConfig {
            enabled: true,
            namespace: Some("default".to_string()),
            port: 9000,
            check_interval_secs: 30,
            selector: selector.clone(),
            prefill_selector: selector.clone(),
            decode_selector: selector.clone(),
            bootstrap_port_annotation: "custom.io/port".to_string(),
        };

        assert!(config.enabled);
        assert_eq!(config.namespace, Some("default".to_string()));
        assert_eq!(config.port, 9000);
        assert_eq!(config.selector.len(), 2);
        assert_eq!(config.selector.get("app"), Some(&"sglang".to_string()));
    }

    #[test]
    fn test_discovery_config_namespace() {
        let config = DiscoveryConfig {
            namespace: None,
            ..Default::default()
        };
        assert!(config.namespace.is_none());

        let config = DiscoveryConfig {
            namespace: Some("production".to_string()),
            ..Default::default()
        };
        assert_eq!(config.namespace, Some("production".to_string()));
    }

    #[test]
    fn test_metrics_config_default() {
        let config = MetricsConfig::default();

        assert_eq!(config.port, 29000);
        assert_eq!(config.host, "0.0.0.0");
    }

    #[test]
    fn test_metrics_config_custom() {
        let config = MetricsConfig {
            port: 9090,
            host: "0.0.0.0".to_string(),
        };

        assert_eq!(config.port, 9090);
        assert_eq!(config.host, "0.0.0.0");
    }

    #[test]
    fn test_mode_type() {
        let config = RouterConfig::builder()
            .regular_mode(vec![])
            .build_unchecked();
        assert_eq!(config.mode_type(), "regular");

        let config = RouterConfig::builder()
            .prefill_decode_mode(vec![], vec![])
            .build_unchecked();
        assert_eq!(config.mode_type(), "prefill_decode");
    }

    #[test]
    fn test_has_service_discovery() {
        let config = RouterConfig::default();
        assert!(!config.has_service_discovery());

        let config = RouterConfig::builder()
            .discovery_config(DiscoveryConfig {
                enabled: false,
                ..Default::default()
            })
            .build_unchecked();
        assert!(!config.has_service_discovery());

        let config = RouterConfig::builder().enable_discovery().build_unchecked();
        assert!(config.has_service_discovery());
    }

    #[test]
    fn test_has_metrics() {
        let config = RouterConfig::default();
        assert!(!config.has_metrics());

        let config = RouterConfig::builder()
            .metrics_config(MetricsConfig::default())
            .build_unchecked();
        assert!(config.has_metrics());
    }

    #[test]
    fn test_large_worker_lists() {
        let large_urls: Vec<String> = (0..1000).map(|i| format!("http://worker{}", i)).collect();

        let config = RouterConfig::builder()
            .regular_mode(large_urls.clone())
            .build_unchecked();

        assert_eq!(config.mode.worker_count(), 1000);

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: RouterConfig = serde_json::from_str(&json).unwrap();

        match deserialized.mode {
            RoutingMode::Regular { worker_urls } => {
                assert_eq!(worker_urls.len(), 1000);
            }
            _ => panic!("Expected Regular mode"),
        }
    }

    #[test]
    fn test_unicode_in_config() {
        let config = RouterConfig::builder()
            .regular_mode(vec![
                "http://работник1".to_string(),
                "http://工作者2".to_string(),
            ])
            .log_dir("/日志/目录")
            .build_unchecked();

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: RouterConfig = serde_json::from_str(&json).unwrap();

        match deserialized.mode {
            RoutingMode::Regular { worker_urls } => {
                assert_eq!(worker_urls[0], "http://работник1");
                assert_eq!(worker_urls[1], "http://工作者2");
            }
            _ => panic!("Expected Regular mode"),
        }

        assert_eq!(deserialized.log_dir, Some("/日志/目录".to_string()));
    }

    #[test]
    fn test_empty_string_fields() {
        let config = RouterConfig::builder()
            .host("")
            .log_dir("")
            .log_level("")
            .build_unchecked();

        assert_eq!(config.host, "");
        assert_eq!(config.log_dir, Some("".to_string()));
        assert_eq!(config.log_level, Some("".to_string()));
    }

    #[test]
    fn test_full_pd_mode_config() {
        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![
                    ("http://prefill1:8000".to_string(), Some(8001)),
                    ("http://prefill2:8000".to_string(), None),
                ],
                vec![
                    "http://decode1:8000".to_string(),
                    "http://decode2:8000".to_string(),
                ],
            )
            .power_of_two_policy(30)
            .host("0.0.0.0")
            .port(3000)
            .max_payload_size(1048576)
            .request_timeout_secs(120)
            .worker_startup_timeout_secs(60)
            .worker_startup_check_interval_secs(5)
            .discovery_config(DiscoveryConfig {
                enabled: true,
                namespace: Some("sglang".to_string()),
                ..Default::default()
            })
            .enable_metrics("0.0.0.0", 9090)
            .log_dir("/var/log/sglang")
            .log_level("info")
            .max_concurrent_requests(64)
            .build_unchecked();

        assert!(config.mode.is_pd_mode());
        assert_eq!(config.mode.worker_count(), 4);
        assert_eq!(config.policy.name(), "power_of_two");
        assert!(config.has_service_discovery());
        assert!(config.has_metrics());
    }

    #[test]
    fn test_full_regular_mode_config() {
        let mut selector = HashMap::new();
        selector.insert("app".to_string(), "sglang".to_string());

        let config = RouterConfig::builder()
            .regular_mode(vec![
                "http://worker1:8000".to_string(),
                "http://worker2:8000".to_string(),
                "http://worker3:8000".to_string(),
            ])
            .cache_aware_policy(0.9, 5, 1.2, 600, 10000)
            .host("0.0.0.0")
            .port(3001)
            .max_payload_size(536870912)
            .request_timeout_secs(300)
            .worker_startup_timeout_secs(180)
            .worker_startup_check_interval_secs(15)
            .discovery_config(DiscoveryConfig {
                enabled: true,
                namespace: None,
                port: 8080,
                check_interval_secs: 45,
                selector,
                ..Default::default()
            })
            .metrics_config(MetricsConfig::default())
            .log_level("debug")
            .max_concurrent_requests(64)
            .build_unchecked();

        assert!(!config.mode.is_pd_mode());
        assert_eq!(config.mode.worker_count(), 3);
        assert_eq!(config.policy.name(), "cache_aware");
        assert!(config.has_service_discovery());
        assert!(config.has_metrics());
    }

    #[test]
    fn test_config_with_all_options() {
        let mut selectors = HashMap::new();
        selectors.insert("env".to_string(), "prod".to_string());
        selectors.insert("version".to_string(), "v1".to_string());

        let config = RouterConfig::builder()
            .regular_mode(vec!["http://worker1".to_string()])
            .round_robin_policy()
            .host("::1") // IPv6
            .port(8888)
            .max_payload_size(1024 * 1024 * 512) // 512MB
            .request_timeout_secs(900)
            .worker_startup_timeout_secs(600)
            .worker_startup_check_interval_secs(20)
            .discovery_config(DiscoveryConfig {
                enabled: true,
                namespace: Some("production".to_string()),
                port: 8443,
                check_interval_secs: 120,
                selector: selectors.clone(),
                prefill_selector: selectors.clone(),
                decode_selector: selectors,
                bootstrap_port_annotation: "mycompany.io/bootstrap".to_string(),
            })
            .enable_metrics("::", 9999) // IPv6 any
            .log_dir("/opt/logs/sglang")
            .log_level("trace")
            .max_concurrent_requests(64)
            .build_unchecked();

        assert!(config.has_service_discovery());
        assert!(config.has_metrics());
        assert_eq!(config.mode_type(), "regular");

        let json = serde_json::to_string_pretty(&config).unwrap();
        let deserialized: RouterConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.host, "::1");
        assert_eq!(deserialized.port, 8888);
        assert_eq!(
            deserialized.discovery.unwrap().namespace,
            Some("production".to_string())
        );
    }

    #[test]
    fn test_pd_policy_fallback_both_specified() {
        let pd = RoutingMode::PrefillDecode {
            prefill_urls: vec![("http://prefill1".to_string(), None)],
            decode_urls: vec!["http://decode1".to_string()],
            prefill_policy: Some(PolicyConfig::CacheAware {
                cache_threshold: 0.5,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                eviction_interval_secs: 60,
                max_tree_size: 1000,
            }),
            decode_policy: Some(PolicyConfig::PowerOfTwo {
                load_check_interval_secs: 60,
            }),
        };

        let main_policy = PolicyConfig::Random;

        match pd.get_prefill_policy(&main_policy) {
            PolicyConfig::CacheAware { .. } => {}
            _ => panic!("Expected CacheAware for prefill"),
        }

        match pd.get_decode_policy(&main_policy) {
            PolicyConfig::PowerOfTwo { .. } => {}
            _ => panic!("Expected PowerOfTwo for decode"),
        }
    }

    #[test]
    fn test_pd_policy_fallback_only_prefill() {
        let pd = RoutingMode::PrefillDecode {
            prefill_urls: vec![("http://prefill1".to_string(), None)],
            decode_urls: vec!["http://decode1".to_string()],
            prefill_policy: Some(PolicyConfig::CacheAware {
                cache_threshold: 0.5,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                eviction_interval_secs: 60,
                max_tree_size: 1000,
            }),
            decode_policy: None,
        };

        let main_policy = PolicyConfig::RoundRobin;

        match pd.get_prefill_policy(&main_policy) {
            PolicyConfig::CacheAware { .. } => {}
            _ => panic!("Expected CacheAware for prefill"),
        }

        match pd.get_decode_policy(&main_policy) {
            PolicyConfig::RoundRobin => {}
            _ => panic!("Expected RoundRobin for decode"),
        }
    }

    #[test]
    fn test_pd_policy_fallback_only_decode() {
        let pd = RoutingMode::PrefillDecode {
            prefill_urls: vec![("http://prefill1".to_string(), None)],
            decode_urls: vec!["http://decode1".to_string()],
            prefill_policy: None,
            decode_policy: Some(PolicyConfig::PowerOfTwo {
                load_check_interval_secs: 60,
            }),
        };

        let main_policy = PolicyConfig::Random;

        match pd.get_prefill_policy(&main_policy) {
            PolicyConfig::Random => {}
            _ => panic!("Expected Random for prefill"),
        }

        match pd.get_decode_policy(&main_policy) {
            PolicyConfig::PowerOfTwo { .. } => {}
            _ => panic!("Expected PowerOfTwo for decode"),
        }
    }

    #[test]
    fn test_pd_policy_fallback_none_specified() {
        let pd = RoutingMode::PrefillDecode {
            prefill_urls: vec![("http://prefill1".to_string(), None)],
            decode_urls: vec!["http://decode1".to_string()],
            prefill_policy: None,
            decode_policy: None,
        };

        let main_policy = PolicyConfig::CacheAware {
            cache_threshold: 0.7,
            balance_abs_threshold: 20,
            balance_rel_threshold: 1.5,
            eviction_interval_secs: 300,
            max_tree_size: 2000,
        };

        match pd.get_prefill_policy(&main_policy) {
            PolicyConfig::CacheAware {
                cache_threshold, ..
            } => {
                assert!((cache_threshold - 0.7).abs() < 0.0001);
            }
            _ => panic!("Expected CacheAware for prefill"),
        }

        match pd.get_decode_policy(&main_policy) {
            PolicyConfig::CacheAware {
                cache_threshold, ..
            } => {
                assert!((cache_threshold - 0.7).abs() < 0.0001);
            }
            _ => panic!("Expected CacheAware for decode"),
        }
    }

    #[test]
    fn test_regular_mode_policy_fallback() {
        let regular = RoutingMode::Regular {
            worker_urls: vec!["http://worker1".to_string()],
        };

        let main_policy = PolicyConfig::RoundRobin;

        match regular.get_prefill_policy(&main_policy) {
            PolicyConfig::RoundRobin => {}
            _ => panic!("Expected RoundRobin for regular mode"),
        }

        match regular.get_decode_policy(&main_policy) {
            PolicyConfig::RoundRobin => {}
            _ => panic!("Expected RoundRobin for regular mode"),
        }
    }
}
