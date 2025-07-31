use super::ConfigResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Main router configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    /// Routing mode configuration
    pub mode: RoutingMode,
    /// Policy configuration
    pub policy: PolicyConfig,
    /// Server host address
    pub host: String,
    /// Server port
    pub port: u16,
    /// Maximum payload size in bytes
    pub max_payload_size: usize,
    /// Request timeout in seconds
    pub request_timeout_secs: u64,
    /// Worker startup timeout in seconds
    pub worker_startup_timeout_secs: u64,
    /// Worker health check interval in seconds
    pub worker_startup_check_interval_secs: u64,
    /// Enable data parallelism aware schedule
    pub dp_aware: bool,
    /// The api key used for the authorization with the worker
    pub api_key: Option<String>,
    /// Service discovery configuration (optional)
    pub discovery: Option<DiscoveryConfig>,
    /// Metrics configuration (optional)
    pub metrics: Option<MetricsConfig>,
    /// Log directory (None = stdout only)
    pub log_dir: Option<String>,
    /// Log level (None = info)
    pub log_level: Option<String>,
    /// Custom request ID headers to check (defaults to common headers)
    pub request_id_headers: Option<Vec<String>>,
    /// Maximum concurrent requests allowed (for rate limiting)
    pub max_concurrent_requests: usize,
    /// CORS allowed origins
    pub cors_allowed_origins: Vec<String>,
}

/// Routing mode configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RoutingMode {
    #[serde(rename = "regular")]
    Regular {
        /// List of worker URLs
        worker_urls: Vec<String>,
    },
    #[serde(rename = "prefill_decode")]
    PrefillDecode {
        /// Prefill worker URLs with optional bootstrap ports
        prefill_urls: Vec<(String, Option<u16>)>,
        /// Decode worker URLs
        decode_urls: Vec<String>,
        /// Optional separate policy for prefill workers
        #[serde(skip_serializing_if = "Option::is_none")]
        prefill_policy: Option<PolicyConfig>,
        /// Optional separate policy for decode workers
        #[serde(skip_serializing_if = "Option::is_none")]
        decode_policy: Option<PolicyConfig>,
    },
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
        /// Minimum prefix match ratio to use cache-based routing
        cache_threshold: f32,
        /// Absolute load difference threshold for load balancing
        balance_abs_threshold: usize,
        /// Relative load ratio threshold for load balancing
        balance_rel_threshold: f32,
        /// Interval between cache eviction cycles (seconds)
        eviction_interval_secs: u64,
        /// Maximum cache tree size per tenant
        max_tree_size: usize,
    },

    #[serde(rename = "power_of_two")]
    PowerOfTwo {
        /// Interval for load monitoring (seconds)
        load_check_interval_secs: u64,
    },
}

impl PolicyConfig {
    pub fn name(&self) -> &'static str {
        match self {
            PolicyConfig::Random => "random",
            PolicyConfig::RoundRobin => "round_robin",
            PolicyConfig::CacheAware { .. } => "cache_aware",
            PolicyConfig::PowerOfTwo { .. } => "power_of_two",
        }
    }
}

/// Service discovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    /// Enable service discovery
    pub enabled: bool,
    /// Kubernetes namespace (None = all namespaces)
    pub namespace: Option<String>,
    /// Service discovery port
    pub port: u16,
    /// Check interval for service discovery
    pub check_interval_secs: u64,
    /// Regular mode selector
    pub selector: HashMap<String, String>,
    /// PD mode prefill selector
    pub prefill_selector: HashMap<String, String>,
    /// PD mode decode selector
    pub decode_selector: HashMap<String, String>,
    /// Bootstrap port annotation key
    pub bootstrap_port_annotation: String,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            namespace: None,
            port: 8000,
            check_interval_secs: 60,
            selector: HashMap::new(),
            prefill_selector: HashMap::new(),
            decode_selector: HashMap::new(),
            bootstrap_port_annotation: "sglang.ai/bootstrap-port".to_string(),
        }
    }
}

/// Metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Prometheus metrics port
    pub port: u16,
    /// Prometheus metrics host
    pub host: String,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            port: 29000,
            host: "127.0.0.1".to_string(),
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
            host: "127.0.0.1".to_string(),
            port: 3001,
            max_payload_size: 268_435_456, // 256MB
            request_timeout_secs: 600,
            worker_startup_timeout_secs: 300,
            worker_startup_check_interval_secs: 10,
            dp_aware: false,
            api_key: None,
            discovery: None,
            metrics: None,
            log_dir: None,
            log_level: None,
            request_id_headers: None,
            max_concurrent_requests: 64,
            cors_allowed_origins: vec![],
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
        }
    }

    /// Check if service discovery is enabled
    pub fn has_service_discovery(&self) -> bool {
        self.discovery.as_ref().map_or(false, |d| d.enabled)
    }

    /// Check if metrics are enabled
    pub fn has_metrics(&self) -> bool {
        self.metrics.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============= RouterConfig Tests =============

    #[test]
    fn test_router_config_default() {
        let config = RouterConfig::default();

        assert!(
            matches!(config.mode, RoutingMode::Regular { worker_urls } if worker_urls.is_empty())
        );
        assert!(matches!(config.policy, PolicyConfig::Random));
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 3001);
        assert_eq!(config.max_payload_size, 268_435_456);
        assert_eq!(config.request_timeout_secs, 600);
        assert_eq!(config.worker_startup_timeout_secs, 300);
        assert_eq!(config.worker_startup_check_interval_secs, 10);
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
        // Other fields should be default
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 3001);
    }

    #[test]
    fn test_router_config_serialization() {
        let config = RouterConfig {
            mode: RoutingMode::Regular {
                worker_urls: vec!["http://worker1".to_string()],
            },
            policy: PolicyConfig::Random,
            host: "0.0.0.0".to_string(),
            port: 8080,
            max_payload_size: 1024,
            request_timeout_secs: 30,
            worker_startup_timeout_secs: 60,
            worker_startup_check_interval_secs: 5,
            dp_aware: false,
            api_key: None,
            discovery: Some(DiscoveryConfig::default()),
            metrics: Some(MetricsConfig::default()),
            log_dir: Some("/var/log".to_string()),
            log_level: Some("debug".to_string()),
            request_id_headers: None,
            max_concurrent_requests: 64,
            cors_allowed_origins: vec![],
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: RouterConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.host, deserialized.host);
        assert_eq!(config.port, deserialized.port);
        assert_eq!(config.max_payload_size, deserialized.max_payload_size);
        assert!(deserialized.discovery.is_some());
        assert!(deserialized.metrics.is_some());
    }

    // ============= RoutingMode Tests =============

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
        // Test Regular mode
        let regular = RoutingMode::Regular {
            worker_urls: vec!["http://worker1".to_string()],
        };
        let json = serde_json::to_string(&regular).unwrap();
        assert!(json.contains("\"type\":\"regular\""));
        assert!(json.contains("\"worker_urls\""));

        // Test PrefillDecode mode
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

    // ============= PolicyConfig Tests =============

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
        // Test Random
        let random = PolicyConfig::Random;
        let json = serde_json::to_string(&random).unwrap();
        assert_eq!(json, r#"{"type":"random"}"#);

        // Test CacheAware with all parameters
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

        // Test PowerOfTwo
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

    // ============= DiscoveryConfig Tests =============

    #[test]
    fn test_discovery_config_default() {
        let config = DiscoveryConfig::default();

        assert!(!config.enabled);
        assert!(config.namespace.is_none());
        assert_eq!(config.port, 8000);
        assert_eq!(config.check_interval_secs, 60);
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
        // Test None namespace (all namespaces)
        let config = DiscoveryConfig {
            namespace: None,
            ..Default::default()
        };
        assert!(config.namespace.is_none());

        // Test specific namespace
        let config = DiscoveryConfig {
            namespace: Some("production".to_string()),
            ..Default::default()
        };
        assert_eq!(config.namespace, Some("production".to_string()));
    }

    // ============= MetricsConfig Tests =============

    #[test]
    fn test_metrics_config_default() {
        let config = MetricsConfig::default();

        assert_eq!(config.port, 29000);
        assert_eq!(config.host, "127.0.0.1");
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

    // ============= RouterConfig Utility Methods Tests =============

    #[test]
    fn test_mode_type() {
        let config = RouterConfig {
            mode: RoutingMode::Regular {
                worker_urls: vec![],
            },
            ..Default::default()
        };
        assert_eq!(config.mode_type(), "regular");

        let config = RouterConfig {
            mode: RoutingMode::PrefillDecode {
                prefill_urls: vec![],
                decode_urls: vec![],
                prefill_policy: None,
                decode_policy: None,
            },
            ..Default::default()
        };
        assert_eq!(config.mode_type(), "prefill_decode");
    }

    #[test]
    fn test_has_service_discovery() {
        let config = RouterConfig::default();
        assert!(!config.has_service_discovery());

        let config = RouterConfig {
            discovery: Some(DiscoveryConfig {
                enabled: false,
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(!config.has_service_discovery());

        let config = RouterConfig {
            discovery: Some(DiscoveryConfig {
                enabled: true,
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(config.has_service_discovery());
    }

    #[test]
    fn test_has_metrics() {
        let config = RouterConfig::default();
        assert!(!config.has_metrics());

        let config = RouterConfig {
            metrics: Some(MetricsConfig::default()),
            ..Default::default()
        };
        assert!(config.has_metrics());
    }

    // ============= Edge Cases =============

    #[test]
    fn test_large_worker_lists() {
        let large_urls: Vec<String> = (0..1000).map(|i| format!("http://worker{}", i)).collect();

        let mode = RoutingMode::Regular {
            worker_urls: large_urls.clone(),
        };

        assert_eq!(mode.worker_count(), 1000);

        // Test serialization with large list
        let config = RouterConfig {
            mode,
            ..Default::default()
        };

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
        let config = RouterConfig {
            mode: RoutingMode::Regular {
                worker_urls: vec!["http://работник1".to_string(), "http://工作者2".to_string()],
            },
            log_dir: Some("/日志/目录".to_string()),
            ..Default::default()
        };

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
        let config = RouterConfig {
            host: "".to_string(),
            log_dir: Some("".to_string()),
            log_level: Some("".to_string()),
            ..Default::default()
        };

        assert_eq!(config.host, "");
        assert_eq!(config.log_dir, Some("".to_string()));
        assert_eq!(config.log_level, Some("".to_string()));
    }

    // ============= Complex Configuration Tests =============

    #[test]
    fn test_full_pd_mode_config() {
        let config = RouterConfig {
            mode: RoutingMode::PrefillDecode {
                prefill_urls: vec![
                    ("http://prefill1:8000".to_string(), Some(8001)),
                    ("http://prefill2:8000".to_string(), None),
                ],
                decode_urls: vec![
                    "http://decode1:8000".to_string(),
                    "http://decode2:8000".to_string(),
                ],
                prefill_policy: None,
                decode_policy: None,
            },
            policy: PolicyConfig::PowerOfTwo {
                load_check_interval_secs: 30,
            },
            host: "0.0.0.0".to_string(),
            port: 3000,
            max_payload_size: 1048576,
            request_timeout_secs: 120,
            worker_startup_timeout_secs: 60,
            worker_startup_check_interval_secs: 5,
            dp_aware: false,
            api_key: None,
            discovery: Some(DiscoveryConfig {
                enabled: true,
                namespace: Some("sglang".to_string()),
                ..Default::default()
            }),
            metrics: Some(MetricsConfig {
                port: 9090,
                host: "0.0.0.0".to_string(),
            }),
            log_dir: Some("/var/log/sglang".to_string()),
            log_level: Some("info".to_string()),
            request_id_headers: None,
            max_concurrent_requests: 64,
            cors_allowed_origins: vec![],
        };

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

        let config = RouterConfig {
            mode: RoutingMode::Regular {
                worker_urls: vec![
                    "http://worker1:8000".to_string(),
                    "http://worker2:8000".to_string(),
                    "http://worker3:8000".to_string(),
                ],
            },
            policy: PolicyConfig::CacheAware {
                cache_threshold: 0.9,
                balance_abs_threshold: 5,
                balance_rel_threshold: 1.2,
                eviction_interval_secs: 600,
                max_tree_size: 10000,
            },
            host: "0.0.0.0".to_string(),
            port: 3001,
            max_payload_size: 536870912,
            request_timeout_secs: 300,
            worker_startup_timeout_secs: 180,
            worker_startup_check_interval_secs: 15,
            dp_aware: false,
            api_key: None,
            discovery: Some(DiscoveryConfig {
                enabled: true,
                namespace: None,
                port: 8080,
                check_interval_secs: 45,
                selector,
                ..Default::default()
            }),
            metrics: Some(MetricsConfig::default()),
            log_dir: None,
            log_level: Some("debug".to_string()),
            request_id_headers: None,
            max_concurrent_requests: 64,
            cors_allowed_origins: vec![],
        };

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

        let config = RouterConfig {
            mode: RoutingMode::Regular {
                worker_urls: vec!["http://worker1".to_string()],
            },
            policy: PolicyConfig::RoundRobin,
            host: "::1".to_string(), // IPv6
            port: 8888,
            max_payload_size: 1024 * 1024 * 512, // 512MB
            request_timeout_secs: 900,
            worker_startup_timeout_secs: 600,
            worker_startup_check_interval_secs: 20,
            dp_aware: false,
            api_key: None,
            discovery: Some(DiscoveryConfig {
                enabled: true,
                namespace: Some("production".to_string()),
                port: 8443,
                check_interval_secs: 120,
                selector: selectors.clone(),
                prefill_selector: selectors.clone(),
                decode_selector: selectors,
                bootstrap_port_annotation: "mycompany.io/bootstrap".to_string(),
            }),
            metrics: Some(MetricsConfig {
                port: 9999,
                host: "::".to_string(), // IPv6 any
            }),
            log_dir: Some("/opt/logs/sglang".to_string()),
            log_level: Some("trace".to_string()),
            request_id_headers: None,
            max_concurrent_requests: 64,
            cors_allowed_origins: vec![],
        };

        assert!(config.has_service_discovery());
        assert!(config.has_metrics());
        assert_eq!(config.mode_type(), "regular");

        // Test round-trip serialization
        let json = serde_json::to_string_pretty(&config).unwrap();
        let deserialized: RouterConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.host, "::1");
        assert_eq!(deserialized.port, 8888);
        assert_eq!(
            deserialized.discovery.unwrap().namespace,
            Some("production".to_string())
        );
    }

    // ============= Policy Fallback Tests =============

    #[test]
    fn test_pd_policy_fallback_both_specified() {
        // When both prefill and decode policies are specified, they should be used
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

        // Both specific policies should be used
        match pd.get_prefill_policy(&main_policy) {
            PolicyConfig::CacheAware { .. } => {} // Success
            _ => panic!("Expected CacheAware for prefill"),
        }

        match pd.get_decode_policy(&main_policy) {
            PolicyConfig::PowerOfTwo { .. } => {} // Success
            _ => panic!("Expected PowerOfTwo for decode"),
        }
    }

    #[test]
    fn test_pd_policy_fallback_only_prefill() {
        // When only prefill policy is specified, decode should use main policy
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

        // Prefill should use specific policy
        match pd.get_prefill_policy(&main_policy) {
            PolicyConfig::CacheAware { .. } => {} // Success
            _ => panic!("Expected CacheAware for prefill"),
        }

        // Decode should fall back to main policy
        match pd.get_decode_policy(&main_policy) {
            PolicyConfig::RoundRobin => {} // Success
            _ => panic!("Expected RoundRobin for decode"),
        }
    }

    #[test]
    fn test_pd_policy_fallback_only_decode() {
        // When only decode policy is specified, prefill should use main policy
        let pd = RoutingMode::PrefillDecode {
            prefill_urls: vec![("http://prefill1".to_string(), None)],
            decode_urls: vec!["http://decode1".to_string()],
            prefill_policy: None,
            decode_policy: Some(PolicyConfig::PowerOfTwo {
                load_check_interval_secs: 60,
            }),
        };

        let main_policy = PolicyConfig::Random;

        // Prefill should fall back to main policy
        match pd.get_prefill_policy(&main_policy) {
            PolicyConfig::Random => {} // Success
            _ => panic!("Expected Random for prefill"),
        }

        // Decode should use specific policy
        match pd.get_decode_policy(&main_policy) {
            PolicyConfig::PowerOfTwo { .. } => {} // Success
            _ => panic!("Expected PowerOfTwo for decode"),
        }
    }

    #[test]
    fn test_pd_policy_fallback_none_specified() {
        // When no specific policies are specified, both should use main policy
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

        // Both should fall back to main policy
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
        // For regular mode, the helper methods should just return the main policy
        let regular = RoutingMode::Regular {
            worker_urls: vec!["http://worker1".to_string()],
        };

        let main_policy = PolicyConfig::RoundRobin;

        // Both methods should return main policy for regular mode
        match regular.get_prefill_policy(&main_policy) {
            PolicyConfig::RoundRobin => {} // Success
            _ => panic!("Expected RoundRobin for regular mode"),
        }

        match regular.get_decode_policy(&main_policy) {
            PolicyConfig::RoundRobin => {} // Success
            _ => panic!("Expected RoundRobin for regular mode"),
        }
    }
}
