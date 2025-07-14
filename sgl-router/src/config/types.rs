use super::{ConfigError, ConfigResult};
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
    /// Service discovery configuration (optional)
    pub discovery: Option<DiscoveryConfig>,
    /// Metrics configuration (optional)
    pub metrics: Option<MetricsConfig>,
    /// Log directory (None = stdout only)
    pub log_dir: Option<String>,
    /// Log level (None = info)
    pub log_level: Option<String>,
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
            } => prefill_urls.len() + decode_urls.len(),
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
            discovery: None,
            metrics: None,
            log_dir: None,
            log_level: None,
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

    /// Convert to routing PolicyConfig for internal use
    pub fn to_routing_policy_config(&self) -> ConfigResult<crate::router::PolicyConfig> {
        match (&self.mode, &self.policy) {
            (
                RoutingMode::PrefillDecode {
                    prefill_urls,
                    decode_urls,
                },
                policy,
            ) => {
                // Map policy to PDSelectionPolicy
                let selection_policy = match policy {
                    PolicyConfig::Random => crate::pd_types::PDSelectionPolicy::Random,
                    PolicyConfig::PowerOfTwo { .. } => {
                        crate::pd_types::PDSelectionPolicy::PowerOfTwo
                    }
                    PolicyConfig::CacheAware { .. } => {
                        return Err(ConfigError::IncompatibleConfig {
                            reason: "CacheAware policy is not supported in PD disaggregated mode"
                                .to_string(),
                        });
                    }
                    PolicyConfig::RoundRobin => {
                        return Err(ConfigError::IncompatibleConfig {
                            reason: "RoundRobin policy is not supported in PD disaggregated mode"
                                .to_string(),
                        });
                    }
                };

                Ok(crate::router::PolicyConfig::PrefillDecodeConfig {
                    selection_policy,
                    prefill_urls: prefill_urls.clone(),
                    decode_urls: decode_urls.clone(),
                    timeout_secs: self.worker_startup_timeout_secs,
                    interval_secs: self.worker_startup_check_interval_secs,
                })
            }
            (RoutingMode::Regular { .. }, PolicyConfig::Random) => {
                Ok(crate::router::PolicyConfig::RandomConfig {
                    timeout_secs: self.worker_startup_timeout_secs,
                    interval_secs: self.worker_startup_check_interval_secs,
                })
            }
            (RoutingMode::Regular { .. }, PolicyConfig::RoundRobin) => {
                Ok(crate::router::PolicyConfig::RoundRobinConfig {
                    timeout_secs: self.worker_startup_timeout_secs,
                    interval_secs: self.worker_startup_check_interval_secs,
                })
            }
            (
                RoutingMode::Regular { .. },
                PolicyConfig::CacheAware {
                    cache_threshold,
                    balance_abs_threshold,
                    balance_rel_threshold,
                    eviction_interval_secs,
                    max_tree_size,
                },
            ) => Ok(crate::router::PolicyConfig::CacheAwareConfig {
                cache_threshold: *cache_threshold,
                balance_abs_threshold: *balance_abs_threshold,
                balance_rel_threshold: *balance_rel_threshold,
                eviction_interval_secs: *eviction_interval_secs,
                max_tree_size: *max_tree_size,
                timeout_secs: self.worker_startup_timeout_secs,
                interval_secs: self.worker_startup_check_interval_secs,
            }),
            (RoutingMode::Regular { .. }, PolicyConfig::PowerOfTwo { .. }) => {
                Err(ConfigError::IncompatibleConfig {
                    reason: "PowerOfTwo policy is only supported in PD disaggregated mode"
                        .to_string(),
                })
            }
        }
    }
}
