use clap::Parser;
use sglang_router_rs::config::{
    CircuitBreakerConfig, ConfigError, ConfigResult, DiscoveryConfig, MetricsConfig,
    PolicyConfig, RetryConfig, RouterConfig, RoutingMode,
};
use sglang_router_rs::metrics::PrometheusConfig;
use sglang_router_rs::server::{self, ServerConfig};
use sglang_router_rs::service_discovery::ServiceDiscoveryConfig;
use std::collections::HashMap;

#[derive(Parser, Debug)]
#[command(name = "sglang-router")]
#[command(about = "SGLang Router - High-performance request distribution across worker nodes")]
#[command(long_about = r#"
SGLang Router - High-performance request distribution across worker nodes

Usage:
This launcher enables starting a router with individual worker instances. It is useful for
multi-node setups or when you want to start workers and router separately.

Examples:
  # Regular mode
  sglang-router --worker-urls http://worker1:8000 http://worker2:8000

  # PD disaggregated mode with same policy for both
  sglang-router --pd-disaggregation \
    --prefill http://prefill1:8000 9000 --prefill http://prefill2:8000 \
    --decode http://decode1:8001 --decode http://decode2:8001 \
    --policy cache_aware

  # PD mode with different policies for prefill and decode
  sglang-router --pd-disaggregation \
    --prefill http://prefill1:8000 --prefill http://prefill2:8000 \
    --decode http://decode1:8001 --decode http://decode2:8001 \
    --prefill-policy cache_aware --decode-policy power_of_two
"#)]
struct CliArgs {
    /// Host address to bind the router server
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port number to bind the router server
    #[arg(long, default_value_t = 30000)]
    port: u16,

    /// List of worker URLs (e.g., http://worker1:8000 http://worker2:8000)
    #[arg(long, value_delimiter = ' ', num_args = 0..)]
    worker_urls: Vec<String>,

    /// Load balancing policy to use
    #[arg(long, default_value = "cache_aware", value_parser = ["random", "round_robin", "cache_aware", "power_of_two"])]
    policy: String,

    /// Enable PD (Prefill-Decode) disaggregated mode
    #[arg(long, default_value_t = false)]
    pd_disaggregation: bool,

    /// Prefill server URLs (format: URL or URL:BOOTSTRAP_PORT, can be specified multiple times)
    #[arg(long, value_delimiter = ',')]
    prefill: Vec<String>,

    /// Decode server URLs (can be specified multiple times)
    #[arg(long, value_delimiter = ',')]
    decode: Vec<String>,

    /// Specific policy for prefill nodes in PD mode
    #[arg(long, value_parser = ["random", "round_robin", "cache_aware", "power_of_two"])]
    prefill_policy: Option<String>,

    /// Specific policy for decode nodes in PD mode
    #[arg(long, value_parser = ["random", "round_robin", "cache_aware", "power_of_two"])]
    decode_policy: Option<String>,

    /// Timeout in seconds for worker startup
    #[arg(long, default_value_t = 300)]
    worker_startup_timeout_secs: u64,

    /// Interval in seconds between checks for worker startup
    #[arg(long, default_value_t = 10)]
    worker_startup_check_interval: u64,

    /// Cache threshold (0.0-1.0) for cache-aware routing
    #[arg(long, default_value_t = 0.5)]
    cache_threshold: f32,

    /// Absolute threshold for load balancing
    #[arg(long, default_value_t = 32)]
    balance_abs_threshold: usize,

    /// Relative threshold for load balancing
    #[arg(long, default_value_t = 1.0001)]
    balance_rel_threshold: f32,

    /// Interval in seconds between cache eviction operations
    #[arg(long, default_value_t = 60)]
    eviction_interval: u64,

    /// Maximum size of the approximation tree for cache-aware routing
    #[arg(long, default_value_t = 16777216)] // 2^24
    max_tree_size: usize,

    /// Maximum payload size in bytes
    #[arg(long, default_value_t = 268435456)] // 256MB
    max_payload_size: usize,

    /// Enable data parallelism aware schedule
    #[arg(long, default_value_t = false)]
    dp_aware: bool,

    /// API key for worker authorization
    #[arg(long)]
    api_key: Option<String>,

    /// Directory to store log files
    #[arg(long)]
    log_dir: Option<String>,

    /// Set the logging level
    #[arg(long, default_value = "info", value_parser = ["debug", "info", "warn", "error"])]
    log_level: String,

    /// Enable Kubernetes service discovery
    #[arg(long, default_value_t = false)]
    service_discovery: bool,

    /// Label selector for Kubernetes service discovery (format: key1=value1 key2=value2)
    #[arg(long, value_delimiter = ' ', num_args = 0..)]
    selector: Vec<String>,

    /// Port to use for discovered worker pods
    #[arg(long, default_value_t = 80)]
    service_discovery_port: u16,

    /// Kubernetes namespace to watch for pods
    #[arg(long)]
    service_discovery_namespace: Option<String>,

    /// Label selector for prefill server pods in PD mode
    #[arg(long, value_delimiter = ' ', num_args = 0..)]
    prefill_selector: Vec<String>,

    /// Label selector for decode server pods in PD mode
    #[arg(long, value_delimiter = ' ', num_args = 0..)]
    decode_selector: Vec<String>,

    /// Port to expose Prometheus metrics
    #[arg(long, default_value_t = 29000)]
    prometheus_port: u16,

    /// Host address to bind the Prometheus metrics server
    #[arg(long, default_value = "127.0.0.1")]
    prometheus_host: String,

    /// Custom HTTP headers to check for request IDs
    #[arg(long, value_delimiter = ' ', num_args = 0..)]
    request_id_headers: Vec<String>,

    /// Request timeout in seconds
    #[arg(long, default_value_t = 600)]
    request_timeout_secs: u64,

    /// Maximum number of concurrent requests allowed
    #[arg(long, default_value_t = 64)]
    max_concurrent_requests: usize,

    /// CORS allowed origins
    #[arg(long, value_delimiter = ' ', num_args = 0..)]
    cors_allowed_origins: Vec<String>,

    // Retry configuration
    /// Maximum number of retries
    #[arg(long, default_value_t = 3)]
    retry_max_retries: u32,

    /// Initial backoff in milliseconds for retries
    #[arg(long, default_value_t = 100)]
    retry_initial_backoff_ms: u64,

    /// Maximum backoff in milliseconds for retries
    #[arg(long, default_value_t = 10000)]
    retry_max_backoff_ms: u64,

    /// Backoff multiplier for exponential backoff
    #[arg(long, default_value_t = 2.0)]
    retry_backoff_multiplier: f32,

    /// Jitter factor for retry backoff
    #[arg(long, default_value_t = 0.1)]
    retry_jitter_factor: f32,

    /// Disable retries
    #[arg(long, default_value_t = false)]
    disable_retries: bool,

    // Circuit breaker configuration
    /// Number of failures before circuit breaker opens
    #[arg(long, default_value_t = 5)]
    cb_failure_threshold: u32,

    /// Number of successes before circuit breaker closes
    #[arg(long, default_value_t = 2)]
    cb_success_threshold: u32,

    /// Timeout duration in seconds for circuit breaker
    #[arg(long, default_value_t = 30)]
    cb_timeout_duration_secs: u64,

    /// Window duration in seconds for circuit breaker
    #[arg(long, default_value_t = 60)]
    cb_window_duration_secs: u64,

    /// Disable circuit breaker
    #[arg(long, default_value_t = false)]
    disable_circuit_breaker: bool,
}

impl CliArgs {
    /// Parse selector strings into HashMap
    fn parse_selector(selector_list: &[String]) -> HashMap<String, String> {
        let mut map = HashMap::new();
        for item in selector_list {
            if let Some(eq_pos) = item.find('=') {
                let key = item[..eq_pos].to_string();
                let value = item[eq_pos + 1..].to_string();
                map.insert(key, value);
            }
        }
        map
    }

    /// Parse prefill URLs with optional bootstrap ports
    /// Format can be either "URL" or "URL@PORT" where PORT is the bootstrap port
    fn parse_prefill_urls(&self) -> Vec<(String, Option<u16>)> {
        self.prefill
            .iter()
            .map(|url_str| {
                // Check if there's a bootstrap port specified with @ separator
                if let Some(at_pos) = url_str.rfind('@') {
                    let url = url_str[..at_pos].to_string();
                    let port_str = &url_str[at_pos + 1..];
                    let bootstrap_port = if port_str.to_lowercase() == "none" {
                        None
                    } else {
                        port_str.parse::<u16>().ok()
                    };
                    (url, bootstrap_port)
                } else {
                    (url_str.clone(), None)
                }
            })
            .collect()
    }

    /// Convert policy string to PolicyConfig
    fn parse_policy(&self, policy_str: &str) -> PolicyConfig {
        match policy_str {
            "random" => PolicyConfig::Random,
            "round_robin" => PolicyConfig::RoundRobin,
            "cache_aware" => PolicyConfig::CacheAware {
                cache_threshold: self.cache_threshold,
                balance_abs_threshold: self.balance_abs_threshold,
                balance_rel_threshold: self.balance_rel_threshold,
                eviction_interval_secs: self.eviction_interval,
                max_tree_size: self.max_tree_size,
            },
            "power_of_two" => PolicyConfig::PowerOfTwo {
                load_check_interval_secs: 5, // Default value
            },
            _ => PolicyConfig::RoundRobin, // Fallback
        }
    }

    /// Convert CLI arguments to RouterConfig
    fn to_router_config(&self) -> ConfigResult<RouterConfig> {
        // Determine routing mode
        let mode = if self.pd_disaggregation {
            let prefill_urls = self.parse_prefill_urls();
            let decode_urls = self.decode.clone();

            // Validate PD configuration if not using service discovery
            if !self.service_discovery && (prefill_urls.is_empty() || decode_urls.is_empty()) {
                return Err(ConfigError::ValidationFailed {
                    reason: "PD disaggregation mode requires --prefill and --decode URLs when not using service discovery".to_string(),
                });
            }

            RoutingMode::PrefillDecode {
                prefill_urls,
                decode_urls,
                prefill_policy: self.prefill_policy.as_ref().map(|p| self.parse_policy(p)),
                decode_policy: self.decode_policy.as_ref().map(|p| self.parse_policy(p)),
            }
        } else {
            // Regular mode
            if !self.service_discovery && self.worker_urls.is_empty() {
                return Err(ConfigError::ValidationFailed {
                    reason: "Regular mode requires --worker-urls when not using service discovery"
                        .to_string(),
                });
            }
            RoutingMode::Regular {
                worker_urls: self.worker_urls.clone(),
            }
        };

        // Main policy
        let policy = self.parse_policy(&self.policy);

        // Service discovery configuration
        let discovery = if self.service_discovery {
            Some(DiscoveryConfig {
                enabled: true,
                namespace: self.service_discovery_namespace.clone(),
                port: self.service_discovery_port,
                check_interval_secs: 60,
                selector: Self::parse_selector(&self.selector),
                prefill_selector: Self::parse_selector(&self.prefill_selector),
                decode_selector: Self::parse_selector(&self.decode_selector),
                bootstrap_port_annotation: "sglang.ai/bootstrap-port".to_string(),
            })
        } else {
            None
        };

        // Metrics configuration
        let metrics = Some(MetricsConfig {
            port: self.prometheus_port,
            host: self.prometheus_host.clone(),
        });

        // Build RouterConfig
        Ok(RouterConfig {
            mode,
            policy,
            host: self.host.clone(),
            port: self.port,
            max_payload_size: self.max_payload_size,
            request_timeout_secs: self.request_timeout_secs,
            worker_startup_timeout_secs: self.worker_startup_timeout_secs,
            worker_startup_check_interval_secs: self.worker_startup_check_interval,
            dp_aware: self.dp_aware,
            api_key: self.api_key.clone(),
            discovery,
            metrics,
            log_dir: self.log_dir.clone(),
            log_level: Some(self.log_level.clone()),
            request_id_headers: if self.request_id_headers.is_empty() {
                None
            } else {
                Some(self.request_id_headers.clone())
            },
            max_concurrent_requests: self.max_concurrent_requests,
            cors_allowed_origins: self.cors_allowed_origins.clone(),
            retry: RetryConfig {
                max_retries: self.retry_max_retries,
                initial_backoff_ms: self.retry_initial_backoff_ms,
                max_backoff_ms: self.retry_max_backoff_ms,
                backoff_multiplier: self.retry_backoff_multiplier,
                jitter_factor: self.retry_jitter_factor,
            },
            circuit_breaker: CircuitBreakerConfig {
                failure_threshold: self.cb_failure_threshold,
                success_threshold: self.cb_success_threshold,
                timeout_duration_secs: self.cb_timeout_duration_secs,
                window_duration_secs: self.cb_window_duration_secs,
            },
            disable_retries: self.disable_retries,
            disable_circuit_breaker: self.disable_circuit_breaker,
        })
    }

    /// Create ServerConfig from CLI args and RouterConfig
    fn to_server_config(&self, router_config: RouterConfig) -> ServerConfig {
        // Create service discovery config if enabled
        let service_discovery_config = if self.service_discovery {
            Some(ServiceDiscoveryConfig {
                enabled: true,
                selector: Self::parse_selector(&self.selector),
                check_interval: std::time::Duration::from_secs(60),
                port: self.service_discovery_port,
                namespace: self.service_discovery_namespace.clone(),
                pd_mode: self.pd_disaggregation,
                prefill_selector: Self::parse_selector(&self.prefill_selector),
                decode_selector: Self::parse_selector(&self.decode_selector),
                bootstrap_port_annotation: "sglang.ai/bootstrap-port".to_string(),
            })
        } else {
            None
        };

        // Create Prometheus config
        let prometheus_config = Some(PrometheusConfig {
            port: self.prometheus_port,
            host: self.prometheus_host.clone(),
        });

        ServerConfig {
            host: self.host.clone(),
            port: self.port,
            router_config,
            max_payload_size: self.max_payload_size,
            log_dir: self.log_dir.clone(),
            log_level: Some(self.log_level.clone()),
            service_discovery_config,
            prometheus_config,
            request_timeout_secs: self.request_timeout_secs,
            request_id_headers: if self.request_id_headers.is_empty() {
                None
            } else {
                Some(self.request_id_headers.clone())
            },
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse CLI arguments
    let cli_args = CliArgs::parse();

    // Print startup info
    println!("SGLang Router starting...");
    println!("Host: {}:{}", cli_args.host, cli_args.port);
    println!("Mode: {}", if cli_args.pd_disaggregation { "PD Disaggregated" } else { "Regular" });
    println!("Policy: {}", cli_args.policy);

    // Convert to RouterConfig
    let router_config = cli_args.to_router_config()?;

    // Validate configuration
    router_config.validate()?;

    // Create ServerConfig
    let server_config = cli_args.to_server_config(router_config);

    // Start server
    server::startup(server_config).await?;

    Ok(())
}