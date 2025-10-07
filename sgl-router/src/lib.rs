use pyo3::prelude::*;
pub mod config;
pub mod logging;
use std::collections::HashMap;

pub mod core;
pub mod data_connector;
#[cfg(feature = "grpc-client")]
pub mod grpc_client;
pub mod mcp;
pub mod metrics;
pub mod middleware;
pub mod policies;
pub mod protocols;
pub mod reasoning_parser;
pub mod routers;
pub mod server;
pub mod service_discovery;
pub mod tokenizer;
pub mod tool_parser;
pub mod tree;
use crate::metrics::PrometheusConfig;

#[pyclass(eq)]
#[derive(Clone, PartialEq, Debug)]
pub enum PolicyType {
    Random,
    RoundRobin,
    CacheAware,
    PowerOfTwo,
}

#[pyclass]
#[derive(Debug, Clone, PartialEq)]
struct Router {
    host: String,
    port: u16,
    worker_urls: Vec<String>,
    policy: PolicyType,
    worker_startup_timeout_secs: u64,
    worker_startup_check_interval: u64,
    cache_threshold: f32,
    balance_abs_threshold: usize,
    balance_rel_threshold: f32,
    eviction_interval_secs: u64,
    max_tree_size: usize,
    max_payload_size: usize,
    dp_aware: bool,
    api_key: Option<String>,
    log_dir: Option<String>,
    log_level: Option<String>,
    service_discovery: bool,
    selector: HashMap<String, String>,
    service_discovery_port: u16,
    service_discovery_namespace: Option<String>,
    prefill_selector: HashMap<String, String>,
    decode_selector: HashMap<String, String>,
    bootstrap_port_annotation: String,
    prometheus_port: Option<u16>,
    prometheus_host: Option<String>,
    request_timeout_secs: u64,
    request_id_headers: Option<Vec<String>>,
    pd_disaggregation: bool,
    prefill_urls: Option<Vec<(String, Option<u16>)>>,
    decode_urls: Option<Vec<String>>,
    prefill_policy: Option<PolicyType>,
    decode_policy: Option<PolicyType>,
    max_concurrent_requests: usize,
    cors_allowed_origins: Vec<String>,
    retry_max_retries: u32,
    retry_initial_backoff_ms: u64,
    retry_max_backoff_ms: u64,
    retry_backoff_multiplier: f32,
    retry_jitter_factor: f32,
    disable_retries: bool,
    cb_failure_threshold: u32,
    cb_success_threshold: u32,
    cb_timeout_duration_secs: u64,
    cb_window_duration_secs: u64,
    disable_circuit_breaker: bool,
    health_failure_threshold: u32,
    health_success_threshold: u32,
    health_check_timeout_secs: u64,
    health_check_interval_secs: u64,
    health_check_endpoint: String,
    enable_igw: bool,
    queue_size: usize,
    queue_timeout_secs: u64,
    rate_limit_tokens_per_second: Option<usize>,
    connection_mode: config::ConnectionMode,
    model_path: Option<String>,
    tokenizer_path: Option<String>,
    reasoning_parser: Option<String>,
    tool_call_parser: Option<String>,
}

impl Router {
    /// Determine connection mode from worker URLs
    fn determine_connection_mode(worker_urls: &[String]) -> config::ConnectionMode {
        for url in worker_urls {
            if url.starts_with("grpc://") || url.starts_with("grpcs://") {
                return config::ConnectionMode::Grpc;
            }
        }
        config::ConnectionMode::Http
    }

    pub fn to_router_config(&self) -> config::ConfigResult<config::RouterConfig> {
        use config::{
            DiscoveryConfig, MetricsConfig, PolicyConfig as ConfigPolicyConfig, RoutingMode,
        };

        let convert_policy = |policy: &PolicyType| -> ConfigPolicyConfig {
            match policy {
                PolicyType::Random => ConfigPolicyConfig::Random,
                PolicyType::RoundRobin => ConfigPolicyConfig::RoundRobin,
                PolicyType::CacheAware => ConfigPolicyConfig::CacheAware {
                    cache_threshold: self.cache_threshold,
                    balance_abs_threshold: self.balance_abs_threshold,
                    balance_rel_threshold: self.balance_rel_threshold,
                    eviction_interval_secs: self.eviction_interval_secs,
                    max_tree_size: self.max_tree_size,
                },
                PolicyType::PowerOfTwo => ConfigPolicyConfig::PowerOfTwo {
                    load_check_interval_secs: 5,
                },
            }
        };

        let mode = if self.enable_igw {
            RoutingMode::Regular {
                worker_urls: vec![],
            }
        } else if self.pd_disaggregation {
            RoutingMode::PrefillDecode {
                prefill_urls: self.prefill_urls.clone().unwrap_or_default(),
                decode_urls: self.decode_urls.clone().unwrap_or_default(),
                prefill_policy: self.prefill_policy.as_ref().map(convert_policy),
                decode_policy: self.decode_policy.as_ref().map(convert_policy),
            }
        } else {
            RoutingMode::Regular {
                worker_urls: self.worker_urls.clone(),
            }
        };

        let policy = convert_policy(&self.policy);

        let discovery = if self.service_discovery {
            Some(DiscoveryConfig {
                enabled: true,
                namespace: self.service_discovery_namespace.clone(),
                port: self.service_discovery_port,
                check_interval_secs: 60,
                selector: self.selector.clone(),
                prefill_selector: self.prefill_selector.clone(),
                decode_selector: self.decode_selector.clone(),
                bootstrap_port_annotation: self.bootstrap_port_annotation.clone(),
            })
        } else {
            None
        };

        let metrics = match (self.prometheus_port, self.prometheus_host.as_ref()) {
            (Some(port), Some(host)) => Some(MetricsConfig {
                port,
                host: host.clone(),
            }),
            _ => None,
        };

        Ok(config::RouterConfig {
            mode,
            policy,
            host: self.host.clone(),
            port: self.port,
            connection_mode: self.connection_mode.clone(),
            max_payload_size: self.max_payload_size,
            request_timeout_secs: self.request_timeout_secs,
            worker_startup_timeout_secs: self.worker_startup_timeout_secs,
            worker_startup_check_interval_secs: self.worker_startup_check_interval,
            dp_aware: self.dp_aware,
            api_key: self.api_key.clone(),
            discovery,
            metrics,
            log_dir: self.log_dir.clone(),
            log_level: self.log_level.clone(),
            request_id_headers: self.request_id_headers.clone(),
            max_concurrent_requests: self.max_concurrent_requests,
            queue_size: self.queue_size,
            queue_timeout_secs: self.queue_timeout_secs,
            rate_limit_tokens_per_second: self.rate_limit_tokens_per_second,
            cors_allowed_origins: self.cors_allowed_origins.clone(),
            retry: config::RetryConfig {
                max_retries: self.retry_max_retries,
                initial_backoff_ms: self.retry_initial_backoff_ms,
                max_backoff_ms: self.retry_max_backoff_ms,
                backoff_multiplier: self.retry_backoff_multiplier,
                jitter_factor: self.retry_jitter_factor,
            },
            circuit_breaker: config::CircuitBreakerConfig {
                failure_threshold: self.cb_failure_threshold,
                success_threshold: self.cb_success_threshold,
                timeout_duration_secs: self.cb_timeout_duration_secs,
                window_duration_secs: self.cb_window_duration_secs,
            },
            disable_retries: self.disable_retries,
            disable_circuit_breaker: self.disable_circuit_breaker,
            health_check: config::HealthCheckConfig {
                failure_threshold: self.health_failure_threshold,
                success_threshold: self.health_success_threshold,
                timeout_secs: self.health_check_timeout_secs,
                check_interval_secs: self.health_check_interval_secs,
                endpoint: self.health_check_endpoint.clone(),
            },
            enable_igw: self.enable_igw,
            model_path: self.model_path.clone(),
            tokenizer_path: self.tokenizer_path.clone(),
            history_backend: config::HistoryBackend::Memory,
            oracle: None,
            reasoning_parser: self.reasoning_parser.clone(),
            tool_call_parser: self.tool_call_parser.clone(),
        })
    }
}

#[pymethods]
impl Router {
    #[new]
    #[pyo3(signature = (
        worker_urls,
        policy = PolicyType::RoundRobin,
        host = String::from("0.0.0.0"),
        port = 3001,
        worker_startup_timeout_secs = 600,
        worker_startup_check_interval = 30,
        cache_threshold = 0.3,
        balance_abs_threshold = 64,
        balance_rel_threshold = 1.5,
        eviction_interval_secs = 120,
        max_tree_size = 2usize.pow(26),
        max_payload_size = 512 * 1024 * 1024,
        dp_aware = false,
        api_key = None,
        log_dir = None,
        log_level = None,
        service_discovery = false,
        selector = HashMap::new(),
        service_discovery_port = 80,
        service_discovery_namespace = None,
        prefill_selector = HashMap::new(),
        decode_selector = HashMap::new(),
        bootstrap_port_annotation = String::from("sglang.ai/bootstrap-port"),
        prometheus_port = None,
        prometheus_host = None,
        request_timeout_secs = 1800,
        request_id_headers = None,
        pd_disaggregation = false,
        prefill_urls = None,
        decode_urls = None,
        prefill_policy = None,
        decode_policy = None,
        max_concurrent_requests = 256,
        cors_allowed_origins = vec![],
        retry_max_retries = 5,
        retry_initial_backoff_ms = 50,
        retry_max_backoff_ms = 30_000,
        retry_backoff_multiplier = 1.5,
        retry_jitter_factor = 0.2,
        disable_retries = false,
        cb_failure_threshold = 10,
        cb_success_threshold = 3,
        cb_timeout_duration_secs = 60,
        cb_window_duration_secs = 120,
        disable_circuit_breaker = false,
        health_failure_threshold = 3,
        health_success_threshold = 2,
        health_check_timeout_secs = 5,
        health_check_interval_secs = 60,
        health_check_endpoint = String::from("/health"),
        enable_igw = false,
        queue_size = 100,
        queue_timeout_secs = 60,
        rate_limit_tokens_per_second = None,
        model_path = None,
        tokenizer_path = None,
        reasoning_parser = None,
        tool_call_parser = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        worker_urls: Vec<String>,
        policy: PolicyType,
        host: String,
        port: u16,
        worker_startup_timeout_secs: u64,
        worker_startup_check_interval: u64,
        cache_threshold: f32,
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
        eviction_interval_secs: u64,
        max_tree_size: usize,
        max_payload_size: usize,
        dp_aware: bool,
        api_key: Option<String>,
        log_dir: Option<String>,
        log_level: Option<String>,
        service_discovery: bool,
        selector: HashMap<String, String>,
        service_discovery_port: u16,
        service_discovery_namespace: Option<String>,
        prefill_selector: HashMap<String, String>,
        decode_selector: HashMap<String, String>,
        bootstrap_port_annotation: String,
        prometheus_port: Option<u16>,
        prometheus_host: Option<String>,
        request_timeout_secs: u64,
        request_id_headers: Option<Vec<String>>,
        pd_disaggregation: bool,
        prefill_urls: Option<Vec<(String, Option<u16>)>>,
        decode_urls: Option<Vec<String>>,
        prefill_policy: Option<PolicyType>,
        decode_policy: Option<PolicyType>,
        max_concurrent_requests: usize,
        cors_allowed_origins: Vec<String>,
        retry_max_retries: u32,
        retry_initial_backoff_ms: u64,
        retry_max_backoff_ms: u64,
        retry_backoff_multiplier: f32,
        retry_jitter_factor: f32,
        disable_retries: bool,
        cb_failure_threshold: u32,
        cb_success_threshold: u32,
        cb_timeout_duration_secs: u64,
        cb_window_duration_secs: u64,
        disable_circuit_breaker: bool,
        health_failure_threshold: u32,
        health_success_threshold: u32,
        health_check_timeout_secs: u64,
        health_check_interval_secs: u64,
        health_check_endpoint: String,
        enable_igw: bool,
        queue_size: usize,
        queue_timeout_secs: u64,
        rate_limit_tokens_per_second: Option<usize>,
        model_path: Option<String>,
        tokenizer_path: Option<String>,
        reasoning_parser: Option<String>,
        tool_call_parser: Option<String>,
    ) -> PyResult<Self> {
        let mut all_urls = worker_urls.clone();

        if let Some(ref prefill_urls) = prefill_urls {
            for (url, _) in prefill_urls {
                all_urls.push(url.clone());
            }
        }

        if let Some(ref decode_urls) = decode_urls {
            all_urls.extend(decode_urls.clone());
        }

        let connection_mode = Self::determine_connection_mode(&all_urls);

        Ok(Router {
            host,
            port,
            worker_urls,
            policy,
            worker_startup_timeout_secs,
            worker_startup_check_interval,
            cache_threshold,
            balance_abs_threshold,
            balance_rel_threshold,
            eviction_interval_secs,
            max_tree_size,
            max_payload_size,
            dp_aware,
            api_key,
            log_dir,
            log_level,
            service_discovery,
            selector,
            service_discovery_port,
            service_discovery_namespace,
            prefill_selector,
            decode_selector,
            bootstrap_port_annotation,
            prometheus_port,
            prometheus_host,
            request_timeout_secs,
            request_id_headers,
            pd_disaggregation,
            prefill_urls,
            decode_urls,
            prefill_policy,
            decode_policy,
            max_concurrent_requests,
            cors_allowed_origins,
            retry_max_retries,
            retry_initial_backoff_ms,
            retry_max_backoff_ms,
            retry_backoff_multiplier,
            retry_jitter_factor,
            disable_retries,
            cb_failure_threshold,
            cb_success_threshold,
            cb_timeout_duration_secs,
            cb_window_duration_secs,
            disable_circuit_breaker,
            health_failure_threshold,
            health_success_threshold,
            health_check_timeout_secs,
            health_check_interval_secs,
            health_check_endpoint,
            enable_igw,
            queue_size,
            queue_timeout_secs,
            rate_limit_tokens_per_second,
            connection_mode,
            model_path,
            tokenizer_path,
            reasoning_parser,
            tool_call_parser,
        })
    }

    fn start(&self) -> PyResult<()> {
        let router_config = self.to_router_config().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Configuration error: {}", e))
        })?;

        router_config.validate().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Configuration validation failed: {}",
                e
            ))
        })?;

        let service_discovery_config = if self.service_discovery {
            Some(service_discovery::ServiceDiscoveryConfig {
                enabled: true,
                selector: self.selector.clone(),
                check_interval: std::time::Duration::from_secs(60),
                port: self.service_discovery_port,
                namespace: self.service_discovery_namespace.clone(),
                pd_mode: self.pd_disaggregation,
                prefill_selector: self.prefill_selector.clone(),
                decode_selector: self.decode_selector.clone(),
                bootstrap_port_annotation: self.bootstrap_port_annotation.clone(),
            })
        } else {
            None
        };

        let prometheus_config = Some(PrometheusConfig {
            port: self.prometheus_port.unwrap_or(29000),
            host: self
                .prometheus_host
                .clone()
                .unwrap_or_else(|| "127.0.0.1".to_string()),
        });

        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        runtime.block_on(async move {
            server::startup(server::ServerConfig {
                host: self.host.clone(),
                port: self.port,
                router_config,
                max_payload_size: self.max_payload_size,
                log_dir: self.log_dir.clone(),
                log_level: self.log_level.clone(),
                service_discovery_config,
                prometheus_config,
                request_timeout_secs: self.request_timeout_secs,
                request_id_headers: self.request_id_headers.clone(),
            })
            .await
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }
}

#[pymodule]
fn sglang_router_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PolicyType>()?;
    m.add_class::<Router>()?;
    Ok(())
}
