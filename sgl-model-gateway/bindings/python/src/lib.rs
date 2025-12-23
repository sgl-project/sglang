use pyo3::prelude::*;
use sgl_model_gateway::*;
use once_cell::sync::OnceCell;
use std::collections::HashMap;

// Define the enums with PyO3 bindings
#[pyclass(eq)]
#[derive(Clone, PartialEq, Debug)]
pub enum PolicyType {
    Random,
    RoundRobin,
    CacheAware,
    PowerOfTwo,
    Bucket,
}

#[pyclass(eq)]
#[derive(Clone, PartialEq, Debug)]
pub enum BackendType {
    Sglang,
    Openai,
}

#[pyclass(eq)]
#[derive(Clone, PartialEq, Debug)]
pub enum HistoryBackendType {
    Memory,
    None,
    Oracle,
    Postgres,
}

#[pyclass]
#[derive(Clone, PartialEq)]
pub struct PyOracleConfig {
    #[pyo3(get, set)]
    pub wallet_path: Option<String>,
    #[pyo3(get, set)]
    pub connect_descriptor: Option<String>,
    #[pyo3(get, set)]
    pub username: Option<String>,
    #[pyo3(get, set)]
    pub password: Option<String>,
    #[pyo3(get, set)]
    pub pool_min: usize,
    #[pyo3(get, set)]
    pub pool_max: usize,
    #[pyo3(get, set)]
    pub pool_timeout_secs: u64,
}

impl std::fmt::Debug for PyOracleConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyOracleConfig")
            .field("wallet_path", &self.wallet_path)
            .field("connect_descriptor", &"<redacted>")
            .field("username", &self.username)
            .field("password", &"<redacted>")
            .field("pool_min", &self.pool_min)
            .field("pool_max", &self.pool_max)
            .field("pool_timeout_secs", &self.pool_timeout_secs)
            .finish()
    }
}

#[pymethods]
impl PyOracleConfig {
    #[new]
    #[pyo3(signature = (
        password = None,
        username = None,
        connect_descriptor = None,
        wallet_path = None,
        pool_min = 1,
        pool_max = 16,
        pool_timeout_secs = 30,
    ))]
    fn new(
        password: Option<String>,
        username: Option<String>,
        connect_descriptor: Option<String>,
        wallet_path: Option<String>,
        pool_min: usize,
        pool_max: usize,
        pool_timeout_secs: u64,
    ) -> PyResult<Self> {
        if pool_min == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pool_min must be at least 1",
            ));
        }
        if pool_max < pool_min {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pool_max must be >= pool_min",
            ));
        }

        Ok(PyOracleConfig {
            wallet_path,
            connect_descriptor,
            username,
            password,
            pool_min,
            pool_max,
            pool_timeout_secs,
        })
    }
}

impl PyOracleConfig {
    pub fn to_config_oracle(&self) -> config::OracleConfig {
        config::OracleConfig {
            wallet_path: self.wallet_path.clone(),
            connect_descriptor: self.connect_descriptor.clone().unwrap_or_default(),
            username: self.username.clone().unwrap_or_default(),
            password: self.password.clone().unwrap_or_default(),
            pool_min: self.pool_min,
            pool_max: self.pool_max,
            pool_timeout_secs: self.pool_timeout_secs,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone, PartialEq)]
pub struct PyPostgresConfig {
    #[pyo3(get, set)]
    pub db_url: Option<String>,

    #[pyo3(get, set)]
    pub pool_max: usize,
}

#[pymethods]
impl PyPostgresConfig {
    #[new]
    #[pyo3(signature = (db_url = None,pool_max = 16,))]
    fn new(db_url: Option<String>, pool_max: usize) -> PyResult<Self> {
        Ok(PyPostgresConfig { db_url, pool_max })
    }
}

impl PyPostgresConfig {
    pub fn to_config_postgres(&self) -> config::PostgresConfig {
        config::PostgresConfig {
            db_url: self.db_url.clone().unwrap_or_default(),
            pool_max: self.pool_max,
        }
    }
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
    worker_load_check_interval: u64,
    cache_threshold: f32,
    balance_abs_threshold: usize,
    balance_rel_threshold: f32,
    eviction_interval_secs: u64,
    max_tree_size: usize,
    max_payload_size: usize,
    dp_aware: bool,
    dp_minimum_tokens_scheduler: bool,
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
    prometheus_duration_buckets: Option<Vec<f64>>,
    request_timeout_secs: u64,
    shutdown_grace_period_secs: u64,
    request_id_headers: Option<Vec<String>>,
    pd_disaggregation: bool,
    bucket_adjust_interval_secs: usize,
    prefill_urls: Option<Vec<(String, Option<u16>)>>,
    decode_urls: Option<Vec<String>>,
    prefill_policy: Option<PolicyType>,
    decode_policy: Option<PolicyType>,
    max_concurrent_requests: i32,
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
    rate_limit_tokens_per_second: Option<i32>,
    connection_mode: core::ConnectionMode,
    model_path: Option<String>,
    tokenizer_path: Option<String>,
    chat_template: Option<String>,
    tokenizer_cache_enable_l0: bool,
    tokenizer_cache_l0_max_entries: usize,
    tokenizer_cache_enable_l1: bool,
    tokenizer_cache_l1_max_memory: usize,
    reasoning_parser: Option<String>,
    tool_call_parser: Option<String>,
    mcp_config_path: Option<String>,
    backend: BackendType,
    history_backend: HistoryBackendType,
    oracle_config: Option<PyOracleConfig>,
    postgres_config: Option<PyPostgresConfig>,
    client_cert_path: Option<String>,
    client_key_path: Option<String>,
    ca_cert_paths: Vec<String>,
    server_cert_path: Option<String>,
    server_key_path: Option<String>,
    enable_trace: bool,
    otlp_traces_endpoint: String,
}

impl Router {
    fn determine_connection_mode(worker_urls: &[String]) -> core::ConnectionMode {
        for url in worker_urls {
            if url.starts_with("grpc://") || url.starts_with("grpcs://") {
                return core::ConnectionMode::Grpc { port: None };
            }
        }
        core::ConnectionMode::Http
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
                PolicyType::Bucket => ConfigPolicyConfig::Bucket {
                    balance_abs_threshold: self.balance_abs_threshold,
                    balance_rel_threshold: self.balance_rel_threshold,
                    bucket_adjust_interval_secs: self.bucket_adjust_interval_secs,
                },
            }
        };

        let mode = if self.enable_igw {
            RoutingMode::Regular {
                worker_urls: vec![],
            }
        } else if matches!(self.backend, BackendType::Openai) {
            RoutingMode::OpenAI {
                worker_urls: self.worker_urls.clone(),
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

        let trace_config = Some(config::TraceConfig {
            enable_trace: self.enable_trace,
            otlp_traces_endpoint: self.otlp_traces_endpoint.clone(),
        });

        let history_backend = match self.history_backend {
            HistoryBackendType::Memory => config::HistoryBackend::Memory,
            HistoryBackendType::None => config::HistoryBackend::None,
            HistoryBackendType::Oracle => config::HistoryBackend::Oracle,
            HistoryBackendType::Postgres => config::HistoryBackend::Postgres,
        };

        let oracle = if matches!(self.history_backend, HistoryBackendType::Oracle) {
            self.oracle_config
                .as_ref()
                .map(|cfg| cfg.to_config_oracle())
        } else {
            None
        };

        let postgres_config = if matches!(self.history_backend, HistoryBackendType::Postgres) {
            self.postgres_config
                .as_ref()
                .map(|cfg| cfg.to_config_postgres())
        } else {
            None
        };

        config::RouterConfig::builder()
            .mode(mode)
            .policy(policy)
            .host(&self.host)
            .port(self.port)
            .connection_mode(self.connection_mode.clone())
            .max_payload_size(self.max_payload_size)
            .request_timeout_secs(self.request_timeout_secs)
            .worker_startup_timeout_secs(self.worker_startup_timeout_secs)
            .worker_startup_check_interval_secs(self.worker_startup_check_interval)
            .worker_load_check_interval_secs(self.worker_load_check_interval)
            .max_concurrent_requests(self.max_concurrent_requests)
            .queue_size(self.queue_size)
            .queue_timeout_secs(self.queue_timeout_secs)
            .cors_allowed_origins(self.cors_allowed_origins.clone())
            .retry_config(config::RetryConfig {
                max_retries: self.retry_max_retries,
                initial_backoff_ms: self.retry_initial_backoff_ms,
                max_backoff_ms: self.retry_max_backoff_ms,
                backoff_multiplier: self.retry_backoff_multiplier,
                jitter_factor: self.retry_jitter_factor,
            })
            .circuit_breaker_config(config::CircuitBreakerConfig {
                failure_threshold: self.cb_failure_threshold,
                success_threshold: self.cb_success_threshold,
                timeout_duration_secs: self.cb_timeout_duration_secs,
                window_duration_secs: self.cb_window_duration_secs,
            })
            .health_check_config(config::HealthCheckConfig {
                failure_threshold: self.health_failure_threshold,
                success_threshold: self.health_success_threshold,
                timeout_secs: self.health_check_timeout_secs,
                check_interval_secs: self.health_check_interval_secs,
                endpoint: self.health_check_endpoint.clone(),
            })
            .tokenizer_cache(config::TokenizerCacheConfig {
                enable_l0: self.tokenizer_cache_enable_l0,
                l0_max_entries: self.tokenizer_cache_l0_max_entries,
                enable_l1: self.tokenizer_cache_enable_l1,
                l1_max_memory: self.tokenizer_cache_l1_max_memory,
            })
            .history_backend(history_backend)
            .maybe_api_key(self.api_key.as_ref())
            .maybe_discovery(discovery)
            .maybe_metrics(metrics)
            .maybe_trace(trace_config)
            .maybe_log_dir(self.log_dir.as_ref())
            .maybe_log_level(self.log_level.as_ref())
            .maybe_request_id_headers(self.request_id_headers.clone())
            .maybe_rate_limit_tokens_per_second(self.rate_limit_tokens_per_second)
            .maybe_model_path(self.model_path.as_ref())
            .maybe_tokenizer_path(self.tokenizer_path.as_ref())
            .maybe_chat_template(self.chat_template.as_ref())
            .maybe_oracle(oracle)
            .maybe_postgres(postgres_config)
            .maybe_reasoning_parser(self.reasoning_parser.as_ref())
            .maybe_tool_call_parser(self.tool_call_parser.as_ref())
            .maybe_mcp_config_path(self.mcp_config_path.as_ref())
            .dp_aware(self.dp_aware)
            .retries(!self.disable_retries)
            .circuit_breaker(!self.disable_circuit_breaker)
            .igw(self.enable_igw)
            .maybe_client_cert_and_key(
                self.client_cert_path.as_ref(),
                self.client_key_path.as_ref(),
            )
            .add_ca_certificates(self.ca_cert_paths.clone())
            .maybe_server_cert_and_key(
                self.server_cert_path.as_ref(),
                self.server_key_path.as_ref(),
            )
            .dp_minimum_tokens_scheduler(self.dp_minimum_tokens_scheduler)
            .build()
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
        worker_load_check_interval = 10,
        cache_threshold = 0.3,
        balance_abs_threshold = 64,
        balance_rel_threshold = 1.5,
        eviction_interval_secs = 120,
        max_tree_size = 2usize.pow(26),
        max_payload_size = 512 * 1024 * 1024,
        dp_aware = false,
        dp_minimum_tokens_scheduler = false,
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
        prometheus_duration_buckets = None,
        request_timeout_secs = 1800,
        shutdown_grace_period_secs = 180,
        request_id_headers = None,
        pd_disaggregation = false,
        bucket_adjust_interval_secs = 5,
        prefill_urls = None,
        decode_urls = None,
        prefill_policy = None,
        decode_policy = None,
        max_concurrent_requests = -1,
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
        chat_template = None,
        tokenizer_cache_enable_l0 = false,
        tokenizer_cache_l0_max_entries = 10000,
        tokenizer_cache_enable_l1 = false,
        tokenizer_cache_l1_max_memory = 52428800,
        reasoning_parser = None,
        tool_call_parser = None,
        mcp_config_path = None,
        backend = BackendType::Sglang,
        history_backend = HistoryBackendType::Memory,
        oracle_config = None,
        postgres_config = None,
        client_cert_path = None,
        client_key_path = None,
        ca_cert_paths = vec![],
        server_cert_path = None,
        server_key_path = None,
        enable_trace = false,
        otlp_traces_endpoint = String::from("localhost:4317"),
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        worker_urls: Vec<String>,
        policy: PolicyType,
        host: String,
        port: u16,
        worker_startup_timeout_secs: u64,
        worker_startup_check_interval: u64,
        worker_load_check_interval: u64,
        cache_threshold: f32,
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
        eviction_interval_secs: u64,
        max_tree_size: usize,
        max_payload_size: usize,
        dp_aware: bool,
        dp_minimum_tokens_scheduler: bool,
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
        prometheus_duration_buckets: Option<Vec<f64>>,
        request_timeout_secs: u64,
        shutdown_grace_period_secs: u64,
        request_id_headers: Option<Vec<String>>,
        pd_disaggregation: bool,
        bucket_adjust_interval_secs: usize,
        prefill_urls: Option<Vec<(String, Option<u16>)>>,
        decode_urls: Option<Vec<String>>,
        prefill_policy: Option<PolicyType>,
        decode_policy: Option<PolicyType>,
        max_concurrent_requests: i32,
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
        rate_limit_tokens_per_second: Option<i32>,
        model_path: Option<String>,
        tokenizer_path: Option<String>,
        chat_template: Option<String>,
        tokenizer_cache_enable_l0: bool,
        tokenizer_cache_l0_max_entries: usize,
        tokenizer_cache_enable_l1: bool,
        tokenizer_cache_l1_max_memory: usize,
        reasoning_parser: Option<String>,
        tool_call_parser: Option<String>,
        mcp_config_path: Option<String>,
        backend: BackendType,
        history_backend: HistoryBackendType,
        oracle_config: Option<PyOracleConfig>,
        postgres_config: Option<PyPostgresConfig>,
        client_cert_path: Option<String>,
        client_key_path: Option<String>,
        ca_cert_paths: Vec<String>,
        server_cert_path: Option<String>,
        server_key_path: Option<String>,
        enable_trace: bool,
        otlp_traces_endpoint: String,
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
            worker_load_check_interval,
            cache_threshold,
            balance_abs_threshold,
            balance_rel_threshold,
            eviction_interval_secs,
            max_tree_size,
            max_payload_size,
            dp_aware,
            dp_minimum_tokens_scheduler,
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
            prometheus_duration_buckets,
            request_timeout_secs,
            shutdown_grace_period_secs,
            request_id_headers,
            pd_disaggregation,
            bucket_adjust_interval_secs,
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
            chat_template,
            tokenizer_cache_enable_l0,
            tokenizer_cache_l0_max_entries,
            tokenizer_cache_enable_l1,
            tokenizer_cache_l1_max_memory,
            reasoning_parser,
            tool_call_parser,
            mcp_config_path,
            backend,
            history_backend,
            oracle_config,
            postgres_config,
            client_cert_path,
            client_key_path,
            ca_cert_paths,
            server_cert_path,
            server_key_path,
            enable_trace,
            otlp_traces_endpoint,
        })
    }

    fn start(&self) -> PyResult<()> {
        use observability::metrics::PrometheusConfig;

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
            duration_buckets: self.prometheus_duration_buckets.clone(),
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
                shutdown_grace_period_secs: self.shutdown_grace_period_secs,
            })
            .await
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }
}

/// Get simple version string (default for --version)
#[pyfunction]
fn get_version_string() -> String {
    version::get_version_string()
}

/// Get verbose version information string with full build details (for --version-verbose)
#[pyfunction]
fn get_verbose_version_string() -> String {
    version::get_verbose_version_string()
}

/// Get the list of available tool call parsers from the Rust factory.
#[pyfunction]
fn get_available_tool_call_parsers() -> Vec<String> {
    static PARSERS: OnceCell<Vec<String>> = OnceCell::new();
    PARSERS
        .get_or_init(|| {
            let factory = tool_parser::ParserFactory::new();
            factory.list_parsers()
        })
        .clone()
}

#[pymodule]
fn sglang_router_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PolicyType>()?;
    m.add_class::<BackendType>()?;
    m.add_class::<HistoryBackendType>()?;
    m.add_class::<PyOracleConfig>()?;
    m.add_class::<PyPostgresConfig>()?;
    m.add_class::<Router>()?;
    m.add_function(wrap_pyfunction!(get_version_string, m)?)?;
    m.add_function(wrap_pyfunction!(get_verbose_version_string, m)?)?;
    m.add_function(wrap_pyfunction!(get_available_tool_call_parsers, m)?)?;
    Ok(())
}
