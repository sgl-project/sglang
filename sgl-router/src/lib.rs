use pyo3::prelude::*;
pub mod logging;
use std::collections::HashMap;
pub mod prometheus;
pub mod router;
pub mod server;
pub mod service_discovery;
pub mod tree;
pub mod pd_types;
use crate::prometheus::PrometheusConfig;

#[pyclass(eq)]
#[derive(Clone, PartialEq, Debug)]
pub enum PolicyType {
    Random,
    RoundRobin,
    CacheAware,
    PrefillDecode,
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
    verbose: bool,
    log_dir: Option<String>,
    service_discovery: bool,
    selector: HashMap<String, String>,
    service_discovery_port: u16,
    service_discovery_namespace: Option<String>,
    prometheus_port: Option<u16>,
    prometheus_host: Option<String>,
    prefill_urls: Option<Vec<String>>,
    decode_urls: Option<Vec<String>>,
    pd_policy: Option<String>,
}

#[pymethods]
impl Router {
    #[new]
    #[pyo3(signature = (
        worker_urls,
        policy = PolicyType::RoundRobin,
        host = String::from("127.0.0.1"),
        port = 3001,
        worker_startup_timeout_secs = 300,
        worker_startup_check_interval = 10,
        cache_threshold = 0.50,
        balance_abs_threshold = 32,
        balance_rel_threshold = 1.0001,
        eviction_interval_secs = 60,
        max_tree_size = 2usize.pow(24),
        max_payload_size = 4 * 1024 * 1024,
        verbose = false,
        log_dir = None,
        service_discovery = false,
        selector = HashMap::new(),
        service_discovery_port = 80,
        service_discovery_namespace = None,
        prometheus_port = None,
        prometheus_host = None
    ))]
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
        verbose: bool,
        log_dir: Option<String>,
        service_discovery: bool,
        selector: HashMap<String, String>,
        service_discovery_port: u16,
        service_discovery_namespace: Option<String>,
        prometheus_port: Option<u16>,
        prometheus_host: Option<String>,
    ) -> PyResult<Self> {
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
            verbose,
            log_dir,
            service_discovery,
            selector,
            service_discovery_port,
            service_discovery_namespace,
            prometheus_port,
            prometheus_host,
            prefill_urls: None,
            decode_urls: None,
            pd_policy: None,
        })
    }

    /// Prefill-Decode router constructor
    #[staticmethod]
    #[pyo3(signature = (
        prefill_urls,
        decode_urls,
        pd_policy = String::from("random"),
        host = String::from("127.0.0.1"),
        port = 3001,
        worker_startup_timeout_secs = 300,
        worker_startup_check_interval = 10,
        cache_threshold = 0.50,
        balance_abs_threshold = 32,
        balance_rel_threshold = 1.0001,
        max_payload_size = 4 * 1024 * 1024,
        verbose = false,
        log_dir = None,
        prometheus_port = None,
        prometheus_host = None
    ))]
    fn new_pd(
        prefill_urls: Vec<String>,
        decode_urls: Vec<String>,
        pd_policy: String,
        host: String,
        port: u16,
        worker_startup_timeout_secs: u64,
        worker_startup_check_interval: u64,
        cache_threshold: f32,
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
        max_payload_size: usize,
        verbose: bool,
        log_dir: Option<String>,
        prometheus_port: Option<u16>,
        prometheus_host: Option<String>,
    ) -> PyResult<Self> {
        let mut worker_urls = prefill_urls.clone();
        worker_urls.extend(decode_urls.clone());

        Ok(Router {
            host,
            port,
            worker_urls,
            policy: PolicyType::PrefillDecode,
            worker_startup_timeout_secs,
            worker_startup_check_interval,
            cache_threshold,
            balance_abs_threshold,
            balance_rel_threshold,
            eviction_interval_secs: 60, // default
            max_tree_size: 2usize.pow(24),
            max_payload_size,
            verbose,
            log_dir,
            service_discovery: false,
            selector: HashMap::new(),
            service_discovery_port: 80,
            service_discovery_namespace: None,
            prometheus_port,
            prometheus_host,
            prefill_urls: Some(prefill_urls),
            decode_urls: Some(decode_urls),
            pd_policy: Some(pd_policy),
        })
    }

    fn start(&self) -> PyResult<()> {
        let policy_config = match &self.policy {
            PolicyType::Random => router::PolicyConfig::RandomConfig {
                timeout_secs: self.worker_startup_timeout_secs,
                interval_secs: self.worker_startup_check_interval,
            },
            PolicyType::RoundRobin => router::PolicyConfig::RoundRobinConfig {
                timeout_secs: self.worker_startup_timeout_secs,
                interval_secs: self.worker_startup_check_interval,
            },
            PolicyType::CacheAware => router::PolicyConfig::CacheAwareConfig {
                timeout_secs: self.worker_startup_timeout_secs,
                interval_secs: self.worker_startup_check_interval,
                cache_threshold: self.cache_threshold,
                balance_abs_threshold: self.balance_abs_threshold,
                balance_rel_threshold: self.balance_rel_threshold,
                eviction_interval_secs: self.eviction_interval_secs,
                max_tree_size: self.max_tree_size,
            },
            PolicyType::PrefillDecode => {
                use crate::pd_types::{EngineInfo, EngineType, PDSelectionPolicy};

                let prefill_urls = self.prefill_urls.clone().unwrap_or_default();
                let decode_urls = self.decode_urls.clone().unwrap_or_default();

                let prefill_workers: Vec<EngineInfo> = prefill_urls
                    .iter()
                    .map(|url| EngineInfo {
                        engine_type: EngineType::Sglang,
                        url: url.clone(),
                        bootstrap_port: None,
                    })
                    .collect();

                let decode_workers: Vec<EngineInfo> = decode_urls
                    .iter()
                    .map(|url| EngineInfo {
                        engine_type: EngineType::Sglang,
                        url: url.clone(),
                        bootstrap_port: None,
                    })
                    .collect();

                let selection_policy = match self.pd_policy.as_deref().unwrap_or("random") {
                    "random" => PDSelectionPolicy::Random,
                    "po2" | "poweroftwo" => PDSelectionPolicy::PowerOfTwo,
                    "cache_aware" | "cacheaware" => PDSelectionPolicy::CacheAware {
                        cache_threshold: self.cache_threshold,
                        balance_abs_threshold: self.balance_abs_threshold,
                        balance_rel_threshold: self.balance_rel_threshold,
                    },
                    _ => PDSelectionPolicy::Random,
                };

                router::PolicyConfig::PrefillDecodeConfig {
                    prefill_workers,
                    decode_workers,
                    selection_policy,
                    timeout_secs: self.worker_startup_timeout_secs,
                    interval_secs: self.worker_startup_check_interval,
                }
            }
        };

        // Create service discovery config if enabled
        let service_discovery_config = if self.service_discovery {
            Some(service_discovery::ServiceDiscoveryConfig {
                enabled: true,
                selector: self.selector.clone(),
                check_interval: std::time::Duration::from_secs(60),
                port: self.service_discovery_port,
                namespace: self.service_discovery_namespace.clone(),
            })
        } else {
            None
        };

        // Create Prometheus config if enabled
        let prometheus_config = Some(PrometheusConfig {
            port: self.prometheus_port.unwrap_or(29000),
            host: self
                .prometheus_host
                .clone()
                .unwrap_or_else(|| "127.0.0.1".to_string()),
        });

        actix_web::rt::System::new().block_on(async move {
            server::startup(server::ServerConfig {
                host: self.host.clone(),
                port: self.port,
                worker_urls: self.worker_urls.clone(),
                policy_config,
                verbose: self.verbose,
                max_payload_size: self.max_payload_size,
                log_dir: self.log_dir.clone(),
                service_discovery_config,
                prometheus_config,
            })
            .await
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(())
        })
    }
}

#[pymodule]
fn sglang_router_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PolicyType>()?;
    m.add_class::<Router>()?;
    Ok(())
}
