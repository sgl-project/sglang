use pyo3::prelude::*;
pub mod logging;
use std::collections::HashMap;
pub mod prometheus;
pub mod router;
pub mod server;
pub mod service_discovery;
pub mod tree;
use crate::prometheus::PrometheusConfig;

use bytes::Bytes;
use pyo3::types::PyBytes;

#[pyclass]
struct InnerRouter {
    inner: router::Router,
}

#[pymethods]
impl InnerRouter {
    fn select_generate_worker<'p>(&self, py: Python<'p>, body: &[u8], route: &str) -> PyResult<String> {
        let py_data = PyBytes::new(py, body);
        Ok(self.inner.select_generate_worker(&Bytes::copy_from_slice(py_data.as_bytes()), route))
    }
}

#[pyclass(eq)]
#[derive(Clone, PartialEq, Debug)]
pub enum PolicyType {
    Random,
    RoundRobin,
    CacheAware,
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
        })
    }

    fn create_innerrouter<'p>(&self) -> PyResult<InnerRouter> {
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
        };
        let inner = router::Router::new(self.worker_urls.clone(), policy_config).unwrap();
        Ok(InnerRouter {
            inner,
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
    m.add_class::<InnerRouter>()?;
    Ok(())
}
