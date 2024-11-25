use pyo3::prelude::*;
pub mod router;
pub mod server;
pub mod tree;

#[pyclass(eq)]
#[derive(Clone, PartialEq)]
pub enum PolicyType {
    Random,
    RoundRobin,
    CacheAware,
}

#[pyclass]
struct Router {
    host: String,
    port: u16,
    worker_urls: Vec<String>,
    policy: PolicyType,
    cache_threshold: f32,
    balance_abs_threshold: usize,
    balance_rel_threshold: f32,
    eviction_interval_secs: u64,
    max_tree_size: usize,
    verbose: bool,
}

#[pymethods]
impl Router {
    #[new]
    #[pyo3(signature = (
        worker_urls,
        policy = PolicyType::RoundRobin,
        host = String::from("127.0.0.1"),
        port = 3001,
        cache_threshold = 0.50,
        balance_abs_threshold = 32,
        balance_rel_threshold = 1.0001,
        eviction_interval_secs = 60,
        max_tree_size = 2usize.pow(24),
        verbose = false
    ))]
    fn new(
        worker_urls: Vec<String>,
        policy: PolicyType,
        host: String,
        port: u16,
        cache_threshold: f32,
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
        eviction_interval_secs: u64,
        max_tree_size: usize,
        verbose: bool,
    ) -> PyResult<Self> {
        Ok(Router {
            host,
            port,
            worker_urls,
            policy,
            cache_threshold,
            balance_abs_threshold,
            balance_rel_threshold,
            eviction_interval_secs,
            max_tree_size,
            verbose,
        })
    }

    fn start(&self) -> PyResult<()> {
        let policy_config = match &self.policy {
            PolicyType::Random => router::PolicyConfig::RandomConfig,
            PolicyType::RoundRobin => router::PolicyConfig::RoundRobinConfig,
            PolicyType::CacheAware => router::PolicyConfig::CacheAwareConfig {
                cache_threshold: self.cache_threshold,
                balance_abs_threshold: self.balance_abs_threshold,
                balance_rel_threshold: self.balance_rel_threshold,
                eviction_interval_secs: self.eviction_interval_secs,
                max_tree_size: self.max_tree_size,
            },
        };

        actix_web::rt::System::new().block_on(async move {
            server::startup(server::ServerConfig {
                host: self.host.clone(),
                port: self.port,
                worker_urls: self.worker_urls.clone(),
                policy_config,
                verbose: self.verbose,
            })
            .await
            .unwrap();
        });

        Ok(())
    }
}

#[pymodule]
fn sglang_router_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PolicyType>()?;
    m.add_class::<Router>()?;
    Ok(())
}
