// Python Binding
use pyo3::prelude::*;
pub mod router;
pub mod server;
pub mod tree;
pub mod multi_tenant_tree;
pub mod multi_tenant_tree_single;

#[pyclass(eq)]
#[derive(Clone, PartialEq)]
pub enum PolicyType {
    Random,
    RoundRobin,
    ApproxTree,
}

#[pyclass]
struct Router {
    host: String,
    port: u16,
    worker_urls: Vec<String>,
    policy: PolicyType,
    cache_threshold: Option<f32>,
}

#[pymethods]
impl Router {
    #[new]
    #[pyo3(signature = (
        worker_urls,
        policy = PolicyType::RoundRobin,
        host = String::from("127.0.0.1"),
        port = 3001,
        cache_threshold = Some(0.50)
    ))]
    fn new(
        worker_urls: Vec<String>,
        policy: PolicyType,
        host: String,
        port: u16,
        cache_threshold: Option<f32>,
    ) -> PyResult<Self> {

        Ok(Router {
            host,
            port,
            worker_urls,
            policy,
            cache_threshold,
        })
    }

    fn start(&self) -> PyResult<()> {
        let host = self.host.clone();
        let port = self.port;
        let worker_urls = self.worker_urls.clone();

        let policy_config = match &self.policy {
            PolicyType::Random => router::PolicyConfig::RandomConfig,
            PolicyType::RoundRobin => router::PolicyConfig::RoundRobinConfig,
            PolicyType::ApproxTree => router::PolicyConfig::ApproxTreeConfig {
                cache_threshold: self
                    .cache_threshold
                    .expect("cache_threshold is required for approx_tree policy"),
            },
        };

        actix_web::rt::System::new().block_on(async move {
            server::startup(host, port, worker_urls, policy_config)
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
