// Python Binding
use pyo3::prelude::*;
pub mod router;
mod server;
pub mod tree;

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
    tokenizer_path: Option<String>,
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
        tokenizer_path = None,
        cache_threshold = Some(0.50)
    ))]
    fn new(
        worker_urls: Vec<String>,
        policy: PolicyType,
        host: String,
        port: u16,
        tokenizer_path: Option<String>,
        cache_threshold: Option<f32>,
    ) -> PyResult<Self> {
        // Validate required parameters for approx_tree policy
        if matches!(policy, PolicyType::ApproxTree) {
            if tokenizer_path.is_none() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "tokenizer_path is required for approx_tree policy",
                ));
            }
        }

        Ok(Router {
            host,
            port,
            worker_urls,
            policy,
            tokenizer_path,
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
                tokenizer_path: self
                    .tokenizer_path
                    .clone()
                    .expect("tokenizer_path is required for approx_tree policy"),
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
