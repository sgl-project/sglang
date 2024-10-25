use pyo3::prelude::*;
mod server;
pub mod router;

// Python binding
#[pyclass]
struct Router {
    host: String,
    port: u16,
    worker_urls: Vec<String>,
    policy: String
}

#[pymethods]
impl Router {
    #[new]
    fn new(host: String, port: u16, worker_urls: Vec<String>, policy: String) -> Self {
        Router {
            host,
            port,
            worker_urls,
            policy
        }
    }

    fn start(&self) -> PyResult<()> {
        let host = self.host.clone();
        let port = self.port;
        let worker_urls = self.worker_urls.clone();
        let policy = self.policy.clone();

        actix_web::rt::System::new().block_on(async move {
            server::startup(host, port, worker_urls, policy).await.unwrap();
        });

        Ok(())
    }
}

// python usage: `from sglang_router import Router`
#[pymodule]
fn sglang_router(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Router>()?;
    Ok(())
}