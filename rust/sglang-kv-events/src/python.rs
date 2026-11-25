//! PyO3 bindings: `sglang.srt.rust_extensions._kv_events`.

use bytes::Bytes;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use tracing_subscriber::EnvFilter;

use crate::publisher::{Publisher, PublisherConfig, PublisherError};

fn to_py_err(error: PublisherError) -> PyErr {
    match error {
        PublisherError::UnsupportedEndpoint(_) => PyValueError::new_err(error.to_string()),
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}

#[pyclass(
    name = "Publisher",
    module = "sglang.srt.rust_extensions._kv_events",
    frozen
)]
struct PyPublisher {
    inner: Publisher,
}

#[pymethods]
impl PyPublisher {
    #[new]
    #[pyo3(signature = (*, endpoint, replay_endpoint, topic, buffer_steps, max_queue_size, hwm))]
    fn new(
        py: Python<'_>,
        endpoint: String,
        replay_endpoint: Option<String>,
        topic: String,
        buffer_steps: usize,
        max_queue_size: usize,
        hwm: usize,
    ) -> PyResult<Self> {
        let config = PublisherConfig {
            endpoint,
            replay_endpoint,
            topic,
            buffer_steps,
            max_queue_size,
            hwm,
        };
        let inner = py.detach(|| Publisher::start(config)).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Queue one encoded batch, waiting while the queue is full.
    fn publish(&self, py: Python<'_>, payload: &[u8]) -> PyResult<()> {
        let payload = Bytes::copy_from_slice(payload);
        py.detach(|| self.inner.publish(payload)).map_err(to_py_err)
    }

    #[getter]
    fn local_pub_endpoint(&self) -> &str {
        self.inner.local_pub_endpoint()
    }

    #[getter]
    fn local_replay_endpoint(&self) -> Option<&str> {
        self.inner.local_replay_endpoint()
    }

    fn shutdown(&self, py: Python<'_>) {
        py.detach(|| self.inner.shutdown());
    }
}

#[pymodule]
fn _kv_events(module: &Bound<'_, PyModule>) -> PyResult<()> {
    // This extension has its own tracing dispatcher: without a subscriber the
    // publisher thread's warnings reach nobody.
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(filter)
        .try_init();
    module.add_class::<PyPublisher>()
}
