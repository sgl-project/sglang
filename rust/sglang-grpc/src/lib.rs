use pyo3::prelude::*;
use std::sync::Arc;
use tokio::sync::Notify;

pub mod proto {
    tonic::include_proto!("sglang.runtime.v1");
}

/// Handle returned by `start_server` — used to shut down the gRPC server.
#[pyclass]
pub struct GrpcServerHandle {
    shutdown: Arc<Notify>,
    join_handle: Option<std::thread::JoinHandle<()>>,
}

#[pymethods]
impl GrpcServerHandle {
    /// Signal the server to stop and wait for the background thread to exit.
    fn shutdown(&mut self) {
        self.shutdown.notify_one();
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }
    }

    /// Returns `true` while the server thread is still running.
    fn is_alive(&self) -> bool {
        self.join_handle
            .as_ref()
            .map_or(false, |h| !h.is_finished())
    }
}

/// Start the gRPC server in a background thread.
///
/// * `host` – bind address (e.g. "0.0.0.0")
/// * `port` – listen port
/// * `runtime_handle` – Python `RuntimeHandle` object (from `grpc_bridge.py`)
///
/// Returns a `GrpcServerHandle` that can be used to shut the server down.
#[pyfunction]
fn start_server(host: String, port: u16, runtime_handle: PyObject) -> PyResult<GrpcServerHandle> {
    let _ = &runtime_handle; // Will be used in Phase 1 PR 2
    let shutdown = Arc::new(Notify::new());
    let shutdown_clone = shutdown.clone();

    let addr_str = format!("{}:{}", host, port);
    let addr: std::net::SocketAddr = addr_str
        .parse()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Bad address: {e}")))?;

    let join_handle = std::thread::Builder::new()
        .name("grpc-server".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(4)
                .enable_all()
                .build()
                .expect("Failed to build Tokio runtime");

            rt.block_on(async move {
                tracing::info!("gRPC server listening on {}", addr);
                // Server implementation will be added in PR 2.
                // For now, just wait for shutdown signal.
                shutdown_clone.notified().await;
                tracing::info!("gRPC server shutting down");
            });
        })
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to spawn thread: {e}"))
        })?;

    Ok(GrpcServerHandle {
        shutdown,
        join_handle: Some(join_handle),
    })
}

/// Python module exported by the Rust extension.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(start_server, m)?)?;
    m.add_class::<GrpcServerHandle>()?;
    Ok(())
}
