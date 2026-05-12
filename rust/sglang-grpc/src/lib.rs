pub mod bridge;
pub mod server;
pub mod tokenizers;
pub mod utils;

pub mod proto {
    tonic::include_proto!("sglang.runtime.v1");
}

use pyo3::prelude::*;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Notify;

use bridge::PyBridge;
use tokenizers::RustTokenizer;

/// Handle returned to Python that controls the running gRPC server.
#[pyclass]
struct GrpcServerHandle {
    shutdown: Arc<Notify>,
    join_handle: Option<std::thread::JoinHandle<()>>,
}

#[pymethods]
impl GrpcServerHandle {
    /// Gracefully shut down the gRPC server.
    fn shutdown(&mut self) {
        self.shutdown.notify_one();
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }
    }

    /// Check if the server thread is still running.
    fn is_alive(&self) -> bool {
        self.join_handle
            .as_ref()
            .map_or(false, |h| !h.is_finished())
    }
}

struct TokenizerInfo {
    tokenizer_path: Option<String>,
    tokenizer_mode: Option<String>,
    context_len: i32,
}

/// Extract tokenizer path/mode and context_len from the Python RuntimeHandle (one-time GIL).
fn extract_tokenizer_info(runtime_handle: &PyObject) -> TokenizerInfo {
    Python::with_gil(|py| {
        let tm = match runtime_handle.getattr(py, "tokenizer_manager") {
            Ok(tm) => tm,
            Err(_) => {
                return TokenizerInfo {
                    tokenizer_path: None,
                    tokenizer_mode: None,
                    context_len: 0,
                };
            }
        };

        let server_args = tm.getattr(py, "server_args").ok();

        let tokenizer_path: Option<String> = server_args
            .as_ref()
            .and_then(|args| args.getattr(py, "tokenizer_path").ok())
            .and_then(|v| v.extract(py).ok())
            .or_else(|| {
                tm.getattr(py, "model_path")
                    .ok()
                    .and_then(|v| v.extract(py).ok())
            });

        let tokenizer_mode: Option<String> = server_args
            .as_ref()
            .and_then(|args| args.getattr(py, "tokenizer_mode").ok())
            .and_then(|v| v.extract(py).ok());

        let context_len: i32 = tm
            .getattr(py, "model_config")
            .ok()
            .and_then(|mc| mc.getattr(py, "context_len").ok())
            .and_then(|v| v.extract(py).ok())
            .unwrap_or(0);

        TokenizerInfo {
            tokenizer_path,
            tokenizer_mode,
            context_len,
        }
    })
}

/// Start the gRPC server in a background thread with its own Tokio runtime.
///
/// Args:
///     host: Bind address (e.g., "0.0.0.0")
///     port: Port number (e.g., 40000)
///     runtime_handle: Python RuntimeHandle object with submit_generate, submit_embed, abort, etc.
///
/// Returns:
///     GrpcServerHandle that can be used to shut down the server.
/// Start the native gRPC server in a background thread.
#[pyfunction]
#[pyo3(signature = (host, port, runtime_handle, worker_threads=4))]
fn start_server(
    host: String,
    port: u16,
    runtime_handle: PyObject,
    worker_threads: usize,
) -> PyResult<GrpcServerHandle> {
    let addr: SocketAddr = format!("{}:{}", host, port).parse().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid address: {}", e))
    })?;

    // Extract tokenizer info from Python (one-time GIL acquisition)
    let tokenizer_info = extract_tokenizer_info(&runtime_handle);

    // Attempt to load the Rust tokenizer
    let rust_tokenizer = tokenizer_info.tokenizer_path.as_deref().and_then(|p| {
        RustTokenizer::from_tokenizer_path(
            p,
            tokenizer_info.tokenizer_mode.as_deref(),
            tokenizer_info.context_len,
        )
    });

    let bridge = Arc::new(PyBridge::new(
        runtime_handle,
        rust_tokenizer,
        tokenizer_info.context_len,
    ));
    let shutdown = Arc::new(Notify::new());
    let shutdown_clone = shutdown.clone();

    let join_handle = std::thread::Builder::new()
        .name("sglang-grpc".to_string())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(worker_threads)
                .enable_all()
                .thread_name("sglang-grpc-tokio")
                .build()
                .expect("Failed to build Tokio runtime for gRPC server");

            if let Err(e) = rt.block_on(server::run_grpc_server(addr, bridge, shutdown_clone)) {
                eprintln!("gRPC server error: {}", e);
            }
        })
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to spawn gRPC thread: {}",
                e
            ))
        })?;

    Ok(GrpcServerHandle {
        shutdown,
        join_handle: Some(join_handle),
    })
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(start_server, m)?)?;
    m.add_class::<GrpcServerHandle>()?;
    Ok(())
}
