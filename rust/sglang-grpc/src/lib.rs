pub mod bridge;
pub mod server;
pub mod tokenizers;
pub(crate) mod utils;

pub mod proto {
    tonic::include_proto!("sglang.runtime.v1");
}

use pyo3::prelude::*;
use std::net::{SocketAddr, TcpListener};
use std::sync::Arc;
use tokio::sync::Notify;
use tokio::time::Duration;
use tracing_subscriber::EnvFilter;

use bridge::{ChunkSendStatus, DEFAULT_RESPONSE_CHANNEL_CAPACITY, PyBridge};
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
        self.join_handle.as_ref().is_some_and(|h| !h.is_finished())
    }
}

struct TokenizerInfo {
    tokenizer_path: Option<String>,
    tokenizer_mode: Option<String>,
    context_len: i32,
}

/// Extract tokenizer path/mode and context_len from the Python RuntimeHandle (one-time GIL).
///
/// Missing `tokenizer_manager` indicates a misconfigured runtime handle and should surface at
/// startup. Sub-fields are best-effort because unsupported native tokenizer backends can still
/// fall back to Python tokenization.
fn try_get_attr(
    py: Python<'_>,
    obj: &Py<PyAny>,
    attr: &'static str,
    context: &'static str,
) -> Option<Py<PyAny>> {
    obj.getattr(py, attr).map(Some).unwrap_or_else(|err| {
        tracing::debug!("{}.{} is unavailable: {}", context, attr, err);
        None
    })
}

fn try_get_attr_str(
    py: Python<'_>,
    obj: &Py<PyAny>,
    attr: &'static str,
    context: &'static str,
) -> Option<String> {
    try_get_attr(py, obj, attr, context).and_then(|value| {
        value.extract(py).map(Some).unwrap_or_else(|err| {
            tracing::debug!("Could not extract {}.{} as string: {}", context, attr, err);
            None
        })
    })
}

fn try_get_attr_i32(
    py: Python<'_>,
    obj: &Py<PyAny>,
    attr: &'static str,
    context: &'static str,
) -> Option<i32> {
    try_get_attr(py, obj, attr, context).and_then(|value| {
        value.extract(py).map(Some).unwrap_or_else(|err| {
            tracing::debug!("Could not extract {}.{} as i32: {}", context, attr, err);
            None
        })
    })
}

fn extract_tokenizer_info(runtime_handle: &Py<PyAny>) -> PyResult<TokenizerInfo> {
    Python::attach(|py| {
        let tm = runtime_handle
            .getattr(py, "tokenizer_manager")
            .map_err(|err| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "runtime_handle.tokenizer_manager is required: {}",
                    err
                ))
            })?;

        let server_args = try_get_attr(py, &tm, "server_args", "tokenizer_manager");

        let tokenizer_path = server_args
            .as_ref()
            .and_then(|args| try_get_attr_str(py, args, "tokenizer_path", "server_args"))
            .or_else(|| {
                server_args
                    .as_ref()
                    .and_then(|args| try_get_attr_str(py, args, "model_path", "server_args"))
            })
            .or_else(|| try_get_attr_str(py, &tm, "model_path", "tokenizer_manager"));
        if tokenizer_path.is_none() {
            tracing::warn!("Could not extract tokenizer path; Rust tokenizer disabled");
        }

        let tokenizer_mode = server_args
            .as_ref()
            .and_then(|args| try_get_attr_str(py, args, "tokenizer_mode", "server_args"));

        let context_len = try_get_attr(py, &tm, "model_config", "tokenizer_manager")
            .and_then(|model_config| {
                try_get_attr_i32(py, &model_config, "context_len", "model_config")
            })
            .unwrap_or_else(|| {
                tracing::warn!("Could not extract model_config.context_len; defaulting to 0");
                0
            });

        Ok(TokenizerInfo {
            tokenizer_path,
            tokenizer_mode,
            context_len,
        })
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
#[pyfunction]
#[pyo3(signature = (host, port, runtime_handle, worker_threads=4, response_channel_capacity=64, response_timeout_secs=300))]
fn start_server(
    host: String,
    port: u16,
    runtime_handle: Py<PyAny>,
    worker_threads: usize,
    response_channel_capacity: usize,
    response_timeout_secs: u64,
) -> PyResult<GrpcServerHandle> {
    // Best-effort: embedding processes may initialize tracing themselves.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .try_init();

    let addr: SocketAddr = format!("{}:{}", host, port)
        .parse()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid address: {}", e)))?;
    let worker_threads = worker_threads.max(1);
    let response_channel_capacity = if response_channel_capacity == 0 {
        tracing::warn!(
            default = DEFAULT_RESPONSE_CHANNEL_CAPACITY,
            "response_channel_capacity must be positive; using default"
        );
        DEFAULT_RESPONSE_CHANNEL_CAPACITY
    } else {
        response_channel_capacity
    };
    let response_timeout_secs = if response_timeout_secs == 0 {
        tracing::warn!(
            default = server::DEFAULT_RESPONSE_TIMEOUT_SECS,
            "response_timeout_secs must be positive; using default"
        );
        server::DEFAULT_RESPONSE_TIMEOUT_SECS
    } else {
        response_timeout_secs
    };
    let response_timeout = Duration::from_secs(response_timeout_secs);
    let listener = TcpListener::bind(addr).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Failed to bind gRPC server to {}: {}",
            addr, e
        ))
    })?;
    listener.set_nonblocking(true).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Failed to configure gRPC listener for {}: {}",
            addr, e
        ))
    })?;

    let tokenizer_info = extract_tokenizer_info(&runtime_handle)?;

    let rust_tokenizer = tokenizer_info.tokenizer_path.as_deref().and_then(|p| {
        RustTokenizer::from_tokenizer_path(
            p,
            tokenizer_info.tokenizer_mode.as_deref(),
            tokenizer_info.context_len,
        )
    });

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(worker_threads)
        .enable_all()
        .thread_name("sglang-grpc-tokio")
        .build()
        .map_err(|err| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to build Tokio runtime for gRPC server: {}",
                err
            ))
        })?;
    let tokio_handle = rt.handle().clone();

    let bridge = Arc::new(PyBridge::new(
        runtime_handle,
        rust_tokenizer,
        tokenizer_info.context_len,
        response_channel_capacity,
        tokio_handle,
    ));
    let shutdown = Arc::new(Notify::new());
    let shutdown_clone = shutdown.clone();
    let bridge_clone = bridge.clone();

    let join_handle = std::thread::Builder::new()
        .name("sglang-grpc".to_string())
        .spawn(move || {
            if let Err(e) = rt.block_on(server::run_grpc_server(
                listener,
                bridge_clone,
                shutdown_clone,
                response_timeout,
            )) {
                tracing::error!("gRPC server exited with error: {}", e);
            }
        })
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to spawn gRPC thread: {}", e))
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
    m.add_class::<ChunkSendStatus>()?;
    Ok(())
}
