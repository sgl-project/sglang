//! sglang-server: a multi-threaded Rust frontend (API server → TokenizerManager
//! → Tokenizer/Detokenizer) embedded in the Python scheduler process.
//!
//! Pipeline stages 1–5 are pure Rust and never touch a `PyObject`, so they run
//! concurrently with the Python scheduler without contending for the GIL. The
//! only GIL crossings are the boundary methods on [`Server`]:
//!   * `recv_requests` — Python scheduler thread drains the ingress ring.
//!   * `push_batch`    — Python scheduler thread pushes one output batch.
//!   * `push_result`   — Python scheduler thread pushes one control result.
//!
//! All are non-blocking, so the GIL is never held across a wait.

mod api_server;
mod detokenizer;
mod error;
mod fsm;
mod ids;
mod message;
mod runtime;
mod tokenizer;
mod tokenizer_manager;

use std::net::SocketAddr;

use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyBytes;

use crate::runtime::{Runtime, RuntimeConfig};

/// Columnar ingress batch handed to Python by [`Server::recv_requests`]:
/// `(headers, ids_buf, lengths)` — per-request scalar msgpack headers, all
/// requests' raw int64 ids concatenated, and per-request token counts.
type IngressBatch<'py> = (Vec<Bound<'py, PyBytes>>, Bound<'py, PyBytes>, Vec<u32>);

/// Handle owned by the Python scheduler process. Construct once via
/// [`Server::start`], then poll it from the scheduler event loop.
#[pyclass]
struct Server {
    rt: Runtime,
}

#[pymethods]
impl Server {
    /// Boot the frontend (spawns all threads) and return immediately.
    #[new]
    #[pyo3(signature = (
        bind = None,
        ingress_ring_cap = 8192,
        egress_ring_cap = 8192,
        channel_cap = 8192,
        pin_cores = true,
        cores = None,

        server_args_json = "{}",
    ))]
    // pyo3 `#[new]` constructor: the wide arg list is the Python-facing boot
    // surface (all optional overrides), not a call-site ergonomics problem.
    #[allow(clippy::too_many_arguments)]
    fn start(
        bind: Option<String>,
        ingress_ring_cap: usize,
        egress_ring_cap: usize,
        channel_cap: usize,
        pin_cores: bool,
        cores: Option<Vec<usize>>,
        server_args_json: &str,
    ) -> PyResult<Self> {
        // Static server metadata (server_args + model_config) dumped by the
        // scheduler; parse and validate mandatory fields now so a bad/missing
        // field is a boot error, not a request-time 500.
        let server_args: runtime::ServerArgs = runtime::ServerArgs::from_json(server_args_json)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "bad server_args_json: {e}"
                ))
            })?;
        server_args.validate_mandatory().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("server_args: {e}"))
        })?;
        // The bind address, tokenizer source/threads/shards all live in the
        // `server_args` blob; resolve them from there so the scheduler doesn't
        // re-pass them. The explicit params stay as optional overrides for
        // standalone callers (tests) that construct a `Server` without a full
        // `server_args`.
        let bind: SocketAddr = bind
            .unwrap_or_else(|| server_args.bind())
            .parse()
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("bad bind: {e}"))
            })?;

        let tokenizer_worker_num = server_args.tokenizer_worker_num();
        let detokenizer_worker_num = server_args.detokenizer_worker_num();
        let api_worker_num = server_args.api_worker_num();

        let tokenizer_path = server_args.tokenizer_path();
        let revision = server_args.revision();

        let server_args = std::sync::Arc::new(server_args);

        let cfg = RuntimeConfig {
            bind,
            api_worker_num,
            tokenizer_worker_num,
            detokenizer_worker_num,
            ingress_ring_cap,
            egress_ring_cap,
            channel_cap,
            pin_cores,
            cores,
            tokenizer_path,
            revision,
            server_args,
        };
        let rt = runtime::start(cfg).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("runtime start failed: {e}"))
        })?;
        Ok(Server { rt })
    }

    /// Non-blocking drain of the ingress ring, returned **columnar** so the large
    /// `input_ids` tensor never goes through msgpack. Yields a 3-tuple:
    ///   * `headers`: `list[bytes]` — one msgpack scalar header per request
    ///     (`input_ids` omitted), decoded individually by the scheduler;
    ///   * `ids_buf`: `bytes` — all requests' raw little-endian int64 ids,
    ///     concatenated; sliced per request and wrapped as `array("q")`;
    ///   * `lengths`: `list[int]` — per-request token count (0 for control reqs),
    ///     so the scheduler can slice `ids_buf`.
    /// The GIL is released for the drain + columnar split; only the `PyBytes`
    /// marshaling needs it. The `ids` cells are copied **directly into the result
    /// `bytes`** (one copy, no intermediate buffer).
    #[pyo3(signature = (max = 256))]
    fn recv_requests<'py>(&self, py: Python<'py>, max: usize) -> PyResult<IngressBatch<'py>> {
        let cols = py.detach(|| self.rt.ingress.drain(max));
        let headers = cols.headers.iter().map(|h| PyBytes::new(py, h)).collect();
        // Single pass: copy each raw ids cell straight into the output `bytes`.
        let ids_buf = PyBytes::new_with(py, cols.ids_total, |buf| {
            let mut pos = 0;
            for cell in &cols.ids {
                let end = pos + cell.len();
                buf[pos..end].copy_from_slice(cell);
                pos = end;
            }
            Ok(())
        })?;
        Ok((headers, ids_buf, cols.lengths))
    }

    /// Park up to `timeout_ms` for an incoming request so the idle scheduler loop
    /// sleeps instead of spinning at 100% CPU. Returns `True` when a request is
    /// ready (the next `recv_requests` includes it). The GIL is released while
    /// parked, and `flume` wakes the moment a request is pushed, so this adds no
    /// latency to real requests — only the idle wait is bounded by `timeout_ms`.
    #[pyo3(signature = (timeout_ms = 1000))]
    fn wait_ingress(&self, py: Python<'_>, timeout_ms: u64) -> bool {
        py.detach(|| {
            self.rt
                .ingress
                .wait(std::time::Duration::from_millis(timeout_ms))
        })
    }

    /// Push a whole decode batch as ONE frame: a columnar msgpack `header` plus
    /// the raw `data_cols` (per-column `bytes`), concatenated here. Blocks for
    /// backpressure; `False` only on shutdown.
    fn push_batch(&self, py: Python<'_>, header: &[u8], data_cols: Vec<PyBackedBytes>) -> bool {
        let cols: Vec<&[u8]> = data_cols.iter().map(|d| d.as_ref()).collect();
        py.detach(|| {
            let bytes = crate::message::frame_egress_batch_cols(header, &cols);
            self.rt.egress.push(bytes)
        })
    }

    /// Push a control-request result. Blocks for backpressure; `False` only on
    /// shutdown. Frames inside `detach` like `push_batch` — the payload is small
    /// here so the copy is cheap either way, but kept uniform so no push holds the
    /// GIL across framing.
    fn push_result(&self, py: Python<'_>, rid: &str, payload: &[u8]) -> bool {
        py.detach(|| {
            self.rt
                .egress
                .push(crate::message::frame_egress_result(rid, payload))
        })
    }

    /// Route a terminal failure back to request `rid`. Blocks for backpressure;
    /// `False` only on shutdown.
    fn push_error(&self, py: Python<'_>, rid: &str, message: &str) -> bool {
        py.detach(|| {
            self.rt
                .egress
                .push(crate::message::frame_egress_error(rid, message))
        })
    }

    /// Signal all threads to stop (best effort).
    fn shutdown(&self) {
        self.rt.request_shutdown();
    }
}

/// Keeps the non-blocking log writer's background thread alive for the process
/// lifetime (dropping the guard would stop log delivery).
static LOG_GUARD: std::sync::OnceLock<tracing_appender::non_blocking::WorkerGuard> =
    std::sync::OnceLock::new();

/// The Python module: `import sglang_server`.
#[pymodule]
fn sglang_server(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize tracing once; ignore if already set by the host process.
    // Non-blocking writer: emitting threads (axum workers, egress, detok) only
    // enqueue; a dedicated thread does the stdout formatting-flush + syscall.
    // The queue is bounded and lossy — under extreme pressure log lines are
    // dropped instead of stalling request threads.
    let (writer, guard) = tracing_appender::non_blocking(std::io::stdout());
    let _ = LOG_GUARD.set(guard);
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(writer)
        .try_init();
    m.add_class::<Server>()?;
    Ok(())
}
