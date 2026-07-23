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
//! All are non-blocking, so the GIL is never held across a wait. The one
//! deliberate exception to "stages never touch a `PyObject`" is the MM worker
//! pool ([`mm`]): its Rust threads call the registered Python `mm_processor`
//! handler until that pipeline is ported to native Rust.

mod api_server;
mod detokenizer;
mod environ;
mod error;
mod fsm;
mod ids;
mod message;
mod mm;
mod runtime;
mod tokenizer;
mod tokenizer_manager;

use std::net::SocketAddr;

use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyBytes;

use crate::runtime::{Runtime, RuntimeConfig};

/// Columnar ingress batch handed to Python by [`Server::recv_requests`].
/// `frozen`: immutable snapshot, so field access never contends on a borrow.
#[pyclass(frozen, get_all)]
struct IngressBatch {
    /// One msgpack scalar header per request (`input_ids` omitted).
    headers: Vec<Py<PyBytes>>,
    /// The raw-data plane today just all requests' raw little-endian int64
    /// ids, concatenated; sliced per request via `lengths`.
    data: Py<PyBytes>,
    /// Per-request token count (0 for control requests).
    lengths: Vec<u32>,
}

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
        http_addr = None,
        ingress_ring_cap = 8192,
        egress_ring_cap = 8192,
        channel_cap = 8192,
        cores = None,

        server_args_json = "{}",
    ))]
    // pyo3 `#[new]` constructor: the wide arg list is the Python-facing boot
    // surface (all optional overrides), not a call-site ergonomics problem.
    #[allow(clippy::too_many_arguments)]
    fn start(
        http_addr: Option<String>,
        ingress_ring_cap: usize,
        egress_ring_cap: usize,
        channel_cap: usize,
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
        // The HTTP listen address, tokenizer source/threads/shards all live in the
        // `server_args` blob; resolve them from there so the scheduler doesn't
        // re-pass them. The explicit params stay as optional overrides for
        // standalone callers (tests) that construct a `Server` without a full
        // `server_args`.
        let http_addr: SocketAddr = http_addr
            .unwrap_or_else(|| server_args.bind())
            .parse()
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("bad http_addr: {e}"))
            })?;

        let tokenizer_worker_num = server_args.tokenizer_worker_num;
        let detokenizer_worker_num = server_args.detokenizer_worker_num;
        let api_worker_num = server_args.api_worker_num();

        // Empty only in minimal standalone blobs (the Python dump always resolves
        // it); empty → no tokenizer, which `runtime::start` allows only under
        // `skip_tokenizer_init`. Repo ids are resolved to a local path in
        // `RustServer._build_server_args` before this blob is parsed.
        let tokenizer_path =
            (!server_args.tokenizer_path.is_empty()).then(|| server_args.tokenizer_path.clone());

        let server_args = std::sync::Arc::new(server_args);

        let cfg = RuntimeConfig {
            http_addr,
            api_worker_num,
            tokenizer_worker_num,
            detokenizer_worker_num,
            ingress_ring_cap,
            egress_ring_cap,
            channel_cap,
            cores,
            tokenizer_path,
            server_args,
        };
        let rt = runtime::start(cfg).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("runtime start failed: {e}"))
        })?;
        Ok(Server { rt })
    }

    /// Non-blocking drain of the ingress ring, returned **columnar** as an
    /// [`IngressBatch`] so the large `input_ids` tensor never goes through
    /// msgpack (see the field docs for the layout). The GIL is released for the
    /// drain + columnar split; only the `PyBytes` marshaling needs it. The `ids`
    /// cells are copied **directly into the result `bytes`** (one copy, no
    /// intermediate buffer).
    #[pyo3(signature = (max = 256))]
    fn recv_requests(&self, py: Python<'_>, max: usize) -> PyResult<IngressBatch> {
        let cols = py.detach(|| self.rt.ingress.drain(max));
        let headers = cols
            .headers
            .iter()
            .map(|h| PyBytes::new(py, h).unbind())
            .collect();
        // Single pass: copy each raw ids cell straight into the output `bytes`.
        let data = PyBytes::new_with(py, cols.ids_total, |buf| {
            let mut pos = 0;
            for cell in &cols.ids {
                let end = pos + cell.len();
                buf[pos..end].copy_from_slice(cell);
                pos = end;
            }
            Ok(())
        })?
        .unbind();
        Ok(IngressBatch {
            headers,
            data,
            lengths: cols.lengths,
        })
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

    /// Register the Python MM handler and spawn the Rust MM worker pool. Each
    /// worker drains one `Encoding`-parked request at a time and calls
    /// `handler(rid, payload) -> bytes` — the msgpack
    /// `[text, input_ids, image_data, video_data, audio_data]` payload in, the
    /// final placeholder-expanded prompt ids (raw little-endian int64 bytes)
    /// out; a Python exception rejects the request as a 400. `workers` bounds
    /// MM concurrency. Call once, before serving multimodal traffic.
    ///
    /// `native_spec_json` (when the model family has a pure-Rust pipeline)
    /// enables the native path: image-only requests are processed entirely in
    /// Rust and their results parked for [`Server::take_native_mm`]; anything
    /// unsupported still drives `handler`.
    #[pyo3(signature = (handler, workers = 8, native_spec_json = None))]
    fn start_mm_workers(
        &self,
        py: Python<'_>,
        handler: Py<PyAny>,
        workers: usize,
        native_spec_json: Option<&str>,
    ) -> PyResult<()> {
        let native = native_spec_json
            .map(|spec| {
                mm::NativeContext::new(spec, self.rt.tokenizer.clone(), self.rt.mm_native.clone())
                    .map(std::sync::Arc::new)
            })
            .transpose()
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        let handles = mm::spawn_workers(
            py,
            self.rt.mm.clone(),
            self.rt.tm.clone(),
            handler,
            workers,
            native,
        );
        self.rt.adopt_threads(handles);
        Ok(())
    }

    /// Pop the native MM result for `rid` (stored strictly before the request
    /// was pushed to the ingress ring). Returns
    /// `(features_f32, grids, hashes, offsets, mrope_i64, mrope_delta)` or
    /// `None` when the request took the Python path. The two numeric buffers
    /// are 1-D numpy arrays that take **ownership** of the Rust vectors — no
    /// copy — and `hashes` are the worker-precomputed per-image feature
    /// hashes. This runs on the scheduler loop (`RustServer.drain`, under the
    /// GIL) between decode steps, so any per-byte work here (memcpy, hashing —
    /// tens of MB per image-heavy request) would stall every running
    /// request's inter-token latency.
    fn take_native_mm<'py>(
        &self,
        py: Python<'py>,
        rid: &str,
    ) -> Option<(
        Bound<'py, numpy::PyArray1<f32>>,
        Vec<(u32, u32, u32)>,
        Vec<u64>,
        Vec<(u32, u32)>,
        Bound<'py, numpy::PyArray1<i64>>,
        i64,
    )> {
        use numpy::IntoPyArray;

        let res = self.rt.mm_native.lock().unwrap().remove(rid)?;
        let features = res.features.into_pyarray(py);
        let mrope = res.mrope.into_pyarray(py);
        let grids = res.grids.iter().map(|g| (g[0], g[1], g[2])).collect();
        Some((
            features,
            grids,
            res.hashes,
            res.offsets,
            mrope,
            res.mrope_delta,
        ))
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
    m.add_class::<IngressBatch>()?;
    Ok(())
}
