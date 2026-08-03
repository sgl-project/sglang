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
mod environ;
mod error;
mod fsm;
mod ids;
mod message;
mod mm;
mod ring;
mod runtime;
mod tokenizer;
mod tokenizer_manager;
mod utils;

use std::net::SocketAddr;

use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyBytes;

use crate::runtime::{Runtime, RuntimeConfig};

/// One drained MM result (see [`Server::take_mm`]), named fields instead of
/// a positional tuple so the Rust/Python schema is explicit. Exactly one of
/// `features`/`shm_names` is `Some`: inline features for single-rank serving
/// (zero-copy into numpy), or one POSIX shm segment name per item when the
/// scheduler broadcasts across TP ranks (the Python side wraps each name in
/// a `ShmPointerMMData` stub).
#[pyclass(frozen, get_all)]
struct MmHandoff {
    features: Option<Py<numpy::PyArray1<f32>>>,
    shm_names: Option<Vec<String>>,
    grids: Vec<(u32, u32, u32)>,
    hashes: Vec<u64>,
    offsets: Vec<(u32, u32)>,
    mrope: Py<numpy::PyArray1<i64>>,
    mrope_delta: i64,
}

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

        let cfg = RuntimeConfig {
            rust_server_args: runtime::RustServerServerArgs {
                http_addr,
                api_worker_num: server_args.api_worker_num(),
                ingress_ring_cap,
                egress_ring_cap,
                channel_cap,
                cores,
            },
            server_args: std::sync::Arc::new(server_args),
        };
        let rt = runtime::start(cfg).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("runtime start failed: {e}"))
        })?;
        Ok(Server { rt })
    }

    /// Non-blocking drain of the ingress ring, returned **columnar** as an
    /// [`IngressBatch`] so the large `input_ids` tensor never goes through
    /// msgpack (see the field docs for the layout). The `ids` cells are copied
    /// **directly into the result `bytes`** (one copy, no intermediate buffer).
    ///
    /// Runs entirely GIL-held, deliberately. `drain` is a `try_recv` loop plus an
    /// uncontended stash lock (the Python thread is the only consumer), so it
    /// cannot block — there is nothing for a detach to overlap with. And detaching
    /// is far from free: reacquiring the GIL waits out the interpreter's switch
    /// interval, so a `py.detach` here cost up to 5 ms whenever another Python
    /// thread was runnable, to cover ~0.2 µs of work. Held, the whole call is a
    /// fraction of a microsecond on an empty ring.
    #[pyo3(signature = (max = 256))]
    fn recv_requests(&self, py: Python<'_>, max: usize) -> PyResult<IngressBatch> {
        let cols = self.rt.ingress.drain(max);
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
    ///
    /// Framed and pushed with the GIL HELD, detaching only if the ring is full.
    /// This runs on the scheduler's CUDA-launch thread every decode step, where the
    /// unconditional detach was the single worst boundary cost: framing is
    /// ~0.1–0.2 µs, but reacquiring the GIL waits out the interpreter's switch
    /// interval (5 ms by default) whenever another Python thread is runnable —
    /// 17–50% of a 10–30 ms decode step, landing nondeterministically. Held, the
    /// whole boundary is ~1.3 µs per step.
    ///
    /// The slow path keeps its detach because a full ring genuinely parks: the
    /// scheduler must feel backpressure rather than drop output it has already
    /// committed to. It essentially never fires — measured headroom is ~100×.
    fn push_batch(&self, py: Python<'_>, header: &[u8], data_cols: Vec<PyBackedBytes>) -> bool {
        let cols: Vec<&[u8]> = data_cols.iter().map(|d| d.as_ref()).collect();
        self.push_frame(py, crate::message::frame_egress_batch_cols(header, &cols))
    }

    /// Push a control-request result. Blocks for backpressure; `False` only on
    /// shutdown.
    fn push_result(&self, py: Python<'_>, rid: &str, payload: &[u8]) -> bool {
        self.push_frame(py, crate::message::frame_egress_result(rid, payload))
    }

    /// Route a terminal failure back to request `rid`. Blocks for backpressure;
    /// `False` only on shutdown.
    fn push_error(&self, py: Python<'_>, rid: &str, message: &str) -> bool {
        self.push_frame(py, crate::message::frame_egress_error(rid, message))
    }

    /// Spawn the MM worker pool for the pipeline described by `spec_json`
    /// (built by the Python side from the resolved processor config; see
    /// `NativeMmHost.resolve_native_spec`). Image-only requests are processed
    /// entirely in Rust and their results parked for [`Server::take_mm`];
    /// requests the pipeline cannot serve (video/audio, precomputed inputs,
    /// undecodable images) are rejected back to the client — there is no
    /// Python fallback.
    fn start_mm_workers(&self, spec_json: &str, workers: usize) -> PyResult<()> {
        let ctx = mm::Context::new(
            spec_json,
            self.rt.tokenizer.clone(),
            self.rt.mm_sidecar.clone(),
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        let handles = mm::spawn_workers(
            self.rt.mm.clone(),
            self.rt.tm.clone(),
            workers,
            std::sync::Arc::new(ctx),
        );
        self.rt.adopt_threads(handles);
        Ok(())
    }

    /// Pop the MM result for `rid` (stored strictly before the request was
    /// pushed to the ingress ring). Returns
    /// `(features_f32, grids, hashes, offsets, mrope_i64, mrope_delta)` or
    /// `None` when no result is parked for `rid`. The two numeric
    /// buffers are 1-D numpy arrays that take **ownership** of the Rust
    /// vectors — no copy — and `hashes` are the worker-precomputed per-image
    /// feature hashes. This runs on the scheduler loop (`RustServer.drain`,
    /// under the GIL) between decode steps, so any per-byte work here (memcpy,
    /// hashing — tens of MB per image-heavy request) would stall every running
    /// request's inter-token latency.
    fn take_mm(&self, py: Python<'_>, rid: &str) -> Option<MmHandoff> {
        use numpy::IntoPyArray;

        let res = self.rt.mm_sidecar.take(rid)?;
        let (features, shm_names) = match res.features {
            mm::FeatureStore::Inline(v) => (Some(v.into_pyarray(py).unbind()), None),
            // Ownership of the segments (and the duty to unlink) moves to
            // Python here: `materialize()` unlinks after the post-broadcast
            // clone on every rank.
            mm::FeatureStore::Shm(segments) => (
                None,
                Some(segments.into_iter().map(|s| s.into_name()).collect()),
            ),
        };
        Some(MmHandoff {
            features,
            shm_names,
            grids: res.grids.iter().map(|g| (g[0], g[1], g[2])).collect(),
            hashes: res.hashes,
            offsets: res.offsets,
            mrope: res.mrope.into_pyarray(py).unbind(),
            mrope_delta: res.mrope_delta,
        })
    }

    /// Signal all threads to stop (best effort).
    fn shutdown(&self) {
        self.rt.request_shutdown();
    }
}

impl Server {
    /// Hand one already-framed egress message to the ring: GIL-held when it fits,
    /// detaching only to park on a full ring. Shared by every push path — they
    /// differ solely in how the frame is built. `false` only on shutdown.
    #[inline]
    fn push_frame(&self, py: Python<'_>, frame: bytes::Bytes) -> bool {
        match self.rt.egress.try_push(frame) {
            Ok(()) => true,
            // Consumer gone (shutdown): the frame is unavoidably lost.
            Err(None) => false,
            // Full: the scheduler must block here so backpressure reaches it, and
            // blocking is exactly when releasing the GIL pays for itself.
            Err(Some(frame)) => py.detach(|| self.rt.egress.push(frame)),
        }
    }
}

/// Keeps the non-blocking log writer's background thread alive for the process
/// lifetime (dropping the guard would stop log delivery).
static LOG_GUARD: std::sync::OnceLock<tracing_appender::non_blocking::WorkerGuard> =
    std::sync::OnceLock::new();

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
    m.add_class::<MmHandoff>()?;
    Ok(())
}
