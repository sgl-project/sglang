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
mod chat;
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

pub use chat::{NativeChatOutput, NativeChatProcessor};
pub use message::MmWorkItem;
pub use message::mm_payload::{ResolvedMediaWork, resolve_media_work};
pub use mm::{
    GenericMmItem, GenericMmSidecarEntry, GenericTensor, GenericTensorData, MmProcessOutput,
    MmSidecarEntry, NativeMmProcessor,
};
pub use tokenizer::TextTokenizer;

use std::net::SocketAddr;

use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyBytes;

use crate::runtime::{Runtime, RuntimeConfig};

/// One drained MM result (see [`Server::take_mm`]). Exactly one of
/// `features`/`shm_names` is `Some`: inline features for single-rank serving
/// (zero-copy into numpy), or one POSIX segment name per item when the scheduler
/// broadcasts across TP ranks and Python wraps each in a `ShmPointerMMData`.
#[pyclass(frozen, get_all)]
pub struct MmHandoff {
    features: Option<Py<numpy::PyArray1<f32>>>,
    shm_names: Option<Vec<String>>,
    grids: Vec<(u32, u32, u32)>,
    hashes: Vec<u64>,
    offsets: Vec<(u32, u32)>,
    mrope: Py<numpy::PyArray1<i64>>,
    mrope_delta: i64,
    generic_items: Vec<Py<GenericMmItemHandoff>>,
    metadata_json: Option<String>,
}

/// One model-agnostic MM item. Exactly one numeric feature field is populated;
/// `shape` gives its logical dimensions and the owning model package interprets
/// the opaque JSON metadata.
#[pyclass(frozen, get_all)]
pub struct GenericMmItemHandoff {
    modality: String,
    feature_f32: Option<Py<numpy::PyArray1<f32>>>,
    feature_i64: Option<Py<numpy::PyArray1<i64>>>,
    feature_u8: Option<Py<numpy::PyArray1<u8>>>,
    feature_u16: Option<Py<numpy::PyArray1<u16>>>,
    shape: Vec<usize>,
    hash: u64,
    offsets: Vec<(u32, u32)>,
    metadata_json: String,
}

/// Columnar ingress batch handed to Python by [`Server::recv_requests`].
/// `frozen`: immutable snapshot, so field access never contends on a borrow.
#[pyclass(frozen, get_all)]
pub struct IngressBatch {
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
pub struct Server {
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
    pub fn start(
        http_addr: Option<String>,
        ingress_ring_cap: usize,
        egress_ring_cap: usize,
        channel_cap: usize,
        cores: Option<Vec<usize>>,
        server_args_json: &str,
    ) -> PyResult<Self> {
        Self::start_with_chat_processor(
            http_addr,
            ingress_ring_cap,
            egress_ring_cap,
            channel_cap,
            cores,
            server_args_json,
            None,
        )
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
    pub fn recv_requests(&self, py: Python<'_>, max: usize) -> PyResult<IngressBatch> {
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
    pub fn wait_ingress(&self, py: Python<'_>, timeout_ms: u64) -> bool {
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
    pub fn push_batch(&self, py: Python<'_>, header: &[u8], data_cols: Vec<PyBackedBytes>) -> bool {
        let cols: Vec<&[u8]> = data_cols.iter().map(|d| d.as_ref()).collect();
        self.push_frame(py, crate::message::frame_egress_batch_cols(header, &cols))
    }

    /// Push a control-request result. Blocks for backpressure; `False` only on
    /// shutdown.
    pub fn push_result(&self, py: Python<'_>, rid: &str, payload: &[u8]) -> bool {
        self.push_frame(py, crate::message::frame_egress_result(rid, payload))
    }

    /// Route a terminal failure back to request `rid`. Blocks for backpressure;
    /// `False` only on shutdown.
    pub fn push_error(&self, py: Python<'_>, rid: &str, message: &str) -> bool {
        self.push_frame(py, crate::message::frame_egress_error(rid, message))
    }

    /// Spawn the MM worker pool for the pipeline in `spec_json` (built from the
    /// resolved processor config; see `NativeMmHost.resolve_native_spec`).
    /// Image-only requests are processed entirely in Rust and parked for
    /// [`Server::take_mm`]; anything the pipeline cannot serve is rejected back to
    /// the client — there is no Python fallback.
    pub fn start_mm_workers(&self, spec_json: &str, workers: usize) -> PyResult<()> {
        let ctx = mm::Context::new(
            spec_json,
            self.rt.tokenizer.clone(),
            self.rt.mm_sidecar.clone(),
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        self.rt.spawn_mm_pool(workers, std::sync::Arc::new(ctx));
        Ok(())
    }

    /// Pop the MM result for `rid` — parked strictly before the request reached
    /// the ingress ring — or `None` if there is none. The numeric buffers become
    /// 1-D numpy arrays that take **ownership** of the Rust vectors, no copy.
    ///
    /// Runs on the scheduler loop (`RustServer.drain`, under the GIL) between
    /// decode steps, so any per-byte work here — memcpy or hashing, tens of MB
    /// per image-heavy request — would stall every running request's ITL. Hence
    /// the worker-precomputed `hashes`.
    pub fn take_mm(&self, py: Python<'_>, rid: &str) -> PyResult<Option<MmHandoff>> {
        use numpy::IntoPyArray;

        let Some(res) = self.rt.mm_sidecar.take(rid) else {
            return Ok(None);
        };
        match res {
            mm::MmSidecarEntry::Qwen(res) => {
                let (features, shm_names) = match res.features {
                    mm::FeatureStore::Inline(v) => (Some(v.into_pyarray(py).unbind()), None),
                    // The segments — and the duty to unlink — move to Python here;
                    // `materialize()` unlinks after the post-broadcast clone on each rank.
                    mm::FeatureStore::Shm(segments) => (
                        None,
                        Some(segments.into_iter().map(|s| s.into_name()).collect()),
                    ),
                };
                Ok(Some(MmHandoff {
                    features,
                    shm_names,
                    grids: res.grids.iter().map(|g| (g[0], g[1], g[2])).collect(),
                    hashes: res.hashes,
                    offsets: res.offsets,
                    mrope: res.mrope.into_pyarray(py).unbind(),
                    mrope_delta: res.mrope_delta,
                    generic_items: Vec::new(),
                    metadata_json: None,
                }))
            }
            mm::MmSidecarEntry::Generic(res) => {
                let mut items = Vec::with_capacity(res.items.len());
                for item in res.items {
                    let (feature_f32, feature_i64, feature_u8, feature_u16) =
                        match item.feature.data {
                            mm::GenericTensorData::F32(v) => {
                                (Some(v.into_pyarray(py).unbind()), None, None, None)
                            }
                            mm::GenericTensorData::I64(v) => {
                                (None, Some(v.into_pyarray(py).unbind()), None, None)
                            }
                            mm::GenericTensorData::U8(v) => {
                                (None, None, Some(v.into_pyarray(py).unbind()), None)
                            }
                            mm::GenericTensorData::U16(v) => {
                                (None, None, None, Some(v.into_pyarray(py).unbind()))
                            }
                        };
                    items.push(Py::new(
                        py,
                        GenericMmItemHandoff {
                            modality: item.modality,
                            feature_f32,
                            feature_i64,
                            feature_u8,
                            feature_u16,
                            shape: item.feature.shape,
                            hash: item.hash,
                            offsets: item.offsets,
                            metadata_json: item.metadata_json,
                        },
                    )?);
                }
                Ok(Some(MmHandoff {
                    features: None,
                    shm_names: None,
                    grids: Vec::new(),
                    hashes: Vec::new(),
                    offsets: Vec::new(),
                    mrope: Vec::<i64>::new().into_pyarray(py).unbind(),
                    mrope_delta: 0,
                    generic_items: items,
                    metadata_json: Some(res.metadata_json),
                }))
            }
        }
    }

    /// Signal all threads to stop (best effort).
    pub fn shutdown(&self) {
        self.rt.request_shutdown();
    }
}

impl Server {
    /// Boot the frontend with an optional model-package chat renderer.
    #[allow(clippy::too_many_arguments)]
    pub fn start_with_chat_processor(
        http_addr: Option<String>,
        ingress_ring_cap: usize,
        egress_ring_cap: usize,
        channel_cap: usize,
        cores: Option<Vec<usize>>,
        server_args_json: &str,
        chat_processor: Option<std::sync::Arc<dyn NativeChatProcessor>>,
    ) -> PyResult<Self> {
        // Parse and validate static metadata at boot, not on the first request.
        let server_args: runtime::ServerArgs = runtime::ServerArgs::from_json(server_args_json)
            .map_err(|error| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "bad server_args_json: {error}"
                ))
            })?;
        server_args.validate_mandatory().map_err(|error| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("server_args: {error}"))
        })?;
        let http_addr: SocketAddr = http_addr
            .unwrap_or_else(|| server_args.bind())
            .parse()
            .map_err(|error| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("bad http_addr: {error}"))
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
            chat_processor,
        };
        let rt = runtime::start(cfg).map_err(|error| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "runtime start failed: {error}"
            ))
        })?;
        Ok(Server { rt })
    }

    /// Start the shared worker pool with a processor supplied by an external
    /// model package. The default Python API uses [`Server::start_mm_workers`]
    /// and therefore retains the built-in Qwen registry behavior.
    pub fn start_mm_workers_with_processor(
        &self,
        processor: std::sync::Arc<dyn NativeMmProcessor>,
        workers: usize,
    ) {
        let ctx = mm::Context::with_processor(
            processor,
            self.rt.tokenizer.clone(),
            self.rt.mm_sidecar.clone(),
        );
        self.rt.spawn_mm_pool(workers, std::sync::Arc::new(ctx));
    }

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

/// Register the Python boundary value types used by [`Server`]. External
/// model-package modules call this once before exposing a wrapper server; the
/// default `_core` module uses the same function, so the two surfaces cannot
/// drift as new boundary types are added.
pub fn register_boundary_types(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IngressBatch>()?;
    m.add_class::<MmHandoff>()?;
    m.add_class::<GenericMmItemHandoff>()?;
    Ok(())
}

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
    register_boundary_types(m)?;
    Ok(())
}
