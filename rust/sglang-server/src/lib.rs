//! sglang-server: a multi-threaded Rust frontend (HTTP server → TokenizerManager
//! → Tokenizer/Detokenizer) embedded in the Python scheduler process.
//!
//! This file is the Python↔Rust boundary: it registers the pyo3 module
//! (`_server`) and the classes exposed to the scheduler — the boot config
//! ([`ServerArgs`] and its parts, constructed by keyword from Python; their
//! `#[pyclass]`es and constructors live in `message::config`), [`Server`]
//! (boot, `recv_requests`/`wait_request`, `push_*`, MM handoff, shutdown),
//! [`RequestBatch`] and [`MmEncodedResult`]. Everything behind that boundary —
//! receiving requests, encoding multimodal inputs, tokenizing, detokenizing,
//! SSE streaming, and so on — is implemented purely in Rust and never touches
//! a `PyObject`.

mod api_server;
mod message;
mod multi_modality;
mod tokenizer_manager;
mod utils;

use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyBytes;

use crate::message::config::{
    DefaultSamplingParams, DisaggregationMode, MmFamily, MmResample, MmSpec, ModelConfig,
    RuntimeConfig, RustServerServerArgs, ServerArgs,
};
use crate::utils::startup::{listen_addr, value_error};
use crate::utils::{logging, runtime};

/// One drained MM result (see [`Server::take_mm_result`]), consumed by
/// `RustMmProcessor.wrap_encoded` to build the scheduler's
/// `MultimodalProcessorOutput`.
#[pyclass(frozen, get_all)]
struct MmEncodedResult {
    // General fields.
    /// All items' `pixel_values` concatenated as flat `f32` with logical shape
    /// `[sum(t*h*w), feature_dim]`; present on the inline (single-rank) path.
    features: Option<Py<numpy::PyArray1<f32>>>,
    /// Per-item POSIX shared-memory segment holding `[t*h*w, feature_dim]` f32
    /// features; present on the TP-broadcast path.
    shm_names: Option<Vec<String>>,
    /// Per-item content hash of the raw source bytes, or the caller-provided
    /// `mm_hashes` override, precomputed so draining never re-hashes.
    hashes: Vec<u64>,
    /// Per-item inclusive `(start, end)` placeholder-token span in the expanded
    /// `input_ids`.
    offsets: Vec<(u32, u32)>,

    // Qwen-VL-specific fields.
    /// Per-item `image_grid_thw` `(t, h, w)` in patch units; `t*h*w` is also the
    /// item's row count in `features`.
    grids: Vec<(u32, u32, u32)>,
    /// M-RoPE position ids as flat `i64` with row-major shape `[3, seq_len]`
    /// (temporal, height, and width rows).
    mrope: Py<numpy::PyArray1<i64>>,
    /// M-RoPE delta, `max(mrope) + 1 - seq_len`, added to the plain sequence
    /// position during decoding.
    mrope_delta: i64,
}

/// Columnar request batch handed to Python by [`Server::recv_requests`].
/// `frozen`: immutable snapshot, so field access never contends on a borrow.
#[pyclass(frozen, get_all)]
struct RequestBatch {
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
    rt: runtime::Runtime,
}

#[pymethods]
impl Server {
    /// Boot the frontend (spawns all threads) and return immediately.
    /// `server_args` is the scheduler's [`ServerArgs`]; the rest are
    /// rust-server-only overrides.
    #[new]
    #[pyo3(signature = (
        server_args,
        port_offset = None,
        to_scheduler_cap = 8192,
        from_scheduler_cap = 8192,
        stage_channel_cap = 8192,
        cores = None,
    ))]
    // pyo3 `#[new]` constructor: the wide arg list is the Python-facing boot
    // surface (all optional overrides), not a call-site ergonomics problem.
    #[allow(clippy::too_many_arguments)]
    fn start(
        server_args: ServerArgs,
        port_offset: Option<u16>, // DP rank; listen on server_args.port + offset
        to_scheduler_cap: usize,
        from_scheduler_cap: usize,
        stage_channel_cap: usize,
        cores: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        // `server_args` already arrived typed (pyo3 rejected any missing/extra/
        // mistyped field when Python constructed it); only value checks remain.
        server_args
            .validate()
            .map_err(|e| value_error("server_args", e))?;
        // The host and base port come from `server_args`; DP ranks only supply
        // their offset so this boundary has one source of truth for the address.
        let http_addr = listen_addr(&server_args, port_offset)
            .map_err(|e| value_error("bad listen address", e))?;

        let cfg = RuntimeConfig {
            rust_server_args: RustServerServerArgs {
                http_addr,
                http_api_worker_num: server_args.http_api_worker_num(),
                to_scheduler_cap,
                from_scheduler_cap,
                stage_channel_cap,
                cores,
            },
            server_args: std::sync::Arc::new(server_args),
        };
        let rt = runtime::start(cfg).map_err(|e| value_error("runtime start failed", e))?;
        Ok(Server { rt })
    }

    /// Non-blocking drain of the to_scheduler channel, returned **columnar** as an
    /// [`RequestBatch`] so the large `input_ids` tensor never goes through
    /// msgpack (see the field docs for the layout).
    #[pyo3(signature = (max = 256))]
    fn recv_requests(&self, py: Python<'_>, max: usize) -> PyResult<RequestBatch> {
        let cols = self.rt.to_scheduler_rx.drain(max);
        let headers = cols
            .headers
            .iter()
            .map(|h| PyBytes::new(py, h).unbind())
            .collect();
        let data = PyBytes::new_with(py, cols.ids_total, |buf| {
            cols.copy_ids_into(buf);
            Ok(())
        })?;
        Ok(RequestBatch {
            headers,
            data: data.unbind(),
            lengths: cols.lengths,
        })
    }

    /// Park up to `timeout_ms` for an incoming request so the idle scheduler loop
    /// sleeps instead of spinning at 100% CPU.
    #[pyo3(signature = (timeout_ms = 1000))]
    fn wait_request(&self, py: Python<'_>, timeout_ms: u64) -> bool {
        py.detach(|| {
            self.rt
                .to_scheduler_rx
                .wait(std::time::Duration::from_millis(timeout_ms))
        })
    }

    /// Push a whole decode batch as ONE frame: a columnar msgpack `header` plus
    /// the raw `data_cols` (per-column `bytes`), concatenated here. Blocks for
    /// backpressure; `False` only on shutdown.
    fn push_decode_result_batch(
        &self,
        py: Python<'_>,
        header: &[u8],
        data_cols: Vec<PyBackedBytes>,
    ) -> bool {
        let cols: Vec<&[u8]> = data_cols.iter().map(|d| d.as_ref()).collect();
        self.push_frame(
            py,
            crate::message::response::frame_decode_batch_cols(header, &cols),
        )
    }

    /// Push a control-request result. Blocks for backpressure; `False` only on
    /// shutdown.
    fn push_control_result(&self, py: Python<'_>, rid: &str, payload: &[u8]) -> bool {
        self.push_frame(
            py,
            crate::message::response::frame_control_result(rid, payload),
        )
    }

    /// Route a terminal failure back to request `rid`. Blocks for backpressure;
    /// `False` only on shutdown.
    fn push_error(&self, py: Python<'_>, rid: &str, message: &str) -> bool {
        self.push_frame(py, crate::message::response::frame_error(rid, message))
    }

    /// Spawn the MM worker pool for the pipeline in `spec` (built from the
    /// resolved processor config; see `RustMmProcessor.resolve_spec` and
    /// `RustServer._build_mm_spec`). Image-only requests are processed entirely
    /// in Rust and parked for [`Server::take_mm_result`]; anything the pipeline
    /// cannot serve is rejected back to the client — there is no Python fallback.
    fn start_mm_workers(&self, spec: MmSpec, workers: usize) -> PyResult<()> {
        self.rt
            .start_mm_workers(spec, workers)
            .map_err(|e| value_error("mm spec", e))
    }

    /// Pop the MM result for `rid` — parked strictly before the request reached
    /// the to_scheduler channel — or `None` if there is none. The numeric
    /// buffers become 1-D numpy arrays that take **ownership** of the Rust
    /// vectors, no copy.
    ///
    /// Runs on the scheduler loop between decode steps, so any per-byte work
    /// here — memcpy or hashing, tens of MB per image-heavy request — would
    /// stall every running request's ITL. Hence the worker-precomputed `hashes`.
    fn take_mm_result(&self, py: Python<'_>, rid: &str) -> Option<MmEncodedResult> {
        use numpy::IntoPyArray;

        let res = self.rt.mm_results.take(rid)?;
        let (features, shm_names) = match res.features {
            multi_modality::result_store::FeatureStore::Inline(v) => {
                (Some(v.into_pyarray(py).unbind()), None)
            }
            // The segments — and the duty to unlink — move to Python here;
            // `materialize()` unlinks after the post-broadcast clone on each rank.
            multi_modality::result_store::FeatureStore::Shm(segments) => (
                None,
                Some(segments.into_iter().map(|s| s.into_name()).collect()),
            ),
        };
        Some(MmEncodedResult {
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
    /// Hand one already-framed message to the ring. Shared by every push path —
    /// they differ solely in how the frame is built. `false` only on shutdown.
    #[inline]
    fn push_frame(&self, py: Python<'_>, frame: bytes::Bytes) -> bool {
        match self.rt.from_scheduler_tx.try_push(frame) {
            Ok(()) => true,
            // Consumer gone (shutdown): the frame is unavoidably lost.
            Err(None) => false,
            // Full: the scheduler must block here so backpressure reaches it.
            Err(Some(frame)) => py.detach(|| self.rt.from_scheduler_tx.push(frame)),
        }
    }
}

#[pymodule]
fn _server(m: &Bound<'_, PyModule>) -> PyResult<()> {
    logging::init_tracing();
    m.add_class::<DisaggregationMode>()?;
    m.add_class::<DefaultSamplingParams>()?;
    m.add_class::<ModelConfig>()?;
    m.add_class::<ServerArgs>()?;
    m.add_class::<MmFamily>()?;
    m.add_class::<MmResample>()?;
    m.add_class::<MmSpec>()?;
    m.add_class::<Server>()?;
    m.add_class::<RequestBatch>()?;
    m.add_class::<MmEncodedResult>()?;
    Ok(())
}
