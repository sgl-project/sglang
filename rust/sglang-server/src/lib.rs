//! sglang-server: a multi-threaded Rust frontend (HTTP server → TokenizerManager
//! → Tokenizer/Detokenizer) embedded in the Python scheduler process.
//!
//! This file is the Python↔Rust boundary: it registers the pyo3 module
//! (`_server`) and the classes exposed to the scheduler — the boot config
//! ([`ServerArgs`] and its parts, constructed by keyword from Python; their
//! `#[pyclass]`es and constructors live in `message::config`), [`Server`]
//! (boot, `recv_requests`/`wait_request`, `push_*`, MM results, shutdown),
//! [`RequestBatch`] and [`MmEncodedResult`]. Everything behind that boundary —
//! receiving requests, encoding multimodal inputs, tokenizing, detokenizing,
//! SSE streaming, and so on — is implemented purely in Rust and never touches
//! a `PyObject`.

mod api_server;
mod message;
mod metrics;
mod multi_modality;
mod tokenizer_manager;
mod utils;

pub use api_server::app::{ChatInput, HttpExtension, RequestPreparation, ResponseProcessor};
pub use api_server::dp::DpIngress;
pub use message::config::{
    DefaultSamplingParams, DisaggregationMode, MmFamily, MmResample, MmSpec, ModelConfig,
    RustServerServerArgs, ServerArgs,
};
pub use message::multimodal::{MmItem, MmProcessorOptions};
pub use message::request::{
    GenerateRequest, MmData, MmFetchTiming, MmPadSpan, MmPrefetchStats, MmWorkItem,
    ProcessorExtensions,
};
pub use message::response::OutputMetadata;
pub use message::sampling::SamplingParams;
pub use multi_modality::payload::{ResolvedMediaWork, resolve_media_work};
pub use multi_modality::result_store::{
    ExternalMmEncodedEntry, ExternalMmItem, MmEncodedEntry, MmModality, MmTokenIds,
};
pub use multi_modality::worker::{MmProcessOutput, MmProcessor};
pub use sglang_mm::pipeline::{Tensor, TensorData};
pub use tokenizer_manager::tokenizer::TextTokenizer;

use std::collections::BTreeMap;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyBytes;

use crate::message::config::RuntimeConfig;
use crate::utils::startup::{listen_addr, value_error};
use crate::utils::{logging, runtime};

/// One drained MM result (see [`Server::take_mm_result`]).
///
/// Built-in results are consumed by `RustMmProcessor.wrap_encoded` to build
/// `MultimodalProcessorOutput`. External integrations consume `external_items`
/// and `external_token_ids` in their own Python wrappers; the built-in fields
/// are empty in that case.
#[pyclass(frozen, get_all)]
pub struct MmEncodedResult {
    // General fields for the built-in processor path.
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

    // Fields for external processor integrations.
    external_items: Vec<Py<ExternalMmItemResult>>,
    external_token_ids: Option<MmTokenIds>,
}

/// One media item returned to an external Python integration. The NumPy view
/// owns the feature allocation.
#[pyclass(frozen, get_all)]
pub struct ExternalMmItemResult {
    modality: MmModality,
    feature: MmFeatureArray,
    shape: Vec<usize>,
    hash: u64,
    offsets: Vec<(u32, u32)>,
    model_specific_data: BTreeMap<String, i64>,
}

#[derive(IntoPyObjectRef)]
enum MmFeatureArray {
    #[pyo3(transparent)]
    F32(Py<numpy::PyArray1<f32>>),
    #[pyo3(transparent)]
    I64(Py<numpy::PyArray1<i64>>),
    #[pyo3(transparent)]
    Bf16(Py<numpy::PyArray1<u16>>),
}

#[pymethods]
impl ExternalMmItemResult {
    /// BF16 is exposed as raw u16 bits for Python to reinterpret without a copy.
    #[getter]
    fn feature_is_bf16(&self) -> bool {
        matches!(self.feature, MmFeatureArray::Bf16(_))
    }
}

impl MmEncodedResult {
    fn from_entry(py: Python<'_>, entry: MmEncodedEntry) -> PyResult<Self> {
        use numpy::IntoPyArray;

        match entry {
            MmEncodedEntry::Qwen(entry) => {
                let (features, shm_names) = match entry.features {
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
                Ok(Self {
                    features,
                    shm_names,
                    hashes: entry.hashes,
                    offsets: entry.offsets,
                    grids: entry.grids.iter().map(|g| (g[0], g[1], g[2])).collect(),
                    mrope: entry.mrope.into_pyarray(py).unbind(),
                    mrope_delta: entry.mrope_delta,
                    external_items: Vec::new(),
                    external_token_ids: None,
                })
            }
            MmEncodedEntry::External(entry) => {
                let external_items = entry
                    .items
                    .into_iter()
                    .map(|item| {
                        let feature = match item.feature.data {
                            TensorData::F32(data) => {
                                MmFeatureArray::F32(data.into_pyarray(py).unbind())
                            }
                            TensorData::I64(data) => {
                                MmFeatureArray::I64(data.into_pyarray(py).unbind())
                            }
                            TensorData::Bf16(data) => {
                                MmFeatureArray::Bf16(data.into_pyarray(py).unbind())
                            }
                        };
                        Py::new(
                            py,
                            ExternalMmItemResult {
                                modality: item.modality,
                                feature,
                                shape: item.feature.shape,
                                hash: item.hash,
                                offsets: item.offsets,
                                model_specific_data: item.model_specific_data,
                            },
                        )
                    })
                    .collect::<PyResult<_>>()?;
                Ok(Self {
                    features: None,
                    shm_names: None,
                    hashes: Vec::new(),
                    offsets: Vec::new(),
                    grids: Vec::new(),
                    mrope: Vec::<i64>::new().into_pyarray(py).unbind(),
                    mrope_delta: 0,
                    external_items,
                    external_token_ids: Some(entry.token_ids),
                })
            }
        }
    }
}

/// Columnar request batch handed to Python by [`Server::recv_requests`].
/// `frozen`: immutable snapshot, so field access never contends on a borrow.
#[pyclass(frozen, get_all)]
pub struct RequestBatch {
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
        http_port = None,
    ))]
    // pyo3 `#[new]` constructor: the wide arg list is the Python-facing boot
    // surface (all optional overrides), not a call-site ergonomics problem.
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        server_args: ServerArgs,
        port_offset: Option<u16>, // DP rank; listen on server_args.port + offset
        to_scheduler_cap: usize,
        from_scheduler_cap: usize,
        stage_channel_cap: usize,
        cores: Option<Vec<usize>>,
        http_port: Option<u16>,
    ) -> PyResult<Self> {
        Self::start_with_http_extension(
            server_args,
            port_offset,
            to_scheduler_cap,
            from_scheduler_cap,
            stage_channel_cap,
            cores,
            None,
            http_port,
        )
    }

    /// Non-blocking drain of the to_scheduler channel, returned **columnar** as an
    /// [`RequestBatch`] so the large `input_ids` tensor never goes through
    /// msgpack (see the field docs for the layout).
    #[pyo3(signature = (max = 256))]
    pub fn recv_requests(&self, py: Python<'_>, max: usize) -> PyResult<RequestBatch> {
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
    pub fn wait_request(&self, py: Python<'_>, timeout_ms: u64) -> bool {
        py.detach(|| {
            self.rt
                .to_scheduler_rx
                .wait(std::time::Duration::from_millis(timeout_ms))
        })
    }

    #[getter]
    pub fn http_port(&self) -> u16 {
        self.rt.http_addr.port()
    }

    /// Called only after the launch parent's warmup has completed.
    pub fn mark_ready(&self) {
        self.rt
            .startup_ready
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// Push a whole decode batch as ONE frame: a columnar msgpack `header` plus
    /// the raw `data_cols` (per-column `bytes`), concatenated here. Blocks for
    /// backpressure; `False` only on shutdown.
    pub fn push_decode_result_batch(
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
    pub fn push_control_result(&self, py: Python<'_>, rid: &str, payload: &[u8]) -> bool {
        self.push_frame(
            py,
            crate::message::response::frame_control_result(rid, payload),
        )
    }

    pub fn push_control_result_part(
        &self,
        py: Python<'_>,
        rid: &str,
        dp_rank: u32,
        payload: &[u8],
    ) -> bool {
        self.push_frame(
            py,
            crate::message::response::frame_control_result_part(rid, dp_rank, payload),
        )
    }

    pub fn push_abort_result(&self, py: Python<'_>, rid: &str, payload: &[u8]) -> bool {
        self.push_frame(
            py,
            crate::message::response::frame_abort_result(rid, payload),
        )
    }

    /// Publish the scheduler's existing public load snapshot for HTTP readers.
    pub fn publish_load_snapshot(&self, py: Python<'_>, payload: &[u8]) -> PyResult<()> {
        py.detach(|| self.rt.load_snapshots.publish(payload))
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Route a terminal failure back to request `rid`. Blocks for backpressure;
    /// `False` only on shutdown.
    pub fn push_error(&self, py: Python<'_>, rid: &str, message: &str) -> bool {
        self.push_frame(py, crate::message::response::frame_error(rid, message))
    }

    /// Spawn the MM worker pool for the pipeline in `spec` (built from the
    /// resolved processor config; see `RustMmProcessor.resolve_spec` and
    /// `RustServer._build_mm_spec`). Image-only requests are processed entirely
    /// in Rust and parked for [`Server::take_mm_result`]; anything the pipeline
    /// cannot serve is rejected back to the client — there is no Python fallback.
    pub fn start_mm_workers(&self, spec: MmSpec, workers: usize) -> PyResult<()> {
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
    pub fn take_mm_result(&self, py: Python<'_>, rid: &str) -> PyResult<Option<MmEncodedResult>> {
        self.rt
            .mm_results
            .take(rid)
            .map(|entry| MmEncodedResult::from_entry(py, entry))
            .transpose()
    }

    /// Signal all threads to stop (best effort).
    pub fn shutdown(&self) {
        self.rt.request_shutdown();
    }
}

impl Server {
    /// External Rust model packages install their HTTP integration before the
    /// listener starts accepting requests.
    #[allow(clippy::too_many_arguments)]
    pub fn start_with_http_extension(
        server_args: ServerArgs,
        port_offset: Option<u16>,
        to_scheduler_cap: usize,
        from_scheduler_cap: usize,
        stage_channel_cap: usize,
        cores: Option<Vec<usize>>,
        http_extension: Option<Arc<dyn HttpExtension>>,
        http_port: Option<u16>,
    ) -> PyResult<Self> {
        server_args
            .validate()
            .map_err(|e| value_error("server_args", e))?;
        let http_addr = listen_addr(&server_args, port_offset, http_port)
            .map_err(|e| value_error("bad listen address", e))?;
        let cfg = RuntimeConfig {
            rust_server_args: RustServerServerArgs {
                http_addr,
                http_api_worker_num: server_args.http_api_worker_num(),
                to_scheduler_cap,
                from_scheduler_cap,
                stage_channel_cap,
                cores,
                http_extension,
            },
            server_args: Arc::new(server_args),
        };
        let rt = runtime::start(cfg).map_err(|e| value_error("runtime start failed", e))?;
        Ok(Self { rt })
    }

    /// Start the shared worker pool with a processor supplied by an external
    /// model package. The default Python API retains the built-in Qwen path.
    pub fn start_mm_workers_with_processor(&self, processor: Arc<dyn MmProcessor>, workers: usize) {
        self.rt.start_mm_workers_with_processor(processor, workers);
    }

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

/// Register all Python boundary types used by [`Server`]. External
/// model-package modules call this before exposing their wrapper server.
pub fn register_boundary_types(m: &Bound<'_, PyModule>) -> PyResult<()> {
    logging::init_tracing();
    m.add_class::<DisaggregationMode>()?;
    m.add_class::<message::types::HiddenStatesMode>()?;
    m.add_class::<DefaultSamplingParams>()?;
    m.add_class::<ModelConfig>()?;
    m.add_class::<ServerArgs>()?;
    m.add_class::<DpIngress>()?;
    m.add_class::<MmFamily>()?;
    m.add_class::<MmResample>()?;
    m.add_class::<MmSpec>()?;
    m.add_class::<RequestBatch>()?;
    m.add_class::<MmEncodedResult>()?;
    m.add_class::<ExternalMmItemResult>()?;
    m.add_class::<MmModality>()?;
    m.add_class::<MmTokenIds>()?;
    Ok(())
}

#[pymodule]
fn _server(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Server>()?;
    register_boundary_types(m)?;
    Ok(())
}
