//! sglang-server: a multi-threaded Rust frontend (HTTP server → TokenizerManager
//! → Tokenizer/Detokenizer) embedded in the Python scheduler process.
//!
//! This file is the Python↔Rust boundary: it registers the pyo3 module
//! (`_server`) and the classes exposed to the scheduler — the boot config
//! ([`ServerArgs`] and its parts, constructed by keyword from Python; their
//! `#[pyclass]`es and constructors live in `message::config`), [`Server`]
//! (boot, `recv_requests`/`wait_request`, `push_*`, shutdown),
//! [`IngressRequest`] and [`ShmBuffer`]. Everything behind that boundary —
//! receiving requests, encoding multimodal inputs, tokenizing, detokenizing,
//! SSE streaming, and so on — is implemented purely in Rust and never touches
//! a `PyObject`.

mod api_server;
mod message;
mod multi_modality;
mod tokenizer_manager;
mod utils;

pub use message::config::{
    DefaultSamplingParams, DisaggregationMode, MmFamily, MmResample, MmSpec, ModelConfig,
    RustServerServerArgs, ServerArgs,
};
pub use message::multimodal::MmItem;
pub use message::request::{MmWorkItem, ProcessorExtensions};
pub use message::types::TokenIds;
pub use multi_modality::encoded::{
    MRope, MmEncodedEntry, MmEncodedItem, MmMetaValue, MmModality, MmTokenIds,
};
pub use multi_modality::payload::{ResolvedMediaWork, resolve_media_work};
pub use multi_modality::worker::{MmProcessOutput, MmProcessor};
pub use sglang_mm::pipeline::{Tensor, TensorData};

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyBytes;

use crate::message::config::RuntimeConfig;
use crate::utils::startup::{listen_addr, value_error};
use crate::utils::{logging, runtime};

/// One drained request handed to Python by [`Server::recv_requests`]: the
/// msgpack scalar header plus every non-scalar payload as a named buffer —
/// `input_ids`, `token_ids_logprob`, and for a multimodal request the
/// `mm.*` set (`mm.feature.{i}` per item, `mm.mrope`, and the `mm.meta`
/// msgpack sidecar) that `RustMmProcessor.wrap_encoded` or an external
/// package's wrapper consumes. An inline buffer is a numpy array, shaped, that
/// **owns** the Rust vector (no copy); a shm buffer is a [`ShmBuffer`] naming
/// the segment to map. `frozen`: immutable snapshot, so field access never
/// contends on a borrow.
#[pyclass(frozen, get_all)]
pub struct IngressRequest {
    header: Py<PyBytes>,
    /// `(name, numpy array | ShmBuffer)` in producer order; empty for control
    /// requests.
    buffers: Vec<(String, Py<PyAny>)>,
}

/// A buffer parked in a POSIX shared-memory segment: `name` is what Python's
/// `SharedMemory(name=…)` opens, `dtype` the numpy dtype to view it with,
/// `shape` its logical shape. The duty to unlink moves to Python with it
/// (`ShmPointerMMData.materialize()` unlinks after the post-broadcast clone).
#[pyclass(frozen, get_all)]
pub struct ShmBuffer {
    name: String,
    dtype: &'static str,
    shape: Vec<usize>,
}

/// Hand one buffer across: the inline vector becomes a numpy array owning it,
/// viewed with the buffer's shape; the shm segment becomes its name.
fn buffer_to_py(py: Python<'_>, buffer: message::buffers::Buffer) -> PyResult<(String, Py<PyAny>)> {
    use message::buffers::{BufferData, BufferStore};
    use numpy::{IntoPyArray, PyArrayMethods};

    fn shaped<T: numpy::Element>(
        py: Python<'_>,
        v: Vec<T>,
        shape: &[usize],
    ) -> PyResult<Py<PyAny>> {
        let array = v.into_pyarray(py);
        Ok(if shape.len() == 1 {
            array.into_any().unbind()
        } else {
            array.reshape(shape.to_vec())?.into_any().unbind()
        })
    }

    let shape = buffer.shape;
    let value = match buffer.store {
        BufferStore::Inline(data) => match data {
            BufferData::I64(v) => shaped(py, v, &shape)?,
            BufferData::F32(v) => shaped(py, v, &shape)?,
            BufferData::U32(v) => shaped(py, v, &shape)?,
            BufferData::U64(v) => shaped(py, v, &shape)?,
            BufferData::U16(v) => shaped(py, v, &shape)?,
            BufferData::U8(v) => shaped(py, v, &shape)?,
        },
        BufferStore::Shm { segment, dtype } => Py::new(
            py,
            ShmBuffer {
                name: segment.into_name(),
                dtype: dtype.numpy(),
                shape,
            },
        )?
        .into_any(),
    };
    Ok((buffer.name, value))
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
        Ok(Self { rt })
    }

    /// Non-blocking drain of the to_scheduler channel: one [`IngressRequest`]
    /// per request, its buffers moved out of Rust rather than copied (see the
    /// type docs for the layout).
    #[pyo3(signature = (max = 256))]
    pub fn recv_requests(&self, py: Python<'_>, max: usize) -> PyResult<Vec<IngressRequest>> {
        self.rt
            .to_scheduler_rx
            .drain(max)
            .into_iter()
            .map(|req| {
                Ok(IngressRequest {
                    header: PyBytes::new(py, &req.header).unbind(),
                    buffers: req
                        .buffers
                        .into_iter()
                        .map(|b| buffer_to_py(py, b))
                        .collect::<PyResult<_>>()?,
                })
            })
            .collect()
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

    /// Route a terminal failure back to request `rid`. Blocks for backpressure;
    /// `False` only on shutdown.
    pub fn push_error(&self, py: Python<'_>, rid: &str, message: &str) -> bool {
        self.push_frame(py, crate::message::response::frame_error(rid, message))
    }

    /// Spawn the MM worker pool for the pipeline in `spec` (built from the
    /// resolved processor config; see `RustMmProcessor.resolve_spec` and
    /// `RustServer._build_mm_spec`). Image-only requests are processed entirely
    /// in Rust; their buffers ride the ring with the request (the `mm.*` set of
    /// [`IngressRequest::buffers`]). Anything the pipeline cannot serve is
    /// rejected back to the client — there is no Python fallback.
    pub fn start_mm_workers(&self, spec: MmSpec, workers: usize) -> PyResult<()> {
        self.rt
            .start_mm_workers(spec, workers)
            .map_err(|e| value_error("mm spec", e))
    }

    /// Signal all threads to stop (best effort).
    pub fn shutdown(&self) {
        self.rt.request_shutdown();
    }
}

impl Server {
    /// Start the shared worker pool with a processor supplied by an external
    /// model package. The default Python API retains the built-in Qwen path.
    /// `feature_shm` is the package's own `_use_feature_shm` answer: place
    /// feature tensors in POSIX shm for the TP broadcast.
    pub fn start_mm_workers_with_processor(
        &self,
        processor: Arc<dyn MmProcessor>,
        workers: usize,
        feature_shm: bool,
    ) {
        self.rt
            .start_mm_workers_with_processor(processor, workers, feature_shm);
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
    m.add_class::<DefaultSamplingParams>()?;
    m.add_class::<ModelConfig>()?;
    m.add_class::<ServerArgs>()?;
    m.add_class::<MmFamily>()?;
    m.add_class::<MmResample>()?;
    m.add_class::<MmSpec>()?;
    m.add_class::<IngressRequest>()?;
    m.add_class::<ShmBuffer>()?;
    Ok(())
}

#[pymodule]
fn _server(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Server>()?;
    register_boundary_types(m)?;
    Ok(())
}
