//! Multimodal worker pool.
//!
//! Rust-owned threads drain requests parked in the `Encoding` stage. Two paths:
//!
//! * **Native** ([`native`]): for model families with a pure-Rust pipeline
//!   (spec registered via `Server.start_mm_workers`), the worker runs fetch →
//!   decode → preprocess → placeholder expansion → M-RoPE entirely in Rust
//!   (GIL-free), stores the result in the rid-keyed [`NativeSidecar`] and
//!   returns the expanded ids. The Python side attaches the buffers at drain
//!   time (`Server.take_native_mm`).
//! * **Python fallback**: everything else (no native spec, video/audio,
//!   precomputed inputs, unsupported shapes) drives the *Python*
//!   `mm_processor` path (`rust_server.MmProcessorHost.handle_sync`: media
//!   download/load, resize, HF AutoProcessor) through the registered handler.
//!   This is the one deliberate exception to the "pipeline stages never touch
//!   a `PyObject`" rule: a worker holds the GIL only for the handler call
//!   (which itself releases it while waiting on the processor's event loop)
//!   and parks GIL-free on the channel.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyBytes;

use crate::message::MmRequest;
use crate::runtime::channels::TmEvent;
use crate::tokenizer::TextTokenizer;

pub mod native;

/// One native result: everything the drain-time Python adapter needs to build
/// the scheduler's `MultimodalProcessorOutput` without any processing logic.
pub struct NativeMmResult {
    /// Per-image `pixel_values`, concatenated (split by grid products).
    pub features: Vec<f32>,
    /// Per-image `[t, h, w]` patch grids.
    pub grids: Vec<[u32; 3]>,
    /// Per-image feature hashes (`MultimodalDataItem.hash`), precomputed in
    /// the worker so the scheduler's `set_pad_value` never hashes the buffer.
    pub hashes: Vec<u64>,
    /// Per-image inclusive `(start, end)` token offsets in the expanded ids.
    pub offsets: Vec<(u32, u32)>,
    /// Flattened row-major `[3, input_len]` M-RoPE positions.
    pub mrope: Vec<i64>,
    pub mrope_delta: i64,
}

/// Results parked between a worker's `MmEncoded` and the scheduler's drain.
/// An entry is stored strictly before `MmEncoded` is emitted and popped by
/// `Server.take_native_mm`; late results for rejected requests are purged by
/// the ingress (same lifecycle as the Python host's `results` table).
pub type NativeSidecar = Arc<Mutex<HashMap<String, NativeMmResult>>>;

/// Shared state of the native path, built once at `start_mm_workers`.
pub struct NativeContext {
    pub pipeline: sglang_mm::registry::NativePipeline,
    /// `None` under `skip_tokenizer_init` (requests must carry `input_ids`).
    pub tokenizer: Option<Arc<dyn TextTokenizer>>,
    pub sidecar: NativeSidecar,
}

impl NativeContext {
    pub fn new(
        spec_json: &str,
        tokenizer: Option<Arc<dyn TextTokenizer>>,
        sidecar: NativeSidecar,
    ) -> Result<Self, String> {
        Ok(Self {
            pipeline: sglang_mm::registry::native_pipeline_from_spec(spec_json)?,
            tokenizer,
            sidecar,
        })
    }
}

/// Spawn `workers` `mm-worker-{i}` threads (unpinned — the heavy work happens
/// off the pinned pools either way) and return their join handles for the
/// runtime's shutdown join.
pub fn spawn_workers(
    py: Python<'_>,
    rx: flume::Receiver<MmRequest>,
    tm: flume::Sender<TmEvent>,
    handler: Py<PyAny>,
    workers: usize,
    native: Option<Arc<NativeContext>>,
) -> Vec<std::thread::JoinHandle<()>> {
    (0..workers.max(1))
        .map(|i| {
            let worker = MmWorker {
                rx: rx.clone(),
                tm: tm.clone(),
                handler: handler.clone_ref(py),
                native: native.clone(),
            };
            std::thread::Builder::new()
                .name(format!("mm-worker-{i}"))
                .spawn(move || worker.run())
                .expect("spawn mm worker")
        })
        .collect()
}

struct MmWorker {
    rx: flume::Receiver<MmRequest>,
    tm: flume::Sender<TmEvent>,
    handler: Py<PyAny>,
    native: Option<Arc<NativeContext>>,
}

impl MmWorker {
    /// Drain until the mm channel closes (tm-ingress drops its sender on
    /// shutdown). Pool size bounds MM concurrency: each worker processes one
    /// request at a time.
    fn run(self) {
        while let Ok(req) = self.rx.recv() {
            let rid = req.rid.clone();
            let event = match self.process(req) {
                Ok(input_ids) => TmEvent::MmEncoded { rid, input_ids },
                Err(message) => {
                    tracing::warn!(%rid, %message, "mm processing failed");
                    TmEvent::MmFailed { rid, message }
                }
            };
            if self.tm.send(event).is_err() {
                return; // tm-ingress gone: shutdown
            }
        }
    }

    /// Try the native pipeline first; unsupported requests fall through to the
    /// Python handler unchanged. Both store their side output (sidecar / rid
    /// table) strictly *before* returning, so `MmEncoded` (and the scheduler
    /// drain that follows) always finds the entry. An error rejects the
    /// request back to the client as a 400.
    fn process(&self, req: MmRequest) -> Result<Vec<i32>, String> {
        if let Some(ctx) = &self.native {
            match native::process(ctx, &req) {
                native::Outcome::Done(ids) => {
                    tracing::debug!(rid = %req.rid, tokens = ids.len(), "native mm: processed");
                    return Ok(ids);
                }
                native::Outcome::Fallback(reason) => {
                    tracing::debug!(rid = %req.rid, %reason, "native mm: python fallback");
                }
                native::Outcome::Failed(message) => return Err(message),
            }
        }
        self.process_python(req)
    }

    /// One handler call: `handler(rid, payload) -> bytes` — the final
    /// placeholder-expanded prompt ids as raw little-endian int64 bytes
    /// (`array("q").tobytes()`). The handler stores the processed `mm_inputs`
    /// in its rid-keyed table before returning.
    fn process_python(&self, req: MmRequest) -> Result<Vec<i32>, String> {
        Python::attach(|py| {
            let out = self
                .handler
                .call1(py, (req.rid.as_str(), PyBytes::new(py, &req.payload)))
                .map_err(|e| e.to_string())?;
            let ids = out
                .extract::<PyBackedBytes>(py)
                .map_err(|e| format!("mm handler must return bytes: {e}"))?;
            decode_ids(&ids)
        })
    }
}

/// Raw little-endian int64 bytes → `Vec<i32>`.
fn decode_ids(bytes: &[u8]) -> Result<Vec<i32>, String> {
    if !bytes.len().is_multiple_of(8) {
        return Err("mm handler returned malformed input_ids bytes".into());
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as i32)
        .collect())
}
