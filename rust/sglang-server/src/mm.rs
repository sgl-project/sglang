//! Multimodal worker pool.
//!
//! Rust-owned threads drain requests parked in the `Encoding` stage and run
//! the native pipeline ([`native`]) for the model family whose spec was
//! registered via `Server.start_mm_workers`: fetch → decode → preprocess →
//! placeholder expansion → M-RoPE, entirely in Rust (GIL-free). The worker
//! stores the result in the rid-keyed [`NativeSidecar`] and returns the
//! expanded ids; the Python side attaches the buffers at drain time
//! (`Server.take_native_mm`). Requests the pipeline cannot serve
//! (video/audio, precomputed inputs, undecodable images) are rejected back to
//! the client — there is no Python fallback path.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::message::MmRequest;
use crate::tokenizer::TextTokenizer;
use crate::tokenizer_manager::TmEvent;

pub mod native;

/// One native result: everything the drain-time Python adapter needs.
pub type NativeMmResult = sglang_mm::native_driver::NativeMmResult;

/// Results parked between a worker's `MmEncoded` and the scheduler's drain.
/// An entry is stored strictly before `MmEncoded` is emitted and popped by
/// `Server.take_native_mm`; late results for rejected requests are purged by
/// the ingress.
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

/// Spawn `workers` `mm-worker-{i}` threads and return their join handles for
/// the runtime's shutdown join. Not pinned here: they inherit the launch
/// thread's affinity, which `RustServer.launch` narrows to the server cores
/// before calling `start_mm_workers` so MM work never preempts the scheduler
/// thread on its reserved cores.
pub fn spawn_workers(
    rx: flume::Receiver<MmRequest>,
    tm: flume::Sender<TmEvent>,
    workers: usize,
    native: Arc<NativeContext>,
) -> Vec<std::thread::JoinHandle<()>> {
    (0..workers.max(1))
        .map(|i| {
            let worker = MmWorker {
                rx: rx.clone(),
                tm: tm.clone(),
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
    native: Arc<NativeContext>,
}

impl MmWorker {
    /// Drain until the mm channel closes (tm-ingress drops its sender on
    /// shutdown). Pool size bounds MM concurrency: each worker processes one
    /// request at a time. The sidecar entry is stored strictly *before*
    /// `MmEncoded` is emitted, so the scheduler drain that follows always
    /// finds it; an error rejects the request back to the client as a 400.
    fn run(self) {
        while let Ok(req) = self.rx.recv() {
            let rid = req.rid.clone();
            let event = match native::process(&self.native, &req) {
                Ok(input_ids) => {
                    tracing::debug!(%rid, tokens = input_ids.len(), "native mm: processed");
                    TmEvent::MmEncoded { rid, input_ids }
                }
                Err(message) => {
                    tracing::warn!(%rid, %message, "mm processing rejected");
                    TmEvent::MmFailed { rid, message }
                }
            };
            if self.tm.send(event).is_err() {
                return; // tm-ingress gone: shutdown
            }
        }
    }
}
