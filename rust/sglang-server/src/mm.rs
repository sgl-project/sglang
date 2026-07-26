//! Multimodal worker pool.
//!
//! Rust-owned threads drain requests parked in the `Encoding` stage and run
//! the `sglang-mm` pipeline for the model family whose spec was registered
//! via `Server.start_mm_workers`: fetch → decode → preprocess → placeholder
//! expansion → M-RoPE, entirely in Rust (GIL-free). The worker stores the
//! result in the rid-keyed [`Sidecar`] and returns the expanded ids; the
//! Python side attaches the buffers at drain time (`Server.take_mm`).
//! Requests the pipeline cannot serve (video/audio, precomputed inputs,
//! undecodable images) are rejected back to the client — there is no Python
//! fallback path.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::message::MmRequest;
use crate::tokenizer::TextTokenizer;
use crate::tokenizer_manager::TmEvent;

/// One mm result: everything the drain-time Python adapter needs. This is
/// the qwen scheduler-drain shape (`sglang_mm::qwen_vl::pack_drain`);
/// it generalizes to a named-tensor handoff when a family needs a
/// different shape.
pub struct MmResult {
    pub features: Vec<f32>,
    pub grids: Vec<[u32; 3]>,
    pub hashes: Vec<u64>,
    pub offsets: Vec<(u32, u32)>,
    pub mrope: Vec<i64>,
    pub mrope_delta: i64,
}

/// Results parked between a worker's `MmEncoded` and the scheduler's drain.
/// An entry is stored strictly before `MmEncoded` is emitted and popped by
/// `Server.take_mm`; late results for rejected requests are purged by the
/// ingress.
pub type Sidecar = Arc<Mutex<HashMap<String, MmResult>>>;

/// Shared state of the mm path, built once at `start_mm_workers`.
pub struct Context {
    pub family: Box<dyn sglang_mm::pipeline::MmFamilyProcessor>,
    /// `None` under `skip_tokenizer_init` (requests must carry `input_ids`).
    pub tokenizer: Option<Arc<dyn TextTokenizer>>,
    pub sidecar: Sidecar,
}

impl Context {
    pub fn new(
        spec_json: &str,
        tokenizer: Option<Arc<dyn TextTokenizer>>,
        sidecar: Sidecar,
    ) -> Result<Self, String> {
        Ok(Self {
            family: sglang_mm::registry::pipeline_from_spec(spec_json)?,
            tokenizer,
            sidecar,
        })
    }
}

/// Run the pipeline for one request. `Ok` returns the final
/// placeholder-expanded ids (the mm buffers are parked in the sidecar
/// strictly before returning); `Err` rejects the request back to the client.
fn process(ctx: &Context, req: &MmRequest) -> Result<Vec<i32>, String> {
    let input = crate::message::mm_payload::parse(&req.payload)?;
    let output = sglang_mm::driver::process(ctx.family.as_ref(), input, |text| {
        let tokenizer = ctx.tokenizer.as_ref().ok_or_else(|| {
            "skip_tokenizer_init is set: multimodal text prompts require input_ids".to_string()
        })?;
        tokenizer.encode(text).map_err(|error| error.to_string())
    })?;
    let drain = sglang_mm::qwen_vl::pack_drain(output)?;
    ctx.sidecar.lock().unwrap().insert(
        req.rid.as_str().to_owned(),
        MmResult {
            features: drain.features,
            grids: drain.grids,
            hashes: drain.hashes,
            offsets: drain.offsets,
            mrope: drain.mrope,
            mrope_delta: drain.mrope_delta,
        },
    );
    Ok(drain.input_ids)
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
    ctx: Arc<Context>,
) -> Vec<std::thread::JoinHandle<()>> {
    (0..workers.max(1))
        .map(|i| {
            let worker = MmWorker {
                rx: rx.clone(),
                tm: tm.clone(),
                ctx: ctx.clone(),
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
    ctx: Arc<Context>,
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
            let event = match process(&self.ctx, &req) {
                Ok(input_ids) => {
                    tracing::debug!(%rid, tokens = input_ids.len(), "mm: processed");
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
