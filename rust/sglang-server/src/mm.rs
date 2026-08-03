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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::message::MmRequest;
use crate::tokenizer::TextTokenizer;
use crate::tokenizer_manager::TmEvent;

/// A named POSIX shared-memory segment owning its name: dropped → unlinked.
///
/// Written by an MM worker (off the scheduler loop) so the TP broadcast can
/// carry a pointer instead of the feature tensor: the scheduler wraps the
/// name in a `ShmPointerMMData` stub, `broadcast_pyobj` moves ~100 bytes
/// instead of ~20 MB, and every TP rank maps + materializes the segment in
/// parallel. Python's `materialize()` unlinks after cloning; this `Drop`
/// covers the paths where the buffers never reach Python (request aborted
/// while parked, late result purged).
pub struct ShmSegment {
    name: String,
}

impl ShmSegment {
    /// Create `/dev/shm/{name}` holding exactly `bytes`. The name must be
    /// usable by Python's `multiprocessing.shared_memory.SharedMemory(name=…)`,
    /// i.e. no leading slash here (shm_open gets one added).
    pub fn create(name: String, bytes: &[u8]) -> Result<Self, String> {
        let c_name = std::ffi::CString::new(format!("/{name}"))
            .map_err(|_| "shm name contains NUL".to_string())?;
        // SAFETY: plain POSIX calls on a name we own; every handle created
        // below is closed/unmapped on all paths.
        unsafe {
            let fd = libc::shm_open(
                c_name.as_ptr(),
                libc::O_CREAT | libc::O_EXCL | libc::O_RDWR,
                0o600,
            );
            if fd < 0 {
                return Err(format!(
                    "shm_open({name}): {}",
                    std::io::Error::last_os_error()
                ));
            }
            let segment = Self { name }; // unlink from here on any failure
            if libc::ftruncate(fd, bytes.len() as libc::off_t) != 0 {
                let e = std::io::Error::last_os_error();
                libc::close(fd);
                return Err(format!("ftruncate({}): {e}", segment.name));
            }
            let ptr = libc::mmap(
                std::ptr::null_mut(),
                bytes.len(),
                libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            );
            libc::close(fd);
            if ptr == libc::MAP_FAILED {
                return Err(format!(
                    "mmap({}): {}",
                    segment.name,
                    std::io::Error::last_os_error()
                ));
            }
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast::<u8>(), bytes.len());
            libc::munmap(ptr, bytes.len());
            Ok(segment)
        }
    }

    /// Hand the segment (and the duty to unlink) to the caller — used when
    /// Python takes ownership at drain time.
    pub fn into_name(self) -> String {
        std::mem::take(&mut std::mem::ManuallyDrop::new(self).name)
    }
}

impl Drop for ShmSegment {
    fn drop(&mut self) {
        if let Ok(c_name) = std::ffi::CString::new(format!("/{}", self.name)) {
            // SAFETY: unlinking a name we created; ENOENT (already unlinked
            // by Python's materialize) is fine to ignore.
            unsafe { libc::shm_unlink(c_name.as_ptr()) };
        }
    }
}

/// Unique-enough segment names: pid guards across server restarts (a crashed
/// process may leak segments with its old pid), the counter within one.
fn shm_name(item: usize) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("sglmm-{}-{n}-{item}", std::process::id())
}

/// One mm result: everything the drain-time Python adapter needs. This is
/// the qwen scheduler-drain shape (`sglang_mm::qwen_vl::pack_drain`);
/// it generalizes to a named-tensor handoff when a family needs a
/// different shape.
pub struct MmResult {
    pub features: FeatureStore,
    pub grids: Vec<[u32; 3]>,
    pub hashes: Vec<u64>,
    pub offsets: Vec<(u32, u32)>,
    pub mrope: Vec<i64>,
    pub mrope_delta: i64,
}

/// Where a result's feature buffers live between worker and drain.
pub enum FeatureStore {
    /// In this process; the drain wraps them zero-copy (single-rank serving,
    /// or the shm fallback). Cheap for rank 0, but under TP the whole buffer
    /// would ride the scheduler's `broadcast_pyobj` to the other ranks.
    Inline(Vec<f32>),
    /// One POSIX segment per item; the drain sends only the names across
    /// ranks. Written by the worker, off the scheduler loop.
    Shm(Vec<ShmSegment>),
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
    /// Park feature buffers in POSIX shm (`"feature_shm": true` in the spec —
    /// set by the Python launcher exactly when the scheduler broadcasts
    /// requests across TP ranks and `ShmPointerMMData` will be unwrapped).
    pub feature_shm: bool,
}

impl Context {
    pub fn new(
        spec_json: &str,
        tokenizer: Option<Arc<dyn TextTokenizer>>,
        sidecar: Sidecar,
    ) -> Result<Self, String> {
        let feature_shm = serde_json::from_str::<serde_json::Value>(spec_json)
            .ok()
            .and_then(|v| v.get("feature_shm").and_then(|b| b.as_bool()))
            .unwrap_or(false);
        Ok(Self {
            family: sglang_mm::registry::pipeline_from_spec(spec_json)?,
            tokenizer,
            sidecar,
            feature_shm,
        })
    }
}

/// Run the pipeline for one request. `Ok` returns the final
/// placeholder-expanded ids (the mm buffers are parked in the sidecar
/// strictly before returning); `Err` rejects the request back to the client.
fn process(ctx: &Context, rid: &crate::ids::Rid, work: crate::message::MmWorkItem) -> Result<Vec<i32>, String> {
    let input = crate::message::mm_payload::to_mm_input(work)?;
    let output = sglang_mm::driver::process(ctx.family.as_ref(), input, |text| {
        let tokenizer = ctx.tokenizer.as_ref().ok_or_else(|| {
            "skip_tokenizer_init is set: multimodal text prompts require input_ids".to_string()
        })?;
        tokenizer.encode(text).map_err(|error| error.to_string())
    })?;
    let drain = sglang_mm::qwen_vl::pack_drain(output)?;
    let features = if ctx.feature_shm {
        park_features_in_shm(&drain.features, &drain.grids)
    } else {
        FeatureStore::Inline(drain.features)
    };
    ctx.sidecar.lock().unwrap().insert(
        rid.as_str().to_owned(),
        MmResult {
            features,
            grids: drain.grids,
            hashes: drain.hashes,
            offsets: drain.offsets,
            mrope: drain.mrope,
            mrope_delta: drain.mrope_delta,
        },
    );
    Ok(drain.input_ids)
}

/// Split the flat feature buffer per item (rows = `t*h*w` per grid) and park
/// each slice in its own segment. Any shm failure (e.g. `/dev/shm` full)
/// falls back to inline transport — same policy as Python's
/// `_wrap_shm_or_inline` — so requests degrade to the slow path instead of
/// erroring.
fn park_features_in_shm(features: &[f32], grids: &[[u32; 3]]) -> FeatureStore {
    let total_rows: usize = grids
        .iter()
        .map(|g| g[0] as usize * g[1] as usize * g[2] as usize)
        .sum();
    if total_rows == 0 || !features.len().is_multiple_of(total_rows) {
        // Shape surprise: keep the request alive on the inline path.
        return FeatureStore::Inline(features.to_vec());
    }
    let dim = features.len() / total_rows;
    let mut segments = Vec::with_capacity(grids.len());
    let mut row = 0usize;
    for (item, grid) in grids.iter().enumerate() {
        let rows = grid[0] as usize * grid[1] as usize * grid[2] as usize;
        let slice = &features[row * dim..(row + rows) * dim];
        row += rows;
        match ShmSegment::create(shm_name(item), bytemuck::cast_slice(slice)) {
            Ok(segment) => segments.push(segment),
            Err(error) => {
                tracing::warn!(%error, "mm: shm feature transport failed; falling back to inline");
                return FeatureStore::Inline(features.to_vec());
            }
        }
    }
    FeatureStore::Shm(segments)
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
            let rid = req.rid;
            let event = match process(&self.ctx, &rid, req.work) {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn shm_path(name: &str) -> std::path::PathBuf {
        std::path::Path::new("/dev/shm").join(name)
    }

    /// The segment holds exactly the written bytes, and dropping it unlinks —
    /// the leak guard for results purged before Python ever takes them.
    #[test]
    fn segment_roundtrip_and_drop_unlinks() {
        let name = shm_name(0);
        let payload: Vec<u8> = (0..255u8).collect();
        let segment = ShmSegment::create(name.clone(), &payload).unwrap();
        assert_eq!(std::fs::read(shm_path(&name)).unwrap(), payload);
        drop(segment);
        assert!(!shm_path(&name).exists(), "drop must unlink");
    }

    /// `into_name` transfers the unlink duty to the caller (Python's
    /// `materialize()`), so the segment must survive the handoff.
    #[test]
    fn into_name_disarms_the_unlink() {
        let segment = ShmSegment::create(shm_name(0), &[1, 2, 3]).unwrap();
        let name = segment.into_name();
        assert!(shm_path(&name).exists(), "handoff must not unlink");
        // manual cleanup for the test
        let c = std::ffi::CString::new(format!("/{name}")).unwrap();
        unsafe { libc::shm_unlink(c.as_ptr()) };
    }

    /// Per-item slicing matches the grid row counts, so Python's
    /// `(rows, feature_dim)` reshape of each segment sees its own item only.
    #[test]
    fn park_splits_features_by_grid() {
        // Two items: grids (1,2,2)=4 rows and (1,1,2)=2 rows, dim=3.
        let features: Vec<f32> = (0..18).map(|i| i as f32).collect();
        let grids = [[1, 2, 2], [1, 1, 2]];
        let FeatureStore::Shm(segments) = park_features_in_shm(&features, &grids) else {
            panic!("expected shm store");
        };
        assert_eq!(segments.len(), 2);
        let read = |seg: &ShmSegment| -> Vec<u8> { std::fs::read(shm_path(&seg.name)).unwrap() };
        assert_eq!(
            read(&segments[0]),
            bytemuck::cast_slice::<f32, u8>(&features[..12])
        );
        assert_eq!(
            read(&segments[1]),
            bytemuck::cast_slice::<f32, u8>(&features[12..])
        );
    }

    /// A degenerate shape must degrade to inline, never a shm-side panic.
    #[test]
    fn shape_surprise_falls_back_inline() {
        let features = vec![0.0f32; 7]; // not divisible by 2 rows
        let grids = [[1, 1, 2]];
        assert!(matches!(
            park_features_in_shm(&features, &grids),
            FeatureStore::Inline(_)
        ));
    }
}
