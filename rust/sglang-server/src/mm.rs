//! Multimodal worker pool.
//!
//! Rust threads drain requests parked in `Encoding` and run the `sglang-mm`
//! pipeline registered by `Server.start_mm_workers` (decode → preprocess →
//! placeholder expansion → M-RoPE, GIL-free). Each worker parks the result
//! buffers in the rid-keyed [`Sidecar`] and returns only the expanded ids;
//! Python attaches the buffers at drain time (`Server.take_mm`). Inputs the
//! pipeline cannot serve are rejected to the client — no Python fallback.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::message::MmRequest;
use crate::runtime::Runnable;
use crate::tokenizer::TextTokenizer;
use crate::tokenizer_manager::TmEvent;

/// A named POSIX shared-memory segment owning its name: dropped → unlinked.
///
/// Written by an MM worker so the TP broadcast carries a ~100-byte
/// `ShmPointerMMData` stub instead of the ~20 MB feature tensor, and every
/// rank maps it in parallel. Python's `materialize()` unlinks after cloning;
/// this `Drop` covers the paths where the buffers never reach Python (aborted
/// while parked, late result purged).
pub struct ShmSegment {
    name: String,
}

impl ShmSegment {
    /// Create `/dev/shm/{name}` holding exactly `bytes`. No leading slash —
    /// the name must suit Python's `SharedMemory(name=…)` (shm_open adds one).
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

    /// Hand the segment — and the duty to unlink — to the caller (Python, at
    /// drain time).
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

/// Unique segment names: the pid separates server restarts (a crash can leak
/// segments under the old pid), the counter separates results within one.
fn shm_name(item: usize) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("sglmm-{}-{n}-{item}", std::process::id())
}

/// Python parity: caller hashes override the computed ones so an external
/// router's keys align with the prefix cache. A length mismatch or malformed
/// entry warns and keeps the computed hash — never blocks the request.
fn apply_caller_hashes(hashes: &mut [u64], caller: &[String]) {
    if caller.is_empty() {
        return;
    }
    if caller.len() != hashes.len() {
        tracing::warn!(
            caller = caller.len(),
            items = hashes.len(),
            "mm_hashes length != mm item count; ignoring caller hashes"
        );
        return;
    }
    for (hash, entry) in hashes.iter_mut().zip(caller) {
        match parse_caller_hash(entry) {
            Some(v) => *hash = v,
            None => tracing::warn!(%entry, "malformed mm_hashes entry; keeping computed hash"),
        }
    }
}

/// Hex of any width, as Python's `int(hex_hash, 16)` takes it (a full SHA-256
/// being the common case), keeping the low 64 bits — only the low 30 are
/// observable, through `_compute_pad_value`.
fn parse_caller_hash(entry: &str) -> Option<u64> {
    let hex = entry.strip_prefix("0x").unwrap_or(entry);
    if hex.is_empty() || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    u64::from_str_radix(&hex[hex.len().saturating_sub(16)..], 16).ok()
}

/// One parked result: the buffers the drain-time Python adapter needs (the
/// expanded `input_ids` travel separately, via `TmEvent::MmEncoded`). The qwen
/// drain shape (`sglang_mm::qwen_vl::pack_drain`); generalizes to a
/// named-tensor handoff once a family needs a different one.
pub struct MmSidecarEntry {
    pub features: FeatureStore,
    pub grids: Vec<[u32; 3]>,
    pub hashes: Vec<u64>,
    pub offsets: Vec<(u32, u32)>,
    pub mrope: Vec<i64>,
    pub mrope_delta: i64,
}

/// Where a result's feature buffers live between worker and drain.
pub enum FeatureStore {
    /// In-process; the drain wraps them zero-copy. Single-rank serving, or the
    /// shm fallback. Under TP the whole buffer would ride `broadcast_pyobj`.
    Inline(Vec<f32>),
    /// One POSIX segment per item, written by the worker; only the names cross
    /// ranks. See [`ShmSegment`].
    Shm(Vec<ShmSegment>),
}

/// Results parked between a worker's `MmEncoded` and the scheduler drain, keyed
/// by rid. Owns the lifecycle so entries never leak: [`park`](Self::park)
/// strictly before `MmEncoded`, [`take`](Self::take) at the drain,
/// [`purge`](Self::purge) for requests that die while parked.
#[derive(Clone, Default)]
pub struct Sidecar(Arc<Mutex<HashMap<String, MmSidecarEntry>>>);

impl Sidecar {
    pub fn park(&self, rid: String, entry: MmSidecarEntry) {
        self.0.lock().unwrap().insert(rid, entry);
    }
    pub fn take(&self, rid: &str) -> Option<MmSidecarEntry> {
        self.0.lock().unwrap().remove(rid)
    }
    pub fn purge(&self, rid: &str) {
        self.0.lock().unwrap().remove(rid);
    }
}

/// Shared state of the mm path, built once at `start_mm_workers`.
pub struct Context {
    pub family: Box<dyn sglang_mm::pipeline::MmFamilyProcessor>,
    /// `None` under `skip_tokenizer_init` (requests must carry `input_ids`).
    pub tokenizer: Option<Arc<dyn TextTokenizer>>,
    pub sidecar: Sidecar,
    /// Park feature buffers in POSIX shm. Set by the Python launcher
    /// (`NativeMmHost._use_feature_shm`) exactly when the scheduler broadcasts
    /// across TP ranks and will unwrap `ShmPointerMMData`.
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

/// Run the pipeline for one request. `Ok` returns the final expanded ids, the
/// buffers already parked; `Err` rejects the request back to the client.
fn process(
    ctx: &Context,
    rid: &crate::ids::Rid,
    mut work: crate::message::MmWorkItem,
) -> Result<Vec<i32>, String> {
    let caller_hashes = std::mem::take(&mut work.mm_hashes);
    let input = crate::message::mm_payload::to_mm_input(work)?;
    let output = sglang_mm::driver::process(ctx.family.as_ref(), input, |text| {
        let tokenizer = ctx.tokenizer.as_ref().ok_or_else(|| {
            "skip_tokenizer_init is set: multimodal text prompts require input_ids".to_string()
        })?;
        tokenizer.encode(text).map_err(|error| error.to_string())
    })?;
    let mut drain = sglang_mm::qwen_vl::pack_drain(output)?;
    apply_caller_hashes(&mut drain.hashes, &caller_hashes);
    let features = if ctx.feature_shm {
        park_features_in_shm(&drain.features, &drain.grids)
    } else {
        FeatureStore::Inline(drain.features)
    };
    ctx.sidecar.park(
        rid.as_str().to_owned(),
        MmSidecarEntry {
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

/// Split the flat feature buffer per item (`t*h*w` rows per grid) and park each
/// slice in its own segment. Any shm failure (`/dev/shm` full, odd shape) falls
/// back to inline, as Python's `_wrap_shm_or_inline` does: degrade to the slow
/// path, never fail the request.
fn park_features_in_shm(features: &[f32], grids: &[[u32; 3]]) -> FeatureStore {
    let total_rows: usize = grids
        .iter()
        .map(|g| g[0] as usize * g[1] as usize * g[2] as usize)
        .sum();
    if total_rows == 0 || !features.len().is_multiple_of(total_rows) {
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

/// One MM worker, spawned via `Runtime::spawn_mm_pool` (which owns the
/// pinning policy for this pool — see its docs).
pub struct MmWorker {
    rx: flume::Receiver<MmRequest>,
    tm: flume::Sender<TmEvent>,
    ctx: Arc<Context>,
}

impl MmWorker {
    pub fn new(
        rx: flume::Receiver<MmRequest>,
        tm: flume::Sender<TmEvent>,
        ctx: Arc<Context>,
    ) -> Self {
        Self { rx, tm, ctx }
    }
}

impl Runnable for MmWorker {
    /// Drain until the mm channel closes (tm-ingress drops its sender on
    /// shutdown). One request at a time, so the pool size bounds MM
    /// concurrency; an error rejects the request back to the client.
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

    /// Caller hashes override computed ones; mismatched lengths and malformed
    /// entries fall back per item, never reject (Python parity).
    #[test]
    fn caller_hashes_override_with_fallback() {
        let mut hashes = vec![1, 2, 3];
        apply_caller_hashes(&mut hashes, &[]);
        assert_eq!(hashes, [1, 2, 3]);

        apply_caller_hashes(&mut hashes, &["ff".into()]); // length mismatch
        assert_eq!(hashes, [1, 2, 3]);

        apply_caller_hashes(&mut hashes, &["ff".into(), "not-hex".into(), "0x10".into()]);
        assert_eq!(hashes, [0xff, 2, 0x10]);
    }

    /// A full SHA-256 (what routers send) keeps its low 64 bits rather than
    /// falling back, so the pad value matches Python's wide `int`.
    #[test]
    fn caller_hashes_accept_arbitrary_width() {
        let sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        let mut hashes = vec![1];
        apply_caller_hashes(&mut hashes, &[sha256.into()]);
        assert_eq!(hashes, [0xa495991b7852b855]);
        assert_eq!(hashes[0] % (1 << 30), 944_945_237); // int(sha256, 16) % (1 << 30)

        // Width alone is never malformed; a non-hex digit still is.
        assert_eq!(parse_caller_hash(&"f".repeat(64)), Some(u64::MAX));
        assert_eq!(parse_caller_hash("0x"), None);
        assert_eq!(parse_caller_hash(""), None);
    }

    fn shm_path(name: &str) -> std::path::PathBuf {
        std::path::Path::new("/dev/shm").join(name)
    }

    /// The segment holds exactly the written bytes and dropping it unlinks —
    /// the leak guard for results purged before Python takes them.
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

    /// Per-item slicing follows the grid row counts, so Python's
    /// `(rows, feature_dim)` reshape of a segment sees only its own item.
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
