//! Multimodal worker pool.
//!
//! Rust threads drain requests parked in `Encoding` and run the `sglang-mm`
//! pipeline registered by `Server.start_mm_workers` (decode → preprocess →
//! placeholder expansion → M-RoPE, GIL-free). Each worker parks the result
//! buffers in the rid-keyed [`MmResultStore`] and returns only the expanded ids;
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
/// Written by an MM worker — the single feature transport across topologies;
/// Python decides at the drain who materializes it. Python unlinks once it
/// owns the buffers; this `Drop` covers the paths where they never reach
/// Python (aborted while parked, late result purged).
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
/// expanded `input_ids` travel separately, via `TmEvent::MmEncoded`).
///
/// TODO(mm-families): these fields are the shape the only current family
/// (qwen_vl) produces; generalize to a named-tensor handoff when a family
/// needs a different one.
pub struct MmEncodedEntry {
    /// One POSIX segment per item; only names cross the drain. See
    /// [`ShmSegment`].
    pub features: Vec<ShmSegment>,
    /// Per item `[t, h, w]` patch grid.
    pub grids: Vec<[u32; 3]>,
    pub hashes: Vec<u64>,
    /// Per item inclusive token range in the expanded prompt.
    pub offsets: Vec<(u32, u32)>,
    /// Flattened row-major `[3, input_len]` M-RoPE positions.
    pub mrope: Vec<i64>,
    pub mrope_delta: i64,
}

/// Results parked between a worker's `MmEncoded` and the scheduler drain, keyed
/// by rid. Owns the lifecycle so entries never leak: [`park`](Self::park)
/// strictly before `MmEncoded`, [`take`](Self::take) at the drain,
/// [`purge`](Self::purge) for requests that die while parked.
#[derive(Clone, Default)]
pub struct MmResultStore(Arc<Mutex<HashMap<String, MmEncodedEntry>>>);

impl MmResultStore {
    pub fn park(&self, rid: String, entry: MmEncodedEntry) {
        self.0.lock().unwrap().insert(rid, entry);
    }
    pub fn take(&self, rid: &str) -> Option<MmEncodedEntry> {
        self.0.lock().unwrap().remove(rid)
    }
    pub fn purge(&self, rid: &str) {
        self.0.lock().unwrap().remove(rid);
    }
}

/// Boot-time wiring of the MM path, held privately by the `Runtime` for the
/// late pool spawn (`Runtime::start_mm_workers`, once Python has resolved
/// the spec).
pub struct MmWiring {
    /// Requests parked in `Encoding`, drained by the worker pool. Stays empty
    /// for non-multimodal models — nothing routes to it.
    pub mm_rx: flume::Receiver<MmRequest>,
    /// Back-channel for the workers' `MmEncoded` / `MmFailed` into tm-ingress.
    pub tm_tx: flume::Sender<TmEvent>,
    /// The loaded tokenizer, shared with the tokenizer pool (`None` under
    /// `skip_tokenizer_init`).
    pub tokenizer: Option<Arc<dyn TextTokenizer>>,
}

/// Shared state of the mm path, built once at `start_mm_workers`.
pub struct MmContext {
    pub family: Box<dyn sglang_mm::pipeline::MmFamilyProcessor>,
    /// `None` under `skip_tokenizer_init` (requests must carry `input_ids`).
    pub tokenizer: Option<Arc<dyn TextTokenizer>>,
    pub results: MmResultStore,
}

impl MmContext {
    pub fn new(
        spec_json: &str,
        tokenizer: Option<Arc<dyn TextTokenizer>>,
        results: MmResultStore,
    ) -> Result<Self, String> {
        Ok(Self {
            family: sglang_mm::registry::pipeline_from_spec(spec_json)?,
            tokenizer,
            results,
        })
    }
}

/// Run the pipeline for one request. `Ok` returns the final expanded ids, the
/// buffers already parked; `Err` rejects the request back to the client.
fn process(
    ctx: &MmContext,
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
    // TODO(mm-families): the one family-specific call in this worker — dispatch
    // on the spec's `family` (as `registry::pipeline_from_spec` does) once a
    // second family lands.
    let mut packed = sglang_mm::qwen_vl::pack_output(output)?;
    apply_caller_hashes(&mut packed.hashes, &caller_hashes);
    let features = park_features_in_shm(packed.features)?;
    ctx.results.park(
        rid.as_str().to_owned(),
        MmEncodedEntry {
            features,
            grids: packed.grids,
            hashes: packed.hashes,
            offsets: packed.offsets,
            mrope: packed.mrope,
            mrope_delta: packed.mrope_delta,
        },
    );
    Ok(packed.input_ids)
}

/// Park each item's feature buffer in its own POSIX segment. A failure — in
/// practice `/dev/shm` exhaustion (the launcher warns when it looks small) —
/// rejects the request; there is no inline fallback.
fn park_features_in_shm(items: Vec<Vec<f32>>) -> Result<Vec<ShmSegment>, String> {
    items
        .into_iter()
        .enumerate()
        .map(|(item, features)| {
            ShmSegment::create(shm_name(item), bytemuck::cast_slice(&features)).map_err(|error| {
                format!("mm feature transport: {error}; is /dev/shm mounted large enough?")
            })
        })
        .collect()
}

/// One MM worker, spawned via `Runtime::spawn_mm_pool` (which owns the
/// pinning policy for this pool — see its docs).
pub struct MmWorker {
    mm_rx: flume::Receiver<MmRequest>,
    tm_tx: flume::Sender<TmEvent>,
    ctx: Arc<MmContext>,
}

impl MmWorker {
    pub fn new(
        mm_rx: flume::Receiver<MmRequest>,
        tm_tx: flume::Sender<TmEvent>,
        ctx: Arc<MmContext>,
    ) -> Self {
        Self { mm_rx, tm_tx, ctx }
    }
}

impl Runnable for MmWorker {
    /// Drain until the mm channel closes (tm-ingress drops its sender on
    /// shutdown). One request at a time, so the pool size bounds MM
    /// concurrency; an error rejects the request back to the client.
    fn run(self) {
        while let Ok(req) = self.mm_rx.recv() {
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
            if self.tm_tx.send(event).is_err() {
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

    /// One segment per item holding exactly that item's bytes, so Python's
    /// `(rows, feature_dim)` reshape of a segment sees only its own item.
    #[test]
    fn park_creates_one_segment_per_item() {
        let items = vec![
            (0..12).map(|i| i as f32).collect::<Vec<_>>(),
            (12..18).map(|i| i as f32).collect::<Vec<_>>(),
        ];
        let expected: Vec<Vec<u8>> = items
            .iter()
            .map(|v| bytemuck::cast_slice::<f32, u8>(v).to_vec())
            .collect();
        let segments = park_features_in_shm(items).unwrap();
        assert_eq!(segments.len(), 2);
        for (segment, bytes) in segments.iter().zip(&expected) {
            assert_eq!(&std::fs::read(shm_path(&segment.name)).unwrap(), bytes);
        }
    }
}
