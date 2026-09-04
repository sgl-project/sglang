//! The worker pool: drain MM requests, run the `sglang-mm` pipeline, park
//! the result buffers.

use std::sync::Arc;

use super::result_store::{FeatureStore, MmEncodedEntry, MmResultStore, park_features_in_shm};
use crate::message::config::MmSpec;
use crate::message::ids::Rid;
use crate::message::request::MmRequest;
use crate::tokenizer_manager::tokenizer::TextTokenizer;
use crate::tokenizer_manager::wiring::TmEvent;
use crate::utils::runtime::Runnable;

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

/// Shared state of the mm path, built once at `start_mm_workers`.
pub struct MmContext {
    pub family: Box<dyn sglang_mm::pipeline::MmFamilyProcessor>,
    /// `None` under `skip_tokenizer_init` (requests must carry `input_ids`).
    pub tokenizer: Option<Arc<dyn TextTokenizer>>,
    pub results: MmResultStore,
    /// Park feature buffers in POSIX shm. Set by the Python launcher
    /// (`RustMmProcessor._use_feature_shm`) exactly when the scheduler broadcasts
    /// across TP ranks and will unwrap `ShmPointerMMData`.
    pub feature_shm: bool,
}

impl MmContext {
    pub fn new(
        spec: MmSpec,
        tokenizer: Option<Arc<dyn TextTokenizer>>,
        results: MmResultStore,
    ) -> Result<Self, String> {
        Ok(Self {
            family: sglang_mm::registry::build_pipeline(spec.pipeline)?,
            tokenizer,
            results,
            feature_shm: spec.feature_shm,
        })
    }
}

/// Run the pipeline for one request. `Ok` returns the final expanded ids, the
/// buffers already parked; `Err` rejects the request back to the client.
fn process(
    ctx: &MmContext,
    rid: &Rid,
    mut work: crate::message::request::MmWorkItem,
) -> Result<Vec<i32>, String> {
    let caller_hashes = std::mem::take(&mut work.mm_hashes);
    let input = super::payload::to_mm_input(work)?;
    let output = sglang_mm::driver::process(ctx.family.as_ref(), input, |text| {
        let tokenizer = ctx.tokenizer.as_ref().ok_or_else(|| {
            "skip_tokenizer_init is set: multimodal text prompts require input_ids".to_string()
        })?;
        tokenizer.encode(text).map_err(|error| error.to_string())
    })?;
    // TODO(mm-families): the one family-specific call in this worker — dispatch
    // on the spec's `family` (as `registry::build_pipeline` does) once a
    // second family lands.
    let mut packed = sglang_mm::qwen_vl::pack_output(output)?;
    apply_caller_hashes(&mut packed.hashes, &caller_hashes);
    let features = if ctx.feature_shm {
        park_features_in_shm(&packed.features, &packed.grids)
    } else {
        FeatureStore::Inline(packed.features)
    };
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

/// Boot-time wiring of the MM path, held privately by the `Runtime` for the
/// late pool spawn (`Runtime::start_mm_workers`, once Python has resolved
/// the spec).
pub struct MmWiring {
    /// Requests parked in `Encoding`, drained by the worker pool. Stays empty
    /// for non-multimodal models — nothing routes to it.
    pub mm_rx: flume::Receiver<MmRequest>,
    /// Back-channel for the workers' `MmEncoded` / `MmFailed` into the
    /// to-scheduler loop.
    pub tm_tx: flume::Sender<TmEvent>,
    /// The loaded tokenizer, shared with the tokenizer pool (`None` under
    /// `skip_tokenizer_init`).
    pub tokenizer: Option<Arc<dyn TextTokenizer>>,
}

/// One MM worker, spawned via `Runtime::start_mm_workers` (which owns the
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
    /// Drain until the mm channel closes (to-scheduler drops its sender on
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
                return; // to-scheduler gone: shutdown
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
}
