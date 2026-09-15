//! The worker pool: drain MM requests, run the `sglang-mm` pipeline, park
//! the result buffers.

use std::sync::Arc;

use super::result_store::{
    FeatureStore, MmEncodedEntry, MmResultStore, QwenMmEncodedEntry, park_features_in_shm,
};
use crate::message::config::MmSpec;
use crate::message::ids::Rid;
use crate::message::request::{MmRequest, MmWorkItem};
use crate::tokenizer_manager::wiring::TmEvent;
use crate::utils::runtime::Runnable;

/// Python parity: caller hashes override the computed ones so an external
/// router's keys align with the prefix cache. A length mismatch or malformed
/// entry warns and keeps the computed hash — never blocks the request.
fn apply_caller_hashes<'a>(hashes: impl ExactSizeIterator<Item = &'a mut u64>, caller: &[String]) {
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
    for (hash, entry) in hashes.zip(caller) {
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

/// Complete result of one multimodal processor invocation.
pub struct MmProcessOutput {
    pub input_ids: Vec<i32>,
    pub result: MmEncodedEntry,
}

/// Multimodal processor shared by built-in and external implementations.
/// Implementations run on the fixed Rust worker pool and must not retain
/// request-scoped Python objects. The work item arrives tokenized (the FSM
/// runs `Tokenizing` before `Encoding` for a text prompt), so a processor
/// never tokenizes: it expands placeholders in ids.
pub trait MmProcessor: Send + Sync {
    fn process(&self, work: MmWorkItem) -> Result<MmProcessOutput, String>;
}

struct QwenMmProcessor {
    family: Box<dyn sglang_mm::pipeline::MmFamilyProcessor>,
    feature_shm: bool,
}

impl QwenMmProcessor {
    fn new(spec: MmSpec) -> Result<Self, String> {
        Ok(Self {
            family: sglang_mm::registry::build_pipeline(spec.pipeline)?,
            feature_shm: spec.feature_shm,
        })
    }
}

impl MmProcessor for QwenMmProcessor {
    fn process(&self, work: MmWorkItem) -> Result<MmProcessOutput, String> {
        let input = super::payload::to_mm_input(work)?;
        let output = sglang_mm::driver::process(self.family.as_ref(), input)?;
        let drain = sglang_mm::qwen_vl::pack_output(output)?;
        let features = if self.feature_shm {
            park_features_in_shm(&drain.features, &drain.grids)
        } else {
            FeatureStore::Inline(drain.features)
        };
        Ok(MmProcessOutput {
            input_ids: drain.input_ids,
            result: MmEncodedEntry::Qwen(QwenMmEncodedEntry {
                features,
                grids: drain.grids,
                hashes: drain.hashes,
                offsets: drain.offsets,
                mrope: drain.mrope,
                mrope_delta: drain.mrope_delta,
            }),
        })
    }
}

/// Shared state of the multimodal path, built once at worker startup.
pub struct MmContext {
    pub processor: Arc<dyn MmProcessor>,
    pub results: MmResultStore,
}

impl MmContext {
    pub fn new(spec: MmSpec, results: MmResultStore) -> Result<Self, String> {
        Ok(Self {
            processor: Arc::new(QwenMmProcessor::new(spec)?),
            results,
        })
    }

    pub fn with_processor(processor: Arc<dyn MmProcessor>, results: MmResultStore) -> Self {
        Self { processor, results }
    }
}

/// Run the processor for one request. `Ok` returns the final expanded ids,
/// the buffers already parked; `Err` rejects the request back to the client.
fn process(ctx: &MmContext, rid: &Rid, mut work: MmWorkItem) -> Result<Vec<i32>, String> {
    let caller_hashes = std::mem::take(&mut work.mm_hashes);
    let mut output = ctx.processor.process(work)?;
    match &mut output.result {
        MmEncodedEntry::Qwen(entry) => apply_caller_hashes(entry.hashes.iter_mut(), &caller_hashes),
        MmEncodedEntry::External(entry) => {
            entry.validate(output.input_ids.len())?;
            apply_caller_hashes(
                entry.items.iter_mut().map(|item| &mut item.hash),
                &caller_hashes,
            );
        }
    }
    ctx.results.park(rid.as_str().to_owned(), output.result);
    Ok(output.input_ids)
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
    use crate::{
        ExternalMmEncodedEntry, ExternalMmItem, MmModality, MmTokenIds, Tensor, TensorData,
    };

    struct ExternalProcessor {
        shape: Vec<usize>,
        offsets: Vec<(u32, u32)>,
    }

    impl MmProcessor for ExternalProcessor {
        fn process(&self, work: MmWorkItem) -> Result<MmProcessOutput, String> {
            Ok(MmProcessOutput {
                input_ids: work.input_ids,
                result: MmEncodedEntry::External(ExternalMmEncodedEntry {
                    items: vec![ExternalMmItem {
                        modality: MmModality::Image,
                        feature: Tensor {
                            shape: self.shape.clone(),
                            data: TensorData::F32(vec![1.0]),
                        },
                        hash: 7,
                        offsets: self.offsets.clone(),
                        model_specific_data: Default::default(),
                    }],
                    token_ids: MmTokenIds::default(),
                }),
            })
        }
    }

    #[test]
    fn external_processor_result_reaches_store() {
        let results = MmResultStore::default();
        let processor = ExternalProcessor {
            shape: vec![1],
            offsets: vec![(1, 1)],
        };
        let ctx = MmContext::with_processor(Arc::new(processor), results.clone());
        let rid = Rid::from_client("external");
        let work = MmWorkItem {
            input_ids: vec![1, 2],
            mm_hashes: vec!["2a".to_owned()],
            ..Default::default()
        };

        assert_eq!(process(&ctx, &rid, work).unwrap(), [1, 2]);
        let Some(MmEncodedEntry::External(entry)) = results.take(rid.as_str()) else {
            panic!("external processor must park an external entry")
        };
        assert_eq!(entry.items.len(), 1);
        assert_eq!(entry.items[0].hash, 0x2a);
    }

    #[test]
    fn malformed_processor_results_are_rejected_before_parking() {
        for (shape, offsets) in [
            (vec![2], vec![(1, 1)]),
            (vec![usize::MAX, 2], vec![(1, 1)]),
            (vec![1], vec![(2, 1)]),
            (vec![1], vec![(1, 2)]),
        ] {
            let results = MmResultStore::default();
            let processor = ExternalProcessor { shape, offsets };
            let ctx = MmContext::with_processor(Arc::new(processor), results.clone());
            let rid = Rid::from_client("invalid");
            let work = MmWorkItem {
                input_ids: vec![1, 2],
                ..Default::default()
            };
            assert!(process(&ctx, &rid, work).is_err());
            assert!(results.take(rid.as_str()).is_none());
        }
    }

    /// Caller hashes override computed ones; mismatched lengths and malformed
    /// entries fall back per item, never reject (Python parity).
    #[test]
    fn caller_hashes_override_with_fallback() {
        let mut hashes = vec![1, 2, 3];
        apply_caller_hashes(hashes.iter_mut(), &[]);
        assert_eq!(hashes, [1, 2, 3]);

        apply_caller_hashes(hashes.iter_mut(), &["ff".into()]); // length mismatch
        assert_eq!(hashes, [1, 2, 3]);

        apply_caller_hashes(
            hashes.iter_mut(),
            &["ff".into(), "not-hex".into(), "0x10".into()],
        );
        assert_eq!(hashes, [0xff, 2, 0x10]);
    }

    /// A full SHA-256 (what routers send) keeps its low 64 bits rather than
    /// falling back, so the pad value matches Python's wide `int`.
    #[test]
    fn caller_hashes_accept_arbitrary_width() {
        let sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        let mut hashes = vec![1];
        apply_caller_hashes(hashes.iter_mut(), &[sha256.into()]);
        assert_eq!(hashes, [0xa495991b7852b855]);
        assert_eq!(hashes[0] % (1 << 30), 944_945_237); // int(sha256, 16) % (1 << 30)

        // Width alone is never malformed; a non-hex digit still is.
        assert_eq!(parse_caller_hash(&"f".repeat(64)), Some(u64::MAX));
        assert_eq!(parse_caller_hash("0x"), None);
        assert_eq!(parse_caller_hash(""), None);
    }
}
