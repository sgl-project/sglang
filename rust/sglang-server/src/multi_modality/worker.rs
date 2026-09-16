//! The worker pool: drain MM requests, run the selected processor, hand the
//! result back as named buffers that ride the ring with the request.

use std::collections::BTreeMap;
use std::sync::Arc;

use sglang_mm::pipeline::{Tensor, TensorData};

use super::encoded::{MRope, MmEncodedEntry, MmEncodedItem, MmMeta, MmMetaValue, MmModality};
use crate::message::buffers::{Buffer, BufferData, BufferStore};
use crate::message::config::MmSpec;
use crate::message::request::{MmRequest, MmWorkItem};
use crate::message::types::TokenIds;
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
    /// The final placeholder-expanded prompt ids.
    pub input_ids: TokenIds,
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
}

impl QwenMmProcessor {
    fn new(spec: MmSpec) -> Result<Self, String> {
        Ok(Self {
            family: sglang_mm::registry::build_pipeline(spec.pipeline)?,
        })
    }
}

impl MmProcessor for QwenMmProcessor {
    fn process(&self, work: MmWorkItem) -> Result<MmProcessOutput, String> {
        let input = super::payload::to_mm_input(work)?;
        let output = sglang_mm::driver::process(self.family.as_ref(), input)?;
        let packed = sglang_mm::qwen_vl::pack_output(output)?;
        let items = packed
            .features
            .into_iter()
            .zip(packed.grids)
            .zip(packed.hashes)
            .zip(packed.offsets)
            .map(|(((feature, [t, h, w]), hash), offsets)| {
                let rows = (t * h * w) as usize;
                let dim = if rows == 0 { 0 } else { feature.len() / rows };
                MmEncodedItem {
                    modality: MmModality::Image,
                    feature: Tensor {
                        shape: vec![rows, dim],
                        data: TensorData::F32(feature),
                    },
                    hash,
                    offsets: vec![offsets],
                    model_specific_data: BTreeMap::from([(
                        "image_grid_thw".to_owned(),
                        MmMetaValue::Ints(vec![t as i64, h as i64, w as i64]),
                    )]),
                }
            })
            .collect();
        Ok(MmProcessOutput {
            input_ids: packed.input_ids,
            result: MmEncodedEntry {
                items,
                token_ids: None,
                mrope: Some(MRope {
                    positions: packed.mrope,
                    delta: packed.mrope_delta,
                }),
            },
        })
    }
}

/// Shared state of the multimodal path, built once at worker startup.
pub struct MmContext {
    pub processor: Arc<dyn MmProcessor>,
    /// Place feature tensors in POSIX shm. Set by the Python launcher
    /// (`RustMmProcessor._use_feature_shm`) exactly when the scheduler broadcasts
    /// across TP ranks and will unwrap `ShmPointerMMData`.
    pub feature_shm: bool,
}

impl MmContext {
    pub fn new(spec: MmSpec) -> Result<Self, String> {
        let feature_shm = spec.feature_shm;
        Ok(Self {
            processor: Arc::new(QwenMmProcessor::new(spec)?),
            feature_shm,
        })
    }

    pub fn with_processor(processor: Arc<dyn MmProcessor>, feature_shm: bool) -> Self {
        Self {
            processor,
            feature_shm,
        }
    }
}

fn tensor_data(data: TensorData) -> BufferData {
    match data {
        TensorData::F32(v) => BufferData::F32(v),
        TensorData::I64(v) => BufferData::I64(v),
        TensorData::Bf16(v) => BufferData::U16(v),
    }
}

/// Lay each item's feature tensor out as `mm.feature.{i}`: in its own shm
/// segment when `shm` is set — the unit Python's `ShmPointerMMData` maps —
/// else inline. Any shm failure (`/dev/shm` full) falls the whole request
/// back to inline, as Python's `_wrap_shm_or_inline` does: degrade to the slow
/// path, never fail the request. `segment_name` names each item's segment.
fn place_features(
    features: Vec<Tensor>,
    shm: bool,
    mut segment_name: impl FnMut(usize) -> String,
) -> Vec<Buffer> {
    let name = |i: usize| format!("mm.feature.{i}");
    let features: Vec<(Vec<usize>, BufferData)> = features
        .into_iter()
        .map(|t| (t.shape, tensor_data(t.data)))
        .collect();
    if shm {
        let parked: Result<Vec<Buffer>, String> = features
            .iter()
            .enumerate()
            .map(|(i, (shape, data))| Buffer::shm(name(i), segment_name(i), shape.clone(), data))
            .collect();
        match parked {
            Ok(buffers) => return buffers,
            Err(error) => {
                tracing::warn!(%error, "mm: shm feature transport failed; falling back to inline");
            }
        }
    }
    features
        .into_iter()
        .enumerate()
        .map(|(i, (shape, data))| Buffer {
            name: name(i),
            shape,
            store: BufferStore::Inline(data),
        })
        .collect()
}

/// The ring's named buffers for one result: the per-item features (see
/// [`place_features`]), the M-RoPE positions, and the `mm.meta` sidecar —
/// always last, so a reader that finds it knows the rest is present.
fn make_buffers(entry: MmEncodedEntry, feature_shm: bool) -> Result<Vec<Buffer>, String> {
    let meta = MmMeta::of(&entry).encode()?;
    let MmEncodedEntry { items, mrope, .. } = entry;
    let features = items.into_iter().map(|item| item.feature).collect();
    let mut buffers = place_features(features, feature_shm, |_| {
        crate::utils::shm::unique_name("mm")
    });
    if let Some(mrope) = mrope {
        let len = mrope.positions.len() / 3;
        buffers.push(Buffer::inline_shaped(
            "mm.mrope",
            vec![3, len],
            mrope.positions,
        )?);
    }
    buffers.push(Buffer::inline("mm.meta", meta));
    Ok(buffers)
}

/// Run the processor for one request. `Ok` returns the final expanded ids and
/// the buffers to ride the ring; `Err` rejects the request back to the client.
fn process(ctx: &MmContext, mut work: MmWorkItem) -> Result<(TokenIds, Vec<Buffer>), String> {
    let caller_hashes = std::mem::take(&mut work.mm_hashes);
    let mut output = ctx.processor.process(work)?;
    output.result.validate(output.input_ids.len())?;
    apply_caller_hashes(
        output.result.items.iter_mut().map(|item| &mut item.hash),
        &caller_hashes,
    );
    let buffers = make_buffers(output.result, ctx.feature_shm)?;
    Ok((output.input_ids, buffers))
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
            let event = match process(&self.ctx, req.work) {
                Ok((input_ids, buffers)) => {
                    tracing::debug!(%rid, tokens = input_ids.len(), "mm: processed");
                    TmEvent::MmEncoded {
                        rid,
                        input_ids,
                        buffers,
                    }
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
    use crate::message::buffers::find;
    use crate::utils::shm::{shm_path, unique_name};

    /// An external-style processor: one image item with a caller-shaped
    /// feature and spans, no M-RoPE.
    struct ExternalProcessor {
        shape: Vec<usize>,
        offsets: Vec<(u32, u32)>,
    }

    impl MmProcessor for ExternalProcessor {
        fn process(&self, work: MmWorkItem) -> Result<MmProcessOutput, String> {
            Ok(MmProcessOutput {
                input_ids: work.input_ids,
                result: MmEncodedEntry {
                    items: vec![MmEncodedItem {
                        modality: MmModality::Image,
                        feature: Tensor {
                            shape: self.shape.clone(),
                            data: TensorData::F32(vec![1.0]),
                        },
                        hash: 7,
                        offsets: self.offsets.clone(),
                        model_specific_data: BTreeMap::from([(
                            "clip_index".to_owned(),
                            MmMetaValue::Int(3),
                        )]),
                    }],
                    token_ids: None,
                    mrope: None,
                },
            })
        }
    }

    /// The built-in Qwen shape: two items with grids, spans and M-RoPE.
    fn qwen_entry() -> MmEncodedEntry {
        let item = |rows: usize, feature: Vec<f32>, grid: [i64; 3], hash, span| MmEncodedItem {
            modality: MmModality::Image,
            feature: Tensor {
                shape: vec![rows, 2],
                data: TensorData::F32(feature),
            },
            hash,
            offsets: vec![span],
            model_specific_data: BTreeMap::from([(
                "image_grid_thw".to_owned(),
                MmMetaValue::Ints(grid.to_vec()),
            )]),
        };
        MmEncodedEntry {
            items: vec![
                item(2, vec![1.0, 2.0, 3.0, 4.0], [1, 2, 1], 11, (1, 2)),
                item(1, vec![5.0, 6.0], [1, 1, 1], 22, (3, 3)),
            ],
            token_ids: None,
            mrope: Some(MRope {
                positions: vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
                delta: -1,
            }),
        }
    }

    fn decoded_meta(buffers: &[Buffer]) -> rmpv::Value {
        let BufferStore::Inline(BufferData::U8(bytes)) = &find(buffers, "mm.meta").unwrap().store
        else {
            panic!("mm.meta must be an inline byte buffer")
        };
        rmpv::decode::read_value(&mut bytes.as_slice()).unwrap()
    }

    /// Inline: one shaped `mm.feature.{i}` per item owning its own tensor, the
    /// M-RoPE positions as `[3, len]`, and the sidecar last, decoding to named
    /// maps with the item metadata in order.
    #[test]
    fn buffers_are_shaped_features_mrope_and_meta_sidecar() {
        let buffers = make_buffers(qwen_entry(), false).unwrap();
        let names: Vec<&str> = buffers.iter().map(|b| b.name.as_str()).collect();
        assert_eq!(
            names,
            ["mm.feature.0", "mm.feature.1", "mm.mrope", "mm.meta"]
        );
        let feature1 = find(&buffers, "mm.feature.1").unwrap();
        assert_eq!(feature1.shape, [1, 2]);
        assert!(
            matches!(&feature1.store, BufferStore::Inline(BufferData::F32(v)) if v == &[5.0, 6.0])
        );
        assert_eq!(find(&buffers, "mm.mrope").unwrap().shape, [3, 3]);

        let meta = decoded_meta(&buffers);
        let get = |m: &rmpv::Value, key: &str| {
            m.as_map()
                .unwrap()
                .iter()
                .find(|(k, _)| k.as_str() == Some(key))
                .map(|(_, v)| v.clone())
                .unwrap()
        };
        let items = get(&meta, "items");
        let items = items.as_array().unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(get(&items[1], "hash").as_u64(), Some(22));
        assert_eq!(get(&items[1], "modality").as_str(), Some("image"));
        assert_eq!(
            get(&items[1], "offsets"),
            rmpv::Value::Array(vec![rmpv::Value::Array(vec![3.into(), 3.into()])])
        );
        assert_eq!(
            get(&get(&items[0], "model_specific_data"), "image_grid_thw"),
            rmpv::Value::Array(vec![1.into(), 2.into(), 1.into()])
        );
        assert_eq!(get(&meta, "mrope_delta").as_i64(), Some(-1));
        assert!(get(&meta, "token_ids").is_nil());
    }

    /// Shm: each item's tensor lands in its own segment holding exactly its
    /// bytes, shaped as the tensor; the segment lives as long as the buffer.
    #[test]
    fn shm_places_each_item_in_its_own_segment() {
        let names: Vec<String> = (0..2).map(|_| unique_name("test")).collect();
        let namer = names.clone();
        let features: Vec<Tensor> = qwen_entry().items.into_iter().map(|i| i.feature).collect();
        let expected: Vec<Vec<f32>> = features
            .iter()
            .map(|t| match &t.data {
                TensorData::F32(v) => v.clone(),
                _ => unreachable!(),
            })
            .collect();
        let buffers = place_features(features, true, move |i| namer[i].clone());
        for (i, name) in names.iter().enumerate() {
            assert!(matches!(buffers[i].store, BufferStore::Shm { .. }));
            assert_eq!(buffers[i].shape, [[2, 2], [1, 2]][i]);
            let bytes = std::fs::read(shm_path(name)).unwrap();
            assert_eq!(bytes, bytemuck::cast_slice::<f32, u8>(&expected[i]));
        }
        drop(buffers);
        assert!(names.iter().all(|n| !shm_path(n).exists()), "drop unlinks");
    }

    /// A segment that cannot be created degrades the whole request to inline
    /// rather than rejecting it (Python's `_wrap_shm_or_inline` parity).
    #[test]
    fn shm_failure_falls_back_to_inline() {
        let features = qwen_entry().items.into_iter().map(|i| i.feature).collect();
        let buffers = place_features(features, true, |_| "bad\0name".into());
        assert_eq!(buffers.len(), 2);
        assert!(
            buffers
                .iter()
                .all(|b| matches!(b.store, BufferStore::Inline(_)))
        );
    }

    /// An external processor's result rides the same buffers: its tensor's
    /// dtype and shape are kept, the caller hash override applies, and the
    /// sidecar carries its scalar metadata.
    #[test]
    fn external_processor_result_becomes_buffers() {
        let ctx = MmContext::with_processor(
            Arc::new(ExternalProcessor {
                shape: vec![1],
                offsets: vec![(1, 1)],
            }),
            false,
        );
        let work = MmWorkItem {
            input_ids: vec![1, 2],
            mm_hashes: vec!["2a".to_owned()],
            ..Default::default()
        };
        let (input_ids, buffers) = process(&ctx, work).unwrap();
        assert_eq!(input_ids, [1, 2]);
        assert_eq!(find(&buffers, "mm.feature.0").unwrap().shape, [1]);
        assert!(find(&buffers, "mm.mrope").is_none());
        let meta = decoded_meta(&buffers);
        let text = format!("{meta}");
        assert!(text.contains("42"), "caller hash 0x2a applied: {text}");
        assert!(text.contains("clip_index"), "{text}");
    }

    /// Malformed results are rejected before any buffer exists, so nothing
    /// reaches the ring for them.
    #[test]
    fn malformed_processor_results_are_rejected() {
        for (shape, offsets) in [
            (vec![2], vec![(1, 1)]),
            (vec![usize::MAX, 2], vec![(1, 1)]),
            (vec![1], vec![(2, 1)]),
            (vec![1], vec![(1, 2)]),
        ] {
            let ctx =
                MmContext::with_processor(Arc::new(ExternalProcessor { shape, offsets }), false);
            let work = MmWorkItem {
                input_ids: vec![1, 2],
                ..Default::default()
            };
            assert!(process(&ctx, work).is_err());
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
