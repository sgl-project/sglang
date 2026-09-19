//! The one result shape every multimodal processor produces — the built-in
//! Qwen pipeline and external model packages alike — and its msgpack
//! sidecar. The worker turns it into the ring's named buffers (see
//! `worker::make_buffers`): one shaped feature buffer per item, the M-RoPE
//! positions, and `mm.meta`, the sidecar carrying everything small and
//! structured.

use std::collections::BTreeMap;

use serde::Serialize;
use sglang_mm::pipeline::Tensor;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum MmModality {
    Image,
    Video,
    Audio,
}

/// Placeholder and boundary tokens consumed by `MultimodalProcessorOutput`.
/// External processors fill these; the built-in path leaves them `None` and
/// Python reads them from its spec.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct MmTokenIds {
    pub im_token_id: Option<i64>,
    pub im_start_id: Option<i64>,
    pub im_end_id: Option<i64>,
    pub video_token_id: Option<i64>,
    pub audio_token_id: Option<i64>,
    pub audio_start_id: Option<i64>,
    pub audio_end_id: Option<i64>,
}

/// A processor-owned per-item attribute: a scalar (a clip index, a count) or
/// a short list (a `[t, h, w]` grid).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum MmMetaValue {
    Int(i64),
    Ints(Vec<i64>),
}

/// One media item: its feature tensor plus the metadata Python's
/// `MultimodalDataItem` needs.
pub struct MmEncodedItem {
    pub modality: MmModality,
    pub feature: Tensor,
    /// Content hash of the raw source bytes, or the caller's `mm_hashes`
    /// override; precomputed so draining never hashes.
    pub hash: u64,
    /// Inclusive token spans in the expanded prompt.
    pub offsets: Vec<(u32, u32)>,
    pub model_specific_data: BTreeMap<String, MmMetaValue>,
}

/// M-RoPE positions for the expanded prompt, row-major `[3, input_len]`, and
/// the delta added to the plain sequence position during decoding.
pub struct MRope {
    pub positions: Vec<i64>,
    pub delta: i64,
}

/// What one processor invocation produced for the scheduler, besides the
/// expanded `input_ids`.
pub struct MmEncodedEntry {
    pub items: Vec<MmEncodedItem>,
    pub token_ids: Option<MmTokenIds>,
    pub mrope: Option<MRope>,
}

impl MmEncodedEntry {
    /// Reject a result whose tensors or spans cannot be what the scheduler
    /// expects, before anything leaves the worker.
    pub(super) fn validate(&self, input_len: usize) -> Result<(), String> {
        for (index, item) in self.items.iter().enumerate() {
            let elements = item
                .feature
                .shape
                .iter()
                .try_fold(1usize, |size, &dim| size.checked_mul(dim));
            if elements != Some(item.feature.data.len()) {
                return Err(format!(
                    "multimodal item {index}: feature shape does not match its data"
                ));
            }
            if item
                .offsets
                .iter()
                .any(|&(start, end)| start > end || end as usize >= input_len)
            {
                return Err(format!(
                    "multimodal item {index}: token offsets are outside the expanded prompt"
                ));
            }
        }
        if let Some(mrope) = &self.mrope
            && mrope.positions.len() != 3 * input_len
        {
            return Err("multimodal M-RoPE positions do not match the expanded prompt".into());
        }
        Ok(())
    }
}

/// The `mm.meta` sidecar, msgpack-encoded with named maps so Python decodes it
/// straight into dicts: per-item metadata in `items` order (the item's feature
/// is the `mm.feature.{i}` buffer), the token ids, and the M-RoPE delta (its
/// positions are the `mm.mrope` buffer).
#[derive(Serialize)]
pub(super) struct MmMeta<'a> {
    pub items: Vec<MmItemMeta<'a>>,
    pub token_ids: Option<&'a MmTokenIds>,
    pub mrope_delta: Option<i64>,
}

#[derive(Serialize)]
pub(super) struct MmItemMeta<'a> {
    pub modality: MmModality,
    pub hash: u64,
    pub offsets: &'a [(u32, u32)],
    pub model_specific_data: &'a BTreeMap<String, MmMetaValue>,
}

impl<'a> MmMeta<'a> {
    pub fn of(entry: &'a MmEncodedEntry) -> Self {
        Self {
            items: entry
                .items
                .iter()
                .map(|item| MmItemMeta {
                    modality: item.modality,
                    hash: item.hash,
                    offsets: &item.offsets,
                    model_specific_data: &item.model_specific_data,
                })
                .collect(),
            token_ids: entry.token_ids.as_ref(),
            mrope_delta: entry.mrope.as_ref().map(|m| m.delta),
        }
    }

    pub fn encode(&self) -> Result<Vec<u8>, String> {
        rmp_serde::to_vec_named(self).map_err(|e| format!("mm.meta encode: {e}"))
    }
}
