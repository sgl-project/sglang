//! The model-family seam of the server MM pipeline.
//!
//! Design rule: **families produce data, the driver owns control flow.** A
//! family never sees the request loop, the thread pool, or the failure
//! protocol — it implements [`MmFamilyProcessor`], turning decoded media into
//! named tensors and describing its prompt geometry as a [`TokenLayout`]
//! value. `driver::process` applies the layout mechanically, so expansion,
//! per-item offsets, and position inputs all derive from one declarative
//! structure and every family gets identical failure semantics for free.

/// Typed tensor payload. Grows a variant per dtype actually produced by a
/// family — not speculatively.
pub enum TensorData {
    F32(Vec<f32>),
    I64(Vec<i64>),
}

impl TensorData {
    /// Raw little-endian bytes, for content hashing and zero-copy handoff.
    pub fn as_bytes(&self) -> &[u8] {
        // Safety: f32/i64 are plain-old-data with no padding.
        unsafe {
            match self {
                TensorData::F32(v) => std::slice::from_raw_parts(
                    v.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(&v[..]),
                ),
                TensorData::I64(v) => std::slice::from_raw_parts(
                    v.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(&v[..]),
                ),
            }
        }
    }
}

pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: TensorData,
}

/// Named auxiliary tensors that reach the model runner as kwargs — the Rust
/// analogue of Python's `MultimodalDataItem.model_specific_data`, e.g. qwen's
/// `image_grid_thw`.
pub type NamedTensors = Vec<(String, Tensor)>;

/// One decoded media item handed to [`MmFamilyProcessor::process_item`].
/// Grows a variant per modality as families that need it are ported.
pub enum DecodedMedia {
    /// HWC u8 RGB.
    Image {
        rgb: Vec<u8>,
        height: usize,
        width: usize,
    },
}

/// Family-internal geometry of one processed item, consumed by
/// [`MmFamilyProcessor::layout`] / [`MmFamilyProcessor::positions`]. Grows a
/// variant per family style; the driver never interprets it.
#[derive(Clone, Debug)]
pub enum Geometry {
    /// `[t, h, w]` patch grid (`t` = 1 for still images).
    Grid([u32; 3]),
}

/// One processed media item, mirroring Python's `MultimodalDataItem`: the
/// primary feature tensor (its bytes are the item's identity — the driver
/// hashes them, standing in for the Python path's `hash_feature`), named
/// auxiliary tensors, and the geometry the family's own `layout`/`positions`
/// hooks need.
pub struct ProcessedItem {
    /// The model's feature tensor for this item (qwen: `pixel_values`).
    pub feature: Tensor,
    pub aux: NamedTensors,
    pub geometry: Geometry,
}

/// The tokens one media item occupies in the expanded prompt.
pub enum TokenPattern {
    /// N copies of one placeholder id (qwen-style).
    Repeat { id: i32, n: usize },
    /// An explicit id sequence — tile markers, row separators, wrapper
    /// tokens (minicpm/internvl-style structured expansions).
    Explicit(Vec<i32>),
}

/// One span of the expanded prompt.
pub enum Segment {
    /// Copy `src` (a range into the original ids) verbatim.
    Text(std::ops::Range<usize>),
    /// Media item `item`'s token span.
    Media { item: usize, pattern: TokenPattern },
}

/// Prompt geometry as data: the family *describes* the expansion, the driver
/// *applies* it (`common::token_layout::apply_layout`) — deriving final input ids
/// and per-item offsets, and validating that every item is placed exactly
/// once.
pub struct TokenLayout {
    pub segments: Vec<Segment>,
}

/// Modalities a family accepts; the server's message layer rejects anything
/// a family does not declare.
#[derive(Clone, Copy, Debug, Default)]
pub struct Capabilities {
    pub video: bool,
    pub audio: bool,
}

/// Position scheme of the expanded prompt.
pub enum PositionOutput {
    /// Plain sequential positions — the scheduler needs nothing extra.
    Rope1D,
    /// M-RoPE: flattened row-major `[3, input_len]` positions + the position
    /// delta (`max + 1 - input_len`).
    MRope { positions: Vec<i64>, delta: i64 },
}

/// The per-model-family hooks of the server pipeline. Adding a family =
/// implementing this in `src/<model>/mod.rs` and adding its `family` arm to
/// [`crate::registry::pipeline_from_spec`]. All parameters come from the
/// runtime spec JSON (resolved from the HF config on the Python side);
/// nothing is hardcoded per model.
pub trait MmFamilyProcessor: Send + Sync {
    /// Modalities beyond images this family accepts. Default: images only.
    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }

    /// Preprocess one decoded media item: the model's HF processor
    /// equivalent (resize/tile/normalize/patchify → named tensors) plus the
    /// geometry `layout`/`positions` will need.
    fn process_item(&self, media: &DecodedMedia) -> Result<ProcessedItem, String>;

    /// Describe how the prompt expands around the processed items (in
    /// prompt order). Sees the full original prompt and all items, so
    /// structured schemes (tile markers, separators) are expressible.
    fn layout(&self, input_ids: &[i32], items: &[Geometry]) -> Result<TokenLayout, String>;

    /// Positions for the expanded prompt. Families without a custom scheme
    /// keep the default.
    fn positions(
        &self,
        input_len: usize,
        offsets: &[(u32, u32)],
        items: &[Geometry],
    ) -> Result<PositionOutput, String> {
        let _ = (input_len, offsets, items);
        Ok(PositionOutput::Rope1D)
    }
}
