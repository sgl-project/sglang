//! Multimodal preprocessing over `dynamo-multimodal`.

use dynamo_multimodal::image::decode::DecodeLimits;
use dynamo_multimodal::processor::{
    DecodedMedia, MmFamilyProcessor, NamedTensors, PositionOutput, ProcessedItem, Tensor,
};
use dynamo_multimodal::{MmError, content_hash_bytes, token_layout};

use crate::ProcessorError;

mod models;

pub use models::MmPipelineSpec;

/// PIL raises `DecompressionBombError` above twice its default `MAX_IMAGE_PIXELS`.
const PIL_DECOMPRESSION_BOMB_PIXELS: usize = 2 * 89_478_485;

/// One model's image pipeline, resolved once at boot and shared by every request.
pub struct MmProcessor {
    family: Box<dyn MmFamilyProcessor>,
    decode: models::Decode,
    limits: DecodeLimits,
}

/// One request's model inputs.
pub struct MmOutput {
    /// The prompt with each placeholder expanded to its item's feature tokens.
    pub input_ids: Vec<i32>,
    /// One per image, in prompt order.
    pub items: Vec<MmOutputItem>,
    /// Inclusive token range of each item, as Python's `get_mm_items_offset`.
    pub offsets: Vec<(u32, u32)>,
    pub positions: PositionOutput,
}

pub struct MmOutputItem {
    pub feature: Tensor,
    /// Model kwargs such as Qwen's `image_grid_thw` (Python's `model_specific_data`).
    pub aux: NamedTensors,
    /// blake3 of the encoded bytes; never comparable with Python's `hash_feature`.
    pub hash: u64,
}

impl MmProcessor {
    /// The pipeline for a spec SGLang resolved from the model's HF processor
    /// (`rust_server/multimodal.py::RustMmProcessor.resolve_spec`).
    pub fn from_spec(spec: MmPipelineSpec) -> Result<Self, ProcessorError> {
        let limits = DecodeLimits {
            max_pixels: Some(PIL_DECOMPRESSION_BOMB_PIXELS),
            ..DecodeLimits::default()
        };
        let (family, decode) = models::build(spec).map_err(mm_error)?;
        Ok(Self {
            family,
            decode,
            limits,
        })
    }

    /// Expand an images-only prompt and preprocess its images, as Python's
    /// `process_mm_data_async` does. `input_ids` hold one placeholder per image.
    /// The prompt length ceiling is the host's pre-send check, as in Python.
    pub fn process(&self, input_ids: &[i32], images: &[&[u8]]) -> Result<MmOutput, ProcessorError> {
        if images.is_empty() {
            return Err("multimodal request without images".into());
        }
        let hashes: Vec<u64> = images
            .iter()
            .map(|bytes| content_hash_bytes(bytes))
            .collect();
        let items = images
            .iter()
            .map(|bytes| self.process_image(bytes))
            .collect::<Result<Vec<_>, _>>()?;

        let layout = self.family.layout(input_ids, &items).map_err(mm_error)?;
        let token_counts: Vec<usize> = items.iter().map(|item| item.feature_token_count).collect();
        let expanded = token_layout::apply_layout(input_ids, &layout, &token_counts, usize::MAX)
            .map_err(mm_error)?;
        let positions = self
            .family
            .positions(expanded.input_ids.len(), &expanded.offsets, &items)
            .map_err(mm_error)?;

        Ok(MmOutput {
            input_ids: expanded.input_ids,
            items: items
                .into_iter()
                .zip(hashes)
                .map(|(item, hash)| MmOutputItem {
                    feature: item.feature,
                    aux: item.aux,
                    hash,
                })
                .collect(),
            offsets: expanded.offsets,
            positions,
        })
    }

    fn process_image(&self, bytes: &[u8]) -> Result<ProcessedItem, ProcessorError> {
        let (rgb, height, width) = (self.decode)(bytes, &self.limits).map_err(mm_error)?;
        self.family
            .process_item(&DecodedMedia::Image { rgb, height, width })
            .map_err(mm_error)
    }
}

fn mm_error(error: MmError) -> ProcessorError {
    match error {
        MmError::Internal { .. } => ProcessorError::Internal(error.to_string()),
        _ => ProcessorError::InvalidRequest(error.to_string()),
    }
}
