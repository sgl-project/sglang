//! Multimodal preprocessing over `dynamo-multimodal`.

pub mod prepared;

use dynamo_multimodal::image::decode::{DecodeLimits, decode_rgb};
use dynamo_multimodal::processor::{
    DecodedMedia, MmFamilyProcessor, Modality, NamedTensors, PositionOutput, ProcessedItem, Tensor,
};
use dynamo_multimodal::registry::{ProcessorSpec, build_processor};
use dynamo_multimodal::{MmError, content_hash_bytes, token_layout};

use crate::ProcessorError;

/// PIL raises `DecompressionBombError` above twice its default `MAX_IMAGE_PIXELS`.
const PIL_DECOMPRESSION_BOMB_PIXELS: usize = 2 * 89_478_485;

/// One model's image pipeline, resolved once at boot and shared by every request.
pub struct MmProcessor {
    family: Box<dyn MmFamilyProcessor>,
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
    pub fn from_spec(spec: ProcessorSpec) -> Result<Self, ProcessorError> {
        let limits = DecodeLimits {
            max_pixels: Some(PIL_DECOMPRESSION_BOMB_PIXELS),
            ..DecodeLimits::default()
        };
        Ok(Self {
            family: build_processor(spec).map_err(mm_error)?,
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
        self.assemble_output(expanded.input_ids, expanded.offsets, items, hashes)
    }

    /// Finish a prepared request after the installed profile has already built
    /// canonical [`ProcessedItem`] values.
    ///
    /// `checked` is **only** structural proof from [`prepared::PreparedInput::check`].
    /// The caller that owns the installed processor profile must complete
    /// model-semantic validation (required tensors, geometry, token alignment,
    /// stage/admission) **before** calling this. This is not a universal
    /// external-tensor admission hook and does not invent a per-model factory.
    ///
    /// Expects already-expanded `input_ids` on `checked`; does not run
    /// `layout` / `apply_layout` again. Prepared outputs use `hash = 0` because
    /// there are no encoded-source bytes — the host must derive its usual
    /// typed-content cache identity and must not trust client-supplied hashes.
    pub fn finish_prepared(
        &self,
        checked: &prepared::CheckedPreparedInput<'_>,
        items: Vec<ProcessedItem>,
    ) -> Result<MmOutput, ProcessorError> {
        let input = checked.input();
        if items.len() != input.items.len() {
            return Err(format!(
                "prepared canonical item count {} does not match checked item count {}",
                items.len(),
                input.items.len()
            )
            .into());
        }

        let caps = self.family.capabilities();
        let mut offsets = Vec::with_capacity(items.len());
        for (index, (prepared_item, item)) in input.items.iter().zip(&items).enumerate() {
            // MmOutput is currently image-only; do not silently accept other
            // modalities even if a family advertises them.
            if item.modality != Modality::Image {
                return Err(format!(
                    "prepared item {index}: finish_prepared currently represents Image only, got {:?}",
                    item.modality
                )
                .into());
            }
            if item.modality != prepared_item.modality {
                return Err(format!(
                    "prepared item {index}: canonical modality {:?} does not match input {:?}",
                    item.modality, prepared_item.modality
                )
                .into());
            }
            if !caps.supports(item.modality) {
                return Err(format!(
                    "prepared item {index}: modality {:?} is outside the installed family capabilities",
                    item.modality
                )
                .into());
            }
            if prepared_item.spans.len() != 1 {
                return Err(format!(
                    "prepared item {index}: finish_prepared requires exactly one feature span, got {}",
                    prepared_item.spans.len()
                )
                .into());
            }
            let span = prepared_item.spans[0];
            let span_len = usize::try_from(span.end - span.start).map_err(|_| {
                format!("prepared item {index}: feature span length does not fit in usize")
            })?;
            if span_len != item.feature_token_count {
                return Err(format!(
                    "prepared item {index}: feature span length {span_len} does not match feature_token_count {}",
                    item.feature_token_count
                )
                .into());
            }
            // Structural check already guarantees start < end, so end - 1 is
            // a valid inclusive upper bound matching apply_layout offsets.
            offsets.push((span.start, span.end - 1));
        }

        self.assemble_output(
            input.input_ids.to_vec(),
            offsets,
            items,
            std::iter::repeat_n(0u64, input.items.len()),
        )
    }

    /// Python's Qwen path decodes with PIL and only `convert("RGB")`s: no EXIF transpose.
    fn process_image(&self, bytes: &[u8]) -> Result<ProcessedItem, ProcessorError> {
        let (rgb, height, width) = decode_rgb(bytes, &self.limits).map_err(mm_error)?;
        self.family
            .process_item(&DecodedMedia::Image { rgb, height, width })
            .map_err(mm_error)
    }

    fn assemble_output(
        &self,
        input_ids: Vec<i32>,
        offsets: Vec<(u32, u32)>,
        items: Vec<ProcessedItem>,
        hashes: impl IntoIterator<Item = u64>,
    ) -> Result<MmOutput, ProcessorError> {
        let positions = self
            .family
            .positions(input_ids.len(), &offsets, &items)
            .map_err(mm_error)?;
        Ok(MmOutput {
            input_ids,
            items: items
                .into_iter()
                .zip(hashes)
                .map(|(item, hash)| MmOutputItem {
                    feature: item.feature,
                    aux: item.aux,
                    hash,
                })
                .collect(),
            offsets,
            positions,
        })
    }
}

fn mm_error(error: MmError) -> ProcessorError {
    match error {
        MmError::Internal { .. } => ProcessorError::Internal(error.to_string()),
        _ => ProcessorError::InvalidRequest(error.to_string()),
    }
}
