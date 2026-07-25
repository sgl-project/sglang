//! Shared multimodal request driver for the server (pure-Rust) pipeline.

use rayon::prelude::*;

use crate::common::payload::ImageSource;
use crate::common::{self, fetch, payload, tokens};
use crate::registry::{MropeItem, Pipeline, ProcessedImage};

/// The per-request mm buffers parked for the scheduler drain.
pub struct MmResult {
    pub features: Vec<f32>,
    pub grids: Vec<[u32; 3]>,
    /// Content hash of each image's `pixel_values` bytes — same identity
    /// domain as the Python path's `hash_feature`, but a different algorithm
    /// (blake3 here vs SHA-256 there): hashes are consistent within the
    /// native path, never comparable across the two paths.
    pub hashes: Vec<u64>,
    pub offsets: Vec<(u32, u32)>,
    pub mrope: Vec<i64>,
    pub mrope_delta: i64,
}

pub struct Output {
    pub input_ids: Vec<i32>,
    pub mm: MmResult,
}

/// Run one request through the pipeline. Any `Err` rejects the request back
/// to the client — including inputs merely outside the pipeline's scope
/// (video/audio, precomputed features, undecodable images), since there is
/// no Python fallback path.
pub fn process(
    pipeline: &Pipeline,
    payload_bytes: &[u8],
    tokenize: impl FnOnce(&str) -> Result<Vec<i32>, String>,
) -> Result<Output, String> {
    let payload = payload::parse(payload_bytes)?;
    let processed: Vec<(ProcessedImage, u64)> = common::pool().install(|| {
        payload
            .images
            .par_iter()
            .map(|source| {
                let bytes: std::borrow::Cow<'_, [u8]> = match source {
                    ImageSource::String(source) => fetch::fetch_bytes(source)?.into(),
                    ImageSource::Bytes(bytes) => bytes.as_slice().into(),
                };
                // The Python (PIL) path decodes more formats (GIF/WebP/BMP,
                // 16-bit PNG); those error here and reject the request.
                let (rgb, height, width) = common::decode_rgb(&bytes)?;
                let image = pipeline.processor.process_image(&rgb, height, width)?;
                let hash = common::sha256_u64(f32_bytes(&image.pixel_values));
                Ok((image, hash))
            })
            .collect::<Result<Vec<_>, String>>()
    })?;

    let input_ids = match payload.input_ids {
        Some(input_ids) if !input_ids.is_empty() => input_ids,
        _ => {
            let text = payload
                .text
                .as_deref()
                .ok_or("multimodal request without text or input_ids")?;
            tokenize(text)?
        }
    };
    let counts = processed
        .iter()
        .map(|(image, _)| pipeline.processor.tokens_per_image(&image.grid_thw))
        .collect::<Vec<_>>();
    let expanded = tokens::expand_placeholders(&input_ids, pipeline.image_token_id, &counts)?;
    let mrope_items = expanded
        .offsets
        .iter()
        .zip(&processed)
        .map(|(&(start, end), (image, _))| MropeItem {
            start,
            end,
            grid: image.grid_thw,
        })
        .collect::<Vec<_>>();
    let (mrope, mrope_delta) = pipeline
        .processor
        .mrope_image_only(expanded.input_ids.len(), &mrope_items)?;

    let mut features = Vec::with_capacity(
        processed
            .iter()
            .map(|(image, _)| image.pixel_values.len())
            .sum(),
    );
    let mut grids = Vec::with_capacity(processed.len());
    let mut hashes = Vec::with_capacity(processed.len());
    for (image, hash) in processed {
        features.extend(image.pixel_values);
        grids.push(image.grid_thw);
        hashes.push(hash);
    }
    Ok(Output {
        input_ids: expanded.input_ids,
        mm: MmResult {
            features,
            grids,
            hashes,
            offsets: expanded.offsets,
            mrope,
            mrope_delta,
        },
    })
}

fn f32_bytes(values: &[f32]) -> &[u8] {
    // Safety: f32 is plain-old-data with no padding.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}
