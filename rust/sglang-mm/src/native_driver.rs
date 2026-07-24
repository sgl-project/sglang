//! Shared native multimodal request driver.

use rayon::prelude::*;

use crate::common::payload::{ImageSource, NativeError};
use crate::common::{self, fetch, payload, tokens};
use crate::registry::{MropeItem, NativePipeline, ProcessedImage};

pub struct NativeMmResult {
    pub features: Vec<f32>,
    pub grids: Vec<[u32; 3]>,
    pub hashes: Vec<u64>,
    pub offsets: Vec<(u32, u32)>,
    pub mrope: Vec<i64>,
    pub mrope_delta: i64,
}

pub struct NativeDriverOutput {
    pub input_ids: Vec<i32>,
    pub mm: NativeMmResult,
}

pub fn process(
    pipeline: &NativePipeline,
    payload_bytes: &[u8],
    tokenize: impl FnOnce(&str) -> Result<Vec<i32>, String>,
) -> Result<NativeDriverOutput, NativeError> {
    let payload = payload::parse(payload_bytes)?;
    let processed: Vec<(ProcessedImage, u64)> = common::pool().install(|| {
        payload
            .images
            .par_iter()
            .map(|source| {
                let bytes = match source {
                    ImageSource::String(source) => {
                        fetch::fetch_bytes(source).map_err(|error| match error {
                            fetch::FetchError::Unsupported(message) => {
                                NativeError::Fallback(message)
                            }
                            fetch::FetchError::Failed(message) => NativeError::Failed(message),
                        })?
                    }
                    ImageSource::Bytes(bytes) => bytes.clone(),
                };
                let (rgb, height, width) =
                    common::decode_rgb(&bytes).map_err(NativeError::Failed)?;
                let image = pipeline
                    .processor
                    .process_image(&rgb, height, width)
                    .map_err(NativeError::Failed)?;
                let hash = common::sha256_u64(f32_bytes(&image.pixel_values));
                Ok((image, hash))
            })
            .collect::<Result<Vec<_>, NativeError>>()
    })?;

    let input_ids = match payload.input_ids {
        Some(input_ids) if !input_ids.is_empty() => input_ids,
        _ => {
            let text = payload.text.as_deref().ok_or_else(|| {
                NativeError::Failed("multimodal request without text or input_ids".into())
            })?;
            tokenize(text).map_err(NativeError::Failed)?
        }
    };
    let counts = processed
        .iter()
        .map(|(image, _)| pipeline.processor.tokens_per_image(&image.grid_thw))
        .collect::<Vec<_>>();
    let expanded = tokens::expand_placeholders(&input_ids, pipeline.image_token_id, &counts)
        .map_err(NativeError::Fallback)?;
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
        .mrope_image_only(expanded.input_ids.len(), &mrope_items)
        .map_err(NativeError::Failed)?;

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
    Ok(NativeDriverOutput {
        input_ids: expanded.input_ids,
        mm: NativeMmResult {
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
