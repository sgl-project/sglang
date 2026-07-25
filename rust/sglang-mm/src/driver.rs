//! Shared multimodal request driver for the server (pure-Rust) pipeline.

use rayon::prelude::*;

use crate::common::{self, fetch, tokens};
use crate::registry::{MropeItem, Pipeline, ProcessedImage};

/// One raw image source from the request.
#[derive(Debug)]
pub enum ImageSource {
    /// `data:`/base64/file/http — resolved by [`fetch::fetch_bytes`].
    String(String),
    /// Already-raw encoded image bytes.
    Bytes(Vec<u8>),
}

/// Typed multimodal request input. The server's message layer owns the wire
/// format and parses its payload into this before calling [`process`].
pub struct MmInput {
    pub text: Option<String>,
    pub input_ids: Option<Vec<i32>>,
    pub images: Vec<ImageSource>,
}

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
    input: MmInput,
    tokenize: impl FnOnce(&str) -> Result<Vec<i32>, String>,
) -> Result<Output, String> {
    if input.images.is_empty() {
        return Err("multimodal request without image sources".into());
    }
    let processed: Vec<(ProcessedImage, u64)> = common::pool().install(|| {
        input
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

    let input_ids = match input.input_ids {
        Some(input_ids) if !input_ids.is_empty() => input_ids,
        _ => {
            let text = input
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::pipeline_from_spec;

    const SPEC: &str = r#"{"family":"qwen_vl","image_token_id":1,"patch_size":2,
        "merge_size":2,"temporal_patch_size":2,"min_pixels":4,
        "max_pixels":1073741824,"image_mean":[0.0,0.0,0.0],"image_std":[1.0,1.0,1.0]}"#;

    fn png(w: u32, h: u32) -> Vec<u8> {
        let img = image::RgbImage::from_fn(w, h, |x, y| image::Rgb([x as u8, y as u8, 7]));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        buf.into_inner()
    }

    #[test]
    fn processes_typed_image_request() {
        let pipeline = pipeline_from_spec(SPEC).unwrap();
        let input = MmInput {
            text: None,
            input_ids: Some(vec![7, 1, 8]),
            images: vec![ImageSource::Bytes(png(8, 8))],
        };
        let out = process(&pipeline, input, |_| Err("no tokenizer".into())).unwrap();
        // 8x8, factor 4 → grid [1, 4, 4] → 16 patches / merge² = 4 tokens.
        assert_eq!(out.mm.grids, vec![[1, 4, 4]]);
        assert_eq!(out.input_ids, vec![7, 1, 1, 1, 1, 8]);
        assert_eq!(out.mm.offsets, vec![(1, 4)]);
        assert_eq!(out.mm.mrope.len(), 3 * out.input_ids.len());
        assert_eq!(out.mm.features.len(), 16 * 3 * 2 * 2 * 2);
        assert_eq!(out.mm.hashes.len(), 1);
    }

    #[test]
    fn image_free_and_mismatched_requests_rejected() {
        let pipeline = pipeline_from_spec(SPEC).unwrap();
        let no_images = MmInput {
            text: None,
            input_ids: Some(vec![7, 1]),
            images: vec![],
        };
        let err = process(&pipeline, no_images, |_| unreachable!())
            .err()
            .unwrap();
        assert!(err.contains("image sources"));
        let no_placeholder = MmInput {
            text: None,
            input_ids: Some(vec![7, 8]),
            images: vec![ImageSource::Bytes(png(8, 8))],
        };
        let err = process(&pipeline, no_placeholder, |_| unreachable!())
            .err()
            .unwrap();
        assert!(err.contains("placeholder"));
    }
}
