//! Shared multimodal request driver for the server (pure-Rust) pipeline.
//!
//! Owns the request control flow — parallel fan-out, layout application,
//! failure semantics — while every model decision lives behind
//! [`MmFamilyProcessor`] (see `pipeline.rs`). Families produce data; they
//! cannot alter orchestration.

use crate::common::{self, fetch, par, token_layout};
use crate::pipeline::{DecodedMedia, MmFamilyProcessor, PositionOutput, ProcessedItem};

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

/// One processed media item at the request boundary.
pub struct OutputItem {
    pub feature: crate::pipeline::Tensor,
    pub aux: crate::pipeline::NamedTensors,
    /// [`common::content_hash_u64`] of the raw encoded source bytes — the same
    /// identity role as the Python path's `hash_feature`, but a different
    /// algorithm, so hashes are consistent within the server pipeline and
    /// never comparable across the two paths.
    pub hash: u64,
}

/// The per-request result parked for the scheduler drain.
pub struct Output {
    pub input_ids: Vec<i32>,
    /// In prompt order; `offsets[i]` is `items[i]`'s inclusive token range.
    pub items: Vec<OutputItem>,
    pub offsets: Vec<(u32, u32)>,
    pub positions: PositionOutput,
}

/// Resolve one image source to raw encoded bytes, borrowing when the request
/// already carries them.
fn resolve(source: &ImageSource) -> Result<std::borrow::Cow<'_, [u8]>, String> {
    match source {
        ImageSource::String(source) => Ok(fetch::fetch_bytes(source)?.into()),
        ImageSource::Bytes(bytes) => Ok(bytes.as_slice().into()),
    }
}

/// Run one request through the pipeline. Any `Err` rejects the request back
/// to the client — including inputs merely outside the pipeline's scope
/// (video/audio, precomputed features, undecodable images), since there is
/// no Python fallback path.
pub fn process(
    family: &dyn MmFamilyProcessor,
    input: MmInput,
    tokenize: impl FnOnce(&str) -> Result<Vec<i32>, String>,
) -> Result<Output, String> {
    if input.images.is_empty() {
        return Err("multimodal request without image sources".into());
    }
    // Stage 1 (fetch) is blocking I/O and runs inline, sequentially: it must
    // NOT go through `par::try_map` — one request's slow URLs would occupy the
    // CPU workers other requests need for decode/resize. The server supplies
    // concurrency across requests, and its request shape is pre-fetched
    // `Bytes` anyway; give fetch its own I/O pool before ever fanning it out.
    let fetched: Vec<std::borrow::Cow<'_, [u8]>> = input
        .images
        .iter()
        .map(resolve)
        .collect::<Result<_, _>>()?;
    let processed: Vec<(ProcessedItem, u64)> =
        par::try_map(&fetched, |bytes| -> Result<(ProcessedItem, u64), String> {
            let hash = common::content_hash_u64(bytes);
            // The Python (PIL) path decodes more formats (GIF/WebP/BMP, 16-bit
            // PNG); those error here and reject the request.
            let (rgb, height, width) = common::decode_rgb(bytes)?;
            let item = family.process_item(&DecodedMedia::Image { rgb, height, width })?;
            Ok((item, hash))
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
    let geometries = processed
        .iter()
        .map(|(item, _)| item.geometry.clone())
        .collect::<Vec<_>>();
    let layout = family.layout(&input_ids, &geometries)?;
    let expanded = token_layout::apply_layout(&input_ids, &layout, processed.len())?;
    let positions = family.positions(expanded.input_ids.len(), &expanded.offsets, &geometries)?;

    Ok(Output {
        input_ids: expanded.input_ids,
        items: processed
            .into_iter()
            .map(|(item, hash)| OutputItem {
                feature: item.feature,
                aux: item.aux,
                hash,
            })
            .collect(),
        offsets: expanded.offsets,
        positions,
    })
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
        let family = pipeline_from_spec(SPEC).unwrap();
        let input = MmInput {
            text: None,
            input_ids: Some(vec![7, 1, 8]),
            images: vec![ImageSource::Bytes(png(8, 8))],
        };
        let out = process(family.as_ref(), input, |_| Err("no tokenizer".into())).unwrap();
        // 8x8, factor 4 → grid [1, 4, 4] → 16 patches / merge² = 4 tokens.
        assert_eq!(out.input_ids, vec![7, 1, 1, 1, 1, 8]);
        assert_eq!(out.offsets, vec![(1, 4)]);
        let item = &out.items[0];
        assert_eq!(item.feature.shape, [16, 3 * 2 * 2 * 2]);
        assert_eq!(item.aux[0].0, "image_grid_thw");
        let crate::pipeline::PositionOutput::MRope { positions, .. } = &out.positions else {
            panic!("qwen emits mrope")
        };
        assert_eq!(positions.len(), 3 * out.input_ids.len());
    }

    #[test]
    fn image_free_and_mismatched_requests_rejected() {
        let family = pipeline_from_spec(SPEC).unwrap();
        let no_images = MmInput {
            text: None,
            input_ids: Some(vec![7, 1]),
            images: vec![],
        };
        let err = process(family.as_ref(), no_images, |_| unreachable!())
            .err()
            .unwrap();
        assert!(err.contains("image sources"));
        let no_placeholder = MmInput {
            text: None,
            input_ids: Some(vec![7, 8]),
            images: vec![ImageSource::Bytes(png(8, 8))],
        };
        let err = process(family.as_ref(), no_placeholder, |_| unreachable!())
            .err()
            .unwrap();
        assert!(err.contains("placeholder"));
    }
}
