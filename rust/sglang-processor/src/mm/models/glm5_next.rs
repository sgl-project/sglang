//! GLM-5.3-Flash images as SGLang's `Glm4vImageProcessor` runs transformers'
//! `Glm5NextImageProcessor`: `smart_to_rgb` decode, resize to the content size,
//! zero-pad to Dynamo's planned canvas, then Qwen-VL's normalize and patch order.

use std::io::Cursor;

use dynamo_multimodal::MmError;
use dynamo_multimodal::image::decode::{DecodeLimits, decode_rgb_exif_transposed};
use dynamo_multimodal::image::resize::resize_rgb;
use dynamo_multimodal::models::glm5_next::{Glm5NextProcessor, Glm5NextSpec};
use dynamo_multimodal::models::qwen_vl::{MropeItem, Resampler, mrope_image_only};
use dynamo_multimodal::processor::{
    DecodedMedia, Geometry, MediaMetadata, MmFamilyProcessor, Modality, PositionOutput,
    ProcessedItem, Tensor, TensorData, TokenLayout,
};
use dynamo_multimodal::token_layout::layout_by_placeholder;
use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader};

/// Dynamo's planning spec plus the normalize and resize knobs it lacks.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct Glm5NextImageSpec {
    /// Only a `glm5_next` spec parses as this variant of `MmPipelineSpec`.
    #[serde(rename = "family")]
    _family: Glm5NextTag,
    #[serde(flatten)]
    planner: Glm5NextSpec,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    #[serde(default)]
    resample: Resampler,
}

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum Glm5NextTag {
    Glm5Next,
}

pub(super) struct Glm5Next {
    planner: Glm5NextProcessor,
    spec: Glm5NextImageSpec,
    /// Per-channel u8 to normalized f32, as Dynamo's Qwen-VL `normalize_lut`.
    lut: [[f32; 256]; 3],
}

impl Glm5Next {
    pub(super) fn new(spec: Glm5NextImageSpec) -> Result<Self, MmError> {
        let planner = Glm5NextProcessor::new(spec.planner.clone())?;
        if spec
            .image_std
            .iter()
            .any(|std| !std.is_finite() || *std <= 0.0)
        {
            return Err(MmError::invalid_input(
                "glm5_next: image_std must be positive",
            ));
        }
        let lut = core::array::from_fn(|c| {
            let (mean, std) = (spec.image_mean[c], spec.image_std[c]);
            match spec.resample {
                Resampler::Pil => core::array::from_fn(|v| (v as f32 / 255.0 - mean) / std),
                // torchvision folds the rescale into mean and std first.
                Resampler::AtenU8 => {
                    core::array::from_fn(|v| (v as f32 - mean * 255.0) / (std * 255.0))
                }
            }
        });
        Ok(Self { planner, spec, lut })
    }

    /// `[grid_h * grid_w, 3 * tps * ps * ps]` in merged-block order, each still
    /// repeated over the temporal patch.
    fn patchify(&self, rgb: &[u8], h: usize, w: usize) -> Vec<f32> {
        let geometry = &self.spec.planner.geometry;
        let (ps, m, tps) = (
            geometry.patch_size as usize,
            geometry.merge_size as usize,
            geometry.temporal_patch_size as usize,
        );
        let (gh, gw) = (h / ps, w / ps);
        let mut out = Vec::with_capacity(gh * gw * 3 * tps * ps * ps);
        for (bh, bw, mh, mw) in (0..gh / m).flat_map(|bh| {
            (0..gw / m).flat_map(move |bw| {
                (0..m).flat_map(move |mh| (0..m).map(move |mw| (bh, bw, mh, mw)))
            })
        }) {
            let (y0, x0) = ((bh * m + mh) * ps, (bw * m + mw) * ps);
            for (c, lut) in self.lut.iter().enumerate() {
                let start = out.len();
                for y in y0..y0 + ps {
                    let row = &rgb[(y * w + x0) * 3..(y * w + x0 + ps) * 3];
                    out.extend(row.iter().skip(c).step_by(3).map(|&v| lut[v as usize]));
                }
                for _ in 1..tps {
                    out.extend_from_within(start..start + ps * ps);
                }
            }
        }
        out
    }
}

impl MmFamilyProcessor for Glm5Next {
    fn num_media_tokens(&self, media: &MediaMetadata) -> Result<usize, MmError> {
        self.planner.num_media_tokens(media)
    }

    fn layout_from_metadata(
        &self,
        input_ids: &[i32],
        media: &[MediaMetadata],
    ) -> Result<TokenLayout, MmError> {
        self.planner.layout_from_metadata(input_ids, media)
    }

    fn process_item(&self, media: &DecodedMedia) -> Result<ProcessedItem, MmError> {
        let DecodedMedia::Image { rgb, height, width } = media else {
            return Err(MmError::unsupported("glm5_next supports only images"));
        };
        let (h, w) = (*height, *width);
        let geometry = &self.spec.planner.geometry;
        let plan = self.planner.plan_image(w as u32, h as u32)?;
        let (canvas_h, canvas_w) = (plan.resized_height as usize, plan.resized_width as usize);

        // `Glm5NextImageProcessor.resize`: images above the token minimum are never upscaled.
        let factor =
            (geometry.patch_size * geometry.merge_size * geometry.patch_expand_factor) as u64;
        let mut scale = (canvas_h as f64 / h as f64).min(canvas_w as f64 / w as f64);
        if (h * w) as u64 >= factor * factor * geometry.min_image_tokens as u64 {
            scale = scale.min(1.0);
        }
        let content =
            |side: usize, canvas: usize| ((side as f64 * scale).floor() as usize).clamp(1, canvas);
        let (content_h, content_w) = (content(h, canvas_h), content(w, canvas_w));
        let resized;
        let content_rgb = if (content_h, content_w) != (h, w) {
            resized = resize_rgb(rgb, h, w, content_h, content_w, self.spec.resample.into());
            &resized
        } else {
            rgb
        };
        let mut canvas = vec![0u8; canvas_h * canvas_w * 3];
        for (y, row) in content_rgb.chunks_exact(content_w * 3).enumerate() {
            canvas[y * canvas_w * 3..][..content_w * 3].copy_from_slice(row);
        }

        let grid = [1, plan.grid_h, plan.grid_w];
        let patch_dim =
            3 * geometry.temporal_patch_size * geometry.patch_size * geometry.patch_size;
        Ok(ProcessedItem {
            modality: Modality::Image,
            feature_token_count: plan.num_image_tokens as usize,
            feature: Tensor {
                shape: vec![(plan.grid_h * plan.grid_w) as usize, patch_dim as usize],
                data: TensorData::F32(self.patchify(&canvas, canvas_h, canvas_w)),
            },
            aux: vec![(
                "image_grid_thw".into(),
                Tensor {
                    shape: vec![3],
                    data: TensorData::I64(grid.iter().map(|&g| g as i64).collect()),
                },
            )],
            geometry: Some(Geometry::Grid(grid)),
        })
    }

    fn layout(&self, input_ids: &[i32], items: &[ProcessedItem]) -> Result<TokenLayout, MmError> {
        let counts: Vec<usize> = items.iter().map(|item| item.feature_token_count).collect();
        layout_by_placeholder(input_ids, self.spec.planner.image_token_id, &counts)
    }

    /// `MRotaryEmbedding.get_rope_index_glm4v` on an image-only prompt.
    fn positions(
        &self,
        input_len: usize,
        offsets: &[(u32, u32)],
        items: &[ProcessedItem],
    ) -> Result<PositionOutput, MmError> {
        let items: Vec<MropeItem> = (offsets.iter().zip(items))
            .map(|(&(start, end), item)| match item.geometry {
                Some(Geometry::Grid(grid)) => Ok(MropeItem { start, end, grid }),
                _ => Err(MmError::invalid_input("glm5_next: item without its grid")),
            })
            .collect::<Result<_, _>>()?;
        let merge_size = self.spec.planner.geometry.merge_size as usize;
        let (positions, delta) = mrope_image_only(input_len, &items, merge_size)?;
        Ok(PositionOutput::MRope { positions, delta })
    }
}

/// `utils/common.py::smart_to_rgb`: EXIF transpose, then any alpha composited
/// onto a background that contrasts with the opaque edge pixels.
pub(super) fn decode(
    data: &[u8],
    limits: &DecodeLimits,
) -> Result<(Vec<u8>, usize, usize), MmError> {
    let (rgb, h, w) = decode_rgb_exif_transposed(data, limits)?;
    match rgba(data)? {
        Some(rgba) if rgba.len() == h * w * 4 => Ok((composite(&rgba, h, w), h, w)),
        Some(_) => Err(MmError::invalid_input(
            "glm5_next: alpha and RGB decodes disagree",
        )),
        None => Ok((rgb, h, w)),
    }
}

/// The EXIF-transposed RGBA pixels of a PNG or WebP that carries alpha
/// (PIL's RGBA, LA, or `transparency`).
fn rgba(data: &[u8]) -> Result<Option<Vec<u8>>, MmError> {
    let invalid = |error: image::ImageError| MmError::invalid_input(error.to_string());
    let reader = ImageReader::new(Cursor::new(data))
        .with_guessed_format()
        .map_err(|error| MmError::invalid_input(error.to_string()))?;
    if !matches!(reader.format(), Some(ImageFormat::Png | ImageFormat::WebP)) {
        return Ok(None);
    }
    let mut decoder = reader.into_decoder().map_err(invalid)?;
    if !decoder.color_type().has_alpha() {
        return Ok(None);
    }
    let orientation = decoder.orientation().map_err(invalid)?;
    let mut image = DynamicImage::from_decoder(decoder).map_err(invalid)?;
    image.apply_orientation(orientation);
    Ok(Some(image.into_rgba8().into_raw()))
}

/// `background.paste(image, mask=alpha)`, with PIL's `BLEND` / `DIV255` rounding.
fn composite(rgba: &[u8], h: usize, w: usize) -> Vec<u8> {
    let alpha = |x: usize, y: usize| &rgba[(y * w + x) * 4..][..4];
    let rows = (0..w)
        .step_by((w / 20).max(1))
        .flat_map(|x| [(x, 0), (x, h - 1)]);
    let columns = (0..h)
        .step_by((h / 20).max(1))
        .flat_map(|y| [(0, y), (w - 1, y)]);
    let edges: Vec<&[u8]> = (rows.chain(columns))
        .map(|(x, y)| alpha(x, y))
        .filter(|pixel| pixel[3] > 128)
        .collect();
    let sum: usize = edges
        .iter()
        .flat_map(|pixel| &pixel[..3])
        .map(|&v| v as usize)
        .sum();
    let background: u32 = match edges.len() {
        0 => 255,
        n if sum > 128 * 3 * n => 32,
        _ => 240,
    };
    (rgba.chunks_exact(4))
        .flat_map(|pixel| {
            let a = pixel[3] as u32;
            pixel[..3].iter().map(move |&c| {
                let t = background * (255 - a) + c as u32 * a + 128;
                (((t >> 8) + t) >> 8) as u8
            })
        })
        .collect()
}
