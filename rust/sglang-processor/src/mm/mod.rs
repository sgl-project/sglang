//! Multimodal preprocessing over `dynamo-multimodal`.

mod video;

use dynamo_multimodal::image::decode::{DecodeLimits, decode_rgb};
use dynamo_multimodal::image::resize::Resample;
use dynamo_multimodal::processor::{
    DecodedMedia, MmFamilyProcessor, Modality, NamedTensors, PositionOutput, ProcessedItem, Tensor,
    TextEncoder,
};
use dynamo_multimodal::registry::{ProcessorSpec, build_processor};
use dynamo_multimodal::video_decode::{FrameResize, RgbConversion, Video, VideoDecodeLimits};
use dynamo_multimodal::{MmError, content_hash_bytes, token_layout};

pub use video::{VideoConfig, VideoHints, VideoSampling, frame_indices};

use crate::ProcessorError;

/// PIL raises `DecompressionBombError` above twice its default `MAX_IMAGE_PIXELS`.
const PIL_DECOMPRESSION_BOMB_PIXELS: usize = 2 * 89_478_485;

/// One model's media pipeline, resolved once at boot and shared by every request.
pub struct MmProcessor {
    family: Box<dyn MmFamilyProcessor>,
    limits: DecodeLimits,
    video_limits: VideoDecodeLimits,
    /// SGLang's frame sampling; `None` refuses video.
    video: Option<VideoSampling>,
}

/// One video of a request: its encoded bytes and `MediaRef` hints.
pub struct MmVideo<'a> {
    pub bytes: &'a [u8],
    pub hints: VideoHints,
}

/// One request's model inputs.
pub struct MmOutput {
    /// The prompt with each placeholder expanded to its item's feature tokens.
    pub input_ids: Vec<i32>,
    /// The images in prompt order, then the videos in prompt order, as Python
    /// collects its `mm_items`.
    pub items: Vec<MmOutputItem>,
    pub positions: PositionOutput,
}

pub struct MmOutputItem {
    pub modality: Modality,
    pub feature: Tensor,
    /// Model kwargs such as Qwen's `image_grid_thw` (Python's `model_specific_data`).
    pub aux: NamedTensors,
    /// Inclusive token range of each feature run, as Python's
    /// `get_mm_items_offset`: one for an image, one per temporal patch of a video.
    pub offsets: Vec<(u32, u32)>,
    /// blake3 of the encoded bytes; never comparable with Python's `hash_feature`.
    pub hash: u64,
}

impl MmProcessor {
    /// The pipeline for a spec SGLang resolved from the model's HF processor
    /// (`rust_server/multimodal.py::RustMmProcessor.resolve_spec`).
    pub fn from_spec(
        spec: ProcessorSpec,
        video: Option<VideoSampling>,
    ) -> Result<Self, ProcessorError> {
        let limits = DecodeLimits {
            max_pixels: Some(PIL_DECOMPRESSION_BOMB_PIXELS),
            ..DecodeLimits::default()
        };
        let mut video_limits = VideoDecodeLimits::default();
        if let Some(sampling) = &video {
            // `preprocess_video` keeps a clip under `FRAME_FACTOR * VIDEO_TOTAL_PIXELS`
            // pixels up to about 1700 frames; at the defaults that tops 512 MiB.
            let clip_bytes = 3.0 * sampling.frame_factor * sampling.total_pixels;
            video_limits.max_output_bytes = video_limits.max_output_bytes.max(clip_bytes as usize);
        }
        Ok(Self {
            family: build_processor(spec).map_err(mm_error)?,
            limits,
            video_limits,
            video,
        })
    }

    /// Expand an images-only prompt; see [`Self::process_media`].
    pub fn process(&self, input_ids: &[i32], images: &[&[u8]]) -> Result<MmOutput, ProcessorError> {
        self.process_media(input_ids, images, &[], &NoText)
    }

    /// Expand a prompt and preprocess its media, as Python's `process_mm_data_async`
    /// does; `text` encodes the text an expansion inserts (video timestamps).
    pub fn process_media(
        &self,
        input_ids: &[i32],
        images: &[&[u8]],
        videos: &[MmVideo<'_>],
        text: &dyn TextEncoder,
    ) -> Result<MmOutput, ProcessorError> {
        if images.is_empty() && videos.is_empty() {
            return Err("multimodal request without media".into());
        }
        let mut items: Vec<(ProcessedItem, u64)> = images
            .iter()
            .map(|bytes| Ok((self.process_image(bytes)?, content_hash_bytes(bytes))))
            .collect::<Result<_, ProcessorError>>()?;
        for video in videos {
            items.push(self.process_video(video)?);
        }
        let order = self.prompt_order(input_ids, images.len(), videos.len())?;
        let mut slots: Vec<Option<(ProcessedItem, u64)>> = items.into_iter().map(Some).collect();
        let prompt_items: Vec<(ProcessedItem, u64)> = order
            .iter()
            .map(|&i| slots[i].take().expect("a permutation"))
            .collect();
        let (processed, hashes): (Vec<ProcessedItem>, Vec<u64>) = prompt_items.into_iter().unzip();

        let layout = (self.family)
            .layout_with_text(input_ids, &processed, text)
            .map_err(mm_error)?;
        let token_counts: Vec<usize> = processed
            .iter()
            .map(|item| item.feature_token_count)
            .collect();
        let expanded = token_layout::apply_layout(input_ids, &layout, &token_counts, usize::MAX)
            .map_err(mm_error)?;
        let positions = (self.family)
            .positions_with_features(
                expanded.input_ids.len(),
                &expanded.offsets,
                &expanded.feature_ranges,
                &processed,
            )
            .map_err(mm_error)?;

        let mut outputs: Vec<Option<MmOutputItem>> = (0..order.len()).map(|_| None).collect();
        for (((item, hash), ranges), &slot) in processed
            .into_iter()
            .zip(hashes)
            .zip(expanded.feature_ranges)
            .zip(&order)
        {
            outputs[slot] = Some(MmOutputItem {
                modality: item.modality,
                feature: item.feature,
                aux: item.aux,
                offsets: ranges.iter().map(|r| (r.start, r.end - 1)).collect(),
                hash,
            });
        }
        Ok(MmOutput {
            input_ids: expanded.input_ids,
            items: outputs
                .into_iter()
                .map(|item| item.expect("every slot filled"))
                .collect(),
            positions,
        })
    }

    /// Python's Qwen path decodes with PIL and only `convert("RGB")`s: no EXIF transpose.
    fn process_image(&self, bytes: &[u8]) -> Result<ProcessedItem, ProcessorError> {
        let (rgb, height, width) = decode_rgb(bytes, &self.limits).map_err(mm_error)?;
        self.family
            .process_item(&DecodedMedia::Image { rgb, height, width })
            .map_err(mm_error)
    }

    /// `QwenVLImageProcessor._process_mm_data_uncached` for one video.
    fn process_video(&self, video: &MmVideo<'_>) -> Result<(ProcessedItem, u64), ProcessorError> {
        let Some(sampling) = &self.video else {
            return Err("this model takes no video input".into());
        };
        let config = sampling.config.with_hints(&video.hints).map_err(mm_error)?;
        let source = Video::open(&video.bytes, &self.video_limits).map_err(mm_error)?;
        let info = source.info();
        // torchcodec's `len()` and `average_fps`, which Python samples from.
        let num_frames = sampling
            .num_frames(&config, info.frame_count, info.average_fps)
            .map_err(mm_error)?;
        let indices = frame_indices(info.frame_count, num_frames);
        let (height, width) = sampling
            .presize(
                &config,
                indices.len(),
                info.height as usize,
                info.width as usize,
            )
            .map_err(mm_error)?;
        let side = |v: usize| {
            u32::try_from(v).map_err(|_| ProcessorError::InvalidRequest("video too large".into()))
        };
        let presize = FrameResize {
            width: side(width)?,
            height: side(height)?,
            filter: Resample::AtenF32Bilinear,
        };
        // torchcodec converts with swscale's x86 SIMD path, except at odd
        // heights, where swscale's generic path (not reproduced) runs.
        let rgb = match info.height % 2 {
            0 => RgbConversion::Swscale,
            _ => RgbConversion::Float,
        };
        let frames = source
            .decode_with(&indices, Some(&presize), &self.video_limits, rgb)
            .map_err(mm_error)?;
        let (height, width) = (frames.height(), frames.width());
        let hash = content_hash_bytes(video.bytes);
        let item = self
            .family
            .process_item(&DecodedMedia::Video {
                rgb: frames.into_rgb(),
                height,
                width,
                frame_indices: indices,
                fps: info.average_fps,
            })
            .map_err(mm_error)?;
        Ok((item, hash))
    }

    /// Indices into `[images.., videos..]` in placeholder order.
    fn prompt_order(
        &self,
        input_ids: &[i32],
        images: usize,
        videos: usize,
    ) -> Result<Vec<usize>, ProcessorError> {
        if videos == 0 {
            return Ok((0..images).collect());
        }
        let modalities = self
            .family
            .placeholder_modalities(input_ids)
            .map_err(mm_error)?;
        let (mut image, mut video) = (0, images);
        let order: Vec<usize> = modalities
            .iter()
            .map(|modality| match modality {
                Modality::Video => (video, video += 1).0,
                _ => (image, image += 1).0,
            })
            .collect();
        if image != images || video != images + videos {
            return Err(format!(
                "prompt has {} image and {} video placeholder(s) but {images} image(s) and {videos} video(s)",
                image,
                video - images
            )
            .into());
        }
        Ok(order)
    }
}

/// The images-only path's encoder: no image expansion has text in it.
struct NoText;

impl TextEncoder for NoText {
    fn encode(&self, _text: &str) -> dynamo_multimodal::Result<Vec<i32>> {
        Err(MmError::internal("an image expansion asked for text"))
    }
}

fn mm_error(error: MmError) -> ProcessorError {
    match error {
        MmError::Internal { .. } => ProcessorError::Internal(error.to_string()),
        _ => ProcessorError::InvalidRequest(error.to_string()),
    }
}
