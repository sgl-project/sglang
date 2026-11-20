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
    /// SGLang's frame sampling; `None` refuses video.
    video: Option<VideoSampling>,
}

/// One video of a request: its encoded bytes and `MediaRef` hints.
pub struct MmVideo<'a> {
    pub bytes: &'a [u8],
    pub hints: VideoHints,
}

/// One decoded item of a request, from any source: [`MmProcessor::process_media`]
/// decodes encoded bytes into these, and a caller that already holds decoded media
/// passes them to [`MmProcessor::process_decoded`].
pub struct MmDecoded {
    pub media: DecodedMedia,
    /// Identifies the source; `process_media` uses blake3 of its encoded bytes.
    pub hash: u64,
}

/// One request's model inputs.
pub struct MmOutput {
    /// The prompt with each placeholder expanded to its item's feature tokens.
    pub input_ids: Vec<i32>,
    /// One per item, in the caller's order. `process_media` gives the images in
    /// prompt order, then the videos in prompt order, as Python collects its `mm_items`.
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
    /// The item's [`MmDecoded::hash`]; never comparable with Python's `hash_feature`.
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
        Ok(Self {
            family: build_processor(spec).map_err(mm_error)?,
            limits,
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
        let decoded = (images.iter().map(|bytes| self.decode_image(bytes)))
            .chain(videos.iter().map(|video| self.decode_video(video)));
        self.expand(input_ids, decoded, text)
    }

    /// Expand a prompt for media decoded elsewhere. The n-th placeholder of a
    /// modality takes that modality's n-th item. Video frames are used as given:
    /// SGLang's frame sampling and pre-resize run only in [`Self::process_media`].
    pub fn process_decoded(
        &self,
        input_ids: &[i32],
        items: Vec<MmDecoded>,
        text: &dyn TextEncoder,
    ) -> Result<MmOutput, ProcessorError> {
        self.expand(input_ids, items.into_iter().map(Ok), text)
    }

    /// Every request's path after decode, whatever its modalities and sources.
    /// Each item is processed as it arrives, so one decoded item is held at a time.
    fn expand(
        &self,
        input_ids: &[i32],
        items: impl Iterator<Item = Result<MmDecoded, ProcessorError>>,
        text: &dyn TextEncoder,
    ) -> Result<MmOutput, ProcessorError> {
        let items: Vec<(ProcessedItem, u64)> = items
            .map(|item| {
                let item = item?;
                let processed = self.family.process_item(&item.media).map_err(mm_error)?;
                Ok((processed, item.hash))
            })
            .collect::<Result<_, ProcessorError>>()?;
        if items.is_empty() {
            return Err("multimodal request without media".into());
        }
        let modalities: Vec<Modality> = items.iter().map(|(item, _)| item.modality).collect();
        let order = self.prompt_order(input_ids, &modalities)?;
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
    fn decode_image(&self, bytes: &[u8]) -> Result<MmDecoded, ProcessorError> {
        let (rgb, height, width) = decode_rgb(bytes, &self.limits).map_err(mm_error)?;
        Ok(MmDecoded {
            media: DecodedMedia::Image { rgb, height, width },
            hash: content_hash_bytes(bytes),
        })
    }

    /// The frames `QwenVLImageProcessor._process_mm_data_uncached` hands HF for one video.
    fn decode_video(&self, video: &MmVideo<'_>) -> Result<MmDecoded, ProcessorError> {
        let Some(sampling) = &self.video else {
            return Err("this model takes no video input".into());
        };
        let config = sampling.config.with_hints(&video.hints).map_err(mm_error)?;
        let mut limits = VideoDecodeLimits::default();
        let source = Video::open(&video.bytes, &limits).map_err(mm_error)?;
        let info = source.info();
        // torchcodec's `len()` and `average_fps`, which Python samples from.
        let plan = |config: &VideoConfig| -> Result<(Vec<u64>, usize, usize), MmError> {
            let num_frames = sampling.num_frames(config, info.frame_count, info.average_fps)?;
            let indices = frame_indices(info.frame_count, num_frames);
            let (height, width) = sampling.presize(
                config,
                indices.len(),
                info.height as usize,
                info.width as usize,
            )?;
            Ok((indices, height, width))
        };
        let (indices, height, width) = plan(&config).map_err(mm_error)?;
        let side = |v: usize| {
            u32::try_from(v).map_err(|_| ProcessorError::InvalidRequest("video too large".into()))
        };
        let presize = FrameResize {
            width: side(width)?,
            height: side(height)?,
            filter: Resample::AtenF32Bilinear,
        };
        // Python decodes every frame the server's config plans, with a per-frame pixel floor;
        // that plan sets the output budget, so a request's hints cannot raise it.
        if let Ok((frames, height, width)) = plan(&sampling.config) {
            limits.max_output_bytes =
                (frames.len().saturating_mul(height * width * 3)).max(limits.max_output_bytes);
        }
        // torchcodec converts with swscale's x86 SIMD path, except at odd
        // heights, where swscale's generic path (not reproduced) runs.
        let rgb = match info.height % 2 {
            0 => RgbConversion::Swscale,
            _ => RgbConversion::Float,
        };
        let frames = source
            .decode_with(&indices, Some(&presize), &limits, rgb)
            .map_err(mm_error)?;
        let (height, width) = (frames.height(), frames.width());
        Ok(MmDecoded {
            media: DecodedMedia::Video {
                rgb: frames.into_rgb(),
                height,
                width,
                frame_indices: indices,
                fps: info.average_fps,
            },
            hash: content_hash_bytes(video.bytes),
        })
    }

    /// Indices into the request's items (`modalities`, in the caller's order) in
    /// placeholder order: each placeholder takes the next unused item of its modality.
    fn prompt_order(
        &self,
        input_ids: &[i32],
        modalities: &[Modality],
    ) -> Result<Vec<usize>, ProcessorError> {
        // Families that take one modality per prompt report no placeholder modalities;
        // their layout checks the items against the placeholders.
        if modalities.iter().all(|&modality| modality == modalities[0]) {
            return Ok((0..modalities.len()).collect());
        }
        let placeholders = self
            .family
            .placeholder_modalities(input_ids)
            .map_err(mm_error)?;
        let mut taken = vec![false; modalities.len()];
        let order: Option<Vec<usize>> = placeholders
            .iter()
            .map(|&placeholder| {
                let next =
                    (0..modalities.len()).find(|&i| !taken[i] && modalities[i] == placeholder)?;
                taken[next] = true;
                Some(next)
            })
            .collect();
        match order {
            Some(order) if order.len() == modalities.len() => Ok(order),
            _ => Err(format!(
                "prompt has {} placeholder(s) but the request has {} item(s)",
                tally(&placeholders),
                tally(modalities)
            )
            .into()),
        }
    }
}

/// Counts per modality, as `1 image, 2 video`.
fn tally(modalities: &[Modality]) -> String {
    let mut counts: Vec<(Modality, usize)> = Vec::new();
    for &modality in modalities {
        match counts.iter_mut().find(|(seen, _)| *seen == modality) {
            Some((_, count)) => *count += 1,
            None => counts.push((modality, 1)),
        }
    }
    let parts: Vec<String> = counts
        .iter()
        .map(|(modality, count)| format!("{count} {}", format!("{modality:?}").to_lowercase()))
        .collect();
    parts.join(", ")
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
