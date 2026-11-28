//! Specs `dynamo-multimodal` resolves from a checkpoint directory, finished
//! for SGLang's pixel pipeline, with SGLang's video policy for the family.

use std::path::Path;

use dynamo_multimodal::models::nemotron_vl::{NemotronVlProcessor, NemotronVlSizing};
use dynamo_multimodal::models::qwen_vl::Resampler;
use dynamo_multimodal::registry::{self, ProcessorSpec};

use super::{VideoPolicy, mm_error};
use crate::ProcessorError;

/// A family whose spec comes from the checkpoint directory.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MmDirFamily {
    /// GLM-5.3-Flash (`glm5_next`), which `Glm4vImageProcessor` serves.
    Glm5Next,
    /// Nemotron 3.5 Super VL (`nemotron_h_omni`), which
    /// `NanoNemotronVLImageProcessor` serves with these server and config values.
    NemotronVl {
        /// `--context-length`, or 8192 when unset.
        max_model_len: u32,
        /// `hf_config.video_target_num_patches` and `video_maintain_aspect_ratio`,
        /// which SGLang reads from `vision_config` and the resolver does not.
        video_target_num_patches: u32,
        video_maintain_aspect_ratio: bool,
        /// `hf_config.video_pruning_rate`, which `--json-model-override-args` can set.
        video_pruning_rate: f64,
    },
}

/// `family`'s spec for the checkpoint in `dir`, with the image `resample` of
/// the processor SGLang loaded, and how SGLang samples its videos (`None`:
/// the checkpoint takes no video).
pub fn spec_from_model_dir(
    family: MmDirFamily,
    dir: &Path,
    resample: Resampler,
) -> Result<(ProcessorSpec, Option<VideoPolicy>), ProcessorError> {
    match family {
        // Not `registry::spec_from_model_dir`: it reads the flat processor file first.
        MmDirFamily::Glm5Next => {
            let config = read(dir, &["config.json"])?;
            let processor = processor_json(dir)?;
            let mut spec = registry::glm5_next_pixel_spec(&config, &processor).map_err(mm_error)?;
            // `--image-processor-backend` can load another class than the checkpoint declares.
            spec.resample = resample;
            let video = (spec.planner.video.as_ref()).map(|video| VideoPolicy::Glm {
                fps: video.fps,
                max_frames: video.max_frames,
            });
            Ok((ProcessorSpec::Glm5NextImage(spec), video))
        }
        MmDirFamily::NemotronVl {
            max_model_len,
            video_target_num_patches,
            video_maintain_aspect_ratio,
            video_pruning_rate,
        } => {
            let ProcessorSpec::NemotronVl(mut spec) =
                registry::spec_from_model_dir(dir).map_err(mm_error)?
            else {
                return Err(
                    format!("{} does not hold a {family:?} checkpoint", dir.display()).into(),
                );
            };
            spec.sizing = NemotronVlSizing::Sglang;
            spec.max_model_len = max_model_len;
            if let Some(video) = &mut spec.video {
                video.target_num_patches = video_target_num_patches;
                video.maintain_aspect_ratio = video_maintain_aspect_ratio;
                video.pruning_rate = video_pruning_rate;
            }
            // A video spec the family refuses turns video off; images still serve.
            if spec.video.is_some() && NemotronVlProcessor::new(spec.clone()).is_err() {
                spec.video = None;
            }
            let video = spec.video.as_ref().map(|_| VideoPolicy::Nemotron);
            Ok((ProcessorSpec::NemotronVl(spec), video))
        }
    }
}

/// The image processor file Transformers loads: `processor_config.json` when
/// it nests a non-null `image_processor`, else `preprocessor_config.json`.
fn processor_json(dir: &Path) -> Result<String, ProcessorError> {
    if let Ok(json) = std::fs::read_to_string(dir.join("processor_config.json")) {
        let nested = serde_json::from_str::<serde_json::Value>(&json).is_ok_and(|top| {
            top.get("image_processor")
                .is_some_and(|nested| !nested.is_null())
        });
        if nested {
            return Ok(json);
        }
    }
    read(dir, &["preprocessor_config.json", "processor_config.json"])
}

/// The first of `names` in `dir`.
fn read(dir: &Path, names: &[&str]) -> Result<String, ProcessorError> {
    let path = (names.iter())
        .map(|name| dir.join(name))
        .find(|path| path.is_file())
        .ok_or_else(|| format!("{} has no {}", dir.display(), names.join(" or ")))?;
    std::fs::read_to_string(&path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()).into())
}
