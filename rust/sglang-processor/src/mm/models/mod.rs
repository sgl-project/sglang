//! Families whose pixel path Dynamo does not ship yet, as Dynamo
//! `MmFamilyProcessor`s; each is deleted once Dynamo's own family processes pixels.

mod deepseek_v41;
mod glm5_next;

use dynamo_multimodal::MmError;
use dynamo_multimodal::image::decode::{DecodeLimits, decode_rgb};
use dynamo_multimodal::processor::MmFamilyProcessor;
use dynamo_multimodal::registry::{ProcessorSpec, build_processor};

/// The pipeline spec SGLang resolves (`RustMmSpec.rust_json()`): a family whose
/// spec Dynamo lacks fields for, or else Dynamo's `ProcessorSpec`.
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(untagged)]
pub enum MmPipelineSpec {
    Glm5Next(glm5_next::Glm5NextImageSpec),
    Dynamo(ProcessorSpec),
}

/// Encoded bytes to HWC RGB, as the family's Python processor loads images.
pub(super) type Decode = fn(&[u8], &DecodeLimits) -> Result<(Vec<u8>, usize, usize), MmError>;

pub(super) fn build(spec: MmPipelineSpec) -> Result<(Box<dyn MmFamilyProcessor>, Decode), MmError> {
    Ok(match spec {
        MmPipelineSpec::Glm5Next(spec) => {
            (Box::new(glm5_next::Glm5Next::new(spec)?), glm5_next::decode)
        }
        MmPipelineSpec::Dynamo(ProcessorSpec::DeepseekV41(spec)) => {
            (Box::new(deepseek_v41::DeepseekV41::new(spec)?), decode_rgb)
        }
        MmPipelineSpec::Dynamo(spec) => (build_processor(spec)?, decode_rgb),
    })
}
