//! Families whose pixel path Dynamo does not ship yet, as Dynamo
//! `MmFamilyProcessor`s; each is deleted once Dynamo's own family processes pixels.

mod deepseek_v41;

use dynamo_multimodal::MmError;
use dynamo_multimodal::processor::MmFamilyProcessor;
use dynamo_multimodal::registry::{ProcessorSpec, build_processor};

pub(super) fn build(spec: ProcessorSpec) -> Result<Box<dyn MmFamilyProcessor>, MmError> {
    match spec {
        ProcessorSpec::DeepseekV41(spec) => Ok(Box::new(deepseek_v41::DeepseekV41::new(spec)?)),
        spec => build_processor(spec),
    }
}
