//! Where SGLang's Python preprocessing differs from Dynamo's, per family.

use dynamo_multimodal::registry::ProcessorSpec;

use super::MmProcessorOptions;
use crate::ProcessorError;

mod qwen_vl;

/// Apply SGLang's server args and Python-side defaults to a spec Dynamo resolved.
pub(super) fn apply_sglang_rules(
    spec: &mut ProcessorSpec,
    options: &MmProcessorOptions,
) -> Result<(), ProcessorError> {
    match spec {
        ProcessorSpec::QwenVl(spec) => qwen_vl::apply(spec, options),
        _ => Ok(()),
    }
}
