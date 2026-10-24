//! Image pipeline selection from a model's HF config files and SGLang's server args.

use dynamo_multimodal::registry::{build_processor, spec_from_hf_configs};
use serde_json::Value;

use super::{MmProcessor, mm_error, models};
use crate::ProcessorError;
use crate::model_files::resolve_model_file;

/// Model sources and the server args that change image preprocessing.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MmProcessorOptions {
    /// Where `config.json` lives; empty means `tokenizer_path`.
    pub model_path: String,
    /// Where the HF processor files live, as Python's `get_processor(tokenizer_path)`.
    pub tokenizer_path: String,
    pub revision: Option<String>,
    /// `--image-processor-backend`: `auto`, `torchvision` or `pil`.
    pub image_processor_backend: Option<String>,
    /// `--mm-process-config`'s `image` object.
    pub image_overrides: Option<Value>,
}

/// Build the model's image pipeline. Errors name what SGLang cannot serve
/// bit-exactly, so hosts can refuse to start rather than diverge from Python.
pub fn select_mm_processor(options: &MmProcessorOptions) -> Result<MmProcessor, ProcessorError> {
    let config = read_model_file(&model_source(options), options, &["config.json"])?;
    let processor_config = read_model_file(
        &options.tokenizer_path,
        options,
        &["preprocessor_config.json", "processor_config.json"],
    )?;
    let mut spec = spec_from_hf_configs(&config, &processor_config).map_err(mm_error)?;
    models::apply_sglang_rules(&mut spec, options)?;
    let family = build_processor(spec).map_err(mm_error)?;
    Ok(MmProcessor::new(family))
}

fn model_source(options: &MmProcessorOptions) -> String {
    if options.model_path.is_empty() {
        options.tokenizer_path.clone()
    } else {
        options.model_path.clone()
    }
}

/// The first of `filenames` found under `path`, read as text.
fn read_model_file(
    path: &str,
    options: &MmProcessorOptions,
    filenames: &[&str],
) -> Result<String, ProcessorError> {
    let file = filenames
        .iter()
        .find_map(|name| resolve_model_file(path, options.revision.as_deref(), name))
        .ok_or_else(|| format!("{path}: none of {filenames:?} found"))?;
    std::fs::read_to_string(&file).map_err(|error| format!("cannot read {file}: {error}").into())
}
