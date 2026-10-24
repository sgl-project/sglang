//! Qwen-VL rules SGLang applies on top of Dynamo's HF-config resolution.

use dynamo_multimodal::models::qwen_vl::{QwenVlSpec, Resampler};
use serde_json::Value;

use crate::ProcessorError;
use crate::mm::MmProcessorOptions;

pub(super) fn apply(
    spec: &mut QwenVlSpec,
    options: &MmProcessorOptions,
) -> Result<(), ProcessorError> {
    if let Some(resample) = backend_resampler(options.image_processor_backend.as_deref())? {
        spec.resample = resample;
    }
    if let Some(overrides) = &options.image_overrides {
        apply_pixel_overrides(spec, overrides)?;
    }
    Ok(())
}

/// Mirrors `utils/hf_transformers/processor.py::_apply_image_processor_backend`:
/// `auto` keeps the checkpoint's class, the others reload the image processor.
fn backend_resampler(backend: Option<&str>) -> Result<Option<Resampler>, ProcessorError> {
    match backend.unwrap_or("auto") {
        "auto" => Ok(None),
        "torchvision" => Ok(Some(Resampler::AtenU8)),
        "pil" => Ok(Some(Resampler::Pil)),
        other => Err(format!(
            "Unsupported image processor backend: {other}. \
             Expected one of ['auto', 'pil', 'torchvision']."
        )
        .into()),
    }
}

/// Mirrors `rust_server/multimodal.py::RustMmProcessor.resolve_spec`: only the
/// pixel limits of `--mm-process-config.image` are honored, anything else is refused.
fn apply_pixel_overrides(spec: &mut QwenVlSpec, overrides: &Value) -> Result<(), ProcessorError> {
    let overrides = overrides
        .as_object()
        .ok_or("--mm-process-config image overrides must be an object")?;
    for (key, value) in overrides {
        let field = match key.as_str() {
            "min_pixels" => &mut spec.min_pixels,
            "max_pixels" => &mut spec.max_pixels,
            _ => {
                return Err(format!(
                    "--mm-process-config image.{key} is not supported by the Rust image pipeline; \
                     only min_pixels and max_pixels are"
                )
                .into());
            }
        };
        *field = value.as_u64().ok_or_else(|| {
            format!("--mm-process-config image.{key} must be a non-negative integer")
        })? as usize;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn spec() -> QwenVlSpec {
        serde_json::from_value(json!({
            "image_token_id": 151655,
            "patch_size": 14,
            "merge_size": 2,
            "temporal_patch_size": 2,
            "min_pixels": 3136,
            "max_pixels": 12845056,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        }))
        .unwrap()
    }

    fn options(backend: Option<&str>, overrides: Option<Value>) -> MmProcessorOptions {
        MmProcessorOptions {
            image_processor_backend: backend.map(str::to_owned),
            image_overrides: overrides,
            ..MmProcessorOptions::default()
        }
    }

    #[test]
    fn backend_selects_resampler() {
        for (backend, expected) in [
            (None, Resampler::AtenU8),
            (Some("auto"), Resampler::AtenU8),
            (Some("torchvision"), Resampler::AtenU8),
            (Some("pil"), Resampler::Pil),
        ] {
            let mut spec = spec();
            apply(&mut spec, &options(backend, None)).unwrap();
            assert_eq!(spec.resample, expected, "{backend:?}");
        }
        assert!(apply(&mut spec(), &options(Some("cv2"), None)).is_err());
    }

    #[test]
    fn pixel_overrides() {
        let mut spec = spec();
        let overrides = json!({"min_pixels": 1024, "max_pixels": 4096});
        apply(&mut spec, &options(None, Some(overrides))).unwrap();
        assert_eq!((spec.min_pixels, spec.max_pixels), (1024, 4096));

        for bad in [
            json!({"do_resize": false}),
            json!({"max_pixels": "4096"}),
            json!([1]),
        ] {
            assert!(
                apply(&mut spec.clone(), &options(None, Some(bad.clone()))).is_err(),
                "{bad}"
            );
        }
    }
}
