//! Native multimodal processing adapter.
//!
//! The model-independent driver lives in `sglang-mm`; this module supplies the
//! server tokenizer and parks successful results in the scheduler sidecar.

use super::NativeContext;
use crate::message::MmRequest;

/// Run the native pipeline for one request. `Ok` returns the final
/// placeholder-expanded ids (the mm buffers are parked in the sidecar strictly
/// before returning). `Err` carries the driver's classification — `fallback:`
/// (outside the native pipeline's scope: video/audio, precomputed inputs,
/// undecodable images) or `failed:` (hard error) — and rejects the request
/// back to the client; there is no Python fallback path.
pub fn process(ctx: &NativeContext, req: &MmRequest) -> Result<Vec<i32>, String> {
    let output = sglang_mm::native_driver::process(&ctx.pipeline, &req.payload, |text| {
        let tokenizer = ctx.tokenizer.as_ref().ok_or_else(|| {
            "skip_tokenizer_init is set: multimodal text prompts require input_ids".to_string()
        })?;
        tokenizer.encode(text).map_err(|error| error.to_string())
    })
    .map_err(|error| error.to_string())?;
    let input_ids = output.input_ids;
    ctx.sidecar
        .lock()
        .unwrap()
        .insert(req.rid.clone(), output.mm);
    Ok(input_ids)
}
