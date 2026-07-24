//! Native multimodal processing adapter.
//!
//! The model-independent driver lives in `sglang-mm`; this module supplies the
//! server tokenizer and parks successful results in the scheduler sidecar.

use sglang_mm::common::payload::NativeError;

use super::NativeContext;
use crate::message::MmRequest;

pub enum Outcome {
    Done(Vec<i32>),
    Fallback(String),
    Failed(String),
}

pub fn process(ctx: &NativeContext, req: &MmRequest) -> Outcome {
    let output = sglang_mm::native_driver::process(&ctx.pipeline, &req.payload, |text| {
        let tokenizer = ctx.tokenizer.as_ref().ok_or_else(|| {
            "skip_tokenizer_init is set: multimodal text prompts require input_ids".to_string()
        })?;
        tokenizer.encode(text).map_err(|error| error.to_string())
    });
    match output {
        Ok(output) => {
            let input_ids = output.input_ids;
            ctx.sidecar
                .lock()
                .unwrap()
                .insert(req.rid.clone(), output.mm);
            Outcome::Done(input_ids)
        }
        Err(NativeError::Fallback(message)) => Outcome::Fallback(message),
        Err(NativeError::Failed(message)) => Outcome::Failed(message),
    }
}
