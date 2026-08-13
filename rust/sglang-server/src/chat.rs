//! Optional model-package chat preprocessing.
//!
//! The default server renders Hugging Face or legacy templates internally.
//! External model packages can instead return an already-tokenized prompt and
//! opaque media sources while retaining the shared OpenAI response pipeline.

use std::fmt::Debug;

/// One chat request after a model package has rendered and tokenized it.
#[derive(Clone, Debug, Default)]
pub struct NativeChatOutput {
    pub input_ids: Vec<i32>,
    pub image_data: Option<rmpv::Value>,
    pub video_data: Option<rmpv::Value>,
    pub audio_data: Option<rmpv::Value>,
    pub multimodal_placeholders: Option<rmpv::Value>,
}

/// Model-owned chat rendering boundary.
///
/// The raw JSON is intentional: private extensions may carry message metadata
/// that is not part of the public OpenAI schema and must survive rendering.
pub trait NativeChatProcessor: Debug + Send + Sync + 'static {
    fn process(&self, request_json: &str) -> Result<NativeChatOutput, String>;
}
