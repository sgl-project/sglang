//! Extended ChatCompletionRequest wrapper that preserves unknown fields during
//! deserialization, enabling transparent pass-through of engine-specific parameters
//! not explicitly modeled in the `openai-protocol` crate.

use std::collections::HashMap;
use std::ops::Deref;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::protocols::chat::ChatCompletionRequest;
use crate::protocols::common::GenerationRequest;

/// A wrapper around [`ChatCompletionRequest`] that captures any extra JSON fields
/// not defined in the upstream struct via `#[serde(flatten)]`.
///
/// On serialization the known fields and extras are merged back into a single
/// JSON object, so forwarding preserves the full original payload.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ExtendedChatCompletionRequest {
    #[serde(flatten)]
    pub inner: ChatCompletionRequest,

    /// Extra fields not modeled in `ChatCompletionRequest` (e.g. `rid`,
    /// `priority`, `input_ids`, `data_parallel_rank`, `stop_regex`, …).
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

impl Deref for ExtendedChatCompletionRequest {
    type Target = ChatCompletionRequest;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl GenerationRequest for ExtendedChatCompletionRequest {
    fn is_stream(&self) -> bool {
        self.inner.is_stream()
    }

    fn get_model(&self) -> Option<&str> {
        self.inner.get_model()
    }

    fn extract_text_for_routing(&self) -> String {
        self.inner.extract_text_for_routing()
    }
}
