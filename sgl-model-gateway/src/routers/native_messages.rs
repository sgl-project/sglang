//! Native Messages requests are forwarded without an OpenAI conversion.
//! The engine owns protocol validation, tools, reasoning and multimodal semantics.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::protocols::common::GenerationRequest;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(transparent)]
pub struct NativeMessagesRequest(pub Value);

impl GenerationRequest for NativeMessagesRequest {
    fn is_stream(&self) -> bool {
        self.0
            .get("stream")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }

    fn get_model(&self) -> Option<&str> {
        self.0.get("model").and_then(Value::as_str)
    }

    fn extract_text_for_routing(&self) -> String {
        // This is only a routing key, never a replacement generation payload.
        // Include system/tools/media identity so distinct prefixes do not alias.
        let mut prefix = self.0.clone();
        if let Some(object) = prefix.as_object_mut() {
            object.retain(|key, _| matches!(key.as_str(), "system" | "tools" | "messages"));
        }
        prefix.to_string()
    }
}
