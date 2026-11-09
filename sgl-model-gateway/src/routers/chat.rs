//! HTTP chat extensions not yet represented by openai-protocol.

use std::ops::Deref;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use validator::{Validate, ValidationErrors};

use crate::protocols::{
    chat::ChatCompletionRequest, common::GenerationRequest, validated::Normalizable,
};

/// Keep SGLang custom sampling parameters alongside the validated OpenAI request.
/// Flattening preserves the backend's top-level JSON layout.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatRequest {
    #[serde(flatten)]
    pub request: ChatCompletionRequest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom_params: Option<Map<String, Value>>,
}

impl From<ChatCompletionRequest> for ChatRequest {
    fn from(request: ChatCompletionRequest) -> Self {
        Self {
            request,
            custom_params: None,
        }
    }
}

impl Deref for ChatRequest {
    type Target = ChatCompletionRequest;

    fn deref(&self) -> &Self::Target {
        &self.request
    }
}

impl Validate for ChatRequest {
    fn validate(&self) -> Result<(), ValidationErrors> {
        self.request.validate()
    }
}

impl Normalizable for ChatRequest {
    fn normalize(&mut self) {
        self.request.normalize();
    }
}

impl GenerationRequest for ChatRequest {
    fn is_stream(&self) -> bool {
        self.request.is_stream()
    }

    fn get_model(&self) -> Option<&str> {
        self.request.get_model()
    }

    fn extract_text_for_routing(&self) -> String {
        self.request.extract_text_for_routing()
    }
}
