// Generate API request types (/generate)

use crate::protocols::common::{GenerationRequest, LoRAPath, StringOrArray};
use crate::protocols::generate::types::{GenerateParameters, InputIds, SamplingParams};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerateRequest {
    /// The prompt to generate from (OpenAI style)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<StringOrArray>,

    /// Text input - SGLang native format
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,

    /// Input IDs for tokenized input
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_ids: Option<InputIds>,

    /// Generation parameters
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<GenerateParameters>,

    /// Sampling parameters (sglang style)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling_params: Option<SamplingParams>,

    /// Whether to stream the response
    #[serde(default)]
    pub stream: bool,

    /// Whether to return logprobs
    #[serde(default)]
    pub return_logprob: bool,

    // ============= SGLang Extensions =============
    /// Path to LoRA adapter(s) for model customization
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_path: Option<LoRAPath>,

    /// Session parameters for continual prompting
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_params: Option<HashMap<String, serde_json::Value>>,

    /// Return model hidden states
    #[serde(default)]
    pub return_hidden_states: bool,

    /// Request ID for tracking
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rid: Option<String>,
}

impl GenerationRequest for GenerateRequest {
    fn is_stream(&self) -> bool {
        self.stream
    }

    fn get_model(&self) -> Option<&str> {
        // Generate requests typically don't have a model field
        None
    }

    fn extract_text_for_routing(&self) -> String {
        // Check fields in priority order: text, prompt, inputs
        if let Some(ref text) = self.text {
            return text.clone();
        }

        if let Some(ref prompt) = self.prompt {
            return match prompt {
                StringOrArray::String(s) => s.clone(),
                StringOrArray::Array(v) => v.join(" "),
            };
        }

        if let Some(ref input_ids) = self.input_ids {
            return match input_ids {
                InputIds::Single(ids) => ids
                    .iter()
                    .map(|&id| id.to_string())
                    .collect::<Vec<String>>()
                    .join(" "),
                InputIds::Batch(batches) => batches
                    .iter()
                    .flat_map(|batch| batch.iter().map(|&id| id.to_string()))
                    .collect::<Vec<String>>()
                    .join(" "),
            };
        }

        // No text input found
        String::new()
    }
}
