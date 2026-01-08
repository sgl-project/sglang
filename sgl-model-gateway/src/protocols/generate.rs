use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use validator::Validate;

use super::{
    common::{default_true, GenerationRequest, InputIds},
    sampling_params::SamplingParams,
};
use crate::protocols::validated::Normalizable;

// ============================================================================
// SGLang Generate API (native format)
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize, Validate)]
#[validate(schema(function = "validate_generate_request"))]
pub struct GenerateRequest {
    /// Text input - SGLang native format
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,

    pub model: Option<String>,

    /// Input IDs for tokenized input
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_ids: Option<InputIds>,

    /// Input embeddings for direct embedding input
    /// Can be a 2D array (single request) or 3D array (batch of requests)
    /// Placeholder for future use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_embeds: Option<Value>,

    /// Image input data
    /// Can be an image instance, file name, URL, or base64 encoded string
    /// Supports single images, lists of images, or nested lists for batch processing
    /// Placeholder for future use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_data: Option<Value>,

    /// Video input data
    /// Can be a file name, URL, or base64 encoded string
    /// Supports single videos, lists of videos, or nested lists for batch processing
    /// Placeholder for future use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub video_data: Option<Value>,

    /// Audio input data
    /// Can be a file name, URL, or base64 encoded string
    /// Supports single audio files, lists of audio, or nested lists for batch processing
    /// Placeholder for future use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_data: Option<Value>,

    /// Sampling parameters (sglang style)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling_params: Option<SamplingParams>,

    /// Whether to return logprobs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_logprob: Option<bool>,

    /// If return logprobs, the start location in the prompt for returning logprobs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprob_start_len: Option<i32>,

    /// If return logprobs, the number of top logprobs to return at each position.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs_num: Option<i32>,

    /// If return logprobs, the token ids to return logprob for.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_ids_logprob: Option<Vec<u32>>,

    /// Whether to detokenize tokens in text in the returned logprobs.
    #[serde(default)]
    pub return_text_in_logprobs: bool,

    /// Whether to stream the response
    #[serde(default)]
    pub stream: bool,

    /// Whether to log metrics for this request (e.g. health_generate calls do not log metrics)
    #[serde(default = "default_true")]
    pub log_metrics: bool,

    /// Return model hidden states
    #[serde(default)]
    pub return_hidden_states: bool,

    /// The modalities of the image data [image, multi-images, video]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<String>>,

    /// Session parameters for continual prompting
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_params: Option<HashMap<String, Value>>,

    /// Path to LoRA adapter(s) for model customization
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_path: Option<String>,

    /// LoRA adapter ID (if pre-loaded)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_id: Option<String>,

    /// Custom logit processor for advanced sampling control. Must be a serialized instance
    /// of `CustomLogitProcessor` in python/sglang/srt/sampling/custom_logit_processor.py
    /// Use the processor's `to_str()` method to generate the serialized string.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_logit_processor: Option<String>,

    /// For disaggregated inference
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bootstrap_host: Option<String>,

    /// For disaggregated inference
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bootstrap_port: Option<i32>,

    /// For disaggregated inference
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bootstrap_room: Option<i32>,

    /// For disaggregated inference
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bootstrap_pair_key: Option<String>,

    /// Data parallel rank routing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_parallel_rank: Option<i32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_parallel_rank_decode: Option<i32>,

    /// Background response
    #[serde(default)]
    pub background: bool,

    /// Conversation ID for tracking
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation_id: Option<String>,

    /// Priority for the request
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority: Option<i32>,

    /// Extra key for classifying the request (e.g. cache_salt)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_key: Option<String>,

    /// Whether to disallow logging for this request (e.g. due to ZDR)
    #[serde(default)]
    pub no_logs: bool,

    /// Custom metric labels
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_labels: Option<HashMap<String, String>>,

    /// Whether to return bytes for image generation
    #[serde(default)]
    pub return_bytes: bool,

    /// Whether to return entropy
    #[serde(default)]
    pub return_entropy: bool,

    /// Request ID for tracking (inherited from BaseReq in Python)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rid: Option<String>,
}

impl Normalizable for GenerateRequest {
    // Use default no-op implementation - no normalization needed for GenerateRequest
}

/// Validation function for GenerateRequest - ensure exactly one input type is provided
fn validate_generate_request(req: &GenerateRequest) -> Result<(), validator::ValidationError> {
    // Exactly one of text or input_ids must be provided
    // Note: input_embeds not yet supported in Rust implementation
    let has_text = req.text.is_some();
    let has_input_ids = req.input_ids.is_some();

    let count = [has_text, has_input_ids].iter().filter(|&&x| x).count();

    if count == 0 {
        return Err(validator::ValidationError::new(
            "Either text or input_ids should be provided.",
        ));
    }

    if count > 1 {
        return Err(validator::ValidationError::new(
            "Either text or input_ids should be provided.",
        ));
    }

    Ok(())
}

impl GenerationRequest for GenerateRequest {
    fn is_stream(&self) -> bool {
        self.stream
    }

    fn get_model(&self) -> Option<&str> {
        // Generate requests have an optional model field
        if let Some(s) = &self.model {
            Some(s.as_str())
        } else {
            None
        }
    }

    fn extract_text_for_routing(&self) -> String {
        // Check fields in priority order: text, input_ids
        if let Some(ref text) = self.text {
            return text.clone();
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

// ============================================================================
// SGLang Generate Response Types
// ============================================================================

/// SGLang generate response (single completion or array for n>1)
///
/// Format for n=1:
/// ```json
/// {
///   "text": "...",
///   "output_ids": [...],
///   "meta_info": { ... }
/// }
/// ```
///
/// Format for n>1:
/// ```json
/// [
///   {"text": "...", "output_ids": [...], "meta_info": {...}},
///   {"text": "...", "output_ids": [...], "meta_info": {...}}
/// ]
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub text: String,
    pub output_ids: Vec<u32>,
    pub meta_info: GenerateMetaInfo,
}

/// Metadata for a single generate completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateMetaInfo {
    pub id: String,
    pub finish_reason: GenerateFinishReason,
    pub prompt_tokens: u32,
    pub weight_version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_token_logprobs: Option<Vec<Vec<Option<f64>>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_token_logprobs: Option<Vec<Vec<Option<f64>>>>,
    pub completion_tokens: u32,
    pub cached_tokens: u32,
    pub e2e_latency: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matched_stop: Option<Value>,
}

/// Finish reason for generate endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum GenerateFinishReason {
    Length {
        length: u32,
    },
    Stop,
    #[serde(untagged)]
    Other(Value),
}
