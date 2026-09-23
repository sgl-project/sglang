use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::protocols::{
    common::{GenerationRequest, InputIds, StringOrArray},
    completion::CompletionRequest as TextCompletionRequest,
};

/// `/v1/completions` request accepted by the gateway.
///
/// `openai-protocol` currently models `prompt` as string-or-string-array,
/// while SGLang's Python endpoint also accepts token IDs. Keep the existing
/// typed request for text prompts and use a transparent envelope for token-ID
/// prompts so extension fields are forwarded unchanged.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum CompletionRequest {
    Text(Box<TextCompletionRequest>),
    TokenIds(TokenIdCompletionRequest),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TokenIdCompletionRequest {
    pub model: String,
    pub prompt: InputIds,
    #[serde(default)]
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<u32>,
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

impl CompletionRequest {
    pub fn model(&self) -> &str {
        match self {
            Self::Text(request) => &request.model,
            Self::TokenIds(request) => &request.model,
        }
    }

    pub fn logprobs(&self) -> Option<u32> {
        match self {
            Self::Text(request) => request.logprobs,
            Self::TokenIds(request) => request.logprobs,
        }
    }

    pub fn batch_size(&self) -> Option<usize> {
        match self {
            Self::Text(request) => match &request.prompt {
                StringOrArray::Array(prompts) if !prompts.is_empty() => Some(prompts.len()),
                _ => None,
            },
            Self::TokenIds(request) => match &request.prompt {
                InputIds::Batch(prompts) if !prompts.is_empty() => Some(prompts.len()),
                _ => None,
            },
        }
    }

    pub fn first_prompt_for_routing(&self) -> Option<String> {
        match self {
            Self::Text(request) => match &request.prompt {
                StringOrArray::String(prompt) => Some(prompt.clone()),
                StringOrArray::Array(prompts) => prompts.first().cloned(),
            },
            // The current cache-aware policy matches character prefixes. Rendering
            // token IDs as text would create false matches between unrelated token
            // sequences, so token-ID requests use the no-request-text fallback.
            Self::TokenIds(_) => None,
        }
    }
}

impl GenerationRequest for CompletionRequest {
    fn is_stream(&self) -> bool {
        match self {
            Self::Text(request) => request.stream,
            Self::TokenIds(request) => request.stream,
        }
    }

    fn get_model(&self) -> Option<&str> {
        Some(self.model())
    }

    fn extract_text_for_routing(&self) -> String {
        self.first_prompt_for_routing().unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn accepts_and_round_trips_all_completion_prompt_shapes() {
        let cases = [
            json!("hello"),
            json!(["hello", "world"]),
            json!([1, 2, 3]),
            json!([[1, 2], [3, 4]]),
        ];

        for prompt in cases {
            let payload = json!({
                "model": "test-model",
                "prompt": prompt,
                "max_tokens": 4,
                "custom_extension": true,
            });
            let request: CompletionRequest = serde_json::from_value(payload.clone()).unwrap();
            let serialized = serde_json::to_value(request).unwrap();
            assert_eq!(serialized["prompt"], payload["prompt"]);
            assert_eq!(serialized["custom_extension"], true);
        }
    }

    #[test]
    fn distinguishes_token_sequence_from_token_batch() {
        let single: CompletionRequest = serde_json::from_value(json!({
            "model": "test-model",
            "prompt": [1, 2, 3]
        }))
        .unwrap();
        let batch: CompletionRequest = serde_json::from_value(json!({
            "model": "test-model",
            "prompt": [[1, 2], [3, 4]]
        }))
        .unwrap();

        assert_eq!(single.batch_size(), None);
        assert_eq!(batch.batch_size(), Some(2));
    }

    #[test]
    fn preserves_text_prompt_routing_behavior() {
        let single: CompletionRequest = serde_json::from_value(json!({
            "model": "test-model",
            "prompt": "shared text prefix"
        }))
        .unwrap();
        let batch: CompletionRequest = serde_json::from_value(json!({
            "model": "test-model",
            "prompt": ["first prompt", "second prompt"]
        }))
        .unwrap();

        assert_eq!(
            single.first_prompt_for_routing().as_deref(),
            Some("shared text prefix")
        );
        assert_eq!(
            batch.first_prompt_for_routing().as_deref(),
            Some("first prompt")
        );
    }

    #[test]
    fn token_id_prompts_do_not_create_text_routing_keys() {
        for prompt in [json!([1]), json!([2]), json!([[1, 2], [3, 4]])] {
            let request: CompletionRequest = serde_json::from_value(json!({
                "model": "test-model",
                "prompt": prompt
            }))
            .unwrap();

            assert_eq!(request.first_prompt_for_routing(), None);
            assert_eq!(request.extract_text_for_routing(), "");
        }
    }
}
