//! Provider abstractions for vendor-specific API transformations.

use std::{collections::HashMap, sync::Arc};

use reqwest::RequestBuilder;
use serde_json::Value;
use thiserror::Error;

use crate::core::{model_type::Endpoint, ProviderType};

const SGLANG_FIELDS: &[&str] = &[
    "request_id",
    "priority",
    "top_k",
    "min_p",
    "min_tokens",
    "regex",
    "ebnf",
    "json_schema",
    "stop_token_ids",
    "no_stop_trim",
    "ignore_eos",
    "continue_final_message",
    "skip_special_tokens",
    "lora_path",
    "session_params",
    "separate_reasoning",
    "stream_reasoning",
    "chat_template",
    "chat_template_kwargs",
    "return_hidden_states",
    "repetition_penalty",
    "sampling_seed",
    "backend_url",
];

fn strip_sglang_fields(payload: &mut Value) {
    if let Some(obj) = payload.as_object_mut() {
        for field in SGLANG_FIELDS {
            obj.remove(*field);
        }
    }
}

#[derive(Error, Debug)]
pub enum ProviderError {
    #[error("Unsupported endpoint: {0:?}")]
    UnsupportedEndpoint(Endpoint),

    #[error("Transform error: {0}")]
    TransformError(String),
}

/// Default `transform_request` strips SGLang fields.
pub trait Provider: Send + Sync {
    fn provider_type(&self) -> ProviderType;

    fn transform_request(
        &self,
        payload: &mut Value,
        _endpoint: Endpoint,
    ) -> Result<(), ProviderError> {
        strip_sglang_fields(payload);
        Ok(())
    }

    fn transform_response(
        &self,
        _response: &mut Value,
        _endpoint: Endpoint,
    ) -> Result<(), ProviderError> {
        Ok(())
    }

    fn apply_headers(&self, builder: RequestBuilder) -> RequestBuilder {
        builder
    }
}

pub struct SGLangProvider;

impl Provider for SGLangProvider {
    fn provider_type(&self) -> ProviderType {
        ProviderType::OpenAI
    }

    fn transform_request(
        &self,
        _payload: &mut Value,
        _endpoint: Endpoint,
    ) -> Result<(), ProviderError> {
        Ok(())
    }
}

pub struct OpenAIProvider;

impl Provider for OpenAIProvider {
    fn provider_type(&self) -> ProviderType {
        ProviderType::OpenAI
    }
}

pub struct AnthropicProvider;

impl Provider for AnthropicProvider {
    fn provider_type(&self) -> ProviderType {
        ProviderType::Anthropic
    }
}

pub struct XAIProvider;

impl Provider for XAIProvider {
    fn provider_type(&self) -> ProviderType {
        ProviderType::XAI
    }

    fn transform_request(
        &self,
        payload: &mut Value,
        endpoint: Endpoint,
    ) -> Result<(), ProviderError> {
        strip_sglang_fields(payload);

        if endpoint == Endpoint::Responses {
            if let Some(obj) = payload.as_object_mut() {
                Self::transform_responses_input(obj);
            }
        }
        Ok(())
    }
}

impl XAIProvider {
    fn transform_responses_input(obj: &mut serde_json::Map<String, Value>) {
        let Some(input_arr) = obj.get_mut("input").and_then(Value::as_array_mut) else {
            return;
        };

        for item in input_arr.iter_mut().filter_map(Value::as_object_mut) {
            item.remove("id");
            item.remove("status");

            let Some(content_arr) = item.get_mut("content").and_then(Value::as_array_mut) else {
                continue;
            };

            for content in content_arr.iter_mut().filter_map(Value::as_object_mut) {
                if content.get("type").and_then(Value::as_str) == Some("output_text") {
                    content.insert("type".to_string(), Value::String("input_text".to_string()));
                }
            }
        }
    }
}

pub struct GeminiProvider;

impl Provider for GeminiProvider {
    fn provider_type(&self) -> ProviderType {
        ProviderType::Gemini
    }

    fn transform_request(
        &self,
        payload: &mut Value,
        endpoint: Endpoint,
    ) -> Result<(), ProviderError> {
        strip_sglang_fields(payload);

        if endpoint == Endpoint::Chat {
            if let Some(obj) = payload.as_object_mut() {
                if obj.get("logprobs").and_then(|v| v.as_bool()) == Some(false) {
                    obj.remove("logprobs");
                }
            }
        }
        Ok(())
    }
}

pub struct ProviderRegistry {
    providers: HashMap<ProviderType, Arc<dyn Provider>>,
    default_provider: Arc<dyn Provider>,
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ProviderRegistry {
    pub fn new() -> Self {
        let mut providers = HashMap::new();

        providers.insert(
            ProviderType::OpenAI,
            Arc::new(OpenAIProvider) as Arc<dyn Provider>,
        );
        providers.insert(
            ProviderType::XAI,
            Arc::new(XAIProvider) as Arc<dyn Provider>,
        );
        providers.insert(
            ProviderType::Gemini,
            Arc::new(GeminiProvider) as Arc<dyn Provider>,
        );
        providers.insert(
            ProviderType::Anthropic,
            Arc::new(AnthropicProvider) as Arc<dyn Provider>,
        );

        Self {
            providers,
            default_provider: Arc::new(SGLangProvider),
        }
    }

    pub fn get(&self, provider_type: &ProviderType) -> &dyn Provider {
        self.providers
            .get(provider_type)
            .map(|p| p.as_ref())
            .unwrap_or(self.default_provider.as_ref())
    }

    pub fn get_for_model(&self, model_name: &str) -> &dyn Provider {
        match ProviderType::from_model_name(model_name) {
            Some(pt) => self.get(&pt),
            None => self.default_provider.as_ref(),
        }
    }

    pub fn default_provider(&self) -> &dyn Provider {
        self.default_provider.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn test_sglang_provider_passthrough() {
        let provider = SGLangProvider;
        let mut payload = json!({"regex": ".*", "top_k": 50});

        provider
            .transform_request(&mut payload, Endpoint::Chat)
            .unwrap();

        assert!(payload.get("regex").is_some());
        assert!(payload.get("top_k").is_some());
    }

    #[test]
    fn test_openai_provider_strips_sglang_fields() {
        let provider = OpenAIProvider;
        let mut payload = json!({"regex": ".*", "top_k": 50, "temperature": 0.7});

        provider
            .transform_request(&mut payload, Endpoint::Chat)
            .unwrap();

        assert!(payload.get("regex").is_none());
        assert!(payload.get("top_k").is_none());
        assert!(payload.get("temperature").is_some());
    }

    #[test]
    fn test_xai_provider_transforms_responses_input() {
        let provider = XAIProvider;
        let mut payload = json!({
            "input": [{
                "id": "msg_123",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Hello"}]
            }]
        });

        provider
            .transform_request(&mut payload, Endpoint::Responses)
            .unwrap();

        let item = &payload["input"][0];
        assert!(item.get("id").is_none());
        assert!(item.get("status").is_none());
        assert_eq!(item["content"][0]["type"], "input_text");
    }

    #[test]
    fn test_gemini_provider_removes_false_logprobs() {
        let provider = GeminiProvider;
        let mut payload = json!({"logprobs": false});

        provider
            .transform_request(&mut payload, Endpoint::Chat)
            .unwrap();

        assert!(payload.get("logprobs").is_none());
    }

    #[test]
    fn test_gemini_provider_keeps_true_logprobs() {
        let provider = GeminiProvider;
        let mut payload = json!({"logprobs": true});

        provider
            .transform_request(&mut payload, Endpoint::Chat)
            .unwrap();

        assert_eq!(payload.get("logprobs").unwrap(), true);
    }

    #[test]
    fn test_provider_registry_lookup() {
        let registry = ProviderRegistry::new();

        assert_eq!(
            registry.get(&ProviderType::OpenAI).provider_type(),
            ProviderType::OpenAI
        );
        assert_eq!(
            registry.get(&ProviderType::XAI).provider_type(),
            ProviderType::XAI
        );

        let custom = ProviderType::Custom("unknown".to_string());
        assert_eq!(registry.get(&custom).provider_type(), ProviderType::OpenAI);
    }

    #[test]
    fn test_provider_registry_get_for_model() {
        let registry = ProviderRegistry::new();

        assert_eq!(
            registry.get_for_model("gpt-4").provider_type(),
            ProviderType::OpenAI
        );
        assert_eq!(
            registry.get_for_model("grok-2").provider_type(),
            ProviderType::XAI
        );
        assert_eq!(
            registry.get_for_model("gemini-pro").provider_type(),
            ProviderType::Gemini
        );
        assert_eq!(
            registry.get_for_model("llama-3.1-8b").provider_type(),
            ProviderType::OpenAI
        );
    }
}
