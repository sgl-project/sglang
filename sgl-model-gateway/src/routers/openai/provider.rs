//! Provider abstractions for vendor-specific API transformations.

use std::{collections::HashMap, sync::Arc};

use reqwest::RequestBuilder;
use serde_json::Value;
use thiserror::Error;

use crate::{
    core::{model_type::Endpoint, ProviderType},
    sglang_extensions::EXTENSION_FIELD_NAMES,
};

/// Strip every SGLang extension field from the request payload before
/// forwarding to OpenAI (which would 400 on unknown fields).
fn strip_sglang_fields(payload: &mut Value) {
    if let Some(obj) = payload.as_object_mut() {
        for field in EXTENSION_FIELD_NAMES {
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

    #[allow(dead_code)]
    pub fn get(&self, provider_type: &ProviderType) -> &dyn Provider {
        self.providers
            .get(provider_type)
            .map(|p| p.as_ref())
            .unwrap_or(self.default_provider.as_ref())
    }

    pub fn get_arc(&self, provider_type: &ProviderType) -> Arc<dyn Provider> {
        self.providers
            .get(provider_type)
            .cloned()
            .unwrap_or_else(|| Arc::clone(&self.default_provider))
    }

    #[allow(dead_code)]
    pub fn get_for_model(&self, model_name: &str) -> &dyn Provider {
        match ProviderType::from_model_name(model_name) {
            Some(pt) => self.get(&pt),
            None => self.default_provider.as_ref(),
        }
    }

    #[allow(dead_code)]
    pub fn default_provider(&self) -> &dyn Provider {
        self.default_provider.as_ref()
    }

    pub fn default_provider_arc(&self) -> Arc<dyn Provider> {
        Arc::clone(&self.default_provider)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn strip_sglang_fields_removes_every_known_extension() {
        // Build a payload that sets every EXTENSION_FIELD_NAMES entry alongside two
        // standard OpenAI fields. After stripping, only the OpenAI fields
        // should remain — otherwise we'd forward unknown fields to OpenAI's
        // API and get a 400 back.
        let mut payload = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
        });

        let payload_obj = payload.as_object_mut().unwrap();
        for field in EXTENSION_FIELD_NAMES {
            payload_obj.insert((*field).to_string(), json!("sentinel"));
        }

        strip_sglang_fields(&mut payload);

        let remaining: Vec<&str> = payload
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(remaining, vec!["model", "messages"]);
    }

    #[test]
    fn strip_sglang_fields_strips_every_new_extension_field() {
        // Spot-check the fields added in the SGLang RL extension PR. If a
        // new field name is misspelled in EXTENSION_FIELD_NAMES, this test fires.
        let new_fields = [
            "return_routed_experts",
            "return_cached_tokens_details",
            "return_prompt_token_ids",
            "return_meta_info",
            "input_ids",
            "stop_regex",
            "custom_logit_processor",
            "custom_params",
            "max_dynamic_patch",
            "min_dynamic_patch",
            "rid",
            "extra_key",
            "cache_salt",
            "bootstrap_host",
            "bootstrap_port",
            "bootstrap_room",
            "routed_dp_rank",
            "disagg_prefill_dp_rank",
            "data_parallel_rank",
        ];

        for field in new_fields {
            assert!(
                EXTENSION_FIELD_NAMES.contains(&field),
                "{field} missing from EXTENSION_FIELD_NAMES — would leak to OpenAI"
            );

            let mut payload = json!({ "model": "gpt-4o", field: 1 });
            strip_sglang_fields(&mut payload);
            assert!(
                payload.get(field).is_none(),
                "strip_sglang_fields did not remove {field}",
            );
        }
    }

    #[test]
    fn strip_sglang_fields_preserves_unknown_keys() {
        // Custom passthrough fields the user adds but aren't in EXTENSION_FIELD_NAMES
        // should survive the strip — only known SGLang extension keys are
        // removed.
        let mut payload = json!({
            "model": "gpt-4o",
            "custom_user_metadata": {"trace_id": "abc"},
            "return_routed_experts": true,
        });

        strip_sglang_fields(&mut payload);

        assert!(payload.get("custom_user_metadata").is_some());
        assert!(payload.get("return_routed_experts").is_none());
    }
}
