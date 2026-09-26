//! SGLang extends the OpenAI chat candidate limit without changing the request.
use openai_protocol::{chat::ChatCompletionRequest, validated::Normalizable};
use serde::Deserialize;
use validator::{Validate, ValidationError, ValidationErrors, ValidationErrorsKind};

/// Keep all protocol validation, allowing the backend's larger candidate head.
#[derive(Debug, Deserialize)]
#[serde(transparent)]
pub struct SglangChatRequest(pub ChatCompletionRequest);

impl Normalizable for SglangChatRequest {
    fn normalize(&mut self) {
        self.0.normalize();
    }
}

impl Validate for SglangChatRequest {
    fn validate(&self) -> Result<(), ValidationErrors> {
        let mut errors = self.0.validate().err().unwrap_or_default();
        // The pinned OpenAI schema imposes a provider-specific maximum of 20.
        // Remove only that range error; preserve every other validation error.
        if let Some(ValidationErrorsKind::Field(field_errors)) =
            errors.errors_mut().get_mut("top_logprobs")
        {
            field_errors.retain(|error| {
                !(error.code == "range" && error.params.get("max") == Some(&serde_json::json!(20)))
            });
            if field_errors.is_empty() {
                errors.errors_mut().remove("top_logprobs");
            }
        }
        if let Some(value) = self.0.top_logprobs.filter(|value| *value > 128) {
            let mut error = ValidationError::new("range");
            error.add_param("min".into(), &0);
            error.add_param("max".into(), &128);
            error.add_param("value".into(), &value);
            errors.add("top_logprobs", error);
        }
        if errors.errors().is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request(top_logprobs: u32) -> SglangChatRequest {
        serde_json::from_value(json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
            "logprobs": true,
            "top_logprobs": top_logprobs
        }))
        .unwrap()
    }

    #[test]
    fn accepts_extended_head_without_modifying_request() {
        for count in [0, 16, 20, 21, 128] {
            let mut body = request(count);
            body.normalize();
            assert!(body.validate().is_ok(), "{count}");
            assert_eq!(
                serde_json::to_value(&body.0).unwrap()["top_logprobs"],
                count
            );
        }
    }

    #[test]
    fn rejects_head_above_backend_limit() {
        for count in [129, u32::MAX] {
            let errors = request(count).validate().unwrap_err();
            assert!(errors.field_errors().contains_key("top_logprobs"));
        }
    }

    #[test]
    fn preserves_other_protocol_validation() {
        let mut body = request(128);
        body.0.temperature = Some(3.0);
        let errors = body.validate().unwrap_err();
        assert!(errors.field_errors().contains_key("temperature"));
    }

    #[test]
    fn optional_head_and_invalid_json_types() {
        let value = json!({"model":"test","messages":[{"role":"user","content":"hello"}]});
        let body: SglangChatRequest = serde_json::from_value(value.clone()).unwrap();
        assert!(body.validate().is_ok());
        for bad in [json!(-1), json!(1.5), json!("128")] {
            let mut value = value.clone();
            value["top_logprobs"] = bad;
            assert!(serde_json::from_value::<SglangChatRequest>(value).is_err());
        }
    }
}
