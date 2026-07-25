//! Shared response formatting logic
//!
//! This module contains common logic for formatting responses, including:
//! - Usage calculation from gRPC responses
//! - ChatCompletionResponse construction

use serde::{ser::Error as _, Serialize, Serializer};
use serde_json::json;

use crate::{
    protocols::{chat::ChatCompletionResponse, common::Usage},
    routers::grpc::proto_wrapper::ProtoGenerateComplete,
};

#[derive(Debug)]
pub(crate) struct ChatCompletionResponseWithUsage {
    response: ChatCompletionResponse,
    cached_tokens: u32,
}

impl ChatCompletionResponseWithUsage {
    pub fn new(response: ChatCompletionResponse, responses: &[ProtoGenerateComplete]) -> Self {
        let cached_tokens = responses
            .iter()
            .map(|response| response.cached_tokens() as u32)
            .sum();

        Self {
            response,
            cached_tokens,
        }
    }

    pub fn into_inner(self) -> ChatCompletionResponse {
        self.response
    }
}

impl Serialize for ChatCompletionResponseWithUsage {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut value = serde_json::to_value(&self.response).map_err(S::Error::custom)?;

        if self.cached_tokens > 0 {
            if let Some(usage) = value
                .get_mut("usage")
                .and_then(|usage| usage.as_object_mut())
            {
                usage.insert(
                    "prompt_tokens_details".to_string(),
                    json!({ "cached_tokens": self.cached_tokens }),
                );
            }
        }

        value.serialize(serializer)
    }
}

/// Build usage information from collected gRPC responses
///
/// Sums prompt_tokens and completion_tokens across all responses.
/// Typically used with n>1 parameter where multiple completions are generated.
///
/// # Arguments
/// * `responses` - Vector of GenerateComplete responses from the backend
///
/// # Returns
/// Usage object with aggregated token counts
pub(crate) fn build_usage(responses: &[ProtoGenerateComplete]) -> Usage {
    let total_prompt_tokens: u32 = responses.iter().map(|r| r.prompt_tokens() as u32).sum();
    let total_completion_tokens: u32 = responses.iter().map(|r| r.completion_tokens() as u32).sum();

    Usage {
        prompt_tokens: total_prompt_tokens,
        completion_tokens: total_completion_tokens,
        total_tokens: total_prompt_tokens + total_completion_tokens,
        completion_tokens_details: None,
    }
}

#[cfg(test)]
mod tests {
    use smg_grpc_client::sglang_proto;

    use super::*;

    fn complete(cached_tokens: i32) -> ProtoGenerateComplete {
        ProtoGenerateComplete::Sglang(sglang_proto::GenerateComplete {
            prompt_tokens: 100,
            completion_tokens: 20,
            cached_tokens,
            ..Default::default()
        })
    }

    fn response(responses: &[ProtoGenerateComplete]) -> ChatCompletionResponseWithUsage {
        let response = ChatCompletionResponse::builder("request-id", "model")
            .usage(build_usage(responses))
            .build();
        ChatCompletionResponseWithUsage::new(response, responses)
    }

    #[test]
    fn serializes_non_zero_cached_tokens() {
        let value = serde_json::to_value(response(&[complete(42)])).unwrap();

        assert_eq!(value["usage"]["prompt_tokens"], 100);
        assert_eq!(value["usage"]["completion_tokens"], 20);
        assert_eq!(value["usage"]["total_tokens"], 120);
        assert_eq!(value["usage"]["prompt_tokens_details"]["cached_tokens"], 42);
    }

    #[test]
    fn omits_zero_cached_tokens() {
        let value = serde_json::to_value(response(&[complete(0)])).unwrap();

        assert!(value["usage"].get("prompt_tokens_details").is_none());
    }

    #[test]
    fn sums_cached_tokens_across_responses() {
        let value = serde_json::to_value(response(&[complete(10), complete(20)])).unwrap();

        assert_eq!(value["usage"]["prompt_tokens"], 200);
        assert_eq!(value["usage"]["completion_tokens"], 40);
        assert_eq!(value["usage"]["total_tokens"], 240);
        assert_eq!(value["usage"]["prompt_tokens_details"]["cached_tokens"], 30);
    }
}
