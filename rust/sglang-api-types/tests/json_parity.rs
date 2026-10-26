//! Cross-language parity: the fixtures in `testdata/json_parity.json` are
//! checked here against the generated Rust types and, case for case, against
//! the generated Python types by `test/registered/unit/api/v1/test_parity.py`.
//! Both emitters must decode each `input` to the same `canonical` bytes (field
//! set, key order, presence and null rules) and keep `canonical` fixed under
//! a second round trip.

use serde::{Serialize, de::DeserializeOwned};
use sglang_api_types::api::v1 as api;

const FIXTURES: &str = include_str!("../testdata/json_parity.json");

fn check<T: DeserializeOwned + Serialize>(name: &str, input: &serde_json::Value, canonical: &str) {
    let decoded: T = serde_json::from_value(input.clone())
        .unwrap_or_else(|error| panic!("{name}: input does not decode: {error}"));
    assert_eq!(
        serde_json::to_string(&decoded).unwrap(),
        canonical,
        "{name}: input -> canonical"
    );
    let again: T = serde_json::from_str(canonical)
        .unwrap_or_else(|error| panic!("{name}: canonical does not decode: {error}"));
    assert_eq!(
        serde_json::to_string(&again).unwrap(),
        canonical,
        "{name}: canonical is not a fixed point"
    );
}

#[test]
fn fixtures_decode_to_their_canonical_bytes() {
    let fixtures: serde_json::Value = serde_json::from_str(FIXTURES).unwrap();
    let cases = fixtures["cases"].as_array().expect("cases array");
    assert!(!cases.is_empty());
    for case in cases {
        let ty = case["type"].as_str().expect("type");
        let name = format!("{ty}: {}", case["name"].as_str().expect("name"));
        let input = &case["input"];
        let canonical = case["canonical"].as_str().expect("canonical");
        match ty {
            "GenerateRequest" => check::<api::GenerateRequest>(&name, input, canonical),
            "SamplingParams" => check::<api::SamplingParams>(&name, input, canonical),
            "GenerateResponse" => check::<api::GenerateResponse>(&name, input, canonical),
            "GenerateStreamError" => check::<api::GenerateStreamError>(&name, input, canonical),
            "GetModelInfoResponse" => check::<api::GetModelInfoResponse>(&name, input, canonical),
            "GetServerInfoResponse" => check::<api::GetServerInfoResponse>(&name, input, canonical),
            "AttachHiCacheStorageRequest" => {
                check::<api::AttachHiCacheStorageRequest>(&name, input, canonical)
            }
            "HiCacheStatus" => check::<api::HiCacheStatus>(&name, input, canonical),
            "HiCacheStorageResponse" => {
                check::<api::HiCacheStorageResponse>(&name, input, canonical)
            }
            "ClearHiCacheStorageRequest" => {
                check::<api::ClearHiCacheStorageRequest>(&name, input, canonical)
            }
            "DetachHiCacheStorageRequest" => {
                check::<api::DetachHiCacheStorageRequest>(&name, input, canonical)
            }
            "GetHiCacheStorageRequest" => {
                check::<api::GetHiCacheStorageRequest>(&name, input, canonical)
            }
            "ListModelsResponse" => check::<api::ListModelsResponse>(&name, input, canonical),
            "DetokenizeRequest" => check::<api::DetokenizeRequest>(&name, input, canonical),
            "DetokenizeResponse" => check::<api::DetokenizeResponse>(&name, input, canonical),
            "AbortRequest" => check::<api::AbortRequest>(&name, input, canonical),
            "AbortResponse" => check::<api::AbortResponse>(&name, input, canonical),
            "SetInternalStateRequest" => {
                check::<api::SetInternalStateRequest>(&name, input, canonical)
            }
            "StartProfileRequest" => check::<api::StartProfileRequest>(&name, input, canonical),
            "SlowDownRequest" => check::<api::SlowDownRequest>(&name, input, canonical),
            "ReleaseMemoryOccupationRequest" => {
                check::<api::ReleaseMemoryOccupationRequest>(&name, input, canonical)
            }
            "GetWeightsByNameRequest" => {
                check::<api::GetWeightsByNameRequest>(&name, input, canonical)
            }
            "InitWeightsUpdateGroupRequest" => {
                check::<api::InitWeightsUpdateGroupRequest>(&name, input, canonical)
            }
            "SendWeightsToRemoteInstanceRequest" => {
                check::<api::SendWeightsToRemoteInstanceRequest>(&name, input, canonical)
            }
            "AnthropicMessagesRequest" => {
                check::<api::AnthropicMessagesRequest>(&name, input, canonical)
            }
            "AnthropicMessagesResponse" => {
                check::<api::AnthropicMessagesResponse>(&name, input, canonical)
            }
            "AnthropicStreamEvent" => check::<api::AnthropicStreamEvent>(&name, input, canonical),
            "AnthropicErrorResponse" => {
                check::<api::AnthropicErrorResponse>(&name, input, canonical)
            }
            "AnthropicCountTokensRequest" => {
                check::<api::AnthropicCountTokensRequest>(&name, input, canonical)
            }
            "AnthropicCountTokensResponse" => {
                check::<api::AnthropicCountTokensResponse>(&name, input, canonical)
            }
            "OpenAiCompletionRequest" => {
                check::<api::OpenAiCompletionRequest>(&name, input, canonical)
            }
            "OpenAiCompletionResponse" => {
                check::<api::OpenAiCompletionResponse>(&name, input, canonical)
            }
            "OpenAiChatCompletionRequest" => {
                check::<api::OpenAiChatCompletionRequest>(&name, input, canonical)
            }
            "OpenAiChatCompletionResponse" => {
                check::<api::OpenAiChatCompletionResponse>(&name, input, canonical)
            }
            "OpenAiChatCompletionStreamResponse" => {
                check::<api::OpenAiChatCompletionStreamResponse>(&name, input, canonical)
            }
            "OpenAiEmbeddingRequest" => {
                check::<api::OpenAiEmbeddingRequest>(&name, input, canonical)
            }
            "OpenAiEmbeddingResponse" => {
                check::<api::OpenAiEmbeddingResponse>(&name, input, canonical)
            }
            "OpenAiScoreRequest" => check::<api::OpenAiScoreRequest>(&name, input, canonical),
            "OpenAiScoreResponse" => check::<api::OpenAiScoreResponse>(&name, input, canonical),
            "OpenAiRerankRequest" => check::<api::OpenAiRerankRequest>(&name, input, canonical),
            "OpenAiRerankResponse" => check::<api::OpenAiRerankResponse>(&name, input, canonical),
            "OpenAiTokenizeRequest" => check::<api::OpenAiTokenizeRequest>(&name, input, canonical),
            "OpenAiTokenizeResponse" => {
                check::<api::OpenAiTokenizeResponse>(&name, input, canonical)
            }
            "OpenAiResponsesRequest" => {
                check::<api::OpenAiResponsesRequest>(&name, input, canonical)
            }
            "OpenAiModelList" => check::<api::OpenAiModelList>(&name, input, canonical),
            "OpenAiErrorResponse" => check::<api::OpenAiErrorResponse>(&name, input, canonical),
            "OllamaChatRequest" => check::<api::OllamaChatRequest>(&name, input, canonical),
            "OllamaChatResponse" => check::<api::OllamaChatResponse>(&name, input, canonical),
            "OllamaGenerateRequest" => check::<api::OllamaGenerateRequest>(&name, input, canonical),
            "OllamaGenerateResponse" => {
                check::<api::OllamaGenerateResponse>(&name, input, canonical)
            }
            "OllamaTagsResponse" => check::<api::OllamaTagsResponse>(&name, input, canonical),
            "OllamaShowRequest" => check::<api::OllamaShowRequest>(&name, input, canonical),
            "OllamaShowResponse" => check::<api::OllamaShowResponse>(&name, input, canonical),
            "ParseFunctionCallRequest" => {
                check::<api::ParseFunctionCallRequest>(&name, input, canonical)
            }
            "ParseFunctionCallResponse" => {
                check::<api::ParseFunctionCallResponse>(&name, input, canonical)
            }
            "AddExternalCorpusRequest" => {
                check::<api::AddExternalCorpusRequest>(&name, input, canonical)
            }
            "ListExternalCorporaResponse" => {
                check::<api::ListExternalCorporaResponse>(&name, input, canonical)
            }
            "PauseGenerationRequest" => {
                check::<api::PauseGenerationRequest>(&name, input, canonical)
            }
            "PdRoleSwitchRequest" => check::<api::PdRoleSwitchRequest>(&name, input, canonical),
            "UpdateWeightVersionRequest" => {
                check::<api::UpdateWeightVersionRequest>(&name, input, canonical)
            }
            other => panic!("fixture type {other} has no Rust mapping; add it here"),
        }
    }
}
