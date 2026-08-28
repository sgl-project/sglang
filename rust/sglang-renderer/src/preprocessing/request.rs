//! Internal and transport request representations.

use std::collections::BTreeMap;

use dynamo_renderer::RenderedPrompt;
use serde::{Deserialize, Serialize};

use crate::{SamplingParams, TokenIds};

/// Request-scoped metadata that must survive protocol lowering and prompt
/// tokenization before the request is submitted to SGLang `/generate`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GenerateRequestMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_salt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bootstrap_host: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bootstrap_port: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bootstrap_room: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub routed_dp_rank: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disagg_prefill_dp_rank: Option<i64>,
}

#[derive(Debug, Clone, Default)]
/// Generation options shared by text and token-ID inputs.
pub struct GenerationOptions {
    pub sampling_params: SamplingParams,
    pub stream: bool,
    pub return_logprob: bool,
    pub logprob_start_len: i64,
    pub top_logprobs_num: i64,
    pub token_ids_logprob: Option<TokenIds>,
    pub return_hidden_states: bool,
    pub return_text_in_logprobs: Option<bool>,
}

#[derive(Debug, Clone)]
/// Internal text-only generation request before tokenization.
///
/// Protocol adapters lower textual completions into this type. Structured chat
/// reaches it only after [`crate::ChatPreprocessor`] renders the messages.
pub struct TextRequest {
    pub rid: String,
    pub prompt: RenderedPrompt,
    pub add_special_tokens: bool,
    pub options: GenerationOptions,
    pub metadata: GenerateRequestMetadata,
}

impl TextRequest {
    pub fn text(
        rid: impl Into<String>,
        text: impl Into<String>,
        add_special_tokens: bool,
        options: GenerationOptions,
    ) -> Self {
        Self {
            rid: rid.into(),
            prompt: RenderedPrompt::text(text.into()),
            add_special_tokens,
            options,
            metadata: GenerateRequestMetadata::default(),
        }
    }

    pub fn rendered(
        rid: impl Into<String>,
        prompt: RenderedPrompt,
        add_special_tokens: bool,
        options: GenerationOptions,
    ) -> Self {
        Self {
            rid: rid.into(),
            prompt,
            add_special_tokens,
            options,
            metadata: GenerateRequestMetadata::default(),
        }
    }

    pub fn with_metadata(mut self, metadata: GenerateRequestMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}

#[derive(Debug, Clone)]
/// A generation request whose prompt is already represented by token IDs.
pub struct TokenIdsRequest {
    pub rid: String,
    pub input_ids: TokenIds,
    pub options: GenerationOptions,
    pub metadata: GenerateRequestMetadata,
}

impl TokenIdsRequest {
    pub fn new(rid: impl Into<String>, input_ids: TokenIds, options: GenerationOptions) -> Self {
        Self {
            rid: rid.into(),
            input_ids,
            options,
            metadata: GenerateRequestMetadata::default(),
        }
    }

    pub fn with_metadata(mut self, metadata: GenerateRequestMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Token-only request sent to the model server's `/generate` endpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub rid: String,
    #[serde(flatten)]
    pub metadata: GenerateRequestMetadata,
    pub input_ids: TokenIds,
    pub sampling_params: GenerateSamplingParams,
    pub stream: bool,
    pub return_logprob: bool,
    pub logprob_start_len: i64,
    pub top_logprobs_num: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_ids_logprob: Option<TokenIds>,
    pub return_hidden_states: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_text_in_logprobs: Option<bool>,
}

impl From<TokenIdsRequest> for GenerateRequest {
    fn from(request: TokenIdsRequest) -> Self {
        let options = request.options;
        Self {
            rid: request.rid,
            metadata: request.metadata,
            input_ids: request.input_ids,
            sampling_params: options.sampling_params.into(),
            stream: options.stream,
            return_logprob: options.return_logprob,
            logprob_start_len: options.logprob_start_len,
            top_logprobs_num: options.top_logprobs_num,
            token_ids_logprob: options.token_ids_logprob,
            return_hidden_states: options.return_hidden_states,
            return_text_in_logprobs: options.return_text_in_logprobs,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerateSamplingParams {
    pub max_new_tokens: Option<i64>,
    pub stop: Vec<String>,
    pub stop_token_ids: Option<Vec<i64>>,
    pub stop_regex: Vec<String>,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: i64,
    pub min_p: f64,
    pub frequency_penalty: f64,
    pub presence_penalty: f64,
    pub repetition_penalty: f64,
    pub min_new_tokens: i64,
    pub n: i64,
    pub json_schema: Option<String>,
    pub regex: Option<String>,
    pub ebnf: Option<String>,
    pub structural_tag: Option<String>,
    pub ignore_eos: bool,
    pub skip_special_tokens: bool,
    pub spaces_between_special_tokens: bool,
    pub no_stop_trim: bool,
    pub stream_interval: Option<i64>,
    pub logit_bias: Option<BTreeMap<String, f64>>,
    pub sampling_seed: Option<i64>,
    pub custom_params: Option<serde_json::Value>,
}

impl From<SamplingParams> for GenerateSamplingParams {
    fn from(params: SamplingParams) -> Self {
        Self {
            max_new_tokens: params.max_new_tokens,
            stop: params.stop_strs,
            stop_token_ids: params.stop_token_ids,
            stop_regex: params.stop_regex_strs,
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k,
            min_p: params.min_p,
            frequency_penalty: params.frequency_penalty,
            presence_penalty: params.presence_penalty,
            repetition_penalty: params.repetition_penalty,
            min_new_tokens: params.min_new_tokens,
            n: params.n,
            json_schema: params.json_schema,
            regex: params.regex,
            ebnf: params.ebnf,
            structural_tag: params.structural_tag,
            ignore_eos: params.ignore_eos,
            skip_special_tokens: params.skip_special_tokens,
            spaces_between_special_tokens: params.spaces_between_special_tokens,
            no_stop_trim: params.no_stop_trim,
            stream_interval: params.stream_interval,
            logit_bias: params.logit_bias,
            sampling_seed: params.sampling_seed,
            custom_params: params.custom_params,
        }
    }
}
