//! Internal and transport request representations.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{SamplingParams, TokenIds};

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
/// One text-generation prompt before tokenizer-dependent preparation.
pub enum TextPrompt {
    Text(String),
    TokenIds(TokenIds),
}

#[derive(Debug, Clone)]
/// Internal text-generation request shared by every protocol frontend.
///
/// Chat preprocessing renders structured messages into `Text`; Completions
/// and gRPC may also supply already-tokenized prompts through `TokenIds`.
pub struct TextRequest {
    pub rid: String,
    pub prompt: TextPrompt,
    pub skip_special_tokens: bool,
    pub options: GenerationOptions,
}

impl TextRequest {
    pub fn text(
        rid: impl Into<String>,
        text: impl Into<String>,
        skip_special_tokens: bool,
        options: GenerationOptions,
    ) -> Self {
        Self {
            rid: rid.into(),
            prompt: TextPrompt::Text(text.into()),
            skip_special_tokens,
            options,
        }
    }

    pub fn token_ids(
        rid: impl Into<String>,
        input_ids: TokenIds,
        skip_special_tokens: bool,
        options: GenerationOptions,
    ) -> Self {
        Self {
            rid: rid.into(),
            prompt: TextPrompt::TokenIds(input_ids),
            skip_special_tokens,
            options,
        }
    }
}

#[derive(Debug, Clone)]
/// A generation request whose prompt is already represented by token IDs.
pub struct TokenIdsRequest {
    pub rid: String,
    pub input_ids: TokenIds,
    pub options: GenerationOptions,
}

/// Prototype text token-in request accepted by SGLang's `/generate` endpoint.
///
/// This is a transport DTO rather than an in-process engine type. It is not yet
/// a field-complete compatibility contract for multimodal or disaggregated
/// serving.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreparedGenerateRequest {
    pub rid: String,
    pub input_ids: TokenIds,
    pub sampling_params: PreparedSamplingParams,
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

impl From<TokenIdsRequest> for PreparedGenerateRequest {
    fn from(request: TokenIdsRequest) -> Self {
        let options = request.options;
        Self {
            rid: request.rid,
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
pub struct PreparedSamplingParams {
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

impl From<SamplingParams> for PreparedSamplingParams {
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
