//! Internal and transport request representations.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{SamplingParams, TokenIds};

#[derive(Debug, Clone, Default)]
pub struct RendererRequest {
    pub rid: String,
    pub text: Option<String>,
    pub input_ids: Option<TokenIds>,
    pub skip_special_tokens: bool,
    pub sampling_params: SamplingParams,
    pub stream: bool,
    pub return_logprob: bool,
    pub logprob_start_len: i64,
    pub top_logprobs_num: i64,
    pub token_ids_logprob: Option<TokenIds>,
    pub return_hidden_states: bool,
    pub return_text_in_logprobs: Option<bool>,
}

impl RendererRequest {
    pub fn already_tokenized(&self) -> bool {
        self.input_ids.as_ref().is_some_and(|ids| !ids.is_empty())
    }
}

/// Stable token-in request accepted by SGLang's `/generate` endpoint.
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

impl From<RendererRequest> for PreparedGenerateRequest {
    fn from(mut request: RendererRequest) -> Self {
        Self {
            rid: request.rid,
            input_ids: request
                .input_ids
                .take()
                .expect("renderer preparation always produces input_ids"),
            sampling_params: request.sampling_params.into(),
            stream: request.stream,
            return_logprob: request.return_logprob,
            logprob_start_len: request.logprob_start_len,
            top_logprobs_num: request.top_logprobs_num,
            token_ids_logprob: request.token_ids_logprob,
            return_hidden_states: request.return_hidden_states,
            return_text_in_logprobs: request.return_text_in_logprobs,
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
