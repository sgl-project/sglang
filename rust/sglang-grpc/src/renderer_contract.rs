//! Lossless mapping between the transport-neutral renderer DTO and protobuf.

use std::collections::BTreeMap;

use sglang_renderer::{PreparedGenerateRequest, PreparedSamplingParams};

use crate::renderer_proto as proto;

impl From<PreparedGenerateRequest> for proto::PreparedGenerateRequest {
    fn from(request: PreparedGenerateRequest) -> Self {
        Self {
            rid: request.rid,
            input_ids: request.input_ids,
            sampling_params: Some(request.sampling_params.into()),
            stream: request.stream,
            return_logprob: request.return_logprob,
            logprob_start_len: request.logprob_start_len,
            top_logprobs_num: request.top_logprobs_num,
            token_ids_logprob: request
                .token_ids_logprob
                .map(|values| proto::OptionalInt32List { values }),
            return_hidden_states: request.return_hidden_states,
            return_text_in_logprobs: request.return_text_in_logprobs,
        }
    }
}

impl TryFrom<proto::PreparedGenerateRequest> for PreparedGenerateRequest {
    type Error = String;

    fn try_from(request: proto::PreparedGenerateRequest) -> Result<Self, Self::Error> {
        Ok(Self {
            rid: request.rid,
            input_ids: request.input_ids,
            sampling_params: request
                .sampling_params
                .ok_or_else(|| "prepared request is missing sampling_params".to_owned())?
                .try_into()?,
            stream: request.stream,
            return_logprob: request.return_logprob,
            logprob_start_len: request.logprob_start_len,
            top_logprobs_num: request.top_logprobs_num,
            token_ids_logprob: request.token_ids_logprob.map(|values| values.values),
            return_hidden_states: request.return_hidden_states,
            return_text_in_logprobs: request.return_text_in_logprobs,
        })
    }
}

impl From<PreparedSamplingParams> for proto::PreparedSamplingParams {
    fn from(params: PreparedSamplingParams) -> Self {
        Self {
            max_new_tokens: params.max_new_tokens,
            stop: params.stop,
            stop_token_ids: params
                .stop_token_ids
                .map(|values| proto::OptionalInt64List { values }),
            stop_regex: params.stop_regex,
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
            logit_bias: params.logit_bias.map(|values| proto::OptionalLogitBias {
                values: values.into_iter().collect(),
            }),
            sampling_seed: params.sampling_seed,
            custom_params_json: params
                .custom_params
                .map(|value| serde_json::to_vec(&value).expect("JSON value serializes")),
        }
    }
}

impl TryFrom<proto::PreparedSamplingParams> for PreparedSamplingParams {
    type Error = String;

    fn try_from(params: proto::PreparedSamplingParams) -> Result<Self, Self::Error> {
        Ok(Self {
            max_new_tokens: params.max_new_tokens,
            stop: params.stop,
            stop_token_ids: params.stop_token_ids.map(|values| values.values),
            stop_regex: params.stop_regex,
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
            logit_bias: params
                .logit_bias
                .map(|values| values.values.into_iter().collect::<BTreeMap<String, f64>>()),
            sampling_seed: params.sampling_seed,
            custom_params: params
                .custom_params_json
                .map(|json| serde_json::from_slice(&json).map_err(|error| error.to_string()))
                .transpose()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prepared(
        optional_ids: Option<Vec<i32>>,
        optional_stops: Option<Vec<i64>>,
    ) -> PreparedGenerateRequest {
        PreparedGenerateRequest {
            rid: "request-1".into(),
            input_ids: vec![1, 2, 3],
            sampling_params: PreparedSamplingParams {
                max_new_tokens: Some(16),
                stop: vec!["END".into()],
                stop_token_ids: optional_stops,
                stop_regex: vec!["done$".into()],
                temperature: 0.25,
                top_p: 0.9,
                top_k: 12,
                min_p: 0.1,
                frequency_penalty: 0.2,
                presence_penalty: 0.3,
                repetition_penalty: 1.1,
                min_new_tokens: 2,
                n: 1,
                json_schema: Some(r#"{"type":"object"}"#.into()),
                regex: None,
                ebnf: None,
                structural_tag: Some("tag".into()),
                ignore_eos: true,
                skip_special_tokens: false,
                spaces_between_special_tokens: true,
                no_stop_trim: true,
                stream_interval: Some(4),
                logit_bias: Some(BTreeMap::from([("7".into(), 1.5)])),
                sampling_seed: Some(42),
                custom_params: Some(serde_json::json!({"processor": [1, true]})),
            },
            stream: true,
            return_logprob: true,
            logprob_start_len: 0,
            top_logprobs_num: 5,
            token_ids_logprob: optional_ids,
            return_hidden_states: true,
            return_text_in_logprobs: Some(false),
        }
    }

    #[test]
    fn protobuf_round_trip_preserves_prepared_request() {
        let expected = prepared(Some(vec![8, 9]), Some(vec![10, 11]));
        let wire = proto::PreparedGenerateRequest::from(expected.clone());
        assert_eq!(PreparedGenerateRequest::try_from(wire).unwrap(), expected);
    }

    #[test]
    fn protobuf_preserves_absent_and_present_empty_lists() {
        for (ids, stops) in [(None, None), (Some(Vec::new()), Some(Vec::new()))] {
            let expected = prepared(ids, stops);
            let wire = proto::PreparedGenerateRequest::from(expected.clone());
            assert_eq!(PreparedGenerateRequest::try_from(wire).unwrap(), expected);
        }
    }
}
