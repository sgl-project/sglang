//! Conversions from the schema-generated API types (sglang-api-types) into
//! the internal pipeline types. The generated types own the HTTP wire; the
//! internal types own normalization and the scheduler IPC encoding — this
//! file is the seam between the two, migrated endpoint by endpoint.

use sglang_api_types::api::v1 as genapi;

use crate::message::request::GenerateBody;
use crate::message::sampling::{SamplingParams, SamplingParamsInput};
use crate::message::types::{OneOrMany, TokenIds};

fn one_or_many_string(v: Option<genapi::StringOrList>) -> Option<OneOrMany<String>> {
    use genapi::string_or_list::Value;
    match v?.value? {
        Value::One(s) => Some(OneOrMany::One(s)),
        Value::Many(l) => Some(OneOrMany::Many(l.items)),
    }
}

fn one_or_many_bool(v: Option<genapi::BoolOrList>) -> Option<OneOrMany<bool>> {
    use genapi::bool_or_list::Value;
    match v?.value? {
        Value::One(b) => Some(OneOrMany::One(b)),
        Value::Many(l) => Some(OneOrMany::Many(l.items)),
    }
}

fn one_or_many_i64(v: Option<genapi::Int64OrList>) -> Option<OneOrMany<i64>> {
    use genapi::int64_or_list::Value;
    match v?.value? {
        Value::One(n) => Some(OneOrMany::One(n)),
        Value::Many(l) => Some(OneOrMany::Many(l.items)),
    }
}

fn one_or_many_token_ids(v: Option<genapi::TokenIdsOrList>) -> Option<OneOrMany<TokenIds>> {
    use genapi::token_ids_or_list::Value;
    match v?.value? {
        Value::One(t) => Some(OneOrMany::One(t.ids)),
        Value::Many(l) => Some(OneOrMany::Many(
            l.items.into_iter().map(|t| t.ids).collect(),
        )),
    }
}

fn one_or_many_opt_string(
    v: Option<genapi::OptionalStringOrList>,
) -> Option<OneOrMany<Option<String>>> {
    use genapi::optional_string_or_list::Value;
    match v?.value? {
        Value::One(s) => Some(OneOrMany::One(s.value)),
        Value::Many(l) => Some(OneOrMany::Many(
            l.items.into_iter().map(|s| s.value).collect(),
        )),
    }
}

fn one_or_many_opt_i64(v: Option<genapi::OptionalInt64OrList>) -> Option<OneOrMany<Option<i64>>> {
    use genapi::optional_int64_or_list::Value;
    match v?.value? {
        Value::One(n) => Some(OneOrMany::One(n.value)),
        Value::Many(l) => Some(OneOrMany::Many(
            l.items.into_iter().map(|n| n.value).collect(),
        )),
    }
}

/// A raw_json field's stored JSON text, reparsed as the pipeline's value
/// type. The text was produced by our own deserializer, so it is valid JSON
/// by construction.
fn raw_json_rmpv(v: Option<String>) -> Option<rmpv::Value> {
    v.map(|s| serde_json::from_str(&s).expect("raw_json field holds valid JSON"))
}

pub(crate) fn sampling_params(p: genapi::SamplingParams) -> SamplingParams {
    // The *_or_default accessors carry the schema defaults for protobuf
    // arrival (unset optional scalars); JSON arrival baked them already.
    // Read before the struct literal moves `p`'s owned fields.
    let temperature = p.temperature_or_default();
    let top_p = p.top_p_or_default();
    let top_k = p.top_k_or_default();
    let min_p = p.min_p_or_default();
    let frequency_penalty = p.frequency_penalty_or_default();
    let presence_penalty = p.presence_penalty_or_default();
    let repetition_penalty = p.repetition_penalty_or_default();
    let min_new_tokens = p.min_new_tokens_or_default();
    let n = p.n_or_default();
    let ignore_eos = p.ignore_eos_or_default();
    let skip_special_tokens = p.skip_special_tokens_or_default();
    let spaces_between_special_tokens = p.spaces_between_special_tokens_or_default();
    let no_stop_trim = p.no_stop_trim_or_default();
    SamplingParams {
        max_new_tokens: p.max_new_tokens,
        stop: one_or_many_string(p.stop),
        stop_token_ids: p.stop_token_ids.map(|l| l.items),
        stop_regex: one_or_many_string(p.stop_regex),
        temperature,
        top_p,
        top_k,
        min_p,
        frequency_penalty,
        presence_penalty,
        repetition_penalty,
        min_new_tokens,
        n,
        json_schema: p.json_schema,
        regex: p.regex,
        ebnf: p.ebnf,
        structural_tag: p.structural_tag,
        ignore_eos,
        skip_special_tokens,
        spaces_between_special_tokens,
        no_stop_trim,
        stream_interval: p.stream_interval,
        // proto maps carry no presence: an empty map reads as "no bias", the
        // same way the scheduler treats an absent one.
        logit_bias: (!p.logit_bias.is_empty()).then(|| p.logit_bias.into_iter().collect()),
        sampling_seed: p.sampling_seed,
        custom_params: p
            .custom_params
            .map(|s| serde_json::from_str(&s).expect("raw_json field holds valid JSON")),
        ..Default::default()
    }
}

fn sampling_params_input(v: Option<genapi::SamplingParamsOrList>) -> Option<SamplingParamsInput> {
    use genapi::sampling_params_or_list::Value;
    match v?.value? {
        Value::One(p) => Some(SamplingParamsInput::One(Box::new(sampling_params(p)))),
        Value::Many(l) => Some(SamplingParamsInput::Many(
            l.items.into_iter().map(sampling_params).collect(),
        )),
    }
}

/// The generated wire request, converted into the internal fan-out input.
/// `into_requests` (normalization, batching, rid minting, DoS budgets) is
/// unchanged: it consumes the converted body exactly as it consumed the
/// hand-parsed one.
pub(crate) fn generate_body(req: genapi::GenerateRequest) -> GenerateBody {
    let stream = req.stream_or_default();
    GenerateBody {
        rid: one_or_many_string(req.rid),
        text: one_or_many_string(req.text),
        input_ids: one_or_many_token_ids(req.input_ids),
        stream,
        sampling_params: sampling_params_input(req.sampling_params),
        return_logprob: one_or_many_bool(req.return_logprob),
        logprob_start_len: one_or_many_i64(req.logprob_start_len),
        top_logprobs_num: one_or_many_i64(req.top_logprobs_num),
        token_ids_logprob: one_or_many_token_ids(req.token_ids_logprob),
        return_hidden_states: one_or_many_bool(req.return_hidden_states),
        return_text_in_logprobs: req.return_text_in_logprobs,
        bootstrap_host: one_or_many_opt_string(req.bootstrap_host),
        bootstrap_port: one_or_many_opt_i64(req.bootstrap_port),
        bootstrap_room: one_or_many_opt_i64(req.bootstrap_room),
        bootstrap_pair_key: one_or_many_opt_string(req.bootstrap_pair_key),
        decode_tp_size: one_or_many_opt_i64(req.decode_tp_size),
        routed_dp_rank: req.routed_dp_rank,
        disagg_prefill_dp_rank: req.disagg_prefill_dp_rank,
        image_data: raw_json_rmpv(req.image_data),
        mm_hashes: raw_json_rmpv(req.mm_hashes),
        video_data: raw_json_rmpv(req.video_data),
        audio_data: raw_json_rmpv(req.audio_data),
    }
}
