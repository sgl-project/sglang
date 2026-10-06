//! `runtime.v1` request conversion for the Rust frontend.
//!
//! The canonical protobuf is wider than the Rust frontend today. Fields with
//! no internal representation are rejected explicitly instead of being
//! silently ignored or expanding frontend behavior in this wiring change.

use sglang_grpc_types::sglang::runtime::v1 as proto;
use tonic::Status;

use crate::frontend::FrontendRequest;
use crate::message::config::PreferredSamplingParams;
use crate::message::ids::Rid;
use crate::message::sampling::SamplingParams;

type ConvertResult<T> = Result<T, ConvertError>;

#[derive(Debug, thiserror::Error)]
pub(super) enum ConvertError {
    #[error("{0}")]
    InvalidArgument(String),
    #[error("{0}")]
    Unsupported(String),
    #[error("{0}")]
    Internal(String),
}

impl ConvertError {
    fn invalid_argument(message: impl Into<String>) -> Self {
        Self::InvalidArgument(message.into())
    }

    fn internal(message: impl Into<String>) -> Self {
        Self::Internal(message.into())
    }
}

impl From<ConvertError> for Status {
    fn from(error: ConvertError) -> Self {
        match error {
            ConvertError::InvalidArgument(message) => Self::invalid_argument(message),
            ConvertError::Unsupported(message) => Self::unimplemented(message),
            ConvertError::Internal(message) => Self::internal(message),
        }
    }
}

pub(super) fn text_generate(
    request: proto::TextGenerateRequest,
    preferred: Option<&PreferredSamplingParams>,
) -> ConvertResult<FrontendRequest> {
    let proto::TextGenerateRequest {
        text,
        sampling_params,
        stream,
        return_logprob,
        top_logprobs_num,
        logprob_start_len,
        return_text_in_logprobs,
        rid,
        lora_path,
        routing_key,
        routed_dp_rank,
        trace_headers,
        session_id,
        disaggregated_params,
        priority,
        require_reasoning,
        max_thinking_tokens,
        kv_hints,
    } = request;

    reject_unsupported([
        ("kv_hints", kv_hints.is_some()),
        ("lora_path", lora_path.is_some()),
        ("routing_key", routing_key.is_some()),
        ("trace_headers", !trace_headers.is_empty()),
        ("session_id", session_id.is_some()),
        ("priority", priority.is_some()),
        ("require_reasoning", require_reasoning.is_some()),
        ("max_thinking_tokens", max_thinking_tokens.is_some()),
    ])?;

    let (bootstrap_host, bootstrap_port, bootstrap_room) =
        disaggregated_fields(disaggregated_params);
    Ok(FrontendRequest {
        rid: request_id(rid),
        text: Some(text),
        input_ids: None,
        skip_special_tokens: false,
        sampling_params: convert_sampling_params(sampling_params, preferred)?,
        stream: stream.unwrap_or(false),
        return_logprob: return_logprob.unwrap_or(false),
        logprob_start_len: logprob_start_len.map(i64::from).unwrap_or(-1),
        top_logprobs_num: top_logprobs_num.map(i64::from).unwrap_or(0),
        token_ids_logprob: None,
        return_sampling_mask: false,
        return_hidden_states: false,
        return_text_in_logprobs,
        bootstrap_host,
        bootstrap_port,
        bootstrap_room,
        bootstrap_pair_key: None,
        decode_tp_size: None,
        routed_dp_rank: routed_dp_rank.map(i64::from),
        disagg_prefill_dp_rank: None,
        mm: None,
        mm_buffers: Vec::new(),
    })
}

pub(super) fn generate(
    request: proto::GenerateRequest,
    preferred: Option<&PreferredSamplingParams>,
) -> ConvertResult<FrontendRequest> {
    let proto::GenerateRequest {
        input_ids,
        sampling_params,
        stream,
        return_logprob,
        top_logprobs_num,
        logprob_start_len,
        rid,
        lora_path,
        routing_key,
        routed_dp_rank,
        trace_headers,
        session_id,
        disaggregated_params,
        priority,
        require_reasoning,
        max_thinking_tokens,
        kv_hints,
    } = request;

    reject_unsupported([
        ("kv_hints", kv_hints.is_some()),
        ("lora_path", lora_path.is_some()),
        ("routing_key", routing_key.is_some()),
        ("trace_headers", !trace_headers.is_empty()),
        ("session_id", session_id.is_some()),
        ("priority", priority.is_some()),
        ("require_reasoning", require_reasoning.is_some()),
        ("max_thinking_tokens", max_thinking_tokens.is_some()),
    ])?;
    if input_ids.is_empty() {
        return Err(ConvertError::invalid_argument("input_ids cannot be empty"));
    }

    let (bootstrap_host, bootstrap_port, bootstrap_room) =
        disaggregated_fields(disaggregated_params);
    Ok(FrontendRequest {
        rid: request_id(rid),
        text: None,
        input_ids: Some(input_ids.into_iter().map(i64::from).collect()),
        skip_special_tokens: false,
        sampling_params: convert_sampling_params(sampling_params, preferred)?,
        stream: stream.unwrap_or(false),
        return_logprob: return_logprob.unwrap_or(false),
        logprob_start_len: logprob_start_len.map(i64::from).unwrap_or(-1),
        top_logprobs_num: top_logprobs_num.map(i64::from).unwrap_or(0),
        token_ids_logprob: None,
        return_sampling_mask: false,
        return_hidden_states: false,
        return_text_in_logprobs: None,
        bootstrap_host,
        bootstrap_port,
        bootstrap_room,
        bootstrap_pair_key: None,
        decode_tp_size: None,
        routed_dp_rank: routed_dp_rank.map(i64::from),
        disagg_prefill_dp_rank: None,
        mm: None,
        mm_buffers: Vec::new(),
    })
}

fn request_id(rid: Option<String>) -> Rid {
    rid.map_or_else(Rid::new, |rid| Rid::from_client(&rid))
}

fn reject_unsupported<const N: usize>(fields: [(&'static str, bool); N]) -> ConvertResult<()> {
    let unsupported = fields
        .into_iter()
        .filter_map(|(name, present)| present.then_some(name))
        .collect::<Vec<_>>();
    if unsupported.is_empty() {
        return Ok(());
    }
    Err(ConvertError::Unsupported(format!(
        "the Rust frontend does not yet support runtime.v1 field(s): {}",
        unsupported.join(", ")
    )))
}

fn disaggregated_fields(
    params: Option<proto::DisaggregatedParams>,
) -> (Option<String>, Option<i64>, Option<i64>) {
    params.map_or((None, None, None), |params| {
        (
            Some(params.bootstrap_host),
            Some(i64::from(params.bootstrap_port)),
            Some(params.bootstrap_room),
        )
    })
}

/// Merge launch-time preferred values first, then overwrite only fields that
/// are explicitly present in protobuf. This is the same precedence used by
/// the Python-backed gRPC path; proto3 repeated fields cannot distinguish an
/// omitted list from an explicit empty list, so empty stop lists remain absent.
#[allow(deprecated)]
fn convert_sampling_params(
    params: Option<proto::SamplingParams>,
    preferred: Option<&PreferredSamplingParams>,
) -> ConvertResult<SamplingParams> {
    let mut output = SamplingParams::default();

    let Some(params) = params else {
        if let Some(preferred) = preferred {
            output.apply_preferred(&preferred.0).map_err(|error| {
                ConvertError::internal(format!("invalid preferred_sampling_params: {error}"))
            })?;
        }
        return Ok(output);
    };
    if params.guided_decoding.is_some() && (params.json_schema.is_some() || params.regex.is_some())
    {
        return Err(ConvertError::invalid_argument(
            "legacy json_schema/regex cannot be combined with guided_decoding",
        ));
    }

    if let Some(value) = params.temperature {
        output.temperature = f64::from(value);
        output.mark_explicit("temperature");
    }
    if let Some(value) = params.top_p {
        output.top_p = f64::from(value);
        output.mark_explicit("top_p");
    }
    if let Some(value) = params.top_k {
        output.top_k = i64::from(value);
        output.mark_explicit("top_k");
    }
    if let Some(value) = params.min_p {
        output.min_p = f64::from(value);
        output.mark_explicit("min_p");
    }
    if let Some(value) = params.frequency_penalty {
        output.frequency_penalty = f64::from(value);
        output.mark_explicit("frequency_penalty");
    }
    if let Some(value) = params.presence_penalty {
        output.presence_penalty = f64::from(value);
        output.mark_explicit("presence_penalty");
    }
    if let Some(value) = params.repetition_penalty {
        output.repetition_penalty = f64::from(value);
        output.mark_explicit("repetition_penalty");
    }
    if let Some(value) = params.max_new_tokens {
        output.max_new_tokens = Some(i64::from(value));
        output.mark_explicit("max_new_tokens");
    }
    if let Some(value) = params.min_new_tokens {
        output.min_new_tokens = i64::from(value);
        output.mark_explicit("min_new_tokens");
    }
    if !params.stop.is_empty() {
        output.stop = Some(crate::message::types::OneOrMany::Many(params.stop));
        output.mark_explicit("stop");
    }
    if !params.stop_token_ids.is_empty() {
        output.stop_token_ids = Some(params.stop_token_ids.into_iter().map(i64::from).collect());
        output.mark_explicit("stop_token_ids");
    }
    if let Some(value) = params.ignore_eos {
        output.ignore_eos = value;
        output.mark_explicit("ignore_eos");
    }
    if let Some(value) = params.n {
        output.n = i64::from(value);
        output.mark_explicit("n");
    }
    if let Some(value) = params.seed {
        output.sampling_seed = Some(value);
        output.mark_explicit("sampling_seed");
    }

    if let Some(guided) = params.guided_decoding {
        apply_guided_decoding(&mut output, guided)?;
    } else {
        if let Some(value) = params.json_schema {
            if value.is_empty() {
                return Err(ConvertError::invalid_argument(
                    "legacy json_schema must not be empty",
                ));
            }
            output.json_schema = Some(value);
            output.mark_explicit("json_schema");
        }
        if let Some(value) = params.regex {
            if value.is_empty() {
                return Err(ConvertError::invalid_argument(
                    "legacy regex must not be empty",
                ));
            }
            output.regex = Some(value);
            output.mark_explicit("regex");
        }
    }

    if let Some(preferred) = preferred {
        output.apply_preferred(&preferred.0).map_err(|error| {
            ConvertError::internal(format!("invalid preferred_sampling_params: {error}"))
        })?;
    }

    Ok(output)
}

fn apply_guided_decoding(
    output: &mut SamplingParams,
    guided: proto::GuidedDecoding,
) -> ConvertResult<()> {
    use proto::guided_decoding::Constraint;

    match guided.constraint {
        Some(Constraint::JsonSchema(value)) if !value.is_empty() => {
            output.json_schema = Some(value);
            output.mark_explicit("json_schema");
        }
        Some(Constraint::Regex(value)) if !value.is_empty() => {
            output.regex = Some(value);
            output.mark_explicit("regex");
        }
        Some(Constraint::Ebnf(value)) if !value.is_empty() => {
            output.ebnf = Some(value);
            output.mark_explicit("ebnf");
        }
        Some(Constraint::StructuralTag(value)) if !value.is_empty() => {
            output.structural_tag = Some(value);
            output.mark_explicit("structural_tag");
        }
        Some(Constraint::Choice(choice))
            if !choice.values.is_empty() && choice.values.iter().all(|value| !value.is_empty()) =>
        {
            let alternatives = choice
                .values
                .iter()
                .map(|value| regex_escape_literal(value))
                .collect::<Vec<_>>()
                .join("|");
            output.regex = Some(format!("(?:{alternatives})"));
            output.mark_explicit("regex");
        }
        Some(Constraint::Choice(_)) => {
            return Err(ConvertError::invalid_argument(
                "guided choice must contain only non-empty values",
            ));
        }
        Some(_) => {
            return Err(ConvertError::invalid_argument(
                "guided decoding constraint must not be empty",
            ));
        }
        None => {
            return Err(ConvertError::invalid_argument(
                "guided decoding constraint must be specified",
            ));
        }
    }
    Ok(())
}

fn regex_escape_literal(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        if matches!(
            character,
            '.' | '+' | '*' | '?' | '^' | '$' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '\\'
        ) {
            escaped.push('\\');
        }
        escaped.push(character);
    }
    escaped
}

#[cfg(test)]
mod tests {
    use super::*;
    use tonic::Code;

    #[test]
    fn text_request_maps_supported_fields_and_preferred_sampling() {
        let preferred = PreferredSamplingParams(serde_json::json!({
            "temperature": 0.2,
            "top_p": 0.7,
            "max_new_tokens": 64,
        }));
        let request = proto::TextGenerateRequest {
            text: "hello".into(),
            sampling_params: Some(proto::SamplingParams {
                temperature: Some(0.8),
                seed: Some(7),
                ..Default::default()
            }),
            stream: Some(true),
            return_logprob: Some(true),
            top_logprobs_num: Some(3),
            logprob_start_len: Some(2),
            return_text_in_logprobs: Some(true),
            rid: Some("client-id".into()),
            routed_dp_rank: Some(4),
            disaggregated_params: Some(proto::DisaggregatedParams {
                bootstrap_host: "10.0.0.1".into(),
                bootstrap_port: 8998,
                bootstrap_room: 12,
            }),
            ..Default::default()
        };

        let request = text_generate(request, Some(&preferred)).unwrap();
        assert_eq!(request.rid.client_facing(), "client-id");
        assert_eq!(request.text.as_deref(), Some("hello"));
        assert!(request.input_ids.is_none());
        assert!(request.stream);
        assert!(request.return_logprob);
        assert_eq!(request.top_logprobs_num, 3);
        assert_eq!(request.logprob_start_len, 2);
        assert_eq!(request.return_text_in_logprobs, Some(true));
        assert_eq!(request.routed_dp_rank, Some(4));
        assert_eq!(request.bootstrap_host.as_deref(), Some("10.0.0.1"));
        assert_eq!(request.bootstrap_port, Some(8998));
        assert_eq!(request.bootstrap_room, Some(12));
        assert_eq!(request.sampling_params.temperature, f64::from(0.8_f32));
        assert_eq!(request.sampling_params.top_p, 0.7);
        assert_eq!(request.sampling_params.max_new_tokens, Some(64));
        assert_eq!(request.sampling_params.sampling_seed, Some(7));
    }

    #[test]
    fn token_request_validates_input_and_rejects_unported_fields() {
        let empty = Status::from(generate(proto::GenerateRequest::default(), None).unwrap_err());
        assert_eq!(empty.code(), Code::InvalidArgument);

        let unsupported = Status::from(
            generate(
                proto::GenerateRequest {
                    input_ids: vec![1],
                    routing_key: Some("worker-a".into()),
                    priority: Some(1),
                    ..Default::default()
                },
                None,
            )
            .unwrap_err(),
        );
        assert_eq!(unsupported.code(), Code::Unimplemented);
        assert!(unsupported.message().contains("routing_key, priority"));
    }

    #[test]
    fn absent_sampling_params_still_uses_preferred_values() {
        let preferred = PreferredSamplingParams(serde_json::json!({
            "temperature": 0.25,
            "max_new_tokens": 32,
        }));
        let request = generate(
            proto::GenerateRequest {
                input_ids: vec![1],
                ..Default::default()
            },
            Some(&preferred),
        )
        .unwrap();

        assert_eq!(request.sampling_params.temperature, 0.25);
        assert_eq!(request.sampling_params.max_new_tokens, Some(32));
    }

    #[test]
    fn guided_choice_preserves_existing_runtime_v1_mapping() {
        let request = generate(
            proto::GenerateRequest {
                input_ids: vec![1],
                sampling_params: Some(proto::SamplingParams {
                    guided_decoding: Some(proto::GuidedDecoding {
                        constraint: Some(proto::guided_decoding::Constraint::Choice(
                            proto::ChoiceConstraint {
                                values: vec!["a+b".into(), "x.y".into()],
                            },
                        )),
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            },
            None,
        )
        .unwrap();
        assert_eq!(
            request.sampling_params.regex.as_deref(),
            Some("(?:a\\+b|x\\.y)")
        );
    }

    #[test]
    #[allow(deprecated)]
    fn guided_and_legacy_constraints_cannot_be_combined() {
        let error = Status::from(
            generate(
                proto::GenerateRequest {
                    input_ids: vec![1],
                    sampling_params: Some(proto::SamplingParams {
                        regex: Some("a".into()),
                        guided_decoding: Some(proto::GuidedDecoding {
                            constraint: Some(proto::guided_decoding::Constraint::Regex("b".into())),
                        }),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                None,
            )
            .unwrap_err(),
        );
        assert_eq!(error.code(), Code::InvalidArgument);
    }
}
