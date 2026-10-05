//! SGLang engine frame parsing and normalization into generation deltas.

use super::internal;
use crate::engine::TokenDelta;
use crate::{
    GenerationFinishReason, GenerationOutputExtras, MatchedStop, PositionLogprobs, ResponseError,
    TokenIds, TokenLogprob,
};
use serde::Deserialize;

type WireLogprob = (Option<f32>, i32, Option<String>);
type WireTopLogprobs = Vec<Option<Vec<WireLogprob>>>;

#[derive(Deserialize)]
struct EngineFrame {
    #[serde(default)]
    output_ids: TokenIds,
    meta_info: EngineMeta,
}

#[derive(Deserialize)]
struct EngineMeta {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u64,
    #[serde(default)]
    finish_reason: Option<EngineFinishReason>,
    #[serde(default)]
    output_token_logprobs: Vec<WireLogprob>,
    #[serde(default)]
    input_token_logprobs: Vec<WireLogprob>,
    #[serde(default)]
    output_top_logprobs: WireTopLogprobs,
    #[serde(default)]
    input_top_logprobs: WireTopLogprobs,
}

#[derive(Deserialize)]
struct EngineFinishReason {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    matched: Option<EngineMatchedStop>,
    #[serde(default)]
    status_code: Option<u16>,
    #[serde(default)]
    message: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum EngineMatchedStop {
    Token(i64),
    Text(String),
    Tokens(Vec<i64>),
}

#[derive(Deserialize)]
struct EngineErrorEnvelope {
    error: EngineError,
}

#[derive(Deserialize)]
struct EngineError {
    #[serde(default = "default_error_code")]
    code: u16,
    message: String,
}

fn default_error_code() -> u16 {
    500
}

pub(super) fn parse_engine_frame(payload: &str) -> Result<TokenDelta, ResponseError> {
    if let Ok(error) = serde_json::from_str::<EngineErrorEnvelope>(payload) {
        return Err(ResponseError {
            kind: crate::ResponseErrorKind::Upstream(crate::UpstreamErrorCode::Http(
                error.error.code,
            )),
            message: error.error.message,
        });
    }
    let frame: EngineFrame = serde_json::from_str(payload)
        .map_err(|error| internal(format!("invalid engine frame: {error}")))?;
    if let Some(reason) = frame.meta_info.finish_reason.as_ref()
        && reason.kind == "abort"
        && let Some(status_code) = reason.status_code
    {
        return Err(ResponseError {
            kind: crate::ResponseErrorKind::Upstream(crate::UpstreamErrorCode::Http(status_code)),
            message: reason
                .message
                .clone()
                .unwrap_or_else(|| "request aborted".to_owned()),
        });
    }
    let finish_reason = frame
        .meta_info
        .finish_reason
        .map(|reason| match reason.kind.as_str() {
            "stop" => GenerationFinishReason::Stop(reason.matched.map(|matched| match matched {
                EngineMatchedStop::Token(id) => MatchedStop::Token(id),
                EngineMatchedStop::Text(text) => MatchedStop::Text(text),
                EngineMatchedStop::Tokens(ids) => MatchedStop::Tokens(ids),
            })),
            "length" => GenerationFinishReason::Length,
            "abort" => GenerationFinishReason::Abort,
            "content_filter" => GenerationFinishReason::ContentFilter,
            other => GenerationFinishReason::Other(other.to_owned()),
        });
    let has_extras = !frame.meta_info.output_token_logprobs.is_empty()
        || !frame.meta_info.input_token_logprobs.is_empty()
        || !frame.meta_info.output_top_logprobs.is_empty()
        || !frame.meta_info.input_top_logprobs.is_empty();
    let output_logprobs = group_logprobs(
        frame.meta_info.output_token_logprobs,
        frame.meta_info.output_top_logprobs,
        "output",
    )?;
    let input_logprobs = group_logprobs(
        frame.meta_info.input_token_logprobs,
        frame.meta_info.input_top_logprobs,
        "input",
    )?;
    let extras = has_extras.then_some(Box::new(GenerationOutputExtras {
        output_logprobs,
        input_logprobs,
    }));
    Ok(TokenDelta {
        token_ids: frame.output_ids,
        finish_reason,
        prompt_tokens: frame.meta_info.prompt_tokens,
        completion_tokens: frame.meta_info.completion_tokens,
        extras,
    })
}

pub(super) fn normalize_engine_output(
    output: &mut TokenDelta,
    emitted_tokens: &mut u64,
) -> Result<(), ResponseError> {
    let total = output.completion_tokens;
    let delta = total.checked_sub(*emitted_tokens).ok_or_else(|| {
        internal(format!(
            "engine completion token count decreased from {} to {total}",
            *emitted_tokens
        ))
    })?;
    let output_len = u64::try_from(output.token_ids.len()).unwrap_or(u64::MAX);
    let trimmed_stop_tokens = match output.finish_reason.as_ref() {
        Some(GenerationFinishReason::Stop(Some(MatchedStop::Token(_)))) => 1,
        Some(GenerationFinishReason::Stop(Some(MatchedStop::Tokens(ids)))) => {
            u64::try_from(ids.len()).unwrap_or(u64::MAX)
        }
        _ => 0,
    };
    let cumulative =
        output_len == total || output_len.checked_add(trimmed_stop_tokens) == Some(total);
    let incremental =
        output_len == delta || output_len.checked_add(trimmed_stop_tokens) == Some(delta);

    if cumulative {
        let prefix = usize::try_from(*emitted_tokens)
            .map_err(|_| internal("engine completion token count exceeds addressable memory"))?;
        if prefix > output.token_ids.len() {
            return Err(internal(format!(
                "engine returned {output_len} cumulative output token IDs after {prefix} were already emitted"
            )));
        }
        output.token_ids.drain(..prefix);
        if let Some(extras) = output.extras.as_deref_mut() {
            trim_cumulative_output_extras(extras, prefix)?;
        }
    } else if !incremental {
        return Err(internal(format!(
            "engine returned {output_len} output token IDs after reporting {delta} new completion tokens"
        )));
    }

    output.completion_tokens = delta;
    *emitted_tokens = total;
    Ok(())
}

fn trim_cumulative_output_extras(
    extras: &mut GenerationOutputExtras,
    prefix: usize,
) -> Result<(), ResponseError> {
    drain_optional_prefix(
        &mut extras.output_logprobs,
        prefix,
        "output logprob positions",
    )
}

fn drain_prefix<T>(
    values: &mut Vec<T>,
    prefix: usize,
    description: &str,
) -> Result<(), ResponseError> {
    if values.len() < prefix {
        return Err(internal(format!(
            "engine returned {} {description} values for a {prefix}-token cumulative prefix",
            values.len()
        )));
    }
    values.drain(..prefix);
    Ok(())
}

fn drain_optional_prefix<T>(
    values: &mut Vec<T>,
    prefix: usize,
    description: &str,
) -> Result<(), ResponseError> {
    if values.is_empty() {
        return Ok(());
    }
    drain_prefix(values, prefix, description)
}

fn wire_logprob((logprob, token_id, text): WireLogprob) -> TokenLogprob {
    TokenLogprob {
        logprob,
        token_id,
        text,
    }
}

fn group_logprobs(
    values: Vec<WireLogprob>,
    top_values: WireTopLogprobs,
    kind: &str,
) -> Result<Vec<PositionLogprobs>, ResponseError> {
    // P/D can send a single null position when top logprobs are disabled.
    if top_values.iter().all(Option::is_none) {
        return Ok(values
            .into_iter()
            .map(|token| PositionLogprobs {
                token: wire_logprob(token),
                top: Vec::new(),
            })
            .collect());
    }

    if top_values.len() != values.len() {
        return Err(internal(format!(
            "engine returned {} {kind} top-logprob positions for {} selected-token positions",
            top_values.len(),
            values.len()
        )));
    }

    Ok(values
        .into_iter()
        .zip(top_values)
        .map(|(token, top)| PositionLogprobs {
            token: wire_logprob(token),
            top: top
                .unwrap_or_default()
                .into_iter()
                .map(wire_logprob)
                .collect(),
        })
        .collect())
}

pub(super) fn engine_error_message(body: &str) -> Option<String> {
    serde_json::from_str::<EngineErrorEnvelope>(body)
        .ok()
        .map(|error| error.error.message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::test_utils::position;

    #[test]
    fn engine_frame_maps_tokens_usage_finish_and_logprobs() {
        let output = parse_engine_frame(
            r#"{
                "output_ids":[7],
                "meta_info":{
                    "prompt_tokens":3,
                    "completion_tokens":1,
                    "finish_reason":{"type":"stop","matched":9},
                    "output_token_logprobs":[[-0.25,7,null]],
                    "output_top_logprobs":[[[-0.25,7,null],[-1.0,8,null]]]
                }
            }"#,
        )
        .unwrap();
        assert_eq!(output.token_ids, [7]);
        assert_eq!(output.prompt_tokens, 3);
        assert_eq!(output.completion_tokens, 1);
        assert_eq!(
            output.finish_reason,
            Some(GenerationFinishReason::Stop(Some(MatchedStop::Token(9))))
        );
        let extras = output.extras.unwrap();
        assert_eq!(extras.output_logprobs.len(), 1);
        assert_eq!(extras.output_logprobs[0].token.token_id, 7);
        assert_eq!(extras.output_logprobs[0].top.len(), 2);
    }

    #[test]
    fn engine_frame_preserves_selected_logprobs_with_absent_top_positions() {
        let output = parse_engine_frame(
            r#"{
                "output_ids":[12095,13],
                "meta_info":{
                    "prompt_tokens":5,
                    "completion_tokens":2,
                    "output_token_logprobs":[
                        [-0.42652416229248047,12095,null],
                        [-0.7053262591362,13,null]
                    ],
                    "output_top_logprobs":[null]
                }
            }"#,
        )
        .unwrap();

        assert_eq!(output.token_ids, [12095, 13]);
        assert_eq!(
            output.extras.unwrap().output_logprobs,
            [
                position(12095, -0.42652416, &[]),
                position(13, -0.70532626, &[])
            ]
        );
    }

    #[test]
    fn engine_frame_rejects_misaligned_logprob_positions() {
        let error = parse_engine_frame(
            r#"{
                "output_ids":[7,8],
                "meta_info":{
                    "completion_tokens":2,
                    "output_token_logprobs":[[-0.25,7,null],[-0.5,8,null]],
                    "output_top_logprobs":[[[-0.25,7,null]]]
                }
            }"#,
        )
        .unwrap_err();

        assert_eq!(error.kind, crate::ResponseErrorKind::Internal);
        assert_eq!(
            error.message,
            "engine returned 1 output top-logprob positions for 2 selected-token positions"
        );
    }

    #[test]
    fn engine_error_frame_preserves_status_and_message() {
        let error = parse_engine_frame(
            r#"{"error":{"message":"too long","type":"BadRequestError","code":400}}"#,
        )
        .unwrap_err();
        assert_eq!(
            error.kind,
            crate::ResponseErrorKind::Upstream(crate::UpstreamErrorCode::Http(400))
        );
        assert_eq!(error.message, "too long");
    }

    #[test]
    fn coded_abort_frame_preserves_status_and_message() {
        let error = parse_engine_frame(
            r#"{"output_ids":[],"meta_info":{"finish_reason":{"type":"abort","status_code":503,"message":"out of memory"}}}"#,
        )
        .unwrap_err();

        assert_eq!(
            error.kind,
            crate::ResponseErrorKind::Upstream(crate::UpstreamErrorCode::Http(503))
        );
        assert_eq!(error.message, "out of memory");
    }

    #[test]
    fn uncoded_abort_frame_remains_a_finish_reason() {
        let output = parse_engine_frame(
            r#"{"output_ids":[],"meta_info":{"finish_reason":{"type":"abort","status_code":null,"message":"cancelled"}}}"#,
        )
        .unwrap();

        assert_eq!(output.finish_reason, Some(GenerationFinishReason::Abort));
    }

    #[test]
    fn cumulative_engine_frames_become_deltas() {
        let mut emitted_tokens = 1;
        let mut output = TokenDelta {
            token_ids: vec![7, 8],
            completion_tokens: 2,
            extras: Some(Box::new(GenerationOutputExtras {
                output_logprobs: vec![
                    position(7, -0.5, &[(7, -0.5), (9, -1.0)]),
                    position(8, -0.25, &[(8, -0.25)]),
                ],
                ..Default::default()
            })),
            ..Default::default()
        };

        normalize_engine_output(&mut output, &mut emitted_tokens).unwrap();

        assert_eq!(output.token_ids, [8]);
        assert_eq!(output.completion_tokens, 1);
        assert_eq!(emitted_tokens, 2);
        let extras = output.extras.unwrap();
        assert_eq!(extras.output_logprobs.len(), 1);
        assert_eq!(extras.output_logprobs[0].token.token_id, 8);
        assert_eq!(extras.output_logprobs[0].top.len(), 1);
    }

    #[test]
    fn token_stops_may_be_trimmed_from_incremental_or_cumulative_frames() {
        for token_ids in [vec![], vec![7]] {
            let mut emitted_tokens = 1;
            let mut output = TokenDelta {
                token_ids,
                completion_tokens: 2,
                finish_reason: Some(GenerationFinishReason::Stop(Some(MatchedStop::Token(9)))),
                ..Default::default()
            };

            normalize_engine_output(&mut output, &mut emitted_tokens).unwrap();

            assert!(output.token_ids.is_empty());
            assert_eq!(output.completion_tokens, 1);
            assert_eq!(emitted_tokens, 2);
        }
    }

    #[test]
    fn inconsistent_engine_token_counts_are_rejected() {
        let mut emitted_tokens = 2;
        let mut output = TokenDelta {
            token_ids: vec![7, 8],
            completion_tokens: 3,
            ..Default::default()
        };

        let error = normalize_engine_output(&mut output, &mut emitted_tokens).unwrap_err();

        assert_eq!(error.kind, crate::ResponseErrorKind::Internal);
        assert!(error.message.contains("2 output token IDs"));
        assert_eq!(emitted_tokens, 2);
    }
}
