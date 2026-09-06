use axum::{http::StatusCode, response::Response};
use futures::{StreamExt, TryStreamExt, stream::BoxStream};

use crate::{GenerateRequest, GenerationOutput, GenerationStream, ResponseError};

use super::{OpenAIHttpFrontend, error::openai_error};

// Bound one OpenAI request's pending HTTP handshakes without duplicating the
// engine scheduler's aggregate admission policy.
const CONCURRENT_ENGINE_SUBMISSIONS: usize = 32;

/// Submit prepared token-only requests in input order.
///
/// All streams are established before either endpoint starts collecting them,
/// preserving concurrent engine execution.
pub(super) async fn submit_generate_requests(
    frontend: &OpenAIHttpFrontend,
    inputs: Vec<GenerateRequest>,
    stream_response: bool,
) -> Result<Vec<GenerationStream>, Response> {
    futures::stream::iter(inputs.into_iter().map(|input| async move {
        frontend
            .generate_client
            .generate(input)
            .await
            .map_err(|error| {
                openai_error(
                    StatusCode::from_u16(error.status_code)
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                    error.message,
                    stream_response,
                )
            })
    }))
    .buffered(CONCURRENT_ENGINE_SUBMISSIONS)
    .try_collect()
    .await
}

pub(super) fn merge_indexed(
    streams: Vec<GenerationStream>,
) -> BoxStream<'static, (usize, Result<GenerationOutput, ResponseError>)> {
    let streams = streams
        .into_iter()
        .enumerate()
        .map(|(index, events)| events.map(move |event| (index, event)).boxed());
    futures::stream::select_all(streams).boxed()
}

pub(super) async fn collect_output(
    mut events: GenerationStream,
) -> Result<GenerationOutput, ResponseError> {
    let mut collected = GenerationOutput::default();
    while let Some(item) = events.next().await {
        let output = item?;
        let finished = output.finish_reason.is_some();
        fold_output(&mut collected, output);
        if finished {
            return Ok(collected);
        }
    }
    Err(ResponseError {
        status_code: 500,
        message: "response truncated before completion".into(),
    })
}

fn fold_output(collected: &mut GenerationOutput, output: GenerationOutput) {
    collected.text.push_str(&output.text);
    collected.token_ids.extend(output.token_ids);
    collected.prompt_tokens = output.prompt_tokens;
    collected.completion_tokens = collected
        .completion_tokens
        .saturating_add(output.completion_tokens);
    if output.finish_reason.is_some() {
        collected.finish_reason = output.finish_reason;
    }
    if let Some(output) = output.extras {
        let collected = collected
            .extras
            .get_or_insert_with(|| Box::new(crate::GenerationOutputExtras::default()));
        collected.output_logprobs.extend(output.output_logprobs);
        if !output.input_logprobs.is_empty() {
            collected.input_logprobs = output.input_logprobs;
        }
    }
}

#[cfg(test)]
mod tests {
    use futures::{StreamExt, stream};

    use super::{fold_output, merge_indexed};
    use crate::{GenerationOutput, GenerationOutputExtras, PositionLogprobs, TokenLogprob};

    #[test]
    fn unary_output_appends_generated_logprobs_and_replaces_prompt_logprobs() {
        let mut collected = GenerationOutput::default();
        for (output_token, input_token) in [(1, 10), (2, 20)] {
            fold_output(
                &mut collected,
                GenerationOutput {
                    extras: Some(Box::new(GenerationOutputExtras {
                        output_logprobs: vec![position(output_token, -0.1)],
                        input_logprobs: vec![position(input_token, -0.2)],
                    })),
                    ..Default::default()
                },
            );
        }
        let extras = collected.extras.unwrap();
        assert_eq!(extras.output_logprobs[0].token.token_id, 1);
        assert_eq!(extras.output_logprobs[1].token.token_id, 2);
        assert_eq!(extras.input_logprobs[0].token.token_id, 20);
    }

    fn position(token_id: i32, logprob: f32) -> PositionLogprobs {
        PositionLogprobs {
            token: TokenLogprob {
                logprob: Some(logprob),
                token_id,
                text: None,
            },
            top: Vec::new(),
        }
    }

    #[tokio::test]
    async fn merged_stream_preserves_choice_indexes() {
        let choice0 = stream::iter([
            Ok(GenerationOutput {
                text: "a".into(),
                ..Default::default()
            }),
            Ok(GenerationOutput {
                text: "b".into(),
                ..Default::default()
            }),
        ])
        .boxed();
        let choice1 = stream::iter([Ok(GenerationOutput {
            text: "x".into(),
            ..Default::default()
        })])
        .boxed();

        let events = merge_indexed(vec![choice0, choice1])
            .collect::<Vec<_>>()
            .await;
        let mut observed = events
            .into_iter()
            .map(|(index, event)| (index, event.unwrap().text))
            .collect::<Vec<_>>();
        observed.sort();
        assert_eq!(
            observed,
            [(0, "a".into()), (0, "b".into()), (1, "x".into())]
        );
    }
}
