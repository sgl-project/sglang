use axum::{http::StatusCode, response::Response};
use futures::{StreamExt, stream::BoxStream};

use crate::{FrontendError, GenerationEvent, GenerationOutput, GenerationStream, TextRequest};

use super::{OpenAIHttpFrontend, error::openai_error};

/// Prepare and submit each request in input order.
///
/// All streams are established before either endpoint starts collecting them,
/// preserving concurrent engine execution without introducing a backend trait.
pub(super) async fn submit_inputs(
    frontend: &OpenAIHttpFrontend,
    inputs: Vec<TextRequest>,
    stream_response: bool,
) -> Result<Vec<GenerationStream>, Response> {
    let mut streams = Vec::with_capacity(inputs.len());
    for input in inputs {
        let prepared = frontend
            .renderer
            .prepare_text_request(input)
            .await
            .map_err(|error| {
                openai_error(
                    super::error::renderer_status(&error),
                    error.to_string(),
                    false,
                )
            })?;
        let events = frontend
            .generate_client
            .generate(prepared)
            .await
            .map_err(|error| {
                openai_error(
                    StatusCode::from_u16(error.status_code)
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                    error.message,
                    stream_response,
                )
            })?;
        streams.push(events);
    }
    Ok(streams)
}

pub(super) fn merge_indexed(
    streams: Vec<GenerationStream>,
) -> BoxStream<'static, (usize, Result<GenerationEvent, FrontendError>)> {
    let streams = streams
        .into_iter()
        .enumerate()
        .map(|(index, events)| events.map(move |event| (index, event)).boxed());
    futures::stream::select_all(streams).boxed()
}

pub(super) async fn collect_output(
    mut events: GenerationStream,
) -> Result<GenerationOutput, FrontendError> {
    let mut collected = GenerationOutput::default();
    while let Some(item) = events.next().await {
        match item? {
            GenerationEvent::Frame(output) => fold_output(&mut collected, output),
            GenerationEvent::Done(output) => {
                fold_output(&mut collected, output);
                return Ok(collected);
            }
        }
    }
    Err(FrontendError {
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
        collected
            .output_logprob_token_ids
            .extend(output.output_logprob_token_ids);
        collected
            .output_logprob_text
            .extend(output.output_logprob_text);
        collected
            .output_top_logprobs
            .extend(output.output_top_logprobs);
        collected
            .output_top_logprob_token_ids
            .extend(output.output_top_logprob_token_ids);
        collected
            .output_top_logprob_lengths
            .extend(output.output_top_logprob_lengths);
        collected
            .output_top_logprob_text
            .extend(output.output_top_logprob_text);
        if !output.input_logprobs.is_empty() {
            collected.input_logprobs = output.input_logprobs;
            collected.input_logprob_token_ids = output.input_logprob_token_ids;
            collected.input_logprob_text = output.input_logprob_text;
        }
        if !output.input_top_logprob_lengths.is_empty() {
            collected.input_top_logprobs = output.input_top_logprobs;
            collected.input_top_logprob_token_ids = output.input_top_logprob_token_ids;
            collected.input_top_logprob_lengths = output.input_top_logprob_lengths;
            collected.input_top_logprob_text = output.input_top_logprob_text;
        }
    }
}

#[cfg(test)]
mod tests {
    use futures::{StreamExt, stream};

    use super::{fold_output, merge_indexed};
    use crate::{GenerationEvent, GenerationOutput, GenerationOutputExtras};

    #[test]
    fn unary_output_appends_generated_logprobs_and_replaces_prompt_logprobs() {
        let mut collected = GenerationOutput::default();
        for (output_token, input_token) in [(1, 10), (2, 20)] {
            fold_output(
                &mut collected,
                GenerationOutput {
                    extras: Some(Box::new(GenerationOutputExtras {
                        output_logprobs: vec![-0.1],
                        output_logprob_token_ids: vec![output_token],
                        input_logprobs: vec![-0.2],
                        input_logprob_token_ids: vec![input_token],
                        ..Default::default()
                    })),
                    ..Default::default()
                },
            );
        }
        let extras = collected.extras.unwrap();
        assert_eq!(extras.output_logprob_token_ids, [1, 2]);
        assert_eq!(extras.input_logprob_token_ids, [20]);
    }

    #[tokio::test]
    async fn merged_stream_preserves_choice_indexes() {
        let choice0 = stream::iter([
            Ok(GenerationEvent::Frame(GenerationOutput {
                text: "a".into(),
                ..Default::default()
            })),
            Ok(GenerationEvent::Done(GenerationOutput {
                text: "b".into(),
                ..Default::default()
            })),
        ])
        .boxed();
        let choice1 = stream::iter([Ok(GenerationEvent::Done(GenerationOutput {
            text: "x".into(),
            ..Default::default()
        }))])
        .boxed();

        let events = merge_indexed(vec![choice0, choice1])
            .collect::<Vec<_>>()
            .await;
        let mut observed = events
            .into_iter()
            .map(|(index, event)| {
                let output = match event.unwrap() {
                    GenerationEvent::Frame(output) | GenerationEvent::Done(output) => output,
                };
                (index, output.text)
            })
            .collect::<Vec<_>>();
        observed.sort();
        assert_eq!(
            observed,
            [(0, "a".into()), (0, "b".into()), (1, "x".into())]
        );
    }
}
