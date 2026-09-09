//! OpenAI completion preparation, response aggregation, and typed chunks.

use std::collections::BTreeMap;

use super::{
    completion_usage,
    protocol::{
        CompletionRequest, lower_text_completion_request, lower_token_ids_completion_request,
    },
    renderer_error,
    submission::{collect_output, merge_indexed},
    unix_seconds_u32,
};
use crate::{
    GenerateRequest, GenerationFinishReason, GenerationOutput, GenerationOutputExtras,
    GenerationStream, MatchedStop, RendererService, ResponseError, engine::HttpGenerateClient,
};
use dynamo_protocols::types::{CompletionUsage, Prompt};
use futures::StreamExt;
use serde::Serialize;

pub(crate) struct SubmittedChoice {
    pub(crate) index: usize,
    pub(crate) prompt_index: usize,
    pub(crate) echo: String,
    pub(crate) events: GenerationStream,
}

pub(crate) fn attach_streams(
    metadata: Vec<(usize, usize, String)>,
    streams: Vec<GenerationStream>,
) -> Vec<SubmittedChoice> {
    metadata
        .into_iter()
        .zip(streams)
        .map(|((index, prompt_index, echo), events)| SubmittedChoice {
            index,
            prompt_index,
            echo,
            events,
        })
        .collect()
}
#[derive(Debug, Serialize)]
#[serde(untagged)]
enum MatchedStopWire {
    Token(i64),
    Text(String),
    Tokens(Vec<i64>),
}

#[derive(Debug, PartialEq, Serialize)]
struct CompletionLogprobsWire {
    tokens: Vec<String>,
    token_logprobs: Vec<Option<f32>>,
    top_logprobs: Vec<Option<BTreeMap<String, f32>>>,
    text_offset: Vec<i32>,
}

#[derive(Debug, Serialize)]
struct CompletionChoiceWire {
    text: String,
    index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<CompletionLogprobsWire>,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
    matched_stop: Option<MatchedStopWire>,
}

#[derive(Debug, Serialize)]
pub(crate) struct CompletionResponseWire {
    id: String,
    choices: Vec<CompletionChoiceWire>,
    created: u32,
    model: String,
    object: &'static str,
    usage: Option<CompletionUsage>,
}

pub(crate) struct PreparedCompletion {
    pub(crate) requests: Vec<GenerateRequest>,
    pub(crate) metadata: Vec<(usize, usize, String)>,
    pub(crate) response_id: String,
    pub(crate) model: String,
    pub(crate) created: u32,
    pub(crate) echo: bool,
    pub(crate) want_logprobs: bool,
    pub(crate) include_usage: bool,
    pub(crate) continuous_usage: bool,
}

pub(crate) async fn prepare_request(
    renderer: &RendererService,
    client: &HttpGenerateClient,
    request: CompletionRequest,
) -> Result<PreparedCompletion, ResponseError> {
    let echo = request.echo.unwrap_or(false);
    let model = request.model.clone();
    let n = request.n.unwrap_or(1) as usize;
    let include_usage = request
        .stream_options
        .as_ref()
        .is_some_and(|options| options.include_usage)
        || renderer.config().stream_response_default_include_usage;
    let continuous_usage = request
        .stream_options
        .as_ref()
        .is_some_and(|options| options.continuous_usage_stats);
    let want_logprobs = request.logprobs.is_some();
    let created = unix_seconds_u32();
    let text_prompt = matches!(&request.prompt, Prompt::String(_) | Prompt::StringArray(_));
    let (response_id, requests, metadata) = if text_prompt {
        let (response_id, completion_requests) =
            lower_text_completion_request(renderer.config(), &request).map_err(renderer_error)?;
        let metadata = completion_requests
            .iter()
            .enumerate()
            .flat_map(|(prompt_index, request)| {
                let prompt_echo = if echo {
                    request.prompt.as_str().to_owned()
                } else {
                    String::new()
                };
                request
                    .requests
                    .iter()
                    .map(move |_| (prompt_index, prompt_echo.clone()))
            })
            .enumerate()
            .map(|(index, (prompt_index, prompt_echo))| (index, prompt_index, prompt_echo))
            .collect();
        let requests = renderer
            .prepare_text_request_groups(completion_requests)
            .await
            .map_err(renderer_error)?;
        (response_id, requests, metadata)
    } else {
        let (response_id, token_requests) =
            lower_token_ids_completion_request(renderer.config(), &request)
                .map_err(renderer_error)?;
        let mut metadata = Vec::with_capacity(token_requests.len());
        let mut prompt_echo = String::new();
        for (index, request) in token_requests.iter().enumerate() {
            let prompt_index = index / n;
            if index % n == 0 {
                prompt_echo = if echo {
                    client.detokenize(request.input_ids.clone())?
                } else {
                    String::new()
                };
            }
            metadata.push((index, prompt_index, prompt_echo.clone()));
        }
        let requests = renderer
            .prepare_token_ids_requests(token_requests)
            .map_err(renderer_error)?;
        (response_id, requests, metadata)
    };
    Ok(PreparedCompletion {
        requests,
        metadata,
        response_id,
        model,
        created,
        echo,
        want_logprobs,
        include_usage,
        continuous_usage,
    })
}

pub(crate) async fn unary_completion(
    submitted: Vec<SubmittedChoice>,
    response_id: String,
    model: String,
    created: u32,
    echo: bool,
    want_logprobs: bool,
) -> Result<CompletionResponseWire, ResponseError> {
    // Every request is already submitted, so draining in choice order does not
    // serialize generation. The non-streaming native path sends one terminal
    // result, and the accumulator also tolerates intermediate frames.
    let mut choices = Vec::with_capacity(submitted.len());
    let mut prompt_tokens = BTreeMap::<usize, u32>::new();
    let mut completion_tokens = 0u64;

    for choice in submitted {
        let output = collect_output(choice.events).await?;

        prompt_tokens
            .entry(choice.prompt_index)
            .or_insert(output.prompt_tokens);
        completion_tokens = completion_tokens.saturating_add(output.completion_tokens);
        let response_choice = completion_choice(
            choice.index,
            if echo {
                choice.echo + &output.text
            } else {
                output.text.clone()
            },
            &output,
            want_logprobs,
            echo,
        );
        choices.push(response_choice);
    }

    let prompt_tokens = prompt_tokens
        .values()
        .copied()
        .fold(0u32, u32::saturating_add);
    let usage = completion_usage(
        prompt_tokens,
        u32::try_from(completion_tokens).unwrap_or(u32::MAX),
    );

    Ok(CompletionResponseWire {
        id: response_id,
        choices,
        created,
        model,
        object: "text_completion",
        usage: Some(usage),
    })
}

fn completion_choice(
    index: usize,
    text: String,
    output: &GenerationOutput,
    want_logprobs: bool,
    include_input_logprobs: bool,
) -> CompletionChoiceWire {
    let reason = output.finish_reason.as_ref();
    let finish_reason = match reason {
        Some(GenerationFinishReason::Stop(_)) => Some("stop".into()),
        Some(GenerationFinishReason::Length) => Some("length".into()),
        Some(GenerationFinishReason::ContentFilter) => Some("content_filter".into()),
        Some(GenerationFinishReason::Abort) => Some("abort".into()),
        Some(GenerationFinishReason::Other(other)) => Some(other.clone()),
        None => None,
    };
    let matched_stop = reason
        .and_then(|reason| match reason {
            GenerationFinishReason::Stop(matched) => matched.as_ref(),
            _ => None,
        })
        .map(|matched| match matched {
            MatchedStop::Token(id) => MatchedStopWire::Token(*id),
            MatchedStop::Text(value) => MatchedStopWire::Text(value.clone()),
            // Python's OpenAI schema supports an integer or string here, not a
            // multi-token list. Preserve the native value rather than dropping it.
            MatchedStop::Tokens(ids) => MatchedStopWire::Tokens(ids.clone()),
        });
    CompletionChoiceWire {
        text,
        index: u32::try_from(index).unwrap_or(u32::MAX),
        logprobs: want_logprobs
            .then(|| completion_logprobs(output.extras.as_deref(), include_input_logprobs)),
        finish_reason,
        matched_stop,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn completion_event_stream(
    submitted: Vec<SubmittedChoice>,
    response_id: String,
    model: String,
    created: u32,
    echo: bool,
    want_logprobs: bool,
    include_usage: bool,
    continuous_usage: bool,
) -> impl futures::Stream<Item = Result<CompletionResponseWire, ResponseError>> {
    async_stream::stream! {
        let count = submitted.len();
        let mut prompt_indexes = Vec::with_capacity(count);
        let mut echoes = Vec::with_capacity(count);
        let mut first_chunks = vec![true; count];
        let mut prompt_tokens_by_prompt = BTreeMap::<usize, u32>::new();
        let mut completion_tokens_by_choice = vec![0u64; count];
        let mut streams = Vec::with_capacity(count);

        for choice in submitted {
            prompt_indexes.push(choice.prompt_index);
            echoes.push(choice.echo);
            streams.push(choice.events);
        }
        let mut events = merge_indexed(streams);

        while let Some((index, item)) = events.next().await {
            let output = match item {
                Ok(output) => output,
                Err(error) => {
                    yield Err(error);
                    break;
                }
            };

            prompt_tokens_by_prompt
                .entry(prompt_indexes[index])
                .or_insert(output.prompt_tokens);
            completion_tokens_by_choice[index] = completion_tokens_by_choice[index]
                .saturating_add(output.completion_tokens);
            let first = std::mem::replace(&mut first_chunks[index], false);
            let text = if echo && first {
                echoes[index].clone() + &output.text
            } else {
                output.text.clone()
            };
            let chunk_usage = continuous_usage.then(|| {
                completion_usage(
                    output.prompt_tokens,
                    u32::try_from(completion_tokens_by_choice[index]).unwrap_or(u32::MAX),
                )
            });
            let choice = completion_choice(
                index,
                text,
                &output,
                want_logprobs,
                echo && first,
            );
            let chunk = CompletionResponseWire {
                id: response_id.clone(),
                choices: vec![choice],
                created,
                model: model.clone(),
                object: "text_completion",
                usage: chunk_usage,
            };
            yield Ok(chunk);
        }

        if include_usage {
            let prompt_tokens = prompt_tokens_by_prompt
                .values()
                .copied()
                .fold(0u32, u32::saturating_add);
            let completion_tokens = completion_tokens_by_choice
                .into_iter()
                .fold(0u64, u64::saturating_add);
            let final_chunk = CompletionResponseWire {
                id: response_id,
                choices: vec![],
                created,
                model,
                object: "text_completion",
                usage: Some(completion_usage(
                    prompt_tokens,
                    u32::try_from(completion_tokens).unwrap_or(u32::MAX),
                )),
            };
            yield Ok(final_chunk);
        }
    }
}

fn completion_logprobs(
    extras: Option<&GenerationOutputExtras>,
    include_input: bool,
) -> CompletionLogprobsWire {
    let mut result = CompletionLogprobsWire {
        tokens: Vec::new(),
        token_logprobs: Vec::new(),
        top_logprobs: Vec::new(),
        text_offset: Vec::new(),
    };
    let Some(extras) = extras else {
        return result;
    };
    if include_input {
        append_logprobs(&mut result, &extras.input_logprobs);
    }
    append_logprobs(&mut result, &extras.output_logprobs);
    result
}

fn append_logprobs(result: &mut CompletionLogprobsWire, positions: &[crate::PositionLogprobs]) {
    for position in positions {
        let selected = &position.token;
        result.tokens.push(
            selected
                .text
                .clone()
                .unwrap_or_else(|| format!("token_id:{}", selected.token_id)),
        );
        result.token_logprobs.push(selected.logprob);
        result.text_offset.push(-1);
        if position.top.is_empty() {
            result.top_logprobs.push(None);
            continue;
        }
        let mut top = BTreeMap::new();
        for candidate in &position.top {
            let Some(logprob) = candidate.logprob else {
                continue;
            };
            top.insert(
                candidate
                    .text
                    .clone()
                    .unwrap_or_else(|| format!("token_id:{}", candidate.token_id)),
                logprob,
            );
        }
        result.top_logprobs.push(Some(top));
    }
}

#[cfg(test)]
mod tests {
    use super::{completion_event_stream, completion_logprobs, unary_completion};
    use crate::GenerationOutputExtras;
    use crate::openai::test_utils::{chunk, submitted};
    use crate::{PositionLogprobs, ResponseError, TokenLogprob};
    use futures::StreamExt;

    #[test]
    fn zero_top_logprobs_keeps_selected_token_and_empty_top_map() {
        let extras = GenerationOutputExtras {
            output_logprobs: vec![PositionLogprobs {
                token: TokenLogprob {
                    logprob: Some(-0.25),
                    token_id: 7,
                    text: Some("x".into()),
                },
                top: Vec::new(),
            }],
            ..Default::default()
        };
        let logprobs = completion_logprobs(Some(&extras), false);
        assert_eq!(logprobs.tokens, ["x"]);
        assert_eq!(logprobs.token_logprobs, [Some(-0.25)]);
        assert_eq!(logprobs.top_logprobs, [None]);
        assert_eq!(logprobs.text_offset, [-1]);
    }

    #[tokio::test]
    async fn unary_fold_orders_choices_and_counts_each_prompt_once() {
        let (choice0, tx0) = submitted(0, 0);
        let (choice1, tx1) = submitted(1, 0);
        tx0.send(chunk("a", false)).await.unwrap();
        tx0.send(chunk("b", true)).await.unwrap();
        tx1.send(chunk("x", false)).await.unwrap();
        tx1.send(chunk("y", true)).await.unwrap();

        let response = unary_completion(
            vec![choice0, choice1],
            "cmpl-test".into(),
            "model".into(),
            1,
            false,
            false,
        )
        .await;
        let value = serde_json::to_value(response.unwrap()).unwrap();
        assert_eq!(value["choices"][0]["text"], "ab");
        assert_eq!(value["choices"][1]["text"], "xy");
        assert_eq!(value["choices"][0]["matched_stop"], "</s>");
        assert!(value.get("system_fingerprint").is_none());
        assert_eq!(value["usage"]["prompt_tokens"], 5);
        assert_eq!(value["usage"]["completion_tokens"], 4);
    }

    #[tokio::test]
    async fn stream_uses_deltas_then_usage() {
        let (choice, tx) = submitted(0, 0);
        tx.send(chunk("a", false)).await.unwrap();
        tx.send(chunk("b", true)).await.unwrap();

        let stream = completion_event_stream(
            vec![choice],
            "cmpl-test".into(),
            "model".into(),
            1,
            false,
            false,
            true,
            false,
        );
        futures::pin_mut!(stream);
        let frames: Vec<_> = stream
            .map(|chunk| serde_json::to_value(chunk.unwrap()).unwrap())
            .collect()
            .await;
        assert_eq!(frames.len(), 3);
        let first = &frames[0];
        let terminal = &frames[1];
        let usage = &frames[2];
        assert_eq!(first["choices"][0]["text"], "a");
        assert_eq!(terminal["choices"][0]["text"], "b");
        assert_eq!(terminal["choices"][0]["finish_reason"], "stop");
        assert!(usage["choices"].as_array().unwrap().is_empty());
        assert_eq!(usage["usage"]["prompt_tokens"], 5);
        assert_eq!(usage["usage"]["completion_tokens"], 2);
    }

    #[tokio::test]
    async fn stream_stops_all_choices_after_error() {
        let (choice0, tx0) = submitted(0, 0);
        let (choice1, tx1) = submitted(1, 0);
        let stream = completion_event_stream(
            vec![choice0, choice1],
            "cmpl-test".into(),
            "model".into(),
            1,
            false,
            false,
            true,
            false,
        );
        futures::pin_mut!(stream);

        tx0.send(Err(ResponseError {
            status_code: 503,
            message: "out of memory".into(),
        }))
        .await
        .unwrap();
        let error = stream.next().await.unwrap().unwrap_err();
        assert_eq!(error.status_code, 503);

        tx1.send(chunk("late", true)).await.unwrap();
        let remaining = stream.collect::<Vec<_>>().await;
        assert_eq!(remaining.len(), 1);
        assert!(
            remaining
                .into_iter()
                .all(|chunk| chunk.unwrap().choices.is_empty())
        );
    }
}
