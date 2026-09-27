//! OpenAI completion preparation, response aggregation, and typed chunks.

use crate::engine::response::{collect_output, merge_indexed};
use std::collections::BTreeMap;

use super::{
    completion_usage,
    protocol::{
        CompletionRequest, lower_text_completion_request, lower_token_ids_completion_request,
        text_completion_prompts, token_ids_completion_prompts,
    },
    unix_seconds_u32,
};
use crate::{
    GenerateRequest, GenerationFinishReason, GenerationOutput, GenerationOutputExtras,
    GenerationStream, MatchedStop, RendererService, ResponseError, engine::TokenDecoder,
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
    token_logprobs: Vec<Option<f64>>,
    top_logprobs: Vec<Option<BTreeMap<String, f64>>>,
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

struct CompletionResponseContext {
    metadata: Vec<(usize, usize, String)>,
    response_id: String,
    model: String,
    created: u32,
    echo: bool,
    want_logprobs: bool,
    include_usage: bool,
    continuous_usage: bool,
}

pub(crate) async fn prepare_request(
    renderer: &RendererService,
    request: &CompletionRequest,
) -> Result<(String, Vec<GenerateRequest>), ResponseError> {
    if matches!(&request.prompt, Prompt::String(_) | Prompt::StringArray(_)) {
        let (response_id, requests) = lower_text_completion_request(renderer.config(), request)?;
        let requests = renderer.prepare_text_request_groups(requests).await?;
        Ok((response_id, requests))
    } else {
        let (response_id, requests) =
            lower_token_ids_completion_request(renderer.config(), request)?;
        let requests = renderer.prepare_token_ids_requests(requests)?;
        Ok((response_id, requests))
    }
}

// Called after request preparation has validated the prompt and choice count.
fn prepare_response(
    renderer: &RendererService,
    tokenizer: &TokenDecoder,
    request: &CompletionRequest,
    response_id: String,
    choice_count: usize,
) -> Result<CompletionResponseContext, ResponseError> {
    let echo = request.echo.unwrap_or(false);
    let n = request.n.unwrap_or(1) as usize;
    // Echo uses the original input, even when preprocessing truncates engine input IDs.
    let prompt_echoes = if !echo {
        vec![String::new(); choice_count / n]
    } else if matches!(&request.prompt, Prompt::String(_) | Prompt::StringArray(_)) {
        text_completion_prompts(&request.prompt).map_err(crate::RendererError::from)?
    } else {
        token_ids_completion_prompts(&request.prompt)
            .map_err(crate::RendererError::from)?
            .into_iter()
            .map(|ids| tokenizer.detokenize_prompt(ids))
            .collect::<Result<Vec<_>, _>>()?
    };
    let metadata = prompt_echoes
        .into_iter()
        .enumerate()
        .flat_map(|(prompt_index, echo)| {
            (0..n).map(move |choice| (prompt_index * n + choice, prompt_index, echo.clone()))
        })
        .collect();
    Ok(CompletionResponseContext {
        metadata,
        response_id,
        model: request.model.clone(),
        created: unix_seconds_u32(),
        echo,
        want_logprobs: request.logprobs.is_some(),
        include_usage: request
            .stream_options
            .as_ref()
            .is_some_and(|options| options.include_usage)
            || renderer.config().stream_response_default_include_usage,
        continuous_usage: request
            .stream_options
            .as_ref()
            .is_some_and(|options| options.continuous_usage_stats),
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
        // Python exposes the engine's f32 values as double-precision JSON numbers.
        result.token_logprobs.push(selected.logprob.map(f64::from));
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
                f64::from(logprob),
            );
        }
        result.top_logprobs.push(Some(top));
    }
}

impl super::OpenAIService {
    pub(crate) async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<
        super::OperationResponse<CompletionResponseWire, CompletionResponseWire>,
        ResponseError,
    > {
        use super::OperationResponse;
        let stream = request.stream.unwrap_or(false);
        let (response_id, requests) = prepare_request(&self.renderer, &request).await?;
        let context = prepare_response(
            &self.renderer,
            &self.generation.decoder,
            &request,
            response_id,
            requests.len(),
        )?;
        let streams = match self.generation.generate_many(requests).await {
            Ok(streams) => streams,
            Err(error) if stream => {
                return Ok(OperationResponse::Stream(
                    futures::stream::once(async { Err(error) }).boxed(),
                ));
            }
            Err(error) => return Err(error),
        };
        let submitted = attach_streams(context.metadata, streams);
        if stream {
            Ok(OperationResponse::Stream(
                completion_event_stream(
                    submitted,
                    context.response_id,
                    context.model,
                    context.created,
                    context.echo,
                    context.want_logprobs,
                    context.include_usage,
                    context.continuous_usage,
                )
                .boxed(),
            ))
        } else {
            unary_completion(
                submitted,
                context.response_id,
                context.model,
                context.created,
                context.echo,
                context.want_logprobs,
            )
            .await
            .map(OperationResponse::Unary)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        completion_event_stream, completion_logprobs, prepare_request, prepare_response,
        unary_completion,
    };
    use crate::GenerationOutputExtras;
    use crate::engine::{TokenDecoder, test_utils::tiny_tokenizer};
    use crate::openai::test_utils::{chunk, renderer_config, submitted};
    use crate::{DynamoTokenizer, PositionLogprobs, RendererService, ResponseError, TokenLogprob};
    use futures::StreamExt;
    use std::sync::Arc;

    #[tokio::test]
    async fn completion_response_preserves_batched_echo_before_truncation() {
        let tokenizer = tiny_tokenizer();
        let prompts = ["hello", "world"];
        let token_ids =
            prompts.map(|prompt| tokenizer.encode(prompt).unwrap().token_ids().to_vec());
        for truncate in [false, true] {
            let mut config = renderer_config();
            if truncate {
                config.limits.context_len = 2;
                config.limits.allow_auto_truncate = true;
                assert!(token_ids.iter().all(|ids| ids.len() > 2));
            }
            let renderer = RendererService::with_tokenizer(
                config,
                Arc::new(DynamoTokenizer::new(tokenizer.clone(), tokenizer.clone())),
                1,
                1,
            );
            for tokenized in [false, true] {
                for echo in [false, true] {
                    let prompt = if tokenized {
                        serde_json::json!(token_ids)
                    } else {
                        serde_json::json!(prompts)
                    };
                    let request = serde_json::from_value(serde_json::json!({
                        "model": "model", "prompt": prompt, "n": 2, "echo": echo,
                        "rid": ["prompt-a", "prompt-b"], "max_tokens": 4, "logprobs": 0
                    }))
                    .unwrap();
                    let (response_id, requests) =
                        prepare_request(&renderer, &request).await.unwrap();
                    let context = prepare_response(
                        &renderer,
                        &TokenDecoder::new(tokenizer.clone()),
                        &request,
                        response_id,
                        requests.len(),
                    )
                    .unwrap();

                    assert_eq!(requests.len(), 4);
                    assert_eq!(context.metadata.len(), 4);
                    assert_eq!(context.echo, echo);
                    for (index, (request, metadata)) in
                        requests.iter().zip(&context.metadata).enumerate()
                    {
                        let prompt_index = index / 2;
                        let expected_echo = if !echo {
                            String::new()
                        } else if tokenized {
                            String::from(tokenizer.decode(&token_ids[prompt_index], true).unwrap())
                        } else {
                            prompts[prompt_index].to_owned()
                        };
                        assert_eq!(metadata, &(index, prompt_index, expected_echo));
                        let mut expected_ids = token_ids[prompt_index]
                            .iter()
                            .map(|&id| id as i32)
                            .collect::<Vec<_>>();
                        if truncate {
                            expected_ids.truncate(2);
                        }
                        assert_eq!(request.input_ids, expected_ids);
                        assert_eq!(request.logprob_start_len, if echo { 0 } else { -1 });
                        assert_eq!(
                            request.rid,
                            format!(
                                "prompt-{}-{}",
                                if prompt_index == 0 { "a" } else { "b" },
                                index % 2
                            )
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn serialized_logprobs_preserve_python_float_values() {
        let selected = -1.586831_f32;
        let alternative = -2.7182817_f32;
        let extras = GenerationOutputExtras {
            output_logprobs: vec![PositionLogprobs {
                token: TokenLogprob {
                    logprob: Some(selected),
                    token_id: 7,
                    text: Some("x".into()),
                },
                top: vec![TokenLogprob {
                    logprob: Some(alternative),
                    token_id: 8,
                    text: Some("y".into()),
                }],
            }],
            ..Default::default()
        };
        // Exercise the wire serializer: to_value widens f32 before encoding it.
        let json = serde_json::to_string(&completion_logprobs(Some(&extras), false)).unwrap();
        let wire: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(
            wire["token_logprobs"][0].as_f64(),
            Some(f64::from(selected))
        );
        assert_eq!(
            wire["top_logprobs"][0]["y"].as_f64(),
            Some(f64::from(alternative))
        );
    }

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
            kind: crate::ResponseErrorKind::Unavailable,
            message: "out of memory".into(),
        }))
        .await
        .unwrap();
        let error = stream.next().await.unwrap().unwrap_err();
        assert_eq!(error.kind, crate::ResponseErrorKind::Unavailable);

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
