//! OpenAI legacy text-completion endpoint and wire shaping.

use std::collections::BTreeMap;
use std::convert::Infallible;

use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    http::{HeaderMap, StatusCode},
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::post,
};
use dynamo_protocols::types::{
    Choice, CompletionFinishReason, CompletionUsage, CreateCompletionRequest,
    CreateCompletionResponse, Logprobs, Prompt, Stop,
};
use futures::StreamExt;
use tokio::sync::mpsc;

use super::super::guard::AbortGuard;
use super::{
    AppState, MAX_OPENAI_CHOICES, authorize, collect_output, indexed_egress_stream, openai_error,
    streaming_error, submit_generation, unix_seconds_u32,
};
use crate::ids::Rid;
use crate::message::{
    ChunkEvent, ChunkExtras, EgressItem, GenerateRequest, Matched, OneOrMany, SamplingParams,
};

pub(super) fn routes() -> Router<AppState> {
    Router::new().route("/v1/completions", post(completions))
}

#[derive(Debug)]
struct PromptSpec {
    text: Option<String>,
    input_ids: Option<Vec<i32>>,
    echo: String,
}

pub(super) struct SubmittedChoice {
    pub(super) index: usize,
    pub(super) prompt_index: usize,
    pub(super) rid: Rid,
    pub(super) echo: String,
    pub(super) rx: mpsc::Receiver<EgressItem>,
}
#[derive(Debug, Default)]
pub(super) struct ChoiceExtensions {
    matched_stop: Option<serde_json::Value>,
    /// Dynamo's enum covers the standard values. Python additionally exposes
    /// `abort`, and native unknown finish types are preserved rather than lost.
    finish_reason_override: Option<String>,
}

async fn completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Result<Json<CreateCompletionRequest>, JsonRejection>,
) -> Response {
    if let Some(response) = authorize(&state, &headers) {
        return response;
    }
    let request = match body {
        Ok(Json(request)) => request,
        Err(rejection) => {
            return openai_error(StatusCode::BAD_REQUEST, rejection.body_text());
        }
    };
    let stream = request.stream.unwrap_or(false);
    let model = request.model.clone();
    if model != state.server_args.served_model_name {
        return openai_error(
            StatusCode::BAD_REQUEST,
            format!("The model `{model}` does not exist"),
        );
    }

    if request.prompt_embeds.is_some() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "prompt_embeds is not supported by the Rust frontend",
        );
    }
    if request.suffix.is_some() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "suffix is not supported by this model",
        );
    }
    if request.best_of.is_some_and(|best_of| best_of != 1) {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "best_of values greater than 1 are not supported",
        );
    }
    if request.max_tokens == Some(0) {
        return openai_error(StatusCode::BAD_REQUEST, "max_tokens must be positive");
    }
    if request.n == Some(0) {
        return openai_error(StatusCode::BAD_REQUEST, "n must be at least 1");
    }
    let prompts = match completion_prompt_specs(
        &request.prompt,
        request.echo.unwrap_or(false),
        state.tokenizer.as_ref(),
    ) {
        Ok(prompts) => prompts,
        Err(message) => return openai_error(StatusCode::BAD_REQUEST, message),
    };
    let mut sampling = match completion_sampling_params(&request) {
        Ok(sampling) => sampling,
        Err(message) => return openai_error(StatusCode::BAD_REQUEST, message),
    };
    if let Err(error) = sampling.normalize(
        state.server_args.skip_tokenizer_init,
        state
            .server_args
            .model_config
            .vocab_size
            .unwrap_or(u64::MAX),
    ) {
        return openai_error(StatusCode::BAD_REQUEST, error.to_string());
    }

    let n = request.n.unwrap_or(1) as usize;
    let choice_count = match prompts.len().checked_mul(n) {
        Some(count) if count <= MAX_OPENAI_CHOICES => count,
        _ => {
            return openai_error(
                StatusCode::BAD_REQUEST,
                format!("prompt count times n exceeds the maximum of {MAX_OPENAI_CHOICES}"),
            );
        }
    };
    let response_id = format!("cmpl-{}", uuid::Uuid::new_v4().simple());
    let created = unix_seconds_u32();
    let mut guard = AbortGuard::new_empty(state.senders.clone());
    let mut submitted = Vec::with_capacity(choice_count);

    for (prompt_index, prompt) in prompts.into_iter().enumerate() {
        for sample_index in 0..n {
            let index = prompt_index * n + sample_index;
            let rid = Rid::from_client(&format!("{response_id}-{index}"));
            let native = GenerateRequest {
                rid: rid.clone(),
                text: prompt.text.clone(),
                input_ids: prompt.input_ids.clone(),
                sampling_params: sampling.clone(),
                stream,
                return_logprob: request.logprobs.is_some(),
                logprob_start_len: if request.echo.unwrap_or(false) && request.logprobs.is_some() {
                    0
                } else {
                    -1
                },
                top_logprobs_num: request.logprobs.unwrap_or(0) as i64,
                return_text_in_logprobs: request.logprobs.map(|_| true),
                ..Default::default()
            };
            let rx = match submit_generation(&state, native, stream, &mut guard).await {
                Ok(rx) => rx,
                Err(response) => return response,
            };
            submitted.push(SubmittedChoice {
                index,
                prompt_index,
                rid,
                echo: prompt.echo.clone(),
                rx,
            });
        }
    }

    if stream {
        let include_usage = request
            .stream_options
            .map(|o| o.include_usage)
            .unwrap_or(false)
            || state.server_args.stream_response_default_include_usage;
        let continuous_usage = request
            .stream_options
            .map(|o| o.continuous_usage_stats)
            .unwrap_or(false);
        let want_logprobs = request.logprobs.is_some();
        let echo = request.echo.unwrap_or(false);
        let s = completion_event_stream(
            submitted,
            guard,
            response_id,
            model,
            created,
            echo,
            want_logprobs,
            include_usage,
            continuous_usage,
        )
        .map(|data| Ok::<_, Infallible>(Event::default().data(data)));
        Sse::new(s).into_response()
    } else {
        unary_completion(
            submitted,
            guard,
            response_id,
            model,
            created,
            request.echo.unwrap_or(false),
            request.logprobs.is_some(),
        )
        .await
    }
}

fn completion_prompt_specs(
    prompt: &Prompt,
    echo: bool,
    tokenizer: Option<&dynamo_tokenizers::Tokenizer>,
) -> Result<Vec<PromptSpec>, String> {
    match prompt {
        Prompt::String(text) => {
            if text.is_empty() {
                return Err("Prompt cannot be empty".into());
            }
            Ok(vec![PromptSpec {
                text: Some(text.clone()),
                input_ids: None,
                echo: if echo { text.clone() } else { String::new() },
            }])
        }
        Prompt::StringArray(texts) => {
            if texts.is_empty() || texts.iter().any(String::is_empty) {
                return Err("Prompt cannot be empty".into());
            }
            Ok(texts
                .iter()
                .map(|text| PromptSpec {
                    text: Some(text.clone()),
                    input_ids: None,
                    echo: if echo { text.clone() } else { String::new() },
                })
                .collect())
        }
        Prompt::IntegerArray(ids) => Ok(vec![token_prompt_spec(ids, echo, tokenizer)?]),
        Prompt::ArrayOfIntegerArray(prompts) => {
            if prompts.is_empty() {
                return Err("Prompt cannot be empty".into());
            }
            prompts
                .iter()
                .map(|ids| token_prompt_spec(ids, echo, tokenizer))
                .collect()
        }
    }
}

fn token_prompt_spec(
    ids: &[u32],
    echo: bool,
    tokenizer: Option<&dynamo_tokenizers::Tokenizer>,
) -> Result<PromptSpec, String> {
    if ids.is_empty() {
        return Err("Prompt cannot be empty".into());
    }
    let input_ids = ids
        .iter()
        .map(|&id| i32::try_from(id).map_err(|_| format!("Token ID {id} is out of range")))
        .collect::<Result<Vec<_>, _>>()?;
    let echo = if echo {
        tokenizer
            .ok_or_else(|| {
                "echo for token-ID prompts is unavailable when skip_tokenizer_init=True".to_string()
            })?
            .decode(ids, true)
            .map_err(|e| format!("failed to decode prompt for echo: {e}"))?
            .as_str()
            .to_owned()
    } else {
        String::new()
    };
    Ok(PromptSpec {
        text: None,
        input_ids: Some(input_ids),
        echo,
    })
}

fn completion_sampling_params(request: &CreateCompletionRequest) -> Result<SamplingParams, String> {
    let mut stop = None;
    let mut stop_token_ids = None;
    match request.stop.as_ref() {
        Some(Stop::String(value)) => stop = Some(OneOrMany::One(value.clone())),
        Some(Stop::StringArray(values)) => stop = Some(OneOrMany::Many(values.clone())),
        Some(Stop::TokenIdArray(values)) => {
            stop_token_ids
                .get_or_insert_with(Vec::new)
                .extend(values.iter().map(|&id| id as i64));
        }
        None => {}
    }

    let mut logit_bias = BTreeMap::new();
    if let Some(values) = request.logit_bias.as_ref() {
        for (token, bias) in values {
            let bias = bias
                .as_f64()
                .ok_or_else(|| format!("logit_bias[{token:?}] must be a number"))?;
            logit_bias.insert(token.clone(), bias);
        }
    }

    Ok(SamplingParams {
        max_new_tokens: Some(request.max_tokens.unwrap_or(16) as i64),
        stop,
        stop_token_ids,
        temperature: request.temperature.unwrap_or(1.0) as f64,
        top_p: request.top_p.unwrap_or(1.0) as f64,
        frequency_penalty: request.frequency_penalty.unwrap_or(0.0) as f64,
        presence_penalty: request.presence_penalty.unwrap_or(0.0) as f64,
        // OpenAI `n` is implemented by fan-out: every native request has one
        // output, avoiding the native path's intentional `n > 1` rejection.
        n: 1,
        logit_bias: (!logit_bias.is_empty()).then_some(logit_bias),
        sampling_seed: request.seed,
        ..Default::default()
    })
}

pub(super) async fn unary_completion(
    submitted: Vec<SubmittedChoice>,
    mut guard: AbortGuard,
    response_id: String,
    model: String,
    created: u32,
    echo: bool,
    want_logprobs: bool,
) -> Response {
    // Every request is already submitted, so draining in choice order does not
    // serialize generation. The non-streaming native path sends one terminal
    // result, and the accumulator also tolerates intermediate frames.
    let mut choices = Vec::with_capacity(submitted.len());
    let mut extensions = Vec::with_capacity(submitted.len());
    let mut prompt_tokens = BTreeMap::<usize, u32>::new();
    let mut completion_tokens = 0u64;

    for choice in submitted {
        let output = match collect_output(choice.rx, &mut guard, &choice.rid).await {
            Ok(output) => output,
            Err((status, message)) => return openai_error(status, message),
        };

        prompt_tokens
            .entry(choice.prompt_index)
            .or_insert(output.prompt_tokens);
        completion_tokens = completion_tokens.saturating_add(output.completion_tokens);
        let (response_choice, extension) = completion_choice(
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
        extensions.push(extension);
    }

    let prompt_tokens = prompt_tokens
        .values()
        .copied()
        .fold(0u32, u32::saturating_add);
    let usage = completion_usage(
        prompt_tokens,
        u32::try_from(completion_tokens).unwrap_or(u32::MAX),
    );

    Json(completion_response_value(
        CreateCompletionResponse {
            id: response_id,
            choices,
            created,
            model,
            system_fingerprint: None,
            object: "text_completion".into(),
            usage: Some(usage),
        },
        &extensions,
    ))
    .into_response()
}

fn completion_choice(
    index: usize,
    text: String,
    output: &ChunkEvent,
    want_logprobs: bool,
    include_input_logprobs: bool,
) -> (Choice, ChoiceExtensions) {
    let reason = output.finish_reason.as_ref();
    let (finish_reason, finish_reason_override) = {
        match reason.and_then(|reason| reason.kind_name()) {
            Some("stop") => (Some(CompletionFinishReason::Stop), None),
            Some("length") => (Some(CompletionFinishReason::Length), None),
            Some("content_filter") => (Some(CompletionFinishReason::ContentFilter), None),
            Some(other) => (None, Some(other.into())),
            None => (None, None),
        }
    };
    let matched_stop = reason
        .and_then(|reason| reason.matched())
        .map(|matched| match matched {
            Matched::Token(id) => serde_json::json!(id),
            Matched::Str(value) => serde_json::json!(value),
            // Python's OpenAI schema supports an integer or string here, not a
            // multi-token list. Preserve the native value rather than dropping it.
            Matched::Tokens(ids) => serde_json::json!(ids),
        });
    (
        Choice {
            text,
            index: u32::try_from(index).unwrap_or(u32::MAX),
            logprobs: want_logprobs
                .then(|| completion_logprobs(output.extras.as_deref(), include_input_logprobs)),
            finish_reason,
        },
        ChoiceExtensions {
            matched_stop,
            finish_reason_override,
        },
    )
}

/// Serialize Dynamo's standard response and add only SGLang/Python fields that
/// its schema cannot represent. `text_offset` is corrected here because Dynamo
/// types it as `u32`, while Python deliberately emits `-1`.
pub(super) fn completion_response_value(
    response: CreateCompletionResponse,
    extensions: &[ChoiceExtensions],
) -> serde_json::Value {
    let mut value = serde_json::to_value(response).expect("OpenAI response must serialize");
    let Some(root) = value.as_object_mut() else {
        return value;
    };
    // Python's Completion response does not expose this OpenAI field.
    root.remove("system_fingerprint");
    let Some(choices) = root
        .get_mut("choices")
        .and_then(serde_json::Value::as_array_mut)
    else {
        return value;
    };
    for (choice, extension) in choices.iter_mut().zip(extensions) {
        let Some(choice) = choice.as_object_mut() else {
            continue;
        };
        if let Some(reason) = &extension.finish_reason_override {
            choice.insert("finish_reason".into(), serde_json::json!(reason));
        }
        choice.insert(
            "matched_stop".into(),
            extension
                .matched_stop
                .clone()
                .unwrap_or(serde_json::Value::Null),
        );
        if let Some(logprobs) = choice
            .get_mut("logprobs")
            .and_then(serde_json::Value::as_object_mut)
        {
            let count = logprobs
                .get("tokens")
                .and_then(serde_json::Value::as_array)
                .map_or(0, Vec::len);
            logprobs.insert("text_offset".into(), serde_json::json!(vec![-1; count]));
        }
    }
    value
}

#[allow(clippy::too_many_arguments)]
pub(super) fn completion_event_stream(
    submitted: Vec<SubmittedChoice>,
    mut guard: AbortGuard,
    response_id: String,
    model: String,
    created: u32,
    echo: bool,
    want_logprobs: bool,
    include_usage: bool,
    continuous_usage: bool,
) -> impl futures::Stream<Item = String> {
    async_stream::stream! {
        let count = submitted.len();
        let mut rids = Vec::with_capacity(count);
        let mut prompt_indexes = Vec::with_capacity(count);
        let mut echoes = Vec::with_capacity(count);
        let mut first_chunks = vec![true; count];
        let mut prompt_tokens_by_prompt = BTreeMap::<usize, u32>::new();
        let mut completion_tokens_by_choice = vec![0u64; count];
        let mut streams = Vec::with_capacity(count);

        for choice in submitted {
            let index = choice.index;
            rids.push(choice.rid);
            prompt_indexes.push(choice.prompt_index);
            echoes.push(choice.echo);
            streams.push(indexed_egress_stream(index, choice.rx));
        }
        let mut events = futures::stream::select_all(streams);

        while let Some((index, item)) = events.next().await {
            let Some(item) = item else {
                yield streaming_error(500, "response truncated before completion");
                continue;
            };
            let output = match item {
                EgressItem::Frame(output) => output,
                EgressItem::Done(output) => {
                    guard.disarm(&rids[index]);
                    output
                }
                EgressItem::Error(error) => {
                    guard.disarm(&rids[index]);
                    yield streaming_error(error.http_status(), error.to_string());
                    continue;
                }
                EgressItem::Control(_) => continue,
            };

            if let Some((code, message)) = output
                .finish_reason
                .as_ref()
                .and_then(|reason| reason.abort_status())
            {
                yield streaming_error(code, message);
                continue;
            }

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
            let (choice, extension) = completion_choice(
                index,
                text,
                &output,
                want_logprobs,
                echo && first,
            );
            let chunk = CreateCompletionResponse {
                id: response_id.clone(),
                choices: vec![choice],
                created,
                model: model.clone(),
                system_fingerprint: None,
                object: "text_completion".into(),
                usage: chunk_usage,
            };
            yield completion_response_value(chunk, &[extension]).to_string();
        }

        if include_usage {
            let prompt_tokens = prompt_tokens_by_prompt
                .values()
                .copied()
                .fold(0u32, u32::saturating_add);
            let completion_tokens = completion_tokens_by_choice
                .into_iter()
                .fold(0u64, u64::saturating_add);
            let final_chunk = CreateCompletionResponse {
                id: response_id,
                choices: vec![],
                created,
                model,
                system_fingerprint: None,
                object: "text_completion".into(),
                usage: Some(completion_usage(
                    prompt_tokens,
                    u32::try_from(completion_tokens).unwrap_or(u32::MAX),
                )),
            };
            yield completion_response_value(final_chunk, &[]).to_string();
        }
        yield "[DONE]".to_string();
    }
}

pub(super) fn completion_usage(prompt_tokens: u32, completion_tokens: u32) -> CompletionUsage {
    CompletionUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens.saturating_add(completion_tokens),
        ..Default::default()
    }
}

pub(super) fn completion_logprobs(extras: Option<&ChunkExtras>, include_input: bool) -> Logprobs {
    let mut result = Logprobs {
        tokens: Vec::new(),
        token_logprobs: Vec::new(),
        top_logprobs: Vec::new(),
        text_offset: Vec::new(),
    };
    let Some(extras) = extras else {
        return result;
    };
    if include_input {
        append_selected_logprobs(
            &mut result,
            &extras.in_lp_val,
            &extras.in_lp_idx,
            &extras.in_lp_txt,
        );
        append_top_logprobs(
            &mut result,
            &extras.in_top_val,
            &extras.in_top_idx,
            &extras.in_top_lens,
            &extras.in_top_txt,
        );
    }
    append_selected_logprobs(
        &mut result,
        &extras.out_lp_val,
        &extras.out_lp_idx,
        &extras.out_lp_txt,
    );
    append_top_logprobs(
        &mut result,
        &extras.out_top_val,
        &extras.out_top_idx,
        &extras.out_top_lens,
        &extras.out_top_txt,
    );
    result
}

fn append_selected_logprobs(result: &mut Logprobs, values: &[f32], ids: &[i32], texts: &[String]) {
    for (index, (&value, &id)) in values.iter().zip(ids).enumerate() {
        result.tokens.push(
            texts
                .get(index)
                .cloned()
                .unwrap_or_else(|| format!("token_id:{id}")),
        );
        result
            .token_logprobs
            .push((!value.is_nan()).then_some(value));
        // Dynamo's field is `u32`; Python's `-1` sentinel is applied once at
        // final wire shaping in `completion_response_value`.
        result.text_offset.push(0);
    }
}

fn append_top_logprobs(
    result: &mut Logprobs,
    values: &[f32],
    ids: &[i32],
    lens: &[u32],
    texts: &[String],
) {
    let mut offset = 0usize;
    for &len in lens {
        let len = len as usize;
        if len == 0 {
            result.top_logprobs.push(serde_json::Value::Null);
            continue;
        }
        let mut top = BTreeMap::new();
        for index in offset..offset.saturating_add(len) {
            let (Some(&value), Some(&id)) = (values.get(index), ids.get(index)) else {
                continue;
            };
            top.insert(
                texts
                    .get(index)
                    .cloned()
                    .unwrap_or_else(|| format!("token_id:{id}")),
                value,
            );
        }
        result.top_logprobs.push(serde_json::json!(top));
        offset = offset.saturating_add(len);
    }
}
