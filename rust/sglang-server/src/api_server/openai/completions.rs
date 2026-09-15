//! OpenAI legacy text-completion endpoint and wire shaping.

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::sync::Arc;

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
    Choice, CompletionFinishReason, CreateCompletionRequest, CreateCompletionResponse, Logprobs,
    Prompt, Stop,
};
use futures::StreamExt;
use serde::Deserialize;

use super::super::guard::AbortGuard;
use super::super::submit::submit;
use super::{
    AppState, MAX_OPENAI_CHOICES, collect_output, error_payload, indexed_decode_stream,
    openai_error, submit_generation, unix_seconds_u32,
};
use crate::message::finish_reason::Matched;
use crate::message::ids::Rid;
use crate::message::request::RequestKind;
use crate::message::response::{ChunkEvent, ChunkExtras, ResponseItem};
use crate::message::sampling::{SamplingParams, SamplingParamsInput};
use crate::message::types::{OneOrMany, TokenIds};
use crate::utils::error::Error;

pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new().route("/v1/completions", post(completions))
}

#[derive(Debug, PartialEq, Eq)]
enum PromptSpec {
    Text(String),
    TokenIds(TokenIds),
}

pub(super) struct SubmittedChoice {
    pub(super) index: usize,
    pub(super) prompt_index: usize,
    pub(super) rid: Rid,
    pub(super) echo: String,
    pub(super) rx: super::ResponseReceiver,
}
#[derive(Debug, Default)]
pub(super) struct ChoiceExtensions {
    matched_stop: Option<serde_json::Value>,
    /// Dynamo's enum covers the standard values. Python additionally exposes
    /// `abort`; unrecognized scheduler finish types are preserved as well.
    finish_reason_override: Option<String>,
}

async fn completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Json<serde_json::Value>, JsonRejection>,
) -> Response {
    let started = std::time::Instant::now();
    let custom_labels = state
        .frontend_metrics
        .as_ref()
        .and_then(|metrics| metrics.custom_labels(&headers));
    let raw_request = match body {
        Ok(Json(request)) => request,
        Err(rejection) => {
            return openai_error(StatusCode::BAD_REQUEST, rejection.body_text(), false);
        }
    };
    let request = match CreateCompletionRequest::deserialize(&raw_request) {
        Ok(request) => request,
        Err(error) => return openai_error(StatusCode::BAD_REQUEST, error.to_string(), false),
    };
    let (mut native_body, output_options) =
        match super::extensions::request_options(&raw_request, &headers, &state.server_args, false)
        {
            Ok(options) => options,
            Err(error) => return openai_error(StatusCode::BAD_REQUEST, error, false),
        };
    let stream = request.stream.unwrap_or(false);
    let echo = request.echo.unwrap_or(false);
    let model = request.model.clone();

    if request.prompt_embeds.is_some() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "prompt_embeds is not supported by the Rust frontend",
            false,
        );
    }
    if request.suffix.is_some() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "suffix is not supported by this model",
            false,
        );
    }
    if request.best_of.is_some_and(|best_of| best_of != 1) {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "best_of values greater than 1 are not supported",
            false,
        );
    }
    if request.n == Some(0) {
        return openai_error(StatusCode::BAD_REQUEST, "n must be at least 1", false);
    }
    let prompts = match completion_prompt_specs(&request.prompt) {
        Ok(prompts) => prompts,
        Err(message) => {
            return openai_error(StatusCode::BAD_REQUEST, &message, false);
        }
    };
    let mut sampling = match completion_sampling_params(&request) {
        Ok(sampling) => sampling,
        Err(message) => {
            return openai_error(StatusCode::BAD_REQUEST, &message, false);
        }
    };
    if let Err(error) = super::extensions::apply_sampling(&raw_request, &mut sampling, None) {
        return openai_error(StatusCode::BAD_REQUEST, error, false);
    }
    if let Err(error) = sampling.normalize(
        state.server_args.skip_tokenizer_init,
        state.server_args.model_config.vocab_size,
    ) {
        return openai_error(StatusCode::BAD_REQUEST, error.to_string(), false);
    }

    let n = request.n.unwrap_or(1) as usize;
    let choice_count = match prompts.len().checked_mul(n) {
        Some(count) if count <= MAX_OPENAI_CHOICES => count,
        _ => {
            return openai_error(
                StatusCode::BAD_REQUEST,
                format!("prompt count times n exceeds the maximum of {MAX_OPENAI_CHOICES}"),
                false,
            );
        }
    };
    if native_body.rid.is_none() {
        native_body.rid = Some(OneOrMany::One(format!(
            "cmpl-{}",
            uuid::Uuid::new_v4().simple()
        )));
    }
    if matches!(prompts.first(), Some(PromptSpec::Text(_))) {
        native_body.text = match serde_json::from_value(raw_request["prompt"].clone()) {
            Ok(text) => Some(text),
            Err(error) => return openai_error(StatusCode::BAD_REQUEST, error.to_string(), false),
        };
    } else {
        native_body.input_ids = match serde_json::from_value(raw_request["prompt"].clone()) {
            Ok(ids) => Some(ids),
            Err(error) => return openai_error(StatusCode::BAD_REQUEST, error.to_string(), false),
        };
    }
    sampling.n = n as i64;
    native_body.sampling_params = Some(SamplingParamsInput::One(Box::new(sampling)));
    native_body.stream = stream;
    let native_requests = match native_body.into_requests() {
        Ok((requests, _)) => requests,
        Err(error) => return openai_error(StatusCode::BAD_REQUEST, error.to_string(), false),
    };
    let response_id = native_requests[0].rid.client_facing().to_owned();
    let created = unix_seconds_u32();
    let mut guard = AbortGuard::new_empty(state.senders.clone());
    let mut submitted = Vec::with_capacity(choice_count);

    let mut echoes = vec![String::new(); prompts.len()];
    for (index, mut native) in native_requests.into_iter().enumerate() {
        let prompt_index = index / n;
        let rid = native.rid.clone();
        if echo && index % n == 0 {
            echoes[prompt_index] = match &prompts[prompt_index] {
                PromptSpec::Text(text) => text.clone(),
                PromptSpec::TokenIds(ids) => match decode_prompt_echo(&state, ids.clone()).await {
                    Ok(echo) => echo,
                    Err(response) => return response,
                },
            };
        }
        native.started = Some(started);
        native.custom_labels = custom_labels.clone();
        native.return_logprob = request.logprobs.is_some();
        native.logprob_start_len = if echo && request.logprobs.is_some() {
            0
        } else {
            -1
        };
        native.top_logprobs_num = request.logprobs.unwrap_or(0) as i64;
        native.return_text_in_logprobs = Some(true);
        let rx = match submit_generation(&state, native, stream, &mut guard).await {
            Ok(rx) => rx,
            Err(response) => return response,
        };
        submitted.push(SubmittedChoice {
            index,
            prompt_index,
            rid,
            echo: echoes[prompt_index].clone(),
            rx,
        });
    }

    if stream {
        if let Err((index, status, message)) =
            super::prime_stream(submitted.iter_mut().map(|choice| &mut choice.rx)).await
        {
            guard.disarm(&submitted[index].rid);
            return openai_error(status, message, false);
        }
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
            output_options,
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
            echo,
            request.logprobs.is_some(),
            output_options,
        )
        .await
    }
}

/// Decode a token-id prompt back to text for `echo=true`, via a
/// `RequestKind::Detokenize` request through the regular submit path — the
/// detok stage answers it with a single `Data` payload (the raw UTF-8 text),
/// or an `Error` (e.g. out-of-range ids → `Validation` → 400).
async fn decode_prompt_echo(state: &AppState, token_ids: TokenIds) -> Result<String, Response> {
    let Ok((_rid, mut rx)) = submit(
        state,
        RequestKind::Detokenize {
            token_ids,
            skip_special_tokens: true,
        },
        false,
    )
    .await
    else {
        // Same rule as `submit_generation`: rebuild the refusal in the OpenAI
        // error shape rather than forwarding the native-shaped response.
        return Err(openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "service unavailable",
            false,
        ));
    };
    match rx.recv().await {
        Some(ResponseItem::Data(payload)) => String::from_utf8(payload.to_vec()).map_err(|_| {
            openai_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "detokenized prompt is not valid UTF-8",
                false,
            )
        }),
        Some(ResponseItem::Error(Error::Validation(message))) => {
            Err(openai_error(StatusCode::BAD_REQUEST, &message, false))
        }
        Some(ResponseItem::Error(error)) => {
            let status = StatusCode::from_u16(error.http_status())
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            Err(openai_error(
                status,
                format!("failed to decode prompt for echo: {error}"),
                false,
            ))
        }
        Some(_) | None => Err(openai_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "failed to decode prompt for echo: reply channel closed",
            false,
        )),
    }
}

fn completion_prompt_specs(prompt: &Prompt) -> Result<Vec<PromptSpec>, String> {
    match prompt {
        Prompt::String(text) => {
            if text.is_empty() {
                return Err("Prompt cannot be empty".into());
            }
            Ok(vec![PromptSpec::Text(text.clone())])
        }
        Prompt::StringArray(texts) => {
            if texts.is_empty() || texts.iter().any(String::is_empty) {
                return Err("Prompt cannot be empty".into());
            }
            Ok(texts.iter().cloned().map(PromptSpec::Text).collect())
        }
        Prompt::IntegerArray(ids) => Ok(vec![token_prompt_spec(ids)?]),
        Prompt::ArrayOfIntegerArray(prompts) => {
            if prompts.is_empty() {
                return Err("Prompt cannot be empty".into());
            }
            prompts.iter().map(|ids| token_prompt_spec(ids)).collect()
        }
    }
}

fn token_prompt_spec(ids: &[u32]) -> Result<PromptSpec, String> {
    if ids.is_empty() {
        return Err("Prompt cannot be empty".into());
    }
    let input_ids = ids
        .iter()
        .map(|&id| i32::try_from(id).map_err(|_| format!("Token ID {id} is out of range")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PromptSpec::TokenIds(input_ids))
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
        // OpenAI already expanded the choices before native admission.
        n: 1,
        logit_bias: (!logit_bias.is_empty()).then_some(logit_bias),
        sampling_seed: request.seed,
        ..Default::default()
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn unary_completion(
    submitted: Vec<SubmittedChoice>,
    mut guard: AbortGuard,
    response_id: String,
    model: String,
    created: u32,
    echo: bool,
    want_logprobs: bool,
    output_options: super::extensions::OutputOptions,
) -> Response {
    // Every request is already submitted, so draining in choice order does not
    // serialize generation. The non-streaming native path sends one terminal
    // result, and the accumulator also tolerates intermediate frames.
    let mut choices = Vec::with_capacity(submitted.len());
    let mut extensions = Vec::with_capacity(submitted.len());
    let n = submitted
        .iter()
        .take_while(|c| c.prompt_index == 0)
        .count()
        .max(1);
    let mut outputs = Vec::with_capacity(submitted.len());

    for choice in submitted {
        let output = match collect_output(choice.rx, &mut guard, &choice.rid).await {
            Ok(output) => output,
            Err((status, message)) => {
                return openai_error(status, &message, false);
            }
        };

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
        outputs.push(super::super::frame::frame_value(
            &output,
            choice.rid.client_facing(),
        ));
        choices.push(response_choice);
        extensions.push(extension);
    }

    let mut value = completion_response_value(
        CreateCompletionResponse {
            id: response_id,
            choices,
            created,
            model,
            system_fingerprint: None,
            object: "text_completion".into(),
            usage: None,
        },
        &extensions,
    );
    output_options.unary_fields(&mut value, &outputs, n, false);
    Json(value).into_response()
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
            // multi-token list. Preserve the original token IDs.
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
    output_options: super::extensions::OutputOptions,
) -> impl futures::Stream<Item = String> {
    async_stream::stream! {
        let mut last_response_id = response_id;
        let count = submitted.len();
        let n = submitted.iter().take_while(|choice| choice.prompt_index == 0).count().max(1);
        let mut snapshots: Vec<super::responses::StreamChoice> = (0..count).map(|_| Default::default()).collect();
        let mut rids = Vec::with_capacity(count);
        let mut echoes = Vec::with_capacity(count);
        let mut first_chunks = vec![true; count];
        let mut streams = Vec::with_capacity(count);

        for choice in submitted {
            let index = choice.index;
            rids.push(choice.rid);
            echoes.push(choice.echo);
            streams.push(indexed_decode_stream(index, choice.rx));
        }
        let mut events = futures::stream::select_all(streams);

        while let Some((index, item)) = events.next().await {
            let Some(item) = item else {
                yield error_payload(StatusCode::INTERNAL_SERVER_ERROR, "response truncated before completion").to_string();
                continue;
            };
            let output = match item {
                ResponseItem::Frame(output) => output,
                ResponseItem::Done(output) => {
                    guard.disarm(&rids[index]);
                    output
                }
                ResponseItem::Error(error) => {
                    guard.disarm(&rids[index]);
                    yield error_payload(StatusCode::from_u16(error.http_status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), error.to_string()).to_string();
                    continue;
                }
                ResponseItem::Control(_) | ResponseItem::Data(_) | ResponseItem::Tokenized(_) => continue,
            };

            if let Some((code, message)) = output
                .finish_reason
                .as_ref()
                .and_then(|reason| reason.abort_status())
            {
                yield error_payload(StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), message).to_string();
                continue;
            }

            snapshots[index].observe(&output, false);
            last_response_id = output.rid.client_facing().into();
            let first = std::mem::replace(&mut first_chunks[index], false);
            let text = if echo && first {
                echoes[index].clone() + &output.text
            } else {
                output.text.clone()
            };
            let (choice, extension) = completion_choice(
                index,
                text,
                &output,
                want_logprobs,
                echo && first,
            );
            let chunk = CreateCompletionResponse {
                id: output.rid.client_facing().into(),
                choices: vec![choice],
                created,
                model: model.clone(),
                system_fingerprint: None,
                object: "text_completion".into(),
                usage: None,
            };
            let mut value = completion_response_value(chunk, &[extension]);
            value["choices"][0].as_object_mut().expect("OpenAI choice must be an object").entry("logprobs").or_insert(serde_json::Value::Null);
            if output_options.return_token_ids {
                value["choices"][0]["token_ids"] = serde_json::json!(output.token_ids);
                if first && let Some(ids) = output.extras.as_ref().and_then(|e| e.prompt_token_ids.as_ref()) {
                    value["choices"][0]["prompt_token_ids"] = serde_json::json!(ids.as_ref());
                }
            }
            if continuous_usage {
                let mut options = output_options.clone();
                options.enable_cache_report = false;
                value["usage"] = options.usage(&[snapshots[index].metadata_item()], 1, false);
            } else { value["usage"] = serde_json::Value::Null; }
            yield value.to_string();
        }

        let items: Vec<_> = snapshots.into_iter().map(|snapshot| snapshot.into_item()).collect();
        for event in output_options.stream_tail(&items, n, false, &last_response_id, &model, created, include_usage) {
            yield event.data;
        }
        yield "[DONE]".to_string();
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

#[cfg(test)]
mod tests {
    use super::super::test_utils::{chunk, senders, submitted};
    use super::{
        ChoiceExtensions, PromptSpec, completion_event_stream, completion_logprobs,
        completion_prompt_specs, completion_response_value, unary_completion,
    };
    use crate::api_server::guard::AbortGuard;
    use crate::message::response::ChunkExtras;
    use axum::http::StatusCode;
    use dynamo_protocols::types::{
        Choice, CreateCompletionRequest, CreateCompletionResponse, Prompt,
    };
    use futures::StreamExt;

    #[test]
    fn dynamo_completion_request_deserializes_directly() {
        let request: CreateCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "prompt": ["a", "b"],
            "max_tokens": 8,
            "n": 2,
            "stream_options": {
                "include_usage": true,
                "continuous_usage_stats": true
            }
        }))
        .unwrap();
        assert!(matches!(request.prompt, Prompt::StringArray(_)));
        assert_eq!(request.n, Some(2));
        assert!(request.stream_options.unwrap().continuous_usage_stats);
    }

    #[test]
    fn token_prompt_is_normalized_without_echo_state() {
        let specs = completion_prompt_specs(&Prompt::IntegerArray(vec![1, 2])).unwrap();
        assert_eq!(specs, [PromptSpec::TokenIds(vec![1, 2])]);
    }

    #[test]
    fn zero_top_logprobs_keeps_selected_token_and_empty_top_map() {
        let extras = ChunkExtras {
            out_lp_val: vec![-0.25],
            out_lp_idx: vec![7],
            out_lp_txt: vec!["x".into()],
            out_top_lens: vec![0],
            ..Default::default()
        };
        let logprobs = completion_logprobs(Some(&extras), false);
        assert_eq!(logprobs.tokens, ["x"]);
        assert_eq!(logprobs.token_logprobs, [Some(-0.25)]);
        assert_eq!(logprobs.top_logprobs, [serde_json::Value::Null]);

        let value = completion_response_value(
            CreateCompletionResponse {
                id: "cmpl-test".into(),
                choices: vec![Choice {
                    text: "x".into(),
                    index: 0,
                    logprobs: Some(logprobs),
                    finish_reason: None,
                }],
                created: 1,
                model: "model".into(),
                system_fingerprint: None,
                object: "text_completion".into(),
                usage: None,
            },
            &[ChoiceExtensions::default()],
        );
        assert_eq!(
            value["choices"][0]["logprobs"]["text_offset"],
            serde_json::json!([-1])
        );
    }

    #[tokio::test]
    async fn unary_fold_orders_choices_and_counts_each_prompt_once() {
        let (choice0, tx0) = submitted(0, 0, "r0");
        let (choice1, tx1) = submitted(1, 0, "r1");
        tx0.send(chunk("r0", "a", false)).await.unwrap();
        tx0.send(chunk("r0", "b", true)).await.unwrap();
        tx1.send(chunk("r1", "x", false)).await.unwrap();
        tx1.send(chunk("r1", "y", true)).await.unwrap();

        let response = unary_completion(
            vec![choice0, choice1],
            AbortGuard::new_empty(senders()),
            "cmpl-test".into(),
            "model".into(),
            1,
            false,
            false,
            Default::default(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["choices"][0]["text"], "ab");
        assert_eq!(value["choices"][1]["text"], "xy");
        assert_eq!(value["choices"][0]["matched_stop"], "</s>");
        assert_eq!(value["usage"]["prompt_tokens"], 5);
        assert_eq!(value["usage"]["completion_tokens"], 4);
    }

    #[tokio::test]
    async fn stream_uses_deltas_then_usage_and_done() {
        let (choice, tx) = submitted(0, 0, "r0");
        tx.send(chunk("r0", "a", false)).await.unwrap();
        tx.send(chunk("r0", "b", true)).await.unwrap();

        let stream = completion_event_stream(
            vec![choice],
            AbortGuard::new_empty(senders()),
            "cmpl-test".into(),
            "model".into(),
            1,
            false,
            false,
            true,
            false,
            Default::default(),
        );
        futures::pin_mut!(stream);
        let frames: Vec<String> = stream.collect().await;
        assert_eq!(frames.len(), 4);
        let first: serde_json::Value = serde_json::from_str(&frames[0]).unwrap();
        let terminal: serde_json::Value = serde_json::from_str(&frames[1]).unwrap();
        let usage: serde_json::Value = serde_json::from_str(&frames[2]).unwrap();
        assert_eq!(first["choices"][0]["text"], "a");
        assert_eq!(terminal["choices"][0]["text"], "b");
        assert_eq!(terminal["choices"][0]["finish_reason"], "stop");
        assert!(usage["choices"].as_array().unwrap().is_empty());
        assert_eq!(usage["usage"]["prompt_tokens"], 5);
        assert_eq!(usage["usage"]["completion_tokens"], 2);
        assert_eq!(frames[3], "[DONE]");
    }
}
