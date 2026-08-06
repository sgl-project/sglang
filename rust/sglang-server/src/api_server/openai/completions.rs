//! OpenAI legacy text-completion endpoint and wire shaping.

use std::collections::BTreeMap;
use std::convert::Infallible;

use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::post,
};
use dynamo_protocols::types::{
    Choice, CompletionFinishReason, CompletionUsage, CreateCompletionRequest,
    CreateCompletionResponse, Logprobs, Prompt,
};
use futures::StreamExt;

use super::super::guard::AbortGuard;
use super::super::submit::submit_openai;
use super::{
    AppState, ChoiceStep, MAX_OPENAI_CHOICES, OpenAiChoice, choice_step_stream, collect_choices,
    error_payload, invalid_request, native_stops, normalize_sampling, parse_logit_bias,
    require_served_model, stream_include_usage, unix_seconds_u32,
};
use crate::ids::Rid;
use crate::message::{
    ChunkEvent, ChunkExtras, EgressItem, GenerateRequest, Matched, SamplingParams, TokenIds,
};
use crate::utils::response::error_response;

pub(super) fn routes() -> Router<AppState> {
    Router::new().route("/v1/completions", post(completions))
}

#[derive(Debug, PartialEq, Eq)]
enum PromptSpec {
    Text(String),
    TokenIds(TokenIds),
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
    body: Result<Json<CreateCompletionRequest>, JsonRejection>,
) -> Response {
    let request = match body {
        Ok(Json(request)) => request,
        Err(rejection) => {
            return invalid_request(rejection.body_text());
        }
    };
    let stream = request.stream.unwrap_or(false);
    let echo = request.echo.unwrap_or(false);
    let model = request.model.clone();
    if let Some(response) = require_served_model(&model, &state.server_args) {
        return response;
    }

    if request.prompt_embeds.is_some() {
        return invalid_request("prompt_embeds is not supported by the Rust frontend");
    }
    if request.suffix.is_some() {
        return invalid_request("suffix is not supported by this model");
    }
    if request.best_of.is_some_and(|best_of| best_of != 1) {
        return invalid_request("best_of values greater than 1 are not supported");
    }
    if request.max_tokens == Some(0) {
        return invalid_request("max_tokens must be positive");
    }
    if request.n == Some(0) {
        return invalid_request("n must be at least 1");
    }
    let prompts = match completion_prompt_specs(&request.prompt) {
        Ok(prompts) => prompts,
        Err(message) => {
            return invalid_request(message);
        }
    };
    let mut sampling = match completion_sampling_params(&request) {
        Ok(sampling) => sampling,
        Err(message) => {
            return invalid_request(message);
        }
    };
    if let Err(error) = normalize_sampling(&mut sampling, &state.server_args) {
        return invalid_request(error);
    }

    let n = request.n.unwrap_or(1) as usize;
    let choice_count = match prompts.len().checked_mul(n) {
        Some(count) if count <= MAX_OPENAI_CHOICES => count,
        _ => {
            return invalid_request(format!(
                "prompt count times n exceeds the maximum of {MAX_OPENAI_CHOICES}"
            ));
        }
    };
    let response_id = format!("cmpl-{}", uuid::Uuid::new_v4().simple());
    let created = unix_seconds_u32();
    let mut guard = AbortGuard::new_empty(state.senders.clone());
    let mut submitted = Vec::with_capacity(choice_count);
    // Choices whose echo arrives as a `Data` prologue on their own sink.
    let mut pending_echo = Vec::new();

    for (prompt_index, prompt) in prompts.into_iter().enumerate() {
        let (text, input_ids, prompt_echo) = match prompt {
            PromptSpec::Text(text) => {
                let prompt_echo = if echo { text.clone() } else { String::new() };
                (Some(text), None, prompt_echo)
            }
            PromptSpec::TokenIds(input_ids) => (None, Some(input_ids), String::new()),
        };
        // A token-id prompt's echo text rides the generation itself: the shard
        // decodes it while the GPU prefills and delivers it as the sink's
        // first item (see `GenerateRequest::return_prompt_text`).
        let echo_from_sink = echo && input_ids.is_some();
        for sample_index in 0..n {
            let index = prompt_index * n + sample_index;
            let rid = Rid::from_client(&format!("{response_id}-{index}"));
            let native = GenerateRequest {
                rid: rid.clone(),
                text: text.clone(),
                input_ids: input_ids.clone(),
                sampling_params: sampling.clone(),
                stream,
                return_logprob: request.logprobs.is_some(),
                logprob_start_len: if echo && request.logprobs.is_some() {
                    0
                } else {
                    -1
                },
                top_logprobs_num: request.logprobs.unwrap_or(0) as i64,
                return_text_in_logprobs: request.logprobs.map(|_| true),
                return_prompt_text: echo_from_sink,
                ..Default::default()
            };
            let rx = match submit_openai(&state, native, stream, &mut guard).await {
                Ok(rx) => rx,
                Err(response) => return response,
            };
            submitted.push(OpenAiChoice {
                index,
                prompt_index,
                rid,
                echo: prompt_echo.clone(),
                rx,
            });
            if echo_from_sink {
                pending_echo.push(submitted.len() - 1);
            }
        }
    }

    // Resolve the sink-delivered echoes AFTER every submit, so the decodes
    // overlapped with prefill. `push_to_ring` dispatched each decode before its
    // ring push, so the `Data` prologue is guaranteed to be the first item on
    // the receiver — the generic drains below never see one. An early return
    // here drops `guard`, aborting every in-flight generation.
    for &slot in &pending_echo {
        let choice = &mut submitted[slot];
        match choice.rx.recv().await {
            Some(EgressItem::Data(payload)) => match String::from_utf8(payload.to_vec()) {
                Ok(text) => choice.echo = text,
                Err(_) => {
                    return error_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        error_payload(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "detokenized prompt is not valid UTF-8",
                        ),
                        false,
                    );
                }
            },
            Some(EgressItem::Error(crate::error::Error::Validation(message))) => {
                return invalid_request(message);
            }
            Some(EgressItem::Error(error)) => {
                let status = StatusCode::from_u16(error.http_status())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                return error_response(
                    status,
                    error_payload(status, format!("failed to decode prompt for echo: {error}")),
                    false,
                );
            }
            _ => {
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    error_payload(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "failed to decode prompt for echo: no prologue before generation output",
                    ),
                    false,
                );
            }
        }
    }

    if stream {
        let include_usage =
            stream_include_usage(request.stream_options.as_ref(), &state.server_args);
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
        )
        .await
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
    let (stop, stop_token_ids) = native_stops(request.stop.as_ref());
    let logit_bias = parse_logit_bias(request.logit_bias.as_ref())?;

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
        logit_bias,
        sampling_seed: request.seed,
        ..Default::default()
    })
}

pub(super) async fn unary_completion(
    submitted: Vec<OpenAiChoice>,
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
    let (outputs, prompt_tokens, completion_tokens) =
        match collect_choices(submitted, &mut guard).await {
            Ok(collected) => collected,
            Err(response) => return response,
        };
    let mut choices = Vec::with_capacity(outputs.len());
    let mut extensions = Vec::with_capacity(outputs.len());

    for (index, prompt_echo, output) in outputs {
        let (response_choice, extension) = completion_choice(
            index,
            if echo {
                prompt_echo + &output.text
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
    submitted: Vec<OpenAiChoice>,
    guard: AbortGuard,
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
        let mut echoes = vec![String::new(); count];
        for choice in &submitted {
            echoes[choice.index] = choice.echo.clone();
        }
        // Per-choice running total, only for `continuous_usage_stats` chunks;
        // the final usage chunk uses the driver's totals.
        let mut completion_tokens_by_choice = vec![0u64; count];

        let steps = choice_step_stream(submitted, guard);
        futures::pin_mut!(steps);
        while let Some(step) = steps.next().await {
            match step {
                ChoiceStep::Error(payload) => yield payload,
                ChoiceStep::Usage { prompt_tokens, completion_tokens } => {
                    if include_usage {
                        let final_chunk = CreateCompletionResponse {
                            id: response_id.clone(),
                            choices: vec![],
                            created,
                            model: model.clone(),
                            system_fingerprint: None,
                            object: "text_completion".into(),
                            usage: Some(completion_usage(
                                prompt_tokens,
                                u32::try_from(completion_tokens).unwrap_or(u32::MAX),
                            )),
                        };
                        yield completion_response_value(final_chunk, &[]).to_string();
                    }
                }
                ChoiceStep::Output { index, output, first } => {
                    completion_tokens_by_choice[index] = completion_tokens_by_choice[index]
                        .saturating_add(output.completion_tokens);
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
                    let (choice, extension) =
                        completion_choice(index, text, &output, want_logprobs, echo && first);
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
            }
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

#[cfg(test)]
mod tests {
    use super::super::test_utils::{chunk, senders, submitted};
    use super::{
        ChoiceExtensions, PromptSpec, completion_event_stream, completion_logprobs,
        completion_prompt_specs, completion_response_value, unary_completion,
    };
    use crate::api_server::guard::AbortGuard;
    use crate::message::ChunkExtras;
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
    fn max_tokens_zero_is_rejected_before_submission() {
        let request: CreateCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "prompt": "hello",
            "max_tokens": 0
        }))
        .unwrap();
        assert_eq!(request.max_tokens, Some(0));
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
