//! OpenAI legacy text-completion endpoint and wire shaping.

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::sync::Arc;

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
    CreateCompletionResponse, Logprobs,
};
use futures::StreamExt;
use tokio::sync::mpsc;

use super::super::submit::submit;
use super::{
    AppState, collect_output, error_payload, indexed_decode_stream, openai_error,
    submit_generation, unix_seconds_u32,
};
use crate::frontend::AbortGuard;
use crate::message::finish_reason::Matched;
use crate::message::ids::Rid;
use crate::message::request::{GenerateRequest, RequestKind};
use crate::message::response::{ChunkEvent, ChunkExtras, ResponseItem};
use crate::message::types::TokenIds;
use crate::renderer::render_http_status;
use crate::utils::error::Error;

pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new().route("/v1/completions", post(completions))
}

pub(super) struct SubmittedChoice {
    pub(super) index: usize,
    pub(super) prompt_index: usize,
    pub(super) rid: Rid,
    pub(super) echo: String,
    pub(super) rx: mpsc::Receiver<ResponseItem>,
}
#[derive(Debug, Default)]
pub(super) struct ChoiceExtensions {
    matched_stop: Option<serde_json::Value>,
    /// Dynamo's enum covers the standard values. Python additionally exposes
    /// `abort`, and native unknown finish types are preserved rather than lost.
    finish_reason_override: Option<String>,
}

async fn completions(
    State(state): State<Arc<AppState>>,
    body: Result<Json<CreateCompletionRequest>, JsonRejection>,
) -> Response {
    let request = match body {
        Ok(Json(request)) => request,
        Err(rejection) => {
            return openai_error(StatusCode::BAD_REQUEST, rejection.body_text(), false);
        }
    };
    let response_id = format!("cmpl-{}", uuid::Uuid::new_v4().simple());
    let completion_requests = match state.lowerer.lower_completions(&request, &response_id) {
        Ok(requests) => requests,
        Err(error) => {
            let status = StatusCode::from_u16(render_http_status(&error))
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            return openai_error(status, error.to_string(), false);
        }
    };
    let stream = request.stream.unwrap_or(false);
    let echo = request.echo.unwrap_or(false);
    let model = request.model.clone();
    let created = unix_seconds_u32();
    let mut guard = state.frontend.empty_abort_guard();
    let mut submitted = Vec::with_capacity(completion_requests.len());
    let n = request.n.unwrap_or(1) as usize;
    let mut prompt_echo = String::new();

    for (index, completion_request) in completion_requests.into_iter().enumerate() {
        let native = GenerateRequest::from(completion_request);
        let prompt_index = index / n;
        let sample_index = index % n;
        if sample_index == 0 {
            prompt_echo = if !echo {
                String::new()
            } else if let Some(text) = &native.text {
                text.clone()
            } else if let Some(token_ids) = &native.input_ids {
                match decode_prompt_echo(&state, token_ids.clone()).await {
                    Ok(echo) => echo,
                    Err(response) => return response,
                }
            } else {
                unreachable!("lowered completion request has a prompt")
            };
        }
        let rid = native.rid.clone();
        let rx = match submit_generation(&state, native, stream, &mut guard).await {
            Ok(rx) => rx,
            Err(response) => return response,
        };
        submitted.push(SubmittedChoice {
            index,
            prompt_index,
            rid,
            echo: prompt_echo.clone(),
            rx,
        });
    }

    if stream {
        let include_usage = request
            .stream_options
            .map(|o| o.include_usage)
            .unwrap_or(false)
            || state.lowerer.config().stream_response_default_include_usage;
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

/// Decode a token-id prompt back to text for `echo=true`, via a
/// `RequestKind::Detokenize` request through the regular submit path — the
/// detok stage answers it with a single `Data` payload (the raw UTF-8 text),
/// or an `Error` (e.g. out-of-range ids → `Validation` → 400).
async fn decode_prompt_echo(state: &AppState, token_ids: TokenIds) -> Result<String, Response> {
    let Ok((_rid, mut rx)) = submit(state, RequestKind::Detokenize { token_ids }, false).await
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
            Err((status, message)) => {
                return openai_error(status, &message, false);
            }
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
                ResponseItem::Control(_) | ResponseItem::Data(_) => continue,
            };

            if let Some((code, message)) = output
                .finish_reason
                .as_ref()
                .and_then(|reason| reason.abort_status())
            {
                yield error_payload(StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), message).to_string();
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

#[cfg(test)]
mod tests {
    use super::super::test_utils::{chunk, senders, submitted};
    use super::{
        ChoiceExtensions, completion_event_stream, completion_logprobs, completion_response_value,
        unary_completion,
    };
    use crate::frontend::AbortGuard;
    use crate::message::response::ChunkExtras;
    use axum::http::StatusCode;
    use dynamo_protocols::types::{
        Choice, CreateCompletionRequest, CreateCompletionResponse, Prompt,
    };
    use futures::StreamExt;
    use sglang_renderer::openai::{PromptSpec, completion_prompt_specs};

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
            AbortGuard::new_empty(senders().abort_tx),
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
            AbortGuard::new_empty(senders().abort_tx),
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
