//! OpenAI legacy text-completion endpoint and wire shaping.

use std::sync::Arc;

use dynamo_protocols::types::CreateCompletionRequest;
use futures::StreamExt;
use http::StatusCode;

use super::super::encode::sse_encode;
use super::super::plumbing::{HttpResponse, json_response, read_json};
use super::{AppState, MAX_OPENAI_CHOICES, openai_error, submit_generation, unix_seconds_u32};
use crate::api_server::core::guard::AbortGuard;
use crate::api_server::core::openai::completions::{
    PromptSpec, SubmittedChoice, completion_event_stream, completion_prompt_specs,
    completion_sampling_params, completion_sse_payload, decode_prompt_echo, unary_completion,
};
use crate::message::ids::Rid;
use crate::message::request::GenerateRequest;

pub(in crate::api_server) async fn completions<B: http_body::Body>(
    state: Arc<AppState>,
    req: http::Request<B>,
) -> HttpResponse {
    let request = match read_json::<CreateCompletionRequest, _>(req).await {
        Ok(request) => request,
        Err(rejection) => {
            return openai_error(StatusCode::BAD_REQUEST, rejection.body_text, false);
        }
    };
    let stream = request.stream.unwrap_or(false);
    let echo = request.echo.unwrap_or(false);
    let model = request.model.clone();
    if model != state.server_args.served_model_name {
        return openai_error(
            StatusCode::BAD_REQUEST,
            format!("The model `{model}` does not exist"),
            false,
        );
    }

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
    if request.max_tokens == Some(0) {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "max_tokens must be positive",
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
    let response_id = format!("cmpl-{}", uuid::Uuid::new_v4().simple());
    let created = unix_seconds_u32();
    let mut guard = AbortGuard::new_empty(state.senders.clone());
    let mut submitted = Vec::with_capacity(choice_count);

    for (prompt_index, prompt) in prompts.into_iter().enumerate() {
        let (text, input_ids, mut prompt_echo) = match prompt {
            PromptSpec::Text(text) => {
                let prompt_echo = if echo { text.clone() } else { String::new() };
                (Some(text), None, prompt_echo)
            }
            PromptSpec::TokenIds(input_ids) => (None, Some(input_ids), String::new()),
        };
        for sample_index in 0..n {
            let index = prompt_index * n + sample_index;
            let rid = Rid::from_client(&format!("{response_id}-{index}"));
            if echo
                && sample_index == 0
                && let Some(token_ids) = &input_ids
            {
                prompt_echo = match decode_prompt_echo(&state, token_ids.clone()).await {
                    Ok(echo) => echo,
                    Err(e) => return openai_error(e.http_status(), e.message, false),
                };
            }
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
                ..Default::default()
            };
            let rx = match submit_generation(&state, native, &mut guard).await {
                Ok(rx) => rx,
                Err(e) => return openai_error(e.http_status(), e.message, stream),
            };
            submitted.push(SubmittedChoice {
                index,
                prompt_index,
                rid,
                echo: prompt_echo.clone(),
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
        );
        sse_encode(s.map(completion_sse_payload))
    } else {
        match unary_completion(
            submitted,
            guard,
            response_id,
            model,
            created,
            echo,
            request.logprobs.is_some(),
        )
        .await
        {
            Ok(value) => json_response(StatusCode::OK, &value),
            Err(e) => openai_error(e.http_status(), e.message, false),
        }
    }
}
