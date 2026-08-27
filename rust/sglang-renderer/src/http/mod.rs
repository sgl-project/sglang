//! OpenAI HTTP frontends built on the engine-neutral renderer contracts.

use std::sync::Arc;

use axum::{
    Json, Router,
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
};
use futures::StreamExt;

use crate::{
    FrontendError, GenerationEvent, GenerationInput, GenerationOutput, GenerationSubmission,
    InferenceBackend, InferenceSession, OpenAIRequestLowerer, RendererErrorKind,
};

mod chat;
mod completions;
mod render;
#[cfg(test)]
mod test_utils;

pub struct OpenAIHttpFrontend<B> {
    pub(crate) lowerer: Arc<OpenAIRequestLowerer>,
    pub(crate) backend: B,
}

impl<B> OpenAIHttpFrontend<B> {
    pub fn new(lowerer: Arc<OpenAIRequestLowerer>, backend: B) -> Self {
        Self { lowerer, backend }
    }
}

pub fn inference_routes<B>(frontend: OpenAIHttpFrontend<B>) -> Router<()>
where
    B: InferenceBackend,
{
    Router::new()
        .merge(chat::routes())
        .merge(completions::routes())
        .with_state(Arc::new(frontend))
}

pub fn render_routes(renderer: Arc<crate::RendererService>) -> Router<()> {
    render::routes(renderer)
}

pub(crate) fn unix_seconds_u32() -> u32 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| u32::try_from(duration.as_secs()).unwrap_or(u32::MAX))
        .unwrap_or(0)
}

pub(crate) fn error_payload(code: StatusCode, message: impl Into<String>) -> serde_json::Value {
    let error_type = if code.is_server_error() {
        "InternalServerError"
    } else {
        "BadRequestError"
    };
    serde_json::json!({
        "error": {
            "object": "error", "message": message.into(), "type": error_type,
            "param": null, "code": code.as_u16(),
        }
    })
}

pub(crate) fn openai_error(code: StatusCode, message: impl Into<String>, stream: bool) -> Response {
    let payload = error_payload(code, message);
    if !stream {
        return (code, Json(payload)).into_response();
    }
    let frames = [payload.to_string(), "[DONE]".to_owned()];
    Sse::new(futures::stream::iter(frames.map(|data| {
        Ok::<_, std::convert::Infallible>(Event::default().data(data))
    })))
    .into_response()
}

pub(crate) fn renderer_status(kind: RendererErrorKind) -> StatusCode {
    match kind {
        RendererErrorKind::InvalidRequest => StatusCode::BAD_REQUEST,
        RendererErrorKind::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
        RendererErrorKind::Tokenize | RendererErrorKind::Internal => {
            StatusCode::INTERNAL_SERVER_ERROR
        }
    }
}

pub(crate) async fn submit_generation<S: InferenceSession>(
    session: &mut S,
    request: GenerationInput,
    stream: bool,
) -> Result<GenerationSubmission, Response> {
    session.submit(request, stream).await.map_err(|error| {
        openai_error(
            StatusCode::from_u16(error.status_code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            error.message,
            stream,
        )
    })
}

pub(crate) async fn collect_output<S: InferenceSession>(
    session: &mut S,
    mut submission: GenerationSubmission,
) -> Result<GenerationOutput, FrontendError> {
    let mut collected = GenerationOutput::default();
    while let Some(item) = submission.events.next().await {
        match item? {
            GenerationEvent::Frame(output) => fold_output(&mut collected, output),
            GenerationEvent::Done(output) => {
                fold_output(&mut collected, output);
                session.complete(&submission.id);
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
    use super::fold_output;
    use crate::{GenerationOutput, GenerationOutputExtras};

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
}
