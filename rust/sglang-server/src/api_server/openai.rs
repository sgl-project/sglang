//! OpenAI-compatible generation endpoints.
//!
//! The HTTP adapter stays deliberately thin: Dynamo owns the standard OpenAI
//! request and response primitives. Scheduler [`ChunkEvent`] values remain the one
//! backend output type for both unary and streaming responses.

use axum::{Router, http::StatusCode, response::Response};
use futures::StreamExt;
use std::sync::Arc;
use tokio::sync::mpsc;

mod chat;
mod completions;
mod extensions;
mod models;
mod reasoning;
mod responses;
mod template;
mod template_builtins;
mod template_legacy;
mod template_loader;
mod tokenize;
mod tools;

pub(super) use template::{ChatFormatter, ChatTemplateKwargs};

use super::app::AppState;
use super::frame::OutputAccumulator;
use super::guard::AbortGuard;
use super::submit::submit;
use crate::message::config::ServerArgs;
use crate::message::ids::Rid;
use crate::message::request::{GenerateRequest, RequestKind};
use crate::message::response::{ChunkEvent, ResponseItem};
use crate::tokenizer_manager::tokenizer;
use crate::utils::response::error_response;

const MAX_OPENAI_CHOICES: usize = 4096;

/// The routes this module owns, mounted by `api_server::serve`.
pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .merge(models::routes())
        .merge(completions::routes())
        .merge(chat::routes())
        .merge(tokenize::routes())
}

/// Resolve the chat formatter, or `None` to disable the OpenAI chat-completions
/// endpoint. Tokenization is the tokenizer pool's job (the api server never
/// encodes); the formatter needs at most `tokenizer_config.json` — a built-in
/// `--chat-template` name or a model-path-inferred legacy template resolve
/// without it, so its absence must not disable chat.
pub(super) fn load_chat_support(server_args: &ServerArgs) -> Option<ChatFormatter> {
    // Chat needs the tokenizer pool behind it: under `skip_tokenizer_init`
    // there is none (text cannot be submitted), so chat is disabled.
    if server_args.skip_tokenizer_init || server_args.tokenizer_path.is_empty() {
        return None;
    }
    let config_file = tokenizer::resolve_model_file(
        &server_args.tokenizer_path,
        server_args.revision.as_deref(),
        "tokenizer_config.json",
    );

    match template::load_chat_formatter(
        config_file.as_deref(),
        (!server_args.model_path.is_empty()).then_some(server_args.model_path.as_str()),
        server_args.model_config.model_type.as_deref(),
        server_args.chat_template.as_deref(),
    ) {
        Ok(formatter) => {
            tracing::info!(
                config = ?config_file.as_deref().unwrap_or("<built-in / inferred>"),
                "loaded OpenAI chat template"
            );
            Some(formatter)
        }
        Err(error) => {
            tracing::warn!(%error, "OpenAI chat completions disabled");
            None
        }
    }
}

fn unix_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn unix_seconds_u32() -> u32 {
    u32::try_from(unix_seconds()).unwrap_or(u32::MAX)
}

/// The OpenAI error payload.
pub(super) fn error_payload(code: StatusCode, message: impl Into<String>) -> serde_json::Value {
    let message = message.into();
    let error_type = if code == StatusCode::UNAUTHORIZED {
        "AuthenticationError"
    } else if code.is_server_error() {
        "InternalServerError"
    } else {
        "BadRequestError"
    };
    serde_json::json!({
        "error": {
            "object": "error",
            "message": message,
            "type": error_type,
            "param": null,
            "code": code.as_u16(),
        }
    })
}

/// Form an OpenAI error response: unary → `code` plus the JSON `body`,
/// streaming → 200 with one SSE error frame + `[DONE]`.
pub(super) fn openai_error(code: StatusCode, message: impl Into<String>, stream: bool) -> Response {
    let mut payload = error_payload(code, message);
    // Python's legacy completion endpoints use a flat unary ErrorResponse;
    // committed SSE errors keep the enclosing `error` field.
    if !stream {
        payload = payload["error"].take();
    }
    error_response(code, payload, stream)
}

pub(super) struct ResponseReceiver {
    rx: mpsc::Receiver<ResponseItem>,
    first: Option<ResponseItem>,
}

impl From<mpsc::Receiver<ResponseItem>> for ResponseReceiver {
    fn from(rx: mpsc::Receiver<ResponseItem>) -> Self {
        Self { rx, first: None }
    }
}

impl ResponseReceiver {
    async fn recv(&mut self) -> Option<ResponseItem> {
        if self.first.is_some() {
            self.first.take()
        } else {
            self.rx.recv().await
        }
    }

    async fn prime(&mut self) -> Result<(), (StatusCode, String)> {
        loop {
            match self.rx.recv().await {
                Some(ResponseItem::Error(error)) => {
                    return Err((
                        StatusCode::from_u16(error.http_status())
                            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                        error.to_string(),
                    ));
                }
                Some(
                    ResponseItem::Control(_) | ResponseItem::Data(_) | ResponseItem::Tokenized(_),
                ) => {}
                item => {
                    self.first = item;
                    return Ok(());
                }
            }
        }
    }
}

/// Python advances the generator once before committing HTTP 200. Prime the
/// first ready choice without serializing parallel generation or losing a frame.
async fn prime_stream<'a>(
    receivers: impl Iterator<Item = &'a mut ResponseReceiver>,
) -> Result<(), (usize, StatusCode, String)> {
    let pending: Vec<_> = receivers.map(|rx| Box::pin(rx.prime())).collect();
    let (result, index, _) = futures::future::select_all(pending).await;
    result.map_err(|(status, message)| (index, status, message))
}

/// Drain one submitted request to its terminal output: fold frames, disarm
/// `guard` on a natural terminal, and map errors / validation aborts /
/// truncation to `(status, message)` for the OpenAI error shape.
async fn collect_output(
    mut rx: ResponseReceiver,
    guard: &mut AbortGuard,
    rid: &Rid,
) -> Result<ChunkEvent, (StatusCode, String)> {
    let mut accumulator = OutputAccumulator::default();
    let output = loop {
        match rx.recv().await {
            Some(ResponseItem::Frame(output)) => accumulator.fold(&output),
            Some(ResponseItem::Done(output)) => {
                accumulator.fold(&output);
                break accumulator.into_output();
            }
            Some(ResponseItem::Error(error)) => {
                guard.disarm(rid);
                let status = StatusCode::from_u16(error.http_status())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                return Err((status, error.to_string()));
            }
            Some(ResponseItem::Control(_))
            | Some(ResponseItem::Data(_))
            | Some(ResponseItem::Tokenized(_)) => {}
            None => {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "response truncated before completion".into(),
                ));
            }
        }
    };
    guard.disarm(rid);
    if let Some((code, message)) = output
        .finish_reason
        .as_ref()
        .and_then(|reason| reason.abort_status())
    {
        return Err((
            StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            message.to_owned(),
        ));
    }
    Ok(output)
}

async fn submit_generation(
    state: &AppState,
    mut request: GenerateRequest,
    stream: bool,
    guard: &mut AbortGuard,
) -> Result<ResponseReceiver, Response> {
    if let Some(metadata) = state
        .http_extension
        .as_ref()
        .and_then(|extension| extension.initial_response_metadata())
    {
        request
            .response_metadata
            .get_or_insert_default()
            .extend(metadata);
    }
    let limits = &state.server_args.limit_mm_data_per_request;
    if let Err(error) = super::prefetch::validate_limits(std::slice::from_ref(&request), limits) {
        return Err(openai_error(StatusCode::BAD_REQUEST, error, false));
    }
    if let Some(extension) = &state.http_extension
        && let Err(error) = extension.prepare_request(&mut request).await
    {
        return Err(openai_error(StatusCode::BAD_REQUEST, error, false));
    }
    if let Err(error) = super::prefetch::prefetch_all(
        std::slice::from_mut(&mut request),
        limits,
        state.http_extension.as_ref(),
    )
    .await
    {
        return Err(openai_error(StatusCode::BAD_REQUEST, error, false));
    }
    match submit(state, RequestKind::Generate(Box::new(request)), stream).await {
        Ok((rid, rx)) => {
            guard.arm(rid);
            Ok(rx.into())
        }
        // Same `error_response` rule: a committed stream gets 200 plus an
        // SSE error frame + `[DONE]`, not a unary 503 — but with the OpenAI
        // error shape, since this is the OpenAI frontend.
        Err(_) => Err(openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "service unavailable",
            stream,
        )),
    }
}

fn indexed_decode_stream(
    index: usize,
    rx: ResponseReceiver,
) -> futures::stream::BoxStream<'static, (usize, Option<ResponseItem>)> {
    futures::stream::unfold((rx, false), move |(mut rx, finished)| async move {
        if finished {
            return None;
        }
        match rx.recv().await {
            Some(item) => {
                let finished = matches!(item, ResponseItem::Done(_) | ResponseItem::Error(_));
                Some(((index, Some(item)), (rx, finished)))
            }
            None => Some(((index, None), (rx, true))),
        }
    })
    .boxed()
}

fn contains_media(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Array(values) => values.iter().any(contains_media),
        serde_json::Value::Object(object) => {
            object.keys().any(|key| {
                matches!(
                    key.as_str(),
                    "image_url" | "video_url" | "input_audio" | "audio_url" | "file"
                )
            }) || object.values().any(contains_media)
        }
        _ => false,
    }
}

#[cfg(test)]
mod test_utils;
