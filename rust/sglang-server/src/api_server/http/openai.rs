//! OpenAI-compatible generation endpoints.
//!
//! The HTTP adapter stays deliberately thin: Dynamo owns the standard OpenAI
//! request and response primitives. [`CoreOutput`] remains the one backend
//! output type for both unary and streaming responses.

use axum::{
    Json, Router,
    body::Bytes,
    extract::{FromRequest, Request, rejection::JsonRejection},
    http::{HeaderMap, StatusCode},
    response::Response,
};
use futures::StreamExt;
use sglang_processor::ChatFormatterOptions;
use std::sync::Arc;

mod chat;
mod completions;
mod lowered;
mod models;
mod pd_routing;
mod template;
mod tokenize;
mod tools;

pub(super) use lowered::OpenAiChat;
pub(super) use template::{ChatFormatter, ChatTemplateKwargs};

use super::app::AppState;
use crate::api_server::core::{CoreError, CoreEvent, CoreOutput, ReceivedAs};
use crate::api_server::handlers::CoreCall;
use crate::message::config::ServerArgs;
use crate::message::request::GenerateRequest;
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
    let (formatter, error) = sglang_processor::select_chat_formatter(&ChatFormatterOptions {
        tokenizer_path: server_args.tokenizer_path.clone(),
        model_path: server_args.model_path.clone(),
        revision: server_args.revision.clone(),
        chat_template: server_args.chat_template.clone(),
    });
    if let Some(error) = error {
        tracing::warn!(%error, "OpenAI chat completions disabled");
    }
    // recipe parses a DeepSeek model's output in place of the server's parsers;
    // Python refuses to start where it cannot.
    let deepseek = formatter.as_ref().map(|formatter| {
        sglang_processor::openai::ChatModel::for_formatter(formatter)
            .deepseek_output(
                server_args.reasoning_parser.as_deref(),
                server_args.tool_call_parser.as_deref(),
            )
            .map(|_| ())
    });
    if let Some(Err(error)) = deepseek {
        tracing::warn!(%error, "OpenAI chat completions disabled");
        return None;
    }
    formatter
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
    error_response(code, error_payload(code, message), stream)
}

/// Drain one submitted request to its terminal output, fold frames, and map
/// semantic failures / truncation to `(status, message)` for the OpenAI error
/// shape. The call owns cancellation and disarms itself.
async fn collect_output(mut call: CoreCall) -> Result<CoreOutput, (StatusCode, String)> {
    let mut accumulated = CoreOutput::default();
    let output = loop {
        match call.recv().await {
            Some(CoreEvent::Delta(output)) => accumulated.append_delta(&output),
            Some(CoreEvent::Finished(output)) => {
                accumulated.append_delta(&output);
                break accumulated;
            }
            Some(CoreEvent::Failed(error)) => {
                let status = error.http_status();
                return Err((status, error.to_string()));
            }
            None => {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "response truncated before completion".into(),
                ));
            }
        }
    };
    Ok(output)
}

/// A JSON request body as `Json` extracts it (the same rejections), with its
/// bytes when the request log logs OpenAI requests as the client sent them.
pub(super) struct LoggedJson<T> {
    pub(super) body: Result<Json<T>, JsonRejection>,
    pub(super) raw: Option<Bytes>,
}

impl<T: serde::de::DeserializeOwned> FromRequest<Arc<AppState>> for LoggedJson<T> {
    type Rejection = std::convert::Infallible;

    async fn from_request(
        request: Request,
        state: &Arc<AppState>,
    ) -> Result<Self, Self::Rejection> {
        // `Json` refuses a body that is not JSON before reading it.
        if !state.core.logs_openai_requests() || !json_content_type(request.headers()) {
            let body = Json::from_request(request, state).await;
            return Ok(Self { body, raw: None });
        }
        let raw = match Bytes::from_request(request, state).await {
            Ok(raw) => raw,
            Err(rejection) => {
                return Ok(Self {
                    body: Err(rejection.into()),
                    raw: None,
                });
            }
        };
        let body = Json::from_bytes(&raw);
        Ok(Self {
            raw: body.is_ok().then_some(raw),
            body,
        })
    }
}

/// The `Content-Type` check of axum's `Json` extractor.
fn json_content_type(headers: &HeaderMap) -> bool {
    headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok()?.parse::<mime::Mime>().ok())
        .is_some_and(|mime| {
            mime.type_() == "application"
                && (mime.subtype() == "json" || mime.suffix().is_some_and(|name| name == "json"))
        })
}

async fn submit_generation(
    state: &AppState,
    headers: &HeaderMap,
    request: GenerateRequest,
    stream: bool,
) -> Result<CoreCall, Response> {
    let received = ReceivedAs::Typed(state.core.logged_headers(headers));
    match state.core.generate(request, received).await {
        Ok(call) => Ok(call),
        // Same `error_response` rule: a committed stream gets 200 plus an
        // SSE error frame + `[DONE]`, not a unary 503 — but with the OpenAI
        // error shape, since this is the OpenAI frontend.
        Err(CoreError::Unavailable) => Err(openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "service unavailable",
            stream,
        )),
        // Admission refused the request (the launch's `GenerateValidator`, or
        // a media client error) before any frame: Python answers the
        // tokenizer manager's `ValueError` with a unary 400, for a stream too.
        Err(CoreError::InvalidArgument(message)) => {
            Err(openai_error(StatusCode::BAD_REQUEST, message, false))
        }
        Err(error) => Err(openai_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            error.to_string(),
            stream,
        )),
    }
}

fn indexed_decode_stream(
    index: usize,
    call: CoreCall,
) -> futures::stream::BoxStream<'static, (usize, CoreEvent)> {
    futures::stream::unfold((call, false), move |(mut call, finished)| async move {
        if finished {
            return None;
        }
        let event = call.recv().await?;
        let finished = event.is_terminal();
        Some(((index, event), (call, finished)))
    })
    .boxed()
}

/// Content-part keys that carry media; images first.
const MEDIA_KEYS: [&str; 5] = ["image_url", "video_url", "input_audio", "audio_url", "file"];

fn contains_media(value: &serde_json::Value) -> bool {
    contains_any_key(value, &MEDIA_KEYS)
}

/// Media the multimodal pipeline does not take yet: anything but images.
fn contains_non_image_media(value: &serde_json::Value) -> bool {
    contains_any_key(value, &MEDIA_KEYS[1..])
}

fn contains_any_key(value: &serde_json::Value, keys: &[&str]) -> bool {
    match value {
        serde_json::Value::Array(values) => values.iter().any(|v| contains_any_key(v, keys)),
        serde_json::Value::Object(object) => {
            object.keys().any(|key| keys.contains(&key.as_str()))
                || object.values().any(|v| contains_any_key(v, keys))
        }
        _ => false,
    }
}

/// Why a chat or completions request's `model` cannot be served, if it
/// cannot. Any name is served, as Python's handlers serve it, except the
/// `base-model:adapter` form: there it asks for that LoRA adapter, which
/// this server does not support, so serving the base model instead would
/// silently change the answer. (Python refuses it the same way when LoRA is
/// off.) The served name itself is always accepted, colon or not.
fn model_error(model: &str, served_model_name: &str) -> Option<String> {
    if model == served_model_name {
        return None;
    }
    let (_, adapter) = model.split_once(':')?;
    let adapter = adapter.trim();
    (!adapter.is_empty()).then(|| {
        format!(
            "model `{model}` asks for LoRA adapter `{adapter}`; LoRA is not supported by the Rust server"
        )
    })
}

#[cfg(test)]
mod test_utils;
