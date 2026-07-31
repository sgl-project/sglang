//! OpenAI-compatible generation endpoints.
//!
//! The HTTP adapter stays deliberately thin: Dynamo owns the standard OpenAI
//! request and response primitives. Native [`ChunkEvent`] values remain the one
//! backend output type for both unary and streaming responses.

use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    Json, Router,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use dynamo_protocols::types::ChatCompletionRequestMessage;
use dynamo_protocols::types::responses::Response as OpenAIResponse;
use futures::StreamExt;
use serde::Serialize;
use tokio::sync::{RwLock, mpsc};

mod chat;
mod completions;
mod models;
mod response_stream;
mod responses;
mod template;
mod tools;

pub(super) use template::ChatFormatter;

use super::AppState;
use super::frame::OutputAccumulator;
use super::guard::AbortGuard;
use super::submit::submit;
use crate::ids::Rid;
use crate::message::{ChunkEvent, EgressItem, GenerateRequest, RequestKind};
use crate::runtime::ServerArgs;

const MAX_OPENAI_CHOICES: usize = 4096;

#[derive(Clone)]
pub(super) struct StoredResponse {
    response: OpenAIResponse,
    messages: Vec<ChatCompletionRequestMessage>,
    rid: Option<Rid>,
}

pub(super) type ResponseStore = Arc<RwLock<HashMap<String, StoredResponse>>>;

pub(super) fn new_response_store() -> ResponseStore {
    Arc::new(RwLock::new(HashMap::new()))
}

/// The routes this module owns, mounted by `api_server::serve`.
pub(super) fn routes() -> Router<AppState> {
    Router::new()
        .merge(models::routes())
        .merge(completions::routes())
        .merge(chat::routes())
        .merge(responses::routes())
}

pub(super) fn load_chat_support(
    server_args: &ServerArgs,
) -> (Option<ChatFormatter>, Option<dynamo_tokenizers::Tokenizer>) {
    if server_args.skip_tokenizer_init || server_args.tokenizer_path.is_empty() {
        return (None, None);
    }
    let Some(tokenizer_file) = crate::tokenizer::resolve_model_file(
        &server_args.tokenizer_path,
        server_args.revision.as_deref(),
        "tokenizer.json",
    ) else {
        return (None, None);
    };
    let Some(config_file) = crate::tokenizer::resolve_model_file(
        &server_args.tokenizer_path,
        server_args.revision.as_deref(),
        "tokenizer_config.json",
    ) else {
        return (None, None);
    };

    let formatter =
        template::load_chat_formatter(&config_file, server_args.chat_template.as_deref());
    let tokenizer = dynamo_tokenizers::Tokenizer::from_file_with_options(
        &tokenizer_file,
        dynamo_tokenizers::TokenizerOptions {
            add_special_tokens: false,
        },
    )
    .map_err(|error| error.to_string());

    let error = match (formatter, tokenizer) {
        (Ok(formatter), Ok(tokenizer)) => {
            tracing::info!(%config_file, "loaded OpenAI chat template");
            return (Some(formatter), Some(tokenizer));
        }
        (Err(error), _) => error.to_string(),
        (_, Err(error)) => error,
    };
    tracing::warn!(%error, "OpenAI chat completions disabled");
    (None, None)
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    object: &'static str,
    message: String,
    #[serde(rename = "type")]
    error_type: &'static str,
    param: Option<String>,
    code: u16,
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

fn openai_error(code: StatusCode, message: impl Into<String>) -> Response {
    let error_type = if code == StatusCode::UNAUTHORIZED {
        "AuthenticationError"
    } else if code.is_server_error() {
        "InternalServerError"
    } else {
        "BadRequestError"
    };
    (
        code,
        Json(ErrorResponse {
            object: "error",
            message: message.into(),
            error_type,
            param: None,
            code: code.as_u16(),
        }),
    )
        .into_response()
}

fn streaming_error(code: u16, message: impl Into<String>) -> String {
    let status = StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    serde_json::json!({
        "error": ErrorResponse {
            object: "error",
            message: message.into(),
            error_type: if status.is_server_error() {
                "InternalServerError"
            } else {
                "BadRequestError"
            },
            param: None,
            code,
        }
    })
    .to_string()
}

fn is_authorized(state: &AppState, headers: &HeaderMap) -> bool {
    let Some(key) = state.server_args.api_key.as_deref() else {
        return true;
    };
    let expected = format!("Bearer {key}");
    headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        == Some(expected.as_str())
}

fn authorize(state: &AppState, headers: &HeaderMap) -> Option<Response> {
    (!is_authorized(state, headers)).then(|| {
        openai_error(
            StatusCode::UNAUTHORIZED,
            "Invalid authentication credentials",
        )
    })
}

async fn collect_output(
    mut rx: mpsc::Receiver<EgressItem>,
    guard: &mut AbortGuard,
    rid: &Rid,
) -> Result<ChunkEvent, (StatusCode, String)> {
    let mut accumulator = OutputAccumulator::default();
    let output = loop {
        match rx.recv().await {
            Some(EgressItem::Frame(output)) => accumulator.fold(&output),
            Some(EgressItem::Done(output)) => {
                accumulator.fold(&output);
                break accumulator.into_output();
            }
            Some(EgressItem::Error(error)) => {
                guard.disarm(rid);
                let status = StatusCode::from_u16(error.http_status())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                return Err((status, error.to_string()));
            }
            Some(EgressItem::Control(_)) => {}
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
    request: GenerateRequest,
    stream: bool,
    guard: &mut AbortGuard,
) -> Result<mpsc::Receiver<EgressItem>, Response> {
    match submit(state, RequestKind::Generate(Box::new(request)), stream).await {
        Ok((rid, rx)) => {
            guard.arm(rid);
            Ok(rx)
        }
        Err(_) => Err(openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "service unavailable",
        )),
    }
}

fn indexed_egress_stream(
    index: usize,
    rx: mpsc::Receiver<EgressItem>,
) -> futures::stream::BoxStream<'static, (usize, Option<EgressItem>)> {
    futures::stream::unfold((rx, false), move |(mut rx, finished)| async move {
        if finished {
            return None;
        }
        match rx.recv().await {
            Some(item) => {
                let finished = matches!(item, EgressItem::Done(_) | EgressItem::Error(_));
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
mod tests;
