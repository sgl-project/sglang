//! Transport-neutral pieces of the OpenAI-compatible frontend: chat-template
//! rendering, tool-call constraint + parsing, and reasoning-content splitting.
//! The HTTP handlers stay in `api_server::openai`.

pub(crate) mod chat;
pub(crate) mod completions;
pub(crate) mod models;
pub(crate) mod reasoning;
pub(crate) mod template;
#[cfg(test)]
pub(crate) mod test_utils;
pub(crate) mod tools;

use futures::StreamExt;
use tokio::sync::mpsc;

use self::template::ChatFormatter;
use crate::api_server::core::error::ApiError;
use crate::api_server::core::frame::OutputAccumulator;
use crate::api_server::core::guard::AbortGuard;
use crate::api_server::core::state::CoreState;
use crate::api_server::core::submit::submit;
use crate::message::config::ServerArgs;
use crate::message::ids::Rid;
use crate::message::request::{GenerateRequest, RequestKind};
use crate::message::response::{ChunkEvent, ResponseItem};
use crate::tokenizer_manager::tokenizer;

pub(crate) fn unix_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

pub(crate) fn unix_seconds_u32() -> u32 {
    u32::try_from(unix_seconds()).unwrap_or(u32::MAX)
}

/// The OpenAI error payload — the body shape every OpenAI-compatible surface
/// answers errors with, regardless of transport framing.
pub(crate) fn error_payload_value(code: u16, message: &str) -> serde_json::Value {
    let error_type = if code == 401 {
        "AuthenticationError"
    } else if (500..600).contains(&code) {
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
            "code": code,
        }
    })
}

/// Resolve the chat formatter, or `None` to disable the OpenAI chat-completions
/// endpoint. Tokenization is the tokenizer pool's job (the api server never
/// encodes); the formatter needs at most `tokenizer_config.json` — a built-in
/// `--chat-template` name or a model-path-inferred legacy template resolve
/// without it, so its absence must not disable chat.
pub(crate) fn load_chat_support(server_args: &ServerArgs) -> Option<ChatFormatter> {
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

    match crate::api_server::core::openai::template::load_chat_formatter(
        config_file.as_deref(),
        (!server_args.model_path.is_empty()).then_some(server_args.model_path.as_str()),
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

/// Drain one submitted request to its terminal output: fold frames, disarm
/// `guard` on a natural terminal, and map errors / validation aborts /
/// truncation to an [`ApiError`] for the caller's transport shaping.
pub(crate) async fn collect_output(
    mut rx: mpsc::Receiver<ResponseItem>,
    guard: &mut AbortGuard,
    rid: &Rid,
) -> Result<ChunkEvent, ApiError> {
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
                return Err(ApiError::from_pipeline(&error));
            }
            Some(ResponseItem::Control(_)) | Some(ResponseItem::Data(_)) => {}
            None => {
                return Err(ApiError::internal("response truncated before completion"));
            }
        }
    };
    guard.disarm(rid);
    if let Some((code, message)) = output
        .finish_reason
        .as_ref()
        .and_then(|reason| reason.abort_status())
    {
        return Err(ApiError::from_abort(code, message));
    }
    Ok(output)
}

pub(crate) async fn submit_generation(
    state: &CoreState,
    request: GenerateRequest,
    guard: &mut AbortGuard,
) -> Result<mpsc::Receiver<ResponseItem>, ApiError> {
    let (rid, rx) = submit(state, RequestKind::Generate(Box::new(request))).await?;
    guard.arm(rid);
    Ok(rx)
}

pub(crate) fn indexed_decode_stream(
    index: usize,
    rx: mpsc::Receiver<ResponseItem>,
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
