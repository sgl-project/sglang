//! Ollama-compatible API endpoints (`/api/chat`, `/api/generate`, `/api/tags`,
//! `/api/show`). A thin HTTP adapter on top of the OpenAI chat-template /
//! submission machinery. Streaming responses use `application/x-ndjson` (one
//! JSON object per line), matching python/sglang/srt/entrypoints/ollama.
//!
//! The module sits under `openai` so it can reuse the shared chat renderer,
//! sampling defaults, and request submission path instead of duplicating the
//! request lifecycle.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use dynamo_protocols::types::{CreateChatCompletionRequest, Stop};
use futures::StreamExt;
use serde::Deserialize;
use tokio::sync::mpsc;

use super::super::frame::OutputAccumulator;
use super::super::guard::AbortGuard;
use super::super::submit::submit;
use super::AppState;
use super::chat::prepare_chat_request;
use crate::message::ids::Rid;
use crate::message::request::{GenerateRequest, RequestKind};
use crate::message::response::{ChunkEvent, ResponseItem};
use crate::message::sampling::SamplingParams;
use crate::message::types::OneOrMany;
use axum::body::Body;

const DEFAULT_MAX_NEW_TOKENS: i64 = 2048;

/// Ollama `/api/chat` request (fields actually read by the Python handler).
#[derive(Debug, Deserialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    options: Option<serde_json::Map<String, serde_json::Value>>,
}

/// Ollama `/api/generate` request.
#[derive(Debug, Deserialize)]
struct OllamaGenerateRequest {
    model: String,
    prompt: String,
    #[serde(default)]
    system: Option<String>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    options: Option<serde_json::Map<String, serde_json::Value>>,
}

/// Ollama `/api/show` request.
#[derive(Debug, Deserialize)]
struct OllamaShowRequest {
    model: String,
}

/// Ollama message.
#[derive(Debug, Deserialize)]
struct OllamaMessage {
    role: String,
    content: String,
}

/// Routes this module owns, mounted by `openai::routes`.
pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/chat", post(chat))
        .route("/api/generate", post(generate))
        .route("/api/tags", get(tags))
        .route("/api/show", post(show))
}

fn timestamp() -> String {
    // Ollama timestamps are `YYYY-MM-DDThh:mm:ss.mmmZ` in UTC. The exact value
    // is not part of parity tests; a close civil conversion is enough to match
    // the Ollama CLI shape.
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}Z", civil_utc(now.as_secs(), now.subsec_millis()))
}

/// Days since 1970-01-01 to (year, month, day) via Howard Hinnant's algorithm.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn civil_utc(secs: u64, millis: u32) -> String {
    let days = secs / 86_400;
    let rem = secs % 86_400;
    let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let (y, mo, d) = civil_from_days(days as i64);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}.{millis:03}Z")
}

/// `POST /api/chat`
async fn chat(
    State(state): State<Arc<AppState>>,
    body: Result<Json<OllamaChatRequest>, JsonRejection>,
) -> Response {
    let ollama = match body {
        Ok(Json(body)) => body,
        Err(rejection) => return ollama_error(StatusCode::BAD_REQUEST, &rejection.body_text()),
    };
    if ollama.model != state.server_args.served_model_name {
        return ollama_error(
            StatusCode::BAD_REQUEST,
            &format!("The model `{}` does not exist", ollama.model),
        );
    }
    if ollama.messages.is_empty() {
        return ollama_error(StatusCode::BAD_REQUEST, "messages cannot be empty");
    }

    // Reuse the OpenAI chat formatter by translating the Ollama message shape
    // into an OpenAI-shaped request. `CreateChatCompletionRequest` is derived
    // from client JSON in the OpenAI handler too, so omitted fields default.
    let openai_value = serde_json::json!({
        "model": ollama.model,
        "messages": ollama
            .messages
            .iter()
            .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
            .collect::<Vec<_>>(),
        "stream": ollama.stream,
    });
    let mut request: CreateChatCompletionRequest = match serde_json::from_value(openai_value) {
        Ok(request) => request,
        Err(error) => return ollama_error(StatusCode::BAD_REQUEST, &error.to_string()),
    };
    let prompt = match prepare_chat_request(&state, request.clone()).await {
        Ok((prepared, prompt)) => {
            request = prepared;
            prompt
        }
        Err(response) => return response,
    };

    let mut sampling = match sampling_params(ollama.options.as_ref()) {
        Ok(sampling) => sampling,
        Err(message) => return ollama_error(StatusCode::BAD_REQUEST, &message),
    };
    // Template stops first, then request stops, mirroring the OpenAI chat path.
    let request_stops = match request.stop.as_ref() {
        Some(Stop::String(value)) => Some(vec![value.clone()]),
        Some(Stop::StringArray(values)) => Some(values.clone()),
        Some(Stop::TokenIdArray(_)) | None => None,
    };
    if let Some(request_stops) = request_stops {
        let mut stops = request_stops;
        if let Some(user_stops) = sampling.stop.take() {
            match user_stops {
                OneOrMany::One(value) => stops.push(value),
                OneOrMany::Many(many) => stops.extend(many),
            }
        }
        sampling.stop = Some(OneOrMany::Many(stops));
    }
    if let Err(error) = sampling.normalize(
        state.server_args.skip_tokenizer_init,
        state.server_args.model_config.vocab_size,
    ) {
        return ollama_error(StatusCode::BAD_REQUEST, &error.to_string());
    }

    let generation = GenerateRequest {
        rid: Rid::from_client(&format!("ollama-{}", uuid::Uuid::new_v4().simple())),
        text: Some(prompt),
        // Template owns special tokens; keep identical to OpenAI chat.
        skip_special_tokens: true,
        sampling_params: sampling,
        stream: ollama.stream,
        ..Default::default()
    };

    if ollama.stream {
        let stream = ollama_stream(state.clone(), generation, ollama.model.clone(), true);
        let body = Body::from_stream(
            stream.map(|line| Ok::<_, std::convert::Infallible>(line.into_bytes())),
        );
        (
            StatusCode::OK,
            [("content-type", "application/x-ndjson")],
            body,
        )
            .into_response()
    } else {
        unary(&state, generation, &ollama.model, true).await
    }
}

/// `POST /api/generate`
async fn generate(
    State(state): State<Arc<AppState>>,
    body: axum::Json<OllamaGenerateRequest>,
) -> Response {
    let ollama = body.0;
    if ollama.model != state.server_args.served_model_name {
        return ollama_error(
            StatusCode::BAD_REQUEST,
            &format!("The model `{}` does not exist", ollama.model),
        );
    }
    // Ollama CLI sends an empty prompt on startup; return an empty response
    // instead of round-tripping a no-op to the scheduler.
    if ollama.prompt.trim().is_empty() {
        let empty = serde_json::json!({
            "model": ollama.model,
            "created_at": timestamp(),
            "response": "",
            "done": true,
            "done_reason": "stop",
        });
        if ollama.stream {
            return (
                StatusCode::OK,
                [("content-type", "application/x-ndjson")],
                format!("{}\n", empty),
            )
                .into_response();
        }
        return (StatusCode::OK, Json(empty)).into_response();
    }
    let prompt = match &ollama.system {
        Some(system) if !system.is_empty() => format!("{system}\n\n{}", ollama.prompt),
        _ => ollama.prompt,
    };

    let mut sampling = match sampling_params(ollama.options.as_ref()) {
        Ok(sampling) => sampling,
        Err(message) => return ollama_error(StatusCode::BAD_REQUEST, &message),
    };
    if let Err(error) = sampling.normalize(
        state.server_args.skip_tokenizer_init,
        state.server_args.model_config.vocab_size,
    ) {
        return ollama_error(StatusCode::BAD_REQUEST, &error.to_string());
    }

    let generation = GenerateRequest {
        rid: Rid::from_client(&format!("ollama-{}", uuid::Uuid::new_v4().simple())),
        text: Some(prompt),
        sampling_params: sampling,
        stream: ollama.stream,
        ..Default::default()
    };

    if ollama.stream {
        let stream = ollama_stream(state.clone(), generation, ollama.model.clone(), false);
        let body = Body::from_stream(
            stream.map(|line| Ok::<_, std::convert::Infallible>(line.into_bytes())),
        );
        (
            StatusCode::OK,
            [("content-type", "application/x-ndjson")],
            body,
        )
            .into_response()
    } else {
        unary(&state, generation, &ollama.model, false).await
    }
}

/// `GET /api/tags`
async fn tags(State(state): State<Arc<AppState>>) -> Response {
    let name = &state.server_args.served_model_name;
    let family = name.rsplit('/').next().unwrap_or(name).to_owned();
    Json(serde_json::json!({
        "models": [{
            "name": name,
            "model": name,
            "modified_at": timestamp(),
            "size": 0,
            "digest": "sha256:sglang0000000000000000000000000000000000000000000000000000000000",
            "details": {
                "format": "sglang",
                "family": family,
                "parameter_size": "unknown"
            }
        }]
    }))
    .into_response()
}

/// `POST /api/show` — Python routes this as POST only, so no GET alias here.
async fn show(
    State(state): State<Arc<AppState>>,
    body: Result<Json<OllamaShowRequest>, JsonRejection>,
) -> Response {
    let model = match body {
        Ok(Json(body)) => body.model,
        Err(rejection) => return ollama_error(StatusCode::BAD_REQUEST, &rejection.body_text()),
    };
    let family = model
        .rsplit('/')
        .next()
        .unwrap_or(&model)
        .replace("-Instruct", "")
        .replace("-Chat", "")
        .replace("-Base", "");
    let context_len = state.server_args.model_config.context_len;
    Json(serde_json::json!({
        "license": "",
        "modelfile": format!("FROM {model}\nPARAMETER num_ctx {context_len}\n"),
        "parameters": format!("num_ctx {context_len}"),
        "template": "",
        "modified_at": timestamp(),
        "details": {
            "parent_model": "",
            "format": "sglang",
            "family": family,
            "families": [family],
            "parameter_size": "unknown",
            "quantization_level": ""
        },
        "model_info": {
            "general.architecture": family,
            "general.name": model,
            "general.parameter_count": 0,
            format!("{family}.context_length"): context_len,
            format!("{family}.block_count"): 0,
            format!("{family}.embedding_length"): 0,
            format!("{family}.attention.head_count"): 0,
        },
        "capabilities": ["completion"]
    }))
    .into_response()
}

async fn unary(state: &AppState, generation: GenerateRequest, model: &str, chat: bool) -> Response {
    let rid = generation.rid.clone();
    let mut guard = AbortGuard::new(state.senders.clone(), rid.clone());
    let rx = match submit(state, RequestKind::Generate(Box::new(generation)), false).await {
        Ok((_, rx)) => rx,
        Err(response) => return response,
    };
    let output = match drain_output(state, rx, &mut guard, &rid).await {
        Ok(output) => output,
        Err((status, message)) => return ollama_error(status, &message),
    };
    let done = output.finish_reason.is_some();
    let mut value = serde_json::json!({
        "model": model,
        "created_at": timestamp(),
        "done": done,
        "total_duration": output.completion_tokens,
        "prompt_eval_count": output.prompt_tokens,
        "eval_count": output.completion_tokens,
    });
    if chat {
        value["message"] = serde_json::json!({"role": "assistant", "content": output.text});
    } else {
        value["response"] = serde_json::json!(output.text);
    }
    if done {
        value["done_reason"] = serde_json::json!("stop");
    }
    (StatusCode::OK, Json(value)).into_response()
}

/// Drain a non-streaming response into the terminal [`ChunkEvent`]. Mostly a
/// local equivalent of `openai::collect_output`: it exists here to keep the
/// Ollama module self-contained while still sharing the response channel.
async fn drain_output(
    _state: &AppState,
    mut rx: mpsc::Receiver<ResponseItem>,
    guard: &mut AbortGuard,
    rid: &Rid,
) -> Result<ChunkEvent, (StatusCode, String)> {
    let mut acc = OutputAccumulator::default();
    let output = loop {
        match rx.recv().await {
            Some(ResponseItem::Frame(delta)) => acc.fold(&delta),
            Some(ResponseItem::Done(delta)) => {
                acc.fold(&delta);
                break acc.into_output();
            }
            Some(ResponseItem::Error(error)) => {
                guard.disarm(rid);
                let status = StatusCode::from_u16(error.http_status())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                return Err((status, error.to_string()));
            }
            Some(ResponseItem::Control(_)) | Some(ResponseItem::Data(_)) => {}
            None => {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "response truncated before completion".into(),
                ));
            }
        }
    };
    guard.disarm(rid);
    Ok(output)
}

/// NDJSON streaming drain, compatible with Ollama's Python implementation:
/// a JSON object per line, with injected `done: true` on the terminal frame.
fn ollama_stream(
    state: Arc<AppState>,
    generation: GenerateRequest,
    model: String,
    chat: bool,
) -> impl futures::Stream<Item = String> {
    // Held for the stream's lifetime so a client disconnect aborts the work.
    let _guard = AbortGuard::new_empty(state.senders.clone());
    async_stream::stream! {
        let mut rx = match submit(&state, RequestKind::Generate(Box::new(generation)), true).await {
            Ok((_, rx)) => rx,
            Err(_) => {
                yield error_ndjson(&model);
                return;
            }
        };
        let mut acc = OutputAccumulator::default();
        let mut previous = String::new();
        while let Some(item) = rx.recv().await {
            match item {
                ResponseItem::Frame(delta) | ResponseItem::Done(delta) => {
                    acc.fold(&delta);
                    let cumulative = acc.snapshot();
                    let text = &cumulative.text;
                    let delta_text = if text.starts_with(&previous) {
                        text[previous.len()..].to_owned()
                    } else {
                        text.clone()
                    };
                    previous = text.clone();
                    let done = cumulative.finish_reason.is_some();
                    let mut value = serde_json::json!({
                        "model": model,
                        "created_at": timestamp(),
                        "done": done,
                    });
                    if chat {
                        value["message"] =
                            serde_json::json!({"role": "assistant", "content": delta_text});
                    } else {
                        value["response"] = serde_json::json!(delta_text);
                    }
                    if done {
                        value["done_reason"] = serde_json::json!("stop");
                    }
                    yield serde_json::to_string(&value).unwrap_or_default() + "\n";
                }
                ResponseItem::Error(error) => {
                    yield serde_json::json!({
                        "model": model,
                        "created_at": timestamp(),
                        "error": error.to_string(),
                        "done": true,
                    })
                    .to_string()
                        + "\n";
                }
                ResponseItem::Control(_) | ResponseItem::Data(_) => {}
            }
        }
    }
}

fn error_ndjson(model: &str) -> String {
    serde_json::json!({
        "model": model,
        "created_at": timestamp(),
        "error": "service unavailable",
        "done": true,
    })
    .to_string()
        + "\n"
}

fn ollama_error(code: StatusCode, message: &str) -> Response {
    (code, serde_json::json!({"error": message}).to_string()).into_response()
}

/// Translate Ollama `options` to [`SamplingParams`]. Mirrors Python's mapping
/// in `ollama/serving.py` plus the default `max_new_tokens = 2048`.
fn sampling_params(
    options: Option<&serde_json::Map<String, serde_json::Value>>,
) -> Result<SamplingParams, String> {
    let mut sampling = SamplingParams {
        max_new_tokens: Some(DEFAULT_MAX_NEW_TOKENS),
        ..Default::default()
    };
    let Some(options) = options else {
        return Ok(sampling);
    };
    if let Some(value) = options.get("temperature") {
        sampling.temperature = value
            .as_f64()
            .ok_or_else(|| "temperature must be a number".to_owned())?;
    }
    if let Some(value) = options.get("top_p") {
        sampling.top_p = value
            .as_f64()
            .ok_or_else(|| "top_p must be a number".to_owned())?;
    }
    if let Some(value) = options.get("top_k") {
        sampling.top_k = value
            .as_i64()
            .ok_or_else(|| "top_k must be an integer".to_owned())?;
    }
    if let Some(value) = options.get("num_predict") {
        sampling.max_new_tokens = Some(
            value
                .as_i64()
                .ok_or_else(|| "num_predict must be an integer".to_owned())?,
        );
    }
    if let Some(value) = options.get("stop") {
        sampling.stop = match value {
            serde_json::Value::String(stop) => Some(OneOrMany::One(stop.clone())),
            serde_json::Value::Array(stops) => {
                let stops: Result<Vec<_>, _> = stops
                    .iter()
                    .map(|stop| {
                        stop.as_str()
                            .map(str::to_owned)
                            .ok_or_else(|| "stop array must contain strings".to_owned())
                    })
                    .collect();
                Some(OneOrMany::Many(stops?))
            }
            _ => return Err("stop must be a string or array of strings".to_owned()),
        };
    }
    if let Some(value) = options.get("presence_penalty") {
        sampling.presence_penalty = value
            .as_f64()
            .ok_or_else(|| "presence_penalty must be a number".to_owned())?;
    }
    if let Some(value) = options.get("frequency_penalty") {
        sampling.frequency_penalty = value
            .as_f64()
            .ok_or_else(|| "frequency_penalty must be a number".to_owned())?;
    }
    if let Some(value) = options.get("seed") {
        sampling.sampling_seed = Some(
            value
                .as_i64()
                .ok_or_else(|| "seed must be an integer".to_owned())?,
        );
    }
    Ok(sampling)
}

#[cfg(test)]
mod tests {
    use super::super::test_utils::{app_state, body_json, oneshot, post_json, senders};
    use super::*;
    use axum::body::Body;
    use axum::http::Request;

    fn app() -> axum::Router<()> {
        super::super::routes().with_state(app_state(senders()))
    }

    #[tokio::test]
    async fn tags_lists_served_model() {
        let response = oneshot(
            app(),
            Request::builder()
                .uri("/api/tags")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let value = body_json(response).await;
        let model = &value["models"][0];
        assert_eq!(model["name"], "model");
        assert_eq!(model["model"], "model");
        assert_eq!(model["details"]["format"], "sglang");
    }

    #[tokio::test]
    async fn generate_empty_prompt_returns_empty_done() {
        let response = post_json(
            app(),
            "/api/generate",
            serde_json::json!({"model": "model", "prompt": "  ", "stream": false}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let value = body_json(response).await;
        assert_eq!(value["response"], "");
        assert_eq!(value["done"], true);
        assert_eq!(value["done_reason"], "stop");
    }

    #[tokio::test]
    async fn generate_unknown_model_rejected() {
        let response = post_json(
            app(),
            "/api/generate",
            serde_json::json!({"model": "other", "prompt": "hi"}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let value = body_json(response).await;
        assert!(value["error"].as_str().unwrap().contains("does not exist"));
    }

    #[tokio::test]
    async fn chat_unknown_model_rejected_without_formatter() {
        let response = post_json(
            app(),
            "/api/chat",
            serde_json::json!({"model": "other", "messages": [{"role":"user","content":"hi"}]}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn sampling_options_map_like_python() {
        let params = sampling_params(None).unwrap();
        assert_eq!(params.max_new_tokens, Some(DEFAULT_MAX_NEW_TOKENS));

        let options: serde_json::Map<String, serde_json::Value> =
            serde_json::from_value(serde_json::json!({
                "temperature": 0.31,
                "num_predict": 17,
                "seed": 42,
                "stop": ["END", "STOP"]
            }))
            .unwrap();
        let sampling = sampling_params(Some(&options)).unwrap();
        assert!((sampling.temperature - 0.31).abs() < 1e-6);
        assert_eq!(sampling.max_new_tokens, Some(17));
        assert_eq!(sampling.sampling_seed, Some(42));
        match sampling.stop.unwrap() {
            OneOrMany::Many(values) => {
                assert_eq!(values, vec!["END".to_owned(), "STOP".to_owned()])
            }
            _ => panic!("expected array stop"),
        }
    }
}
