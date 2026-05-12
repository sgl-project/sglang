// Copyright 2023-2024 SGLang Team
// Licensed under the Apache License, Version 2.0

//! axum HTTP server exposing OpenAI-compatible API endpoints.
//!
//! Endpoints:
//!   GET  /health
//!   GET  /v1/models
//!   POST /v1/completions          (streaming + non-streaming)
//!   POST /v1/chat/completions     (streaming + non-streaming)

pub mod protocol;

use crate::http_server::protocol::*;
use crate::ipc::wire::SamplingParams;
use crate::tokenizer_manager::{ResponseChunk, TmConfig, TokenizerManager};
use crate::HttpServerConfig;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use log::info;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tower_http::cors::CorsLayer;

// ──────────────────────────── shared state ──────────────────────────────────

#[derive(Clone)]
pub struct AppState {
    pub tm: Arc<TokenizerManager>,
    pub model_name: String,
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn gen_id(prefix: &str) -> String {
    format!("{prefix}-{}", uuid::Uuid::new_v4().simple())
}

// ─────────────────────────── router factory ─────────────────────────────────

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/completions", post(v1_completions))
        .route("/v1/chat/completions", post(v1_chat_completions))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

// ─────────────────────────────── handlers ───────────────────────────────────

async fn health() -> &'static str {
    "OK"
}

async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelList> {
    Json(ModelList {
        object: "list".into(),
        data: vec![ModelCard {
            id: state.model_name.clone(),
            object: "model".into(),
            created: now_secs(),
            owned_by: "sglang".into(),
        }],
    })
}

// ─────────────────────────── /v1/completions ────────────────────────────────

async fn v1_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    let stream_mode = req.stream.unwrap_or(false);

    // Build text prompt from request
    let text = match &req.prompt {
        PromptInput::Text(s) => s.clone(),
        PromptInput::Batch(v) => v.first().cloned().unwrap_or_default(),
        PromptInput::Tokens(_) => {
            return error_response(StatusCode::BAD_REQUEST, "Token ID prompts not supported");
        }
        PromptInput::Empty => String::new(),
    };

    let input_ids = match state.tm.tokenize(&text) {
        Ok(ids) => ids,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e),
    };

    let sp = sampling_params_from_completion(&req);
    let model = req.model.clone();
    let (rid, mut rx) = state.tm.submit(input_ids, sp, stream_mode).await;

    if stream_mode {
        let id = gen_id("cmpl");
        let (sse_tx, sse_rx) = mpsc::unbounded_channel::<Result<Event, Infallible>>();
        let id2 = id.clone();
        let model2 = model.clone();
        tokio::spawn(async move {
            while let Some(chunk) = rx.recv().await {
                let (data, done) = match &chunk {
                    ResponseChunk::Delta { text } => {
                        let c = CompletionChunk {
                            id: id2.clone(),
                            object: "text_completion".into(),
                            created: now_secs(),
                            model: model2.clone(),
                            choices: vec![CompletionChunkChoice {
                                text: text.clone(),
                                index: 0,
                                finish_reason: None,
                                logprobs: None,
                            }],
                        };
                        (serde_json::to_string(&c).unwrap_or_default(), false)
                    }
                    ResponseChunk::Done { text, finish_reason } => {
                        let c = CompletionChunk {
                            id: id2.clone(),
                            object: "text_completion".into(),
                            created: now_secs(),
                            model: model2.clone(),
                            choices: vec![CompletionChunkChoice {
                                text: text.clone(),
                                index: 0,
                                finish_reason: Some(finish_reason.clone()),
                                logprobs: None,
                            }],
                        };
                        (serde_json::to_string(&c).unwrap_or_default(), true)
                    }
                    ResponseChunk::Error(e) => (format!("{{\"error\":\"{e}\"}}"), true),
                };
                if sse_tx.send(Ok(Event::default().data(data))).is_err() || done {
                    break;
                }
            }
            let _ = sse_tx.send(Ok(Event::default().data("[DONE]")));
        });

        let _ = rid;
        Sse::new(UnboundedReceiverStream::new(sse_rx)).into_response()
    } else {
        let (full_text, finish_reason) = collect_text(rx).await;
        let resp = CompletionResponse {
            id: gen_id("cmpl"),
            object: "text_completion".into(),
            created: now_secs(),
            model: req.model.clone(),
            choices: vec![CompletionChoice {
                text: full_text,
                index: 0,
                finish_reason: Some(finish_reason),
                logprobs: None,
            }],
            usage: UsageInfo::default(),
        };
        let _ = rid;
        Json(resp).into_response()
    }
}

// ──────────────────────── /v1/chat/completions ──────────────────────────────

async fn v1_chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let stream_mode = req.stream.unwrap_or(false);

    // Build message list for template application
    let msgs: Vec<(&str, &str)> = req
        .messages
        .iter()
        .map(|m| {
            let role: &str = m.role.as_str();
            let content: &str = m
                .content
                .as_ref()
                .map(|c| match c {
                    ChatContent::Text(s) => s.as_str(),
                    ChatContent::Parts(_) => "",
                })
                .unwrap_or("");
            (role, content)
        })
        .collect();

    // We need owned strings because template may not live long enough
    let msg_owned: Vec<(String, String)> = msgs
        .iter()
        .map(|(r, c)| (r.to_string(), c.to_string()))
        .collect();

    let msg_refs: Vec<(&str, &str)> = msg_owned
        .iter()
        .map(|(r, c)| (r.as_str(), c.as_str()))
        .collect();

    let prompt = state.tm.apply_chat_template(&msg_refs, true);

    let input_ids = match state.tm.tokenize(&prompt) {
        Ok(ids) => ids,
        Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e),
    };

    let sp = sampling_params_from_chat(&req);
    let model = req.model.clone();
    let (rid, mut rx) = state.tm.submit(input_ids, sp, stream_mode).await;

    if stream_mode {
        let id = gen_id("chatcmpl");
        // Bridge: send all SSE events (role delta + text chunks + [DONE]) into one channel.
        let (sse_tx, sse_rx) = mpsc::unbounded_channel::<Result<Event, Infallible>>();

        let id2 = id.clone();
        let model2 = model.clone();
        tokio::spawn(async move {
            // First chunk: role delta (required by OpenAI spec for streaming)
            let role_chunk = ChatChunk {
                id: id2.clone(),
                object: "chat.completion.chunk".into(),
                created: now_secs(),
                model: model2.clone(),
                choices: vec![ChatChunkChoice {
                    index: 0,
                    delta: DeltaMessage {
                        role: Some("assistant".into()),
                        content: Some(String::new()),
                        tool_calls: None,
                    },
                    finish_reason: None,
                    logprobs: None,
                }],
                usage: None,
            };
            let _ = sse_tx.send(Ok(Event::default().data(
                serde_json::to_string(&role_chunk).unwrap_or_default(),
            )));

            // Content chunks from the detokenizer
            while let Some(chunk) = rx.recv().await {
                let data = match &chunk {
                    ResponseChunk::Delta { text } => {
                        let c = ChatChunk {
                            id: id2.clone(),
                            object: "chat.completion.chunk".into(),
                            created: now_secs(),
                            model: model2.clone(),
                            choices: vec![ChatChunkChoice {
                                index: 0,
                                delta: DeltaMessage {
                                    role: None,
                                    content: Some(text.clone()),
                                    tool_calls: None,
                                },
                                finish_reason: None,
                                logprobs: None,
                            }],
                            usage: None,
                        };
                        serde_json::to_string(&c).unwrap_or_default()
                    }
                    ResponseChunk::Done { text, finish_reason } => {
                        let c = ChatChunk {
                            id: id2.clone(),
                            object: "chat.completion.chunk".into(),
                            created: now_secs(),
                            model: model2.clone(),
                            choices: vec![ChatChunkChoice {
                                index: 0,
                                delta: DeltaMessage {
                                    role: None,
                                    content: Some(text.clone()),
                                    tool_calls: None,
                                },
                                finish_reason: Some(finish_reason.clone()),
                                logprobs: None,
                            }],
                            usage: None,
                        };
                        let _ = sse_tx.send(Ok(Event::default().data(
                            serde_json::to_string(&c).unwrap_or_default(),
                        )));
                        break;
                    }
                    ResponseChunk::Error(e) => format!("{{\"error\":\"{e}\"}}"),
                };
                if sse_tx.send(Ok(Event::default().data(data))).is_err() {
                    break;
                }
            }
            // [DONE] sentinel
            let _ = sse_tx.send(Ok(Event::default().data("[DONE]")));
        });

        let _ = rid;
        Sse::new(UnboundedReceiverStream::new(sse_rx)).into_response()
    } else {
        let (full_text, finish_reason) = collect_text(rx).await;
        let resp = ChatCompletionResponse {
            id: gen_id("chatcmpl"),
            object: "chat.completion".into(),
            created: now_secs(),
            model: req.model.clone(),
            choices: vec![ChatChoice {
                index: 0,
                message: AssistantMessage {
                    role: "assistant".into(),
                    content: Some(full_text),
                    tool_calls: None,
                },
                finish_reason: Some(finish_reason),
                logprobs: None,
            }],
            usage: UsageInfo::default(),
        };
        let _ = rid;
        Json(resp).into_response()
    }
}

// ─────────────────────────── helper utilities ───────────────────────────────

/// Drain a response channel to completion, accumulating text and finish reason.
async fn collect_text(mut rx: mpsc::UnboundedReceiver<ResponseChunk>) -> (String, String) {
    let mut full = String::new();
    let mut reason = "stop".to_string();
    while let Some(chunk) = rx.recv().await {
        match chunk {
            ResponseChunk::Delta { text } => full.push_str(&text),
            ResponseChunk::Done { text, finish_reason } => {
                full.push_str(&text);
                reason = finish_reason;
                break;
            }
            ResponseChunk::Error(e) => {
                reason = format!("error: {e}");
                break;
            }
        }
    }
    (full, reason)
}

fn error_response(status: StatusCode, msg: &str) -> Response {
    let body = ErrorResponse::bad_request(msg);
    (status, Json(body)).into_response()
}

fn sampling_params_from_completion(req: &CompletionRequest) -> SamplingParams {
    let mut sp = SamplingParams::default();
    if let Some(v) = req.max_tokens { sp.max_new_tokens = v; }
    if let Some(v) = req.temperature { sp.temperature = v; }
    if let Some(v) = req.top_p { sp.top_p = v; }
    if let Some(v) = req.top_k { sp.top_k = v; }
    if let Some(v) = req.min_p { sp.min_p = v; }
    if let Some(v) = req.presence_penalty { sp.presence_penalty = v; }
    if let Some(v) = req.frequency_penalty { sp.frequency_penalty = v; }
    if let Some(v) = req.repetition_penalty { sp.repetition_penalty = v; }
    if let Some(v) = req.seed { sp.seed = Some(v); }
    if let Some(v) = req.ignore_eos { sp.ignore_eos = v; }
    if let Some(v) = req.skip_special_tokens { sp.skip_special_tokens = v; }
    if let Some(stop) = &req.stop { sp.stop = Some(stop.as_vec()); }
    if let Some(rf) = &req.response_format {
        sp.json_schema = rf.json_schema_str();
    }
    sp
}

fn sampling_params_from_chat(req: &ChatCompletionRequest) -> SamplingParams {
    let mut sp = SamplingParams::default();
    let max_tokens = req.max_completion_tokens.or(req.max_tokens);
    if let Some(v) = max_tokens { sp.max_new_tokens = v; }
    if let Some(v) = req.temperature { sp.temperature = v; }
    if let Some(v) = req.top_p { sp.top_p = v; }
    if let Some(v) = req.top_k { sp.top_k = v; }
    if let Some(v) = req.min_p { sp.min_p = v; }
    if let Some(v) = req.presence_penalty { sp.presence_penalty = v; }
    if let Some(v) = req.frequency_penalty { sp.frequency_penalty = v; }
    if let Some(v) = req.repetition_penalty { sp.repetition_penalty = v; }
    if let Some(v) = req.seed { sp.seed = Some(v); }
    if let Some(v) = req.ignore_eos { sp.ignore_eos = v; }
    if let Some(v) = req.skip_special_tokens { sp.skip_special_tokens = v; }
    if let Some(stop) = &req.stop { sp.stop = Some(stop.as_vec()); }
    if let Some(rf) = &req.response_format {
        sp.json_schema = rf.json_schema_str();
    }
    sp
}

// ─────────────────────────────────── start ──────────────────────────────────

pub fn start(config: HttpServerConfig) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(config.worker_threads.unwrap_or(8) as usize)
        .enable_all()
        .thread_name("sglang-http")
        .build()
        .expect("Failed to build tokio runtime");

    rt.block_on(async move {
        let tm_cfg = TmConfig {
            tokenizer_ipc_name: config.tokenizer_ipc_name.clone(),
            scheduler_ipc_name: config.scheduler_ipc_name.clone(),
            tokenizer_path: config.tokenizer_path.clone(),
            skip_tokenizer_init: config.skip_tokenizer_init,
            model_name: config.model_name.clone(),
        };

        let tm = TokenizerManager::start(tm_cfg);

        let state = Arc::new(AppState {
            tm,
            model_name: config.model_name.clone(),
        });

        let app = build_router(state);
        let addr = format!("{}:{}", config.host, config.port);
        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .unwrap_or_else(|e| panic!("Failed to bind HTTP on {addr}: {e}"));

        info!("HTTP server listening on {addr}");
        axum::serve(listener, app)
            .await
            .expect("HTTP server error");
    });
}
