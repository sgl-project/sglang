//! Tokenizer services share the generation workers and never schedule GPU work.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use super::super::{guard::AbortGuard, submit::submit};
use super::{AppState, ChatTemplateKwargs, chat::prepare_chat_request, openai_error};
use crate::message::request::RequestKind;
use crate::message::response::ResponseItem;
use crate::message::types::{OneOrMany, TokenIds};
use crate::utils::error::Error;

pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/tokenize", post(tokenize))
        .route("/v1/tokenize", post(tokenize))
        .route("/detokenize", post(detokenize))
        .route("/v1/detokenize", post(detokenize))
}

fn default_true() -> bool {
    true
}

#[derive(Deserialize)]
struct TokenizeRequest {
    prompt: Option<OneOrMany<String>>,
    messages: Option<Vec<Value>>,
    chat_template_kwargs: Option<ChatTemplateKwargs>,
    #[serde(default = "default_true")]
    add_special_tokens: bool,
    #[serde(flatten)]
    chat_options: Map<String, Value>,
}

#[derive(Deserialize)]
struct DetokenizeRequest {
    tokens: OneOrMany<TokenIds>,
    #[serde(default = "default_true")]
    skip_special_tokens: bool,
}

async fn service(state: &AppState, kind: RequestKind) -> Result<ResponseItem, Response> {
    let (rid, mut rx) = submit(state, kind, false).await.map_err(|_| {
        openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "service unavailable",
            false,
        )
    })?;
    let mut guard = AbortGuard::new(state.senders.clone(), rid.clone());
    let item = rx.recv().await.ok_or_else(|| {
        openai_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "tokenizer worker disconnected",
            false,
        )
    })?;
    guard.disarm(&rid);
    if let ResponseItem::Error(error) = item {
        let status = match error {
            Error::Detokenize(_) => StatusCode::BAD_REQUEST,
            _ => StatusCode::from_u16(error.http_status())
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        };
        return Err(openai_error(status, error.to_string(), false));
    }
    Ok(item)
}

async fn tokenize(
    State(state): State<Arc<AppState>>,
    body: Result<Json<TokenizeRequest>, JsonRejection>,
) -> Response {
    let request = match body {
        Ok(Json(request)) => request,
        Err(error) => return openai_error(StatusCode::BAD_REQUEST, error.body_text(), false),
    };
    let (prompts, batched, add_special_tokens) = match (request.prompt, request.messages) {
        (Some(OneOrMany::One(prompt)), None) => (vec![prompt], false, request.add_special_tokens),
        (Some(OneOrMany::Many(prompts)), None) => (prompts, true, request.add_special_tokens),
        (None, Some(messages)) => {
            if messages.is_empty() {
                return openai_error(StatusCode::BAD_REQUEST, "messages cannot be empty", false);
            }
            let mut chat = request.chat_options;
            chat.entry("model").or_insert_with(|| json!("default"));
            chat.insert("messages".into(), json!(messages));
            let chat = match serde_json::from_value(Value::Object(chat)) {
                Ok(chat) => chat,
                Err(error) => {
                    return openai_error(StatusCode::BAD_REQUEST, error.to_string(), false);
                }
            };
            let (_, prompt) =
                match prepare_chat_request(&state, chat, request.chat_template_kwargs.as_ref())
                    .await
                {
                    Ok(prepared) => prepared,
                    Err(response) => return response,
                };
            // Chat templates own their BOS/EOS tokens, as in generation.
            (vec![prompt], false, false)
        }
        _ => {
            return openai_error(
                StatusCode::BAD_REQUEST,
                "Exactly one of 'prompt' or 'messages' must be provided.",
                false,
            );
        }
    };
    let mut tokens = Vec::with_capacity(prompts.len());
    for text in prompts {
        match service(
            &state,
            RequestKind::Tokenize {
                text,
                add_special_tokens,
            },
        )
        .await
        {
            Ok(ResponseItem::Tokenized(ids)) => tokens.push(ids),
            Ok(_) => {
                return openai_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "unexpected tokenizer response",
                    false,
                );
            }
            Err(response) => return response,
        }
    }
    let count: Vec<_> = tokens.iter().map(Vec::len).collect();
    let (tokens, count) = if batched {
        (json!(tokens), json!(count))
    } else {
        (json!(tokens[0]), json!(count[0]))
    };
    Json(json!({
        "tokens": tokens,
        "count": count,
        "max_model_len": state.server_args.tokenizer_model_max_length,
    }))
    .into_response()
}

async fn detokenize(
    State(state): State<Arc<AppState>>,
    body: Result<Json<DetokenizeRequest>, JsonRejection>,
) -> Response {
    let request = match body {
        Ok(Json(request)) => request,
        Err(error) => return openai_error(StatusCode::BAD_REQUEST, error.body_text(), false),
    };
    let (sequences, batched) = match request.tokens {
        OneOrMany::One(ids) => (vec![ids], false),
        OneOrMany::Many(ids) => (ids, true),
    };
    let mut texts = Vec::with_capacity(sequences.len());
    for token_ids in sequences {
        let kind = RequestKind::Detokenize {
            token_ids,
            skip_special_tokens: request.skip_special_tokens,
        };
        match service(&state, kind).await {
            Ok(ResponseItem::Data(bytes)) => match String::from_utf8(bytes.to_vec()) {
                Ok(text) => texts.push(text),
                Err(error) => {
                    return openai_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        error.to_string(),
                        false,
                    );
                }
            },
            Ok(_) => {
                return openai_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "unexpected detokenizer response",
                    false,
                );
            }
            Err(response) => return response,
        }
    }
    let text = if batched {
        json!(texts)
    } else {
        json!(texts[0])
    };
    Json(json!({"text": text})).into_response()
}

#[cfg(test)]
mod tests {
    use super::super::test_utils::{app_state, body_json, post_json};
    use super::*;
    use crate::runtime::Runnable;
    use crate::tokenizer_manager::{
        channel::to_scheduler,
        detokenizer::{DetokenizerBackend, DetokenizerWorker},
        to_scheduler::{Intake, Limits, MmDispatch},
        tokenizer::{TokenizerWorker, test_tokenizer},
        wiring::Senders,
    };

    #[tokio::test]
    async fn aliases_round_trip_batches_and_special_tokens_without_scheduler_work() {
        let (tm_tx, tm_rx) = flume::unbounded();
        let (tok_tx, tok_rx) = flume::unbounded();
        let (detok_tx, detok_rx) = flume::unbounded();
        let (lifecycle_tx, lifecycle_rx) = flume::unbounded();
        let (shutdown_tx, shutdown_rx) = flume::unbounded();
        let senders = Senders {
            tok_manager_tx: tm_tx.clone(),
            lifecycle_tx: lifecycle_tx.clone(),
            tokenizer_tx: tok_tx,
            detokenizer_tx: vec![detok_tx],
        };
        let state = app_state(senders.clone());
        let (scheduler_tx, scheduler_rx) = to_scheduler(8);
        let intake = Intake::new(
            tm_rx,
            lifecycle_rx,
            senders,
            scheduler_tx,
            Limits::from(state.server_args.as_ref()),
            MmDispatch {
                enabled: false,
                tx: flume::unbounded().0,
                results: Default::default(),
            },
            shutdown_rx,
            None,
        );
        let tokenizer = test_tokenizer();
        let detok = DetokenizerWorker::new(
            0,
            detok_rx,
            DetokenizerBackend::Dynamo {
                tokenizer: tokenizer.decoder(),
                vocab_size: Some(7),
            },
            lifecycle_tx,
        );
        let tok = TokenizerWorker::new(tok_rx, tm_tx, Arc::new(tokenizer));
        let threads = [
            std::thread::spawn(move || intake.run()),
            std::thread::spawn(move || tok.run()),
            std::thread::spawn(move || detok.run()),
        ];
        let cases = [
            (
                "/tokenize",
                json!({"prompt": "hello 世界"}),
                json!({"tokens": [0,3,5,1], "count": 4, "max_model_len": -1}),
            ),
            (
                "/v1/tokenize",
                json!({"prompt": ["hello 世界", ""], "add_special_tokens": false}),
                json!({"tokens": [[3,5],[]], "count": [2,0], "max_model_len": -1}),
            ),
            (
                "/tokenize",
                json!({"prompt": ""}),
                json!({"tokens": [0,1], "count": 2, "max_model_len": -1}),
            ),
            (
                "/v1/tokenize",
                json!({"prompt": []}),
                json!({"tokens": [], "count": [], "max_model_len": -1}),
            ),
            (
                "/detokenize",
                json!({"tokens": [0,3,5,1]}),
                json!({"text": "hello 世界"}),
            ),
            (
                "/v1/detokenize",
                json!({"tokens": [[0,3,5,1],[]], "skip_special_tokens": false}),
                json!({"text": ["<s> hello 世界 </s>", ""]}),
            ),
            ("/detokenize", json!({"tokens": []}), json!({"text": ""})),
        ];
        for (path, request, expected) in cases {
            let response = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                post_json(routes().with_state(state.clone()), path, request),
            )
            .await
            .unwrap();
            assert_eq!(response.status(), StatusCode::OK, "{path}");
            assert_eq!(body_json(response).await, expected, "{path}");
        }
        for (path, request) in [
            ("/tokenize", json!({})),
            ("/tokenize", json!({"prompt": "hello", "messages": []})),
            ("/tokenize", json!({"prompt": ["hello", 1]})),
            ("/detokenize", json!({"tokens": [-1]})),
            ("/detokenize", json!({"tokens": [1, [2]]})),
        ] {
            let response = post_json(routes().with_state(state.clone()), path, request).await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{path}");
        }
        assert!(
            scheduler_rx.drain(8).headers.is_empty(),
            "tokenizer services must not enqueue GPU requests"
        );
        drop(state);
        drop(shutdown_tx);
        for thread in threads {
            thread.join().unwrap();
        }
    }
}
