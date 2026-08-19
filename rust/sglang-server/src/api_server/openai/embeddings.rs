//! OpenAI-compatible dense embedding endpoint backed by the existing Python
//! scheduler.

use std::collections::HashSet;
use std::sync::{Arc, LazyLock};

use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
};
use base64::Engine;
use dynamo_protocols::types::{
    CreateEmbeddingResponse, Embedding, EmbeddingInput, EmbeddingUsage, EmbeddingVector,
    EncodingFormat,
};
use futures::{StreamExt, stream::FuturesUnordered};
use serde::Deserialize;
use tokio::sync::mpsc;

use super::{AppState, openai_error};
use crate::api_server::guard::AbortGuard;
use crate::api_server::submit::submit;
use crate::message::ids::Rid;
use crate::message::request::{EmbeddingRequest as SchedulerEmbeddingRequest, RequestKind};
use crate::message::response::{EmbeddingEvent, ResponseItem};
use crate::message::types::OneOrMany;
use crate::utils::environ::env_u64;

static MAX_BATCH_REQS_PER_HTTP_REQ: LazyLock<usize> =
    LazyLock::new(|| env_u64("SGLANG_MAX_BATCH_REQS_PER_HTTP_REQ", 4096) as usize);
const MAX_RID_BROADCAST_BYTES: usize = 64 << 20;

pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new().route("/v1/embeddings", post(embeddings))
}

#[derive(Debug, Deserialize)]
struct MultimodalEmbeddingInput {
    #[allow(dead_code)]
    text: Option<String>,
    #[allow(dead_code)]
    image: Option<String>,
    #[allow(dead_code)]
    video: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EmbeddingRequestInput {
    OpenAi(EmbeddingInput),
    Multimodal(Vec<MultimodalEmbeddingInput>),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CreateEmbeddingRequest {
    input: EmbeddingRequestInput,
    #[serde(default = "default_model")]
    model: String,
    #[serde(default)]
    encoding_format: EncodingFormat,
    dimensions: Option<u32>,
    #[allow(dead_code)]
    user: Option<String>,
    rid: Option<OneOrMany<String>>,
    priority: Option<i64>,
    lora_path: Option<OneOrMany<Option<String>>>,
    embed_override_token_id: Option<i32>,
    embed_overrides: Option<Vec<Option<Vec<Vec<f32>>>>>,
}

fn default_model() -> String {
    "default".into()
}

#[derive(Debug)]
enum InputItem {
    Text(String),
    TokenIds(Vec<i32>),
}

struct ValidatedEmbeddingRequest {
    items: Vec<InputItem>,
    rids: Vec<Rid>,
    encoding_format: EncodingFormat,
    dimensions: Option<i64>,
    priority: Option<i64>,
}

async fn embeddings(
    State(state): State<Arc<AppState>>,
    body: Result<Json<CreateEmbeddingRequest>, JsonRejection>,
) -> Response {
    let Json(request) = match body {
        Ok(request) => request,
        Err(error) => {
            return openai_error(StatusCode::BAD_REQUEST, error.body_text(), false);
        }
    };
    let parsed = match validate_request(request, &state) {
        Ok(parsed) => parsed,
        Err((status, message)) => return openai_error(status, message, false),
    };

    let mut guard = AbortGuard::new_empty(state.senders.clone());
    let mut pending = FuturesUnordered::new();
    let count = parsed.items.len();
    for (index, (item, rid)) in parsed.items.into_iter().zip(parsed.rids).enumerate() {
        let (text, input_ids) = match item {
            InputItem::Text(text) => (Some(text), None),
            InputItem::TokenIds(ids) => (None, Some(ids)),
        };
        let request = SchedulerEmbeddingRequest::new(
            rid,
            text,
            input_ids,
            parsed.dimensions,
            parsed.priority,
        );
        let (rid, rx) = match submit(&state, RequestKind::Embedding(Box::new(request)), false).await
        {
            Ok(submitted) => submitted,
            Err(_) => {
                return openai_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "service unavailable",
                    false,
                );
            }
        };
        guard.arm(rid.clone());
        pending.push(wait_embedding(index, rid, rx));
    }

    let mut results: Vec<Option<EmbeddingEvent>> = (0..count).map(|_| None).collect();
    while let Some((index, rid, result)) = pending.next().await {
        guard.disarm(&rid);
        let event = match result {
            Ok(event) => event,
            Err((status, message)) => return openai_error(status, message, false),
        };
        if let Some((code, message)) = event
            .finish_reason
            .as_ref()
            .and_then(|reason| reason.abort_status())
        {
            let status = StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            return openai_error(status, message, false);
        }
        results[index] = Some(event);
    }

    let mut prompt_tokens = 0u32;
    let mut data = Vec::with_capacity(count);
    for (index, event) in results.into_iter().enumerate() {
        let Some(event) = event else {
            return openai_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "embedding response truncated before completion",
                false,
            );
        };
        prompt_tokens = match prompt_tokens.checked_add(event.prompt_tokens) {
            Some(total) => total,
            None => {
                return openai_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "embedding token usage overflowed the OpenAI response type",
                    false,
                );
            }
        };
        let embedding = match parsed.encoding_format {
            EncodingFormat::Float => EmbeddingVector::Float(event.embedding),
            EncodingFormat::Base64 => {
                let mut bytes = Vec::with_capacity(event.embedding.len() * 4);
                for value in event.embedding {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                EmbeddingVector::Base64(base64::engine::general_purpose::STANDARD.encode(bytes))
            }
        };
        let index = match u32::try_from(index) {
            Ok(index) => index,
            Err(_) => {
                return openai_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "embedding index overflowed the OpenAI response type",
                    false,
                );
            }
        };
        data.push(Embedding {
            object: "embedding".into(),
            embedding,
            index,
        });
    }
    Json(CreateEmbeddingResponse {
        object: "list".into(),
        data,
        model: state.server_args.served_model_name.clone(),
        usage: EmbeddingUsage {
            prompt_tokens,
            total_tokens: prompt_tokens,
        },
    })
    .into_response()
}

async fn wait_embedding(
    index: usize,
    rid: Rid,
    mut rx: mpsc::Receiver<ResponseItem>,
) -> (usize, Rid, Result<EmbeddingEvent, (StatusCode, String)>) {
    let result = loop {
        match rx.recv().await {
            Some(ResponseItem::Embedding(event)) => break Ok(event),
            Some(ResponseItem::Error(error)) => {
                let status = StatusCode::from_u16(error.http_status())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                break Err((status, error.to_string()));
            }
            Some(_) => continue,
            None => {
                break Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "embedding response truncated before completion".into(),
                ));
            }
        }
    };
    (index, rid, result)
}

fn validate_request(
    request: CreateEmbeddingRequest,
    state: &AppState,
) -> Result<ValidatedEmbeddingRequest, (StatusCode, String)> {
    if request.model != "default" && request.model != state.server_args.served_model_name {
        return Err((
            StatusCode::NOT_FOUND,
            format!("The model `{}` does not exist", request.model),
        ));
    }

    reject_unsupported_fields(&request)?;
    let (items, is_batch) = parse_input(request.input)?;
    if items.len() > *MAX_BATCH_REQS_PER_HTTP_REQ {
        return bad(format!(
            "batch size {} exceeds the maximum of {}",
            items.len(),
            *MAX_BATCH_REQS_PER_HTTP_REQ
        ));
    }
    let rids = normalize_embedding_rids(request.rid, items.len(), is_batch)?;

    Ok(ValidatedEmbeddingRequest {
        items,
        rids,
        encoding_format: request.encoding_format,
        dimensions: request.dimensions.map(i64::from),
        priority: request.priority,
    })
}

fn normalize_embedding_rids(
    rid: Option<OneOrMany<String>>,
    n: usize,
    is_batch: bool,
) -> Result<Vec<Rid>, (StatusCode, String)> {
    match rid {
        None => Ok((0..n).map(|_| Rid::default()).collect()),
        Some(OneOrMany::One(rid)) if !is_batch => Ok(vec![Rid::from_client(&rid)]),
        Some(OneOrMany::One(rid)) => {
            if rid.len().saturating_mul(n) > MAX_RID_BROADCAST_BYTES {
                return bad(format!(
                    "rid ({} bytes) broadcast to {n} inputs would exceed the \
                     {MAX_RID_BROADCAST_BYTES}-byte limit",
                    rid.len()
                ));
            }
            Ok((0..n)
                .map(|index| Rid::from_client(&format!("{rid}_{index}")))
                .collect())
        }
        Some(OneOrMany::Many(rids)) => {
            if !is_batch || rids.len() != n {
                return bad(format!(
                    "rid list length {} does not match batch size {n}",
                    rids.len()
                ));
            }
            let mut seen = HashSet::with_capacity(rids.len());
            let duplicates: Vec<&String> = rids.iter().filter(|rid| !seen.insert(*rid)).collect();
            if !duplicates.is_empty() {
                return bad(format!(
                    "duplicate request IDs detected within the request: {duplicates:?}"
                ));
            }
            Ok(rids.iter().map(|rid| Rid::from_client(rid)).collect())
        }
    }
}

fn reject_unsupported_fields(request: &CreateEmbeddingRequest) -> Result<(), (StatusCode, String)> {
    if request.lora_path.is_some() {
        return bad("LoRA embeddings are not supported by the Rust frontend");
    }
    if request.embed_override_token_id.is_some() || request.embed_overrides.is_some() {
        return bad("embed_overrides are not supported by the Rust frontend");
    }
    Ok(())
}

fn parse_input(
    input: EmbeddingRequestInput,
) -> Result<(Vec<InputItem>, bool), (StatusCode, String)> {
    match input {
        EmbeddingRequestInput::Multimodal(items) if items.is_empty() => {
            bad("input batch cannot be empty")
        }
        EmbeddingRequestInput::Multimodal(_) => bad("multimodal embedding input is not supported"),
        EmbeddingRequestInput::OpenAi(EmbeddingInput::String(text)) => {
            validate_text(&text)?;
            Ok((vec![InputItem::Text(text)], false))
        }
        EmbeddingRequestInput::OpenAi(EmbeddingInput::StringArray(texts)) => {
            if texts.is_empty() {
                return bad("input batch cannot be empty");
            }
            let items = texts
                .into_iter()
                .map(|text| {
                    validate_text(&text)?;
                    Ok(InputItem::Text(text))
                })
                .collect::<Result<_, _>>()?;
            Ok((items, true))
        }
        EmbeddingRequestInput::OpenAi(EmbeddingInput::IntegerArray(ids)) => {
            if ids.is_empty() {
                return bad("token ID input cannot be empty");
            }
            Ok((vec![InputItem::TokenIds(convert_token_ids(ids)?)], false))
        }
        EmbeddingRequestInput::OpenAi(EmbeddingInput::ArrayOfIntegerArray(id_lists)) => {
            if id_lists.is_empty() {
                return bad("input batch cannot be empty");
            }
            let items = id_lists
                .into_iter()
                .map(|ids| {
                    if ids.is_empty() {
                        return bad("token ID sequences cannot be empty");
                    }
                    convert_token_ids(ids).map(InputItem::TokenIds)
                })
                .collect::<Result<_, _>>()?;
            Ok((items, true))
        }
    }
}

fn convert_token_ids(ids: Vec<u32>) -> Result<Vec<i32>, (StatusCode, String)> {
    ids.into_iter()
        .map(|id| {
            i32::try_from(id).map_err(|_| {
                (
                    StatusCode::BAD_REQUEST,
                    format!("token ID {id} is out of range"),
                )
            })
        })
        .collect()
}

fn validate_text(text: &str) -> Result<(), (StatusCode, String)> {
    if text.trim().is_empty() {
        return bad("input strings cannot be empty or whitespace-only");
    }
    Ok(())
}

fn bad<T>(message: impl Into<String>) -> Result<T, (StatusCode, String)> {
    Err((StatusCode::BAD_REQUEST, message.into()))
}

#[cfg(test)]
mod tests {
    use super::super::routes;
    use super::super::test_utils::{app_state, body_json, post_json};
    use super::*;
    use crate::message::response::ResponseSink;
    use crate::tokenizer_manager::wiring::{AbortSource, Senders, TmEvent};
    use crate::utils::error::Error;
    use serde_json::json;

    fn live_senders() -> (
        Senders,
        flume::Receiver<TmEvent>,
        flume::Receiver<AbortSource>,
    ) {
        let (tm, tm_rx) = flume::unbounded();
        let (abort, abort_rx) = flume::unbounded();
        let (tok, _tok_rx) = flume::unbounded();
        (
            Senders {
                tok_manager_tx: tm,
                abort_tx: abort,
                tokenizer_tx: tok,
                detokenizer_tx: vec![],
            },
            tm_rx,
            abort_rx,
        )
    }

    #[test]
    fn input_shapes_are_unambiguous() {
        let input = |value| serde_json::from_value::<EmbeddingRequestInput>(value).unwrap();
        assert!(matches!(
            parse_input(input(json!([1, 2, 3]))).unwrap(),
            (items, false) if matches!(items.as_slice(), [InputItem::TokenIds(ids)] if ids == &[1, 2, 3])
        ));
        let (nested, nested_is_batch) = parse_input(input(json!([[1, 2], [3, 4]]))).unwrap();
        assert_eq!(nested.len(), 2);
        assert!(nested_is_batch);
        let (strings, strings_are_batch) = parse_input(input(json!(["hello", "world"]))).unwrap();
        assert_eq!(strings.len(), 2);
        assert!(strings_are_batch);
    }

    #[test]
    fn invalid_input_shapes_are_rejected() {
        for value in [json!(["ok", 1]), json!([1, [2]]), json!([-1])] {
            assert!(serde_json::from_value::<EmbeddingRequestInput>(value).is_err());
        }
        for value in [
            json!([]),
            json!("  "),
            json!([[1], []]),
            json!([{"image": "x"}]),
        ] {
            let input: EmbeddingRequestInput = serde_json::from_value(value).unwrap();
            assert!(parse_input(input).is_err());
        }
    }

    #[test]
    fn request_defaults_and_extensions_are_typed() {
        let request: CreateEmbeddingRequest = serde_json::from_value(json!({
            "input": "hello",
            "rid": "client-rid",
            "priority": 7,
        }))
        .unwrap();
        assert_eq!(request.model, "default");
        assert_eq!(request.encoding_format, EncodingFormat::Float);
        assert!(matches!(request.rid, Some(OneOrMany::One(rid)) if rid == "client-rid"));
        assert_eq!(request.priority, Some(7));

        assert!(
            serde_json::from_value::<CreateEmbeddingRequest>(json!({
                "input": "hello",
                "unknown": true,
            }))
            .is_err()
        );
    }

    #[test]
    fn request_rid_list_is_normalized_per_item() {
        let request: CreateEmbeddingRequest = serde_json::from_value(json!({
            "input": ["left", "right"],
            "rid": ["left-rid", "right-rid"],
        }))
        .unwrap();
        let state = app_state(live_senders().0);
        let validated = validate_request(request, &state).unwrap();
        assert_eq!(validated.rids[0].client_facing(), "left-rid");
        assert_eq!(validated.rids[1].client_facing(), "right-rid");
    }

    #[test]
    fn base64_is_little_endian_f32() {
        let values = [0.5f32, -2.0];
        let mut bytes = Vec::new();
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
        assert_eq!(
            base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .unwrap(),
            bytes
        );
    }

    #[tokio::test]
    async fn handler_preserves_order_usage_and_float_values() {
        let (senders, tm_rx, _abort_rx) = live_senders();
        let app = routes().with_state(app_state(senders));
        let responder = tokio::spawn(async move {
            let mut requests = Vec::new();
            for _ in 0..2 {
                let TmEvent::Intake(request) = tm_rx.recv_async().await.unwrap() else {
                    panic!("expected ingress")
                };
                let RequestKind::Embedding(embedding) = &request.kind else {
                    panic!("expected embedding request")
                };
                assert_eq!(embedding.dimensions, Some(64));
                requests.push(request);
            }
            assert_eq!(requests[0].rid.client_facing(), "embed-batch_0");
            assert_eq!(requests[1].rid.client_facing(), "embed-batch_1");
            // Deliberately complete in reverse scheduler order.
            for (value, tokens, request) in [
                (2.0, 5, requests.pop().unwrap()),
                (1.0, 3, requests.pop().unwrap()),
            ] {
                let event = EmbeddingEvent {
                    rid: request.rid,
                    embedding: vec![value, value + 0.5],
                    prompt_tokens: tokens,
                    finish_reason: None,
                };
                let ResponseSink::Local(sink) = request.sink;
                sink.send(ResponseItem::Embedding(event)).await.unwrap();
            }
        });
        let response = post_json(
            app,
            "/v1/embeddings",
            json!({
                "model": "model",
                "input": ["first", "second"],
                "dimensions": 64,
                "rid": "embed-batch",
            }),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body: CreateEmbeddingResponse =
            serde_json::from_value(body_json(response).await).unwrap();
        assert_eq!(body.model, "model");
        assert_eq!(body.data[0].index, 0);
        assert_eq!(
            body.data[0].embedding,
            EmbeddingVector::Float(vec![1.0, 1.5])
        );
        assert_eq!(body.data[1].index, 1);
        assert_eq!(
            body.data[1].embedding,
            EmbeddingVector::Float(vec![2.0, 2.5])
        );
        assert_eq!(body.usage.prompt_tokens, 8);
        assert_eq!(body.usage.total_tokens, 8);
        responder.await.unwrap();
    }

    #[tokio::test]
    async fn handler_base64_and_validation_errors() {
        let (senders, tm_rx, _abort_rx) = live_senders();
        let app = routes().with_state(app_state(senders));
        tokio::spawn(async move {
            let TmEvent::Intake(request) = tm_rx.recv_async().await.unwrap() else {
                panic!("expected ingress")
            };
            let event = EmbeddingEvent {
                rid: request.rid,
                embedding: vec![0.5, -2.0],
                prompt_tokens: 2,
                finish_reason: None,
            };
            let ResponseSink::Local(sink) = request.sink;
            sink.send(ResponseItem::Embedding(event)).await.unwrap();
        });
        let response = post_json(
            app.clone(),
            "/v1/embeddings",
            json!({"model": "model", "input": [1, 2], "encoding_format": "base64"}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body: CreateEmbeddingResponse =
            serde_json::from_value(body_json(response).await).unwrap();
        let EmbeddingVector::Base64(encoded) = &body.data[0].embedding else {
            panic!("expected base64 embedding")
        };
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .unwrap();
        assert_eq!(&decoded[..4], &0.5f32.to_le_bytes());
        assert_eq!(&decoded[4..], &(-2.0f32).to_le_bytes());

        for invalid in [
            json!({"model": "model", "input": " "}),
            json!({"model": "model", "input": ["ok", 1]}),
            json!({"model": "model", "input": [1, [2]]}),
            json!({"model": "model", "input": "ok", "encoding_format": "hex"}),
            json!({"model": "model", "input": "ok", "dimensions": -1}),
            json!({"model": "model", "input": "ok", "return_pooled_hidden_states": true}),
            json!({"model": "model", "input": [{"text": "ok"}]}),
            json!({"model": "model", "input": "ok", "lora_path": "adapter"}),
            json!({"model": "model", "input": ["a", "b"], "rid": ["only-one"]}),
            json!({"model": "model", "input": ["a", "b"], "rid": ["same", "same"]}),
            json!({"model": "model", "input": "ok", "unknown": true}),
        ] {
            let response = post_json(app.clone(), "/v1/embeddings", invalid).await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            assert!(body_json(response).await["error"].is_object());
        }
        let unknown = post_json(
            app,
            "/v1/embeddings",
            json!({"model": "unknown", "input": "ok"}),
        )
        .await;
        assert_eq!(unknown.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn scheduler_error_aborts_other_submitted_items() {
        let (senders, tm_rx, abort_rx) = live_senders();
        let app = routes().with_state(app_state(senders));
        let responder = tokio::spawn(async move {
            let TmEvent::Intake(first) = tm_rx.recv_async().await.unwrap() else {
                panic!("expected ingress")
            };
            let TmEvent::Intake(second) = tm_rx.recv_async().await.unwrap() else {
                panic!("expected ingress")
            };
            let second_rid = second.rid.clone();
            let ResponseSink::Local(sink) = first.sink;
            sink.send(ResponseItem::Error(Error::Internal(
                "scheduler failed".into(),
            )))
            .await
            .unwrap();
            second_rid
        });
        let response = post_json(
            app,
            "/v1/embeddings",
            json!({"model": "model", "input": ["first", "second"]}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let second_rid = responder.await.unwrap();
        let aborted = abort_rx.recv_async().await.unwrap();
        assert!(matches!(aborted, AbortSource::Guard(rid) if rid == second_rid));
    }

    #[tokio::test]
    async fn client_disconnect_aborts_submitted_embedding() {
        let (senders, tm_rx, abort_rx) = live_senders();
        let app = routes().with_state(app_state(senders));
        let handler = tokio::spawn(async move {
            post_json(
                app,
                "/v1/embeddings",
                json!({"model": "model", "input": "hello"}),
            )
            .await
        });

        let TmEvent::Intake(request) = tm_rx.recv_async().await.unwrap() else {
            panic!("expected ingress")
        };
        let rid = request.rid;
        handler.abort();
        assert!(handler.await.unwrap_err().is_cancelled());

        let aborted =
            tokio::time::timeout(std::time::Duration::from_secs(1), abort_rx.recv_async())
                .await
                .expect("disconnect must trigger abort")
                .unwrap();
        assert!(matches!(aborted, AbortSource::Guard(aborted) if aborted == rid));
    }
}
