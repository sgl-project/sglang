//! `/add_external_corpus`, `/remove_external_corpus`, `/list_external_corpora`.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use sglang_api_types::api::v1 as api;

use super::app::AppState;
use crate::api_server::handlers::CorpusSource;

pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/add_external_corpus", post(add_external_corpus))
        .route("/remove_external_corpus", post(remove_external_corpus))
        .route("/list_external_corpora", get(list_external_corpora))
}

/// Python answers a bad body as `{"success": false, "message": ...}` 400,
/// and every outcome with 200 or 400 by its `success`.
fn failed(message: &str) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({"success": false, "message": message})),
    )
        .into_response()
}

fn by_success<T: serde::Serialize>(success: bool, body: T) -> Response {
    let status = if success {
        StatusCode::OK
    } else {
        StatusCode::BAD_REQUEST
    };
    (status, Json(body)).into_response()
}

async fn add_external_corpus(
    State(state): State<Arc<AppState>>,
    body: Result<Json<api::AddExternalCorpusRequest>, JsonRejection>,
) -> Response {
    let request = match body {
        Ok(Json(request)) => request,
        Err(rejection) => return failed(&rejection.body_text()),
    };
    let outcome = state
        .core
        .add_external_corpus(CorpusSource {
            corpus_id: request.corpus_id,
            file_path: request.file_path,
            documents: request.documents.map(|list| list.items),
            token_chunks: request
                .token_chunks
                .map(|list| list.items.into_iter().map(|ids| ids.ids).collect()),
        })
        .await;
    by_success(
        outcome.success,
        api::AddExternalCorpusResponse {
            success: outcome.success,
            corpus_id: outcome.corpus_id,
            message: outcome.message,
            loaded_token_count: outcome.loaded_token_count,
        },
    )
}

async fn remove_external_corpus(
    State(state): State<Arc<AppState>>,
    body: Result<Json<api::RemoveExternalCorpusRequest>, JsonRejection>,
) -> Response {
    let corpus_id = match body {
        Ok(Json(request)) => request.corpus_id.filter(|id| !id.is_empty()),
        Err(rejection) => return failed(&rejection.body_text()),
    };
    let Some(corpus_id) = corpus_id else {
        return failed("corpus_id is required.");
    };
    let outcome = state.core.remove_external_corpus(corpus_id).await;
    by_success(
        outcome.success,
        api::RemoveExternalCorpusResponse {
            success: outcome.success,
            message: outcome.message,
        },
    )
}

async fn list_external_corpora(State(state): State<Arc<AppState>>) -> Response {
    let listing = state.core.list_external_corpora().await;
    by_success(
        listing.success,
        api::ListExternalCorporaResponse {
            success: listing.success,
            corpus_token_counts: listing.corpus_token_counts.into_iter().collect(),
            message: listing.message,
        },
    )
}

#[cfg(test)]
mod tests {
    use axum::{
        body::{Body, to_bytes},
        http::{Request as HttpRequest, header::CONTENT_TYPE},
    };
    use tower::ServiceExt;

    use super::*;
    use crate::api_server::handlers::{CoreConfig, CoreHandle, CoreMetadata};
    use crate::message::config::ServerArgs;
    use crate::message::request::GenerationPolicy;

    /// Without NGRAM speculative decoding every route answers Python's
    /// `success: false` 400 with the reason, and a missing corpus id is the
    /// request's own 400.
    #[tokio::test]
    async fn corpus_routes_answer_python_shapes_without_ngram() {
        let server_args = Arc::new(ServerArgs::default());
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, _abort_rx) = flume::unbounded();
        let state = Arc::new(AppState {
            core: CoreHandle::new(
                intake_tx,
                abort_tx,
                CoreConfig {
                    response_capacity: 8,
                    response_activity: Default::default(),
                    startup_ready: true,
                    is_disaggregation: false,
                    mm_limits: Default::default(),
                    peer_endpoints: Vec::new(),
                    metadata: CoreMetadata::from(server_args.as_ref()),
                },
            ),
            generation_policy: GenerationPolicy::from(server_args.as_ref()),
            server_args,
            chat_formatter: None,
        });
        for (method, uri, body, message) in [
            (
                "POST",
                "/add_external_corpus",
                r#"{"documents": ["a"]}"#,
                "Ngram speculative decoding is not enabled.",
            ),
            (
                "POST",
                "/remove_external_corpus",
                r#"{"corpus_id": "c"}"#,
                "Ngram speculative decoding is not enabled.",
            ),
            (
                "POST",
                "/remove_external_corpus",
                "{}",
                "corpus_id is required.",
            ),
            (
                "GET",
                "/list_external_corpora",
                "",
                "Ngram speculative decoding is not enabled.",
            ),
        ] {
            let mut builder = HttpRequest::builder().method(method).uri(uri);
            if !body.is_empty() {
                builder = builder.header(CONTENT_TYPE, "application/json");
            }
            let response = routes()
                .with_state(state.clone())
                .oneshot(builder.body(Body::from(body)).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{uri}");
            let bytes = to_bytes(response.into_body(), 1 << 16).await.unwrap();
            let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(json["success"], false, "{uri}");
            assert_eq!(json["message"], message, "{uri}");
        }
        assert!(intake_rx.try_recv().is_err(), "nothing reaches a scheduler");
    }
}
