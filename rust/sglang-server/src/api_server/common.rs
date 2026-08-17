//! Common control-plane endpoints — `/server_info`, `/get_model_info`
//! (+ `/model_info` alias), plus the control-request submission path
//! (`await_control_result`, on the shared `submit`). Data-plane endpoints (incl. `/health*`,
//! which round-trips a generate probe) live in the sibling `native_api` and
//! `openai` modules; the shared `AppState` lives in the parent
//! `api_server` module.

use axum::{
    Json,
    Router,
    extract::{DefaultBodyLimit, FromRequestParts, State, rejection::JsonRejection},
    http::{StatusCode, header::AUTHORIZATION, request::Parts},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::Deserialize;

use super::AppState;
use super::guard::AbortGuard;
use super::submit::submit;
use crate::message::{
    AddExternalCorpusReqInput, ControlRequest, EgressItem, GetInternalStateReq,
    ListExternalCorporaReqInput, RemoveExternalCorpusReqInput, RequestKind,
};
use crate::runtime::ServerArgs;

// A signed-i32 token needs at most 11 decimal bytes. Even when every token is in
// its own chunk, brackets and separators keep compact JSON below 16 bytes/token.
// The fixed allowance covers field names and framing.
const EXTERNAL_CORPUS_ADD_BODY_BYTES_PER_TOKEN: usize = 16;
const EXTERNAL_CORPUS_ADD_BODY_FIXED_BYTES: usize = 1 << 20;

/// The routes this module owns, mounted by `api_server::serve`.
pub(super) fn routes(server_args: &ServerArgs) -> Router<AppState> {
    routes_with_external_body_limit(external_corpus_add_body_limit(server_args))
}

fn external_corpus_add_body_limit(server_args: &ServerArgs) -> usize {
    server_args
        .speculative_ngram_external_corpus_max_tokens
        .saturating_mul(EXTERNAL_CORPUS_ADD_BODY_BYTES_PER_TOKEN)
        .saturating_add(EXTERNAL_CORPUS_ADD_BODY_FIXED_BYTES)
}

fn routes_with_external_body_limit(add_body_limit: usize) -> Router<AppState> {
    Router::new()
        // Control-plane: reuses the ingress FSM (no tokenization), returns one
        // non-streamed JSON result. Adding one = a route line + its struct tag.
        .route("/server_info", get(server_info))
        // Static config, no scheduler round-trip. `/get_model_info` (+ `/model_info`
        // alias).
        .route("/get_model_info", get(model_info))
        .route("/model_info", get(model_info))
        .route(
            "/add_external_corpus",
            post(add_external_corpus).layer(DefaultBodyLimit::max(add_body_limit)),
        )
        .route("/remove_external_corpus", post(remove_external_corpus))
        .route("/list_external_corpora", get(list_external_corpora))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AddExternalCorpusBody {
    #[serde(default)]
    corpus_id: Option<String>,
    #[serde(default)]
    file_path: Option<String>,
    #[serde(default)]
    documents: Option<Vec<String>>,
    #[serde(default)]
    token_chunks: Option<Vec<Vec<i32>>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RemoveExternalCorpusBody {
    corpus_id: String,
}

/// Authentication guard for the Python endpoints' `ADMIN_OPTIONAL` level.
///
/// This is a parts extractor (and is the first handler argument), so an
/// unauthorized upload is rejected before Axum/serde reads or allocates its
/// request body.
struct ExternalCorpusAuth;

impl FromRequestParts<AppState> for ExternalCorpusAuth {
    type Rejection = Response;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &AppState,
    ) -> Result<Self, Self::Rejection> {
        if external_corpus_authorized(parts, &state.server_args) {
            Ok(Self)
        } else {
            Err((
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"error": "Unauthorized"})),
            )
                .into_response())
        }
    }
}

fn external_corpus_authorized(parts: &Parts, server_args: &ServerArgs) -> bool {
    // ADMIN_OPTIONAL precedence matches Python: an admin key, when configured,
    // supersedes the ordinary API key for these management endpoints.
    let expected = server_args
        .admin_api_key
        .as_deref()
        .filter(|key| !key.is_empty())
        .or_else(|| {
            server_args
                .api_key
                .as_deref()
                .filter(|key| !key.is_empty())
        });
    let Some(expected) = expected else {
        return true;
    };
    let Some(value) = parts
        .headers
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
    else {
        return false;
    };
    let Some((scheme, token)) = value.split_once(' ') else {
        return false;
    };
    scheme.eq_ignore_ascii_case("bearer")
        && constant_time_eq::constant_time_eq(token.as_bytes(), expected.as_bytes())
}

#[derive(Default, Deserialize)]
struct ExternalCorpusControlResult {
    success: bool,
    #[serde(default)]
    corpus_id: String,
    #[serde(default)]
    message: String,
    #[serde(default)]
    loaded_token_count: u64,
    #[serde(default)]
    corpus_token_counts: std::collections::BTreeMap<String, u64>,
}

enum ExternalCorpusResponseKind {
    Add,
    Remove,
    List,
}

fn external_corpus_response(
    value: ExternalCorpusControlResult,
    kind: ExternalCorpusResponseKind,
    status: StatusCode,
) -> Response {
    let public = match kind {
        ExternalCorpusResponseKind::Add => serde_json::json!({
            "success": value.success,
            "corpus_id": value.corpus_id,
            "message": value.message,
            "loaded_token_count": value.loaded_token_count,
        }),
        ExternalCorpusResponseKind::Remove => serde_json::json!({
            "success": value.success,
            "message": value.message,
        }),
        ExternalCorpusResponseKind::List => serde_json::json!({
            "success": value.success,
            "corpus_token_counts": value.corpus_token_counts,
            "message": value.message,
        }),
    };
    (status, Json(public)).into_response()
}

fn corpus_error(kind: ExternalCorpusResponseKind, message: impl Into<String>) -> Response {
    external_corpus_response(
        ExternalCorpusControlResult {
            message: message.into(),
            ..Default::default()
        },
        kind,
        StatusCode::BAD_REQUEST,
    )
}

fn corpus_json_rejection(kind: ExternalCorpusResponseKind, error: JsonRejection) -> Response {
    let status = error.status();
    external_corpus_response(
        ExternalCorpusControlResult {
            message: error.body_text(),
            ..Default::default()
        },
        kind,
        status,
    )
}

fn supports_dynamic_external_corpus(server_args: &ServerArgs) -> bool {
    server_args.pp_size == 1 && server_args.dp_size == 1
}

fn external_corpus_backend_error(
    state: &AppState,
    kind: ExternalCorpusResponseKind,
) -> Option<Response> {
    (state.server_args.speculative_algorithm.as_deref() != Some("NGRAM")).then(|| {
        corpus_error(kind, "Ngram speculative decoding is not enabled.")
    })
}

fn external_corpus_topology_error(
    state: &AppState,
    kind: ExternalCorpusResponseKind,
) -> Option<Response> {
    (!supports_dynamic_external_corpus(&state.server_args)).then(|| {
        corpus_error(
            kind,
            "dynamic external corpus management requires pp_size=1 and dp_size=1; use the startup external-corpus option for DP/PP deployments",
        )
    })
}

fn corpus_control_response(bytes: bytes::Bytes, kind: ExternalCorpusResponseKind) -> Response {
    match rmp_serde::from_slice::<ExternalCorpusControlResult>(&bytes) {
        // Match the Python endpoints: operational failures are public 400s,
        // while successful scheduler replies are 200s. Typed decoding also
        // strips internal BaseReq fields such as rid/http_worker_ipc.
        Ok(value) => {
            let status = if value.success {
                StatusCode::OK
            } else {
                StatusCode::BAD_REQUEST
            };
            external_corpus_response(value, kind, status)
        }
        Err(error) => {
            tracing::error!(%error, "invalid external-corpus control response");
            external_corpus_response(
                ExternalCorpusControlResult {
                    message: "bad external-corpus response".into(),
                    ..Default::default()
                },
                kind,
                StatusCode::INTERNAL_SERVER_ERROR,
            )
        }
    }
}

fn validate_corpus_id(corpus_id: &str) -> Result<(), &'static str> {
    if corpus_id
        .chars()
        .any(|ch| matches!(ch, '\t' | '\n' | '\r'))
    {
        return Err("corpus_id must not contain tabs or newlines");
    }
    Ok(())
}

fn validate_token_chunks(
    chunks: &[Vec<i32>],
    max_tokens: usize,
    vocab_size: u64,
) -> Result<(), String> {
    if chunks.is_empty() || chunks.iter().any(Vec::is_empty) {
        return Err("token_chunks must be non-empty and contain no empty chunks".into());
    }
    let total_tokens = chunks
        .iter()
        .try_fold(0usize, |total, chunk| total.checked_add(chunk.len()))
        .ok_or_else(|| "token_chunks token count overflow".to_string())?;
    if total_tokens > max_tokens {
        return Err(format!(
            "token_chunks total {total_tokens} exceeds external corpus token limit {max_tokens}"
        ));
    }
    for (chunk_index, chunk) in chunks.iter().enumerate() {
        for (token_index, &token) in chunk.iter().enumerate() {
            if token != i32::MIN && !(token >= 0 && (token as u64) < vocab_size) {
                return Err(format!(
                    "token_chunks[{chunk_index}][{token_index}]={token} is outside the model vocabulary"
                ));
            }
        }
    }
    Ok(())
}

async fn add_external_corpus(
    _auth: ExternalCorpusAuth,
    State(state): State<AppState>,
    body: Result<Json<AddExternalCorpusBody>, JsonRejection>,
) -> Response {
    let body = match body {
        Ok(Json(body)) => body,
        Err(error) => {
            return corpus_json_rejection(ExternalCorpusResponseKind::Add, error);
        }
    };
    if let Some(response) =
        external_corpus_backend_error(&state, ExternalCorpusResponseKind::Add)
    {
        return response;
    }
    if let Some(response) =
        external_corpus_topology_error(&state, ExternalCorpusResponseKind::Add)
    {
        return response;
    }
    if state.server_args.speculative_ngram_external_sam_budget == 0 {
        return corpus_error(
            ExternalCorpusResponseKind::Add,
            "dynamic external corpus loading is disabled; set --speculative-ngram-external-sam-budget to a positive value",
        );
    }
    if body.file_path.is_some() || body.documents.is_some() {
        return corpus_error(
            ExternalCorpusResponseKind::Add,
            "the Rust server accepts token_chunks only; tokenize file_path/documents client-side",
        );
    }
    let Some(chunks) = body.token_chunks else {
        return corpus_error(
            ExternalCorpusResponseKind::Add,
            "token_chunks must be provided",
        );
    };
    let vocab_size = state.server_args.model_config.vocab_size.unwrap_or(0);
    if let Err(message) = validate_token_chunks(
        &chunks,
        state.server_args.speculative_ngram_external_corpus_max_tokens,
        vocab_size,
    ) {
        return corpus_error(ExternalCorpusResponseKind::Add, message);
    }
    let corpus_id = body
        .corpus_id
        .filter(|id| !id.is_empty())
        .unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string());
    if let Err(message) = validate_corpus_id(&corpus_id) {
        return corpus_error(ExternalCorpusResponseKind::Add, message);
    }
    let control = ControlRequest::AddExternalCorpusReqInput(AddExternalCorpusReqInput::new(
        crate::ids::Rid::new().to_string(),
        corpus_id,
        chunks,
    ));
    match await_control_result(&state, control).await {
        Ok(bytes) => corpus_control_response(bytes, ExternalCorpusResponseKind::Add),
        Err(response) => response,
    }
}

async fn remove_external_corpus(
    _auth: ExternalCorpusAuth,
    State(state): State<AppState>,
    body: Result<Json<RemoveExternalCorpusBody>, JsonRejection>,
) -> Response {
    let body = match body {
        Ok(Json(body)) => body,
        Err(error) => {
            return corpus_json_rejection(ExternalCorpusResponseKind::Remove, error);
        }
    };
    if let Some(response) =
        external_corpus_backend_error(&state, ExternalCorpusResponseKind::Remove)
    {
        return response;
    }
    if let Some(response) =
        external_corpus_topology_error(&state, ExternalCorpusResponseKind::Remove)
    {
        return response;
    }
    if body.corpus_id.is_empty() {
        return corpus_error(
            ExternalCorpusResponseKind::Remove,
            "corpus_id is required",
        );
    }
    if let Err(message) = validate_corpus_id(&body.corpus_id) {
        return corpus_error(ExternalCorpusResponseKind::Remove, message);
    }
    let control = ControlRequest::RemoveExternalCorpusReqInput(RemoveExternalCorpusReqInput::new(
        crate::ids::Rid::new().to_string(),
        body.corpus_id,
    ));
    match await_control_result(&state, control).await {
        Ok(bytes) => corpus_control_response(bytes, ExternalCorpusResponseKind::Remove),
        Err(response) => response,
    }
}

async fn list_external_corpora(
    _auth: ExternalCorpusAuth,
    State(state): State<AppState>,
) -> Response {
    if let Some(response) =
        external_corpus_backend_error(&state, ExternalCorpusResponseKind::List)
    {
        return response;
    }
    let control = ControlRequest::ListExternalCorporaReqInput(ListExternalCorporaReqInput::new(
        crate::ids::Rid::new().to_string(),
    ));
    match await_control_result(&state, control).await {
        Ok(bytes) => corpus_control_response(bytes, ExternalCorpusResponseKind::List),
        Err(response) => response,
    }
}

/// Submit a control request through the ingress FSM (no tokenization) and await the
/// scheduler's single msgpack result (a `structs.asdict` named map). Returns the
/// raw bytes, or an error `Response` to return as-is.
async fn await_control_result(
    state: &AppState,
    control: ControlRequest,
) -> Result<bytes::Bytes, Response> {
    let (rid, mut rx) = submit(state, RequestKind::Control(Box::new(control)), false).await?;
    // Control requests register a detok entry like any other, and only
    // `handle_result` removes it — so a request that never produces one (a stalled
    // scheduler, a client that hangs up mid-await) leaves the entry behind. A
    // monitor polling `/server_info` then leaks one `DetokState` per poll, forever.
    // The guard deregisters on drop; it is disarmed below when the result lands.
    let mut guard = AbortGuard::new(state.senders.clone(), rid.clone());
    let received = rx.recv().await;
    if received.is_some() {
        guard.disarm(&rid); // completed normally — nothing to abort
    }
    match received {
        Some(EgressItem::Control(bytes)) => Ok(bytes),
        Some(EgressItem::Error(e)) => {
            let code =
                StatusCode::from_u16(e.http_status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            Err((code, e.to_string()).into_response())
        }
        // A control request never receives generation frames or service-call data.
        Some(EgressItem::Frame(_)) | Some(EgressItem::Done(_)) | Some(EgressItem::Data(_)) => {
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                "unexpected generation output for control request",
            )
                .into_response())
        }
        None => Err((StatusCode::from_u16(499).unwrap(), "request aborted").into_response()),
    }
}

/// `GET /get_model_info` (+ `/model_info` alias) — static model metadata from
/// `server_args` (no scheduler round-trip); `is_generation` always true.
async fn model_info(State(state): State<AppState>) -> Response {
    let sa = &state.server_args;
    let body = serde_json::json!({
        "model_path": sa.model_path,
        "tokenizer_path": sa.tokenizer_path,
        "is_generation": true,
        // Python's `TokenizerManager` merges this into every request
        // (`{**preferred, **client}`); this server has no equivalent yet, so
        // `RustServer.launch` REFUSES to start when it is set. It can therefore
        // only be null here — echoing it keeps the field's shape.
        "preferred_sampling_params": sa.preferred_sampling_params,
        "weight_version": serde_json::Value::Null,
    });
    (
        StatusCode::OK,
        [("content-type", "application/json")],
        serde_json::to_vec(&body).unwrap_or_default(),
    )
        .into_response()
}

/// `GET /server_info` — surface only an allowlist ([`INTERNAL_STATE_ALLOWLIST`] +
/// curated [`ServerArgs`] accessors), never the raw server-args dump (embeds
/// `api_key`/`admin_api_key`; see [`shape_server_info`]).
///
/// TODO(server_info): Python also includes `kv_events`; add once plumbed.
async fn server_info(State(state): State<AppState>) -> Response {
    let bytes = match await_control_result(
        &state,
        ControlRequest::GetInternalStateReq(GetInternalStateReq::new(
            crate::ids::Rid::new().to_string(),
        )),
    )
    .await
    {
        Ok(b) => b,
        Err(resp) => return resp,
    };
    match shape_server_info(&bytes, &state.server_args) {
        Ok(json) => (StatusCode::OK, [("content-type", "application/json")], json).into_response(),
        Err(e) => {
            tracing::error!(error = %e, "server_info: shaping failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "bad server_info response",
            )
                .into_response()
        }
    }
}

/// Runtime-metric keys `get_internal_state` adds atop the server-args dump. We copy
/// ONLY these out of `internal_state` (an allowlist), so the co-mingled
/// `api_key`/`admin_api_key` can never reach the response.
const INTERNAL_STATE_ALLOWLIST: &[&str] = &[
    "last_gen_throughput",
    "memory_usage",
    "effective_max_running_requests_per_dp",
    "avg_spec_accept_length",
    "step_time_dict",
];

fn shape_server_info(msgpack: &[u8], server_args: &ServerArgs) -> Result<Vec<u8>, String> {
    // GetInternalStateReqOutput asdict → `{ "internal_state": { server-args dump +
    // metrics }, ... }`. Pull that inner map out (it is NOT safe to expose whole).
    let mut obj: serde_json::Map<String, serde_json::Value> =
        rmp_serde::from_slice(msgpack).map_err(|e| e.to_string())?;
    let internal = match obj.remove("internal_state") {
        Some(serde_json::Value::Object(m)) => m,
        _ => serde_json::Map::new(),
    };

    // Copy only the allowlisted runtime metrics — never the raw server-args dump.
    let mut state_out = serde_json::Map::new();
    for &k in INTERNAL_STATE_ALLOWLIST {
        match internal.get(k) {
            Some(v) if !v.is_null() => {
                state_out.insert(k.to_string(), v.clone());
            }
            _ => {}
        }
    }

    // Top-level config is an explicit non-secret allowlist. Never serialize the
    // whole typed ServerArgs here: it also holds the API/admin keys used by the
    // management-route auth guard.
    let response = serde_json::json!({
        "model_path": server_args.model_path,
        "served_model_name": server_args.served_model_name,
        "tokenizer_path": server_args.tokenizer_path,
        "max_context_length": server_args.model_config.context_len,
        "max_total_num_tokens": server_args.max_total_num_tokens,
        "version": server_args.version,
        "internal_states": [serde_json::Value::Object(state_out)],
    });
    serde_json::to_vec(&response).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    use super::*;
    use crate::tokenizer_manager::Senders;

    fn senders() -> Senders {
        Senders {
            tm: flume::unbounded().0,
            abort: flume::unbounded().0,
            tok: flume::unbounded().0,
            detok: vec![],
        }
    }

    fn app_state(senders: Senders) -> AppState {
        AppState {
            senders,
            egress_buf: 8,
            server_args: Arc::new(
                ServerArgs::from_json(r#"{"speculative_algorithm":"NGRAM"}"#).unwrap(),
            ),
            chat_formatter: None,
            egress_activity: Default::default(),
        }
    }

    #[test]
    fn external_corpus_topology_requires_single_pp_and_dp() {
        let single = ServerArgs::from_json(r#"{"pp_size": 1, "dp_size": 1}"#).unwrap();
        let pp = ServerArgs::from_json(r#"{"pp_size": 2, "dp_size": 1}"#).unwrap();
        let dp = ServerArgs::from_json(r#"{"pp_size": 1, "dp_size": 2}"#).unwrap();
        assert!(supports_dynamic_external_corpus(&single));
        assert!(!supports_dynamic_external_corpus(&pp));
        assert!(!supports_dynamic_external_corpus(&dp));
    }

    #[tokio::test]
    async fn external_corpus_operational_failure_is_public_http_400() {
        let payload = rmp_serde::to_vec(&serde_json::json!({
            "success": false,
            "corpus_id": "",
            "message": "duplicate",
            "loaded_token_count": 0,
            "rid": "internal-rid",
            "http_worker_ipc": "internal-ipc"
        }))
        .unwrap();
        let payload = bytes::Bytes::from(payload);
        let response = corpus_control_response(payload.clone(), ExternalCorpusResponseKind::Add);
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["success"], false);
        assert_eq!(value["message"], "duplicate");
        assert!(value.get("rid").is_none());
        assert!(value.get("http_worker_ipc").is_none());
        assert_eq!(value.as_object().unwrap().len(), 4);

        let response = corpus_control_response(payload.clone(), ExternalCorpusResponseKind::Remove);
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value.as_object().unwrap().len(), 2);
        assert!(value.get("corpus_id").is_none());

        let response = corpus_control_response(payload, ExternalCorpusResponseKind::List);
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value.as_object().unwrap().len(), 3);
        assert_eq!(value["corpus_token_counts"], serde_json::json!({}));
    }

    #[test]
    fn external_corpus_auth_matches_admin_optional_precedence() {
        fn authorized(server_args: &ServerArgs, value: Option<&str>) -> bool {
            let mut request = Request::builder();
            if let Some(value) = value {
                request = request.header(AUTHORIZATION, value);
            }
            let (parts, ()) = request.body(()).unwrap().into_parts();
            external_corpus_authorized(&parts, server_args)
        }

        let none = ServerArgs::from_json("{}").unwrap();
        assert!(authorized(&none, None));

        let api = ServerArgs::from_json(r#"{"api_key":"api-secret"}"#).unwrap();
        assert!(!authorized(&api, None));
        assert!(authorized(&api, Some("Bearer api-secret")));
        assert!(authorized(&api, Some("bearer api-secret")));
        assert!(!authorized(&api, Some("Bearer wrong")));

        let both = ServerArgs::from_json(
            r#"{"api_key":"api-secret","admin_api_key":"admin-secret"}"#,
        )
        .unwrap();
        assert!(!authorized(&both, Some("Bearer api-secret")));
        assert!(authorized(&both, Some("Bearer admin-secret")));
    }

    #[tokio::test]
    async fn add_external_corpus_body_limit_overrides_global_disable() {
        let app = routes_with_external_body_limit(16)
            .layer(DefaultBodyLimit::disable())
            .with_state(app_state(senders()));
        let request = Request::builder()
            .method("POST")
            .uri("/add_external_corpus")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"token_chunks":[[1]]}"#))
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn external_corpus_auth_rejects_before_body_validation() {
        let mut state = app_state(senders());
        let server_args = ServerArgs::from_json(
            r#"{"api_key":"api-secret","speculative_algorithm":"NGRAM"}"#,
        )
        .unwrap();
        state.server_args = Arc::new(server_args);
        let app = routes(&state.server_args).with_state(state);
        let request = Request::builder()
            .method("POST")
            .uri("/add_external_corpus")
            .header("content-type", "application/json")
            .body(Body::from("not json"))
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn add_external_corpus_requires_positive_sam_budget() {
        let state = app_state(senders());
        let app = routes(&state.server_args).with_state(state);
        let request = Request::builder()
            .method("POST")
            .uri("/add_external_corpus")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"token_chunks":[[1]]}"#))
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn add_external_corpus_requires_ngram_backend() {
        let mut state = app_state(senders());
        state.server_args = Arc::new(ServerArgs::from_json("{}").unwrap());
        let app = routes(&state.server_args).with_state(state);
        let request = Request::builder()
            .method("POST")
            .uri("/add_external_corpus")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"token_chunks":[[1]]}"#))
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn external_corpus_body_limit_tracks_configured_token_limit() {
        let args = ServerArgs::from_json(
            r#"{"speculative_ngram_external_corpus_max_tokens":20000000}"#,
        )
        .unwrap();
        assert_eq!(
            external_corpus_add_body_limit(&args),
            20_000_000 * EXTERNAL_CORPUS_ADD_BODY_BYTES_PER_TOKEN
                + EXTERNAL_CORPUS_ADD_BODY_FIXED_BYTES
        );
    }

    #[test]
    fn exact_token_chunks_are_strictly_validated() {
        assert!(validate_token_chunks(&[vec![1, 2], vec![i32::MIN, 3]], 5, 4).is_ok());
        assert!(validate_token_chunks(&[], 5, 4).is_err());
        assert!(validate_token_chunks(&[Vec::new()], 5, 4).is_err());
        assert!(validate_token_chunks(&[vec![1, 2, 3]], 2, 4).is_err());
        assert!(validate_token_chunks(&[vec![-1]], 5, 4).is_err());
        assert!(validate_token_chunks(&[vec![4]], 5, 4).is_err());
        for body in [
            r#"{"token_chunks": [[true]]}"#,
            r#"{"token_chunks": [[1.0]]}"#,
            r#"{"token_chunks": [["1"]]}"#,
            r#"{"token_chunks": [[2147483648]]}"#,
            r#"{"token_chunks": [[1]], "unknown": true}"#,
        ] {
            assert!(serde_json::from_str::<AddExternalCorpusBody>(body).is_err());
        }
        assert!(validate_corpus_id("docs").is_ok());
        assert!(validate_corpus_id("bad\tname").is_err());
        assert!(validate_corpus_id("bad\nname").is_err());
    }

    /// The scheduler's `internal_state` embeds the full server-args dump (incl.
    /// `api_key`/`admin_api_key`). `/server_info` must surface only the allowlisted
    /// runtime metrics + curated config — never the secrets — and must not re-nest
    /// the dump under `internal_states[].internal_state`.
    #[test]
    fn shape_server_info_excludes_secrets_and_dump() {
        // GetInternalStateReqOutput.asdict → { "internal_state": { …dump+metrics… } }.
        let internal = rmpv::Value::Map(vec![
            (
                rmpv::Value::from("api_key"),
                rmpv::Value::from("secret-token"),
            ),
            (
                rmpv::Value::from("admin_api_key"),
                rmpv::Value::from("admin-token"),
            ),
            (rmpv::Value::from("model_path"), rmpv::Value::from("/m")),
            (
                rmpv::Value::from("last_gen_throughput"),
                rmpv::Value::from(1.5),
            ),
            (
                rmpv::Value::from("effective_max_running_requests_per_dp"),
                rmpv::Value::from(32),
            ),
        ]);
        let outer = rmpv::Value::Map(vec![(rmpv::Value::from("internal_state"), internal)]);
        let mut msgpack = Vec::new();
        rmpv::encode::write_value(&mut msgpack, &outer).unwrap();

        let sa =
            ServerArgs::from_json(r#"{"model_path": "/m", "api_key": "secret-token"}"#).unwrap();
        let out = shape_server_info(&msgpack, &sa).unwrap();
        let text = String::from_utf8(out.clone()).unwrap();
        // No secret leaks anywhere in the serialized response.
        assert!(!text.contains("secret-token"), "api_key leaked: {text}");
        assert!(
            !text.contains("admin-token"),
            "admin_api_key leaked: {text}"
        );

        let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
        // Allowlisted metric surfaced; the whole dump did not.
        let state0 = &v["internal_states"][0];
        assert_eq!(state0["last_gen_throughput"], 1.5);
        assert_eq!(state0["effective_max_running_requests_per_dp"], 32);
        assert!(
            state0.get("internal_state").is_none(),
            "must not re-nest the dump under internal_state"
        );
        assert!(state0.get("api_key").is_none());
        // Curated top-level config comes from typed accessors, not the dump.
        assert_eq!(v["model_path"], "/m");
    }
}
