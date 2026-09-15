//! Common control-plane endpoints — `/server_info`, `/get_model_info`
//! (+ `/model_info` alias), plus the control-request submission path
//! (`await_control_result`, on the shared `submit`). Data-plane endpoints (incl. `/health*`,
//! which round-trips a generate probe) live in the sibling `native_api` and
//! `openai` modules; the shared `AppState` lives in the parent
//! `api_server` module.

use axum::{
    Json, Router,
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use std::sync::Arc;
use std::time::Duration;

use serde::Deserialize;

use super::app::AppState;
use super::guard::AbortGuard;
use super::submit::submit;
use crate::message::config::ServerArgs;
use crate::message::ids::Rid;
use crate::message::io_struct::{
    ClearHiCacheReqInput, ControlRequest, FlushCacheReqInput, GetInternalStateReq,
};
use crate::message::request::RequestKind;
use crate::message::response::ResponseItem;
use crate::tokenizer_manager::wiring::TmEvent;

/// The routes this module owns, mounted by `api_server::serve`.
pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        // Control-plane: reuses the request FSM (no tokenization), returns one
        // non-streamed JSON result. Adding one = a route line + its struct tag.
        .route("/server_info", get(server_info))
        .route("/get_server_info", get(server_info))
        .route("/flush_cache", get(flush_cache).post(flush_cache))
        .route(
            "/hicache/storage-backend/clear",
            post(clear_hicache_storage),
        )
        .route(
            "/clear_hicache_storage_backend",
            get(clear_hicache_storage_deprecated).post(clear_hicache_storage_deprecated),
        )
        .route("/abort_request", post(abort_request))
        // Static config, no scheduler round-trip. `/get_model_info` (+ `/model_info`
        // alias).
        .route("/get_model_info", get(model_info))
        .route("/model_info", get(model_info))
}

/// Submit a control request through the request FSM (no tokenization) and await the
/// scheduler's single msgpack result (a `structs.asdict` named map). Returns the
/// raw bytes, or an error `Response` to return as-is.
async fn await_control_result(
    state: &AppState,
    control: ControlRequest,
    timeout: Duration,
) -> Result<Vec<bytes::Bytes>, Response> {
    let Some(deadline) = tokio::time::Instant::now().checked_add(timeout) else {
        return Err((StatusCode::BAD_REQUEST, "timeout is too large").into_response());
    };
    let timeout_response = || {
        (
            StatusCode::GATEWAY_TIMEOUT,
            "scheduler control request timed out",
        )
            .into_response()
    };
    let rid: Rid = control.rid().into();
    // Control requests register a detok entry like any other, and only
    // `handle_result` removes it — so a request that never produces one (a stalled
    // scheduler, a client that hangs up mid-await) leaves the entry behind. A
    // monitor polling `/server_info` then leaks one `DetokState` per poll, forever.
    // The guard deregisters on drop; it is disarmed below when the result lands.
    let mut guard = AbortGuard::new(state.senders.clone(), rid.clone());
    let (_, mut rx) = tokio::time::timeout_at(
        deadline,
        submit(state, RequestKind::Control(Box::new(control)), false),
    )
    .await
    .map_err(|_| timeout_response())??;
    let received = tokio::time::timeout_at(deadline, rx.recv())
        .await
        .map_err(|_| timeout_response())?;
    if received.is_some() {
        guard.disarm(&rid); // completed normally — nothing to abort
    }
    match received {
        Some(ResponseItem::Control(bytes)) => Ok(bytes),
        Some(ResponseItem::Error(e)) => {
            let code =
                StatusCode::from_u16(e.http_status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            Err((code, e.to_string()).into_response())
        }
        // A control request never receives generation frames or service-call data.
        Some(ResponseItem::Frame(_))
        | Some(ResponseItem::Done(_))
        | Some(ResponseItem::Tokenized(_))
        | Some(ResponseItem::Data(_)) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "unexpected generation output for control request",
        )
            .into_response()),
        None => Err((StatusCode::from_u16(499).unwrap(), "request aborted").into_response()),
    }
}

/// `GET /get_model_info` (+ `/model_info` alias) — static model metadata from
/// `server_args` (no scheduler round-trip).
async fn model_info(State(state): State<Arc<AppState>>) -> Response {
    let sa = &state.server_args;
    let body = serde_json::json!({
        "model_path": sa.model_path,
        "served_model_name": sa.served_model_name,
        "tokenizer_path": sa.public_tokenizer_path,
        "is_generation": sa.model_config.is_generation,
        "has_image_understanding": sa.model_config.has_image_understanding,
        "has_audio_understanding": sa.model_config.has_audio_understanding,
        "model_type": sa.model_config.model_type,
        "architectures": sa.model_config.architectures,
        "preferred_sampling_params": sa.preferred_sampling_params,
        // Python answers this through `config_value`, so a control-plane write
        // moves it there; here it is the launch value.
        "weight_version": sa.weight_version,
        "load_format": sa.load_format,
        // `auto` never reaches the blob: `resolve_auto_parsers` writes the
        // selected parser into `server_args` before the scheduler forks.
        "reasoning_parser": sa.reasoning_parser,
        "tool_call_parser": sa.tool_call_parser,
    });
    Json(body).into_response()
}

/// `GET /server_info` — surface only an allowlist ([`INTERNAL_STATE_ALLOWLIST`] +
/// curated [`ServerArgs`] accessors), never the raw server-args dump (embeds
/// `api_key`/`admin_api_key`; see [`shape_server_info`]).
///
/// TODO(server_info): Python also includes `kv_events`; add once plumbed.
async fn server_info(State(state): State<Arc<AppState>>) -> Response {
    let bytes = match await_control_result(
        &state,
        ControlRequest::GetInternalStateReq(GetInternalStateReq::new(Rid::new().to_string())),
        Duration::from_secs(30),
    )
    .await
    {
        Ok(b) => b,
        Err(resp) => return resp,
    };
    match shape_server_infos(&bytes, &state.server_args) {
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

#[derive(Default, Deserialize)]
struct FlushCacheQuery {
    #[serde(default)]
    timeout: f64,
}

#[derive(Deserialize)]
struct CacheControlResult {
    success: bool,
    #[serde(default)]
    message: String,
}

#[derive(Default, Deserialize)]
struct AbortRequest {
    #[serde(default)]
    rid: Option<String>,
    #[serde(default)]
    abort_all: bool,
}

pub(super) const ABORT_DISPATCHED_HEADER: &str = "x-sglang-abort-dispatched";

async fn abort_request(
    State(state): State<Arc<AppState>>,
    body: Result<Json<AbortRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let request = match body {
        Ok(Json(request)) => request,
        Err(error) => return (StatusCode::BAD_REQUEST, error.body_text()).into_response(),
    };
    let (reply, received) = tokio::sync::oneshot::channel();
    let cancel = async {
        state
            .senders
            .tok_manager_tx
            .send_async(TmEvent::Abort {
                rid_prefix: request.rid.unwrap_or_default(),
                abort_all: request.abort_all,
                reply,
            })
            .await
            .map_err(|_| ())?;
        received.await.map_err(|_| ())
    };
    match tokio::time::timeout(Duration::from_secs(30), cancel).await {
        Ok(Ok(dispatched)) => {
            if state.server_args.dp_size > 1 {
                // The public DP listener owns the operation counter. An abort
                // can match many workers, but is still one client operation.
                (
                    [(ABORT_DISPATCHED_HEADER, if dispatched { "1" } else { "0" })],
                    StatusCode::OK,
                )
                    .into_response()
            } else {
                if dispatched && let Some(metrics) = &state.frontend_metrics {
                    metrics.aborted();
                }
                StatusCode::OK.into_response()
            }
        }
        Ok(Err(())) => (
            StatusCode::SERVICE_UNAVAILABLE,
            "request intake is unavailable",
        )
            .into_response(),
        Err(_) => (
            StatusCode::GATEWAY_TIMEOUT,
            "request cancellation timed out",
        )
            .into_response(),
    }
}

async fn flush_cache(
    State(state): State<Arc<AppState>>,
    query: Result<Query<FlushCacheQuery>, axum::extract::rejection::QueryRejection>,
) -> Response {
    let timeout_s = match query {
        Ok(Query(query)) if query.timeout.is_finite() && query.timeout >= 0.0 => query.timeout,
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                "timeout must be a finite non-negative number",
            )
                .into_response();
        }
    };
    let Some(timeout) = Duration::try_from_secs_f64(timeout_s)
        .ok()
        .and_then(|duration| duration.checked_add(Duration::from_secs(30)))
    else {
        return (StatusCode::BAD_REQUEST, "timeout is too large").into_response();
    };
    let control = ControlRequest::FlushCacheReqInput(FlushCacheReqInput::new(
        Rid::new().to_string(),
        timeout_s,
    ));
    let bytes = match await_control_result(&state, control, timeout).await {
        Ok(bytes) => bytes,
        Err(response) => return response,
    };
    let results = bytes
        .iter()
        .map(|bytes| rmp_serde::from_slice::<CacheControlResult>(bytes))
        .collect::<Result<Vec<_>, _>>();
    match results {
        Ok(results) => match results.into_iter().find(|result| !result.success) {
            None => (StatusCode::OK, "Cache flushed.\nPlease check backend logs for more details. (When there are running or waiting requests, the operation will not be performed.)\n").into_response(),
            Some(result) => (StatusCode::BAD_REQUEST, if result.message.is_empty() { "Flush cache failed.\n".to_owned() } else { result.message }).into_response(),
        },
        Err(error) => {
            tracing::error!(%error, "invalid flush_cache response");
            (StatusCode::INTERNAL_SERVER_ERROR, "invalid flush_cache response").into_response()
        }
    }
}

async fn clear_hicache_storage(State(state): State<Arc<AppState>>) -> Response {
    clear_hicache_storage_impl(&state, false).await
}

async fn clear_hicache_storage_deprecated(State(state): State<Arc<AppState>>) -> Response {
    clear_hicache_storage_impl(&state, true).await
}

async fn clear_hicache_storage_impl(state: &AppState, deprecated: bool) -> Response {
    let control =
        ControlRequest::ClearHiCacheReqInput(ClearHiCacheReqInput::new(Rid::new().to_string()));
    let bytes = match await_control_result(state, control, Duration::from_secs(30)).await {
        Ok(bytes) => bytes,
        Err(response) => return response,
    };
    let results = bytes
        .iter()
        .map(|bytes| rmp_serde::from_slice::<CacheControlResult>(bytes))
        .collect::<Result<Vec<_>, _>>();
    let status = match results {
        Ok(results) if results.iter().all(|result| result.success) => StatusCode::OK,
        Ok(_) => StatusCode::BAD_REQUEST,
        Err(error) => {
            tracing::error!(%error, "invalid clear_hicache_storage response");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "invalid clear_hicache_storage response",
            )
                .into_response();
        }
    };
    let body = if deprecated {
        "Deprecated endpoint. Use POST /hicache/storage-backend/clear.\nHierarchical cache storage backend cleared.\n"
    } else {
        "Hierarchical cache storage backend cleared.\n"
    };
    (status, body).into_response()
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
    "startup_time",
    "dspark_info_record",
];

/// Public configuration used by cache, speculative-decoding, and topology
/// clients. The scheduler snapshot also contains credentials and backend
/// connection options, so additions must be explicit.
const PUBLIC_CONFIG_FIELDS: &[&str] = &[
    "dtype",
    "quantization",
    "kv_cache_dtype",
    "device",
    "context_length",
    "is_embedding",
    "tp_size",
    "pp_size",
    "dp_size",
    "ep_size",
    "world_size",
    "enable_dp_attention",
    "enable_dp_attention_local_control_broadcast",
    "disaggregation_mode",
    "max_running_requests",
    "max_total_tokens",
    "chunked_prefill_size",
    "page_size",
    "disable_radix_cache",
    "enable_deterministic_inference",
    "grammar_backend",
    "enable_hierarchical_cache",
    "hicache_ratio",
    "hicache_size",
    "hicache_write_policy",
    "hicache_io_backend",
    "hicache_mem_layout",
    "hicache_storage_backend",
    "hicache_host_memory_mode",
    "speculative_algorithm",
    "speculative_num_steps",
    "speculative_eagle_topk",
    "speculative_num_draft_tokens",
    "speculative_accept_threshold_single",
    "speculative_accept_threshold_acc",
    "enable_metrics",
    "enable_metrics_for_all_schedulers",
];

fn shape_server_infos(
    replies: &[bytes::Bytes],
    server_args: &ServerArgs,
) -> Result<Vec<u8>, String> {
    let mut response = None;
    let mut states = Vec::with_capacity(replies.len());
    for reply in replies {
        let mut shaped = shape_server_info_value(reply, server_args)?;
        let Some(serde_json::Value::Array(rank_states)) = shaped.get_mut("internal_states") else {
            return Err("missing internal_states".into());
        };
        states.append(rank_states);
        response.get_or_insert(shaped);
    }
    let mut response = response.ok_or("missing scheduler replies")?;
    response["internal_states"] = states.into();
    serde_json::to_vec(&response).map_err(|e| e.to_string())
}

fn shape_server_info_value(
    msgpack: &[u8],
    server_args: &ServerArgs,
) -> Result<serde_json::Value, String> {
    // GetInternalStateReqOutput asdict → `{ "internal_state": { server-args dump +
    // metrics }, ... }`. Pull that inner map out (it is NOT safe to expose whole).
    let mut obj: serde_json::Map<String, serde_json::Value> =
        rmp_serde::from_slice(msgpack).map_err(|e| e.to_string())?;
    let internal = match obj.remove("internal_state") {
        Some(serde_json::Value::Object(m)) => m,
        _ => return Err("scheduler response is missing its internal_state object".into()),
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

    // Top-level non-secret config from typed accessors (structurally can't surface
    // a key field, unlike the raw dump).
    let mut response = serde_json::json!({
        "model_path": server_args.model_path,
        "served_model_name": server_args.served_model_name,
        "tokenizer_path": server_args.public_tokenizer_path,
        "max_context_length": server_args.model_config.context_len,
        "max_total_num_tokens": server_args.max_total_num_tokens,
        "version": server_args.version,
        "internal_states": [serde_json::Value::Object(state_out)],
        "frontend": "rust",
        "enable_http2": server_args.enable_http2,
        "http2_max_concurrent_streams": server_args.http2_max_concurrent_streams,
        "http2_initial_connection_window_size": server_args.http2_initial_connection_window_size,
    });
    for &key in PUBLIC_CONFIG_FIELDS {
        if let Some(value) = internal.get(key) {
            response[key] = value.clone();
            response["internal_states"][0][key] = value.clone();
        }
    }
    if let Some(startup_time) = internal.get("startup_time") {
        response["startup_time"] = startup_time.clone();
    }
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer_manager::wiring::{LifecycleEvent, Senders};
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    fn test_state() -> (
        Arc<AppState>,
        flume::Receiver<TmEvent>,
        flume::Receiver<LifecycleEvent>,
    ) {
        let (tm_tx, tm_rx) = flume::unbounded();
        let (lifecycle_tx, lifecycle_rx) = flume::unbounded();
        let state = Arc::new(AppState {
            senders: Senders {
                tok_manager_tx: tm_tx,
                lifecycle_tx,
                tokenizer_tx: flume::unbounded().0,
                detokenizer_tx: vec![],
            },
            server_args: Arc::new(ServerArgs::default()),
            response_buf: 8,
            chat_formatter: None,
            http_extension: None,
            response_activity: Default::default(),
            startup_ready: Arc::new(true.into()),
            frontend_metrics: None,
        });
        (state, tm_rx, lifecycle_rx)
    }

    #[tokio::test]
    async fn storage_clear_preserves_methods_and_requires_success_from_every_rank() {
        for (path, method, deprecated) in [
            ("/hicache/storage-backend/clear", "POST", false),
            ("/clear_hicache_storage_backend", "GET", true),
            ("/clear_hicache_storage_backend", "POST", true),
        ] {
            for replies in [
                serde_json::json!([{"success": true}]),
                serde_json::json!([{"success": true}, {"success": true}]),
                serde_json::json!([{"success": false}, {"success": true}]),
                serde_json::json!([{"success": true}, {"success": false}]),
                serde_json::json!([{"success": true}, {"other": true}]),
            ] {
                let (state, requests, _lifecycle) = test_state();
                let app = routes().with_state(state);
                let task = tokio::spawn(
                    app.oneshot(
                        Request::builder()
                            .method(method)
                            .uri(path)
                            .body(Body::empty())
                            .unwrap(),
                    ),
                );
                let TmEvent::Intake(request) = requests.recv_async().await.unwrap() else {
                    panic!("control request");
                };
                let RequestKind::Control(control) = request.kind else {
                    panic!("control request");
                };
                let wire: Vec<rmpv::Value> =
                    rmp_serde::from_slice(&control.encode().unwrap()).unwrap();
                assert_eq!(wire.len(), 3);
                assert_eq!(wire[0].as_str(), Some("ClearHiCacheReqInput"));
                assert_eq!(wire[1].as_str(), Some(request.rid.as_str()));
                assert!(wire[2].is_nil());
                let replies = replies.as_array().unwrap();
                request
                    .sink
                    .try_send(ResponseItem::Control(
                        replies
                            .iter()
                            .map(|reply| rmp_serde::to_vec_named(reply).unwrap().into())
                            .collect(),
                    ))
                    .unwrap();
                let response = task.await.unwrap().unwrap();
                let invalid = replies.iter().any(|reply| reply["success"].is_null());
                let status = if invalid {
                    StatusCode::INTERNAL_SERVER_ERROR
                } else if replies.iter().all(|reply| reply["success"] == true) {
                    StatusCode::OK
                } else {
                    StatusCode::BAD_REQUEST
                };
                assert_eq!(response.status(), status, "{path} {replies:?}");
                let body = axum::body::to_bytes(response.into_body(), 4096)
                    .await
                    .unwrap();
                let expected = if invalid {
                    "invalid clear_hicache_storage response"
                } else if deprecated {
                    "Deprecated endpoint. Use POST /hicache/storage-backend/clear.\nHierarchical cache storage backend cleared.\n"
                } else {
                    "Hierarchical cache storage backend cleared.\n"
                };
                assert_eq!(body, expected);
            }
        }
        let (state, requests, _lifecycle) = test_state();
        let response = routes()
            .with_state(state)
            .oneshot(
                Request::builder()
                    .uri("/hicache/storage-backend/clear")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
        assert!(requests.try_recv().is_err());
    }

    #[tokio::test]
    async fn cache_flush_encodes_timeout_and_surfaces_scheduler_results() {
        for (method, timeout, reply, status, text) in [
            (
                "GET",
                "0",
                serde_json::json!({"success": true}),
                StatusCode::OK,
                "Cache flushed.",
            ),
            (
                "POST",
                "0.5",
                serde_json::json!({"success": false, "message": "Busy"}),
                StatusCode::BAD_REQUEST,
                "Busy",
            ),
            (
                "POST",
                "30",
                serde_json::json!({"success": false}),
                StatusCode::BAD_REQUEST,
                "Flush cache failed.",
            ),
            (
                "GET",
                "0",
                serde_json::json!({"other": true}),
                StatusCode::INTERNAL_SERVER_ERROR,
                "invalid flush_cache response",
            ),
        ] {
            let (state, requests, _lifecycle) = test_state();
            let app = routes().with_state(state);
            let task = tokio::spawn(
                app.oneshot(
                    Request::builder()
                        .method(method)
                        .uri(format!("/flush_cache?timeout={timeout}"))
                        .body(Body::empty())
                        .unwrap(),
                ),
            );
            let TmEvent::Intake(request) = requests.recv_async().await.unwrap() else {
                panic!("control request");
            };
            let RequestKind::Control(control) = request.kind else {
                panic!("control request");
            };
            let wire: Vec<rmpv::Value> = rmp_serde::from_slice(&control.encode().unwrap()).unwrap();
            assert_eq!(wire.len(), 4);
            assert_eq!(wire[0].as_str(), Some("FlushCacheReqInput"));
            assert_eq!(wire[1].as_str(), Some(request.rid.as_str()));
            assert!(wire[2].is_nil());
            assert_eq!(wire[3].as_f64(), Some(timeout.parse::<f64>().unwrap()));
            request
                .sink
                .try_send(ResponseItem::Control(vec![
                    rmp_serde::to_vec_named(&reply).unwrap().into(),
                ]))
                .unwrap();
            let response = task.await.unwrap().unwrap();
            assert_eq!(response.status(), status);
            let bytes = axum::body::to_bytes(response.into_body(), 4096)
                .await
                .unwrap();
            assert!(std::str::from_utf8(&bytes).unwrap().starts_with(text));
        }
        let (state, requests, _lifecycle) = test_state();
        for timeout in ["-1", "NaN", "inf", "no", "1e100"] {
            let response = routes()
                .with_state(state.clone())
                .oneshot(
                    Request::builder()
                        .uri(format!("/flush_cache?timeout={timeout}"))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        }
        assert!(
            requests.try_recv().is_err(),
            "invalid queries are never scheduled"
        );
    }

    #[tokio::test]
    async fn expired_control_request_cleans_up_and_rejects_late_replies() {
        let (state, requests, lifecycle) = test_state();
        let task = tokio::spawn(async move {
            await_control_result(
                &state,
                ControlRequest::GetInternalStateReq(GetInternalStateReq::new("expired".into())),
                Duration::from_millis(1),
            )
            .await
        });
        let TmEvent::Intake(request) = requests.recv_async().await.unwrap() else {
            panic!("control request");
        };
        assert_eq!(
            task.await.unwrap().unwrap_err().status(),
            StatusCode::GATEWAY_TIMEOUT
        );
        assert!(
            matches!(lifecycle.recv_async().await.unwrap(), LifecycleEvent::GuardAbort(rid) if rid.as_str() == "expired")
        );
        assert!(
            request
                .sink
                .try_send(ResponseItem::Control(vec![bytes::Bytes::new()]))
                .is_err()
        );

        let (mut state, _requests, lifecycle) = test_state();
        let (tx, requests) = flume::bounded(0);
        Arc::get_mut(&mut state).unwrap().senders.tok_manager_tx = tx;
        let response = await_control_result(
            &state,
            ControlRequest::GetInternalStateReq(GetInternalStateReq::new("queue-expired".into())),
            Duration::from_millis(1),
        )
        .await
        .unwrap_err();
        assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
        assert!(requests.try_recv().is_err());
        assert!(
            matches!(lifecycle.try_recv().unwrap(), LifecycleEvent::GuardAbort(rid) if rid.as_str() == "queue-expired")
        );
    }

    #[tokio::test]
    async fn abort_http_waits_for_intake_acknowledgement() {
        for body in [
            serde_json::json!({"rid": "batch-"}),
            serde_json::json!({"rid": null, "abort_all": true}),
        ] {
            let (state, requests, _lifecycle) = test_state();
            let task = tokio::spawn(
                routes().with_state(state).oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/abort_request")
                        .header("content-type", "application/json")
                        .body(Body::from(body.to_string()))
                        .unwrap(),
                ),
            );
            let TmEvent::Abort {
                rid_prefix,
                abort_all,
                reply,
            } = requests.recv_async().await.unwrap()
            else {
                panic!("abort request");
            };
            assert_eq!(rid_prefix, body["rid"].as_str().unwrap_or_default());
            assert_eq!(abort_all, body["abort_all"].as_bool().unwrap_or_default());
            assert!(!task.is_finished());
            reply.send(true).unwrap();
            assert_eq!(task.await.unwrap().unwrap().status(), StatusCode::OK);
        }
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

        // `api_key` is deliberately NOT a `ServerArgs` field — the typed schema
        // cannot carry it — so the only place it could leak from is the raw
        // scheduler dump shaped above.
        let sa = ServerArgs {
            model_path: "/m".into(),
            enable_http2: true,
            http2_max_concurrent_streams: 17,
            http2_initial_connection_window_size: 2 * 1024 * 1024,
            ..Default::default()
        };
        let out = shape_server_infos(&[msgpack.into()], &sa).unwrap();
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
        assert_eq!(v["enable_http2"], true);
        assert_eq!(v["http2_max_concurrent_streams"], 17);
        assert_eq!(v["http2_initial_connection_window_size"], 2 * 1024 * 1024);
    }

    #[test]
    fn server_info_preserves_feature_flags_and_distributed_configuration() {
        let internal = serde_json::json!({
            "enable_hierarchical_cache": true,
            "hicache_host_memory_mode": "file",
            "hicache_size": 64,
            "hicache_ratio": 2.0,
            "hicache_storage_backend": "file",
            "disable_radix_cache": false,
            "enable_deterministic_inference": false,
            "grammar_backend": "llguidance",
            "speculative_algorithm": "EAGLE3",
            "speculative_num_steps": 3,
            "speculative_num_draft_tokens": 5,
            "speculative_eagle_topk": 1,
            "tp_size": 4,
            "dp_size": 2,
            "ep_size": 4,
            "world_size": 4,
            "enable_dp_attention": true,
            "disaggregation_mode": "null",
            "quantization": null,
            "startup_time": 12.5,
            "avg_spec_accept_length": 3.75,
            "api_key": "secret-token",
            "hicache_storage_backend_extra_config": {"credential": "storage-secret"}
        });
        let msgpack = rmp_serde::to_vec_named(&serde_json::json!({
            "internal_state": internal
        }))
        .unwrap();
        let output = shape_server_infos(&[msgpack.into()], &ServerArgs::default()).unwrap();
        let info: serde_json::Value = serde_json::from_slice(&output).unwrap();
        for key in [
            "enable_hierarchical_cache",
            "hicache_host_memory_mode",
            "hicache_size",
            "hicache_ratio",
            "hicache_storage_backend",
            "disable_radix_cache",
            "enable_deterministic_inference",
            "grammar_backend",
            "speculative_algorithm",
            "speculative_num_steps",
            "speculative_num_draft_tokens",
            "speculative_eagle_topk",
            "tp_size",
            "dp_size",
            "ep_size",
            "world_size",
            "enable_dp_attention",
            "disaggregation_mode",
            "quantization",
        ] {
            assert_eq!(info.get(key), internal.get(key), "{key}");
            assert_eq!(
                info["internal_states"][0].get(key),
                internal.get(key),
                "{key}"
            );
        }
        assert_eq!(info["startup_time"], 12.5);
        assert_eq!(info["internal_states"][0]["avg_spec_accept_length"], 3.75);
        assert_eq!(info["frontend"], "rust");
        let text = String::from_utf8(output).unwrap();
        assert!(!text.contains("secret-token"));
        assert!(!text.contains("storage-secret"));
    }

    #[test]
    fn server_info_rejects_missing_scheduler_state() {
        for response in [
            serde_json::json!({}),
            serde_json::json!({"internal_state": null}),
        ] {
            let msgpack = rmp_serde::to_vec_named(&response).unwrap();
            assert!(shape_server_infos(&[msgpack.into()], &ServerArgs::default()).is_err());
        }
    }
}
