//! Common control-plane endpoints — `/server_info`, `/get_model_info`
//! (+ `/model_info` alias), `/flush_cache`, `/weights_checker`, backed by typed
//! operations on
//! [`crate::api_server::core::CoreHandle`]. Data-plane endpoints (incl. `/health*`,
//! which round-trips a generate probe) live in the sibling `native_api` and
//! `openai` modules; the shared `AppState` lives in the parent
//! `api_server` module.

use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::{Query, State, rejection::QueryRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
};
use serde::Deserialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::sync::Arc;

use super::app::AppState;
use super::native_api::native_error;
use crate::api_server::core::CoreError;
use crate::message::info::ServerInfo;

/// The routes this module owns, mounted by `api_server::serve`.
pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        // Control-plane: the core performs the typed operation and this
        // adapter renders its result as one non-streamed JSON response.
        .route("/server_info", get(server_info))
        // Static config, no scheduler round-trip. `/get_model_info` (+ `/model_info`
        // alias).
        .route("/get_model_info", get(model_info))
        .route("/model_info", get(model_info))
        .route("/flush_cache", get(flush_cache).post(flush_cache))
        .route("/weights_checker", get(check_weights).post(check_weights))
}

/// Await the core's complete semantic server-info result. HTTP status and
/// serialization stay here; scheduler control bytes stay behind the contract.
async fn await_server_info(state: &AppState) -> Result<ServerInfo, Response> {
    match state.core.server_info().await {
        Ok(server_info) => Ok(server_info),
        Err(CoreError::Unavailable) => Err(native_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "service unavailable",
            false,
        )),
        // Preserve the established HTTP behavior for a scheduler-side control
        // stream that disappears, while non-HTTP adapters see Internal via
        // CoreError::kind().
        Err(CoreError::ResponseTruncated) => {
            Err((StatusCode::from_u16(499).unwrap(), "request aborted").into_response())
        }
        Err(error @ CoreError::InvalidResponse(_)) => {
            tracing::error!(%error, "server_info: invalid runtime response");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                "bad server_info response",
            )
                .into_response())
        }
        Err(error) => Err((error.http_status(), error.to_string()).into_response()),
    }
}

/// `GET /get_model_info` (+ `/model_info` alias) — serialize the shared static
/// model metadata (no scheduler round-trip).
async fn model_info(State(state): State<Arc<AppState>>) -> Response {
    let info = state.core.model_info();
    (
        StatusCode::OK,
        [("content-type", "application/json")],
        serde_json::to_vec(&info).unwrap_or_default(),
    )
        .into_response()
}

/// `GET /server_info` — Python's launch-time fields (resolved `server_args`,
/// `launch_command`, scheduler init info, `kv_events`), as Python reports
/// them, beneath the runtime's startup timings, typed allowlisted scheduler
/// metrics and launch metadata. The raw scheduler state is never visible to
/// this HTTP adapter.
async fn server_info(State(state): State<Arc<AppState>>) -> Response {
    let info = match await_server_info(&state).await {
        Ok(info) => info,
        Err(resp) => return resp,
    };
    match info.to_json().and_then(|body| serde_json::to_vec(&body)) {
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

#[derive(Deserialize)]
struct FlushCacheQuery {
    #[serde(default)]
    timeout: f64,
}

/// `GET|POST /flush_cache?timeout=<s>` — the scheduler flushes its caches in
/// its own loop, after waiting up to `timeout` seconds for running requests to
/// finish. Python's status and plain-text body: 200 once flushed, 400 with the
/// scheduler's reason otherwise.
async fn flush_cache(
    State(state): State<Arc<AppState>>,
    query: Result<Query<FlushCacheQuery>, QueryRejection>,
) -> Response {
    let timeout = match query {
        Ok(Query(query)) if query.timeout >= 0.0 => query.timeout,
        Ok(_) => {
            return native_error(
                StatusCode::BAD_REQUEST,
                "timeout must be greater than or equal to 0",
                false,
            );
        }
        Err(rejection) => {
            return native_error(StatusCode::BAD_REQUEST, &rejection.body_text(), false);
        }
    };
    let outcome = match state.core.flush_cache(timeout).await {
        Ok(outcome) => outcome,
        Err(CoreError::Unavailable) => {
            return native_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "service unavailable",
                false,
            );
        }
        Err(error) => return native_error(error.http_status(), &error.to_string(), false),
    };
    let (status, body) = if outcome.success {
        (
            StatusCode::OK,
            "Cache flushed.\nPlease check backend logs for more details. (When there are \
             running or waiting requests, the operation will not be performed.)\n"
                .to_owned(),
        )
    } else if outcome.message.is_empty() {
        (StatusCode::BAD_REQUEST, "Flush cache failed.\n".to_owned())
    } else {
        (StatusCode::BAD_REQUEST, outcome.message)
    };
    // Python's bare `Response(content=...)`: no content type.
    let mut response = Response::new(Body::from(body));
    *response.status_mut() = status;
    response
}

/// Python's `CheckWeightsReqInput` body, every field optional.
#[derive(Deserialize)]
#[serde(default)]
struct CheckWeightsBody {
    action: String,
    allow_quant_error: bool,
    skip_tensor_list: Option<Vec<String>>,
    selector: CheckWeightsSelector,
}

impl Default for CheckWeightsBody {
    fn default() -> Self {
        Self {
            action: "checksum".into(),
            allow_quant_error: false,
            skip_tensor_list: None,
            selector: CheckWeightsSelector::All,
        }
    }
}

/// Which runners the check applies to (Python's `selector` literal).
#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum CheckWeightsSelector {
    Target,
    Draft,
    All,
}

impl CheckWeightsSelector {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Target => "target",
            Self::Draft => "draft",
            Self::All => "all",
        }
    }
}

/// `GET|POST /weights_checker` — every scheduler rank runs the weight
/// checker's `action` (default `checksum`) on the `selector` runners. Python's
/// body: `success` and `message`, plus each rank's record as `ranks` and the
/// SHA-256 of their `per_gpu_checksum`s, in rank order, as
/// `per_engine_checksum` when the ranks returned records. 200 on success,
/// else 400. An absent or `null` body checks with every default.
async fn check_weights(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let body = if body.is_empty() {
        CheckWeightsBody::default()
    } else {
        match serde_json::from_slice::<Option<CheckWeightsBody>>(&body) {
            Ok(body) => body.unwrap_or_default(),
            Err(error) => {
                return native_error(StatusCode::BAD_REQUEST, &error.to_string(), false);
            }
        }
    };
    let outcome = match state
        .core
        .check_weights(
            body.action,
            body.allow_quant_error,
            body.skip_tensor_list,
            body.selector.as_str().to_owned(),
        )
        .await
    {
        Ok(outcome) => outcome,
        Err(CoreError::Unavailable) => {
            return native_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "service unavailable",
                false,
            );
        }
        Err(error) => return native_error(error.http_status(), &error.to_string(), false),
    };
    let status = if outcome.success {
        StatusCode::OK
    } else {
        StatusCode::BAD_REQUEST
    };
    let mut response = Map::new();
    response.insert("success".into(), outcome.success.into());
    response.insert("message".into(), outcome.message.into());
    if let Some(ranks) = outcome.payload {
        let mut digest = Sha256::new();
        for rank in &ranks {
            let Some(checksum) = rank.get("per_gpu_checksum").and_then(Value::as_str) else {
                tracing::error!("weights_checker: a rank record has no per_gpu_checksum");
                return native_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "bad weights_checker response",
                    false,
                );
            };
            digest.update(checksum.as_bytes());
        }
        response.insert(
            "ranks".into(),
            Value::Array(ranks.into_iter().map(Value::Object).collect()),
        );
        response.insert(
            "per_engine_checksum".into(),
            format!("{:x}", digest.finalize()).into(),
        );
    }
    (status, Json(Value::Object(response))).into_response()
}

#[cfg(test)]
mod tests {
    use axum::{
        body::{Body, to_bytes},
        http::{Request as HttpRequest, header::CONTENT_TYPE},
    };
    use bytes::Bytes;
    use tower::ServiceExt;

    use super::*;
    use crate::api_server::core::{CoreConfig, CoreHandle, CoreMetadata};
    use crate::message::config::{DisaggregationMode, ModelConfig, ServerArgs};
    use crate::message::request::RequestKind;
    use crate::message::response::ResponseItem;
    use crate::tokenizer_manager::wiring::TmEvent;

    fn test_state(
        server_args: ServerArgs,
    ) -> (
        Arc<AppState>,
        flume::Receiver<TmEvent>,
        flume::Receiver<crate::tokenizer_manager::wiring::AbortSource>,
    ) {
        let server_args = Arc::new(server_args);
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, abort_rx) = flume::unbounded();
        let core = CoreHandle::new(
            intake_tx,
            abort_tx,
            CoreConfig {
                response_capacity: 8,
                response_activity: Default::default(),
                startup_ready: true,
                is_disaggregation: false,
                mm_limits: Default::default(),
                metadata: CoreMetadata::from(server_args.as_ref()),
            },
        );
        (
            Arc::new(AppState {
                core,
                generation_policy: crate::message::request::GenerationPolicy::from(
                    server_args.as_ref(),
                ),
                server_args,
                chat_formatter: None,
            }),
            intake_rx,
            abort_rx,
        )
    }

    async fn response_json(response: Response) -> serde_json::Value {
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(CONTENT_TYPE).unwrap(),
            "application/json"
        );
        let body = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    #[tokio::test]
    async fn model_info_routes_render_the_typed_frontend_contract() {
        let (state, _, _) = test_state(ServerArgs {
            model_path: "/model".into(),
            served_model_name: "served".into(),
            tokenizer_path: "/tokenizer".into(),
            weight_version: Some("weights-v1".into()),
            load_format: Some("safetensors".into()),
            reasoning_parser: Some("reasoner".into()),
            tool_call_parser: Some("tools".into()),
            disaggregation_mode: DisaggregationMode::Prefill,
            ..Default::default()
        });
        let app = routes().with_state(state);

        for path in ["/get_model_info", "/model_info"] {
            let response = app
                .clone()
                .oneshot(
                    HttpRequest::builder()
                        .uri(path)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            let body = response_json(response).await;
            assert_eq!(body["model_path"], "/model");
            assert_eq!(body["served_model_name"], "served");
            assert_eq!(body["tokenizer_path"], "/tokenizer");
            assert_eq!(body["is_generation"], true);
            assert_eq!(body["weight_version"], "weights-v1");
            assert_eq!(body["load_format"], "safetensors");
            assert_eq!(body["reasoning_parser"], "reasoner");
            assert_eq!(body["tool_call_parser"], "tools");
            assert_eq!(body["disaggregation_mode"], "prefill");
            assert!(body["preferred_sampling_params"].is_null());
        }
    }

    #[tokio::test]
    async fn server_info_route_serializes_only_the_typed_public_subset() {
        // Native floats stay JSON numbers; NumPy scalars encoded by Python's
        // msgspec enc_hook=str must remain JSON strings, as before the refactor.
        assert_server_info_response(rmpv::Value::from(23.616), serde_json::json!(23.616)).await;
        assert_server_info_response(rmpv::Value::from("23.616"), serde_json::json!("23.616")).await;
    }

    async fn assert_server_info_response(
        kvcache: rmpv::Value,
        expected_kvcache: serde_json::Value,
    ) {
        let launch_fields = serde_json::json!({
            "model_path": "/model",
            "tp_size": 4,
            "max_total_num_tokens": 1,
            "launch_command": "launch --tp-size 4",
        });
        let (state, intake_rx, abort_rx) = test_state(ServerArgs {
            model_path: "/model".into(),
            served_model_name: "served".into(),
            tokenizer_path: "/tokenizer".into(),
            model_config: ModelConfig {
                context_len: 4096,
                ..Default::default()
            },
            max_total_num_tokens: 8192,
            version: "1.2.3".into(),
            launch_server_info: crate::message::config::LaunchServerInfo(Arc::new(
                launch_fields.as_object().unwrap().clone(),
            )),
            ..Default::default()
        });
        let request = tokio::spawn(
            routes().with_state(state).oneshot(
                HttpRequest::builder()
                    .uri("/server_info")
                    .body(Body::empty())
                    .unwrap(),
            ),
        );

        let TmEvent::Intake {
            request: runtime_request,
            admission,
        } = intake_rx.recv_async().await.unwrap()
        else {
            panic!("server_info must submit one control request");
        };
        assert!(admission.try_accept());
        assert!(matches!(runtime_request.kind, RequestKind::Control(_)));

        let internal_state = rmpv::Value::Map(vec![
            (
                rmpv::Value::from("rust_mm_transport"),
                rmpv::Value::Map(vec![(
                    rmpv::Value::from("inline_features"),
                    rmpv::Value::from(3),
                )]),
            ),
            (
                rmpv::Value::from("api_key"),
                rmpv::Value::from("must-not-leak"),
            ),
            (
                rmpv::Value::from("last_gen_throughput"),
                rmpv::Value::from(1.5),
            ),
            (
                rmpv::Value::from("rust_server_startup_time"),
                rmpv::Value::Map(vec![
                    (rmpv::Value::from("load_weight"), rmpv::Value::from(1.5)),
                    (rmpv::Value::from("tokenizer_e2e"), rmpv::Value::from(9.0)),
                ]),
            ),
            (
                rmpv::Value::from("memory_usage"),
                rmpv::Value::Map(vec![
                    (rmpv::Value::from("kvcache"), kvcache),
                    (rmpv::Value::from("token_capacity"), rmpv::Value::from(8192)),
                    (rmpv::Value::from("token_capacity_swa"), rmpv::Value::Nil),
                    (
                        rmpv::Value::from("future_uncontracted_metric"),
                        rmpv::Value::from(99),
                    ),
                ]),
            ),
            (
                rmpv::Value::from("step_time_dict"),
                rmpv::Value::Map(vec![(
                    rmpv::Value::from(4),
                    rmpv::Value::Array(vec![rmpv::Value::from(0.01)]),
                )]),
            ),
        ]);
        let envelope =
            rmpv::Value::Map(vec![(rmpv::Value::from("internal_state"), internal_state)]);
        let mut payload = Vec::new();
        rmpv::encode::write_value(&mut payload, &envelope).unwrap();
        runtime_request
            .sink
            .try_send(ResponseItem::Control(Bytes::from(payload)))
            .unwrap();

        let response = request.await.unwrap().unwrap();
        let body = response_json(response).await;
        // Python's launch-time fields come first, in order; a runtime field
        // replaces a launch-time one in place, and the rest follow.
        let keys: Vec<&str> = body
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys[..5],
            [
                "model_path",
                "tp_size",
                "max_total_num_tokens",
                "launch_command",
                "served_model_name"
            ]
        );
        assert_eq!(body["tp_size"], 4);
        assert_eq!(body["launch_command"], "launch --tp-size 4");
        // The embedded server's startup timings move to the top level.
        assert_eq!(
            body["startup_time"],
            serde_json::json!({"load_weight": 1.5, "tokenizer_e2e": 9.0})
        );
        assert!(
            body["internal_states"][0]
                .get("rust_server_startup_time")
                .is_none()
        );
        assert_eq!(body["model_path"], "/model");
        assert_eq!(body["served_model_name"], "served");
        assert_eq!(body["tokenizer_path"], "/tokenizer");
        assert_eq!(body["max_context_length"], 4096);
        assert_eq!(body["max_total_num_tokens"], 8192);
        assert_eq!(body["version"], "1.2.3");
        assert_eq!(body["frontend"], "rust");
        assert_eq!(
            body["internal_states"][0]["rust_mm_transport"]["inline_features"],
            3
        );
        assert_eq!(body["internal_states"][0]["last_gen_throughput"], 1.5);
        assert_eq!(
            body["internal_states"][0]["memory_usage"]["kvcache"],
            expected_kvcache
        );
        assert_eq!(
            body["internal_states"][0]["memory_usage"]["token_capacity"],
            8192
        );
        assert_eq!(
            body["internal_states"][0]["memory_usage"]["token_capacity_swa"],
            serde_json::Value::Null
        );
        assert_eq!(
            body["internal_states"][0]["step_time_dict"]["4"],
            serde_json::json!([0.01])
        );
        assert!(!body.to_string().contains("must-not-leak"));
        assert!(
            body["internal_states"][0]["memory_usage"]
                .get("future_uncontracted_metric")
                .is_none(),
            "uncontracted scheduler fields must not silently become public API"
        );
        assert!(abort_rx.try_recv().is_err());
    }

    /// `/flush_cache` relays one `FlushCacheReqInput` carrying the query
    /// timeout and answers with Python's status and plain-text body; a
    /// negative timeout is refused before anything reaches the scheduler.
    #[tokio::test]
    async fn flush_cache_relays_the_scheduler_outcome() {
        for (method, uri, timeout, outcome, status, body) in [
            (
                "POST",
                "/flush_cache?timeout=30",
                30.0,
                (true, ""),
                StatusCode::OK,
                "Cache flushed.\nPlease check backend logs for more details. (When there \
                 are running or waiting requests, the operation will not be performed.)\n",
            ),
            (
                "GET",
                "/flush_cache",
                0.0,
                (false, "Timed out waiting for idle state."),
                StatusCode::BAD_REQUEST,
                "Timed out waiting for idle state.",
            ),
            (
                "POST",
                "/flush_cache",
                0.0,
                (false, ""),
                StatusCode::BAD_REQUEST,
                "Flush cache failed.\n",
            ),
        ] {
            let (state, intake_rx, _abort_rx) = test_state(ServerArgs::default());
            let response = tokio::spawn(
                routes().with_state(state).oneshot(
                    HttpRequest::builder()
                        .method(method)
                        .uri(uri)
                        .body(Body::empty())
                        .unwrap(),
                ),
            );
            let TmEvent::Intake {
                request: runtime_request,
                admission,
            } = intake_rx.recv_async().await.unwrap()
            else {
                panic!("flush_cache must submit one control request");
            };
            assert!(admission.try_accept());
            let RequestKind::Control(control) = &runtime_request.kind else {
                panic!("flush_cache must submit a control request");
            };
            let header = control.encode().unwrap();
            let header = rmpv::decode::read_value(&mut &header[..]).unwrap();
            assert_eq!(header[0].as_str(), Some("FlushCacheReqInput"));
            assert_eq!(header[3].as_f64(), Some(timeout), "{uri}");

            let mut payload = Vec::new();
            rmpv::encode::write_value(
                &mut payload,
                &rmpv::Value::Map(vec![
                    (rmpv::Value::from("rid"), rmpv::Value::from("x")),
                    (rmpv::Value::from("http_worker_ipc"), rmpv::Value::Nil),
                    (rmpv::Value::from("success"), rmpv::Value::from(outcome.0)),
                    (rmpv::Value::from("message"), rmpv::Value::from(outcome.1)),
                ]),
            )
            .unwrap();
            runtime_request
                .sink
                .try_send(ResponseItem::Control(Bytes::from(payload)))
                .unwrap();

            let response = response.await.unwrap().unwrap();
            assert_eq!(response.status(), status, "{uri}");
            assert!(response.headers().get(CONTENT_TYPE).is_none());
            let bytes = to_bytes(response.into_body(), 1024).await.unwrap();
            assert_eq!(std::str::from_utf8(&bytes).unwrap(), body);
        }

        let (state, intake_rx, _abort_rx) = test_state(ServerArgs::default());
        let response = routes()
            .with_state(state)
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri("/flush_cache?timeout=-1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert!(intake_rx.try_recv().is_err());
    }

    /// `/weights_checker` relays one `CheckWeightsReqInput` carrying the body's
    /// fields (Python's defaults for a missing body) and answers with Python's
    /// JSON: the rank records as sent, in order, with the SHA-256 of their
    /// `per_gpu_checksum`s; no `ranks` when no rank sent one; 400 on failure.
    /// A malformed body is refused before anything reaches the scheduler.
    #[tokio::test]
    async fn weights_checker_relays_the_scheduler_outcome() {
        fn rank(checksum: &str, tp_rank: i64) -> rmpv::Value {
            let info = rmpv::Value::Map(
                [
                    ("role", rmpv::Value::from("target")),
                    ("tp_rank", rmpv::Value::from(tp_rank)),
                    ("tp_size", rmpv::Value::from(2)),
                ]
                .into_iter()
                .map(|(key, value)| (rmpv::Value::from(key), value))
                .collect(),
            );
            rmpv::Value::Map(vec![
                (
                    rmpv::Value::from("checksums"),
                    rmpv::Value::Map(vec![(
                        rmpv::Value::from("model.w"),
                        rmpv::Value::from(checksum),
                    )]),
                ),
                (
                    rmpv::Value::from("per_gpu_checksum"),
                    rmpv::Value::from(checksum),
                ),
                (
                    rmpv::Value::from("parallelism_info"),
                    rmpv::Value::Array(vec![info]),
                ),
            ])
        }
        let ranks = rmpv::Value::Array(vec![rank("a", 0), rank("b", 1)]);
        let ranks_json = serde_json::json!([
            {"checksums": {"model.w": "a"}, "per_gpu_checksum": "a",
             "parallelism_info": [{"role": "target", "tp_rank": 0, "tp_size": 2}]},
            {"checksums": {"model.w": "b"}, "per_gpu_checksum": "b",
             "parallelism_info": [{"role": "target", "tp_rank": 1, "tp_size": 2}]},
        ]);
        for (method, body, request, outcome, status, response_body) in [
            (
                "POST",
                r#"{"action": "fingerprint", "allow_quant_error": true,
                    "skip_tensor_list": ["lm_head"], "selector": "target", "rid": "x"}"#,
                ("fingerprint", true, Some("lm_head"), "target"),
                (true, "Success.", ranks.clone()),
                StatusCode::OK,
                serde_json::json!({
                    "success": true,
                    "message": "Success.",
                    "ranks": ranks_json,
                    "per_engine_checksum":
                        "fb8e20fc2e4c3f248c60c39bd652f3c1347298bb977b8b4d5903b85055620603",
                }),
            ),
            (
                "GET",
                "",
                ("checksum", false, None, "all"),
                (false, "rank 1 failed", rmpv::Value::Nil),
                StatusCode::BAD_REQUEST,
                serde_json::json!({"success": false, "message": "rank 1 failed"}),
            ),
            (
                "POST",
                "null",
                ("checksum", false, None, "all"),
                (true, "Success.", rmpv::Value::Array(vec![])),
                StatusCode::OK,
                serde_json::json!({
                    "success": true,
                    "message": "Success.",
                    "ranks": [],
                    "per_engine_checksum":
                        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                }),
            ),
        ] {
            let (state, intake_rx, _abort_rx) = test_state(ServerArgs::default());
            let response = tokio::spawn(
                routes().with_state(state).oneshot(
                    HttpRequest::builder()
                        .method(method)
                        .uri("/weights_checker")
                        .body(Body::from(body))
                        .unwrap(),
                ),
            );
            let TmEvent::Intake {
                request: runtime_request,
                admission,
            } = intake_rx.recv_async().await.unwrap()
            else {
                panic!("weights_checker must submit one control request");
            };
            assert!(admission.try_accept());
            let RequestKind::Control(control) = &runtime_request.kind else {
                panic!("weights_checker must submit a control request");
            };
            let header = control.encode().unwrap();
            let header = rmpv::decode::read_value(&mut &header[..]).unwrap();
            assert_eq!(header[0].as_str(), Some("CheckWeightsReqInput"));
            assert_eq!(header[3].as_str(), Some(request.0), "{body}");
            assert_eq!(header[4].as_bool(), Some(request.1), "{body}");
            assert_eq!(header[5][0].as_str(), request.2, "{body}");
            assert_eq!(header[6].as_str(), Some(request.3), "{body}");

            let mut payload = Vec::new();
            rmpv::encode::write_value(
                &mut payload,
                &rmpv::Value::Map(vec![
                    (rmpv::Value::from("rid"), rmpv::Value::from("x")),
                    (rmpv::Value::from("http_worker_ipc"), rmpv::Value::Nil),
                    (rmpv::Value::from("success"), rmpv::Value::from(outcome.0)),
                    (rmpv::Value::from("message"), rmpv::Value::from(outcome.1)),
                    (rmpv::Value::from("payload"), outcome.2),
                ]),
            )
            .unwrap();
            runtime_request
                .sink
                .try_send(ResponseItem::Control(Bytes::from(payload)))
                .unwrap();

            let response = response.await.unwrap().unwrap();
            assert_eq!(response.status(), status, "{body}");
            let bytes = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
            // Field order too: Python's response is an ordered dict.
            assert_eq!(
                std::str::from_utf8(&bytes).unwrap(),
                response_body.to_string(),
                "{body}"
            );
        }

        for body in [r#"{"selector": "both"}"#, r#"{"action": null}"#, "{"] {
            let (state, intake_rx, _abort_rx) = test_state(ServerArgs::default());
            let response = routes()
                .with_state(state)
                .oneshot(
                    HttpRequest::builder()
                        .method("POST")
                        .uri("/weights_checker")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{body}");
            assert!(intake_rx.try_recv().is_err(), "{body}");
        }
    }

    #[tokio::test]
    async fn server_info_preserves_legacy_499_for_a_truncated_control_response() {
        let (state, intake_rx, abort_rx) = test_state(ServerArgs::default());
        let response = tokio::spawn(
            routes().with_state(state).oneshot(
                HttpRequest::builder()
                    .uri("/server_info")
                    .body(Body::empty())
                    .unwrap(),
            ),
        );

        let TmEvent::Intake {
            request: runtime_request,
            admission,
        } = intake_rx.recv_async().await.unwrap()
        else {
            panic!("server_info must submit one control request");
        };
        assert!(admission.try_accept());
        let rid = runtime_request.rid.clone();
        drop(runtime_request);

        let response = response.await.unwrap().unwrap();
        assert_eq!(response.status().as_u16(), 499);
        let body = to_bytes(response.into_body(), 1024).await.unwrap();
        assert_eq!(body.as_ref(), b"request aborted");
        assert!(matches!(
            abort_rx.recv_async().await.unwrap(),
            crate::tokenizer_manager::wiring::AbortSource::Guard(aborted) if aborted == rid
        ));
    }
}
