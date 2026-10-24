//! Common control-plane endpoints — `/server_info`, `/get_model_info`
//! (+ `/model_info` alias), `/flush_cache`, `/weights_checker`,
//! `/abort_request`, backed by typed operations on
//! [`crate::api_server::handlers::CoreHandle`]. Data-plane endpoints (incl. `/health*`,
//! which round-trips a generate probe) live in the sibling `native_api` and
//! `openai` modules; the shared `AppState` lives in the parent
//! `api_server` module.

use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::{
        Query, State,
        rejection::{JsonRejection, QueryRejection},
    },
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::Deserialize;
use serde_json::{Map, Value};
use sglang_api_types::api::v1 as api;
use sha2::{Digest, Sha256};
use std::sync::Arc;

use super::app::AppState;
use super::native_api::native_error;
use crate::api_server::core::{CoreError, FORWARD_PEER_PATH};
use crate::api_server::handlers::ProfileOutcome;
use crate::message::info::ServerInfo;
use crate::message::io_struct::{ExpertDistributionAction, ProfileOptions};

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
        .route(
            "/set_internal_state",
            post(set_internal_state).put(set_internal_state),
        )
        .route("/start_profile", get(start_profile).post(start_profile))
        .route("/stop_profile", get(stop_profile).post(stop_profile))
        .route(
            "/start_expert_distribution_record",
            get(start_expert_distribution_record).post(start_expert_distribution_record),
        )
        .route(
            "/stop_expert_distribution_record",
            get(stop_expert_distribution_record).post(stop_expert_distribution_record),
        )
        .route(
            "/dump_expert_distribution_record",
            get(dump_expert_distribution_record).post(dump_expert_distribution_record),
        )
        .route("/slow_down", get(slow_down).post(slow_down))
        .route(
            "/release_memory_occupation",
            get(release_memory_occupation).post(release_memory_occupation),
        )
        .route(
            "/resume_memory_occupation",
            get(resume_memory_occupation).post(resume_memory_occupation),
        )
        .route("/abort_request", post(abort_request))
        // Listener-to-listener: another DP rank's control request for this
        // rank's scheduler (see `api_server::core::peer`).
        .route(FORWARD_PEER_PATH, post(forward_control_req))
}

/// `POST|PUT /set_internal_state` — Python's `SetInternalStateReq` body; the
/// response is the bare list of per-rank `updated` flags, 200 whatever the
/// flags say (a rejected key is `false`, not an error, as in Python).
async fn set_internal_state(
    State(state): State<Arc<AppState>>,
    body: Result<Json<api::SetInternalStateRequest>, JsonRejection>,
) -> Response {
    let body = match body {
        Ok(Json(body)) => body,
        Err(rejection) => {
            return native_error(StatusCode::BAD_REQUEST, &rejection.body_text(), false);
        }
    };
    match state
        .core
        .set_internal_state(body.server_args.as_deref())
        .await
    {
        Ok(updated) => Json(api::SetInternalStateResponse { updated }).into_response(),
        Err(error) => native_error(error.http_status(), &error.to_string(), false),
    }
}

/// `GET|POST /start_profile` — Python's optional `ProfileReq` body (absent,
/// empty, or `null` means every default); 200 with Python's text line, or the
/// merged failure as a 500 with its message (Python raises, which FastAPI
/// renders as a bare 500).
async fn start_profile(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let request = if body.is_empty() {
        api::StartProfileRequest::default()
    } else {
        match serde_json::from_slice::<Option<api::StartProfileRequest>>(&body) {
            Ok(request) => request.unwrap_or_default(),
            Err(error) => return native_error(StatusCode::BAD_REQUEST, &error.to_string(), false),
        }
    };
    let options = ProfileOptions {
        output_dir: request.output_dir,
        start_step: request.start_step,
        num_steps: request.num_steps,
        activities: request.activities.map(|list| list.items),
        profile_by_stage: request.profile_by_stage,
        with_stack: request.with_stack,
        record_shapes: request.record_shapes,
        profile_id: request.profile_id,
        merge_profiles: request.merge_profiles,
        profile_prefix: request.profile_prefix,
        profile_stages: request.profile_stages.map(|list| list.items),
        detailed_annotations: request.detailed_annotations,
    };
    profile_response(
        state.core.start_profile(options).await,
        "Start profiling.\n",
    )
}

/// `GET|POST /stop_profile` — the counterpart of `/start_profile`.
async fn stop_profile(State(state): State<Arc<AppState>>) -> Response {
    profile_response(
        state.core.stop_profile().await,
        "Stop profiling. This will take some time.\n",
    )
}

fn profile_response(outcome: Result<ProfileOutcome, CoreError>, done: &'static str) -> Response {
    match outcome {
        Ok(outcome) if outcome.success => {
            // Python's bare `Response(content=...)`: no content type.
            let mut response = Response::new(Body::from(done));
            *response.status_mut() = StatusCode::OK;
            response
        }
        Ok(outcome) => native_error(StatusCode::INTERNAL_SERVER_ERROR, &outcome.message, false),
        Err(error) => native_error(error.http_status(), &error.to_string(), false),
    }
}

async fn start_expert_distribution_record(State(state): State<Arc<AppState>>) -> Response {
    text_ack(
        state
            .core
            .expert_distribution_record(ExpertDistributionAction::Start)
            .await,
        "Start recording the expert distribution.\n",
    )
}

async fn stop_expert_distribution_record(State(state): State<Arc<AppState>>) -> Response {
    text_ack(
        state
            .core
            .expert_distribution_record(ExpertDistributionAction::Stop)
            .await,
        "Stop recording the expert distribution.\n",
    )
}

async fn dump_expert_distribution_record(State(state): State<Arc<AppState>>) -> Response {
    text_ack(
        state
            .core
            .expert_distribution_record(ExpertDistributionAction::Dump)
            .await,
        "Dump expert distribution record.\n",
    )
}

/// Python's `Response(content=<line>)` on success: a bare 200 text body.
fn text_ack(outcome: Result<(), CoreError>, done: &'static str) -> Response {
    match outcome {
        Ok(()) => {
            let mut response = Response::new(Body::from(done));
            *response.status_mut() = StatusCode::OK;
            response
        }
        Err(error) => native_error(error.http_status(), &error.to_string(), false),
    }
}

/// `GET|POST /slow_down` — Python answers the handler's `None` as a JSON
/// `null`, and a failure as its `{"error": {...}}` 400 body.
async fn slow_down(
    State(state): State<Arc<AppState>>,
    body: Result<Json<api::SlowDownRequest>, JsonRejection>,
) -> Response {
    let body = match body {
        Ok(Json(body)) => body,
        Err(rejection) => {
            return native_error(StatusCode::BAD_REQUEST, &rejection.body_text(), false);
        }
    };
    null_ack(state.core.slow_down(body.forward_sleep_time).await)
}

/// `GET|POST /release_memory_occupation` — see `slow_down` for the shape.
async fn release_memory_occupation(
    State(state): State<Arc<AppState>>,
    body: Result<Json<api::ReleaseMemoryOccupationRequest>, JsonRejection>,
) -> Response {
    let body = match body {
        Ok(Json(body)) => body,
        Err(rejection) => {
            return native_error(StatusCode::BAD_REQUEST, &rejection.body_text(), false);
        }
    };
    null_ack(
        state
            .core
            .release_memory_occupation(body.tags.map(|list| list.items))
            .await,
    )
}

/// `GET|POST /resume_memory_occupation` — see `slow_down` for the shape.
async fn resume_memory_occupation(
    State(state): State<Arc<AppState>>,
    body: Result<Json<api::ResumeMemoryOccupationRequest>, JsonRejection>,
) -> Response {
    let body = match body {
        Ok(Json(body)) => body,
        Err(rejection) => {
            return native_error(StatusCode::BAD_REQUEST, &rejection.body_text(), false);
        }
    };
    null_ack(
        state
            .core
            .resume_memory_occupation(body.tags.map(|list| list.items))
            .await,
    )
}

/// FastAPI's rendering of a handler that returns `None`: 200 with `null`.
fn null_ack(outcome: Result<(), CoreError>) -> Response {
    match outcome {
        Ok(()) => Json(Value::Null).into_response(),
        Err(error) => native_error(error.http_status(), &error.to_string(), false),
    }
}

/// `POST /internal/peer` — the body is a control message's wire bytes,
/// encoded by another DP rank's listener; the reply body is this scheduler's
/// encoded output. Errors are plain text so the relaying side can carry the
/// message into its own merged error.
async fn forward_control_req(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    match state.core.forward_control_req(body).await {
        Ok(reply) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/msgpack")],
            reply,
        )
            .into_response(),
        Err(error) => (error.http_status(), error.to_string()).into_response(),
    }
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

/// Python's `AbortReq` body: the fields `/abort_request` reads.
#[derive(Deserialize)]
struct AbortRequestBody {
    #[serde(default)]
    rid: Option<String>,
    #[serde(default)]
    abort_all: bool,
}

/// `POST /abort_request` — cancel the live generations submitted under the
/// public rid `rid` (every item of a batch sent under it), or all of them with
/// `abort_all`, through the core's cancellation registry, as the gRPC `Abort`
/// does. Python's contract: 200 with an empty body whether or not anything
/// matched, and an empty rid without `abort_all` is ignored rather than
/// matching every request, as `TokenizerManager.abort_request` does.
async fn abort_request(
    State(state): State<Arc<AppState>>,
    body: Result<Json<AbortRequestBody>, JsonRejection>,
) -> Response {
    let body = match body {
        Ok(Json(body)) => body,
        Err(rejection) => {
            return native_error(StatusCode::BAD_REQUEST, &rejection.body_text(), false);
        }
    };
    let rid = body.rid.unwrap_or_default();
    if body.abort_all {
        state.core.abort(None);
    } else if rid.is_empty() {
        tracing::warn!("ignoring abort_request with an empty rid and abort_all=false");
    } else {
        state.core.abort(Some(&rid));
    }
    StatusCode::OK.into_response()
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
    use crate::api_server::core::CoreEvent;
    use crate::api_server::handlers::{CoreConfig, CoreHandle, CoreMetadata};
    use crate::message::config::{DisaggregationMode, ModelConfig, ServerArgs};
    use crate::message::ids::Rid;
    use crate::message::request::{GenerateRequest, RequestKind};
    use crate::message::response::ResponseItem;
    use crate::tokenizer_manager::wiring::{AbortSource, TmEvent};

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
                peer_endpoints: Vec::new(),
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

    /// `/abort_request` cancels the live generation submitted under the body's
    /// rid, or every one with `abort_all`, and answers Python's empty 200: the
    /// runtime is told to abort the request and the client's call ends as
    /// cancelled. A rid nothing runs under is still a 200 with nothing
    /// aborted, an empty rid without `abort_all` is ignored, and a malformed
    /// body is a 400.
    #[tokio::test]
    async fn abort_request_cancels_the_named_generations() {
        for (body, status, aborted) in [
            (r#"{"rid": "client-rid"}"#, StatusCode::OK, true),
            (r#"{"abort_all": true}"#, StatusCode::OK, true),
            (r#"{"rid": "other-rid"}"#, StatusCode::OK, false),
            (r#"{"rid": ""}"#, StatusCode::OK, false),
            ("{}", StatusCode::OK, false),
            (r#"{"rid": 3}"#, StatusCode::BAD_REQUEST, false),
        ] {
            let (state, intake_rx, abort_rx) = test_state(ServerArgs::default());
            let mut call = state
                .core
                .generate(GenerateRequest {
                    rid: Rid::from_client("client-rid"),
                    ..Default::default()
                })
                .await
                .unwrap();
            let TmEvent::Intake { request, admission } = intake_rx.recv_async().await.unwrap()
            else {
                panic!("{body}: generate must submit one request");
            };
            assert!(admission.try_accept(), "{body}");

            let response = routes()
                .with_state(state)
                .oneshot(
                    HttpRequest::builder()
                        .method("POST")
                        .uri("/abort_request")
                        .header(CONTENT_TYPE, "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), status, "{body}");
            let bytes = to_bytes(response.into_body(), 1024).await.unwrap();
            if status == StatusCode::OK {
                assert!(bytes.is_empty(), "{body}");
            }
            if aborted {
                assert!(
                    matches!(abort_rx.try_recv(), Ok(AbortSource::Guard(rid)) if rid == request.rid),
                    "{body}: the runtime must be told to abort the request"
                );
                assert!(
                    matches!(
                        call.recv().await,
                        Some(CoreEvent::Failed(CoreError::Cancelled(_)))
                    ),
                    "{body}: the client's call ends as cancelled"
                );
            } else {
                assert!(abort_rx.try_recv().is_err(), "{body}: nothing is aborted");
            }
            drop(request);
        }
    }

    /// `/set_internal_state` relays the object to the scheduler and answers
    /// Python's bare list of flags with 200; a non-object body is a 400 that
    /// reaches no scheduler.
    #[tokio::test]
    async fn set_internal_state_answers_the_per_rank_flags() {
        let (state, intake_rx, _abort_rx) = test_state(ServerArgs::default());
        let response = tokio::spawn(
            routes().with_state(state.clone()).oneshot(
                HttpRequest::builder()
                    .method("PUT")
                    .uri("/set_internal_state")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        r#"{"server_args": {"pp_max_micro_batch_size": 4}}"#,
                    ))
                    .unwrap(),
            ),
        );
        let TmEvent::Intake { request, admission } = intake_rx.recv_async().await.unwrap() else {
            panic!("set_internal_state must submit one control request");
        };
        assert!(admission.try_accept());
        assert!(matches!(request.kind, RequestKind::Control(_)));
        request
            .sink
            .try_send(ResponseItem::Control(Bytes::from(
                rmp_serde::to_vec_named(&serde_json::json!({"updated": true})).unwrap(),
            )))
            .unwrap();
        let response = response.await.unwrap().unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response_json(response).await, serde_json::json!([true]));

        let response = routes()
            .with_state(state)
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri("/set_internal_state")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(r#"{"server_args": [1, 2]}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert!(intake_rx.try_recv().is_err());
    }

    /// `/start_profile` takes an absent, empty, or null body as all defaults
    /// and answers Python's text line; `/stop_profile` too; a failing rank is
    /// a 500 carrying the scheduler's reason.
    #[tokio::test]
    async fn profile_routes_answer_python_text_lines() {
        for (uri, method, body, reply, status, expected) in [
            (
                "/start_profile",
                "GET",
                "",
                (true, ""),
                StatusCode::OK,
                "Start profiling.\n",
            ),
            (
                "/start_profile",
                "POST",
                "null",
                (true, ""),
                StatusCode::OK,
                "Start profiling.\n",
            ),
            (
                "/start_profile",
                "POST",
                r#"{"num_steps": 3, "activities": ["CPU"]}"#,
                (true, ""),
                StatusCode::OK,
                "Start profiling.\n",
            ),
            (
                "/stop_profile",
                "POST",
                "",
                (true, ""),
                StatusCode::OK,
                "Stop profiling. This will take some time.\n",
            ),
            (
                "/stop_profile",
                "GET",
                "",
                (false, "Profiler is not running"),
                StatusCode::INTERNAL_SERVER_ERROR,
                "Profiler is not running",
            ),
        ] {
            let (state, intake_rx, _abort_rx) = test_state(ServerArgs::default());
            let mut builder = HttpRequest::builder().method(method).uri(uri);
            if !body.is_empty() {
                builder = builder.header(CONTENT_TYPE, "application/json");
            }
            let response = tokio::spawn(
                routes()
                    .with_state(state)
                    .oneshot(builder.body(Body::from(body)).unwrap()),
            );
            let TmEvent::Intake { request, admission } = intake_rx.recv_async().await.unwrap()
            else {
                panic!("{uri}: must submit one control request");
            };
            assert!(admission.try_accept());
            request
                .sink
                .try_send(ResponseItem::Control(Bytes::from(
                    rmp_serde::to_vec_named(
                        &serde_json::json!({"success": reply.0, "message": reply.1}),
                    )
                    .unwrap(),
                )))
                .unwrap();
            let response = response.await.unwrap().unwrap();
            assert_eq!(response.status(), status, "{method} {uri} {body:?}");
            let bytes = to_bytes(response.into_body(), 4096).await.unwrap();
            let text = String::from_utf8_lossy(&bytes);
            assert!(text.contains(expected), "{method} {uri}: {text}");
        }

        let (state, intake_rx, _abort_rx) = test_state(ServerArgs::default());
        let response = routes()
            .with_state(state)
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri("/start_profile")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(r#"{"num_steps": "many"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert!(intake_rx.try_recv().is_err());
    }

    /// The acknowledged routes answer Python's shapes: a text line for the
    /// expert-distribution recorder, a JSON `null` for the others.
    #[tokio::test]
    async fn acknowledged_routes_answer_python_shapes() {
        let bare = Bytes::from(
            rmp_serde::to_vec_named(&serde_json::json!({"rid": "x", "http_worker_ipc": null}))
                .unwrap(),
        );
        for (uri, body, expected) in [
            (
                "/start_expert_distribution_record",
                "",
                "Start recording the expert distribution.\n",
            ),
            (
                "/stop_expert_distribution_record",
                "",
                "Stop recording the expert distribution.\n",
            ),
            (
                "/dump_expert_distribution_record",
                "",
                "Dump expert distribution record.\n",
            ),
            ("/slow_down", r#"{"forward_sleep_time": 0.5}"#, "null"),
            (
                "/release_memory_occupation",
                r#"{"tags": ["weights"]}"#,
                "null",
            ),
            ("/resume_memory_occupation", "{}", "null"),
        ] {
            let (state, intake_rx, _abort_rx) = test_state(ServerArgs::default());
            let mut builder = HttpRequest::builder().method("POST").uri(uri);
            if !body.is_empty() {
                builder = builder.header(CONTENT_TYPE, "application/json");
            }
            let response = tokio::spawn(
                routes()
                    .with_state(state)
                    .oneshot(builder.body(Body::from(body)).unwrap()),
            );
            let TmEvent::Intake { request, admission } = intake_rx.recv_async().await.unwrap()
            else {
                panic!("{uri}: must submit one control request");
            };
            assert!(admission.try_accept());
            request
                .sink
                .try_send(ResponseItem::Control(bare.clone()))
                .unwrap();
            let response = response.await.unwrap().unwrap();
            assert_eq!(response.status(), StatusCode::OK, "{uri}");
            let bytes = to_bytes(response.into_body(), 4096).await.unwrap();
            assert_eq!(String::from_utf8_lossy(&bytes), expected, "{uri}");
        }
    }

    /// A forwarded control request reaches the local scheduler under the rid
    /// its sender encoded, and the scheduler's bytes come back as the body.
    #[tokio::test]
    async fn forward_control_relays_the_sender_rid_and_the_reply_bytes() {
        let (state, intake_rx, _abort_rx) = test_state(ServerArgs::default());
        let header = crate::message::io_struct::ControlRequest::FlushCacheReqInput(
            crate::message::io_struct::FlushCacheReqInput::new("sender-rid".into(), Some(2.0)),
        )
        .encode()
        .unwrap();
        let response = tokio::spawn(
            routes().with_state(state.clone()).oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri(FORWARD_PEER_PATH)
                    .body(Body::from(header.clone()))
                    .unwrap(),
            ),
        );
        let TmEvent::Intake { request, admission } = intake_rx.recv_async().await.unwrap() else {
            panic!("forwarded control must submit one control request");
        };
        assert!(admission.try_accept());
        assert_eq!(request.rid.as_str(), "sender-rid");
        let RequestKind::Control(control) = &request.kind else {
            panic!("expected a control request");
        };
        assert_eq!(control.encode().unwrap(), header, "relayed unchanged");
        request
            .sink
            .try_send(ResponseItem::Control(Bytes::from_static(
                b"scheduler-reply",
            )))
            .unwrap();
        let response = response.await.unwrap().unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 1024).await.unwrap();
        assert_eq!(body.as_ref(), b"scheduler-reply");

        // A body without a rid cannot be routed back: rejected before intake.
        let response = routes()
            .with_state(state)
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri(FORWARD_PEER_PATH)
                    .body(Body::from(Bytes::from_static(b"\xc0")))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert!(intake_rx.try_recv().is_err());
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
