//! Common control-plane endpoints — `/server_info`, `/get_model_info`
//! (+ `/model_info` alias), `/flush_cache`, `/weights_checker`,
//! `/start_profile`, `/stop_profile`, `/abort_request`, backed by typed operations on
//! [`crate::api_server::handlers::CoreHandle`]. Data-plane endpoints (incl. `/health*`,
//! which round-trips a generate probe) live in the sibling `native_api` and
//! `openai` modules; the shared `AppState` lives in the parent
//! `api_server` module.

use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::{
        Path, Query, State,
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
use super::auth::DeclaredRoute;
use super::native_api::native_error;
use super::openai::openai_error;
use crate::api_server::core::{CoreError, FORWARD_PEER_PATH};
use crate::api_server::handlers::{AbortEnding, DumperOutcome, ElasticEpState, ProfileOutcome};
use crate::api_server::openai::unix_seconds_u32;
use crate::message::info::ServerInfo;
use crate::message::io_struct::{
    ExpertDistributionAction, LoggingConfig, PdRoleSwitch, ProfileOptions, SessionOpen,
};

/// The routes this module owns, mounted by `api_server::serve`.
pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        // Control-plane: the core performs the typed operation and this
        // adapter renders its result as one non-streamed JSON response.
        .declared_route("/server_info", get(server_info))
        .declared_route("/get_server_info", get(get_server_info))
        // Liveness and readiness flags, no scheduler round-trip: `/ping` is
        // the SageMaker probe (a paused server still answers 200), `/ready`
        // also needs the server unpaused.
        .declared_route("/ping", get(ping))
        .declared_route("/ready", get(ready))
        .declared_route(
            "/get_hicache_l3_cache_layout",
            get(get_hicache_l3_cache_layout),
        )
        .declared_route(
            "/configure_logging",
            get(configure_logging).post(configure_logging),
        )
        .declared_route("/open_session", get(open_session).post(open_session))
        .declared_route("/close_session", get(close_session).post(close_session))
        .declared_route("/scale_elastic_ep", post(scale_elastic_ep))
        .declared_route("/is_scaling_elastic_ep", get(is_scaling_elastic_ep))
        .merge(dumper_routes())
        // Static config, no scheduler round-trip. `/get_model_info` (+ `/model_info`
        // alias).
        .declared_route("/get_model_info", get(model_info))
        .declared_route("/model_info", get(model_info))
        // The same cards in the OpenAI shape. `{*model}`: a served name may
        // carry a slash (`org/model`), as Python's `{model:path}` takes it.
        .declared_route("/v1/models", get(openai_models))
        .declared_route("/v1/models/{*model}", get(openai_model))
        .declared_route("/flush_cache", get(flush_cache).post(flush_cache))
        .declared_route("/weights_checker", get(check_weights).post(check_weights))
        .declared_route(
            "/set_internal_state",
            post(set_internal_state).put(set_internal_state),
        )
        .declared_route("/start_profile", get(start_profile).post(start_profile))
        .declared_route("/stop_profile", get(stop_profile).post(stop_profile))
        .declared_route(
            "/start_expert_distribution_record",
            get(start_expert_distribution_record).post(start_expert_distribution_record),
        )
        .declared_route(
            "/stop_expert_distribution_record",
            get(stop_expert_distribution_record).post(stop_expert_distribution_record),
        )
        .declared_route(
            "/dump_expert_distribution_record",
            get(dump_expert_distribution_record).post(dump_expert_distribution_record),
        )
        .declared_route("/slow_down", get(slow_down).post(slow_down))
        .declared_route(
            "/release_memory_occupation",
            get(release_memory_occupation).post(release_memory_occupation),
        )
        .declared_route(
            "/resume_memory_occupation",
            get(resume_memory_occupation).post(resume_memory_occupation),
        )
        .declared_route("/freeze_gc", get(freeze_gc).post(freeze_gc))
        .declared_route("/pause_generation", post(pause_generation))
        .declared_route("/continue_generation", post(continue_generation))
        .declared_route("/pd_role_switch", post(pd_role_switch))
        .declared_route("/update_weight_version", post(update_weight_version))
        .declared_route("/abort_request", post(abort_request))
        // Listener-to-listener: another DP rank's control request for this
        // rank's scheduler (see `api_server::core::peer`).
        .declared_route(FORWARD_PEER_PATH, post(forward_control_req))
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

/// Python's `ProfileReq` body for `/start_profile`. `req_type` is accepted
/// only as one of Python's `ProfileReqType` values and then ignored: the route
/// always starts.
#[derive(Deserialize)]
struct StartProfileBody {
    #[serde(default)]
    req_type: Option<u8>,
    #[serde(flatten)]
    options: ProfileOptions,
}

/// `GET|POST /start_profile` — the scheduler ranks start their profilers
/// with the body's options (Python's defaults for an absent or `null` body);
/// which ranks under attention DP: see `ProfileReq`.
/// Python's plain-text 200 once started.
async fn start_profile(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let options = if body.is_empty() {
        ProfileOptions::default()
    } else {
        match serde_json::from_slice::<Option<StartProfileBody>>(&body) {
            Ok(None) => ProfileOptions::default(),
            Ok(Some(StartProfileBody {
                req_type: None | Some(1 | 2),
                options,
            })) => options,
            Ok(Some(_)) => {
                return native_error(
                    StatusCode::BAD_REQUEST,
                    "req_type: Input should be 1 or 2",
                    false,
                );
            }
            Err(error) => {
                return native_error(StatusCode::BAD_REQUEST, &error.to_string(), false);
            }
        }
    };
    profile_response(
        state.core.start_profile(options).await,
        "Start profiling.\n",
    )
}

/// `GET|POST /stop_profile` — the scheduler ranks stop their profilers and
/// write their traces (which ranks under attention DP: see `ProfileReq`);
/// any body is ignored, as Python ignores it.
async fn stop_profile(State(state): State<Arc<AppState>>) -> Response {
    profile_response(
        state.core.stop_profile().await,
        "Stop profiling. This will take some time.\n",
    )
}

/// Python's responses for both profile routes: 200 with `ok_body` and no
/// content type. A scheduler refusal (e.g. already profiling) is the
/// `RuntimeError` Python raises in the route, which reaches the client as the
/// framework's bare 500 `Internal Server Error`; the reason is only logged.
fn profile_response(outcome: Result<ProfileOutcome, CoreError>, ok_body: &'static str) -> Response {
    match outcome {
        Ok(ProfileOutcome { success: true, .. }) => Response::new(Body::from(ok_body)),
        Ok(ProfileOutcome { message, .. }) => {
            tracing::error!(%message, "profile request failed");
            (StatusCode::INTERNAL_SERVER_ERROR, "Internal Server Error").into_response()
        }
        Err(CoreError::Unavailable) => native_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "service unavailable",
            false,
        ),
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

async fn freeze_gc(State(state): State<Arc<AppState>>) -> Response {
    text_ack(
        state.core.freeze_gc_all().await,
        "Garbage collection frozen.\n",
    )
}

/// `POST /pause_generation` — Python's `{"message", "status": "ok"}` body.
async fn pause_generation(
    State(state): State<Arc<AppState>>,
    body: Result<Json<api::PauseGenerationRequest>, JsonRejection>,
) -> Response {
    let body = match body {
        Ok(Json(body)) => body,
        Err(rejection) => {
            return native_error(StatusCode::BAD_REQUEST, &rejection.body_text(), false);
        }
    };
    let mode = body.mode.unwrap_or_else(|| "abort".into());
    match state.core.pause_generation(&mode).await {
        Ok(()) => Json(api::GenerationPauseResponse {
            message: "Generation paused successfully.".into(),
            status: "ok".into(),
        })
        .into_response(),
        Err(error) => native_error(error.http_status(), &error.to_string(), false),
    }
}

/// `POST /continue_generation` — the counterpart of `/pause_generation`.
async fn continue_generation(
    State(state): State<Arc<AppState>>,
    body: Result<Json<api::ContinueGenerationRequest>, JsonRejection>,
) -> Response {
    let body = match body {
        Ok(Json(body)) => body,
        Err(rejection) => {
            return native_error(StatusCode::BAD_REQUEST, &rejection.body_text(), false);
        }
    };
    match state
        .core
        .continue_generation(body.torch_empty_cache.unwrap_or(true))
        .await
    {
        Ok(()) => Json(api::GenerationPauseResponse {
            message: "Generation continued successfully.".into(),
            status: "ok".into(),
        })
        .into_response(),
        Err(error) => native_error(error.http_status(), &error.to_string(), false),
    }
}

/// `POST /pd_role_switch` — Python's `PdRoleSwitchReqOutput` as JSON, 200 on
/// success and 400 otherwise; a frontend failure is its `{"error": ...}` 400.
async fn pd_role_switch(
    State(state): State<Arc<AppState>>,
    body: Result<Json<api::PdRoleSwitchRequest>, JsonRejection>,
) -> Response {
    let body = match body {
        Ok(Json(body)) => body,
        Err(rejection) => {
            return native_error(StatusCode::BAD_REQUEST, &rejection.body_text(), false);
        }
    };
    let switch = PdRoleSwitch {
        new_role: body.new_role.unwrap_or_default(),
        decode_cuda_graph_bs: body.decode_cuda_graph_bs.map(|list| list.items),
        decode_cuda_graph_memory_gb: body.decode_cuda_graph_memory_gb,
        prepare_only: body.prepare_only,
    };
    match state.core.pd_role_switch(switch).await {
        Ok(outcome) => {
            let status = if outcome.success {
                StatusCode::OK
            } else {
                StatusCode::BAD_REQUEST
            };
            (
                status,
                Json(api::PdRoleSwitchResponse {
                    success: outcome.success,
                    message: outcome.message,
                    old_role: outcome.old_role,
                    new_role: outcome.new_role,
                    safe_to_restore: outcome.safe_to_restore,
                }),
            )
                .into_response()
        }
        Err(error) => native_error(StatusCode::BAD_REQUEST, &error.to_string(), false),
    }
}

/// `POST /update_weight_version` — Python's `{"success", "message", "new_version"}`
/// on success, `{"success": false, "message"}` 400 on failure.
async fn update_weight_version(
    State(state): State<Arc<AppState>>,
    body: Result<Json<api::UpdateWeightVersionRequest>, JsonRejection>,
) -> Response {
    let body = match body {
        Ok(Json(body)) => body,
        Err(rejection) => {
            return native_error(StatusCode::BAD_REQUEST, &rejection.body_text(), false);
        }
    };
    let new_version = body.new_version;
    match state
        .core
        .update_weight_version(new_version.clone(), body.abort_all_requests.unwrap_or(true))
        .await
    {
        Ok(()) => Json(api::UpdateWeightVersionResponse {
            success: true,
            message: format!("Weight version updated to {new_version}"),
            new_version: Some(new_version),
        })
        .into_response(),
        Err(error) => (
            StatusCode::BAD_REQUEST,
            Json(api::UpdateWeightVersionResponse {
                success: false,
                message: format!("Failed to update weight version: {error}"),
                new_version: None,
            }),
        )
            .into_response(),
    }
}

/// `POST /internal/peer` — the body is a control message's wire bytes,
/// encoded by another DP rank's listener; the reply body is this scheduler's
/// encoded output. Errors are plain text so the relaying side can carry the
/// message into its own merged error.
async fn forward_control_req(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let Some(routes) = forwarded_tag(&body).and_then(|tag| relayed_by(&tag)) else {
        return (
            StatusCode::BAD_REQUEST,
            "not a control message a peer listener relays",
        )
            .into_response();
    };
    // Without an admin key the forced admin routes refuse before relaying,
    // so only a request sent straight to this route carries their message;
    // refuse it too, or the peer route would bypass that refusal.
    if state.server_args.api_keys.admin_api_key.is_none()
        && routes.iter().copied().any(super::auth::is_admin_force)
    {
        return super::auth::admin_key_missing();
    }
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

/// Each control message a peer listener relays, with the HTTP routes that fan it
/// out; a relayed message is held to their levels in `auth_levels.json`.
const RELAYED_BY: &[(&str, &[&str])] = &[
    ("AbortReq", &["/abort_request"]),
    ("GetInternalStateReq", &["/server_info"]),
    ("FlushCacheReqInput", &["/flush_cache"]),
    ("CheckWeightsReqInput", &["/weights_checker"]),
    ("FreezeGCReq", &["/freeze_gc"]),
    ("SetInternalStateReq", &["/set_internal_state"]),
    ("ProfileReq", &["/start_profile", "/stop_profile"]),
    (
        "ExpertDistributionReq",
        &[
            "/start_expert_distribution_record",
            "/stop_expert_distribution_record",
            "/dump_expert_distribution_record",
        ],
    ),
    ("SlowDownReqInput", &["/slow_down"]),
    (
        "ReleaseMemoryOccupationReqInput",
        &["/release_memory_occupation"],
    ),
    (
        "ResumeMemoryOccupationReqInput",
        &["/resume_memory_occupation"],
    ),
    ("GetWeightsByNameReqInput", &["/get_weights_by_name"]),
    (
        "InitWeightsUpdateGroupReqInput",
        &["/init_weights_update_group"],
    ),
    (
        "DestroyWeightsUpdateGroupReqInput",
        &["/destroy_weights_update_group"],
    ),
    (
        "InitWeightsSendGroupForRemoteInstanceReqInput",
        &["/init_weights_send_group_for_remote_instance"],
    ),
    (
        "SendWeightsToRemoteInstanceReqInput",
        &["/send_weights_to_remote_instance"],
    ),
    ("AddExternalCorpusReqInput", &["/add_external_corpus"]),
    ("RemoveExternalCorpusReqInput", &["/remove_external_corpus"]),
    ("ListExternalCorporaReqInput", &["/list_external_corpora"]),
    ("OpenSessionReqInput", &["/open_session"]),
    ("CloseSessionReqInput", &["/close_session"]),
    ("GetHiCacheL3LayoutReq", &["/get_hicache_l3_cache_layout"]),
    ("ConfigureLoggingReq", &["/configure_logging"]),
    ("DumperControlReqInput", &["/dumper/{method}"]),
    (
        "UpdateWeightFromDiskReqInput",
        &["/update_weights_from_disk"],
    ),
    (
        "UpdateWeightsFromDistributedReqInput",
        &["/update_weights_from_distributed"],
    ),
    (
        "UpdateWeightsFromTensorReqInput",
        &["/update_weights_from_tensor"],
    ),
    (
        "UpdateWeightsFromIPCReqInput",
        &["/update_weights_from_ipc"],
    ),
    ("BeginWeightUpdateReqInput", &["/begin_weight_update"]),
    ("EndWeightUpdateReqInput", &["/end_weight_update"]),
    ("PluginWeightUpdateReqInput", &["/plugin_weight_update"]),
    ("PauseGenerationReqInput", &["/pause_generation"]),
    ("ContinueGenerationReqInput", &["/continue_generation"]),
    ("PdRoleSwitchReqInput", &["/pd_role_switch"]),
    ("UpdateWeightVersionReqInput", &["/update_weight_version"]),
    ("ClearHiCacheReqInput", &["/hicache/storage-backend/clear"]),
    (
        "DetachHiCacheStorageReqInput",
        &["/hicache/storage-backend"],
    ),
    (
        "AttachHiCacheStorageReqInput",
        &["/hicache/storage-backend"],
    ),
];

fn relayed_by(tag: &str) -> Option<&'static [&'static str]> {
    RELAYED_BY
        .iter()
        .find(|(relayed, _)| *relayed == tag)
        .map(|(_, routes)| *routes)
}

/// A forwarded control message's type tag, the first element of its array.
fn forwarded_tag(body: &[u8]) -> Option<String> {
    match rmpv::decode::read_value(&mut &body[..]).ok()? {
        rmpv::Value::Array(items) => items.first()?.as_str().map(str::to_owned),
        _ => None,
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
/// model metadata, Python's launch-time model fields last (no scheduler
/// round-trip).
async fn model_info(State(state): State<Arc<AppState>>) -> Response {
    let info = state.core.model_info().await;
    (
        StatusCode::OK,
        [("content-type", "application/json")],
        info.to_json()
            .and_then(|body| serde_json::to_vec(&body))
            .unwrap_or_default(),
    )
        .into_response()
}

/// `GET /server_info` — Python's launch-time fields (resolved `server_args`,
/// `launch_command`, scheduler init info, `kv_events`), as Python reports
/// them, beneath the runtime's startup timings, typed allowlisted scheduler
/// metrics and launch metadata. The raw scheduler state is never visible to
/// this HTTP adapter. Under DP, the listener of DP rank 0 adds every rank's
/// listener as `dp_listeners` (see `CoreHandle::server_info`).
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

/// The deprecated alias Python keeps, warning on each use.
async fn get_server_info(state: State<Arc<AppState>>) -> Response {
    tracing::warn!(
        "Endpoint '/get_server_info' is deprecated and will be removed in a future version. \
         Please use '/server_info' instead."
    );
    server_info(state).await
}

/// `GET /ping` — Python's `is_healthy()`: up and not shutting down; an
/// intentional pause still answers 200. An empty body either way.
async fn ping(State(state): State<Arc<AppState>>) -> Response {
    flag_status(state.core.is_ready() && !state.core.is_draining())
}

/// `GET /ready` — Python's `is_ready()`: healthy and not paused.
async fn ready(State(state): State<Arc<AppState>>) -> Response {
    flag_status(state.core.is_ready() && !state.core.is_draining() && !state.core.is_paused())
}

fn flag_status(ok: bool) -> Response {
    if ok {
        StatusCode::OK.into_response()
    } else {
        StatusCode::SERVICE_UNAVAILABLE.into_response()
    }
}

/// `GET /get_hicache_l3_cache_layout` — the merged L3 layout, always 200.
async fn get_hicache_l3_cache_layout(State(state): State<Arc<AppState>>) -> Response {
    match state.core.hicache_l3_cache_layout().await {
        Ok(layout) => Json(layout).into_response(),
        Err(error) => native_error(error.http_status(), &error.to_string(), false),
    }
}

/// `GET|POST /configure_logging` — Python's empty 200; an unknown level is
/// refused (Python's `setLevel` raises, which its route does not catch).
async fn configure_logging(
    State(state): State<Arc<AppState>>,
    body: Result<Json<api::ConfigureLoggingRequest>, JsonRejection>,
) -> Response {
    let body = match body {
        Ok(Json(body)) => body,
        Err(rejection) => {
            return native_error(StatusCode::BAD_REQUEST, &rejection.body_text(), false);
        }
    };
    let outcome = state
        .core
        .configure_logging(LoggingConfig {
            log_requests: body.log_requests,
            log_requests_level: body.log_requests_level,
            log_requests_format: body.log_requests_format,
            log_level: body.log_level,
            dump_requests_folder: body.dump_requests_folder,
            dump_requests_threshold: body.dump_requests_threshold,
            crash_dump_folder: body.crash_dump_folder,
            dump_requests_exclude_meta_keys: body
                .dump_requests_exclude_meta_keys
                .map(|list| list.items),
        })
        .await;
    match outcome {
        Ok(()) => StatusCode::OK.into_response(),
        Err(error) => native_error(error.http_status(), &error.to_string(), false),
    }
}

/// `POST /dumper/{method}`, mounted only under `DUMPER_SERVER_PORT=reuse`
/// as Python mounts it.
fn dumper_routes() -> Router<Arc<AppState>> {
    let router = Router::new();
    if std::env::var("DUMPER_SERVER_PORT").as_deref() == Ok("reuse") {
        router.declared_route("/dumper/{method}", post(dumper_control))
    } else {
        router
    }
}

/// Python's rendering: the ranks' responses flattened as a JSON array, or a
/// 400 carrying every failed rank's error as `{"error": [...]}`. The body
/// must be a JSON object (empty when absent).
async fn dumper_control(
    State(state): State<Arc<AppState>>,
    Path(method): Path<String>,
    body: Bytes,
) -> Response {
    let body = if body.is_empty() {
        Map::new()
    } else {
        match serde_json::from_slice::<Value>(&body) {
            Ok(Value::Object(object)) => object,
            Ok(_) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({"error": "Request body must be a JSON object."})),
                )
                    .into_response();
            }
            Err(error) => return native_error(StatusCode::BAD_REQUEST, &error.to_string(), false),
        }
    };
    match state.core.dumper_control(method, body).await {
        Ok(DumperOutcome::Responses(responses)) => Json(Value::Array(responses)).into_response(),
        Ok(DumperOutcome::Failed(errors)) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": errors})),
        )
            .into_response(),
        Err(error) => native_error(error.http_status(), &error.to_string(), false),
    }
}

/// Python's `_create_error_response`: `{"error": {"message": ...}}`, 400.
fn session_error(message: &str) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({"error": {"message": message}})),
    )
        .into_response()
}

/// `GET|POST /open_session` — the new session's id as a bare JSON string,
/// or Python's 400 when it could not be opened.
async fn open_session(
    State(state): State<Arc<AppState>>,
    body: Result<Json<api::OpenSessionRequest>, JsonRejection>,
) -> Response {
    let body = match body {
        Ok(Json(body)) => body,
        Err(rejection) => return session_error(&rejection.body_text()),
    };
    let outcome = state
        .core
        .open_session(SessionOpen {
            capacity_of_str_len: body.capacity_of_str_len,
            session_id: body.session_id,
            streaming: body.streaming,
            timeout: body.timeout,
        })
        .await;
    match outcome {
        Ok(Some(session_id)) => Json(session_id).into_response(),
        Ok(None) => session_error(
            "Failed to open the session. Check if a session with the same id is still open.",
        ),
        Err(error) => session_error(&error.to_string()),
    }
}

/// `GET|POST /close_session` — Python's empty 200.
async fn close_session(
    State(state): State<Arc<AppState>>,
    body: Result<Json<api::CloseSessionRequest>, JsonRejection>,
) -> Response {
    let body = match body {
        Ok(Json(body)) => body,
        Err(rejection) => return session_error(&rejection.body_text()),
    };
    match state.core.close_session(body.session_id).await {
        Ok(()) => StatusCode::OK.into_response(),
        Err(error) => session_error(&error.to_string()),
    }
}

/// Python's `{"error": ...}` 404 without `--elastic-ep-backend`.
fn elastic_ep_disabled() -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({"error": "elastic EP is not enabled (set --elastic-ep-backend)"})),
    )
        .into_response()
}

/// `POST /scale_elastic_ep` — the body is checked as Python checks it (a
/// JSON object with a positive integer `new_ep_size`) before the backend
/// gate. The Rust frontend serves no scale: with a backend, the refusal
/// says so.
async fn scale_elastic_ep(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let error = |status: StatusCode, message: String| {
        (status, Json(serde_json::json!({"error": message}))).into_response()
    };
    let body = match serde_json::from_slice::<Value>(&body) {
        Ok(Value::Object(object)) => object,
        Ok(_) => {
            return error(
                StatusCode::BAD_REQUEST,
                "request body must be a JSON object".into(),
            );
        }
        Err(e) => return error(StatusCode::BAD_REQUEST, format!("Invalid JSON: {e}")),
    };
    let positive = body
        .get("new_ep_size")
        .and_then(Value::as_i64)
        .filter(|size| *size > 0 && !body["new_ep_size"].is_boolean())
        .is_some();
    if !positive {
        return error(
            StatusCode::BAD_REQUEST,
            "new_ep_size must be a positive integer".into(),
        );
    }
    if state.core.elastic_ep_state().is_none() {
        return elastic_ep_disabled();
    }
    error(
        StatusCode::BAD_REQUEST,
        "elastic EP scaling is not served by the Rust frontend".into(),
    )
}

/// `GET /is_scaling_elastic_ep` — Python's `get_elastic_ep_state`.
async fn is_scaling_elastic_ep(State(state): State<Arc<AppState>>) -> Response {
    match state.core.elastic_ep_state() {
        None => elastic_ep_disabled(),
        Some(ElasticEpState { effective_ep_size }) => Json(serde_json::json!({
            "is_scaling_elastic_ep": false,
            "effective_ep_size": effective_ep_size,
            "pending_ep_size": null,
            "scale_phase": "idle",
            "last_error": null,
        }))
        .into_response(),
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
    names: Option<Vec<String>>,
}

impl Default for CheckWeightsBody {
    fn default() -> Self {
        Self {
            action: "checksum".into(),
            allow_quant_error: false,
            skip_tensor_list: None,
            selector: CheckWeightsSelector::All,
            names: None,
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
            body.names,
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
/// matching every request, as `TokenizerManager.abort_request` does. Each
/// aborted request ends as Python ends it, on the scheduler's terminal output
/// with the abort as its finish reason and the partial generation, rather than
/// on the cancelled error the gRPC `Abort` reports. Under DP, the listener
/// of DP rank 0 then relays the abort to every other rank's listener through
/// the control relay (`CoreHandle::abort_peers`), since each one feeds only
/// its own DP group and tracks only its own generations; a peer's failure is
/// logged, not answered, and a relayed abort is served locally and never
/// relayed again.
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
    let ending = AbortEnding::AbortFinishReason;
    if body.abort_all {
        state.core.abort(None, ending);
    } else if rid.is_empty() {
        tracing::warn!("ignoring abort_request with an empty rid and abort_all=false");
    } else {
        state.core.abort(Some(&rid), ending);
    }
    // Detached: the reply does not wait on peers, and a client hanging up
    // cannot cancel the relay.
    let core = state.core.clone();
    tokio::spawn(async move { core.abort_peers(rid, body.abort_all).await });
    StatusCode::OK.into_response()
}

/// `GET /v1/models`: the OpenAI model list, from core metadata.
async fn openai_models(State(state): State<Arc<AppState>>) -> Response {
    let cards = state.core.list_models().await;
    let models: Vec<Value> = cards.into_iter().map(openai_model_card).collect();
    Json(serde_json::json!({ "object": "list", "data": models })).into_response()
}

/// `GET /v1/models/{model}`: one card, or Python's 404 for an unknown name.
async fn openai_model(State(state): State<Arc<AppState>>, Path(model): Path<String>) -> Response {
    let cards = state.core.list_models().await;
    let Some(card) = cards.into_iter().find(|card| card.id == model) else {
        let message = format!("The model `{model}` does not exist");
        return openai_error(StatusCode::NOT_FOUND, message, false);
    };
    Json(openai_model_card(card)).into_response()
}

fn openai_model_card(card: api::ModelCard) -> Value {
    serde_json::json!({
        "id": card.id,
        "object": "model",
        "created": unix_seconds_u32(),
        "owned_by": "sglang",
        "root": card.root,
        "parent": card.parent,
        "max_model_len": card.max_model_len,
    })
}

#[cfg(test)]
mod tests {
    use axum::{
        body::{Body, to_bytes},
        http::{Request as HttpRequest, header::CONTENT_TYPE},
    };

    use futures::StreamExt;
    use tower::ServiceExt;

    use super::*;
    use crate::api_server::core::fanout::ControlTarget;
    use crate::api_server::core::peer_endpoints;
    use crate::api_server::handlers::{CoreConfig, CoreHandle, CoreMetadata};
    use crate::message::config::{DeploymentInfo, RustServerEndpoint, ServerArgs};
    use crate::message::ids::Rid;

    use crate::message::request::Request;

    use crate::message::response::{
        CachedTokensDetails, ChunkEvent, ChunkExtras, ResponseItem, WeightVersionSpan,
    };
    use crate::tokenizer_manager::wiring::{AbortSource, TmEvent};

    fn test_state(
        server_args: ServerArgs,
    ) -> (
        Arc<AppState>,
        flume::Receiver<TmEvent>,
        flume::Receiver<crate::tokenizer_manager::wiring::AbortSource>,
    ) {
        test_state_with_peers(server_args, Vec::new(), Vec::new())
    }

    /// A state whose core answers for the DP ranks `peer_endpoints` reach as
    /// well as its own, and reports `server_info_state_keys` on `/server_info`.
    fn test_state_with_peers(
        server_args: ServerArgs,
        peer_endpoints: Vec<Arc<dyn ControlTarget>>,
        server_info_state_keys: Vec<String>,
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
                multimodal: false,
                peer_endpoints,
                metadata: CoreMetadata::from(server_args.as_ref())
                    .with_server_info_state_keys(server_info_state_keys),
                metrics: None,
                request_log: None,
                generate_validator: None,
                abort_hook: None,
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
                generate_hook: None,
                openai: Default::default(),
            }),
            intake_rx,
            abort_rx,
        )
    }

    fn endpoint(dp_rank: u32, endpoint: &str) -> RustServerEndpoint {
        RustServerEndpoint {
            dp_rank,
            endpoint: endpoint.to_owned(),
        }
    }

    /// A table that points a listener back at itself (a peer published at
    /// this listener's own address) cannot loop: its own relayed abort is
    /// served on the local registry and never relayed again, so one client
    /// abort is two requests served, the client's and its own.
    #[tokio::test]
    async fn a_self_dial_is_one_local_abort() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let own = format!("http://{}", listener.local_addr().unwrap());
        let info = DeploymentInfo {
            dp_rank: Some(0),
            rust_server_endpoints: vec![endpoint(0, &own), endpoint(1, &own)],
            ..Default::default()
        };
        let peers = peer_endpoints(&info, 0, &Default::default()).unwrap();
        let (state, _intake_rx, _abort_rx) = test_state_with_peers(
            ServerArgs {
                deployment_info: info,
                ..Default::default()
            },
            peers,
            Vec::new(),
        );
        // Aborts are idempotent in the core, so count the requests served.
        let served = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = served.clone();
        let app = routes()
            .with_state(state)
            .layer(axum::middleware::map_request(
                move |request: HttpRequest<Body>| {
                    counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    async move { request }
                },
            ));
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let response = reqwest::Client::new()
            .post(format!("{own}/abort_request"))
            .header(CONTENT_TYPE, "application/json")
            .body(r#"{"abort_all": true}"#)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        assert_eq!(served.load(std::sync::atomic::Ordering::Relaxed), 2);
    }

    /// The terminal output Python's scheduler sends for a running request an
    /// `/abort_request` names: the step's tokens, the request's counts, cache
    /// split and weight-version span, and the plain abort as finish reason.
    fn scheduler_abort(
        rid: &Rid,
        text: &str,
        token_ids: Vec<i64>,
        completion_tokens: u64,
    ) -> ResponseItem {
        ResponseItem::Done(ChunkEvent {
            rid: rid.clone(),
            text: text.into(),
            token_ids,
            prompt_tokens: 5,
            cached_tokens: 2,
            completion_tokens,
            finish_reason: Some(
                serde_json::from_value(serde_json::json!({
                    "type": "abort",
                    "message": "Aborted",
                    "status_code": null,
                    "err_type": null
                }))
                .unwrap(),
            ),
            extras: Some(Box::new(ChunkExtras {
                cached_tokens_details: Some(CachedTokensDetails {
                    device: 0,
                    host: 2,
                    storage: None,
                    storage_backend: None,
                }),
                weight_versions: vec![WeightVersionSpan {
                    version: "v1".into(),
                    start: 0,
                    end: 17,
                }],
                ..Default::default()
            })),
            ..Default::default()
        })
    }

    /// An `/abort_request` leaves the request's sink in place and asks the
    /// scheduler to abort it, so each surface ends on the scheduler's terminal
    /// output as Python's server ends it: a native stream on a final frame
    /// with all the output, the counts, the cache split and the abort as its
    /// finish reason, then `[DONE]`; a unary `/generate` aborted before any
    /// intermediate output with a 200 carrying the partial generation; a
    /// `/v1/completions` stream on a final chunk whose `finish_reason` is
    /// `abort`, then `[DONE]`.
    #[tokio::test]
    async fn abort_request_ends_requests_on_the_schedulers_abort() {
        let (state, intake_rx, abort_rx) = test_state(ServerArgs::default());
        let app = routes()
            .merge(super::super::native_api::routes())
            .merge(super::super::openai::routes())
            .with_state(state);
        let post = |path: &'static str, body: serde_json::Value| {
            tokio::spawn(
                app.clone().oneshot(
                    HttpRequest::builder()
                        .method("POST")
                        .uri(path)
                        .header(CONTENT_TYPE, "application/json")
                        .body(Body::from(body.to_string()))
                        .unwrap(),
                ),
            )
        };
        // Accept the next generation, optionally streaming one frame of it.
        let start = |frame: bool| {
            let intake_rx = intake_rx.clone();
            async move {
                let TmEvent::Intake { request, admission } = intake_rx.recv_async().await.unwrap()
                else {
                    panic!("expected a generation");
                };
                assert!(admission.try_accept());
                if frame {
                    request
                        .sink
                        .try_send(ResponseItem::Frame(ChunkEvent {
                            rid: request.rid.clone(),
                            text: "partial".into(),
                            token_ids: vec![7],
                            prompt_tokens: 5,
                            cached_tokens: 2,
                            completion_tokens: 1,
                            ..Default::default()
                        }))
                        .unwrap();
                }
                request
            }
        };
        // Abort through the route, then answer as the scheduler does.
        let (post, abort_rx) = (&post, &abort_rx);
        let abort = |body: serde_json::Value, request: Request, terminal: ResponseItem| async move {
            let response = post("/abort_request", body).await.unwrap().unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert!(
                matches!(abort_rx.try_recv(), Ok(AbortSource::Request(rid)) if rid == request.rid)
            );
            request.sink.try_send(terminal).unwrap();
        };
        let data = |chunk: &[u8]| -> Vec<String> {
            std::str::from_utf8(chunk)
                .unwrap()
                .split("\n\n")
                .filter_map(|event| event.strip_prefix("data: ").map(str::to_owned))
                .collect()
        };

        let response = post(
            "/generate",
            serde_json::json!({"text": "hi", "rid": "stream", "stream": true}),
        );
        let request = start(true).await;
        let mut body = response
            .await
            .unwrap()
            .unwrap()
            .into_body()
            .into_data_stream();
        let first = data(&body.next().await.unwrap().unwrap());
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&first[0]).unwrap()["text"],
            "partial"
        );
        let terminal = scheduler_abort(&request.rid, " end", vec![8], 1);
        abort(serde_json::json!({"rid": "stream"}), request, terminal).await;
        let mut rest = Vec::new();
        while let Some(chunk) = body.next().await {
            rest.extend(data(&chunk.unwrap()));
        }
        let [last, done] = rest.as_slice() else {
            panic!("expected a final frame and [DONE]: {rest:?}");
        };
        let last: serde_json::Value = serde_json::from_str(last).unwrap();
        assert!(last.get("error").is_none(), "{last}");
        assert_eq!(last["text"], "partial end");
        assert_eq!(last["output_ids"], serde_json::json!([7, 8]));
        let meta = &last["meta_info"];
        assert_eq!(meta["finish_reason"]["type"], "abort");
        assert_eq!(meta["finish_reason"]["message"], "Aborted");
        assert_eq!(meta["prompt_tokens"], 5);
        assert_eq!(meta["completion_tokens"], 2);
        assert_eq!(
            meta["cached_tokens_details"],
            serde_json::json!({"device": 0, "host": 2})
        );
        assert_eq!(meta["weight_versions"][0]["version"], "v1");
        assert!(meta["e2e_latency"].is_f64());
        assert_eq!(done, "[DONE]");

        let response = post(
            "/generate",
            serde_json::json!({"text": "hi", "rid": "unary"}),
        );
        let request = start(false).await;
        let terminal = scheduler_abort(&request.rid, "seventeen tokens", (1..=17).collect(), 17);
        abort(serde_json::json!({"rid": "unary"}), request, terminal).await;
        let response = response.await.unwrap().unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 64 * 1024).await.unwrap())
                .unwrap();
        assert_eq!(body["text"], "seventeen tokens");
        assert_eq!(body["output_ids"].as_array().unwrap().len(), 17);
        assert_eq!(body["meta_info"]["completion_tokens"], 17);
        assert_eq!(body["meta_info"]["finish_reason"]["type"], "abort");

        let response = post(
            "/v1/completions",
            serde_json::json!({"model": "m", "prompt": "hi", "stream": true}),
        );
        let request = start(true).await;
        let mut body = response
            .await
            .unwrap()
            .unwrap()
            .into_body()
            .into_data_stream();
        body.next().await.unwrap().unwrap();
        let terminal = scheduler_abort(&request.rid, "", Vec::new(), 0);
        abort(serde_json::json!({"abort_all": true}), request, terminal).await;
        let mut rest = Vec::new();
        while let Some(chunk) = body.next().await {
            rest.extend(data(&chunk.unwrap()));
        }
        let [last, done] = rest.as_slice() else {
            panic!("expected a final chunk and [DONE]: {rest:?}");
        };
        let last: serde_json::Value = serde_json::from_str(last).unwrap();
        assert_eq!(last["choices"][0]["finish_reason"], "abort", "{last}");
        assert_eq!(done, "[DONE]");
    }

    #[test]
    fn relayed_by_covers_every_control_message() {
        use crate::message::io_struct::ControlRequest;
        let mut relayed: Vec<&str> = RELAYED_BY.iter().map(|(tag, _)| *tag).collect();
        relayed.sort_unstable();
        let mut tags = ControlRequest::TAGS.to_vec();
        tags.sort_unstable();
        assert_eq!(relayed, tags);
        for route in RELAYED_BY.iter().flat_map(|(_, routes)| *routes) {
            assert!(super::super::auth::has_level(route), "{route}");
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

    /// `/ping` answers the liveness flag and `/ready` the readiness flag,
    /// both as bare statuses: a pause keeps `/ping` at 200 and turns
    /// `/ready` to 503; a drain turns both.
    #[tokio::test]
    async fn ping_and_ready_answer_bare_flag_statuses() {
        let (state, _intake_rx, _abort_rx) = test_state(ServerArgs::default());
        let probe = |uri: &'static str| {
            let app = routes().with_state(state.clone());
            async move {
                app.oneshot(
                    HttpRequest::builder()
                        .method("GET")
                        .uri(uri)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap()
            }
        };
        for uri in ["/ping", "/ready"] {
            let response = probe(uri).await;
            assert_eq!(response.status(), StatusCode::OK, "{uri}");
            assert!(
                to_bytes(response.into_body(), 16).await.unwrap().is_empty(),
                "{uri}: empty body"
            );
        }
        state.core.set_paused_for_test(true);
        assert_eq!(probe("/ping").await.status(), StatusCode::OK);
        assert_eq!(
            probe("/ready").await.status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        state.core.set_paused_for_test(false);
        state.core.begin_drain(true);
        assert_eq!(
            probe("/ping").await.status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            probe("/ready").await.status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }
}
