//! Weight-management control routes: the state-free subset, backed by the
//! typed operations in `handlers::weights`.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{
        Query, State,
        rejection::{JsonRejection, QueryRejection},
    },
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde_json::Value;
use sglang_api_types::api::v1 as api;

use super::app::AppState;
use super::native_api::native_error;
use crate::api_server::core::CoreError;
use crate::api_server::handlers::{
    DiskUpdateOutcome, LORA_DISABLED, PluginUpdateOutcome, WeightGroupOutcome, WeightUpdateOutcome,
};
use crate::message::io_struct::{
    DiskWeightUpdate, DistributedWeightUpdate, IpcWeightUpdate, PluginWeightUpdate,
    TensorWeightUpdate, WeightsSendGroup, WeightsUpdateGroup, WireBytes,
};

pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route(
            "/get_weights_by_name",
            get(get_weights_by_name).post(get_weights_by_name),
        )
        .route("/update_weights_from_disk", post(update_weights_from_disk))
        .route(
            "/update_weights_from_distributed",
            post(update_weights_from_distributed),
        )
        .route(
            "/update_weights_from_tensor",
            post(update_weights_from_tensor),
        )
        .route("/update_weights_from_ipc", post(update_weights_from_ipc))
        .route("/begin_weight_update", post(begin_weight_update))
        .route("/end_weight_update", post(end_weight_update))
        .route("/plugin_weight_update", post(plugin_weight_update))
        .route("/load_lora_adapter", post(load_lora_adapter))
        .route(
            "/load_lora_adapter_from_tensors",
            post(load_lora_adapter_from_tensors),
        )
        .route("/unload_lora_adapter", post(unload_lora_adapter))
        .route(
            "/init_weights_update_group",
            post(init_weights_update_group),
        )
        .route(
            "/destroy_weights_update_group",
            post(destroy_weights_update_group),
        )
        .route(
            "/init_weights_send_group_for_remote_instance",
            post(init_weights_send_group_for_remote_instance),
        )
        .route(
            "/send_weights_to_remote_instance",
            post(send_weights_to_remote_instance),
        )
        .route(
            "/remote_instance_transfer_engine_info",
            get(remote_instance_transfer_engine_info),
        )
        .route(
            "/get_remote_instance_transfer_engine_info",
            get(get_remote_instance_transfer_engine_info),
        )
}

#[derive(serde::Deserialize)]
struct TransferEngineInfoQuery {
    rank: Option<i64>,
}

/// `GET /remote_instance_transfer_engine_info?rank=N` — the bootstrap server's
/// JSON verbatim, or Python's `{"error": {"message"}}` 400.
async fn remote_instance_transfer_engine_info(
    State(state): State<Arc<AppState>>,
    query: Result<Query<TransferEngineInfoQuery>, QueryRejection>,
) -> Response {
    let rank = match query {
        Ok(Query(query)) => query.rank,
        Err(_) => None,
    };
    let Some(rank) = rank.filter(|rank| *rank >= 0) else {
        return python_error(StatusCode::BAD_REQUEST, "Missing or invalid rank parameter");
    };
    match state.core.remote_instance_transfer_engine_info(rank).await {
        Ok(info) => (
            StatusCode::OK,
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            info,
        )
            .into_response(),
        Err(error) => python_error(StatusCode::BAD_REQUEST, &error.to_string()),
    }
}

/// The deprecated alias Python keeps, warning on each use.
async fn get_remote_instance_transfer_engine_info(
    state: State<Arc<AppState>>,
    query: Result<Query<TransferEngineInfoQuery>, QueryRejection>,
) -> Response {
    tracing::warn!(
        "Endpoint '/get_remote_instance_transfer_engine_info' is deprecated and will be removed \
         in a future version. Please use '/remote_instance_transfer_engine_info' instead."
    );
    remote_instance_transfer_engine_info(state, query).await
}

/// Python's `_create_error_response`: `{"error": {"message": ...}}` with no code.
fn python_error(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(serde_json::json!({"error": {"message": message}})),
    )
        .into_response()
}

/// The parsed body, or the rejection text for a 400.
fn body<T>(body: Result<Json<T>, JsonRejection>) -> Result<T, String> {
    body.map(|Json(body)| body)
        .map_err(|rejection| rejection.body_text())
}

/// `GET|POST /get_weights_by_name` — Python answers the bare parameter on a
/// single-DP deployment and one per rank otherwise, and a parameter no rank
/// has as its `{"error": ...}` 400.
async fn get_weights_by_name(
    State(state): State<Arc<AppState>>,
    body: Result<Json<api::GetWeightsByNameRequest>, JsonRejection>,
) -> Response {
    let request = match self::body(body) {
        Ok(request) => request,
        Err(message) => return native_error(StatusCode::BAD_REQUEST, &message, false),
    };
    let truncate_size = request.truncate_size.unwrap_or(100);
    match state
        .core
        .get_weights_by_name(request.name, truncate_size)
        .await
    {
        Ok(mut parameters) if parameters.len() == 1 => match parameters.pop() {
            // Python's `_create_error_response`: no `code` in the body.
            Some(Value::Null) | None => {
                python_error(StatusCode::BAD_REQUEST, "Get parameter by name failed")
            }
            Some(parameter) => Json(parameter).into_response(),
        },
        Ok(parameters) => Json(Value::Array(parameters)).into_response(),
        Err(error) => native_error(error.http_status(), &error.to_string(), false),
    }
}

async fn init_weights_update_group(
    State(state): State<Arc<AppState>>,
    body: Result<Json<api::InitWeightsUpdateGroupRequest>, JsonRejection>,
) -> Response {
    let request = match self::body(body) {
        Ok(request) => request,
        Err(message) => return native_error(StatusCode::BAD_REQUEST, &message, false),
    };
    group_response(
        state
            .core
            .init_weights_update_group(WeightsUpdateGroup {
                master_address: request.master_address,
                master_port: request.master_port,
                rank_offset: request.rank_offset,
                world_size: request.world_size,
                group_name: request
                    .group_name
                    .unwrap_or_else(|| "weight_update_group".into()),
                backend: request.backend.unwrap_or_else(|| "nccl".into()),
            })
            .await,
    )
}

async fn destroy_weights_update_group(
    State(state): State<Arc<AppState>>,
    body: Result<Json<api::DestroyWeightsUpdateGroupRequest>, JsonRejection>,
) -> Response {
    let request = match self::body(body) {
        Ok(request) => request,
        Err(message) => return native_error(StatusCode::BAD_REQUEST, &message, false),
    };
    group_response(
        state
            .core
            .destroy_weights_update_group(
                request
                    .group_name
                    .unwrap_or_else(|| "weight_update_group".into()),
            )
            .await,
    )
}

async fn init_weights_send_group_for_remote_instance(
    State(state): State<Arc<AppState>>,
    body: Result<Json<api::InitWeightsSendGroupForRemoteInstanceRequest>, JsonRejection>,
) -> Response {
    let request = match self::body(body) {
        Ok(request) => request,
        Err(message) => return native_error(StatusCode::BAD_REQUEST, &message, false),
    };
    group_response(
        state
            .core
            .init_weights_send_group_for_remote_instance(WeightsSendGroup {
                master_address: request.master_address,
                ports: request.ports,
                group_rank: request.group_rank,
                world_size: request.world_size,
                group_name: request
                    .group_name
                    .unwrap_or_else(|| "weight_send_group".into()),
                backend: request.backend.unwrap_or_else(|| "nccl".into()),
            })
            .await,
    )
}

async fn send_weights_to_remote_instance(
    State(state): State<Arc<AppState>>,
    body: Result<Json<api::SendWeightsToRemoteInstanceRequest>, JsonRejection>,
) -> Response {
    let request = match self::body(body) {
        Ok(request) => request,
        Err(message) => return native_error(StatusCode::BAD_REQUEST, &message, false),
    };
    group_response(
        state
            .core
            .send_weights_to_remote_instance(
                request.master_address,
                request.ports,
                request
                    .group_name
                    .unwrap_or_else(|| "weight_send_group".into()),
            )
            .await,
    )
}

/// Python's `Base64Bytes` validator: each payload is strict base64 text,
/// one per TP rank; anything else is the 422 the request type raises.
pub(crate) fn decode_payloads(payloads: &[String]) -> Result<Vec<WireBytes>, String> {
    use base64::Engine;
    payloads
        .iter()
        .map(|payload| {
            base64::engine::general_purpose::STANDARD
                .decode(payload)
                .map(WireBytes)
                .map_err(|_| "serialized_named_tensors: Expected base64-encoded bytes".to_owned())
        })
        .collect()
}

/// A raw-JSON object field (`Dict[str, Any]` in Python), or the 400 for
/// anything but an object.
pub(crate) fn json_object(
    field: &str,
    text: Option<&str>,
) -> Result<Option<serde_json::Map<String, Value>>, String> {
    match text {
        None => Ok(None),
        Some(text) => match serde_json::from_str::<Value>(text) {
            Ok(Value::Object(object)) => Ok(Some(object)),
            Ok(Value::Null) => Ok(None),
            _ => Err(format!("{field}: Input should be a valid dictionary")),
        },
    }
}

/// The `{"success", "message"}` body of a weight update: 200 on success,
/// 400 otherwise, as Python's routes answer.
fn update_response(outcome: Result<WeightUpdateOutcome, CoreError>) -> Response {
    match outcome {
        Ok(outcome) => (
            update_status(outcome.success),
            Json(api::WeightUpdateResponse {
                success: outcome.success,
                message: outcome.message,
            }),
        )
            .into_response(),
        Err(error) => native_error(error.http_status(), &error.to_string(), false),
    }
}

fn update_status(success: bool) -> StatusCode {
    if success {
        StatusCode::OK
    } else {
        StatusCode::BAD_REQUEST
    }
}

/// `POST /update_weights_from_disk` — Python's `{success, message,
/// num_paused_requests}`, the count bare on one DP rank and one per rank
/// otherwise.
async fn update_weights_from_disk(
    State(state): State<Arc<AppState>>,
    body: Result<Json<api::UpdateWeightsFromDiskRequest>, JsonRejection>,
) -> Response {
    let request = match self::body(body) {
        Ok(request) => request,
        Err(message) => return native_error(StatusCode::BAD_REQUEST, &message, false),
    };
    let manifest = match json_object("manifest", request.manifest.as_deref()) {
        Ok(manifest) => manifest,
        Err(message) => return native_error(StatusCode::BAD_REQUEST, &message, false),
    };
    let outcome = state
        .core
        .update_weights_from_disk(DiskWeightUpdate {
            model_path: request.model_path,
            load_format: request.load_format,
            abort_all_requests: request.abort_all_requests,
            weight_version: request.weight_version,
            is_async: request.is_async,
            torch_empty_cache: request.torch_empty_cache,
            keep_pause: request.keep_pause,
            recapture_cuda_graph: request.recapture_cuda_graph,
            token_step: request.token_step,
            flush_cache: request.flush_cache.unwrap_or(true),
            manifest,
        })
        .await;
    match outcome {
        Ok(DiskUpdateOutcome {
            success,
            message,
            num_paused_requests,
        }) => {
            let num_paused_requests = match num_paused_requests.as_slice() {
                [one] => Value::from(*one),
                many => Value::from(many),
            };
            (
                update_status(success),
                Json(serde_json::json!({
                    "success": success,
                    "message": message,
                    "num_paused_requests": num_paused_requests,
                })),
            )
                .into_response()
        }
        Err(error) => native_error(error.http_status(), &error.to_string(), false),
    }
}

async fn update_weights_from_distributed(
    State(state): State<Arc<AppState>>,
    body: Result<Json<api::UpdateWeightsFromDistributedRequest>, JsonRejection>,
) -> Response {
    let request = match self::body(body) {
        Ok(request) => request,
        Err(message) => return native_error(StatusCode::BAD_REQUEST, &message, false),
    };
    update_response(
        state
            .core
            .update_weights_from_distributed(DistributedWeightUpdate {
                names: request.names,
                dtypes: request.dtypes,
                shapes: request.shapes.into_iter().map(|list| list.items).collect(),
                group_name: request
                    .group_name
                    .unwrap_or_else(|| "weight_update_group".into()),
                flush_cache: request.flush_cache.unwrap_or(true),
                abort_all_requests: request.abort_all_requests,
                weight_version: request.weight_version,
                load_format: request.load_format,
                selector: request.selector.unwrap_or_else(|| "all".into()),
                torch_empty_cache: request.torch_empty_cache,
            })
            .await,
    )
}

async fn update_weights_from_tensor(
    State(state): State<Arc<AppState>>,
    body: Result<Json<api::UpdateWeightsFromTensorRequest>, JsonRejection>,
) -> Response {
    let request = match self::body(body) {
        Ok(request) => request,
        Err(message) => return native_error(StatusCode::BAD_REQUEST, &message, false),
    };
    let serialized_named_tensors = match decode_payloads(&request.serialized_named_tensors) {
        Ok(payloads) => payloads,
        Err(message) => return native_error(StatusCode::BAD_REQUEST, &message, false),
    };
    update_response(
        state
            .core
            .update_weights_from_tensor(TensorWeightUpdate {
                serialized_named_tensors,
                load_format: request.load_format,
                flush_cache: request.flush_cache.unwrap_or(true),
                abort_all_requests: request.abort_all_requests,
                weight_version: request.weight_version,
                selector: request.selector.unwrap_or_else(|| "all".into()),
                torch_empty_cache: request.torch_empty_cache,
            })
            .await,
    )
}

async fn update_weights_from_ipc(
    State(state): State<Arc<AppState>>,
    body: Result<Json<api::UpdateWeightsFromIpcRequest>, JsonRejection>,
) -> Response {
    let request = match self::body(body) {
        Ok(request) => request,
        Err(message) => return native_error(StatusCode::BAD_REQUEST, &message, false),
    };
    update_response(
        state
            .core
            .update_weights_from_ipc(IpcWeightUpdate {
                zmq_handles: request.zmq_handles.into_iter().collect(),
                flush_cache: request.flush_cache.unwrap_or(true),
                weight_version: request.weight_version,
                torch_empty_cache: request.torch_empty_cache,
            })
            .await,
    )
}

async fn begin_weight_update(
    State(state): State<Arc<AppState>>,
    body: Result<Json<api::BeginWeightUpdateRequest>, JsonRejection>,
) -> Response {
    let request = match self::body(body) {
        Ok(request) => request,
        Err(message) => return native_error(StatusCode::BAD_REQUEST, &message, false),
    };
    update_response(
        state
            .core
            .begin_weight_update(request.selector.unwrap_or_else(|| "all".into()))
            .await,
    )
}

/// `POST /end_weight_update` — Python's request has no fields; any body
/// (or none) is accepted.
async fn end_weight_update(State(state): State<Arc<AppState>>) -> Response {
    update_response(state.core.end_weight_update().await)
}

/// `POST /plugin_weight_update` — Python's `{success, message,
/// received_tensor_counts}`.
async fn plugin_weight_update(
    State(state): State<Arc<AppState>>,
    body: Result<Json<api::PluginWeightUpdateRequest>, JsonRejection>,
) -> Response {
    let request = match self::body(body) {
        Ok(request) => request,
        Err(message) => return native_error(StatusCode::BAD_REQUEST, &message, false),
    };
    let payload = match json_object("payload", request.payload.as_deref()) {
        Ok(payload) => payload.unwrap_or_default(),
        Err(message) => return native_error(StatusCode::BAD_REQUEST, &message, false),
    };
    match state
        .core
        .plugin_weight_update(PluginWeightUpdate {
            handler: request.handler,
            payload,
            flush_cache: request.flush_cache.unwrap_or(true),
            abort_all_requests: request.abort_all_requests,
            weight_version: request.weight_version,
            selector: request.selector.unwrap_or_else(|| "all".into()),
            torch_empty_cache: request.torch_empty_cache,
        })
        .await
    {
        Ok(PluginUpdateOutcome {
            success,
            message,
            received_tensor_counts,
            // Hand-rendered: Python emits the counts even when empty.
        }) => (
            update_status(success),
            Json(serde_json::json!({
                "success": success,
                "message": message,
                "received_tensor_counts": received_tensor_counts,
            })),
        )
            .into_response(),
        Err(error) => native_error(error.http_status(), &error.to_string(), false),
    }
}

/// Python's LoRA routes dump their whole `LoRAUpdateOutput`, `BaseReq`
/// preamble included; with LoRA disabled every one is this 400.
fn lora_disabled() -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({
            "rid": null,
            "http_worker_ipc": null,
            "success": false,
            "error_message": LORA_DISABLED,
            "loaded_adapters": null,
        })),
    )
        .into_response()
}

/// The LoRA routes. Rust-server mode refuses `--enable-lora` at launch, so
/// each answers as Python does without it; the body is still parsed, so a
/// malformed one is the 400 Python's request type gives it.
async fn load_lora_adapter(
    body: Result<Json<api::LoadLoraAdapterRequest>, JsonRejection>,
) -> Response {
    match self::body(body) {
        Ok(_) => lora_disabled(),
        Err(message) => native_error(StatusCode::BAD_REQUEST, &message, false),
    }
}

async fn load_lora_adapter_from_tensors(
    body: Result<Json<api::LoadLoraAdapterFromTensorsRequest>, JsonRejection>,
) -> Response {
    match self::body(body) {
        Ok(_) => lora_disabled(),
        Err(message) => native_error(StatusCode::BAD_REQUEST, &message, false),
    }
}

async fn unload_lora_adapter(
    body: Result<Json<api::UnloadLoraAdapterRequest>, JsonRejection>,
) -> Response {
    match self::body(body) {
        Ok(_) => lora_disabled(),
        Err(message) => native_error(StatusCode::BAD_REQUEST, &message, false),
    }
}

/// Python's `{"success", "message"}` body: 200 on success, 400 otherwise.
fn group_response(outcome: Result<WeightGroupOutcome, CoreError>) -> Response {
    match outcome {
        Ok(outcome) => {
            let status = if outcome.success {
                StatusCode::OK
            } else {
                StatusCode::BAD_REQUEST
            };
            (
                status,
                Json(api::WeightGroupResponse {
                    success: outcome.success,
                    message: outcome.message,
                }),
            )
                .into_response()
        }
        Err(error) => native_error(error.http_status(), &error.to_string(), false),
    }
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
    use crate::api_server::handlers::{CoreConfig, CoreHandle, CoreMetadata};
    use crate::message::config::ServerArgs;
    use crate::message::request::GenerationPolicy;
    use crate::message::response::ResponseItem;
    use crate::tokenizer_manager::wiring::TmEvent;

    fn test_state() -> (Arc<AppState>, flume::Receiver<TmEvent>) {
        let server_args = Arc::new(ServerArgs::default());
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, _abort_rx) = flume::unbounded();
        let core = CoreHandle::new(
            intake_tx,
            abort_tx,
            CoreConfig {
                response_capacity: 8,
                response_activity: Default::default(),
                startup_ready: true,
                is_disaggregation: false,
                multimodal: false,
                metrics: None,
                request_log: None,
                generate_validator: None,
                mm_limits: Default::default(),
                peer_endpoints: Vec::new(),
                metadata: CoreMetadata::from(server_args.as_ref()),
            },
        );
        (
            Arc::new(AppState {
                core,
                generation_policy: GenerationPolicy::from(server_args.as_ref()),
                server_args,
                chat_formatter: None,
                generate_hook: None,
                openai_chat: None,
            }),
            intake_rx,
        )
    }

    async fn call(
        uri: &str,
        body: &str,
        reply: serde_json::Value,
    ) -> (StatusCode, serde_json::Value, Vec<rmpv::Value>) {
        let (state, intake_rx) = test_state();
        let response = tokio::spawn(
            routes().with_state(state).oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri(uri)
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(body.to_owned()))
                    .unwrap(),
            ),
        );
        let TmEvent::Intake { request, admission } = intake_rx.recv_async().await.unwrap() else {
            panic!("{uri}: must submit one control request");
        };
        assert!(admission.try_accept());
        let crate::message::request::RequestKind::Control(control) = &request.kind else {
            panic!("expected a control request");
        };
        let rmpv::Value::Array(items) =
            rmpv::decode::read_value(&mut &control.encode().unwrap()[..]).unwrap()
        else {
            panic!("positional wire");
        };
        request
            .sink
            .try_send(ResponseItem::Control(Bytes::from(
                rmp_serde::to_vec_named(&reply).unwrap(),
            )))
            .unwrap();
        let response = response.await.unwrap().unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), 1 << 16).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap(), items)
    }

    /// A missing, negative, or non-numeric rank is Python's 400 before any
    /// lookup; the deprecated alias behaves the same.
    #[tokio::test]
    async fn transfer_engine_info_rejects_a_bad_rank_before_looking_it_up() {
        for uri in [
            "/remote_instance_transfer_engine_info",
            "/remote_instance_transfer_engine_info?rank=-1",
            "/remote_instance_transfer_engine_info?rank=abc",
            "/get_remote_instance_transfer_engine_info",
        ] {
            let (state, _intake_rx) = test_state();
            let response = routes()
                .with_state(state)
                .oneshot(
                    HttpRequest::builder()
                        .method("GET")
                        .uri(uri)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{uri}");
            let bytes = to_bytes(response.into_body(), 1 << 16).await.unwrap();
            let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(
                body["error"]["message"], "Missing or invalid rank parameter",
                "{uri}"
            );
        }
    }

    /// A single-rank read answers the bare parameter; a missing one is
    /// Python's 400; the group routes answer `{success, message}` with 200
    /// or 400 and fill Python's defaults onto the wire.
    #[tokio::test]
    async fn weight_routes_keep_python_shapes_and_defaults() {
        let (status, body, items) = call(
            "/get_weights_by_name",
            r#"{"name": "w"}"#,
            serde_json::json!({"parameter": [1.5, 2]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, serde_json::json!([1.5, 2]));
        assert_eq!(items[4].as_i64(), Some(100), "default truncate_size");

        let (status, body, _) = call(
            "/get_weights_by_name",
            r#"{"name": "w", "truncate_size": 3}"#,
            serde_json::json!({"parameter": null}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["message"], "Get parameter by name failed");

        let (status, body, items) = call(
            "/init_weights_update_group",
            r#"{"master_address": "h", "master_port": 1, "rank_offset": 0, "world_size": 2}"#,
            serde_json::json!({"success": true, "message": "ok"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, serde_json::json!({"success": true, "message": "ok"}));
        assert_eq!(items[7].as_str(), Some("weight_update_group"));
        assert_eq!(items[8].as_str(), Some("nccl"));

        let (status, body, items) = call(
            "/send_weights_to_remote_instance",
            r#"{"master_address": "h", "ports": "1,2"}"#,
            serde_json::json!({"success": false, "message": "busy"}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["success"], false);
        assert_eq!(items[5].as_str(), Some("weight_send_group"));
    }

    /// Each update route answers Python's body and status: a bare
    /// `num_paused_requests` on one rank, `{success, message}` for the
    /// session updates, the plugin counts, defaults filled onto the wire,
    /// and the body's base64 decoded.
    #[tokio::test]
    async fn update_routes_keep_python_shapes_and_defaults() {
        let (status, body, items) = call(
            "/update_weights_from_disk",
            r#"{"model_path": "/m", "weight_version": "v9"}"#,
            serde_json::json!({"success": true, "message": "done", "num_paused_requests": 0}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body,
            serde_json::json!({
                "success": true,
                "message": "done Weight version updated to v9.",
                "num_paused_requests": 0
            })
        );
        assert_eq!(items[12].as_bool(), Some(true), "flush_cache defaults on");

        let (status, body, items) = call(
            "/begin_weight_update",
            "{}",
            serde_json::json!({"success": false, "message": "already open"}),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(
            body,
            serde_json::json!({"success": false, "message": "already open"})
        );
        assert_eq!(items[3].as_str(), Some("all"), "selector defaults to all");

        let (status, body, _) = call(
            "/end_weight_update",
            "",
            serde_json::json!({"success": true, "message": "Success"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["success"], true);

        let (status, body, items) = call(
            "/update_weights_from_tensor",
            r#"{"serialized_named_tensors": ["gAU="], "load_format": "flattened_bucket"}"#,
            serde_json::json!({"success": true, "message": "ok"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, serde_json::json!({"success": true, "message": "ok"}));
        assert_eq!(
            items[3],
            rmpv::Value::Array(vec![rmpv::Value::Binary(vec![0x80, 0x05])])
        );
        assert_eq!(items[4].as_str(), Some("flattened_bucket"));
        assert_eq!(items[8].as_str(), Some("all"));

        let (status, body, items) = call(
            "/plugin_weight_update",
            r#"{"handler": "h", "payload": {"k": 1}, "abort_all_requests": true}"#,
            serde_json::json!({"success": true, "message": "ok", "received_tensor_counts": [1]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body,
            serde_json::json!({"success": true, "message": "ok", "received_tensor_counts": [1]})
        );
        assert_eq!(items[6].as_bool(), Some(true));

        let (status, body, items) = call(
            "/update_weights_from_ipc",
            r#"{"zmq_handles": {"0": "ipc:///a"}}"#,
            serde_json::json!({"success": true, "message": "ok"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["success"], true);
        assert_eq!(
            items[3],
            rmpv::Value::Map(vec![("0".into(), "ipc:///a".into())])
        );

        let (status, body, items) = call(
            "/update_weights_from_distributed",
            r#"{"names": ["w"], "dtypes": ["bfloat16"], "shapes": [[2, 3]]}"#,
            serde_json::json!({"success": true, "message": "ok"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["success"], true);
        assert_eq!(
            items[5],
            rmpv::Value::Array(vec![rmpv::Value::Array(vec![2.into(), 3.into()])])
        );
        assert_eq!(items[6].as_str(), Some("weight_update_group"));
    }

    /// A malformed body is refused before any scheduler sees it: bad base64,
    /// a non-object payload, a missing required field, a bad selector.
    #[tokio::test]
    async fn update_routes_refuse_malformed_bodies_before_the_scheduler() {
        for (uri, body) in [
            (
                "/update_weights_from_tensor",
                r#"{"serialized_named_tensors": ["not base64!"]}"#,
            ),
            (
                "/plugin_weight_update",
                r#"{"handler": "h", "payload": [1]}"#,
            ),
            ("/update_weights_from_disk", r#"{"load_format": "auto"}"#),
            ("/begin_weight_update", r#"{"selector": "sideways"}"#),
            ("/update_weights_from_ipc", r#"{}"#),
        ] {
            let (state, intake_rx) = test_state();
            let response = routes()
                .with_state(state)
                .oneshot(
                    HttpRequest::builder()
                        .method("POST")
                        .uri(uri)
                        .header(CONTENT_TYPE, "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{uri} {body}");
            assert!(
                intake_rx.try_recv().is_err(),
                "{uri}: refused before the scheduler"
            );
        }
    }

    /// Without `--enable-lora` (which Rust-server mode refuses at launch),
    /// the LoRA routes answer Python's whole `LoRAUpdateOutput` as a 400;
    /// `/get_weights_by_name` takes GET as Python does.
    #[tokio::test]
    async fn lora_routes_answer_pythons_disabled_refusal() {
        for (uri, body) in [
            (
                "/load_lora_adapter",
                r#"{"lora_name": "a", "lora_path": "/a"}"#,
            ),
            (
                "/load_lora_adapter_from_tensors",
                r#"{"lora_name": "a", "config_dict": {}, "serialized_named_tensors": []}"#,
            ),
            ("/unload_lora_adapter", r#"{"lora_name": "a"}"#),
        ] {
            let (state, intake_rx) = test_state();
            let response = routes()
                .with_state(state)
                .oneshot(
                    HttpRequest::builder()
                        .method("POST")
                        .uri(uri)
                        .header(CONTENT_TYPE, "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{uri}");
            let bytes = to_bytes(response.into_body(), 1 << 16).await.unwrap();
            let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(
                body,
                serde_json::json!({
                    "rid": null, "http_worker_ipc": null, "success": false,
                    "error_message": "LoRA is not enabled. Please set `--enable-lora` to enable LoRA.",
                    "loaded_adapters": null
                }),
                "{uri}"
            );
            assert!(intake_rx.try_recv().is_err());
        }

        let (state, intake_rx) = test_state();
        let response = tokio::spawn(
            routes().with_state(state).oneshot(
                HttpRequest::builder()
                    .method("GET")
                    .uri("/get_weights_by_name")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(r#"{"name": "w"}"#))
                    .unwrap(),
            ),
        );
        let TmEvent::Intake { request, admission } = intake_rx.recv_async().await.unwrap() else {
            panic!("GET must submit the read");
        };
        assert!(admission.try_accept());
        request
            .sink
            .try_send(ResponseItem::Control(Bytes::from(
                rmp_serde::to_vec_named(&serde_json::json!({"parameter": [1]})).unwrap(),
            )))
            .unwrap();
        assert_eq!(response.await.unwrap().unwrap().status(), StatusCode::OK);
    }
}
