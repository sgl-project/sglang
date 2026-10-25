//! Weight-management control routes: the state-free subset, backed by the
//! typed operations in `handlers::weights`.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
};
use serde_json::Value;
use sglang_api_types::api::v1 as api;

use super::app::AppState;
use super::native_api::native_error;
use crate::api_server::core::CoreError;
use crate::api_server::handlers::WeightGroupOutcome;
use crate::message::io_struct::{WeightsSendGroup, WeightsUpdateGroup};

pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route(
            "/get_weights_by_name",
            post(get_weights_by_name).put(get_weights_by_name),
        )
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
}

/// The parsed body, or the rejection text for a 400.
fn body<T>(body: Result<Json<T>, JsonRejection>) -> Result<T, String> {
    body.map(|Json(body)| body)
        .map_err(|rejection| rejection.body_text())
}

/// `POST|PUT /get_weights_by_name` — Python answers the bare parameter on a
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
            Some(Value::Null) | None => native_error(
                StatusCode::BAD_REQUEST,
                "Get parameter by name failed",
                false,
            ),
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
}
