//! HiCache routes with dedicated state shared with the gRPC adapter.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use sglang_api_types::api::v1 as api;

use crate::api_server::core::{CoreError, HiCacheManager};

pub(super) fn routes(manager: Arc<HiCacheManager>) -> Router {
    Router::new()
        .route(
            "/hicache/storage-backend",
            get(status).put(attach).delete(detach),
        )
        .route("/hicache/storage-backend/clear", post(clear))
        .with_state(manager)
}

async fn status(State(manager): State<Arc<HiCacheManager>>) -> Json<api::HiCacheStatus> {
    Json(manager.status().await)
}

async fn attach(
    State(manager): State<Arc<HiCacheManager>>,
    body: Result<Json<api::AttachHiCacheStorageRequest>, JsonRejection>,
) -> Response {
    let body = match body {
        Ok(Json(body)) => body,
        Err(error) => return (StatusCode::UNPROCESSABLE_ENTITY, error.body_text()).into_response(),
    };
    if body.hicache_storage_backend.is_none() {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            "missing field `hicache_storage_backend`",
        )
            .into_response();
    }
    storage_response(manager.attach(body).await)
}

async fn detach(State(manager): State<Arc<HiCacheManager>>) -> Response {
    storage_response(manager.detach().await)
}

async fn clear(State(manager): State<Arc<HiCacheManager>>) -> Response {
    storage_response(manager.clear().await)
}

fn storage_response(result: Result<api::HiCacheStorageResponse, CoreError>) -> Response {
    match result {
        Ok(output) => {
            let status = if output.success {
                StatusCode::OK
            } else {
                StatusCode::BAD_REQUEST
            };
            (status, Json(output)).into_response()
        }
        Err(error) => (error.http_status(), error.to_string()).into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{Body, to_bytes},
        http::Request,
    };
    use bytes::Bytes;
    use serde_json::{Value, json};
    use tower::ServiceExt;

    use crate::api_server::core::{CoreConfig, CoreHandle};
    use crate::message::{config::ServerArgs, request::RequestKind, response::ResponseItem};
    use crate::tokenizer_manager::wiring::TmEvent;

    fn app() -> (Router, flume::Receiver<TmEvent>) {
        app_with_args(ServerArgs::default())
    }

    fn app_with_args(args: ServerArgs) -> (Router, flume::Receiver<TmEvent>) {
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, _) = flume::unbounded();
        let core = CoreHandle::new(
            intake_tx,
            abort_tx,
            CoreConfig {
                response_capacity: 8,
                response_activity: Default::default(),
                startup_ready: true,
                is_disaggregation: false,
                mm_limits: Default::default(),
                metadata: (&args).into(),
                metrics: None,
            },
        );
        (
            routes(Arc::new(HiCacheManager::new(core, &args))),
            intake_rx,
        )
    }

    // Exercise the real router/core boundary, replacing only the scheduler reply.
    async fn exchange(
        method: &str,
        uri: &str,
        body: Value,
        tag: &str,
        fields: Value,
        reply: ResponseItem,
    ) -> (StatusCode, String) {
        exchange_on(&app(), method, uri, body, tag, fields, reply).await
    }

    async fn exchange_on(
        app: &(Router, flume::Receiver<TmEvent>),
        method: &str,
        uri: &str,
        body: Value,
        tag: &str,
        fields: Value,
        reply: ResponseItem,
    ) -> (StatusCode, String) {
        let (router, intake) = app;
        let response = tokio::spawn(
            router.clone().oneshot(
                Request::builder()
                    .method(method)
                    .uri(uri)
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            ),
        );
        let TmEvent::Intake { request, admission } =
            tokio::time::timeout(std::time::Duration::from_secs(5), intake.recv_async())
                .await
                .expect("handler did not submit a control request")
                .unwrap()
        else {
            panic!("expected control intake");
        };
        assert!(admission.try_accept());
        let RequestKind::Control(control) = &request.kind else {
            panic!("expected control request")
        };
        let header: Value = rmp_serde::from_slice(&control.encode().unwrap()).unwrap();
        let mut expected = vec![json!(tag), json!(request.rid.as_str()), Value::Null];
        expected.extend(fields.as_array().unwrap().iter().cloned());
        assert_eq!(header, Value::Array(expected));
        request.sink.try_send(reply).unwrap();
        let response = tokio::time::timeout(std::time::Duration::from_secs(5), response)
            .await
            .expect("handler did not finish")
            .unwrap()
            .unwrap();
        let status = response.status();
        if matches!(status, StatusCode::OK | StatusCode::BAD_REQUEST) {
            assert_eq!(response.headers()["content-type"], "application/json");
        }
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        // Normalize JSON key order for comparisons, preserving plain-text errors.
        let body =
            serde_json::from_str::<Value>(&body).map_or(body.clone(), |value| value.to_string());
        (status, body)
    }

    fn reply(value: Value) -> ResponseItem {
        ResponseItem::Control(Bytes::from(rmp_serde::to_vec_named(&value).unwrap()))
    }

    /// Clear maps scheduler success to HTTP 200 and failure to HTTP 400.
    #[tokio::test]
    async fn clear_relays_success_and_failure() {
        for (success, status) in [(true, StatusCode::OK), (false, StatusCode::BAD_REQUEST)] {
            assert_eq!(
                exchange(
                    "POST",
                    "/hicache/storage-backend/clear",
                    Value::Null,
                    "ClearHiCacheReqInput",
                    json!([]),
                    reply(json!({"success": success}))
                )
                .await,
                (
                    status,
                    json!({"success": success, "message": ""}).to_string()
                )
            );
        }
    }

    /// Detach returns the generated response, defaulting absent messages to empty.
    #[tokio::test]
    async fn detach_preserves_message_and_handles_empty_or_missing_message() {
        for payload in [
            json!({"success": true, "message": "Already detached."}),
            json!({"success": false, "message": "scheduler is not idle"}),
            json!({"success": true, "message": ""}),
            json!({"success": false}),
        ] {
            let (status, body) = exchange(
                "DELETE",
                "/hicache/storage-backend",
                Value::Null,
                "DetachHiCacheStorageReqInput",
                json!([]),
                reply(payload.clone()),
            )
            .await;
            let success = payload["success"].as_bool().unwrap();
            assert_eq!(
                status,
                if success {
                    StatusCode::OK
                } else {
                    StatusCode::BAD_REQUEST
                }
            );
            assert_eq!(
                serde_json::from_str::<Value>(&body).unwrap(),
                json!({
                    "success": success,
                    "message": payload.get("message").cloned().unwrap_or(json!("")),
                })
            );
        }
    }

    /// Malformed replies and scheduler errors produce HTTP 500 for all three operations.
    #[tokio::test]
    async fn control_operations_report_scheduler_errors() {
        for (method, uri, tag, operation, input, fields) in [
            (
                "POST",
                "/hicache/storage-backend/clear",
                "ClearHiCacheReqInput",
                "clear",
                Value::Null,
                json!([]),
            ),
            (
                "DELETE",
                "/hicache/storage-backend",
                "DetachHiCacheStorageReqInput",
                "detach",
                Value::Null,
                json!([]),
            ),
            (
                "PUT",
                "/hicache/storage-backend",
                "AttachHiCacheStorageReqInput",
                "attach",
                json!({"hicache_storage_backend": "file"}),
                json!(["file", null, null, null]),
            ),
        ] {
            for (response, expected_error) in [
                (
                    reply(json!({"message": "missing success"})),
                    format!("invalid HiCache {operation} response"),
                ),
                (
                    reply(json!({"success": "false"})),
                    format!("invalid HiCache {operation} response"),
                ),
                (
                    ResponseItem::Control(Bytes::from_static(&[0xc1])),
                    format!("invalid HiCache {operation} response"),
                ),
                (
                    ResponseItem::Error(crate::utils::error::Error::Internal(
                        "backend unavailable".into(),
                    )),
                    "backend unavailable".into(),
                ),
            ] {
                let (status, body) =
                    exchange(method, uri, input.clone(), tag, fields.clone(), response).await;
                assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
                assert!(body.contains(&expected_error), "{body}");
            }
        }
    }

    /// Attach forwards all options and returns the scheduler result and message.
    #[tokio::test]
    async fn attach_http_forwards_options_and_relays_result() {
        let body = json!({
            "hicache_storage_backend": "file",
            "hicache_storage_backend_extra_config_json": "{}",
            "hicache_storage_prefetch_policy": "timeout",
            "hicache_write_policy": "write_through",
        });
        for (success, status) in [(true, StatusCode::OK), (false, StatusCode::BAD_REQUEST)] {
            assert_eq!(
                exchange(
                    "PUT",
                    "/hicache/storage-backend",
                    body.clone(),
                    "AttachHiCacheStorageReqInput",
                    json!(["file", "{}", "timeout", "write_through"]),
                    reply(json!({"success": success, "message": "backend detail"}))
                )
                .await,
                (
                    status,
                    json!({"success": success, "message": "backend detail"}).to_string()
                )
            );
        }
    }

    /// Python accepts empty names and null options; backend validation belongs to the scheduler.
    #[tokio::test]
    async fn attach_preserves_empty_strings_and_accepts_null_options() {
        for (body, fields) in [
            (
                json!({"hicache_storage_backend": ""}),
                json!(["", null, null, null]),
            ),
            (
                json!({
                    "hicache_storage_backend": "file",
                    "hicache_storage_backend_extra_config_json": null,
                    "hicache_storage_prefetch_policy": null,
                    "hicache_write_policy": null,
                }),
                json!(["file", null, null, null]),
            ),
            (
                json!({
                    "hicache_storage_backend": "file",
                    "hicache_storage_backend_extra_config_json": "",
                    "hicache_storage_prefetch_policy": "",
                    "hicache_write_policy": "",
                }),
                json!(["file", "", "", ""]),
            ),
        ] {
            assert_eq!(
                exchange(
                    "PUT",
                    "/hicache/storage-backend",
                    body,
                    "AttachHiCacheStorageReqInput",
                    fields,
                    reply(json!({"success": true}))
                )
                .await
                .0,
                StatusCode::OK
            );
        }
    }

    /// Empty bodies, invalid JSON, and missing backend fields fail before scheduler submission.
    #[tokio::test]
    async fn attach_rejects_invalid_body_before_scheduler_submission() {
        for (body, expected_error) in [
            ("", "EOF while parsing"),
            ("{", "EOF while parsing"),
            ("{}", "missing field `hicache_storage_backend`"),
            (
                r#"{"hicache_storage_backend":null}"#,
                "missing field `hicache_storage_backend`",
            ),
            (r#"{"hicache_storage_backend":1}"#, "invalid type"),
            (
                r#"{"hicache_storage_backend":"file","hicache_write_policy":{}}"#,
                "invalid type",
            ),
        ] {
            let (router, intake) = app();
            let response = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                router.oneshot(
                    Request::builder()
                        .method("PUT")
                        .uri("/hicache/storage-backend")
                        .header("content-type", "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                ),
            )
            .await
            .expect("invalid attach request must not wait for scheduler")
            .unwrap();
            assert_eq!(
                response.status(),
                StatusCode::UNPROCESSABLE_ENTITY,
                "{body:?}"
            );
            let response_body = to_bytes(response.into_body(), 4096).await.unwrap();
            let response_body = std::str::from_utf8(&response_body).unwrap();
            assert!(response_body.contains(expected_error), "{response_body}");
            assert!(
                intake.is_empty(),
                "invalid request reached scheduler: {body:?}"
            );
        }
    }

    /// GET returns a JSON object with the same four configuration fields as Python.
    #[tokio::test]
    async fn get_status_returns_python_configuration_fields() {
        let (router, _intake) = app();
        let body = read_status(&router).await;
        assert_eq!(
            body,
            json!({
                "hicache_storage_backend": null,
                "hicache_storage_backend_extra_config": null,
                "hicache_storage_prefetch_policy": null,
                "hicache_write_policy": null,
            })
        );
    }

    /// Only successful attach/detach operations update GET. Omitted options and
    /// clear preserve the current configuration.
    #[tokio::test]
    async fn get_status_tracks_only_successful_control_operations() {
        let app = app_with_args(ServerArgs {
            hicache_storage_backend: Some("file".into()),
            hicache_storage_backend_extra_config: Some("{}".into()),
            hicache_storage_prefetch_policy: Some("timeout".into()),
            hicache_write_policy: Some("write_through".into()),
            ..ServerArgs::default()
        });
        let initial = json!({
            "hicache_storage_backend": "file",
            "hicache_storage_backend_extra_config": "{}",
            "hicache_storage_prefetch_policy": "timeout",
            "hicache_write_policy": "write_through",
        });
        assert_eq!(read_status(&app.0).await, initial);
        assert!(app.1.is_empty(), "GET must not submit scheduler work");

        let body = json!({
            "hicache_storage_backend": "file",
            "hicache_storage_backend_extra_config_json": "{\"path\":\"/cache\"}",
            "hicache_storage_prefetch_policy": "wait_complete",
            "hicache_write_policy": "write_back",
        });
        let fields = json!([
            "file",
            "{\"path\":\"/cache\"}",
            "wait_complete",
            "write_back"
        ]);
        for success in [false, true] {
            exchange_on(
                &app,
                "PUT",
                "/hicache/storage-backend",
                body.clone(),
                "AttachHiCacheStorageReqInput",
                fields.clone(),
                reply(json!({"success": success})),
            )
            .await;
            if !success {
                assert_eq!(read_status(&app.0).await, initial);
            }
        }
        let attached = json!({
            "hicache_storage_backend": "file",
            "hicache_storage_backend_extra_config": "{\"path\":\"/cache\"}",
            "hicache_storage_prefetch_policy": "wait_complete",
            "hicache_write_policy": "write_back",
        });
        assert_eq!(read_status(&app.0).await, attached);
        // Omitted options retain their current values, like Python's config updates.
        exchange_on(
            &app,
            "PUT",
            "/hicache/storage-backend",
            json!({"hicache_storage_backend": "file"}),
            "AttachHiCacheStorageReqInput",
            json!(["file", null, null, null]),
            reply(json!({"success": true})),
        )
        .await;
        assert_eq!(read_status(&app.0).await, attached);
        exchange_on(
            &app,
            "POST",
            "/hicache/storage-backend/clear",
            Value::Null,
            "ClearHiCacheReqInput",
            json!([]),
            reply(json!({"success": true})),
        )
        .await;
        assert_eq!(read_status(&app.0).await, attached);
        for success in [false, true] {
            exchange_on(
                &app,
                "DELETE",
                "/hicache/storage-backend",
                Value::Null,
                "DetachHiCacheStorageReqInput",
                json!([]),
                reply(json!({"success": success})),
            )
            .await;
            let mut expected = attached.clone();
            if success {
                expected["hicache_storage_backend"] = Value::Null;
                expected["hicache_storage_backend_extra_config"] = Value::Null;
            }
            assert_eq!(read_status(&app.0).await, expected);
        }
    }

    async fn read_status(router: &Router) -> Value {
        let response = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            router.clone().oneshot(
                Request::builder()
                    .uri("/hicache/storage-backend")
                    .body(Body::empty())
                    .unwrap(),
            ),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert!(
            response.headers()["content-type"]
                .to_str()
                .unwrap()
                .starts_with("application/json")
        );
        serde_json::from_slice(&to_bytes(response.into_body(), 4096).await.unwrap()).unwrap()
    }
}
