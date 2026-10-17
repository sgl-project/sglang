//! HiCache storage management endpoints. The core submits attach, detach, and
//! clear operations to the scheduler and tracks the configuration returned by GET.

use std::sync::Arc;

use axum::{
    Router,
    body::Bytes,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};

use serde::Deserialize;

use super::app::AppState;

#[derive(Deserialize)]
struct AttachHiCacheStorageReqInputBody {
    hicache_storage_backend: String,
    hicache_storage_backend_extra_config_json: Option<String>,
    hicache_storage_prefetch_policy: Option<String>,
    hicache_write_policy: Option<String>,
}

/// The routes this module owns, mounted by `api_server::serve`.
pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route(
            "/hicache/storage-backend",
            get(status).put(attach).delete(detach),
        )
        .route("/hicache/storage-backend/clear", post(clear))
}

async fn status(State(state): State<Arc<AppState>>) -> Response {
    axum::Json(state.core.hicache_status().await).into_response()
}

async fn attach(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    if body.is_empty() {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            "Request body is required.\n",
        )
            .into_response();
    }
    let body: AttachHiCacheStorageReqInputBody = match serde_json::from_slice(&body) {
        Ok(body) => body,
        Err(error) => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                format!("Invalid attach request: {error}\n"),
            )
                .into_response();
        }
    };
    match state
        .core
        .attach_hicache_storage(
            body.hicache_storage_backend,
            body.hicache_storage_backend_extra_config_json,
            body.hicache_storage_prefetch_policy,
            body.hicache_write_policy,
        )
        .await
    {
        Ok(output) => {
            let (status, summary) = if output.success {
                (StatusCode::OK, "HiCache storage backend attached.")
            } else {
                (
                    StatusCode::BAD_REQUEST,
                    "Failed to attach HiCache storage backend.",
                )
            };
            storage_response(status, summary, &output.message)
        }
        Err(error) => (error.http_status(), error.to_string()).into_response(),
    }
}

async fn detach(State(state): State<Arc<AppState>>) -> Response {
    match state.core.detach_hicache_storage().await {
        Ok(output) => {
            let (status, summary) = if output.success {
                (StatusCode::OK, "HiCache storage backend detached.")
            } else {
                (
                    StatusCode::BAD_REQUEST,
                    "Failed to detach HiCache storage backend.",
                )
            };
            storage_response(status, summary, &output.message)
        }
        Err(error) => (error.http_status(), error.to_string()).into_response(),
    }
}

async fn clear(State(state): State<Arc<AppState>>) -> Response {
    match state.core.clear_hicache_storage().await {
        Ok(output) => {
            let (status, summary) = if output.success {
                (
                    StatusCode::OK,
                    "Hierarchical cache storage backend cleared.\n",
                )
            } else {
                (
                    StatusCode::BAD_REQUEST,
                    "Failed to clear hierarchical cache storage backend.\n",
                )
            };
            (status, summary).into_response()
        }
        Err(error) => (error.http_status(), error.to_string()).into_response(),
    }
}

fn storage_response(status: StatusCode, summary: &str, message: &str) -> Response {
    let mut body = format!("{summary}\n");
    if !message.is_empty() {
        body.push_str(message);
        body.push('\n');
    }
    (status, body).into_response()
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
    use crate::message::{
        config::ServerArgs,
        io_struct::{AttachHiCacheStorageReqInput, ControlRequest},
        request::RequestKind,
        response::ResponseItem,
    };
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
                hicache: (&args).into(),
            },
        );
        (
            routes().with_state(Arc::new(AppState {
                core,
                server_args: Arc::new(args),
                chat_formatter: None,
            })),
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
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        (status, String::from_utf8(body.to_vec()).unwrap())
    }

    fn reply(value: Value) -> ResponseItem {
        ResponseItem::Control(Bytes::from(rmp_serde::to_vec_named(&value).unwrap()))
    }

    /// Clear maps scheduler success to HTTP 200 and failure to HTTP 400.
    #[tokio::test]
    async fn clear_relays_success_and_failure() {
        for (success, status, text) in [
            (
                true,
                StatusCode::OK,
                "Hierarchical cache storage backend cleared.\n",
            ),
            (
                false,
                StatusCode::BAD_REQUEST,
                "Failed to clear hierarchical cache storage backend.\n",
            ),
        ] {
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
                (status, text.into())
            );
        }
    }

    /// Detach preserves backend details without adding a blank line for an empty message.
    #[tokio::test]
    async fn detach_preserves_message_and_handles_empty_or_missing_message() {
        for (payload, status, text) in [
            (
                json!({"success": true, "message": "Already detached."}),
                StatusCode::OK,
                "HiCache storage backend detached.\nAlready detached.\n",
            ),
            (
                json!({"success": false, "message": "Reject detach: scheduler is not idle."}),
                StatusCode::BAD_REQUEST,
                "Failed to detach HiCache storage backend.\nReject detach: scheduler is not idle.\n",
            ),
            (
                json!({"success": true, "message": ""}),
                StatusCode::OK,
                "HiCache storage backend detached.\n",
            ),
            (
                json!({"success": false}),
                StatusCode::BAD_REQUEST,
                "Failed to detach HiCache storage backend.\n",
            ),
        ] {
            assert_eq!(
                exchange(
                    "DELETE",
                    "/hicache/storage-backend",
                    Value::Null,
                    "DetachHiCacheStorageReqInput",
                    json!([]),
                    reply(payload)
                )
                .await,
                (status, text.into())
            );
        }
    }

    /// Malformed scheduler replies produce HTTP 500 for attach, detach, and clear.
    #[tokio::test]
    async fn control_operations_reject_invalid_scheduler_replies() {
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
            for bytes in [
                rmp_serde::to_vec_named(&json!({"message": "missing success"})).unwrap(),
                rmp_serde::to_vec_named(&json!({"success": "false"})).unwrap(),
                vec![0xc1], // Reserved MessagePack byte: not a valid response.
            ] {
                let (status, body) = exchange(
                    method,
                    uri,
                    input.clone(),
                    tag,
                    fields.clone(),
                    ResponseItem::Control(Bytes::from(bytes)),
                )
                .await;
                assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
                assert!(
                    body.contains(&format!("invalid HiCache {operation} response")),
                    "{body}"
                );
            }
        }
    }

    /// Scheduler errors reach the HTTP response with their diagnostic message.
    #[tokio::test]
    async fn control_operations_relay_scheduler_errors() {
        for (method, uri, tag, input, fields) in [
            (
                "POST",
                "/hicache/storage-backend/clear",
                "ClearHiCacheReqInput",
                Value::Null,
                json!([]),
            ),
            (
                "DELETE",
                "/hicache/storage-backend",
                "DetachHiCacheStorageReqInput",
                Value::Null,
                json!([]),
            ),
            (
                "PUT",
                "/hicache/storage-backend",
                "AttachHiCacheStorageReqInput",
                json!({"hicache_storage_backend": "file"}),
                json!(["file", null, null, null]),
            ),
        ] {
            let (status, body) = exchange(
                method,
                uri,
                input,
                tag,
                fields,
                ResponseItem::Error(crate::utils::error::Error::Internal(
                    "backend unavailable".into(),
                )),
            )
            .await;
            assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
            assert!(body.contains("backend unavailable"), "{body}");
        }
    }

    /// Attach uses the Python MessagePack field order, including null optional fields.
    #[test]
    fn attach_wire_matches_python_field_order_and_optional_values() {
        for optional in [
            None,
            Some(("{\"path\":\"/tmp/cache\"}", "timeout", "write_through")),
        ] {
            let (config, prefetch, write) = optional.map_or((None, None, None), |(c, p, w)| {
                (Some(c.to_owned()), Some(p.to_owned()), Some(w.to_owned()))
            });
            let request =
                ControlRequest::AttachHiCacheStorageReqInput(AttachHiCacheStorageReqInput::new(
                    "attach-test".into(),
                    "file".into(),
                    config.clone(),
                    prefetch.clone(),
                    write.clone(),
                ));
            let actual: Value = rmp_serde::from_slice(&request.encode().unwrap()).unwrap();
            assert_eq!(
                actual,
                json!([
                    "AttachHiCacheStorageReqInput",
                    "attach-test",
                    null,
                    "file",
                    config,
                    prefetch,
                    write
                ])
            );
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
        for (success, status, summary) in [
            (true, StatusCode::OK, "HiCache storage backend attached."),
            (
                false,
                StatusCode::BAD_REQUEST,
                "Failed to attach HiCache storage backend.",
            ),
        ] {
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
                (status, format!("{summary}\nbackend detail\n"))
            );
        }
    }

    /// Empty bodies, invalid JSON, and missing backend fields fail before scheduler submission.
    #[tokio::test]
    async fn attach_rejects_invalid_body_before_scheduler_submission() {
        for (body, expected_error) in [
            ("", "Request body is required.\n"),
            ("{", "Invalid attach request:"),
            ("{}", "missing field `hicache_storage_backend`"),
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
        let fields = body.as_object().expect("status must be a JSON object");
        assert_eq!(fields.len(), 4);
        for name in [
            "hicache_storage_backend",
            "hicache_storage_backend_extra_config",
            "hicache_storage_prefetch_policy",
            "hicache_write_policy",
        ] {
            assert!(fields.contains_key(name), "missing {name}");
        }
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
