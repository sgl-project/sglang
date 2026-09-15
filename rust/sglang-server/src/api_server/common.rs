//! Common control-plane endpoints — `/server_info`, `/get_model_info`
//! (+ `/model_info` alias), backed by typed operations on
//! [`crate::frontend::FrontendHandle`]. Data-plane endpoints (incl. `/health*`,
//! which round-trips a generate probe) live in the sibling `native_api` and
//! `openai` modules; the shared `AppState` lives in the parent
//! `api_server` module.

use axum::{
    Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
};
use std::sync::Arc;

use super::app::AppState;
use super::frontend_error_status;
use super::native_api::native_error;
use crate::frontend::{FrontendError, ServerInfo};

/// The routes this module owns, mounted by `api_server::serve`.
pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        // Control-plane: the frontend performs the typed operation and this
        // adapter renders its result as one non-streamed JSON response.
        .route("/server_info", get(server_info))
        // Static config, no scheduler round-trip. `/get_model_info` (+ `/model_info`
        // alias).
        .route("/get_model_info", get(model_info))
        .route("/model_info", get(model_info))
}

/// Await the frontend's complete semantic server-info result. HTTP status and
/// serialization stay here; scheduler control bytes stay behind the contract.
async fn await_server_info(state: &AppState) -> Result<ServerInfo, Response> {
    match state.frontend.server_info().await {
        Ok(server_info) => Ok(server_info),
        Err(FrontendError::Unavailable) => Err(native_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "service unavailable",
            false,
        )),
        // Preserve the established HTTP behavior for a scheduler-side control
        // stream that disappears, while non-HTTP adapters see Internal via
        // FrontendError::kind().
        Err(FrontendError::ResponseTruncated) => {
            Err((StatusCode::from_u16(499).unwrap(), "request aborted").into_response())
        }
        Err(error @ FrontendError::InvalidResponse(_)) => {
            tracing::error!(%error, "server_info: invalid runtime response");
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                "bad server_info response",
            )
                .into_response())
        }
        Err(error) => Err((frontend_error_status(&error), error.to_string()).into_response()),
    }
}

/// `GET /get_model_info` (+ `/model_info` alias) — serialize the shared static
/// model metadata (no scheduler round-trip).
async fn model_info(State(state): State<Arc<AppState>>) -> Response {
    let info = state.frontend.model_info();
    (
        StatusCode::OK,
        [("content-type", "application/json")],
        serde_json::to_vec(&info).unwrap_or_default(),
    )
        .into_response()
}

/// `GET /server_info` — serialize typed, allowlisted runtime metrics plus
/// curated launch metadata. The raw scheduler state embeds
/// `api_key`/`admin_api_key` and is never visible to this HTTP adapter.
///
/// TODO(server_info): Python also includes `kv_events`; add once plumbed.
async fn server_info(State(state): State<Arc<AppState>>) -> Response {
    let info = match await_server_info(&state).await {
        Ok(info) => info,
        Err(resp) => return resp,
    };
    match serde_json::to_vec(&info) {
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

#[cfg(test)]
mod tests {
    use axum::{
        body::{Body, to_bytes},
        http::{Request as HttpRequest, header::CONTENT_TYPE},
    };
    use bytes::Bytes;
    use tower::ServiceExt;

    use super::*;
    use crate::frontend::{FrontendConfig, FrontendHandle, FrontendMetadata};
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
        let frontend = FrontendHandle::new(
            intake_tx,
            abort_tx,
            FrontendConfig {
                response_capacity: 8,
                response_activity: Default::default(),
                startup_ready: true,
                is_disaggregation: false,
                mm_limits: Default::default(),
                metadata: FrontendMetadata::from(server_args.as_ref()),
            },
        );
        (
            Arc::new(AppState {
                frontend,
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
                rmpv::Value::from("api_key"),
                rmpv::Value::from("must-not-leak"),
            ),
            (
                rmpv::Value::from("last_gen_throughput"),
                rmpv::Value::from(1.5),
            ),
            (
                rmpv::Value::from("memory_usage"),
                rmpv::Value::Map(vec![(
                    rmpv::Value::from("token_capacity"),
                    rmpv::Value::from(8192),
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
        assert_eq!(body["model_path"], "/model");
        assert_eq!(body["served_model_name"], "served");
        assert_eq!(body["tokenizer_path"], "/tokenizer");
        assert_eq!(body["max_context_length"], 4096);
        assert_eq!(body["max_total_num_tokens"], 8192);
        assert_eq!(body["version"], "1.2.3");
        assert_eq!(body["internal_states"][0]["last_gen_throughput"], 1.5);
        assert_eq!(
            body["internal_states"][0]["memory_usage"]["token_capacity"],
            8192
        );
        assert!(!body.to_string().contains("must-not-leak"));
        assert!(abort_rx.try_recv().is_err());
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
