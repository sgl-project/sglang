//! Vertex AI custom prediction endpoint.
//!
//! The Vertex wire envelope (`instances` + `parameters`) is translated into the
//! protocol-neutral generation requests used by the native batch executor.
//! Successful unary output is wrapped in `{"predictions": ...}`; streaming
//! responses pass through unchanged, matching the Python frontend.

use axum::{
    Json, Router,
    body::to_bytes,
    extract::{State, rejection::JsonRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
};
use serde::{Deserialize, de::IgnoredAny};
use serde_json::{Value, json};
use thiserror::Error;

use crate::{
    api_server::{AppState, native_api::generate_batch},
    message::{GenerateBody, OneOrMany, TokenIds},
};

const DEFAULT_PREDICT_ROUTE: &str = "/vertex_generate";

/// The routes this module owns, mounted by `api_server::serve`.
pub(super) fn routes() -> Router<AppState> {
    let route =
        std::env::var("AIP_PREDICT_ROUTE").unwrap_or_else(|_| DEFAULT_PREDICT_ROUTE.to_string());
    routes_at(&route)
}

fn routes_at(route: &str) -> Router<AppState> {
    Router::new().route(route, post(vertex_generate))
}

#[derive(Debug, Deserialize)]
struct VertexGenerateRequest {
    instances: Vec<VertexInstance>,
    #[serde(default)]
    parameters: Option<GenerateBody>,
}

#[derive(Debug, Deserialize)]
struct VertexInstance {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    input_ids: Option<TokenIds>,
    #[serde(default)]
    input_embeds: Option<Vec<IgnoredAny>>,
    #[serde(default)]
    image_data: Option<rmpv::Value>,
}

#[derive(Debug, Error)]
enum VertexError {
    #[error("{0}")]
    InvalidJson(String),

    #[error("`instances` must not be empty")]
    EmptyInstances,

    #[error("Either text, input_ids or input_embeds should be provided.")]
    MissingInput,

    #[error("`{0}` is not supported by the Rust frontend yet")]
    UnsupportedField(&'static str),

    #[error("`{0}` must be supplied through `instances`, not `parameters`")]
    InputInParameters(&'static str),

    #[error("Vertex instance {index} is missing `{name}`")]
    MissingInstanceInput { index: usize, name: &'static str },

    #[error(transparent)]
    InvalidRequest(#[from] crate::error::Error),
}

impl IntoResponse for VertexError {
    fn into_response(self) -> Response {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": {"message": self.to_string()}})),
        )
            .into_response()
    }
}

impl TryFrom<VertexGenerateRequest> for GenerateBody {
    type Error = VertexError;

    /// Translate the Vertex envelope into the native generation body.
    fn try_from(vertex: VertexGenerateRequest) -> Result<Self, Self::Error> {
        let VertexGenerateRequest {
            mut instances,
            parameters,
        } = vertex;
        let mut parameters = parameters.unwrap_or_default();
        for (present, name) in [
            (parameters.text.is_some(), "text"),
            (parameters.input_ids.is_some(), "input_ids"),
            (parameters.image_data.is_some(), "image_data"),
        ] {
            if present {
                return Err(VertexError::InputInParameters(name));
            }
        }
        if parameters.lora_path.is_some() {
            return Err(VertexError::UnsupportedField("lora_path"));
        }
        if parameters.return_routed_experts.unwrap_or_default() {
            return Err(VertexError::UnsupportedField("return_routed_experts"));
        }

        let image_data = instances
            .iter_mut()
            .filter_map(|instance| instance.image_data.take())
            .collect::<Vec<_>>();
        if !image_data.is_empty() {
            parameters.image_data = Some(rmpv::Value::Array(image_data));
        }

        let first = instances.first().ok_or(VertexError::EmptyInstances)?;
        if first.text.as_ref().is_some_and(|text| !text.is_empty()) {
            let texts = instances
                .into_iter()
                .enumerate()
                .map(|(index, instance)| {
                    instance.text.ok_or(VertexError::MissingInstanceInput {
                        index,
                        name: "text",
                    })
                })
                .collect::<Result<_, _>>()?;
            parameters.text = Some(OneOrMany::Many(texts));
        } else if first
            .input_ids
            .as_ref()
            .is_some_and(|input_ids| !input_ids.is_empty())
        {
            let input_ids = instances
                .into_iter()
                .enumerate()
                .map(|(index, instance)| {
                    instance.input_ids.ok_or(VertexError::MissingInstanceInput {
                        index,
                        name: "input_ids",
                    })
                })
                .collect::<Result<_, _>>()?;
            parameters.input_ids = Some(OneOrMany::Many(input_ids));
        } else if first
            .input_embeds
            .as_ref()
            .is_some_and(|embeds| !embeds.is_empty())
        {
            return Err(VertexError::UnsupportedField("input_embeds"));
        } else {
            return Err(VertexError::MissingInput);
        };

        Ok(parameters)
    }
}

/// Wrap a successful unary native result. Errors and SSE are already complete
/// HTTP responses and must remain unwrapped, as in Python's `isinstance(ret,
/// Response)` branch.
async fn vertex_response(response: Response, stream: bool) -> Response {
    if stream || !response.status().is_success() {
        return response;
    }

    let (parts, body) = response.into_parts();
    let bytes = match to_bytes(body, usize::MAX).await {
        Ok(bytes) => bytes,
        Err(error) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": {"message": error.to_string()}})),
            )
                .into_response();
        }
    };
    let predictions = match serde_json::from_slice::<Value>(&bytes) {
        Ok(value) => value,
        Err(error) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": {"message": error.to_string()}})),
            )
                .into_response();
        }
    };

    let mut response = Json(json!({"predictions": predictions})).into_response();
    *response.status_mut() = parts.status;
    response
}

async fn vertex_generate(
    State(state): State<AppState>,
    body: Result<Json<VertexGenerateRequest>, JsonRejection>,
) -> Result<Response, VertexError> {
    let Json(body) = body.map_err(|rejection| VertexError::InvalidJson(rejection.body_text()))?;
    if body.instances.is_empty() {
        return Ok(Json(Value::Array(Vec::new())).into_response());
    }
    let body = GenerateBody::try_from(body)?;
    let stream = body.stream;
    let (requests, _) = body.into_requests()?;
    Ok(vertex_response(generate_batch(&state, requests, stream).await, stream).await)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        message::{ChunkEvent, EgressItem, GenerateRequest, RequestKind},
        tokenizer_manager::{Senders, TmEvent},
    };
    use axum::{
        body::Body,
        http::{Method, Request},
    };
    use std::sync::{Arc, atomic::AtomicU64};
    use tower::ServiceExt as _;

    fn app_at(route: &str) -> (Router, flume::Receiver<TmEvent>) {
        let (tm, tm_rx) = flume::unbounded();
        let state = AppState {
            senders: Senders {
                tm,
                abort: flume::unbounded().0,
                tok: flume::unbounded().0,
                detok: vec![flume::unbounded().0],
            },
            egress_buf: 8,
            server_args: Arc::new(
                crate::runtime::ServerArgs::from_json(r#"{"model_path": "/m"}"#).unwrap(),
            ),
            egress_activity: Arc::new(AtomicU64::new(0)),
            live_rids: Default::default(),
        };
        (routes_at(route).with_state(state), tm_rx)
    }

    async fn post_json(app: Router, path: &str, body: &str) -> Response {
        app.oneshot(
            Request::builder()
                .method(Method::POST)
                .uri(path)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap()
    }

    async fn response_json(response: Response) -> Value {
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    fn request(value: Value) -> VertexGenerateRequest {
        serde_json::from_value(value).unwrap()
    }

    fn converted_result(value: Value) -> Result<(Vec<GenerateRequest>, bool), VertexError> {
        let body = GenerateBody::try_from(request(value))?;
        let stream = body.stream;
        let (requests, _) = body.into_requests()?;
        Ok((requests, stream))
    }

    fn converted(value: Value) -> (Vec<GenerateRequest>, bool) {
        converted_result(value).unwrap()
    }

    #[tokio::test]
    async fn invalid_requests_return_the_vertex_error_shape() {
        let (app, tm_rx) = app_at(DEFAULT_PREDICT_ROUTE);
        let response = post_json(app.clone(), DEFAULT_PREDICT_ROUTE, r#"{"instances":"#).await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert!(
            body.pointer("/error/message")
                .and_then(Value::as_str)
                .is_some_and(|message| !message.is_empty())
        );

        let response = post_json(
            app,
            DEFAULT_PREDICT_ROUTE,
            r#"{"instances":[{"prompt":"The capital of France is"}]}"#,
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            response_json(response).await,
            json!({
                "error": {
                    "message": "Either text, input_ids or input_embeds should be provided."
                }
            })
        );
        assert!(
            tm_rx.try_recv().is_err(),
            "invalid requests must not reach generation"
        );
    }

    #[tokio::test]
    async fn custom_route_replaces_the_default_and_handles_empty_instances() {
        let custom_route = "/custom-vertex-predict";
        let (app, tm_rx) = app_at(custom_route);

        let missing = post_json(app.clone(), DEFAULT_PREDICT_ROUTE, r#"{"instances":[]}"#).await;
        assert_eq!(missing.status(), StatusCode::NOT_FOUND);

        let response = post_json(app, custom_route, r#"{"instances":[]}"#).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response_json(response).await, json!([]));
        assert!(
            tm_rx.try_recv().is_err(),
            "empty instances must not reach generation"
        );
    }

    #[tokio::test]
    async fn batch_generation_traverses_the_handler_and_wraps_every_prediction() {
        let (app, tm_rx) = app_at(DEFAULT_PREDICT_ROUTE);
        let backend = tokio::spawn(async move {
            for (rid, text, output) in [
                ("vertex_0", "The capital of France is", " Paris"),
                ("vertex_1", "The capital of China is", " Beijing"),
            ] {
                let event =
                    tokio::time::timeout(std::time::Duration::from_secs(1), tm_rx.recv_async())
                        .await
                        .expect("every Vertex instance must reach the TM inbox")
                        .unwrap();
                let TmEvent::Ingress(request) = event else {
                    panic!("Vertex generation must enter through the TM inbox");
                };
                let RequestKind::Generate(generate) = &request.kind else {
                    panic!("Vertex generation must submit a generate request");
                };
                assert_eq!(generate.rid, rid);
                assert_eq!(generate.text.as_deref(), Some(text));
                request
                    .sink
                    .try_send(EgressItem::Done(ChunkEvent {
                        text: output.to_string(),
                        completion_tokens: 1,
                        ..Default::default()
                    }))
                    .unwrap();
            }
        });

        let response = post_json(
            app,
            DEFAULT_PREDICT_ROUTE,
            r#"{
                "instances": [
                    {"text": "The capital of France is"},
                    {"text": "The capital of China is"}
                ],
                "parameters": {"rid": "vertex"}
            }"#,
        )
        .await;
        backend.await.unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response_json(response).await;
        assert_eq!(body["predictions"].as_array().unwrap().len(), 2);
        assert_eq!(body.pointer("/predictions/0/text"), Some(&json!(" Paris")));
        assert_eq!(
            body.pointer("/predictions/1/text"),
            Some(&json!(" Beijing"))
        );
    }

    #[test]
    fn text_instances_become_a_batch_and_parameters_are_forwarded() {
        let (requests, stream) = converted(json!({
            "instances": [{"text": "one"}, {"text": "two"}],
            "parameters": {
                "rid": "vertex",
                "stream": true,
                "return_logprob": true,
                "sampling_params": {"temperature": 0.0, "max_new_tokens": 7}
            }
        }));

        assert!(stream);
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].text.as_deref(), Some("one"));
        assert_eq!(requests[1].text.as_deref(), Some("two"));
        assert_eq!(requests[0].rid, "vertex_0");
        assert_eq!(requests[1].rid, "vertex_1");
        assert!(requests.iter().all(|request| request.stream));
        assert!(
            requests
                .iter()
                .all(|request| request.return_logprob == Some(true))
        );
        assert!(
            requests
                .iter()
                .all(|request| request.sampling_params.max_new_tokens == Some(7))
        );
    }

    #[test]
    fn input_ids_instances_remain_a_batch() {
        let (requests, stream) = converted(json!({
            "instances": [{"input_ids": [1, 2]}, {"input_ids": [3, 4]}]
        }));

        assert!(!stream);
        assert_eq!(requests[0].input_ids.as_deref(), Some(&[1, 2][..]));
        assert_eq!(requests[1].input_ids.as_deref(), Some(&[3, 4][..]));
    }

    #[test]
    fn input_selection_uses_non_empty_values_and_python_order() {
        let (requests, _) = converted(json!({
            "instances": [{"text": "", "input_ids": [1, 2]}]
        }));

        assert!(requests[0].text.is_none());
        assert_eq!(requests[0].input_ids.as_deref(), Some(&[1, 2][..]));
    }

    #[test]
    fn missing_selected_field_in_later_instance_is_an_error_not_a_panic() {
        let result = converted_result(json!({
            "instances": [{"text": "one"}, {"input_ids": [1, 2]}]
        }));

        assert!(matches!(
            result,
            Err(VertexError::MissingInstanceInput {
                index: 1,
                name: "text"
            })
        ));
    }

    #[test]
    fn input_embeds_are_rejected_explicitly() {
        let embeds = converted_result(json!({
            "instances": [{"input_embeds": [[0.1, 0.2]]}]
        }))
        .unwrap_err();
        assert!(matches!(
            embeds,
            VertexError::UnsupportedField("input_embeds")
        ));
    }

    #[test]
    fn typed_parameters_reject_invalid_and_unknown_fields() {
        let invalid = serde_json::from_value::<VertexGenerateRequest>(json!({
            "instances": [{"text": "hello"}],
            "parameters": {"stream": "yes"}
        }));
        assert!(invalid.is_err());

        let unknown = serde_json::from_value::<VertexGenerateRequest>(json!({
            "instances": [{"text": "hello"}],
            "parameters": {"not_a_parameter": true}
        }));
        assert!(unknown.is_err());
    }

    #[test]
    fn instance_inputs_are_rejected_inside_parameters() {
        for (name, value) in [
            ("text", json!("other")),
            ("input_ids", json!([1, 2])),
            ("image_data", json!("image.jpg")),
        ] {
            let result = converted_result(json!({
                "instances": [{"text": "hello"}],
                "parameters": {name: value}
            }));

            assert!(matches!(
                result,
                Err(VertexError::InputInParameters(field)) if field == name
            ));
        }
    }

    #[test]
    fn unsupported_native_extension_parameters_are_rejected_explicitly() {
        let lora = converted_result(json!({
            "instances": [{"text": "hello"}],
            "parameters": {"lora_path": "adapter"}
        }))
        .unwrap_err();
        assert!(matches!(lora, VertexError::UnsupportedField("lora_path")));

        let routed = converted_result(json!({
            "instances": [{"text": "hello"}],
            "parameters": {"return_routed_experts": true}
        }))
        .unwrap_err();
        assert!(matches!(
            routed,
            VertexError::UnsupportedField("return_routed_experts")
        ));
    }

    #[test]
    fn empty_or_null_unsupported_fields_remain_noops() {
        let (requests, _) = converted(json!({
            "instances": [{
                "text": "hello",
                "input_embeds": [],
                "image_data": null
            }],
            "parameters": {
                "lora_path": null,
                "return_routed_experts": false
            }
        }));

        assert_eq!(requests[0].text.as_deref(), Some("hello"));
    }

    #[tokio::test]
    async fn streaming_and_error_responses_are_not_wrapped() {
        let stream = Response::new(Body::from("event stream"));
        let stream = vertex_response(stream, true).await;
        assert_eq!(
            to_bytes(stream.into_body(), usize::MAX).await.unwrap(),
            "event stream"
        );

        let error = (StatusCode::BAD_REQUEST, Json(json!({"error": "bad"}))).into_response();
        let error = vertex_response(error, false).await;
        let bytes = to_bytes(error.into_body(), usize::MAX).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&bytes).unwrap(),
            json!({"error": "bad"})
        );
    }
}
