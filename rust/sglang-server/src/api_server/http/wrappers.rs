//! Python's generation wrappers, served by rewriting the request and sending
//! it back through the whole router: `/vertex_generate`, Vertex AI's
//! `instances` envelope around `/generate`, and `/invocations`, SageMaker's
//! one inference route, dispatched to an OpenAI route by body key. The
//! rewritten request takes the same path a direct call would (DP relay,
//! decompression, the route's own parsing), so the wrappers add only the
//! translation Python's handlers do.

use axum::{
    Router,
    body::{Body, to_bytes},
    extract::Request,
    http::{StatusCode, Uri, header},
    response::{IntoResponse, Response},
    routing::post,
};
use serde_json::{Map, Value};
use tower::ServiceExt;

/// Python's `_SAGEMAKER_INVOCATION_ROUTES`: checked in order, so a body with
/// `messages` always reaches chat.
const SAGEMAKER_ROUTES: [(&str, &str); 3] = [
    ("messages", "/v1/chat/completions"),
    ("prompt", "/v1/completions"),
    ("input", "/v1/embeddings"),
];

const SAGEMAKER_NO_MATCH: &str = "Cannot determine the request type: expected one of 'messages' \
     (chat completions), 'prompt' (completions) or 'input' (embeddings)";

/// Mount the wrappers over `app`. `/vertex_generate` takes its path from
/// `AIP_PREDICT_ROUTE`, as Python's does.
pub(super) fn apply(app: Router) -> Router {
    let vertex_path = std::env::var("AIP_PREDICT_ROUTE")
        .ok()
        .filter(|path| path.starts_with('/'))
        .unwrap_or_else(|| "/vertex_generate".into());
    let inner = app.clone();
    let vertex = post(move |request: Request| vertex_generate(inner.clone(), request));
    let inner = app.clone();
    let invocations = post(move |request: Request| invocations(inner.clone(), request));
    app.route(&vertex_path, vertex)
        .route("/invocations", invocations)
}

/// Python's `ErrorResponse` for a body its request model refuses: a 400.
fn validation_error(message: impl Into<String>) -> Response {
    (
        StatusCode::BAD_REQUEST,
        axum::Json(serde_json::json!({
            "object": "error",
            "message": message.into(),
            "type": "BadRequestError",
            "param": null,
            "code": 400,
        })),
    )
        .into_response()
}

/// The request's JSON object, or Python's validation 400.
async fn json_object(
    request: Request,
) -> Result<(http::request::Parts, Map<String, Value>), Response> {
    let (parts, body) = request.into_parts();
    let bytes = to_bytes(body, usize::MAX)
        .await
        .map_err(|error| validation_error(format!("body: {error}")))?;
    match serde_json::from_slice::<Value>(&bytes) {
        Ok(Value::Object(object)) => Ok((parts, object)),
        Ok(_) => Err(validation_error("body: Input should be a valid dictionary")),
        Err(error) => Err(validation_error(format!("body: {error}"))),
    }
}

/// Send `body` to `path` through the router, with the original request's
/// headers and extensions.
async fn redispatch(
    inner: Router,
    mut parts: http::request::Parts,
    path: &str,
    body: Value,
) -> Response {
    let body = body.to_string();
    parts.uri = match path.parse::<Uri>() {
        Ok(uri) => uri,
        Err(error) => return validation_error(error.to_string()),
    };
    parts.headers.remove(header::CONTENT_LENGTH);
    parts.headers.remove(header::CONTENT_ENCODING);
    parts.headers.insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("application/json"),
    );
    let request = Request::from_parts(parts, Body::from(body));
    match inner.oneshot(request).await {
        Ok(response) => response,
        Err(never) => match never {},
    }
}

/// Python's truthiness, which picks the prompt key of `instances[0]`.
fn truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|n| n != 0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

/// `POST /vertex_generate` — Python's handler: the prompt key is the first
/// of `text`, `input_ids`, `input_embeds` that `instances[0]` sets, taken
/// from every instance as a list; `image_data` is every instance's that has
/// one; `parameters` are the `/generate` fields. A unary answer is wrapped
/// as `{"predictions": ...}`; a stream passes through; no instances is `[]`.
async fn vertex_generate(inner: Router, request: Request) -> Response {
    let (parts, body) = match json_object(request).await {
        Ok(parsed) => parsed,
        Err(response) => return response,
    };
    let instances: Vec<&Map<String, Value>> = match body.get("instances") {
        Some(Value::Array(items)) => {
            let mut instances = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    Value::Object(instance) => instances.push(instance),
                    _ => return validation_error("instances: Input should be a valid dictionary"),
                }
            }
            instances
        }
        _ => return validation_error("instances: Field required"),
    };
    if instances.is_empty() {
        return axum::Json(Value::Array(Vec::new())).into_response();
    }
    let mut generate = Map::new();
    for key in ["text", "input_ids", "input_embeds"] {
        if truthy(instances[0].get(key)) {
            generate.insert(
                key.into(),
                Value::Array(
                    instances
                        .iter()
                        .map(|instance| instance.get(key).cloned().unwrap_or(Value::Null))
                        .collect(),
                ),
            );
            break;
        }
    }
    let image_data: Vec<Value> = instances
        .iter()
        .filter_map(|instance| instance.get("image_data"))
        .filter(|value| !value.is_null())
        .cloned()
        .collect();
    if !image_data.is_empty() {
        generate.insert("image_data".into(), Value::Array(image_data));
    }
    if let Some(Value::Object(parameters)) = body.get("parameters") {
        for (key, value) in parameters {
            generate.insert(key.clone(), value.clone());
        }
    }
    let response = redispatch(inner, parts, "/generate", Value::Object(generate)).await;
    let is_json = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("application/json"));
    if response.status() != StatusCode::OK || !is_json {
        return response;
    }
    let (parts, body) = response.into_parts();
    let bytes = match to_bytes(body, usize::MAX).await {
        Ok(bytes) => bytes,
        Err(error) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()).into_response();
        }
    };
    match serde_json::from_slice::<Value>(&bytes) {
        Ok(predictions) => {
            axum::Json(serde_json::json!({"predictions": predictions})).into_response()
        }
        // Not JSON after all: hand it back as it came.
        Err(_) => Response::from_parts(parts, Body::from(bytes)),
    }
}

/// `POST /invocations` — SageMaker's one inference route: the body goes to
/// chat completions, completions or embeddings by which key it carries.
async fn invocations(inner: Router, request: Request) -> Response {
    let (parts, body) = match json_object(request).await {
        Ok(parsed) => parsed,
        Err(response) => return response,
    };
    let Some((_, path)) = SAGEMAKER_ROUTES
        .iter()
        .find(|(key, _)| body.contains_key(*key))
    else {
        return validation_error(SAGEMAKER_NO_MATCH);
    };
    redispatch(inner, parts, path, Value::Object(body)).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::routing::get;

    /// A router whose generation routes echo the path they were called on
    /// and the body they received.
    fn echo_router() -> Router {
        let echo = |path: &'static str| {
            post(move |request: Request| async move {
                let body = to_bytes(request.into_body(), usize::MAX).await.unwrap();
                let body: Value = serde_json::from_slice(&body).unwrap();
                axum::Json(serde_json::json!({"path": path, "body": body}))
            })
        };
        let router = Router::new()
            .route("/generate", echo("/generate"))
            .route("/v1/chat/completions", echo("/v1/chat/completions"))
            .route("/v1/completions", echo("/v1/completions"))
            .route(
                "/stream",
                get(|| async {
                    (
                        [(header::CONTENT_TYPE, "text/event-stream")],
                        "data: [DONE]\n\n",
                    )
                }),
            );
        apply(router)
    }

    async fn post_json(app: &Router, path: &str, body: &str) -> (StatusCode, Value) {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(path)
                    .header(header::CONTENT_TYPE, "application/json")
                    .header(header::CONTENT_LENGTH, body.len())
                    .body(Body::from(body.to_owned()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    /// Vertex picks the prompt key from the first instance by truthiness,
    /// lists it over every instance, compacts `image_data`, splats
    /// `parameters`, wraps the answer as `predictions`, and answers no
    /// instances with `[]`.
    #[tokio::test]
    async fn vertex_generate_translates_the_instances_envelope() {
        let app = echo_router();
        let (status, body) = post_json(
            &app,
            "/vertex_generate",
            r#"{"instances": [{"text": "a", "image_data": "img"}, {"text": "b"}],
                "parameters": {"sampling_params": {"max_new_tokens": 2}, "stream": false}}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body,
            serde_json::json!({"predictions": {"path": "/generate", "body": {
                "text": ["a", "b"], "image_data": ["img"],
                "sampling_params": {"max_new_tokens": 2}, "stream": false
            }}})
        );

        // An empty `text` is falsy: the key falls through to `input_ids`.
        let (_, body) = post_json(
            &app,
            "/vertex_generate",
            r#"{"instances": [{"text": "", "input_ids": [1]}, {"input_ids": [2]}]}"#,
        )
        .await;
        assert_eq!(
            body["predictions"]["body"],
            serde_json::json!({"input_ids": [[1], [2]]})
        );

        let (status, body) = post_json(&app, "/vertex_generate", r#"{"instances": []}"#).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, serde_json::json!([]));

        for bad in [r#"{"parameters": {}}"#, r#"{"instances": [1]}"#, "[1]"] {
            let (status, body) = post_json(&app, "/vertex_generate", bad).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{bad}");
            assert_eq!(body["object"], "error", "{bad}");
        }
    }

    /// SageMaker dispatch goes by key in Python's order, passes the body
    /// through unchanged, and refuses a body naming none of the keys.
    #[tokio::test]
    async fn invocations_dispatch_by_body_key() {
        let app = echo_router();
        for (body, path) in [
            (r#"{"messages": [], "prompt": "x"}"#, "/v1/chat/completions"),
            (r#"{"prompt": "x", "input": "y"}"#, "/v1/completions"),
        ] {
            let (status, got) = post_json(&app, "/invocations", body).await;
            assert_eq!(status, StatusCode::OK, "{body}");
            assert_eq!(got["path"], path, "{body}");
            assert_eq!(got["body"], serde_json::from_str::<Value>(body).unwrap());
        }
        let (status, body) = post_json(&app, "/invocations", r#"{"model": "m"}"#).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["message"], SAGEMAKER_NO_MATCH);
        // `input` reaches `/v1/embeddings`, served or not.
        let (status, _) = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/invocations")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(r#"{"input": "x"}"#))
                    .unwrap(),
            )
            .await
            .map(|response| (response.status(), ()))
            .unwrap();
        assert_eq!(status, StatusCode::NOT_FOUND);
    }
}
