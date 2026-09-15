//! Prometheus exposition from the launch process's private collector service.

use std::sync::Arc;
use std::time::Duration;

use axum::{
    Router,
    extract::State,
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::get,
};

use crate::message::config::ServerArgs;

const MAX_SCRAPE_BYTES: usize = 64 * 1024 * 1024;

pub(crate) fn router(
    args: &ServerArgs,
    source_owner: bool,
    native: Option<Arc<crate::metrics::FrontendMetrics>>,
) -> Result<Router, String> {
    if !args.enable_metrics {
        return Ok(Router::new());
    }
    let native_router = native
        .map(|metrics| {
            Router::new()
                .route("/metrics/native", get(native_scrape))
                .with_state(metrics)
        })
        .unwrap_or_default();
    if !source_owner {
        return Ok(native_router);
    }
    let socket = args
        .metrics_socket
        .as_ref()
        .ok_or("Rust metrics collector was not initialized before scheduler launch")?;
    let client = reqwest::Client::builder()
        .no_proxy()
        .unix_socket(socket.as_str())
        .timeout(Duration::from_secs(6))
        .build()
        .map_err(|error| format!("initializing metrics client: {error}"))?;
    Ok(Router::new()
        .route("/metrics", get(scrape))
        .route("/metrics/", get(scrape))
        .with_state(client)
        .merge(native_router))
}

async fn native_scrape(State(metrics): State<Arc<crate::metrics::FrontendMetrics>>) -> Response {
    match metrics.render() {
        Ok(body) => (
            [(
                header::CONTENT_TYPE,
                "text/plain; version=0.0.4; charset=utf-8",
            )],
            body,
        )
            .into_response(),
        Err(error) => {
            tracing::error!(%error, "native metrics scrape failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Native metrics collection failed",
            )
                .into_response()
        }
    }
}

async fn collect(client: &reqwest::Client, headers: HeaderMap) -> Result<Response, String> {
    let mut request = client.get("http://localhost/metrics");
    if let Some(accept) = headers.get(header::ACCEPT) {
        request = request.header(header::ACCEPT, accept);
    }
    let mut response = request
        .send()
        .await
        .map_err(|error| format!("contacting Python metrics collector: {error}"))?;
    if !response.status().is_success() {
        return Err(format!(
            "Python metrics collector returned {}",
            response.status()
        ));
    }
    let content_type = response
        .headers()
        .get(header::CONTENT_TYPE)
        .ok_or("Python metrics collector omitted Content-Type")?
        .clone();
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|error| format!("reading metrics: {error}"))?
    {
        if body.len().saturating_add(chunk.len()) > MAX_SCRAPE_BYTES {
            return Err("Python metrics scrape exceeded 64 MiB".into());
        }
        body.extend_from_slice(&chunk);
    }
    Ok(([(header::CONTENT_TYPE, content_type)], body).into_response())
}

async fn scrape(State(client): State<reqwest::Client>, headers: HeaderMap) -> Response {
    match collect(&client, headers).await {
        Ok(response) => response,
        Err(error) => {
            tracing::error!(%error, "metrics scrape failed");
            (
                StatusCode::BAD_GATEWAY,
                "Metrics collection failed; see server logs",
            )
                .into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tower::ServiceExt;

    #[tokio::test]
    async fn private_collection_preserves_exposition_and_propagates_failure() {
        let socket = std::env::temp_dir().join(format!("metrics-{}.sock", uuid::Uuid::new_v4()));
        let listener = tokio::net::UnixListener::bind(&socket).unwrap();
        let backend = tokio::spawn(async move {
            for expected_openmetrics in [false, true] {
                let (mut connection, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                while !request.ends_with(b"\r\n\r\n") {
                    request.push(connection.read_u8().await.unwrap());
                    assert!(request.len() < 4096);
                }
                let request = String::from_utf8_lossy(&request);
                assert_eq!(
                    request.contains("application/openmetrics-text"),
                    expected_openmetrics
                );
                let content_type = if expected_openmetrics {
                    "application/openmetrics-text; version=1.0.0; charset=utf-8"
                } else {
                    "text/plain; version=0.0.4; charset=utf-8"
                };
                let body = if expected_openmetrics {
                    "sglang:requests 3\n# EOF\n"
                } else {
                    "sglang:requests 3\n"
                };
                connection.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
            }
        });
        let args = ServerArgs {
            enable_metrics: true,
            metrics_socket: Some(socket.to_string_lossy().into()),
            ..Default::default()
        };
        let app = router(&args, true, None).unwrap();
        for (path, accept) in [
            ("/metrics", "text/plain"),
            ("/metrics/", "application/openmetrics-text; version=1.0.0"),
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri(path)
                        .header(header::ACCEPT, accept)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert!(
                response.headers()[header::CONTENT_TYPE]
                    .to_str()
                    .unwrap()
                    .starts_with(accept.split(';').next().unwrap())
            );
            let bytes = axum::body::to_bytes(response.into_body(), 1024)
                .await
                .unwrap();
            assert!(bytes.starts_with(b"sglang:requests 3\n"));
            assert_eq!(bytes.ends_with(b"# EOF\n"), path.ends_with('/'));
        }
        backend.await.unwrap();
        std::fs::remove_file(&socket).unwrap();
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    }

    #[tokio::test]
    async fn only_the_engine_source_owner_exposes_python_metrics() {
        let mut args = ServerArgs {
            enable_metrics: true,
            ..Default::default()
        };
        assert!(
            router(&args, true, None).is_err(),
            "enabled collection requires the private exporter"
        );
        for owner in [false, true] {
            if owner {
                args.enable_metrics = false;
            }
            let response = router(&args, owner, None)
                .unwrap()
                .oneshot(
                    Request::builder()
                        .uri("/metrics")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::NOT_FOUND);
        }
    }
}
