//! Prometheus endpoint and HTTP request metrics middleware for the Rust server.

use std::sync::Arc;

use axum::{
    Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
};

use super::AppState;
use crate::metrics::{HttpEndpointLabel, HttpMethodLabel, MetricsState};

const PROMETHEUS_TEXT_CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

pub(super) fn routes() -> Router<AppState> {
    Router::new().route("/metrics", get(metrics))
}

pub(super) fn apply_http_metrics(app: Router, metrics: Arc<MetricsState>) -> Router {
    app.layer(axum::middleware::from_fn_with_state(
        metrics,
        track_http_metrics,
    ))
}

async fn metrics(State(state): State<AppState>) -> Response {
    (
        StatusCode::OK,
        [("content-type", PROMETHEUS_TEXT_CONTENT_TYPE)],
        state.metrics.render_prometheus(),
    )
        .into_response()
}

async fn track_http_metrics(
    State(metrics): State<Arc<MetricsState>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let Some(endpoint) = HttpEndpointLabel::from_path(req.uri().path()) else {
        return next.run(req).await;
    };
    let method = HttpMethodLabel::from_method(req.method());
    metrics.http_request_started(endpoint, method);
    let res = next.run(req).await;
    metrics.http_request_finished(endpoint, method, res.status().as_u16());
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{RuntimeConfig, RustServerServerArgs, ServerArgs};
    use std::io::{Read, Write};

    const TEST_SERVER_ARGS: &str = r#"{
        "skip_tokenizer_init": true,
        "served_model_name": "test-model",
        "model_config": {"context_len": 2048, "vocab_size": 1000}
    }"#;
    const TEST_SERVER_ARGS_WITH_METRICS: &str = r#"{
        "skip_tokenizer_init": true,
        "served_model_name": "test-model",
        "enable_metrics": true,
        "model_config": {"context_len": 2048, "vocab_size": 1000}
    }"#;

    fn get(path: &str, addr: std::net::SocketAddr) -> String {
        let mut conn = std::net::TcpStream::connect(addr).expect("connect");
        let req = format!("GET {path} HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n");
        conn.write_all(req.as_bytes()).unwrap();
        conn.flush().unwrap();
        let mut response = String::new();
        conn.read_to_string(&mut response).unwrap();
        response
    }

    fn status_code(response: &str) -> u16 {
        response
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|code| code.parse().ok())
            .unwrap_or(0)
    }

    #[test]
    fn metrics_route_is_gated_by_enable_metrics() {
        let probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = probe.local_addr().unwrap();
        drop(probe);

        let server_args = ServerArgs::from_json(TEST_SERVER_ARGS).unwrap();
        let cfg = RuntimeConfig {
            rust_server_args: RustServerServerArgs {
                http_addr: addr,
                api_worker_num: 1,
                ..Default::default()
            },
            server_args: Arc::new(server_args),
        };
        let rt = crate::runtime::start(cfg).expect("start runtime");
        assert_eq!(status_code(&get("/metrics", addr)), 404);
        rt.request_shutdown();

        let probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = probe.local_addr().unwrap();
        drop(probe);
        let server_args = ServerArgs::from_json(TEST_SERVER_ARGS_WITH_METRICS).unwrap();
        let cfg = RuntimeConfig {
            rust_server_args: RustServerServerArgs {
                http_addr: addr,
                api_worker_num: 1,
                ..Default::default()
            },
            server_args: Arc::new(server_args),
        };
        let rt = crate::runtime::start(cfg).expect("start runtime");
        let response = get("/metrics", addr);
        assert_eq!(status_code(&response), 200);
        assert!(response.contains("content-type: text/plain; version=0.0.4; charset=utf-8"));
        assert!(response.contains("sglang:rust_server_ring_capacity{ring=\"ingress\"}"));
        rt.request_shutdown();
    }
}
