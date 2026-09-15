//! Public HTTP ingress for scheduler-embedded DP frontends.

mod proxy;
mod routing;
#[cfg(test)]
mod tests;

use std::net::{SocketAddr, TcpListener};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use axum::body::{Body, to_bytes};
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::{Json, Router};
use pyo3::prelude::*;
use serde_json::{Value, json};
use tokio::sync::oneshot;

use super::transport::{Http2Settings, serve_listener};
use routing::{LoadBalanceMethod, RoutingState};

struct DpState {
    workers: Vec<String>,
    client: reqwest::Client,
    routing: Mutex<RoutingState>,
    header_overrides: bool,
    preferred_sampling: Option<Value>,
    metrics: Option<Arc<crate::metrics::FrontendMetrics>>,
}

struct IngressThread {
    stop: oneshot::Sender<()>,
    thread: JoinHandle<()>,
}

/// Owns the public listener; workers keep ephemeral, independently bound ports.
#[pyclass(frozen)]
pub struct DpIngress {
    address: SocketAddr,
    running: Mutex<Option<IngressThread>>,
}

#[pymethods]
impl DpIngress {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (*, host, port, workers, load_balance_method, enable_request_header_overrides, enable_request_decompression, enable_http2, http2_max_concurrent_streams, http2_initial_connection_window_size, preferred_sampling_params=None, metrics_config=None))]
    fn new(
        py: Python<'_>,
        host: String,
        port: u16,
        workers: Vec<(usize, String)>,
        load_balance_method: &str,
        enable_request_header_overrides: bool,
        enable_request_decompression: bool,
        enable_http2: bool,
        http2_max_concurrent_streams: u32,
        http2_initial_connection_window_size: u32,
        preferred_sampling_params: Option<&str>,
        metrics_config: Option<String>,
    ) -> PyResult<Self> {
        let method = load_balance_method
            .parse()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let preferred_sampling = preferred_sampling_params
            .map(serde_json::from_str)
            .transpose()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        if http2_max_concurrent_streams == 0
            || !(1024..=(1 << 31) - 1).contains(&http2_initial_connection_window_size)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "invalid HTTP/2 settings",
            ));
        }
        py.detach(|| {
            let metrics = metrics_config
                .map(|config| {
                    crate::metrics::FrontendMetrics::new(&crate::ServerArgs {
                        enable_metrics: true,
                        metrics_config: Some(config),
                        ..Default::default()
                    })
                })
                .transpose()
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            let listener = TcpListener::bind((host.as_str(), port))
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            Self::start(
                listener,
                workers,
                method,
                enable_request_header_overrides,
                enable_request_decompression,
                preferred_sampling,
                enable_http2.then_some(Http2Settings {
                    max_concurrent_streams: http2_max_concurrent_streams,
                    initial_connection_window_size: http2_initial_connection_window_size,
                }),
                metrics,
            )
            .map_err(pyo3::exceptions::PyValueError::new_err)
        })
    }

    #[getter]
    fn http_port(&self) -> u16 {
        self.address.port()
    }

    fn close(&self, py: Python<'_>) {
        py.detach(|| self.stop());
    }
}

impl DpIngress {
    #[allow(clippy::too_many_arguments)]
    fn start(
        listener: TcpListener,
        workers: Vec<(usize, String)>,
        method: LoadBalanceMethod,
        header_overrides: bool,
        enable_request_decompression: bool,
        preferred_sampling: Option<Value>,
        http2: Option<Http2Settings>,
        metrics: Option<Arc<crate::metrics::FrontendMetrics>>,
    ) -> Result<Self, String> {
        let address = listener.local_addr().map_err(|e| e.to_string())?;
        listener.set_nonblocking(true).map_err(|e| e.to_string())?;
        let workers = routing::validate_workers(workers, address)?;
        let metric_routes = super::metrics::router(
            &crate::ServerArgs {
                enable_metrics: metrics.is_some(),
                ..Default::default()
            },
            false,
            metrics.clone(),
        )?;
        let client = reqwest::Client::builder()
            .no_proxy()
            .no_gzip()
            .no_brotli()
            .no_deflate()
            .no_zstd()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(3))
            .build()
            .map_err(|e| e.to_string())?;
        let state = Arc::new(DpState {
            routing: Mutex::new(RoutingState::new(method, workers.len())),
            workers,
            client,
            header_overrides,
            preferred_sampling,
            metrics: metrics.clone(),
        });
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("rust-dp-http")
            .enable_all()
            .build()
            .map_err(|e| e.to_string())?;
        let (stop, stopped) = oneshot::channel();
        let thread = std::thread::Builder::new()
            .name("rust-dp-ingress".into())
            .spawn(move || {
                runtime.block_on(async {
                    let app = Router::new()
                        .fallback(dispatch)
                        .with_state(state.clone())
                        .merge(metric_routes);
                    let app = super::decompression::apply(app, enable_request_decompression);
                    let app = match metrics {
                        Some(metrics) => crate::metrics::apply_http(app, metrics),
                        None => app,
                    };
                    tokio::select! {
                        result = serve_listener(listener, app, http2) => {
                            if let Err(error) = result {
                                tracing::error!(%error, "DP HTTP listener failed");
                            }
                        }
                        _ = routing::refresh_loads(state) => {}
                        _ = stopped => {}
                    }
                });
                runtime.shutdown_timeout(Duration::from_secs(5));
            })
            .map_err(|e| e.to_string())?;
        Ok(Self {
            address,
            running: Mutex::new(Some(IngressThread { stop, thread })),
        })
    }

    fn stop(&self) {
        let running = self
            .running
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        if let Some(running) = running {
            let _ = running.stop.send(());
            if running.thread.join().is_err() {
                tracing::error!("DP HTTP thread panicked");
            }
        }
    }
}

impl Drop for DpIngress {
    fn drop(&mut self) {
        self.stop();
    }
}

fn error(status: StatusCode, message: impl ToString) -> Response {
    (status, Json(json!({ "detail": message.to_string() }))).into_response()
}

async fn dispatch(State(state): State<Arc<DpState>>, request: Request<Body>) -> Response {
    let (parts, body) = request.into_parts();
    // Matches the worker listener's disabled DefaultBodyLimit; batch expansion
    // and media byte budgets are validated by the shared native intake.
    let body = match to_bytes(body, usize::MAX).await {
        Ok(body) => body,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    match parts.uri.path() {
        "/generate" if matches!(parts.method.as_str(), "POST" | "PUT") => {
            proxy::generate(&state, parts, body).await
        }
        "/v1/loads" | "/get_load" => proxy::loads(&state, parts).await,
        "/health" | "/health_generate" | "/abort_request" => {
            proxy::all_workers(&state, parts, body).await
        }
        "/v1/completions" | "/v1/chat/completions" => {
            let value: Value = match serde_json::from_slice(&body) {
                Ok(value) => value,
                Err(e) => return error(StatusCode::BAD_REQUEST, e),
            };
            let rank = match routing::rank_hint(&value, &parts.headers, state.header_overrides) {
                Ok(rank) => rank,
                Err(e) => return error(StatusCode::BAD_REQUEST, e),
            };
            let rank = match routing::choose(&state, rank, None, 0) {
                Ok(rank) => rank,
                Err(e) => return error(e.status, e.message),
            };
            proxy::one_worker(&state, rank, parts, body).await
        }
        "/disagg_input_metadata" => proxy::input_metadata(&state, parts).await,
        // Global controls enter the DP controller from one origin only. Static
        // model APIs and the engine's local multiprocess metrics share rank 0.
        _ => proxy::one_worker(&state, 0, parts, body).await,
    }
}
