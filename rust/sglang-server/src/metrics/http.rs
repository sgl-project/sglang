//! Public HTTP accounting lives until the response body completes or is dropped.

use std::collections::{BTreeMap, HashMap};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Instant;

use axum::Router;
use axum::body::{Body, HttpBody};
use axum::extract::{MatchedPath, Request, State};
use axum::http::StatusCode;
use axum::middleware::{Next, from_fn_with_state};
use axum::response::Response;
use prometheus::{IntCounterVec, IntGauge, IntGaugeVec, Opts, Registry};

use super::FrontendMetrics;

#[derive(Debug)]
pub(crate) struct HttpMetrics {
    requests: IntCounterVec,
    responses: IntCounterVec,
    active: IntGaugeVec,
    routing_active: IntGauge,
    routing_keys: Mutex<HashMap<String, usize>>,
}

impl HttpMetrics {
    pub fn new(registry: &Registry, labels: &BTreeMap<String, String>) -> Result<Self, String> {
        let opts = |name: &str, help: &str| {
            Opts::new(format!("sglang:{name}"), help)
                .const_labels(labels.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        };
        let requests = IntCounterVec::new(
            opts(
                "http_requests_total",
                "Total number of HTTP requests by endpoint and method",
            ),
            &["endpoint", "method"],
        )
        .map_err(|error| error.to_string())?;
        let responses = IntCounterVec::new(
            opts(
                "http_responses_total",
                "Total number of HTTP responses by endpoint and status code",
            ),
            &["endpoint", "status_code", "method"],
        )
        .map_err(|error| error.to_string())?;
        let active = IntGaugeVec::new(
            opts(
                "http_requests_active",
                "Number of currently active HTTP requests",
            ),
            &["endpoint", "method"],
        )
        .map_err(|error| error.to_string())?;
        let routing_active = IntGauge::new(
            "sglang:routing_keys_active",
            "Number of unique routing keys with active requests",
        )
        .map_err(|error| error.to_string())?;
        for collector in [
            Box::new(requests.clone()) as Box<dyn prometheus::core::Collector>,
            Box::new(responses.clone()),
            Box::new(active.clone()),
            Box::new(routing_active.clone()),
        ] {
            registry
                .register(collector)
                .map_err(|error| error.to_string())?;
        }
        Ok(Self {
            requests,
            responses,
            active,
            routing_active,
            routing_keys: Mutex::new(HashMap::new()),
        })
    }

    fn start(self: &Arc<Self>, request: &Request) -> HttpGuard {
        let endpoint = request
            .extensions()
            .get::<MatchedPath>()
            .map(|path| path.as_str().to_owned())
            .unwrap_or_else(|| public_endpoint(request.uri().path()).to_owned());
        let method = match request.method().as_str() {
            method @ ("GET" | "POST" | "PUT" | "DELETE" | "HEAD" | "OPTIONS" | "PATCH"
            | "CONNECT" | "TRACE") => method,
            _ => "OTHER",
        }
        .to_owned();
        self.requests.with_label_values(&[&endpoint, &method]).inc();
        let active = self.active.with_label_values(&[&endpoint, &method]);
        active.inc();
        let routing_key = request
            .headers()
            .get("x-smg-routing-key")
            .and_then(|value| value.to_str().ok())
            .filter(|value| !value.is_empty())
            .map(str::to_owned);
        if let Some(key) = &routing_key {
            let mut keys = self
                .routing_keys
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let count = keys.entry(key.clone()).or_default();
            if *count == 0 {
                self.routing_active.inc();
            }
            *count += 1;
        }
        HttpGuard {
            metrics: self.clone(),
            active,
            endpoint,
            method,
            routing_key,
            status: None,
        }
    }
}

// DP ingress has a fallback router, so it cannot use MatchedPath. Unknown URLs
// share a bounded label instead of exposing caller-controlled metric series.
fn public_endpoint(path: &str) -> &str {
    match path {
        "/generate"
        | "/v1/completions"
        | "/v1/chat/completions"
        | "/v1/models"
        | "/tokenize"
        | "/detokenize"
        | "/v1/tokenize"
        | "/v1/detokenize"
        | "/health"
        | "/health_generate"
        | "/health_receive"
        | "/get_model_info"
        | "/model_info"
        | "/get_server_info"
        | "/server_info"
        | "/get_pd_match_key"
        | "/flush_cache"
        | "/hicache/storage-backend/clear"
        | "/clear_hicache_storage_backend"
        | "/abort_request"
        | "/get_load"
        | "/v1/loads"
        | "/metrics"
        | "/metrics/"
        | "/register"
        | "/query" => path,
        _ => "unmatched",
    }
}

struct HttpGuard {
    metrics: Arc<HttpMetrics>,
    active: IntGauge,
    endpoint: String,
    method: String,
    routing_key: Option<String>,
    status: Option<StatusCode>,
}

impl Drop for HttpGuard {
    fn drop(&mut self) {
        self.active.dec();
        if let Some(key) = &self.routing_key {
            let mut keys = self
                .metrics
                .routing_keys
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(count) = keys.get_mut(key) {
                *count -= 1;
                if *count == 0 {
                    keys.remove(key);
                    self.metrics.routing_active.dec();
                }
            }
        }
        if let Some(status) = self.status {
            self.metrics
                .responses
                .with_label_values(&[&self.endpoint, status.as_str(), &self.method])
                .inc();
        }
    }
}

struct ObservedBody {
    inner: Body,
    guard: Option<HttpGuard>,
}

impl HttpBody for ObservedBody {
    type Data = <Body as HttpBody>::Data;
    type Error = <Body as HttpBody>::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
        let result = Pin::new(&mut self.inner).poll_frame(cx);
        if matches!(result, Poll::Ready(None) | Poll::Ready(Some(Err(_))))
            || self.inner.is_end_stream()
        {
            self.guard.take();
        }
        result
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

pub(crate) fn apply(router: Router, metrics: Arc<FrontendMetrics>) -> Router {
    router.layer(from_fn_with_state(metrics, track))
}

async fn track(
    State(metrics): State<Arc<FrontendMetrics>>,
    request: Request,
    next: Next,
) -> Response {
    if request.uri().path() == "/metrics/native" {
        return next.run(request).await;
    }
    let started = Instant::now();
    let is_load = request.uri().path() == "/v1/loads";
    let mut guard = metrics.http.start(&request);
    let response = next.run(request).await;
    if is_load {
        metrics.load_duration(started.elapsed().as_secs_f64());
    }
    guard.status = Some(response.status());
    let (parts, body) = response.into_parts();
    Response::from_parts(
        parts,
        Body::new(ObservedBody {
            inner: body,
            guard: Some(guard),
        }),
    )
}
