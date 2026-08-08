//! Embedded PD load balancer: the decode rust-server is the PD front door. An unrouted
//! request gets mini_lb-parity bootstrap params injected, its prefill copy forwarded to a
//! worker registered via `/prefill_workers`, and is served locally (routed requests
//! bypass); a failed forward aborts its rids and downs the worker until `/health` passes.

use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwap;
use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    http::{Request, StatusCode, header},
    response::{IntoResponse, Response},
    routing::get,
};
use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper_util::client::legacy::{Client, connect::HttpConnector};
use hyper_util::rt::TokioExecutor;
use rand::Rng;
use serde::Deserialize;

use crate::api_server::AppState;
use crate::environ;
use crate::ids::Rid;
use crate::message::DetokMsg;
use crate::runtime::ServerArgs;
use crate::tokenizer_manager::Senders;
use crate::utils::response::json_error;

/// Bytes of a non-2xx prefill response body kept for the failure message.
const ERROR_SNIPPET_CAP: usize = 2048;

/// How often the auto-recovery sweeper re-probes down workers, and each
/// probe's own deadline.
const RECOVERY_PROBE_INTERVAL: Duration = Duration::from_secs(5);
const RECOVERY_PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// Prefill endpoint, parse from String.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(try_from = "String")]
pub(crate) struct PrefillEndpoint {
    /// Scheme + authority, no trailing slash.
    base_url: String,
    bootstrap_host: String,
    bootstrap_port: i64,
}

impl TryFrom<String> for PrefillEndpoint {
    type Error = String;
    fn try_from(entry: String) -> Result<Self, String> {
        Self::parse(&entry)
    }
}

impl PrefillEndpoint {
    /// Parse one registration entry, no additional bootstrap port.
    fn parse(entry: &str) -> Result<Self, String> {
        let uri: axum::http::Uri = entry
            .trim()
            .parse()
            .map_err(|e| format!("invalid prefill url '{entry}': {e}"))?;
        if uri.scheme_str() != Some("http") {
            return Err(format!(
                "prefill url '{entry}' must be plain http:// (the embedded PD \
                 load balancer has no TLS support)"
            ));
        }
        if uri.path() != "/" || uri.query().is_some() {
            return Err(format!(
                "prefill url '{entry}' must not carry a path or query"
            ));
        }
        let host = uri
            .host()
            .ok_or_else(|| format!("prefill url '{entry}' has no host"))?;
        let Some(port) = uri.port_u16() else {
            return Err(format!(
                "prefill url '{entry}' must include an explicit port"
            ));
        };
        // ipv6 hosts keep their brackets.
        let bootstrap_host = if host.contains(':') && !host.starts_with('[') {
            format!("[{host}]")
        } else {
            host.to_string()
        };
        Ok(Self {
            base_url: format!("http://{bootstrap_host}:{port}"),
            bootstrap_port: i64::from(port),
            bootstrap_host,
        })
    }
}

/// A registered prefill worker plus its live health mark.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Registration {
    endpoint: PrefillEndpoint,
    healthy: bool,
}

/// The prefill workers this decode api server forwards to, plus the shared
/// pooled HTTP client.
pub(crate) struct PrefillWorkerPool {
    /// Copy-on-write prefill endpoint list.
    endpoints: ArcSwap<Vec<Registration>>,
    /// Pooled outbound HTTP client.
    client: Client<HttpConnector, Full<Bytes>>,
    /// Deadline for one whole forward.
    timeout: Duration,
}

/// Constructible only on a decode node with `SGLANG_ENABLE_EMBEDDED_PD_LB=1`
/// (default off — gateway-fronted decode nodes don't mount the front door);
/// starts empty.
impl TryFrom<&ServerArgs> for PrefillWorkerPool {
    // No error message need to return.
    type Error = ();

    fn try_from(args: &ServerArgs) -> Result<Self, Self::Error> {
        if args.disaggregation_mode != "decode"
            || !environ::env_bool("SGLANG_ENABLE_EMBEDDED_PD_LB", false)
        {
            return Err(());
        }
        let timeout =
            Duration::from_secs(environ::env_u64("SGLANG_PD_LB_PREFILL_TIMEOUT_SECS", 1800));
        Ok(Self {
            endpoints: ArcSwap::from_pointee(Vec::new()),
            client: Client::builder(TokioExecutor::new()).build_http(),
            timeout,
        })
    }
}

impl PrefillWorkerPool {
    /// Random policy over the current registration set.
    pub(crate) fn pick(&self) -> Option<PrefillEndpoint> {
        let entries = self.endpoints.load();
        if entries.is_empty() {
            return None;
        }
        let healthy: Vec<&Registration> = entries.iter().filter(|r| r.healthy).collect();
        let pick_from = |slice: &[&Registration]| {
            slice[rand::rng().random_range(0..slice.len())]
                .endpoint
                .clone()
        };
        if healthy.is_empty() {
            Some(pick_from(&entries.iter().collect::<Vec<_>>()))
        } else {
            Some(pick_from(&healthy))
        }
    }

    fn list(&self) -> Arc<Vec<Registration>> {
        self.endpoints.load_full()
    }

    /// Flip a worker's health mark.
    fn set_health(&self, base_url: &str, healthy: bool) -> bool {
        if !self
            .endpoints
            .load()
            .iter()
            .any(|r| r.endpoint.base_url == base_url && r.healthy != healthy)
        {
            return false;
        }
        let mut changed = false;
        self.endpoints.rcu(|current| {
            changed = false;
            current
                .iter()
                .map(|r| {
                    if r.endpoint.base_url == base_url && r.healthy != healthy {
                        changed = true;
                        Registration {
                            healthy,
                            ..r.clone()
                        }
                    } else {
                        r.clone()
                    }
                })
                .collect::<Vec<_>>()
        });
        changed
    }

    /// One recovery pass: probe every down worker and revive the ones
    /// answering `/health` with a 2xx.
    async fn revive_down_workers(&self) {
        let down: Vec<String> = self
            .endpoints
            .load()
            .iter()
            .filter(|r| !r.healthy)
            .map(|r| r.endpoint.base_url.clone())
            .collect();
        for url in down {
            if self.health_check(&url).await && self.set_health(&url, true) {
                tracing::info!(%url, "prefill worker recovered.");
            }
        }
    }

    /// Check the prefill worker is healthy or not.
    async fn health_check(&self, base_url: &str) -> bool {
        let Ok(request) = Request::get(format!("{base_url}/health")).body(Full::new(Bytes::new()))
        else {
            return false;
        };
        matches!(
            tokio::time::timeout(RECOVERY_PROBE_TIMEOUT, self.client.request(request)).await,
            Ok(Ok(response)) if response.status().is_success()
        )
    }

    /// Register the prefill worker.
    pub(crate) fn register(&self, endpoint: PrefillEndpoint) -> bool {
        let mut added = false;
        self.endpoints.rcu(|current| {
            let mut next = (**current).clone();
            let registration = Registration {
                endpoint: endpoint.clone(),
                healthy: true,
            };
            match next
                .iter_mut()
                .find(|existing| existing.endpoint.base_url == endpoint.base_url)
            {
                Some(existing) => {
                    *existing = registration;
                    added = false;
                }
                None => {
                    next.push(registration);
                    added = true;
                }
            }
            next
        });
        added
    }

    /// Deregister the prefill worker.
    pub(crate) fn deregister(&self, base_url: &str) -> bool {
        let mut removed = false;
        self.endpoints.rcu(|current| {
            let next: Vec<_> = current
                .iter()
                .filter(|r| r.endpoint.base_url != base_url)
                .cloned()
                .collect();
            removed = next.len() != current.len();
            next
        });
        removed
    }

    /// Forward a request to the prefill worker.
    async fn forward(&self, url: &str, payload: Bytes) -> Result<(), ForwardFailureReason> {
        let request = Request::post(url)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Full::new(payload))
            .map_err(|e| ForwardFailureReason::worker_alive(e.to_string()))?;
        let response = self
            .client
            .request(request)
            .await
            .map_err(|e| ForwardFailureReason::worker_down(e.to_string()))?;
        let status = response.status();
        let mut body = response.into_body();
        let mut error_snippet = Vec::new();
        while let Some(frame) = body.frame().await {
            let frame = frame.map_err(|e| {
                ForwardFailureReason::worker_down(format!("reading prefill response: {e}"))
            })?;
            if !status.is_success()
                && error_snippet.len() < ERROR_SNIPPET_CAP
                && let Some(data) = frame.data_ref()
            {
                let take = data.len().min(ERROR_SNIPPET_CAP - error_snippet.len());
                error_snippet.extend_from_slice(&data[..take]);
            }
        }
        if status.is_success() {
            Ok(())
        } else {
            Err(ForwardFailureReason::worker_alive(format!(
                "status {status}: {}",
                String::from_utf8_lossy(&error_snippet)
            )))
        }
    }
}

/// Forward to prefill worker failure reason.
/// Display renders the human message; `worker_down` stays out of it — the
/// classification is logged by the mark-down warn, not repeated per line.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
struct ForwardFailureReason {
    message: String,
    worker_down: bool,
}

impl ForwardFailureReason {
    fn worker_down(message: String) -> Self {
        Self {
            message,
            worker_down: true,
        }
    }
    fn worker_alive(message: String) -> Self {
        Self {
            message,
            worker_down: false,
        }
    }
}

/// Async run the the prefill request.
pub(crate) fn spawn_forward(
    pool: Arc<PrefillWorkerPool>,
    endpoint: PrefillEndpoint,
    path: &'static str,
    body: &serde_json::Value,
    rids: Vec<Rid>,
    senders: Senders,
) {
    let payload = Bytes::from(serde_json::to_vec(body).expect("request JSON reserializes"));
    tokio::spawn(async move {
        let url = format!("{}{}", endpoint.base_url, path);
        let failure = match tokio::time::timeout(pool.timeout, pool.forward(&url, payload)).await {
            Ok(Ok(())) => {
                // When succeeded, put the worker back in rotation without waiting for the sweeper.
                if pool.set_health(&endpoint.base_url, true) {
                    tracing::info!(url = %endpoint.base_url,
                        "prefill worker recovered (forward succeeded); back in rotation");
                }
                return;
            }
            Ok(Err(failure)) => failure,
            // Timeout.
            Err(_) => {
                ForwardFailureReason::worker_alive(format!("timed out after {:?}", pool.timeout))
            }
        };
        if failure.worker_down && pool.set_health(&endpoint.base_url, false) {
            tracing::warn!(url = %endpoint.base_url,
                "prefill worker marked down; out of rotation until /health passes");
        }
        tracing::error!(%url, error = %failure,
            "prefill forward failed; aborting local decode request(s)");
        for rid in rids {
            let message = format!("prefill forward to {url} failed: {failure}");
            let _ = senders
                .detok_for(&rid)
                .send(DetokMsg::Fail { rid, message });
        }
    });
}

/// Uniform `[0, 2^63)`, mini_lb's `random.randint(0, 2**63 - 1)`.
fn random_room() -> i64 {
    rand::rng().random_range(0..=i64::MAX)
}

/// Whether the request already carries PD routing.
pub(crate) fn has_bootstrap(value: &serde_json::Value) -> bool {
    ["bootstrap_host", "bootstrap_room"]
        .iter()
        .any(|key| value.get(key).is_some_and(|v| !v.is_null()))
}

/// Get batsh size from the request body.
fn get_batch_size(value: &serde_json::Value) -> Option<usize> {
    match value.get("text") {
        Some(serde_json::Value::Null) | None => {}
        Some(serde_json::Value::Array(items)) => return Some(items.len()),
        Some(_) => return None, // scalar string prompt
    }
    match value.get("input_ids") {
        Some(serde_json::Value::Array(items)) => match items.first() {
            Some(serde_json::Value::Array(_)) => Some(items.len()),
            // A flat id list is a single prompt; an empty/invalid list is left
            // for `into_requests` to reject.
            _ => None,
        },
        _ => None,
    }
}

/// Inject bootstrap params into a raw `/generate` body.
pub(crate) fn inject_bootstrap_params(value: &mut serde_json::Value, endpoint: &PrefillEndpoint) {
    let batch = get_batch_size(value);
    let Some(object) = value.as_object_mut() else {
        return;
    };
    let (host, port, room) = match batch {
        Some(n) => (
            serde_json::json!(vec![endpoint.bootstrap_host.as_str(); n]),
            serde_json::json!(vec![endpoint.bootstrap_port; n]),
            serde_json::json!((0..n).map(|_| random_room()).collect::<Vec<_>>()),
        ),
        None => (
            serde_json::json!(endpoint.bootstrap_host),
            serde_json::json!(endpoint.bootstrap_port),
            serde_json::json!(random_room()),
        ),
    };
    object.insert("bootstrap_host".into(), host);
    object.insert("bootstrap_port".into(), port);
    object.insert("bootstrap_room".into(), room);
}

/// The scalar bootstrap params an OpenAI request resolves to.
#[derive(Clone, Debug, Default, Deserialize)]
pub(crate) struct BootstrapParams {
    pub(crate) bootstrap_host: Option<String>,
    pub(crate) bootstrap_port: Option<i64>,
    pub(crate) bootstrap_room: Option<i64>,
    pub(crate) bootstrap_pair_key: Option<String>,
}

impl BootstrapParams {
    /// Attach to one fan-out choice: scalars broadcast, the room advances by
    /// the choice index.
    pub(crate) fn for_choice(&self, index: usize) -> Self {
        Self {
            bootstrap_room: self
                .bootstrap_room
                .map(|room| room.wrapping_add(index as i64)),
            ..self.clone()
        }
    }
}

/// Resolve PD routing for an OpenAI request body.
pub(crate) fn resolve_openai_bootstrap(
    pool: &Option<Arc<PrefillWorkerPool>>,
    value: &mut serde_json::Value,
) -> (
    BootstrapParams,
    Option<(Arc<PrefillWorkerPool>, PrefillEndpoint)>,
) {
    if has_bootstrap(value) {
        let params = BootstrapParams::deserialize(&*value).unwrap_or_default();
        return (params, None);
    }
    let Some(pool) = pool else {
        return (BootstrapParams::default(), None);
    };
    let Some(endpoint) = pool.pick() else {
        return (BootstrapParams::default(), None);
    };
    let params = BootstrapParams {
        bootstrap_host: Some(endpoint.bootstrap_host.clone()),
        bootstrap_port: Some(endpoint.bootstrap_port),
        bootstrap_room: Some(random_room()),
        bootstrap_pair_key: None,
    };
    if let Some(object) = value.as_object_mut() {
        object.insert(
            "bootstrap_host".into(),
            serde_json::json!(params.bootstrap_host),
        );
        object.insert(
            "bootstrap_port".into(),
            serde_json::json!(params.bootstrap_port),
        );
        object.insert(
            "bootstrap_room".into(),
            serde_json::json!(params.bootstrap_room),
        );
    }
    (params, Some((pool.clone(), endpoint)))
}

/// Auto-recovery loop.
async fn recovery_sweeper(pool: Arc<PrefillWorkerPool>) {
    loop {
        tokio::time::sleep(RECOVERY_PROBE_INTERVAL).await;
        pool.revive_down_workers().await;
    }
}

// ---------------------------------------------------------------------------
// Admin API — runtime management of the prefill list.
// ---------------------------------------------------------------------------

/// `GET/POST/DELETE /prefill_workers`, mounted only when the pool exists
/// (decode nodes).
fn routes() -> Router<AppState> {
    Router::new().route(
        "/prefill_workers",
        get(list_prefill_workers)
            .post(add_prefill_workers)
            .delete(remove_prefill_worker),
    )
}

pub(crate) fn router_and_sweeper(
    pool: Arc<PrefillWorkerPool>,
) -> (Router<AppState>, impl std::future::Future<Output = ()>) {
    (routes(), recovery_sweeper(pool))
}

fn prefill_workers_json(pool: &PrefillWorkerPool) -> serde_json::Value {
    serde_json::json!({
        "prefill_workers": pool
            .list()
            .iter()
            .map(|r| serde_json::json!({
                "url": r.endpoint.base_url,
                "bootstrap_host": r.endpoint.bootstrap_host,
                "bootstrap_port": r.endpoint.bootstrap_port,
                "healthy": r.healthy,
            }))
            .collect::<Vec<_>>()
    })
}

async fn list_prefill_workers(State(state): State<AppState>) -> Response {
    let Some(pool) = state.prefill_worker_pool.as_ref() else {
        return json_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "this server is not a PD decode",
        );
    };
    Json(prefill_workers_json(pool)).into_response()
}

#[derive(Deserialize)]
struct PrefillWorkersBody {
    #[serde(default)]
    url: Option<PrefillEndpoint>,
    #[serde(default)]
    urls: Vec<PrefillEndpoint>,
}

impl PrefillWorkersBody {
    fn into_entries(self) -> impl Iterator<Item = PrefillEndpoint> {
        self.url.into_iter().chain(self.urls)
    }
}

async fn add_prefill_workers(
    State(state): State<AppState>,
    body: Result<Json<PrefillWorkersBody>, JsonRejection>,
) -> Response {
    let Some(pool) = state.prefill_worker_pool.as_ref() else {
        return json_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "this server is not a PD decode",
        );
    };
    let body = match body {
        Ok(Json(body)) => body,
        Err(rejection) => return json_error(StatusCode::BAD_REQUEST, &rejection.body_text()),
    };
    let endpoints: Vec<PrefillEndpoint> = body.into_entries().collect();
    if endpoints.is_empty() {
        return json_error(
            StatusCode::BAD_REQUEST,
            "provide 'url' or 'urls' (entry format: http://host:port)",
        );
    }
    let (mut added, mut updated) = (0, 0);
    for endpoint in endpoints {
        let url = endpoint.base_url.clone();
        if pool.register(endpoint) {
            added += 1;
            tracing::info!(%url, "PD LB prefill endpoint registered");
        } else {
            updated += 1;
            tracing::info!(%url, "PD LB prefill endpoint updated");
        }
    }
    let mut response = prefill_workers_json(pool);
    response["added"] = serde_json::json!(added);
    response["updated"] = serde_json::json!(updated);
    Json(response).into_response()
}

async fn remove_prefill_worker(
    State(state): State<AppState>,
    body: Result<Json<PrefillWorkersBody>, JsonRejection>,
) -> Response {
    let Some(pool) = state.prefill_worker_pool.as_ref() else {
        return json_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "this server is not a PD decode",
        );
    };
    let body = match body {
        Ok(Json(body)) => body,
        Err(rejection) => return json_error(StatusCode::BAD_REQUEST, &rejection.body_text()),
    };
    let Some(endpoint) = body.url else {
        return json_error(StatusCode::BAD_REQUEST, "provide 'url'");
    };
    let base_url = endpoint.base_url;
    if !pool.deregister(&base_url) {
        return json_error(
            StatusCode::NOT_FOUND,
            &format!("prefill url {base_url} is not registered"),
        );
    }
    tracing::info!(url = %base_url, "PD LB prefill endpoint deregistered");
    Json(prefill_workers_json(pool)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn endpoint() -> PrefillEndpoint {
        PrefillEndpoint::parse("http://10.0.0.7:30000").unwrap()
    }

    /// Serializes flips of the process-global LB gate env var: tests run
    /// concurrently, and a gate-off assertion must not race a construction.
    static LB_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn pool_with(urls: &[&str]) -> Arc<PrefillWorkerPool> {
        let args = crate::runtime::ServerArgs::from_json(
            &json!({"disaggregation_mode": "decode"}).to_string(),
        )
        .unwrap();
        let pool = {
            let _env = LB_ENV_LOCK.lock().unwrap();
            unsafe { std::env::set_var("SGLANG_ENABLE_EMBEDDED_PD_LB", "1") };
            Arc::new(PrefillWorkerPool::try_from(&args).expect("decode node owns a pool"))
        };
        for url in urls {
            pool.register(PrefillEndpoint::parse(url).unwrap());
        }
        pool
    }

    fn senders() -> Senders {
        Senders {
            tm: flume::unbounded().0,
            abort: flume::unbounded().0,
            tok: flume::unbounded().0,
            detok: vec![],
        }
    }

    /// Entry grammar: plain `http://host:port`, explicit port, no path; the
    /// port doubles as the bootstrap port (rust prefill serves the registry
    /// on its api port); IPv6 hosts keep their brackets (mini_lb
    /// `maybe_wrap_ipv6_address` parity — the scheduler joins host:port to
    /// reach the registry).
    #[test]
    fn prefill_entry_parsing() {
        let ep = PrefillEndpoint::parse("http://10.0.0.7:30000").unwrap();
        assert_eq!(ep.base_url, "http://10.0.0.7:30000");
        assert_eq!(ep.bootstrap_host, "10.0.0.7");
        assert_eq!(ep.bootstrap_port, 30000, "bootstrap port IS the api port");

        let ep = PrefillEndpoint::parse("http://[2001:db8::1]:30000/").unwrap();
        assert_eq!(ep.bootstrap_host, "[2001:db8::1]", "IPv6 keeps brackets");
        assert_eq!(ep.base_url, "http://[2001:db8::1]:30000");

        for bad in [
            "https://10.0.0.7:30000", // no TLS support
            "http://10.0.0.7",        // no explicit port
            "http://10.0.0.7:30000/v1",
            // The old URL,BOOTSTRAP_PORT suffix is gone: with a rust prefill
            // the bootstrap port is the api port, never a separate one.
            "http://10.0.0.7:30000,8998",
            "10.0.0.7:30000",
        ] {
            assert!(PrefillEndpoint::parse(bad).is_err(), "{bad}");
        }
    }

    /// Every decode node owns a pool (the admin API fills it at runtime); an
    /// empty pool picks nothing, and non-decode roles get no pool at all.
    #[test]
    fn pool_gating_and_empty_pick() {
        let empty = pool_with(&[]);
        assert!(empty.pick().is_none(), "empty pool routes nothing");

        let _env = LB_ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("SGLANG_ENABLE_EMBEDDED_PD_LB", "1") };
        let prefill_args = crate::runtime::ServerArgs::from_json(
            &json!({"disaggregation_mode": "prefill"}).to_string(),
        )
        .unwrap();
        assert!(
            PrefillWorkerPool::try_from(&prefill_args).is_err(),
            "non-decode role is refused even with the env gate on"
        );
        let decode_args = crate::runtime::ServerArgs::from_json(
            &json!({"disaggregation_mode": "decode"}).to_string(),
        )
        .unwrap();
        unsafe { std::env::set_var("SGLANG_ENABLE_EMBEDDED_PD_LB", "0") };
        assert!(
            PrefillWorkerPool::try_from(&decode_args).is_err(),
            "decode without SGLANG_ENABLE_EMBEDDED_PD_LB stays LB-less"
        );
        unsafe { std::env::set_var("SGLANG_ENABLE_EMBEDDED_PD_LB", "1") };
    }

    /// Runtime registration: idempotent by URL, deregistration by normalized
    /// base_url, and `pick` reflects the live set.
    #[test]
    fn registration_round_trip() {
        let pool = pool_with(&[]);
        assert!(pool.register(endpoint()), "first registration adds");
        assert_eq!(pool.pick().as_ref(), Some(&endpoint()));

        assert!(
            !pool.register(endpoint()),
            "same URL is an update, not an add"
        );
        assert_eq!(
            *pool.list(),
            vec![Registration {
                endpoint: endpoint(),
                healthy: true
            }]
        );

        // Re-registering a down worker revives it (operator's assertion).
        assert!(pool.set_health("http://10.0.0.7:30000", false));
        assert!(!pool.register(endpoint()));
        assert!(pool.list()[0].healthy, "re-registration revives");

        assert!(pool.deregister("http://10.0.0.7:30000"));
        assert!(!pool.deregister("http://10.0.0.7:30000"), "already gone");
        assert!(pool.pick().is_none());
    }

    /// Health drives rotation: a down worker is never picked while a healthy
    /// one exists; with every worker down the pick fails open (the forward's
    /// failure is the diagnostic and a success revives); `set_health` no-ops
    /// on unknown URLs and already-set states.
    #[test]
    fn pick_prefers_healthy_and_fails_open() {
        let pool = pool_with(&["http://10.0.0.7:30000", "http://10.0.0.8:30000"]);
        assert!(pool.set_health("http://10.0.0.8:30000", false));
        for _ in 0..32 {
            assert_eq!(
                pool.pick().unwrap().base_url,
                "http://10.0.0.7:30000",
                "down worker picked while a healthy one exists"
            );
        }
        assert!(pool.set_health("http://10.0.0.7:30000", false));
        assert!(pool.pick().is_some(), "all-down pool must fail open");
        assert!(
            !pool.set_health("http://10.0.0.7:30000", false),
            "already down"
        );
        assert!(!pool.set_health("http://unknown:1", true), "unknown URL");
    }

    /// One sweeper pass revives exactly the down workers whose `/health`
    /// answers 2xx; unreachable ones stay out of rotation.
    #[tokio::test]
    async fn health_probe_revives_down_worker() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = axum::Router::new().route(
            "/health",
            axum::routing::get(|| async { axum::http::StatusCode::OK }),
        );
        tokio::spawn(async move { axum::serve(listener, app).await });

        let dead = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let dead_addr = dead.local_addr().unwrap();
        drop(dead);

        let alive_url = format!("http://{addr}");
        let dead_url = format!("http://{dead_addr}");
        let pool = pool_with(&[&alive_url, &dead_url]);
        assert!(pool.set_health(&alive_url, false));
        assert!(pool.set_health(&dead_url, false));

        pool.revive_down_workers().await;

        let health: std::collections::HashMap<String, bool> = pool
            .list()
            .iter()
            .map(|r| (r.endpoint.base_url.clone(), r.healthy))
            .collect();
        assert!(health[&alive_url], "responding worker revived");
        assert!(!health[&dead_url], "unreachable worker stays down");
    }

    /// Single-prompt `/generate` gets scalar injection, exactly mini_lb's
    /// shape; the room is in `[0, 2^63)`.
    #[test]
    fn inject_bootstrap_params_scalars_for_single_prompt() {
        for mut body in [
            json!({"text": "hi"}),
            json!({"input_ids": [1, 2, 3]}),
            json!({}),
        ] {
            inject_bootstrap_params(&mut body, &endpoint());
            assert_eq!(body["bootstrap_host"], "10.0.0.7");
            assert_eq!(body["bootstrap_port"], 30000);
            assert!(body["bootstrap_room"].is_i64(), "{body}");
            assert!(body["bootstrap_room"].as_i64().unwrap() >= 0);
        }
    }

    /// Batch `/generate` (list `text` / nested `input_ids`) gets per-item
    /// lists: host/port broadcast, one fresh room per item (mini_lb parity —
    /// no reliance on the scalar-room+i rule across implementations).
    #[test]
    fn inject_bootstrap_params_lists_for_batch() {
        for (mut body, n) in [
            (json!({"text": ["a", "b", "c"]}), 3),
            (json!({"input_ids": [[1], [2]]}), 2),
        ] {
            inject_bootstrap_params(&mut body, &endpoint());
            assert_eq!(body["bootstrap_host"], json!(vec!["10.0.0.7"; n]), "{body}");
            assert_eq!(body["bootstrap_port"], json!(vec![30000; n]));
            let rooms = body["bootstrap_room"].as_array().unwrap();
            assert_eq!(rooms.len(), n);
            assert!(rooms.iter().all(|room| room.is_i64()));
        }
    }

    /// A body already routed (external router / fake-bootstrap warmup) is left
    /// byte-identical.
    #[test]
    fn existing_bootstrap_bypasses_injection() {
        let routed = json!({
            "text": "hi",
            "bootstrap_host": "2.2.2.2",
            "bootstrap_port": null,
            "bootstrap_room": 0,
        });
        assert!(has_bootstrap(&routed));
        // Explicit nulls (a router deferring every field) do NOT count as routed.
        assert!(!has_bootstrap(
            &json!({"bootstrap_host": null, "bootstrap_room": null})
        ));
        assert!(!has_bootstrap(&json!({"text": "hi"})));
    }

    /// OpenAI intake: externally-supplied scalars are picked up verbatim (no
    /// forward), and `for_choice` advances the room by the choice index —
    /// the cross-server pairing contract with a rust prefill peer.
    #[test]
    fn openai_bootstrap_external_intake_and_choice_rooms() {
        let mut value = json!({
            "model": "m",
            "bootstrap_host": "10.0.0.9",
            "bootstrap_port": 31000,
            "bootstrap_room": 500,
        });
        let (params, forward) = resolve_openai_bootstrap(&None, &mut value);
        assert!(forward.is_none());
        assert_eq!(params.bootstrap_host.as_deref(), Some("10.0.0.9"));
        assert_eq!(params.bootstrap_port, Some(31000));
        let second = params.for_choice(2);
        assert_eq!(second.bootstrap_room, Some(502));
        assert_eq!(second.bootstrap_host.as_deref(), Some("10.0.0.9"));
        assert_eq!(params.for_choice(0).bootstrap_room, Some(500));
    }

    /// OpenAI intake with the pool: scalars are injected into the raw JSON
    /// (the forwarded body) and returned for local attachment; an
    /// already-routed body is never re-forwarded; an empty pool falls through.
    #[test]
    fn openai_bootstrap_injects_when_front_door() {
        let pool = Some(pool_with(&["http://10.0.0.7:30000"]));

        let mut value = json!({"model": "m", "messages": []});
        let (params, forward) = resolve_openai_bootstrap(&pool, &mut value);
        let (_, picked) = forward.expect("front door must forward");
        assert_eq!(picked.base_url, "http://10.0.0.7:30000");
        assert_eq!(value["bootstrap_host"], "10.0.0.7");
        assert_eq!(value["bootstrap_port"], 30000);
        assert_eq!(value["bootstrap_room"], json!(params.bootstrap_room));
        assert_eq!(params.bootstrap_host.as_deref(), Some("10.0.0.7"));

        let (_, forward) = resolve_openai_bootstrap(&pool, &mut value);
        assert!(forward.is_none(), "already-routed body is not re-forwarded");

        let empty = Some(pool_with(&[]));
        let mut unrouted = json!({"model": "m"});
        let (params, forward) = resolve_openai_bootstrap(&empty, &mut unrouted);
        assert!(forward.is_none(), "empty pool falls through unrouted");
        assert!(params.bootstrap_room.is_none());
        assert!(!has_bootstrap(&unrouted), "body left untouched");
    }

    /// End-to-end failure path: a prefill answering 500 (and one refusing the
    /// connection) turns into a `DetokMsg::Fail` per rid on the right detok
    /// shard, carrying the upstream diagnostic.
    #[tokio::test]
    async fn failed_forward_fails_the_local_rids() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = axum::Router::new().route(
            "/generate",
            axum::routing::post(|| async {
                (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "kv panic")
            }),
        );
        tokio::spawn(async move { axum::serve(listener, app).await });

        let (detok_tx, detok_rx) = flume::unbounded();
        let senders = Senders {
            tm: flume::unbounded().0,
            abort: flume::unbounded().0,
            tok: flume::unbounded().0,
            detok: vec![detok_tx],
        };
        let pool = pool_with(&[&format!("http://{addr}")]);
        let rid = Rid::from("r1");
        spawn_forward(
            pool.clone(),
            pool.pick().unwrap(),
            "/generate",
            &json!({"text": "hi"}),
            vec![rid.clone()],
            senders.clone(),
        );
        match detok_rx.recv_async().await.unwrap() {
            DetokMsg::Fail {
                rid: failed,
                message,
            } => {
                assert_eq!(failed, rid);
                assert!(message.contains("500"), "{message}");
                assert!(message.contains("kv panic"), "{message}");
            }
            _ => panic!("expected DetokMsg::Fail"),
        }
        assert!(
            pool.list()[0].healthy,
            "an HTTP error status means the worker answered — it stays in rotation"
        );

        // Connection refused (nothing listens on the ephemeral port we just
        // closed by rebinding elsewhere) also fails the rid.
        let dead = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let dead_addr = dead.local_addr().unwrap();
        drop(dead);
        let pool = pool_with(&[&format!("http://{dead_addr}")]);
        spawn_forward(
            pool.clone(),
            pool.pick().unwrap(),
            "/generate",
            &json!({"text": "hi"}),
            vec![Rid::from("r2")],
            senders,
        );
        assert!(matches!(
            detok_rx.recv_async().await.unwrap(),
            DetokMsg::Fail { .. }
        ));
        assert!(
            !pool.list()[0].healthy,
            "a refused connection marks the worker down (out of rotation)"
        );
    }

    /// Sequential forwards to the same worker ride ONE TCP connection: the
    /// shared hyper-util client pools keep-alive connections, and the
    /// drain-to-EOF in `forward` is what returns them to the pool. Guards the
    /// pooling assumption against a per-forward client or a dropped drain.
    #[tokio::test]
    async fn forwards_reuse_pooled_connection() {
        use std::collections::HashSet;
        use std::net::SocketAddr;
        use std::sync::Mutex;

        use axum::extract::ConnectInfo;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let peers: Arc<Mutex<HashSet<SocketAddr>>> = Arc::default();
        let seen = peers.clone();
        let app = axum::Router::new().route(
            "/generate",
            axum::routing::post(move |ConnectInfo(peer): ConnectInfo<SocketAddr>| {
                let seen = seen.clone();
                async move {
                    seen.lock().unwrap().insert(peer);
                    "ok"
                }
            }),
        );
        tokio::spawn(async move {
            axum::serve(
                listener,
                app.into_make_service_with_connect_info::<SocketAddr>(),
            )
            .await
        });

        let pool = pool_with(&[&format!("http://{addr}")]);
        for _ in 0..3 {
            pool.forward(
                &format!("http://{addr}/generate"),
                Bytes::from_static(b"{}"),
            )
            .await
            .expect("forward succeeds");
        }
        assert_eq!(
            peers.lock().unwrap().len(),
            1,
            "three sequential forwards must reuse one pooled connection"
        );
    }

    /// A 200 forward (unary or SSE-shaped) is drained without failing any rid.
    #[tokio::test]
    async fn successful_forward_is_silent() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = axum::Router::new().route(
            "/generate",
            axum::routing::post(|| async { axum::Json(serde_json::json!({"text": "ok"})) }),
        );
        tokio::spawn(async move { axum::serve(listener, app).await });

        let (detok_tx, detok_rx) = flume::unbounded();
        let senders = Senders {
            tm: flume::unbounded().0,
            abort: flume::unbounded().0,
            tok: flume::unbounded().0,
            detok: vec![detok_tx],
        };
        let pool = pool_with(&[&format!("http://{addr}")]);
        // Pre-mark the worker down: the fail-open pick + successful forward
        // must put it back in rotation (passive revive).
        assert!(pool.set_health(&format!("http://{addr}"), false));
        spawn_forward(
            pool.clone(),
            pool.pick().unwrap(),
            "/generate",
            &json!({"text": "hi"}),
            vec![Rid::from("r1")],
            senders,
        );
        // Give the spawned forward a beat to complete, then assert silence.
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        assert!(detok_rx.try_recv().is_err(), "no Fail on success");
        assert!(
            pool.list()[0].healthy,
            "a successful forward revives a down worker"
        );
    }

    // -- Admin API ---------------------------------------------------------

    fn admin_app(pool: Option<Arc<PrefillWorkerPool>>) -> Router {
        let state = AppState {
            senders: senders(),
            egress_buf: 8,
            server_args: Arc::new(
                crate::runtime::ServerArgs::from_json("{}").expect("empty blob parses"),
            ),
            chat_formatter: None,
            egress_activity: Default::default(),
            prefill_worker_pool: pool,
        };
        routes().with_state(state)
    }

    async fn admin_call(
        app: &Router,
        method: &str,
        body: Option<serde_json::Value>,
    ) -> (StatusCode, serde_json::Value) {
        use tower::ServiceExt;
        let builder = Request::builder()
            .method(method)
            .uri("/prefill_workers")
            .header(header::CONTENT_TYPE, "application/json");
        let request = builder
            .body(axum::body::Body::from(
                body.map(|b| b.to_string()).unwrap_or_default(),
            ))
            .unwrap();
        let response = app.clone().oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let value = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
        (status, value)
    }

    /// The curl surface: register → list → deregister round-trips through the
    /// HTTP handlers; a bad entry rejects the whole batch (atomic), and an
    /// unknown URL deletion is a 404.
    #[tokio::test]
    async fn admin_routes_register_list_remove() {
        let app = admin_app(Some(pool_with(&["http://10.0.0.7:30000"])));

        let (status, value) = admin_call(&app, "GET", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(value["prefill_workers"][0]["url"], "http://10.0.0.7:30000");
        assert_eq!(value["prefill_workers"][0]["bootstrap_port"], 30000);
        assert_eq!(value["prefill_workers"][0]["healthy"], true);

        // Batch add; a trailing slash is normalized away, and the bootstrap
        // port is always the URL's own port (rust prefill = api port).
        let (status, value) = admin_call(
            &app,
            "POST",
            Some(json!({"urls": ["http://10.0.0.8:30000/", "http://10.0.0.9:31000"]})),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{value}");
        assert_eq!(value["added"], 2);
        assert_eq!(value["updated"], 0);
        assert_eq!(value["prefill_workers"].as_array().unwrap().len(), 3);
        assert_eq!(value["prefill_workers"][2]["bootstrap_port"], 31000);

        // Atomic: the valid first entry must NOT be applied when the second
        // entry is malformed.
        let (status, _) = admin_call(
            &app,
            "POST",
            Some(json!({"urls": ["http://10.0.0.10:30000", "https://nope:1"]})),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let (_, value) = admin_call(&app, "GET", None).await;
        assert_eq!(value["prefill_workers"].as_array().unwrap().len(), 3);

        let (status, value) = admin_call(
            &app,
            "DELETE",
            Some(json!({"url": "http://10.0.0.8:30000"})),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{value}");
        assert_eq!(value["prefill_workers"].as_array().unwrap().len(), 2);

        let (status, _) = admin_call(
            &app,
            "DELETE",
            Some(json!({"url": "http://10.0.0.8:30000"})),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND, "already deregistered");
    }
}
