//! Regression tests for the `smg_http_connections_active` gauge.
//!
//! Guards against the leak that shipped in the original implementation, where
//! the gauge was decremented via a bare `AtomicU64::fetch_sub` *after*
//! `inner.call(req).await`. When the request future was dropped mid-flight
//! (client disconnect, timeout, connection abort) the decrement never ran, so
//! the gauge grew monotonically. The fix wraps the increment/decrement in an
//! RAII `HttpConnectionGuard`, so Rust's guaranteed `Drop` on future
//! cancellation keeps the gauge in balance.
//!
//! The recorder is a process-global singleton, so these tests run serially and
//! assert on *delta* from a baseline rather than absolute values.

use std::{
    convert::Infallible,
    future::Future,
    pin::Pin,
    sync::{Arc, OnceLock},
    task::{Context, Poll},
    time::Duration,
};

use axum::{body::Body, extract::Request, http::StatusCode, response::Response};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use serial_test::serial;
use smg::{middleware::HttpMetricsLayer, observability::inflight_tracker::InFlightRequestTracker};
use tokio::sync::Notify;
use tower::{Layer, Service, ServiceExt};

/// Install a Prometheus recorder exactly once per test binary and hand back a
/// handle we can render to inspect gauge values.
fn recorder() -> &'static PrometheusHandle {
    static HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();
    HANDLE.get_or_init(|| {
        PrometheusBuilder::new()
            .install_recorder()
            .expect("install prometheus recorder for tests")
    })
}

/// Parse the current value of `smg_http_connections_active` from the rendered
/// Prometheus text. Returns 0.0 if the gauge has not been touched yet.
fn read_gauge(handle: &PrometheusHandle) -> f64 {
    let rendered = handle.render();
    for line in rendered.lines() {
        // Skip `# HELP` / `# TYPE` metadata lines.
        if line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("smg_http_connections_active") {
            // The gauge has no labels, so the remainder is " <value>".
            let value = rest.trim_start_matches([' ', '{', '}']);
            return value.trim().parse::<f64>().unwrap_or(0.0);
        }
    }
    0.0
}

/// A minimal inner service whose response is gated on a `Notify` we hold.
/// This lets the test decide precisely when — or whether — the request's
/// `.await` completes.
#[derive(Clone)]
struct GatedService {
    gate: Arc<Notify>,
}

impl Service<Request> for GatedService {
    type Response = Response;
    type Error = Infallible;
    type Future =
        Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send + 'static>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, _req: Request) -> Self::Future {
        let gate = self.gate.clone();
        Box::pin(async move {
            gate.notified().await;
            Ok(Response::builder()
                .status(StatusCode::OK)
                .body(Body::from("ok"))
                .unwrap())
        })
    }
}

fn build_middleware(
    gate: Arc<Notify>,
) -> impl Service<
    Request,
    Response = Response,
    Error = Infallible,
    Future = Pin<Box<dyn Future<Output = Result<Response, Infallible>> + Send + 'static>>,
> + Clone
       + Send
       + 'static {
    let tracker = InFlightRequestTracker::new();
    let layer = HttpMetricsLayer::new(tracker);
    layer.layer(GatedService { gate })
}

fn make_request() -> Request {
    Request::builder()
        .method("GET")
        .uri("/")
        .body(Body::empty())
        .unwrap()
}

/// Positive control: after a request completes normally, the gauge returns to
/// its baseline.
#[tokio::test]
#[serial]
async fn test_gauge_returns_to_baseline_after_completed_request() {
    let handle = recorder();
    let baseline = read_gauge(handle);

    let gate = Arc::new(Notify::new());
    let svc = build_middleware(gate.clone());

    // Fire the request; it will wait on the gate.
    let call_handle = tokio::spawn({
        let mut svc = svc.clone();
        let req = make_request();
        async move { svc.ready().await.unwrap().call(req).await }
    });

    // Give the middleware a chance to enter the async block and increment.
    tokio::time::sleep(Duration::from_millis(50)).await;
    let during = read_gauge(handle);
    assert_eq!(
        during - baseline,
        1.0,
        "gauge should read baseline+1 while a request is in flight (baseline={baseline}, during={during})"
    );

    // Let the inner service finish.
    gate.notify_one();
    let resp = call_handle.await.unwrap().unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Give the runtime a tick to drop the future's state machine.
    tokio::time::sleep(Duration::from_millis(50)).await;
    let after = read_gauge(handle);
    assert_eq!(
        after, baseline,
        "gauge should return to baseline after a normal request (baseline={baseline}, after={after})"
    );
}

/// Regression: dropping the request future mid-flight (client disconnect,
/// timeout, abort) must not leak the gauge. Before the fix, the decrement
/// lived on a code path after `.await` that never ran on cancellation, so this
/// test asserts the gauge falls back to baseline instead of leaking +1.
#[tokio::test]
#[serial]
async fn test_gauge_returns_to_baseline_when_future_cancelled() {
    let handle = recorder();
    let baseline = read_gauge(handle);

    let gate = Arc::new(Notify::new());
    let svc = build_middleware(gate.clone());

    // Start the request in a task, then abort the task without ever notifying
    // the gate. This drops the middleware's future while it is suspended on
    // `.await` — the exact scenario that used to leak.
    let call_handle = tokio::spawn({
        let mut svc = svc.clone();
        let req = make_request();
        async move {
            let _ = svc.ready().await.unwrap().call(req).await;
        }
    });

    tokio::time::sleep(Duration::from_millis(50)).await;
    let during = read_gauge(handle);
    assert_eq!(
        during - baseline,
        1.0,
        "gauge should read baseline+1 while a request is in flight (baseline={baseline}, during={during})"
    );

    // Simulate cancellation: drop the future mid-`.await`.
    call_handle.abort();
    let _ = call_handle.await;

    // Give the runtime a tick to run Drop for the aborted task's state
    // machine, which is what decrements the gauge under the fix.
    tokio::time::sleep(Duration::from_millis(50)).await;
    let after = read_gauge(handle);
    assert_eq!(
        after, baseline,
        "gauge must return to baseline after future cancellation (baseline={baseline}, after={after}); \
         a non-zero delta indicates the leak has regressed"
    );
}

/// Concurrent bursts of completed and cancelled requests must still return the
/// gauge to its baseline. Guards against subtle races between the RAII
/// increment/decrement.
#[tokio::test]
#[serial]
async fn test_gauge_stable_under_mixed_completion_and_cancellation() {
    let handle = recorder();
    let baseline = read_gauge(handle);

    let gate = Arc::new(Notify::new());
    let svc = build_middleware(gate.clone());

    // Half the requests will be aborted, half will be allowed to complete.
    let mut tasks = Vec::new();
    for _ in 0..20 {
        let mut svc = svc.clone();
        tasks.push(tokio::spawn(async move {
            let req = make_request();
            let _ = svc.ready().await.unwrap().call(req).await;
        }));
    }

    tokio::time::sleep(Duration::from_millis(50)).await;
    let during = read_gauge(handle);
    assert_eq!(
        during - baseline,
        20.0,
        "gauge should show 20 in-flight requests (baseline={baseline}, during={during})"
    );

    // Abort the first 10 tasks; wake the remaining 10 via the gate.
    for task in tasks.iter().take(10) {
        task.abort();
    }
    for _ in 0..10 {
        gate.notify_one();
    }
    for task in tasks {
        let _ = task.await;
    }

    tokio::time::sleep(Duration::from_millis(100)).await;
    let after = read_gauge(handle);
    assert_eq!(
        after, baseline,
        "gauge must return to baseline after mixed completions and cancellations \
         (baseline={baseline}, after={after})"
    );
}
