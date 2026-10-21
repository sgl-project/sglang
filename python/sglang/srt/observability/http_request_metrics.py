from typing import Dict, Optional, Tuple


class RefCountedGauge:
    def __init__(self, gauge):
        self._gauge = gauge
        self._refcount: Dict[str, int] = {}

    def inc(self, key: str):
        if key in self._refcount:
            self._refcount[key] += 1
        else:
            self._refcount[key] = 1
            self._gauge.inc()

    def dec(self, key: str):
        if key in self._refcount:
            self._refcount[key] -= 1
            if self._refcount[key] == 0:
                del self._refcount[key]
                self._gauge.dec()


class _HttpRequestMetrics:
    def __init__(self):
        from prometheus_client import Counter, Gauge

        self.requests_total = Counter(
            name="sglang:http_requests_total",
            documentation="Total number of HTTP requests by endpoint and method",
            labelnames=["endpoint", "method"],
        )

        self.responses_total = Counter(
            name="sglang:http_responses_total",
            documentation="Total number of HTTP responses by endpoint and status code",
            labelnames=["endpoint", "status_code", "method"],
        )

        self.requests_active = Gauge(
            name="sglang:http_requests_active",
            documentation="Number of currently active HTTP requests",
            labelnames=["endpoint", "method"],
            multiprocess_mode="livesum",
        )

        self.routing_keys_active = RefCountedGauge(
            Gauge(
                name="sglang:routing_keys_active",
                documentation="Number of unique routing keys with active requests",
                multiprocess_mode="livesum",
            )
        )


_http_request_metrics: Optional[_HttpRequestMetrics] = None


def init_http_request_metrics():
    # Call after PROMETHEUS_MULTIPROC_DIR is set: prometheus_client picks its
    # value backend on import.
    global _http_request_metrics
    if _http_request_metrics is None:
        _http_request_metrics = _HttpRequestMetrics()


def add_http_request_metrics_middleware(app):
    """Records nothing until init_http_request_metrics() runs in this process."""
    # Fix: replace BaseHTTPMiddleware's call_next with a pure ASGI version
    # that passes `receive` through, so request.is_disconnected() keeps working.
    from sglang.srt.utils.http_middleware_patch import patch_app_http_middleware

    patch_app_http_middleware(app)

    @app.middleware("http")
    async def track_http_status_code(request, call_next):
        metrics = _http_request_metrics
        if metrics is None:
            return await call_next(request)

        # With recording all requests, we have the risk of high cardinality if requests have arbitrary unhandled paths.
        # But given that SGLang engines with metrics enabled are usually behind routers this looks safe.
        path, is_handled_path = _get_fastapi_request_path(request)
        method = request.method
        routing_key = request.headers.get("x-smg-routing-key")

        metrics.requests_total.labels(endpoint=path, method=method).inc()
        metrics.requests_active.labels(endpoint=path, method=method).inc()
        if routing_key:
            metrics.routing_keys_active.inc(routing_key)

        try:
            response = await call_next(request)

            metrics.responses_total.labels(
                endpoint=path,
                method=method,
                status_code=str(response.status_code),
            ).inc()

            return response
        finally:
            metrics.requests_active.labels(endpoint=path, method=method).dec()
            if routing_key:
                metrics.routing_keys_active.dec(routing_key)


# https://github.com/blueswen/fastapi-observability/blob/132a3c576f8b09e5311c68bd553215013bc75685/fastapi_app/utils.py#L98
def _get_fastapi_request_path(request) -> Tuple[str, bool]:
    from starlette.routing import Match

    for route in request.app.routes:
        match, child_scope = route.matches(request.scope)
        if match == Match.FULL:
            return getattr(route, "path", request.url.path), True

    return request.url.path, False
