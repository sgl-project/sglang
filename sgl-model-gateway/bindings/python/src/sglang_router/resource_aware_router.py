"""Resource-aware routing for the local multi-model supervisor.

The Rust Router remains the authoritative worker registry and handles replica
selection inside a concrete model.  This module is a small Router-edge service
that resolves an explicitly configured *virtual* model to one of several
compatible concrete model IDs before proxying the request to that Router.

It intentionally relies on SGLang's ``/v1/loads`` endpoint rather than GPU
process accounting.  The runtime is the only component that can accurately
attribute KV-cache and queue pressure to one of several processes that may
share a physical GPU.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import threading
import time
from collections.abc import Iterable
from contextlib import asynccontextmanager
from typing import Any

import aiohttp
import orjson
import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from setproctitle import setproctitle

from sglang_router.multi_model import ModelResolverConfig, RoutingProfile

logger = logging.getLogger(__name__)

_HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
_STREAM_CHUNK_SIZE = 64 * 1024


class ResourceUnavailableError(RuntimeError):
    """No fresh runtime state has enough declared capacity for a profile."""


@dataclasses.dataclass(frozen=True)
class RuntimeEndpoint:
    """One directly observable SGLang Runtime registered with the Router."""

    model_id: str
    url: str


@dataclasses.dataclass(frozen=True)
class RuntimeLoadState:
    """Normalized resource state collected from one Runtime's ``/v1/loads``."""

    model_id: str
    url: str
    observed_at: float
    healthy: bool
    num_running_reqs: int = 0
    num_waiting_reqs: int = 0
    num_total_tokens: int = 0
    max_total_num_tokens: int = 0
    max_running_requests: int = 0
    token_usage: float = 0.0
    utilization: float = 0.0
    memory: dict[str, Any] | None = None
    error: str | None = None

    @property
    def free_tokens(self) -> int:
        return max(0, self.max_total_num_tokens - self.num_total_tokens)

    def as_dict(self, *, now: float) -> dict[str, Any]:
        age_secs = max(0.0, now - self.observed_at)
        return {
            "model_id": self.model_id,
            "url": self.url,
            "healthy": self.healthy,
            "age_secs": round(age_secs, 3),
            "num_running_reqs": self.num_running_reqs,
            "num_waiting_reqs": self.num_waiting_reqs,
            "num_total_tokens": self.num_total_tokens,
            "max_total_num_tokens": self.max_total_num_tokens,
            "free_tokens": self.free_tokens,
            "max_running_requests": self.max_running_requests,
            "token_usage": self.token_usage,
            "utilization": self.utilization,
            "memory": self.memory,
            "error": self.error,
        }


def _int_field(payload: dict[str, Any], field: str) -> int:
    value = payload.get(field, 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))


def _float_field(payload: dict[str, Any], field: str) -> float:
    value = payload.get(field, 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return max(0.0, float(value))


def parse_runtime_load(
    endpoint: RuntimeEndpoint, payload: dict[str, Any], *, observed_at: float
) -> RuntimeLoadState:
    """Aggregate the per-DP-rank SGLang load response into one Runtime state."""

    loads = payload.get("loads")
    if not isinstance(loads, list) or not loads:
        raise ValueError("/v1/loads did not return a non-empty loads list")
    if any(not isinstance(load, dict) for load in loads):
        raise ValueError("/v1/loads contains a non-object load record")

    records = list(loads)
    max_total_num_tokens = sum(
        _int_field(record, "max_total_num_tokens") for record in records
    )
    num_total_tokens = sum(_int_field(record, "num_total_tokens") for record in records)
    token_usage = (
        num_total_tokens / max_total_num_tokens
        if max_total_num_tokens > 0
        else 1.0
    )
    memories = [record.get("memory") for record in records]
    memory_records = [memory for memory in memories if isinstance(memory, dict)]
    memory: dict[str, Any] | None = None
    if memory_records:
        memory = {
            "weight_gb": round(
                sum(_float_field(record, "weight_gb") for record in memory_records), 3
            ),
            "kv_cache_gb": round(
                sum(_float_field(record, "kv_cache_gb") for record in memory_records),
                3,
            ),
            "graph_gb": round(
                sum(_float_field(record, "graph_gb") for record in memory_records), 3
            ),
            "token_capacity": sum(
                _int_field(record, "token_capacity") for record in memory_records
            ),
        }

    return RuntimeLoadState(
        model_id=endpoint.model_id,
        url=endpoint.url,
        observed_at=observed_at,
        healthy=True,
        num_running_reqs=sum(_int_field(record, "num_running_reqs") for record in records),
        num_waiting_reqs=sum(_int_field(record, "num_waiting_reqs") for record in records),
        num_total_tokens=num_total_tokens,
        max_total_num_tokens=max_total_num_tokens,
        max_running_requests=sum(
            _int_field(record, "max_running_requests") for record in records
        ),
        token_usage=token_usage,
        utilization=max(_float_field(record, "utilization") for record in records),
        memory=memory,
    )


class ResourceAwareModelResolver:
    """Select a concrete model from fresh, runtime-owned load snapshots."""

    def __init__(self, config: ModelResolverConfig):
        self._config = config
        self._profiles = {profile.model_id: profile for profile in config.profiles}
        self._states: dict[tuple[str, str], RuntimeLoadState] = {}
        self._lock = threading.Lock()

    @property
    def config(self) -> ModelResolverConfig:
        return self._config

    def update(self, state: RuntimeLoadState) -> None:
        with self._lock:
            self._states[(state.model_id, state.url)] = state

    def mark_unhealthy(self, endpoint: RuntimeEndpoint, error: str, *, now: float) -> None:
        self.update(
            RuntimeLoadState(
                model_id=endpoint.model_id,
                url=endpoint.url,
                observed_at=now,
                healthy=False,
                error=error,
            )
        )

    def is_virtual_model(self, model_id: str) -> bool:
        return model_id in self._profiles

    def _eligible_score(
        self,
        state: RuntimeLoadState,
        profile: RoutingProfile,
        required_free_tokens: int,
        *,
        now: float,
    ) -> float | None:
        if not state.healthy:
            return None
        if now - state.observed_at > self._config.stale_after_secs:
            return None
        if state.max_total_num_tokens <= 0:
            return None
        if state.free_tokens < required_free_tokens:
            return None
        if state.token_usage >= profile.max_kv_utilization:
            return None
        if state.num_waiting_reqs >= profile.max_waiting_requests:
            return None

        running_ratio = state.num_running_reqs / max(1, state.max_running_requests)
        waiting_ratio = state.num_waiting_reqs / max(1, state.max_running_requests)
        capacity_ratio = state.num_total_tokens / state.max_total_num_tokens
        # Queueing and KV exhaustion are intentionally stronger signals than
        # instantaneous scheduler utilization, which can oscillate rapidly.
        return (
            state.token_usage * 1_000.0
            + capacity_ratio * 100.0
            + waiting_ratio * 50.0
            + running_ratio * 10.0
            + state.utilization
        )

    def resolve(
        self,
        requested_model_id: str,
        *,
        max_tokens: int | None = None,
        now: float | None = None,
    ) -> str:
        """Return a concrete model ID, preserving explicit model requests."""

        profile = self._profiles.get(requested_model_id)
        if profile is None:
            return requested_model_id

        timestamp = time.monotonic() if now is None else now
        required_free_tokens = max(profile.min_free_tokens, max_tokens or 1)
        with self._lock:
            states = tuple(self._states.values())

        candidate_scores: dict[str, float] = {}
        for candidate in profile.candidates:
            scores = [
                score
                for state in states
                if state.model_id == candidate
                and (
                    score := self._eligible_score(
                        state,
                        profile,
                        required_free_tokens,
                        now=timestamp,
                    )
                )
                is not None
            ]
            if scores:
                candidate_scores[candidate] = min(scores)

        if not candidate_scores:
            raise ResourceUnavailableError(
                f"No eligible Runtime for virtual model {requested_model_id!r}; "
                "all candidates are unhealthy, stale, queued, or KV-constrained"
            )

        # Preserve declared candidate order as a stable final tie breaker.
        _index, selected = min(
            enumerate(profile.candidates),
            key=lambda item: (candidate_scores.get(item[1], float("inf")), item[0]),
        )
        return selected

    def snapshot(self, *, now: float | None = None) -> dict[str, Any]:
        timestamp = time.monotonic() if now is None else now
        with self._lock:
            states = tuple(self._states.values())
        return {
            "profiles": [
                {
                    "model_id": profile.model_id,
                    "candidates": list(profile.candidates),
                    "max_kv_utilization": profile.max_kv_utilization,
                    "max_waiting_requests": profile.max_waiting_requests,
                    "min_free_tokens": profile.min_free_tokens,
                }
                for profile in self._config.profiles
            ],
            "runtimes": [
                state.as_dict(now=timestamp)
                for state in sorted(states, key=lambda item: (item.model_id, item.url))
            ],
            "stale_after_secs": self._config.stale_after_secs,
        }

    def readiness(self, *, now: float | None = None) -> dict[str, Any]:
        """Report whether every virtual model still has a fresh candidate."""

        timestamp = time.monotonic() if now is None else now
        with self._lock:
            states = tuple(self._states.values())

        profiles = []
        for profile in self._config.profiles:
            fresh_candidates = sorted(
                {
                    state.model_id
                    for state in states
                    if state.model_id in profile.candidates
                    and state.healthy
                    and timestamp - state.observed_at <= self._config.stale_after_secs
                }
            )
            profiles.append(
                {
                    "model_id": profile.model_id,
                    "ready": bool(fresh_candidates),
                    "fresh_candidates": fresh_candidates,
                }
            )
        return {
            "ready": all(profile["ready"] for profile in profiles),
            "profiles": profiles,
        }


class RuntimeLoadCollector:
    """Poll direct Runtime load endpoints in a bounded background thread."""

    def __init__(
        self,
        resolver: ResourceAwareModelResolver,
        endpoints: Iterable[RuntimeEndpoint],
        *,
        refresh_interval_secs: float,
        request_timeout_secs: float,
    ):
        self._resolver = resolver
        self._endpoints = tuple(endpoints)
        self._refresh_interval_secs = refresh_interval_secs
        self._request_timeout_secs = request_timeout_secs
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def refresh_once(self) -> None:
        for endpoint in self._endpoints:
            now = time.monotonic()
            try:
                response = requests.get(
                    f"{endpoint.url}/v1/loads?include=core,memory,queues",
                    timeout=self._request_timeout_secs,
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("/v1/loads returned a non-object JSON value")
                self._resolver.update(parse_runtime_load(endpoint, payload, observed_at=now))
            except (requests.RequestException, ValueError) as exc:
                message = str(exc)
                logger.warning("Failed to collect Runtime load from %s: %s", endpoint.url, message)
                self._resolver.mark_unhealthy(endpoint, message, now=now)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.refresh_once()
            self._stop.wait(self._refresh_interval_secs)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="sglang-runtime-load-collector",
            daemon=True,
        )
        self._thread.start()

    @property
    def endpoints(self) -> tuple[RuntimeEndpoint, ...]:
        return self._endpoints

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._request_timeout_secs + 1.0)
            self._thread = None


def _forward_request_headers(request: Request) -> dict[str, str]:
    return {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _HOP_BY_HOP_HEADERS
    }


def _response_headers(response: aiohttp.ClientResponse, resolved_model: str) -> dict[str, str]:
    headers = {"x-sglang-resolved-model": resolved_model}
    content_type = response.headers.get("content-type")
    if content_type:
        headers["content-type"] = content_type
    return headers


def _max_tokens_from_body(body: dict[str, Any]) -> int | None:
    value = body.get("max_tokens", body.get("max_completion_tokens"))
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


async def _proxy_resolved_request(
    request: Request,
    body: dict[str, Any],
    *,
    backend_url: str,
    path: str,
    resolved_model: str,
    request_timeout_secs: float,
) -> Response:
    upstream_url = f"{backend_url}{path}"
    headers = _forward_request_headers(request)
    encoded_body = orjson.dumps(body)
    timeout = aiohttp.ClientTimeout(total=request_timeout_secs)
    is_streaming = body.get("stream") is True

    if not is_streaming:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(upstream_url, data=encoded_body, headers=headers) as upstream:
                content = await upstream.read()
                return Response(
                    content=content,
                    status_code=upstream.status,
                    headers=_response_headers(upstream, resolved_model),
                )

    session = aiohttp.ClientSession(timeout=timeout)
    try:
        upstream = await session.post(upstream_url, data=encoded_body, headers=headers)
    except Exception:
        await session.close()
        raise

    async def stream_response():
        try:
            async for chunk in upstream.content.iter_chunked(_STREAM_CHUNK_SIZE):
                yield chunk
        finally:
            upstream.release()
            await session.close()

    return StreamingResponse(
        stream_response(),
        status_code=upstream.status,
        headers=_response_headers(upstream, resolved_model),
    )


def create_resource_aware_app(
    resolver: ResourceAwareModelResolver,
    collector: RuntimeLoadCollector,
    *,
    backend_url: str,
    request_timeout_secs: float,
) -> FastAPI:
    """Create the public HTTP endpoint for virtual-model resolution."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        collector.start()
        try:
            yield
        finally:
            collector.stop()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> JSONResponse:
        """Liveness only; use /ready for candidate availability."""

        return JSONResponse({"status": "ok"})

    @app.get("/health/verbose")
    async def health_verbose() -> JSONResponse:
        readiness = resolver.readiness()
        return JSONResponse(
            {
                "status": "ok",
                "ready": readiness["ready"],
                "profiles": readiness["profiles"],
                "runtimes": resolver.snapshot()["runtimes"],
            }
        )

    @app.get("/ready")
    async def ready() -> JSONResponse:
        readiness = resolver.readiness()
        return JSONResponse(
            {"status": "ready" if readiness["ready"] else "not_ready", **readiness},
            status_code=200 if readiness["ready"] else 503,
        )

    @app.get("/v1/models")
    async def models() -> JSONResponse:
        """List virtual routing profiles and directly addressable Runtime models."""

        data = [
            {
                "id": profile.model_id,
                "object": "model",
                "created": 0,
                "owned_by": "sglang-resource-resolver",
            }
            for profile in resolver.config.profiles
        ]
        concrete_model_ids = list(
            dict.fromkeys(endpoint.model_id for endpoint in collector.endpoints)
        )
        data.extend(
            {
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "sglang-runtime",
            }
            for model_id in concrete_model_ids
        )
        return JSONResponse({"object": "list", "data": data})

    @app.get("/v1/runtime-loads")
    async def runtime_loads() -> JSONResponse:
        return JSONResponse(resolver.snapshot())

    async def handle_model_request(request: Request, path: str) -> Response:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "Request body must be valid JSON"}},
            )
        if not isinstance(body, dict) or not isinstance(body.get("model"), str):
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "Request must contain string field 'model'"}},
            )

        requested_model = body["model"]
        try:
            resolved_model = resolver.resolve(
                requested_model,
                max_tokens=_max_tokens_from_body(body),
            )
        except ResourceUnavailableError as exc:
            return JSONResponse(
                status_code=503,
                content={"error": {"message": str(exc), "type": "resource_unavailable"}},
            )

        body["model"] = resolved_model
        response = await _proxy_resolved_request(
            request,
            body,
            backend_url=backend_url,
            path=path,
            resolved_model=resolved_model,
            request_timeout_secs=request_timeout_secs,
        )
        response.headers["x-sglang-requested-model"] = requested_model
        response.headers["x-sglang-resolved-model"] = resolved_model
        return response

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await handle_model_request(request, "/v1/chat/completions")

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        return await handle_model_request(request, "/v1/completions")

    return app


def run_resource_aware_router(
    config: ModelResolverConfig,
    endpoints: tuple[RuntimeEndpoint, ...],
    backend_url: str,
) -> None:
    """Entrypoint used by the supervisor in a dedicated process group."""

    os.setpgrp()
    setproctitle("sglang::resource-router")
    resolver = ResourceAwareModelResolver(config)
    collector = RuntimeLoadCollector(
        resolver,
        endpoints,
        refresh_interval_secs=config.refresh_interval_secs,
        request_timeout_secs=min(config.request_timeout_secs, 10.0),
    )
    # Publish an initial state before accepting requests.  A failed poll is
    # intentionally visible as a 503 rather than silently selecting a model.
    collector.refresh_once()
    app = create_resource_aware_app(
        resolver,
        collector,
        backend_url=backend_url,
        request_timeout_secs=config.request_timeout_secs,
    )
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")
