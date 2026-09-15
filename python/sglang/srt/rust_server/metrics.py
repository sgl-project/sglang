"""Python-owned collectors for the embedded Rust frontend.

Collection runs in the launch process. Scheduler processes only inherit the
private socket path and continue writing their existing Prometheus mmap files.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import json
import logging
import math
import os
import socket
import tempfile
import threading
import uuid
from contextlib import AbstractContextManager
from functools import partial

logger = logging.getLogger(__name__)

METRICS_SOCKET_ENV = "SGLANG_RUST_METRICS_SOCKET"
METRICS_SOURCE_ENV = "SGLANG_RUST_METRICS_SOURCE"
COLLECTION_TIMEOUT_SECONDS = 5.0


def frontend_metrics_config() -> str | None:
    from sglang.srt.disaggregation.utils import DisaggregationMode
    from sglang.srt.observability.utils import generate_buckets
    from sglang.srt.runtime_context import (
        get_disagg,
        get_observability,
        get_schedule,
        get_serving,
    )

    config = get_observability()
    if not config.enable_metrics:
        return None
    labels = {
        "model_name": get_serving().served_model_name,
        "engine_type": DisaggregationMode.to_engine_type(
            get_disagg().disaggregation_mode
        ),
    }
    if get_schedule().enable_priority_scheduling:
        labels["priority"] = ""
    for name in config.tokenizer_metrics_allowed_custom_labels or []:
        labels[name] = ""
    labels.update(config.extra_metric_labels or {})

    def buckets(rule):
        return generate_buckets(rule, []) if rule and rule[0] != "default" else None

    def finite_buckets(values):
        if values is None:
            return None
        # Both clients add the implicit +Inf bucket. JSON must not carry
        # Infinity as a number, even when the CLI specified it explicitly.
        return [value for value in values if value != math.inf]

    return json.dumps(
        {
            "labels": {name: str(value) for name, value in labels.items()},
            "http_labels": {
                name: str(value)
                for name, value in (config.extra_metric_labels or {}).items()
            },
            "allowed_custom_labels": config.tokenizer_metrics_allowed_custom_labels
            or [],
            "custom_labels_header": config.tokenizer_metrics_custom_labels_header,
            "priority_enabled": get_schedule().enable_priority_scheduling,
            "bucket_time_to_first_token": finite_buckets(
                config.bucket_time_to_first_token
            ),
            "bucket_inter_token_latency": finite_buckets(
                config.bucket_inter_token_latency
            ),
            "bucket_e2e_request_latency": finite_buckets(
                config.bucket_e2e_request_latency
            ),
            "prompt_tokens_buckets": finite_buckets(
                buckets(config.prompt_tokens_buckets)
            ),
            "generation_tokens_buckets": finite_buckets(
                buckets(config.generation_tokens_buckets)
            ),
        },
        allow_nan=False,
    )


class _EscapedCollector:
    """Apply negotiated name escaping consistently to families and samples."""

    def __init__(self, registry, escaping):
        self.registry = registry
        self.escaping = escaping

    def _sample(self, sample):
        from prometheus_client.openmetrics import exposition

        labels = {
            exposition.escape_label_name(key, self.escaping): value
            for key, value in sample.labels.items()
        }
        if len(labels) != len(sample.labels):
            raise ValueError(f"Escaped label names collide in metric {sample.name}")
        return sample._replace(
            name=exposition.escape_metric_name(sample.name, self.escaping),
            labels=labels,
        )

    def collect(self):
        from prometheus_client.openmetrics import exposition

        for original in self.registry.collect():
            if self.escaping == exposition.ALLOWUTF8:
                yield original
                continue
            metric = copy.copy(original)
            metric.name = exposition.escape_metric_name(original.name, self.escaping)
            metric.samples = [self._sample(sample) for sample in original.samples]
            yield metric


def encode_metrics(registry, encoder):
    from prometheus_client.openmetrics import exposition

    if isinstance(encoder, partial) and encoder.func is exposition.generate_latest:
        # prometheus_client 0.24–0.26 escapes sample names as label names,
        # replacing the legal ':' in sglang metric names but not their HELP/TYPE
        # names. Escape the public collector objects first, then use lossless
        # encoding. This also preserves the client's requested escaping scheme.
        escaping = encoder.keywords.get("escaping", exposition.UNDERSCORES)
        return encoder(
            _EscapedCollector(registry, escaping), escaping=exposition.ALLOWUTF8
        )
    return encoder(registry)


def make_metrics_app(metrics_dir: str, collection=None, source_path: str | None = None):
    from aiohttp import ClientSession, web
    from prometheus_client.exposition import choose_encoder

    from sglang.srt.rust_server.metrics_sources import (
        MAX_SCRAPE_BYTES,
        MetricsCollection,
    )

    collection = collection or MetricsCollection(metrics_dir, uuid.uuid4().hex)
    pending: dict[str, asyncio.Task[bytes]] = {}
    session = None

    async def client_lifetime(app):
        nonlocal session
        async with ClientSession() as session:
            yield

    async def collect(encoder):
        registry = await collection.collect(session)
        body = await asyncio.to_thread(encode_metrics, registry, encoder)
        if len(body) > MAX_SCRAPE_BYTES:
            raise ValueError("Metrics exposition exceeds the 64 MiB scrape budget")
        return body

    def finished(content_type, task):
        if pending.get(content_type) is task:
            del pending[content_type]
        if not task.cancelled():
            task.exception()

    async def metrics(request):
        encoder, content_type = choose_encoder(request.headers.get("Accept", ""))
        task = pending.get(content_type)
        if task is None:
            task = asyncio.create_task(collect(encoder))
            pending[content_type] = task
            task.add_done_callback(lambda done: finished(content_type, done))
        try:
            data = await asyncio.wait_for(
                asyncio.shield(task), timeout=COLLECTION_TIMEOUT_SECONDS
            )
        except TimeoutError:
            logger.error("Prometheus collection exceeded the scrape deadline")
            return web.Response(status=503, text="Metrics collection timed out")
        except Exception:
            logger.exception("Prometheus collection failed")
            return web.Response(status=500, text="Metrics collection failed")
        return web.Response(body=data, headers={"Content-Type": content_type})

    app = web.Application()
    app.cleanup_ctx.append(client_lifetime)
    app.router.add_get("/metrics", metrics)
    app.router.add_get("/metrics/", metrics)
    if source_path is not None:

        async def snapshot_bytes():
            snapshot = await collection.snapshot()
            data = await asyncio.to_thread(json.dumps, snapshot, allow_nan=False)
            data = data.encode("utf-8")
            if len(data) > MAX_SCRAPE_BYTES:
                raise ValueError(
                    "Private metric snapshot exceeds the 64 MiB scrape budget"
                )
            return data

        async def snapshot(request):
            task = pending.get("private-source")
            if task is None:
                task = asyncio.create_task(snapshot_bytes())
                pending["private-source"] = task
                task.add_done_callback(lambda done: finished("private-source", done))
            try:
                data = await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=COLLECTION_TIMEOUT_SECONDS,
                )
                return web.Response(body=data, content_type="application/json")
            except TimeoutError:
                return web.Response(status=503, text="Metrics collection timed out")
            except Exception:
                logger.exception("Private metric snapshot failed")
                return web.Response(status=500, text="Metrics collection failed")

        app.router.add_get(source_path, snapshot)
    return app


class MetricsExporter(AbstractContextManager):
    """Own one private collector service for the lifetime of a launched engine."""

    def __init__(self, host: str = "127.0.0.1"):
        self.host = host
        self.source = None
        self.socket_path: str | None = None
        self._collection = None
        self._directory: tempfile.TemporaryDirectory | None = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._ready: concurrent.futures.Future[None] = concurrent.futures.Future()
        self._previous_socket: str | None = None
        self._previous_source: str | None = None

    def __enter__(self):
        from sglang.srt.utils.common import set_prometheus_multiproc_dir

        set_prometheus_multiproc_dir()
        self._directory = tempfile.TemporaryDirectory(prefix="sglang-metrics-")
        self.socket_path = os.path.join(self._directory.name, "collector.sock")
        self._previous_socket = os.environ.get(METRICS_SOCKET_ENV)
        self._previous_source = os.environ.get(METRICS_SOURCE_ENV)
        self._thread = threading.Thread(
            target=self._run, name="rust-metrics-collector", daemon=True
        )
        self._thread.start()
        try:
            self._ready.result(timeout=10)
        except BaseException:
            self.close()
            raise
        os.environ[METRICS_SOCKET_ENV] = self.socket_path
        os.environ[METRICS_SOURCE_ENV] = json.dumps(self.source.as_dict())
        return self

    def _run(self):
        try:
            asyncio.run(self._serve())
        except BaseException as error:
            if not self._ready.done():
                self._ready.set_exception(error)
            else:
                logger.exception("Rust metrics collector stopped unexpectedly")

    async def _serve(self):
        from aiohttp import web

        from sglang.srt.rust_server.metrics_sources import (
            MetricsCollection,
            MetricSource,
        )
        from sglang.srt.utils.network import NetworkAddress

        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        source_id = uuid.uuid4().hex
        source_path = f"/source/{source_id}"
        metrics_dir = os.environ["PROMETHEUS_MULTIPROC_DIR"]
        self._collection = MetricsCollection(metrics_dir, source_id)
        app = make_metrics_app(metrics_dir, self._collection, source_path)
        runner = web.AppRunner(
            app, access_log=None, shutdown_timeout=COLLECTION_TIMEOUT_SECONDS
        )
        await runner.setup()
        try:
            site = web.UnixSite(runner, self.socket_path)
            await site.start()
            os.chmod(self.socket_path, 0o600)
            family = socket.AF_INET6 if ":" in self.host else socket.AF_INET
            with socket.create_server((self.host, 0), family=family) as listener:
                listener.setblocking(False)
                await web.SockSite(runner, listener).start()
                address = NetworkAddress(self.host, listener.getsockname()[1]).to_url()
                self.source = MetricSource(source_id, address + source_path)
                self._ready.set_result(None)
                await self._stop.wait()
        finally:
            await runner.cleanup()

    def configure(
        self, scheduler_infos, ingress_url: str | None = None, *, tokenizer_e2e: float
    ):
        from sglang.srt.observability.startup_time import build_engine_startup_time
        from sglang.srt.rust_server.metrics_sources import MetricSource

        sources = []
        for info in scheduler_infos:
            for worker in info["rust_worker_infos"]:
                url = worker["url"]
                sources.append(
                    MetricSource(f"rust:{url}", url + "/metrics/native", "native")
                )
                sources.extend(
                    MetricSource(**source) for source in worker["metrics_sources"]
                )
        if ingress_url is not None:
            sources.append(
                MetricSource("rust:ingress", ingress_url + "/metrics/native", "native")
            )

        startup_time = build_engine_startup_time(
            (info.get("startup_time") for info in scheduler_infos),
            tokenizer_e2e=tokenizer_e2e,
        )
        labels = json.loads(frontend_metrics_config())["labels"]

        async def update():
            self._collection.configure(sources)
            self._collection.set_startup_time(startup_time, labels)

        future = asyncio.run_coroutine_threadsafe(update(), self._loop)
        future.result(timeout=10)

    def close(self):
        if (
            self._loop is not None
            and not self._loop.is_closed()
            and self._stop is not None
        ):
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread is not None:
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                raise RuntimeError("Rust metrics collector did not stop")
        if self._previous_socket is None:
            os.environ.pop(METRICS_SOCKET_ENV, None)
        else:
            os.environ[METRICS_SOCKET_ENV] = self._previous_socket
        if self._previous_source is None:
            os.environ.pop(METRICS_SOURCE_ENV, None)
        else:
            os.environ[METRICS_SOURCE_ENV] = self._previous_source
        if self._directory is not None:
            self._directory.cleanup()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
