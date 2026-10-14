# SPDX-License-Identifier: Apache-2.0
"""Experimental warm-standby gateway for resident-weight clients.

Run with ``python -m sglang.srt.weight_cache.standby --config config.json``.
The weight daemon is managed separately. Commands must launch independent client
process groups with distinct HTTP/NCCL ports and bounded private KV pools.
Only stateless inference is routed. Failed requests are never replayed.
"""

import argparse
import asyncio
import contextlib
import json
import logging
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp
from aiohttp import web

logger = logging.getLogger(__name__)
HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}
ROUTES = {
    "/generate",
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/models",
    "/get_model_info",
}


def forward_headers(headers):
    excluded = HOP_HEADERS | {
        value.strip().lower() for value in headers.get("Connection", "").split(",")
    }
    return {key: value for key, value in headers.items() if key.lower() not in excluded}


@dataclass
class Worker:
    name: str
    url: str
    command: list
    env: dict = field(default_factory=dict)
    state: str = "starting"
    process: object = None
    generation: int = 0
    ready_at: float = 0.0
    last_error: str = ""
    next_probe: float = 0.0
    next_start: float = 0.0
    log_file: object = None


class StandbyGateway:
    def __init__(self, config):
        specs = config["workers"]
        if len(specs) != 2:
            raise ValueError("Exactly two workers are required")
        if len({s["name"] for s in specs}) != 2 or len({s["url"] for s in specs}) != 2:
            raise ValueError("Worker names and URLs must be distinct")
        for spec in specs:
            parsed = urlsplit(spec["url"])
            if (
                parsed.scheme != "http"
                or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parsed.path not in {"", "/"}
            ):
                raise ValueError("Workers must use loopback HTTP URLs")
            if not spec.get("command") or not all(
                isinstance(x, str) for x in spec["command"]
            ):
                raise ValueError("Each worker needs a non-empty command argument list")
        self.workers = [
            Worker(s["name"], s["url"].rstrip("/"), s["command"], s.get("env", {}))
            for s in specs
        ]
        self.config = config
        self.active = None
        self.epoch = 0
        self.events = []
        self.session = None
        self.tasks = []
        self.log_dir = Path(config.get("log_dir", "standby-logs"))
        self.warmup = config.get(
            "warmup",
            {
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 4},
            },
        )
        if self.warmup.get("stream") or "session_params" in self.warmup:
            raise ValueError("Warmup must be non-streaming and stateless")
        for key in (
            "poll_interval",
            "health_interval",
            "health_timeout",
            "restart_backoff",
            "startup_timeout",
            "warmup_timeout",
        ):
            if float(config.get(key, 1)) <= 0:
                raise ValueError(f"{key} must be positive")

    def event(self, kind, **fields):
        record = {
            "event": kind,
            "monotonic": time.monotonic(),
            "time": time.time(),
            **fields,
        }
        self.events.append(record)
        self.events = self.events[-100:]
        logger.info(json.dumps(record))

    def choose(self):
        for worker in self.workers:
            if (
                worker.state == "ready"
                and worker.process is not None
                and worker.process.returncode is not None
            ):
                worker.state = "failed"
                worker.last_error = f"launcher exited {worker.process.returncode}"
                self.event(
                    "failed",
                    worker=worker.name,
                    generation=worker.generation,
                    reason=worker.last_error,
                )
                worker.next_start = time.monotonic() + self.config.get(
                    "restart_backoff", 2
                )
        if self.active is not None and self.active.state == "ready":
            return self.active
        old = self.active.name if self.active else None
        self.active = next((w for w in self.workers if w.state == "ready"), None)
        if self.active is not None:
            self.epoch += 1
            self.event(
                "promoted", previous=old, worker=self.active.name, epoch=self.epoch
            )
        return self.active

    def fail(self, worker, error):
        if worker.state == "failed":
            return
        worker.state = "failed"
        worker.last_error = str(error)
        self.event(
            "failed",
            worker=worker.name,
            generation=worker.generation,
            reason=str(error),
        )
        self.choose()

    async def probe(self, worker):
        timeout = aiohttp.ClientTimeout(total=self.config.get("health_timeout", 2.0))
        try:
            async with self.session.get(
                worker.url + self.config.get("health_path", "/health"), timeout=timeout
            ) as response:
                await response.read()
                return response.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    async def stop_worker(self, worker):
        process = worker.process
        if process is not None:
            # Kill descendants even if the launcher has already exited.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
            worker.process = None
        if worker.log_file is not None:
            worker.log_file.close()
            worker.log_file = None

    async def prepare(self, worker):
        worker.state = "starting"
        worker.generation += 1
        worker.log_file = open(
            self.log_dir / f"{worker.name}.{worker.generation}.log", "ab", buffering=0
        )
        try:
            worker.process = await asyncio.create_subprocess_exec(
                *worker.command,
                env={**os.environ, **worker.env},
                stdout=worker.log_file,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            self.event(
                "started",
                worker=worker.name,
                generation=worker.generation,
                pid=worker.process.pid,
            )
            deadline = time.monotonic() + self.config.get("startup_timeout", 600)
            while time.monotonic() < deadline:
                if worker.process.returncode is not None:
                    raise RuntimeError(f"launcher exited {worker.process.returncode}")
                if await self.probe(worker):
                    async with self.session.post(
                        worker.url + "/generate",
                        json=self.warmup,
                        timeout=aiohttp.ClientTimeout(
                            total=self.config.get("warmup_timeout", 180)
                        ),
                    ) as response:
                        body = await response.read()
                        if response.status != 200:
                            raise RuntimeError(
                                f"warmup HTTP {response.status}: {body[:200]!r}"
                            )
                        result = json.loads(body)
                        if (
                            not isinstance(result, dict)
                            or "text" not in result
                            or "error" in result
                        ):
                            raise RuntimeError(
                                "warmup did not return a generation response"
                            )
                    if worker.process.returncode is not None:
                        raise RuntimeError("launcher exited during warmup")
                    worker.ready_at = time.monotonic()
                    worker.next_probe = worker.ready_at + self.config.get(
                        "health_interval", 1
                    )
                    worker.state = "ready"
                    self.event(
                        "ready", worker=worker.name, generation=worker.generation
                    )
                    self.choose()
                    return
                await asyncio.sleep(0.5)
            raise TimeoutError("worker startup timeout")
        except (
            OSError,
            RuntimeError,
            ValueError,
            aiohttp.ClientError,
            asyncio.TimeoutError,
        ) as error:
            self.fail(worker, error)
            await self.stop_worker(worker)
            worker.next_start = time.monotonic() + self.config.get("restart_backoff", 2)

    async def maintain(self, worker, predecessor=None):
        # Sequential initial preparation avoids simultaneous model initialization.
        if predecessor is not None:
            while predecessor.state != "ready":
                await asyncio.sleep(0.1)
        try:
            while True:
                if worker.state in {"starting", "failed"}:
                    await self.stop_worker(worker)
                    await asyncio.sleep(max(0, worker.next_start - time.monotonic()))
                    await self.prepare(worker)
                elif worker.process.returncode is not None:
                    self.fail(worker, f"launcher exited {worker.process.returncode}")
                    worker.next_start = time.monotonic() + self.config.get(
                        "restart_backoff", 2
                    )
                elif time.monotonic() >= worker.next_probe:
                    generation = worker.generation
                    healthy = await self.probe(worker)
                    if (
                        generation == worker.generation
                        and worker.state == "ready"
                        and not healthy
                    ):
                        self.fail(worker, "health probe failed")
                        worker.next_start = time.monotonic() + self.config.get(
                            "restart_backoff", 2
                        )
                    worker.next_probe = time.monotonic() + self.config.get(
                        "health_interval", 1
                    )
                await asyncio.sleep(self.config.get("poll_interval", 0.05))
        finally:
            await self.stop_worker(worker)

    async def start(self, app):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=1),
            auto_decompress=False,
        )
        self.tasks = [
            asyncio.create_task(self.maintain(self.workers[0])),
            asyncio.create_task(self.maintain(self.workers[1], self.workers[0])),
        ]

    async def close(self, app):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        for worker in self.workers:
            await self.stop_worker(worker)
        if self.session is not None:
            await self.session.close()

    async def status(self, request):
        self.choose()
        return web.json_response(
            {
                "active": self.active.name if self.active else None,
                "epoch": self.epoch,
                "standby_ready": sum(w.state == "ready" for w in self.workers) == 2,
                "workers": [
                    {
                        "name": w.name,
                        "state": w.state,
                        "generation": w.generation,
                        "pid": w.process.pid if w.process else None,
                        "ready_at": w.ready_at,
                        "last_error": w.last_error,
                    }
                    for w in self.workers
                ],
                "events": self.events,
            }
        )

    async def health(self, request):
        return web.Response(status=200 if self.choose() else 503)

    async def forward(self, request):
        if request.path not in ROUTES:
            raise web.HTTPNotFound()
        body = await request.read()
        if request.method == "POST":
            try:
                payload = json.loads(body)
            except (ValueError, UnicodeDecodeError):
                raise web.HTTPBadRequest(text="Invalid JSON")
            if not isinstance(payload, dict):
                raise web.HTTPBadRequest(text="Expected a JSON object")
            if "session_params" in payload:
                raise web.HTTPConflict(
                    text="Session state is not recoverable across workers"
                )
        worker = self.choose()
        if worker is None:
            raise web.HTTPServiceUnavailable(text="No warmed worker available")
        generation = worker.generation
        epoch = self.epoch
        outgoing = None
        try:
            async with self.session.request(
                request.method,
                worker.url + request.path_qs,
                data=body,
                headers=forward_headers(request.headers),
                allow_redirects=False,
            ) as response:
                headers = forward_headers(response.headers)
                headers.update(
                    {
                        "X-SGLang-Recovery-Worker": worker.name,
                        "X-SGLang-Recovery-Epoch": str(epoch),
                    }
                )
                outgoing = web.StreamResponse(status=response.status, headers=headers)
                await outgoing.prepare(request)
                async for chunk in response.content.iter_any():
                    await outgoing.write(chunk)
                await outgoing.write_eof()
                return outgoing
        except (aiohttp.ClientError, asyncio.TimeoutError) as error:
            if worker.generation == generation:
                self.fail(worker, error)
            # No retry: the failed worker may already have accepted the request.
            if outgoing is not None and outgoing.prepared:
                if request.transport is not None:
                    request.transport.close()
                return outgoing
            raise web.HTTPServiceUnavailable(
                text="Worker failed; request was not replayed"
            )

    def app(self):
        app = web.Application(
            client_max_size=self.config.get("client_max_size", 16 * 1024 * 1024)
        )
        app.on_startup.append(self.start)
        app.on_cleanup.append(self.close)
        app.router.add_get("/recovery/status", self.status)
        app.router.add_get("/health", self.health)
        app.router.add_route("*", "/{path:.*}", self.forward)
        return app


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    gateway = StandbyGateway(config)
    web.run_app(
        gateway.app(),
        host=config.get("host", "127.0.0.1"),
        port=config.get("port", 22940),
    )


if __name__ == "__main__":
    main()
