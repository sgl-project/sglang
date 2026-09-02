# SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
# SPDX-License-Identifier: Apache-2.0

"""Content-based routing test for both cache-aware-zmq index backends.

Two SGLang workers publish KV events to two routers at once: one runs the
local ``KvEventIndex`` (SUB straight to the workers) and one runs against an
external KV Indexer fed by a ``kv-indexer-bridge`` per worker. Every
subscriber attaches before the single warmup, so one pair of disjoint
prefixes exercises both index backends without a second model load.

Assert on content, not on convergence: a broken event path degrades
``cache_aware_zmq`` to content-blind min-load, which routes both prefixes to
one worker and so fails at least one assertion below.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from infra.gateway import Gateway
from infra.model_pool import spawn_worker
from infra.model_specs import get_model_spec

# Disjoint prefixes — share no common content. Under the chat template both
# render with the same leading role header (``<|im_start|>user`` ...; Qwen3 has
# no BOS token), so the first block(s) may hash identically; the disjoint
# content then diverges
# well within the matched region, making each worker's HashTree contribution
# uniquely identifying.
#
# Length matters: each prefix must span ≥2 SGLang blocks at the default
# block_size of 64 tokens so the worker actually emits BlockStored
# events. Below that, the publisher stays quiet and we'd be testing
# min-load by accident — the exact failure mode this test exists to
# rule out.
_PREFIX_X_BODY = (
    "Apricot bouquet cinnamon dewdrop elderflower fennel garlic "
    "hibiscus indigo jasmine kumquat lavender mint nutmeg oregano "
    "paprika quince rosemary saffron tarragon. "
)
PREFIX_X = (_PREFIX_X_BODY * 8).strip()

_PREFIX_Y_BODY = (
    "Zephyr yellow xylophone wombat vortex umbrella thistle saffron "
    "quartz peppermint orchid nightshade marigold lemongrass kale "
    "juniper iris hyacinth gardenia foxglove. "
)
PREFIX_Y = (_PREFIX_Y_BODY * 8).strip()


_REQ_TOTAL_RE = re.compile(
    r"^sgl_router_worker_requests_total\{([^}]*)\}\s+(\d+(?:\.\d+)?)\s*$"
)
_LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


def _open_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def _run(binary: Path, env: dict[str, str], log_path: Path):
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [str(binary)],
            env={**os.environ, **env},
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            yield process
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def _wait_for_indexer(process: subprocess.Popen, port: int, log_path: Path) -> None:
    deadline = time.time() + 10
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"KV Indexer exited during startup:\n{log_path.read_text()}"
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("timed out waiting for KV Indexer")


def _wait_for_bridge(process: subprocess.Popen, log_path: Path) -> None:
    deadline = time.time() + 10
    while time.time() < deadline:
        output = log_path.read_text(errors="replace")
        if "bridge session established" in output:
            # ZMQ connect is asynchronous; let the subscription reach the PUB.
            time.sleep(0.5)
            return
        if process.poll() is not None:
            raise RuntimeError(f"KV Indexer Bridge exited during startup:\n{output}")
        time.sleep(0.1)
    raise RuntimeError(f"timed out waiting for KV Indexer Bridge:\n{output}")


def _dump_logs(logs: dict[str, Path]) -> None:
    """Print the tail of each Indexer/Bridge log so a routing failure is debuggable."""
    for name, path in logs.items():
        tail = path.read_text(errors="replace")[-4000:] if path.exists() else "<no log>"
        print(f"\n----- {name} -----\n{tail}")


def _success_counts_by_worker(router_url: str) -> dict[str, int]:
    """Scrape ``/metrics`` and return ``{worker_url: success_count}``."""
    r = httpx.get(f"{router_url}/metrics", timeout=5.0)
    r.raise_for_status()
    counts: dict[str, int] = {}
    for line in r.text.splitlines():
        m = _REQ_TOTAL_RE.match(line)
        if not m:
            continue
        labels = dict(_LABEL_RE.findall(m.group(1)))
        if labels.get("outcome") != "success":
            continue
        worker = labels.get("worker_url")
        if not worker:
            continue
        try:
            counts[worker] = counts.get(worker, 0) + int(float(m.group(2)))
        except ValueError:
            continue
    return counts


def _send_chat(url: str, model_id: str, prompt: str) -> int:
    """POST one chat completion; return the HTTP status."""
    r = httpx.post(
        f"{url}/v1/chat/completions",
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4,
            "stream": False,
        },
        timeout=60.0,
    )
    return r.status_code


def _direct_warm(worker_url: str, model_id: str, prefix: str) -> None:
    """Send one ``/v1/chat/completions`` request with ``prefix`` DIRECTLY to a worker.

    The KV-event publisher emits ``BlockStored`` as the request's
    prompt blocks commit to that worker's cache; routers subscribed to
    the publisher receive the event and add ``(block_hash → worker)``
    entries to their ``HashTree``. The test then exercises those
    entries by routing through the router.

    Direct-warming (rather than going through a router) is the load-
    bearing detail: routing through a router would itself choose which
    worker to populate, so the two workers' HashTree state would no
    longer be uniquely identifying.

    Token alignment with the router — the workers run with the model's
    real chat template (no override), so the engine caches blocks keyed
    on chat-templated tokens (role markers + content + generation prompt).
    ``cache_aware_zmq`` mirrors this: for a chat request on a model that
    ships a chat template, it renders the same template and tokenizes the
    result before hashing, so warm and route hash the same blocks.
    """
    r = httpx.post(
        f"{worker_url}/v1/chat/completions",
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": prefix}],
            "max_tokens": 4,
            "stream": False,
        },
        timeout=60.0,
    )
    assert (
        r.status_code == 200
    ), f"direct warm to {worker_url} failed: HTTP {r.status_code} {r.text!r}"


def _route_through(router_url: str, model_id: str, prompt: str) -> str:
    """Send one request through ``router_url``; return which worker handled it.

    Computed by diffing the per-worker success-counter on ``/metrics``
    around the call. Asserts exactly one worker absorbed the request
    (no partial counts, no cancellation race).
    """
    before = _success_counts_by_worker(router_url)
    code = _send_chat(router_url, model_id, prompt)
    assert code == 200, f"request to {router_url} failed: HTTP {code}"
    after = _success_counts_by_worker(router_url)
    deltas = {w: after.get(w, 0) - before.get(w, 0) for w in set(after) | set(before)}
    winners = [w for w, d in deltas.items() if d > 0]
    assert (
        len(winners) == 1
    ), f"expected exactly one worker delta on {router_url}, got {deltas}"
    return winners[0]


@pytest.mark.real_gpu
@pytest.mark.slow
def test_routers_route_by_prefix_content(
    router_binary,
    gpu_allocator,
    tmp_path,
):
    """Both the local ZMQ index and the external Indexer must route by content."""
    spec = get_model_spec("qwen3-0.6b")
    gpus = gpu_allocator.acquire(2)
    indexer_port = _open_port()
    indexer_endpoint = f"http://127.0.0.1:{indexer_port}"
    indexer_binary = router_binary.parent / "kv-indexer-server"
    bridge_binary = router_binary.parent / "kv-indexer-bridge"
    logs = {
        name: tmp_path / f"{name}.log" for name in ("indexer", "bridge-x", "bridge-y")
    }
    try:
        with (
            spawn_worker(
                "qwen3-0.6b",
                gpu_ids=[gpus[0]],
                enable_kv_events=True,
            ) as worker_x,
            spawn_worker(
                "qwen3-0.6b",
                gpu_ids=[gpus[1]],
                enable_kv_events=True,
            ) as worker_y,
            _run(
                indexer_binary,
                {"KV_INDEXER_LISTEN_ADDR": f"127.0.0.1:{indexer_port}"},
                logs["indexer"],
            ) as indexer,
        ):
            _wait_for_indexer(indexer, indexer_port, logs["indexer"])
            worker_urls = [worker_x.url, worker_y.url]

            def bridge_env(worker, worker_id: str) -> dict[str, str]:
                assert worker.kv_events_endpoint is not None
                return {
                    "KV_INDEXER_WORKER_ID": worker_id,
                    "KV_INDEXER_WORKER_ADDRESS": worker.url,
                    "KV_INDEXER_ENDPOINT": indexer_endpoint,
                    "SGLANG_KV_EVENT_ENDPOINT": worker.kv_events_endpoint.replace(
                        "*", "127.0.0.1"
                    ),
                    "SGLANG_KV_EVENT_TOPIC": "kv",
                }

            with (
                _run(
                    bridge_binary, bridge_env(worker_x, "worker-x"), logs["bridge-x"]
                ) as bridge_x,
                _run(
                    bridge_binary, bridge_env(worker_y, "worker-y"), logs["bridge-y"]
                ) as bridge_y,
                Gateway() as local,
                Gateway() as external,
            ):
                local.start_regular(
                    model_id=spec["model"],
                    tokenizer_path=spec["model"],
                    worker_urls=worker_urls,
                    policy="cache_aware_zmq",
                    timeout=120.0,
                )
                external.start_regular(
                    model_id=spec["model"],
                    tokenizer_path=spec["model"],
                    worker_urls=worker_urls,
                    policy="cache_aware_zmq",
                    kv_indexer_endpoint=indexer_endpoint,
                    timeout=120.0,
                )

                _wait_for_bridge(bridge_x, logs["bridge-x"])
                _wait_for_bridge(bridge_y, logs["bridge-y"])

                _direct_warm(worker_x.url, spec["model"], PREFIX_X)
                _direct_warm(worker_y.url, spec["model"], PREFIX_Y)
                time.sleep(2.0)

                try:
                    for router, label in (
                        (local, "local-index"),
                        (external, "external-indexer"),
                    ):
                        landed = _route_through(
                            router.base_url, spec["model"], PREFIX_X
                        )
                        assert (
                            landed == worker_x.url
                        ), f"router {label}: PREFIX_X must route to {worker_x.url}; landed on {landed}"
                        landed = _route_through(
                            router.base_url, spec["model"], PREFIX_Y
                        )
                        assert (
                            landed == worker_y.url
                        ), f"router {label}: PREFIX_Y must route to {worker_y.url}; landed on {landed}"
                except Exception:
                    _dump_logs(logs)
                    raise
    finally:
        gpu_allocator.release(gpus)
