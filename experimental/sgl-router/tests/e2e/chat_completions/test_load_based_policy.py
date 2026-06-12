# SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
# SPDX-License-Identifier: Apache-2.0

"""Real-GPU acceptance coverage for ``policy = "load_based"``."""

from __future__ import annotations

import re
import threading
import time

import httpx
import pytest
from infra.gateway import Gateway
from infra.model_pool import spawn_worker
from infra.model_specs import get_model_spec

_ACTIVE_RE = re.compile(
    r'^sgl_router_active_load\{worker_url="([^"]+)",kind="prefill_tokens"\}\s+(-?\d+)'
)
_REQ_TOTAL_RE = re.compile(
    r"^sgl_router_requests_total\{([^}]*)\}\s+(\d+(?:\.\d+)?)\s*$"
)
_LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


def _chat(
    router_url: str, model_id: str, prompt: str, max_tokens: int
) -> httpx.Response:
    return httpx.post(
        f"{router_url}/v1/chat/completions",
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "ignore_eos": True,
            "stream": False,
        },
        timeout=240.0,
    )


def _active_prefill_loads(router_url: str) -> dict[str, int]:
    resp = httpx.get(f"{router_url}/metrics", timeout=5.0)
    resp.raise_for_status()
    loads: dict[str, int] = {}
    for line in resp.text.splitlines():
        match = _ACTIVE_RE.match(line)
        if match:
            loads[match.group(1)] = int(match.group(2))
    return loads


def _success_counts(router_url: str) -> dict[str, int]:
    resp = httpx.get(f"{router_url}/metrics", timeout=5.0)
    resp.raise_for_status()
    counts: dict[str, int] = {}
    for line in resp.text.splitlines():
        match = _REQ_TOTAL_RE.match(line)
        if not match:
            continue
        labels = dict(_LABEL_RE.findall(match.group(1)))
        if labels.get("outcome") != "success":
            continue
        worker_url = labels.get("worker_url")
        if worker_url:
            counts[worker_url] = counts.get(worker_url, 0) + int(float(match.group(2)))
    return counts


def _wait_for_busy_worker(router_url: str, worker_urls: list[str]) -> str:
    deadline = time.time() + 60.0
    while time.time() < deadline:
        loads = _active_prefill_loads(router_url)
        busy = [url for url in worker_urls if loads.get(url, 0) > 0]
        if busy:
            return busy[0]
        time.sleep(0.2)
    raise AssertionError(
        f"no worker became busy; last loads={_active_prefill_loads(router_url)}"
    )


def _single_success_delta(before: dict[str, int], after: dict[str, int]) -> str:
    deltas = {
        url: after.get(url, 0) - before.get(url, 0) for url in set(before) | set(after)
    }
    winners = [url for url, delta in deltas.items() if delta == 1]
    assert len(winners) == 1, f"expected exactly one success delta, got {deltas}"
    return winners[0]


@pytest.mark.real_gpu
@pytest.mark.slow
def test_load_based_routes_to_the_cooler_worker(
    router_binary,  # noqa: ARG001 - fixture forces release-binary presence
    gpu_allocator,
) -> None:
    spec = get_model_spec("qwen3-0.6b")
    gpus = gpu_allocator.acquire(2)
    try:
        with (
            spawn_worker("qwen3-0.6b", gpu_ids=[gpus[0]]) as worker_a,
            spawn_worker("qwen3-0.6b", gpu_ids=[gpus[1]]) as worker_b,
            Gateway() as router,
        ):
            worker_urls = [worker_a.url, worker_b.url]
            router.start_regular(
                model_id=spec["model"],
                tokenizer_path=spec["model"],
                worker_urls=worker_urls,
                policy="load_based",
                timeout=120.0,
            )

            errors: list[BaseException] = []

            def long_request() -> None:
                try:
                    resp = _chat(
                        router.base_url,
                        spec["model"],
                        "Write a long numbered list of routing test facts.",
                        1024,
                    )
                    assert resp.status_code == 200, resp.text
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            thread = threading.Thread(target=long_request, daemon=True)
            thread.start()
            busy_worker = _wait_for_busy_worker(router.base_url, worker_urls)

            before = _success_counts(router.base_url)
            resp = _chat(
                router.base_url,
                spec["model"],
                "Answer with one short sentence.",
                8,
            )
            assert resp.status_code == 200, resp.text
            routed_worker = _single_success_delta(
                before, _success_counts(router.base_url)
            )
            assert routed_worker != busy_worker

            thread.join(timeout=240.0)
            assert not thread.is_alive(), "long request did not finish"
            assert not errors, errors
    finally:
        gpu_allocator.release(gpus)
