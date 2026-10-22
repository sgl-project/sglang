# SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
# SPDX-License-Identifier: Apache-2.0

"""Real-model coverage for the Tair KVCache backend sitting behind the router.

Every process of the deployment shape is real here (``infra.kv_indexer``
starts the stack): two engines publish their event streams, one bridge per
engine forwards them, the indexer serves them through the Tair KVCache
backend, and the router routes on what the indexer answers.

That makes a routing hit a claim about Tair KVCache rather than about a cache
in front of it: the backend keeps no placements of its own — every prefix
answer is a ``GetHostCacheState`` RPC to the manager — and
``--kv-indexer-endpoint`` leaves the router with no KV tree of its own, so
who serves the repeat can only have been decided by what the manager holds.

Which engine owns the prefix is read off the router's own per-worker dispatch
counter rather than assumed, so the test does not depend on which engine the
first request picked.
"""

from __future__ import annotations

import os
import re
import time
from contextlib import ExitStack

import httpx
import pytest
from infra.gateway import Gateway
from infra.kv_indexer import (
    Bridge,
    IndexerService,
    KvcmManager,
    missing_stack_binaries,
    stack_logs,
)
from infra.model_pool import spawn_worker
from infra.model_specs import get_model_spec

MODEL_ID = "qwen3-0.6b"

# Long enough to span several blocks, so "the owner" is a claim about a range
# of the prefix rather than about a single block, and short enough that two
# engines prefill it in well under a second.
_PROMPT_SENTENCE = (
    "A prefix owner is only worth remembering if the memory of it outlives "
    "the process that learned it. "
)
PROMPT = _PROMPT_SENTENCE * 100

# Repeats asserted after the prefix is owned. Each one is an independent
# routing decision against the same fact, so requiring all of them turns a
# missing fact (a coin flip back to min-load) into a 1-in-2^REPEATS accident
# rather than a coin flip the test might win.
REPEATS = 5

_METRIC_RE = re.compile(r"^(\w+)\{([^}]*)\}\s+(-?\d+(?:\.\d+)?)\s*$")
_BARE_METRIC_RE = re.compile(r"^(\w+)\s+(-?\d+(?:\.\d+)?)\s*$")
_LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


def _scrape(router_url: str) -> str:
    resp = httpx.get(f"{router_url}/metrics", timeout=10.0)
    resp.raise_for_status()
    return resp.text


def _samples(text: str, name: str) -> list[tuple[dict[str, str], float]]:
    """Every sample of `name`, as (labels, value)."""
    out: list[tuple[dict[str, str], float]] = []
    for line in text.splitlines():
        if not line.startswith(name) or line.startswith("#"):
            continue
        match = _METRIC_RE.match(line)
        if match and match.group(1) == name:
            out.append((dict(_LABEL_RE.findall(match.group(2))), float(match.group(3))))
            continue
        bare = _BARE_METRIC_RE.match(line)
        if bare and bare.group(1) == name:
            out.append(({}, float(bare.group(2))))
    return out


def _sum_where(text: str, name: str, **labels: str) -> float:
    total = 0.0
    for got, value in _samples(text, name):
        if all(got.get(k) == v for k, v in labels.items()):
            total += value
    return total


def _cache_hits(text: str) -> float:
    """Decisions where the policy pinned a prefix owner instead of falling back."""
    return _sum_where(
        text, "sgl_router_cache_aware_decisions_total", decision="cache_hit"
    )


def _success_counts(text: str) -> dict[str, int]:
    """Successful dispatches per worker, from one already-fetched scrape."""
    counts: dict[str, int] = {}
    for labels, value in _samples(text, "sgl_router_worker_requests_total"):
        if labels.get("outcome") != "success":
            continue
        url = labels.get("worker_url")
        if url:
            counts[url] = counts.get(url, 0) + int(value)
    return counts


def _wait_until(predicate, *, timeout: float, what: str):
    """Poll `predicate` until it returns a truthy value.

    Everything here travels asynchronously — engine to ZMQ, bridge to indexer,
    indexer to manager — so assertions about it have to tolerate a lag rather
    than read once and hope.
    """
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(0.5)
    raise AssertionError(f"timed out waiting for {what}; last observed: {last}")


def _chat(router_url: str, model_id: str, prompt: str, max_tokens: int = 8) -> None:
    resp = httpx.post(
        f"{router_url}/v1/chat/completions",
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        },
        timeout=240.0,
    )
    assert resp.status_code == 200, resp.text


def _route_once(
    router_url: str, model_id: str, worker_urls: list[str]
) -> tuple[str, float]:
    """Send one request; report who served it and whether a prefix owner was pinned.

    The cache-hit counter is what makes the assertion deterministic: it is
    booked when the policy found an owner and pinned it, which with
    ``--kv-indexer-endpoint`` can only come from the indexer's answer. The
    dispatch counter alone would leave a miss to the power-of-two fallback,
    i.e. a coin flip the test could win by accident.

    Both are booked as the response completes, which trails the client's own
    completion, so they are waited on rather than scraped once.
    """
    before = _scrape(router_url)
    _chat(router_url, model_id, PROMPT)

    def _settled() -> tuple[str, float] | None:
        after = _scrape(router_url)
        counts_before, counts_after = _success_counts(before), _success_counts(after)
        moved = [
            url
            for url in worker_urls
            if counts_after.get(url, 0) - counts_before.get(url, 0) >= 1
        ]
        if len(moved) > 1:
            raise AssertionError(
                f"one request was dispatched to several workers: {moved}"
            )
        if len(moved) != 1:
            return None
        return moved[0], _cache_hits(after) - _cache_hits(before)

    return _wait_until(
        _settled, timeout=30.0, what="the dispatch to be counted against a worker"
    )


def _in_ci() -> bool:
    """``SGLANG_IS_IN_CI`` is an ``EnvBool``, so its false spellings are false."""
    return os.environ.get("SGLANG_IS_IN_CI", "").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
        "n",
    )


def _worker_health(text: str) -> dict[str, float]:
    return {
        labels["worker_url"]: value
        for labels, value in _samples(text, "sgl_router_worker_health")
        if "worker_url" in labels
    }


def _block_size(worker_url: str) -> int:
    """The block size the router hashes with, as the engine reports it.

    ``kv_events.block_size`` is the contract the router consumes (it folds in
    the DCP factor), and it is absent when the engine publishes no events —
    which is exactly the misconfiguration this guard is here to catch.
    """
    info = httpx.get(f"{worker_url}/server_info", timeout=30.0).json()
    return int(info["kv_events"]["block_size"])


@pytest.mark.real_gpu
@pytest.mark.slow
def test_repeat_returns_to_the_owner_through_tair_kvcache(
    router_binary,  # noqa: ARG001 - fixture forces release-binary presence
    gpu_allocator,
    tmp_path,
) -> None:
    missing = missing_stack_binaries()
    if missing:
        # A missing stack in CI means the artifact the job downloads is gone,
        # which is a wiring bug rather than a reason to skip quietly.
        reason = "KV Indexer stack not available: " + ", ".join(missing)
        if _in_ci():
            pytest.fail(reason)
        pytest.skip(reason)

    spec = get_model_spec(MODEL_ID)
    gpus = gpu_allocator.acquire(2)
    try:
        with ExitStack() as stack:
            manager = stack.enter_context(KvcmManager(workdir=tmp_path / "kvcm"))
            # Engine and manager have to key blocks the same way, or the facts
            # one reports and the queries the other answers describe different
            # blocks and never meet.
            block_size = manager.env["TAIR_KVCACHE_LIVE_BLOCK_SIZE"]
            worker_args = ["--page-size", block_size]

            worker_a = stack.enter_context(
                spawn_worker(
                    MODEL_ID,
                    gpu_ids=[gpus[0]],
                    enable_kv_events=True,
                    extra_args=worker_args,
                )
            )
            worker_b = stack.enter_context(
                spawn_worker(
                    MODEL_ID,
                    gpu_ids=[gpus[1]],
                    enable_kv_events=True,
                    extra_args=worker_args,
                )
            )
            worker_urls = [worker_a.url, worker_b.url]

            primary = stack.enter_context(
                IndexerService(kvcm_env=manager.env, log_dir=tmp_path)
            )
            bridges = [
                stack.enter_context(
                    Bridge(
                        worker_url=worker.url,
                        kv_events_endpoint=worker.kv_events_endpoint,
                        indexer_endpoint=primary.endpoint,
                        log_dir=tmp_path,
                    )
                )
                for worker in (worker_a, worker_b)
            ]

            # The engine may derive its page size from the model rather than
            # take ours; a silent mismatch would make every query miss for a
            # reason that has nothing to do with this backend.
            for worker in (worker_a, worker_b):
                reported = _block_size(worker.url)
                assert reported == int(block_size), (
                    f"{worker.url} reports block_size={reported}, but the "
                    f"manager's instance is keyed at {block_size}; the events and "
                    f"the router's hashes would describe different blocks"
                )

            with Gateway() as router:
                router.start_regular(
                    model_id=spec["model"],
                    tokenizer_path=spec["model"],
                    worker_urls=worker_urls,
                    policy="cache_aware",
                    kv_indexer_endpoint=primary.endpoint,
                    timeout=120.0,
                )

                # Both engines have to be live, not just registered: with one
                # of them down every request trivially goes to the survivor and
                # "the repeat came back" proves nothing. The direct request
                # below is what makes the second engine's liveness a fact the
                # test checked rather than one it assumed.
                def _both_live() -> bool:
                    health = _worker_health(_scrape(router.base_url))
                    return set(health) == set(worker_urls) and all(
                        value > 0 for value in health.values()
                    )

                _wait_until(
                    _both_live,
                    timeout=120.0,
                    what="both engines to register with the router and report healthy",
                )
                httpx.post(
                    f"{worker_urls[1]}/generate",
                    json={"text": "liveness", "max_new_tokens": 1},
                    timeout=120.0,
                ).raise_for_status()

                # The priming request is the only one that may miss: nothing is
                # in the manager yet. Its landing is what makes an engine the
                # owner, so it is read rather than assumed.
                owner, _ = _route_once(router.base_url, spec["model"], worker_urls)

                for _ in range(REPEATS):
                    landed, pinned = _route_once(
                        router.base_url, spec["model"], worker_urls
                    )
                    assert pinned >= 1, (
                        "the router dispatched without pinning a prefix owner, so the "
                        "indexer answered this prefix with no match.\n"
                        f"{stack_logs(manager, primary, *bridges)}"
                    )
                    assert landed == owner, (
                        f"a repeat of the prefix primed on {owner} was dispatched "
                        f"to {landed}. The backend answers prefix queries with a "
                        f"GetHostCacheState RPC and holds no placements of its own, "
                        f"so a miss here is the manager's answer.\n"
                        f"{stack_logs(manager, primary, *bridges)}"
                    )

                # The router has no KV tree here — the series is not exported
                # at all — so every repeat above was decided by the indexer's
                # answer rather than by a router-side copy of the same fact.
                assert not _samples(
                    _scrape(router.base_url), "sgl_router_kv_tree_blocks"
                ), (
                    "the router kept KV state of its own while pointed at an "
                    "external indexer, so this test would pass without the stack"
                )
    finally:
        gpu_allocator.release(gpus)
