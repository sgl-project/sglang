# SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
# SPDX-License-Identifier: Apache-2.0

"""HA-shaped convergence test for cache-aware-zmq.

Two routers + two SGLang workers + one shared model. Each router runs an
independent ``cache_aware_zmq`` policy whose ``KvEventIndex`` subscribes
to **both** workers' KV publishers. The test drives an overlapping-prefix
workload through router A first, lets the KV events propagate, then
drives the same prefix through router B. Both routers' ``/metrics``
endpoint must show the *same* dominant worker — the one whose KV cache
holds the warm prefix.

This pins the v1 design assumption that two routers can reach a
consistent cache-aware routing decision without any cross-router state
sync (which is deferred to v2 per the slim-design spec). The only
mechanism enforcing the agreement is ZMQ PUB/SUB fan-out: both routers'
subscribers receive every BlockStored event from every worker, so their
HashTrees converge on the same `(seq_hash → worker_set)` mapping.
"""

from __future__ import annotations

import re
import time

import httpx
import pytest
from infra.gateway import Gateway
from infra.model_pool import spawn_worker
from infra.model_specs import get_model_spec

# Long enough to span multiple SGLang blocks at the default block_size
# (64 tokens). Below that, no BlockStored event is emitted and the
# cache-aware policy degrades to round-robin — which would defeat the
# convergence assertion below.
_PREFIX_BODY = (
    "The slim-router project replaces the legacy gateway with a focused "
    "Rust binary whose only job is to route OpenAI-compatible chat "
    "completions to a pool of SGLang workers. Cache-aware routing "
    "depends on per-worker KV-event publishers feeding a hash-keyed "
    "radix tree. "
)
WARM_PREFIX = (_PREFIX_BODY * 8).strip()

# How many requests to drive through each router. Needs to be enough
# that the cache-aware fast-path picks up the warm worker on every
# subsequent request, and that the /metrics convergence ratio passes
# the threshold below even if the first 1–2 requests round-robin.
N_REQUESTS_PER_ROUTER = 10

# Each router's dominant worker must absorb ≥ this fraction of its
# successful requests. With N=10 and a 1–2 request warm-up tail, 0.8
# leaves headroom for one stray pick before convergence.
CONVERGENCE_RATIO = 0.8


def _send_chat(router_url: str, model_id: str, suffix: str) -> int:
    """Send one chat completion; return HTTP status code."""
    r = httpx.post(
        f"{router_url}/v1/chat/completions",
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": f"{WARM_PREFIX} {suffix}"}],
            "max_tokens": 4,
            "stream": False,
        },
        timeout=60.0,
    )
    return r.status_code


_REQ_TOTAL_RE = re.compile(
    r"^sgl_router_requests_total\{([^}]*)\}\s+(\d+(?:\.\d+)?)\s*$"
)
_LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


def _success_counts_by_worker(router_url: str) -> dict[str, int]:
    """Scrape ``/metrics`` and return ``{worker_url: success_count}``.

    Only the ``outcome="success"`` slice is counted — error/cancelled
    requests don't tell us anything about routing convergence.
    """
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


@pytest.mark.real_gpu
@pytest.mark.slow
def test_two_routers_cache_aware_converge_to_same_worker(
    router_binary,  # noqa: ARG001 — fixture forces release-binary presence
    gpu_allocator,
):
    """Two routers, two workers, identical cache-aware-zmq policy.

    With both routers' subscribers seeing the same KV-event stream from
    both workers, an overlapping-prefix workload through one router
    must steer subsequent traffic through the OTHER router to the same
    worker. Validates the ZMQ-fan-out HA contract end-to-end.
    """
    spec = get_model_spec("qwen3-0.6b")
    gpus = gpu_allocator.acquire(2)
    try:
        with (
            spawn_worker(
                "qwen3-0.6b", gpu_ids=[gpus[0]], enable_kv_events=True
            ) as worker_x,
            spawn_worker(
                "qwen3-0.6b", gpu_ids=[gpus[1]], enable_kv_events=True
            ) as worker_y,
            Gateway() as router_a,
            Gateway() as router_b,
        ):
            worker_urls = [worker_x.url, worker_y.url]
            for gw in (router_a, router_b):
                gw.start_regular(
                    model_id=spec["model"],
                    tokenizer_path=spec["model"],
                    worker_urls=worker_urls,
                    policy="cache_aware_zmq",
                    timeout=120.0,
                )

            # 1. Warm up via router A. The first 1–2 requests distribute
            #    round-robin (the tree is empty); thereafter the
            #    cache-aware policy should stick to whichever worker
            #    absorbed the first traffic, because its KV cache now
            #    holds the prefix and the publisher emitted BlockStored
            #    events for those blocks.
            for i in range(N_REQUESTS_PER_ROUTER):
                code = _send_chat(router_a.base_url, spec["model"], f"a-{i}")
                assert code == 200, f"router A request {i} failed: {code}"

            # Snapshot router A's dominant worker BEFORE any traffic
            # touches router B. The first-request assertion below uses
            # this as ground truth — without snapshotting, the test
            # would be testing "do both routers agree" (PROPERTY 1
            # later), which can pass even if KV fan-out is broken (e.g.
            # both routers independently round-robin to worker_x first).
            counts_a_warm = _success_counts_by_worker(router_a.base_url)
            assert counts_a_warm, "router A produced no metrics after warmup"
            chosen_a_warm = max(counts_a_warm, key=counts_a_warm.get)

            # 2. Give router B's KvEventIndex time to drain its SUB
            #    queue + apply the events router A's traffic produced.
            #    Loopback ZMQ + the subscriber mpsc are sub-second
            #    under normal conditions; a 2s settle is generous.
            time.sleep(2.0)

            # 3. PROPERTY 0 — first-request convergence: router B's very
            #    first request must land on the worker router A warmed.
            #    cache_aware_zmq has nothing else to base this on
            #    except the BlockStored events fanned out from worker A
            #    to router B's subscriber. If ZMQ fan-out is broken
            #    (single-subscriber socket, drop on backpressure,
            #    silent subscribe failure, etc.), this fails — whereas
            #    PROPERTY 1/2 below could still pass if router B
            #    happened to round-robin its own traffic to the same
            #    worker independently.
            code = _send_chat(router_b.base_url, spec["model"], "b-0")
            assert code == 200, f"router B first request failed: {code}"
            counts_b_first = _success_counts_by_worker(router_b.base_url)
            assert (
                counts_b_first
            ), "router B produced no metrics after its first request"
            chosen_b_first = max(counts_b_first, key=counts_b_first.get)
            assert chosen_b_first == chosen_a_warm, (
                "router B's first request landed on a different worker "
                "than router A warmed — KV-event fan-out is broken or "
                "subscribers are not converging. "
                f"chosen_a_warm={chosen_a_warm}, "
                f"chosen_b_first={chosen_b_first}, "
                f"counts_b_first={counts_b_first}"
            )

            # 4. Drive the remaining requests through router B. The
            #    bulk assertions below exercise the steady-state
            #    behaviour (1 stray pick out of N is tolerated).
            for i in range(1, N_REQUESTS_PER_ROUTER):
                code = _send_chat(router_b.base_url, spec["model"], f"b-{i}")
                assert code == 200, f"router B request {i} failed: {code}"

            # 4. Scrape /metrics on both routers. Each router only
            #    counts the requests *it* dispatched, so router A's
            #    dominant worker is computed independently of router B's.
            counts_a = _success_counts_by_worker(router_a.base_url)
            counts_b = _success_counts_by_worker(router_b.base_url)
            assert counts_a, (
                f"router A produced no successful request samples; "
                f"raw counts: {counts_a}"
            )
            assert counts_b, (
                f"router B produced no successful request samples; "
                f"raw counts: {counts_b}"
            )

            chosen_a = max(counts_a, key=counts_a.get)
            chosen_b = max(counts_b, key=counts_b.get)
            total_a = sum(counts_a.values())
            total_b = sum(counts_b.values())

            # PROPERTY 1: both routers agree on the dominant worker.
            assert chosen_a == chosen_b, (
                "routers disagreed on dominant worker — KV-event "
                f"convergence broken. router_a→{chosen_a} "
                f"(counts {counts_a}); router_b→{chosen_b} "
                f"(counts {counts_b})"
            )

            # PROPERTY 2: that worker absorbed the bulk of each router's
            # traffic. The threshold is loose enough to tolerate the
            # initial round-robin tail but tight enough to fail loudly
            # if cache-aware degenerates to load-balanced selection.
            ratio_a = counts_a[chosen_a] / total_a
            ratio_b = counts_b[chosen_b] / total_b
            assert ratio_a >= CONVERGENCE_RATIO, (
                f"router A did not converge: {counts_a[chosen_a]}/{total_a} "
                f"= {ratio_a:.2%} on {chosen_a} (threshold {CONVERGENCE_RATIO:.0%})"
            )
            assert ratio_b >= CONVERGENCE_RATIO, (
                f"router B did not converge: {counts_b[chosen_b]}/{total_b} "
                f"= {ratio_b:.2%} on {chosen_b} (threshold {CONVERGENCE_RATIO:.0%})"
            )
    finally:
        gpu_allocator.release(gpus)
