# SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
# SPDX-License-Identifier: Apache-2.0

"""Content-based cross-router routing test for cache-aware-zmq.

Two routers + two SGLang workers + one shared model. Each router runs an
independent ``cache_aware_zmq`` policy whose ``KvEventIndex`` subscribes
to **both** workers' KV publishers.

The test warms each worker with a DIFFERENT prefix DIRECTLY (bypassing
both routers), then sends those prefixes through each router and
asserts that routing follows the prefix CONTENT: ``PREFIX_X`` lands on
the worker holding X, ``PREFIX_Y`` lands on the worker holding Y, on
both routers.

# Why content-based, not convergence

An earlier version of this test asserted that both routers converged on
the *same dominant worker* after a one-prefix warmup. That property
sounds like it pins the ZMQ-fan-out contract, but it doesn't: when the
KV-event path is broken (subscribers never opened, e.g. a worker's
``/server_info`` lacks the ``kv_events`` block), ``cache_aware_zmq``
silently degrades to **min-load** — which, with sequential requests
holding ``active_load`` at zero, picks the same worker deterministically
on every call within a router. Both routers' min-load picks happened to
agree often enough (about half the time, modulo HashSet seed) to make
the convergence assertion pass even when no event ever flowed.

Content-based routing is uniquely sensitive to the KV-event path. Two
disjoint prefixes warmed on two different workers can only be routed
correctly if the router knows *which worker holds which content* — the
only mechanism that supplies that information is the ``BlockStored``
event stream. Under min-load fallback, both prefixes route to the same
default worker on each router, so the ``PREFIX_Y → worker_y`` assertion
fails regardless of which worker min-load defaults to.
"""

from __future__ import annotations

import re
import time

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
    r"^sgl_router_requests_total\{([^}]*)\}\s+(\d+(?:\.\d+)?)\s*$"
)
_LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


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
def test_two_routers_route_by_prefix_content(
    router_binary,  # noqa: ARG001 — fixture forces release-binary presence
    gpu_allocator,
):
    """Each router must route by prefix CONTENT, agreeing across routers.

    With each worker direct-warmed by a different disjoint prefix, the
    only way a router can route ``PREFIX_X → worker_x`` AND
    ``PREFIX_Y → worker_y`` is by consulting a HashTree populated from
    the BlockStored events the workers emit. Min-load fallback (the
    failure mode when no SUB socket opened) is content-blind and would
    route both prefixes to whichever worker its tiebreaker prefers.
    """
    spec = get_model_spec("qwen3-0.6b")
    gpus = gpu_allocator.acquire(2)
    # Workers run with the model's REAL chat template (no override): the engine
    # caches chat-templated tokens, and the router renders the same template
    # (loaded from the model's tokenizer_config.json) before hashing. This
    # exercises the production chat-template tokenization path, which aligns
    # router query hashes with the engine's templated blocks.
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

            # 1. Direct-warm each worker with its own prefix. Must happen
            #    AFTER both routers have started — ZMQ PUB/SUB doesn't
            #    replay messages emitted before SUB attaches, so any
            #    BlockStored event predating subscription is lost and
            #    the HashTree never sees it.
            _direct_warm(worker_x.url, spec["model"], PREFIX_X)
            _direct_warm(worker_y.url, spec["model"], PREFIX_Y)

            # 2. Drain the SUB mpsc + pump-apply path. Sub-second under
            #    loopback ZMQ; 2 s leaves comfortable headroom.
            time.sleep(2.0)

            # 3. Content-routing assertion (×4): each prefix must land
            #    on the worker that holds it, on either router.
            #
            #    The four assertions below are independently strong:
            #    min-load fallback routes both prefixes on a given
            #    router to a single default worker, so for ANY broken-
            #    fan-out scenario at least one of the four fails.
            for router, label in ((router_a, "A"), (router_b, "B")):
                landed = _route_through(router.base_url, spec["model"], PREFIX_X)
                assert landed == worker_x.url, (
                    f"router {label}: PREFIX_X must route to worker_x "
                    f"({worker_x.url}); landed on {landed}. "
                    f"Likely cause: HashTree is empty — KV-event "
                    f"subscriber never opened, or BlockStored events "
                    f"never reached the pump."
                )
                landed = _route_through(router.base_url, spec["model"], PREFIX_Y)
                assert landed == worker_y.url, (
                    f"router {label}: PREFIX_Y must route to worker_y "
                    f"({worker_y.url}); landed on {landed}. "
                    f"Likely cause: HashTree is empty — KV-event "
                    f"subscriber never opened, or BlockStored events "
                    f"never reached the pump."
                )
    finally:
        gpu_allocator.release(gpus)
