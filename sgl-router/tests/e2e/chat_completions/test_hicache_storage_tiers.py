# SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
# SPDX-License-Identifier: Apache-2.0

"""Real-GPU coverage for storage-tier-aware cache routing (hicache L2).

An engine running ``--enable-hierarchical-cache`` publishes each KV-cache
tier transition as its own event, tagged with a ``medium``: a ``CPU_PINNED``
``BlockStored`` when a block's host backup lands, a ``GPU`` ``BlockRemoved``
when its device copy is evicted. The router's radix tree used to apply every
``BlockRemoved`` as a full removal, so a worker lost ownership of a prefix the
instant its device copy went — even though the host tier still held it and
could load it back at memory speed.

This is the end-to-end half of that fix. Every other test of the tier logic
synthesises the event sequence; here a real engine produces it, which is the
only way to catch the tag going missing anywhere along
``hiradix_cache -> kv_events -> ZMQ -> subscriber -> pump -> tree``, and the
only way to check the payoff — that a repeat is still served from the tier
that holds it — rather than just the tree state behind it.

Two workers, so the routing assertion has somewhere else to go wrong: with a
tier-blind tree the owner is forgotten the moment its device copy is evicted
and the repeat falls through to min-load, a coin flip. Asserting the tree
state alone would need only one worker, but it is a strictly weaker claim and
this test establishes it on the way (see ``_sole_device_owner``).

The payoff is read off the engine's own per-tier accounting rather than off
the tree: ``return_cached_tokens_details`` makes each response carry
``sglext.cached_tokens_details``, the split of the cached prompt tokens by the
tier that served them. ``host > 0`` is the claim in the test's name, and it is
the form a tier-blind tree cannot also satisfy — while a prefix is still on
device, forgetting the host tier costs nothing and the repeat comes home
anyway, so a routing-only assertion passes either way. The tree metrics stay
in the assertions because the router seeing the tier stream is what this file
uniquely covers; the engine's split is what makes the check non-vacuous.

The engine is launched with a deliberately tiny device KV pool
(``--max-total-tokens``) so a handful of requests forces real device eviction
in seconds, while ``--hicache-ratio`` keeps the host pool large enough that
those blocks are retained on L2 rather than dropped outright. That is the
whole point: the device tier must turn over while the host tier does not.

Both write policies are covered, because they publish the SAME two events in
OPPOSITE orders and the tree has to converge either way:

* ``write_through`` — the pending D2H copy holds a lock ref that blocks
  eviction, so the ``CPU_PINNED`` store is published first and the ``GPU``
  removal second. The worker is never not-an-owner.
* ``write_back`` — ``_detach_backuped`` publishes the ``GPU`` removal as soon
  as host slots are reserved, and the ``CPU_PINNED`` store follows only when
  the copy actually lands. The worker is briefly dropped from the chain (the
  node may even be pruned) and the later store re-adds it.

A tree that only handled the write_through order would look correct in every
steady-state check and still lose the prefix on a write_back fleet.
"""

from __future__ import annotations

import re
import time

import httpx
import pytest
from infra.gateway import Gateway
from infra.model_pool import spawn_worker
from infra.model_specs import get_model_spec

# Device KV pool, in tokens. Small enough that the filler prompts below evict
# the primed prefix within seconds, large enough to hold several requests at
# once so nothing wedges on admission.
DEVICE_KV_TOKENS = 8192
# Tokens per block hash. The router keys its tree on page-sized blocks, so a
# page of 1 would make the tree one node per token.
PAGE_SIZE = 64
# Host pool as a multiple of the device pool. Everything evicted from device
# during a test must still fit on host, or the engine drops it and there is no
# host tier left to observe.
HICACHE_RATIO = 4

_METRIC_RE = re.compile(r"^(\w+)\{([^}]*)\}\s+(-?\d+(?:\.\d+)?)\s*$")
_BARE_METRIC_RE = re.compile(r"^(\w+)\s+(-?\d+(?:\.\d+)?)\s*$")
_LABEL_RE = re.compile(r'(\w+)="([^"]*)"')

# The two orders in which an engine can publish a backup + eviction pair.
WRITE_POLICIES = ["write_through", "write_back"]


def worker_args(write_policy: str) -> list[str]:
    return [
        "--enable-hierarchical-cache",
        "--hicache-ratio",
        str(HICACHE_RATIO),
        "--hicache-write-policy",
        write_policy,
        "--max-total-tokens",
        str(DEVICE_KV_TOKENS),
        "--page-size",
        str(PAGE_SIZE),
    ]


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


def _events(text: str, event: str, medium: str) -> float:
    return _sum_where(text, "sgl_router_kv_events_total", event=event, medium=medium)


def _tree_blocks(text: str, tier: str, worker_url: str | None = None) -> float:
    labels = {"tier": tier}
    if worker_url is not None:
        labels["worker_url"] = worker_url
    return _sum_where(text, "sgl_router_kv_tree_blocks", **labels)


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


def _registered_workers(text: str) -> set[str]:
    """Worker URLs the router has registered, from one scrape."""
    return {
        labels["worker_url"]
        for labels, _ in _samples(text, "sgl_router_worker_health")
        if "worker_url" in labels
    }


def _tier_summary(text: str) -> str:
    """One-line view of the tier state, logged before each assertion so a
    failure (and a pass) shows the numbers it was judged on rather than only
    the predicate that tripped."""
    tiers = {
        tier: _tree_blocks(text, tier)
        for tier in ("device", "host", "disk", "external")
    }
    events = {
        f"{ev}/{med}": _events(text, ev, med)
        for ev in ("block_stored", "block_removed")
        for med in ("GPU", "CPU_PINNED")
    }
    lost = _sum_where(text, "sgl_router_kv_event_batches_lost_total")
    errs = _sum_where(text, "sgl_router_kv_tree_accounting_errors_total")
    return (
        f"tree_blocks={tiers} events={events} "
        f"batches_lost={lost} accounting_errors={errs}"
    )


def _chat(router_url: str, model_id: str, prompt: str, max_tokens: int = 8) -> dict:
    """Send one non-streaming chat request and return the parsed body.

    ``return_cached_tokens_details`` asks the engine for the per-tier split of
    the prompt tokens it served from cache. It survives the trip in both
    directions: the router forwards the request body rather than reserializing
    it from a typed struct, and proxies a non-streaming response verbatim.
    """
    resp = httpx.post(
        f"{router_url}/v1/chat/completions",
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
            "return_cached_tokens_details": True,
        },
        timeout=240.0,
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _cached_tiers(body: dict) -> dict[str, int]:
    """The engine's split of the cached prompt tokens by serving tier, as
    ``{"device": N, "host": M}``.

    Empty when the prompt hit nothing — the engine omits the block entirely
    rather than reporting zeroes, so "absent" and "no cache hit" are the same
    observation here. The non-integer members (``storage_backend``) are
    dropped so callers can compare values without type-checking each one.
    """
    details = (body.get("sglext") or {}).get("cached_tokens_details") or {}
    return {k: v for k, v in details.items() if isinstance(v, int)}


# Words per prompt. Each `tag<i>` word costs several tokens. Measured against
# the Qwen3 tokenizer this lands at ~1.1k tokens for the primed prompt and
# ~2.3k for a filler — the tags differ in how they split, so do not assume one
# figure covers both (FILLERS_PER_PROBE is sized on the filler). Both are
# comfortably under the context ceiling that --max-total-tokens implies (the
# engine rejects anything longer), and small enough that a handful of prompts
# turns the device pool over in stages rather than in one step.
PROMPT_WORDS = 300


def _long_prompt(tag: str) -> str:
    """A prompt with a distinct leading token, so two prompts built with
    different tags share no block hash and cannot be confused for cache hits
    of one another."""
    filler = " ".join(f"{tag}{i}" for i in range(PROMPT_WORDS))
    return f"Context {tag}: {filler}\nReply with one word."


def _wait_until(predicate, *, timeout: float, what: str):
    """Poll `predicate` until it returns a truthy value. KV events are
    asynchronous (engine -> ZMQ -> pump), so every assertion about them has to
    tolerate a lag rather than read once and hope."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(0.5)
    raise AssertionError(f"timed out waiting for {what}; last observed: {last}")


def _chat_and_attribute(
    router_url: str,
    model_id: str,
    prompt: str,
    worker_urls: list[str],
) -> tuple[dict[str, int], dict[str, int]]:
    """Send one request and report where it landed and what served it: the
    per-worker change in successful dispatches, and the response's per-tier
    cached-token split.

    The counter is booked as the router finishes the response, which can trail
    the client's own completion, so wait for the dispatch to be attributed
    rather than scraping once and reading zeroes everywhere.
    """
    before = _success_counts(_scrape(router_url))
    body = _chat(router_url, model_id, prompt)

    def _deltas() -> dict[str, int] | None:
        after = _success_counts(_scrape(router_url))
        deltas = {url: after.get(url, 0) - before.get(url, 0) for url in worker_urls}
        return deltas if sum(deltas.values()) >= 1 else None

    deltas = _wait_until(
        _deltas, timeout=30.0, what="the dispatch to be counted against a worker"
    )
    return deltas, _cached_tiers(body)


# Filler requests between two probes of the primed prefix. A probe prefills
# that prefix again, which makes it the most recently used entry on its
# worker, so the next probe only means something once enough filler has since
# passed through to turn that worker's WHOLE device pool over.
#
# Sized on the OWNER's share, not the fleet's. Filler prompts miss the tree, so
# the cache-aware policy has no candidates and falls back to power-of-two
# choices — roughly half of each burst lands on the owner. At ~2.3k tokens per
# filler that is 16 * 2.3k / 2 ~= 18k against an 8192-token pool, a bit over
# 2x. Sizing on the fleet total instead leaves ~1.1x, where one unlucky split
# makes a whole cycle evict nothing.
FILLERS_PER_PROBE = 16
# Probe cycles before giving up.
MAX_PROBE_CYCLES = 5


def _drive_until_host_served(
    router_url: str,
    model_id: str,
    primed: str,
    *,
    worker_urls: list[str],
) -> tuple[dict[str, int], dict[str, int]]:
    """Apply device pressure until a repeat of `primed` comes back served from
    the host tier, and report that probe's (dispatch deltas, tier split).

    Driven by the observed effect rather than a fixed request count: how many
    requests it takes to turn the device tier over depends on the tokenizer,
    the page size and how the scheduler batches, none of which this test
    should be asserting. A fixed count is either flaky or needlessly slow.

    The probe carries the assertion, so it cannot be a passive read — asking
    whether the prefix is served from host is also what puts it back on
    device. Hence the filler burst between cycles: a probe that re-warmed the
    prefix must not be the reason the next one finds it on device.
    """
    tiers: dict[str, int] = {}
    for cycle in range(MAX_PROBE_CYCLES):
        for i in range(FILLERS_PER_PROBE):
            _chat(router_url, model_id, _long_prompt(f"evict{cycle}-{i}"))
        deltas, tiers = _chat_and_attribute(router_url, model_id, primed, worker_urls)
        if tiers.get("host", 0) > 0:
            return deltas, tiers
    # A failing drive loop and an unreachable router look the same from here,
    # so the router state is best-effort: scraping it is exactly what fails
    # when the router is the reason, and an exception raised while building
    # the message would replace this diagnostic with a connection error.
    try:
        state = _tier_summary(_scrape(router_url))
    except Exception as exc:  # noqa: BLE001
        state = f"(unavailable: {exc!r})"
    raise AssertionError(
        f"no repeat of the primed prefix was served from the host tier after "
        f"{MAX_PROBE_CYCLES} eviction cycles of {FILLERS_PER_PROBE} requests; "
        f"last tier split={tiers}; router state: {state}"
    )


@pytest.mark.real_gpu
@pytest.mark.slow
@pytest.mark.parametrize("write_policy", WRITE_POLICIES)
def test_repeat_returns_to_the_host_tier_owner(
    router_binary,  # noqa: ARG001 - fixture forces release-binary presence
    gpu_allocator,
    write_policy: str,
) -> None:
    """The routing payoff: after a device eviction, a repeat of the evicted
    prefix still goes back to the worker holding it on host.

    This is what the tier split buys. With two workers and a tier-blind tree
    the owner is forgotten the moment its device copy goes, and the repeat
    falls through to min-load — a coin flip that prefills the whole prompt
    cold half the time.
    """
    spec = get_model_spec("qwen3-0.6b")
    gpus = gpu_allocator.acquire(2)
    try:
        with (
            spawn_worker(
                "qwen3-0.6b",
                gpu_ids=[gpus[0]],
                enable_kv_events=True,
                extra_args=worker_args(write_policy),
            ) as worker_a,
            spawn_worker(
                "qwen3-0.6b",
                gpu_ids=[gpus[1]],
                enable_kv_events=True,
                extra_args=worker_args(write_policy),
            ) as worker_b,
            Gateway() as router,
        ):
            worker_urls = [worker_a.url, worker_b.url]
            router.start_regular(
                model_id=spec["model"],
                tokenizer_path=spec["model"],
                worker_urls=worker_urls,
                policy="cache_aware",
                timeout=120.0,
            )

            # Both workers must actually be registered, or the test proves
            # nothing: if one fails introspection the router keeps the other,
            # /readyz is still satisfied, every request lands on the survivor
            # and it trivially is the "sole owner" every assertion below looks
            # for. The two-worker premise has to be checked, not assumed.
            # Waited on rather than read once: registration lands per worker,
            # so a single scrape can catch a half-registered router and fail a
            # fleet that was about to be complete.
            expected_workers = set(worker_urls)
            _wait_until(
                lambda: (lambda seen: seen if seen == expected_workers else None)(
                    _registered_workers(_scrape(router.base_url))
                ),
                timeout=60.0,
                what=(
                    f"the router to register both workers ({expected_workers}); "
                    "without both, every request lands on the survivor and it "
                    "is trivially the sole owner each assertion below looks for"
                ),
            )

            primed = _long_prompt("owner")

            def _sole_device_owner() -> str | None:
                text = _scrape(router.base_url)
                owners = [
                    url
                    for url in worker_urls
                    if _tree_blocks(text, "device", worker_url=url) > 0
                ]
                return owners[0] if len(owners) == 1 else None

            # The first request lands by min-load (the tree is empty), and its
            # cache events reach the router only after the response does. Wait
            # for the prefix to be indexed before repeating it: a repeat sent
            # inside that gap is still routed by load, so it can land on the
            # other worker and leave both owning the prefix with no sole owner
            # to name.
            _chat(router.base_url, spec["model"], primed)
            owner = _wait_until(
                _sole_device_owner,
                timeout=120.0,
                what="exactly one worker to own the primed prefix on device",
            )

            # With the prefix indexed and still on device, cache-aware routing
            # must send the repeat back to its owner. Establishing that here,
            # before any eviction, separates the two ways the assertion at the
            # end of the test can fail: routing that never honoured the tree at
            # all, versus a tree that forgot the owner once its device copy
            # went.
            deltas, tiers = _chat_and_attribute(
                router.base_url, spec["model"], primed, worker_urls
            )
            assert deltas.get(owner, 0) == 1, (
                "repeat of a device-resident prefix did not return to its "
                f"owner {owner}; per-worker deltas={deltas}, tier split={tiers}"
            )

            # Turn the device tier over until a repeat of the primed prefix
            # is actually served back from host. Driving on the probe rather
            # than on tree occupancy is what keeps the final assertion honest:
            # the per-worker tier gauges are aggregates over everything a
            # worker holds, so filler traffic alone can satisfy any inequality
            # between them while the primed prefix sits untouched on device.
            deltas, tiers = _drive_until_host_served(
                router.base_url,
                spec["model"],
                primed,
                worker_urls=worker_urls,
            )

            text = _scrape(router.base_url)
            print(f"[{write_policy}] after eviction: {_tier_summary(text)}")
            print(f"[{write_policy}] repeat served from: {tiers}")

            # The engine really did back blocks up to host. Without this the
            # test proves nothing: a fleet with no L2 traffic cannot exercise
            # the tier split at all, and the assertions below would hold
            # trivially on a device-only cache.
            assert _events(text, "block_stored", "CPU_PINNED") > 0, (
                "engine published no CPU_PINNED stores; hierarchical cache is "
                "not backing blocks up to host and this test is vacuous"
            )
            # The occupancy counters are booked at four mutation sites, all of
            # which the eviction churn above exercised.
            assert (
                _sum_where(text, "sgl_router_kv_tree_accounting_errors_total") == 0
            ), "tree occupancy accounting contradicted itself"

            # `_drive_until_host_served` has already established that the
            # host tier served the repeat; this pins down that it was the
            # owner's host tier, which is the routing half of the claim.
            assert deltas.get(owner, 0) == 1, (
                "repeat of a host-resident prefix did not return to its owner "
                f"{owner}; per-worker deltas={deltas}, tier split={tiers}"
            )
    finally:
        gpu_allocator.release(gpus)
