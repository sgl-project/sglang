"""Shared helpers for the Intel XPU HiCache E2E tests.

Split out of a single test module so each per-scenario file stays inside
run_suite.py's --timeout-per-file budget.
"""

import json
import os
import time

import requests
import torch

from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    find_available_port,
    popen_launch_server,
)

XPU_AVAILABLE = torch.xpu.is_available() if hasattr(torch, "xpu") else False

# TODO(kpjeeja): the intel/sglang-dev image stage-b-test-1-gpu-xpu runs on ships no
# nixl, so every NIXL-gated file below skips there; install it in docker/xpu.Dockerfile
# to turn the storage-tier scenarios into signal.
try:
    from nixl._api import nixl_agent  # noqa: F401

    NIXL_AVAILABLE = True
except ImportError:
    NIXL_AVAILABLE = False

# Device KV pool, in tokens, for the offload/reload scenarios. --mem-fraction-static
# cannot size it small enough to matter: 0.5 of a 22 GB Arc holds ~285k tokens at
# Qwen2.5-1.5B's 28 KB/token, three orders of magnitude above any prompt here, so
# nothing is ever evicted and the load kernel never runs. 1024 tokens is 64 pages
# at --page-size 16, below every scenario's prompt set, so the prefix under test
# leaves the device tier and the reload has to come from the host.
EVICT_DEVICE_POOL_TOKENS = 1024
# Pinning the device pool small makes a large ratio cheap (8 x 1024 tokens is
# ~230 MB), and the host tier has to stay well clear of every scenario's working
# set -- otherwise the oldest node evicted is the prefix under test, and the
# reload measures a recompute instead of a host->device transfer.
EVICT_HICACHE_RATIO = 8
# Continuation length, in tokens, for every golden-vs-restore comparison.
# A restore cannot be expected to match greedily for an unbounded window even
# though the transfer is an identity: the restored prefix occupies different
# pages, so attention reduces over them in a different order, and bf16 rounding
# eventually flips a near-tie token. Measured on Arc with Qwen2.5-1.5B at
# page_size 16: a 48-token comparison diverges around char 147 (~30 tokens),
# deterministically and on both io backends, while a corrupted restore diverges
# by char 16. 16 tokens sits well inside that gap, so the comparison still
# catches corruption without asserting an FP-order coincidence.
# The exact-identity claim lives where it can actually be checked:
# test/registered/xpu/test_hicache_transfer_round_trip_xpu.py compares the
# transferred tensors directly.
COMPARE_TOKENS = 16


def resolve_base_url() -> str:
    """A probed-free port, so back-to-back launches in one file cannot collide."""
    default_port = int(DEFAULT_URL_FOR_TEST.rsplit(":", 1)[1])
    return f"http://127.0.0.1:{find_available_port(default_port)}"


def nixl_posix_config() -> str:
    """Fully-qualified NIXL config -- the first plugin with active=True is
    selected. use_direct_io=False avoids the O_DIRECT alignment requirement.
    The flat ``{"plugin": "posix"}`` form raises AttributeError at construction.
    """
    return json.dumps(
        {
            "plugin": {
                "posix": {
                    "active": True,
                    "use_uring": True,
                    "use_direct_io": False,
                }
            }
        }
    )


def launch_server(model: str, base_url: str, other_args: list, env_extra=None):
    """Launch a server with env_extra layered over the current environment.

    other_args is a flat argv list, as in the neighbouring hicache tests;
    popen_launch_server stringifies each element.
    """
    return popen_launch_server(
        model=model,
        base_url=base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
        env={**os.environ, **(env_extra or {})},
    )


def complete(base_url, model, prompt, max_tokens=32, timeout=90, want_cached=False):
    resp = requests.post(
        f"{base_url}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["text"]
    if not want_cached:
        return text
    cached = 0
    usage = data.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    if details:
        cached = details.get("cached_tokens", 0) or 0
    return text, cached


def prime_cache(base_url, model, prompt, max_tokens=COMPARE_TOKENS, timeout=90) -> None:
    """Serve `prompt` once so the next identical request is a cache hit.

    A restore must be compared against another cache-served run, never against a
    cold one: a full prefill and a prefix-reuse prefill are different numeric
    paths, and greedy decode amplifies the difference within a couple of tokens.
    Measured on Arc with Qwen2.5-1.5B at page_size 16, cold vs restored diverge
    at char 16 -- inside COMPARE_TOKENS, so a cold baseline would fail the
    comparison even when the transfer is exact.
    """
    complete(base_url, model, prompt, max_tokens=max_tokens, timeout=timeout)


def flush_cache(base_url, timeout_s: float = 30.0) -> None:
    """Flush the radix cache, waiting until it actually happens.

    `timeout` makes the endpoint wait out in-flight requests rather than decline
    the flush; a declined flush answers 400, which raise_for_status catches.

    Resets tiers 1 and 2 only: HiRadixCache.reset clears the tree, controller
    and host pool but not the storage backend, and cached_tokens counts storage
    hits. A prompt reused across a flush in one server can therefore still come
    back cached -- keep prompts unique per scenario.
    """
    resp = requests.post(
        f"{base_url}/flush_cache",
        params={"timeout": timeout_s},
        timeout=timeout_s + 30,
    )
    resp.raise_for_status()


def load_back_tokens(base_url) -> float:
    """Tokens loaded host (L2) -> device (L1) since launch, all pools summed.

    Requires --enable-metrics. This is the only direct evidence the load kernel
    ran: an output comparison and cached_tokens > 0 are both satisfied by a plain
    device-tier radix hit, and storage files are written by the write-through
    backup path, so none of them can tell a reload from a prefix that never left
    the device.
    """
    body = requests.get(f"{base_url}/metrics", timeout=30).text
    return sum(
        float(line.rsplit(" ", 1)[1])
        for line in body.splitlines()
        if line.startswith("sglang:load_back_tokens_total")
    )


def wait_load_back_tokens(base_url, above: float, timeout_s: float = 20.0) -> float:
    """Poll load_back_tokens_total until it passes `above`, then return it.

    The counter is incremented when the scheduler reaps the load ack
    (hiradix_cache.py load_back bookkeeping), which can trail the HTTP response
    by a loop iteration, so a bare read right after the completion races.
    Returns the last value observed either way; the caller does the asserting.
    """
    deadline = time.time() + timeout_s
    seen = load_back_tokens(base_url)
    while seen <= above and time.time() < deadline:
        time.sleep(0.5)
        seen = load_back_tokens(base_url)
    return seen


def count_storage_files(storage_dir) -> int:
    n = 0
    for _root, _dirs, names in os.walk(storage_dir):
        n += len(names)
    return n


def shared_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i
