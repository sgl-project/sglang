"""Shared helpers for the Intel XPU HiCache E2E tests."""

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

# Device KV pool, in tokens. --mem-fraction-static cannot size it small enough
# to evict anything on a 22 GB Arc; 1024 tokens (64 pages at --page-size 16) is
# below every scenario's prompt set, so the reload has to come from the host.
EVICT_DEVICE_POOL_TOKENS = 1024
# Keeps the host tier (~230 MB) well clear of every working set, so the prefix
# under test is not evicted from the host too and recomputed.
EVICT_HICACHE_RATIO = 8
# Tokens compared golden-vs-restore. A restored prefix sits on different pages,
# so bf16 reduction order eventually flips a near-tie token even on an exact
# transfer. Measured on Arc (Qwen2.5-1.5B, page_size 16): an exact restore holds
# for ~30 tokens, a corrupted one diverges by char 16. Exact identity is checked
# in test_hicache_transfer_round_trip_xpu.py.
COMPARE_TOKENS = 16


def resolve_base_url() -> str:
    """A probed-free port, so back-to-back launches in one file cannot collide."""
    default_port = int(DEFAULT_URL_FOR_TEST.rsplit(":", 1)[1])
    return f"http://127.0.0.1:{find_available_port(default_port)}"


def launch_server(model: str, base_url: str, other_args: list, env_extra=None):
    """Launch a server with env_extra layered over the current environment."""
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

    The golden must be a cache hit too: cold and prefix-reuse prefill are
    different numeric paths and diverge within COMPARE_TOKENS on Arc.
    """
    complete(base_url, model, prompt, max_tokens=max_tokens, timeout=timeout)


def load_back_tokens(base_url) -> float:
    """Tokens loaded host (L2) -> device (L1) since launch, all pools summed.

    Requires --enable-metrics. The only evidence the load kernel ran: output
    agreement and cached_tokens > 0 are both satisfied by a device-tier hit.
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


def shared_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i
