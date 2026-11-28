"""Shared helpers for the Intel XPU HiCache E2E tests."""

import functools
import json
import os
import time
from typing import Optional

import requests
import torch

from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    find_available_port,
    popen_launch_server,
)

XPU_AVAILABLE = torch.xpu.is_available() if hasattr(torch, "xpu") else False


# POSIX I/O queues to try, in order. Docker's default seccomp profile blocks io_uring
# but allows Linux AIO, so AIO keeps NIXL storage usable in the XPU CI containers.
_NIXL_POSIX_IO_QUEUES = ("use_uring", "use_aio")


@functools.cache
def _nixl_posix_probe() -> tuple[Optional[str], Optional[str]]:
    """(io-queue parameter a POSIX backend starts with, None), or (None, why none does).

    Starts the backend the way nixl_posix_config() asks the server to, so a usable
    queue here means the server can start too.
    """
    try:
        from nixl._api import nixl_agent, nixl_agent_config
    except ImportError as e:
        # Includes the meta package picking a missing nixl_cuXX backend.
        return None, f"NIXL not importable: {e}"
    errors = []
    for param in _NIXL_POSIX_IO_QUEUES:
        agent = nixl_agent(
            f"hicache_xpu_probe_{os.getpid()}_{param}", nixl_agent_config(backends=[])
        )
        if "POSIX" not in agent.get_plugin_list():
            return None, "NIXL POSIX plugin not installed"
        try:
            agent.create_backend("POSIX", {param: "true"})
            return param, None
        except Exception as e:
            errors.append(f"{param}: {e}")
    return None, "NIXL POSIX backend cannot start: " + "; ".join(errors)


def nixl_posix_unavailable_reason() -> Optional[str]:
    """Why NIXL-POSIX storage cannot run in this process, or None if it can."""
    return _nixl_posix_probe()[1]


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
# for ~30 tokens, so 16 leaves margin. Exact identity is checked in
# test_hicache_transfer_round_trip_xpu.py.
COMPARE_TOKENS = 16
# Server args for tests that compare a restored continuation against a golden.
# Triton's default decode picks its KV split count at runtime, and with the
# overlap scheduler a prefix-cache hit can decode with a different reduction
# order than the request that built the cache, even with no HiCache. On a prompt
# with bf16 near-ties (Qwen2.5-1.5B on B60) that flips a token within the first
# few, so a golden-vs-restore mismatch would not mean the KV was corrupted.
# Fixed split sizes make the two decodes bit-identical, so equality is exact.
DETERMINISTIC_DECODE_ARGS = [
    "--enable-deterministic-inference",
    # The XPU default under deterministic inference is intel_xpu, which turns
    # the radix cache off; HiCache needs it.
    "--attention-backend",
    "triton",
]


def resolve_base_url() -> str:
    """A probed-free port, so back-to-back launches in one file cannot collide."""
    default_port = int(DEFAULT_URL_FOR_TEST.rsplit(":", 1)[1])
    return f"http://127.0.0.1:{find_available_port(default_port)}"


def nixl_posix_config() -> str:
    """Fully-qualified NIXL config -- the first plugin with active=True is
    selected. use_direct_io=False avoids the O_DIRECT alignment requirement;
    SGLang reads it at the top level, not from the plugin section.
    The I/O queue is the first one _nixl_posix_probe() could start.
    The flat ``{"plugin": "posix"}`` form raises AttributeError at construction.
    """
    return json.dumps(
        {
            "plugin": {
                "posix": {
                    "active": True,
                    _nixl_posix_probe()[0] or "use_uring": True,
                }
            },
            "use_direct_io": False,
        }
    )


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


def count_storage_files(storage_dir) -> int:
    n = 0
    for _root, _dirs, names in os.walk(storage_dir):
        n += len(names)
    return n


def wait_storage_settled(storage_dir, above: int = 0, timeout_s: float = 30.0) -> int:
    """Wait until the file count passes `above` and stops growing, then return it.

    Write-through backs up to storage on the cache controller's thread after the
    request returns, one node at a time, so an immediate count can miss pages.
    """
    deadline = time.time() + timeout_s
    prev, seen = -1, count_storage_files(storage_dir)
    while (seen <= above or seen != prev) and time.time() < deadline:
        time.sleep(1.0)
        prev, seen = seen, count_storage_files(storage_dir)
    return seen


def shared_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i
