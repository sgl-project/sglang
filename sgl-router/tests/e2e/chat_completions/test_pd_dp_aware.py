# SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
# SPDX-License-Identifier: Apache-2.0

"""Real-GPU coverage for PD disaggregation under ``--dp-aware``.

Topology, four cards: one prefill server with ``--dp-size 2`` and one decode
server with ``--dp-size 2``, behind a router in PD mode with ``--dp-aware``.
The router then picks the prefill rank (a cached prefix first, else the fewest
in-flight requests) and the decode rank (the fewest in-flight requests), and
sends each as ``X-Data-Parallel-Rank``; the engine's DP controller dispatches
straight to that rank.

Which rank served a request is read from each worker's own ``/metrics``: every
scheduler labels ``sglang:num_requests_total`` with its ``dp_rank``, so a delta
there is the engine's word, not the router's.
"""

from __future__ import annotations

import logging
import os
import re
import socket
import threading
import time
from collections.abc import Iterator
from typing import NamedTuple

import httpx
import pytest
from infra import model_pool
from infra.model_pool import ModelInstance, spawn_worker
from infra.model_specs import get_model_spec
from infra.router import Router

logger = logging.getLogger(__name__)

MODEL_ID = "qwen3-0.6b"
DP_SIZE = 2
RANKS = frozenset(range(DP_SIZE))

_METRIC_RE = re.compile(r"^(\w+)\{([^}]*)\}\s+(-?\d+(?:\.\d+)?)\s*$")
_LABEL_RE = re.compile(r'(\w+)="([^"]*)"')

# What the engine's own PD CI fixture exports: mooncake's handshake port range
# kept off the test ports, and the TP rank-consensus checker on.
_MOONCAKE_ENV = {
    "MC_TCP_ENABLE_CONNECTION_POOL": "true",
    "MC_MIN_RPC_PORT": "61000",
    "MC_MAX_RPC_PORT": "65000",
    "SGLANG_ENABLE_RANK_CONSENSUS_CHECKER": "1",
}


class Topology(NamedTuple):
    router: Router
    prefill: ModelInstance
    decode: ModelInstance
    model: str


# --- launch ------------------------------------------------------------------


def _free_port_pair() -> int:
    """A port ``p`` with ``p`` and ``p + 1`` both free: the engine publishes KV
    events for rank ``r`` on ``base + r``."""
    for _ in range(50):
        with socket.socket() as a:
            a.bind(("", 0))
            base = a.getsockname()[1]
            with socket.socket() as b:
                try:
                    b.bind(("", base + 1))
                except OSError:
                    continue
                return base
    raise RuntimeError("no free consecutive port pair")


def _transfer_args(gpu_ids: list[int]) -> tuple[list[str], list[str]]:
    """Mooncake transfer args for a server on ``gpu_ids`` and the env keys this
    call exported for it.

    Reuses the engine PD fixture's NIC discovery so the server pins the RDMA
    devices next to its cards; without any, mooncake takes its TCP path.
    """
    # Lazy: the module imports sglang.test, which needs the engine installed.
    from sglang.test.server_fixtures import disaggregation_fixture as pd_fixture

    args = ["--disaggregation-transfer-backend", "mooncake"]
    if not pd_fixture._get_available_ib_devices():
        logger.warning("no RDMA device found; mooncake will use TCP")
        return args, []
    devices = model_pool.resolve_device_ids(gpu_ids)
    try:
        physical: list[int] | None = [int(d) for d in devices]
    except ValueError:
        # UUID-shaped allotment: let the fixture read CUDA_VISIBLE_DEVICES.
        physical = None
    ib_devices = pd_fixture.get_rdma_devices_args(physical)
    exported = []
    if pd_fixture._maybe_set_roce_gid_index(ib_devices):
        exported.append("MC_GID_INDEX")
    if pd_fixture._maybe_set_ucx_net_devices(ib_devices):
        exported.append("UCX_NET_DEVICES")
    return args + ["--disaggregation-ib-device", ib_devices], exported


@pytest.fixture(scope="module")
def topology(router_binary, gpu_allocator) -> Iterator[Topology]:  # noqa: ARG001
    """2P + 2D as dp ranks: prefill on cards 0-1, decode on cards 2-3."""
    gpus = gpu_allocator.acquire(2 * DP_SIZE)
    prefill_gpus, decode_gpus = gpus[:DP_SIZE], gpus[DP_SIZE:]
    spec = get_model_spec(MODEL_ID)
    bootstrap_port = model_pool._get_open_port()
    exported = [k for k in _MOONCAKE_ENV if k not in os.environ]
    os.environ.update({k: v for k, v in _MOONCAKE_ENV.items() if k in exported})
    try:
        prefill_transfer, prefill_env = _transfer_args(prefill_gpus)
        decode_transfer, decode_env = _transfer_args(decode_gpus)
        exported += prefill_env + decode_env
        common = ["--dp-size", str(DP_SIZE), "--enable-metrics"]
        with (
            spawn_worker(
                MODEL_ID,
                gpu_ids=prefill_gpus,
                disagg_mode="prefill",
                bootstrap_port=bootstrap_port,
                enable_kv_events=True,
                kv_events_port=_free_port_pair(),
                # Distinct rendezvous ports: both sides picking one at random
                # can collide at init_process_group.
                extra_args=common
                + prefill_transfer
                + ["--nccl-port", str(model_pool._get_open_port())],
            ) as prefill,
            spawn_worker(
                MODEL_ID,
                gpu_ids=decode_gpus,
                disagg_mode="decode",
                bootstrap_port=bootstrap_port,
                extra_args=common
                + decode_transfer
                + ["--nccl-port", str(model_pool._get_open_port())],
            ) as decode,
            Router(proxy_request_timeout_secs=180) as router,
        ):
            router.start_pd(
                model_id=spec["model"],
                tokenizer_path=spec["model"],
                prefill_urls=[prefill.url],
                decode_urls=[decode.url],
                extra_args=["--dp-aware"],
                timeout=180.0,
            )
            yield Topology(router, prefill, decode, spec["model"])
    finally:
        for key in exported:
            os.environ.pop(key, None)
        gpu_allocator.release(gpus)


# --- observation ---------------------------------------------------------------


def _scrape(url: str) -> str:
    resp = httpx.get(f"{url}/metrics", timeout=10.0)
    resp.raise_for_status()
    return resp.text


def _per_rank(text: str, name: str) -> dict[int, float]:
    """``name`` summed over every label set but ``dp_rank``."""
    out: dict[int, float] = {}
    for line in text.splitlines():
        match = _METRIC_RE.match(line)
        if not match or match.group(1) != name:
            continue
        labels = dict(_LABEL_RE.findall(match.group(2)))
        if "dp_rank" not in labels:
            continue
        rank = int(labels["dp_rank"])
        out[rank] = out.get(rank, 0.0) + float(match.group(3))
    return out


def _served(url: str) -> dict[int, float]:
    return _per_rank(_scrape(url), "sglang:num_requests_total")


def _running(url: str) -> dict[int, float]:
    return _per_rank(_scrape(url), "sglang:num_running_reqs")


def _delta(before: dict[int, float], after: dict[int, float]) -> dict[int, int]:
    return {r: int(after.get(r, 0) - before.get(r, 0)) for r in RANKS}


def _only_rank(delta: dict[int, int]) -> int:
    hit = [r for r, n in delta.items() if n]
    assert len(hit) == 1, f"expected exactly one rank to serve, got {delta}"
    return hit[0]


def _prefill_tree_blocks(router_url: str, worker_url: str) -> float:
    total = 0.0
    for line in _scrape(router_url).splitlines():
        match = _METRIC_RE.match(line)
        if not match or match.group(1) != "sgl_router_kv_tree_blocks":
            continue
        labels = dict(_LABEL_RE.findall(match.group(2)))
        if labels.get("worker_url") == worker_url:
            total += float(match.group(3))
    return total


def _chat(
    topo: Topology, prompt: str, *, max_tokens: int, stream: bool = False
) -> httpx.Response:
    return httpx.post(
        f"{topo.router.base_url}/v1/chat/completions",
        json={
            "model": topo.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "ignore_eos": True,
            "temperature": 0,
            "stream": stream,
        },
        timeout=240.0,
    )


def _assert_completed(resp: httpx.Response, max_tokens: int) -> None:
    assert resp.status_code == 200, resp.text
    choice = resp.json()["choices"][0]
    assert choice["message"]["content"], f"empty content: {choice!r}"
    # ignore_eos: decode ran the full budget, so the reply came through PD.
    assert choice["finish_reason"] == "length", choice
    assert resp.json()["usage"]["completion_tokens"] == max_tokens


def _long_prompt(tag: str) -> str:
    """A few hundred tokens that no other test shares, so the prefill ranks'
    caches start cold for it."""
    body = " ".join(f"{tag}-{i}" for i in range(160))
    return f"Read this list and reply with its length: {body}"


# --- tests -----------------------------------------------------------------------


@pytest.mark.real_gpu
@pytest.mark.pd_mode
@pytest.mark.slow
def test_concurrent_requests_use_every_prefill_and_decode_rank(topology):
    """Eight requests in flight at once: least-in-flight puts work on both
    prefill ranks and both decode ranks, and every reply completes."""
    topo = topology
    max_tokens = 48
    p0, d0 = _served(topo.prefill.url), _served(topo.decode.url)
    results: list[httpx.Response | Exception] = [None] * 8  # type: ignore[list-item]

    def one(i: int) -> None:
        try:
            results[i] = _chat(topo, _long_prompt(f"spread{i}"), max_tokens=max_tokens)
        except Exception as exc:  # noqa: BLE001
            results[i] = exc

    threads = [threading.Thread(target=one, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=300)
    for r in results:
        assert isinstance(r, httpx.Response), r
        _assert_completed(r, max_tokens)

    prefill, decode = (
        _delta(p0, _served(topo.prefill.url)),
        _delta(d0, _served(topo.decode.url)),
    )
    assert sum(prefill.values()) == 8, prefill
    assert sum(decode.values()) == 8, decode
    assert all(prefill[r] > 0 for r in RANKS), f"a prefill rank sat idle: {prefill}"
    assert all(decode[r] > 0 for r in RANKS), f"a decode rank sat idle: {decode}"


@pytest.mark.real_gpu
@pytest.mark.pd_mode
@pytest.mark.slow
def test_repeated_prefix_sticks_to_the_prefill_rank_that_cached_it(topology):
    """The first request lands on a rank by load; once that rank's KV events
    reach the router, every repeat of the prompt goes back to it."""
    topo = topology
    prompt = _long_prompt("affinity")
    before = _served(topo.prefill.url)
    blocks = _prefill_tree_blocks(topo.router.base_url, topo.prefill.url)
    _assert_completed(_chat(topo, prompt, max_tokens=8), 8)
    owner = _only_rank(_delta(before, _served(topo.prefill.url)))

    # The prompt is new to the tree, so its KV events can only grow it; a
    # repeat sent before they land would be placed by load, not by cache.
    deadline = time.time() + 60.0
    while _prefill_tree_blocks(topo.router.base_url, topo.prefill.url) <= blocks:
        assert time.time() < deadline, "router saw no KV events for the prompt"
        time.sleep(0.5)

    repeats = 8
    before = _served(topo.prefill.url)
    for _ in range(repeats):
        _assert_completed(_chat(topo, prompt, max_tokens=8), 8)
    delta = _delta(before, _served(topo.prefill.url))
    # By load alone a sequential repeat picks a rank at random, so eight in a
    # row on the owner is a 1-in-256 accident; the cached prefix makes it certain.
    assert delta == {owner: repeats, 1 - owner: 0}, f"owner={owner} delta={delta}"


@pytest.mark.real_gpu
@pytest.mark.pd_mode
@pytest.mark.slow
def test_decode_rank_avoids_the_rank_holding_a_long_request(topology):
    """While one decode rank streams a long reply, a new request goes to the
    other rank: the decode rank follows the router's in-flight count."""
    topo = topology
    long_tokens = 400
    chunks: list[str] = []
    errors: list[Exception] = []
    started = threading.Event()

    def stream_long() -> None:
        try:
            with httpx.stream(
                "POST",
                f"{topo.router.base_url}/v1/chat/completions",
                json={
                    "model": topo.model,
                    "messages": [{"role": "user", "content": _long_prompt("hold")}],
                    "max_tokens": long_tokens,
                    "ignore_eos": True,
                    "stream": True,
                },
                timeout=300.0,
            ) as resp:
                assert resp.status_code == 200, resp.read()
                for line in resp.iter_lines():
                    if line.startswith("data:"):
                        chunks.append(line)
                        started.set()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
            started.set()

    thread = threading.Thread(target=stream_long)
    thread.start()
    try:
        assert started.wait(timeout=120), "long request produced no chunk"
        assert not errors, errors
        deadline = time.time() + 30.0
        while True:
            busy = [r for r, n in _running(topo.decode.url).items() if n > 0]
            if busy:
                break
            assert time.time() < deadline, "no decode rank reports the long request"
            time.sleep(0.2)
        assert len(busy) == 1, f"long request on several decode ranks: {busy}"
        holder = busy[0]

        before = _served(topo.decode.url)
        _assert_completed(_chat(topo, _long_prompt("short"), max_tokens=8), 8)
        delta = _delta(before, _served(topo.decode.url))
        assert delta[1 - holder] == 1, f"holder={holder} delta={delta}"
    finally:
        thread.join(timeout=300)
    assert not errors, errors
    assert chunks and chunks[-1].strip() == "data: [DONE]", chunks[-3:]
