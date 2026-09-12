"""Hierarchical cache on the unified memory pool.

HiCache addresses the device buffers with the ids the controller holds, and
under `--enable-unified-memory` those are VIRTUAL while the L2 kernels index
per-layer views in kernel-facing space (and the state pool by physical slot).
On top of that the pool RELOCATES pages under compaction, and the conv/SSM
views are envelope-strided rather than a contiguous per-slot array.

So the guard has to be numerical, and it has to force a real host round trip:
a small device pool plus a dozen distinct long prefixes evicts the target off
the device, and re-requesting it can only be served by loading back through
L2. If any of the translate, the staging, or the move gate were wrong, the
reloaded KV would differ.

The reference is the SAME pool with HiCache off -- not the static pool. Unified
and static legitimately differ in reduction order here (measured up to 1.8 in
logprob on a fresh prompt for the GDN model), so a static baseline would drown
the signal; against unified-without-HiCache the expectation is bit-equality.

Both full-attention families get a cell: MHA reaches the L2 kernels through
`data_ptrs` the unified subclass has to build itself, MLA through ones the base
builds in `__init__`.

    python -m pytest test/registered/hicache/test_hicache_unified_memory.py -v
"""

import os
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=1500, stage="extra-b", runner_config="2-gpu-large")

_COMMON_ARGS = [
    "--trust-remote-code",
    "--enable-unified-memory",
    "--enable-cache-report",
]

# A small device pool is what makes the host tier reachable at all: the dozen
# fillers below have to push the target off the device.
_SMALL_POOL = ["--max-total-tokens", "8192"]

_PREFIX = (
    "The following is a detailed technical description of a distributed inference "
    "system with paged attention, radix prefix caching and hierarchical offload. "
) * 90
_TARGET = _PREFIX + " Question one:"


def _generate(base_url, text, max_new_tokens=32, logprobs=True):
    payload = {
        "text": text,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new_tokens},
    }
    if logprobs:
        payload["return_logprob"] = True
        payload["logprob_start_len"] = 0
    resp = requests.post(f"{base_url}/generate", json=payload, timeout=600)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    lp = (
        [t[0] for t in data["meta_info"]["output_token_logprobs"]] if logprobs else None
    )
    return data["text"], lp


class UnifiedMemoryHiCacheBase(CustomTestCase):
    """Two servers on the same pool, one with HiCache and one without."""

    model: str = ""
    extra_args: list = []
    server_env: dict = {}

    @classmethod
    def setUpClass(cls):
        if cls is UnifiedMemoryHiCacheBase:
            raise unittest.SkipTest("base class")
        base_args = _COMMON_ARGS + cls.extra_args
        cls.hicache_url = "http://127.0.0.1:8157"
        cls.reference_url = "http://127.0.0.1:8158"
        env = {**os.environ, **cls.server_env} if cls.server_env else None
        hicache_args = ["--enable-hierarchical-cache"]
        if "--hicache-size" not in base_args:
            hicache_args += ["--hicache-ratio", "4"]
        cls.process_hicache = popen_launch_server(
            cls.model,
            cls.hicache_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=base_args + hicache_args,
            env=env,
        )
        cls.process_reference = popen_launch_server(
            cls.model,
            cls.reference_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=base_args + ["--base-gpu-id", "1"],
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        for proc in (
            getattr(cls, "process_hicache", None),
            getattr(cls, "process_reference", None),
        ):
            if proc is not None:
                kill_process_tree(proc.pid)

    def _force_host_round_trip(self):
        """Evict the target off the device so the next hit must come from L2."""
        for i in range(12):
            _generate(
                self.hicache_url,
                _PREFIX + f" filler variant {i}. Question:",
                max_new_tokens=8,
                logprobs=False,
            )

    def _flush_both(self):
        """Compare from equal cache state. A populated radix tree shifts the
        chunked-prefill boundaries and with them the reduction order, which is
        a real effect but not the one under test."""
        for url in (self.hicache_url, self.reference_url):
            requests.post(f"{url}/flush_cache", timeout=180)
        time.sleep(3)

    def test_load_back_matches_no_hicache(self):
        """The sharp one: KV that made a device->host->device round trip must
        produce the same logprobs as a run that never left the device."""
        self._flush_both()
        cold_text, cold_lp = _generate(self.hicache_url, _TARGET)
        self._force_host_round_trip()
        warm_text, warm_lp = _generate(self.hicache_url, _TARGET)
        ref_text, ref_lp = _generate(self.reference_url, _TARGET)

        self.assertEqual(cold_text, ref_text)
        self.assertEqual(warm_text, ref_text)
        for label, lp in (("cold", cold_lp), ("after-L2-reload", warm_lp)):
            delta = max(abs(a - b) for a, b in zip(lp, ref_lp))
            self.assertAlmostEqual(
                delta,
                0.0,
                places=5,
                msg=f"{label} diverged from the no-HiCache reference by {delta}",
            )

    def test_server_survives_the_round_trip(self):
        """A wrong move gate or a missed free shows up as the idle memory-leak
        invariant aborting the scheduler rather than as bad output."""
        self._force_host_round_trip()
        for url in (self.hicache_url, self.reference_url):
            resp = requests.get(f"{url}/health", timeout=30)
            self.assertEqual(resp.status_code, 200)


class TestUnifiedMemoryHiCacheGDN(UnifiedMemoryHiCacheBase):
    """MHA full attention + gated-delta-net state: a per-layer-view sub-pool
    (kernel-facing ids) alongside an envelope-strided state sub-pool (physical
    slots, staged through a contiguous buffer)."""

    model = "Qwen/Qwen3.5-0.8B"
    extra_args = _SMALL_POOL + [
        "--linear-attn-backend",
        "triton",
        "--mamba-backend",
        "triton",
        "--max-mamba-cache-size",
        "64",
        "--mem-fraction-static",
        "0.6",
    ]


class TestUnifiedMemoryHiCacheSWA(UnifiedMemoryHiCacheBase):
    """Hybrid sliding-window attention. The sharpest of the four: the SWA side
    has no id space of its own here, so its load-back rows are BOUND for the
    anchor's virtual ids rather than allocated or translated. A translate-only
    derivation silently yields sink ids (the translate clamps an unbound page
    rather than failing), which reads correct on the wire and only surfaces
    later as the tree owning rows the allocator never had live."""

    model = "openai/gpt-oss-20b"
    extra_args = _SMALL_POOL + [
        "--attention-backend",
        "triton",
        "--mem-fraction-static",
        "0.7",
    ]


class TestUnifiedMemoryHiCacheTriPool(UnifiedMemoryHiCacheBase):
    """All three components at once (full + sliding-window + ShortConv state),
    each with its own compaction and its own id treatment on the host path."""

    model = "thinkingmachines/Inkling"
    server_env = {"SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1"}
    extra_args = [
        "--revision",
        "test",
        "--attention-backend",
        "triton",
        "--page-size",
        "128",
        "--mamba-radix-cache-strategy",
        "extra_buffer",
        "--swa-full-tokens-ratio",
        "0.1",
        "--mamba-full-memory-ratio",
        "0.1",
        "--mem-fraction-static",
        "0.5",
        "--cuda-graph-backend-prefill",
        "disabled",
        # The tri-pool's full cap runs to millions of tokens, and the host pool
        # is a multiple of it; bound it or the pair asks for hundreds of GB.
        "--hicache-size",
        "8",
    ]


class TestUnifiedMemoryHiCacheMLA(UnifiedMemoryHiCacheBase):
    """MLA full attention + KDA state. The MLA sub-pool reaches HiCache through
    the same translate, but builds its `data_ptrs` in `MLATokenToKVPool.
    __init__` rather than in the `_create_buffers` the unified subclass
    overrides -- so it is a genuinely different wiring path from the MHA cell
    above, worth its own cell rather than an assumed equivalence."""

    model = "yujiepan/kimi-linear-tiny-random"
    extra_args = _SMALL_POOL + [
        "--max-mamba-cache-size",
        "64",
        "--mem-fraction-static",
        "0.5",
        "--linear-attn-backend",
        "triton",
        "--mamba-backend",
        "triton",
        "--attention-backend",
        "triton",
        "--cuda-graph-backend-decode",
        "disabled",
        "--cuda-graph-backend-prefill",
        "disabled",
    ]


if __name__ == "__main__":
    unittest.main()
