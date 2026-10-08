"""Compare unified-memory HiCache reloads against a resident-cache reference.

Evict a target prefix with distinct filler requests, require a host hit on
reload, and compare generated text and output logprobs. Both servers use the
same unified-memory configuration to keep attention reduction order comparable.
Covers GDN, SWA, tri-pool, and MLA layouts.
"""

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

register_cuda_ci(est_time=600, stage="extra-a", runner_config="2-gpu-large")

_COMMON_ARGS = [
    "--trust-remote-code",
    "--enable-unified-memory",
    "--enable-cache-report",
    "--max-running-requests",
    "1",
    "--context-length",
    "4096",
]

# Distinct filler prefixes must evict the target from device memory.
_SMALL_POOL = ["--max-total-tokens", "8192"]

_PREFIX = (
    "The following is a detailed technical description of a distributed inference "
    "system with paged attention, radix prefix caching and hierarchical offload. "
) * 90
_TARGET = _PREFIX + " Question one:"
_CONTINUATION = _TARGET + " Explain how it works."


def _generate(base_url, text, max_new_tokens=32, logprobs=True):
    payload = {
        "text": text,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new_tokens},
    }
    if logprobs:
        payload["return_logprob"] = True
        # Output logprobs suffice; asking for prompt logprobs from zero
        # caps the reusable prefix at zero and bypasses HiCache entirely.
        payload["logprob_start_len"] = -1
    resp = requests.post(f"{base_url}/generate", json=payload, timeout=600)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    lp = (
        [t[0] for t in data["meta_info"]["output_token_logprobs"]] if logprobs else None
    )
    return data["text"], lp, data["meta_info"]


class UnifiedMemoryHiCacheBase(CustomTestCase):
    """Compare identical unified-memory configurations with and without HiCache."""

    model: str = ""
    extra_args: list = []

    @classmethod
    def setUpClass(cls):
        if cls is UnifiedMemoryHiCacheBase:
            raise unittest.SkipTest("base class")
        base_args = _COMMON_ARGS + cls.extra_args
        cls.hicache_url = "http://127.0.0.1:8157"
        cls.reference_url = "http://127.0.0.1:8158"
        hicache_args = ["--enable-hierarchical-cache"]
        if "--hicache-size" not in base_args:
            hicache_args += ["--hicache-ratio", "4"]
        cls.process_hicache = popen_launch_server(
            cls.model,
            cls.hicache_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=base_args + hicache_args,
        )
        cls.addClassCleanup(kill_process_tree, cls.process_hicache.pid)
        cls.process_reference = popen_launch_server(
            cls.model,
            cls.reference_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=base_args + ["--base-gpu-id", "1"],
        )
        cls.addClassCleanup(kill_process_tree, cls.process_reference.pid)

    def _force_host_round_trip(self):
        """Evict the target off the device so the next hit must come from L2."""
        for i in range(8):
            _generate(
                self.hicache_url,
                f"Document {i}. "
                + (f"Unique filler {i} about an unrelated subject. " * 300),
                max_new_tokens=8,
                logprobs=False,
            )

    def _flush_both(self):
        """Reset cache state to match prefill boundaries and reduction order."""
        for url in (self.hicache_url, self.reference_url):
            requests.post(f"{url}/flush_cache", timeout=180)
        time.sleep(3)

    def test_load_back_matches_no_hicache(self):
        """Host reloads preserve generated text and logprobs within tolerance."""
        self._flush_both()
        cold_text, cold_lp, _ = _generate(self.hicache_url, _TARGET)
        ref_cold_text, ref_cold_lp, _ = _generate(self.reference_url, _TARGET)
        self._force_host_round_trip()
        # Extend the prefix so both servers compute new KV rows. Repeating it
        # would let only the resident reference reuse its original final-token KV.
        warm_text, warm_lp, warm_meta = _generate(self.hicache_url, _CONTINUATION)
        ref_text, ref_lp, _ = _generate(self.reference_url, _CONTINUATION)

        self.assertGreater(
            (warm_meta.get("cached_tokens_details") or {}).get("host", 0),
            0,
            msg=f"Target did not reload from host: {warm_meta}",
        )
        self.assertEqual(cold_text, ref_cold_text)
        self.assertEqual(warm_text, ref_text)
        # Match the reference's prefill boundary in each comparison: cold
        # against cold, and an L2 prefix hit against a resident prefix hit.
        for label, lp, reference in (
            ("cold", cold_lp, ref_cold_lp),
            ("after-L2-reload", warm_lp, ref_lp),
        ):
            self.assertEqual(len(lp), len(reference))
            delta = max(abs(a - b) for a, b in zip(lp, reference))
            self.assertAlmostEqual(
                delta,
                0.0,
                places=5,
                msg=f"{label} diverged from the no-HiCache reference by {delta}",
            )

    def test_server_survives_the_round_trip(self):
        """Cache churn must leave both schedulers healthy."""
        self._force_host_round_trip()
        for url in (self.hicache_url, self.reference_url):
            resp = requests.get(f"{url}/health", timeout=30)
            self.assertEqual(resp.status_code, 200)


class TestUnifiedMemoryHiCacheGDN(UnifiedMemoryHiCacheBase):
    """MHA full attention with envelope-strided gated-delta-net state."""

    model = "yujiepan/qwen3.5-tiny-random"
    extra_args = _SMALL_POOL + [
        "--linear-attn-backend",
        "triton",
        "--mamba-backend",
        "triton",
        "--max-mamba-cache-size",
        "8",
        "--mem-fraction-static",
        "0.6",
    ]


class TestUnifiedMemoryHiCacheSWA(UnifiedMemoryHiCacheBase):
    """Hybrid SWA reloads bind pages to the full-attention pool's virtual IDs."""

    model = "yujiepan/gemma-4e-tiny-random"
    extra_args = _SMALL_POOL + [
        "--attention-backend",
        "triton",
        "--mem-fraction-static",
        "0.7",
    ]


class TestUnifiedMemoryHiCacheTriPool(UnifiedMemoryHiCacheBase):
    """Full attention, sliding-window attention, and ShortConv state together."""

    # The test revision is the reduced checkpoint used by Inkling CI.
    model = "thinkingmachines/Inkling"
    extra_args = _SMALL_POOL + [
        "--revision",
        "test",
        "--attention-backend",
        "triton",
        "--page-size",
        "128",
        "--mamba-radix-cache-strategy",
        "extra_buffer",
        "--swa-full-tokens-ratio",
        "0.8",
        "--max-mamba-cache-size",
        "8",
        "--mamba-full-memory-ratio",
        "0.1",
        "--mem-fraction-static",
        "0.5",
        "--cuda-graph-backend-prefill",
        "disabled",
        # Bound total host memory across all three component pools.
        "--hicache-size",
        "8",
    ]


class TestUnifiedMemoryHiCacheMLA(UnifiedMemoryHiCacheBase):
    """MLA full attention with KDA state and MLA-specific transfer pointers."""

    model = "yujiepan/kimi-linear-tiny-random"
    extra_args = _SMALL_POOL + [
        "--max-mamba-cache-size",
        "8",
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
