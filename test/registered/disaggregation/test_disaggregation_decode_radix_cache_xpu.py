"""PD incremental KV transfer (decode radix cache) on Intel XPU.

With --disaggregation-decode-enable-radix-cache the decode server reports the
prefix it already holds as ``decode_prefix_len`` and prefill sends only the pages
after it. These tests check, on Mooncake (TENT) and NIXL (UCX + Level Zero):

* a request that extends a cached prompt moves far fewer KV bytes than the cold
  request (read from prefill's ``sglang:kv_transfer_total_mb`` histogram);
* the partial transfer lands at the right offset: a fact in the first page of the
  cached prefix still reaches the answer;
* a repeat of a fully cached prompt (aux/state only, no KV pages) still completes.

Requirements: as test_disaggregation_xpu.py (docker/xpu.Dockerfile).

Usage:
    python3 -m pytest test/registered/disaggregation/test_disaggregation_decode_radix_cache_xpu.py -v
"""

import importlib
import time
import unittest

import requests
import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN, is_in_ci

register_xpu_ci(est_time=800, suite="nightly-kernel-main-test-2-gpu-xpu", nightly=True)

_XPU_AVAILABLE = torch.xpu.is_available()
_XPU_DEVICE_COUNT = torch.xpu.device_count() if _XPU_AVAILABLE else 0
_SKIP_REASON = (
    "Intel XPU not available (torch.xpu.is_available() returned False)"
    if not _XPU_AVAILABLE
    else f"PD disaggregation needs 2 XPUs, found {_XPU_DEVICE_COUNT}"
)
_requires_two_xpus = unittest.skipUnless(
    _XPU_DEVICE_COUNT >= 2 or is_in_ci(), _SKIP_REASON
)

_MOONCAKE_XPU_ENV = {"MC_USE_TENT": "1"}
# Deferred destination publish is on by default but needs the WAR barrier,
# which is on by default only on CUDA.
_DEFERRED_PUBLISH_ENV = {
    "SGLANG_ENABLE_WAR_BARRIER": "1",
    "SGLANG_ENABLE_DISAGG_DECODE_DEFERRED_PUBLISH": "1",
}

_METRIC = "sglang:kv_transfer_total_mb"
# ~900 tokens: seven 128-token intel_xpu pages, so a hit skips most of the KV.
_FILLER = "The sky was clear and the market was quiet that day. " * 80


def _prompt(code: str, question: str) -> str:
    return f"Remember this: the secret code is {code}.\n{_FILLER}\n{question}"


def _read_histogram(url: str) -> tuple[float, float]:
    """(sum, count) of the prefill KV transfer size histogram, over all labels."""
    text = requests.get(f"{url}/metrics", timeout=10).text
    total, count = 0.0, 0.0
    for line in text.splitlines():
        if line.startswith(f"{_METRIC}_sum"):
            total += float(line.rsplit(" ", 1)[1])
        elif line.startswith(f"{_METRIC}_count"):
            count += float(line.rsplit(" ", 1)[1])
    return total, count


class _DecodeRadixCacheXpuTestMixin:
    transfer_backend_name: str
    backend_module: str
    use_rdma = False
    # Cold vs warm transfer ratio the warm request must stay under. With seven
    # pages cached a hit sends ~1 page; 0.5 leaves room for page rounding.
    max_warm_ratio = 0.5

    @classmethod
    def setUpClass(cls):
        if _XPU_DEVICE_COUNT < 2:
            raise RuntimeError(_SKIP_REASON)
        importlib.import_module(cls.backend_module)
        super().setUpClass()
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN
        cls.transfer_backend = [
            "--disaggregation-transfer-backend",
            cls.transfer_backend_name,
        ]
        xpu_args = ["--device", "xpu", "--attention-backend", "intel_xpu"]
        cls.extra_prefill_args = xpu_args + [
            "--chunked-prefill-size",
            "256",
            "--enable-metrics",
        ]
        cls.extra_decode_args = xpu_args + [
            "--disaggregation-decode-enable-radix-cache"
        ]
        cls.launch_all()

    def _generate(self, text: str, max_new_tokens: int = 8) -> str:
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": text,
                "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
            },
            timeout=120,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["text"]

    def _transfer_mb(self, text: str) -> tuple[str, float]:
        """Run one request; return its output and the KV MB prefill sent for it."""
        before_sum, before_count = _read_histogram(self.prefill_url)
        output = self._generate(text)
        # Prefill observes the histogram after the transfer settles, which can
        # trail the response.
        deadline = time.monotonic() + 10
        while True:
            after_sum, after_count = _read_histogram(self.prefill_url)
            if after_count > before_count or time.monotonic() > deadline:
                break
            time.sleep(0.2)
        self.assertEqual(
            after_count - before_count, 1, "expected one KV transfer observation"
        )
        return output, after_sum - before_sum

    def test_incremental_transfer_skips_cached_prefix(self):
        """Extending a cached prompt must send far less KV, and stay correct."""
        cold, cold_mb = self._transfer_mb(
            _prompt("7342", "Question: What is the secret code? Answer:")
        )
        self.assertIn("7342", cold)
        warm, warm_mb = self._transfer_mb(
            _prompt("7342", "Question: Repeat the secret code exactly. Answer:")
        )
        print(
            f"{self.transfer_backend_name}: cold={cold_mb:.3f}MB warm={warm_mb:.3f}MB"
        )
        self.assertGreater(cold_mb, 0)
        self.assertLess(
            warm_mb,
            cold_mb * self.max_warm_ratio,
            f"decode prefix hit did not shrink the transfer "
            f"(cold={cold_mb:.3f}MB, warm={warm_mb:.3f}MB)",
        )
        # The fact sits in the first cached page: a misplaced offset loses it.
        self.assertIn("7342", warm, f"wrong answer after partial transfer: {warm!r}")

    def test_full_prefix_hit_completes(self):
        """A repeat of a cached prompt (no KV pages to send) must still finish."""
        prompt = _prompt("5819", "Question: What is the secret code? Answer:")
        first = self._generate(prompt)
        self.assertIn("5819", first)
        for _ in range(3):
            self.assertIn("5819", self._generate(prompt))

    def test_distinct_prompts_are_not_mixed(self):
        """Prompts sharing the filler but not the fact must not reuse each other's KV."""
        for code in ("1111", "2468", "9035"):
            out = self._generate(
                _prompt(code, "Question: What is the secret code? Answer:")
            )
            self.assertIn(code, out, f"expected {code}, got {out!r}")


@_requires_two_xpus
class TestDecodeRadixCacheXpuMooncake(
    _DecodeRadixCacheXpuTestMixin, PDDisaggregationServerBase
):
    transfer_backend_name = "mooncake"
    backend_module = "mooncake.engine"
    extra_prefill_env = _MOONCAKE_XPU_ENV
    extra_decode_env = _MOONCAKE_XPU_ENV


@_requires_two_xpus
class TestDecodeRadixCacheXpuMooncakeDeferredPublish(
    _DecodeRadixCacheXpuTestMixin, PDDisaggregationServerBase
):
    """Mooncake with the deferred publish path, enabled on XPU by the WAR barrier."""

    transfer_backend_name = "mooncake"
    backend_module = "mooncake.engine"
    extra_prefill_env = _MOONCAKE_XPU_ENV
    extra_decode_env = {**_MOONCAKE_XPU_ENV, **_DEFERRED_PUBLISH_ENV}


@_requires_two_xpus
class TestDecodeRadixCacheXpuNixl(
    _DecodeRadixCacheXpuTestMixin, PDDisaggregationServerBase
):
    transfer_backend_name = "nixl"
    backend_module = "nixl._api"


@_requires_two_xpus
class TestDecodeRadixCacheXpuNixlDeferredPublish(
    _DecodeRadixCacheXpuTestMixin, PDDisaggregationServerBase
):
    """The overlap-scheduler deferred publish path, enabled on XPU by the WAR barrier."""

    transfer_backend_name = "nixl"
    backend_module = "nixl._api"
    extra_decode_env = _DEFERRED_PUBLISH_ENV


if __name__ == "__main__":
    unittest.main()
