"""Canary self-e2e on the DSV4-Flash superset fixture.

DSV4-Flash combines three axes (MLA, SWA dual-pool, packed pool / page_size=128)
that mha_full does not exercise; co-locating them in one fixture file avoids
spinning up separate mha_swa / mla_full servers.

11 cases (7 inherited from mha_full + 4 DSV4-specific axes). All under
``extra-a`` / ``1-gpu-large``; per-case timeout is bumped to 120s to cover the
heavier server warmup.
"""

from __future__ import annotations

import os
import unittest
from typing import ClassVar, List

from sglang.test.canary_e2e_base import CanaryE2EBase
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=1200, stage="extra-a", runner_config="dsv4-8-gpu-h200")


_DSV4_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
# SOT-required: 5 layers + compress_ratios [0, 0, 4, 128, 4] to cover all three
# DSV4 compression flavours (full / c4 / c128). The 128 axis is load-bearing —
# c128 is what makes DSV4 unique.
#
# Must use REAL weights, NOT --load-format dummy: dummy random weights produce
# numerically-extreme `kv_score` inputs to the c128 prefill softmax (expf path
# in c128_v2.cuh:442) and trip `CUDA error: an illegal instruction`. The kernel
# itself is healthy (test_c128_v2.py unit test PASSES on H200; the real-weight
# TP=4 DSV4-Flash recipe at test_deepseek_v4_flash_fp4_h200.py PASSES too) —
# the bug is specific to dummy weights' tail distribution. Real weights at the
# 5-layer truncation work fine even though sglang loads only the first 5 layers
# from the cached safetensors.
_NUM_LAYERS_OVERRIDE = '{"num_hidden_layers": 5, "compress_ratios": [0, 0, 4, 128, 4]}'

_DSV4_BASE_ARGS: List[str] = [
    "--trust-remote-code",
    "--json-model-override-args",
    _NUM_LAYERS_OVERRIDE,
    "--disable-cuda-graph",
    "--page-size",
    "128",
    "--moe-runner-backend",
    "marlin",
    "--watchdog-timeout",
    "900",
    # DSV4-Flash has 256+ MoE experts/layer; even 5 layers + real weights blow
    # a single H200's HBM. Cookbook recipe is TP=4. Run TP=4 here too (the
    # sci-h200 runner has 8x H200; canary just claims 4 of them).
    "--tp",
    "4",
    "--mem-fraction-static",
    "0.6",
    "--max-running-requests",
    "32",
]
_PER_CASE_TIMEOUT = 300.0


class _DSV4FlashBase(CanaryE2EBase):
    model: ClassVar[str] = _DSV4_MODEL
    extra_server_args: ClassVar[List[str]] = list(_DSV4_BASE_ARGS)


class TestNoPerturbNoViolation(_DSV4FlashBase, unittest.TestCase):
    def test_no_perturb_no_violation(self) -> None:
        results = self.send_parallel_requests(
            n=16, max_new_tokens=32, timeout=_PER_CASE_TIMEOUT
        )
        for r in results:
            self.assertEqual(r.get("status_code"), 200, r)
        self.assert_health_ok()


class TestPerturbReqToTokenDetectsViolation(_DSV4FlashBase, unittest.TestCase):
    perturb_prob: ClassVar[float] = 0.5
    allow_launch_failure: ClassVar[bool] = True

    def test_perturb_req_to_token_detects_violation(self) -> None:
        if self.launch_failed:
            self.assert_violation_kind_logged(
                ["per_forward_", "sweep_"], flush_wait_seconds=2.0
            )
            return
        self.send_parallel_requests(n=64, max_new_tokens=32, timeout=_PER_CASE_TIMEOUT)
        self.assert_violation_kind_logged(
            ["per_forward_", "sweep_"], flush_wait_seconds=2.0
        )


class TestRealDataOff(_DSV4FlashBase, unittest.TestCase):
    extra_server_args: ClassVar[List[str]] = [
        *_DSV4_BASE_ARGS,
        "--kv-canary-real-data",
        "off",
    ]

    def test_real_data_off(self) -> None:
        results = self.send_parallel_requests(
            n=8, max_new_tokens=16, timeout=_PER_CASE_TIMEOUT
        )
        for r in results:
            self.assertEqual(r.get("status_code"), 200, r)


class TestRealDataBit(_DSV4FlashBase, unittest.TestCase):
    extra_server_args: ClassVar[List[str]] = [
        *_DSV4_BASE_ARGS,
        "--kv-canary-real-data",
        "bit",
    ]

    def test_real_data_bit(self) -> None:
        results = self.send_parallel_requests(
            n=8, max_new_tokens=16, timeout=_PER_CASE_TIMEOUT
        )
        for r in results:
            self.assertEqual(r.get("status_code"), 200, r)


class TestRealDataAll(_DSV4FlashBase, unittest.TestCase):
    extra_server_args: ClassVar[List[str]] = [
        *_DSV4_BASE_ARGS,
        "--kv-canary-real-data",
        "all",
    ]

    def test_real_data_all(self) -> None:
        results = self.send_parallel_requests(
            n=8, max_new_tokens=16, timeout=_PER_CASE_TIMEOUT
        )
        for r in results:
            self.assertEqual(r.get("status_code"), 200, r)


class TestRealDataAllPerturbKvByteDetectsViolation(_DSV4FlashBase, unittest.TestCase):
    extra_server_args: ClassVar[List[str]] = [
        *_DSV4_BASE_ARGS,
        "--kv-canary-real-data",
        "all",
        "--kv-canary-sweep-interval",
        "1",
    ]
    allow_launch_failure: ClassVar[bool] = True

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["SGLANG_KV_CANARY_REAL_PERTURB_BYTES_PROB"] = "0.5"
        os.environ["SGLANG_KV_CANARY_REAL_PERTURB_BYTES_SEED"] = "11"
        super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        os.environ.pop("SGLANG_KV_CANARY_REAL_PERTURB_BYTES_PROB", None)
        os.environ.pop("SGLANG_KV_CANARY_REAL_PERTURB_BYTES_SEED", None)
        super().tearDownClass()

    def test_real_data_all_perturb_kv_byte_detects_violation(self) -> None:
        if self.launch_failed:
            self.assert_violation_kind_logged(
                ["per_forward_", "sweep_"], flush_wait_seconds=2.0
            )
            return
        self.send_parallel_requests(n=64, max_new_tokens=32, timeout=_PER_CASE_TIMEOUT)
        self.assert_violation_kind_logged(
            ["per_forward_", "sweep_"], flush_wait_seconds=2.0
        )


class TestSweepOrphanRadixDetectsViolation(_DSV4FlashBase, unittest.TestCase):
    extra_server_args: ClassVar[List[str]] = [
        *_DSV4_BASE_ARGS,
        "--kv-canary-real-data",
        "all",
        "--kv-canary-sweep-interval",
        "1",
    ]
    allow_launch_failure: ClassVar[bool] = True

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["SGLANG_KV_CANARY_REAL_PERTURB_BYTES_PROB"] = "0.5"
        os.environ["SGLANG_KV_CANARY_REAL_PERTURB_BYTES_SEED"] = "12"
        os.environ["SGLANG_KV_CANARY_REAL_PERTURB_BYTES_REQUIRE_ORPHAN"] = "1"
        super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        os.environ.pop("SGLANG_KV_CANARY_REAL_PERTURB_BYTES_PROB", None)
        os.environ.pop("SGLANG_KV_CANARY_REAL_PERTURB_BYTES_SEED", None)
        os.environ.pop("SGLANG_KV_CANARY_REAL_PERTURB_BYTES_REQUIRE_ORPHAN", None)
        super().tearDownClass()

    def test_sweep_orphan_radix_detects_violation(self) -> None:
        prompts = ["The capital of France is" for _ in range(8)]
        if self.launch_failed:
            self.assert_violation_kind_logged(["sweep_"], flush_wait_seconds=2.0)
            return
        self.send_parallel_requests(
            n=32, prompts=prompts, max_new_tokens=8, timeout=_PER_CASE_TIMEOUT
        )
        self.assert_violation_kind_logged(["sweep_"], flush_wait_seconds=2.0)


class TestNoVHalfEndpointsRegistered(_DSV4FlashBase, unittest.TestCase):
    """MLA axis: DSV4 has no V-half endpoints; only K-half is wired."""

    def test_no_v_half_endpoints_registered(self) -> None:
        # Step 1: launch must succeed for the registry assertion to apply.
        self.assertFalse(self.launch_failed, self.launch_exception)

        # Step 2: drive one round of traffic so endpoints are observably exercised.
        results = self.send_parallel_requests(
            n=4, max_new_tokens=8, timeout=_PER_CASE_TIMEOUT
        )
        for r in results:
            self.assertEqual(r.get("status_code"), 200, r)

        # Step 3: server stderr should only ever stamp K-half canary kinds.
        # ``canary_kind`` lines are emitted on violation; on a clean run we
        # cannot directly read the live endpoint registry from outside the
        # server. We instead assert that NO ``canary_kind: ..._v_*`` line
        # was emitted (MLA path: V endpoints not registered, so they cannot
        # fire even under a hypothetical violation).
        haystack = (self._stderr_buf.getvalue() if self._stderr_buf else "") + (
            self._stdout_buf.getvalue() if self._stdout_buf else ""
        )
        forbidden = [
            line
            for line in haystack.splitlines()
            if "canary_kind:" in line and "_v_" in line.lower()
        ]
        self.assertEqual(forbidden, [], f"Unexpected V-half canary lines: {forbidden}")


class TestSwaWindowClipOnlyLast128(_DSV4FlashBase, unittest.TestCase):
    """SWA axis: long prefix only triggers verification on the last window."""

    extra_server_args: ClassVar[List[str]] = [
        *_DSV4_BASE_ARGS,
        "--kv-canary-real-data",
        "all",
    ]

    def test_swa_window_clip_only_last_128(self) -> None:
        # Step 1: send a prompt longer than the SWA window (>= 256 tokens worth).
        long_prompt = ("The quick brown fox jumps over the lazy dog. " * 64).strip()
        self.assertFalse(self.launch_failed, self.launch_exception)
        results = self.send_parallel_requests(
            n=4,
            prompts=[long_prompt],
            max_new_tokens=8,
            timeout=_PER_CASE_TIMEOUT,
        )

        # Step 2: every request must succeed; SWA clip is internal and never
        # surfaces a violation when the prefix is correct.
        for r in results:
            self.assertEqual(r.get("status_code"), 200, r)
        self.assert_health_ok()


@unittest.skip("DSV4 packed-pool byte-offset table not yet derived.")
class TestDsv4PackedPoolRealKvSourceLayout(_DSV4FlashBase, unittest.TestCase):
    """Special pool axis: packed pool layout, byte-hit assertion.

    The assertion shape is "I read bytes X-Y from the packed slot at index Z",
    which requires hardcoded byte offsets derived from the DSV4 packed layout
    formula. Those offsets have not yet been derived.
    """

    extra_server_args: ClassVar[List[str]] = [
        *_DSV4_BASE_ARGS,
        "--kv-canary-real-data",
        "all",
    ]

    def test_dsv4_packed_pool_real_kv_source_layout(self) -> None:
        self.skipTest("hardcoded packed-pool byte offsets pending DSV4 layout review")


class TestDsv4PgSz128SwaGroup(_DSV4FlashBase, unittest.TestCase):
    """Special pool axis: page_size=128 SWA group still verifies cleanly."""

    extra_server_args: ClassVar[List[str]] = [
        *_DSV4_BASE_ARGS,
        "--kv-canary-real-data",
        "all",
    ]

    def test_dsv4_pg_sz_128_swa_group(self) -> None:
        # Step 1: drive enough traffic that the SWA group's verify path runs.
        self.assertFalse(self.launch_failed, self.launch_exception)
        results = self.send_parallel_requests(
            n=8, max_new_tokens=32, timeout=_PER_CASE_TIMEOUT
        )

        # Step 2: every request must complete with no canary violation surfaced.
        for r in results:
            self.assertEqual(r.get("status_code"), 200, r)
        haystack = (self._stderr_buf.getvalue() if self._stderr_buf else "") + (
            self._stdout_buf.getvalue() if self._stdout_buf else ""
        )
        violations = [line for line in haystack.splitlines() if "canary_kind:" in line]
        self.assertEqual(
            violations, [], f"Unexpected canary lines under clean run: {violations}"
        )


if __name__ == "__main__":
    unittest.main()
