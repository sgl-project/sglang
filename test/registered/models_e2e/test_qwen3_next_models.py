import os
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.kl_divergence_kit import KLDivergenceMixin
from sglang.test.kits.prefix_cache_branching_kit import PrefixCacheBranchingMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(est_time=296, stage="base-c", runner_config="4-gpu-h100")

QWEN3_NEXT_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"

_COMMON_ARGS = [
    "--trust-remote-code",
    "--tp-size",
    "4",
    "--chunked-prefill-size",
    "2048",
    "--mamba-scheduler-strategy",
    "extra_buffer_lazy",
    "--attention-backend",
    "triton",
]


def _make_args(*, page_size=1, track_interval=2):
    return [
        *_COMMON_ARGS,
        "--mamba-track-interval",
        str(track_interval),
        "--page-size",
        str(page_size),
    ]


class TestQwen3NextLazyExtraBuffer(
    GSM8KMixin, KLDivergenceMixin, PrefixCacheBranchingMixin, DefaultServerBase
):
    model = QWEN3_NEXT_MODEL
    cache_chunk_size = 64
    gsm8k_accuracy_thres = 0.93
    kl_div_thres = 0.002
    other_args = _make_args(page_size=1, track_interval=2)


class TestQwen3NextLazyExtraBufferLargePage(
    GSM8KMixin, KLDivergenceMixin, PrefixCacheBranchingMixin, DefaultServerBase
):
    model = QWEN3_NEXT_MODEL
    cache_chunk_size = 64
    gsm8k_accuracy_thres = 0.93
    kl_div_thres = 0.002
    other_args = _make_args(page_size=2, track_interval=2)


@unittest.skip("Manual-only: forces all lazy preallocs to fail")
class TestQwen3NextLazyExtraBufferAllocFail(
    GSM8KMixin, KLDivergenceMixin, PrefixCacheBranchingMixin, DefaultServerBase
):
    model = QWEN3_NEXT_MODEL
    cache_chunk_size = 64
    gsm8k_accuracy_thres = 0.93
    kl_div_thres = 0.002
    other_args = _make_args(page_size=1, track_interval=2)

    @classmethod
    def setUpClass(cls):
        os.environ["SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL"] = "1"
        os.environ["SGLANG_TEST_SKIP_CACHE_HIT_ASSERT"] = "1"
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        os.environ.pop("SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL", None)
        os.environ.pop("SGLANG_TEST_SKIP_CACHE_HIT_ASSERT", None)


@unittest.skip("Manual-only: forces all lazy preallocs to fail")
class TestQwen3NextLazyExtraBufferLargePageAllocFail(
    GSM8KMixin, KLDivergenceMixin, PrefixCacheBranchingMixin, DefaultServerBase
):
    model = QWEN3_NEXT_MODEL
    cache_chunk_size = 64
    gsm8k_accuracy_thres = 0.93
    kl_div_thres = 0.002
    other_args = _make_args(page_size=2, track_interval=2)

    @classmethod
    def setUpClass(cls):
        os.environ["SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL"] = "1"
        os.environ["SGLANG_TEST_SKIP_CACHE_HIT_ASSERT"] = "1"
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        os.environ.pop("SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL", None)
        os.environ.pop("SGLANG_TEST_SKIP_CACHE_HIT_ASSERT", None)


if __name__ == "__main__":
    unittest.main()
