import unittest

from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.kl_divergence_kit import KLDivergenceMixin
from sglang.test.kits.prefix_cache_branching_kit import PrefixCacheBranchingMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

QWEN3_NEXT_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"


class TestQwen3Next(
    GSM8KMixin, KLDivergenceMixin, PrefixCacheBranchingMixin, DefaultServerBase
):
    model = QWEN3_NEXT_MODEL
    cache_chunk_size = 64
    gsm8k_accuracy_thres = 0.93
    kl_div_thres = 0.0025
    other_args = [
        "--tp-size",
        "4",
        "--chunked-prefill-size",
        "1024",
        "--mamba-scheduler-strategy",
        "extra_buffer",
        "--mamba-track-interval",
        "2",
        "--page-size",
        "1",
        "--attention-backend",
        "triton",
        "--moe-runner-backend",
        "triton",
    ]


class TestQwen3NextLazyExtraBuffer(
    GSM8KMixin, KLDivergenceMixin, PrefixCacheBranchingMixin, DefaultServerBase
):
    model = QWEN3_NEXT_MODEL
    cache_chunk_size = 64
    gsm8k_accuracy_thres = 0.93
    kl_div_thres = 0.0025
    other_args = [
        "--tp-size",
        "4",
        "--chunked-prefill-size",
        "1024",
        "--mamba-scheduler-strategy",
        "extra_buffer_lazy",
        "--mamba-track-interval",
        "2",
        "--page-size",
        "1",
        "--attention-backend",
        "triton",
        "--moe-runner-backend",
        "triton",
    ]


if __name__ == "__main__":
    unittest.main()
