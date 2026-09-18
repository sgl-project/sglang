import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.kl_divergence_kit import KLDivergenceMixin
from sglang.test.kits.prefix_cache_branching_kit import PrefixCacheBranchingMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(est_time=463, stage="base-c", runner_config="4-gpu-h100")

QWEN3_NEXT_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"

_COMMON_ARGS = [
    "--trust-remote-code",
    "--tp-size",
    "4",
    "--chunked-prefill-size",
    "2048",
    "--mamba-radix-cache-strategy",
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


class TestQwen3NextPrefillCP(GSM8KMixin, DefaultServerBase):
    model = QWEN3_NEXT_MODEL
    gsm8k_num_examples = 40
    gsm8k_num_threads = 1
    gsm8k_score_threshold = 0.9
    other_args = [
        "--trust-remote-code",
        "--tp-size",
        "4",
        "--enable-prefill-cp",
        "--attn-cp-size",
        "2",
        "--cp-strategy",
        "zigzag",
    ]


if __name__ == "__main__":
    unittest.main()
