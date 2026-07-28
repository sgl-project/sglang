import os
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.kl_divergence_kit import KLDivergenceMixin
from sglang.test.kits.prefix_cache_branching_kit import PrefixCacheBranchingMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(est_time=430, stage="base-c", runner_config="4-gpu-h100")

QWEN3_NEXT_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"


def _mtp_args(*, strategy, steps, topk, draft_tokens, track_interval):
    return [
        "--trust-remote-code",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        str(steps),
        "--speculative-eagle-topk",
        str(topk),
        "--speculative-num-draft-tokens",
        str(draft_tokens),
        "--mem-fraction-static",
        "0.8",
        "--tp",
        "4",
        "--chunked-prefill-size",
        "2048",
        "--mamba-scheduler-strategy",
        strategy,
        "--mamba-track-interval",
        str(track_interval),
    ]


class TestQwen3NextMTPTopk(
    GSM8KMixin, KLDivergenceMixin, PrefixCacheBranchingMixin, DefaultServerBase
):
    # topk > 1 (tree) MTP on a hybrid-GDN model, on spec v2: the tree-aware mamba
    # state update lives in the spec v2 verify path, so mamba + topk > 1 no longer
    # falls back to spec v1.
    model = QWEN3_NEXT_MODEL
    cache_chunk_size = 64
    gsm8k_accuracy_thres = 0.93
    kl_div_thres = 0.008
    other_args = [
        "--trust-remote-code",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "5",
        "--speculative-eagle-topk",
        "4",
        "--speculative-num-draft-tokens",
        "8",
        "--mem-fraction-static",
        "0.8",
        "--tp",
        "4",
        "--chunked-prefill-size",
        "2048",
        "--mamba-scheduler-strategy",
        "extra_buffer",
        "--mamba-track-interval",
        "128",
    ]


class TestQwen3NextMTPV2(GSM8KMixin, KLDivergenceMixin, DefaultServerBase):
    model = QWEN3_NEXT_MODEL
    gsm8k_accuracy_thres = 0.93
    kl_div_thres = 0.0035
    other_args = [
        "--trust-remote-code",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--mem-fraction-static",
        "0.8",
        "--tp",
        "4",
        "--chunked-prefill-size",
        "2048",
        "--mamba-scheduler-strategy",
        "extra_buffer",
        "--mamba-track-interval",
        "128",
    ]


class TestQwen3NextMTPLazyV2(
    GSM8KMixin, KLDivergenceMixin, PrefixCacheBranchingMixin, DefaultServerBase
):
    # extra_buffer_lazy + MTP: the pending-slot track plan
    # (mamba_lazy_spec_prepare / _mamba_lazy_spec_confirm_crossing) replaces
    # the per-req second ping-pong slot. Branching mixin exercises donation
    # of lazily tracked states.
    model = QWEN3_NEXT_MODEL
    cache_chunk_size = 64
    gsm8k_accuracy_thres = 0.93
    kl_div_thres = 0.0035
    other_args = _mtp_args(
        strategy="extra_buffer_lazy",
        steps=3,
        topk=1,
        draft_tokens=4,
        track_interval=128,
    )


@unittest.skip("Manual-only: forces all lazy spec preallocs to fail")
class TestQwen3NextMTPLazyAllocFail(GSM8KMixin, KLDivergenceMixin, DefaultServerBase):
    # Small track interval (= draft tokens) maximizes boundary crossings;
    # forced alloc failure exercises the in-place fallback and the
    # finished-req skip-insert path on every crossing.
    model = QWEN3_NEXT_MODEL
    cache_chunk_size = 64
    gsm8k_accuracy_thres = 0.93
    kl_div_thres = 0.0035
    other_args = _mtp_args(
        strategy="extra_buffer_lazy",
        steps=3,
        topk=1,
        draft_tokens=4,
        track_interval=4,
    )

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
