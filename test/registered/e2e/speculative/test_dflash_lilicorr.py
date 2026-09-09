"""End-to-end serving of a DFLASH drafter carrying a LiLiCorr reranker head.

The head is selected by the checkpoint declaring
``architectures: ["LiLiCorrDraftModel"]``, so the launch arguments below are the
plain DFLASH ones with only the draft path changed. That is the first thing this
test pins: if LiLiCorr ever needed a flag, this server would not come up.

GSM8K is the losslessness check rather than a quality measurement. Verify is
unmodified, so a reranked drafter cannot change which tokens are emitted, only
how many are accepted per step; a score below the target's own means the change
leaked into the output path. The accept-length floor is the complementary check,
that the head actually loaded instead of silently serving as its head-free
parent.
"""

import unittest
from contextlib import ExitStack

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.spec_decoding_kit import SpecDecodingMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(
    est_time=300,
    stage="base-b",
    runner_config="1-gpu-large",
    disabled="no LiLiCorr draft checkpoint is published yet",
)

# The target is public; the drafter is not. Kept here rather than in test_utils.py
# because an unpublished id is not something other tests should reach for, and a
# non-repo string fails loudly if this test is ever enabled before it is filled in.
# Set it and drop the `disabled=` argument above in the same change.
TARGET_MODEL = "Qwen/Qwen3-8B"
DRAFT_MODEL = "<unpublished>"


class TestLiLiCorrServer(CustomTestCase, GSM8KMixin, SpecDecodingMixin):
    model = TARGET_MODEL
    draft_model = DRAFT_MODEL
    # fa3 opts out of the host seq_lens mirror; a backend that does not pays a
    # device-to-host sync per block, and backends differ in attention numerics,
    # so the accept length below is only comparable on this one.
    attention_backend = "fa3"
    # Both thresholds are floors under a measured value, not targets. On an H100 this
    # eval scores 0.955 with a standard error of about 0.015 over its 200 questions.
    gsm8k_accuracy_thres = 0.85
    # Measured on this eval, same target and hardware: 5.07 with the head against 4.17
    # for the head-free DFLASH control. The floor sits between them, so a head that
    # failed to load and served as its head-free parent fails rather than reporting a
    # believable number. Note this is the mixin's own accept length over 200 short
    # 5-shot completions, which is not the same quantity as block-weighted acceptance
    # over long generations -- do not copy a threshold between the two.
    gsm8k_accept_length_thres = 4.6
    # `test_bs_1_speed` is one prompt at 2048 tokens, so it reports a third accept
    # length again: 8.19 with the head against 7.31 for the head-free control. It is
    # bit-reproducible here (greedy, batch of one), so the floor can sit close.
    accept_length_thres = 7.7
    # Measured 883.8 tok/s against the control's 788.0 on an H100. Throughput is not
    # reproducible, so this floor is deliberately loose and only catches a collapse.
    bs_1_speed_thres = 600.0

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        with ExitStack() as stack:
            stack.enter_context(envs.SGLANG_ENABLE_ASYNC_ASSERT.override(True))
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--attention-backend",
                    cls.attention_backend,
                    "--speculative-algorithm",
                    "DFLASH",
                    "--speculative-draft-model-path",
                    cls.draft_model,
                    "--mem-fraction-static",
                    "0.7",
                ],
            )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
