"""End-to-end serving of a DFLASH drafter carrying a LiLiCorr reranker head.

Two black-box behaviours must not come back: LiLiCorr needing a launch flag, since the
args below are plain DFLASH with only the draft path changed; and the head loading
silently as its head-free parent, which the accept-length floor catches. The GSM8K score
is the losslessness check, not a quality measurement.
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
    stage="extra-b",
    runner_config="4-gpu-h100",
    disabled="no LiLiCorr draft checkpoint is published yet",
)

# The drafter is unpublished. Set it and drop the `disabled=` argument above in the same
# change; the non-repo string fails loudly if the test is enabled before it is filled in.
TARGET_MODEL = "Qwen/Qwen3-8B"
DRAFT_MODEL = "<unpublished>"


class TestLiLiCorrServer(CustomTestCase, GSM8KMixin, SpecDecodingMixin):
    model = TARGET_MODEL
    draft_model = DRAFT_MODEL
    # Backends differ in attention numerics, and one publishing seq_lens_cpu costs a
    # device-to-host sync per block, so the floors below hold on fa3 only.
    attention_backend = "fa3"
    # Every threshold is a floor under an H100 measurement. gsm8k: 0.955 scored, se ~0.015
    # over 200 questions.
    gsm8k_accuracy_thres = 0.85
    # 5.07 with the head against 4.17 for the head-free DFLASH control, so a head that
    # failed to load fails here. This is the mixin's accept length over 200 short 5-shot
    # completions, not block-weighted acceptance; do not copy a threshold between the two.
    gsm8k_accept_length_thres = 4.6
    # One prompt at 2048 tokens: 8.19 with the head against 7.31 for the control.
    accept_length_thres = 7.7
    # 883.8 tok/s against the control's 788.0. Throughput is not reproducible, so this
    # floor only catches a collapse.
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
