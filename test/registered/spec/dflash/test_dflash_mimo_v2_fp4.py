import os
import unittest

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=1800, suite="nightly-8-gpu-common", nightly=True)

# Public checkpoints: target = https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash
# (the FP4 mxfp4 target at the repo root), draft = the repo's `dflash/` subfolder. Because
# the draft is a subfolder (not its own repo id), download the repo and point
# MIMO_V2_DFLASH_DRAFT at the local `<repo>/dflash` path:
#   hf download XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash --local-dir /models/mimo-fp4-dflash
#   export MIMO_V2_FP4_TARGET=/models/mimo-fp4-dflash
#   export MIMO_V2_DFLASH_DRAFT=/models/mimo-fp4-dflash/dflash
MIMO_V2_FP4_TARGET = os.environ.get(
    "MIMO_V2_FP4_TARGET", "XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash"
)
MIMO_V2_DFLASH_DRAFT = os.environ.get("MIMO_V2_DFLASH_DRAFT", "")


@unittest.skipUnless(
    MIMO_V2_DFLASH_DRAFT,
    "Set MIMO_V2_DFLASH_DRAFT to the local DFlash draft subfolder "
    "(<repo>/dflash) to run. Needs a 16-GPU (TP16/DP2) runner.",
)
class TestMiMoV2FP4DFlash(CustomTestCase, GSM8KMixin):
    """GSM8K + accept-length check for the FP4 MiMo-V2-Pro target with its DFlash draft.

    Mirrors the internal v0.5.10 deployment eval (TP16/DP2/EP16, dp-attention, fa3),
    which scored AIME25 28/30 at accept-length ~4. The 1T FP4 target cannot fit a
    single device, so this needs a 16-GPU (TP16/DP2 => attn_tp=8) runner.
    """

    model = MIMO_V2_FP4_TARGET
    draft_model = MIMO_V2_DFLASH_DRAFT
    gsm8k_accuracy_thres = 0.85
    gsm8k_accept_length_thres = 3.5

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        launch_args = [
            "--trust-remote-code",
            "--speculative-algorithm",
            "DFLASH",
            "--speculative-draft-model-path",
            cls.draft_model,
            "--speculative-num-draft-tokens",
            "8",
            "--quantization",
            "fp8",
            "--attention-backend",
            "fa3",
            "--tensor-parallel-size",
            "16",
            "--data-parallel-size",
            "2",
            "--ep-size",
            "16",
            "--enable-dp-attention",
            "--enable-dp-lm-head",
            "--moe-dense-tp-size",
            "1",
            "--reasoning-parser",
            "qwen3",
            "--tool-call-parser",
            "mimo",
            "--page-size",
            "1",
            "--disable-overlap-schedule",
        ]
        old_value = os.environ.get("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN")
        os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
        try:
            with envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(1):
                cls.process = popen_launch_server(
                    cls.model,
                    cls.base_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=launch_args,
                )
        finally:
            if old_value is None:
                del os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"]
            else:
                os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = old_value

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
