"""H200 CI: DeepSeek-V4-Flash NVFP4 through the SM90 FlashInfer NVFP4 MegaMoE.

Launches the DeepSeek-V4 cookbook SM90 NVFP4 MegaMoE command (TP=8, DP-attention 8,
EP=8, --moe-a2a-backend megamoe) and runs decode probes plus a GSM8K gate.

Registry: extra-b-test-8-gpu-h200 (label-gated, 8x H200)
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.basic_decode_correctness_kit import BasicDecodeCorrectnessMixin
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
    try_cached_model,
)

register_cuda_ci(est_time=900, stage="extra-b", runner_config="8-gpu-h200")


def _flashinfer_has_sm90_megamoe() -> bool:
    try:
        from flashinfer.megamoe import MegaMoESymmBuffer
    except ImportError:
        return False
    return hasattr(MegaMoESymmBuffer, "commit_staged")


MODEL = "nvidia/DeepSeek-V4-Flash-NVFP4"
SERVER_LAUNCH_TIMEOUT = 3600


@unittest.skipUnless(
    _flashinfer_has_sm90_megamoe(),
    "FlashInfer build lacks flashinfer.megamoe with "
    "flashinfer-ai/flashinfer#6343 and #6517",
)
class TestDSV4FlashNVFP4MegaMoEH200(
    BasicDecodeCorrectnessMixin, GSM8KMixin, CustomTestCase
):
    """SM90 FlashInfer NVFP4 W4A8 MegaMoE on the ModelOpt NVFP4 checkpoint."""

    gsm8k_backend = "sgl_eval"
    gsm8k_score_threshold = 0.93

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "8",
                "--dp",
                "8",
                "--enable-dp-attention",
                "--moe-a2a-backend",
                "megamoe",
                "--disable-shared-experts-fusion",
                "--chunked-prefill-size",
                "8192",
            ],
            env={"SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK": "4096"},
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)


if __name__ == "__main__":
    unittest.main()
