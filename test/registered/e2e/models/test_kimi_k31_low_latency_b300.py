"""B300 CI coverage for the Kimi-K3.1 Low Latency recipe.

Runs the TP8/DCP8 DFLASH recipe (block size 6) on eight B300 GPUs and checks
MMMU-Pro quality, speculative acceptance, and single-request decode speed.
The recipe's decode role runs colocated here: PD needs a second 8-GPU node.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import MMMUProMixin
from sglang.test.kits.spec_decoding_kit import SpecDecodingMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    _wait_for_gpu_idle_in_ci,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=1533, stage="extra-b", runner_config="8-gpu-b300")

MODEL_PATH = "moonshotai/Kimi-K3.1-Pro"
DFLASH_DRAFT_MODEL = "RadixArk/Kimi-K3.1-Dflash2-Preview"
MODEL_LOADER_EXTRA_CONFIG = '{"enable_multithread_load": true}'
SERVER_LAUNCH_TIMEOUT = 3600
GPU_IDLE_TIMEOUT = 120


class TestKimiK31LowLatencyB300(MMMUProMixin, SpecDecodingMixin, CustomTestCase):
    """TP8/DCP8 Low Latency recipe with DFLASH linear ReplaySSM speculation."""

    mmmu_pro_score_threshold = 0.75
    mmmu_pro_num_examples = 200
    # sgl-eval has no Kimi-K3.1 preset yet; K3.1 shares K3's chat template and
    # recommended sampling, and chat completions ignore the request model name.
    mmmu_pro_load_preset_from_model_id = "moonshotai/Kimi-K3"
    accept_length_thres = 4.0
    bs_1_speed_thres = 300

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "8",
                "--dcp-size",
                "8",
                "--mem-fraction-static",
                "0.92",
                "--max-running-requests",
                "72",
                "--weight-loader-prefetch-checkpoints",
                "--model-loader-extra-config",
                MODEL_LOADER_EXTRA_CONFIG,
                "--reasoning-parser",
                "kimi_k3",
                "--tool-call-parser",
                "kimi_k3",
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-draft-model-path",
                DFLASH_DRAFT_MODEL,
                "--speculative-dflash-block-size",
                "6",
                "--speculative-draft-kv-cache-dtype",
                "fp8_e4m3",
                "--speculative-draft-attention-backend",
                "cutedsl_mla",
                "--enable-linear-replayssm-spec",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)
            _wait_for_gpu_idle_in_ci(timeout=GPU_IDLE_TIMEOUT)


if __name__ == "__main__":
    unittest.main()
