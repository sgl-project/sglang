"""DFLASH speculative decoding on CPU (intel_amx): block-diffusion draft with
chain verify on the synchronous (non-overlap) path, the draft kept eager.
Mirrors test/registered/core/test_basic_sanity_dflash.py at CPU-CI scale.
"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.basic_api_contract_kit import BasicAPIContractMixin
from sglang.test.kits.basic_decode_correctness_kit import BasicDecodeCorrectnessMixin
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_DFLASH,
    DEFAULT_TARGET_MODEL_DFLASH,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Measured 287s all-green on a 40-core GNR socket (1 launch + 14 methods).
register_cpu_ci(
    est_time=350,
    suite="base-b-test-cpu",
    disabled="needs gated meta-llama/Llama-3.1-8B-Instruct",
)


class TestDFlashCPU(
    BasicAPIContractMixin,
    BasicDecodeCorrectnessMixin,
    GSM8KMixin,
    CustomTestCase,
):
    served_model_name = DEFAULT_TARGET_MODEL_DFLASH
    model = DEFAULT_TARGET_MODEL_DFLASH

    gsm8k_num_questions = 200
    gsm8k_accuracy_thres = 0.65
    # CPU accept must match CUDA: same greedy chain accept, same draft.
    gsm8k_accept_length_thres = 2.0

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_TARGET_MODEL_DFLASH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                "intel_amx",
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-draft-model-path",
                DEFAULT_DRAFT_MODEL_DFLASH,
                "--disable-overlap-schedule",
                # CPU decode is compute-bound; a wider batch buys nothing here.
                "--max-running-requests",
                "8",
                # The z-lab draft config derives a 40960 context; stay under it.
                "--context-length",
                "8192",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
