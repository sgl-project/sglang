"""DSpark speculative decoding on CPU (intel_amx): markov-head draft on the
synchronous (non-overlap) path, the draft kept eager. The default class runs
the static ragged-verify mode; the cap-accept class covers the per-request
(ragged) extend metadata path in the intel_amx backend. Mirrors
test/registered/core/test_basic_sanity_dspark.py at CPU-CI scale.
"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.basic_api_contract_kit import BasicAPIContractMixin
from sglang.test.kits.basic_decode_correctness_kit import BasicDecodeCorrectnessMixin
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Measured 261s all-green on a 40-core GNR socket (2 launches + 21 methods).
register_cpu_ci(est_time=320, suite="base-b-test-cpu")

TARGET_MODEL = "Qwen/Qwen3-14B"
DRAFT_MODEL = "deepseek-ai/dspark_qwen3_14b_block7"


class DSparkCPUBase(CustomTestCase):
    ragged_verify_mode = "static"

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            TARGET_MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                "intel_amx",
                "--speculative-algorithm",
                "DSPARK",
                "--speculative-draft-model-path",
                DRAFT_MODEL,
                "--disable-overlap-schedule",
                # CPU decode is compute-bound; a wider batch buys nothing here.
                "--max-running-requests",
                "8",
                "--context-length",
                "8192",
            ],
            env={"SGLANG_RAGGED_VERIFY_MODE": cls.ragged_verify_mode},
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestDSparkCPU(
    BasicAPIContractMixin,
    BasicDecodeCorrectnessMixin,
    GSM8KMixin,
    DSparkCPUBase,
):
    served_model_name = TARGET_MODEL
    model = TARGET_MODEL

    gsm8k_num_questions = 64
    gsm8k_accuracy_thres = 0.80
    gsm8k_accept_length_thres = 2.0


class TestDSparkCPUCapAccept(BasicDecodeCorrectnessMixin, DSparkCPUBase):
    """Cap-accept plans per-request verify budgets, so every decode step
    drives the ragged extend path; correctness probes are enough here."""

    ragged_verify_mode = "cap-accept"


if __name__ == "__main__":
    unittest.main()
