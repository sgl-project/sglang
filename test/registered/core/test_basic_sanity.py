"""Basic sanity: small-but-broad server smoke that downstream stages
depend on. Three sanity kits, one shared server, covering protocol
contract, decode correctness, and scheduler stress paths."""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.basic_api_contract_kit import BasicAPIContractMixin
from sglang.test.kits.basic_decode_correctness_kit import BasicDecodeCorrectnessMixin
from sglang.test.kits.basic_scheduler_stress_kit import BasicSchedulerStressMixin
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=120, stage="base-a", runner_config="1-gpu-small")
register_amd_ci(est_time=120, suite="stage-a-test-1-gpu-small-amd")


class TestBasicSanity(
    BasicAPIContractMixin,
    BasicDecodeCorrectnessMixin,
    BasicSchedulerStressMixin,
    CustomTestCase,
):
    served_model_name = DEFAULT_MODEL_NAME_FOR_TEST

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_MODEL_NAME_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--cuda-graph-max-bs",
                "4",
                "--mem-fraction-static",
                "0.7",
                "--enable-metrics",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_accuracy_floor(self):
        # Stage-a-private accuracy guard: hellaswag via the frontend DSL
        # bound to this server. Catches systematic regressions that pass
        # every cheap probe in the mixed-in kits but tank multi-choice
        # reasoning. Not part of any reusable mixin -- accuracy gating
        # is the gate test's own responsibility.
        import sglang as sgl
        from sglang.test.test_programs import test_hellaswag_select

        sgl.set_default_backend(sgl.RuntimeEndpoint(self.base_url))
        try:
            accuracy, _ = test_hellaswag_select()
        finally:
            sgl.set_default_backend(None)
        self.assertGreater(
            accuracy,
            0.60,
            f"hellaswag accuracy floor breached: {accuracy:.3f}",
        )


if __name__ == "__main__":
    unittest.main()
