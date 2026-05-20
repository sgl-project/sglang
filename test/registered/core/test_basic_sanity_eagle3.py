"""Basic sanity for stage-a with EAGLE3 spec decoding. Mirrors
test_basic_sanity.py but launches the same target model with EAGLE3
draft so the gate covers the spec-decoding path."""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.basic_api_contract_kit import BasicAPIContractMixin
from sglang.test.kits.basic_decode_correctness_kit import BasicDecodeCorrectnessMixin
from sglang.test.kits.basic_scheduler_stress_kit import BasicSchedulerStressMixin
from sglang.test.kits.fwd_occupancy_kit import FwdOccupancyMixin
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE3,
    DEFAULT_TARGET_MODEL_EAGLE3,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=200, stage="base-a", runner_config="1-gpu-small")
register_amd_ci(est_time=200, suite="stage-a-test-1-gpu-small-amd")


class TestBasicSanityEagle3(
    BasicAPIContractMixin,
    BasicDecodeCorrectnessMixin,
    BasicSchedulerStressMixin,
    FwdOccupancyMixin,
    CustomTestCase,
):
    served_model_name = DEFAULT_TARGET_MODEL_EAGLE3
    # EAGLE3 spec decoding does more GPU work per wall-clock step than
    # vanilla decode -- threshold can stay tight. Start at 97 like the
    # vanilla gate; adjust per CI calibration.
    fwd_occupancy_threshold = 97.0

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_TARGET_MODEL_EAGLE3,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--speculative-algorithm",
                "EAGLE3",
                "--speculative-draft-model-path",
                DEFAULT_DRAFT_MODEL_EAGLE3,
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
                "--cuda-graph-max-bs",
                "4",
                "--mem-fraction-static",
                "0.7",
                "--enable-metrics",
            ],
            env={"SGLANG_ENABLE_METRICS_DEVICE_TIMER": "1"},
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_accuracy_floor(self):
        # Stage-a's own accuracy gate -- catches systematic regressions
        # that pass every cheap probe but tank multi-choice reasoning.
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
