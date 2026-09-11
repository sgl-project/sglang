import unittest
from types import SimpleNamespace
from typing import ClassVar

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST

register_cuda_ci(est_time=900, stage="base-b", runner_config="2-gpu-large")

# Matches the non-pipelined PD Llama baseline in test_disaggregation_basic.py.
GSM8K_ACCURACY_FLOOR = 0.62
# Force a fixed group size so pipelining actually engages: gsm8k prompts fall
# below SGLANG_PIPELINE_MIN_TOKENS, so the adaptive sizer would otherwise skip
# pipelining and the "enabled" run would silently reduce to the baseline path.
PIPELINE_GROUP_SIZE = "8"


class _GSM8KAccuracyMixin:
    """Shared launch + gsm8k eval; concrete classes flip ``pipeline_enabled``."""

    pipeline_enabled: ClassVar[bool] = False

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        if cls.pipeline_enabled:
            cls.extra_prefill_env = {
                "SGLANG_ENABLE_PIPELINED_KV_TRANSFER": "1",
                "SGLANG_PIPELINE_GROUP_SIZE": PIPELINE_GROUP_SIZE,
            }
        cls.launch_all()

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=f"http://{self.base_host}:{self.lb_port}",
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"pipeline_enabled={self.pipeline_enabled} metrics: {metrics}")
        self.assertGreater(metrics["score"], GSM8K_ACCURACY_FLOOR)


class TestPipelinedKVTransferDisabled(
    _GSM8KAccuracyMixin, PDDisaggregationServerBase
):
    pipeline_enabled = False


class TestPipelinedKVTransferEnabled(
    _GSM8KAccuracyMixin, PDDisaggregationServerBase
):
    pipeline_enabled = True


if __name__ == "__main__":
    unittest.main()
