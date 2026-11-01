import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.lora_utils import (
    CI_MULTI_LORA_MODELS,
    run_lora_batch_splitting_equivalence_test,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=59, stage="extra-a", runner_config="1-gpu-small")
register_amd_ci(est_time=100, suite="extra-a-test-1-gpu-small-amd")


class TestLoRADrainer(CustomTestCase):
    def test_batch_splitting_with_drainer(self):
        run_lora_batch_splitting_equivalence_test(
            model_cases=CI_MULTI_LORA_MODELS,
            attention_backend="torch_native",
            disable_cuda_graph=True,
            disable_radix_cache=True,
            lora_drain_wait_threshold=3.0,
        )


if __name__ == "__main__":
    unittest.main()
