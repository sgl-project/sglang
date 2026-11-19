"""FP8 block-alignment check for TP-sharded routed experts (check_quantized_moe_compatibility)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

from sglang.srt.model_executor.model_runner_components.moe_ep_setup import (
    check_quantized_moe_compatibility,
)
from sglang.test.test_utils import CustomTestCase


def _model_config(*, is_fp4_experts):
    # Ling-3.0-flash-VL-fp4 shape: FP8 128x128 dense blocks, 768-wide routed experts.
    return SimpleNamespace(
        hf_config=SimpleNamespace(
            quantization_config={"weight_block_size": [128, 128]}
        ),
        hf_text_config=SimpleNamespace(moe_intermediate_size=768),
        is_fp4_experts=is_fp4_experts,
    )


class TestQuantizedMoeBlockCheck(CustomTestCase):
    def test_mxfp4_experts_accept_a_128_unaligned_tp_shard(self):
        """MXFP4 routed experts (32-wide scale groups) were rejected at TP4 because
        768 / 4 = 192 is not a multiple of the FP8 block size."""
        check_quantized_moe_compatibility(
            model_config=_model_config(is_fp4_experts=True),
            tp_size=4,
            moe_ep_size=1,
            moe_dp_size=1,
        )

    def test_fp8_experts_still_reject_a_128_unaligned_tp_shard(self):
        with self.assertRaisesRegex(ValueError, "weight_block_size_n"):
            check_quantized_moe_compatibility(
                model_config=_model_config(is_fp4_experts=False),
                tp_size=4,
                moe_ep_size=1,
                moe_dp_size=1,
            )


if __name__ == "__main__":
    unittest.main()
