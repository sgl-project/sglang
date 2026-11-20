"""Non-power-of-two LoRA ranks on the experimental TRT-LLM virtual-experts path.

The stock path is covered by test_virtual_experts_kernels.py. Its mirror keeps
a separate stage-config fallback, and a LoRA rank such as 24 lands on both sides
of a stage: `N` for the shrink weight and `K` for the expand weight. Triton
requires a power-of-two `tl.arange` range, so a tile capped at the raw rank
fails to compile. The kernels mask or wrap the padded rows/columns, so rounding
the tile is enough.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.utils import get_device, is_cuda
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")


class TestTrtllmVirtualExpertFallbackRanks(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not is_cuda():
            raise unittest.SkipTest("CUDA Triton kernels required")
        cls.device = get_device(0)

    @unittest.skipUnless(is_cuda(), "CUDA Triton kernels required")
    def test_fallback_accepts_non_power_of_two_ranks(self):
        import sglang.kernels.ops.lora.moe.trtllm_lora_temp.virtual_experts as temp
        from sglang.srt.layers.moe.moe_runner.triton_utils import (
            fused_moe_triton_config,
        )

        hidden, width, experts = 16, 64, 4
        # Tuning has no entry for these shapes; the fallback config is what
        # reached the kernel before the fix, and it capped the expand tile at
        # the raw rank.
        with patch.object(
            fused_moe_triton_config,
            "try_get_optimal_moe_config",
            side_effect=ValueError,
        ):
            for rank in (12, 16, 24, 40, 48):
                with self.subTest(rank=rank):
                    x = torch.ones(1, hidden, device=self.device, dtype=torch.float16)
                    a = torch.ones(
                        1, experts, rank, hidden, device=self.device, dtype=x.dtype
                    )
                    b = torch.ones(
                        1, experts, width, rank, device=self.device, dtype=x.dtype
                    )
                    out = torch.zeros(1, 1, width, device=self.device, dtype=x.dtype)
                    route = torch.zeros(1, 1, dtype=torch.int32, device=self.device)
                    temp.merged_experts_fused_moe_lora_add(
                        out,
                        x,
                        a,
                        b,
                        route,
                        torch.ones_like(route, dtype=torch.float32),
                        torch.zeros(1, dtype=torch.int32, device=self.device),
                        mul_routed_weight=True,
                        experts_shared_outer_loras_a=False,
                        experts_shared_outer_loras_b=False,
                        adapter_enabled=torch.ones(
                            1, dtype=torch.int32, device=self.device
                        ),
                    )
                    # Ones through both factors: `y = x @ A^T @ B^T` sums
                    # `hidden` over the rank axis.
                    torch.testing.assert_close(
                        out,
                        torch.full_like(out, float(rank * hidden)),
                        rtol=0,
                        atol=0,
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
