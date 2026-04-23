import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.deep_gemm_mega import (
    DeepGemmMegaMoeQuantInfo,
    fused_experts_none_to_deep_gemm_mega_mxfp4,
)


class TestDeepGemmMegaMxfp4(unittest.TestCase):
    def test_rejects_mismatched_packed_activation_scales(self):
        num_tokens = 3
        hidden = 128
        dispatch_output = SimpleNamespace(
            hidden_states=torch.zeros(num_tokens, hidden, dtype=torch.bfloat16),
            topk_output=SimpleNamespace(
                topk_ids=torch.zeros(num_tokens, 2, dtype=torch.int32),
                topk_weights=torch.zeros(num_tokens, 2, dtype=torch.float32),
            ),
        )
        runtime = SimpleNamespace(
            max_num_tokens_per_rank=8,
            symm_buffer=SimpleNamespace(
                x=torch.zeros(8, hidden, dtype=torch.float32),
                x_sf=torch.zeros(8, 1, dtype=torch.int32),
                topk_idx=torch.zeros(8, 2, dtype=torch.int64),
                topk_weights=torch.zeros(8, 2, dtype=torch.float32),
            ),
            transformed_l1_weights=None,
            transformed_l2_weights=None,
        )
        quant_info = DeepGemmMegaMoeQuantInfo(runtime=runtime)
        runner_config = MoeRunnerConfig(activation="silu", is_gated=True)

        with mock.patch(
            "sglang.srt.layers.moe.topk.TopKOutputChecker.format_is_standard",
            return_value=True,
        ), mock.patch(
            "sglang.srt.layers.moe.moe_runner.deep_gemm_mega.sglang_per_token_group_quant_fp8",
            return_value=(
                torch.zeros(num_tokens, hidden, dtype=torch.float32),
                torch.zeros(num_tokens, 2, dtype=torch.int32),
            ),
        ):
            with self.assertRaisesRegex(
                ValueError, "expected packed UE8M0 activation scales"
            ):
                fused_experts_none_to_deep_gemm_mega_mxfp4(
                    dispatch_output, quant_info, runner_config
                )


if __name__ == "__main__":
    unittest.main()
