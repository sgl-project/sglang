"""CPU tests for FlashInfer MXFP4 CUTLASS 0-token early return."""

import unittest
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
    FlashInferCutlassMxfp4MoeQuantInfo,
    _fused_experts_flashinfer_mxfp4_cutlass,
)
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_KERNEL = (
    "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass._flashinfer_cutlass_fused_moe"
)


def _dispatch(num_tokens: int, hidden: int = 256) -> StandardDispatchOutput:
    return StandardDispatchOutput(
        hidden_states=torch.empty(num_tokens, hidden, dtype=torch.bfloat16),
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=torch.empty(num_tokens, 4),
            topk_ids=torch.empty(num_tokens, 4, dtype=torch.int32),
            router_logits=torch.empty(num_tokens, 8),
        ),
    )


def _quant_info() -> FlashInferCutlassMxfp4MoeQuantInfo:
    return FlashInferCutlassMxfp4MoeQuantInfo(
        w13_weight=torch.empty(0),
        w2_weight=torch.empty(0),
        w13_weight_scale=torch.empty(0),
        w2_weight_scale=torch.empty(0),
    )


class TestFlashInferMxfp4EmptyTokens(CustomTestCase):
    def test_empty_tokens_skip_kernel(self):
        dispatch = _dispatch(0)
        with patch(_KERNEL) as mock_kernel:
            mock_kernel.side_effect = RuntimeError("kernel should not be called")
            result = _fused_experts_flashinfer_mxfp4_cutlass(
                dispatch, _quant_info(), MoeRunnerConfig()
            )
        mock_kernel.assert_not_called()
        self.assertIs(result.hidden_states, dispatch.hidden_states)

    def test_nonzero_tokens_reach_kernel(self):
        with patch(_KERNEL) as mock_kernel:
            mock_kernel.side_effect = RuntimeError("kernel reached")
            with self.assertRaisesRegex(RuntimeError, "kernel reached"):
                _fused_experts_flashinfer_mxfp4_cutlass(
                    _dispatch(1), _quant_info(), MoeRunnerConfig()
                )


if __name__ == "__main__":
    unittest.main()
