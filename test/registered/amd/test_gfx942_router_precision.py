"""Router logits must be repeatable before discrete expert selection."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.utils import is_gfx942_supported
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd")


def _bf16_gate(is_deepseek_v4):
    import sglang.srt.models.deepseek_v2 as dsv2
    from sglang.srt.server_args import (
        ServerArgs,
        set_global_server_args_for_scheduler,
    )

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    config = SimpleNamespace(
        n_routed_experts=256, hidden_size=4096, topk_method="noaux_tc"
    )
    # Serving allocates the gate in the model dtype; an FP32 weight would take the early
    # router_fp32 return and never reach the branches under test.
    gate = dsv2.MoEGate(config, quant_config=None, is_deepseek_v4=is_deepseek_v4).to(
        device="cuda", dtype=torch.bfloat16
    )
    gate.weight.data.normal_(std=0.01)
    return gate


def _route(gate, x, *, use_aiter, gfx942, tuned_gemm):
    import sglang.srt.models.deepseek_v2 as dsv2

    with (
        patch.object(dsv2, "_use_aiter", use_aiter),
        patch.object(dsv2, "_is_gfx942_supported", gfx942),
        patch.object(dsv2, "aiter_dsv3_router_gemm", tuned_gemm, create=True),
    ):
        return gate(x)


@unittest.skipUnless(is_gfx942_supported(), "requires gfx942")
class TestGfx942RouterPrecision(unittest.TestCase):
    """AITER's tuned router GEMM may accumulate split-K partials in BF16, so identical
    forwards can disagree. That non-determinism cannot be triggered on demand, so these pin
    the contract that removes it for DeepSeek-V4: on gfx942 its gate never reaches the tuned
    GEMM, with AITER on or off, and returns FP32 logits. Every other model keeps the tuned
    GEMM, and so does DeepSeek-V4 on every other target.
    """

    def test_v4_gate_is_fp32_and_never_reaches_the_tuned_gemm(self):
        tuned = MagicMock(side_effect=AssertionError("V4 on gfx942 must not use it"))
        torch.manual_seed(123)
        gate = _bf16_gate(is_deepseek_v4=True)
        for use_aiter in (True, False):
            for rows in (1, 7, 16, 256):
                with self.subTest(use_aiter=use_aiter, rows=rows):
                    x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)
                    out = _route(
                        gate, x, use_aiter=use_aiter, gfx942=True, tuned_gemm=tuned
                    )
                    self.assertEqual(out.dtype, torch.float32)
                    # Independent reference: matmul against the transposed weight.
                    reference = torch.matmul(x.float(), gate.weight.float().t())
                    torch.testing.assert_close(out, reference, rtol=0, atol=0)
        tuned.assert_not_called()

    def test_other_models_keep_the_tuned_gemm_on_gfx942(self):
        logits = torch.empty(0)
        tuned = MagicMock(return_value=logits)
        gate = _bf16_gate(is_deepseek_v4=False)
        x = torch.randn(7, 4096, device="cuda", dtype=torch.bfloat16)

        out = _route(gate, x, use_aiter=True, gfx942=True, tuned_gemm=tuned)

        self.assertIs(out, logits)
        tuned.assert_called_once()

    def test_v4_keeps_the_tuned_gemm_on_other_targets(self):
        logits = torch.empty(0)
        tuned = MagicMock(return_value=logits)
        gate = _bf16_gate(is_deepseek_v4=True)
        x = torch.randn(7, 4096, device="cuda", dtype=torch.bfloat16)

        out = _route(gate, x, use_aiter=True, gfx942=False, tuned_gemm=tuned)

        self.assertIs(out, logits)
        tuned.assert_called_once()

    def test_the_shared_aiter_router_gemm_is_unchanged(self):
        """Every model that routes through the helper still gets tgemm.mm in its dtype."""
        import sglang.srt.layers.rocm_linear_utils as utils

        x = torch.ones(1, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.ones(8, 128, device="cuda", dtype=torch.bfloat16)
        with patch.object(utils.tgemm, "mm", return_value=x) as gemm:
            self.assertIs(utils.aiter_dsv3_router_gemm(x, weight), x)

        self.assertIs(gemm.call_args.args[0], x)
        self.assertEqual(gemm.call_args.kwargs["otype"], x.dtype)


if __name__ == "__main__":
    unittest.main()
