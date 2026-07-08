"""Numerical tests for the HPC-Ops FP8 MoE runner backend.

Compares the hpc_ops fused func (hpc.fuse_moe_blockwise) against the triton
fused_experts reference and an fp32 exact reference on realistic blockwise
FP8 quantized weights. Skipped when HPC-Ops (https://github.com/Tencent/hpc-ops)
is not installed or the GPU is older than sm90.
"""

import unittest

import torch

from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.hpc_ops import (
    HpcOpsMoeQuantInfo,
    fused_experts_none_to_hpc_ops,
    has_hpc_ops,
    pad_hpc_ops_block_scale,
)
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")

# Qwen3-30B-A3B-FP8 MoE shapes.
E, TOPK, H, I = 128, 8, 2048, 768


def _sm90_or_newer() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


def _quant_blockwise(w: torch.Tensor, block: int = 128):
    """Proper 128x128 blockwise fp8 quantization of a fp32 weight [E, N, K]."""
    num_experts, n, k = w.shape
    wb = w.view(num_experts, n // block, block, k // block, block)
    amax = wb.abs().amax(dim=(2, 4), keepdim=True).clamp(min=1e-6)
    scale = amax / 448.0
    wq = (wb / scale).clamp(-448, 448).view(num_experts, n, k).to(torch.float8_e4m3fn)
    return wq, scale.squeeze(-1).squeeze(2)


@unittest.skipUnless(
    has_hpc_ops() and _sm90_or_newer(),
    "requires HPC-Ops (pip install from https://github.com/Tencent/hpc-ops) and sm90+",
)
class TestHpcOpsMoeBlockwise(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        torch.manual_seed(0)

    def _make_case(self, num_tokens: int):
        device = "cuda"
        x = torch.randn(num_tokens, H, dtype=torch.bfloat16, device=device)
        w13_f = torch.randn(E, 2 * I, H, dtype=torch.float32, device=device) * 0.02
        w2_f = torch.randn(E, H, I, dtype=torch.float32, device=device) * 0.02
        w13, w13_scale = _quant_blockwise(w13_f)
        w2, w2_scale = _quant_blockwise(w2_f)
        logits = torch.randn(num_tokens, E, dtype=torch.float32, device=device)
        topk_weights, topk_ids = torch.topk(logits.softmax(dim=-1), TOPK, dim=-1)
        topk_weights = (topk_weights / topk_weights.sum(dim=-1, keepdim=True)).float()
        return x, w13, w2, w13_scale, w2_scale, topk_weights, topk_ids.to(torch.int32)

    def _runner_config(self):
        return MoeRunnerConfig(
            num_experts=E,
            num_local_experts=E,
            hidden_size=H,
            intermediate_size_per_partition=I,
            layer_id=0,
            top_k=TOPK,
            num_fused_shared_experts=0,
            params_dtype=torch.bfloat16,
            activation="silu",
            is_gated=True,
            inplace=False,
        )

    def _run_hpc_ops(self, x, w13, w2, w13_scale, w2_scale, topk_weights, topk_ids):
        dispatch_output = StandardDispatchOutput(
            hidden_states=x,
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(topk_weights, topk_ids, None),
        )
        quant_info = HpcOpsMoeQuantInfo(
            w13_weight=w13,
            w2_weight=w2,
            block_quant=True,
            global_num_experts=E,
            moe_ep_rank=0,
            w13_weight_scale_inv=pad_hpc_ops_block_scale(w13_scale),
            w2_weight_scale_inv=pad_hpc_ops_block_scale(w2_scale),
            block_shape=[128, 128],
        )
        return fused_experts_none_to_hpc_ops(
            dispatch_output, quant_info, self._runner_config()
        ).hidden_states

    def _run_triton(self, x, w13, w2, w13_scale, w2_scale, topk_weights, topk_ids):
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            fused_experts,
        )

        return fused_experts(
            hidden_states=x.clone(),
            w1=w13,
            w2=w2,
            topk_output=StandardTopKOutput(topk_weights, topk_ids, None),
            moe_runner_config=self._runner_config(),
            use_fp8_w8a8=True,
            w1_scale=w13_scale,
            w2_scale=w2_scale,
            block_shape=[128, 128],
        )

    def test_blockwise_fp8_matches_triton(self):
        for num_tokens in (7, 64, 512):
            with self.subTest(num_tokens=num_tokens):
                case = self._make_case(num_tokens)
                out_hpc = self._run_hpc_ops(*case).float()
                out_triton = self._run_triton(*case).float()
                cos = torch.nn.functional.cosine_similarity(
                    out_hpc.flatten(), out_triton.flatten(), dim=0
                )
                self.assertGreater(cos.item(), 0.999)
                torch.testing.assert_close(out_hpc, out_triton, rtol=0.05, atol=0.05)


if __name__ == "__main__":
    unittest.main()
