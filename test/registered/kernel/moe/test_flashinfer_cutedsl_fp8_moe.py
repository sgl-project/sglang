"""Numerical tests for the flashinfer_cutedsl_fp8 MoE runner backend.

Runs the fused func (FlashInfer CuTe DSL contiguous grouped FP8 GEMM on
SM100/SM103) on Qwen3-30B-A3B-FP8 expert shapes and compares it against the
triton fused_experts reference and a torch reference that mirrors the fp8
quantization. Skipped off SM100/SM103 or when FlashInfer lacks the kernel.
"""

import unittest

import torch

from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl_fp8 import (
    FlashInferCuteDslFp8MoeQuantInfo,
    fused_experts_none_to_flashinfer_cutedsl_fp8,
    has_flashinfer_cutedsl_fp8_group_gemm,
)
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.layer_ut_utils import init_single_process_dist
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

# Qwen3-30B-A3B-FP8 MoE shapes.
E, TOPK, H, I = 128, 8, 2048, 768
BLOCK = 128


def _sm100() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] == 10


def _quant_blockwise(w: torch.Tensor, block: int = BLOCK):
    """128x128 blockwise fp8 quantization of a fp32 weight [E, N, K]."""
    num_experts, n, k = w.shape
    wb = w.view(num_experts, n // block, block, k // block, block)
    amax = wb.abs().amax(dim=(2, 4), keepdim=True).clamp(min=1e-6)
    scale = amax / 448.0
    wq = (wb / scale).clamp(-448, 448).view(num_experts, n, k).to(torch.float8_e4m3fn)
    return wq, scale.squeeze(-1).squeeze(2).contiguous()


def _dequant_blockwise(wq: torch.Tensor, scale: torch.Tensor, block: int = BLOCK):
    """Inverse of _quant_blockwise for a single expert [N, K]."""
    n, k = wq.shape
    s = scale.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
    return wq.float() * s[:n, :k]


def _quant_dequant_act(x: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """Round-trip a [T, K] activation through the runner's 1x128 fp8 quant."""
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    xq, xs = sglang_per_token_group_quant_fp8(x.contiguous(), block)
    return xq.float() * xs.repeat_interleave(block, dim=1)


@unittest.skipUnless(
    _sm100() and has_flashinfer_cutedsl_fp8_group_gemm(),
    "requires an SM100/SM103 GPU and a FlashInfer build with "
    "group_gemm_fp8_nt_groupwise_contiguous",
)
class TestFlashInferCuteDslFp8Moe(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        # The reference and the runner allocate their outputs under
        # use_symmetric_memory(get_tp_group(), ...), which needs the TP group.
        init_single_process_dist(master_port=29634)
        torch.manual_seed(0)

    def _make_weights(self, num_experts: int = E, intermediate: int = I):
        device = "cuda"
        w13_f = torch.randn(num_experts, 2 * intermediate, H, device=device) * 0.02
        w2_f = torch.randn(num_experts, H, intermediate, device=device) * 0.02
        w13, w13_scale = _quant_blockwise(w13_f)
        w2, w2_scale = _quant_blockwise(w2_f)
        return w13, w2, w13_scale, w2_scale

    def _make_routing(self, num_tokens: int, num_experts: int = E):
        logits = torch.randn(num_tokens, num_experts, device="cuda")
        topk_weights, topk_ids = torch.topk(logits.softmax(dim=-1), TOPK, dim=-1)
        topk_weights = (topk_weights / topk_weights.sum(dim=-1, keepdim=True)).float()
        return topk_weights, topk_ids.to(torch.int32)

    def _runner_config(self, num_local_experts: int = E, intermediate: int = I):
        return MoeRunnerConfig(
            num_experts=E,
            num_local_experts=num_local_experts,
            hidden_size=H,
            intermediate_size_per_partition=intermediate,
            layer_id=0,
            top_k=TOPK,
            num_fused_shared_experts=0,
            params_dtype=torch.bfloat16,
            activation="silu",
            is_gated=True,
            inplace=False,
        )

    def _run_cutedsl(
        self, x, w13, w2, w13_scale, w2_scale, topk_weights, topk_ids, config
    ):
        dispatch_output = StandardDispatchOutput(
            hidden_states=x,
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(topk_weights, topk_ids, None),
        )
        quant_info = FlashInferCuteDslFp8MoeQuantInfo(
            w13_weight=w13,
            w2_weight=w2,
            w13_weight_scale_inv=w13_scale,
            w2_weight_scale_inv=w2_scale,
            block_shape=[BLOCK, BLOCK],
        )
        return fused_experts_none_to_flashinfer_cutedsl_fp8(
            dispatch_output, quant_info, config
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
            block_shape=[BLOCK, BLOCK],
        )

    def _run_torch_reference(
        self, x, w13, w2, w13_scale, w2_scale, topk_weights, topk_ids
    ):
        """fp32 reference with the runner's fp8 activation quant; id -1 contributes nothing."""
        num_local_experts, two_i, _ = w13.shape
        intermediate = two_i // 2
        x_deq = _quant_dequant_act(x)
        out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        for e in range(num_local_experts):
            token_idx, slot_idx = (topk_ids == e).nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            w13_e = _dequant_blockwise(w13[e], w13_scale[e])
            w2_e = _dequant_blockwise(w2[e], w2_scale[e])
            h = x_deq[token_idx] @ w13_e.t()
            act = torch.nn.functional.silu(h[:, :intermediate]) * h[:, intermediate:]
            act = _quant_dequant_act(act.to(torch.bfloat16).float())
            y = act @ w2_e.t()
            out.index_add_(0, token_idx, y * topk_weights[token_idx, slot_idx, None])
        return out.to(torch.bfloat16)

    def _assert_close(self, out, ref):
        out, ref = out.float(), ref.float()
        cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0)
        self.assertGreater(cos.item(), 0.999)
        torch.testing.assert_close(out, ref, rtol=0.05, atol=0.05)

    def test_blockwise_fp8_matches_triton(self):
        """Token counts around one 128-row tile per expert: empty experts, partial tiles, slack rows."""
        w13, w2, w13_scale, w2_scale = self._make_weights()
        for num_tokens in (7, 64, 512):
            with self.subTest(num_tokens=num_tokens):
                x = torch.randn(num_tokens, H, dtype=torch.bfloat16, device="cuda")
                topk_weights, topk_ids = self._make_routing(num_tokens)
                out = self._run_cutedsl(
                    x,
                    w13,
                    w2,
                    w13_scale,
                    w2_scale,
                    topk_weights,
                    topk_ids,
                    self._runner_config(),
                )
                ref = self._run_triton(
                    x, w13, w2, w13_scale, w2_scale, topk_weights, topk_ids
                )
                self._assert_close(out, ref)

    def test_ep_local_ids_skip_non_local_experts(self):
        """EP hands the runner local ids with -1 for non-local experts; those rows
        must neither enter the grouped GEMM nor the gathered output."""
        num_local_experts, ep_rank = 32, 1
        w13, w2, w13_scale, w2_scale = self._make_weights(num_experts=num_local_experts)
        num_tokens = 96
        x = torch.randn(num_tokens, H, dtype=torch.bfloat16, device="cuda")
        topk_weights, global_ids = self._make_routing(num_tokens)
        local_ids = global_ids - ep_rank * num_local_experts
        local_ids = torch.where(
            (local_ids >= 0) & (local_ids < num_local_experts),
            local_ids,
            torch.full_like(local_ids, -1),
        )
        self.assertTrue((local_ids == -1).any())
        out = self._run_cutedsl(
            x,
            w13,
            w2,
            w13_scale,
            w2_scale,
            topk_weights,
            local_ids,
            self._runner_config(num_local_experts=num_local_experts),
        )
        ref = self._run_torch_reference(
            x, w13, w2, w13_scale, w2_scale, topk_weights, local_ids
        )
        self._assert_close(out, ref)

    def test_rejects_unaligned_intermediate_size(self):
        """A TP shard with a non-128-multiple GEMM dim must fail loudly."""
        intermediate = 192  # Qwen3-30B-A3B intermediate size sharded 4 ways.
        w13 = torch.empty(
            E, 2 * intermediate, H, dtype=torch.float8_e4m3fn, device="cuda"
        )
        w2 = torch.empty(E, H, intermediate, dtype=torch.float8_e4m3fn, device="cuda")
        w13_scale = torch.ones(E, 2 * intermediate // BLOCK, H // BLOCK, device="cuda")
        w2_scale = torch.ones(E, H // BLOCK, -(-intermediate // BLOCK), device="cuda")
        x = torch.randn(4, H, dtype=torch.bfloat16, device="cuda")
        topk_weights, topk_ids = self._make_routing(4)
        with self.assertRaisesRegex(ValueError, "multiple of 128"):
            self._run_cutedsl(
                x,
                w13,
                w2,
                w13_scale,
                w2_scale,
                topk_weights,
                topk_ids,
                self._runner_config(intermediate=intermediate),
            )


if __name__ == "__main__":
    unittest.main()
