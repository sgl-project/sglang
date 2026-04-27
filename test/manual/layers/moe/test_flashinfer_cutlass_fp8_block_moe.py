"""
Test FlashInfer CUTLASS fused MoE for FP8 block-quantized models on SM90.

Tests:
1. Kernel-level: cutlass_fused_moe vs reference BF16 computation
2. E2E: server with --moe-runner-backend flashinfer_cutlass on FP8 block-quant model

Requirements:
- SM90 GPU (H100/H800)
- FlashInfer >= 0.6.6
- nvcc in PATH
"""

import os
import sys
import unittest

import torch

from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_FP8_WITH_MOE,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


def is_sm90():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] == 9


def has_flashinfer_cutlass():
    try:
        from flashinfer.fused_moe import cutlass_fused_moe

        return True
    except ImportError:
        return False


@unittest.skipUnless(
    is_sm90() and has_flashinfer_cutlass(),
    "Requires SM90 GPU and FlashInfer with CUTLASS MoE support",
)
class TestCutlassFusedMoeFP8BlockKernel(CustomTestCase):
    """Kernel-level correctness test for cutlass_fused_moe with FP8 block scales."""

    def _block_quant(self, w, block_size=128):
        """Per-block FP8 quantization."""
        E, M, K = w.shape
        fp8_max = 448.0
        w_blocks = w.float().reshape(E, M // block_size, block_size, K // block_size, block_size)
        absmax = w_blocks.abs().amax(dim=(2, 4), keepdim=True).clamp(min=1e-10)
        scale = (absmax / fp8_max).squeeze(2).squeeze(-1)
        w_q = (w_blocks / (absmax / fp8_max)).clamp(-fp8_max, fp8_max)
        w_fp8 = w_q.reshape(E, M, K).to(torch.float8_e4m3fn)
        return w_fp8, scale.float()

    def _dequant_block(self, w_fp8, scale, block_size=128):
        """Dequantize block-quantized FP8 tensor."""
        N, K = w_fp8.shape
        w = w_fp8.float().reshape(N // block_size, block_size, K // block_size, block_size)
        s = scale.reshape(N // block_size, 1, K // block_size, 1)
        return (w * s).reshape(N, K).bfloat16()

    def _reference_moe(self, x, gate_w, up_w, down_w, gate_s, up_s, down_s):
        """Reference BF16 MoE computation: silu(x @ gate.T) * (x @ up.T) @ down.T"""
        gate_bf16 = self._dequant_block(gate_w, gate_s)
        up_bf16 = self._dequant_block(up_w, up_s)
        down_bf16 = self._dequant_block(down_w, down_s)
        gate_out = x @ gate_bf16.T
        up_out = x @ up_bf16.T
        return (torch.nn.functional.silu(gate_out) * up_out) @ down_bf16.T

    def test_cutlass_fused_moe_correctness(self):
        """Compare cutlass_fused_moe output with reference BF16 computation."""
        from flashinfer.fused_moe import cutlass_fused_moe

        torch.manual_seed(42)
        E, N, K = 4, 256, 512
        BS = 8
        topk = 2

        # Create per-expert weights
        gate_ws = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda") * 0.01
        up_ws = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda") * 0.01
        down_ws = torch.randn(E, K, N, dtype=torch.bfloat16, device="cuda") * 0.01

        # Block quantize
        gate_fp8, gate_s = [], []
        up_fp8, up_s = [], []
        for e in range(E):
            gq, gs = self._block_quant(gate_ws[e:e+1])
            uq, us = self._block_quant(up_ws[e:e+1])
            gate_fp8.append(gq[0]); gate_s.append(gs[0])
            up_fp8.append(uq[0]); up_s.append(us[0])

        gate_fp8 = torch.stack(gate_fp8)
        up_fp8 = torch.stack(up_fp8)
        gate_s = torch.stack(gate_s)
        up_s = torch.stack(up_s)

        down_fp8, down_s = self._block_quant(down_ws)

        # w13 = [up, gate] (FlashInfer CUTLASS SwiGLU convention)
        w13_fp8 = torch.cat([up_fp8, gate_fp8], dim=1)  # [E, 2N, K]
        w13_s = torch.cat([up_s, gate_s], dim=1)  # [E, 2N/b, K/b]

        x = torch.randn(BS, K, dtype=torch.bfloat16, device="cuda")
        scores = torch.randn(BS, E, dtype=torch.float32, device="cuda")
        topk_weights, topk_ids = torch.topk(scores, topk, dim=-1)
        topk_weights = torch.softmax(topk_weights, dim=-1)

        # CUTLASS fused MoE
        output = cutlass_fused_moe(
            input=x,
            token_selected_experts=topk_ids.to(torch.int32),
            token_final_scales=topk_weights,
            fc1_expert_weights=w13_fp8,
            fc2_expert_weights=down_fp8,
            output_dtype=x.dtype,
            quant_scales=[w13_s, down_s],
            use_deepseek_fp8_block_scale=True,
        )[0]

        # Reference: weighted sum of per-expert outputs
        ref = torch.zeros_like(x)
        for i in range(BS):
            for j in range(topk):
                eid = topk_ids[i, j].item()
                w = topk_weights[i, j].item()
                expert_out = self._reference_moe(
                    x[i:i+1],
                    gate_fp8[eid], up_fp8[eid], down_fp8[eid],
                    gate_s[eid], up_s[eid], down_s[eid],
                )
                ref[i] += w * expert_out[0]

        # FP8 quantization introduces error, use relaxed tolerance
        max_diff = (output - ref).abs().max().item()
        mean_diff = (output - ref).abs().mean().item()
        self.assertLess(max_diff, 1.0, f"Max diff {max_diff} too large")
        self.assertLess(mean_diff, 0.1, f"Mean diff {mean_diff} too large")

    def test_weight_swap_correctness(self):
        """Verify [gate,up] → [up,gate] swap produces correct results."""
        from flashinfer.fused_moe import cutlass_fused_moe

        torch.manual_seed(123)
        E, N, K = 2, 256, 256
        BS = 4

        gate_w = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda") * 0.01
        up_w = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda") * 0.01
        down_w = torch.randn(E, K, N, dtype=torch.bfloat16, device="cuda") * 0.01

        w13_gate_up = torch.cat([gate_w, up_w], dim=1)  # SGLang default: [gate, up]
        w13_up_gate = torch.cat([up_w, gate_w], dim=1)   # FlashInfer CUTLASS: [up, gate]

        w13_gu_fp8, w13_gu_s = self._block_quant(w13_gate_up)
        w13_ug_fp8, w13_ug_s = self._block_quant(w13_up_gate)
        down_fp8, down_s = self._block_quant(down_w)

        x = torch.randn(BS, K, dtype=torch.bfloat16, device="cuda")
        ids = torch.zeros(BS, 1, dtype=torch.int32, device="cuda")
        wts = torch.ones(BS, 1, dtype=torch.float32, device="cuda")

        # Reference
        ref = self._reference_moe(
            x, gate_w[0].to(torch.float8_e4m3fn), up_w[0].to(torch.float8_e4m3fn),
            down_w[0].to(torch.float8_e4m3fn),
            torch.ones(N // 128, K // 128, device="cuda"),
            torch.ones(N // 128, K // 128, device="cuda"),
            torch.ones(K // 128, N // 128, device="cuda"),
        )

        # [gate, up] — wrong order for CUTLASS
        out_gu = cutlass_fused_moe(
            x, ids, wts, w13_gu_fp8, down_fp8, x.dtype,
            quant_scales=[w13_gu_s, down_s], use_deepseek_fp8_block_scale=True,
        )[0]

        # [up, gate] — correct order for CUTLASS
        out_ug = cutlass_fused_moe(
            x, ids, wts, w13_ug_fp8, down_fp8, x.dtype,
            quant_scales=[w13_ug_s, down_s], use_deepseek_fp8_block_scale=True,
        )[0]

        diff_gu = (out_gu - ref).abs().max().item()
        diff_ug = (out_ug - ref).abs().max().item()

        # [up, gate] should be more accurate
        self.assertLess(
            diff_ug, diff_gu,
            f"[up,gate] diff ({diff_ug:.4f}) should be less than [gate,up] diff ({diff_gu:.4f})"
        )


@unittest.skipUnless(
    is_sm90() and has_flashinfer_cutlass(),
    "Requires SM90 GPU and FlashInfer with CUTLASS MoE support",
)
class TestCutlassFusedMoeFP8BlockE2E(CustomTestCase):
    """E2E test: launch server with --moe-runner-backend flashinfer_cutlass."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_FP8_WITH_MOE
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=600,
            other_args=[
                "--trust-remote-code",
                "--moe-runner-backend",
                "flashinfer_cutlass",
                "--attention-backend",
                "triton",
                "--sampling-backend",
                "pytorch",
                "--disable-cuda-graph",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        from sglang.srt.utils import kill_process_tree

        kill_process_tree(cls.process.pid)

    def test_generate(self):
        """Basic generation test."""
        import requests

        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"max_new_tokens": 32, "temperature": 0},
            },
        )
        self.assertEqual(response.status_code, 200)
        text = response.json()["text"]
        self.assertIn("Paris", text, f"Expected 'Paris' in output, got: {text}")

    def test_batch_generate(self):
        """Batch generation test."""
        import requests

        prompts = [
            "What is 1+1?",
            "The largest planet is",
            "Python is a",
        ]
        for prompt in prompts:
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "text": prompt,
                    "sampling_params": {"max_new_tokens": 16, "temperature": 0},
                },
            )
            self.assertEqual(response.status_code, 200)
            text = response.json()["text"]
            self.assertTrue(len(text) > 0, f"Empty output for prompt: {prompt}")


if __name__ == "__main__":
    unittest.main()
