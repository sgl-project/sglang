import itertools
import unittest

import torch

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant
from sglang.test.test_utils import CustomTestCase


def native_w8a8_per_token_matmul(A, B, As, Bs, output_dtype=torch.float16):
    """Matrix multiplication function that supports per-token input quantization and per-column weight quantization"""
    A = A.to(torch.float32)
    B = B.to(torch.float32)

    assert A.shape[-1] == B.shape[-1], "Dimension mismatch"
    assert B.ndim == 2 and B.is_contiguous(), "B must be a 2D contiguous tensor"

    # Reshape input
    M = A.numel() // A.shape[-1]
    B = B.t()  # Transpose weight matrix
    N, K = B.shape
    origin_C_shape = A.shape[:-1] + (K,)
    A = A.reshape(M, N)

    # As is per-token [M, 1], Bs is per-column [1, K]
    C = torch.matmul(A, B)  # [M, K]
    C = As * C * Bs.view(1, -1)  # Broadcast per-column scale

    return C.reshape(origin_C_shape).to(output_dtype)


def fp8_mask(a, mask):
    dtype = a.dtype
    return a.view(torch.int8)[mask].view(dtype)


def torch_w8a8_per_column_moe(a, w1, w2, w1_s, w2_s, score, topk):
    """This function performs fused moe with per-column int8 quantization using native torch."""

    B, D = a.shape
    # Perform per-token quantization
    a_q, a_s = scaled_fp8_quant(a, use_per_token_if_dynamic=True)
    # Repeat tokens to match topk
    a_q = a_q.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    # Also repeat the scale
    a_s = a_s.view(B, -1, 1).repeat(1, topk, 1).reshape(-1, 1)  # [B*topk, 1]

    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)

    # Calculate routing
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    # Process each expert
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            # First MLP layer: note that a_s is now per-token
            inter_out = native_w8a8_per_token_matmul(
                fp8_mask(a_q, mask),
                w1[i],
                fp8_mask(a_s, mask),
                w1_s[i],
                output_dtype=a.dtype,
            )
            # Activation function
            act_out = SiluAndMul().forward_native(inter_out)
            # Quantize activation output with per-token
            act_out_q, act_out_s = scaled_fp8_quant(
                act_out, use_per_token_if_dynamic=True
            )

            # Second MLP layer
            out[mask] = native_w8a8_per_token_matmul(
                act_out_q, w2[i], act_out_s, w2_s[i], output_dtype=a.dtype
            )
    # Apply routing weights and sum
    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)


class TestW8A8FP8FusedMoE(CustomTestCase):
    DTYPES = [torch.half, torch.bfloat16]
    M = [1, 33]
    N = [128, 1024]
    K = [256, 4096]
    E = [8]
    TOP_KS = [2, 6]
    BLOCK_SIZE = [[64, 64], [64, 128], [128, 64], [128, 128]]
    BLOCK_SIZE = [[128, 128]]
    SEEDS = [0]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _w8a8_fp8_fused_moe(self, M, N, K, E, topk, block_size, dtype, seed):
        torch.manual_seed(seed)
        # Initialize int8 quantization parameters
        factor_for_scale = 1e-2
        finfo = torch.finfo(torch.float8_e4m3fn)
        fp8_max = finfo.max
        fp8_min = finfo.min

        # Input tensor
        # M * K
        a = torch.randn((M, K), dtype=dtype) / 10

        # Generate int8 weights
        w1_fp32 = (torch.rand((E, 2 * N, K), dtype=torch.float32) - 0.5) * 2
        w1 = (w1_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        w2_fp32 = (torch.rand((E, K, N), dtype=torch.float32) - 0.5) * 2
        w2 = (w2_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        # Generate scale for each column (per-column quantization)
        w1_s = torch.rand(E, 2 * N, device=w1_fp32.device) * factor_for_scale
        w2_s = torch.rand(E, K, device=w2_fp32.device) * factor_for_scale
        score = torch.randn((M, E), dtype=dtype)

        with torch.inference_mode():
            ref_out = torch_w8a8_per_column_moe(a, w1, w2, w1_s, w2_s, score, topk)
            topk_output = select_experts(
                hidden_states=a,
                router_logits=score,
                topk_config=TopKConfig(top_k=topk, renormalize=False),
            )
            out = fused_moe(
                a,
                w1,
                w2,
                topk_output,
                use_fp8_w8a8=True,  # using fp8
                use_int8_w8a16=False,
                use_int8_w8a8=False,
                per_channel_quant=True,
                w1_scale=w1_s,
                w2_scale=w2_s,
                block_shape=None,  # Not using block quantization
            )

        # Check results
        self.assertTrue(
            torch.mean(torch.abs(out.to(torch.float32) - ref_out.to(torch.float32)))
            / torch.mean(torch.abs(ref_out.to(torch.float32)))
            < 0.05
        )

    def test_w8a8_fp8_fused_moe(self):
        for params in itertools.product(
            self.M,
            self.N,
            self.K,
            self.E,
            self.TOP_KS,
            self.BLOCK_SIZE,
            self.DTYPES,
            self.SEEDS,
        ):
            with self.subTest(
                M=params[0],
                N=params[1],
                K=params[2],
                E=params[3],
                topk=params[4],
                block_size=params[5],
                dtype=params[6],
                seed=params[7],
            ):
                self._w8a8_fp8_fused_moe(*params)


if __name__ == "__main__":
    unittest.main(verbosity=2)
