import itertools
import sys
import unittest

import torch

sys.path.insert(0, "/home/hadoop-hmart-waimai-rank/vllm")

# from sglang.srt.layers.moe.topk import select_experts
from sgl_kernel import fused_marlin_moe
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk

# from vllm.model_executor.layers. import select_experts
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    marlin_quantize,
)
from vllm.scalar_type import scalar_types


def stack_and_dev(tensors: list[torch.Tensor]):
    dev = tensors[0].device
    return torch.stack(tensors, dim=0).to(dev)


def torch_moe(a, w1, w2, score, topk, expert_map):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    if expert_map is not None:
        topk_ids = expert_map[topk_ids]
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = SiluAndMul()(a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(
                0, 1
            )
    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)


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


def torch_w8a8_per_column_moe(a, w1, w2, w1_s, w2_s, score, topk):
    """This function performs fused moe with per-column int8 quantization using native torch."""

    B, D = a.shape
    # Perform per-token quantization
    a_q, a_s = per_token_quant_int8(a)
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
                a_q[mask], w1[i], a_s[mask], w1_s[i], output_dtype=a.dtype
            )
            # Activation function
            act_out = SiluAndMul().forward_native(inter_out)
            # Quantize activation output with per-token
            act_out_q, act_out_s = per_token_quant_int8(act_out)

            # Second MLP layer
            out[mask] = native_w8a8_per_token_matmul(
                act_out_q, w2[i], act_out_s, w2_s[i], output_dtype=a.dtype
            )
    # Apply routing weights and sum
    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)


def marlin_fused_moe(
    N, E, K, a, w1, w2, num_bits, group_size, act_order, score, topk, ep_size
):
    quant_type = scalar_types.uint4b8 if num_bits == 4 else scalar_types.uint8b128
    if ep_size > 1:
        local_e = E // ep_size
        e_ids = torch.randperm(E, device="cuda", dtype=torch.int32)[:local_e]
        e_map = torch.full((E,), -1, device="cuda", dtype=torch.int32)
        e_map[e_ids] = torch.arange(local_e, device="cuda", dtype=torch.int32)
        w1 = w1[e_ids]
        w2 = w2[e_ids]
    else:
        e_map = None
    w_ref1_l = []
    qweight1_l = []
    scales1_l = []
    zeros1_l = []
    g_idx1_l = []
    sort_indices1_l = []
    s1_l = []
    for i in range(w1.shape[0]):
        test_perm = torch.randperm(n=K)
        quant_res = marlin_quantize(
            w1[i].transpose(1, 0), quant_type, group_size, act_order, test_perm
        )
        w_ref1, qweight1, scales1, g_idx1, sort_indices1, _ = quant_res
        w_ref1_l.append(w_ref1.T)
        qweight1_l.append(qweight1)
        scales1_l.append(scales1)
        g_idx1_l.append(g_idx1)
        sort_indices1_l.append(sort_indices1)
    w_ref1 = stack_and_dev(w_ref1_l)
    qweight1 = stack_and_dev(qweight1_l).contiguous()
    scales1 = stack_and_dev(scales1_l)
    g_idx1 = stack_and_dev(g_idx1_l) if g_idx1_l else None
    zeros1 = stack_and_dev(zeros1_l) if zeros1_l else None
    sort_indices1 = stack_and_dev(sort_indices1_l) if sort_indices1_l else None

    w_ref2_l = []
    qweight2_l = []
    scales2_l = []
    zeros2_l = []
    g_idx2_l = []
    sort_indices2_l = []
    for i in range(w2.shape[0]):
        test_perm = torch.randperm(n=N)
        quant_res = marlin_quantize(
            w2[i].transpose(1, 0), quant_type, group_size, act_order, test_perm
        )
        w_ref2, qweight2, scales2, g_idx2, sort_indices2, _ = quant_res

        w_ref2_l.append(w_ref2.T)
        qweight2_l.append(qweight2)
        scales2_l.append(scales2)
        g_idx2_l.append(g_idx2)
        sort_indices2_l.append(sort_indices2)

    w_ref2 = stack_and_dev(w_ref2_l)
    qweight2 = stack_and_dev(qweight2_l).contiguous()
    scales2 = stack_and_dev(scales2_l)
    g_idx2 = stack_and_dev(g_idx2_l) if g_idx2_l else None
    zeros2 = stack_and_dev(zeros2_l) if zeros2_l else None
    sort_indices2 = stack_and_dev(sort_indices2_l) if sort_indices2_l else None

    topk_weights, topk_ids = fused_topk(a, score, topk, False)
    # topk_weights, topk_ids = FusedMoE.select_experts(
    #     hidden_states=a,
    #     router_logits=score,
    #     top_k=topk,
    #     num_expert_group=E,
    #     use_grouped_topk=False,
    #     renormalize=False,
    #     topk_group=None,
    #     )

    torch_output = torch_moe(a, w_ref1, w_ref2, score, topk, e_map)
    marlin_output = fused_marlin_moe(
        a,
        qweight1,
        qweight2,
        scales1,
        scales2,
        score,
        topk_weights,
        topk_ids,
        global_num_experts=E,
        expert_map=e_map,
        g_idx1=g_idx1,
        g_idx2=g_idx2,
        sort_indices1=sort_indices1,
        sort_indices2=sort_indices2,
        w1_zeros=zeros1,
        w2_zeros=zeros2,
        num_bits=num_bits,
        is_k_full=True,
    )
    return marlin_output, torch_output


class TestW8A8Int8FusedMoE(unittest.TestCase):
    DTYPES = [torch.float16]
    M = [1, 16]
    N = [128]
    K = [256]
    E = [4, 10]
    TOP_KS = [2, 4]
    BLOCK_SIZE = [[128, 128]]
    SEEDS = [0]
    NUM_BITS = [4]
    EP_SIZE = [1, 4]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _w4a8_int8_fused_moe(
        self, M, N, K, E, topk, block_size, dtype, seed, num_bits, ep_size
    ):
        torch.manual_seed(seed)
        a = torch.randn((M, K), dtype=dtype) / 10

        # Generate int8 weights
        w1_fp16 = (torch.rand((E, 2 * N, K), dtype=dtype) - 0.5) * 2
        w2_fp16 = (torch.rand((E, K, N), dtype=dtype) - 0.5) * 2

        score = torch.randn((M, E), dtype=dtype)

        with torch.inference_mode():
            marlin_out, ref_out = marlin_fused_moe(
                N=N,
                E=E,
                K=K,
                a=a,
                w1=w1_fp16,
                w2=w2_fp16,
                num_bits=num_bits,
                group_size=-1,
                act_order=False,
                score=score,
                topk=topk,
                ep_size=ep_size,
            )
        # Check results
        if (
            torch.mean(
                torch.abs(marlin_out.to(torch.float32) - ref_out.to(torch.float32))
            )
            / torch.mean(torch.abs(ref_out.to(torch.float32)))
            > 0.1
        ):
            print(f"marlin_out: {marlin_out}")
            print(f"ref_out: {ref_out}")
            print(
                torch.mean(
                    torch.abs(marlin_out.to(torch.float32) - ref_out.to(torch.float32))
                )
                / torch.mean(torch.abs(ref_out.to(torch.float32)))
            )
        torch.testing.assert_close(marlin_out, ref_out, atol=2e-2, rtol=0)

    def test_w4a8_int8_fused_moe(self):
        for params in itertools.product(
            self.M,
            self.N,
            self.K,
            self.E,
            self.TOP_KS,
            self.BLOCK_SIZE,
            self.DTYPES,
            self.SEEDS,
            self.NUM_BITS,
            self.EP_SIZE,
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
                num_bits=params[8],
                ep_size=params[9],
            ):
                self._w4a8_int8_fused_moe(*params)


if __name__ == "__main__":
    unittest.main(verbosity=2)
