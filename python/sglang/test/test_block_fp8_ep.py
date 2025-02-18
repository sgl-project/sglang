import itertools
import unittest
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe
from sglang.srt.layers.quantization.fp8_kernel import (
    per_token_group_quant_fp8,
    w8a8_block_fp8_matmul,
)
from sglang.srt.layers.moe.topk import select_experts
from sglang.srt.layers.moe.ep_moe.kernels import (
    grouped_gemm_triton,
    post_reorder_triton_kernel,
    pre_reorder_triton_kernel,
    run_moe_ep_preproess,
    silu_and_mul_triton_kernel,
)
from sglang.srt.layers.moe.ep_moe.layer import EPMoE

# For test
def torch_w8a8_block_fp8_ep_moe(
    a, w1, w2, w1_s, w2_s, score, topk, block_shape,
    # ep config
    num_experts: int = 256,
    fp8_dtype: torch.types = torch.float8_e4m3fn,
    num_experts_per_partition: int = 128, # num_experts/ep_size
    start_expert_id: int = 0,
    end_expert_id: int = 127,    
):
    """This function performs fused moe with block-wise quantization using native torch."""

    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)

    # run_moe_ep_preproess
    reorder_topk_ids, reorder_ids = torch.sort(topk_ids, stable=True)
    seg_indptr = torch.zeros(num_experts + 1, device=topk_ids.device, dtype=torch.int64)
    src2dst = torch.empty(topk_ids.numel(), device=topk_ids.device, dtype=torch.int32)
    num_toks = topk_ids.numel()
    for expert in range(num_experts):
        # compute_seg_indptr_triton_kernel
        low = 0
        high = num_toks - 1
        target_location = -1
        while low <= high:
            mid = (low + high) // 2
            if reorder_topk_ids[mid] > expert:
                high = mid - 1
            else:
                low = mid + 1
                target_location = mid
        seg_indptr[expert + 1] = target_location + 1
    src2dst[reorder_ids] = torch.arange(num_toks)

    _, block_k = block_shape[0], block_shape[1]
    a_q, a_s = native_per_token_group_quant_fp8(a, block_k)
    # NOTE(HandH1998): Since "index_cuda" not implemented for 'Float8_e4m3fn', we need to cast `float8`` to `float32``.
    a_q = a_q.to(torch.float32)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            inter_out = native_w8a8_block_fp8_matmul(
                a_q[mask], w1[i], a_s[mask], w1_s[i], block_shape, output_dtype=a.dtype
            )
            act_out = SiluAndMul().forward_native(inter_out)
            act_out_q, act_out_s = native_per_token_group_quant_fp8(act_out, block_k)
            act_out = act_out.to(torch.float32)
            out[mask] = native_w8a8_block_fp8_matmul(
                act_out_q, w2[i], act_out_s, w2_s[i], block_shape, output_dtype=a.dtype
            )
    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)

def ep_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,

    #ep config
    num_experts: int = 256,
    fp8_dtype: torch.types = torch.float8_e4m3fn,
    num_experts_per_partition: int = 128, # num_experts/ep_size
    start_expert_id: int = 0,
    end_expert_id: int = 127,

    use_grouped_topk: bool = False,
    num_expert_group: Optional[int] = None,
    topk_group: Optional[int] = None,
    custom_routing_function: Optional[Callable] = None,
    use_fp8_w8a8: bool = False,
    w1_scale_inv: Optional[torch.Tensor] = None,
    w2_scale_inv: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
):
    use_blockwise_fp8 = block_shape is not None
    topk_weights, topk_ids = select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=top_k,
        use_grouped_topk=use_grouped_topk,
        renormalize=renormalize,
        topk_group=topk_group,
        num_expert_group=num_expert_group,
        # correction_bias=correction_bias, #skip this in test
        custom_routing_function=custom_routing_function,
    )

    reorder_topk_ids, src2dst, seg_indptr = run_moe_ep_preproess(
        topk_ids, num_experts
    )

    gateup_input = torch.empty(
        (int(hidden_states.shape[0] * top_k), hidden_states.shape[1]),
        device=hidden_states.device,
        dtype=fp8_dtype if use_blockwise_fp8 else hidden_states.dtype,
    )
    # activation_scheme == "dynamic":
    if use_blockwise_fp8:
        max_value = (
            torch.max(hidden_states)
            .repeat(num_experts_per_partition)
            .to(torch.float32)
        )
        w1_input_scale = max_value / torch.finfo(fp8_dtype).max
    else:
        w1_input_scale = None

    # PreReorder
    pre_reorder_triton_kernel[(hidden_states.shape[0],)](
        hidden_states,
        gateup_input,
        src2dst,
        topk_ids,
        w1_input_scale,
        start_expert_id,
        end_expert_id,
        top_k,
        hidden_states.shape[1],
        BLOCK_SIZE=512,
    )

    seg_indptr_cur_rank = seg_indptr[start_expert_id : end_expert_id + 2]
    weight_indices_cur_rank = torch.arange(
        0,
        num_experts_per_partition,
        device=hidden_states.device,
        dtype=torch.int64,
    )
    # GroupGemm-0
    gateup_output = torch.empty(
        gateup_input.shape[0],
        w1.shape[1],
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    if use_blockwise_fp8:
        gateup_output = grouped_gemm_triton(
            a=gateup_input,
            b=w1,
            c=gateup_output,
            batch_size=num_experts_per_partition,
            weight_column_major=True,
            seg_indptr=seg_indptr_cur_rank,
            weight_indices=weight_indices_cur_rank,
            use_fp8_w8a8=False,
            scale_a=None,#(None if self.quant_method.block_quant else self.w13_weight_scale),
            scale_b=w1_scale_inv,#if self.quant_method.block_quant
            block_shape=block_shape
        )
    else:
        gateup_output = grouped_gemm_triton(
            a=gateup_input,
            b=w1,
            c=gateup_output,
            batch_size=num_experts_per_partition,
            weight_column_major=True,
            seg_indptr=seg_indptr_cur_rank,
            weight_indices=weight_indices_cur_rank,
            use_fp8_w8a8=False,
            scale_a=None,
            scale_b=None,
            block_shape=None
        )

    # Act
    down_input = torch.empty(
        gateup_output.shape[0],
        gateup_output.shape[1] // 2,
        device=gateup_output.device,
        dtype=fp8_dtype,
    )
    if use_blockwise_fp8:
        w2_input_scale = torch.ones(
            num_experts_per_partition,
            dtype=torch.float32,
            device=hidden_states.device,
        )
    else:
        w2_input_scale = None

    # self.activation == "silu":
    silu_and_mul_triton_kernel[(gateup_output.shape[0],)](
        gateup_output,
        down_input,
        gateup_output.shape[1],
        reorder_topk_ids,
        w2_input_scale,
        start_expert_id,
        end_expert_id,
        BLOCK_SIZE=512,
    )

    # GroupGemm-1
    down_output = torch.empty(
        down_input.shape[0],
        w2.shape[1],
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    if use_blockwise_fp8:
        down_output = grouped_gemm_triton(
            a=down_input,
            b=w2,
            c=down_output,
            batch_size=num_experts_per_partition,
            weight_column_major=True,
            seg_indptr=seg_indptr_cur_rank,
            weight_indices=weight_indices_cur_rank,
            use_fp8_w8a8=False,
            scale_a=None,#(None if self.quant_method.block_quant else self.w2_input_scale),
            scale_b=w2_scale_inv,#if self.quant_method.block_quant
            block_shape=block_shape,
        )
    else:
        down_output = grouped_gemm_triton(
            a=down_input,
            b=w2,
            c=down_output,
            batch_size=num_experts_per_partition,
            weight_column_major=True,
            seg_indptr=seg_indptr_cur_rank,
            weight_indices=weight_indices_cur_rank,
            use_fp8_w8a8=False,
            scale_a=None,
            scale_b=None,
            block_shape=None,
        )

    # PostReorder
    output = torch.empty_like(hidden_states)
    post_reorder_triton_kernel[(hidden_states.size(0),)](
        down_output,
        output,
        src2dst,
        topk_ids,
        topk_weights,
        start_expert_id,
        end_expert_id,
        top_k,
        hidden_states.size(1),
        BLOCK_SIZE=512,
    )
    return output

class TestW8A8BlockFP8EPMoE(unittest.TestCase):
    DTYPES = [torch.float32]#, torch.half, torch.bfloat16]
    M = [1, 33, 64]#, 222, 1024 * 128]
    N = [128]#, 1024, 2048]
    K = [256]#, 4096, 5120]
    E = [8, 24]
    TOP_KS = [2, 6]
    # BLOCK_SIZE = [[64, 64], [64, 128], [128, 64], [128, 128]]
    BLOCK_SIZE = [[128, 128]]
    SEEDS = [0]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _w8a8_block_fp8_fused_moe(self, M, N, K, E, topk, block_size, dtype, seed):
        torch.manual_seed(seed)
        # NOTE(HandH1998): to avoid overflow when out_dtype = torch.half
        factor_for_scale = 1e-2
        fp8_info = torch.finfo(torch.float8_e4m3fn)
        fp8_max, fp8_min = fp8_info.max, fp8_info.min

        a = torch.randn((M, K), dtype=dtype) / 10

        w1_fp32 = (torch.rand((E, 2 * N, K), dtype=torch.float32) - 0.5) * 2 * fp8_max
        w1 = w1_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        w2_fp32 = (torch.rand((E, K, N), dtype=torch.float32) - 0.5) * 2 * fp8_max
        w2 = w2_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        block_n, block_k = block_size[0], block_size[1]
        n_tiles_w1 = (2 * N + block_n - 1) // block_n
        n_tiles_w2 = (K + block_n - 1) // block_n
        k_tiles_w1 = (K + block_k - 1) // block_k
        k_tiles_w2 = (N + block_k - 1) // block_k

        w1_s = (
            torch.rand((E, n_tiles_w1, k_tiles_w1), dtype=torch.float32)
            * factor_for_scale
        )
        w2_s = (
            torch.rand((E, n_tiles_w2, k_tiles_w2), dtype=torch.float32)
            * factor_for_scale
        )

        score = torch.randn((M, E), dtype=dtype)

        with torch.inference_mode():
            # out = ep_moe(
            #     hidden_states=a,
            #     w1=w1,
            #     w2=w2,
            #     router_logits=score,
            #     top_k=topk,
            #     renormalize=False,
            #     use_fp8_w8a8=False,
            #     w1_scale_inv=w1_s,
            #     w2_scale_inv=w2_s,
            #     block_shape=block_size,
            # )
            ref_out = ep_moe(
                hidden_states=a,
                w1=w1_fp32,
                w2=w2_fp32,
                router_logits=score,
                top_k=topk,
                renormalize=False,
                use_fp8_w8a8=False,
                w1_scale_inv=None,
                w2_scale_inv=None,
                block_shape=None,
            )

        self.assertTrue(
            torch.mean(torch.abs(out.to(torch.float32) - ref_out.to(torch.float32)))
            / torch.mean(torch.abs(ref_out.to(torch.float32)))
            < 0.02
        )

    def test_w8a8_block_fp8_fused_moe(self):
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
                self._w8a8_block_fp8_fused_moe(*params)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main(verbosity=2)
