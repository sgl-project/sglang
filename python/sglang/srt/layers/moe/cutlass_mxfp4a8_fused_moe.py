# SPDX-License-Identifier: Apache-2.0
"""CUTLASS MXFP4A8 fused MoE runner.

For EP=1 this runner uses the explicit fused MXFP4A8 SM90 grouped GEMM entry with
load-time-interleaved weights, folded E8M0 offsets, and per-token FP8 activation
scales. EP keeps the complete legacy MXFP4A8 protocol.

  * ``prepare_moe_input`` builds expert offsets, GEMM problem sizes, and both
    permutations in one CUDA path.
  * ``shuffle_rows`` performs the gate/up input reorder.
  * A fused MXFP4A8-only CUDA op fuses input quantization with the expert residual
    scale lookup; older kernels fall back to the original CUDA/Triton sequence.
  * ``apply_shuffle_mul_sum`` performs the final top-k reorder, router-weight
    multiply, and reduction.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_cuda_alike

_is_cuda_alike = is_cuda_alike()

if _is_cuda_alike:
    from sgl_kernel import (
        cutlass_mxfp4a8_moe_mm,
        get_cutlass_w4a8_moe_mm_data,
        prepare_moe_input,
        sgl_per_token_quant_fp8,
    )

    try:
        from sgl_kernel import compact_cutlass_w4a8_moe_mm_data
    except ImportError:
        compact_cutlass_w4a8_moe_mm_data = None

    try:
        from sgl_kernel import get_cutlass_w4a8_moe_mm_data_with_permutation
    except ImportError:
        get_cutlass_w4a8_moe_mm_data_with_permutation = None

@triton.jit
def _apply_shuffle_mul_sum_fp32_factors_kernel(
    input_ptr,
    output_ptr,
    perm_ptr,
    factors_ptr,
    m,
    topk: tl.constexpr,
    row_stride: tl.constexpr,
    routed_scaling_factor: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    block = tl.program_id(1)
    offs = block * BLOCK + tl.arange(0, BLOCK)
    mask = (token < m) & (offs < row_stride)

    acc = tl.zeros((BLOCK,), tl.float32)
    for j in tl.range(0, topk):
        token_major_idx = token * topk + j
        src_row = tl.load(perm_ptr + token_major_idx).to(tl.int64)
        vals = tl.load(
            input_ptr + src_row * row_stride + offs, mask=mask, other=0.0
        ).to(tl.float32)
        factor = tl.load(factors_ptr + token_major_idx).to(tl.float32)
        acc += vals * factor * routed_scaling_factor

    tl.store(output_ptr + token * row_stride + offs, acc, mask=mask)


@triton.jit
def _mul_per_token_scale_by_expert_kernel(
    scale_ptr,
    residual_ptr,
    expert_offsets_ptr,
    total_m,
    num_experts: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = rows < total_m
    expert = tl.zeros((BLOCK,), tl.int32)
    for e in tl.range(0, num_experts):
        next_offset = tl.load(expert_offsets_ptr + e + 1)
        expert = tl.where(rows >= next_offset, e + 1, expert)
    expert = tl.minimum(expert, num_experts - 1)
    scale = tl.load(scale_ptr + rows, mask=mask)
    residual = tl.load(residual_ptr + expert, mask=mask)
    tl.store(scale_ptr + rows, scale * residual, mask=mask)


class CutlassMxfp4A8FusedMoeRunner:
    """Stateful MXFP4A8 MoE runner with reusable per-layer workspaces."""

    def __init__(self):
        self._workspace: Dict[Tuple[str, Tuple[int, ...], torch.dtype, int], torch.Tensor] = {}

    def _empty(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        key = ("tensor", name, dtype, device.index or 0)
        numel = 1
        for dim in shape:
            numel *= dim
        tensor = self._workspace.get(key)
        if tensor is None or tensor.device != device or tensor.numel() < numel:
            tensor = torch.empty((numel,), dtype=dtype, device=device)
            self._workspace[key] = tensor
        return tensor[:numel].view(shape)

    def _compact_moe_metadata(
        self,
        expert_offsets: torch.Tensor,
        problem_sizes1: torch.Tensor,
        problem_sizes2: torch.Tensor,
        num_experts: int,
        max_groups: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        compact_expert_offsets = self._empty(
            "compact_expert_offsets", (max_groups,), torch.int32, expert_offsets.device
        )
        compact_problem_sizes1 = self._empty(
            "compact_problem_sizes1", (max_groups, 3), torch.int32, expert_offsets.device
        )
        compact_problem_sizes2 = self._empty(
            "compact_problem_sizes2", (max_groups, 3), torch.int32, expert_offsets.device
        )
        compact_expert_ids = self._empty(
            "compact_expert_ids", (max_groups,), torch.int32, expert_offsets.device
        )
        compact_cutlass_w4a8_moe_mm_data(
            expert_offsets,
            problem_sizes1,
            problem_sizes2,
            compact_expert_offsets,
            compact_problem_sizes1,
            compact_problem_sizes2,
            compact_expert_ids,
            num_experts,
            max_groups,
        )
        return (
            compact_expert_offsets,
            compact_problem_sizes1,
            compact_problem_sizes2,
            compact_expert_ids,
        )

    def _quantize_fp8_per_token_into(
        self,
        x: torch.Tensor,
        out_q: torch.Tensor,
        out_s: torch.Tensor,
    ) -> None:
        if x.numel() == 0:
            return
        sgl_per_token_quant_fp8(x, out_q, out_s.view(-1, 1))

    def _mul_per_token_scale_by_expert(
        self,
        scale: torch.Tensor,
        residual: torch.Tensor,
        expert_offsets: torch.Tensor,
        num_experts: int,
    ) -> None:
        if scale.numel() == 0:
            return
        block = 256
        _mul_per_token_scale_by_expert_kernel[(triton.cdiv(scale.numel(), block),)](
            scale,
            residual,
            expert_offsets,
            scale.numel(),
            num_experts,
            BLOCK=block,
        )

    @staticmethod
    def _fused_configs(num_tokens: int) -> Tuple[int, int]:
        if num_tokens <= 64:
            return 100, 100
        if num_tokens == 2048:
            return 313, 313
        if num_tokens == 4096:
            return 320, 334
        if num_tokens >= 8192:
            return 322, 334
        return 101, 101

    def _apply_shuffle_mul_sum_fp32_factors(
        self,
        c2: torch.Tensor,
        output: torch.Tensor,
        c_map: torch.Tensor,
        factors: torch.Tensor,
        routed_scaling_factor: float,
        topk: int,
    ) -> None:
        if output.numel() == 0:
            return
        m, row_stride = output.shape
        block = 256
        _apply_shuffle_mul_sum_fp32_factors_kernel[
            (m, triton.cdiv(row_stride, block))
        ](
            c2,
            output,
            c_map,
            factors,
            m,
            topk,
            row_stride,
            routed_scaling_factor,
            BLOCK=block,
        )

    def __call__(
        self,
        a: torch.Tensor,
        w1_q: torch.Tensor,
        w2_q: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        w1_fused: torch.Tensor,
        w2_fused: torch.Tensor,
        w1_scale_fused: torch.Tensor,
        w2_scale_fused: torch.Tensor,
        w1_residual_fused: torch.Tensor,
        w2_residual_fused: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        a_strides1: torch.Tensor,
        b_strides1: torch.Tensor,
        c_strides1: torch.Tensor,
        a_strides2: torch.Tensor,
        b_strides2: torch.Tensor,
        c_strides2: torch.Tensor,
        s_strides13: torch.Tensor,
        s_strides2: torch.Tensor,
        expert_offsets: torch.Tensor,
        problem_sizes1: torch.Tensor,
        problem_sizes2: torch.Tensor,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        routed_scaling_factor: float = 1.0,
        swiglu_limit: Optional[float] = None,
    ) -> torch.Tensor:
        assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
        assert w1_q.dtype == torch.int8
        assert w2_q.dtype == torch.int8
        assert a.shape[1] // 2 == w1_q.shape[2], "Hidden size mismatch w1"
        assert w1_q.shape[2] * 2 == w2_q.shape[1], "Hidden size mismatch w2"
        assert w1_q.shape[0] == w2_q.shape[0], "Expert number mismatch"
        assert w1_q.shape[0] == w1_scale.shape[0], "w1 scales expert number mismatch"
        assert w1_q.shape[0] == w2_scale.shape[0], "w2 scales expert number mismatch"

        num_local_experts = w1_q.size(0)
        m = a.size(0)
        k = w1_q.size(2) * 2
        n = w2_q.size(2) * 2
        topk = topk_ids.size(1)
        device = a.device

        if apply_router_weight_on_input:
            assert topk == 1, "apply_router_weight_on_input is only implemented for topk=1"

        # The AOT prepare/apply path does not materialize valid c_map entries for
        # the EP sentinel (-1 -> num_local_experts). Keep the legacy Triton path
        # for EP until the sentinel handling is added to prepare_moe_input.
        if get_parallel().moe_ep_size > 1:
            from sglang.srt.layers.moe.cutlass_mxfp4a8_moe import cutlass_mxfp4a8_moe

            return cutlass_mxfp4a8_moe(
                a,
                w1_q,
                w2_q,
                w1_scale,
                w2_scale,
                topk_weights,
                topk_ids,
                a_strides1,
                b_strides1,
                c_strides1,
                a_strides2,
                b_strides2,
                c_strides2,
                s_strides13,
                s_strides2,
                expert_offsets,
                problem_sizes1,
                problem_sizes2,
                a1_scale,
                a2_scale,
                apply_router_weight_on_input,
                routed_scaling_factor,
                swiglu_limit,
            )

        assert w1_fused.dtype == torch.int8
        assert w2_fused.dtype == torch.int8
        assert w1_scale_fused.dtype == torch.uint8
        assert w2_scale_fused.dtype == torch.uint8
        assert w1_residual_fused.shape == (num_local_experts,)
        assert w2_residual_fused.shape == (num_local_experts,)

        topk_ids_i32 = topk_ids.contiguous()
        if topk_ids_i32.dtype != torch.int32:
            topk_ids_i32 = topk_ids_i32.to(torch.int32)

        a_map = self._empty("a_map", (topk_ids.numel(),), torch.int32, device)
        c_map = self._empty("c_map", (topk_ids.numel(),), torch.int32, device)
        prepare_inputs_in_core = m <= 64
        if not prepare_inputs_in_core:
            if get_cutlass_w4a8_moe_mm_data_with_permutation is None:
                prepare_moe_input(
                    topk_ids_i32,
                    expert_offsets,
                    problem_sizes1,
                    problem_sizes2,
                    a_map,
                    c_map,
                    num_local_experts,
                    n,
                    k,
                )
                get_cutlass_w4a8_moe_mm_data(
                    topk_ids_i32,
                    expert_offsets,
                    problem_sizes1,
                    problem_sizes2,
                    a_map,
                    c_map,
                    num_local_experts,
                    n,
                    k,
                )
            else:
                get_cutlass_w4a8_moe_mm_data_with_permutation(
                    topk_ids_i32,
                    expert_offsets,
                    problem_sizes1,
                    problem_sizes2,
                    a_map,
                    c_map,
                    num_local_experts,
                    n,
                    k,
                )

        gateup_input = self._empty(
            "gateup_input_fp8", (m * topk, k), torch.float8_e4m3fn, device
        )
        a1_scale_per_token = self._empty(
            "a1_scale_per_token", (m * topk,), torch.float32, device
        )
        # Compact fused MXFP4A8 groups use one prebuilt weight TMA descriptor per
        # logical group, mapped through active_expert_ids to the source expert.
        # Compact only when routing is sparse enough to remove at least half of
        # the expert groups. At high coverage, scanning and publishing one
        # weight descriptor per logical group costs more than the empty groups.
        use_compact_groups = False
        if use_compact_groups:
            max_compact_groups = max(1, min(num_local_experts, topk_ids.numel()))
            (
                expert_offsets_gemm,
                problem_sizes1_gemm,
                problem_sizes2_gemm,
                active_expert_ids,
            ) = self._compact_moe_metadata(
                expert_offsets,
                problem_sizes1,
                problem_sizes2,
                num_local_experts,
                max_compact_groups,
            )
        else:
            expert_offsets_gemm = expert_offsets[:-1]
            problem_sizes1_gemm = problem_sizes1
            problem_sizes2_gemm = problem_sizes2
            active_expert_ids = None

        a_strides1_gemm = a_strides1
        b_strides1_gemm = b_strides1
        c_strides1_gemm = c_strides1
        s_strides13_gemm = s_strides13
        a_strides2_gemm = a_strides2
        b_strides2_gemm = b_strides2
        c_strides2_gemm = c_strides2
        s_strides2_gemm = s_strides2
        c1_width = n * 2
        c1 = self._empty("c1", (m * topk, c1_width), torch.bfloat16, device)
        c2 = self._empty("c2", (m * topk, k), torch.bfloat16, device)
        gemm1_config, gemm2_config = self._fused_configs(m)

        intermediate_q = self._empty(
            "intermediate_q", (m * topk, n), torch.float8_e4m3fn, device
        )
        a2_scale_per_token = self._empty(
            "a2_scale_per_token", (m * topk,), torch.float32, device
        )
        core_op = torch.ops.sgl_kernel.cutlass_mxfp4a8_fused_moe_core.default

        core_op(
            c1,
            c2,
            a,
            topk_ids_i32,
            a_map,
            c_map,
            a,
            gateup_input,
            a1_scale_per_token,
            intermediate_q,
            a2_scale_per_token,
            w1_fused,
            w1_scale_fused,
            w1_residual_fused,
            w2_fused,
            w2_scale_fused,
            w2_residual_fused,
            expert_offsets,
            expert_offsets_gemm,
            problem_sizes1_gemm,
            problem_sizes2_gemm,
            a_strides1_gemm,
            b_strides1_gemm,
            c_strides1_gemm,
            s_strides13_gemm,
            a_strides2_gemm,
            b_strides2_gemm,
            c_strides2_gemm,
            s_strides2_gemm,
            topk,
            gemm1_config,
            gemm2_config,
            num_local_experts,
            n,
            k,
            float(swiglu_limit or 0.0),
            swiglu_limit is not None,
            prepare_inputs_in_core,
            active_expert_ids,
        )

        output = self._empty("output", tuple(a.shape), a.dtype, device)
        factors = topk_weights.reshape(-1).contiguous()
        self._apply_shuffle_mul_sum_fp32_factors(
            c2, output, c_map, factors, float(routed_scaling_factor), topk
        )
        return output


_DEFAULT_RUNNER = CutlassMxfp4A8FusedMoeRunner()


def cutlass_mxfp4a8_fused_moe(*args, **kwargs) -> torch.Tensor:
    return _DEFAULT_RUNNER(*args, **kwargs)
