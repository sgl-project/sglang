# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL FP8 Paged MQA Logits runner and custom op.

Ported from TensorRT-LLM https://github.com/NVIDIA/TensorRT-LLM/pull/13219
Provides ``torch.ops.sglang.cute_dsl_fp8_paged_mqa_logits`` as an alternative
to ``deep_gemm.fp8_paged_mqa_logits`` on Blackwell SM100. It performs well, when the bs is low,
and the context is long.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass.utils import HardwareInfo

from sglang.jit_kernel.cutedsl_fp8_paged_mqa_logits import FP8MQALogitsKernel
from sglang.srt.utils import is_sm100_supported

logger = logging.getLogger(__name__)


def _pick_dsl_expand(
    next_n: int,
    batch_size: int = 0,
    max_ctx: int = 0,
    num_sms: int = 148,
    kernel_atoms: Tuple[int, ...] = (1, 2, 3, 4),
) -> Tuple[int, int]:
    """Pick (expand_factor, effective_next_n) for the DSL paged kernel
    using a wave-aware strategy.

    The DSL FP8 kernel natively supports ``effective_next_n ∈ kernel_atoms``
    (default ``(1, 2, 3, 4)``). When SM utilization can be improved, reshape
    ``[B, next_n, ...]`` -> ``[B * expand_factor, effective_next_n, ...]``
    caller-side.

    Strategy: enumerate ``(expand_factor, effective_next_n)`` pairs with
    ``expand_factor * effective_next_n == next_n`` and ``effective_next_n
    in kernel_atoms``. Score each by ``(waves, -expand_factor)`` where
    ``waves = ceil(B * expand_factor * ceil(max_ctx/256) / num_sms)``.
    Pick min waves; on tie, prefer LARGER expand_factor (more SMs busy per
    wave; pays HBM cost of expand_factor x KV re-reads).

    When ``batch_size == 0`` or ``max_ctx == 0`` (workload unknown), fall
    back to the legacy HBM-minimizing heuristic: largest effective_next_n
    that divides next_n cleanly (still constrained to ``kernel_atoms``).
    """
    if batch_size <= 0 or max_ctx <= 0:
        for eff in sorted(kernel_atoms, reverse=True):
            if next_n % eff == 0:
                return next_n // eff, eff
        return next_n, 1

    SPLIT_KV_TOKENS = 256
    cands = []
    for eff in kernel_atoms:
        if next_n % eff == 0:
            factor = next_n // eff
            ntask = (
                batch_size
                * factor
                * ((max_ctx + SPLIT_KV_TOKENS - 1) // SPLIT_KV_TOKENS)
            )
            waves = (ntask + num_sms - 1) // num_sms
            cands.append((waves, factor, eff))
    if not cands:
        return next_n, 1
    cands.sort(key=lambda x: (x[0], -x[1]))
    _, factor, eff = cands[0]
    return factor, eff


_TORCH_TO_CUTLASS_DTYPE = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


class CuteDSLPagedMQALogitsRunner:
    """Runner for CuTe DSL FP8 Paged MQA Logits kernel (Blackwell SM100).

    Caches compiled kernels keyed by static params
    (compute_block_kv, phys_block_kv, num_heads, head_dim, next_n, num_sms).
    """

    kernel_cache: Dict[Tuple, object] = dict()

    @classmethod
    def _compile(
        cls,
        compute_block_kv,
        phys_block_kv,
        num_heads,
        head_dim,
        next_n,
        num_sms,
        num_epi_subtiles,
        epi_dtype,
        acc_dtype,
        output_dtype,
    ):
        """Compile kernel using fake tensors + TVM FFI."""
        key = (
            compute_block_kv,
            phys_block_kv,
            num_heads,
            head_dim,
            next_n,
            num_sms,
            num_epi_subtiles,
            epi_dtype,
            acc_dtype,
            output_dtype,
        )
        if key in cls.kernel_cache:
            return

        to_cutlass = _TORCH_TO_CUTLASS_DTYPE
        N = next_n * num_heads
        block_bytes = phys_block_kv * (head_dim + 4)

        sym_num_phys_blocks = cute.sym_int()
        sym_B = cute.sym_int()
        max_ctx = cute.sym_int()
        max_blocks_per_seq = cute.sym_int()
        num_ctas = cute.sym_int()

        kv_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Uint8,
            (sym_num_phys_blocks, block_bytes),
            stride_order=(1, 0),
        )

        q_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Uint8, (N, head_dim, sym_B), stride_order=(1, 0, 2)
        )

        w_dtype = (
            cutlass.Float16 if epi_dtype == torch.float16 else to_cutlass[epi_dtype]
        )
        w_fake = cute.runtime.make_fake_compact_tensor(
            w_dtype, (N, sym_B), stride_order=(0, 1)
        )

        logits_fake = cute.runtime.make_fake_tensor(
            to_cutlass[output_dtype],
            (cute.sym_int(), max_ctx),
            stride=(cute.sym_int64(), 1),
        )

        bt_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32, (sym_B, max_blocks_per_seq), stride_order=(1, 0)
        )

        cl_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32, (sym_B,), stride_order=(0,)
        )

        sm_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32, (num_ctas, 2), stride_order=(1, 0)
        )

        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

        kernel = FP8MQALogitsKernel(
            block_kv=compute_block_kv,
            phys_block_kv=phys_block_kv,
            num_heads=num_heads,
            head_dim=head_dim,
            next_n=next_n,
            num_sms=num_sms,
            num_epi_subtiles=num_epi_subtiles,
            epi_dtype=to_cutlass[epi_dtype],
            acc_dtype=to_cutlass[acc_dtype],
            output_dtype=to_cutlass[output_dtype],
        )

        compiled = cute.compile(
            kernel,
            kv_fake,
            q_fake,
            w_fake,
            logits_fake,
            bt_fake,
            cl_fake,
            sm_fake,
            cutlass.Int32(1),
            cutlass.Int32(1),
            fake_stream,
            options="--enable-tvm-ffi",
        )
        cls.kernel_cache[key] = compiled
        logger.debug(
            f"[compile cute_dsl fp8_paged_mqa_logits] {key}"
            f" kv_stages={kernel.num_kv_stages}"
            f" umma_stages={kernel.num_umma_stages}"
        )

    @classmethod
    def forward(
        cls,
        q: torch.Tensor,
        kv_fused: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        max_context_len: int,
        num_epi_subtiles: int = 1,
        epi_dtype: torch.dtype = torch.float32,
        acc_dtype: torch.dtype = torch.float32,
        output_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Execute FP8 paged MQA logits kernel.

        Args:
            q: [B, next_n, H, D] FP8
            kv_fused: [num_blocks, phys_block_kv, 1, D+4] uint8
            weights: [B*next_n, H] float32
            context_lens: [B] int32
            block_table: [B, max_blocks] int32
            schedule_meta: [num_sms+1, 2] int32
            max_context_len: int
            num_epi_subtiles: epilogue sub-tile count (1, 2, or 4)
            epi_dtype: epilogue compute dtype
            acc_dtype: MMA accumulator dtype
            output_dtype: output logits dtype
        Returns:
            logits: [B*next_n, max_context_len] output_dtype
        """
        B, next_n, H, D = q.shape
        N = next_n * H
        phys_block_kv = kv_fused.shape[1]
        compute_block_kv = 128
        num_phys_blocks = kv_fused.shape[0]
        num_sms = HardwareInfo().get_device_multiprocessor_count()

        # Reshape Q: [B, next_n, H, D] -> [B, N, D] -> [N, D, B]
        q_3d = q.reshape(B, N, D).permute(1, 2, 0)

        # Reshape weights: [B*next_n, H] -> [B, N] -> [N, B]
        if epi_dtype == torch.float16:
            # TODO: move type conversion to weight loading
            w_2d = weights.reshape(B, N).half().t()
        else:
            w_2d = weights.reshape(B, N).t()

        # Flatten fused KV to [num_phys_blocks, block_bytes]
        kv_flat = kv_fused.reshape(num_phys_blocks, -1)

        # Allocate output with alignment padding
        SPLIT_KV = compute_block_kv * 2  # NUM_MATH_WG = 2
        aligned_max_ctx = ((max_context_len + SPLIT_KV - 1) // SPLIT_KV) * SPLIT_KV
        logits = torch.empty(
            (B * next_n, aligned_max_ctx),
            device=q.device,
            dtype=output_dtype,
        )
        logits = logits[:, :max_context_len]

        key = (
            compute_block_kv,
            phys_block_kv,
            H,
            D,
            next_n,
            num_sms,
            num_epi_subtiles,
            epi_dtype,
            acc_dtype,
            output_dtype,
        )
        if key not in cls.kernel_cache:
            cls._compile(
                compute_block_kv,
                phys_block_kv,
                H,
                D,
                next_n,
                num_sms,
                num_epi_subtiles,
                epi_dtype,
                acc_dtype,
                output_dtype,
            )
        compiled = cls.kernel_cache[key]

        # FP8 q needs uint8 view to match compile-time dtype
        q_for_ffi = (
            q_3d.view(torch.uint8)
            if q_3d.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            else q_3d
        )

        compiled(
            kv_flat,
            q_for_ffi,
            w_2d,
            logits,
            block_table,
            context_lens,
            schedule_meta,
            num_phys_blocks,
            B,
        )
        return logits


@torch.library.custom_op(
    "sglang::cute_dsl_fp8_paged_mqa_logits",
    mutates_args=(),
    device_types="cuda",
)
def cute_dsl_fp8_paged_mqa_logits(
    q: torch.Tensor,
    kv_fused: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_meta: torch.Tensor,
    max_context_len: int,
    num_epi_subtiles: int = 1,
    epi_dtype: torch.dtype = torch.float32,
    acc_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if not is_sm100_supported():
        raise ValueError("CuteDSL FP8 Paged MQA Logits only supports SM 100 family.")
    return CuteDSLPagedMQALogitsRunner.forward(
        q,
        kv_fused,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_context_len,
        num_epi_subtiles=num_epi_subtiles,
        epi_dtype=epi_dtype,
        acc_dtype=acc_dtype,
        output_dtype=output_dtype,
    )


@torch.library.register_fake("sglang::cute_dsl_fp8_paged_mqa_logits")
def _(
    q: torch.Tensor,
    kv_fused: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_meta: torch.Tensor,
    max_context_len: int,
    num_epi_subtiles: int = 1,
    epi_dtype: torch.dtype = torch.float32,
    acc_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    B = q.shape[0]
    next_n = q.shape[1]
    return torch.empty(
        B * next_n,
        max_context_len,
        dtype=output_dtype,
        device=q.device,
    )
