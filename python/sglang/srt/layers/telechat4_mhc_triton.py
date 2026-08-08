"""Triton Ascend kernels for TeleChat4's 3584-wide mHC layers."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.runtime.driver as driver

MHC_STREAMS = 4
MHC_HIDDEN_SIZE = 3584
MHC_FLAT_SIZE = MHC_STREAMS * MHC_HIDDEN_SIZE
MHC_OUTPUT_SIZE = MHC_STREAMS * (MHC_STREAMS + 2)
MHC_PADDED_OUTPUT_SIZE = 32


@triton.jit
def _telechat4_mhc_pre_gemm_kernel(
    residual_ptr,
    fn_ptr,
    partial_logits_ptr,
    partial_sqrsum_ptr,
    num_tokens,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_m_blocks = tl.cdiv(num_tokens, BLOCK_M)
    num_tasks = num_m_blocks * SPLIT_K
    split_size: tl.constexpr = K // SPLIT_K

    offs_m_base = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    for task_id in range(pid, num_tasks, num_programs):
        split_id = task_id % SPLIT_K
        block_m_id = task_id // SPLIT_K
        offs_m = block_m_id * BLOCK_M + offs_m_base
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        sqrsum = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for k_offset in range(0, split_size, BLOCK_K):
            offs_k_global = split_id * split_size + k_offset + offs_k
            residual = tl.load(
                residual_ptr + offs_m[:, None] * K + offs_k_global[None, :],
                mask=(offs_m[:, None] < num_tokens) & (offs_k_global[None, :] < K),
                other=0.0,
            )
            fn = tl.load(
                fn_ptr + offs_n[None, :] * K + offs_k_global[:, None],
                mask=(offs_n[None, :] < N) & (offs_k_global[:, None] < K),
                other=0.0,
            )
            acc += tl.dot(residual, fn)
            residual_fp32 = residual.to(tl.float32)
            sqrsum += tl.sum(residual_fp32 * residual_fp32, axis=1)

        logits_offsets = (
            split_id * num_tokens * BLOCK_N
            + offs_m[:, None] * BLOCK_N
            + offs_n[None, :]
        )
        tl.store(
            partial_logits_ptr + logits_offsets,
            acc,
            mask=offs_m[:, None] < num_tokens,
        )
        tl.store(
            partial_sqrsum_ptr + split_id * num_tokens + offs_m,
            sqrsum,
            mask=offs_m < num_tokens,
        )


@triton.jit
def _telechat4_mhc_pre_finalize_kernel(
    residual_ptr,
    partial_logits_ptr,
    partial_sqrsum_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    post_mix_ptr,
    comb_mix_ptr,
    layer_input_ptr,
    num_tokens,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    SPLIT_K: tl.constexpr,
    SINKHORN_REPEAT: tl.constexpr,
    STREAMS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    FLAT_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    POST_LAYOUT: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    split_offsets = tl.arange(0, SPLIT_K)
    stream_offsets = tl.arange(0, STREAMS)
    comb_offsets = tl.arange(0, STREAMS * STREAMS)
    hidden_offsets = tl.arange(0, BLOCK_H)

    for token_id in range(pid, num_tokens, num_programs):
        partial_sqrsum = tl.load(
            partial_sqrsum_ptr + split_offsets * num_tokens + token_id
        )
        inv_rms = tl.rsqrt(tl.sum(partial_sqrsum, axis=0) / FLAT_SIZE + rms_eps)

        pre_partial_offsets = (
            split_offsets[:, None] * num_tokens * BLOCK_N
            + token_id * BLOCK_N
            + stream_offsets[None, :]
        )
        pre_logits = (
            tl.sum(tl.load(partial_logits_ptr + pre_partial_offsets), axis=0) * inv_rms
        )
        pre_scale = tl.load(hc_scale_ptr)
        pre_base = tl.load(hc_base_ptr + stream_offsets)
        pre_mix = tl.sigmoid(pre_logits * pre_scale + pre_base) + hc_pre_eps

        post_partial_offsets = pre_partial_offsets + STREAMS
        post_logits = (
            tl.sum(tl.load(partial_logits_ptr + post_partial_offsets), axis=0) * inv_rms
        )
        post_scale = tl.load(hc_scale_ptr + 1)
        post_base = tl.load(hc_base_ptr + STREAMS + stream_offsets)
        post_mix = tl.sigmoid(post_logits * post_scale + post_base) * hc_post_mult_value

        comb_partial_offsets = (
            split_offsets[:, None] * num_tokens * BLOCK_N
            + token_id * BLOCK_N
            + 2 * STREAMS
            + comb_offsets[None, :]
        )
        comb_logits = (
            tl.sum(tl.load(partial_logits_ptr + comb_partial_offsets), axis=0) * inv_rms
        )
        comb_scale = tl.load(hc_scale_ptr + 2)
        comb_base = tl.load(hc_base_ptr + 2 * STREAMS + comb_offsets)
        comb_mix = tl.reshape(
            comb_logits * comb_scale + comb_base,
            (STREAMS, STREAMS),
        )

        row_max = tl.max(comb_mix, axis=1)
        comb_mix = tl.exp(comb_mix - row_max[:, None])
        row_sum = tl.sum(comb_mix, axis=1)
        comb_mix = comb_mix / row_sum[:, None] + hc_sinkhorn_eps
        col_sum = tl.sum(comb_mix, axis=0)
        comb_mix = comb_mix / (col_sum[None, :] + hc_sinkhorn_eps)
        for _ in range(SINKHORN_REPEAT - 1):
            row_sum = tl.sum(comb_mix, axis=1)
            comb_mix = comb_mix / (row_sum[:, None] + hc_sinkhorn_eps)
            col_sum = tl.sum(comb_mix, axis=0)
            comb_mix = comb_mix / (col_sum[None, :] + hc_sinkhorn_eps)

        residual_offsets = (
            token_id * FLAT_SIZE
            + stream_offsets[:, None] * HIDDEN_SIZE
            + hidden_offsets[None, :]
        )
        residual = tl.load(
            residual_ptr + residual_offsets,
            mask=hidden_offsets[None, :] < HIDDEN_SIZE,
            other=0.0,
        ).to(tl.float32)
        layer_input = tl.sum(residual * pre_mix[:, None], axis=0)

        tl.store(
            post_mix_ptr + token_id * STREAMS + stream_offsets,
            post_mix,
        )
        if POST_LAYOUT:
            comb_output_offsets = (
                comb_offsets % STREAMS
            ) * STREAMS + comb_offsets // STREAMS
        else:
            comb_output_offsets = comb_offsets
        tl.store(
            comb_mix_ptr + token_id * STREAMS * STREAMS + comb_output_offsets,
            tl.reshape(comb_mix, (STREAMS * STREAMS,)),
        )
        tl.store(
            layer_input_ptr + token_id * HIDDEN_SIZE + hidden_offsets,
            layer_input,
            mask=hidden_offsets < HIDDEN_SIZE,
        )


@triton.jit
def _telechat4_mhc_post_kernel(
    x_ptr,
    residual_ptr,
    post_mix_ptr,
    comb_mix_ptr,
    output_ptr,
    num_tokens,
    STREAMS: tl.constexpr,
    FLAT_SIZE: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_tasks = num_tokens * STREAMS
    hidden_offsets = tl.arange(0, BLOCK_H)
    old_stream_offsets = tl.arange(0, STREAMS)

    for task_id in range(pid, num_tasks, num_programs):
        token_id = task_id // STREAMS
        new_stream_id = task_id % STREAMS
        hidden_mask = hidden_offsets < HIDDEN_SIZE
        x = tl.load(
            x_ptr + token_id * HIDDEN_SIZE + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        post_mix = tl.load(post_mix_ptr + token_id * STREAMS + new_stream_id)
        residual_offsets = (
            token_id * FLAT_SIZE
            + old_stream_offsets[:, None] * HIDDEN_SIZE
            + hidden_offsets[None, :]
        )
        residual = tl.load(
            residual_ptr + residual_offsets,
            mask=hidden_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        comb_offsets = (
            token_id * STREAMS * STREAMS + old_stream_offsets * STREAMS + new_stream_id
        )
        comb_mix = tl.load(comb_mix_ptr + comb_offsets)
        output = post_mix * x + tl.sum(residual * comb_mix[:, None], axis=0)
        tl.store(
            output_ptr
            + token_id * FLAT_SIZE
            + new_stream_id * HIDDEN_SIZE
            + hidden_offsets,
            output,
            mask=hidden_mask,
        )


def _validate_pre_inputs(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
) -> None:
    if residual.shape[-2:] != (MHC_STREAMS, MHC_HIDDEN_SIZE):
        raise ValueError(
            "TeleChat4 Triton mHC requires residual[..., 4, 3584], "
            f"got {tuple(residual.shape)}"
        )
    if residual.dtype != torch.bfloat16 or fn.dtype != torch.bfloat16:
        raise TypeError("TeleChat4 Triton mHC requires BF16 residual and fn")
    if fn.shape != (MHC_OUTPUT_SIZE, MHC_FLAT_SIZE):
        raise ValueError(
            f"TeleChat4 Triton mHC requires fn[24, 14336], got {tuple(fn.shape)}"
        )
    if hc_scale.shape != (3,) or hc_scale.dtype != torch.float32:
        raise ValueError("TeleChat4 Triton mHC requires FP32 hc_scale[3]")
    if hc_base.shape != (MHC_OUTPUT_SIZE,) or hc_base.dtype != torch.float32:
        raise ValueError("TeleChat4 Triton mHC requires FP32 hc_base[24]")
    if not all(t.is_contiguous() for t in (residual, fn, hc_scale, hc_base)):
        raise ValueError("TeleChat4 Triton mHC inputs must be contiguous")


def _telechat4_mhc_pre_split(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    split_k: int = 8,
    direct_post_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if split_k not in (1, 2, 4, 8, 16):
        raise ValueError(f"split_k must be one of 1, 2, 4, 8, 16, got {split_k}")
    if MHC_FLAT_SIZE % split_k != 0:
        raise ValueError(f"split_k={split_k} does not divide {MHC_FLAT_SIZE}")

    num_tokens = residual.numel() // MHC_FLAT_SIZE
    residual_flat = residual.view(num_tokens, MHC_FLAT_SIZE)
    partial_logits = torch.empty(
        split_k,
        num_tokens,
        MHC_PADDED_OUTPUT_SIZE,
        dtype=torch.float32,
        device=residual.device,
    )
    partial_sqrsum = torch.empty(
        split_k, num_tokens, dtype=torch.float32, device=residual.device
    )
    post_mix = torch.empty(
        num_tokens,
        MHC_STREAMS,
        1,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_post_layout = torch.empty(
        num_tokens,
        MHC_STREAMS,
        MHC_STREAMS,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input = torch.empty(
        num_tokens,
        MHC_HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    num_aicore = driver.active.utils.get_device_properties(residual.device.index or 0)[
        "num_aicore"
    ]
    block_m = 16
    num_tasks = triton.cdiv(num_tokens, block_m) * split_k
    _telechat4_mhc_pre_gemm_kernel[(min(num_tasks, num_aicore),)](
        residual_flat,
        fn,
        partial_logits,
        partial_sqrsum,
        num_tokens,
        SPLIT_K=split_k,
        BLOCK_M=block_m,
        BLOCK_N=MHC_PADDED_OUTPUT_SIZE,
        BLOCK_K=128,
        K=MHC_FLAT_SIZE,
        N=MHC_OUTPUT_SIZE,
        num_stages=2,
    )

    num_vectorcore = driver.active.utils.get_device_properties(
        residual.device.index or 0
    )["num_vectorcore"]
    _telechat4_mhc_pre_finalize_kernel[(min(num_tokens, num_vectorcore),)](
        residual_flat,
        partial_logits,
        partial_sqrsum,
        hc_scale,
        hc_base,
        post_mix,
        comb_post_layout,
        layer_input,
        num_tokens,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        SPLIT_K=split_k,
        SINKHORN_REPEAT=sinkhorn_repeat,
        STREAMS=MHC_STREAMS,
        HIDDEN_SIZE=MHC_HIDDEN_SIZE,
        FLAT_SIZE=MHC_FLAT_SIZE,
        BLOCK_H=4096,
        BLOCK_N=MHC_PADDED_OUTPUT_SIZE,
        POST_LAYOUT=direct_post_layout,
        num_stages=1,
    )
    comb_mix = (
        comb_post_layout.transpose(-1, -2) if direct_post_layout else comb_post_layout
    )
    return post_mix, comb_mix, layer_input


def telechat4_mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    split_k: int = 8,
    implementation: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run TeleChat4 mHC pre-mixing with split-K Ascend Triton kernels."""
    _validate_pre_inputs(residual, fn, hc_scale, hc_base)
    if implementation not in ("auto", "split", "split_direct"):
        raise ValueError(
            "implementation must be 'auto', 'split', or 'split_direct', "
            f"got {implementation!r}"
        )

    if implementation == "auto":
        from sglang.srt.runtime_context import get_forward

        direct_post_layout = bool(get_forward().is_extend_in_batch)
    else:
        direct_post_layout = implementation == "split_direct"
    return _telechat4_mhc_pre_split(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        split_k,
        direct_post_layout,
    )


def telechat4_mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
) -> torch.Tensor:
    """Run the 3584-wide TeleChat4 mHC post-mixing kernel on Ascend."""
    if x.ndim != 2 or x.shape[1] != MHC_HIDDEN_SIZE:
        raise ValueError(f"expected x[num_tokens, 3584], got {tuple(x.shape)}")
    if residual.shape != (x.shape[0], MHC_STREAMS, MHC_HIDDEN_SIZE):
        raise ValueError(
            f"expected residual[{x.shape[0]}, 4, 3584], got {tuple(residual.shape)}"
        )
    if post_mix.shape not in (
        (x.shape[0], MHC_STREAMS),
        (x.shape[0], MHC_STREAMS, 1),
    ):
        raise ValueError(f"unexpected post_mix shape {tuple(post_mix.shape)}")
    if comb_mix.shape != (x.shape[0], MHC_STREAMS, MHC_STREAMS):
        raise ValueError(f"unexpected comb_mix shape {tuple(comb_mix.shape)}")
    if x.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16:
        raise TypeError("TeleChat4 Triton mHC post requires BF16 x and residual")
    if post_mix.dtype != torch.float32 or comb_mix.dtype != torch.float32:
        raise TypeError("TeleChat4 Triton mHC post requires FP32 mixing coefficients")
    if not all(t.is_contiguous() for t in (x, residual, post_mix, comb_mix)):
        raise ValueError("TeleChat4 Triton mHC post inputs must be contiguous")

    output = torch.empty(residual.shape, device=residual.device, dtype=residual.dtype)
    num_tokens = x.shape[0]
    num_vectorcore = driver.active.utils.get_device_properties(x.device.index or 0)[
        "num_vectorcore"
    ]
    _telechat4_mhc_post_kernel[(min(num_tokens * MHC_STREAMS, num_vectorcore),)](
        x,
        residual,
        post_mix,
        comb_mix,
        output,
        num_tokens,
        STREAMS=MHC_STREAMS,
        FLAT_SIZE=MHC_FLAT_SIZE,
        HIDDEN_SIZE=MHC_HIDDEN_SIZE,
        BLOCK_H=4096,
        num_stages=1,
    )
    return output
