"""
DCP All-to-All communication backend for Decode Context Parallelism.

Alternative to the AG+RS path (cp_lse_ag_out_rs in utils.py).
After local attention produces partial outputs and LSEs for all heads
over a local KV shard, A2A exchanges head partials across DCP ranks,
then a Triton kernel combines them locally using LSE-weighted merging.

Ported from vLLM's dcp_alltoall.py with adaptations for SGLang's
GroupCoordinator API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator


# ---------------------------------------------------------------------------
# FlashInfer MNNVL all-to-all backend (fi_a2a)
#
# Optional: delegates only the cross-rank EXCHANGE to FlashInfer's LL128 FIFO
# all-to-all kernel (flashinfer #2951). Requires SM90+ and MNNVL fabric memory
# (GB200 NVL72) — it would deadlock on non-MNNVL clusters. FlashInfer exchanges
# partial_o + softmax_stats separately; the LSE-weighted combine still uses the
# local Triton kernel below. All FlashInfer imports are lazy so the a2a / ag_rs
# paths work without a FlashInfer install.
# ---------------------------------------------------------------------------

# Per-process singleton: MNNVL workspace + this rank's cp position. Populated
# once, pre-CUDA-graph-capture, by init_fi_a2a_workspace().
_FI_A2A_STATE: Optional[dict] = None


def init_fi_a2a_workspace(cp_group: "GroupCoordinator") -> None:
    """Allocate + initialize the FlashInfer MNNVL DCP all-to-all workspace.

    MUST be called exactly once per process, BEFORE any CUDA-graph capture: the
    FlashInfer init synchronizes the stream and this routine issues a cross-rank
    barrier, neither of which is capturable. Raises a clear error if FlashInfer
    is unavailable or the platform lacks MNNVL fabric memory (the kernel would
    otherwise deadlock at runtime).
    """
    global _FI_A2A_STATE
    if _FI_A2A_STATE is not None:
        return
    if cp_group.world_size == 1:
        return

    import torch.distributed as dist

    try:
        from flashinfer.comm.dcp_alltoall import (
            decode_cp_a2a_allocate_mnnvl_workspace,
            decode_cp_a2a_init_workspace,
        )
        from flashinfer.comm.mapping import Mapping
        from flashinfer.comm.mnnvl import MnnvlConfig, is_mnnvl_fabric_supported
    except ImportError as e:
        raise ImportError(
            "--dcp-comm-backend fi_a2a requires FlashInfer with the DCP "
            "all-to-all kernel (flashinfer #2951); could not import "
            "flashinfer.comm.dcp_alltoall."
        ) from e

    from sglang.srt.layers.moe.token_dispatcher.flashinfer_utils import (
        TorchDistributedCommBackend,
    )

    if not is_mnnvl_fabric_supported(torch.cuda.current_device()):
        raise RuntimeError(
            "--dcp-comm-backend fi_a2a requires MNNVL fabric memory (e.g. "
            "GB200 NVL72); is_mnnvl_fabric_supported() returned False. Use "
            "--dcp-comm-backend a2a or ag_rs on clusters without MNNVL."
        )

    cp_size = cp_group.world_size
    cp_rank = cp_group.rank_in_group
    # Pure-DCP mapping: CP is the only parallel axis of this group, so
    # world_size == cp_size and tp_size == pp_size == 1.
    mapping = Mapping(
        world_size=cp_size,
        rank=cp_rank,
        gpus_per_node=torch.cuda.device_count(),
        cp_size=cp_size,
        tp_size=1,
        pp_size=1,
    )
    workspace = decode_cp_a2a_allocate_mnnvl_workspace(
        mapping,
        mnnvl_config=MnnvlConfig(
            comm_backend=TorchDistributedCommBackend(cp_group.device_group)
        ),
    )
    decode_cp_a2a_init_workspace(workspace, cp_rank, cp_size)
    # REQUIRED cross-rank barrier before the first alltoall (flashinfer
    # dcp_alltoall docstring): every rank must finish init first, else a rank
    # may write a peer's FIFO before that peer is ready -> deadlock.
    dist.barrier(group=cp_group.device_group)
    _FI_A2A_STATE = {
        "workspace": workspace,
        "cp_rank": cp_rank,
        "cp_size": cp_size,
    }


# ---------------------------------------------------------------------------
# Triton kernel: LSE-weighted combine of N partial attention outputs
# ---------------------------------------------------------------------------


@triton.jit
def _dcp_lse_combine_kernel(
    recv_output_ptr,
    recv_lse_ptr,
    out_ptr,
    out_lse_ptr,
    recv_output_stride_N,
    recv_output_stride_B,
    recv_output_stride_H,
    recv_output_stride_D,
    recv_lse_stride_N,
    recv_lse_stride_B,
    recv_lse_stride_H,
    out_stride_B,
    out_stride_H,
    out_stride_D,
    N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_BASE_E: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    """Combine N partial attention outputs weighted by their LSE values.

    Grid: (B, H_local).
    Each program handles one (batch, head) position across all N shards.

    Two-pass approach:
    Pass 1: find max LSE and weight sum across shards
    Pass 2: accumulate weighted outputs
    """
    batch_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)

    lse_base = batch_idx * recv_lse_stride_B + head_idx * recv_lse_stride_H

    # Pass 1: find max LSE across N shards
    lse_max = tl.load(recv_lse_ptr + lse_base).to(tl.float32)
    lse_max = tl.where(
        (lse_max != lse_max) | (lse_max == float("inf")), -float("inf"), lse_max
    )
    for i in tl.static_range(1, N):
        lse_i = tl.load(recv_lse_ptr + lse_base + i * recv_lse_stride_N).to(tl.float32)
        lse_i = tl.where(
            (lse_i != lse_i) | (lse_i == float("inf")), -float("inf"), lse_i
        )
        lse_max = tl.where(lse_i > lse_max, lse_i, lse_max)

    lse_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)

    # Pass 2: accumulate weighted outputs
    weight_sum = tl.zeros([], dtype=tl.float32)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for i in tl.static_range(N):
        lse_i = tl.load(recv_lse_ptr + lse_base + i * recv_lse_stride_N).to(tl.float32)
        lse_i = tl.where(
            (lse_i != lse_i) | (lse_i == float("inf")), -float("inf"), lse_i
        )
        centered = lse_i - lse_max
        if IS_BASE_E:
            w = tl.exp(centered)
        else:
            w = tl.exp2(centered)
        weight_sum += w

        o_offsets = (
            i * recv_output_stride_N
            + batch_idx * recv_output_stride_B
            + head_idx * recv_output_stride_H
            + d_offsets * recv_output_stride_D
        )
        partial_out = tl.load(recv_output_ptr + o_offsets).to(tl.float32)
        acc += partial_out * w

    acc = acc / weight_sum

    out_offsets = (
        batch_idx * out_stride_B + head_idx * out_stride_H + d_offsets * out_stride_D
    )
    tl.store(out_ptr + out_offsets, acc.to(out_ptr.dtype.element_ty))

    if RETURN_LSE:
        if IS_BASE_E:
            global_lse = tl.log(weight_sum) + lse_max
        else:
            global_lse = tl.log2(weight_sum) + lse_max
        out_lse_offset = batch_idx * recv_lse_stride_B + head_idx * recv_lse_stride_H
        tl.store(out_lse_ptr + out_lse_offset, global_lse)


# ---------------------------------------------------------------------------
# Triton launcher
# ---------------------------------------------------------------------------


def dcp_lse_combine_triton(
    recv_output: torch.Tensor,
    recv_lse: torch.Tensor,
    is_lse_base_on_e: bool = True,
    return_lse: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Launch the Triton LSE-combine kernel.

    Args:
        recv_output: [N, B, H_local, D] partial outputs from each DCP rank.
        recv_lse:    [N, B, H_local]    log-sum-exp from each DCP rank.
        is_lse_base_on_e: True if LSE uses base-e (FlashAttention),
                          False if base-2 (FlashInfer).
        return_lse: If True, also return the combined global LSE.

    Returns:
        (combined_output [B, H_local, D], combined_lse [B, H_local] or None)
    """
    N, B, H_local, D = recv_output.shape
    out = torch.empty(
        (B, H_local, D), device=recv_output.device, dtype=recv_output.dtype
    )
    out_lse = (
        torch.empty((B, H_local), device=recv_lse.device, dtype=recv_lse.dtype)
        if return_lse
        else recv_lse.new_empty(0)
    )

    grid = (B, H_local)
    _dcp_lse_combine_kernel[grid](
        recv_output,
        recv_lse,
        out,
        out_lse,
        recv_output.stride(0),
        recv_output.stride(1),
        recv_output.stride(2),
        recv_output.stride(3),
        recv_lse.stride(0),
        recv_lse.stride(1),
        recv_lse.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        N=N,
        HEAD_DIM=D,
        IS_BASE_E=is_lse_base_on_e,
        RETURN_LSE=return_lse,
    )
    return out, (out_lse if return_lse else None)


# ---------------------------------------------------------------------------
# Orchestrator — main entry point
# ---------------------------------------------------------------------------


def _lse_pack_dim(output_dtype: torch.dtype) -> int:
    """Number of output-dtype elements needed to store one fp32 LSE value."""
    return torch.finfo(torch.float32).bits // torch.finfo(output_dtype).bits


def dcp_a2a_lse_reduce(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: "GroupCoordinator",
    is_lse_base_on_e: bool = True,
    cuda_graph_buffers: Optional[dict] = None,
    comm_backend: str = "a2a",
) -> torch.Tensor:
    """A2A-based DCP reduce: exchange head partials, then local combine.

    Fuses output + LSE into a single all_to_all call by packing fp32 LSE
    as reinterpreted output-dtype elements along the D dimension:
      combined = [N, B, H_per_rank, D + lse_pack_dim]
    This halves the NCCL calls (1 instead of 2 per layer, 27 fewer per step).

    Args:
        cp_attn_out: [B, H, D]  attention output (all heads, local KV shard)
        cp_attn_lse: [B, H]     log-sum-exp values (fp32)
        cp_group:    DCP GroupCoordinator
        is_lse_base_on_e: True for FlashAttention (base-e), False for FlashInfer (base-2)
        cuda_graph_buffers: Pre-allocated buffers for CUDA graph mode.
            Keys: send_combined, recv_combined  [N, bs, H_per_rank, D+lse_pack_dim]
                  send_lse, recv_lse            [N, bs, H_per_rank] fp32 staging

    Returns:
        [B, H_local, D] combined attention output for this rank's local heads.
    """
    if cp_group.world_size == 1:
        return cp_attn_out

    if comm_backend == "fi_a2a":
        return _dcp_fi_a2a_lse_reduce(
            cp_attn_out, cp_attn_lse, cp_group, is_lse_base_on_e
        )

    N = cp_group.world_size
    B, H, D = cp_attn_out.shape
    assert H % N == 0, f"num_heads ({H}) must be divisible by dcp_size ({N})"
    H_per_rank = H // N
    out_dtype = cp_attn_out.dtype
    lpd = _lse_pack_dim(out_dtype)  # 2 for bf16/fp16

    # Reshape [B, H, D] -> [N, B, H/N, D] — split heads across ranks
    reshaped_out = cp_attn_out.view(B, N, H_per_rank, D).permute(1, 0, 2, 3)
    reshaped_lse = cp_attn_lse.view(B, N, H_per_rank).permute(1, 0, 2)

    if cuda_graph_buffers is not None:
        # CUDA graph path with pre-allocated fused buffers.
        send_combined = cuda_graph_buffers["send_combined"]
        recv_combined = cuda_graph_buffers["recv_combined"]
        send_lse_stg = cuda_graph_buffers["send_lse"]
        recv_lse_stg = cuda_graph_buffers["recv_lse"]

        # Pack output into [:D] columns
        send_combined[:, :B, :, :D].copy_(reshaped_out)
        # Pack LSE: fp32 → view as output dtype → copy into [D:] columns
        send_lse_stg[:, :B, :].copy_(reshaped_lse)
        send_combined[:, :, :, D:].copy_(
            send_lse_stg.view(out_dtype).view(N, -1, H_per_rank, lpd)
        )

        # Single fused all_to_all
        cp_group.all_to_all_single(recv_combined.view(-1), send_combined.view(-1))

        # Unpack output (non-contiguous view — Triton handles strides)
        recv_output = recv_combined[:, :B, :, :D]
        # Unpack LSE: copy [D:] columns back to fp32 staging buffer
        recv_lse_stg.view(out_dtype).view(N, -1, H_per_rank, lpd).copy_(
            recv_combined[:, :, :, D:]
        )
        recv_lse = recv_lse_stg[:, :B, :]
    else:
        # Eager path: allocate fused buffer on the fly
        send_lse_contig = reshaped_lse.contiguous()  # [N, B, H_per_rank] fp32
        send_combined = torch.empty(
            N,
            B,
            H_per_rank,
            D + lpd,
            dtype=out_dtype,
            device=cp_attn_out.device,
        )
        recv_combined = torch.empty_like(send_combined)

        send_combined[:, :, :, :D].copy_(reshaped_out)
        send_combined[:, :, :, D:].copy_(
            send_lse_contig.view(out_dtype).view(N, B, H_per_rank, lpd)
        )

        cp_group.all_to_all_single(recv_combined.view(-1), send_combined.view(-1))

        recv_output = recv_combined[:, :, :, :D]
        recv_lse_stg = torch.empty(
            N,
            B,
            H_per_rank,
            dtype=torch.float32,
            device=cp_attn_out.device,
        )
        recv_lse_stg.view(out_dtype).view(N, B, H_per_rank, lpd).copy_(
            recv_combined[:, :, :, D:]
        )
        recv_lse = recv_lse_stg

    combined, _ = dcp_lse_combine_triton(
        recv_output, recv_lse, is_lse_base_on_e=is_lse_base_on_e
    )
    return combined


def _dcp_fi_a2a_lse_reduce(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: "GroupCoordinator",
    is_lse_base_on_e: bool = True,
) -> torch.Tensor:
    """fi_a2a variant: delegate only the cross-rank EXCHANGE to FlashInfer's
    MNNVL all-to-all kernel, then reuse the same local Triton LSE combine.

    FlashInfer takes output and LSE as SEPARATE tensors (no manual packing):
      partial_o     : [B, H_per_rank, cp_size, D]   (peer axis second-to-last)
      softmax_stats : [B, H_per_rank, cp_size, 2]   fp32, S padded 1->2
    """
    from flashinfer.comm.dcp_alltoall import decode_cp_a2a_alltoall

    state = _FI_A2A_STATE
    assert state is not None, (
        "fi_a2a workspace not initialized — call init_fi_a2a_workspace(dcp_group) "
        "at model-runner init (before CUDA graph capture)."
    )

    N = cp_group.world_size
    B, H, D = cp_attn_out.shape
    assert H % N == 0, f"num_heads ({H}) must be divisible by dcp_size ({N})"
    H_per_rank = H // N

    # partial_o: FlashInfer sends partial_o[..., peer, :] to `peer`. Head h maps
    # to (rank=h//H_per_rank, local_head=h%H_per_rank), so the peer axis is the
    # outer head split. [B, N, H_per_rank, D] -> [B, H_per_rank, N, D].
    partial_o = (
        cp_attn_out.view(B, N, H_per_rank, D).permute(0, 2, 1, 3).contiguous()
    )
    # softmax_stats: fp32 [B, H_per_rank, N, S=2] (FI requires S>=2 & even);
    # carry the LSE in lane 0, lane 1 is ignored by the combine.
    lse_view = cp_attn_lse.view(B, N, H_per_rank).permute(0, 2, 1)  # [B,H_pr,N]
    softmax_stats = torch.zeros(
        B, H_per_rank, N, 2, dtype=torch.float32, device=cp_attn_out.device
    )
    softmax_stats[..., 0] = lse_view

    o_out, stats_out = decode_cp_a2a_alltoall(
        partial_o,
        softmax_stats,
        state["workspace"],
        state["cp_rank"],
        N,
    )

    # o_out[b, hpr, src, :] = rank `src`'s partial for this rank's local head
    # hpr -> reshape to the combine kernel's [N, B, H_local, D] / [N, B, H_local].
    recv_output = o_out.permute(2, 0, 1, 3).contiguous()  # [N, B, H_per_rank, D]
    recv_lse = stats_out[..., 0].permute(2, 0, 1).contiguous()  # [N, B, H_per_rank]

    combined, _ = dcp_lse_combine_triton(
        recv_output, recv_lse, is_lse_base_on_e=is_lse_base_on_e
    )
    return combined


# ---------------------------------------------------------------------------
# CPU reference implementation (for unit testing)
# ---------------------------------------------------------------------------


def _lse_weighted_combine_cpu(
    partial_outputs: torch.Tensor,
    partial_lses: torch.Tensor,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor:
    """CPU reference: combine N partial attention outputs using LSE weights.

    Args:
        partial_outputs: [N, B, H_local, D]
        partial_lses:    [N, B, H_local]
        is_lse_base_on_e: base-e (True) or base-2 (False)

    Returns:
        [B, H_local, D] combined output
    """
    N, B, H_local, D = partial_outputs.shape
    partial_outputs = partial_outputs.float()
    partial_lses = partial_lses.float()

    # Sanitize
    partial_lses = torch.where(
        torch.isnan(partial_lses) | torch.isinf(partial_lses),
        torch.full_like(partial_lses, float("-inf")),
        partial_lses,
    )

    # Max LSE for numerical stability: [B, H_local]
    lse_max, _ = partial_lses.max(dim=0)
    lse_max = torch.where(lse_max == float("-inf"), torch.zeros_like(lse_max), lse_max)

    # Compute weights: [N, B, H_local]
    centered = partial_lses - lse_max.unsqueeze(0)
    if is_lse_base_on_e:
        weights = torch.exp(centered)
    else:
        weights = torch.pow(2.0, centered)

    weight_sum = weights.sum(dim=0, keepdim=True)
    weights = weights / weight_sum

    # Weighted sum: [N, B, H_local, D] * [N, B, H_local, 1] -> sum -> [B, H_local, D]
    combined = (partial_outputs * weights.unsqueeze(-1)).sum(dim=0)
    return combined
