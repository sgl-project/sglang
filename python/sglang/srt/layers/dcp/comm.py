# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Group accessors, LSE-merge and all-gather collectives for decode CP (DCP).

The two LSE-merge variants kept separate (bodies are backend-forced, see
PR #25090 vs #14194):
  - cp_lse_ag_out_rs_mha: torch / natural-log logsumexp / all-reduce + head slice
  - cp_lse_ag_out_rs_mla: Triton (log2/exp2) correction / reduce-scatter
"""

import warnings
from typing import Optional

import torch

from sglang.kernels.ops.attention.dcp_kernels import (
    CPTritonContext,
    _lse_pack_dim,
    correct_attn_out,
    dcp_lse_combine_triton,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.runtime_context import get_parallel


def _warn_deprecated_dcp_accessor(name: str, replacement: str) -> None:
    warnings.warn(
        f"{name} is deprecated; use {replacement} instead.",
        DeprecationWarning,
        stacklevel=2,
    )


def dcp_enabled() -> bool:
    """Deprecated: use ``get_parallel().dcp_enabled``."""
    _warn_deprecated_dcp_accessor("dcp_enabled()", "get_parallel().dcp_enabled")
    return get_parallel().dcp_enabled


def get_attention_dcp_world_size() -> int:
    """Deprecated: use ``get_parallel().attn_dcp_size``."""
    _warn_deprecated_dcp_accessor(
        "get_attention_dcp_world_size()", "get_parallel().attn_dcp_size"
    )
    return get_parallel().attn_dcp_size


def get_attention_dcp_rank() -> int:
    """Deprecated: use ``get_parallel().attn_dcp_rank``."""
    _warn_deprecated_dcp_accessor(
        "get_attention_dcp_rank()", "get_parallel().attn_dcp_rank"
    )
    return get_parallel().attn_dcp_rank


def _ag_lse(cp_attn_lse: torch.Tensor, cp_group: GroupCoordinator) -> torch.Tensor:
    """All-gather each rank's LSE into a ``[world_size, *lse.shape]`` stack.

    Shared prologue of both ``cp_lse_ag_out_rs_{mha,mla}``. Callers do their own
    pre-processing (``contiguous()`` for MHA, fp32 cast for MLA) before calling.
    """
    return cp_group.all_gather(cp_attn_lse, dim=0).view(
        (cp_group.world_size,) + cp_attn_lse.shape
    )


def cp_lse_ag_out_rs_mha(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    return_lse: bool = False,
):
    """Merge DCP partial attention outputs using natural-log LSE (PR #25090)."""
    if cp_group.world_size == 1:
        return (cp_attn_out, cp_attn_lse) if return_lse else cp_attn_out

    cp_attn_lse = cp_attn_lse.contiguous()
    lses = _ag_lse(cp_attn_lse, cp_group)
    global_lse = torch.logsumexp(lses, dim=0)
    scale = torch.exp(cp_attn_lse - global_lse).unsqueeze(-1)
    scale = torch.nan_to_num(scale, nan=0.0, posinf=0.0, neginf=0.0)

    out = torch.nan_to_num(cp_attn_out, nan=0.0, posinf=0.0, neginf=0.0) * scale
    out = cp_group.all_reduce(out)

    cp_num_heads = global_lse.shape[1] // cp_group.world_size
    cp_rank = cp_group.rank_in_group
    head_start = cp_num_heads * cp_rank
    head_end = cp_num_heads * (cp_rank + 1)
    out = out[:, head_start:head_end, :].contiguous()
    if return_lse:
        return out, global_lse[:, head_start:head_end].contiguous()
    return out


def cp_lse_ag_out_rs_mla(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    ctx: Optional[CPTritonContext] = None,
):
    """Merge DCP partial attention outputs via Triton correction (PR #14194).

    cp_attn_out: [ B, H, D ]
    cp_attn_lse: [ B, H ]
    """
    if cp_group.world_size == 1:
        return cp_attn_out

    if ctx is None:
        ctx = CPTritonContext()

    with use_symmetric_memory(cp_group):
        # cp_attn_out is [B,H,D], we want to transpose it to [H,B,D] for the kernel, and then transpose back after correction.
        new_output = cp_attn_out.new_empty(
            cp_attn_out.transpose(0, 1).shape, dtype=torch.float32
        )
        cp_attn_lse = cp_attn_lse.to(torch.float32)
    lses = _ag_lse(cp_attn_lse, cp_group)
    out, _ = correct_attn_out(
        cp_attn_out, lses, cp_group.rank_in_group, ctx, new_output
    )
    out = cp_group.reduce_scatter_along_dim(out, dim=0)
    return out.to(cp_attn_out.dtype)


def _all_gather_dcp_kv_cache(kv_a: torch.Tensor):
    parallel = get_parallel()
    dcp_world_size = parallel.dcp_size
    # not use symmetric_memory unless torch mem_pool updated, see https://github.com/pytorch/pytorch/issues/178138
    gathered_kv_a = kv_a.new_empty(
        (kv_a.shape[0] * dcp_world_size, *kv_a.shape[1:]),
    )
    # pynccl's ncclDataTypeEnum has no fp8 entry, but all-gather is a pure byte
    # copy — transport an fp8 KV cache (fp8_e4m3 / fp8_e5m2) as raw bytes via a
    # uint8 view (shared storage) so DCP works with --kv-cache-dtype fp8_*.
    if kv_a.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        parallel.dcp_group.all_gather_into_tensor(
            gathered_kv_a.view(torch.uint8), kv_a.contiguous().view(torch.uint8)
        )
    else:
        parallel.dcp_group.all_gather_into_tensor(gathered_kv_a, kv_a)
    gathered_kv_a = (
        gathered_kv_a.reshape((dcp_world_size,) + kv_a.shape)
        .transpose(0, 1)
        .reshape(-1, *kv_a.shape[1:])
    )
    return gathered_kv_a


def all_gather_kv_cache_for_mha_chunk_extend(
    kv_a: torch.Tensor,
    k_pe: torch.Tensor,
    prefix_kv_lens_cpu: torch.Tensor,
    prefix_starts_cpu: torch.Tensor = None,
):
    if get_parallel().dcp_enabled:
        kv_a = kv_a.unsqueeze(1)
        gathered_kv = all_gather_kv_cache_for_dcp(
            kv_a,
            k_pe,
            prefix_kv_lens_cpu,
            prefix_starts_cpu,
        )
        kv_a, k_pe = gathered_kv.split([kv_a.shape[-1], k_pe.shape[-1]], dim=-1)
        kv_a = kv_a.squeeze(1)
    return kv_a.contiguous(), k_pe.contiguous()


def all_gather_kv_cache_for_mha_extend(
    token_to_kv_pool,
    attn_mqa,
    dcp_local_prefix_kv_indices,
    seq_lens,
    extend_prefix_lens,
    extend_prefix_lens_cpu: list[int],
    extend_seq_lens,
    kv_a: torch.Tensor,
    k_pe: torch.Tensor,
):
    prefix_kv_a, prefix_k_pe = token_to_kv_pool.get_mla_kv_buffer(
        attn_mqa, dcp_local_prefix_kv_indices, dst_dtype=kv_a.dtype
    )
    extend_prefix_lens_cpu = torch.tensor(extend_prefix_lens_cpu)
    gathered_kv_cache = all_gather_kv_cache_for_dcp(
        prefix_kv_a,
        prefix_k_pe,
        extend_prefix_lens_cpu,
    )
    prefix_kv_a, prefix_k_pe = gathered_kv_cache.split(
        [kv_a.shape[-1], k_pe.shape[-1]], dim=-1
    )
    prefix_kv_a = prefix_kv_a.squeeze(1)
    # fp8 KV cache: the gathered prefix is fp8 (from the pool) while the current
    # extend kv_a/k_pe are bf16 — torch.cat below cannot promote fp8+bf16, so
    # align dtypes (dequantize the fp8 prefix; exact for the scale=1.0 default).
    if prefix_kv_a.dtype != kv_a.dtype:
        prefix_kv_a = prefix_kv_a.to(kv_a.dtype)
    if prefix_k_pe.dtype != k_pe.dtype:
        prefix_k_pe = prefix_k_pe.to(k_pe.dtype)
    # re-organize kv with query orders
    prefix_lens_cu = torch.zeros(
        len(seq_lens) + 1,
        dtype=torch.int32,
        device=kv_a.device,
    )
    extend_lens_cu = torch.zeros_like(prefix_lens_cu)
    prefix_lens_cu[1:] = torch.cumsum(extend_prefix_lens, dim=0)
    extend_lens_cu[1:] = torch.cumsum(extend_seq_lens, dim=0)
    kv_a_tuple = ()
    k_pe_tuple = ()
    for i in range(len(seq_lens)):
        kv_a_tuple += (
            prefix_kv_a[prefix_lens_cu[i] : prefix_lens_cu[i + 1]],
            kv_a[extend_lens_cu[i] : extend_lens_cu[i + 1]],
        )
        k_pe_tuple += (
            prefix_k_pe[prefix_lens_cu[i] : prefix_lens_cu[i + 1]],
            k_pe[extend_lens_cu[i] : extend_lens_cu[i + 1]],
        )
    kv_a = torch.cat(kv_a_tuple, dim=0)
    k_pe = torch.cat(k_pe_tuple, dim=0)
    return kv_a.contiguous(), k_pe.contiguous()


def all_gather_q_for_mla_decode(
    q_nope_out: torch.Tensor,
    q_pe: torch.Tensor,
):
    group = get_parallel().dcp_group
    with use_symmetric_memory(group):
        # transpose q_pe and q_nope_out from [B, H, L] to [H, B, L]
        combined = torch.cat([q_pe.transpose(0, 1), q_nope_out.transpose(0, 1)], dim=-1)
    gathered = group.all_gather(combined, dim=0)
    d_pe = q_pe.size(-1)
    d_nope = q_nope_out.size(-1)
    q_pe, q_nope_out = gathered.split([d_pe, d_nope], dim=-1)
    q_pe = q_pe.transpose(0, 1)
    q_nope_out = q_nope_out.transpose(0, 1)
    return q_nope_out, q_pe


def all_gather_kv_cache_for_mla_extend(
    token_to_kv_pool,
    attn_mqa,
    extend_prefix_lens_cpu: list[int],
    dcp_local_prefix_kv_indices,
    dcp_extend_prefix_lens_sum,
    dcp_kv_buffer,
    kv_lora_rank,
    k_nope,
    k_pe,
):
    cache_k_nope, cache_k_rope = token_to_kv_pool.get_mla_kv_buffer(
        attn_mqa,
        dcp_local_prefix_kv_indices,
    )
    extend_prefix_lens_cpu = torch.tensor(extend_prefix_lens_cpu)
    # all gather kv cache into forward_batch.attn_dcp_metadata.dcp_kv_buffer
    gathered_kv = all_gather_kv_cache_for_dcp(
        cache_k_nope,
        cache_k_rope,
        extend_prefix_lens_cpu,
        prefix_starts_cpu=torch.zeros_like(extend_prefix_lens_cpu),
    )
    dcp_kv_buffer[:dcp_extend_prefix_lens_sum] = gathered_kv

    # copy local kv cache into forward_batch.attn_dcp_metadata.dcp_kv_buffer
    dcp_kv_buffer[
        dcp_extend_prefix_lens_sum:,
        ...,
        :kv_lora_rank,
    ] = k_nope
    dcp_kv_buffer[
        dcp_extend_prefix_lens_sum:,
        ...,
        kv_lora_rank:,
    ] = k_pe


# all gather kv cache and re-org to query orders
def all_gather_kv_cache_for_dcp(
    prefix_kv_a: torch.Tensor,
    prefix_k_pe: torch.Tensor,
    prefix_kv_lens_cpu: torch.Tensor,
    prefix_starts_cpu: torch.Tensor = None,
):
    """
    prefix_kv_a and prefix_k_pe should have same shape, expect for last dim
    """
    parallel = get_parallel()
    if not parallel.dcp_enabled:
        return torch.cat([prefix_kv_a, prefix_k_pe], dim=-1)
    # 1. compute max kv_lens for each seq
    dcp_world_size = parallel.dcp_size
    dcp_rank = parallel.dcp_rank

    if prefix_starts_cpu is None:
        prefix_starts_cpu = torch.zeros_like(prefix_kv_lens_cpu)

    left_pads = prefix_starts_cpu % dcp_world_size > dcp_rank
    left_pads = left_pads.to(torch.int32)
    right_pads = (
        prefix_starts_cpu + prefix_kv_lens_cpu - 1
    ) % dcp_world_size < dcp_rank
    right_pads = right_pads.to(torch.int32)
    padded_lens = (
        prefix_kv_lens_cpu + (prefix_starts_cpu % dcp_world_size) + dcp_world_size - 1
    ) // dcp_world_size

    local_kv_lens = padded_lens - left_pads - right_pads
    local_kv_lens_cu = torch.zeros(
        len(prefix_kv_lens_cpu) + 1,
        dtype=torch.int32,
    )
    local_kv_lens_cu[1:] = torch.cumsum(local_kv_lens, dim=0)

    padded_kv_cache_arr = []
    prefix_kv_cache = torch.cat([prefix_kv_a, prefix_k_pe], dim=-1)
    for req_idx in range(len(prefix_kv_lens_cpu)):
        padded_tensor = prefix_kv_cache.new_empty(
            (padded_lens[req_idx].item(),) + prefix_kv_cache.size()[1:]
        )
        padded_tensor[
            left_pads[req_idx] : left_pads[req_idx] + local_kv_lens[req_idx]
        ] = prefix_kv_cache[local_kv_lens_cu[req_idx] : local_kv_lens_cu[req_idx + 1]]
        padded_kv_cache_arr.append(padded_tensor)

    padded_kv_cache = torch.cat(padded_kv_cache_arr, dim=0)

    gatherd_kv_cache = _all_gather_dcp_kv_cache(padded_kv_cache)

    # 2. re-org kv cache to query orders
    padded_lens_cu = torch.zeros(
        len(prefix_kv_lens_cpu) + 1,
        dtype=torch.int32,
    )
    padded_lens_cu[1:] = torch.cumsum(padded_lens, dim=0)
    kv_cache_tuple = ()
    for req_idx in range(len(prefix_kv_lens_cpu)):
        kv_cache_tuple += (
            gatherd_kv_cache[
                padded_lens_cu[req_idx] * dcp_world_size
                + (prefix_starts_cpu[req_idx] % dcp_world_size) :
            ][: prefix_kv_lens_cpu[req_idx]],
        )
    gatherd_kv_cache = torch.cat(kv_cache_tuple, dim=0)

    return gatherd_kv_cache


# ---------------------------------------------------------------------------
# A2A communication backend for DCP decode (alternative to AG+RS above).
# After local attention produces per-head partial outputs + LSEs over a local
# KV shard, A2A exchanges head partials across DCP ranks then combines them
# locally with the Triton LSE kernel (kernels.dcp_lse_combine_triton). fi_a2a
# delegates the exchange to FlashInfer's MNNVL kernel (flashinfer #2951).
# ---------------------------------------------------------------------------

# Per-process singleton: MNNVL workspace + this rank's cp position. Populated
# once, pre-CUDA-graph-capture, by init_fi_a2a_workspace().
_FI_A2A_STATE: Optional[dict] = None


def init_fi_a2a_workspace(cp_group: "GroupCoordinator") -> None:
    # Call once per process BEFORE CUDA-graph capture: the FlashInfer init syncs
    # the stream and barriers cross-rank, neither of which is capturable.
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

    # Reuse the MoE adapter: its Split() returns a CommBackend (what FlashInfer's
    # Mapping expects); the flashinfer_comm_fusion copy has drifted to return a
    # raw ProcessGroup, so don't swap without re-checking the Split() contract.
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
    # REQUIRED barrier before the first alltoall: every rank must finish init,
    # else a rank writes a peer's FIFO before it is ready -> deadlock.
    dist.barrier(group=cp_group.device_group)
    _FI_A2A_STATE = {
        "workspace": workspace,
        "cp_rank": cp_rank,
    }


def dcp_a2a_lse_reduce(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: "GroupCoordinator",
    is_lse_base_on_e: bool = True,
    cuda_graph_buffers: Optional[dict] = None,
    comm_backend: str = "a2a",
) -> torch.Tensor:
    """A2A DCP reduce: all-to-all exchange of head partials, then local Triton
    combine. Output + fp32 LSE are packed into ONE all_to_all (LSE reinterpreted
    as output-dtype columns along D) -> 1 NCCL call/layer instead of 2.
    is_lse_base_on_e: True=base-e (FlashAttention), False=base-2 (FlashInfer-MLA).
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

        send_combined[:, :B, :, :D].copy_(reshaped_out)
        send_lse_stg[:, :B, :].copy_(reshaped_lse)
        send_combined[:, :, :, D:].copy_(
            send_lse_stg.view(out_dtype).view(N, -1, H_per_rank, lpd)
        )

        cp_group.all_to_all_single(
            recv_combined.reshape(-1).view(torch.uint8),
            send_combined.reshape(-1).view(torch.uint8),
        )
        recv_output = recv_combined[:, :B, :, :D]
        recv_lse_stg.view(out_dtype).view(N, -1, H_per_rank, lpd).copy_(
            recv_combined[:, :, :, D:]
        )
        recv_lse = recv_lse_stg[:, :B, :]
    else:
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

        # Transport as raw bytes (uint8): the output may be fp8 (fp8 KV cache),
        # which pynccl's dtype enum can't send; byte a2a is exact for equal chunks.
        cp_group.all_to_all_single(
            recv_combined.reshape(-1).view(torch.uint8),
            send_combined.reshape(-1).view(torch.uint8),
        )

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
    """fi_a2a: delegate only the cross-rank exchange to FlashInfer's MNNVL kernel,
    then reuse the local Triton LSE combine. FlashInfer takes output + LSE as
    separate tensors: partial_o [B, H_per_rank, cp_size, D] (peer axis 2nd-to-last),
    softmax_stats [B, H_per_rank, cp_size, 2] fp32 (S padded 1->2).
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
    partial_o = cp_attn_out.view(B, N, H_per_rank, D).permute(0, 2, 1, 3).contiguous()
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
