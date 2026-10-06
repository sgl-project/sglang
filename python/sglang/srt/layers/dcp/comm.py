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

import logging
from typing import Optional

import msgspec
import torch

from sglang.kernels.ops.attention.dcp_kernels import (
    CPTritonContext,
    _lse_pack_dim,
    correct_attn_out,
    dcp_lse_combine_triton,
    dcp_pack_a2a_send,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_hip, log_info_on_rank0
from sglang.srt.utils.custom_op import register_custom_op

logger = logging.getLogger(__name__)

_is_hip = is_hip()


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
    if cp_group.world_size == 1:
        return (cp_attn_out, cp_attn_lse) if return_lse else cp_attn_out

    cp_attn_lse = cp_attn_lse.contiguous()
    lses = _ag_lse(cp_attn_lse, cp_group)
    global_lse = torch.logsumexp(lses, dim=0)
    scale = torch.exp(cp_attn_lse - global_lse).unsqueeze(-1)
    scale = torch.nan_to_num(scale, nan=0.0, posinf=0.0, neginf=0.0)

    out = cp_attn_out.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    out.mul_(scale)
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
    is_lse_base_on_e: bool = False,
):
    """Merge DCP partial attention outputs with the LSE's actual log base.

    cp_attn_out: [ B, H, D ]
    cp_attn_lse: [ B, H ]

    FlashInfer MLA returns base-2 LSE, while FlashMLA returns natural-log LSE.
    The correction kernel must use the matching exp/log pair or it computes
    incorrect cross-rank softmax weights.
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
        cp_attn_out,
        lses,
        cp_group.rank_in_group,
        ctx,
        new_output,
        is_lse_base_on_e=is_lse_base_on_e,
    )
    out = cp_group.reduce_scatter_along_dim(out, dim=0)
    return out.to(cp_attn_out.dtype)


def cp_lse_ag_out_rs_mla_npu(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
) -> torch.Tensor:
    """Merge NPU DCP partial outputs and return the local head slice."""
    if cp_group.world_size == 1:
        return cp_attn_out

    import torch_npu

    batch_size, total_heads, head_dim = cp_attn_out.shape
    world_size = cp_group.world_size
    local_heads = total_heads // world_size
    packed = torch.cat([cp_attn_out.float(), cp_attn_lse.float().unsqueeze(-1)], dim=-1)
    packed = packed.permute(1, 2, 0).contiguous()
    gathered = torch.empty_like(packed)
    cp_group.all_to_all_single(gathered, packed)
    # all_to_all_single splits the leading head dimension. After the exchange,
    # the heads are grouped by source rank inside every token. Move that source
    # rank in front before flattening tokens and local heads for the update op.
    gathered = gathered.permute(2, 0, 1).contiguous()
    gathered = (
        gathered.view(batch_size, world_size, local_heads, head_dim + 1)
        .permute(1, 0, 2, 3)
        .contiguous()
        .view(world_size, batch_size * local_heads, head_dim + 1)
    )
    out_flat, lse_flat = torch.split(gathered, [head_dim, 1], dim=-1)
    merged, _ = torch_npu.npu_attention_update(
        lse_flat.squeeze(-1).unbind(0), out_flat.unbind(0), 0
    )
    return merged.view(batch_size, local_heads, head_dim).to(cp_attn_out.dtype)


def _all_gather_dcp_kv_cache(kv_a: torch.Tensor):
    parallel = get_parallel()
    dcp_world_size = parallel.dcp_size
    # not use symmetric_memory unless torch mem_pool updated, see https://github.com/pytorch/pytorch/issues/178138
    gathered_kv_a = kv_a.new_empty(
        (kv_a.shape[0] * dcp_world_size, *kv_a.shape[1:]),
    )
    # pynccl has no fp8 dtype; all-gather is a byte copy, so transport an fp8 KV
    # cache as raw bytes via a uint8 view (works with --kv-cache-dtype fp8_*).
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
    # torch.cat can't promote fp8 (gathered prefix) + bf16 (current extend), so
    # align dtypes first (dequant the fp8 prefix; exact for the scale=1.0 default).
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
    # On hip, skip the all-gather when there is no cached prefix to avoid crash
    if not _is_hip or dcp_extend_prefix_lens_sum > 0:
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
# A2A communication backend for DCP decode (alternative to AG+RS above): exchange
# per-head partial outputs + LSEs across DCP ranks and merge them. a2a sends one
# NCCL all-to-all and merges with the Triton LSE kernel; fi_a2a hands exchange and
# merge to FlashInfer's fused decode_cp_a2a_lse_reduce, one kernel that writes into
# its peers' torch symmetric memory.
# ---------------------------------------------------------------------------


class _FiA2aState(msgspec.Struct, frozen=True, kw_only=True):
    cp_rank: int
    # Rows (tokens x local heads) one fused call carries.
    capacity_rows: int
    # A workspace serves one ordered CUDA stream: graphs captured on capture_stream
    # use capture_workspace, also when replayed; eager calls use serving_workspace.
    capture_stream: int
    capture_workspace: torch.Tensor
    serving_workspace: torch.Tensor


# Set once per process, before CUDA graph capture, by init_fi_a2a_workspace().
_FI_A2A_STATE: Optional[_FiA2aState] = None


def init_fi_a2a_workspace(
    cp_group: GroupCoordinator,
    *,
    max_tokens: int,
    local_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    capture_stream: torch.cuda.Stream,
) -> None:
    """Create the fused reduce's workspaces on every DCP rank. Collective, and it
    synchronizes the current stream, so it runs before graph capture."""
    global _FI_A2A_STATE
    if _FI_A2A_STATE is not None or cp_group.world_size == 1:
        return
    import torch.distributed._symmetric_memory as symm_mem
    from flashinfer.comm import (
        decode_cp_a2a_lse_reduce_create_workspace,
        decode_cp_a2a_lse_reduce_workspace_size,
    )

    geometry = dict(
        max_tokens=max_tokens,
        local_heads=local_heads,
        cp_size=cp_group.world_size,
        head_dim=head_dim,
        dtype=dtype,
    )
    device = torch.device("cuda", torch.cuda.current_device())

    def create_workspace() -> torch.Tensor:
        return decode_cp_a2a_lse_reduce_create_workspace(
            **geometry, group=cp_group.device_group
        )

    try:
        capture_workspace = create_workspace()
        serving_workspace = create_workspace()
    except RuntimeError as e:
        raise RuntimeError(
            "--dcp-comm-backend fi_a2a: FlashInfer could not create its fused-reduce "
            "workspace on this process's torch symmetric-memory backend "
            f"({symm_mem.get_backend(device)}): {str(e).rstrip('.')}. Use a "
            "flashinfer-python release whose decode_cp_a2a_lse_reduce does not "
            "select a backend, or pass --dcp-comm-backend a2a."
        ) from e
    _FI_A2A_STATE = _FiA2aState(
        cp_rank=cp_group.rank_in_group,
        capacity_rows=max_tokens * local_heads,
        capture_stream=capture_stream.cuda_stream,
        capture_workspace=capture_workspace,
        serving_workspace=serving_workspace,
    )
    workspace_mib = decode_cp_a2a_lse_reduce_workspace_size(**geometry) / 2**20
    log_info_on_rank0(
        logger,
        "DCP fi_a2a: fused all-to-all + LSE reduce ready "
        f"(backend={symm_mem.get_backend(device)}, capacity "
        f"{max_tokens * local_heads} rows ({max_tokens} tokens x {local_heads} "
        f"heads), 2 workspaces x {workspace_mib:.2f} MiB)",
    )


def _fi_a2a_peer_views(
    cp_attn_out: torch.Tensor, cp_attn_lse: torch.Tensor, cp_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """View [B, H, D] partials and their [B, H] LSE as the fused reduce's
    [B, H/cp, cp, D] and [B, H/cp, cp]; head h goes to peer h // (H/cp)."""
    batch, heads, _ = cp_attn_out.shape
    local_heads = heads // cp_size
    itemsize = cp_attn_out.element_size()
    # The kernel reads every row in 8-byte words along a unit-stride last dim.
    if cp_attn_out.stride(-1) != 1 or any(
        n % 8
        for n in (
            cp_attn_out.data_ptr(),
            cp_attn_out.stride(0) * itemsize,
            cp_attn_out.stride(1) * itemsize,
        )
    ):
        cp_attn_out = cp_attn_out.clone(memory_format=torch.contiguous_format)
    partial_o = cp_attn_out.unflatten(1, (cp_size, local_heads)).transpose(1, 2)
    partial_lse = (
        cp_attn_lse.reshape(batch, heads)
        .unflatten(1, (cp_size, local_heads))
        .transpose(1, 2)
    )
    return partial_o, partial_lse


def _fi_a2a_lse_reduce_fake(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_size: int,
    is_lse_base_on_e: bool,
) -> torch.Tensor:
    batch, heads, head_dim = cp_attn_out.shape
    return cp_attn_out.new_empty(batch, heads // cp_size, head_dim)


# A custom op, so torch.compile sees one opaque collective rather than tracing the
# workspace choice.
@register_custom_op(fake_impl=_fi_a2a_lse_reduce_fake)
def fi_a2a_lse_reduce(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_size: int,
    is_lse_base_on_e: bool,
) -> torch.Tensor:
    from flashinfer.comm import decode_cp_a2a_lse_reduce

    state = _FI_A2A_STATE
    assert state is not None, (
        "BaseRunner.warmup() creates the fi_a2a workspaces before graph capture"
    )
    batch, heads, head_dim = cp_attn_out.shape
    local_heads = heads // cp_size
    # The kernel rejects an empty batch; every DCP rank sees the same batch.
    if batch == 0:
        return cp_attn_out.new_empty(0, local_heads, head_dim)
    partial_o, partial_lse = _fi_a2a_peer_views(
        cp_attn_out=cp_attn_out, cp_attn_lse=cp_attn_lse, cp_size=cp_size
    )
    if torch.cuda.current_stream().cuda_stream == state.capture_stream:
        workspace = state.capture_workspace
    else:
        workspace = state.serving_workspace
    # Eager batches wider than the widest captured graph run as several calls,
    # in the same order on every DCP rank.
    tokens_per_call = state.capacity_rows // local_heads
    outputs = [
        decode_cp_a2a_lse_reduce(
            partial_o=partial_o[start : start + tokens_per_call],
            partial_lse=partial_lse[start : start + tokens_per_call],
            workspace=workspace,
            cp_rank=state.cp_rank,
            cp_size=cp_size,
            lse_mode="basee" if is_lse_base_on_e else "base2",
        )
        for start in range(0, batch, tokens_per_call)
    ]
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs)


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
        return fi_a2a_lse_reduce(
            cp_attn_out=cp_attn_out,
            cp_attn_lse=cp_attn_lse,
            cp_size=cp_group.world_size,
            is_lse_base_on_e=is_lse_base_on_e,
        )

    N = cp_group.world_size
    B, H, D = cp_attn_out.shape
    assert H % N == 0, f"num_heads ({H}) must be divisible by dcp_size ({N})"
    H_per_rank = H // N
    out_dtype = cp_attn_out.dtype
    lpd = _lse_pack_dim(out_dtype)  # 2 for bf16/fp16

    if cuda_graph_buffers is not None:
        send_combined = cuda_graph_buffers["send_combined"]
        recv_combined = cuda_graph_buffers["recv_combined"]
    else:
        send_combined = torch.empty(
            N,
            B,
            H_per_rank,
            D + lpd,
            dtype=out_dtype,
            device=cp_attn_out.device,
        )
        recv_combined = torch.empty_like(send_combined)

    send_words = send_combined.view(torch.float32)
    dcp_pack_a2a_send(
        cp_attn_out,
        cp_attn_lse,
        send_combined[:, :, :, :D],
        send_words[:, :, :, D // lpd],
    )

    # Transport as raw bytes (uint8): the output may be fp8 (fp8 KV cache),
    # which pynccl's dtype enum can't send; byte a2a is exact for equal chunks.
    cp_group.all_to_all_single(
        recv_combined.reshape(-1).view(torch.uint8),
        send_combined.reshape(-1).view(torch.uint8),
    )

    recv_output = recv_combined[:, :B, :, :D]
    recv_lse = recv_combined.view(torch.float32)[:, :B, :, D // lpd]

    combined, _ = dcp_lse_combine_triton(
        recv_output, recv_lse, is_lse_base_on_e=is_lse_base_on_e
    )
    return combined
