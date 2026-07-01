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

from typing import Optional

import torch

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.distributed.parallel_state import (
    GroupCoordinator,
    get_dcp_group,
    get_dcp_group_no_assert,
    get_dcp_rank,
    get_dcp_world_size,
)
from sglang.srt.layers.cp.dcp.kernels import CPTritonContext, correct_attn_out
from sglang.srt.utils import is_cuda


def dcp_enabled() -> bool:
    """
    only checks whether dcp enabled for cuda platform
    """
    if get_dcp_group_no_assert() is None:
        return False
    if not is_cuda():
        return False
    return get_dcp_world_size() > 1


def get_attention_dcp_world_size() -> int:
    if not dcp_enabled():
        return 1
    return get_dcp_world_size()


def get_attention_dcp_rank() -> int:
    if not dcp_enabled():
        return 0
    return get_dcp_rank()


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
    dcp_world_size = get_dcp_world_size()
    # not use symmetric_memory unless torch mem_pool updated, see https://github.com/pytorch/pytorch/issues/178138
    gathered_kv_a = kv_a.new_empty(
        (kv_a.shape[0] * dcp_world_size, *kv_a.shape[1:]),
    )
    get_dcp_group().all_gather_into_tensor(gathered_kv_a, kv_a)
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
    if dcp_enabled():
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
        attn_mqa, dcp_local_prefix_kv_indices
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
    with use_symmetric_memory(get_dcp_group()):
        # transpose q_pe and q_nope_out from [B, H, L] to [H, B, L]
        combined = torch.cat([q_pe.transpose(0, 1), q_nope_out.transpose(0, 1)], dim=-1)
    gathered = get_dcp_group().all_gather(combined, dim=0)
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
    if not dcp_enabled():
        return torch.cat([prefix_kv_a, prefix_k_pe], dim=-1)
    # 1. compute max kv_lens for each seq
    dcp_world_size = get_dcp_world_size()
    dcp_rank = get_dcp_rank()

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
