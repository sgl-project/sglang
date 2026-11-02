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

import statistics
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch

from sglang.kernels.ops.attention.dcp_kernels import (
    CPTritonContext,
    _lse_pack_dim,
    correct_attn_out,
    dcp_lse_combine_triton,
    dcp_pack_a2a_send,
    dcp_pack_q,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_hip
from sglang.srt.utils.common import is_fi_a2a_supported

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
    prefix_kv_a, prefix_k_pe = token_to_kv_pool.layer_router.read(
        attn_mqa.layer_id
    ).read_mla(attn_mqa, dcp_local_prefix_kv_indices, dst_dtype=kv_a.dtype)
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
    dtype: Optional[torch.dtype] = None,
):
    """All-gather the DCP group's query heads; ``dtype`` casts q before the wire."""
    group = get_parallel().dcp_group
    if dtype is None or dtype == q_pe.dtype:
        with use_symmetric_memory(group):
            # transpose q_pe and q_nope_out from [B, H, L] to [H, B, L]
            combined = torch.cat(
                [q_pe.transpose(0, 1), q_nope_out.transpose(0, 1)], dim=-1
            )
        gathered = group.all_gather(combined, dim=0)
    else:
        B, H, d_nope = q_nope_out.shape
        with use_symmetric_memory(group):
            combined = torch.empty(
                (H, B, q_pe.shape[-1] + d_nope), dtype=dtype, device=q_pe.device
            )
        dcp_pack_q(q_nope_out, q_pe, combined)
        # The aiter custom all-gather only takes 16/32-bit floats and pynccl has
        # no FP8 dtype: pair the bytes into fp16 when the row allows, an exact copy.
        wire = torch.float16 if combined.shape[-1] % 2 == 0 else torch.uint8
        gathered = group.all_gather(combined.view(wire), dim=0).view(dtype)
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
        cache_k_nope, cache_k_rope = token_to_kv_pool.layer_router.read(
            attn_mqa.layer_id
        ).read_mla(
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

    # Attention TP may append dummy tokens to align an extend batch to its
    # group width. The DCP buffer tracks only real sequence tokens.
    extend_num_tokens = dcp_kv_buffer.shape[0] - dcp_extend_prefix_lens_sum
    dcp_kv_buffer[
        dcp_extend_prefix_lens_sum:,
        ...,
        :kv_lora_rank,
    ] = k_nope[:extend_num_tokens]
    dcp_kv_buffer[
        dcp_extend_prefix_lens_sum:,
        ...,
        kv_lora_rank:,
    ] = k_pe[:extend_num_tokens]


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
# per-head partial outputs + LSEs across DCP ranks, then combine locally with the
# Triton LSE kernel. fi_a2a delegates the exchange to FlashInfer MNNVL (#2951).
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
        from flashinfer.comm.mnnvl import MnnvlConfig
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

    cp_size = cp_group.world_size
    cp_rank = cp_group.rank_in_group
    parallel = get_parallel()

    if not is_fi_a2a_supported(
        dcp_size=cp_size,
        tp_size=parallel.tp_size,
        pp_size=parallel.pp_size,
        nnodes=parallel.nnodes,
    ):
        raise RuntimeError(
            "--dcp-comm-backend fi_a2a needs a Blackwell system whose DCP group "
            "shares one MNNVL domain: either MNNVL fabric memory (GB200/GB300) "
            f"or a DCP group inside one node (got dcp_size={cp_size}, "
            f"tp_size={parallel.tp_size}, pp_size={parallel.pp_size}, "
            f"nnodes={parallel.nnodes}). Use --dcp-comm-backend a2a or ag_rs "
            "otherwise."
        )

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


_side_stream: Optional[torch.cuda.Stream] = None


def dcp_side_stream() -> torch.cuda.Stream:
    """The one stream DCP collectives overlap compute on. A single stream: ROCm
    multiplexes streams over a few hardware queues, and an extra stream can land
    on the compute stream's queue and serialize behind it."""
    global _side_stream
    if _side_stream is None:
        _side_stream = torch.cuda.Stream()
    return _side_stream


class DcpA2aChunkPipeline:
    """``dcp_a2a_lse_reduce`` over row chunks of one batch.

    ``run`` attends one row chunk at a time and V-projects and packs it; its
    exchange runs on a side stream while the next chunk attends. The combine of
    every chunk then writes straight into ``[rows, H // N, D]``. Every DCP rank
    must run the same chunks in the same order.
    """

    def __init__(
        self,
        chunks: List[Tuple[int, int]],
        cp_group: GroupCoordinator,
        project: Callable[[torch.Tensor], torch.Tensor],
        is_lse_base_on_e: bool,
        timed: bool = False,
    ):
        self.chunks = chunks
        self.rows = chunks[-1][1]
        self.cp_group = cp_group
        self.project = project
        self.is_lse_base_on_e = is_lse_base_on_e
        self.consumed = False
        self.timing = _ChunkTiming() if timed else None
        self._pending = []

    def run(
        self, attend: Callable[[int, int], Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        for lo, hi in self.chunks:
            if self.timing is not None:
                self.timing.start_chunk()
            self.submit(lo, hi, *attend(lo, hi))
            if self.timing is not None:
                # Timed chunks run one after another, so each stage is timed
                # without the other stream contending for the GPU.
                torch.cuda.current_stream().wait_event(self._pending[-1][4])
        if self.timing is not None:
            self.timing.host.append(time.perf_counter())
        return self.finish()

    def submit(
        self, lo: int, hi: int, attn_out: torch.Tensor, attn_lse: torch.Tensor
    ) -> None:
        projected = self.project(attn_out)
        B, H, D = projected.shape
        N = self.cp_group.world_size
        lpd = _lse_pack_dim(projected.dtype)
        send = torch.empty(
            (N, B, H // N, D + lpd), dtype=projected.dtype, device=projected.device
        )
        recv = torch.empty_like(send)
        dcp_pack_a2a_send(
            projected,
            attn_lse,
            send[:, :, :, :D],
            send.view(torch.float32)[:, :, :, D // lpd],
        )
        timed = self.timing is not None
        if timed:
            self.timing.end_compute()
        stream = dcp_side_stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            start = _timing_event() if timed else None
            self.cp_group.all_to_all_single(
                recv.view(-1).view(torch.uint8), send.view(-1).view(torch.uint8)
            )
            done = torch.cuda.Event(enable_timing=timed)
            done.record()
        if timed:
            self.timing.a2a.append((start, done))
        # send/recv stay referenced until finish() has ordered this stream
        # after the exchange, so the allocator cannot hand them out early.
        self._pending.append((lo, hi, send, recv, done))

    def finish(self) -> torch.Tensor:
        self.consumed = True
        recv0 = self._pending[0][3]
        _, _, H_local, row = recv0.shape
        lpd = _lse_pack_dim(recv0.dtype)
        D = row - lpd
        out = torch.empty(
            (self.rows, H_local, D), dtype=recv0.dtype, device=recv0.device
        )
        stream = torch.cuda.current_stream()
        timed = self.timing is not None
        for lo, hi, _send, recv, done in self._pending:
            stream.wait_event(done)
            start = _timing_event() if timed else None
            dcp_lse_combine_triton(
                recv[:, :, :, :D],
                recv.view(torch.float32)[:, :, :, D // lpd],
                is_lse_base_on_e=self.is_lse_base_on_e,
                out=out[lo:hi],
            )
            if timed:
                self.timing.combine.append((start, _timing_event()))
        self._pending.clear()
        return out


def _timing_event() -> torch.cuda.Event:
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


class _ChunkTiming:
    """Per-chunk GPU time of each pipeline stage and host time to issue a chunk."""

    def __init__(self):
        self.host: List[float] = []
        self.compute: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self.a2a: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self.combine: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def start_chunk(self) -> None:
        self.host.append(time.perf_counter())
        self._compute_start = _timing_event()

    def end_compute(self) -> None:
        self.compute.append((self._compute_start, _timing_event()))

    def stage_us(self) -> List[List[float]]:
        """GPU time (us) of each chunk on the compute, a2a and combine stage."""
        self.combine[-1][1].synchronize()
        return [
            [start.elapsed_time(end) * 1e3 for start, end in stage]
            for stage in (self.compute, self.a2a, self.combine)
        ]

    def host_us(self) -> List[float]:
        return [(b - a) * 1e6 for a, b in zip(self.host, self.host[1:])]


def _median_costs(
    timings: List["_ChunkTiming"], chunks: List[Tuple[int, int]]
) -> List[float]:
    """``plan_pipeline_chunks`` costs from the median over layers of each chunk's
    stage times; the first and last chunk must differ in size."""
    rows = [hi - lo for lo, hi in chunks]
    stages = [t.stage_us() for t in timings]
    costs = []
    for s in range(3):
        us = [statistics.median(st[s][i] for st in stages) for i in range(len(rows))]
        costs += _fit_line(rows, us)
    return costs + [statistics.median(h for t in timings for h in t.host_us())]


def _fit_line(rows: List[int], us: List[float]) -> List[float]:
    """(fixed, per-row) cost through the first and last chunk, clamped at 0."""
    per_row = max(0.0, (us[-1] - us[0]) / (rows[-1] - rows[0]))
    return [max(0.0, us[0] - per_row * rows[0]), per_row]


def _even_chunks(rows: int, n: int) -> List[Tuple[int, int]]:
    per = -(-rows // n)
    return [(lo, min(rows, lo + per)) for lo in range(0, rows, per)]


def modeled_pipeline_us(costs: List[float], chunks: List[Tuple[int, int]]) -> float:
    """Finish time of the pipeline schedule under ``costs``.

    The compute stream attends, projects and packs chunk i once the host has
    issued it; the side stream exchanges it after both the chunk and the
    previous exchange; the compute stream then combines each chunk after its
    exchange. One chunk is the unpipelined reduce.
    """
    c0, c1, x0, x1, m0, m1, host = costs
    compute = side = 0.0
    exchanged = []
    for i, (lo, hi) in enumerate(chunks):
        compute = max(compute, (i + 1) * host) + c0 + c1 * (hi - lo)
        side = max(side, compute) + x0 + x1 * (hi - lo)
        exchanged.append(side)
    for done, (lo, hi) in zip(exchanged, chunks):
        compute = max(compute, done) + m0 + m1 * (hi - lo)
    return compute


def plan_pipeline_chunks(costs: List[float], rows: int) -> List[Tuple[int, int]]:
    """The even row chunking whose modeled schedule finishes first."""
    best = _even_chunks(rows, 1)
    best_us = modeled_pipeline_us(costs, best)
    host = costs[-1]
    for n in range(2, rows + 1):
        if n * host >= best_us:
            # Issuing n chunks alone takes longer than the best schedule.
            break
        chunks = _even_chunks(rows, n)
        us = modeled_pipeline_us(costs, chunks)
        if us >= best_us:
            break
        best, best_us = chunks, us
    return best


class DcpA2aChunkPlanner:
    """Row chunks for ``DcpA2aChunkPipeline`` from the measured cost of its stages.

    Each stage of a chunk costs a fixed part plus a per-row part: attention,
    projection and pack on the compute stream, the a2a on the side stream, the
    combine back on the compute stream, and the host time to issue the chunk.
    The planner times every layer of one prefill run as two unequal chunks and
    takes the median over the layers after the first, which takes the
    first-use costs. It then picks the chunk count whose modeled schedule
    finishes first; one chunk means no pipeline. A prefill larger than the
    measured one is measured again, so the per-row costs come from where they
    dominate.

    Every DCP rank must plan the same chunks, so the ranks agree on each cost:
    the slowest rank's compute, combine and host time, and the fastest rank's
    a2a, which on the others also holds the wait for the last rank to arrive.
    """

    def __init__(self, cp_group: GroupCoordinator):
        self.cp_group = cp_group
        self.costs: Optional[List[float]] = None
        self.measured_rows = 0
        self._measuring: List[Tuple[int, DcpA2aChunkPipeline]] = []
        self._num_chunks: Dict[int, int] = {}

    def settle(self, layer_id: int) -> None:
        """Take the costs measured from ``layer_id`` on in an earlier forward."""
        if not self._measuring or self._measuring[0][0] != layer_id:
            return
        rows = self._measuring[0][1].rows
        measured = [p for _, p in self._measuring[1:] if p.consumed]
        self._measuring = []
        # Not measured again at these rows, whether or not it took.
        self.measured_rows = rows
        if not measured:
            return
        costs = torch.tensor(
            _median_costs([p.timing for p in measured], measured[0].chunks)
        )
        a2a = slice(2, 4)
        if self.cp_group.world_size > 1:
            costs[a2a] = -costs[a2a]
            torch.distributed.all_reduce(
                costs, op=torch.distributed.ReduceOp.MAX, group=self.cp_group.cpu_group
            )
            costs[a2a] = -costs[a2a]
        self.costs = costs.tolist()
        self._num_chunks.clear()

    def pipeline(
        self,
        layer_id: int,
        rows: int,
        project: Callable[[torch.Tensor], torch.Tensor],
        is_lse_base_on_e: bool,
    ) -> Optional[DcpA2aChunkPipeline]:
        if self._measuring:
            measuring = self._measuring[0][1].rows == rows
        else:
            # Two chunks of at least one row each and unequal size.
            measuring = rows > self.measured_rows and rows >= 3
        if measuring:
            # Two chunks, one twice the other: the fixed and per-row costs
            # separate, and neither chunk includes the attention set-up.
            pipeline = DcpA2aChunkPipeline(
                [(0, rows // 3), (rows // 3, rows)],
                self.cp_group,
                project,
                is_lse_base_on_e,
                timed=True,
            )
            self._measuring.append((layer_id, pipeline))
            return pipeline
        if self.costs is None or rows < 2:
            return None
        n = self._num_chunks.get(rows)
        if n is None:
            n = len(plan_pipeline_chunks(self.costs, rows))
            self._num_chunks[rows] = n
        if n == 1:
            return None
        return DcpA2aChunkPipeline(
            _even_chunks(rows, n), self.cp_group, project, is_lse_base_on_e
        )


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

    # Note(kpham-sgl): empty(), not zeros() -- the pack below fills partial_o and
    # stats slot 0, and slot 1 is never read by anyone. The a2a moves the stats
    # field as opaque bytes and we only ever read slot 0 back off the wire.
    partial_o = torch.empty(
        B, H_per_rank, N, D, dtype=cp_attn_out.dtype, device=cp_attn_out.device
    )
    softmax_stats = torch.empty(
        B, H_per_rank, N, 2, dtype=torch.float32, device=cp_attn_out.device
    )
    dcp_pack_a2a_send(
        cp_attn_out,
        cp_attn_lse,
        partial_o.permute(2, 0, 1, 3),
        softmax_stats[..., 0].permute(2, 0, 1),
    )

    o_out, stats_out = decode_cp_a2a_alltoall(
        partial_o,
        softmax_stats,
        state["workspace"],
        state["cp_rank"],
        N,
    )

    recv_output = o_out.permute(2, 0, 1, 3)
    recv_lse = stats_out[..., 0].permute(2, 0, 1)

    combined, _ = dcp_lse_combine_triton(
        recv_output, recv_lse, is_lse_base_on_e=is_lse_base_on_e
    )
    return combined
