# SPDX-License-Identifier: Apache-2.0

import logging
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch._C._distributed_c10d import _SymmetricMemory
import dataclasses
from typing import List, Optional, Tuple
import triton
import triton.language as tl
from triton.language import core

logger = logging.getLogger(__name__)


@core.extern
def __syncthreads(_semantic=None):
    return tl.tensor(_semantic.builder.create_barrier(), tl.void)


@triton.jit
def tid(axis: tl.constexpr = 0):
    """PTX threadIdx.x/y/z."""
    if axis == 0:
        return tl.inline_asm_elementwise(
            asm="mov.u32 $0, %tid.x;",
            constraints=("=r"),
            args=[],
            dtype=tl.int32,
            is_pure=True,
            pack=1
        )
    elif axis == 1:
        return tl.inline_asm_elementwise(
            asm="mov.u32 $0, %tid.y;",
            constraints=("=r"),
            args=[],
            dtype=tl.int32,
            is_pure=True,
            pack=1
        )
    else:
        return tl.inline_asm_elementwise(
            asm="mov.u32 $0, %tid.z;",
            constraints=("=r"),
            args=[],
            dtype=tl.int32,
            is_pure=True,
            pack=1
        )


@triton.jit
def ld_sys(ptr):
    """ld.global.acquire.sys.b32 — system-scope acquire load."""
    return tl.inline_asm_elementwise(
        asm="ld.global.acquire.sys.b32 $0, [$1];",
        constraints=("=r,l"),
        args=[ptr],
        dtype=tl.int32,
        is_pure=False,
        pack=1
    )


@triton.jit
def st_sys(ptr, val):
    """st.global.release.sys.b32 — system-scope release store."""
    tl.inline_asm_elementwise(
        asm="""
        st.global.release.sys.b32 [$1], $2;
        mov.u32 $0, 0;
        """,
        constraints=("=r,l,r"),
        args=[ptr, val],
        dtype=tl.int32,
        is_pure=False,
        pack=1
    )


@dataclasses.dataclass
class AllGatherGemmContextSymmMem:
    """Context for AG->GEMM overlap. Holds symm_mem buffers and handles."""

    rank: int
    num_ranks: int
    NUM_COMM_SMS: int
    NUM_GEMM_SMS: int
    ag_stream: torch.cuda.Stream

    symm_input_buf: torch.Tensor           # [world_size, M, K]
    symm_ag_a_buf: torch.Tensor            # [M * world_size, K]
    ag_signal_buf: torch.Tensor            # [world_size] uint32

    # symm_mem rendezvous handles
    input_hdl: object = None
    ag_hdl: object = None
    signal_hdl: object = None

    mc_ag_a_buf: Optional[torch.Tensor] = None  # deprecated (NVLS)
    peer_signal_ptrs: Optional[torch.Tensor] = None
    peer_symm_input_bufs: Optional[List[torch.Tensor]] = None
    group: Optional[object] = None

    def finalize(self):
        """Release symm_mem resources with a barrier for lock-step teardown."""
        self.symm_input_buf = None
        self.symm_ag_a_buf = None
        self.ag_signal_buf = None
        self.input_hdl = None
        self.ag_hdl = None
        self.signal_hdl = None
        self.mc_ag_a_buf = None
        self.peer_signal_ptrs = None
        self.peer_symm_input_bufs = None
        if dist.is_initialized():
            dist.barrier(group=self.group)

    def get_input_buf(self, M, K):
        """Return this rank's [M, K] shard view in symm_input_buf."""
        return self.symm_input_buf[self.rank, :M, :]


def create_allgather_gemm_context_symm_mem(
    ag_stream: torch.cuda.Stream,
    rank: int,
    world_size: int,
    max_M: int,
    K: int,
    NUM_COMM_SMS: int = 0,
    enable_multicast: bool = False,
    group: Optional[object] = None,
):
    """Create AG->GEMM context with symm_mem buffers.

    group: process group for symm_mem rendezvous (defaults to WORLD;
        set to TP group when pipeline parallelism is used).
    """
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before creating the context.")

    device = torch.cuda.current_device()
    rendezvous_group = group if group is not None else dist.group.WORLD

    symm_input_buf = symm_mem.empty(
        (world_size, max_M, K), dtype=torch.bfloat16, device=device,
    )
    symm_ag_a_buf = symm_mem.empty(
        (max_M * world_size, K), dtype=torch.bfloat16, device=device,
    )
    ag_signal_buf = symm_mem.empty(
        (world_size,), dtype=torch.uint32, device=device,
    )

    symm_input_buf.zero_()
    symm_ag_a_buf.zero_()
    ag_signal_buf.zero_()

    input_hdl = symm_mem.rendezvous(symm_input_buf, group=rendezvous_group)
    ag_hdl = symm_mem.rendezvous(symm_ag_a_buf, group=rendezvous_group)
    signal_hdl = symm_mem.rendezvous(ag_signal_buf, group=rendezvous_group)

    input_hdl.barrier()

    mc_ag_a_buf = None
    if enable_multicast:
        logger.warning(
            "enable_multicast is deprecated in the symm_mem migration; "
            "mc_ag_a_buf will be None. Reactivate via ag_hdl.multicast_ptr if needed."
        )

    peer_signal_ptrs = torch.tensor(
        [signal_hdl.get_buffer(r, (world_size,), torch.uint32).data_ptr()
         for r in range(world_size)],
        dtype=torch.int64,
        device=device,
    )

    peer_symm_input_bufs = [
        input_hdl.get_buffer(r, (world_size, max_M, K), torch.bfloat16)
        for r in range(world_size)
    ]

    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_gemm_sms = num_sms - NUM_COMM_SMS

    ctx = AllGatherGemmContextSymmMem(
        rank=rank,
        num_ranks=world_size,
        NUM_COMM_SMS=NUM_COMM_SMS,
        NUM_GEMM_SMS=num_gemm_sms,
        ag_stream=ag_stream,
        symm_input_buf=symm_input_buf,
        symm_ag_a_buf=symm_ag_a_buf,
        ag_signal_buf=ag_signal_buf,
        input_hdl=input_hdl,
        ag_hdl=ag_hdl,
        signal_hdl=signal_hdl,
        mc_ag_a_buf=mc_ag_a_buf,
        peer_signal_ptrs=peer_signal_ptrs,
        peer_symm_input_bufs=peer_symm_input_bufs,
        group=group,
    )

    return ctx


def cp_engine_full_mesh_pull_ag(
    rank: int,
    world_size: int,
    M_local: int,
    K: int,
    symm_input: torch.Tensor,
    symm_ag_a: torch.Tensor,
    peer_symm_input_bufs: List[torch.Tensor],
    ag_signal: torch.Tensor,
):
    """AllGather via Copy Engine full-mesh pull (PtoP data + PtoP signal).

    Signal writes via cuStreamWriteValue32.
    Caller must wrap in `with torch.cuda.stream(ag_stream):`.
    """
    # Self-shard: local copy + PtoP signal via cuStreamWriteValue32
    local_dst = symm_ag_a[rank * M_local : (rank + 1) * M_local, :]
    local_dst.copy_(symm_input)
    _SymmetricMemory.stream_write_value32(ag_signal, rank, 1)

    # Remote shards in rotated order
    for offset in range(1, world_size):
        src_rank = (rank + offset) % world_size
        remote_src = peer_symm_input_bufs[src_rank][src_rank, :M_local, :]
        local_dst = symm_ag_a[src_rank * M_local : (src_rank + 1) * M_local, :]
        local_dst.copy_(remote_src)
        # cuStreamWriteValue32 issues a system level fence before the write
        _SymmetricMemory.stream_write_value32(ag_signal, src_rank, 1)


@triton.jit
def consumer_bf16_a_block_fp8_matmul(
    # Pointers
    A_ptr,           # bf16 [M, K] gathered A (symm_mem)
    B_ptr,           # fp8 [N, K] weight (col-major)
    C_ptr,           # output [M, N]
    Bs_ptr,          # B scale [N//group_n, K//group_k]
    ag_signal_ptr,   # [world_size] uint32
    # Dimensions
    M, N, K,
    M_local,
    M_local_tiles,
    # Block quantization
    group_n, group_k,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_Bs_k, stride_Bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    needs_masking: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
):
    """Non-persistent consumer GEMM: polls AG signal per rank-shard, then
    computes bf16-A on-the-fly-quantized block-FP8 matmul.

    Rank-aware tile rotation aligns CTA scheduling with AG signal arrival
    order (self-shard first, then peers in rotated order).
    """
    fp8_max = 448.0
    pid = tl.program_id(axis=0)

    num_pid_m = M_local_tiles * world_size
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Rank-aware tile rotation
    logical_src_rank = min(pid_m // M_local_tiles, world_size - 1)
    tile_in_rank = pid_m - logical_src_rank * M_local_tiles
    src_rank = (rank + logical_src_rank) % world_size
    pid_m = src_rank * M_local_tiles + tile_in_rank

    tile_row_start = src_rank * M_local + tile_in_rank * BLOCK_SIZE_M

    # Poll AG signal for src_rank's shard
    if tid(0) == 0:
        while ld_sys(ag_signal_ptr + src_rank) != 1:
            pass
    __syncthreads()

    offs_am = (tile_row_start + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs_ptr + offs_bsn * stride_Bs_n
    n_tiles_k_per_group_k = group_k // BLOCK_SIZE_K

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if needs_masking:
            a_bf16 = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        else:
            a_bf16 = tl.load(a_ptrs)
            b = tl.load(b_ptrs)

        # On-the-fly A quantization: per-row absmax
        a_absmax = tl.max(tl.abs(a_bf16), axis=1)
        a_s = tl.maximum(a_absmax, 1e-12) / fp8_max
        a_fp8 = tl.clamp(a_bf16 / a_s[:, None], -fp8_max, fp8_max)
        a_fp8 = a_fp8.to(B_ptr.dtype.element_ty)

        b_s = tl.load(Bs_ptrs)

        scale_step_k = tl.where((k + 1) % n_tiles_k_per_group_k == 0, 1, 0)
        accumulator += tl.dot(a_fp8, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        Bs_ptrs += scale_step_k * stride_Bs_k

    c = accumulator.to(tl.bfloat16)

    offs_cm = tile_row_start + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    rank_row_end = (src_rank + 1) * M_local
    c_mask = (offs_cm[:, None] < rank_row_end) & (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def allgather_gemm_op_symm_mem(
    ctx: AllGatherGemmContextSymmMem,
    a_bf16: torch.Tensor,
    b_fp8: torch.Tensor,
    b_scale: torch.Tensor,
    block_size: list,
    output_dtype: torch.dtype = torch.bfloat16,
    GROUP_SIZE_M: int = 32,
):
    """AG->GEMM overlap: CE-driven AllGather on ag_stream + consumer GEMM on
    current_stream with per-rank-shard signal polling for compute-comm overlap.

    Returns (gathered_A [M*world_size, K], C [M*world_size, N]).
    """
    assert len(block_size) == 3
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = block_size

    M_local, K = a_bf16.shape
    N = b_fp8.shape[0]
    M = M_local * ctx.num_ranks

    assert a_bf16.dtype == torch.bfloat16, f"Expected bf16 A, got {a_bf16.dtype}"
    assert a_bf16.shape[1] == b_fp8.shape[1], "K dimension mismatch"
    assert a_bf16.is_contiguous()

    num_pid_n = triton.cdiv(N, BLOCK_SIZE_N)

    # Copy A shard to symmetric memory
    symm_input = ctx.get_input_buf(M_local, K)
    symm_input.copy_(a_bf16)

    symm_ag_a = ctx.symm_ag_a_buf
    ag_signal = ctx.ag_signal_buf

    assert ag_signal.numel() >= ctx.num_ranks

    c = torch.empty((M, N), dtype=output_dtype, device=a_bf16.device)

    current_stream = torch.cuda.current_stream()
    ag_stream = ctx.ag_stream

    ctx.input_hdl.barrier()
    ag_signal.fill_(0)
    ctx.input_hdl.barrier()

    ag_stream.wait_stream(current_stream)

    # Step 1: AG on ag_stream (CE, no SM consumption)
    with torch.cuda.stream(ag_stream):
        cp_engine_full_mesh_pull_ag(
            rank=ctx.rank,
            world_size=ctx.num_ranks,
            M_local=M_local,
            K=K,
            symm_input=symm_input,
            symm_ag_a=symm_ag_a,
            peer_symm_input_bufs=ctx.peer_symm_input_bufs,
            ag_signal=ag_signal,
        )

    # Step 2: consumer GEMM on current_stream
    needs_masking = bool(K % BLOCK_SIZE_K != 0)
    M_local_tiles = triton.cdiv(M_local, BLOCK_SIZE_M)
    num_pid_m_grid = M_local_tiles * ctx.num_ranks
    num_tiles = num_pid_m_grid * num_pid_n

    consumer_bf16_a_block_fp8_matmul[(num_tiles,)](
        symm_ag_a,
        b_fp8,
        c,
        b_scale,
        ag_signal,
        M, N, K,
        M_local,
        M_local_tiles,
        BLOCK_SIZE_N,  # group_n
        BLOCK_SIZE_K,  # group_k
        symm_ag_a.stride(0),
        symm_ag_a.stride(1),
        b_fp8.stride(1),
        b_fp8.stride(0),
        c.stride(0),
        c.stride(1),
        b_scale.stride(1),
        b_scale.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        needs_masking=needs_masking,
        rank=ctx.rank,
        world_size=ctx.num_ranks,
        num_warps=4,
        num_stages=3,
    )

    current_stream.wait_stream(ag_stream)

    return symm_ag_a[:M_local*ctx.num_ranks,:], c


def maybe_fused_ag_shared_experts(
    hidden_states: torch.Tensor,
    shared_experts_is_fp8: bool,
    w_fp8: Optional[torch.Tensor],
    w_scale: Optional[torch.Tensor],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Fused AG + fp8-quant + gate_up gemm. Returns (None, None) to fall back.

    Standalone wrapper around allgather_gemm_op_symm_mem that handles
    communicator lookup, context creation, and output allocation.
    """
    from sglang.srt.distributed.device_communicators.torch_symm_mem import (
        TorchSymmMemCommunicator,
    )

    comm = TorchSymmMemCommunicator.get_active_comm()
    if comm is None:
        return None, None
    if not shared_experts_is_fp8:
        return None, None
    if w_fp8 is None or w_scale is None:
        return None, None
    _, K = hidden_states.shape

    ctx = comm.get_or_create_ag_gemm_ctx(K=K)
    if ctx is None:
        return None, None

    try:
        block_size = [64, 128, 128]
        allgather_output, gate_up_full = allgather_gemm_op_symm_mem(
            ctx=ctx,
            a_bf16=hidden_states,
            b_fp8=w_fp8.contiguous(),
            b_scale=w_scale,
            block_size=block_size,
        )
    except Exception:
        return None, None

    return allgather_output, gate_up_full



# ============================================================================
# Consumer BF16 Matmul Kernel (non-persistent, polls AG signal, then computes
# bf16 matmul C = A @ W^T where W is [N, K] row-major / F.linear format)
# ============================================================================
@triton.jit
def consumer_bf16_matmul(
    # Pointers
    A_ptr,           # bf16 [M, K] gathered A (symm_mem)
    B_ptr,           # bf16 [N, K] weight (row-major, F.linear format)
    C_ptr,           # output [M, N]
    ag_signal_ptr,   # [world_size] uint32
    # Dimensions
    M, N, K,
    M_local,
    M_local_tiles,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    needs_masking: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
):
    """Non-persistent consumer BF16 GEMM: polls AG signal per rank-shard, then
    computes bf16 matmul C = A @ W^T where W is [N, K] row-major.
    Rank-aware tile rotation aligns CTA scheduling with AG signal arrival
    order (self-shard first, then peers in rotated order).
    """
    pid = tl.program_id(axis=0)
    num_pid_m = M_local_tiles * world_size
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    # Rank-aware tile rotation
    logical_src_rank = min(pid_m // M_local_tiles, world_size - 1)
    tile_in_rank = pid_m - logical_src_rank * M_local_tiles
    src_rank = (rank + logical_src_rank) % world_size
    pid_m = src_rank * M_local_tiles + tile_in_rank
    tile_row_start = src_rank * M_local + tile_in_rank * BLOCK_SIZE_M
    # Poll AG signal for src_rank's shard
    if tid(0) == 0:
        while ld_sys(ag_signal_ptr + src_rank) != 1:
            pass
    __syncthreads()
    offs_am = (tile_row_start + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if needs_masking:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        else:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.bfloat16)
    offs_cm = tile_row_start + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    rank_row_end = (src_rank + 1) * M_local
    c_mask = (offs_cm[:, None] < rank_row_end) & (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
# ============================================================================
# AllGather -> BF16 GEMM Operation (AG first, BF16 GEMM follows with tile-level signaling)
# ============================================================================
def allgather_bf16_gemm_op_symm_mem(
    ctx: AllGatherGemmContextSymmMem,
    a_bf16: torch.Tensor,
    w_bf16: torch.Tensor,
    block_size: list,
    output_dtype: torch.dtype = torch.bfloat16,
    GROUP_SIZE_M: int = 32,
):
    """AG->BF16 Matmul overlap: CE-driven AllGather on ag_stream + consumer BF16
    GEMM on current_stream with per-rank-shard signal polling.
    Unlike allgather_gemm_op_symm_mem which uses FP8 block-wise matmul, this
    performs a pure BF16 matmul (C = A @ W^T) without quantization.
    Args:
        ctx: AllGatherGemmContextSymmMem context
        a_bf16: bf16 activation tensor [M, K] (local shard for this rank)
        w_bf16: bf16 weight tensor [N, K] (row-major, F.linear format)
        block_size: [BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K]
        output_dtype: output dtype
        GROUP_SIZE_M: swizzle group size for GEMM
    Returns:
        (gathered_A, C) where gathered_A is [M * world_size, K] and
        C is [M * world_size, N] with C = gathered_A @ W^T.
    """
    assert len(block_size) == 3
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = block_size
    M_local, K = a_bf16.shape
    N = w_bf16.shape[0]
    M = M_local * ctx.num_ranks
    assert a_bf16.dtype == torch.bfloat16, f"Expected bf16 A, got {a_bf16.dtype}"
    assert w_bf16.dtype == torch.bfloat16, f"Expected bf16 W, got {w_bf16.dtype}"
    assert a_bf16.shape[1] == w_bf16.shape[1], "K dimension mismatch between A and W"
    assert a_bf16.is_contiguous()
    num_pid_n = triton.cdiv(N, BLOCK_SIZE_N)
    # Copy A shard to symmetric memory
    symm_input = ctx.get_input_buf(M_local, K)
    symm_input.copy_(a_bf16)
    symm_ag_a = ctx.symm_ag_a_buf
    ag_signal = ctx.ag_signal_buf
    assert ag_signal.numel() >= ctx.num_ranks
    c = torch.empty((M, N), dtype=output_dtype, device=a_bf16.device)
    current_stream = torch.cuda.current_stream()
    ag_stream = ctx.ag_stream
    ctx.input_hdl.barrier()
    ag_signal.fill_(0)
    ctx.input_hdl.barrier()
    ag_stream.wait_stream(current_stream)
    # Step 1: AG on ag_stream (CE, no SM consumption)
    with torch.cuda.stream(ag_stream):
        cp_engine_full_mesh_pull_ag(
            rank=ctx.rank,
            world_size=ctx.num_ranks,
            M_local=M_local,
            K=K,
            symm_input=symm_input,
            symm_ag_a=symm_ag_a,
            peer_symm_input_bufs=ctx.peer_symm_input_bufs,
            ag_signal=ag_signal,
        )
    # Step 2: consumer BF16 GEMM on current_stream
    needs_masking = bool(K % BLOCK_SIZE_K != 0)
    M_local_tiles = triton.cdiv(M_local, BLOCK_SIZE_M)
    num_pid_m_grid = M_local_tiles * ctx.num_ranks
    num_tiles = num_pid_m_grid * num_pid_n
    consumer_bf16_matmul[(num_tiles,)](
        symm_ag_a,
        w_bf16,
        c,
        ag_signal,
        M, N, K,
        M_local,
        M_local_tiles,
        symm_ag_a.stride(0),
        symm_ag_a.stride(1),
        w_bf16.stride(1),  # stride_bk (K is dim 1 in [N, K] row-major)
        w_bf16.stride(0),  # stride_bn (N is dim 0 in [N, K] row-major)
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        needs_masking=needs_masking,
        rank=ctx.rank,
        world_size=ctx.num_ranks,
        num_warps=4,
        num_stages=3,
    )
    current_stream.wait_stream(ag_stream)
    return symm_ag_a[:M, :], c


def maybe_fused_ag_gate_mm(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Fused AllGather + gate matmul.  Returns ``(ag_out, router_logits)``
    or ``(None, None)`` on fallback.

    This is a convenience wrapper that:
      1. Retrieves the active ``TorchSymmMemCommunicator``
      2. Gets/creates the AG+GEMM context (cached per K, shared with
         shared-experts path via ``get_or_create_ag_gemm_ctx``)
      3. Calls :func:`allgather_bf16_gemm_op_symm_mem`
      4. Falls back to ``(None, None)`` on any error

    Args:
        hidden_states: Local shard [M_local, K] in bf16.
        gate_weight: Gate projection weight [N, K] in bf16.

    Returns:
        ``(ag_out, router_logits)`` on success, ``(None, None)`` on failure.
    """
    from sglang.srt.distributed.device_communicators.torch_symm_mem import (
        TorchSymmMemCommunicator,
    )

    comm = TorchSymmMemCommunicator.get_active_comm()
    if comm is None:
        return None, None
    if gate_weight.dtype != torch.bfloat16:
        return None, None

    _, K = hidden_states.shape

    ctx = comm.get_or_create_ag_gemm_ctx(K=K)
    if ctx is None:
        return None, None

    try:
        block_size = [64, 128, 128]
        allgather_output, router_logits = allgather_bf16_gemm_op_symm_mem(
            ctx=ctx,
            a_bf16=hidden_states,
            w_bf16=gate_weight,
            block_size=block_size,
        )
    except Exception:
        logger.warning(
            "maybe_fused_ag_gate_mm failed, falling back to separate AG + matmul",
            exc_info=True,
        )
        return None, None

    return allgather_output, router_logits