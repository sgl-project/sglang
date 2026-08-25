# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

import triton
import triton.language as tl
from triton.language import core

logger = logging.getLogger(__name__)


@core.extern
def __syncthreads(_semantic=None):
    return tl.tensor(_semantic.builder.create_barrier(), tl.void)


@triton.jit
def _get_tid():
    return tl.inline_asm_elementwise(
        """
        mov.u32 $0, %tid.x;
        mov.u32 $1, %tid.y;
        mov.u32 $2, %tid.z;
        """,
        "=r,=r,=r",
        [],
        dtype=(tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _get_ntid():
    return tl.inline_asm_elementwise(
        """
        mov.u32 $0, %ntid.x;
        mov.u32 $1, %ntid.y;
        mov.u32 $2, %ntid.z;
        """,
        "=r,=r,=r",
        [],
        dtype=(tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _get_flat_tid():
    tid_x, tid_y, tid_z = _get_tid()
    ntid_x, ntid_y, _ = _get_ntid()
    return tid_z * ntid_y * ntid_x + tid_y * ntid_x + tid_x


@triton.jit
def _get_flat_bid():
    return (
        tl.program_id(2) * tl.num_programs(1) * tl.num_programs(0)
        + tl.program_id(1) * tl.num_programs(0)
        + tl.program_id(0)
    )


@triton.jit
def _atomic_add_release(ptr, val):
    """atom.global.release.gpu.add.u32"""
    tl.inline_asm_elementwise(
        """
        {
            .reg .u32   %inc;
            atom.global.release.gpu.add.u32 %inc, [$1], $2;
        }
        """,
        "=r, l, r",
        [ptr, tl.cast(val, tl.uint32)],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _load_acquire(ptr):
    """ld.global.acquire.gpu.u32"""
    return tl.inline_asm_elementwise(
        """
        {
            .reg .u32   %val;
            ld.global.acquire.gpu.u32 %val, [$1];
            mov.u32 $0, %val;
        }
        """,
        "=r, l",
        [ptr],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _store_release_with_highbit(ptr, expected):
    """st.release.gpu with high-bit set (expected | 0x80000000)."""
    tl.inline_asm_elementwise(
        """
        {
            .reg .u32   %mask, %val;
            mov.u32  %mask, 0x80000000;
            or.b32   %val, %mask, $2;
            st.global.release.gpu.u32   [$1], %val;
        }
        """,
        "=r, l, r",
        [ptr, tl.cast(expected, tl.uint32)],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def barrier_on_this_grid(barrier_ptr):
    """Synchronize all CTAs in this kernel launch (split-counter pattern).

    All CTAs atomically increment, CTA-0 spins until count == expected
    then sets high-bit to release everyone.
    """
    __syncthreads()
    pid_size_x = tl.num_programs(axis=0)
    pid_size_y = tl.num_programs(axis=1)
    pid_size_z = tl.num_programs(axis=2)
    expected = pid_size_x * pid_size_y * pid_size_z

    if _get_flat_tid() == 0:
        _atomic_add_release(barrier_ptr, 1)
        if _get_flat_bid() == 0:
            while _load_acquire(barrier_ptr) != expected:
                pass
            _store_release_with_highbit(barrier_ptr, expected)
        else:
            while (_load_acquire(barrier_ptr) & 0x80000000) == 0:
                pass

    __syncthreads()


@triton.jit
def _cas_sys_release(addrs, expected, desired):
    """atom.global.release.sys.cas.b32 — spins until CAS succeeds."""
    return tl.inline_asm_elementwise(
        f"""
        {{
            .reg .u32   %tmp32_<1>;
            .reg .pred  %p<1>;
            cas_loop:
                atom.global.release.sys.cas.b32 %tmp32_0, [$1], {expected}, {desired};
                setp.eq.u32 %p0, %tmp32_0, {expected};
                @!%p0 bra cas_loop;
            mov.u32 $0, %tmp32_0;
        }}
        """,
        "=r, l",
        [addrs],
        dtype=addrs.dtype,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _cas_sys_acquire(addrs, expected, desired):
    """atom.global.acquire.sys.cas.b32 — spins until CAS succeeds."""
    return tl.inline_asm_elementwise(
        f"""
        {{
            .reg .u32   %tmp32_<1>;
            .reg .pred  %p<1>;
            cas_loop:
                atom.global.acquire.sys.cas.b32 %tmp32_0, [$1], {expected}, {desired};
                setp.eq.u32 %p0, %tmp32_0, {expected};
                @!%p0 bra cas_loop;
            mov.u32 $0, %tmp32_0;
        }}
        """,
        "=r, l",
        [addrs],
        dtype=addrs.dtype,
        is_pure=False,
        pack=1,
    )


@triton.jit
def barrier_all_intra_node_atomic_cas_block(
    local_rank: tl.constexpr,
    rank: tl.constexpr,
    local_world_size: tl.constexpr,
    symm_flag_ptr,
):
    """Cross-rank intra-node barrier via CAS on symmetric memory flags.

    Phase 1: CAS remote peer's flag[local_rank] 0->1 (notify peer).
    Phase 2: CAS local flag[thread_idx] 1->0 (wait for peer, reset).
    """
    __syncthreads()
    flat_tid = _get_flat_tid()
    local_rank_offset = rank - local_rank

    ptrs = symm_flag_ptr.to(tl.pointer_type(tl.uint64))

    if flat_tid < local_world_size:
        peer = flat_tid + local_rank_offset
        remote_base = tl.load(ptrs + peer).to(tl.pointer_type(tl.uint32))
        remote_addr = remote_base + local_rank
        _cas_sys_release(remote_addr, 0, 1)

    if flat_tid < local_world_size:
        local_base = tl.load(ptrs + rank).to(tl.pointer_type(tl.uint32))
        local_addr = local_base + flat_tid
        _cas_sys_acquire(local_addr, 1, 0)

    __syncthreads()


@dataclass
class MoEReduceRSSymmMemContext:
    """Context for symm_mem-based MoE reduce-scatter. Holds pre-allocated
    symmetric buffers and synchronization resources."""

    max_M: int   # max tokens (NOT ntokens * topk)
    N: int
    num_experts: int
    topk: int
    dtype: torch.dtype
    # distributed
    rank: int
    num_ranks: int
    num_local_ranks: int
    n_chunks_max: int
    # local sync primitives
    grid_barrier: torch.Tensor   # [1] int32
    group: Optional[object] = None

    # Computed in __post_init__
    local_rank: int = field(init=False)
    nnodes: int = field(init=False)
    num_sms: int = field(init=False)  # GPU SM count, queried once at init

    # symm_mem handle and buffers
    symm_mem_hdl: Optional[object] = field(default=None, init=False)
    symm_reduce_scatter_buffer: Optional[torch.Tensor] = field(default=None, init=False)

    buf_tuple: Optional[Tuple[torch.Tensor, ...]] = field(default=None, init=False)
    signal_pad_tuple: Optional[Tuple[torch.Tensor, ...]] = field(default=None, init=False)

    # GPU-side pointer arrays for Triton kernel (int64)
    buf_ptrs: Optional[torch.Tensor] = field(default=None, init=False)
    signal_pad_ptrs: Optional[torch.Tensor] = field(default=None, init=False)

    def __post_init__(self):
        assert self.dtype in [torch.bfloat16, torch.float16], \
            "Only float16 / bfloat16 are supported"

        self.local_rank = self.rank % self.num_local_ranks
        self.nnodes = self.num_ranks // self.num_local_ranks
        self.num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count

        ntokens = self.max_M

        self.symm_reduce_scatter_buffer = symm_mem.empty(
            (ntokens, self.N),
            dtype=self.dtype,
            device=torch.cuda.current_device(),
        )
        rendezvous_group = self.group if self.group is not None else dist.group.WORLD
        self.symm_mem_hdl = symm_mem.rendezvous(
            self.symm_reduce_scatter_buffer,
            group=rendezvous_group,
        )

        self.buf_tuple = tuple(
            self.symm_mem_hdl.get_buffer(i, (ntokens, self.N), self.dtype)
            for i in range(self.num_ranks)
        )

        self.signal_pad_tuple = tuple(
            self.symm_mem_hdl.get_signal_pad(i, (self.num_ranks,), torch.int32)
            for i in range(self.num_ranks)
        )

        self.buf_ptrs = torch.tensor(
            [buf.data_ptr() for buf in self.buf_tuple],
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        self.signal_pad_ptrs = torch.tensor(
            [pad.data_ptr() for pad in self.signal_pad_tuple],
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )

        barrier_group = self.group if self.group is not None else None
        dist.barrier(group=barrier_group)

    def finalize(self):
        """Release resources. symm_mem tensors freed by Python GC."""
        barrier_group = self.group if self.group is not None else None
        dist.barrier(group=barrier_group)
        self.symm_reduce_scatter_buffer = None
        self.symm_mem_hdl = None
        self.buf_tuple = None
        self.signal_pad_tuple = None
        self.buf_ptrs = None
        self.signal_pad_ptrs = None


def create_moe_rs_symm_mem_context(
    rank: int,
    world_size: int,
    local_world_size: int,
    max_token_num: int,
    hidden_dim: int,
    num_experts: int,
    topk: int,
    input_dtype: torch.dtype,
    n_chunks_max: int = 8,
    group: Optional[object] = None,
) -> MoEReduceRSSymmMemContext:
    """Create MoEReduceRSSymmMemContext (symm_mem replacement for NVSHMEM version).

    group: process group for symm_mem rendezvous (defaults to WORLD;
        set to TP group when pipeline parallelism is used).
    """
    device = torch.cuda.current_device()
    grid_barrier = torch.zeros((1,), dtype=torch.int32, device=device)

    return MoEReduceRSSymmMemContext(
        max_M=max_token_num,
        N=hidden_dim,
        num_experts=num_experts,
        topk=topk,
        dtype=input_dtype,
        rank=rank,
        num_ranks=world_size,
        num_local_ranks=local_world_size,
        n_chunks_max=n_chunks_max,
        grid_barrier=grid_barrier,
        group=group,
    )


@triton.jit
def moe_reduce_rs_symm_mem_kernel(
    # input
    x_ptr,                          # [M * topk, N]
    shared_expert_out_ptr,          # [M, N]
    routed_scaling_factor,          # scalar float
    # symmetric buffer pointer array
    buf_ptrs,                       # int64[world_size]
    # sync
    signal_pad_ptrs,                # int64[world_size]
    grid_barrier_ptr,               # int32[1]
    # problem sizes
    M, N, topk,
    N_CHUNKS: tl.constexpr,
    # strides
    stride_xm: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_bm: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_se_m: tl.constexpr,
    stride_se_n: tl.constexpr,
    # distributed
    rank: tl.constexpr,
    world_size: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    TOPK: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Fused topk-reduce + add shared_expert_out + A2A push into symm buffers.

    Equivalent to: reduce_scatter(routed_scaling_factor * sum_topk(x) + shared_expert_out)
    Phase 1 pushes reduced tiles into peer symm buffers; after grid+cross-rank
    barriers, the host sums contributions for the final reduce-scatter output.
    """
    pid = tl.program_id(axis=0)
    npid = tl.num_programs(axis=0)

    M_per_rank = M // world_size
    N_per_chunk = N // N_CHUNKS
    N_per_chunk = tl.multiple_of(N_per_chunk, 16)

    n_blocks_m = tl.cdiv(M_per_rank, BLOCK_SIZE_M)
    n_blocks_n = tl.cdiv(N_per_chunk, BLOCK_SIZE_N)
    blocks_per_rank = n_blocks_m * n_blocks_n

    dst_segment_offset = M_per_rank.to(tl.int64) * stride_bm * rank

    # Phase 1: topk-reduce + add shared_expert + A2A push
    for n_chunk in tl.range(0, N_CHUNKS, step=1, loop_unroll_factor=1):
        offs_n_chunk = n_chunk * N_per_chunk * stride_xn

        total_tiles = world_size * blocks_per_rank

        for tile_id in range(pid, total_tiles, npid):
            peer = tile_id // blocks_per_rank
            bid = tile_id % blocks_per_rank

            bid_m = bid // n_blocks_n
            bid_n = bid % n_blocks_n

            offs_m = bid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_n = bid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            mask_m = offs_m < M_per_rank
            mask_n = offs_n < N_per_chunk
            mask = mask_m[:, None] & mask_n[None, :]

            # topk reduce (fp32 accumulator for precision)
            offs_in = offs_m[:, None].to(tl.int64) * stride_xm * TOPK + offs_n[None, :].to(tl.int64) * stride_xn
            src_segment_offset = peer.to(tl.int64) * M_per_rank * TOPK * stride_xm
            input_ptrs = x_ptr + offs_n_chunk + src_segment_offset + offs_in

            reduced = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)
            for j in tl.static_range(1, TOPK):
                reduced += tl.load(input_ptrs + j * stride_xm, mask=mask, other=0.0).to(tl.float32)
            reduced = reduced * routed_scaling_factor

            # add shared expert output
            se_segment_offset = peer.to(tl.int64) * M_per_rank * stride_se_m
            se_ptrs = (shared_expert_out_ptr
                       + offs_n_chunk
                       + se_segment_offset
                       + offs_m[:, None].to(tl.int64) * stride_se_m
                       + offs_n[None, :].to(tl.int64) * stride_se_n)
            shared_expert_val = tl.load(se_ptrs, mask=mask, other=0.0).to(tl.float32)
            reduced = reduced + shared_expert_val

            # A2A push into peer's symm buffer
            peer_buf_ptr = tl.load(buf_ptrs + peer).to(tl.pointer_type(DTYPE))
            peer_buf_ptr = tl.multiple_of(peer_buf_ptr, 16)
            dst_ptrs = (peer_buf_ptr
                        + offs_n_chunk
                        + dst_segment_offset
                        + offs_m[:, None].to(tl.int64) * stride_bm
                        + offs_n[None, :].to(tl.int64) * stride_bn)
            tl.store(dst_ptrs, reduced, mask=mask)

    # Grid barrier (all local CTAs done writing)
    barrier_on_this_grid(grid_barrier_ptr)

    # Cross-rank barrier (CAS-based intra-node sync)
    if pid == 0:
        barrier_all_intra_node_atomic_cas_block(
            rank,
            rank,
            world_size,
            signal_pad_ptrs,
        )


@triton.jit
def moe_reduce_rs_without_se_symm_mem_kernel(
    # input
    x_ptr,                          # [M * topk, N]
    routed_scaling_factor,          # scalar float
    # symmetric buffer pointer array
    buf_ptrs,                       # int64[world_size]
    # sync
    signal_pad_ptrs,                # int64[world_size]
    grid_barrier_ptr,               # int32[1]
    # problem sizes
    M, N, topk,
    N_CHUNKS: tl.constexpr,
    # strides
    stride_xm: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_bm: tl.constexpr,
    stride_bn: tl.constexpr,
    # distributed
    rank: tl.constexpr,
    world_size: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    TOPK: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Fused topk-reduce + add shared_expert_out + A2A push into symm buffers.

    Equivalent to: reduce_scatter(routed_scaling_factor * sum_topk(x) + shared_expert_out)
    Phase 1 pushes reduced tiles into peer symm buffers; after grid+cross-rank
    barriers, the host sums contributions for the final reduce-scatter output.
    """
    pid = tl.program_id(axis=0)
    npid = tl.num_programs(axis=0)

    M_per_rank = M // world_size
    N_per_chunk = N // N_CHUNKS
    N_per_chunk = tl.multiple_of(N_per_chunk, 16)

    n_blocks_m = tl.cdiv(M_per_rank, BLOCK_SIZE_M)
    n_blocks_n = tl.cdiv(N_per_chunk, BLOCK_SIZE_N)
    blocks_per_rank = n_blocks_m * n_blocks_n

    dst_segment_offset = M_per_rank.to(tl.int64) * stride_bm * rank

    # Phase 1: topk-reduce + add shared_expert + A2A push
    for n_chunk in tl.range(0, N_CHUNKS, step=1, loop_unroll_factor=1):
        offs_n_chunk = n_chunk * N_per_chunk * stride_xn

        total_tiles = world_size * blocks_per_rank

        for tile_id in range(pid, total_tiles, npid):
            peer = tile_id // blocks_per_rank
            bid = tile_id % blocks_per_rank

            bid_m = bid // n_blocks_n
            bid_n = bid % n_blocks_n

            offs_m = bid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_n = bid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            mask_m = offs_m < M_per_rank
            mask_n = offs_n < N_per_chunk
            mask = mask_m[:, None] & mask_n[None, :]

            # topk reduce (fp32 accumulator for precision)
            offs_in = offs_m[:, None].to(tl.int64) * stride_xm * TOPK + offs_n[None, :].to(tl.int64) * stride_xn
            src_segment_offset = peer.to(tl.int64) * M_per_rank * TOPK * stride_xm
            input_ptrs = x_ptr + offs_n_chunk + src_segment_offset + offs_in

            reduced = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)
            for j in tl.static_range(1, TOPK):
                reduced += tl.load(input_ptrs + j * stride_xm, mask=mask, other=0.0).to(tl.float32)
            reduced = reduced * routed_scaling_factor

            # A2A push into peer's symm buffer
            peer_buf_ptr = tl.load(buf_ptrs + peer).to(tl.pointer_type(DTYPE))
            peer_buf_ptr = tl.multiple_of(peer_buf_ptr, 16)
            dst_ptrs = (peer_buf_ptr
                        + offs_n_chunk
                        + dst_segment_offset
                        + offs_m[:, None].to(tl.int64) * stride_bm
                        + offs_n[None, :].to(tl.int64) * stride_bn)
            tl.store(dst_ptrs, reduced, mask=mask)

    # Grid barrier (all local CTAs done writing)
    barrier_on_this_grid(grid_barrier_ptr)

    # Cross-rank barrier (CAS-based intra-node sync)
    if pid == 0:
        barrier_all_intra_node_atomic_cas_block(
            rank,
            rank,
            world_size,
            signal_pad_ptrs,
        )


def moe_reduce_rs_symm_mem(
    grouped_gemm_out: torch.Tensor,
    shared_expert_out: torch.Tensor,
    ctx: MoEReduceRSSymmMemContext,
    ntokens: int,
    n_chunks: int,
    out: torch.Tensor,
    routed_scaling_factor: float,
    BLOCK_SIZE_M: int = 64,
    BLOCK_SIZE_N: int = 128,
) -> torch.Tensor:
    """Fused reduce-topk + add shared expert + reduce-scatter via symm_mem.

    output = reduce_scatter(routed_scaling_factor * sum_topk(grouped_gemm_out)
                            + shared_expert_out)
    """

    N = ctx.N
    topk = ctx.topk
    world_size = ctx.num_ranks
    rank = ctx.rank

    my_buf = ctx.buf_tuple[rank]
    buf_ptrs = ctx.buf_ptrs

    ctx.grid_barrier.zero_()

    if shared_expert_out is not None:
        moe_reduce_rs_symm_mem_kernel[(ctx.num_sms,)](
            grouped_gemm_out,
            shared_expert_out,
            routed_scaling_factor,
            buf_ptrs,
            ctx.signal_pad_ptrs,
            ctx.grid_barrier,
            M=ntokens,
            N=N,
            topk=topk,
            N_CHUNKS=n_chunks,
            stride_xm=N,
            stride_xn=1,
            stride_bm=my_buf.stride(0),
            stride_bn=my_buf.stride(1),
            stride_se_m=shared_expert_out.stride(0),
            stride_se_n=shared_expert_out.stride(1),
            rank=rank,
            world_size=world_size,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            TOPK=topk,
            DTYPE=tl.bfloat16 if grouped_gemm_out.dtype == torch.bfloat16 else tl.float16,
            num_warps=32,
            num_stages=1,
        )
    else:
        moe_reduce_rs_without_se_symm_mem_kernel[(ctx.num_sms,)](
            grouped_gemm_out,
            routed_scaling_factor,
            buf_ptrs,
            ctx.signal_pad_ptrs,
            ctx.grid_barrier,
            M=ntokens,
            N=N,
            topk=topk,
            N_CHUNKS=n_chunks,
            stride_xm=N,
            stride_xn=1,
            stride_bm=my_buf.stride(0),
            stride_bn=my_buf.stride(1),
            rank=rank,
            world_size=world_size,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            TOPK=topk,
            DTYPE=tl.bfloat16 if grouped_gemm_out.dtype == torch.bfloat16 else tl.float16,
            num_warps=32,
            num_stages=1,
        )

    # Phase 2: local reduce — sum world_size contributions
    ntokens_per_rank = ntokens // world_size
    torch.sum(
        ctx.symm_reduce_scatter_buffer[:ntokens].view(world_size, ntokens_per_rank, N),
        dim=0,
        out=out,
    )
    return out


# ---------------------------------------------------------------------------
# MoE mul-sum + add shared output + reduce-scatter overlap kernel (for humming)
# ---------------------------------------------------------------------------

@triton.jit
def moe_mul_sum_add_rs_overlap_kernel(
    # ---- input/output pointers ----
    inputs_ptr,                         # [M * topk, N] flattened grouped_gemm_out
    topk_weights_ptr,                   # [M, topk]
    topk_ids_ptr,                       # [M, topk]
    expert_map_ptr,                     # [num_experts] or None
    shared_expert_out_ptr,              # [M, N] or None
    routed_scaling_factor,              # scalar float
    # ---- symmetric buffer pointer arrays ----
    buf_ptrs,                           # int64[world_size]
    # ---- sync ----
    signal_pad_ptrs,                    # int64[world_size]
    grid_barrier_ptr,                   # int32[1]
    # ---- problem sizes (runtime) ----
    M,                                  # ntokens (logical)
    N,                                  # hidden dimension
    topk,
    N_CHUNKS: tl.constexpr,
    # ---- strides (constexpr) ----
    stride_xm: tl.constexpr,           # stride for flattened [M*topk, N] input (row stride = N)
    stride_xn: tl.constexpr,           # column stride for input (1)
    stride_bm: tl.constexpr,           # symm buffer row stride
    stride_bn: tl.constexpr,           # symm buffer column stride
    stride_se_m: tl.constexpr,         # shared_expert_out row stride
    stride_se_n: tl.constexpr,         # shared_expert_out column stride
    # ---- distributed (constexpr) ----
    rank: tl.constexpr,
    world_size: tl.constexpr,
    # ---- tile sizes (constexpr) ----
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    TOPK: tl.constexpr,
    DTYPE: tl.constexpr,
    # ---- gating flags ----
    HAS_EXPERT_MAP: tl.constexpr,
    IS_EP: tl.constexpr,
    HAS_SHARED_EXPERT: tl.constexpr,
):
    """Fused moe_fused_mul_sum + add_shared_output + A2A push for reduce-scatter.

    Equivalent to:
        output = reduce_scatter(
            routed_scaling_factor * sum_topk(inputs, topk_weights, topk_ids)
            + shared_expert_out
        )

    Phase 1: Each CTA computes a tile — topk weighted sum (with expert_map/is_ep
             gating) + optional shared expert add — then pushes the result into
             the appropriate peer's symmetric memory buffer.
    Phase 2: Grid barrier + cross-rank barrier.
    (Host performs the final local reduction via torch.sum.)
    """
    pid = tl.program_id(axis=0)
    npid = tl.num_programs(axis=0)

    M_per_rank = M // world_size
    N_per_chunk = N // N_CHUNKS
    N_per_chunk = tl.multiple_of(N_per_chunk, 16)

    n_blocks_m = tl.cdiv(M_per_rank, BLOCK_SIZE_M)
    n_blocks_n = tl.cdiv(N_per_chunk, BLOCK_SIZE_N)
    blocks_per_rank = n_blocks_m * n_blocks_n

    # Pre-compute destination segment offset for A2A push
    dst_segment_offset = M_per_rank.to(tl.int64) * stride_bm * rank

    # === Phase 1: topk weighted sum + optional shared expert add + A2A push ===
    for n_chunk in tl.range(0, N_CHUNKS, step=1, loop_unroll_factor=1):
        offs_n_chunk = n_chunk * N_per_chunk * stride_xn

        total_tiles = world_size * blocks_per_rank

        for tile_id in range(pid, total_tiles, npid):
            peer = tile_id // blocks_per_rank
            bid = tile_id % blocks_per_rank

            bid_m = bid // n_blocks_n
            bid_n = bid % n_blocks_n

            offs_m = bid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_n = bid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            mask_m = offs_m < M_per_rank
            mask_n = offs_n < N_per_chunk
            mask = mask_m[:, None] & mask_n[None, :]

            # --- topk weighted sum (adapted from moe_fused_mul_sum_kernel) ---
            # Input layout: [M * topk, N] where stride_xm = N (row-major)
            # For token `offs_m` and topk index `n`:
            #   row offset in flattened input = offs_m * TOPK * stride_xm + n * stride_xm
            # Source segment for peer `p` starts at: p * M_per_rank * TOPK * stride_xm
            src_segment_offset = peer.to(tl.int64) * M_per_rank * TOPK * stride_xm
            base_input_offset = (offs_m[:, None].to(tl.int64) * TOPK * stride_xm
                                 + offs_n[None, :].to(tl.int64) * stride_xn)
            input_ptrs = inputs_ptr + offs_n_chunk + src_segment_offset + base_input_offset

            # topk_ids / topk_weights: [M, topk], row-major
            # Token index within peer's segment: peer * M_per_rank + offs_m
            # For token t, topk slot n: index = t * TOPK + n
            token_offset_in_peer = peer.to(tl.int64) * M_per_rank + offs_m.to(tl.int64)

            # Accumulate topk weighted sum in fp32
            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            for n in tl.static_range(TOPK):
                # Load weight for this topk slot
                w_val = tl.load(
                    topk_weights_ptr + token_offset_in_peer * TOPK + n,
                    mask=mask_m,
                    other=0.0,
                ).to(tl.float32)
                if routed_scaling_factor != 1.0:
                    w_val = w_val * routed_scaling_factor

                # Load expert output with optional gating mask
                if HAS_EXPERT_MAP:
                    id_val = tl.load(
                        topk_ids_ptr + token_offset_in_peer * TOPK + n,
                        mask=mask_m,
                        other=0,
                    )
                    expert_mask = tl.load(expert_map_ptr + id_val) >= 0
                    a_vec = tl.load(
                        input_ptrs + n * stride_xm,
                        mask=mask & expert_mask[:, None],
                        other=0.0,
                    ).to(tl.float32)
                elif IS_EP:
                    id_val = tl.load(
                        topk_ids_ptr + token_offset_in_peer * TOPK + n,
                        mask=mask_m,
                        other=0,
                    )
                    expert_mask = id_val >= 0
                    a_vec = tl.load(
                        input_ptrs + n * stride_xm,
                        mask=mask & expert_mask[:, None],
                        other=0.0,
                    ).to(tl.float32)
                else:
                    a_vec = tl.load(
                        input_ptrs + n * stride_xm,
                        mask=mask,
                        other=0.0,
                    ).to(tl.float32)
                acc += a_vec * w_val[:, None]

            # --- add shared expert output (optional) ---
            if HAS_SHARED_EXPERT:
                se_segment_offset = peer.to(tl.int64) * M_per_rank * stride_se_m
                se_ptrs = (shared_expert_out_ptr
                           + offs_n_chunk
                           + se_segment_offset
                           + offs_m[:, None].to(tl.int64) * stride_se_m
                           + offs_n[None, :].to(tl.int64) * stride_se_n)
                shared_expert_val = tl.load(se_ptrs, mask=mask, other=0.0).to(tl.float32)
                acc = acc + shared_expert_val

            # --- A2A push into peer's symmetric memory buffer ---
            peer_buf_ptr = tl.load(buf_ptrs + peer).to(tl.pointer_type(DTYPE))
            peer_buf_ptr = tl.multiple_of(peer_buf_ptr, 16)
            dst_ptrs = (peer_buf_ptr
                        + offs_n_chunk
                        + dst_segment_offset
                        + offs_m[:, None].to(tl.int64) * stride_bm
                        + offs_n[None, :].to(tl.int64) * stride_bn)
            tl.store(dst_ptrs, acc.to(DTYPE), mask=mask)

    # === Phase 2: Synchronize ===
    # Grid barrier: all CTAs on this device must finish pushing
    barrier_on_this_grid(grid_barrier_ptr)

    # Cross-rank barrier: only CTA 0 participates
    if pid == 0:
        barrier_all_intra_node_atomic_cas_block(
            rank,
            rank,
            world_size,
            signal_pad_ptrs,
        )


def moe_mul_sum_add_rs_overlap(
    grouped_gemm_out: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_expert_out: Optional[torch.Tensor],
    ctx: MoEReduceRSSymmMemContext,
    ntokens: int,
    n_chunks: int,
    out: torch.Tensor,
    routed_scaling_factor: float,
    expert_map: Optional[torch.Tensor] = None,
    is_ep: bool = False,
    BLOCK_SIZE_M: int = 64,
    BLOCK_SIZE_N: int = 128,
) -> torch.Tensor:
    """Fused moe_fused_mul_sum + add_shared_output + reduce-scatter via symm_mem.

    output = reduce_scatter(
        routed_scaling_factor * sum_topk(grouped_gemm_out, topk_weights, topk_ids)
        + shared_expert_out
    )

    This kernel reuses MoEReduceRSSymmMemContext, so it shares the same
    symmetric memory buffers and synchronization resources with the
    non-humming reduce-scatter path.

    Args:
        grouped_gemm_out: [M, topk, N] or [M*topk, N] expert outputs.
        topk_weights: [M, topk] gating weights.
        topk_ids: [M, topk] expert indices.
        shared_expert_out: [M, N] shared expert output, or None.
        ctx: MoEReduceRSSymmMemContext with pre-allocated symmetric buffers.
        ntokens: M (number of tokens).
        n_chunks: number of chunks along N dimension.
        out: pre-allocated output tensor [M // world_size, N].
        routed_scaling_factor: scaling factor for routed experts.
        expert_map: optional expert mapping (values < 0 mean invalid).
        is_ep: whether expert parallelism is used.
        BLOCK_SIZE_M: tile size in M dimension (auto if None).
        BLOCK_SIZE_N: tile size in N dimension (auto if None).

    Returns:
        out: [M // world_size, N] reduce-scatter output.
    """
    N = ctx.N
    topk = ctx.topk
    world_size = ctx.num_ranks
    rank = ctx.rank
    has_expert_map = expert_map is not None
    has_shared_expert = shared_expert_out is not None

    assert ntokens % world_size == 0, \
        f"ntokens ({ntokens}) must be divisible by world_size ({world_size})"
    assert N % n_chunks == 0, \
        f"N ({N}) must be divisible by n_chunks ({n_chunks})"
    assert ntokens <= ctx.max_M, \
        f"ntokens ({ntokens}) exceeds max_M ({ctx.max_M})"

    # Flatten grouped_gemm_out to [M * topk, N] if needed
    if grouped_gemm_out.ndim == 3:
        M_in, topk_in, N_in = grouped_gemm_out.shape
        assert topk_in == topk, f"topk mismatch: got {topk_in}, expected {topk}"
        x_flat = grouped_gemm_out.reshape(M_in * topk_in, N_in)
    else:
        x_flat = grouped_gemm_out

    my_buf = ctx.buf_tuple[rank]
    buf_ptrs = ctx.buf_ptrs

    # Reset grid barrier before launch
    ctx.grid_barrier.zero_()

    # Launch single persistent kernel (intra-SM: all SMs)
    moe_mul_sum_add_rs_overlap_kernel[(ctx.num_sms,)](
        x_flat,
        topk_weights,
        topk_ids,
        expert_map,
        shared_expert_out,
        routed_scaling_factor,
        buf_ptrs,
        ctx.signal_pad_ptrs,
        ctx.grid_barrier,
        M=ntokens,
        N=N,
        topk=topk,
        N_CHUNKS=n_chunks,
        stride_xm=N,
        stride_xn=1,
        stride_bm=my_buf.stride(0),
        stride_bn=my_buf.stride(1),
        stride_se_m=shared_expert_out.stride(0) if has_shared_expert else 1,
        stride_se_n=shared_expert_out.stride(1) if has_shared_expert else 1,
        rank=rank,
        world_size=world_size,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        TOPK=topk,
        DTYPE=tl.bfloat16 if x_flat.dtype == torch.bfloat16 else tl.float16,
        HAS_EXPERT_MAP=has_expert_map,
        IS_EP=is_ep,
        HAS_SHARED_EXPERT=has_shared_expert,
        num_warps=32,
        num_stages=1,
    )

    # Phase 2: local reduce — sum world_size contributions for reduce-scatter
    ntokens_per_rank = ntokens // world_size
    torch.sum(
        ctx.symm_reduce_scatter_buffer[:ntokens].view(world_size, ntokens_per_rank, N),
        dim=0,
        out=out,
    )
    return out


def maybe_fused_shared_add_rs(
    final_hidden_states: torch.Tensor,
    shared_output: Optional[torch.Tensor],
    tp_size: int,
    n_shared_experts: int,
    top_k: int,
    routed_scaling_factor: float,
    is_humming: bool = False,
    topk_weights: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
    expert_map: Optional[torch.Tensor] = None,
    is_ep: bool = False,
) -> Optional[torch.Tensor]:
    """Fused add-shared + reduce-scatter. Returns None to fall back.

    Standalone wrapper that handles communicator lookup, context creation,
    and output allocation. When is_humming=True, uses the mul-sum overlap
    kernel that performs per-token topk weighted sum with topk_weights/topk_ids.
    Otherwise, uses the simple reduce-topk kernel.
    """
    from sglang.srt.distributed.device_communicators.torch_symm_mem import (
        TorchSymmMemCommunicator,
    )

    comm = TorchSymmMemCommunicator.get_active_comm()
    if comm is None:
        return None

    M, _, N = final_hidden_states.shape
    if M == 0 or (M % tp_size) != 0:
        return None

    ctx = comm.get_or_create_moe_rs_ctx(
        N=N,
        num_experts=n_shared_experts,
        topk=top_k,
        dtype=final_hidden_states.dtype,
    )
    if ctx is None:
        return None

    out = torch.empty(
        (M // tp_size, N),
        dtype=final_hidden_states.dtype,
        device=final_hidden_states.device,
    )
    try:
        if is_humming:
            # humming path: per-token topk weighted sum
            assert topk_weights is not None, \
                "topk_weights is required when is_humming=True"
            assert topk_ids is not None, \
                "topk_ids is required when is_humming=True"
            moe_mul_sum_add_rs_overlap(
                grouped_gemm_out=final_hidden_states.view(M * top_k, N)
                    if final_hidden_states.ndim == 2
                    else final_hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                shared_expert_out=shared_output,
                ctx=ctx,
                ntokens=M,
                n_chunks=N // 512,
                out=out,
                routed_scaling_factor=routed_scaling_factor,
                expert_map=expert_map,
                is_ep=is_ep,
            )
        else:
            moe_reduce_rs_symm_mem(
                grouped_gemm_out=final_hidden_states.view(M * top_k, N),
                shared_expert_out=shared_output,
                ctx=ctx,
                ntokens=M,
                n_chunks=N // 512,
                out=out,
                routed_scaling_factor=routed_scaling_factor,
            )
    except Exception as e:
        logger.warning(
            "maybe_fused_shared_add_rs failed, falling back to default path: %s", e
        )
        return None
    return out
