"""
Fused TopK Reduce-Sum + Add Shared Expert Output + Reduce-Scatter Overlap Kernel
================================================================================
Intra-SM Mode: compute and communication are fused into a single kernel.

Compute:
  1. topk_reduce_sum: sum routed expert outputs across topk dimension
     (gating weights have already been applied by the expert GEMM kernels,
      so only a plain sum is needed here)
  2. add shared_expert_output: element-wise add the shared expert output

Communication:
  reduce-scatter: distribute the combined result across ranks

Architecture:
  Single Kernel (one CTA per tile, persistent loop over all tiles)
  +---------------------------------------------------+
  | Phase 1: Compute (topk reduce-sum + add shared)   |
  |          + A2A push to peer symmetric buffers      |
  | Phase 2: CTA-grid barrier + cross-rank barrier    |
  | Phase 3: Local reduce (host-side torch.sum)       |
  +---------------------------------------------------+

Input shapes (per rank):
  - expert_outputs:       [M, TOPK, N] or [M * TOPK, N]  (routed expert outputs)
  - shared_expert_output: [M, N]                          (shared expert output)

Output shape (per rank):
  - output: [M_per_rank, N]  where M_per_rank = M // world_size

Symmetric memory layout (per rank):
  - [M, N]  (= [world_size * M_per_rank, N])
    Each source rank i writes its segment into rows
    [i * M_per_rank : (i+1) * M_per_rank] of the destination rank's buffer.
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl
from dataclasses import dataclass, field
from typing import Optional


# ===========================================================================
# Section 1: PTX Instruction Primitives
# ===========================================================================

@triton.jit
def _get_flat_tid():
    """Flattened thread index within the CTA."""
    tid_x, tid_y, tid_z = tl.inline_asm_elementwise(
        "mov.u32 $0, %tid.x; mov.u32 $1, %tid.y; mov.u32 $2, %tid.z;",
        "=r,=r,=r", [], dtype=(tl.uint32, tl.uint32, tl.uint32),
        is_pure=True, pack=1,
    )
    ntid_x, ntid_y, _ = tl.inline_asm_elementwise(
        "mov.u32 $0, %ntid.x; mov.u32 $1, %ntid.y; mov.u32 $2, %ntid.z;",
        "=r,=r,=r", [], dtype=(tl.uint32, tl.uint32, tl.uint32),
        is_pure=True, pack=1,
    )
    return tid_z * ntid_y * ntid_x + tid_y * ntid_x + tid_x


@triton.jit
def _get_flat_bid():
    """Flattened CTA index within the grid."""
    return (
        tl.program_id(2) * tl.num_programs(1) * tl.num_programs(0)
        + tl.program_id(1) * tl.num_programs(0)
        + tl.program_id(0)
    )


@triton.jit
def _atomic_add_release(ptr, val):
    """GPU-scope release atomic add (u32)."""
    tl.inline_asm_elementwise(
        """
        {
            .reg .u32 %inc;
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
    """GPU-scope acquire load (u32)."""
    return tl.inline_asm_elementwise(
        """
        {
            .reg .u32 %val;
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
    """Store expected | 0x80000000 with GPU-scope release."""
    tl.inline_asm_elementwise(
        """
        {
            .reg .u32 %mask, %val;
            mov.u32  %mask, 0x80000000;
            or.b32   %val, %mask, $2;
            st.global.release.gpu.u32 [$1], %val;
        }
        """,
        "=r, l, r",
        [ptr, tl.cast(expected, tl.uint32)],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _cas_sys_release(addrs, expected, desired):
    """System-scope release CAS with PTX-level spin-loop."""
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
    """System-scope acquire CAS with PTX-level spin-loop."""
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
def barrier_on_this_grid(barrier_ptr):
    """CTA-grid barrier using master-CTA pattern (primitives.md S5.2).

    All CTAs increment a counter; bid==0 spins until all arrive,
    then flips the high bit to release everyone.
    Host must zero the counter before each kernel launch.
    """
    expected = tl.num_programs(0) * tl.num_programs(1) * tl.num_programs(2)

    if _get_flat_tid() == 0:
        _atomic_add_release(barrier_ptr, 1)
        if _get_flat_bid() == 0:
            # Master CTA: spin until all CTAs have arrived
            while _load_acquire(barrier_ptr) != expected:
                pass
            _store_release_with_highbit(barrier_ptr, expected)
        else:
            # Other CTAs: spin until the high bit is set
            while (_load_acquire(barrier_ptr) & 0x80000000) == 0:
                pass

    tl.debug_barrier()


@triton.jit
def barrier_all_intra_node_atomic_cas_block(
    local_rank: tl.constexpr,
    rank: tl.constexpr,
    local_world_size: tl.constexpr,
    symm_flag_ptr,
):
    """Cross-rank barrier via system-scope CAS on symmetric-memory signal pads.

    Phase 1: each thread (tid < world_size) signals one peer with CAS(0->1).
    Phase 2: each thread waits for its peer's signal with CAS(1->0).
    Self-resetting: no host-side reset needed between iterations.
    """
    flat_tid = _get_flat_tid()
    local_rank_offset = rank - local_rank

    ptrs = symm_flag_ptr.to(tl.pointer_type(tl.uint64))

    # Phase 1: each thread with tid < world_size signals one peer (parallel)
    if flat_tid < local_world_size:
        peer = flat_tid + local_rank_offset
        remote_base = tl.load(ptrs + peer).to(tl.pointer_type(tl.uint32))
        remote_addr = remote_base + local_rank
        _cas_sys_release(remote_addr, 0, 1)

    # Phase 2: each thread waits for its corresponding peer's signal (parallel)
    if flat_tid < local_world_size:
        local_base = tl.load(ptrs + rank).to(tl.pointer_type(tl.uint32))
        local_addr = local_base + flat_tid
        _cas_sys_acquire(local_addr, 1, 0)

    tl.debug_barrier()


# ===========================================================================
# Section 2: Context Dataclass & Initialization
# ===========================================================================

@dataclass
class TopkRsSOverlapContext:
    """
    Pre-allocated state for the TopK Reduce-Sum + Add Shared + Reduce-Scatter
    overlap kernel.

    Holds symmetric memory buffers, signal pads, and local sync primitives.
    Reusable across multiple kernel invocations (only barrier zero-out needed
    between calls).
    """
    # Problem dimensions
    max_M: int
    N: int
    topk: int
    dtype: torch.dtype

    # Distributed info
    rank: int
    num_ranks: int
    num_local_ranks: int

    # Local sync primitives
    grid_barrier: torch.Tensor  # [1], int32

    # Symmetric memory (set in __post_init__)
    symm_mem_hdl: Optional[object] = field(default=None, init=False)
    symm_buffer: Optional[torch.Tensor] = field(default=None, init=False)
    buf_ptrs: Optional[torch.Tensor] = field(default=None, init=False)          # [num_ranks], int64
    signal_pad_ptrs: Optional[torch.Tensor] = field(default=None, init=False)   # [num_ranks], int64

    def __post_init__(self):
        device = torch.cuda.current_device()
        world_size = self.num_ranks

        # Allocate symmetric buffer: [M, N] per rank for reduce-scatter
        # Each rank's buffer receives one M_per_rank-row segment from each source rank
        buf = symm_mem.empty(
            (self.max_M, self.N),
            dtype=self.dtype,
            device=device,
        )
        hdl = symm_mem.rendezvous(buf, group=dist.group.WORLD)
        self.symm_mem_hdl = hdl
        self.symm_buffer = buf

        # Build per-rank buffer pointer array (int64) for Triton dynamic indexing
        self.buf_ptrs = torch.tensor(
            [hdl.get_buffer(r, sizes=(self.max_M, self.N), dtype=self.dtype,
                            storage_offset=0).data_ptr()
             for r in range(world_size)],
            dtype=torch.int64, device=device,
        )

        # Build per-rank signal pad pointer array (int64)
        self.signal_pad_ptrs = torch.tensor(
            [hdl.get_signal_pad(r, (world_size,), torch.int32).data_ptr()
             for r in range(world_size)],
            dtype=torch.int64, device=device,
        )

        dist.barrier()

    def finalize(self):
        """Release symmetric memory resources."""
        dist.barrier()
        self.symm_mem_hdl = None
        self.symm_buffer = None
        self.buf_ptrs = None
        self.signal_pad_ptrs = None


def create_topk_rss_overlap_context(
    max_M: int,
    N: int,
    topk: int,
    dtype: torch.dtype = torch.bfloat16,
) -> TopkRsSOverlapContext:
    """Factory: allocate local tensors, construct and return context.

    Must be called after ``dist.init_process_group``.
    ``max_M`` is the maximum total token count (same across all ranks).
    Must be divisible by ``world_size``.
    ``N`` is the hidden dimension and must be divisible by ``n_chunks`` (default 2).
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_world_size = world_size  # assuming single-node

    assert max_M % world_size == 0, (
        f"max_M ({max_M}) must be divisible by world_size ({world_size})"
    )

    device = torch.cuda.current_device()
    grid_barrier = torch.zeros((1,), dtype=torch.int32, device=device)

    return TopkRsSOverlapContext(
        max_M=max_M,
        N=N,
        topk=topk,
        dtype=dtype,
        rank=rank,
        num_ranks=world_size,
        num_local_ranks=local_world_size,
        grid_barrier=grid_barrier,
    )


# ===========================================================================
# Section 3: Triton Kernel Implementation
# ===========================================================================

@triton.jit
def fused_topk_reduce_sum_add_shared_reduce_scatter_kernel(
    # ---- input/output pointers ----
    expert_outputs_ptr,       # [M * TOPK, N] — flattened routed expert outputs
    shared_expert_ptr,        # [M, N]        — shared expert output
    output_ptr,               # [M_per_rank, N] — reduce-scatter output (written by host-side torch.sum)
    routed_scaling_factor,    # scalar float — applied after topk reduce-sum
    # ---- symmetric buffer pointer arrays ----
    buf_ptrs,                 # [world_size], int64
    # ---- sync ----
    grid_barrier_ptr,         # [1], int32
    signal_pad_ptrs,          # [world_size], int64
    # ---- problem sizes (runtime) ----
    M,                        # total token count (same across all ranks)
    N,
    # ---- strides (constexpr) ----
    stride_xm: tl.constexpr,      # expert_outputs row stride (in [M*TOPK, N] layout)
    stride_xn: tl.constexpr,      # expert_outputs col stride
    stride_shared_m: tl.constexpr,  # shared_expert row stride
    stride_shared_n: tl.constexpr,  # shared_expert col stride
    stride_buf_m: tl.constexpr,   # symm_buffer row stride
    stride_buf_n: tl.constexpr,   # symm_buffer col stride
    # ---- distributed (constexpr) ----
    rank: tl.constexpr,
    world_size: tl.constexpr,
    # ---- tile sizes (constexpr) ----
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_CHUNKS: tl.constexpr,       # number of chunks along N dimension
    TOPK: tl.constexpr,           # compile-time topk for full unrolling
    DTYPE: tl.constexpr,          # output dtype (tl.bfloat16 or tl.float16)
):
    """
    Fused kernel: topk_reduce_sum + add shared_expert_output + reduce-scatter (A2A push).

    Phase 1: Each CTA (persistent) iterates over tiles for all destination peers.
             For each tile:
               a) Load expert_outputs, reduce across TOPK dimension (plain sum),
                  then multiply by routed_scaling_factor
               b) Load and add shared_expert_output
               c) Push the result to the destination peer's symmetric buffer
    Phase 2: Grid barrier + cross-rank barrier
    Phase 3: Local reduce is done on the host side via torch.sum
    """
    pid = tl.program_id(0)
    npid = tl.num_programs(0)

    M_per_rank = M // world_size
    N_per_chunk = N // N_CHUNKS
    N_per_chunk = tl.multiple_of(N_per_chunk, 16)  # alignment hint for vectorized loads

    num_tiles_m = tl.cdiv(M_per_rank, BLOCK_M)
    num_tiles_n = tl.cdiv(N_per_chunk, BLOCK_N)
    blocks_per_rank = num_tiles_m * num_tiles_n

    # Pre-compute destination segment offset: this rank writes at
    # offset rank * M_per_rank in the peer's buffer (int64 for overflow safety)
    dst_segment_offset = tl.cast(M_per_rank, tl.int64) * stride_buf_m * rank

    # === Phase 1: Compute (topk reduce-sum + add shared) + A2A push ===
    for n_chunk in tl.range(0, N_CHUNKS, step=1, loop_unroll_factor=1):
        n_chunk_off_x = n_chunk * N_per_chunk * stride_xn
        n_chunk_off_shared = n_chunk * N_per_chunk * stride_shared_n
        n_chunk_off_buf = n_chunk * N_per_chunk * stride_buf_n

        total_tiles = world_size * blocks_per_rank

        for tile_id in range(pid, total_tiles, npid):
            peer = tile_id // blocks_per_rank
            bid = tile_id % blocks_per_rank
            bid_m = bid // num_tiles_n
            bid_n = bid % num_tiles_n

            offs_m = bid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = bid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_m = offs_m < M_per_rank
            mask_n = offs_n < N_per_chunk
            mask = mask_m[:, None] & mask_n[None, :]

            # Global row index: peer p's segment starts at peer * M_per_rank
            global_row = peer * M_per_rank + offs_m  # [BLOCK_M]

            # --- Load expert outputs and reduce across TOPK ---
            # Input layout: [M * TOPK, N]
            # For token at global_row r, expert k: row = r * TOPK + k
            # Base pointer for the first expert of each token in this tile:
            base_row_off = (global_row.to(tl.int64)[:, None] * TOPK * stride_xm
                            + offs_n[None, :].to(tl.int64) * stride_xn)
            expert_ptrs = expert_outputs_ptr + n_chunk_off_x + base_row_off

            # Sum across TOPK dimension (gating weights already applied upstream)
            accum = tl.load(expert_ptrs, mask=mask, other=0.0).to(tl.float32)
            for k in tl.static_range(1, TOPK):
                accum += tl.load(expert_ptrs + k * stride_xm,
                                 mask=mask, other=0.0).to(tl.float32)
            accum = accum * routed_scaling_factor

            # --- Add shared expert output ---
            shared_row_off = (global_row.to(tl.int64)[:, None] * stride_shared_m
                              + offs_n[None, :].to(tl.int64) * stride_shared_n)
            shared_ptrs = shared_expert_ptr + n_chunk_off_shared + shared_row_off
            shared_val = tl.load(shared_ptrs, mask=mask, other=0.0).to(tl.float32)
            accum += shared_val

            # --- Push to peer's symmetric buffer ---
            # In peer's buffer, this rank (source) writes at offset
            # rank * M_per_rank (the segment belonging to this source)
            peer_buf = tl.load(buf_ptrs + peer).to(tl.pointer_type(DTYPE))
            peer_buf = tl.multiple_of(peer_buf, 16)  # enable vectorized stores
            dst_ptrs = (peer_buf
                        + n_chunk_off_buf
                        + dst_segment_offset
                        + offs_m[:, None].to(tl.int64) * stride_buf_m
                        + offs_n[None, :].to(tl.int64) * stride_buf_n)
            tl.store(dst_ptrs, accum.to(DTYPE), mask=mask)

    # === Phase 2: Synchronize ===
    # Grid barrier: all CTAs on this device must finish pushing (primitives.md S5.2)
    barrier_on_this_grid(grid_barrier_ptr)
    # Cross-rank barrier: only CTA 0 participates -- the signal pad slots are
    # binary (0<->1) and can only track one barrier invocation per rank pair.
    # If all CTAs called this, multiple CTAs' threads would CAS-contend on the
    # same slot, causing deadlock (primitives.md S5.3).
    if pid == 0:
        barrier_all_intra_node_atomic_cas_block(
            rank, rank, world_size, signal_pad_ptrs,
        )


# ===========================================================================
# Section 4: Python Entry Point
# ===========================================================================

def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def fused_topk_reduce_sum_add_shared_reduce_scatter(
    ctx: TopkRsSOverlapContext,
    expert_outputs: torch.Tensor,
    shared_expert_output: torch.Tensor,
    routed_scaling_factor: float = 1.0,
    output: Optional[torch.Tensor] = None,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    num_warps: int = 32,
    num_stages: int = 1,
    n_chunks: int = 2,
) -> torch.Tensor:
    """
    Host-side launcher for the fused topk_reduce_sum + add_shared + reduce_scatter
    overlap kernel (intra-SM mode).

    Args:
        ctx:           pre-allocated overlap context
        expert_outputs:  routed expert outputs, shape [M, TOPK, N] or [M*TOPK, N]
        shared_expert_output: shared expert output, shape [M, N]
        routed_scaling_factor: scalar applied after topk reduce-sum (default 1.0)
        output:          pre-allocated output tensor [M_per_rank, N], or None
        block_m:         tile size in M dimension (auto-computed if None)
        block_n:         tile size in N dimension (auto-computed if None)
        num_warps:       warps per CTA (default: 32 for memory-bound workload)
        num_stages:      pipeline stages (default: 1)
        n_chunks:        number of chunks along N dimension (default: 2)

    Returns:
        output tensor of shape [M_per_rank, N]
    """
    # Flatten input if 3D
    if expert_outputs.ndim == 3:
        M, topk, N = expert_outputs.shape
        x_flat = expert_outputs.view(M * topk, N)
    elif expert_outputs.ndim == 2:
        M_topk, N = expert_outputs.shape
        topk = ctx.topk
        M = M_topk // topk
        x_flat = expert_outputs
    else:
        raise ValueError(f"expert_outputs must be 2D or 3D, got {expert_outputs.ndim}D")

    assert topk == ctx.topk, f"topk mismatch: got {topk}, expected {ctx.topk}"
    assert M % ctx.num_ranks == 0, (
        f"M ({M}) must be divisible by world_size ({ctx.num_ranks})"
    )
    assert N == ctx.N, f"N mismatch: got {N}, expected {ctx.N}"
    assert M <= ctx.max_M, f"M ({M}) exceeds max_M ({ctx.max_M})"
    assert N % n_chunks == 0, f"N ({N}) must be divisible by n_chunks ({n_chunks})"

    M_per_rank = M // ctx.num_ranks
    N_per_chunk = N // n_chunks

    # Auto-compute block sizes
    elem_size = x_flat.element_size()
    if block_n is None:
        block_n = _next_power_of_2(N_per_chunk)
    if block_m is None:
        block_m = _next_power_of_2(16 * 1024 // block_n // elem_size)
    # Cap block sizes to avoid excessive register pressure
    block_m = min(block_m, 128)
    block_n = min(block_n, 512)

    # Resolve dtype
    DTYPE = tl.bfloat16 if x_flat.dtype == torch.bfloat16 else tl.float16

    # Allocate output
    if output is None:
        output = torch.empty(
            (M_per_rank, N), dtype=x_flat.dtype, device=x_flat.device,
        )

    # Reset sync tensors (must be 0 before each launch)
    ctx.grid_barrier.zero_()

    # Compute grid dimensions
    num_tiles_m = triton.cdiv(M_per_rank, block_m)
    num_tiles_n = triton.cdiv(N_per_chunk, block_n)
    blocks_per_rank = num_tiles_m * num_tiles_n
    total_tiles = ctx.num_ranks * blocks_per_rank
    # Cap CTAs to SM count — this is a persistent kernel with a grid barrier,
    # so ALL CTAs must be concurrently resident on SMs to avoid scheduling
    # deadlock.  is_pure=False on _load_acquire increases register pressure,
    # which can reduce occupancy to 1 CTA/SM; if num_ctas > SMs, later CTAs
    # cannot be scheduled and the barrier deadlocks.
    num_sm = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    num_ctas = min(total_tiles, num_sm)

    grid = (num_ctas,)

    # Launch kernel
    fused_topk_reduce_sum_add_shared_reduce_scatter_kernel[grid](
        x_flat,
        shared_expert_output,
        output,
        routed_scaling_factor,
        ctx.buf_ptrs,
        ctx.grid_barrier,
        ctx.signal_pad_ptrs,
        M, N,
        stride_xm=x_flat.stride(0),
        stride_xn=x_flat.stride(1),
        stride_shared_m=shared_expert_output.stride(0),
        stride_shared_n=shared_expert_output.stride(1),
        stride_buf_m=ctx.symm_buffer.stride(0),
        stride_buf_n=ctx.symm_buffer.stride(1),
        rank=ctx.rank,
        world_size=ctx.num_ranks,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        N_CHUNKS=n_chunks,
        TOPK=topk,
        DTYPE=DTYPE,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    # Phase 3: Local reduce — sum across source ranks in symmetric buffer
    # Buffer layout: [M, N] = [world_size * M_per_rank, N]
    # Rows [i * M_per_rank : (i+1) * M_per_rank] contain source rank i's contribution
    torch.sum(
        ctx.symm_buffer[:M].view(ctx.num_ranks, M_per_rank, N),
        dim=0,
        out=output,
    )

    return output