# Output Format Convention

The generated overlap kernel file **must** be organized into the following four sections, in order. Each section is clearly separated by comment banners. This structure ensures consistency, readability, and testability.

## Section 1: PTX Instruction Primitives (Dependencies)

All low-level PTX inline assembly helpers needed by the kernel. These are self-contained `@triton.jit` functions with no external dependencies beyond `triton` and `tl`.

**The canonical implementations live in `references/primitives.md`.** Read that file to get the exact code for each helper. Copy only the primitives the kernel actually uses into this section.

Available primitives (choose based on mode):

| Category | Primitives | Used by |
|----------|-----------|---------|
| Thread/block ID | `_get_flat_tid`, `_get_flat_bid` | all in-kernel modes |
| GPU-scope atomics | `_atomic_add_release`, `_load_acquire`, `_store_release_with_highbit` | intra-sm (grid barrier) |
| System-scope CAS | `_cas_sys_release`, `_cas_sys_acquire` | intra-sm only |
| Per-tile signals | `_send_signal`, `_wait_signal` | inter-sm only |
| CTA-Grid barrier | `barrier_on_this_grid` | intra-sm |
| Cross-rank barrier (in-kernel) | `barrier_all_intra_node_atomic_cas_block` | intra-sm only |
| Cross-rank barrier (host-side) | `symm_mem_hdl.barrier()` | inter-sm, without-sm (around `signal.zero_()`) |
| Signal polling/writing (without-sm) | `ld_sys`, `st_sys`, `__syncthreads`, `tid` | without-sm only (defined in `without_sm.md`) |
| CE-side signal ops (without-sm) | `stream_write_value32`, `cuStreamWaitValue32` | without-sm only (host-side, in `without_sm.md`) |

See `primitives.md` §8 for the full mode-to-primitive mapping.

## Section 2: Context Dataclass & Initialization

A `@dataclass` holding all pre-allocated buffers, symmetric memory handles, and synchronization state. Followed by a factory function `create_*_context()` that performs allocation and rendezvous.

```python
# ---------------------------------------------------------------------------
# Context Dataclass
# ---------------------------------------------------------------------------

@dataclass
class <Op>OverlapContext:
    """
    Pre-allocated state for the overlap kernel.
    Holds symmetric memory buffers, signal pads, and local sync primitives.
    """
    # Problem dimensions
    max_M: int
    N: int
    dtype: torch.dtype

    # Distributed info
    rank: int
    num_ranks: int
    num_local_ranks: int

    # SM allocation — semantics differ by overlap mode:
    #   intra-sm:   num_comm_sms = total SMs (single kernel uses all SMs);
    #               num_comp_sms is unused (0).
    #   inter-sm:   num_comm_sms = small set of SMs for persistent comm kernel;
    #               num_comp_sms = remaining SMs for compute kernel.
    #   without-sm: comm is on copy engine (zero SMs); compute kernel uses all SMs;
    #               num_comm_sms = 0, num_comp_sms = total SMs.
    num_comm_sms: int
    num_comp_sms: int = field(default=0, init=False)  # computed in __post_init__

    # Local sync primitives (non-symmetric, device tensors)
    grid_barrier: torch.Tensor       # [1], int32 — intra-kernel CTA barrier
    # ... additional counters/flags as needed ...

    # Symmetric memory (set in __post_init__)
    symm_mem_hdl: Optional[object] = field(default=None, init=False)
    symm_buffer: Optional[torch.Tensor] = field(default=None, init=False)
    buf_ptrs: Optional[torch.Tensor] = field(default=None, init=False)       # [num_ranks], int64
    signal_pad_ptrs: Optional[torch.Tensor] = field(default=None, init=False) # [num_ranks], int64

    # Communication stream (created once, reused across calls; used by both inter-SM and without-SM overlap)
    comm_stream: Optional[torch.cuda.Stream] = field(default=None, init=False)

    def __post_init__(self):
        device = torch.cuda.current_device()
        # Calculate SM allocation: total SMs minus communication SMs
        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.num_comp_sms = num_sms - self.num_comm_sms
        # Create communication stream (for two-stream overlap: inter-SM or without-SM/copy-engine)
        self.comm_stream = torch.cuda.Stream(device=device)
        # Allocate signal tensor for two-stream tile-level sync (one slot per tile, reused across calls)
        # Size based on max possible tiles: cdiv(max_M, block_m) * cdiv(N, block_n)
        # Reset to 0 before each kernel launch via signal.zero_()
        # self.signal = torch.zeros(max_total_tiles, dtype=torch.int32, device=device)
        # 1. Allocate symmetric buffer via symm_mem.empty()
        #    IMPORTANT: The symm_mem buffer shape must match the collective communication's
        #    data layout, NOT the compute kernel's input/output shape. The buffer is the
        #    medium for cross-rank data exchange, so its shape should reflect what the
        #    communication pattern needs:
        #      - All-Gather:      [M_per_rank, N]  (each rank's local contribution)
        #      - Reduce-Scatter:  [M, N] = [world_size * M_per_rank, N]  (receives from all peers)
        #      - All-Reduce:      [M, N]  (full data shape)
        # 2. Rendezvous across ranks via symm_mem.rendezvous()
        # 3. Build per-rank buffer pointer array (int64) for Triton dynamic indexing
        # 4. Build per-rank signal pad pointer array (int64)
        # 5. dist.barrier() to ensure all ranks complete setup
        ...

    def finalize(self):
        """Release symmetric memory resources."""
        dist.barrier()
        # Set all handles/buffers to None (freed by GC)
        ...


def create_<op>_overlap_context(...) -> <Op>OverlapContext:
    """Factory: allocate local tensors, construct and return context."""
    device = torch.cuda.current_device()
    grid_barrier = torch.zeros((1,), dtype=torch.int32, device=device)
    # ... allocate other local sync tensors ...
    return <Op>OverlapContext(...)
```

Key principles:
- Triton cannot index Python tuples dynamically — always expose pointer arrays (`buf_ptrs`, `signal_pad_ptrs`) as int64 tensors for `tl.load(ptrs + index)`
- Keep `max_M`, `N` etc. as the **logical** problem dimensions (not multiplied by topk/world_size) to avoid confusion
- The context is **reusable** across multiple kernel invocations with different actual sizes ≤ max

## Section 3: Triton Kernel Implementation

The `@triton.jit` kernel function(s). For intra-SM mode this is a single kernel; for inter-SM mode there may be two (compute kernel + persistent comm kernel).

```python
# ---------------------------------------------------------------------------
# Triton Kernel
# ---------------------------------------------------------------------------

@triton.jit
def <op>_overlap_kernel(
    # ---- input/output pointers ----
    input_ptr,
    output_ptr,
    # ---- symmetric buffer pointer arrays ----
    buf_ptrs,
    # ---- sync ----
    signal_ptr,
    signal_pad_ptrs,
    grid_barrier_ptr,
    # ---- problem sizes (runtime) ----
    M, N,
    # ---- strides (constexpr where possible) ----
    stride_im: tl.constexpr,
    stride_in: tl.constexpr,
    # ---- distributed (constexpr) ----
    rank: tl.constexpr,
    world_size: tl.constexpr,
    # ---- tile sizes (constexpr) ----
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    # ---- dtype ----
    DTYPE: tl.constexpr,
):
    """
    Overlap kernel: <describe compute> + <describe communication>.
    """
    # Phase 1: Compute + push to peer symmetric buffers
    ...

    # Grid barrier: ensure all CTAs complete before cross-rank sync
    barrier_on_this_grid(grid_barrier_ptr)

    # Cross-rank barrier
    barrier_all_intra_node_atomic_cas_block(rank, rank, world_size, signal_pad_ptrs)

    # Phase 2: Local reduction (if needed, e.g., reduce-scatter)
    ...
```

Design rules:
- Use `tl.constexpr` only for values fixed across calls (strides, rank, world_size, tile sizes). Frequently-changing quantities (`M`, `M_per_rank`, `n_chunks`, etc.) must be regular parameters or computed inside the kernel — each unique constexpr combination triggers a recompilation.
- Use `tl.multiple_of` hints for aligned pointers/offsets to enable vectorized loads
- Accumulate in `tl.float32` for numerical stability, cast to native dtype before store
- Use `int64` arithmetic for offset calculations to avoid overflow with large tensors
- Include `N_CHUNKS` loop if the N dimension exceeds what a single tile can cover

## Section 4: Python Entry Point

The host-side function that prepares arguments, resets barriers, and launches the kernel.

**Critical performance settings:**
- `num_warps=32`: These kernels are memory-bound; 32 warps (1024 threads) per CTA maximizes memory bandwidth utilization. Using the default (4 warps / 128 threads) can cause 50%+ performance loss.
- `num_stages=1`: No software pipelining needed for these simple load/store patterns.
- `n_chunks=N dimension // 1024`: Split the N dimension into chunks to reduce register pressure.
- Auto-compute `BLOCK_M`/`BLOCK_N` based on problem size to balance register usage.

```python
# ---------------------------------------------------------------------------
# Python Entry Point
# ---------------------------------------------------------------------------

def <op>_overlap(
    ctx: <Op>OverlapContext,
    input_tensor: torch.Tensor,
    # ... problem-specific args ...
    output: torch.Tensor,
    block_size: list,           # [block_m, block_n, block_k], e.g. [64, 128, 128]
    num_warps: int = 32,        # 32 warps for memory-bound workloads
    num_stages: int = 1,        # No software pipeline needed
    n_chunks: int = 2,          # N-dimension chunks
) -> torch.Tensor:
    """
    Host-side launcher for the overlap kernel.

    Args:
        ctx: pre-allocated overlap context (contains num_comm_sms, num_comp_sms, comm_stream)
        input_tensor: input data
        output: pre-allocated output tensor
        block_size: [block_m, block_n, block_k] tile dimensions, e.g. [64, 128, 128]
        num_warps: warps per CTA (default: 32 for memory-bound workload)
        num_stages: pipeline stages (default: 1)
        n_chunks: number of chunks along N dimension (default: 2)

    Returns:
        output tensor (same as input `output` argument)
    """
    assert len(block_size) == 3
    block_m, block_n, block_k = block_size[0], block_size[1], block_size[2]

    # Handle both [M, topk, N] and [M*topk, N] input layouts
    if input_tensor.ndim == 3:
        M, topk, N = input_tensor.shape
        x_flat = input_tensor.view(M * topk, N)
    elif input_tensor.ndim == 2:
        M_topk, N = input_tensor.shape
        topk = ctx.topk
        M = M_topk // topk
        x_flat = input_tensor

    N_per_chunk = N // n_chunks

    # Reset sync tensors (must be 0 before each launch)
    ctx.signal.zero_()
    ctx.grid_barrier.zero_()

    # Compute grid dimensions
    # SM allocation differs by overlap mode:
    #   intra-sm:   single kernel uses ALL SMs — num_ctas = min(total_tiles, num_sms)
    #   inter-sm:   comm kernel uses num_comm_sms; compute kernel uses num_comp_sms
    #   without-sm: compute kernel uses all SMs; comm is on copy engine (zero SMs)
    n_blocks_m = triton.cdiv(M_per_rank, block_m)
    n_blocks_n = triton.cdiv(N_per_chunk, block_n)
    total_tiles = world_size * n_blocks_m * n_blocks_n

    # For intra-sm: the single fused kernel handles both compute and communication,
    # so it should use all SMs on the device. The kernel contains a grid barrier,
    # so num_ctas must not exceed the number of SMs (otherwise scheduling deadlock).
    num_sm = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    num_ctas = min(total_tiles, num_sm)

    grid = (num_ctas,)

    # Launch kernel with explicit num_warps and num_stages
    <op>_overlap_kernel[grid](
        x_flat,
        output,
        ctx.buf_ptrs,
        ctx.signal,
        ctx.signal_pad_ptrs,
        ctx.grid_barrier,
        M, N,
        x_flat.stride(0), x_flat.stride(1),  # strides from flattened input
        ctx.symm_buffer.stride(0), ctx.symm_buffer.stride(1),
        rank=ctx.rank,
        world_size=ctx.num_ranks,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        TOPK=topk,
        N_CHUNKS=n_chunks,
        DTYPE=tl.bfloat16 if input_tensor.dtype == torch.bfloat16 else tl.float16,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output
```