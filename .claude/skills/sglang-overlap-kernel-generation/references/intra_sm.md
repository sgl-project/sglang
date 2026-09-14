# Intra-SM Overlap: Fused Single Kernel

Compute and communication are fused into one kernel. Each CTA computes a tile, then
immediately performs communication via load/store to peer symmetric memory — no extra
streams or separate kernels needed.

> **Primitives**: all PTX helpers (`_get_flat_tid`, `_get_flat_bid`,
> `_atomic_add_release`, `_load_acquire`, `_cas_sys_release`, `_cas_sys_acquire`,
> `barrier_on_this_grid`, `barrier_all_intra_node_atomic_cas_block`, etc.) are defined
> in `references/primitives.md`. Read that file first for the canonical implementations.

## Architecture

```
Single Kernel (one CTA per tile)
┌──────────────────────────────────────┐
│ Phase 1: Compute tile                │
│ Phase 2: Push result to peer buffers │  ← A2A via symm_mem pointers
│ Phase 3: CTA barrier + rank barrier  │
│ Phase 4: Local reduce (if needed)    │  ← e.g., reduce-scatter
└──────────────────────────────────────┘
```

## Synchronization Overview

Intra-SM overlap needs two levels of synchronization (both defined in `primitives.md`):

1. **CTA-Grid barrier** (`barrier_on_this_grid`, primitives.md §5.2) — GPU-scope atomics
   ensure all CTAs on this device finish pushing before any CTA starts reducing.
2. **Cross-rank barrier** (`barrier_all_intra_node_atomic_cas_block`, primitives.md §5.3) —
   system-scope CAS ensures all ranks have completed their pushes.

See `primitives.md` §7 for memory ordering rules (release/acquire/relaxed).

## Step 1: Peer Access via Pointer Arrays

Symmetric memory provides a pointer array where `peer_ptrs[i]` points to rank `i`'s
buffer. In Triton, load the pointer and cast to the appropriate type:

```python
# In kernel arguments: peer_ptrs is a tensor of uint64 pointers, one per rank
peer_buf_addr = tl.load(peer_ptrs + peer_rank).to(tl.pointer_type(tl.float16))

# Use tl.multiple_of hint for aligned pointers to enable vectorized loads/stores
peer_buf_addr = tl.multiple_of(peer_buf_addr, 16)

# Write to peer's buffer at the appropriate offset
tl.store(peer_buf_addr + write_offset, tile_data, mask=mask)
```

## Step 2: Fused Kernel Template

### Reduce-Scatter Pattern

Each CTA computes a tile, pushes partial results to peer buffers (A2A), waits for all
ranks to finish pushing, then reduces contributions from all peers locally.

**Key performance optimizations** applied in this template:
1. **N_CHUNKS loop**: Split the N dimension into chunks to reduce register pressure and improve data locality
2. **Incremental pointer update**: Use `tl.static_range` + `ptr + j * stride` instead of recomputing full addresses each iteration
3. **`tl.multiple_of` hints**: Enable vectorized 128-bit loads/stores for aligned pointers
4. **`tl.constexpr` strides**: Let the compiler optimize address calculations at compile time
5. **Input layout as `[M * topk, N]`**: Flatten the topk dimension for contiguous memory access

```python
@triton.jit
def fused_compute_reduce_scatter(
    input_ptr, output_ptr,
    peer_ptrs,          # symm_mem buffer pointers (one per rank)
    barrier_ptr,        # grid-level barrier counter
    signal_pad_ptrs,    # cross-rank signal pad
    M, N, K,
    stride_xm: tl.constexpr,    # strides as constexpr for compiler optimization
    stride_xn: tl.constexpr,
    stride_buf_m: tl.constexpr,
    stride_buf_n: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    N_CHUNKS: tl.constexpr,     # number of chunks along N dimension
    TOPK: tl.constexpr,         # compile-time topk for full unrolling
    DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    npid = tl.num_programs(0)

    M_per_rank = M // world_size
    N_per_chunk = N // N_CHUNKS
    N_per_chunk = tl.multiple_of(N_per_chunk, 16)  # alignment hint

    num_tiles_m = tl.cdiv(M_per_rank, BLOCK_M)
    num_tiles_n = tl.cdiv(N_per_chunk, BLOCK_N)
    blocks_per_rank = num_tiles_m * num_tiles_n

    # Pre-compute destination segment offset (int64 to avoid overflow)
    dst_segment_offset = M_per_rank.to(tl.int64) * stride_buf_m * rank

    # === Phase 1: Compute + A2A push (with N_CHUNKS loop) ===
    for n_chunk in tl.range(0, N_CHUNKS, step=1, loop_unroll_factor=1):
        offs_n_chunk = n_chunk * N_per_chunk * stride_xn

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

            # Incremental pointer update for topk reduce:
            # Input layout: [M * topk, N], compute base pointer once,
            # then use ptr + j * stride_xm for each topk iteration
            offs_in = (offs_m[:, None].to(tl.int64) * stride_xm * TOPK
                       + offs_n[None, :].to(tl.int64) * stride_xn)
            src_segment_offset = peer.to(tl.int64) * M_per_rank * TOPK * stride_xm
            input_ptrs = input_ptr + offs_n_chunk + src_segment_offset + offs_in

            # Accumulate in float32 for numerical stability
            accum = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)
            for j in tl.static_range(1, TOPK):
                accum += tl.load(input_ptrs + j * stride_xm, mask=mask, other=0.0).to(tl.float32)

            # Apply scaling (scalar or per-token)
            accum *= scaling_factor

            # Push to peer's buffer with alignment hint
            peer_buf = tl.load(peer_ptrs + peer).to(tl.pointer_type(DTYPE))
            peer_buf = tl.multiple_of(peer_buf, 16)  # enable vectorized stores
            dst_ptrs = (peer_buf
                        + offs_n_chunk
                        + dst_segment_offset
                        + offs_m[:, None].to(tl.int64) * stride_buf_m
                        + offs_n[None, :].to(tl.int64) * stride_buf_n)
            tl.store(dst_ptrs, accum, mask=mask)

    # === Phase 2: Synchronize ===
    # Grid barrier: all CTAs on this device must finish pushing (primitives.md §5.2)
    barrier_on_this_grid(barrier_ptr)
    # Cross-rank barrier: only CTA 0 participates — the signal pad slots are
    # binary (0↔1) and can only track one barrier invocation per rank pair.
    # ALL CTAs must NOT call this; only pid == 0 does the cross-rank sync.
    # (primitives.md §5.3)
    if pid == 0:
        barrier_all_intra_node_atomic_cas_block(rank, rank, world_size, signal_pad_ptrs)

    # === Phase 3: Local reduce (can also be done host-side via torch.sum) ===
    # Host-side torch.sum is often simpler and equally fast:
    #   torch.sum(symm_buffer[:M].view(world_size, M_per_rank, N), dim=0, out=output)
```

**Performance notes on the template above:**
- `tl.static_range(1, TOPK)` enables full compile-time unrolling of the topk loop
- `offs_n_chunk` is added to both input and output pointers to handle N-dimension chunking
- The `N_CHUNKS` outer loop processes the N dimension in smaller pieces, reducing the
  register footprint of each tile and improving SM occupancy

### All-Gather Pattern

Simpler — each CTA pushes its local tile to all peers, barrier, then reads the full
gathered result.

```python
@triton.jit
def fused_compute_allgather(input_ptr, output_ptr, peer_ptrs, ...):
    pid = tl.program_id(0)

    # Phase 1: Compute
    result = compute_tile(input_ptr, pid, ...)

    # Phase 2: Push local result to all peers
    for peer in range(world_size):
        peer_buf = tl.load(peer_ptrs + peer).to(tl.pointer_type(result.dtype))
        tl.store(peer_buf + rank * chunk_size + local_offset, result)

    # Phase 3: Barrier (primitives.md §5.2 + §5.3)
    barrier_on_this_grid(barrier_ptr)
    # Only CTA 0 does cross-rank sync — signal pad slots are single-use binary flags
    if pid == 0:
        barrier_all_intra_node_atomic_cas_block(rank, rank, world_size, signal_pad_ptrs)

    # Phase 4: Read full gathered tensor (optional — may just use the buffer directly)
```

## Step 3: Host-Side Launch

Single kernel launch, no stream management needed. The fused kernel handles both
compute and communication, so it uses **all SMs** on the device.

```python
def launch_intra_sm_overlap(kernel_fn, input_tensor, ...):
    # Symmetric buffer shape should match the collective's communication layout:
    #   - Reduce-Scatter: [M, N] = [world_size * M_per_rank, N]
    #     (each rank's buffer receives one M_per_rank segment from every peer)
    #   - All-Gather: [M_per_rank, N]
    #     (each rank's buffer holds its local contribution for peers to read)
    #   - All-Reduce: [M, N]
    #     (full data shape for in-place reduction)
    buf = symm_mem.empty(buffer_shape, dtype=input_tensor.dtype, device=device)
    hdl = symm_mem.rendezvous(buf, group=dist.group.WORLD)

    # Collect peer buffer pointers
    peer_ptrs = torch.tensor(
        [hdl.get_buffer(r, sizes=buf[r].shape, dtype=buf.dtype,
                         storage_offset=0).data_ptr()
         for r in range(world_size)],
        dtype=torch.int64, device=device,
    )

    # Collect signal pad pointers (for cross-rank barrier)
    signal_pad_ptrs_device = torch.tensor(
        [hdl.get_signal_pad(i, (world_size,), torch.int32).data_ptr() for i in range(world_size)],
        dtype=torch.int64, device=device,
    )

    barrier = torch.zeros(1, dtype=torch.int32, device=device)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=device)

    # Intra-SM: single kernel uses ALL SMs (compute + communication are fused).
    # num_ctas must not exceed SM count — this is a persistent kernel with a grid barrier,
    # so all CTAs must be concurrently schedulable to avoid deadlock.
    num_sm = torch.cuda.get_device_properties(device).multi_processor_count
    num_ctas = min(num_tiles, num_sm)
    grid = (num_ctas,)
    kernel_fn[grid](
        input_tensor, output, peer_ptrs, barrier,
        signal_pad_ptrs_device,
        M, N, K, rank, world_size,
        BLOCK_M=64, BLOCK_N=128,
    )
```