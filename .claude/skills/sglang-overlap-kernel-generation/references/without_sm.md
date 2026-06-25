# Without-SM Overlap: Copy Engine (DMA) Based

Communication is offloaded to the **copy engine** (CE), consuming zero SM resources. A
background stream performs `copy_()` on symmetric memory (routed through DMA), while the
compute kernel runs simultaneously on the main stream. Synchronization between the two
streams is achieved through **signal tensors** — written by `cuStreamWriteValue32` on the
CE stream and polled by the compute kernel via `ld_sys` (system-scope acquire load).
Signal tensor shape determines granularity: `[world_size * splits_per_rank]` where
`splits_per_rank=1` gives one signal per rank and `splits_per_rank>1` gives finer-grained
chunk signaling.

## Architecture

The without-sm mode supports two execution orders:

### Comm-First: CE → Compute (e.g., AllGather → GEMM)

```
Main Stream (SM)                         Backend Stream (Copy Engine)
┌──────────────────────────────┐         ┌─────────────────────────────────────────┐
│ compute_kernel               │         │ for each peer in rotated order:         │
│   process local (no wait)    │         │   copy_(peer_symm_buf → local_dst)      │
│   poll signal[chunk_idx]     │ ◄────── │   stream_write_value32(progress, 1)     │
│   process remote chunk       │         │                                         │
└──────────────────────────────┘         └─────────────────────────────────────────┘
```

### Comp-First: Compute → CE (e.g., GEMM → ReduceScatter)

```
Main Stream (SM)                         Backend Stream (Copy Engine)
┌──────────────────────────────┐         ┌─────────────────────────────────────────┐
│ compute_kernel               │         │ for each tile:                          │
│   compute tile               │         │   cuStreamWaitValue32(signal[tile]==1)  │
│   st_sys signal[tile] = 1    │ ──────► │   copy_(local_tile → peer_symm_buf)     │
│                              │         │                                         │
└──────────────────────────────┘         └─────────────────────────────────────────┘
```

## Signal Mechanism: `stream_write_value32`

The compute kernel and CE stream synchronize via `stream_write_value32` (which wraps
`cuStreamWriteValue32`) — a 32-bit write on the CE stream with a **system-level fence**
before the write, guaranteeing that all prior `copy_()` operations on that stream are
visible before the signal value becomes `1`.

**Comm-First**: After the CE copies data from a peer, it writes `1` to the progress tensor
via `hdl.stream_write_value32`. The compute kernel polls via `ld_sys`.

**Comp-First**: The compute kernel writes `1` to a signal slot via `st_sys` after each
tile's stores complete. The CE stream enqueues `cuStreamWaitValue32` — a GPU-side hardware
wait that stalls the CE channel until the signal equals the expected value.

**⚠️ CRITICAL: The `progress` tensor MUST use `dtype=torch.uint32`, NOT `torch.int32`.**
`cuStreamWriteValue32` writes an unsigned 32-bit value, and PyTorch's `symm_mem` binding
validates that the input tensor is a flat, contiguous `uint32` tensor. Passing an `int32`
tensor raises: `RuntimeError: symm_mem::stream_write_value32_: input must be a flat,
contiguous uint32 tensor.` Always double-check that the signal tensor is `uint32`.

**Buffers required (in addition to data buffers):**

| Field | Shape / Type | Description |
|-------|-------------|-------------|
| `progress` | `[world_size * splits_per_rank]` uint32, **local** | Progress/signal tensor on the local device. No symmetric memory needed. **Must be `uint32` — `int32` causes a RuntimeError in `stream_write_value32`.** Reset via `progress.fill_(0)` between iterations. |

## Primitives Used

Without-sm does NOT use `_send_signal` / `_wait_signal` / `barrier_all_intra_node_atomic_cas_block`.
Instead, it uses the following from `primitives.md`:

- **Kernel-side**: `ld_sys`, `st_sys`, `tid(axis)`, `__syncthreads` (§2 & §5)
- **CE-side (host)**: `hdl.stream_write_value32` (comm-first) or `cuda.bindings.driver.cuStreamWaitValue32` (comp-first)
- **Cross-rank sync**: `symm_mem_hdl.barrier()` (§6.4) between iterations

---

## Path A: CE → Compute (Comm-First)

Example: AllGather (CE) → GEMM (compute). The CE pulls data from peers and writes
signals via `stream_write_value32`; the compute kernel polls signals and processes chunks
as they arrive.

### Step 1: Context (Buffers Required)

| Field | Shape / Type | Description |
|-------|-------------|-------------|
| `symm_input_buf` | `[world_size, M_local, K]` symmetric | Input shards. Each rank writes into `[rank, :, :]`. |
| `symm_ag_a_buf` | `[M_local * world_size, K]` symmetric | Compact gathered output. |
| `progress` | `[world_size * splits_per_rank]` uint32, **local** | Progress/signal tensor. **No symmetric memory needed. Must be `uint32`.** |
| `peer_symm_input_bufs` | `List[Tensor]` | `input_hdl.get_buffer(i, ...)` views. |
| `ag_stream` | `torch.cuda.Stream` | Dedicated stream for CE copies. |
| `input_hdl` | handle | `rendezvous(symm_input_buf)`. |
| `ag_hdl` | handle | `rendezvous(symm_ag_a_buf)`. |
| `splits_per_rank` | `int` | Number of chunk splits per rank (≥1). Higher = finer overlap. |

The `progress` tensor is a regular local device tensor — it does not need to be in
symmetric memory because it is only written by the local CE (via `stream_write_value32`)
and read by the local compute kernel (via `ld_sys`).

### Step 2: CE Communication Helper

```python
from torch._C._distributed_c10d import _SymmetricMemory
def cp_engine_full_mesh_pull_ag(
    rank: int,
    world_size: int,
    M_local: int,
    K: int,
    symm_input: torch.Tensor,              # this rank's [M_local, K] shard (view into symm_input_buf[rank])
    symm_ag_output: torch.Tensor,          # local [M_local * world_size, K] gathered buffer
    peer_symm_input_bufs: List[torch.Tensor],  # peer views: peer_symm_input_bufs[i] is rank i's symm_input_buf
    ag_signal: torch.Tensor,               # local ag_signal [world_size] int32
):
    """
    AllGather via Copy Engine (full-mesh pull).
    For each peer rank src_rank in rotated order:
      - PtoP data copy from peer_symm_input_bufs[src_rank][src_rank, :, :]
        into symm_ag_output[src_rank * M_local : (src_rank+1) * M_local, :]
      - Signal write via _SymmetricMemory.stream_write_value32
    Signal writes use cuStreamWriteValue32, which issues a system-scope fence
    before the write. This ensures all prior data copies are visible to all GPUs
    before the signal value becomes visible, correctly pairing with ld.sys in
    the consumer kernel.
    NOTE: caller must wrap this call in `with torch.cuda.stream(ag_stream):`
    so that all enqueued copy_ ops land on ag_stream.
    """
    # Self-shard: local copy + signal via stream_write_value32
    local_dst = symm_ag_output[rank * M_local : (rank + 1) * M_local, :]
    local_dst.copy_(symm_input)
    _SymmetricMemory.stream_write_value32(ag_signal, rank, 1)
    # Remote shards in rotated order
    for offset in range(1, world_size):
        src_rank = (rank + offset) % world_size
        remote_src = peer_symm_input_bufs[src_rank][src_rank, :M_local, :]
        local_dst = symm_ag_output[src_rank * M_local : (src_rank + 1) * M_local, :]
        local_dst.copy_(remote_src)
        # cuStreamWriteValue32 issues a system level fence before the write
        _SymmetricMemory.stream_write_value32(ag_signal, src_rank, 1)
```

**Key details:**
- `copy_()` from a peer symmetric tensor routes through the **Memcpy PtoP** CE channel
  (DMA, zero SM cost).
- Rotated iteration order `(rank + offset) % world_size` spreads NVLink traffic across
  links, reducing hot-spots.
- Signal granularity: controlled by `splits_per_rank`. Signal tensor shape is always
  `[world_size * splits_per_rank]`. `splits_per_rank=1` gives one signal per rank;
  `splits_per_rank>1` splits each rank's shard into finer chunks for earlier compute
  overlap.
- **Signal tensor dtype**: **must** use `uint32` for `progress` — `cuStreamWriteValue32`
  writes an unsigned 32-bit value and PyTorch's binding validates the dtype; `int32`
  raises `RuntimeError`. The compute kernel's `ld_sys` (PTX `ld.global.acquire.sys.b32`)
  reads 32 bits regardless of signedness, so it works with `uint32`.

### Step 3: Compute Kernel with Signal Polling

The compute kernel polls the signal tensor before touching remote data. Signal granularity
is controlled by `splits_per_rank`: `=1` means one signal per rank (per-rank), `>1` means
one signal per chunk within each rank (per-chunk). Local data (this rank's own shard) needs
no waiting — the signal is already set to 1 for all local chunks.

**Critical**: Only **one thread** (threadIdx.x == 0) polls the signal via `ld_sys`. After
the poll exits, `__syncthreads()` ensures all threads in the CTA see the signal before
proceeding. If all threads poll independently, each thread issues its own
`ld.global.acquire.sys` — wasting memory bandwidth and adding unnecessary synchronization
traffic.

```python
@triton.jit
def consumer_gemm_kernel(
    A_ptr,              # symm_ag_a_buf: [M, K] gathered A (compact layout)
    B_ptr,              # weight [N, K]
    C_ptr,              # output [M, N]
    signal_ptr,         # signal tensor [world_size * splits_per_rank] — uint32
    M, N, K,
    M_local,            # rows per rank
    M_split,            # rows per split = M_local // splits_per_rank
    M_split_tiles,      # = cdiv(M_split, BLOCK_SIZE_M)
    splits_per_rank,    # number of splits per rank (1 = per-rank, >1 = per-chunk)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # Chunk-aware tile rotation: local chunks first, then peers in rotated order.
    total_chunks = world_size * splits_per_rank
    num_pid_m = M_split_tiles * total_chunks
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Map from logical tile to (src_rank, split_id)
    chunk_id = min(pid_m // M_split_tiles, total_chunks - 1)
    tile_in_chunk = pid_m - chunk_id * M_split_tiles
    logical_rank = chunk_id // splits_per_rank
    split_id = chunk_id % splits_per_rank
    src_rank = (rank + logical_rank) % world_size
    pid_m = (src_rank * splits_per_rank + split_id) * M_split_tiles + tile_in_chunk

    # Poll signal for this chunk (system-scope acquire load).
    # Only tid(0) polls; __syncthreads() broadcasts the result.
    signal_idx = src_rank * splits_per_rank + split_id
    if tid(0) == 0:
        while ld_sys(signal_ptr + signal_idx) != 1:
            pass
    __syncthreads()

    # Compute row offset
    tile_row_start = src_rank * M_local + split_id * M_split + tile_in_chunk * BLOCK_SIZE_M

    # ... standard GEMM tile computation using A[tile_row_start:, :] x B ...
```

**Key points:**
- **`splits_per_rank=1`** is the per-rank case: `M_split = M_local`, `total_chunks = world_size`,
  each CTA polls `signal_ptr + src_rank`. This is the simplest default.
- **`splits_per_rank>1`** is the per-chunk case: each rank's shard is split into finer chunks,
  allowing the compute kernel to start processing earlier as each chunk's copy completes.
- Signal poll uses `ld_sys(...)` = `ld.global.acquire.sys.b32` — not `tl.load(...)`.
  System-scope acquire is required because the signal was written by the CE
  (`cuStreamWriteValue32`); plain `tl.load` has no cross-device visibility guarantee
  and may read stale L2 cache data.
- Only `tid(0) == 0` polls to reduce `ld.global.acquire.sys` traffic to 1-per-CTA.
- Chunk-aware rotation ensures CTAs for the local shard process first (signal
  already set), then peers in the same rotated order as the CE AG helper — minimizing
  spin-wait time.

### Step 4: Host-Side Orchestration

```python
def ag_gemm_overlap(ctx, a_input, compute_kernel, block_size):
    """Launch copy-engine AG and consumer compute kernel with overlap."""
    rank, world_size = ctx.rank, ctx.num_ranks
    M_local, K = a_input.shape[0], a_input.shape[1]
    N = ctx.weight.shape[0]
    M = M_local * world_size

    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = block_size

    # Copy input shard into symmetric buffer (so peers can pull it)
    symm_input = ctx.get_input_buf(M_local, K)
    symm_input.copy_(a_input)

    current_stream = torch.cuda.current_stream()
    ag_stream = ctx.ag_stream

    # ---- Barrier + signal reset pattern ----
    # Must happen on the default stream BEFORE enqueuing anything on ag_stream.
    ctx.input_hdl.barrier()          # ensure all ranks finished previous iteration
    ctx.progress.fill_(0)
    ctx.input_hdl.barrier()          # ensure all ranks see the reset before launching

    ag_stream.wait_stream(current_stream)

    # Step 1: enqueue full-mesh pull AG on ag_stream
    with torch.cuda.stream(ag_stream):
        cp_engine_full_mesh_pull_ag(
            rank=rank, world_size=world_size,
            M_local=M_local, K=K,
            splits_per_rank=ctx.splits_per_rank,
            symm_input=symm_input,
            output=ctx.symm_ag_a_buf,
            peer_symm_input_bufs=ctx.peer_symm_input_bufs,
            progress=ctx.progress,
            hdl=ctx.ag_hdl,
            ag_stream=ag_stream,
        )

    # Step 2: launch consumer compute kernel on current_stream
    # Without-SM: communication is on copy engine (zero SMs), compute uses ALL SMs.
    # Non-persistent kernel (one CTA per tile), so num_tiles > SMs is safe.
    M_local_tiles = triton.cdiv(M_local, BLOCK_SIZE_M)
    num_tiles = M_local_tiles * world_size * triton.cdiv(N, BLOCK_SIZE_N)

    c = torch.empty((M, N), dtype=a_input.dtype, device=a_input.device)
    compute_kernel[(num_tiles,)](
        ctx.symm_ag_a_buf, ctx.weight, c,
        ctx.progress,
        M, N, K, M_local, M_local_tiles,
        # ... strides, block sizes, etc.
        rank=rank, world_size=world_size,
        num_warps=32, num_stages=1,
    )

    current_stream.wait_stream(ag_stream)
    return ctx.symm_ag_a_buf[:M_local * world_size, :], c
```

---

## Path B: Compute → CE (Comp-First)

Example: GEMM (compute) → ReduceScatter (CE). The compute kernel writes per-tile signals
as tiles complete; the CE stream enqueues `cuStreamWaitValue32` + `copy_()` per tile to
pipeline copies behind computation.

### Anti-patterns

❌ **Host-side CPU spin-wait** — each `.item()` call triggers GPU→CPU sync (~5–50 μs),
and `pass` in the loop burns a CPU core at 100%:
```python
# ❌ BAD: CPU spin-wait — D2H latency × num_tiles
for tile_id in range(num_tiles):
    while signal_pad[tile_id].item() != 1:
        pass
    issue_ce_copy(tile_id)
```

❌ **Chunked kernel launch + CUDA event** — per-chunk kernel launch overhead is
prohibitively expensive for fine-grained tile-level overlap. The whole point of
without-sm is a single compute kernel launch with CE pipelining behind it.

### Recommended: GPU-side signal + `cuStreamWaitValue32`

Single compute kernel launch. Each CTA writes a per-tile signal via `st_sys`
(after the tile's `tl.store` completes). The CE stream enqueues
`cuStreamWaitValue32` per tile — a **GPU-side hardware wait** that blocks the CE
channel until the memory-mapped signal equals the expected value. The host call
returns immediately; no CPU spinning, no extra kernel launches.

#### Step A: Compute kernel — write signal per tile

```python
@triton.jit
def compute_kernel_with_push_signal(
    # ... existing args (input, output, weight pointers, dimensions, strides, etc.) ...
    signal_pad_ptr,    # [num_tiles] int32 — one slot per tile
    # ... dimensions, block sizes, etc. ...
):
    pid = tl.program_id(0)

    # ... compute tile, tl.store results to symm_output ...

    # After all tile stores are done, signal this tile's completion.
    # st_sys = st.global.release.sys.b32 — ensures tile data is visible before signal.
    # Only one thread per CTA needs to write the signal.
    if tid(0) == 0:
        st_sys(signal_pad_ptr + pid, 1)
```

#### Step B: CE push — `cuStreamWaitValue32` per tile

```python
from typing import List
from cuda.bindings import driver

def cp_engine_full_mesh_push(
    rank: int,
    world_size: int,
    M_local: int,
    K: int,
    symm_output: torch.Tensor,                 # this rank's computed [M_local, K] in symmetric memory
    peer_symm_output_bufs: List[torch.Tensor],  # peer views of symm_output
    signal_pad: torch.Tensor,                  # [num_tiles] int32 — compute kernel writes 1 per tile
    ce_stream: torch.cuda.Stream,              # dedicated stream for CE copies
):
    """
    Push computed tiles to all peers via Copy Engine.

    The compute kernel (running on current_stream) writes signal_pad[tile_id] = 1
    via st_sys after each tile's stores complete. This function enqueues
    per-tile cuStreamWaitValue32 + copy_() on ce_stream so the CE channel
    pipelines naturally behind the compute kernel — zero CPU spin, zero extra
    kernel launches.
    """
    num_tiles = signal_pad.numel()
    tile_numel = (M_local * K) // num_tiles
    assert (M_local * K) % num_tiles == 0, "Output must be evenly divisible by num_tiles"
    ce_stream_handle = driver.CUstream(ce_stream.cuda_stream)

    with torch.cuda.stream(ce_stream):
        for tile_id in range(num_tiles):
            # GPU-side hardware wait: CE channel stalls until signal_pad[tile_id] == 1.
            # Host call returns immediately — no CPU blocking.
            wait_addr = driver.CUdeviceptr(signal_pad[tile_id].data_ptr())
            driver.cuStreamWaitValue32(
                ce_stream_handle,
                wait_addr,
                1,
                driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
            )

            # PtoP push: copy this tile to every peer's symmetric buffer
            tile_offset = tile_id * tile_numel
            local_tile = symm_output.flatten()[tile_offset : tile_offset + tile_numel]

            for offset in range(1, world_size):
                dst_rank = (rank + offset) % world_size
                remote_dst = peer_symm_output_bufs[dst_rank].flatten()[
                    tile_offset : tile_offset + tile_numel
                ]
                remote_dst.copy_(local_tile)  # PtoP CE push

    # Cleanup: current stream waits for all CE copies to finish
    current_stream = torch.cuda.current_stream()
    current_stream.wait_stream(ce_stream)
```

**End-to-end timeline:**

```
current_stream (SM)                ce_stream (Copy Engine)
─────────────────────              ─────────────────────
kernel launches →
  CTA-0: compute tile 0
         st_sys(pad[0], 1)  ───►  cuStreamWaitValue32(pad[0], ==1)  ← fires!
                                      copy_(tile_0 → peer_1, peer_2, …)
  CTA-1: compute tile 1
         st_sys(pad[1], 1)  ───►  cuStreamWaitValue32(pad[1], ==1)  ← fires!
                                      copy_(tile_1 → peer_1, peer_2, …)
  ...                               ...
kernel exits                       all copies complete
```

The CE channel pipelines naturally: `cuStreamWaitValue32` is a hardware wait that gates
the subsequent `copy_()` on the CE stream. The host thread enqueues all `num_tiles`
wait+copy pairs up front (returns immediately), and the GPU scheduler progresses each
tile's copy as soon as the compute kernel's signal lands in memory.

**Key details:**
- `st_sys` uses `st.global.release.sys.b32` — system-scope release ensures data stores
  are visible before the signal.
- `cuStreamWaitValue32` is a **GPU-side hardware wait** — the CE stream's DMAs are gated
  by the memory-mapped value. No CPU involvement, no D2H transfers.
- Signal granularity is **per-tile** (fine-grained), giving the CE stream maximum overlap
  opportunity — each tile's copy starts as soon as that tile's signal fires.
- `copy_()` into a peer symmetric tensor routes through the **Memcpy PtoP** CE channel.
- Reset `signal_pad.zero_()` between iterations (protected by the
  `barrier() → zero_() → barrier()` pattern). If the reset is missing, the CE will
  see stale `1` values and issue copies before the compute kernel has finished — a
  correctness bug.

#### Step C: Host-Side Orchestration (Comp-First)

```python
def gemm_rs_overlap(ctx, a_input, b_weight, compute_kernel, signal_pad, block_size):
    """Launch compute kernel (GEMM) and CE push (reduce-scatter) with overlap."""
    rank, world_size = ctx.rank, ctx.num_ranks
    M_local, K = a_input.shape
    N = b_weight.shape[0]

    current_stream = torch.cuda.current_stream()
    ce_stream = ctx.ce_stream

    # Barrier + signal reset
    ctx.output_hdl.barrier()
    signal_pad.zero_()
    ctx.output_hdl.barrier()

    # Step 1: launch compute kernel on current_stream (writes per-tile signals)
    num_tiles = triton.cdiv(M_local, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N)
    symm_output = ctx.get_output_buf(M_local, K)

    compute_kernel[(num_tiles,)](
        a_input, b_weight, symm_output, signal_pad,
        M_local, N, K,
        # ... strides, block sizes, etc.
        num_warps=32, num_stages=1,
    )

    # Step 2: enqueue CE push on ce_stream (per-tile wait + copy)
    ce_stream.wait_stream(current_stream)
    cp_engine_full_mesh_push(
        rank=rank, world_size=world_size,
        M_local=M_local, K=K,
        symm_output=symm_output,
        peer_symm_output_bufs=ctx.peer_symm_output_bufs,
        signal_pad=signal_pad,
        ce_stream=ce_stream,
    )

    current_stream.wait_stream(ce_stream)
```

---

## Host-Side Cross-Rank Barrier Pattern

Both paths require host-side `symm_mem_hdl.barrier()` calls to synchronize across ranks:

1. **Before launching**: `barrier() → signal_reset → barrier()` ensures all ranks have
   finished the previous iteration before resetting signals, and all ranks see the reset
   before launching new kernels.

2. **After completion**: `current_stream.wait_stream(ce_stream)` ensures the host thread
   does not return until both streams complete.

```python
# Reset pattern (before each iteration):
hdl.barrier()              # all ranks finished previous iteration
progress.fill_(0)          # or signal_pad.zero_() for comp-first
hdl.barrier()              # all ranks see reset before launching

# Sync pattern (after each iteration):
current_stream.wait_stream(ce_stream)   # wait for CE to finish
```

**Why not in-kernel barriers?** In without-sm mode, the compute kernel and CE stream
run as separate entities with no shared in-kernel phase after communication. The
barrier needs to happen on the host side between iterations. Using in-kernel
`barrier_all_intra_node_atomic_cas_block` is incorrect here — there is no single
kernel that encompasses both compute and communication. Use host-side
`symm_mem_hdl.barrier()` instead (see `primitives.md` §6.4).