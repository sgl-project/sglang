# Without-SM Overlap: Copy Engine (DMA) Based

Communication is offloaded to the copy engine, consuming zero SM resources. The compute
kernel on the main stream polls a signal tensor to process data as it arrives.

## Architecture

```
Main Stream (SM)                      Backend Stream (Copy Engine / DMA)
┌──────────────────────────┐          ┌──────────────────────────────────────┐
│ compute_kernel           │          │ for each peer in rotated order:      │
│  process local (no wait) │          │   copy_(peer_symm_buf → local_dst)   │
│  poll signal[src_rank]   │◄──────── │   signal[src_rank].copy_(peer_one)   │
│  process remote chunk    │          │     (PtoP pull — same CE channel)    │
└──────────────────────────┘          └──────────────────────────────────────┘
```

## Step 1: Buffers Required (ctx)

The orchestration function (Step 4) assumes a pre-initialized context `ctx` with:

| Field | Shape / Type | Description |
|-------|-------------|-------------|
| `symm_input_buf` | `[world_size, M_local, K]` symmetric | Allocated via `symm_mem.empty()` + `rendezvous()`. Each rank writes its shard into `[rank, :, :]`. |
| `symm_ag_output` | `[M_local * world_size, K]` local | Compact gathered output buffer. |
| `ag_signal` | `[world_size]` int32, local | Per-rank-shard signal (consumer polls for `== 1`). |
| `peer_symm_input_bufs` | `List[Tensor]` (len = world_size) | `hdl.get_buffer(i, ...)` views into each peer's `symm_input_buf`. |
| `peer_signal_one_bufs` | `List[Tensor]` (len = world_size) | Views into a **symmetric all-ones int32 buffer** — used as PtoP source for signal writes. Allocate a separate symmetric buffer filled with `1`, rendezvous it, then `get_buffer(i, ...)` for each peer. |
| `ag_stream` | `torch.cuda.Stream` | Dedicated stream for CE copies. |
| `symm_mem_hdl` | handle | The symmetric memory handle from `rendezvous(symm_input_buf)`. |

**Why `peer_signal_one_bufs`?** Signal writes must stay on the same Memcpy PtoP CE
channel as data copies. Pulling a pre-filled `1` from a peer's symmetric buffer
achieves this — using `fill_()` or `stream_write_value32` would route through a
different channel (DtoD) and introduce a serialization point that breaks overlap.

## Step 2: Copy Engine Communication Helper (Full-Mesh Pull)

Enqueues PtoP (Peer-to-Peer) DMA copies on the **current stream** (caller is responsible
for stream context). Both data and signal writes are sourced from **peer symmetric
memory**, keeping all operations on the same CE (Memcpy PtoP) channel. This avoids the
DtoD serialization point that `fill_()` or local `copy_()` would introduce — which
empirically prevents overlap with the consumer compute kernel.

```python
def cp_engine_full_mesh_pull_ag(
    rank: int,
    world_size: int,
    M_local: int,
    K: int,
    symm_input: torch.Tensor,              # this rank's [M_local, K] shard (view into symm_input_buf[rank])
    symm_ag_output: torch.Tensor,          # local [M_local * world_size, K] gathered buffer
    peer_symm_input_bufs: List[torch.Tensor],  # peer views: peer_symm_input_bufs[i] is rank i's symm_input_buf
    ag_signal: torch.Tensor,               # local ag_signal [world_size] int32
    peer_signal_one_bufs: List[torch.Tensor],  # peer views of symmetric all-ones int32 [world_size]
):
    """
    AllGather via Copy Engine (full-mesh pull).

    For each peer rank src_rank in rotated order:
      - PtoP data copy from peer_symm_input_bufs[src_rank][src_rank, :, :]
        into symm_ag_output[src_rank * M_local : (src_rank+1) * M_local, :]
      - PtoP signal pull of 4B from peer_signal_one_bufs[src_rank]

    CRITICAL: Both data and signal writes are routed through PtoP (Memcpy PtoP
    channel) by sourcing them from peer symmetric memory. This keeps all ops
    on the same CE channel and avoids DtoD serialization.

    NOTE: caller must wrap this call in `with torch.cuda.stream(ag_stream):`
    so that all enqueued copy_ ops land on ag_stream.
    """
    # Self-shard: data is already local, just signal via PtoP pull.
    local_dst = symm_ag_output[rank * M_local : (rank + 1) * M_local, :]
    local_dst.copy_(symm_input)
    ag_signal[rank:rank + 1].copy_(peer_signal_one_bufs[rank][rank:rank + 1])

    # Pull remote shards in rotated order: (rank+1, rank+2, ..., rank-1) % world_size
    # Rotated order spreads NVLink traffic: not all ranks pull from rank 0 first.
    for offset in range(1, world_size):
        src_rank = (rank + offset) % world_size
        # Remote source: rank src_rank's own shard at peer's symm_input_buf[src_rank, :, :]
        remote_src = peer_symm_input_bufs[src_rank][src_rank, :M_local, :]
        local_dst = symm_ag_output[src_rank * M_local : (src_rank + 1) * M_local, :]
        local_dst.copy_(remote_src)  # PtoP CE data copy

        # Signal: PtoP pull 4B from peer's signal_one_buf — same channel as data.
        ag_signal[src_rank:src_rank + 1].copy_(
            peer_signal_one_bufs[src_rank][src_rank:src_rank + 1]
        )
```

**Key details:**
- `copy_()` from a peer symmetric tensor routes through the **Memcpy PtoP** CE channel
  (DMA, zero SM cost)
- Signal is written by pulling a pre-filled `1` from the peer's symmetric `signal_one_buf`
  — this stays on the same PtoP channel as data, preserving ordering without explicit
  fences. The consumer kernel sees `ag_signal[src_rank] == 1` only after the preceding
  data copy completes on the same channel.
- Rotated iteration order `(rank + offset) % world_size` spreads NVLink traffic across
  links, reducing hot-spots
- Signal granularity is **per-rank-shard** (one slot per peer), not per-chunk. The
  consumer kernel polls `ag_signal[src_rank]` before processing that rank's rows.

## Step 3: Compute Kernel with Signal Polling

The compute kernel polls the **per-rank-shard** signal tensor before touching remote
data. Signal granularity matches the AG helper: one slot per peer rank. Local data
(this rank's own shard) needs no waiting.

The example below shows a non-persistent GEMM consumer (one CTA per output tile) with
rank-aware tile rotation so that tiles targeting the local shard launch first (signaled
immediately), then tiles for peers in rotated order — matching the AG signal-arrival
order and minimizing spin-wait.

```python
@triton.jit
def consumer_gemm_compute_kernel(
    A_ptr,              # symm_ag_output: [M, K] gathered A (compact layout)
    B_ptr,              # weight [N, K]
    C_ptr,              # output [M, N]
    ag_signal_ptr,      # signal from AG [world_size] int32 (per-rank-shard)
    M, N, K,
    M_local,            # rows per rank
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    M_per_rank_tiles: tl.constexpr,  # = cdiv(M_local, BLOCK_SIZE_M)
    rank: tl.constexpr,
    world_size: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Rank-aware tile rotation: remap pid_m so local tiles come first
    # tile_in_rank = pid_m % M_per_rank_tiles
    # rank_offset = pid_m // M_per_rank_tiles
    # src_rank = (rank + rank_offset) % world_size
    tile_in_rank = pid_m % M_per_rank_tiles
    rank_offset = pid_m // M_per_rank_tiles
    src_rank = (rank + rank_offset) % world_size

    # Poll AG signal for this rank's shard (skip for local)
    if rank_offset != 0:
        while tl.load(ag_signal_ptr + src_rank, volatile=True) == 0:
            pass  # spin-wait until src_rank's shard is gathered

    # Compute row offset in compact layout
    row_start = src_rank * M_local + tile_in_rank * BLOCK_SIZE_M

    # ... standard GEMM tile computation using A[row_start:, :] x B ...
```

**Key points:**
- Signal poll is `ag_signal_ptr + src_rank` (per-rank-shard granularity)
- Rank-aware rotation ensures CTAs for the local shard (rank_offset == 0) skip the poll
  entirely, and subsequent CTAs target peers in the same rotated order as the AG helper
- For a simpler (non-GEMM) persistent kernel, the same pattern applies: determine which
  rank owns the chunk, poll `ag_signal[src_rank]`, then process

## Step 4: Host-Side Orchestration

Launch both communication and compute in a single host function. The
`barrier() → signal.fill_(0) → barrier()` pattern ensures all ranks have finished
the previous iteration's communication before resetting, and all ranks see the reset
before launching the next iteration.

```python
def ag_gemm_overlap(ctx, a_input, compute_kernel):
    """Launch copy-engine AG and consumer compute kernel with overlap."""
    rank, world_size = ctx.rank, ctx.num_ranks
    hdl = ctx.symm_mem_hdl
    M_local, K = a_input.shape[0], a_input.shape[1]

    symm_input = ctx.symm_input_buf          # symmetric [world_size, M_local, K]
    symm_ag_output = ctx.symm_ag_output      # local [M_local * world_size, K]
    ag_signal = ctx.ag_signal                 # local [world_size] int32
    peer_symm_input_bufs = ctx.peer_symm_input_bufs
    peer_signal_one_bufs = ctx.peer_signal_one_bufs

    current_stream = torch.cuda.current_stream()
    ag_stream = ctx.ag_stream

    # Copy input shard into symmetric buffer (so peers can pull it)
    symm_input[rank, :M_local, :].copy_(a_input)

    # Host-side cross-rank barrier: ensure all ranks finished previous iteration
    hdl.barrier()
    # Reset signal (safe now — all ranks' CE copies have completed)
    ag_signal.fill_(0)
    # Host-side barrier: ensure all ranks see the reset before launching
    hdl.barrier()

    # Ensure ag_stream waits for signal reset and input copy
    ag_stream.wait_stream(current_stream)

    # Step 1: enqueue full-mesh pull AG on ag_stream
    with torch.cuda.stream(ag_stream):
        cp_engine_full_mesh_pull_ag(
            rank=rank,
            world_size=world_size,
            M_local=M_local,
            K=K,
            symm_input=symm_input[rank, :M_local, :],
            symm_ag_output=symm_ag_output,
            peer_symm_input_bufs=peer_symm_input_bufs,
            ag_signal=ag_signal,
            peer_signal_one_bufs=peer_signal_one_bufs,
        )

    # Step 2: launch consumer compute kernel on current_stream (polls ag_signal)
    M = M_local * world_size
    N = ctx.N
    M_per_rank_tiles = triton.cdiv(M_local, BLOCK_SIZE_M)
    num_tiles = M_per_rank_tiles * world_size * triton.cdiv(N, BLOCK_SIZE_N)

    c = torch.empty((M, N), dtype=ctx.output_dtype, device=a_input.device)
    compute_kernel[(num_tiles,)](
        symm_ag_output, ctx.weight, c, ag_signal,
        M, N, K, M_local,
        # ... strides, block sizes, etc.
        M_per_rank_tiles=M_per_rank_tiles,
        rank=rank,
        world_size=world_size,
    )

    # Wait for AG stream to complete before returning
    current_stream.wait_stream(ag_stream)
    return c
```

## Signal Variant: Compute-First

If compute runs first and communication follows (e.g., GEMM → reduce-scatter / all-gather
via DMA), the direction reverses: the compute kernel signals tile completion, and the
CE stream waits for each signal before issuing the corresponding `copy_()`.

```python
from typing import List

def cp_engine_full_mesh_push(
    rank: int,
    world_size: int,
    M_local: int,
    K: int,
    symm_output: torch.Tensor,             # this rank's computed [M_local, K] in symmetric memory
    peer_symm_output_bufs: List[torch.Tensor],  # peer views of symm_output
    signal_pad: torch.Tensor,              # [num_tiles] int32 — compute kernel writes 1 per tile
    ce_stream: torch.cuda.Stream,          # the backend CE stream (caller sets stream context)
):
    """
    Push computed tiles to all peers via Copy Engine, waiting for each tile's
    compute signal before issuing the copy.

    Host-side while loop polls signal_pad until the compute kernel writes 1,
    then issues PtoP push via copy_(). No external dependencies required.

    NOTE: caller must wrap this call in `with torch.cuda.stream(ce_stream):`
    """
    num_tiles = signal_pad.numel()
    tile_numel = (M_local * K) // num_tiles  # elements per tile

    for tile_id in range(num_tiles):
        # Host-side poll: wait for compute kernel to signal this tile
        while signal_pad[tile_id].item() != 1:
            pass

        # PtoP push: copy this tile to every peer's symmetric buffer
        tile_offset = tile_id * tile_numel
        local_tile = symm_output.flatten()[tile_offset : tile_offset + tile_numel]

        for offset in range(1, world_size):
            dst_rank = (rank + offset) % world_size
            remote_dst = peer_symm_output_bufs[dst_rank].flatten()[tile_offset : tile_offset + tile_numel]
            remote_dst.copy_(local_tile)  # PtoP CE push
```

**Key details:**
- Host-side `while signal_pad[tile_id].item() != 1: pass` polls the compute kernel's
  per-tile signal
- `copy_()` into a peer symmetric tensor routes through the **Memcpy PtoP** CE channel,
  same as the pull direction in `cp_engine_full_mesh_pull_ag`
- Reset `signal_pad.zero_()` between iterations (protected by the same
  `barrier() → zero_() → barrier()` pattern as the pull variant)