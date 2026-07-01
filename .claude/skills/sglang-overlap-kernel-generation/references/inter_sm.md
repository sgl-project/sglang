# Inter-SM Overlap: Two Kernels on Separate Streams

Compute and communication run as separate kernels on two CUDA streams, synchronized via
per-tile signals for fine-grained overlap.

> **Primitives**: all PTX helpers (`_send_signal`, `_wait_signal`, `_get_flat_tid`, etc.)
> are defined in `references/primitives.md`. Read that file first for the canonical
> implementations.

## Architecture

```
Stream A (compute)          Stream B (comm)
┌──────────────────┐        ┌──────────────────────────┐
│ compute_kernel   │        │ persistent_comm_kernel   │
│  tile 0 → signal │──────▶ │  poll signal → comm tile │
│  tile 1 → signal │──────▶ │  poll signal → comm tile │
│  ...             │        │  ...                     │
└──────────────────┘        └──────────────────────────┘
                            ↓ wait_stream
                            symm_mem_hdl.barrier()  (host-side rank sync)
```

## Step 1: Add Signals to Compute Kernel

After each CTA finishes computing a tile, it writes a signal to indicate the output data
is ready. Use `_send_signal` (see `primitives.md` §6) with `"release"` semantics so that
preceding data stores are visible to the consumer.

### Signal Insertion Pattern

In the compute kernel, after writing the output tile:

```python
@triton.jit
def compute_kernel_with_signal(out_ptr, signal_ptr, ...):
    pid = tl.program_id(0)
    # ... compute tile ...
    tl.store(out_ptr + offsets, result, mask=mask)
    # Signal this tile is ready (only thread 0)
    flat_tid = _get_flat_tid()
    if flat_tid == 0:
        _send_signal(signal_ptr + pid, "release")
```

## Step 2: Implement Persistent Communication Kernel

The comm kernel is **persistent** — it stays alive and polls signals, processing tiles
as they become available. It uses `symm_mem` pointer arrays for peer buffer access.

```python
@triton.jit
def persistent_comm_kernel(
    data_ptr, peer_ptrs, signal_ptr,
    tiles_per_rank, rank: tl.constexpr, world_size: tl.constexpr, ...
):
    pid = tl.program_id(0)

    # Each CTA handles one or more tiles
    for tile_id in range(pid, total_tiles, tl.num_programs(0)):
        # Wait for compute to finish this tile
        flat_tid = _get_flat_tid()
        if flat_tid == 0:
            _wait_signal(signal_ptr + tile_id, "acquire")
        tl.debug_barrier()  # broadcast readiness to all threads in CTA

        # Load tile data from local output
        tile_data = tl.load(data_ptr + tile_offsets)

        # Push to peer(s) via symmetric memory
        for peer in range(world_size):
            if peer != rank:
                peer_buf = tl.load(peer_ptrs + peer).to(tl.pointer_type(out_dtype))
                tl.store(peer_buf + dest_offsets, tile_data)

    # No in-kernel cross-rank barrier needed — host-side barrier handles it
    tl.debug_barrier()
```

**Key points about the persistent kernel:**
- The `for tile_id in range(pid, total_tiles, tl.num_programs(0))` loop distributes tiles
  across CTAs in a round-robin fashion. The loop terminates when all tiles are processed —
  this is the bounded exit condition that prevents hangs.
- `_wait_signal` + `tl.debug_barrier()` ensures all threads in the CTA see the data before
  loading it.
- No in-kernel cross-rank barrier is needed. Since the comm kernel has no subsequent phase
  after communication, the host-side `symm_mem_hdl.barrier()` (around `signal.zero_()`) provides
  the same rank-level synchronization with less kernel complexity.

## Step 3: Host-Side Launch

Use a two-stream pattern where compute runs on the **current stream** and
communication runs on a **separate stream**. This avoids creating unnecessary
extra streams and follows standard PyTorch stream management conventions.

```python
import torch
import torch.distributed._symmetric_memory as symm_mem

def launch_inter_sm_overlap(ctx, compute_fn, comm_fn, data, ...):
    # --- Stream setup ---
    # Compute on current stream; comm on a separate stream (from ctx).
    current_stream = torch.cuda.current_stream()
    comm_stream = ctx.comm_stream

    # Host-side barrier: ensure all ranks finished previous iteration
    ctx.symm_mem_hdl.barrier()
    # Reset signal tensor (safe now — all ranks' comm kernels have exited)
    ctx.signal.zero_()
    # Host-side barrier: ensure all ranks see the reset before launching
    ctx.symm_mem_hdl.barrier()

    # Ensure comm_stream sees signal reset
    comm_stream.wait_stream(current_stream)

    # Launch compute kernel on current_stream (producer of signals).
    # Compute kernel uses ctx.num_comp_sms (= total SMs - num_comm_sms).
    # This leaves num_comm_sms SMs free for the concurrent comm kernel.
    compute_grid = (ctx.num_comp_sms,)
    compute_fn[compute_grid](data, ctx.symm_buffer, ctx.signal, ...)

    # Launch persistent comm kernel on comm_stream.
    # Comm kernel uses ctx.num_comm_sms SMs — a small subset of total SMs.
    # This kernel is persistent: CTAs loop over all tiles, polling signals.
    # Grid size must not exceed num_comm_sms to avoid scheduling deadlock.
    with torch.cuda.stream(comm_stream):
        comm_fn[(ctx.num_comm_sms,)](
            ctx.symm_buffer,
            ctx.mc_ptr,
            ctx.signal,
            ...
        )

    # Wait for comm stream to finish (needed for correctness of next iteration's barrier)
    current_stream.wait_stream(comm_stream)
```

## Barrier Level Selection

Choose the barrier granularity based on the collective and data dependencies.
See `primitives.md` §5 for the full barrier hierarchy and implementations.

| Barrier level | When to use | Mechanism |
|---------------|------------|-----------|
| **Thread** | Independent per-element operations | `tl.debug_barrier()` within CTA |
| **CTA** | All CTAs on this device must finish | `barrier_on_this_grid()` (primitives.md §5.2) |
| **Rank (host-side)** | All ranks must finish, no in-kernel follow-up | `symm_mem_hdl.barrier()` around `signal.zero_()` |
| **Rank (in-kernel)** | All ranks must finish, kernel has a subsequent phase (e.g. reduce) | `barrier_all_intra_node_atomic_cas_block()` (primitives.md §5.3) |

**Inter-sm** and **without-sm** use host-side rank barrier: the comm kernel (or copy-engine)
completes communication and exits, then `barrier()` → `signal.zero_()` → `barrier()` ensures all ranks
have finished and see the reset before the next iteration launches. **Intra-sm** uses
in-kernel rank barrier: the same kernel must continue to a reduce/read phase after all
ranks push, so it cannot exit first.