# Shared Primitives Library

This file is the **single source of truth** for all low-level synchronization helpers used by `inter-sm`, `intra-sm`, and `without-sm` overlap kernels — thread/block ID helpers, PTX inline-asm atomics, signals, system-scope load/store, and the three-level barrier hierarchy.

Read this file whenever you are about to emit any of these helpers into a generated kernel, regardless of which mode you picked. The mode files (`inter_sm.md`, `intra_sm.md`) reference primitives by name and only describe *how* they are composed.

## Table of Contents

1. [Why factor primitives out](#1-why-factor-primitives-out)
2. [Thread/Block ID helpers](#2-threadblock-id-helpers)
3. [GPU-scope memory ordering atomics](#3-gpu-scope-memory-ordering-atomics)
4. [System-scope CAS (cross-rank)](#4-system-scope-cas-cross-rank)
5. [System-scope load/store (without-sm)](#5-system-scope-loadstore-without-sm)
6. [Three-level barrier hierarchy](#6-three-level-barrier-hierarchy)
   - 6.1 Thread-level (intra-CTA)
   - 6.2 CTA-Grid level: `barrier_on_this_grid`
   - 6.3 Rank level: `barrier_all_intra_node_atomic_cas_block`
7. [Per-tile signal pattern (inter-sm only)](#7-per-tile-signal-pattern-inter-sm-only)
8. [Memory order rules — when to use release / acquire / relaxed](#8-memory-order-rules)
9. [Which primitives each mode needs (quick reference)](#9-which-primitives-each-mode-needs)

---

## 1. Why factor primitives out

These helpers (store/load spin-loops, grid barriers, signal pads) are mode-agnostic: the same `barrier_on_this_grid` works in any fused kernel; CAS primitives back `barrier_all_intra_node_atomic_cas_block` (cross-rank sync), while `_send_signal`/`_wait_signal` use simple store/load spin (inter-sm). Defining them once here avoids signature drift and ensures the generated kernel uses the fastest known implementation of each primitive.

When emitting a kernel, include only the primitives that mode actually uses (see [section 8](#8-which-primitives-each-mode-needs)) — do not paste this whole catalog into the output file.

---

## 2. Thread/Block ID helpers

Both `_get_flat_tid` and `_get_flat_bid` are needed whenever a kernel must restrict an action to "one thread per CTA" (signals) or "one CTA per grid" (master CTA in grid barrier).

```python
@triton.jit
def _get_flat_tid():
    """Flattened thread index within the CTA: tid.z * ntid.y * ntid.x + tid.y * ntid.x + tid.x."""
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
```

### `tid`

Lightweight per-axis thread index accessor. Used in without-sm mode where only a single axis tid is needed (e.g., `tid(0) == 0` to gate signal polling to one thread).

```python
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
            pack=1,
        )
    elif axis == 1:
        return tl.inline_asm_elementwise(
            asm="mov.u32 $0, %tid.y;",
            constraints=("=r"),
            args=[],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
    else:
        return tl.inline_asm_elementwise(
            asm="mov.u32 $0, %tid.z;",
            constraints=("=r"),
            args=[],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
```

### `__syncthreads`

PTX `bar.sync` — block-level `__syncthreads`. Used in without-sm mode to broadcast a signal poll result from one thread to all threads in the CTA.

```python
@triton.jit
def __syncthreads():
    """PTX bar.sync (block-level __syncthreads)."""
    tl.inline_asm_elementwise(
        asm="bar.sync 0;",
        constraints="=r",
        args=[],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
```

**Signedness pitfall:** `_get_flat_tid()` returns `tl.uint32`. Triton forbids `//`, `%` between `uint32` and `int32`. When using `flat_tid` in arithmetic (e.g., `flat_tid // vec_per_row`), cast first: `flat_tid = _get_flat_tid().to(tl.int32)`. Same applies to `%ntid.x` if read via inline asm as a loop stride. Simple comparisons (`== 0`, `< N`) work without casting.

---

## 3. GPU-scope memory ordering atomics

Used to build the CTA-grid barrier. **GPU scope** is correct here because all participating CTAs run on the same device.

### `_atomic_add_release`

Used to count arrivals at a barrier point. The `release` semantics ensure all preceding stores are visible to other CTAs that observe the counter increment.

```python
@triton.jit
def _atomic_add_release(ptr, val):
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
```

### `_load_acquire`

Spin-wait on a barrier counter. The `acquire` semantics ensure that reads after this load see all stores that happened before the corresponding `release`.

```python
@triton.jit
def _load_acquire(ptr):
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
```

### `_store_release_with_highbit`

Master CTA broadcasts "release all waiters" by storing `expected | 0x80000000` (high-bit flag pattern).

```python
@triton.jit
def _store_release_with_highbit(ptr, expected):
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
```

---

## 4. System-scope CAS (cross-rank)

These back the **cross-rank barrier** (`barrier_all_intra_node_atomic_cas_block`). **System scope** is required because the target memory lives in another GPU's symmetric memory.

The spin-loop is kept **inside PTX** (`@!%p0 bra cas_loop`). A Python-level `while` around a single-shot CAS incurs significant Triton control-flow overhead and is measurably slower.

```python
@triton.jit
def _cas_sys_release(addrs, expected, desired):
    """atom.global.release.sys.cas.b32 with PTX-level spin-loop."""
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
    """atom.global.acquire.sys.cas.b32 with PTX-level spin-loop."""
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
```

**Anti-pattern** — do NOT use a Python-level spin-loop with a single-shot CAS:

```python
# BAD: Python-level spin-loop is slow
old = tl.cast(1, tl.uint32)
while old != 0:
    old = _cas_single_shot(peer_signal_addr, 0, 1)  # single CAS, no internal loop
```

---

## 5. System-scope load/store (without-sm)

These helpers are used by without-sm (copy engine) overlap kernels. The compute kernel needs `ld_sys` to poll signals written by the CE (cross-GPU PtoP copy or `cuStreamWriteValue32`), and `st_sys` to write per-tile completion signals that the CE waits on via `cuStreamWaitValue32`.

Plain `tl.load` / `tl.store` lack cross-device visibility guarantees — the L2 cache may return stale data when the writer is on a different GPU. System-scope acquire/release is required.

### `ld_sys`

```python
@triton.jit
def ld_sys(ptr):
    """ld.global.acquire.sys.b32 — system-scope acquire load.

    Required for reading signals written by the CE (cross-GPU PtoP copy or
    cuStreamWriteValue32). Plain tl.load does NOT guarantee visibility of
    writes that arrived over NVLink — the L2 cache may return stale data.
    """
    return tl.inline_asm_elementwise(
        asm="ld.global.acquire.sys.b32 $0, [$1];",
        constraints="=r,l",
        args=[ptr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
```

### `st_sys`

```python
@triton.jit
def st_sys(ptr, val):
    """st.global.release.sys.b32 — system-scope release store.

    Used by the compute kernel (comp-first path) to write a per-tile signal
    that the CE waits on via cuStreamWaitValue32. Release semantics ensure
    all prior tl.store (tile data) are visible before the signal write.
    """
    tl.inline_asm_elementwise(
        asm="""
        st.global.release.sys.b32 [$1], $2;
        mov.u32 $0, 0;
        """,
        constraints="=r,l,r",
        args=[ptr, val],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
```

**Memory ordering rationale:**
- `ld_sys` → `ld.global.acquire.sys`: the CE's `stream_write_value32` or PtoP `copy_()` is a release operation; the compute kernel's acquire load ensures it sees the data that was copied before the signal was written.
- `st_sys` → `st.global.release.sys`: the compute kernel's release store ensures all prior tile data stores are visible before the signal `1` becomes observable by the CE's `cuStreamWaitValue32`.

**Why `st_sys` instead of `_send_signal` (CAS)?** In without-sm comp-first mode, each tile has exactly one writer (the CTA that computed it), and the signal tensor is reset to 0 before each launch. So there is no contention — a simple store suffices. The CAS spin-loop in `_send_signal` is overkill and adds latency.

---

## 6. Three-level barrier hierarchy

A kernel that synchronizes "everyone" must say *which* "everyone": threads in this CTA, CTAs on this device, or ranks across devices. Pick the smallest scope that covers your data dependency.

### 6.1 Thread-level (intra-CTA)

Built-in. Use `tl.debug_barrier()` to re-converge threads after warp-divergent control flow, or to broadcast a value written by `if flat_tid == 0` to the rest of the CTA. No custom helper needed.

```python
if flat_tid == 0:
    _wait_signal(signal_ptr + tile_id, "acquire")
tl.debug_barrier()  # broadcast readiness to all threads in CTA
```

### 6.2 CTA-Grid level: `barrier_on_this_grid`

Synchronizes all CTAs in a single kernel launch **without** requiring `cooperative_groups`. Uses a split-counter pattern: each CTA atomically increments a global counter; the master CTA (`bid == 0`) spins until all arrive, then flips the high bit to release everyone.

**Performance note**: This implementation uses the *master CTA pattern* (only `bid==0` does the spin-wait, others just check the high bit) with a single `tl.debug_barrier()` at the end. Avoid the naive variant where every CTA checks if it's the last to arrive — that requires 3 separate `tl.debug_barrier()` calls and is significantly slower.

```python
@triton.jit
def barrier_on_this_grid(barrier_ptr):
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

    tl.debug_barrier()  # single sync point at the end
```

**Host-side requirement**: the counter must be reset to 0 between kernel launches (`ctx.grid_barrier.zero_()`).

**⚠️ CRITICAL: `num_ctas` must not exceed SM for persistent kernels.** This barrier requires ALL CTAs to arrive before any can proceed. If `num_ctas > SMs`, some CTAs cannot be scheduled (SMs are fully occupied by earlier CTAs that are already spinning at the barrier) and the kernel deadlocks. Always cap the grid size:

```python
num_sm = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
num_ctas = min(total_tiles, num_sm)  # conservative: assume 1 CTA/SM
# or query actual occupancy and use: min(total_tiles, num_sm * occupancy)
```

### 6.3 Rank level: `barrier_all_intra_node_atomic_cas_block`

The single canonical cross-rank barrier used by both `inter-sm` and `intra-sm` kernels. Each CTA independently participates: thread `i` (where `i < local_world_size`) signals peer `i` by CAS(0→1) on the peer's barrier slot for our rank, then waits for the peer to signal us back by CAS(1→0) on our own slot.

The barrier is **self-resetting** (CAS 1→0 in phase 2 returns the slot to 0), so it is reusable across iterations with no host-side reset.

**⚠️ CRITICAL: Only ONE CTA per rank must call this function.** The signal pad slots are binary (0↔1) and can only track a single barrier invocation per rank pair. If multiple CTAs call this function concurrently, their threads will CAS-contend on the same slots:
1. CTA 0's thread CAS(0→1) succeeds (slot → 1)
2. CTA 1's thread CAS(0→1) fails (slot already 1) → spins waiting for reset
3. Peer's phase-2 CAS(1→0) resets the slot → CTA 1's CAS succeeds (slot → 1 again)
4. But the peer will NOT reset the slot a second time → **CTA 2+ deadlock**

Always guard the call with `if pid == 0:` (or equivalent single-CTA condition) after `barrier_on_this_grid`:

```python
barrier_on_this_grid(barrier_ptr)  # ALL CTAs participate
if pid == 0:                        # Only CTA 0 does cross-rank sync
    barrier_all_intra_node_atomic_cas_block(rank, rank, world_size, signal_pad_ptrs)
```

**Performance**: uses **multiple threads in parallel** (one per peer rank), not a single thread serially iterating over all peers. With 8 ranks, the parallel version is ~8× faster than the serial variant.

Parameters `local_rank`, `rank`, and `local_world_size` are declared `tl.constexpr` so the compiler can fully unroll the thread-to-peer mapping.

```python
@triton.jit
def barrier_all_intra_node_atomic_cas_block(
    local_rank: tl.constexpr,
    rank: tl.constexpr,
    local_world_size: tl.constexpr,
    symm_flag_ptr,                  # tensor of int64 ptrs → each peer's barrier base
):
    """Cross-rank barrier via system-scope CAS on symmetric-memory signal pads.

    `symm_flag_ptr` is the pointer array built by:
        symm_flag_ptr = torch.tensor(
            [hdl.get_signal_pad(i, (world_size,), torch.int32).data_ptr()
             for i in range(world_size)],
            dtype=torch.int64, device=device,
        )
    Each rank's signal pad needs `local_world_size` int32 slots (one per remote rank).
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
```

**Anti-pattern** — do NOT use single-threaded serial iteration:

```python
# BAD: single thread (flat_tid == 0) serially loops over all peers
if flat_tid == 0:
    for peer in range(local_world_size):
        if peer != local_rank:
            ...  # signal peer one by one (serial, slow)
```

> **Note on naming**: an older variant `barrier_all_intra_node_atomic_cas_block_symm_mem` (with `(rank, num_ranks, symm_flag_ptr, peer_barrier_ptrs)` signature, using Python-level `while atomic_cas(...) != 0` spin-loops) appeared in earlier drafts. It is superseded by the function above — same semantics, faster spin-loop, fewer arguments. Always emit the canonical form.

### 6.4 Host-side cross-rank barrier: `symm_mem_hdl.barrier()`

When a kernel does **not** have a subsequent in-kernel phase after communication (i.e., the kernel exits once the communication phase is done), use **host-side** `symm_mem_hdl.barrier()` instead of the in-kernel `barrier_all_intra_node_atomic_cas_block`. This is the standard PyTorch symmetric memory barrier and is the correct choice for **inter-sm** and **without-sm** modes.

`symm_mem_hdl` is obtained via `symm_mem.rendezvous(<symm_mem_tensor>, group=<communication_group>)`:

```python
import torch.distributed._symmetric_memory as symm_mem

buf = symm_mem.empty(shape, dtype=dtype, device=device)       # allocate symmetric memory
symm_mem_hdl = symm_mem.rendezvous(buf, group=dist.group.WORLD)  # get the handler
```

The handler provides `barrier()`, `get_buffer()`, `get_signal_pad()`, and other symmetric memory APIs.

**Underlying implementation** (from PyTorch's `CUDASymmetricMemory.cu`):

```cuda
static __global__ void barrier_kernel(
    uint32_t** signal_pads, int channel, int rank, int world_size, size_t timeout_ms) {
    if (threadIdx.x < world_size) {
        auto target_rank = threadIdx.x;
        if (target_rank == rank) return;
        // Signal target_rank (release)
        try_put_signal<memory_order_release>(
            signal_pads[target_rank] + world_size * channel + rank, timeout_ms);
        // Wait for target_rank's signal (acquire)
        try_wait_signal<memory_order_acquire>(
            signal_pads[rank] + world_size * channel + target_rank, timeout_ms);
    }
}
```

It launches a small CUDA kernel (1 block, `world_size` threads) that performs pairwise signal/wait on the symmetric memory signal pads. This is a **host-side API** — you call it from Python, and it launches + synchronizes internally.

**Key characteristics:**

| Property | `symm_mem_hdl.barrier()` | `barrier_all_intra_node_atomic_cas_block` |
|---|---|---|
| **Where it runs** | Host side (launches a CUDA kernel) | Inside a Triton kernel |
| **Channel support** | Yes — `barrier(channel=N)` for multi-phase pipelines | No — uses CAS on dedicated signal pad slots |
| **Self-resetting** | Yes — signal pads are toggled and reset by the barrier kernel itself | Yes — CAS(1→0) in phase 2 returns the slot to 0 |
| **Overhead** | Extra kernel launch + stream sync | Zero extra kernel launches (runs inside existing kernel) |
| **Best for** | inter-sm, without-sm (kernel exits after comm) | intra-sm (kernel has post-comm compute phase) |

**Usage pattern:**

```python
# Host side: use symm_mem_hdl.barrier() around signal.zero_() for cross-rank sync
# between iterations when the kernel itself has no post-comm compute phase.

# Reset per-tile signals (must happen after all ranks finished consuming)
ctx.signal_tensor.zero_()
ctx.symm_mem_hdl.barrier()  # ensure all ranks see zeroed signals

# Launch kernel — it exits when communication is done
kernel[grid](...)
```

**Channel usage for multi-phase pipelines:**

The `barrier(channel=N)` parameter supports multiple independent barrier channels within the same signal pad. This is used by PyTorch's `_pipelined_produce_and_all2all` and `_pipelined_all_gather_and_consume` to manage multi-phase pipelines:

```python
symm_mem.barrier(channel=0)  # phase 0: ensure workspace is ready
# ... copy local shard ...
symm_mem.barrier(channel=1)  # phase 1: all local shards copied
# ... consume remote shards ...
symm_mem.barrier(channel=0)  # final sync
```

For overlap kernels, `channel=0` is typically sufficient. The number of channels is bounded by `signal_pad_size / (sizeof(uint32_t) * world_size)`.

**Signal pad size requirement:** Each channel uses `world_size * world_size` uint32 slots in the signal pad. The default signal pad size is typically sufficient for a few channels. If more channels are needed, call `symm_mem.set_signal_pad_size()` before any allocations.

**Memory ordering:** The barrier kernel uses `memory_order_release` for puts and `memory_order_acquire` for waits. This establishes a full happens-before relationship: all stores before the barrier on any rank are visible to all loads after the barrier on every other rank. This is equivalent to the ordering guarantees of the in-kernel `barrier_all_intra_node_atomic_cas_block`.

**Decision rule:**

| Scenario | Use |
|---|---|
| Kernel has a post-comm compute phase (intra-sm) | In-kernel `barrier_all_intra_node_atomic_cas_block` (single CTA) |
| Kernel exits after communication (inter-sm, without-sm) | Host-side `symm_mem_hdl.barrier()` |
| Multi-phase pipelined host-side communication | Host-side `symm_mem_hdl.barrier(channel=N)` |

---

## 7. Per-tile signal pattern (inter-sm only)

The inter-sm mode uses an additional **per-tile** signal protocol on top of the rank barrier: the producer kernel stores 1 to a signal slot as each tile completes, the consumer kernel spins on it before consuming. This is what enables fine-grained pipelining between two kernels on different streams.

`_send_signal` and `_wait_signal` use system-scope store/load (not CAS). They are not needed by intra-sm or without-sm kernels.

```python
@triton.jit
def _send_signal(addrs, sem: tl.constexpr):
    """Store 1 to signal slot. System-scope.

    Valid sem values:
      - "release"  — flush prior writes before signal (use after producing data)
      - "relaxed"  — no ordering guarantee (use for pure event signaling)
      "acquire" is NOT valid for send.
    """
    if sem == "release":
        tl.inline_asm_elementwise(
            "st.global.release.sys.b32 [$1], 1;",
            "=r, l", [addrs], dtype=addrs.dtype, is_pure=False, pack=1,
        )
    else:  # relaxed
        tl.inline_asm_elementwise(
            "st.global.relaxed.sys.b32 [$1], 1;",
            "=r, l", [addrs], dtype=addrs.dtype, is_pure=False, pack=1,
        )


@triton.jit
def _wait_signal(addrs, sem: tl.constexpr):
    """Spin until signal slot becomes 1 (ld spin). System-scope.

    Valid sem values:
      - "acquire"  — ensure subsequent loads see remote writes (use before consuming data)
      - "relaxed"  — no ordering guarantee
      "release" is NOT valid for wait.
    """
    if sem == "acquire":
        tl.inline_asm_elementwise(
            """
            {
                .reg .u32 %tmp; .reg .pred %p;
                wait: ld.global.acquire.sys.b32 %tmp, [$1];
                setp.eq.u32 %p, %tmp, 1; @!%p bra wait;
            }
            """, "=r, l", [addrs], dtype=tl.int32, is_pure=False, pack=1,
        )
    else:  # relaxed
        tl.inline_asm_elementwise(
            """
            {
                .reg .u32 %tmp; .reg .pred %p;
                wait: ld.global.relaxed.sys.b32 %tmp, [$1];
                setp.eq.u32 %p, %tmp, 1; @!%p bra wait;
            }
            """, "=r, l", [addrs], dtype=tl.int32, is_pure=False, pack=1,
        )
```

### Usage pattern

```python
# Producer (compute kernel): one thread signals per tile, after the store
tl.store(out_ptr + offsets, result, mask=mask)
flat_tid = _get_flat_tid()
if flat_tid == 0:
    _send_signal(signal_ptr + pid, "release")

# Consumer (persistent comm kernel): one thread waits, then broadcasts to CTA
for tile_id in range(pid, total_tiles, tl.num_programs(0)):
    flat_tid = _get_flat_tid()
    if flat_tid == 0:
        _wait_signal(signal_ptr + tile_id, "acquire")
    tl.debug_barrier()  # broadcast readiness to all threads in CTA
    # ... consume tile ...
```

The signal tensor is reset by the host via `.zero_()` before each launch. `_send_signal` stores 1, `_wait_signal` spins until it sees 1. No in-kernel reset is needed.

---

## 8. Memory order rules

| Operation | Order | Why |
|---|---|---|
| Plain `tl.store` of data | (none) | Visibility is established by the subsequent signal/barrier |
| Barrier counter increment (`_atomic_add_release`) | release | Make preceding data stores visible to observers of the counter |
| Barrier counter read (`_load_acquire`) | acquire | Subsequent loads must see remote stores released before the counter |
| Cross-rank signal **send** after data store | release | Same reason — flush data before raising the flag |
| Cross-rank signal **wait** before data load | acquire | Reads after wait must see the producer's data |
| Cross-rank signal for pure ordering (no data) | relaxed | Cheaper; only the happens-before edge is needed |

Rule of thumb: any flag whose **point** is to advertise "data is ready" pairs **release** (writer) with **acquire** (reader). Use **relaxed** only when you've already proven there's no data dependency through that flag.

---

## 9. Which primitives each mode needs

| Primitive | inter-sm | intra-sm | without-sm |
|---|:---:|:---:|:---:|
| `_get_flat_tid` | ✅ | ✅ | — |
| `_get_flat_bid` | (optional) | ✅ | — |
| `tid` | — | — | ✅ (single-axis thread index for signal gating) |
| `__syncthreads` | — | — | ✅ (broadcast signal poll to CTA) |
| `_atomic_add_release` / `_load_acquire` / `_store_release_with_highbit` | (only if you add a grid barrier) | ✅ | — |
| `_cas_sys_release` / `_cas_sys_acquire` | — | ✅ (backs rank barrier) | — |
| `ld_sys` / `st_sys` | — | — | ✅ (signal polling & writing for CE sync) |
| `_send_signal` / `_wait_signal` | ✅ (core mechanism) | — | — |
| `barrier_on_this_grid` | usually ❌ (per-tile signal replaces it) | ✅ (separates compute/push from reduce) | — |
| `barrier_all_intra_node_atomic_cas_block` | ❌ (use host-side `symm_mem_hdl.barrier()`) | ✅ (after grid barrier) | — |
| `symm_mem_hdl.barrier()` (host-side) | ✅ (around `signal.zero_()`) | ❌ (use in-kernel rank barrier) | ✅ (between iterations) |
| `tl.debug_barrier()` | ✅ | ✅ | — |
| `stream_write_value32` / `cuStreamWaitValue32` | — | — | ✅ (CE-side signal ops) |

Without-sm uses: **kernel-side** `ld_sys`/`st_sys`/`tid`/`__syncthreads` from this catalog for signal polling and writing; **CE-side (host)** `symm_mem_hdl.stream_write_value32` (comm-first) or `cuda.bindings.driver.cuStreamWaitValue32` (comp-first) for CPU-side CE signal operations; and **cross-rank sync** `symm_mem_hdl.barrier()` (host-side) around `progress.fill_(0)` between iterations.