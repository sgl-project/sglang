# Shared Primitives Library

This file is the **single source of truth** for all low-level synchronization helpers used by overlap kernels — thread/block ID helpers, PTX inline-asm atomics, signals, system-scope load/store, and the three-level barrier hierarchy.

Read this file whenever you are about to emit any of these helpers into a generated kernel. The mode files reference primitives by name and only describe *how* they are composed. When emitting a kernel, include only the primitives that mode actually uses (see [§9](#9-which-primitives-each-mode-needs)) — do not paste this whole catalog into the output file.

## Table of Contents

1. [Thread/Block ID helpers](#1-threadblock-id-helpers)
2. [GPU-scope memory ordering atomics](#2-gpu-scope-memory-ordering-atomics)
3. [System-scope CAS (cross-rank)](#3-system-scope-cas-cross-rank)
4. [System-scope load/store (without-sm)](#4-system-scope-loadstore-without-sm)
5. [Three-level barrier hierarchy](#5-three-level-barrier-hierarchy)
   - 5.1 Thread-level (intra-CTA)
   - 5.2 CTA-Grid level: `barrier_on_this_grid`
   - 5.3 Rank level: `barrier_all_intra_node_atomic_cas_block`
   - 5.4 Host-side cross-rank barrier: `symm_mem_hdl.barrier()`
6. [Per-tile signal pattern (inter-sm only)](#6-per-tile-signal-pattern-inter-sm-only)
7. [Memory order rules](#7-memory-order-rules)
8. [Which primitives each mode needs](#8-which-primitives-each-mode-needs)

---

## 1. Thread/Block ID helpers

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


@triton.jit
def __syncthreads():
    """PTX bar.sync — broadcasts signal poll result from one thread to all in CTA."""
    tl.inline_asm_elementwise(asm="bar.sync 0;", constraints="=r", args=[], dtype=tl.int32, is_pure=False, pack=1)
```

**Signedness pitfall:** `_get_flat_tid()` returns `tl.uint32`. Triton forbids `//`, `%` between `uint32` and `int32`. Cast first: `flat_tid = _get_flat_tid().to(tl.int32)`. Simple comparisons (`== 0`, `< N`) work without casting.

---

## 2. GPU-scope memory ordering atomics

Used to build the CTA-grid barrier. GPU scope is correct because all participating CTAs run on the same device.

```python
@triton.jit
def _atomic_add_release(ptr, val):
    """Release-semantics atomic add — counts arrivals at a barrier point."""
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
    """Acquire-semantics load — spin-wait on a barrier counter."""
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
    """Master CTA broadcasts 'release all waiters' by storing expected | 0x80000000."""
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

## 3. System-scope CAS (cross-rank)

Backs the **cross-rank barrier** (`barrier_all_intra_node_atomic_cas_block`). System scope is required because the target memory lives in another GPU's symmetric memory. The spin-loop is kept **inside PTX** (`@!%p0 bra cas_loop`) — a Python-level `while` around a single-shot CAS incurs significant Triton control-flow overhead.

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

---

## 4. System-scope load/store (without-sm)

Without-sm (copy engine) overlap kernels use these for CE ↔ compute kernel synchronization. Plain `tl.load`/`tl.store` lack cross-device visibility — the L2 cache may return stale data when the writer is on a different GPU. System-scope acquire/release is required.

```python
@triton.jit
def ld_sys(ptr):
    """ld.global.acquire.sys.b32 — poll signal written by CE (cuStreamWriteValue32 or PtoP copy)."""
    return tl.inline_asm_elementwise(
        asm="ld.global.acquire.sys.b32 $0, [$1];",
        constraints="=r,l",
        args=[ptr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def st_sys(ptr, val):
    """st.global.release.sys.b32 — write per-tile completion signal for CE cuStreamWaitValue32.

    No contention (each tile has exactly one writer, signal reset to 0 before launch),
    so a simple store suffices — the CAS spin-loop in `_send_signal` is overkill here."""
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

---

## 5. Three-level barrier hierarchy

Pick the smallest scope that covers your data dependency.

### 5.1 Thread-level (intra-CTA)

Built-in. Use `tl.debug_barrier()` to re-converge threads after warp-divergent control flow, or to broadcast a value written by `if flat_tid == 0` to the rest of the CTA.

```python
if flat_tid == 0:
    _wait_signal(signal_ptr + tile_id, "acquire")
tl.debug_barrier()  # broadcast readiness to all threads in CTA
```

### 5.2 CTA-Grid level: `barrier_on_this_grid`

Synchronizes all CTAs in a single kernel launch **without** requiring `cooperative_groups`. Uses a split-counter pattern: each CTA atomically increments a global counter; the master CTA (`bid == 0`) spins until all arrive, then flips the high bit to release everyone. Uses the *master CTA pattern* (only `bid==0` spins, others check the high bit) with a single `tl.debug_barrier()` at the end.

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

**Host-side**: reset counter to 0 between launches (`ctx.grid_barrier.zero_()`).

**⚠️ CRITICAL: `num_ctas` must not exceed SM for persistent kernels.** If `num_ctas > SMs`, some CTAs cannot be scheduled (SMs occupied by earlier CTAs already spinning at the barrier) and the kernel deadlocks. Cap: `num_ctas = min(total_tiles, num_sm)`.

### 5.3 Rank level: `barrier_all_intra_node_atomic_cas_block`

Cross-rank barrier for `inter-sm` and `intra-sm` kernels. Thread `i` (where `i < local_world_size`) signals peer `i` by CAS(0→1), then waits for the peer to signal us back by CAS(1→0). Self-resetting (CAS 1→0 returns slot to 0), reusable across iterations.

**⚠️ CRITICAL: Only ONE CTA per rank must call this.** Signal pad slots are binary (0↔1) — multiple CTAs calling concurrently will CAS-contend and **deadlock**. Guard with `if pid == 0:` after `barrier_on_this_grid`.

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

### 5.4 Host-side cross-rank barrier: `symm_mem_hdl.barrier()`

Use when the kernel has **no** subsequent in-kernel phase after communication (inter-sm, without-sm). It launches a small CUDA kernel (1 block, `world_size` threads) that performs pairwise signal/wait on symmetric memory signal pads — a host-side API that launches + synchronizes internally. Self-resetting; supports `barrier(channel=N)` for multi-phase pipelines (typically `channel=0` is sufficient for overlap kernels).

`symm_mem_hdl` is obtained via `symm_mem.rendezvous(buf, group=dist.group.WORLD)`. The handler provides `barrier()`, `get_buffer()`, `get_signal_pad()`, etc.

| Property | `symm_mem_hdl.barrier()` | `barrier_all_intra_node_atomic_cas_block` |
|---|---|---|
| **Where** | Host side (launches CUDA kernel) | Inside a Triton kernel |
| **Overhead** | Extra kernel launch + stream sync | Zero extra launches |
| **Best for** | inter-sm, without-sm | intra-sm (has post-comm compute) |

**Usage pattern:**

```python
# Reset + barrier before launch
hdl.barrier()              # all ranks finished previous iteration
progress.fill_(0)          # or signal.zero_()
hdl.barrier()              # all ranks see reset before launching

# Launch kernels ...
```

---

## 6. Per-tile signal pattern (inter-sm only)

Inter-sm uses a per-tile signal protocol: producer stores 1 as each tile completes, consumer spins on it before consuming. `_send_signal`/`_wait_signal` use system-scope store/load (not CAS). Not needed by intra-sm or without-sm.

```python
@triton.jit
def _send_signal(addrs, sem: tl.constexpr):
    """Store 1 to signal slot. System-scope. sem: "release" (after data) or "relaxed" (pure event)."""
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
    """Spin until signal slot becomes 1. System-scope. sem: "acquire" (before data) or "relaxed"."""
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

Signal tensor is reset by host `.zero_()` before each launch. No in-kernel reset needed.

---

## 7. Memory order rules

| Operation | Order | Why |
|---|---|---|
| Plain `tl.store` of data | (none) | Visibility established by subsequent signal/barrier |
| Barrier counter increment (`_atomic_add_release`) | release | Make preceding data stores visible to counter observers |
| Barrier counter read (`_load_acquire`) | acquire | Subsequent loads must see stores released before the counter |
| Cross-rank signal **send** after data store | release | Flush data before raising the flag |
| Cross-rank signal **wait** before data load | acquire | Reads after wait must see the producer's data |
| Cross-rank signal for pure ordering (no data) | relaxed | Cheaper; only the happens-before edge is needed |

Rule of thumb: any flag whose **point** is to advertise "data is ready" pairs **release** (writer) with **acquire** (reader). Use **relaxed** only when there's no data dependency through that flag.

---

## 8. Which primitives each mode needs

| Primitive | inter-sm | intra-sm | without-sm |
|---|:---:|:---:|:---:|
| `_get_flat_tid` | ✅ | ✅ | ✅ |
| `_get_flat_bid` | (optional) | ✅ | — |
| `__syncthreads` | — | — | ✅ (broadcast signal poll to CTA) |
| `_atomic_add_release` / `_load_acquire` / `_store_release_with_highbit` | (grid barrier only) | ✅ | — |
| `_cas_sys_release` / `_cas_sys_acquire` | — | ✅ (rank barrier) | — |
| `ld_sys` / `st_sys` | — | — | ✅ (CE signal poll/write) |
| `_send_signal` / `_wait_signal` | ✅ (core) | — | — |
| `barrier_on_this_grid` | usually ❌ | ✅ | — |
| `barrier_all_intra_node_atomic_cas_block` | ❌ (host barrier) | ✅ | ❌ (host barrier) |
| `symm_mem_hdl.barrier()` | ✅ | ❌ (in-kernel) | ✅ |
| `tl.debug_barrier()` | ✅ | ✅ | — |
| `stream_write_value32` / `cuStreamWaitValue32` | — | — | ✅ (CE-side) |