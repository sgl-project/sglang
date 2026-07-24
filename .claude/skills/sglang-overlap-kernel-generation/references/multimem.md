# Multimem (NVLS Hardware Multicast) Reference

> **On-demand reference.** Load this file **only when** the user explicitly asks the collective communication portion (the "comm" half of intra-sm / inter-sm overlap) to use **multimem / NVLS / hardware multicast / in-network reduction**. Otherwise stay on the default `tl.store` + hand-written reduce path.

> **Self-contained** The generated kernel file must paste the implementations from §D directly. The only project-external dependencies remain `torch`, `torch.distributed`, `triton`, `triton.language`.

This reference covers the three operations that drive NVSwitch in-network compute:

- `hdl.multicast_ptr` — host-side, extract the multicast (MC) pointer from a symmetric-memory handle (it is a PyTorch symm_mem property, no extra import needed).
- `multimem_st(ptr, val0[, val1, val2, val3], dtype=...)` — kernel-side, one PTX instruction broadcasts a value to **all** ranks. **Paste the implementation from §D into the generated file.**
- `multimem_ld_reduce(ptr, OP, DTYPE, ACC_DTYPE)` — kernel-side, one PTX instruction loads + reduces values **from all ranks** inside the NVSwitch fabric. **Paste the implementation from §D into the generated file.**

The semantics are exactly equivalent to "N peer stores" / "N peer loads + sum" but cost **one** instruction and **zero** extra SM cycles on the issuer; the data movement and the reduction happen inside the switch.

---

## (A) API Quick Reference

### A.1 Prerequisites

If `hdl.multicast_ptr` returns `0`. The host code **must** check for this and fall back to the default path (or raise a clear error). See §B.3.

### A.2 Multicast pointer from symm_mem

```python
import torch.distributed._symmetric_memory as symm_mem

buf = symm_mem.empty(M, N, dtype=dtype, device=device)
hdl = symm_mem.rendezvous(buf, group=group.group_name)
mc_ptr = hdl.multicast_ptr   # int64; 0 means "unsupported"
```

- Returns a Python `int` (already an `int64` device address). Pass it directly into a Triton kernel; Triton will treat it as an `int64` scalar — the kernel side should reinterpret it as `*` of the element dtype via `tl.cast(mc_ptr, tl.pointer_type(dtype))` (or just use pointer arithmetic on the `int64` and cast at the store/load site).
- The pointer covers the **same logical region** as `hdl.get_buffer(...)`, i.e. byte offsets line up with the local symm_mem buffer.
- This is a stock PyTorch API on the symm_mem handle. 

### A.3 `multimem_st(ptr, val0[, val1, val2, val3], dtype=...)`

Three variants are selected by how many `val*` arguments you pass:

| Variant | Args | PTX | Bytes written | Alignment of `ptr` |
|---------|------|-----|---------------|--------------------|
| scalar  | `val0`                         | `multimem.st.global.{suffix}`     | 4   | 4 B  |
| v2      | `val0, val1`                   | `multimem.st.global.v2.{suffix}`  | 8   | 8 B  |
| v4      | `val0, val1, val2, val3`       | `multimem.st.global.v4.{suffix}`  | 16  | **16 B** |

Supported `dtype` (drives the PTX suffix):

| `dtype`                  | PTX suffix | Per-register payload          |
|--------------------------|------------|-------------------------------|
| `tl.float32`             | `f32`      | 1 fp32 per reg → v4 = 4×fp32 (16 B) |
| `tl.bfloat16`            | `bf16x2`   | 2 bf16 packed in 1 int32 reg → v4 = 8×bf16 (16 B) |
| `tl.float16`             | `f16x2`    | 2 fp16 packed in 1 int32 reg → v4 = 8×fp16 (16 B) |
| `tl.int32` / `tl.uint32` | `b32`      | 1 int32 per reg → v4 = 4×int32 (16 B) |
| `tl.int64` / `tl.uint64` | `b64`      | 1 int64 per reg (scalar/v2 only) |

Semantics: one instruction → the value lands at the same byte offset in **every** rank's symmetric-memory region. There is no per-rank loop, no `for peer in range(world_size)`. This is the key reason intra-sm push phases shrink from `O(world_size)` stores to `O(1)`.

### A.4 `multimem_ld_reduce(ptr, OP, DTYPE, ACC_DTYPE)`

All three parameters are `tl.constexpr` (compile-time constants), matching Triton's convention for dispatch parameters.

- `OP`: **only `"sum"` is supported by hardware today** (`tl.static_assert` enforces this in the inlined source).
- `DTYPE`: `tl.float32 | tl.bfloat16 | tl.float16` — selects the PTX operand suffix (integer reduce is not exposed here).
- `ACC_DTYPE`: `tl.float32` — in-switch accumulator precision. Currently only `tl.float32` is supported (`tl.static_assert` enforces this in the inlined source). The `.acc::f32` PTX modifier is used for bf16/fp16 to preserve reduction precision. Ignored for `DTYPE == tl.float32` (no `.acc::` modifier needed).
- The default variant emitted is **v4**: returns a tuple `(v0, v1, v2, v3)` of the reduced values (16 B worth).

```python
v0, v1, v2, v3 = multimem_ld_reduce(
    mc_ptr + byte_off,            # pointer arithmetic in element units; see §C.4
    OP="sum",
    DTYPE=tl.bfloat16,
    ACC_DTYPE=tl.float32,         # .acc::f32 for bf16/fp16
)
```

One instruction → NVSwitch reads the value at this offset from every rank, sums them in network, returns the reduced result to the issuing thread. The N-1 cross-rank bandwidth that a naïve hand-written reduce would burn is saved.

### A.5 Key performance properties

- **1 instruction, all ranks.** No peer loop on the issuer side.
- **Zero SM occupancy for the comm itself.** Compute can keep using the same warps; the switch does the broadcast / reduction.
- **Bandwidth-optimal.** `multimem.st` consumes 1× NVLink BW for an N-rank broadcast (vs. N-1×). `multimem.ld_reduce` returns reduced data, saving (N-1)/N of the ingress BW.
- **Alignment-sensitive.** Misaligned `ptr` produces undefined behaviour; the v4 path requires 16 B alignment of both `ptr` and the multicast region's base.
- **Visibility model.** Multimem is a memory operation, not a barrier — cross-rank barriers are still required. See §C.6 for details.

---

## (B) Host-side Setup

### B.1 Extend the existing context dataclass

Add `mc_ptr` next to the existing symm_mem fields. Do **not** remove the regular `peer_buf` / `signal_pad_ptrs` plumbing — multimem replaces only the bulk data movement, not the signaling.

```python
from dataclasses import dataclass
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


@dataclass
class OverlapCtx:
    # ... existing fields ...
    symm_buf: torch.Tensor
    hdl: object
    signal_pad_ptrs: torch.Tensor
    grid_barrier: torch.Tensor
    # NEW: multicast pointer (int64 device address; 0 if unsupported)
    mc_ptr: int
```

### B.2 Rendezvous + extract `mc_ptr`

The order is fixed: `symm_mem.empty` → `symm_mem.rendezvous` → `hdl.multicast_ptr`. Accessing `multicast_ptr` before rendezvous is undefined.

```python
def create_overlap_context(M, N, dtype, group, device):
    world_size = dist.get_world_size(group)

    # 1. Allocate symmetric memory (shape selection per SKILL.md table)
    symm_buf = symm_mem.empty(M, N, dtype=dtype, device=device)

    # 2. Rendezvous across the group
    hdl = symm_mem.rendezvous(symm_buf, group=group.group_name)

    # 3. Extract multicast pointer (must come AFTER rendezvous)
    mc_ptr = hdl.multicast_ptr      # stock PyTorch API

    # 4. Same signal-pad / grid-barrier plumbing as the default path
    signal_pad_ptrs = torch.tensor(
        [hdl.get_signal_pad(i, (world_size,), torch.int32).data_ptr()
         for i in range(world_size)],
        dtype=torch.int64, device=device,
    )
    grid_barrier = torch.zeros(1, dtype=torch.int32, device=device)

    return OverlapCtx(
        symm_buf=symm_buf,
        hdl=hdl,
        signal_pad_ptrs=signal_pad_ptrs,
        grid_barrier=grid_barrier,
        mc_ptr=mc_ptr,
    )
```

### B.3 Runtime support check

```python
ctx = create_overlap_context(...)
if ctx.mc_ptr == 0:
    raise RuntimeError(
        "Multimem (NVLS) is not supported on this system: "
        "requires H100+ GPU, NVSwitch V3+, CUDA 12.3+, and PyTorch 2.11+. "
        "Re-run without the --multimem flag to use the default tl.store path."
    )
```

For a richer pre-flight diagnostic you may probe `torch.cuda.get_device_capability()[0] >= 9` and `torch.version.cuda` before allocating anything.

### B.4 Passing `mc_ptr` into the kernel

```python
kernel[grid](
    # ... regular tensor args ...
    ctx.signal_pad_ptrs,
    ctx.grid_barrier,
    ctx.mc_ptr,           # plain int64; Triton accepts it as a scalar arg
    M, N,
    num_warps=32,
    num_stages=1,
)
```

The kernel signature simply declares it as a regular argument (no special annotation). Inside the kernel, treat `mc_ptr` as an `int64` device address and apply byte-offset arithmetic, then cast to the appropriate pointer type at the store/load site (see §C).

---

## (C) Kernel-side Usage

### C.1 Where the implementations come from

**Do not import `multimem_st` / `multimem_ld_reduce` from any external module.** Paste the source from §D verbatim into the generated kernel file's "PTX Primitives" section (the first of the four sections defined in `references/kernel_format.md`). After that, just use the names:

```python
import triton
import triton.language as tl
# (Paste the helpers + multimem_st + multimem_ld_reduce from §D here.)

@triton.jit
def my_overlap_kernel(...):
    ...
    multimem_st(mc_addr, v0, v1, v2, v3, dtype=tl.bfloat16)
    a, b, c, d = multimem_ld_reduce(mc_addr, OP="sum",
                                     DTYPE=tl.bfloat16, ACC_DTYPE=tl.float32)
```

### C.2 Replacing the intra-sm "push" phase

Default intra-sm pattern — one store per peer (`O(world_size)` traffic on the issuer):

```python
# BEFORE: hand-written push to every peer
for peer in tl.static_range(WORLD_SIZE):
    peer_ptr = tl.load(peer_buf_ptrs + peer).to(tl.pointer_type(tl.bfloat16))
    tl.store(peer_ptr + dst_off + cols, vals, mask=mask)
```

Multimem-equivalent — one instruction broadcasts to every peer:

```python
# AFTER: single multimem store, switch broadcasts to all ranks
# vals must be already packed into 4 registers (16 B). For bf16, that's 8 bf16
# elements packed as 4 × int32 (see §C.4).
mc_addr = mc_ptr + (dst_off + cols_base) * tl.constexpr(2)  # byte offset for bf16
tl.multiple_of(mc_addr, 16)
multimem_st(mc_addr, v0, v1, v2, v3, dtype=tl.bfloat16)
```

Behavioural consequences:

- The per-CTA push cost drops from `world_size × store` to `1 × store`.
- The cross-rank barrier you used after the push loop is still needed (see §C.5).
- If the local rank also needs the value, multimem already wrote it to the local symm_mem region — no extra `tl.store` to the local buffer.

### C.3 Replacing the reduce phase (reduce-scatter / all-reduce)

Default pattern — load from every peer, sum in registers:

```python
acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
for peer in tl.static_range(WORLD_SIZE):
    peer_ptr = tl.load(peer_buf_ptrs + peer).to(tl.pointer_type(tl.bfloat16))
    acc += tl.load(peer_ptr + src_off + cols, mask=mask, other=0.0).to(tl.float32)
out = acc.to(tl.bfloat16)
tl.store(out_ptr + ..., out, mask=mask)
```

Multimem-equivalent — one instruction loads + sums in the switch:

```python
mc_addr = mc_ptr + (src_off + cols_base) * tl.constexpr(2)
tl.multiple_of(mc_addr, 16)
v0, v1, v2, v3 = multimem_ld_reduce(
    mc_addr, OP="sum", DTYPE=tl.bfloat16, ACC_DTYPE=tl.float32,
)
# v0..v3 are bf16 values (returned as 4 × int32 holding 8 bf16 elements).
# Either store them back to the local output via 4 × tl.store, or repack
# into a vector and store once (preferred).
```

Notes:

- `ACC_DTYPE=tl.float32` is required for bf16/fp16 to match the precision rules already enforced by the SKILL.md "Accumulator dtype" check.
- `OP="sum"` is the only operation currently supported by hardware. For mean/scaling, do the divide **after** `multimem_ld_reduce` returns (in fp32, then cast).

### C.4 Data packing rules (the 16-byte unit)

`multimem_st` / `multimem_ld_reduce` always operate on 4 registers in the v4 path. The mapping from "elements" to "registers" depends on dtype:

| dtype          | Elements per `multimem_*_v4` call | How to obtain the 4 values |
|----------------|------------------------------------|----------------------------|
| `tl.float32`   | 4                                  | Load 4 fp32 elements as a 4-vector, unpack into `v0..v3`. |
| `tl.bfloat16`  | 8                                  | Reinterpret 8 bf16 as 4 × int32, pass the 4 int32 values. The PTX suffix `bf16x2` tells the switch to treat each int32 as 2 bf16. |
| `tl.float16`   | 8                                  | Same as bf16 but with `dtype=tl.float16` → PTX suffix `f16x2`. |

Practical loading pattern for bf16:

```python
# Load 8 bf16 values as 4 packed int32s
data_i32 = tl.load(
    (input_ptr + src_off + cols_base).to(tl.pointer_type(tl.int32)),
    mask=mask_i32,
    other=0,
)  # shape [..., 4], dtype int32, each int32 = 2 bf16

multimem_st(mc_addr, data_i32[..., 0], data_i32[..., 1],
            data_i32[..., 2], data_i32[..., 3], dtype=tl.bfloat16)
```

For fp32 you skip the packing and pass the 4 fp32 values directly.

#### Scalar (per-row) loading: use `ld.global.v4` instead of 4× `tl.load`

When the multimem push is done in a serial per-row loop (e.g., inter-SM comm kernel iterating `tl.static_range(0, BLOCK_M)` rows × `tl.static_range(0, N_I32_PER_ROW, 4)` groups), each iteration loads 4 scalar int32 values and passes them to `multimem_st`. **Do NOT use 4 separate `tl.load` calls** — this generates 4 independent load instructions and wastes memory bandwidth.

Instead, use a single `ld.global.v4.b32` inline-asm helper that loads 16 bytes in one instruction:

```python
@triton.jit
def _load_v4_b32(ptr):
    """ld.global.v4.b32 — load 4 × int32 (16 bytes) in one instruction."""
    return tl.inline_asm_elementwise(
        asm="ld.global.v4.b32 {$0,$1,$2,$3}, [$4];",
        constraints="=r,=r,=r,=r,l",
        args=[ptr],
        dtype=(tl.int32, tl.int32, tl.int32, tl.int32),
        is_pure=True, pack=1)
```

Usage:

```python
# GOOD: one v4 load → one multimem_st (16 B load + 16 B broadcast)
load_addr = buf_i32_ptr + row_i32_off + g
v0, v1, v2, v3 = _load_v4_b32(load_addr)
multimem_st(mc_addr, v0, v1, v2, v3, DTYPE)

# BAD: 4 × scalar tl.load (generates 4 separate ld.global.b32 instructions)
v0 = tl.load(buf_i32_ptr + row_i32_off + g)
v1 = tl.load(buf_i32_ptr + row_i32_off + g + 1)
v2 = tl.load(buf_i32_ptr + row_i32_off + g + 2)
v3 = tl.load(buf_i32_ptr + row_i32_off + g + 3)
```

This pairs the 16 B vectorized load with the 16 B `multimem_st`, maximizing memory bandwidth utilization. The `_load_v4_b32` helper should be included in the PTX Primitives section of the generated kernel file whenever the multimem push uses scalar per-row iteration.

### C.5 Alignment hints & offset rules

These hints are **mandatory** for the v4 variant — the PTX instruction is UB on misaligned addresses.

```python
tl.multiple_of(mc_ptr, 16)              # base pointer is 16 B aligned
tl.multiple_of(byte_offset, 16)         # the offset added to it is also 16 B aligned
tl.max_contiguous(cols_base, 16)        # helps the compiler vectorize the surrounding loads
```

Rules of thumb when computing offsets:

- Keep `BLOCK_N` such that one tile row spans an integer number of 16 B groups (i.e. multiple of 8 for bf16/fp16, multiple of 4 for fp32).
- Compute offsets in elements, then multiply by `element_size` when adding to `mc_ptr` (which is in bytes). Use `tl.cast(off, tl.int64)` when the multiplication could overflow int32 (see SKILL.md "Overflow in offsets" check).
- Place the `tl.multiple_of` hint immediately before the `multimem_st` / `multimem_ld_reduce` call so the compiler does not lose it through intermediate ops.

### C.6 Synchronization is still required

`multimem_st` is a memory store, not a release. `multimem_ld_reduce` is a memory load, not an acquire. Cross-rank visibility is not free.

Keep the existing primitives from `primitives.md`:

- After a multimem push, run the **cross-rank barrier** (`barrier_all_intra_node_atomic_cas_block` or equivalent) before any consumer issues `multimem_ld_reduce` / peer loads.
- After a producer CTA finishes its tile, still do the **grid-level barrier** (`barrier_on_this_grid`) before the consumer phase begins on the same rank.
- The signal-pad / CAS protocol is unchanged; multimem only swaps out the bulk data ops.

A correct intra-sm reduce-scatter sequence with multimem becomes:

```
1. compute tile → results in regs
2. multimem_st  (push results to all ranks' symm_buf at this tile offset)
3. barrier_on_this_grid                     # local CTAs synced
4. barrier_all_intra_node_atomic_cas_block  # cross-rank synced
5. multimem_ld_reduce on the rows assigned to this rank
6. cast + tl.store to the final output tensor
```

Skipping step 3 or step 4 is the most common cause of intermittent wrong results when migrating from the default path to multimem.

---

## (D) Inlined Implementation Source (paste into the generated kernel file)

Copy the block below verbatim into the **PTX Primitives** section of the generated file. It only depends on `triton` and `triton.language` — no nvshmem, no other project module.

```python
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# multimem.st — v4 variants (16 B broadcast to all ranks)
# ---------------------------------------------------------------------------

@triton.jit
def _multimem_st_v4_bf16(ptr, val0, val1, val2, val3):
    """multimem.st.global.v4.bf16x2 [ptr], {val0, val1, val2, val3};
    Each val is an int32 holding 2 packed bf16 values. Total: 8 bf16 = 16 B.
    """
    tl.inline_asm_elementwise(
        asm="""
        multimem.st.global.v4.bf16x2 [$1], {$2, $3, $4, $5};
        mov.u32 $0, 0;
        """,
        constraints="=r,l,r,r,r,r",
        args=[ptr, val0, val1, val2, val3],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _multimem_st_v4_f16(ptr, val0, val1, val2, val3):
    """multimem.st.global.v4.f16x2 [ptr], {val0, val1, val2, val3};
    Each val is an int32 holding 2 packed fp16 values. Total: 8 fp16 = 16 B.
    """
    tl.inline_asm_elementwise(
        asm="""
        multimem.st.global.v4.f16x2 [$1], {$2, $3, $4, $5};
        mov.u32 $0, 0;
        """,
        constraints="=r,l,r,r,r,r",
        args=[ptr, val0, val1, val2, val3],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _multimem_st_v4_f32(ptr, val0, val1, val2, val3):
    """multimem.st.global.v4.f32 [ptr], {val0, val1, val2, val3};
    Each val is one fp32. Total: 4 fp32 = 16 B.
    """
    tl.inline_asm_elementwise(
        asm="""
        multimem.st.global.v4.f32 [$1], {$2, $3, $4, $5};
        mov.u32 $0, 0;
        """,
        constraints="=r,l,f,f,f,f",
        args=[ptr, val0, val1, val2, val3],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def multimem_st(ptr, val0, val1, val2, val3, DTYPE: tl.constexpr):
    """Store 16B to all ranks via NVLS multicast (v4 variant, one PTX instruction).

    Args:
        ptr: multicast pointer (byte address, 16 B aligned).
        val0-val3: 4 register values. For bf16/fp16: each is int32 holding 2 packed elements.
                   For fp32: each is one fp32 value.
        DTYPE: tl.bfloat16 / tl.float16 / tl.float32 — selects the PTX suffix.
    """
    if DTYPE == tl.bfloat16:
        _multimem_st_v4_bf16(ptr, val0, val1, val2, val3)
    elif DTYPE == tl.float16:
        _multimem_st_v4_f16(ptr, val0, val1, val2, val3)
    elif DTYPE == tl.float32:
        _multimem_st_v4_f32(ptr, val0, val1, val2, val3)


# ---------------------------------------------------------------------------
# multimem.ld_reduce — v4 variants (16 B in-switch reduction from all ranks)
# ---------------------------------------------------------------------------

@triton.jit
def _multimem_ld_reduce_v4_bf16_acc32(ptr):
    """multimem.ld_reduce.global.add.acc::f32.v4.bf16x2 {r0,r1,r2,r3}, [ptr];
    Returns 4 × int32, each holding 2 reduced bf16 values (8 bf16 total = 16 B).
    The .acc::f32 modifier ensures the in-switch accumulator runs in fp32.
    """
    return tl.inline_asm_elementwise(
        asm="multimem.ld_reduce.global.add.acc::f32.v4.bf16x2 {$0,$1,$2,$3}, [$4];",
        constraints="=r,=r,=r,=r,l",
        args=[ptr],
        dtype=(tl.int32, tl.int32, tl.int32, tl.int32),
        is_pure=False,
        pack=1,
    )


@triton.jit
def _multimem_ld_reduce_v4_f16_acc32(ptr):
    """multimem.ld_reduce.global.add.acc::f32.v4.f16x2 {r0,r1,r2,r3}, [ptr];
    Returns 4 × int32, each holding 2 reduced fp16 values (8 fp16 total = 16 B).
    """
    return tl.inline_asm_elementwise(
        asm="multimem.ld_reduce.global.add.acc::f32.v4.f16x2 {$0,$1,$2,$3}, [$4];",
        constraints="=r,=r,=r,=r,l",
        args=[ptr],
        dtype=(tl.int32, tl.int32, tl.int32, tl.int32),
        is_pure=False,
        pack=1,
    )


@triton.jit
def _multimem_ld_reduce_v4_f32(ptr):
    """multimem.ld_reduce.global.add.v4.f32 {r0,r1,r2,r3}, [ptr];
    Returns 4 × fp32 reduced values (16 B total).
    """
    return tl.inline_asm_elementwise(
        asm="multimem.ld_reduce.global.add.v4.f32 {$0,$1,$2,$3}, [$4];",
        constraints="=f,=f,=f,=f,l",
        args=[ptr],
        dtype=(tl.float32, tl.float32, tl.float32, tl.float32),
        is_pure=False,
        pack=1,
    )


@triton.jit
def multimem_ld_reduce(ptr, OP: tl.constexpr, DTYPE: tl.constexpr, ACC_DTYPE: tl.constexpr):
    """Load + in-switch reduce from all ranks (v4 variant, one PTX instruction).

    NVSwitch reads 16 B at `ptr` from every rank, sums them in network, and
    returns the reduced 16 B to the issuing thread.

    Args:
        ptr: multicast pointer (byte address, 16 B aligned).
        OP: reduction operation. Only "sum" is supported by hardware.
        DTYPE: tl.bfloat16 / tl.float16 / tl.float32 — selects the PTX suffix.
        ACC_DTYPE: in-switch accumulator precision. Only tl.float32 is
                   supported currently. The .acc::f32 PTX modifier is used
                   for bf16/fp16 to preserve reduction precision.

    Returns:
        Tuple (v0, v1, v2, v3):
          - bf16/fp16: 4 × int32, each holding 2 packed reduced elements.
          - fp32: 4 × fp32 reduced values.
    """
    tl.static_assert(OP == "sum", "multimem_ld_reduce: only OP='sum' is supported by hardware")
    tl.static_assert(ACC_DTYPE == tl.float32, "multimem_ld_reduce: only ACC_DTYPE=tl.float32 is supported currently")
    if DTYPE == tl.bfloat16:
        return _multimem_ld_reduce_v4_bf16_acc32(ptr)
    elif DTYPE == tl.float16:
        return _multimem_ld_reduce_v4_f16_acc32(ptr)
    elif DTYPE == tl.float32:
        return _multimem_ld_reduce_v4_f32(ptr)
```

After pasting, the rest of the kernel file uses `multimem_st(ptr, v0, v1, v2, v3, DTYPE)` / `multimem_ld_reduce(ptr, OP, DTYPE, ACC_DTYPE)` directly. 

---

## Checklist when generating a multimem-enabled overlap kernel

- [ ] Source from §D is pasted into the file's PTX Primitives section; 
- [ ] Host: accessed `hdl.multicast_ptr` **after** `symm_mem.rendezvous`, stored in `ctx.mc_ptr`.
- [ ] Host: runtime check `ctx.mc_ptr != 0` with a clear error message.
- [ ] Kernel signature: `mc_ptr` declared as a plain scalar (Triton treats it as int64).
- [ ] Inside kernel: `tl.multiple_of(mc_ptr, 16)` and offset alignment hints in place.
- [ ] Push path: replaced the `for peer in range(world_size): tl.store(peer_buf, ...)` loop with a single `multimem_st(...)`.
- [ ] Reduce path: replaced the hand-written `acc += tl.load(peer_buf, ...)` loop with `multimem_ld_reduce(..., OP="sum", ACC_DTYPE=tl.float32)`.
- [ ] Packing: bf16/fp16 v4 loads reinterpreted as 4 × int32; fp32 v4 passes 4 fp32 directly.
- [ ] Barriers: grid-level + cross-rank barriers are still present around the multimem ops.
- [ ] Dtype: `ACC_DTYPE=tl.float32` used for bf16/fp16 reductions to preserve precision.