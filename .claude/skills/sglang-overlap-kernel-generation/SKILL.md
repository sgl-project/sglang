---
 name: sglang-overlap-kernel-generation
description: |
  Generate compute-communication overlap kernels for distributed GPU workloads (currently NVIDIA GPU only).
  Given a compute kernel, a collective communication operation, their execution order,
  and an overlap mode (inter-sm / intra-sm / without-sm), produce a complete overlap
  implementation based on PyTorch symmetric memory. Use this skill whenever the user
  wants to overlap compute with communication, fuse a collective into a kernel, hide
  communication latency, or mentions "overlap", "fused collective", "compute-comm",
  "persistent kernel", "copy engine", "symmetric memory", "signal/wait", "inter-sm",
  "intra-sm", "fp8", "block-wise fp8", "w8a8", "fp8 gemm", "fp8 matmul" in the context
  of multi-GPU kernel development.
---
# Compute-Communication Overlap Kernel Generator

> **Platform limitation**: Currently only NVIDIA GPUs are supported.

## Input

The user's request may only partially specify the configuration. For any item below that the user has **not** explicitly provided, ask them about it before generating code. Ask each missing item **one by one in order** (do not bundle multiple questions into one message, and do not skip any). If the user skips a question (e.g., replies "skip", or similar), use the default value for that item and move on to the next.

When presenting options to the user via `AskUserQuestion`, put the full explanation directly into each option's **label** field (not just the description field, which may be hidden behind a tooltip in the UI). For example, use `"label": "inter-sm — two kernels, two streams"` rather than a bare `"label": "inter-sm"`. This ensures the user sees the explanation directly without needing to hover or click.

1. **Compute kernel**: what kernel to overlap (e.g., GEMM, topk-reduce), users can provide the path to the target compute kernel.
2. **Collective operation**: all-gather, reduce-scatter, all-reduce, etc.
3. **Communication mechanism**: which hardware path for the collective? *(default: register)*
   - `register` — load/store via symmetric memory (SM-driven)
   - `copy_engine` — DMA copy on a background stream (zero SM cost)
   - `tma` — Tensor Memory Accelerator async bulk copy **(not yet supported)**
   - `nvswitch` — multimem via NVSwitch hardware (NVLS)
4. **Overlap mode**: how compute and comm are scheduled: *(default: inter-sm)*
   - `inter-sm` — two separate kernels, two streams
   - `intra-sm` — fused single kernel, single stream
   - `without-sm` — copy engine (DMA), two streams
5. **Execution order**: compute-first or comm-first
6. **Overlap triton kernel type**: **persistent** (one CTA loops over multiple tiles) or **non-persistent** (one CTA handles one tile)? *(default: persistent)*

**Constraint 1**: if communication mechanism = `copy_engine`, overlap mode is forced to `without-sm` (copy engine runs on DMA, not SMs) — inform the user and skip the overlap mode question.

**Constraint 2**: if communication mechanism = `tma`, inform the user that TMA (Tensor Memory Accelerator) async bulk copy is **not yet supported** and ask them to choose a different mechanism (`register`, `copy_engine`, or `nvswitch`). Do not proceed with TMA.

## Mode Overview

### Inter-SM — two kernels, two streams

Compute and communication run as **separate kernels on separate CUDA streams**. The first-to-execute kernel writes per-tile signals as blocks complete; the second is a **persistent kernel** that polls signals and processes data as it becomes ready. Coarse ordering via `torch.cuda.Event`, fine-grained overlap via store/load PTX signals on the signal tensor (host `.zero_()` reset before each launch). Barrier level (thread / CTA / rank) is chosen based on the collective.

Read `references/inter_sm.md` for signal primitives, cross-device barrier, host-side launch pattern, and full implementation guide. If the user wants the collective communication portion to use hardware multicast (NVLS), additionally read `references/multimem.md`.

### Intra-SM — fused single kernel

Compute and communication are **fused into one kernel**. Each CTA computes a tile, then directly writes results to peer symmetric memory buffers via load/store (or multimem load/store). Synchronization uses GPU-scope atomics (`atomic_add_release` / `load_acquire`) for intra-node CTA barriers and `barrier_all_intra_node_atomic_cas_block` for cross-rank barriers. If the collective requires reduction (e.g., reduce-scatter), a final local-reduce phase sums contributions from all peers.

Read `references/intra_sm.md` for barrier primitives, A2A push pattern, kernel structure template, and full implementation guide. The default push/reduce path uses `tl.store` / hand-written reduction; if the user wants the collective communication portion to use hardware multicast (NVLS) — i.e. `multimem.st` for the push phase and `multimem.ld_reduce` for the reduce phase — additionally read `references/multimem.md`.

### Without-SM — copy engine (DMA)

Communication is offloaded to the **copy engine**, consuming zero SM resources. A background stream performs `copy_()` on symmetric memory (routed through DMA). A **signal tensor** (written via `cuStreamWriteValue32` / polled via `ld_sys`) lets the compute kernel poll and process data as it arrives — local data first (no wait), then remote chunks as they become ready. The signal tensor shape is `[world_size * splits_per_rank]`, where `splits_per_rank=1` gives per-rank granularity and `splits_per_rank>1` gives finer per-chunk granularity.

Read `references/without_sm.md` for copy engine setup, progress tracking, compute kernel polling pattern, and full implementation guide.

## Common Dependencies

All three modes use PyTorch symmetric memory:

```python
import torch.distributed._symmetric_memory as symm_mem

buf = symm_mem.empty(shape, dtype=dtype, device=device)       # allocate
hdl = symm_mem.rendezvous(buf, group=dist.group.WORLD)        # rendezvous
peer_buf = hdl.get_buffer(peer_rank, sizes=shape, dtype=dtype,
                        storage_offset=offset)               # peer access
signal_pad_ptrs = torch.tensor(                                    # cross-rank sync
    [hdl.get_signal_pad(i, (world_size,), torch.int32).data_ptr()
     for i in range(world_size)],
    dtype=torch.int64, device=device,
)
```

### Symmetric Memory Shape Selection Principle

The symmetric memory buffer (`symm_mem.empty(shape, ...)`) is primarily used for the **communication** phase — it is the medium through which data is exchanged between ranks. Therefore, its shape should match the shape required by the collective communication operation, not the compute kernel's input/output shape. This avoids unnecessary memory waste and offset misalignment.

Choose the shape based on the collective:

| Collective | symm_mem shape per rank | Rationale |
|-----------|------------------------|-----------|
| **All-Gather** | `[M_per_rank, N]` (each rank's local contribution) | Each rank writes its local data into its own symm_mem buffer; peers read from it. The gathered output `[M, N]` is assembled from all ranks' buffers. |
| **Reduce-Scatter** | `[M, N]` (i.e., `[world_size * M_per_rank, N]`) | Each rank's buffer receives partial results from all peers (one `M_per_rank`-row segment per source rank), then reduces locally. The buffer layout is logically `[world_size, M_per_rank, N]`. |
| **All-Reduce** | `[M, N]` (full data shape) | Each rank's buffer holds the full tensor; peers write their contributions and a reduction is performed in-place or into a separate output. |

The key insight: symm_mem shape = the shape that the communication pattern needs to read/write across ranks. The compute kernel's input/output may have a different shape (e.g., the compute input might be `[M, K]` while the symm_mem buffer for reduce-scatter is `[M, N]` where N is the output dimension after compute).

## Selection Guide

| Mode | When to use | Pros | Cons |
|------|------------|------|------|
| **inter-sm** | Compute and comm have different granularity; need flexibility | Clear separation, easier to debug | Two kernels, stream overhead |
| **intra-sm** | Tile-aligned data, same CTA can do both | Zero extra SM cost, tightest overlap | Complex kernel, register pressure |
| **without-sm** | Large contiguous transfers, want zero SM overhead for comm | Copy engine is free, simple compute kernel | Coarser granularity, needs large transfers |

---

## Output Format Convention

The generated overlap kernel file is organized into **four sections** in order, each separated by comment banners:

1. **PTX Primitives** — `@triton.jit` inline-asm helpers (signal, barrier, memory ordering)
2. **Context Dataclass** — `@dataclass` + `create_*_context()` factory for symmetric memory setup
3. **Triton Kernel** — the `@triton.jit` kernel function(s) implementing the overlap logic
4. **Python Entry Point** — host-side launcher that resets barriers and calls the kernel

Read `references/kernel_format.md` for the complete code templates, design rules, and available PTX primitives catalog.

---

## Workflow Steps

After gathering user input and selecting the overlap mode, execute these steps **in order**:

### Step 1: Generate Overlap Kernel

Read the appropriate reference file for the selected mode, then generate a complete kernel file following the four-section format above. **Before emitting any PTX primitives or barrier calls, read `references/primitives.md`** to use the canonical implementations — do not re-derive them from the mode files. The file should be self-contained (no imports from other overlap implementations in this repo except standard libraries: `torch`, `torch.distributed`, `triton`, `triton.language`).

**Multimem (NVLS) trigger — load `references/multimem.md` on demand.**

Additionally read `references/multimem.md` **before** generating the kernel when the user select the Communication mechanism as `nvswitch`.

When triggered:
1. Read `references/multimem.md` in addition to the selected mode file.
2. **Inline the implementation source from `multimem.md` §D** (helpers + `multimem_st` + `multimem_ld_reduce` + their `_v*` impls) into the generated kernel file's PTX Primitives section.
3. On the host side: extend the context dataclass with `mc_ptr`, access `hdl.multicast_ptr` (stock PyTorch symm_mem property, no extra import) after `symm_mem.rendezvous`, and guard against unsupported hardware (`mc_ptr == 0`).
4. In the kernel: replace the per-peer `tl.store` push loop with `multimem_st(...)`, and replace the hand-written `acc += tl.load(peer_buf, ...)` reduction loop with `multimem_ld_reduce(..., OP="sum", ACC_DTYPE=tl.float32)`. Keep all existing grid-level and cross-rank barriers — multimem swaps out only the bulk data movement, not the synchronization.

When **not** triggered: stay on the default path (per-peer `tl.store` for the push phase, hand-written reduction for the reduce phase) and do **not** load `references/multimem.md`.

**FP8 GEMM trigger — load `references/triton_fp8_gemm.md` on demand.**

Additionally read `references/triton_fp8_gemm.md` **before** generating the kernel when the compute kernel in the overlap is a **GEMM / matmul that operates on FP8 quantized inputs** (e.g., AllGather + FP8 GEMM, reduce-scatter + FP8 GEMM, linear layers with FP8 weight). This covers:

- **Keyword trigger** — the user's request mentions any of: `fp8`, `block-wise fp8`, `w8a8`, `fp8 gemm`, `fp8 matmul`, `block fp8`.
- **Intent trigger** — the user describes a quantized GEMM workload (e.g., "weight is in fp8", "block-wise quantized matmul", "fp8 weight and activation with per-block scaling", "quantize activation on the fly inside GEMM", "fuse bf16 to fp8 quantization into the matmul kernel").

When triggered:
1. Read `references/triton_fp8_gemm.md` in addition to the selected mode file and `references/primitives.md`.
2. Inline the `_w8a8_block_fp8_matmul` kernel code from the reference into the generated file. The generated file must remain self-contained.
3. Choose the activation quantization strategy based on the overlap pattern (see `references/triton_fp8_gemm.md` §C.1):
   - **Separate quantize kernel + pre-quantized `_w8a8_block_fp8_matmul`**: run a `per_token_group_quant_fp8` kernel to produce `A_fp8` and `A_scale` first, then feed them into the standard kernel. Use this when exact per-`group_k` quantization granularity is required, or when A is already available in fp8 format.
   - **On-the-fly quantization inside the GEMM kernel**: modify the K-loop to load bf16 A, compute per-row `absmax`, cast to fp8 inline, then `tl.dot(fp8_A, fp8_B) * scale * B_scale`. Use this when A arrives as bf16 from symmetric memory and you want to save one kernel launch + one global memory round-trip. This is the recommended default for overlap kernels. When `BLOCK_SIZE_K == group_k == 128` (standard setting), numerical accuracy is equivalent to the separate-kernel approach.

When **not** triggered: if the compute kernel uses bf16/fp16 natively (not block-wise FP8), do **not** load `references/triton_fp8_gemm.md`.

### Step 2: Overlap Kernel Correctness & Hang Check

After generating the kernel, produce a **verification checklist** and verify each item:

#### Hang Prevention Checks

| Check | What to verify |
|-------|---------------|
| **Grid barrier correctness** | `barrier_on_this_grid` is called by ALL CTAs before cross-rank sync; the barrier counter is reset to 0 before each kernel launch |
| **Cross-rank barrier single-CTA** | In-kernel `barrier_all_intra_node_atomic_cas_block` is called by **exactly ONE CTA per rank** (guarded by `if pid == 0:`). The signal pad slots are binary (0↔1) and can only track one barrier invocation per rank pair; multiple CTAs calling it concurrently will CAS-contend on the same slots and **deadlock**. For inter-sm and without-sm modes, do NOT use this in-kernel barrier — use host-side `symm_mem_hdl.barrier()` instead (see `primitives.md` §5.4) |
| **Cross-rank barrier symmetry** | Every rank executes the same number of barrier calls; no conditional paths that skip barriers on some ranks. The `if pid == 0:` guard does NOT violate this — all ranks have a pid==0 CTA |
| **Signal reset** | Signal tensor is reset by host `.zero_()` before each launch; `_send_signal` stores 1, `_wait_signal` spins until 1 — no in-kernel reset needed |
| **No deadlock ordering** | If multiple barriers exist, all ranks execute them in the same order; no circular dependencies between barrier phases |
| **Persistent kernel termination** | For inter-SM persistent kernels: the polling loop has a bounded exit condition (tile counter reaches total_tiles) |
| **Thread convergence** | `tl.debug_barrier()` is called after any warp-divergent section to re-converge threads before shared operations |
| **Triton AST compatibility** | Triton JIT does NOT support `break`, `continue`, or `return` inside `@triton.jit` functions. Replace `while True: ... if cond: break` with `while not cond:` pattern. All loop exits must be expressed as `while <condition>` |

#### Precision Checks

| Check | What to verify |
|-------|---------------|
| **Accumulator dtype** | Reductions (sum, mean) use `tl.float32` accumulator even if input is fp16/bf16 |
| **Cast placement** | Cast to native dtype happens AFTER all arithmetic, immediately before `tl.store` |
| **Scaling factor** | Multiplicative scaling (e.g., `1/topk`) is applied in float32 before final cast |
| **Overflow in offsets** | All pointer offset calculations use int64 when `M * stride` could exceed int32 range (~2B elements). For Triton tensors (e.g., `tl.arange`, `tl.load` results), use `.to(tl.int64)`. For values that may be Python `int` at compile time (e.g., results of `//` or `%` on `tl.constexpr` args, loop variables from `tl.static_range` or `range`), use `tl.cast(val, tl.int64)` instead — Python `int` has no `.to()` method and will raise `AttributeError` |
| **tl.cast vs .to() usage** | `.to(tl.int64)` is only valid on Triton tensor values (results of `tl.load`, `tl.arange`, tensor arithmetic). For scalar values that may be Python `int` at JIT time, always use `tl.cast(val, tl.int64)`. Mixing these up causes `AttributeError: 'int' object has no attribute 'to'` at kernel compile time |
| **Mask correctness** | Boundary masks cover both M and N dimensions; `other=0.0` is used for masked loads in reductions |
| **Output copy shape** | Ensure output shape matches collective semantics: <br>- **AllReduce**: same as input (full data, reduced) <br>- **AllGather**: concatenated across ranks <br>- **ReduceScatter**: local portion only |

#### Memory Safety Checks (Illegal Memory Access Prevention)

These checks prevent `CUDA error: an illegal memory access was encountered`. This is the most common runtime crash in Triton kernels and almost always means some thread/lane is dereferencing a pointer outside allocated memory.

| Check | What to verify |
|-------|---------------|
| **Power-of-2 lane overflow** | `tl.arange(0, BLOCK_X)` always produces a power-of-2-length tensor. When `BLOCK_X > actual_size` (which is the norm — `BLOCK_X = next_power_of_2(actual_size)`), excess lanes exist. For standard `tl.load`/`tl.store` this is handled by the `mask` parameter. But **any operation that bypasses Triton's mask mechanism — PTX `inline_asm_elementwise`, raw pointer arithmetic passed to inline asm, manual `tl.store` without mask — will execute on ALL lanes including out-of-bounds ones**. Every pointer derived from `tl.arange` that feeds into an unmasked operation must be guarded |
| **Unmasked PTX inline_asm** | PTX inline assembly (`tl.inline_asm_elementwise`) has no built-in mask support. If it receives a tensor of addresses, ALL addresses are dereferenced unconditionally. Two valid fix patterns: (1) **Address clamping** — `addr = tl.where(mask, real_addr, safe_addr)` before calling the asm, where `safe_addr` is a known-valid address (e.g., element 0). (2) **Predicated execution** — pass mask as an integer arg and use `setp + @%p` guard inside the PTX asm so the instruction only fires for valid lanes |
| **Tile boundary overflow** | When tiling a dimension (M or N), the last tile often extends past the real extent. Verify: (a) loads from input use `mask` + `other=0.0`, (b) stores to output use `mask`, (c) pointer offsets for the last tile don't produce negative values or wrap around due to unsigned arithmetic. (d) For compact layout where each rank occupies exactly `M_local` rows (no padding to `BLOCK_SIZE_M`), tile row offsets must account for the real per-rank size, not assume uniform tile-aligned strides across ranks |
| **Cross-buffer pointer confusion** | Kernels with multiple buffers (input, output, symm_mem, signal pads) must not mix up strides or base pointers. Verify each pointer offset expression uses the stride that matches its target buffer. Common bug: applying `stride_in_m` to `symm_buffer` offsets when `stride_buf_m` is different |
| **Byte-level pointer arithmetic** | When computing byte offsets for raw pointers (common with multicast/DMA), verify: (a) multiplication by `ELEM_BYTES` is applied consistently, (b) division back to element-level (e.g., `// 4` for int32 reinterpret) is correct, (c) alignment requirements (16B for v4 ops) are satisfied by the computed addresses |
| **Pointer array bounds** | Arrays like `buf_ptrs[world_size]` or `signal_pad_ptrs[world_size]` have fixed length. Any index used to load from them must be provably `< array_length`. Verify loop bounds and conditional indices |
| **Chunk/split divisibility** | When splitting a dimension into chunks (`N_per_chunk = N // N_CHUNKS`), the division must be exact. Non-exact division causes: the last chunk to be short (missing mask → OOB reads), or `BLOCK_X = next_power_of_2(N_per_chunk)` to overshoot into the next chunk's address space. Enforce `assert dim % num_chunks == 0` on the host side |
| **Dynamic vs allocated size mismatch** | If a buffer is pre-allocated for `max_M` rows but the kernel is called with `M < max_M`, this is safe (under-utilization). But if `M > max_M` at runtime, stores will exceed the buffer. Enforce `assert M <= ctx.max_M` on the host side. Same applies to N dimension |
| **Grid-stride loop with raw pointers** | In persistent kernel patterns (`for tile_id in range(pid, total, npid)`), each iteration computes new pointer offsets. Verify that `total` is correctly derived from the actual tensor shape, not from an unrelated `BLOCK` constant. Off-by-one in `total` means the last CTA accesses memory one tile beyond the allocation |

**Fix patterns:**
- **Address clamping for inline asm**: `safe_addr = tl.where(mask, real_addr, base_addr)` — redirect OOB lanes to element 0
- **Predicated PTX**: Add `setp.ne.s32 %p0, $mask_arg, 0; @%p0 <instruction>` inside the asm string; pass `mask.to(tl.int32)` as an extra arg
- **Host-side guards**: `assert N % n_chunks == 0`, `assert M <= ctx.max_M`, `assert input.shape[-1] == ctx.N`

#### Performance Checks

| Check | What to verify |
|-------|---------------|
| **num_warps=32** | The kernel launch specifies `num_warps=32` (not the default 4). These kernels are memory-bound; 32 warps (1024 threads/CTA) is needed to saturate memory bandwidth. Using default 4 warps causes 50%+ performance loss |
| **num_stages=1** | Explicitly set `num_stages=1`. No software pipelining is needed for simple load/store patterns |
| **N_CHUNKS splitting** | The N dimension is split into chunks (default N dimension // 1024) via an outer `tl.range` loop to reduce register pressure and improve SM occupancy |
| **tl.multiple_of hints** | `tl.multiple_of(N_per_chunk, 16)` and `tl.multiple_of(peer_buf, 16)` are used to enable vectorized 128-bit loads/stores |
| **tl.constexpr strides** | Stride parameters are declared as `tl.constexpr` so the compiler can optimize address calculations at compile time |
| **Incremental pointer update** | topk/reduction loops use `tl.static_range` + `ptr + j * stride` instead of recomputing full addresses each iteration |
| **Input layout flattening** | 3D inputs like `[M, topk, N]` are flattened to `[M*topk, N]` for contiguous memory access in the topk dimension |
| **Master CTA barrier** | `barrier_on_this_grid` uses the master CTA pattern (bid==0 spins, others wait for high bit) with a single `tl.debug_barrier()`, not the naive 3-barrier pattern (see `primitives.md` §5.2) |
| **Multi-threaded CAS barrier** | `barrier_all_intra_node_atomic_cas_block` uses `flat_tid < world_size` for parallel peer signaling, not single-threaded serial iteration (see `primitives.md` §5.3) |
| **Cross-rank barrier placement** | In-kernel `barrier_all_intra_node_atomic_cas_block` is only needed when the kernel has a subsequent phase after communication (intra-sm). For inter-sm and without-sm, use the host-side `barrier() → signal.zero_() → barrier()` pattern (two barriers flanking the reset, not just one) instead — the comm kernel exits once communication is done, so no in-kernel cross-rank sync is required. See `primitives.md` §5.3 for the in-kernel barrier and §5.4 for the host-side barrier |
| **PTX-level CAS spin-loop** | `_cas_sys_release`/`_cas_sys_acquire` have the spin-loop inside PTX (`@!%p0 bra cas_loop`), not in Python-level `while` loops (see `primitives.md` §3) |
| **Auto block size** | `BLOCK_M`/`BLOCK_N` default to `None` and are auto-computed based on problem size: `BLOCK_N = next_power_of_2(N_per_chunk)`, `BLOCK_M = next_power_of_2(16K / BLOCK_N / element_size)` |

If any check fails, fix the generated kernel before proceeding.

### Step 3: Generate Benchmark Test File

Generate a test file with four modes: `correctness`, `performance`, `multi_size`, `stability`.

The performance test must separately benchmark:
1. **Compute-only**: the compute kernel in isolation
2. **Comm-only**: the collective (NCCL) in isolation
3. **Non-overlap**: compute + comm sequentially (no overlap)
4. **Overlap**: the fused overlap kernel

This enables calculating **speedup** = `(compute_time + comm_time) / overlap_time` and **overlap efficiency** = `max(compute_time, comm_time) / overlap_time` (how close to the ideal of fully overlapping compute and comm).

**Compute kernel selection for benchmark (important):** The compute-only and non-overlap benchmarks **must prefer the user-provided compute kernel implementation** for timing, not a PyTorch reference implementation. The overlap kernel internally uses the custom compute kernel (e.g., a Triton kernel), so the non-overlap baseline must use the same compute path to ensure a fair comparison. If the benchmark measures compute-only with PyTorch ops while the overlap uses Triton, the speedup metric becomes misleading. Implementation pattern:

1. If the user provides a compute kernel (path or function), import it from the overlap kernel file and use it directly in `compute_only()` and `non_overlap()`.
2. Keep a separate PyTorch reference implementation for **correctness verification only** (comparing overlap output against ground truth).
3. Only fall back to PyTorch ops for performance benchmarking when no user-provided compute kernel is available.

Read `references/benchmark_template.md` for the complete test file structure, argument conventions, and output format.

---

## Important Constraints

- **No hardcoded paths**: all file paths, module imports, and resource locations must be relative or derived from runtime context (e.g., `os.path.dirname(__file__)`). The skill must work regardless of where the repo is cloned.
- **Self-contained kernel files**: each generated kernel file imports only from standard packages (`torch`, `triton`, `triton.language`, `torch.distributed`). No cross-imports between overlap kernel implementations.
- **Reusable context**: the context dataclass must support repeated kernel launches without re-allocation. Only `grid_barrier.zero_()` and similar resets are needed between calls.

## References

| File | When to read |
|------|-------------|
| `references/primitives.md` | Always — before emitting any PTX helpers, barriers, or signals into the kernel |
| `references/kernel_format.md` | Step 1 — generating the kernel file (4-section code templates) |
| `references/benchmark_template.md` | Step 3 — generating the benchmark test file |
| `references/inter_sm.md` | User selects inter-sm mode |
| `references/intra_sm.md` | User selects intra-sm mode |
| `references/without_sm.md` | User selects without-sm mode |
| `references/multimem.md` | On-demand — only when the user asks the collective communication portion to use multimem / NVLS / hardware multicast (see Step 1 trigger conditions) |
| `references/triton_fp8_gemm.md` | On-demand — only when the compute kernel is a GEMM with FP8 quantized inputs (block-wise W8A8), e.g. AllGather + FP8 GEMM overlap (see Step 1 trigger conditions) |