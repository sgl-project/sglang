# Making a JIT kernel fast

`SKILL.md` covers the mechanics of adding a kernel. This file is the short version of what matters once it has to be fast. Unless marked otherwise, observations are from nvcc / ptxas 13.4 on `sm_100a` (B200).

**Measurement decides.** The estimates and rules below are a starting point. Accept or reject every optimization by measured time, read together with the SASS and Nsight Compute counters. A claim like "faster" or "no gain" in a PR needs a measurement behind it.

## Estimate before implementing

Classify each shape bucket the kernel serves, including padded sizes (CUDA-graph batch padding), as low latency or high throughput.

**Low latency** (under one wave, or a few CTAs per SM). Time is dominated by dependency chains, not bandwidth:

- Count the dependent memory round trips on the critical path (read a page table, then the KV; read an index, then gather). Kernel time ≈ round trips × latency. The goal is fewer round trips, not fewer bytes.
- Little's law: bytes in flight = bandwidth × latency. One SM can keep only so much in flight, so even small problems often run faster when split across more CTAs.
- Issue independent loads (weights, constants) as early as possible, before the dependency chain starts. Pair prefetching with wide (128-bit) accesses, or the narrow loads fill the LSU queues (`lg_throttle`).
- Stores leave an SM at about 32 B/cycle. A large output written by a few SMs adds a drain time after the last store.

**High throughput** (several waves, or a persistent kernel). Divide the work of each resource (HBM / L2 bytes, shared-memory bytes, tensor-core time, issue slots, per-pipe cycles, store rate) by its rate; the largest term is the bound. Each producer-consumer pipeline also needs enough data in flight: with S stages of B bytes and fetch latency L, the rate is at most S · B / L. Compute residency with the occupancy API (`runtime::get_blocks_per_sm`), not by dividing totals.

**Use the estimate to steer the work.** A kernel takes at least the larger of two terms: its critical path (dependent memory round trips, dependent compute, synchronization hops, launch and prologue) and its most loaded resource (work divided by rate, per GPU and per SM). The larger term says what to optimize: shorten the chain, or cut or spread the work on the bound resource. Drop a design whose bound is already worse than the baseline before writing it. Compare each measurement with the bound: a large gap means the kernel does not do what the design says, or the estimate misses an effect, and either is the next thing to look at; a small gap means little is left to gain on this design.

**Cross-CTA reduction**, from lowest to highest latency: a cluster with DSMEM (size capped, must fit in one GPC); global atomics (last-arriving CTA reduces; floating-point atomics are nondeterministic); a separate kernel.

When two implementations serve different shape ranges, write each one's time as a function of the dispatched dimension and solve for the crossover. A rule derived that way carries over to shapes that were not measured; a table of measured thresholds does not.

### Reference numbers

Throughputs measured with microbenchmarks on one B200 (148 SMs, SM clock about 1.96 GHz, CUDA 13.4). They are provisional starting points for an estimate; re-measure when a decision depends on one.

- **FMA:** FFMA / FADD / FMUL issue 1 warp-instruction per cycle per SMSP (128 lane-ops per cycle per SM). Packed FFMA2 / HFMA2 do the same lane-ops in half the warp-instructions, so they save issue slots, not math.
- **MUFU** (`ex2`, `rcp`, `rsqrt`, `tanh`, ...): 1 warp-instruction per 8 cycles per SMSP, one sixteenth of the FMA rate, shared by all MUFU functions.
- **LDG:** up to 128 B/cycle per SM, further bounded by HBM at about 7.05 TB/s per GPU.
- **LDS / STS:** 128 B/cycle per SM when conflict-free (on Blackwell, STS.128 only reaches 112). LDG returns share this 128 B/cycle path with shared-memory traffic.
- **Bytes in flight:** about 32 KB per SM reaches about 90 % of the HBM read peak. Warps and ILP are interchangeable at equal bytes in flight.
- **256-bit LDG** has the same bandwidth as 128-bit at every level; it only saves instructions.
- **STG:** 32 B/cycle per SM at any width; HBM write about 6.3 TB/s per GPU. Observed: a short burst of stores costs little when compute follows it, while a long burst adds its drain time; overwriting a store's source registers right after issuing it can stall. Write whole 32-byte sectors: partial sectors lose 30-63 % of HBM write bandwidth.
- **The L1 knee:** when the 128-byte lines requested by LDG in flight on one SM exceed what L1 can hold, every LDS / STS / SHFL on that SM waits about a global-memory latency. With a large shared-memory carveout this already happens at 32-40 KB of contiguous LDG.128 in flight.
- **Other parts:** H200 matches these FMA, MUFU, LDS and STG numbers, except that packed bf16x2 / f16x2 FMA runs at twice the B200 rate and STS.128 reaches the full 128 B/cycle. B300 matches B200 except that `MUFU.EX2` runs at twice the rate.

## Read the SASS

Log registers, spills and stack frame with `SGLANG_JIT_LOG_RESOURCE_USAGE=1 SGLANG_JIT_FORCE_RECOMPILE=1`. Read the hot loop with `cuobjdump -sass` or the Nsight Compute source view. Check for:

- no spills and a 0-byte stack frame. A register array punned to a wider type, or indexed at run time, ends up on the stack;
- global loads and stores at the intended width (128-bit, or 256-bit on SM100). A generic `LD.E` instead of `LDG` splits a 32-byte access in two. It appears when a pointer passed through a shuffle or shared memory. It also appears for `__grid_constant__` struct pointer fields loaded after an asm with a `"memory"` clobber, such as `PDLWaitPrimary`; fix that by wrapping the pointer in `device::assume_global` after the asm (from `utils.cuh`);
- loads issued ahead of their use. The compiler does not hoist a load above a dynamic branch (it might fault), so prefetch manually across data-dependent loops;
- no `BSSY` / `BRA` around short per-element conditionals. Select the operand, not the result: `f(c ? a : b)` stays branch-free, `c ? f(a) : b` branches once `f` is longer than a few instructions;
- no lane loop (`ELECT` / `BRA.U.ANY`) around single-thread ops (TMA, `tcgen05.mma`, `tcgen05.commit`). Guard them with `warp::elect_one_lane()`, not `threadIdx.x == 0`.

For a stall, separate waiting from contention by the Nsight Compute stall reason; more concurrency hides the first and worsens the second:

| stall reason | cause |
|---|---|
| `long_scoreboard` | waiting on global memory (a dependent chain) |
| `short_scoreboard` | waiting on shared memory or MUFU |
| `lg_throttle` / `mio_throttle` | full LSU / MIO queues (too many narrow memory instructions) |
| `wait` | fixed instruction latency |

Fix in this order, and stop at the first that works: restructure the source (compile-time constants, `__restrict__`, integer types, a manual hoist); then a header-library helper or intrinsic; then single-instruction inline PTX, after checking the PTX ISA. Never rely on undefined behaviour.

Other compiler facts worth knowing:

- The compiler never reorders floating-point accumulation, even with `--use_fast_math`. Multiple accumulators must be written by hand. FMA contraction is on by default.
- Making trip counts and tile sizes template arguments lets loops fully unroll, which is why JIT kernels template them (see the Conventions in `SKILL.md`).
- Shared-memory rows accessed with vectors need a compile-time pitch that is a multiple of the access width. When each lane reads its own row, a pitch that is an odd multiple of 16 B avoids bank conflicts. Check the padding against both the producer's and the consumer's access pattern.
- To match a Triton reference bitwise, use the same approximate operators it lowers to (`ex2.approx`, `div.full`). For example, `a * rcp.approx.ftz(b)` matches `div.full.f32` for normal operands.

## SM100 (tcgen05, TMA, clusters, PDL)

- **TMEM:** any `tcgen05` instruction in the kernel (even on a branch that never runs) limits it to one CTA per SM until `tcgen05.relinquish_alloc_permit`. Allocate once at kernel start and deallocate at the end. Fence the address handoff with `tcgen05.fence::before_thread_sync`, then a barrier, then `fence::after_thread_sync`.
- **`setmaxnreg`** needs `__launch_bounds__(threads, min_blocks)` with both arguments; otherwise ptxas silently drops it. Issue it at the top of each role's branch, in one `if` / `else if` chain on the warp role. Code after paths with different budgets merge runs under the smaller budget.
- **Tensor maps** are `const __grid_constant__ CUtensorMap` kernel parameters, each in its own parameter. Issue `prefetch.tensormap` at kernel entry. Use one `mbar_arrive_expect_tx` per stage, not one per copy.
- **MMA loop:** build descriptors once and advance them by adding the address offset. Commit each stage to its "empty" barrier, and do not wait for MMA completion between stages. `.ws` shapes allow only N = 64 / 128 / 256; any other N is an illegal instruction at run time.
- **PDL:** everything that does not read the previous kernel's output can go before `PDLWaitPrimary()`. In general, call `PDLTriggerSecondary()` late, just before the final stores. An early trigger can make the next kernel pack its CTAs onto the few free SMs and run slower than without PDL.
- **Clusters** larger than 2 leave SMs idle on B200 / B300, because GPCs have uneven SM counts. Count clusters per wave with `cudaOccupancyMaxActiveClusters`. Pass the cluster size as a template argument and read the rank from `%cluster_ctarank`.

## Comparing variants fairly

`.claude/rules/kernel-benchmark.md` covers L2 reuse. In addition:

- **Quick round after every change:** a correctness check and timing at a few representative shapes. Change one thing at a time.
- **Build all variants first** with `parallel_compile` (see `SKILL.md`); serial first-use compiles otherwise dominate iteration.
- **Correctness:** compare against the production kernel. Require equivalence within a justified tolerance, not bitwise, unless bitwise is needed. Cover tails, empty and length-1 inputs, and the largest size. A quantity fed back into its own input (recurrent state, KV written then read) needs a chained-drift test over many steps.
- **Check what actually ran:** the import path of the package (an editable install may point at another checkout), and which variant a dispatcher picked.
- **Timing:**
  - Time A and B interleaved in the same process conditions.
  - Keep profilers (CUPTI, torch.profiler) out of timing processes, since they slow later graph replays.
  - Autotune in a separate process.
  - For cold-cache numbers, rotate outputs and workspaces too, not just inputs.
  - Watch for clock throttling under sustained load.
- **PDL chains:** a kernel that overlaps under PDL is timed in its real producer -> kernel -> consumer chain, with the producer triggering where production's does. Compare whole chains against a baseline that includes every op the new kernel replaces.
- **Realistic inputs** where the time depends on values (routing, top-k, variable lengths).
