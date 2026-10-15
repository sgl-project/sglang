# Overlap Heuristics

This analyzer is intentionally conservative.

## What Comes From Which Trace

### Mapping trace

Used for:

- `kernel -> cpu_op -> python scope`
- launch-site call chains

This trace should be easier to read, even if it is not the exact final serving schedule.

### Formal trace

Used for:

- hidden ratio
- exclusive ratio
- overlap headroom
- ASCII timelines

This trace should reflect the real serving shape.

## What It Treats As Hidden

A kernel is treated as hidden for a segment if:

- it is active during that segment
- at least one kernel on a different stream is also active

If the overlapping kernel is compute-like, the analyzer separately records that it is hidden under compute.

## Category Heuristics

The analyzer classifies kernels by name:

- `compute`: GEMM, attention, cutlass, cublas, Triton matmul-like kernels
- `communication`: NCCL, all-reduce, reduce-scatter, all-gather, DeepEP dispatch/combine
- `elementwise`: sigmoid, top-k, gate, rmsnorm, layernorm, rope, casts
- `memory`: memcpy, memset, fill, copy
- `other`: everything else

These categories are for prioritization only.

## How To Read The Action Table

The overlap-opportunity table is intentionally not a full kernel dump.

It only keeps rows that already have an action-oriented label:

- `headroom`
- `low-roi-hidden`

It also prunes very small `headroom` rows after prioritization.

- if a `headroom` row would end up as `P5` because it is below the default `1%` share bar, it is omitted from the table
- `low-roi-hidden` rows can still remain even when they are small, because they are useful as "do not chase this first" signals

### `headroom`

Interpretation:

- the kernel still spends meaningful time exposed in the formal trace
- the mapped Python scope is a good place to inspect scheduling or fusion opportunities
- the dependency signal should still be checked before treating it as a serious overlap candidate

### `low-roi-hidden`

Interpretation:

- the kernel is already mostly hidden by another stream
- optimizing it in isolation is less likely to move end-to-end latency
- focus on fusion, launch reduction, or the surrounding schedule instead

## Dependency Signal

The table includes a dependency-oriented adjacency signal from the formal trace.

It is built from the nearest previous and next kernels on the same stream plus the mapping-trace source attribution.

Communication kernels are treated more conservatively than before:

- if a tight adjacent kernel looks like a likely producer or consumer, the table will raise the dependency risk even when the Python scope names differ
- this avoids over-claiming that an all-reduce-like kernel is a clean overlap candidate just because its neighbors map to different functions

Typical labels:

- `serial risk low`: adjacent kernels do not look like a tight same-code serial chain
- `prev-side serial risk`: the previous adjacent kernel looks tightly tied to the same code path
- `next-side serial risk`: the next adjacent kernel looks tightly tied to the same code path
- `both-side serial risk`: both sides look like a tight serial chain
- `adjacency unclear`: the timing is tight but source attribution is too weak to trust a stronger claim

Treat this as a strong heuristic, not proof of dataflow.

The readable table compresses those into shorter labels:

- `low`
- `high`
- `unclear`

The recommendation labels are also intentionally short:

- `try overlap`
- `try fusion`
- `check deps`
- `skip overlap`
- `manual check`
- `observe later`

## Important Limits

- A trace shows what overlapped, not what could legally overlap.
- Two kernels on different streams do not prove they are dependency-free.
- A mapped Python scope is a launch-site clue, not the only relevant code location.
- A hidden kernel can still matter if it changes occupancy, launch count, or surrounding schedule.
