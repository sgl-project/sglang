---
name: sglang-diffusion-ako4all-kernel
description: Use when optimizing an existing SGLang diffusion kernel with AKO4ALL, including AKO4ALL repo hygiene, custom microbench setup, ncu-guided iteration, and end-to-end denoise validation. Also use when a sibling AKO4ALL repo must be cloned or refreshed before starting kernel tuning work.
---

# SGLang Diffusion AKO4ALL Kernel

Use this skill to run the full AKO4ALL-based optimization loop for an existing SGLang diffusion kernel.
It is the default implementation path once the benchmark/profile skill has already shown that a hotspot is real and not covered by an existing fast path. This workflow bootstraps a custom AKO harness, benchmarks and profiles the kernel, iterates with `ncu`, ports the best version back to `sglang`, then validates with targeted tests and model-level denoise runs.

This skill assumes a sibling repo layout like:

```text
<base-dir>/
├── sglang/
└── AKO4ALL/
```

If `AKO4ALL/` is missing under the current base directory, clone it first.

## Use This Skill When

- tuning an existing diffusion Triton, CUDA JIT, CuTeDSL, or runtime-integrated kernel in `sglang`
- `sglang-diffusion-benchmark-profile` has already ruled out an existing in-repo fast path or overlap family
- creating a custom AKO4ALL harness for a real diffusion kernel instead of using the default benchmark tasks
- validating that a kernel-level win transfers to Qwen, FLUX, Wan, Hunyuan, MOVA, or other diffusion denoise latency
- preparing PR artifacts such as microbench tables, `ncu` before/after data, and proof image outputs

Do not start here when the bottleneck has not been proven yet.
First use [../sglang-diffusion-benchmark-profile/SKILL.md](../sglang-diffusion-benchmark-profile/SKILL.md) to:
- measure the real denoise regression
- collect the perf dump baseline
- capture one representative `torch.profiler` trace
- rule out existing merged fast paths

If a future specialized optimization skill matches the kernel family better than AKO4ALL, hand off there instead. The diagnosis contract stays the same.

## Mandatory AKO4ALL Preflight

Before any AKO work:

1. Run `scripts/ensure_ako4all_clean.sh [base-dir]`.
2. If `<base-dir>/AKO4ALL` does not exist, the script clones it.
3. Do not continue unless `AKO4ALL` is:
   - on the upstream default branch, usually `main`
   - fully clean with no tracked or untracked local changes
   - exactly synced to `upstream/<default-branch>`
4. If the script reports local commits, divergence, or a dirty worktree, stop and clean or re-clone the repo before continuing.

The script creates an `upstream` remote automatically when missing.
By default it uses the existing `origin` URL, or `AKO4ALL_URL` if you need to override the clone source.

## Workflow

### 1. Scope the Kernel

- Identify the exact kernel entry point and runtime call sites in `sglang`.
- Record the target shapes, dtypes, model families, and whether the kernel is on a hot path.
- Reuse existing unit tests and benchmark entry points when they already exist.

### 2. Bootstrap the AKO Harness

Inside the clean `AKO4ALL` repo:

- read `TASK.md` and `HINTS.md`
- create a custom harness instead of relying on the stock benchmark tasks
- mirror the real SGLang kernel into:
  - `input/reference.py`
  - `input/<kernel>.py`
  - `solution/<kernel>.py`
  - `bench/bench_<kernel>.py`
- keep a short context note in `context/` when the kernel has model-specific shape assumptions or perf conclusions

The custom benchmark should:

- cover representative diffusion shapes
- check correctness against the reference kernel
- report aggregate runtime plus per-shape results when useful

### 3. Establish the Baseline

- run the AKO custom microbench before changing the kernel
- capture one representative `ncu` baseline on the hottest meaningful shape
- note whether the bottleneck looks like registers, occupancy, instruction count, launch config, or memory latency

### 4. Iterate in AKO4ALL

- change one idea at a time
- rerun the microbench after every change
- update `ITERATIONS.md` with hypothesis, result, and next step
- prefer simple, explainable wins over clever rewrites that do not transfer

After 3 consecutive no-improvement or regression iterations:

- rerun `ncu`
- re-read `ITERATIONS.md`
- change direction instead of continuing blind sweeps

### 5. Port the Best Version Back to SGLang

- apply the best candidate to the real `sglang` kernel file
- run import or syntax checks and targeted tests first
- keep the AKO `solution/` version aligned with the main-tree version you actually want to keep

### 6. Validate on Real Models

- use the benchmark/profile skill for denoise perf dumps and before/after comparison
- prefer exact local snapshot validation when testing local edits on a GPU box
- run targeted kernel tests first
- run model-level denoise benchmarks with perf dumps
- compare baseline vs optimized runs with `compare_perf.py`
- if the PR needs proof that generation still works, save one real model output image

### 7. Prepare PR Artifacts

At minimum, keep:

- one microbench table
- one denoise-stage table
- one end-to-end table
- one `ncu` before/after pair on the most representative kernel shape
- one generated image when the kernel affects production inference

See [references/ako-loop.md](references/ako-loop.md) for the checklist and common stop rules.

## Operating Rules

- Treat AKO4ALL repo hygiene as a gate, not a suggestion.
- Prefer exact local snapshot validation over hand-wavy “remote tree is close enough”.
- Keep model-level validation honest: if microbench improves but denoise does not, do not keep the AKO-only variant in the main code path.
- When writing conclusions, explain the win in terms of measurable causes such as lower registers per thread, higher occupancy, fewer executed instructions, or better scheduler eligibility.
