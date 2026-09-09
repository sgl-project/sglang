# Share the CuTe DSL AR fusion core, and bound the deferred MoE finalize

## What this changes

This branch does two separate things.

First, it makes the FlashInfer MNNVL CuTe DSL AllReduce fusion available to more
than one model family. The fusion arrived with the Qwen3.5 stack, and it lived
in a module named after that model. Almost none of its content was specific to
Qwen3.5. This branch moves the mechanism into a shared module and wires the
DeepSeek-V3 family to it, which covers GLM-5.x without further work.

Second, it adds an upper token bound to the deferred MoE finalize. That bound
fixes a performance defect that exists on main today, and the defect is
independent of the fusion.

## The fusion refactor

This branch deletes `layers/moe/qwen35_flashinfer_fusion.py`. Its content
moves to `layers/moe/cutedsl_ar_fusion.py`, which names no model.

The only real difference between model families is the attribute of their
RMSNorm that holds the gamma value. `fused_norm_gamma()` reads that attribute
from the norm module itself. GemmaRMSNorm returns its pre-folded weight, and a
plain RMSNorm returns its weight. The fusion needs no per-architecture subclass.

`--flashinfer-allreduce-fusion-backend cute-dsl` now selects the fusion. This
branch also deletes the environment variable
`SGLANG_FLASHINFER_MNNVL_CUTEDSL_AR_FUSION`, together with the resolution pass
that had to suppress the backend argument when a user set both switches.

## Two defects the refactor exposed

Both live in the Qwen module on main today.

`prepare_attn()` returned early on the deferred path, and it therefore skipped
the tail that publishes `attn_inputs`. Qwen3.5 never noticed, because it does
not use `qkv_latent_func`. DeepSeek does use it, and it asserts. The tail is now
a shared method that both paths call.

FlashInfer selects its kernel routing profile by an exact match on tensor
parallel size, hidden size, top-k and dtype. It ships profiles for hidden size
8192 and top-k 10 only. Every other shape raises "No MNNVL CuTe DSL profile
supports this static shape". `_config_for_shape()` rebuilds one profile at the
running shape from the shipped presets. Hidden size 8192 with top-k 10 still
receives the shipped profile unchanged.

## The deferred finalize defect

A MoE layer expands each token into 8 rows, one for each expert that the router
picks. The finalize collects those 8 rows back into 1 row for each token.

The FlashInfer TRT-LLM kernel normally does that collection inside the epilogue
of its second matrix multiply. The expanded tensor never reaches GPU memory.

A deferred finalize skips that epilogue. It writes the full expanded tensor to
GPU memory instead, and a later kernel reads the tensor again. The later kernel
can absorb the finalize into the all-reduce, which saves one kernel launch for
each layer.

The saving stays about the same at every batch size. The added memory traffic
grows in direct proportion to the batch size, because the expanded tensor is 8
times larger than the layer output. At batch 512 that tensor holds 50 MB for each
layer, or 3.8 GB across the 75 MoE layers.

On main the decision to defer reads four conditions, and none of them looks at
the batch size. SGLang therefore defers at every batch size.

TensorRT-LLM stops deferring above 128 tokens, although that number is a kernel
capacity limit rather than a measured crossover. TokenSpeed stops above 32
tokens, and its comment names that number as a measured profit edge.

This branch adds `SGLANG_MOE_DEFERRED_FINALIZE_MAX_TOKENS`, which defaults to
192. A value of 0 turns the bound off.

## Results

Model: `nvidia/GLM-5.2-NVFP4`, hidden size 6144, top-k 8, 75 MoE layers.
Hardware: 8x B300, single node. Workload: 1024 input and 1024 output tokens,
every request the same length.

The numbers are median inter-token latency in milliseconds. Lower is better.
The speedup column divides the stock time by the cute-dsl-plus-bound time.

### TP8

| batch | stock | stock + bound | cute-dsl | cute-dsl + bound | speedup |
|---|---|---|---|---|---|
| 1 | 5.91 | 5.91 | 5.65 | 5.64 | **1.047x** |
| 2 | 6.14 | 6.14 | 5.82 | 5.83 | **1.054x** |
| 4 | 6.76 | 6.76 | 6.42 | 6.43 | **1.051x** |
| 8 | 7.84 | 7.84 | 7.46 | 7.47 | **1.050x** |
| 16 | 9.61 | 9.61 | 9.42 | 9.39 | **1.023x** |
| 32 | 12.03 | 12.03 | 11.81 | 11.78 | **1.021x** |
| 64 | 15.04 | 15.04 | 14.66 | 14.67 | **1.025x** |
| 128 | 17.84 | 17.84 | 17.42 | 17.42 | **1.024x** |
| 256 | 24.00 | 23.02 | 23.71 | 22.98 | **1.045x** |
| 384 | 30.41 | 28.10 | 29.88 | 28.15 | **1.080x** |
| 512 | 36.63 | 32.66 | 35.48 | 32.47 | **1.128x** |

Range 1.021x to 1.128x. Mean 1.050x.

### TP4

| batch | stock | stock + bound | cute-dsl | cute-dsl + bound | speedup |
|---|---|---|---|---|---|
| 1 | 6.66 | 6.66 | 6.36 | 6.36 | **1.047x** |
| 2 | 7.00 | 7.00 | 6.76 | 6.75 | **1.037x** |
| 4 | 7.95 | 7.95 | 7.65 | 7.64 | **1.040x** |
| 8 | 9.50 | 9.50 | 9.14 | 9.14 | **1.040x** |
| 16 | 12.19 | 12.19 | 12.02 | 12.03 | **1.014x** |
| 32 | 15.93 | 15.93 | 15.75 | 15.73 | **1.013x** |
| 64 | 20.16 | 20.16 | 19.84 | 19.78 | **1.019x** |
| 128 | 23.99 | 23.99 | 23.74 | 23.73 | **1.011x** |
| 256 | 31.04 | 30.11 | 30.81 | 30.28 | **1.025x** |
| 384 | 37.78 | 35.74 | 37.36 | 35.78 | **1.056x** |
| 512 | 45.05 | 40.43 | 44.23 | 40.55 | **1.111x** |

Range 1.011x to 1.111x. Mean 1.037x.

### The bound, measured on cute-dsl at TP8

| batch | defer on | defer off | effect of turning defer off |
|---|---|---|---|
| 8 | 7.87 | 8.17 | -3.75% |
| 16 | 9.42 | 10.28 | -8.40% |
| 32 | 11.80 | 12.63 | -6.60% |
| 64 | 14.65 | 15.32 | -4.41% |
| 96 | 16.11 | 17.14 | -6.01% |
| 128 | 17.36 | 18.01 | -3.61% |
| 160 | 20.43 | 20.58 | -0.71% |
| 192 | 21.48 | 21.25 | +1.11% |
| 256 | 23.49 | 22.87 | +2.68% |
| 384 | 29.88 | 28.07 | +6.47% |
| 512 | 35.46 | 32.43 | +9.36% |

## Where the gain comes from

The two changes win in different places, and they do not overlap.

The fusion wins below the bound, where it is worth about 1.05x at TP8. The bound
wins above it, where it is worth up to 1.12x. The bound does nothing below 192
by construction, and the fusion adds little above it.

At batch 512 the bound supplies about 11 of the 12.8 points. A user on main can
reach most of that today with `SGLANG_ENABLE_MOE_DEFERRED_FINALIZE=0`, and that
switch needs no part of this branch.

## Method

Every server launched with `SGLANG_FLASHINFER_AUTOTUNE_CACHE=0`. Without that
setting one launch inherits another launch's tuning results, and the arms differ
by up to 3.6% for that reason alone.

Each batch size ran 3 discarded warmup passes and 2 recorded passes. The tables
report the median of the 2. The largest spread between the 2 passes was 0.52% at
TP8 and 0.52% at TP4.

The measurement uses median inter-token latency rather than mean time per output
token. The mean includes rare slow steps, and one arm showed an 11% difference
between two passes for that reason. The median of the same two passes differed by
0.4%.

Correctness: GSM8K scores 0.930 to 0.945 across runs, with no invalid outputs.

## What this branch does not cover

- Qwen3.5 has no measurement here. The refactor changes its code path, and this
  branch deletes the Qwen final-norm debug scaffolding.
- The bound of 192 comes from one model at one hidden size. The cost scales with
  batch size, top-k and hidden size, so another model crosses over elsewhere.
- Prefill is not measured. These numbers cover decode only.
- A simple bandwidth model predicts a smaller penalty than the measurement shows,
  and the gap grows with the batch size. The gather reads scattered rows, which
  may explain the gap. No profile confirms that explanation.
