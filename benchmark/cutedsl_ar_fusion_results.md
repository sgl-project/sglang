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

The stock column holds decode throughput in output tokens for each second. It
divides the batch size by the median inter-token latency, so it excludes the
prefill time.

The other three columns hold the gain of that arm over stock, as a percentage.
Higher is better. Each percentage comes from the median inter-token latency
rather than from the rounded throughput.

### TP8

| batch | stock (tok/s) | stock + bound | cute-dsl | cute-dsl + bound |
|---|---|---|---|---|
| 1 | 169 | +0.0% | +4.6% | **+4.7%** |
| 2 | 326 | +0.0% | +5.4% | **+5.4%** |
| 4 | 592 | +0.0% | +5.2% | **+5.1%** |
| 8 | 1021 | +0.0% | +5.1% | **+5.0%** |
| 16 | 1665 | +0.0% | +2.1% | **+2.3%** |
| 32 | 2659 | +0.0% | +1.9% | **+2.1%** |
| 64 | 4256 | +0.0% | +2.6% | **+2.5%** |
| 128 | 7176 | +0.0% | +2.4% | **+2.4%** |
| 256 | 10667 | +4.2% | +1.2% | **+4.5%** |
| 384 | 12629 | +8.2% | +1.8% | **+8.0%** |
| 512 | 13978 | +12.2% | +3.2% | **+12.8%** |

The last column ranges from +2.1% to +12.8%. The mean is +5.0%.

### TP4

| batch | stock (tok/s) | stock + bound | cute-dsl | cute-dsl + bound |
|---|---|---|---|---|
| 1 | 150 | +0.0% | +4.7% | **+4.7%** |
| 2 | 286 | +0.0% | +3.6% | **+3.7%** |
| 4 | 503 | +0.0% | +3.9% | **+4.0%** |
| 8 | 842 | +0.0% | +3.9% | **+4.0%** |
| 16 | 1312 | +0.0% | +1.4% | **+1.4%** |
| 32 | 2008 | +0.0% | +1.2% | **+1.3%** |
| 64 | 3174 | +0.0% | +1.6% | **+1.9%** |
| 128 | 5336 | +0.0% | +1.0% | **+1.1%** |
| 256 | 8249 | +3.1% | +0.7% | **+2.5%** |
| 384 | 10163 | +5.7% | +1.1% | **+5.6%** |
| 512 | 11365 | +11.4% | +1.8% | **+11.1%** |

The last column ranges from +1.1% to +11.1%. The mean is +3.7%.

### The bound, measured on cute-dsl at TP8

| batch | defer on (tok/s) | defer off |
|---|---|---|
| 8 | 1017 | -3.75% |
| 16 | 1698 | -8.40% |
| 32 | 2713 | -6.60% |
| 64 | 4369 | -4.41% |
| 96 | 5958 | -6.01% |
| 128 | 7372 | -3.61% |
| 160 | 7832 | -0.71% |
| 192 | 8938 | +1.11% |
| 256 | 10899 | +2.68% |
| 384 | 12850 | +6.47% |
| 512 | 14439 | +9.36% |

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
