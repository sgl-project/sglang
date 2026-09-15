# Startup weight-load overlap checks

Reusable checks for loading real weights **after** CUDA graph capture. These are
manual tests, not a new GPU CI suite or a model × dtype × TP/EP matrix. They use
small tensors or a locally generated checkpoint; no model download is needed.

## When to run which check

| Change | Check | What it covers |
| --- | --- | --- |
| Admission, fallback, storage validation, or startup ordering | [CPU unit tests](../../registered/unit/model_executor/model_runner_components/test_startup_weight_load.py) | Decisions, failure paths, and storage/constant checks |
| GDN converted/aligned parameter caches | `test_gdn_weight_refresh.py` | Real FlashInfer decode replay observes refreshed values without changing cache addresses |
| MLA weight-derived tensors | `test_mla_reload.py` | Repeated postprocess, stable storage, and replay parity for unquantized and 128x128 block-FP8 weights |
| CPU offload, device staging, or mixed-device validation | `test_cpu_offload.py` | V1 partial offload, real H2D copies, BF16/block-FP8 replay parity, and postprocess staging |
| Loader/runner lifecycle integration | `test_cpu_offload_engine.py` | Two engine boots, serial/forced-overlap parity with V1 offload, repeated batches 1 and 3 |
| MegaMoE packing or MoE LoRA wrapper ownership | Explicit repros below | Known unsupported paths, **not** passing support tests |

Adding a model that reuses an existing loading/postprocess implementation does
not require copying these tests or expanding a model matrix. Keep its normal
model correctness coverage. If it introduces a new packing step, weight copy,
wrapper, or storage owner, exercise that mechanism and add a focused regression
only where existing checks do not cover it. A new launcher/worker lifecycle also
needs a representative engine comparison; a layer test cannot cover that wiring.

Manual tests do not run automatically. In a PR that changes one of these paths,
record the relevant command, GPU/backend, result, and any skipped checks. A model
or dtype name alone does not establish that a particular kernel path ran.

## Run

Use an installed SGLang development environment and run from the repository root.
The GPU checks were developed on an H100 with CUDA, FlashInfer, and `sgl-kernel`.
Other GPUs must support the selected kernels; a skipped test is not coverage.
Tests sharing process-global runtime state should not be run concurrently in one
process. Run engine tests separately from layer tests.

```bash
# Existing CPU mechanism regressions (no GPU).
python -m pytest -q \
  test/registered/unit/model_executor/model_runner_components/test_startup_weight_load.py

# Small GPU layer checks; no server or model download.
CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
  test/manual/startup_weight_load/test_gdn_weight_refresh.py \
  test/manual/startup_weight_load/test_mla_reload.py \
  test/manual/startup_weight_load/test_cpu_offload.py

# Generated BF16 Qwen MoE checkpoint, serial vs overlap, with CPU offload.
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python \
  test/manual/startup_weight_load/test_cpu_offload_engine.py -v
```

The existing [registered engine test](../../registered/model_loading/test_startup_weight_load.py)
continues to cover the ordinary dense-model path in CI. It uses a downloaded
model and is separate from these download-free manual checks.

## What a reload check must establish

1. Start with capture-safe weights, run the real postprocess, and capture.
2. Load distinct real values through the relevant loader and repeat postprocess
   and cache refresh. Compare with fresh serial loading of those same values.
3. Check graph-visible storage, layout, aliases, and scalar constants as well as
   replay output. Unchanged pointers alone cannot detect stale or mispacked values.
4. Replay more than once. Lifecycle changes must also complete before serving;
   use the CPU failure-path tests and a representative engine check for that.

The reference path must itself work without overlap. Existing serial limitations
are not a reason to add an overlap-only configuration gate.

## Deferred support

The following paths still need overlap-specific handling: HPC-Ops block-FP8
scale caches, non-routed TRT-LLM unquantized MoE packing (including unquantized
experts in FP8 checkpoints), MLA non-128x128 block-FP8 scales, DCP replicated-Q
buffers, and EAGLE token-mapped target heads. Auto mode selects serial loading;
forced overlap rejects them before capture-safe preparation. Ordinary dtype and
parallelism validation remains in the normal startup pipeline.

## Known-limit reproductions

These files deliberately do not match `test_*.py` discovery. Run them explicitly:

```bash
# SM90/H100 only: unchanged addresses but incorrect packed-weight replay.
CUDA_VISIBLE_DEVICES=0 python \
  test/manual/startup_weight_load/repros/repro_megamoe.py -v

# Real MoE LoRA wrapper and expert loader: names and postprocess ownership.
CUDA_VISIBLE_DEVICES=0 python \
  test/manual/startup_weight_load/repros/repro_moe_lora.py -v
```

**PASS means the known limitation was reproduced, not that overlap is supported.**
After implementing support, replace the corresponding reproduction with a
positive serial/replay regression; do not preserve the old failure expectation.

- MegaMoE's one-time packing flag can skip packing newly loaded canonical
  weights. Support must handle repeated and partial loads without double-packing
  untouched layers. The repro uses a packed-layout reference consumer, not the
  fused DeepGEMM MegaMoE kernel.
- MoE LoRA wrappers change expert parameter paths and postprocess ownership.
  The repro uses minimally initialized real classes, not adapter inference.
  Restoring the loader view is not sufficient evidence for every runner or UNO;
  a real adapter/captured-forward check remains necessary.
- UE8M0 conversion, other online/packed quantization, and group/shared-memory/meta
  offload have no GPU support proof here. They remain separate loading-lifecycle
  work; an H100 check cannot establish a different GPU's conversion path.

## Coverage limits and development record

After merging main `cebca698e` on 2026-09-13, all five retained GPU layer checks
and the CPU-offload engine comparison passed again on H100, with no skips.
The environment below was reused with `sgl-deep-gemm` 0.1.7 added as required by
main. The engine comparison retained exact token parity and the `1e-5` logprob
tolerance for batches 1, 3, and 3. The seven focused CPU test files covering
startup, prefetching, GDN, and configuration migration passed 177 tests, each
file run in its own process.

On 2026-09-13, the pre-reduction revision `415412660` passed on H100 80 GB with
PyTorch 2.13.0+cu130, FlashInfer 0.6.18, Transformers 5.12.1, and `sgl-kernel`
0.4.6.post1: 10 layer checks, including the five retained above, and one generated
BF16 Qwen MoE engine comparison with CPU offload. Serial and forced overlap
produced identical token IDs and logprobs within `1e-5` for repeated batches of
1, 3, and 3, with captured decode confirmed in both runs. None of these checks
were skipped.

The deferred-support probes were removed with their implementations. Their source
and results remain associated with that revision; these historical results do not
validate subsequent admission or lifecycle changes. Rerun the retained checks
when changing those paths.

The explicit repros also reproduced all three known MegaMoE/MoE LoRA limitations.
These are correctness observations, not performance measurements or a requirement
to pin those package versions. In particular, the two engine boots share compiled
kernel caches, so their startup times are not an overlap speedup measurement.

The MLA checks use the non-DeepGEMM BMM path. GDN exercises a real FlashInfer decode
kernel but not every attention backend. The generated engine fixture is a wiring
check, not full-model accuracy coverage. Keep these limits when reusing results.
