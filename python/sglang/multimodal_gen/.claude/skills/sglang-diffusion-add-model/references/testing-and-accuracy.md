# Testing And Accuracy

Use this reference after a new diffusion model or pipeline variant can already
produce a non-noise image or video.

## Test Placement

- Add concrete GPU integration cases in `python/sglang/multimodal_gen/test/server/gpu_cases.py`.
- Keep reusable dataclasses, constants, thresholds, and testcase factory helpers in `python/sglang/multimodal_gen/test/server/testcase_configs.py`.
- Add the case id to `python/sglang/multimodal_gen/test/server/accuracy_testcase_configs.py`
  only when it should be part of component-accuracy coverage. Adding a GPU case
  alone does not enroll it there.
- Let `python/sglang/multimodal_gen/test/run_suite.py` own suite selection, runtime-based partitioning, and standalone test files. Do not hard-code CI shard lists elsewhere.
- If a new standalone test file is added to a suite, update `STANDALONE_FILE_EST_TIMES` after the first measured CI/runtime value is known.

Useful local entrypoints from repo root:

```bash
PYTHONPATH=python python3 python/sglang/multimodal_gen/test/run_suite.py --suite unit
PYTHONPATH=python python3 python/sglang/multimodal_gen/test/run_suite.py --suite component-accuracy-1-gpu -k <case_id>
PYTHONPATH=python python3 python/sglang/multimodal_gen/test/run_suite.py --suite 1-gpu --total-partitions 1 --partition-id 0 -k <case_id>
```

## Component Accuracy When Adding A GPU Case

If you add a new entry to `ONE_GPU_CASES`, `TWO_GPU_CASES`, or a B200-specific
case group in `gpu_cases.py`, treat component accuracy as part of the
model-adding workflow. Do not assume the new testcase will automatically fit or
enter the existing component-accuracy harness.

The component-accuracy harness compares SGLang components against Diffusers/HF
reference components. This is stricter than pipeline-level inference. New GPU
cases commonly fail here for one of three reasons:

1. The model family needs explicit hook wiring in `python/sglang/multimodal_gen/test/server/accuracy_hooks.py`.
   - Add hook logic only when the harness cannot call the raw component correctly without it.
   - Valid reasons include missing required forward arguments, required autocast/runtime context, or family-specific input preparation for the same component contract.
   - Do not change the compared output mode or add harness-side behavior that changes the component contract just to make the test pass.

2. The component is already covered by another testcase with the same source component and topology.
   - Do not add redundant component-accuracy coverage.
   - Add a skip entry in `python/sglang/multimodal_gen/test/server/accuracy_config.py` with a concrete reason such as `Representative VAE accuracy is already covered by ... for the same source component and topology`.
   - This is the preferred path for variant-only cases such as LoRA, Cache-DiT, upscaling, or other cases that reuse the same underlying component weights and topology.

3. The HF/Diffusers reference component cannot be loaded or compared faithfully in the harness.
   - Add a skip entry in `accuracy_config.py` with the exact technical failure.
   - Good reasons include missing/unsupported HF component layout, incomplete checkpoints, unsupported raw component contract, or proven divergence after matched weight transfer and matching output shape.
   - Keep the skip reason concrete and technical. Do not write vague reasons like "component accuracy flaky" or "needs investigation."

When adding a new GPU case, make this decision explicitly:

- if the case should have component-accuracy coverage, add its case id to
  `accuracy_testcase_configs.py`
- if the family needs minimal harness wiring, add the smallest possible change in `accuracy_hooks.py`
- if the case is only a variant of an already covered source component and topology, add a skip in `accuracy_config.py`
- if the HF/Diffusers reference component cannot be compared faithfully, add a skip in `accuracy_config.py`
- if the case is intentionally GPU-smoke-only, leave it out of `accuracy_testcase_configs.py` and keep that choice explicit in the PR notes

Do not add a new GPU case and wait for CI to discover missing component-accuracy
wiring.

## Follow-up Scope

Once the model is working and output quality is verified, cover the follow-up
scope the user requested. If the user did not specify test or benchmark depth,
propose the smallest useful validation set before launching long GPU runs.

Tests should cover:

- pipeline construction and stage wiring
- single-GPU inference producing non-noise output
- multi-GPU inference if TP/SP is supported
- relevant unit tests for new math, parsing, scheduling, or loader behavior

For performance data:

- use the `warmup excluded` latency line for command-line generation
- keep prompt, seed, shape, step count, model path, backend, and GPU topology fixed
- use `sglang-diffusion-benchmark-profile` for denoise perf dumps and profiler traces
- use `python/sglang/multimodal_gen/benchmarks/bench_serving.py` for serving benchmarks
