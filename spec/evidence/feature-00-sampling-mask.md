Status: implemented

# Sampling-mask verification

Validation on NVIDIA H200, Python 3.12, PyTorch 2.13.0, sglang-kernel
0.4.6.post1, FlashInfer 0.6.18, and Transformers 5.12.1. Server integration
uses the repository pins TVM-FFI 0.1.11 and CUTLASS DSL 4.6.2.

## Unit and GPU capture regression tests

From the repository root, with the checkout installed in `.venv`:

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONPATH="$PWD/python" .venv/bin/python -m pytest -q \
  test/registered/unit/sampling/test_sampling_batch_info.py \
  test/registered/unit/managers/test_batch_result_processor_hidden_states.py \
  test/registered/unit/managers/test_batch_result_processor_mamba_boundary.py \
  test/registered/unit/managers/test_generation_auxiliary_output.py \
  test/registered/unit/disaggregation/test_disaggregation_wire.py \
  test/registered/unit/server_args/test_server_args.py \
  test/registered/sampling/test_sampling_mask.py::TestSamplingMaskCapture \
  test/registered/sampling/test_sampling_mask.py::TestSamplingMaskPacking \
  --disable-warnings --tb=short
```

Result: **311 passed, 38 subtests passed**, 15 warnings, in **19.35 seconds**.
This includes real CUDA sampling and asynchronous copying, cutoff ties, min-p,
mixed opt-in rows, overflow and invalid support, synchronized-token logprobs,
batch filtering/merging, abort cleanup, pipeline payload reconstruction, and
disaggregation metadata transport.

## Server integration matrix

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONPATH="$PWD/python" \
  SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0 \
  .venv/bin/python spec/evidence/run_sampling_mask.py
```

The checked-in runner uses public `Qwen/Qwen2.5-0.5B-Instruct` weights. It runs
FlashInfer and PyTorch with overlap enabled and disabled, then seeded sampling
parity with mask capture enabled and disabled. Prefill graph capture
is configured up to 32 tokens and decode capture up to batch size 8, subject to
the server's mode-compatibility rules. KV capacity is 4096 tokens.

Multi-process pipeline and prefill/decode deployments are covered here by
transport unit tests, not live deployment tests.

| Backend / test | Overlap | Passed |
| --- | --- | ---: |
| FlashInfer | Enabled | 6 |
| PyTorch | Enabled | 6 |
| FlashInfer | Disabled | 6 |
| PyTorch | Disabled | 6 |
| Seeded parity, PyTorch with Triton matrix multiplication | Enabled | 1 |

Result: **25 passed**. The seeded-parity test passed in **31.562 seconds** and
preserved both token IDs and text. The command uses Triton matrix multiplication
for deterministic inference because the installed DeepGEMM binary is incompatible
with the local PyTorch build. DeepGEMM deterministic inference was not validated
in this environment.
