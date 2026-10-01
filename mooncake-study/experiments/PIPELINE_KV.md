# Pipeline Target-KV Source Assembly

The injector now handles target KV distributed over PP stages, as a prerequisite
for pipeline DSpark. Full PP speculative execution and capture remain disabled.

Run the complete CPU files in a matching SGLang environment with Gloo:

```bash
PYTHONPATH=python python test/registered/unit/spec/test_dspark_target_kv_pipeline.py -v -f
PYTHONPATH=python python test/registered/unit/spec/test_dspark_target_kv.py -v -f
```

The new test spawns four CPU processes. It exercises the production injector,
identity/geometry assembly, coordinated policy validation and PP broadcast method.
TP collection uses Gloo all-gather through a small group adapter. Local target
artifact inspection and draft safetensors hashing use explicit test fixtures;
this is not a real-model or NCCL test.

The topology/dtype cases are:

| TP | PP | Global KV heads | Dtype |
| --- | --- | --- | --- |
| 2 | 2 | 2 | BF16 |
| 2 | 2 | 1 | BF16 |
| 1 | 4 | 2 | BF16 |
| 4 | 1 | 2 | BF16 |
| 4 | 1 | 8 | BF16 |
| 2 | 2 | 2 | FP16 |

Selected layers are deliberately ordered `[3, 1]`. Each rank has a different
logical-to-physical slot permutation. Noncanonical TP replicas contain poisoned
values, so using or averaging the wrong copy fails exact tensor comparison.
PP4 stages zero and two own no selected layers and still receive all four K/V
tensors. Assembled tensors and encoded context match the full logical reference
with zero tolerance. Incremental writes cover `[0:3)` and `[3:5)`, preserve the
rank-local destination slots and advance the committed projection state once.

PP4 stage zero separately fails local binding, draft pool validation and draft
weight agreement. All four ranks must report the expected phase/ranks, without
exposing source buffers; a subsequent valid attempt uses the same groups.

Run the existing one-GPU serving regression with Mooncake SDK, `mooncake_master`
on `PATH`, and local Qwen3-0.6B artifacts:

```bash
PYTHONPATH=python \
TRAINING_CAPTURE_TEST_MODEL=/path/to/Qwen3-0.6B \
python test/registered/storage/test_training_capture_pd_dspark.py -v -f
```

Its four cases cross eager/verify CUDA graphs with AR-only P or KV-input DSpark
on both P and D. They use actual TCP Mooncake P/D transfer and Store, plus an
HTTP Catalog test double. This guards the existing PP1 serving path after the
injector change. It cannot establish PP serving, NCCL, latency SLOs or trained
draft quality. Retained run results are in `pipeline-target-kv.json`.
