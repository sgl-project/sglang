# DSpark Markov candidate validation

The candidate path prunes each Markov step to the union of base-logit Top-K and
static bias Top-M. It supports NVIDIA CUDA, TP=1, VanillaMarkov heads and directly
readable FP16/BF16/FP32 weights, with K<=64, M<=128, rank<=1024 and gamma<=32.
K=0 and unsupported configurations retain the existing dense proposal path.

The proposal cache is still dense FP32 `[capacity, gamma, target_vocab_size]`;
only its updates are sparse. The LM head and Top-K remain separate operations,
and the candidate walk retains `enable_fp_fusion=False`. This change does not
implement a CSR cache or a fused LM-head/Top-K kernel.

## Correctness

Run from the repository root in a complete SGLang environment:

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/unit/spec/test_dspark_checkpoint.py \
  test/registered/kernels/ops/speculative/test_dspark_markov_candidates.py \
  test/registered/e2e/spec/test_dspark_candidate_runtime.py \
  test/registered/spec/dspark/test_dspark_kernel_parity.py
```

The tests cover checkpoint configuration and vocabulary mappings, independent
proposal oracles, actual GPU proposal/rejection distributions, cache reuse,
weight refresh, and CUDA Graph input/RNG updates. Statistical failures must be
investigated, not rerun until passing. CUDA skips do not establish correctness.

For local target/draft checkpoints, run the existing serving smoke explicitly:

```bash
SGLANG_TEST_DSPARK_USER_CHECKPOINT=1 \
SGLANG_TEST_DSPARK_TARGET_PATH=/path/to/target \
SGLANG_TEST_DSPARK_DRAFT_PATH=/path/to/draft \
PYTHONPATH=python python -m pytest -q -s \
  test/registered/core/test_basic_sanity_dspark.py::TestDSparkUserCheckpointCandidates
```

This checks TP1/DP1 greedy token parity against target-only serving, mixed
requests, and eager/graph execution with K32/M128. It requires candidate dispatch
in startup logs. It does not establish task accuracy, DP8 behavior or speedup.

## Performance

Run correctness tests before timing. The synthetic benchmark compares the existing
dense proposal with the production candidate wrapper, including its RNG and Top-K:

```bash
PYTHONPATH=python python benchmark/kernels/bench_dspark_markov_candidates.py \
  --vocab 151936 --rank 256 --dtype bfloat16 --base-dtype float32 \
  --k 32 --m 0 --gamma 8 --batch 1 4 16 32 \
  --sampling probabilistic --temperature 1 --cache hot --graph
```

The example shapes are synthetic: choose rank, dtype and scale for the workload.
This benchmark assumes equal target/draft vocabularies and does not represent
reduced-vocabulary checkpoint memory. It excludes the LM head, verifier, model
forward and serving overhead. Report the emitted timer, cache policy, graph mode,
initialization memory and p50/p95 with the result.

For end-to-end measurements, compare K=0 against K32/M0 on the same commit,
checkpoint, dataset, sampling parameters and request-length policy. Start with DP1,
then repeat with DP8; alternate baseline/candidate runs. Record output tokens/s,
TPOT/ITL, accepted length, memory and task accuracy at temperatures 0 and 1.
Attach the exact commit, GPU/driver/runtime versions, launch commands and raw
results to the PR. No measured H200 result is recorded here yet.
