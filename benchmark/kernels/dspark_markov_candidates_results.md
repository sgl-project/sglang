# DSpark Markov candidate validation

The candidate path prunes each Markov step to the union of base-logit Top-K and
static bias Top-M. It supports NVIDIA CUDA, TP=1, VanillaMarkov heads and directly
readable FP16/BF16/FP32 weights, with K<=64, M<=128, rank<=1024 and gamma<=32.
K=0 and unsupported configurations retain the existing dense proposal path.

The proposal cache is still dense FP32 `[capacity, gamma, target_vocab_size]`;
only its updates are sparse. Top-K does not sort the selected candidates, and the
walk uses rank tiles of `min(next_power_of_2(rank), 64)`. The LM head and Top-K
remain separate operations, and the candidate walk retains `enable_fp_fusion=False`.

CUDA verification stores each draft row's scaled maximum and inverse exponential
sum instead of materializing `[batch, gamma, target_vocab_size]` draft probabilities.
Acceptance reads only the proposed token's probability; on rejection, the existing
CDF kernel computes draft probabilities for that row as it scans the residual.
Normalization still reads the dense logits. Target probabilities and the NPU
verification path retain their existing representation.

## Correctness

End-to-end serving benchmarks were run on a GPU server. Commands and performance
results are recorded in [PR #40513](https://github.com/sgl-project/sglang/pull/40513).

Run the existing DSpark kernel parity and block-verification regressions from the
repository root in a complete SGLang environment:

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/kernels/ops/speculative/test_block_verification.py \
  test/registered/spec/dspark/test_dspark_kernel_parity.py
```

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
results to the PR.
