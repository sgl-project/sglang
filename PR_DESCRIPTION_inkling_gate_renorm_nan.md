# [Fix] Stabilize Inkling gate normalization for extreme logits

## Motivation

Inkling's fused gate can produce NaN routing weights from finite logits. Expert selection uses `sigmoid(raw_logit) + expert_bias`, but normalization uses the unbiased raw logits of the six selected routed experts and two shared experts. A large selection bias can therefore select an expert whose sigmoid weight is tiny; top-k selection does not guarantee a nonzero normalization denominator.

The CUDA gate evaluates sigmoid as `1 / (1 + exp(-x))`. At `x = -100`, the intermediate `exp(100)` overflows FP32 and the computed sigmoid becomes zero. If this happens for all eight active experts, normalization divides by zero and produces NaNs. Adding `1e-20` to the denominator avoids the NaNs but changes the model's output: eight equal finite logits should always receive equal normalized weights, not zero weights. Even before overflow, the epsilon attenuates the total weight by `S / (S + 1e-20)`, where `S` is the sigmoid sum. With eight logits equal to `-50`, the unscaled weights sum to about `0.1337` instead of `1`.

The non-fused path has a separate cancellation issue. Reconstructing `max_log_prob + log(sum_exp)` loses the normalization term when the maximum is very negative, such as `-1e8`. Eight equal logits can then receive unscaled weights of `1` each instead of `1/8`.

This work was prompted by an Inkling request that reached sampling with an invalid row and triggered a device-side assert in `torch.multinomial`. Disabling the overlap scheduler did not resolve that failure. The gate defect is reproducible independently, but the sampling logs alone do not establish it as the first source of non-finite values in the original request.

## Modifications

Replace the epsilon workaround with a scaled sigmoid computation that is mathematically equivalent to `softmax(logsigmoid(active_logits))`:

```text
m   = max_i(min(x_i, 0))
q_i = exp(min(x_i, 0) - m) / (1 + exp(-abs(x_i)))
w_i = (q_i / sum_j(q_j)) * route_scale * global_scale
```

The common scale factor cancels during normalization. For finite active logits, every `q_i` is at most `1`, and at least one is at least `0.5`, so the eight-expert denominator stays in `[0.5, 8]`. This avoids both a zero denominator and an arbitrary epsilon without adding logarithms or kernel launches to the fused paths.

- Apply the stable normalization to CUDA JIT v1/v2 and the Triton fused gate. JIT v2 and the fused GEMV path share the corrected epilogue.
- Mask inactive Triton slots before both the maximum and sum reductions so padding cannot affect normalization.
- Normalize shifted exponentials directly in the non-fused Triton path, and use `log_probs.softmax(-1)` in the CPU fallback to avoid cancellation.
- Add independent FP64 regression tests and a packed-gate microbenchmark using production-style padded inputs.

Expert selection, output layouts, packed top-k support, and CUDA Graph support are unchanged. Results are numerically equivalent to the intended normalization, not necessarily bitwise identical to the old implementation. This is not a sanitizer for logits or expert weights that are already non-finite, or for an invalid global scale.

## Accuracy Tests

The GPU results below were recorded on the source branch before this cherry-pick, using NVIDIA H20-3e and PyTorch `2.13.0+cu130`. They have not been rerun on this destination branch.

- All six test methods passed with the stable normalization.
- As a negative control, the new launch-shape and non-fused tests failed 32 subtests against the earlier epsilon implementation. These covered JIT v1/v2, Triton, both output formats, and CPU/CUDA non-fused normalization.
- Coverage includes random logits and bias, equal and unequal negative tails, finite logits down to `-1e30`, mixed routed/shared values, weight-sum preservation, padded row strides, inactive slots, CUDA Graph replay with changed logits/scales, and the actual fused GEMV entry point.
- The reference computes expert selection separately and uses FP64 `logsigmoid` followed by `softmax`; it does not reuse the implementation's scaled sigmoid or epsilon formula.

For example, eight active logits equal to `-100` with `route_scale = 1.5` and `global_scale = 0.75` must each receive weight `0.140625`. The earlier epsilon-based CUDA JIT implementation returned zeros for this case.

Run from the repository root:

```bash
PYTHONPATH=python python3 test/registered/kernel/moe/test_inkling_gate_topk_renorm.py
```

Scoped pre-commit checks and `git diff --check` were also reported as passing on the source branch. No B200 run or replay of the original failing MMMU-Pro request has been completed. Confirming that the production incident is fully resolved still requires replaying that request with the deployed checkpoint and configuration.

## Speed Tests and Profiling

The source-branch microbenchmark uses `marker.do_bench`, CUDA Graphs, and production-style padded input layouts. The initial H20-3e comparison for JIT v2 with normal logits was:

| Tokens | Before | Stable normalization |
| ---: | ---: | ---: |
| 1 | 1.959 us | 1.979 us |
| 16 | 2.157 us | 2.204 us |
| 1,024 | 3.366 us | 3.487 us |

The fix is not strictly zero-cost. Other workloads were running on the GPU, and the 16K-token measurements were too variable to support a regression conclusion. These are kernel timings, not end-to-end throughput measurements or B200 results.

```bash
PYTHONPATH=python python3 test/registered/kernel/moe/bench_inkling_gate_topk_renorm.py
```
