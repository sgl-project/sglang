# Frequently Asked Questions

## The results are not deterministic, even with a temperature of 0

You may notice that when you send the same request twice, the results from the engine will be slightly different, even when the temperature is set to 0.

From our initial investigation, this indeterminism arises from two factors: dynamic batching and prefix caching. Roughly speaking, dynamic batching accounts for about 95% of the indeterminism, while prefix caching accounts for the remaining portion. The server runs dynamic batching under the hood. Different batch sizes can cause PyTorch/CuBLAS to dispatch to different CUDA kernels, which can lead to slight numerical differences. This difference accumulates across many layers, resulting in nondeterministic output when the batch size changes. Similarly, when prefix caching is enabled, it can also dispatch to different kernels. Even when the computations are mathematically equivalent, small numerical differences from different kernel implementations lead to the final nondeterministic outputs.

To achieve more deterministic outputs in the current code, you can add `--disable-radix-cache` and send only one request at a time. The results will be mostly deterministic under this setting.

We are still investigating the root causes and potential solutions. In the short term, we may introduce a "deterministic mode" that uses more padding to address the variance caused by dynamic batching. This mode will be more deterministic but slower.

We have two issues to track our progress:
- The deterministic mode is tracked at [https://github.com/sgl-project/sglang/issues/1729](https://github.com/sgl-project/sglang/issues/1729).
- The per-request random seed is tracked at [https://github.com/sgl-project/sglang/issues/1335](https://github.com/sgl-project/sglang/issues/1335).
