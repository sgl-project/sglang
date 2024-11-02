# Frequently Asked Questions

## The results are not deterministic even with temperature 0

When you run decoding with a temperature of 0, obtaining the logprob of input tokens or output tokens, you might notice that the results returned by the engine are not deterministic.
You may observe that when you send the same request twice, the results will be slightly different.

From our early investigation, this indeterminism arises from two factors: dynamic batching and prefix caching.
Roughly speaking, dynamic batching can account for 95% of the indeterminism, while prefix caching accounts for the remaining portion. The server runs dynamic batching under the hood. Different batch sizes can cause PyTorch/CuBLAS to dispatch to different CUDA kernels, which can lead to slight numerical differences. This difference accumulates throughout many layers and results in nondeterministic output when the batch size changes. Similarly, when prefix caching is turned on, it will also dispatch to different kernels.

We are still investigating the root cause and possible solutions. In the short term, we might introduce a "deterministic mode" that uses more padding to address the variance from dynamic batching. This mode will be more deterministic but slower.

On the other hand, if you add `--disable-radix-cache` and only send one request at a time, the results will be mostly deterministic.

We have two issues to track our progress:
- The deterministic mode is tracked at [https://github.com/sgl-project/sglang/issues/1729](https://github.com/sgl-project/sglang/issues/1729)
- The per-request random seed is tracked at [https://github.com/sgl-project/sglang/issues/1335](https://github.com/sgl-project/sglang/issues/1335)
