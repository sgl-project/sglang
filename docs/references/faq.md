# Troubleshooting and Frequently Asked Questions

## Troubleshooting

This page lists common errors and tips for resolving them.

### CUDA Out of Memory
If you encounter out-of-memory (OOM) errors, you can adjust the following parameters:

- If OOM occurs during prefill, try reducing `--chunked-prefill-size` to `4096` or `2048`. This saves memory but slows down the prefill speed for long prompts.
- If OOM occurs during decoding, try lowering `--max-running-requests`.
- You can also reduce `--mem-fraction-static` to a smaller value, such as 0.8 or 0.7. This decreases the memory usage of the KV cache memory pool and helps prevent OOM errors during both prefill and decoding. However, it limits maximum concurrency and reduces peak throughput.
- Another common case for OOM is requesting input logprobs for a long prompt as it requires significant memory. To address this, set `logprob_start_len` in your sampling parameters to include only the necessary parts. If you do need input logprobs for a long prompt, try reducing `--mem-fraction-static`.

### CUDA Error: Illegal Memory Access Encountered
This error may result from kernel errors or out-of-memory issues:
- If it is a kernel error, resolving it may be challenging. Please file an issue on GitHub.
- If it is an out-of-memory issue, it may sometimes be reported as this error instead of "Out of Memory." Refer to the section above for guidance on avoiding OOM issues.


## Frequently Asked Questions

### The results are not deterministic, even with a temperature of 0

You may notice that when you send the same request twice, the results from the engine will be slightly different, even when the temperature is set to 0.

From our initial investigation, this indeterminism arises from two factors: dynamic batching and prefix caching. Roughly speaking, dynamic batching accounts for about 95% of the indeterminism, while prefix caching accounts for the remaining portion. The server runs dynamic batching under the hood. Different batch sizes can cause PyTorch/CuBLAS to dispatch to different CUDA kernels, which can lead to slight numerical differences. This difference accumulates across many layers, resulting in nondeterministic output when the batch size changes. Similarly, when prefix caching is enabled, it can also dispatch to different kernels. Even when the computations are mathematically equivalent, small numerical differences from different kernel implementations lead to the final nondeterministic outputs.

To achieve more deterministic outputs in the current code, you can add `--disable-radix-cache` and send only one request at a time. The results will be mostly deterministic under this setting.

We are still investigating the root causes and potential solutions. In the short term, we may introduce a "deterministic mode" that uses more padding to address the variance caused by dynamic batching. This mode will be more deterministic but slower.

We have two issues to track our progress:
- The deterministic mode is tracked at [https://github.com/sgl-project/sglang/issues/1729](https://github.com/sgl-project/sglang/issues/1729).
- The per-request random seed is tracked at [https://github.com/sgl-project/sglang/issues/1335](https://github.com/sgl-project/sglang/issues/1335).
