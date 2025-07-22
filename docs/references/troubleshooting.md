# Troubleshooting

This page lists common errors and tips for resolving them.

## CUDA Out of Memory
If you encounter out-of-memory (OOM) errors, you can adjust the following parameters:

- If OOM occurs during prefill, try reducing `--chunked-prefill-size` to `4096` or `2048`. This saves memory but slows down the prefill speed for long prompts.
- If OOM occurs during decoding, try lowering `--max-running-requests`.
- You can also reduce `--mem-fraction-static` to a smaller value, such as 0.8 or 0.7. This decreases the memory usage of the KV cache memory pool and helps prevent OOM errors during both prefill and decoding. However, it limits maximum concurrency and reduces peak throughput.
- Another common case for OOM is requesting input logprobs for a long prompt as it requires significant memory. To address this, set `logprob_start_len` in your sampling parameters to include only the necessary parts. If you do need input logprobs for a long prompt, try reducing `--mem-fraction-static`.

## CUDA Error: Illegal Memory Access Encountered
This error may result from kernel errors or out-of-memory issues:
- If it is a kernel error, resolving it may be challenging. Please file an issue on GitHub.
- If it is an out-of-memory issue, it may sometimes be reported as this error instead of "Out of Memory." Refer to the section above for guidance on avoiding OOM issues.
