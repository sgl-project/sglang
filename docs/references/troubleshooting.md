# Troubleshooting

This page lists some common errors and tips for fixing them.

## CUDA out of memory
If you see out of memory (OOM) errors, you can try to tune the following parameters.
- If OOM happens during prefill, try to decrease `--chunked-prefill-size` to `4096` or `2048`.
- If OOM happens during decoding, try to decrease `--max-running-requests`.
- You can also try to decrease `--mem-fraction-static`, which reduces the memory usage of the KV cache memory pool and helps both prefill and decoding.

## CUDA error: an illegal memory access was encountered
This error may be due to kernel errors or out-of-memory issues.
- If it is a kernel error, it is not easy to fix. Please file an issue on the GitHub.
- If it is out-of-memory, sometimes it will report this error instead of "Out-of-memory." Please refer to the above section to avoid the OOM.
