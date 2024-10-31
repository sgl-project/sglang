# Troubleshooting

This page lists some common errors and tips for fixing them.

## CUDA error: an illegal memory access was encountered
This error may be due to kernel errors or out-of-memory issues.
- If it is a kernel error, it is not easy to fix.
- If it is out-of-memory, sometimes it will report this error instead of "Out-of-memory." In this case, try setting a smaller value for `--mem-fraction-static`. The default value of `--mem-fraction-static` is around 0.8 - 0.9.

## The server hangs
If the server hangs, try disabling some optimizations when launching the server.
- Add `--disable-cuda-graph`.
- Add `--sampling-backend pytorch`.
