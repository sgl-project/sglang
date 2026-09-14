"""Hot-path dispatch overhead microbenchmark for the unified ``BaseFusedOp``.

Compares per-call overhead of:

1. a plain bound-method call (theoretical floor),
2. a minimal reproduction of the former ``MultiPlatformOp`` hot path
   (``nn.Module.__call__`` -> ``self._forward_method(*args)``),
3. the unified ``BaseFusedOp`` (adds the forced-backend check, the cached
   dispatch lookup, and the trace flag check).

The op body is a no-op so the numbers isolate pure dispatch overhead; real
kernels are microseconds+, so the delta reported here is the worst case.

Run locally (CPU is fine):
    python test/manual/kernels/bench_fused_op_dispatch.py
"""

import time

import torch
from torch import nn

from sglang.kernels.fused_op import BaseFusedOp

N_WARMUP = 10_000
N_ITERS = 200_000


class _OldStyleOp(nn.Module):
    """Minimal replica of the retired MultiPlatformOp hot path."""

    def __init__(self):
        super().__init__()
        self._forward_method = self.forward_cuda

    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)

    def forward_cuda(self, x):
        return x


class _NewOp(BaseFusedOp):
    op = "bench.dispatch"

    def forward_native(self, x):
        return x

    def forward_cuda(self, x):
        return x


def _bench(fn, x) -> float:
    for _ in range(N_WARMUP):
        fn(x)
    start = time.perf_counter()
    for _ in range(N_ITERS):
        fn(x)
    return (time.perf_counter() - start) / N_ITERS * 1e9  # ns/call


def main():
    x = torch.zeros(1)

    old_op = _OldStyleOp()
    new_op = _NewOp()
    new_op(x)  # resolve + cache dispatch
    bound = new_op.forward_native

    floor_ns = _bench(bound, x)
    old_ns = _bench(old_op, x)
    new_ns = _bench(new_op, x)

    print(f"plain bound method      : {floor_ns:8.1f} ns/call")
    print(f"MultiPlatformOp replica : {old_ns:8.1f} ns/call")
    print(f"unified BaseFusedOp     : {new_ns:8.1f} ns/call")
    print(f"delta (new - old)       : {new_ns - old_ns:8.1f} ns/call")


if __name__ == "__main__":
    main()
