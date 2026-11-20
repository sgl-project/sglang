"""Microbenchmark: MPS Metal GDN recurrence vs MLX on Qwen3.5 shapes.

Not a CI test. Runs the in-tree Metal kernel (fused gating + q/k L2 norm),
``mx.fast.gated_delta_update`` and ``mlx_lm.models.gated_delta.gated_delta_kernel``
on fp32 inputs. The MLX entries are timed with g/beta and the normalized q/k
precomputed, so they exclude the work the Metal kernel fuses in.

    PYTHONPATH=python python test/manual/kernels/bench_gdn_mps.py

Calls repeat on the same tensors, which is the hot-L2 pattern of decode
serving; the prefill working sets (up to ~130 MB) already exceed L2.
"""

import argparse
import time

import mlx.core as mx
import numpy as np
import torch
from mlx_lm.models.gated_delta import gated_delta_kernel, normalize_qk

from sglang.kernels.ops.attention.linear.gdn_mps import gated_delta_rule_mps

DEV = "mps"


def _inputs(batch, seq_len, hk, hv, dk=128, dv=128, seed=0):
    gen = torch.Generator().manual_seed(seed)
    total = batch * seq_len

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, generator=gen) * scale

    return {
        "q": randn(total, hk, dk),
        "k": randn(total, hk, dk),
        "v": randn(total, hv, dv),
        "a": randn(total, hv, scale=0.5),
        "b": randn(total, hv),
        "A_log": randn(hv, scale=0.5),
        "dt_bias": randn(hv, scale=0.1),
        "state": randn(batch, hv, dv, dk, scale=0.1),
    }


def _metal_call(tensors, batch, seq_len):
    total = batch * seq_len
    device = {name: t.to(DEV) for name, t in tensors.items()}
    qsl = torch.arange(0, total + 1, seq_len, dtype=torch.int32, device=DEV)
    indices = torch.arange(batch, dtype=torch.int32, device=DEV)
    initial = torch.ones(batch, dtype=torch.bool, device=DEV)

    def run():
        gated_delta_rule_mps(
            device["q"],
            device["k"],
            device["v"],
            device["a"],
            device["b"],
            device["A_log"],
            device["dt_bias"],
            device["state"],
            indices,
            qsl,
            initial,
        )

    return run


def _mlx_arrays(tensors, batch, seq_len):
    q, k, v = (
        tensors[n].numpy().reshape(batch, seq_len, *tensors[n].shape[1:]) for n in "qkv"
    )
    a = tensors["a"].numpy().reshape(batch, seq_len, -1)
    b = tensors["b"].numpy().reshape(batch, seq_len, -1)
    A_log, dt_bias, state = (
        mx.array(tensors[n].numpy()) for n in ("A_log", "dt_bias", "state")
    )
    q, k, v, a, b = (mx.array(t) for t in (q, k, v, a, b))
    g = mx.exp(-mx.exp(A_log) * mx.log(1 + mx.exp(a + dt_bias)))
    beta = mx.sigmoid(b)
    q, k = normalize_qk(q, k, inv_scale=q.shape[-1] ** -0.5, eps=1e-6)
    return q, k, v, g, beta, state


def _assert_finite(fn):
    outputs = fn()
    mx.eval(outputs)
    for output in outputs:
        fp32 = torch.from_numpy(np.asarray(output, dtype=np.float32))
        assert torch.isfinite(fp32).all(), "MLX baseline produced non-finite output"


def _time(fn, sync, iters):
    for _ in range(3):
        fn()
        sync()
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        sync()
        best = min(best, (time.perf_counter() - t0) / iters)
    return best * 1e3


def _time_mlx(fn, iters):
    # mlx is lazy: eval the outputs of every call or nothing is measured.
    return _time(lambda: mx.eval(fn()), lambda: None, iters)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    cases = []
    for hk, hv in ((16, 16), (16, 32)):
        for batch in (1, 8, 16):
            cases.append(("decode", batch, 1, hk, hv))
        for seq_len in (512, 4096):
            cases.append(("prefill", 1, seq_len, hk, hv))

    header = (
        f"{'mode':8} {'B':>3} {'T':>5} {'Hk':>3} {'Hv':>3}"
        f" {'metal':>9} {'mx.fast':>9} {'mlx-lm':>9}"
    )
    print(header + "  (ms per call)")
    for mode, batch, seq_len, hk, hv in cases:
        tensors = _inputs(batch, seq_len, hk, hv)
        iters = args.iters if seq_len < 4096 else max(3, args.iters // 4)
        metal_ms = _time(
            _metal_call(tensors, batch, seq_len), torch.mps.synchronize, iters
        )
        q, k, v, g, beta, state = _mlx_arrays(tensors, batch, seq_len)
        _assert_finite(lambda: mx.fast.gated_delta_update(q, k, v, g, beta, state))
        _assert_finite(lambda: gated_delta_kernel(q, k, v, g, beta, state))
        fast_ms = _time_mlx(
            lambda: mx.fast.gated_delta_update(q, k, v, g, beta, state), iters
        )
        lm_ms = _time_mlx(lambda: gated_delta_kernel(q, k, v, g, beta, state), iters)
        row = f"{mode:8} {batch:>3} {seq_len:>5} {hk:>3} {hv:>3}"
        print(f"{row} {metal_ms:>9.3f} {fast_ms:>9.3f} {lm_ms:>9.3f}", flush=True)


if __name__ == "__main__":
    main()
