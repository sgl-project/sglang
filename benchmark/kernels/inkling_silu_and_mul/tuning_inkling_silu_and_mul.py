"""Tune the Inkling MoE ``silu_and_mul`` Triton kernel and write the JSON configs.

Run on one idle GPU, with clocks locked for stable rankings:

    nvidia-smi -i 0 -lgc <max_sm_clock>
    python benchmark/kernels/inkling_silu_and_mul/tuning_inkling_silu_and_mul.py --control
    nvidia-smi -i 0 -rgc

Writes ``layout={interleaved,contiguous},device_name=<device>.json`` into
``python/sglang/kernels/ops/moe/configs/``. Timings are the min over several
rounds; ``--control`` re-measures one config twice to report the noise floor,
below which rankings are meaningless.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from collections import defaultdict

import torch
import triton
from triton.testing import do_bench

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.kernels.ops.moe.inkling_moe import _silu_and_mul_triton_kernel
from sglang.kernels.ops.moe.inkling_silu_config import (
    config_file_name,
    configs_dir,
    default_config,
)

# Inkling MoE intermediate sizes plus the surrounding power-of-two grid.
DEFAULT_NS = [384, 512, 768, 1024, 2048, 48 * 96, 6144, 8192]
# Large M decides the config; small M is insensitive and inherits it.
DEFAULT_MS = [1024, 4096, 16384]

BLOCK_MS = [1, 2, 4, 8, 16]
BLOCK_NS = [128, 256, 512, 1024]
NUM_WARPS = [1, 2, 4, 8]


def candidate_configs(half_dim: int) -> list[dict[str, int]]:
    out = []
    for block_m, block_n, warps in itertools.product(BLOCK_MS, BLOCK_NS, NUM_WARPS):
        if block_n > max(128, triton.next_power_of_2(half_dim)):
            continue
        # Huge tiles on few warps spill registers and compile slowly.
        per_lane = (block_m * block_n) / (warps * 32)
        if not 1 <= per_lane <= 64:
            continue
        out.append(
            {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "num_warps": warps}
        )
    return out


def make_runner(gateup, topk_weights, use_interleaved, config):
    m = gateup.shape[0]
    n = gateup.shape[1] // 2
    block_m = config["BLOCK_SIZE_M"]
    block_n = config["BLOCK_SIZE_N"]
    if use_interleaved and block_n % 8 != 0:
        return None
    out = torch.empty((m, n), device=gateup.device, dtype=gateup.dtype)
    grid = triton.cdiv(m, block_m) * triton.cdiv(n, block_n)
    kwargs = dict(
        gateup_out_ptr=gateup,
        topk_weights_ptr=topk_weights,
        down_inp_ptr=out,
        M=m,
        N=n,
        TOPK_WEIGHTS=topk_weights is not None,
        INTERLEAVED=use_interleaved,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        EVEN_N=n % block_n == 0,
        INT64_INDEX=gateup.nbytes >= 2**31,
        num_warps=config["num_warps"],
        **({"USE_PDL": True, "launch_pdl": True} if is_arch_support_pdl() else {}),
    )
    try:
        _silu_and_mul_triton_kernel[(grid,)](**kwargs)
    except Exception:
        return None
    return out, (lambda: _silu_and_mul_triton_kernel[(grid,)](**kwargs))


def reference(gateup, topk_weights, use_interleaved):
    x = gateup.float()
    if use_interleaved:
        gate, up = x[:, 0::2], x[:, 1::2]
    else:
        n = x.shape[1] // 2
        gate, up = x[:, :n], x[:, n:]
    out = gate * torch.sigmoid(gate) * up
    if topk_weights is not None:
        out = out * topk_weights.float()[:, None]
    return out.to(gateup.dtype)


def bench(fn, rounds: int) -> float:
    return min(do_bench(fn, warmup=25, rep=100) * 1e3 for _ in range(rounds))


def tune_one(use_interleaved, half_dim, ms, dtype, rounds, control):
    """Score every candidate over the M sweep; return (config, report)."""
    scores: dict[tuple, float] = defaultdict(float)
    noise = 0.0
    for m, with_w in itertools.product(ms, [False, True]):
        torch.manual_seed(0)
        gateup = torch.randn(m, 2 * half_dim, dtype=dtype, device="cuda")
        w = torch.randn(m, dtype=dtype, device="cuda") if with_w else None
        expect = reference(gateup, w, use_interleaved)

        timings = {}
        for config in candidate_configs(half_dim):
            made = make_runner(gateup, w, use_interleaved, config)
            if made is None:
                continue
            out, fn = made
            # A wrong config must not win on speed.
            torch.testing.assert_close(
                out.float(), expect.float(), rtol=2**-7, atol=1e-6
            )
            timings[tuple(sorted(config.items()))] = bench(fn, rounds)
        if control:
            incumbent = make_runner(
                gateup, w, use_interleaved, default_config(half_dim)
            )
            if incumbent is not None:
                a = bench(incumbent[1], rounds)
                b = bench(incumbent[1], rounds)
                noise = max(noise, abs(a - b) / min(a, b))
        best = min(timings.values())
        for key, us in timings.items():
            scores[key] += best / us

    best_key = max(scores, key=scores.get)
    return dict(best_key), {"noise": noise, "num_candidates": len(scores)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns", type=int, nargs="+", default=DEFAULT_NS)
    parser.add_argument("--ms", type=int, nargs="+", default=DEFAULT_MS)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--control",
        action="store_true",
        help="re-measure the incumbent config to expose the noise floor",
    )
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    out_dir = args.out_dir or configs_dir()
    os.makedirs(out_dir, exist_ok=True)

    for use_interleaved in (True, False):
        table = {}
        for half_dim in args.ns:
            config, report = tune_one(
                use_interleaved, half_dim, args.ms, dtype, args.rounds, args.control
            )
            table[str(half_dim)] = config
            layout = "interleaved" if use_interleaved else "contiguous"
            extra = f" noise={report['noise'] * 100:.1f}%" if args.control else ""
            print(
                f"{layout:>11} N={half_dim:>5}: {config}"
                f"  ({report['num_candidates']} candidates{extra})",
                flush=True,
            )
        path = os.path.join(out_dir, config_file_name(use_interleaved))
        with open(path, "w") as f:
            json.dump({str(dtype): table}, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
