#!/usr/bin/env python3
"""Geometry / split-K sweep for DeepSeek-V4.1-Flash decode Marlin.

Shapes match fused_marlin_moe on SM70 (M=1, top-6, hidden=5120, intermediate=2304):

  w13: size_m=1, size_n=4608, size_k=5120, top_k=6, moe_block_size=8
  w2:  size_m=6, size_n=5120, size_k=2304, top_k=1, moe_block_size=8

Does not load the 476 GiB checkpoint. Uses random MXFP4-shaped Marlin buffers.
Set SM70_MARLIN_MOE_* around each launch so marlin_v100 picks that CTA; the
sglang shape pin is skipped via _sm70_marlin_user_tuning.

Usage (GPU must be free; stop any running server first):

  CUDA_VISIBLE_DEVICES=0 python scripts/dsv41_sm70_marlin_decode_sweep.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

import torch

# Geometry table from marlin_v100 tests/sm70_env_sweep.py (MoE: CTA_M in {32,64}).
SM70_MOE_GEOMETRY_LABELS: tuple[str, ...] = (
    "32x64x32x4x32x32x16",
    "32x64x64x4x32x32x32",
    "32x64x64x4x32x64x16",
    "32x64x128x4x32x64x32",
    "32x128x32x4x32x32x32",
    "32x128x32x4x32x64x16",
    "32x128x64x4x32x64x32",
    "32x128x64x8x32x32x32",
    "32x128x64x8x32x64x16",
    "32x128x128x8x32x64x32",
    "32x256x32x4x32x64x32",
    "32x256x64x8x32x64x32",
    "64x64x32x4x32x32x32",
    "64x64x32x4x32x64x16",
    "64x64x32x4x64x32x16",
    "64x64x32x8x32x32x16",
    "64x64x64x4x32x64x32",
    "64x64x64x4x64x32x32",
    "64x64x64x4x64x64x16",
    "64x64x64x8x32x32x32",
    "64x64x64x8x32x64x16",
    "64x64x128x4x64x64x32",
    "64x64x128x8x32x64x32",
    "64x128x32x4x32x64x32",
    "64x128x32x4x64x32x32",
    "64x128x32x4x64x64x16",
    "64x128x32x8x32x32x32",
    "64x128x32x8x32x64x16",
    "64x128x32x8x64x32x16",
    "64x128x64x4x64x64x32",
    "64x128x64x8x32x64x32",
    "64x128x64x8x64x32x32",
    "64x128x64x8x64x64x16",
    "64x128x128x8x64x64x32",
    "64x256x32x4x64x64x32",
    "64x256x32x8x32x64x32",
    "64x256x32x8x64x32x32",
    "64x256x32x8x64x64x16",
    "64x256x64x8x64x64x32",
)
SPLIT_K_VALUES = (1, 2, 4, 8)
CACHE_VALUES = ("vector_words", "lane_vectors")
ENV_NAMES = (
    "SM70_MARLIN_MOE_CTA_GEOMETRY",
    "SM70_MARLIN_MOE_SPLIT_K",
    "SM70_MARLIN_MOE_METADATA_CACHE",
)

HIDDEN = 5120
INTERMEDIATE = 2304
GATE_UP_N = 2 * INTERMEDIATE
TOPK = 6
BLOCK = 8
EXPERTS = 6
WARMUP = 8
ITERS = 20


@dataclass
class SweepRow:
    gemm: str
    geometry: str
    split_k: int
    cache: str
    us: float
    ok: bool
    err: str = ""
    max_abs: float = 0.0


def _parse_geom(label: str) -> tuple[int, int, int]:
    cta_m, cta_n, cta_k, *_ = (int(x) for x in label.split("x"))
    return cta_m, cta_n, cta_k


def _shape_ok(label: str, size_n: int, size_k: int) -> bool:
    _cta_m, cta_n, cta_k = _parse_geom(label)
    packed_macro_n = 256 if size_n % 256 == 0 else 128 if size_n % 128 == 0 else 64
    return (
        size_n % cta_n == 0
        and size_n % 64 == 0
        and size_k % cta_k == 0
        and packed_macro_n % cta_n == 0
    )


def _set_env(geometry: str | None, split_k: int | None, cache: str | None) -> None:
    if geometry is None:
        for name in ENV_NAMES:
            os.environ.pop(name, None)
        return
    os.environ[ENV_NAMES[0]] = geometry
    os.environ[ENV_NAMES[1]] = str(split_k)
    os.environ[ENV_NAMES[2]] = cache


def _time_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _build_problem(device: torch.device):
    from sglang.kernels.ops.moe.moe_align_single_token import moe_align_single_token
    from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
    from sglang.srt.layers.quantization.utils import get_scalar_types

    _, scalar_types = get_scalar_types()
    b_q_type = scalar_types.float4_e2m1f
    torch.manual_seed(0)

    w13 = torch.randint(
        -(2**31),
        2**31 - 1,
        (EXPERTS, HIDDEN // 16, GATE_UP_N * 2),
        dtype=torch.int32,
        device=device,
    )
    w2 = torch.randint(
        -(2**31),
        2**31 - 1,
        (EXPERTS, INTERMEDIATE // 16, HIDDEN * 2),
        dtype=torch.int32,
        device=device,
    )
    s13 = torch.full(
        (EXPERTS, HIDDEN // 32, GATE_UP_N),
        127,
        dtype=torch.uint8,
        device=device,
    ).view(torch.float8_e8m0fnu)
    s2 = torch.full(
        (EXPERTS, INTERMEDIATE // 32, HIDDEN),
        127,
        dtype=torch.uint8,
        device=device,
    ).view(torch.float8_e8m0fnu)

    hidden = torch.randn(1, HIDDEN, dtype=torch.float16, device=device) * 0.1
    topk_ids = torch.arange(TOPK, dtype=torch.int32, device=device).view(1, -1)
    topk_weights = torch.full((1, TOPK), 1.0 / TOPK, dtype=torch.float32, device=device)
    sorted_ids, expert_ids, n_post = moe_align_single_token(topk_ids, BLOCK)
    workspace = marlin_make_workspace(device, 4)

    out13 = torch.empty((TOPK, GATE_UP_N), dtype=torch.float16, device=device)
    act = torch.empty((TOPK, INTERMEDIATE), dtype=torch.float16, device=device)
    out2 = torch.empty((TOPK, HIDDEN), dtype=torch.float16, device=device)

    def gemm_w13():
        return moe_wna16_marlin_gemm(
            hidden,
            out13,
            w13,
            None,
            s13,
            None,
            None,
            None,
            None,
            workspace,
            sorted_ids,
            expert_ids,
            n_post,
            topk_weights,
            moe_block_size=BLOCK,
            top_k=TOPK,
            mul_topk_weights=False,
            is_ep=False,
            b_q_type=b_q_type,
            size_m=1,
            size_n=GATE_UP_N,
            size_k=HIDDEN,
            is_k_full=True,
            use_atomic_add=False,
            use_fp32_reduce=True,
            is_zp_float=False,
        )

    def gemm_w2():
        return moe_wna16_marlin_gemm(
            act,
            out2,
            w2,
            None,
            s2,
            None,
            None,
            None,
            None,
            workspace,
            sorted_ids,
            expert_ids,
            n_post,
            topk_weights,
            moe_block_size=BLOCK,
            top_k=1,
            mul_topk_weights=True,
            is_ep=False,
            b_q_type=b_q_type,
            size_m=TOPK,
            size_n=HIDDEN,
            size_k=INTERMEDIATE,
            is_k_full=True,
            use_atomic_add=False,
            use_fp32_reduce=True,
            is_zp_float=False,
        )

    return gemm_w13, gemm_w2, out13, out2, act


def _candidates(size_n: int, size_k: int, exhaustive: bool, seed_geoms: list[str]):
    geoms = [g for g in SM70_MOE_GEOMETRY_LABELS if _shape_ok(g, size_n, size_k)]
    if exhaustive:
        for g in geoms:
            for sk in SPLIT_K_VALUES:
                for cache in CACHE_VALUES:
                    yield g, sk, cache
        return
    # Phase-1: all geoms, split_k=1, vector_words. Caller then re-invokes
    # with seed_geoms for the fine pass.
    if not seed_geoms:
        for g in geoms:
            yield g, 1, "vector_words"
        return
    for g in seed_geoms:
        for sk in SPLIT_K_VALUES:
            for cache in CACHE_VALUES:
                yield g, sk, cache


def _run_gemm(
    name: str,
    fn,
    out: torch.Tensor,
    ref: torch.Tensor | None,
    size_n: int,
    size_k: int,
    exhaustive: bool,
    top_k_fine: int,
) -> list[SweepRow]:
    rows: list[SweepRow] = []
    phase1_seed: list[str] = []
    passes = [None]
    if not exhaustive:
        passes = [None, "fine"]

    for phase in passes:
        seeds = phase1_seed if phase == "fine" else []
        seen: set[tuple[str, int, str]] = set()
        cand = list(_candidates(size_n, size_k, exhaustive, seeds))
        print(f"== {name} {phase or 'coarse'}: {len(cand)} configs", flush=True)
        for i, (geom, sk, cache) in enumerate(cand, 1):
            key = (geom, sk, cache)
            if key in seen:
                continue
            seen.add(key)
            _set_env(geom, sk, cache)
            try:
                out.zero_()
                fn()
                torch.cuda.synchronize()
                if not torch.isfinite(out).all():
                    raise RuntimeError("non-finite output")
                cur = out.detach().float().clone()
                max_abs = 0.0
                if ref is not None:
                    max_abs = float((cur - ref).abs().max())
                    if max_abs > 0.25:
                        raise RuntimeError(f"max_abs={max_abs:.4f} vs auto")
                us = _time_ms(fn, WARMUP, ITERS) * 1000.0
                rows.append(
                    SweepRow(name, geom, sk, cache, us, True, max_abs=max_abs)
                )
                print(
                    f"  [{i}/{len(cand)}] {geom} sk={sk} {cache}: {us:.1f} us "
                    f"max_abs={max_abs:.4f}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).split("\n")[0][:160]
                rows.append(SweepRow(name, geom, sk, cache, float("inf"), False, err=msg))
                print(f"  [{i}/{len(cand)}] {geom} sk={sk} {cache}: FAIL {msg}", flush=True)
        if phase != "fine":
            ok = [r for r in rows if r.ok and r.gemm == name]
            ok.sort(key=lambda r: r.us)
            phase1_seed = []
            for r in ok:
                if r.geometry not in phase1_seed:
                    phase1_seed.append(r.geometry)
                if len(phase1_seed) >= top_k_fine:
                    break
            print(f"  fine seeds: {phase1_seed}", flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exhaustive", action="store_true")
    parser.add_argument("--top-k-fine", type=int, default=8)
    parser.add_argument(
        "--out",
        default="/tmp/dsv41_marlin_decode_sweep.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        print("need SM70", file=sys.stderr)
        return 2

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = torch.device("cuda")

    import sglang.kernels.ops.moe.moe_wna16_marlin as marlin_mod

    marlin_mod._sm70_marlin_user_tuning = True
    if marlin_mod._load_marlin_v100_op() is None:
        print("marlin_v100 missing", file=sys.stderr)
        return 2

    gemm_w13, gemm_w2, out13, out2, act = _build_problem(device)

    _set_env(None, None, None)
    gemm_w13()
    torch.cuda.synchronize()
    ref13 = out13.detach().float().clone()
    auto13 = _time_ms(gemm_w13, WARMUP, ITERS) * 1000.0
    print(f"auto w13: {auto13:.1f} us", flush=True)

    # Activation for w2 comes from w13; keep one snapshot so every w2 config
    # sees the same input (and matches the auto reference).
    act.copy_(out13[:, :INTERMEDIATE].contiguous())
    gemm_w2()
    torch.cuda.synchronize()
    ref2 = out2.detach().float().clone()
    auto2 = _time_ms(gemm_w2, WARMUP, ITERS) * 1000.0
    print(f"auto w2:  {auto2:.1f} us", flush=True)

    t0 = time.time()
    rows13 = _run_gemm(
        "w13",
        gemm_w13,
        out13,
        ref13,
        GATE_UP_N,
        HIDDEN,
        args.exhaustive,
        args.top_k_fine,
    )
    # Restore act from auto w13 so w2 reference stays valid.
    _set_env(None, None, None)
    gemm_w13()
    torch.cuda.synchronize()
    act.copy_(out13[:, :INTERMEDIATE].contiguous())
    rows2 = _run_gemm(
        "w2",
        gemm_w2,
        out2,
        ref2,
        HIDDEN,
        INTERMEDIATE,
        args.exhaustive,
        args.top_k_fine,
    )

    def best(rows: list[SweepRow]) -> SweepRow:
        ok = [r for r in rows if r.ok]
        if not ok:
            raise RuntimeError("no successful configs")
        return min(ok, key=lambda r: r.us)

    b13, b2 = best(rows13), best(rows2)
    payload = {
        "auto_us": {"w13": auto13, "w2": auto2, "sum": auto13 + auto2},
        "best": {
            "w13": asdict(b13),
            "w2": asdict(b2),
            "sum_us": b13.us + b2.us,
        },
        "layers40_ms": {
            "auto": 40 * (auto13 + auto2) / 1000.0,
            "best_one_marlin": 40 * (b13.us + b2.us) / 1000.0,
            "best_two_marlin_landing": 80 * (b13.us + b2.us) / 1000.0,
        },
        "elapsed_s": time.time() - t0,
        "rows": [asdict(r) for r in rows13 + rows2],
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload["best"], indent=2))
    print(json.dumps(payload["layers40_ms"], indent=2))
    print(f"wrote {args.out} in {payload['elapsed_s']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
