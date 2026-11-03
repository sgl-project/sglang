# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tuned A8W4 MoE dispatch (gfx942, FlyDSL): picks the fastest kernel combination
per batch size from a CSV produced by aiter's ``doc-mimo/kernels/a8w4_moe_tune.py``
(copied here as ``configs/a8w4_moe_tuned.txt``).

Every candidate consumes the SAME fd2 weight dict (``w1``/``w2``/``w1_scale``/
``w2_scale``/``w1_exp`` after ``prep_w1_fast`` / ``prep_w2_fast(..., 2)``), so the
kernel can change per batch size without re-preparing weights.

Kinds (CSV column ``kind``):
  2k   moe_sorting -> x row-quant -> gemm1 (q8s epilogue) -> gemm2 (atomic scatter);
       block_m, gemm1 b_cache_mod (g1_bcm), gemm2 "base" | "nt4"
  1k   fused single kernel (test_a8w4_fmoe_bm64): bcm, waves_per_eu, ksplit
  1kq  persistent work-queue fused kernel (test_a8w4_fmoe_bm64q): bcm. Not CUDA-graph
       safe (per-launch epoch / queue counters): replaced by GRAPH_SAFE_1KQ while a
       graph is being captured.
num_tokens is rounded up to 2^N (clamped to the table range) and the row of
(model_dim, inter_dim, E, topk, routing) is used; DEFAULT_CFG without a match.
"""

import csv
import functools
import os

import torch
from aiter import dtypes
from aiter.fused_moe import moe_sorting
from aiter.ops.flydsl.kernels.moe_2stage_a16wmix.a8w4 import quantize_rows

from . import test_a8w4_fmoe_bm64 as F
from . import test_a8w4_fmoe_bm64q as Q
from . import test_a8w4_gemm1 as G1
from . import test_a8w4_gemm2 as G2

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "configs", "a8w4_moe_tuned.txt")
DEFAULT_CFG = dict(kind="2k", block_m=64, g1_bcm=1, g2="base")
GRAPH_SAFE_1KQ = dict(kind="1k", block_m=64, bcm=1, wpe=2, ksplit=1)


def config_name(cfg):
    if cfg["kind"] == "2k":
        return f"2k bm{cfg['block_m']} g1bcm{cfg['g1_bcm']} g2{cfg['g2']}"
    if cfg["kind"] == "1k":
        return f"1k bm64 bcm{cfg['bcm']} w{cfg['wpe']} ks{cfg['ksplit']}"
    return f"1kq bm64q bcm{cfg['bcm']}"


def _row_to_cfg(r):
    kind = r["kind"]
    if kind == "2k":
        return dict(kind=kind, block_m=int(r["block_m"]), g1_bcm=int(r["g1_bcm"]), g2=r["g2"])
    if kind == "1k":
        return dict(kind=kind, block_m=64, bcm=int(r["bcm"]), wpe=int(r["wpe"]),
                    ksplit=int(r["ksplit"]))
    return dict(kind=kind, block_m=64, bcm=int(r["bcm"]))


@functools.lru_cache(maxsize=8)
def _load_table(path):
    """{(H, I, E, topk, routing): sorted [(num_tokens, cfg)]}; {} if no file."""
    table = {}
    if not path or not os.path.exists(path):
        return table
    with open(path) as fh:
        for r in csv.DictReader(fh):
            key = (int(r["model_dim"]), int(r["inter_dim"]), int(r["experts"]),
                   int(r["topk"]), r["routing"])
            table.setdefault(key, []).append((int(r["num_tokens"]), _row_to_cfg(r)))
    return {k: sorted(v, key=lambda t: t[0]) for k, v in table.items()}


@functools.lru_cache(maxsize=1024)
def lookup_config(num_tokens, model_dim, inter_dim, experts, topk, routing="uniform",
                  csv_path=DEFAULT_CSV):
    rows = _load_table(csv_path).get((model_dim, inter_dim, experts, topk, routing))
    if not rows:
        return DEFAULT_CFG
    t = 1 << max(0, (int(num_tokens) - 1).bit_length())
    for n, cfg in rows:
        if n >= t:
            return cfg
    return rows[-1][1]


def run_config(cfg, x, W, topk_weights, topk_ids):
    """Run one kernel combination end to end (moe_sorting + x quant + kernels);
    returns bf16 [tokens, model_dim]."""
    E, model_dim, _ = W["w2"].shape
    topk = topk_ids.shape[1]
    if cfg["kind"] == "1kq" and torch.cuda.is_current_stream_capturing():
        cfg = GRAPH_SAFE_1KQ
    if cfg["kind"] == "1k":
        return F.fused_moe_a8w4_bm64(x, W, topk_weights, topk_ids, b_cache_mod=cfg["bcm"],
                                     waves_per_eu=cfg["wpe"], ksplit=cfg["ksplit"])
    if cfg["kind"] == "1kq":
        return Q.fused_moe_a8w4_bm64q(x, W, topk_weights, topk_ids, b_cache_mod=cfg["bcm"])
    assert cfg["kind"] == "2k", cfg
    bm = cfg["block_m"]
    inter = W["w1"].shape[1] // 2
    groups = inter // 128
    sids, sw, seids, nv, out = moe_sorting(topk_ids, topk_weights, E, model_dim,
                                           dtypes.bf16, bm)[:5]
    x8, xs = quantize_rows(x)
    sorted_size = seids.numel() * bm
    buf = torch.empty(sorted_size * (inter + groups * 4), dtype=torch.uint8, device=x.device)
    G1.a8w4_gemm1(x8, xs, W["w1"], W["w1_scale"], W["w1_exp"], sids, seids, nv, buf,
                  num_experts=E, tile_m=bm, epilog="q8s", inter_dim=inter, topk=topk,
                  tile_n=128, b_cache_mod=cfg["g1_bcm"], xcd_swizzle=1, dyn_m=bm > 32,
                  fast_dec=W["w1_fast_dec"])
    nd = sorted_size * inter
    a2 = buf[:nd].view(torch.float8_e4m3fnuz).view(sorted_size, inter)
    a2s = buf[nd:].view(torch.float32).view(sorted_size, groups)
    kw = dict(tile_n=128, b_cache_mod=2, xcd_swizzle=0, dyn_rows=True)
    if cfg["g2"] == "nt4":
        kw.update(n_tiles=4, epilog="direct64", row_vgpr=True)
    G2.a8w4_gemm2(a2, a2s, W["w2"], W["w2_scale"], sids, sw, seids, nv, out,
                  num_experts=E, tile_m=bm, a_group=128, fast_dec=W["w2_fast_dec"], **kw)
    return out


def tuned_moe(x, weights, topk_weights, topk_ids, *, csv_path=DEFAULT_CSV, routing="uniform"):
    """A8W4 MoE (bf16 x [tokens, H] -> bf16 [tokens, H]) with the tuned kernel
    combination for this batch size. ``weights``: fd2 dict (w1/w2_fast_dec == 2)."""
    assert weights["w1_fast_dec"] == 2 and weights["w2_fast_dec"] == 2, "needs fd2 weights"
    E, model_dim, _ = weights["w2"].shape
    cfg = lookup_config(x.shape[0], model_dim, weights["w1"].shape[1] // 2, E,
                        topk_ids.shape[1], routing, csv_path)
    return run_config(cfg, x, weights, topk_weights, topk_ids)
