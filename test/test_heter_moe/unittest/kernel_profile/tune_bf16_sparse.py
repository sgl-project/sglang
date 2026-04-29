"""Task 2.prelim: autotune BF16 (Triton) sparse-active path.

For each (n_active, m_per_expert) cell, sweeps the 1920-config compute-bound
search space and writes the best tile. Output keys: "n{n}_bse{bse}".

Single-GPU per invocation; shell launcher (run_tune.sh) splits cells across
GPUs 0..7 and merges per-shard JSONs.

Usage:
  CUDA_VISIBLE_DEVICES=0 python tune_bf16_sparse.py \
      --n-active 8 --m-per-expert 32 \
      --out results/bf16_sparse_configs.shard.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import triton
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..",
    "benchmark", "kernels", "fused_moe_triton")))

from common_utils import get_configs_compute_bound  # noqa: E402

from _utils import (  # noqa: E402
    KERN_E,
    bench,
    make_bf16_weights,
    make_uniform_inputs,
    sparse_active_dispatch,
    write_json,
)


def get_reduced_search_space():
    """A reduced search space to keep per-cell autotune wall under ~10 min.

    Drops num_stages={2,5}, BLOCK_SIZE_K=256, GROUP_SIZE_M=1 from the full
    1920-config space. Keeps the dominant axis BLOCK_SIZE_M ∈ {16,32,64,128,256}
    and BLOCK_SIZE_N ∈ {32,64,128,256}. Result: 2*5*2*4*2*3 = 480 configs.
    """
    out = []
    for num_stages in [3, 4]:
        for block_m in [16, 32, 64, 128, 256]:
            for block_k in [64, 128]:
                for block_n in [32, 64, 128, 256]:
                    for num_warps in [4, 8]:
                        for group_size in [16, 32, 64]:
                            out.append({
                                "BLOCK_SIZE_M": block_m,
                                "BLOCK_SIZE_N": block_n,
                                "BLOCK_SIZE_K": block_k,
                                "GROUP_SIZE_M": group_size,
                                "num_warps": num_warps,
                                "num_stages": num_stages,
                            })
    return out

N_ACTIVE_VALUES = [8, 16, 32, 48, 64]
M_PER_EXPERT_VALUES = [32, 64, 128, 256, 512, 1024, 2048, 4096]


def bench_one_config(config, x, w13, w2, hot_w, hot_ids, device):
    """Time outplace_fused_experts with a forced tile config via override."""
    from sglang.srt.layers.moe.fused_moe_triton import override_config
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
        outplace_fused_experts,
    )

    def fn():
        with override_config(config):
            outplace_fused_experts(x, w13, w2, hot_w, hot_ids)

    try:
        return bench(fn, device, warmup=8, iters=20)
    except triton.runtime.autotuner.OutOfResources:
        return float("inf")
    except Exception:
        return float("inf")


def tune_cell(n_active: int, m_per_expert: int, device: torch.device):
    w13, w2 = make_bf16_weights(device, seed=0)
    x, topk_w, topk_ids, _ = make_uniform_inputs(m_per_expert, device, seed=0)
    active_set = torch.arange(n_active, device=device, dtype=topk_ids.dtype)
    hot_ids, hot_w = sparse_active_dispatch(topk_ids, topk_w, active_set, "triton")

    search = get_reduced_search_space()
    best_lat = float("inf")
    best_cfg = None

    print(f"[tune] n={n_active} bse={m_per_expert} | search={len(search)} configs")
    iter_ = tqdm(search, desc=f"n{n_active}_bse{m_per_expert}", leave=False)
    for cfg in iter_:
        lat = bench_one_config(cfg, x, w13, w2, hot_w, hot_ids, device)
        if lat < best_lat:
            best_lat = lat
            best_cfg = cfg
            iter_.set_postfix(best_ms=f"{best_lat:.4f}")
    return best_cfg, best_lat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-active", type=int, required=True)
    parser.add_argument("--m-per-expert", type=int, required=True)
    parser.add_argument("--out", type=str, required=True,
                        help="path to per-shard JSON")
    args = parser.parse_args()

    assert torch.cuda.is_available()
    device = torch.device("cuda")
    print(f"[tune] device={torch.cuda.get_device_name(0)}")
    t0 = time.perf_counter()

    cfg, lat = tune_cell(args.n_active, args.m_per_expert, device)
    elapsed = time.perf_counter() - t0
    key = f"n{args.n_active}_bse{args.m_per_expert}"

    out = {
        key: {
            **cfg,
            "_lat_ms": round(lat, 5),
            "_tune_seconds": round(elapsed, 1),
        }
    }
    write_json(args.out, out)
    print(f"[tune] {key}: best={cfg} lat={lat:.4f}ms in {elapsed:.0f}s → {args.out}")


if __name__ == "__main__":
    main()
