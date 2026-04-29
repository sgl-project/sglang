"""Task 4: separated up/down BF16 sparse-active autotune.

For each (n_active, m_per_expert) cell:
  - Sweep 480 configs grouped by BLOCK_SIZE_M.
  - For each BLOCK_SIZE_M, run moe_align_block_size once (its output depends
    on BLOCK_SIZE_M).
  - For each config in the BLOCK_SIZE_M group, time the up GEMM and the
    down GEMM SEPARATELY via invoke_fused_moe_kernel.
  - Track best up tile and best down tile per BLOCK_SIZE_M.
  - Pick BLOCK_SIZE_M* minimizing (best_up + best_down).

Output (one shard per cell):
  {"n{n}_bse{bse}": {"up": {tile_dict}, "down": {tile_dict},
                     "_t_up_ms": float, "_t_down_ms": float}}

Production fused kernel reads per-direction tiles via
try_get_optimal_moe_config(return_down_config=True), which constrains
up.BLOCK_SIZE_M == down.BLOCK_SIZE_M (asserted at fused_moe_triton_config.py:300-303).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import triton
import triton.language as tl
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _utils import (  # noqa: E402
    KERN_E,
    KERN_K,
    KERN_N,
    KERN_TOP_K,
    bench,
    make_bf16_weights,
    make_uniform_inputs,
    sparse_active_dispatch,
    write_json,
)
from tune_bf16_sparse import get_reduced_search_space  # noqa: E402

N_ACTIVE_VALUES = list(range(4, 64 + 1, 4))  # 4, 8, 12, ..., 64 (16 values)
M_PER_EXPERT_VALUES = [32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536]  # 12 values


def time_kernel(invoke_fn, A, B, bias, C, topk_weights, topk_ids,
                sorted_ids, expert_ids, num_pad, mul_routed_weight, top_k,
                cfg, device):
    def fn():
        invoke_fn(
            A=A, B=B, bias=bias, C=C,
            A_scale=None, B_scale=None, B_zp=None,
            topk_weights=topk_weights, topk_ids=topk_ids,
            sorted_token_ids=sorted_ids, expert_ids=expert_ids,
            num_tokens_post_padded=num_pad,
            mul_routed_weight=mul_routed_weight, top_k=top_k,
            config=cfg, compute_type=tl.bfloat16,
            use_fp8_w8a8=False, use_int8_w8a8=False,
            use_int8_w8a16=False, use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=None,
            b_use_tma=False, c_sorted=False, filter_expert=True,
        )
    try:
        return bench(fn, device, warmup=5, iters=15)
    except Exception:
        return float("inf")


TOP_K_REFINE = 5


def tune_cell(n_active: int, m_per_expert: int, device: torch.device):
    """Two-stage tune: (1) per-direction isolated sweep tracks top-K
    candidates per BLOCK_SIZE_M; (2) K×K refinement pairs top-up with
    top-down and benches the *full fused pipeline* (up + silu + down
    captured as one CUDA graph), picking the (BS_M, up, down) triple
    that minimizes fused latency.

    Why stage 2: stage 1's per-direction sweep at iters=15 has enough
    noise that the top-1 pick can differ from the actually-best tile.
    Pairing the top-K with full-pipeline measurement at higher iters
    de-noises both the per-direction picks AND the BS_M coupling
    (best-up often wants a different BS_M than best-down — they must
    share, and stage 1 doesn't see the trade-off).
    """
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
        outplace_fused_experts,
    )
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels import (
        invoke_fused_moe_kernel,
    )

    from _utils import override_split_config

    w13, w2 = make_bf16_weights(device, seed=0)
    x, topk_w, topk_ids, _ = make_uniform_inputs(m_per_expert, device, seed=0)
    active_set = torch.arange(n_active, device=device, dtype=topk_ids.dtype)
    hot_ids, hot_w = sparse_active_dispatch(topk_ids, topk_w, active_set, "triton")
    M = x.shape[0]

    search = get_reduced_search_space()
    by_block_m = {}
    for c in search:
        by_block_m.setdefault(c["BLOCK_SIZE_M"], []).append(c)

    # Stage 1: per-direction isolated sweep, tracking top-K per BS_M.
    topK_up = {}    # block_m -> list of (cfg, t_ms) sorted ascending
    topK_down = {}

    iter_outer = tqdm(sorted(by_block_m.keys()),
                      desc=f"n{n_active}_bse{m_per_expert} stage1", leave=False)
    for block_m in iter_outer:
        cfgs = by_block_m[block_m]
        sorted_ids, expert_ids, num_pad = moe_align_block_size(
            hot_ids, block_m, KERN_E)
        max_padded = (M * KERN_TOP_K) + (KERN_E + 1) * (block_m - 1)
        cache1 = torch.empty(max_padded, 2 * KERN_N,
                             device=device, dtype=torch.bfloat16)
        cache2 = torch.empty(max_padded, KERN_N,
                             device=device, dtype=torch.bfloat16)
        cache3 = torch.empty(M, KERN_TOP_K, KERN_K,
                             device=device, dtype=torch.bfloat16)

        ups = []
        downs = []
        for cfg in cfgs:
            t_up = time_kernel(
                invoke_fused_moe_kernel,
                A=x, B=w13, bias=None, C=cache1,
                topk_weights=hot_w, topk_ids=hot_ids,
                sorted_ids=sorted_ids, expert_ids=expert_ids, num_pad=num_pad,
                mul_routed_weight=False, top_k=KERN_TOP_K,
                cfg=cfg, device=device,
            )
            t_down = time_kernel(
                invoke_fused_moe_kernel,
                A=cache2, B=w2, bias=None, C=cache3,
                topk_weights=hot_w, topk_ids=hot_ids,
                sorted_ids=sorted_ids, expert_ids=expert_ids, num_pad=num_pad,
                mul_routed_weight=True, top_k=1,
                cfg=cfg, device=device,
            )
            ups.append((cfg, t_up))
            downs.append((cfg, t_down))

        ups.sort(key=lambda p: p[1])
        downs.sort(key=lambda p: p[1])
        topK_up[block_m] = ups[:TOP_K_REFINE]
        topK_down[block_m] = downs[:TOP_K_REFINE]
        iter_outer.set_postfix(
            blkM=block_m,
            up=f"{ups[0][1]:.3f}",
            dn=f"{downs[0][1]:.3f}",
        )

    # Stage 2: paired-fused refinement. For each BS_M, K×K = 25 fused-
    # pipeline benches with override_split_config injected. Higher iters
    # for noise reduction (the search space is now small).
    best_overall = None  # (block_m, up_cfg, down_cfg, t_fused)

    iter_refine = tqdm(sorted(topK_up.keys()),
                       desc=f"n{n_active}_bse{m_per_expert} stage2", leave=False)
    for block_m in iter_refine:
        for up_cfg, _t_up in topK_up[block_m]:
            for down_cfg, _t_down in topK_down[block_m]:
                # The runtime asserts up.BS_M == down.BS_M; both came from
                # the same block_m group so this is satisfied by construction.
                def fn():
                    with override_split_config(up_cfg, down_cfg):
                        outplace_fused_experts(x, w13, w2, hot_w, hot_ids)
                try:
                    t_fused = bench(fn, device, warmup=8, iters=30)
                except Exception:
                    t_fused = float("inf")
                if best_overall is None or t_fused < best_overall[3]:
                    best_overall = (block_m, up_cfg, down_cfg, t_fused)
        iter_refine.set_postfix(
            blkM=block_m,
            best=f"{best_overall[3]:.3f}",
        )

    block_m_star, up_cfg, down_cfg, t_fused = best_overall
    # Find the matching isolated timings for diagnostic record.
    t_up = next(t for c, t in topK_up[block_m_star] if c is up_cfg)
    t_down = next(t for c, t in topK_down[block_m_star] if c is down_cfg)
    return up_cfg, down_cfg, t_up, t_down, t_fused


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-active", type=int, required=True)
    parser.add_argument("--m-per-expert", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    assert torch.cuda.is_available()
    device = torch.device("cuda")
    print(f"[tune-sep] device={torch.cuda.get_device_name(0)} "
          f"n={args.n_active} bse={args.m_per_expert}")
    t0 = time.perf_counter()

    up_cfg, down_cfg, t_up, t_down, t_fused = tune_cell(
        args.n_active, args.m_per_expert, device)
    elapsed = time.perf_counter() - t0
    key = f"n{args.n_active}_bse{args.m_per_expert}"
    out = {key: {
        "up": up_cfg,
        "down": down_cfg,
        "_t_up_ms": round(t_up, 5),
        "_t_down_ms": round(t_down, 5),
        "_t_sum_ms": round(t_up + t_down, 5),
        "_t_fused_ms": round(t_fused, 5),
        "_tune_seconds": round(elapsed, 1),
    }}
    write_json(args.out, out)
    print(f"[tune-sep] {key}: up={up_cfg} t_up={t_up:.4f}ms")
    print(f"[tune-sep] {key}: dn={down_cfg} t_down={t_down:.4f}ms")
    print(f"[tune-sep] {key}: t_fused={t_fused:.4f}ms (sum={t_up+t_down:.4f}ms) "
          f"in {elapsed:.0f}s → {args.out}")


if __name__ == "__main__":
    main()
