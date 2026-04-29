"""Shared helpers for kernel_profile/ — sparse-active dispatch, weight
factories, bench harness wrapper, Zipf routing, table I/O, and tile lookup.

Shapes are fixed to Qwen3-30B-A3B (E=128, K=2048, N=768, top_k=8).
"""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Initialize global server args once per process — outplace_fused_experts
# (and downstream) reads `get_global_server_args().enable_fused_moe_sum_all_reduce`
# at kernel-launch time and crashes if not set.
_server_args_initialized = False


def _ensure_server_args():
    global _server_args_initialized
    if _server_args_initialized:
        return
    from sglang.srt.server_args import (
        ServerArgs,
        set_global_server_args_for_scheduler,
    )
    set_global_server_args_for_scheduler(
        ServerArgs(model_path="dummy", tp_size=1)
    )
    _server_args_initialized = True


_ensure_server_args()


# Qwen3-30B-A3B kernel-level dimensions
KERN_K = 2048
KERN_N = 768
KERN_E = 128
KERN_TOP_K = 8
KERN_GROUP_SIZE = 128
KERN_NUM_BITS = 4

# Bench harness tuning (matches test_efficiency.py)
WARMUP = 20
ITERS = 50
_L2_FLUSH_SIZE = 50 * 1024 * 1024  # 50 MiB

# ---------------------------------------------------------------------------
# Bench harness
# ---------------------------------------------------------------------------

_l2_flush_buf: Optional[torch.Tensor] = None


def _flush_l2(device: torch.device) -> None:
    global _l2_flush_buf
    if _l2_flush_buf is None or _l2_flush_buf.device != device:
        _l2_flush_buf = torch.empty(_L2_FLUSH_SIZE, dtype=torch.int8, device=device)
    _l2_flush_buf.zero_()


def bench(fn, device: torch.device, warmup: int = WARMUP, iters: int = ITERS,
          use_cuda_graph: bool = True) -> float:
    """Median-of-N latency (ms) with L2 flush between iters and optional CUDA graph."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()

    graph = None
    if use_cuda_graph:
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            torch.cuda.synchronize()
            for _ in range(warmup):
                graph.replay()
            torch.cuda.synchronize()
        except Exception:
            graph = None

    if graph is None:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        _flush_l2(device)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        if graph is not None:
            graph.replay()
        else:
            fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Weight factories
# ---------------------------------------------------------------------------


def make_bf16_weights(device: torch.device,
                      seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    w13 = torch.randn(KERN_E, 2 * KERN_N, KERN_K,
                      dtype=torch.bfloat16, device=device, generator=g)
    w2 = torch.randn(KERN_E, KERN_K, KERN_N,
                     dtype=torch.bfloat16, device=device, generator=g)
    return w13, w2


def make_int4_weights(device: torch.device,
                      seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor,
                                              torch.Tensor, torch.Tensor]:
    """Random Marlin-format INT4 weights (packed int32) + dummy bf16 scales."""
    g = torch.Generator(device=device).manual_seed(seed)
    w1 = torch.randint(
        0, 2**31,
        (KERN_E, KERN_K // 16, 2 * KERN_N * (KERN_NUM_BITS // 2)),
        dtype=torch.int32, device=device, generator=g,
    )
    w2 = torch.randint(
        0, 2**31,
        (KERN_E, KERN_N // 16, KERN_K * (KERN_NUM_BITS // 2)),
        dtype=torch.int32, device=device, generator=g,
    )
    s1 = torch.ones(KERN_E, KERN_K // KERN_GROUP_SIZE, 2 * KERN_N,
                    dtype=torch.bfloat16, device=device) * 0.01
    s2 = torch.ones(KERN_E, KERN_N // KERN_GROUP_SIZE, KERN_K,
                    dtype=torch.bfloat16, device=device) * 0.01
    return w1, w2, s1, s2


# ---------------------------------------------------------------------------
# Dispatch builders
# ---------------------------------------------------------------------------


def make_uniform_inputs(m_per_expert: int, device: torch.device,
                        seed: int = 0):
    """Each of E=128 experts gets exactly m_per_expert tokens. Returns
    (x, topk_w, topk_ids, gating). Used for measuring a single-precision path
    where every active expert is loaded uniformly."""
    M_global = m_per_expert * KERN_E // KERN_TOP_K
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(M_global, KERN_K, dtype=torch.bfloat16, device=device, generator=g)
    topk_w = torch.ones(M_global, KERN_TOP_K, dtype=torch.bfloat16, device=device) / KERN_TOP_K
    all_expert_ids = torch.arange(KERN_E, device=device).repeat(m_per_expert)
    perm = torch.randperm(len(all_expert_ids), device=device, generator=g)
    topk_ids = all_expert_ids[perm].reshape(M_global, KERN_TOP_K)
    gating = torch.randn(M_global, KERN_E, dtype=torch.bfloat16, device=device, generator=g)
    return x, topk_w, topk_ids, gating


def sparse_active_dispatch(topk_ids: torch.Tensor, topk_w: torch.Tensor,
                           active_set: torch.Tensor,
                           kernel: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask topk_ids/weights so that only experts in `active_set` are routed.
    Inactive expert IDs are replaced by a sentinel:
      - kernel="marlin": sentinel = E (Marlin treats E as "no expert")
      - kernel="triton": sentinel = -1 (outplace_fused_experts)
    """
    device = topk_ids.device
    is_active = torch.zeros(KERN_E, dtype=torch.bool, device=device)
    is_active[active_set] = True
    in_set = is_active[topk_ids]
    if kernel == "marlin":
        sentinel = torch.tensor(KERN_E, device=device)
    elif kernel == "triton":
        sentinel = torch.tensor(-1, device=device)
    else:
        raise ValueError(f"unknown kernel '{kernel}'")
    new_ids = torch.where(in_set, topk_ids, sentinel)
    new_w = topk_w * in_set.to(topk_w.dtype)
    return new_ids, new_w


def make_zipf_routing(M_global: int, device: torch.device,
                      seed: int = 42, alpha: float = 1.1):
    """Zipf-shaped per-expert routing for one synthetic prefill chunk.

    Returns (x, topk_w, topk_ids, gating, expert_freq) where expert_freq is
    a length-E count tensor of how many times each expert was routed across
    the whole batch.
    Matches the synthetic generator in scripts/heter_moe_collect_routing.py.
    """
    rng = np.random.default_rng(seed + M_global)
    zipf_weights = 1.0 / np.arange(1, KERN_E + 1) ** alpha
    zipf_weights /= zipf_weights.sum()
    perm = rng.permutation(KERN_E)
    shuffled = zipf_weights[perm]
    total_tokens = M_global * KERN_TOP_K
    counts = rng.multinomial(total_tokens, shuffled).astype(np.int64)
    # Build a topk_ids with that many tokens routed to each expert,
    # interpreting "tokens" as flat slots across (M_global, top_k).
    flat_ids = np.repeat(np.arange(KERN_E), counts)
    rng.shuffle(flat_ids)
    flat_ids = flat_ids[: M_global * KERN_TOP_K]
    if flat_ids.shape[0] < M_global * KERN_TOP_K:
        # Pad with random experts (shouldn't happen with multinomial)
        extra = np.random.randint(0, KERN_E,
                                  M_global * KERN_TOP_K - flat_ids.shape[0])
        flat_ids = np.concatenate([flat_ids, extra])
    topk_ids = torch.from_numpy(flat_ids).reshape(M_global, KERN_TOP_K).to(
        device=device, dtype=torch.int32)
    topk_w = (
        torch.ones(M_global, KERN_TOP_K, dtype=torch.bfloat16, device=device)
        / KERN_TOP_K
    )
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(M_global, KERN_K, dtype=torch.bfloat16, device=device, generator=g)
    gating = torch.randn(M_global, KERN_E, dtype=torch.bfloat16, device=device, generator=g)
    expert_freq = torch.from_numpy(counts).to(device=device)
    return x, topk_w, topk_ids, gating, expert_freq


# ---------------------------------------------------------------------------
# Tile config lookup (Task 2.bench / Task 3)
# ---------------------------------------------------------------------------


def hierarchical_lookup(configs: Dict[str, dict], n_active: int,
                        m_per_expert: int) -> Tuple[str, dict]:
    """Find nearest-(n_active, m_per_expert) tile config in the JSON.
    Hierarchical: nearest n first, then nearest bse within that n.
    Returns (key_used, tile_dict).
    """
    parsed = {}  # n -> {bse -> (key, tile)}
    for k, v in configs.items():
        # key format "n{n}_bse{m}"
        n_str, m_str = k.split("_")
        n = int(n_str[1:])
        m = int(m_str[3:])
        parsed.setdefault(n, {})[m] = (k, v)
    if not parsed:
        raise ValueError("empty config dict")
    n_keys = sorted(parsed.keys())
    n_best = min(n_keys, key=lambda nn: abs(nn - n_active))
    bse_keys = sorted(parsed[n_best].keys())
    m_best = min(bse_keys, key=lambda mm: abs(mm - m_per_expert))
    return parsed[n_best][m_best]


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------


def write_csv(path: str, header: List[str], rows: List[list]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def read_csv(path: str) -> Tuple[List[str], List[list]]:
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        rows = [row for row in r]
    return header, rows


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def read_json(path: str):
    with open(path) as f:
        return json.load(f)
