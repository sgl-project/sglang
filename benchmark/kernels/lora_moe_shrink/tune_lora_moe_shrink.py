"""
Auto-tuning script for the MoE LoRA shrink (LoRA A) kernel.

The kernel tuned here is `_moe_lora_shrink_splitk_kernel` from
sglang/srt/lora/triton_ops/virtual_experts.py. It performs the LoRA A
(shrink) grouped GEMM of the merged-experts MoE LoRA path: it projects the
MoE hidden states down to the LoRA rank, one virtual expert per routed block.

Kernel shape terminology:
    E  num (virtual) experts (kernel weight dim 0, e.g. 64, 128, 256)
    N  max LoRA rank      (kernel N / output dim, e.g. 16, 32, 64)
    K  hidden size        (kernel K / reduction dim, e.g. 512, 768, 2048, 7168)
    M  number of tokens   (the config key; routed rows are M * top_k)

This produces one JSON config file per (E, N, K, device), keyed by token count M,
in the same spirit as the fused_moe_triton configs. Each config file is split
into two regimes:

  * Decode  (small M): BLOCK_SIZE_M and BLOCK_SIZE_K are pinned (bm=16, bk=256).
                       We tune num_warps, num_stages and SPLIT_K.
  * Prefill (large M): BLOCK_SIZE_K is pinned (bk=256). We tune BLOCK_SIZE_M,
                       num_warps, num_stages and SPLIT_K.

BLOCK_SIZE_N is always the rank (N) and BLOCK_SIZE_K always 256, so neither is
tuned or stored; GROUP_SIZE_M is pinned to 1.

Config files are written to
    python/sglang/srt/lora/triton_ops/moe_shrink_configs/triton_<ver>/
and loaded automatically at runtime by moe_lora_shrink_config.py.

Usage:
    # Tune the default grid:
    #   experts {64,96,128,192,256,384} x ranks {16,32,64} x hidden {512,768,2048,7168}
    python benchmark/kernels/lora_moe_shrink/tune_lora_moe_shrink.py

    # Restrict the grid
    python benchmark/kernels/lora_moe_shrink/tune_lora_moe_shrink.py \
        --num-experts 128 256 --ranks 16 64 --hidden-sizes 2048 7168

    # Adjust the representative MoE top_k used while benchmarking
    python benchmark/kernels/lora_moe_shrink/tune_lora_moe_shrink.py --top-k 8
"""

import argparse
import json
import math
import os
import statistics
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import triton
import triton.testing

from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)
from sglang.srt.lora.triton_ops.moe_lora_shrink_config import (
    get_moe_lora_shrink_config_file_name,
)
from sglang.srt.lora.triton_ops.virtual_experts import _moe_lora_shrink_splitk_kernel

# Pinned (non-tuned) block dimensions.
BLOCK_SIZE_K = 256
GROUP_SIZE_M = 1


def estimate_shrink_smem_bytes(
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    elem_size: int = 2,
) -> int:
    """Approximate Triton shared-memory use of the shrink MMA operand pipeline.

    The kernel multi-buffers both GEMM operands (A tile BLOCK_M x BLOCK_K and
    B tile BLOCK_K x BLOCK_N) across ``num_stages`` pipeline stages. For bf16/fp16
    (``elem_size=2``) this matches the observed launch limit:
    ``bm=64, bk=256, bn=64, stages=4`` -> 262144 B, which exceeds the 227 KB
    opt-in cap and fails to launch. Used here to prune configs that cannot fit
    before we waste time compiling them.
    """
    per_stage = (block_m * block_k + block_k * block_n) * elem_size
    return num_stages * per_stage


def get_device_smem_cap_bytes(device_index: int = 0) -> int:
    """Per-block shared-memory budget for the given CUDA/HIP device.

    Prefers the opt-in maximum (227 KB on Hopper/Blackwell); falls back to the
    regular per-block limit (e.g. 64 KB LDS on AMD) and finally a conservative
    64 KB when device properties expose neither.
    """
    props = torch.cuda.get_device_properties(device_index)
    cap = getattr(props, "shared_memory_per_block_optin", 0)
    if not cap:
        cap = getattr(props, "shared_memory_per_block", 0)
    return int(cap) if cap else 64 * 1024


# Decode pins BLOCK_SIZE_M; prefill tunes it from this candidate set.
DECODE_BLOCK_SIZE_M = 16
PREFILL_BLOCK_SIZE_M = [16, 32, 64, 128]
# Tuning axes. num_warps=1 is excluded: with BLOCK_SIZE_K=256 the reduction and
# the BLOCK_M x rank output tile need N/K work spread across warps, so a single
# warp serializes the MMA and never wins, even at bs=16.
NUM_WARPS = [2, 4]
NUM_STAGES = [1, 2, 3, 4]
SPLIT_K = [1, 2, 3, 4, 5, 6, 7, 8]


# --- L2-cold benchmarking (adapted from sgl-project/sglang#26706, which adapts
# the flashinfer L2-rotation utility). Two effects combined:
#   1. flush L2 (zero a 5x-L2 buffer) before each timed graph replay, and
#   2. rotate the read/written buffers within the captured graph so back-to-back
#      calls don't reuse hot-L2 weights.
# Without this, capturing N calls into the SAME weight buffer measures L2-HOT
# latency -- optimistic, especially for small E whose weight fits in L2.
_L2_SAFE_RATIO = 5
_L2_BYTES: Dict[int, int] = {}
_FLUSH_BUFFERS: Dict[int, torch.Tensor] = {}


def _l2_bytes(device_index: int) -> int:
    if device_index not in _L2_BYTES:
        _L2_BYTES[device_index] = torch.cuda.get_device_properties(
            device_index
        ).L2_cache_size
    return _L2_BYTES[device_index]


def _flush_buffer(device_index: int) -> torch.Tensor:
    """Cached 5x-L2 uint8 buffer; zeroing it evicts the L2 cache."""
    if device_index not in _FLUSH_BUFFERS:
        size = int(_l2_bytes(device_index) * _L2_SAFE_RATIO)
        _FLUSH_BUFFERS[device_index] = torch.empty(
            size, device=f"cuda:{device_index}", dtype=torch.uint8
        )
    return _FLUSH_BUFFERS[device_index]


def rotation_count(
    nbytes: int, device_index: int, min_rot: int = 2, cap: int = 100
) -> int:
    """Number of buffer copies so the working set exceeds ~5x L2 (cold cache).

    Returns 1 when the buffers already dwarf L2 (cache effects negligible).
    """
    thresh = _l2_bytes(device_index) * _L2_SAFE_RATIO
    if nbytes <= 0 or nbytes >= thresh:
        return 1
    return min(cap, max(min_rot, math.ceil(thresh / nbytes) + 1))


def bench(
    launch_i: Callable[[int], None],
    rotate_count: int,
    device_index: int,
    target: int = 100,
    timed_reps: int = 20,
) -> float:
    """Per-call milliseconds with a cold L2.

    Captures ``loop_count`` rotated ``launch_i(i % rotate_count)`` calls in ONE
    CUDA graph (amortizing per-launch overhead), flushes L2 before each timed
    replay, and divides replay time by ``loop_count``. Returns the median.
    """
    loop_count = math.ceil(target / rotate_count) * rotate_count
    flush = _flush_buffer(device_index)
    stream = torch.cuda.current_stream()

    # Warm/compile every rotation index on a side stream before capture (capture
    # cannot trigger Triton compilation).
    s = torch.cuda.Stream()
    s.wait_stream(stream)
    with torch.cuda.stream(s):
        for i in range(rotate_count):
            launch_i(i)
    stream.wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(loop_count):
            launch_i(i % rotate_count)
    g.replay()  # warm the graph once
    torch.cuda.synchronize()

    times: List[float] = []
    for _ in range(timed_reps):
        flush.zero_()  # cold the L2 cache before each measurement
        torch.cuda.synchronize()
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)
        tic.record(stream)
        g.replay()
        toc.record(stream)
        stream.synchronize()
        times.append(tic.elapsed_time(toc) / loop_count)
    return float(statistics.median(times))


def align_routing(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size_m: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align fixed routing to a given BLOCK_SIZE_M.

    Returns (sorted_token_ids, expert_ids, num_tokens_post_padded). Must be
    recomputed per BLOCK_SIZE_M, since expert_ids / sorted_token_ids are aligned
    to the block size the kernel will be launched with.

    Match the production routing path by trimming the native align helper's
    worst-case allocation to the tight upper bound used for the launch grid.
    """
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_size_m, num_experts
    )

    num_tokens = topk_ids.numel()
    max_nonempty = min(num_tokens, num_experts)
    tight_padded = (
        triton.cdiv(num_tokens + max_nonempty * (block_size_m - 1), block_size_m)
        * block_size_m
    )
    return (
        sorted_token_ids[:tight_padded],
        expert_ids[: tight_padded // block_size_m],
        num_tokens_post_padded,
    )


def benchmark_config(
    config: Dict[str, Any],
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k: int,
    N: int,
    K: int,
) -> Optional[float]:
    """Benchmark a single shrink config. Returns per-call ms or None on failure."""
    kernel = _moe_lora_shrink_splitk_kernel

    block_size_m = config["BLOCK_SIZE_M"]
    # BLOCK_SIZE_N and BLOCK_SIZE_K are not tuned: N is always the LoRA rank
    # (a power of two) and K is pinned to 256, matching the runtime launcher.
    block_size_n = triton.next_power_of_2(N)
    block_size_k = BLOCK_SIZE_K
    group_size_m = config["GROUP_SIZE_M"]
    split_k = config["SPLIT_K"]

    base_grid = triton.cdiv(sorted_token_ids.shape[0], block_size_m) * triton.cdiv(
        N, block_size_n
    )
    grid = (split_k * base_grid,)

    device_index = (
        weight.device.index
        if weight.device.index is not None
        else torch.cuda.current_device()
    )
    # Rotate the read/written buffers (the weight dominates) so back-to-back graph
    # calls hit a cold L2. Routing tensors are tiny/read-only and left shared.
    nbytes = (
        weight.numel() * weight.element_size()
        + hidden_states.numel() * hidden_states.element_size()
        + output.numel() * output.element_size()
    )
    rot = rotation_count(nbytes, device_index)
    weights = [weight] + [weight.clone() for _ in range(rot - 1)]
    hiddens = [hidden_states] + [hidden_states.clone() for _ in range(rot - 1)]
    outputs = [output] + [torch.zeros_like(output) for _ in range(rot - 1)]
    num_valid_tokens = topk_ids.numel()

    def launch(i: int):
        h, w, o = hiddens[i], weights[i], outputs[i]
        kernel[grid](
            h,
            w,
            o,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            N,
            K,
            num_valid_tokens,
            h.stride(0),
            h.stride(1),
            w.stride(0),
            w.stride(1),
            w.stride(2),
            o.stride(0),
            o.stride(1),
            top_k=top_k,
            BLOCK_SIZE_M=block_size_m,
            BLOCK_SIZE_N=block_size_n,
            BLOCK_SIZE_K=block_size_k,
            GROUP_SIZE_M=group_size_m,
            SPLIT_K=split_k,
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
        )

    try:
        # Validate the launch (catches SMEM/compile failures), then time it with
        # the L2-cold CUDA-graph bench (rotated buffers + L2 flush per replay).
        launch(0)
        torch.cuda.synchronize()
        return bench(launch, rot, device_index)
    except Exception:
        return None


def get_search_space(
    N: int,
    K: int,
    is_decode: bool,
    smem_cap: int,
    elem_size: int,
) -> List[Dict[str, Any]]:
    """Candidate configs for one regime.

    Decode pins BLOCK_SIZE_M=16; prefill sweeps PREFILL_BLOCK_SIZE_M.
    SPLIT_K candidates are clamped to what K/BLOCK_SIZE_K can support.
    num_stages combos whose operand pipeline would exceed the device shared-memory
    cap are dropped up front (they would fail to launch anyway).
    """
    max_split_k = max(1, K // BLOCK_SIZE_K)
    split_k_choices = [s for s in SPLIT_K if s <= max_split_k] or [1]
    bm_choices = [DECODE_BLOCK_SIZE_M] if is_decode else PREFILL_BLOCK_SIZE_M
    # Mirror the runtime launcher, which uses next_power_of_2(N) for BLOCK_SIZE_N.
    block_size_n = triton.next_power_of_2(N)

    configs = []
    dropped = 0
    for bm in bm_choices:
        for split_k in split_k_choices:
            for num_warps in NUM_WARPS:
                for num_stages in NUM_STAGES:
                    if (
                        estimate_shrink_smem_bytes(
                            bm, block_size_n, BLOCK_SIZE_K, num_stages, elem_size
                        )
                        > smem_cap
                    ):
                        dropped += 1
                        continue
                    # BLOCK_SIZE_N (=rank) and BLOCK_SIZE_K (=256) are derived by
                    # the runtime, so they are intentionally not stored.
                    configs.append(
                        {
                            "BLOCK_SIZE_M": bm,
                            "GROUP_SIZE_M": GROUP_SIZE_M,
                            "num_warps": num_warps,
                            "num_stages": num_stages,
                            "SPLIT_K": split_k,
                        }
                    )
    if dropped:
        print(
            f"  (skipped {dropped} configs exceeding SMEM cap "
            f"{smem_cap // 1024} KB for N={N}, K={K})"
        )
    return configs


def sort_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Order config keys for consistent JSON output."""
    ordered = {}
    for key in [
        "BLOCK_SIZE_M",
        "GROUP_SIZE_M",
        "num_warps",
        "num_stages",
        "SPLIT_K",
    ]:
        if key in config:
            ordered[key] = config[key]
    return ordered


def config_signature(config: Dict[str, Any]) -> str:
    """Stable compact representation used to group identical tuned configs."""
    return json.dumps(sort_config(config), sort_keys=True, separators=(",", ":"))


def format_range(start: int, end: int) -> str:
    """Human-readable inclusive integer range."""
    return str(start) if start == end else f"{start}-{end}"


def routed_counts_to_m_values(
    name: str,
    routed_counts: Optional[List[int]],
    top_k: int,
) -> Optional[List[int]]:
    """Convert routed-row counts to input-token counts for runtime-compatible keys."""
    if routed_counts is None:
        return None

    invalid = [count for count in routed_counts if count <= 0 or count % top_k != 0]
    if invalid:
        raise ValueError(
            f"{name} values must be positive multiples of top_k={top_k}; "
            f"invalid values: {invalid}"
        )
    return sorted(set(count // top_k for count in routed_counts))


def make_balanced_topk_ids(
    tokens_per_expert: int,
    active_experts: int,
    top_k: int,
    device: torch.device,
) -> torch.Tensor:
    """Create routing with exactly tokens_per_expert rows per active expert."""
    if active_experts < top_k:
        raise ValueError(
            f"active_experts={active_experts} must be >= top_k={top_k} to avoid "
            "duplicate experts within a token"
        )

    num_routed_tokens = tokens_per_expert * active_experts
    if num_routed_tokens % top_k != 0:
        raise ValueError(
            f"tokens_per_expert * active_experts must be divisible by top_k; "
            f"got {tokens_per_expert} * {active_experts} for top_k={top_k}"
        )

    flat = torch.arange(active_experts, dtype=torch.int32, device=device).repeat(
        tokens_per_expert
    )
    return flat.view(num_routed_tokens // top_k, top_k)


def save_config(configs: Dict[int, Dict[str, Any]], E: int, N: int, K: int) -> str:
    """Write tuned configs (keyed by M) to the standard config dir. Returns path."""
    filename = get_moe_lora_shrink_config_file_name(E, N, K)

    triton_version = triton.__version__
    version_dir = f"triton_{triton_version.replace('.', '_')}"
    config_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "..",
        "python",
        "sglang",
        "srt",
        "lora",
        "triton_ops",
        "moe_shrink_configs",
        version_dir,
    )
    config_dir = os.path.normpath(config_dir)
    os.makedirs(config_dir, exist_ok=True)

    filepath = os.path.join(config_dir, filename)
    # JSON object keys must be strings; sort numerically for readability.
    out = {str(m): configs[m] for m in sorted(configs.keys())}
    with open(filepath, "w") as f:
        json.dump(out, f, indent=4)
        f.write("\n")
    return filepath


def tune_one(
    E: int,
    N: int,
    K: int,
    decode_batch_sizes: List[int],
    prefill_num_tokens: List[int],
    top_k: int,
    dtype: torch.dtype,
    output_dtype: torch.dtype,
    device: torch.device,
    seed: int,
    decode_tokens_per_expert: Optional[List[int]],
    active_experts: Optional[int],
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Tuple[float, Dict[str, Any]]]]:
    """Tune the shrink kernel for one (experts E, rank N, hidden K).

    Returns (configs, results).
    """
    num_experts = E
    print(f"\n{'='*80}")
    print(f"Tuning MoE LoRA shrink — E(experts)={E}, N(rank)={N}, K(hidden)={K}")
    print(f"{'='*80}")

    weight = torch.randn(num_experts, N, K, device=device, dtype=dtype)
    elem_size = weight.element_size()
    smem_cap = get_device_smem_cap_bytes(device.index or 0)

    best_configs: Dict[int, Dict[str, Any]] = {}
    results: Dict[int, Tuple[float, Dict[str, Any]]] = {}

    controlled_active_experts = min(active_experts or num_experts, num_experts)
    if decode_tokens_per_expert is None:
        regimes = [(m, True, None) for m in decode_batch_sizes]
    else:
        regimes = []
        for tokens_per_expert in decode_tokens_per_expert:
            routed_tokens = tokens_per_expert * controlled_active_experts
            if routed_tokens % top_k != 0:
                raise ValueError(
                    f"tokens_per_expert={tokens_per_expert} with "
                    f"active_experts={controlled_active_experts} produces "
                    f"{routed_tokens} routed rows, not divisible by top_k={top_k}"
                )
            regimes.append((routed_tokens // top_k, True, tokens_per_expert))
    regimes += [(m, False, None) for m in prefill_num_tokens]

    for M, is_decode, tokens_per_expert in regimes:
        search = get_search_space(N, K, is_decode, smem_cap, elem_size)
        # Deterministic per-shape routing: the chosen config (esp. split_k) is
        # sensitive to how tokens cluster across experts, so seed per (seed, E, N, M)
        # to make picks reproducible and independent of what else is tuned.
        torch.manual_seed(
            (seed * 1_000_003 + num_experts * 9176 + N * 251 + M) & 0x7FFFFFFF
        )
        hidden_states = torch.randn(M, K, device=device, dtype=dtype)
        output = torch.zeros(M * top_k, N, device=device, dtype=output_dtype)

        # Deterministic routing only (no randomness). Default path: balanced
        # round-robin keyed by M -- token i routes to experts
        # [i*top_k .. i*top_k+top_k-1] mod E. This covers both the under-saturated
        # (M*top_k < E -> M*top_k experts, 1 token each) and saturated (all E
        # experts, evenly loaded) regimes, for decode and prefill alike. The
        # tokens_per_expert path forces an exact per-expert load instead.
        if tokens_per_expert is None:
            topk_ids = (
                torch.arange(M * top_k, device=device, dtype=torch.int32) % num_experts
            ).reshape(M, top_k)
        else:
            topk_ids = make_balanced_topk_ids(
                tokens_per_expert,
                controlled_active_experts,
                top_k,
                device,
            )
        # Cache alignment per BLOCK_SIZE_M to avoid recomputing it for every
        # (num_warps, num_stages, split_k) combination.
        align_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        best_config = None
        best_time = float("inf")

        for i, config in enumerate(search):
            bm = config["BLOCK_SIZE_M"]
            if bm not in align_cache:
                align_cache[bm] = align_routing(topk_ids, num_experts, bm)
            sorted_token_ids, expert_ids, num_tokens_post_padded = align_cache[bm]

            t = benchmark_config(
                config,
                hidden_states,
                weight,
                output,
                topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                top_k,
                N,
                K,
            )
            if t is not None and t < best_time:
                best_time = t
                best_config = config
            if (i + 1) % 20 == 0:
                routing_desc = (
                    f"tokens/expert={tokens_per_expert}, "
                    f"active_experts={controlled_active_experts}, "
                    if tokens_per_expert is not None
                    else ""
                )
                print(
                    f"  M={M} ({'decode' if is_decode else 'prefill'}): "
                    f"{routing_desc}{i+1}/{len(search)} tested, "
                    f"best={best_time*1e3:.2f}us"
                )

        if best_config is None:
            print(f"  M={M}: all configs failed, skipping")
            continue

        best_configs[M] = sort_config(best_config)
        results[M] = (best_time, best_configs[M])
        routing_desc = (
            f", tokens/expert={tokens_per_expert}, "
            f"active_experts={controlled_active_experts}"
            if tokens_per_expert is not None
            else ""
        )
        print(
            f"  M={M} ({'decode' if is_decode else 'prefill'}): "
            f"routed={M * top_k}{routing_desc}, best={best_time*1e3:.2f}us, "
            f"config={best_configs[M]}"
        )

    return best_configs, results


def print_config_ranges(
    all_results: List[Tuple[int, int, int, Dict[int, Tuple[float, Dict[str, Any]]]]],
    top_k: int,
) -> None:
    """Print adjacent tested M/routed ranges that share an identical config."""
    print(f"\n{'='*80}")
    print("CONFIG RANGES")
    print(f"{'='*80}")
    print(
        f"\n{'E':>5} {'N(rank)':>8} {'K(hidden)':>10} "
        f"{'M range':>14} {'routed range':>18}  config"
    )
    print("-" * 120)

    for E, N, K, results in all_results:
        current_start = None
        current_end = None
        current_sig = None
        current_cfg = None

        def flush_range():
            if current_start is None or current_end is None or current_cfg is None:
                return
            routed_start = current_start * top_k
            routed_end = current_end * top_k
            print(
                f"{E:>5} {N:>8} {K:>10} "
                f"{format_range(current_start, current_end):>14} "
                f"{format_range(routed_start, routed_end):>18}  {current_cfg}"
            )

        for M in sorted(results.keys()):
            _, cfg = results[M]
            sig = config_signature(cfg)
            if current_sig is None:
                current_start = current_end = M
                current_sig = sig
                current_cfg = cfg
            elif sig == current_sig:
                current_end = M
            else:
                flush_range()
                current_start = current_end = M
                current_sig = sig
                current_cfg = cfg
        flush_range()


def main(args: argparse.Namespace):
    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    output_dtype = torch.float32 if args.fp32_atomic_add else dtype
    decode_batch_sizes = (
        routed_counts_to_m_values(
            "--decode-routed-token-counts",
            args.decode_routed_token_counts,
            args.top_k,
        )
        or args.decode_batch_sizes
    )
    prefill_num_tokens = (
        routed_counts_to_m_values(
            "--prefill-routed-token-counts",
            args.prefill_routed_token_counts,
            args.top_k,
        )
        or args.prefill_num_tokens
    )

    print("MoE LoRA shrink tuning")
    print(f"  num_experts(E)={args.num_experts}")
    print(f"  ranks={args.ranks}, hidden_sizes={args.hidden_sizes}")
    if args.decode_tokens_per_expert is None:
        print(f"  decode batch sizes (M==bs)={decode_batch_sizes}")
        print(f"  decode routed rows={[m * args.top_k for m in decode_batch_sizes]}")
    else:
        print(f"  decode tokens/expert={args.decode_tokens_per_expert}")
        print(f"  decode active_experts={args.active_experts or 'all E'}")
    print(f"  prefill num_tokens M={prefill_num_tokens}")
    print(f"  prefill routed rows={[m * args.top_k for m in prefill_num_tokens]}")
    print(
        f"  top_k={args.top_k}, dtype={args.dtype}, "
        f"output_dtype={'float32' if args.fp32_atomic_add else args.dtype}"
    )

    all_results = []  # (E, N, K, results)
    for E in args.num_experts:
        for N in args.ranks:
            for K in args.hidden_sizes:
                best_configs, results = tune_one(
                    E,
                    N,
                    K,
                    decode_batch_sizes,
                    prefill_num_tokens,
                    args.top_k,
                    dtype,
                    output_dtype,
                    device,
                    args.seed,
                    args.decode_tokens_per_expert,
                    args.active_experts,
                )
                if best_configs and not args.no_save:
                    filepath = save_config(best_configs, E, N, K)
                    print(f"  Saved to: {filepath}")
                elif best_configs:
                    print("  Skipped saving configs (--no-save)")
                all_results.append((E, N, K, results))

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(
        f"\n{'E':>5} {'N(rank)':>8} {'K(hidden)':>10} "
        f"{'M':>6} {'routed':>8} {'tuned(us)':>11}  config"
    )
    print("-" * 112)
    for E, N, K, results in all_results:
        for M in sorted(results.keys()):
            best, cfg = results[M]
            print(
                f"{E:>5} {N:>8} {K:>10} {M:>6} {M * args.top_k:>8} "
                f"{best*1e3:>10.2f}  {cfg}"
            )

    print_config_ranges(all_results, args.top_k)

    print(f"\nTuning completed at {datetime.now().ctime()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto-tune the MoE LoRA shrink (LoRA A) split-K kernel"
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=[16, 32, 64],
        help="LoRA ranks to tune (kernel N). Default: 16 32 64",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[512, 768, 2048, 7168],
        help="Hidden sizes to tune (kernel K). Default: 512 768 2048 7168",
    )
    parser.add_argument(
        "--decode-batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256],
        help="Decode batch sizes tuned in the decode regime (bm pinned to 16). "
        "In decode the config key M == batch size (1 token/request).",
    )
    parser.add_argument(
        "--decode-routed-token-counts",
        type=int,
        nargs="+",
        default=None,
        help="Optional decode sweep expressed as routed rows instead of input "
        "batch size. Values must be multiples of top_k and are converted to "
        "runtime-compatible M keys by dividing by top_k.",
    )
    parser.add_argument(
        "--decode-tokens-per-expert",
        type=int,
        nargs="+",
        default=None,
        help="Optional decode sweep with controlled balanced routing. Each value "
        "is the exact number of routed rows assigned to each active expert. "
        "The runtime-compatible M key is active_experts * value / top_k.",
    )
    parser.add_argument(
        "--active-experts",
        type=int,
        default=None,
        help="Number of active experts to use with --decode-tokens-per-expert. "
        "Defaults to all E virtual experts for each tuned shape.",
    )
    parser.add_argument(
        "--prefill-num-tokens",
        type=int,
        nargs="*",
        default=[512, 1024, 1536, 2048, 3072, 4096, 8192],
        help="Token counts M tuned in the prefill regime (bm tuned). "
        "Pass with no values to skip the prefill sweep entirely.",
    )
    parser.add_argument(
        "--prefill-routed-token-counts",
        type=int,
        nargs="+",
        default=None,
        help="Optional prefill sweep expressed as routed rows instead of input "
        "token count. Values must be multiples of top_k and are converted to "
        "runtime-compatible M keys by dividing by top_k.",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        nargs="+",
        default=[64, 96, 128, 192, 256, 384],
        help="Expert counts E to tune (kernel weight dim 0 = num_virtual_experts). "
        "Default: 64 96 128 192 256 384",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Representative MoE top_k used while benchmarking.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16"],
        default="bfloat16",
        help="Activation/weight dtype used while benchmarking.",
    )
    parser.add_argument(
        "--fp32-atomic-add",
        action="store_true",
        help="Benchmark split-K accumulation with a float32 output buffer, so "
        "atomic_add operates on fp32 instead of the activation dtype. This is an "
        "experiment for the shrink stage and should usually be paired with --no-save.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run tuning and print results without writing JSON config files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for the (deterministic, per-shape) benchmark routing.",
    )
    args = parser.parse_args()
    main(args)
