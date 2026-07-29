"""
Auto-tuning script for LoRA CSGMV (Chunked Segmented Matrix-Vector) kernels.

LoRA adds low-rank adapters to linear layers. The two kernels are:
  - Shrink (lora_a): x @ A^T, projecting from input_dim down to rank
  - Expand (lora_b): (x @ A^T) @ B^T, projecting from rank back up to output_dim

Terminology / dimensions:
  K          For shrink: input_dim (the large dimension, e.g. hidden_size).
             For expand: output_dim (e.g. hidden_size or qkv_output_dim).
  R          Max LoRA rank (e.g. 16, 32, 64). The small dimension.
  S          num_slices — how many weight slices a layer fuses together:
               qkv_proj → 3 (q, k, v), gate_up_proj → 2, others → 1.
             Affects the Triton grid (N = S * R for shrink, grid dim for expand).
  chunk_size BLOCK_M — the max segment length in the chunked batch. Sequences
             are split into fixed-size chunks for load-balanced GPU scheduling.
             Typical values: 16, 32, 64, 128.

Tuned parameters (per kernel, K, R, S, chunk_size):
  BLOCK_N    Tile size along the N (output) dimension.
  BLOCK_K    Tile size along the K (reduction) dimension.
  num_warps  Number of warps per Triton program instance.
  num_stages Number of software pipelining stages.
  maxnreg    (expand only) Register cap to improve occupancy.

Config files are saved as JSON keyed by chunk_size, e.g.:
  lora_shrink,K=1024,R=64,S=3,device=NVIDIA_H100.json

The server loads these at startup via lora_tuning_config.py. If no tuned
config exists, hardcoded defaults are used.

Usage:
    # Tune from model name (auto-derives hidden_size, QKV dims)
    python benchmark/kernels/lora_csgmv/tune_lora_csgmv.py \
        --model Qwen/Qwen3-0.6B --rank 64

    # Tune with explicit dimensions
    python benchmark/kernels/lora_csgmv/tune_lora_csgmv.py \
        --hidden-size 1024 --rank 64

    # Tune for specific chunk sizes
    python benchmark/kernels/lora_csgmv/tune_lora_csgmv.py \
        --model Qwen/Qwen3-0.6B --rank 64 --chunk-sizes 32 64 128

    # Another model
    python benchmark/kernels/lora_csgmv/tune_lora_csgmv.py \
        --model meta-llama/Llama-2-7b-hf --rank 32
"""

import argparse
import json
import math
import os
import statistics
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
import triton

from sglang.srt.lora.triton_ops.chunked_sgmv_expand import _chunked_lora_expand_kernel
from sglang.srt.lora.triton_ops.chunked_sgmv_shrink import _chunked_lora_shrink_kernel
from sglang.srt.lora.triton_ops.lora_tuning_config import (
    DEFAULT_EXPAND_CONFIG,
    DEFAULT_SHRINK_CONFIG,
    get_lora_config_file_name,
)
from sglang.srt.lora.utils import LoRABatchInfo


def _get_raw_kernel(cached_kernel):
    """Get the underlying triton.jit function, bypassing cached_triton_kernel."""
    return getattr(cached_kernel, "fn", cached_kernel)


def build_batch_info(
    total_tokens: int,
    chunk_size: int,
    rank: int,
    device: torch.device,
) -> LoRABatchInfo:
    """Build a LoRABatchInfo for benchmarking with a single LoRA adapter."""
    num_segments = math.ceil(total_tokens / chunk_size)

    seg_indptr = []
    for i in range(num_segments):
        seg_indptr.append(i * chunk_size)
    seg_indptr.append(total_tokens)
    seg_indptr = torch.tensor(seg_indptr, dtype=torch.int32, device=device)

    weight_indices = torch.ones(num_segments, dtype=torch.int32, device=device)
    lora_ranks = torch.tensor([0, rank], dtype=torch.int32, device=device)
    scalings = torch.ones(2, dtype=torch.float32, device=device)
    permutation = torch.arange(total_tokens, dtype=torch.int32, device=device)

    return LoRABatchInfo(
        use_cuda_graph=False,
        bs=1,
        num_segments=num_segments,
        max_len=chunk_size,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        lora_ranks=lora_ranks,
        scalings=scalings,
        seg_lens=None,
        permutation=permutation,
    )


def timed_cuda_ms(fn, warmup: int = 10, trials: int = 50) -> float:
    """Time a GPU function using CUDA events. Returns median time in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------


def get_shrink_search_space() -> List[Dict[str, Any]]:
    """Generate candidate configs for the shrink kernel."""
    configs = []
    for block_n in [16, 32, 64]:
        for block_k in [64, 128, 256]:
            for num_warps in [4, 8]:
                for num_stages in [2, 3, 4]:
                    configs.append(
                        {
                            "BLOCK_N": block_n,
                            "BLOCK_K": block_k,
                            "num_warps": num_warps,
                            "num_stages": num_stages,
                        }
                    )
    return configs


def get_expand_search_space() -> List[Dict[str, Any]]:
    """Generate candidate configs for the expand kernel."""
    configs = []
    for block_n in [32, 64]:
        for block_k in [16, 32]:
            for num_warps in [4, 8]:
                for num_stages in [1, 2, 3]:
                    # Without maxnreg
                    configs.append(
                        {
                            "BLOCK_N": block_n,
                            "BLOCK_K": block_k,
                            "num_warps": num_warps,
                            "num_stages": num_stages,
                        }
                    )
                    # With maxnreg (register capping for occupancy)
                    for maxnreg in [96, 112, 128, 160]:
                        configs.append(
                            {
                                "BLOCK_N": block_n,
                                "BLOCK_K": block_k,
                                "num_warps": num_warps,
                                "num_stages": num_stages,
                                "maxnreg": maxnreg,
                            }
                        )
    return configs


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------


def benchmark_shrink_config(
    config: Dict[str, Any],
    x: torch.Tensor,
    weights: torch.Tensor,
    batch_info: LoRABatchInfo,
    num_slices: int,
    N: int,
    K: int,
) -> Optional[float]:
    """Benchmark a single shrink config. Returns median ms or None on failure."""
    kernel = _get_raw_kernel(_chunked_lora_shrink_kernel)
    S = x.shape[0]
    num_segments = batch_info.num_segments

    grid = (triton.cdiv(N, config["BLOCK_N"]), num_segments)
    output = torch.empty((S, N), device=x.device, dtype=x.dtype)

    extra_kwargs = {}
    if "num_warps" in config:
        extra_kwargs["num_warps"] = config["num_warps"]
    if "num_stages" in config:
        extra_kwargs["num_stages"] = config["num_stages"]

    try:
        kernel[grid](
            x=x,
            weights=weights,
            output=output,
            seg_indptr=batch_info.seg_indptr,
            weight_indices=batch_info.weight_indices,
            lora_ranks=batch_info.lora_ranks,
            permutation=batch_info.permutation,
            num_segs=num_segments,
            N=N,
            K=K,
            NUM_SLICES=num_slices,
            BLOCK_M=batch_info.max_len,
            BLOCK_N=config["BLOCK_N"],
            BLOCK_K=config["BLOCK_K"],
            **extra_kwargs,
        )
        torch.cuda.synchronize()
    except Exception:
        return None

    def run():
        kernel[grid](
            x=x,
            weights=weights,
            output=output,
            seg_indptr=batch_info.seg_indptr,
            weight_indices=batch_info.weight_indices,
            lora_ranks=batch_info.lora_ranks,
            permutation=batch_info.permutation,
            num_segs=num_segments,
            N=N,
            K=K,
            NUM_SLICES=num_slices,
            BLOCK_M=batch_info.max_len,
            BLOCK_N=config["BLOCK_N"],
            BLOCK_K=config["BLOCK_K"],
            **extra_kwargs,
        )

    return timed_cuda_ms(run, warmup=10, trials=50)


def benchmark_expand_config(
    config: Dict[str, Any],
    x: torch.Tensor,
    weights: torch.Tensor,
    batch_info: LoRABatchInfo,
    slice_offsets: torch.Tensor,
    max_slice_size: int,
    output_dim: int,
    num_slices: int,
    max_rank: int,
) -> Optional[float]:
    """Benchmark a single expand config. Returns median ms or None on failure."""
    kernel = _get_raw_kernel(_chunked_lora_expand_kernel)
    M = x.shape[0]
    num_segments = batch_info.num_segments

    grid = (
        triton.cdiv(max_slice_size, config["BLOCK_N"]),
        num_slices,
        num_segments,
    )
    output = torch.zeros((M, output_dim), device=x.device, dtype=x.dtype)

    extra_kwargs = {}
    if "num_warps" in config:
        extra_kwargs["num_warps"] = config["num_warps"]
    if "num_stages" in config:
        extra_kwargs["num_stages"] = config["num_stages"]
    if "maxnreg" in config:
        extra_kwargs["maxnreg"] = config["maxnreg"]

    try:
        kernel[grid](
            x=x,
            weights=weights,
            output=output,
            seg_indptr=batch_info.seg_indptr,
            weight_indices=batch_info.weight_indices,
            lora_ranks=batch_info.lora_ranks,
            permutation=batch_info.permutation,
            num_segs=num_segments,
            scalings=batch_info.scalings,
            slice_offsets=slice_offsets,
            NUM_SLICES=num_slices,
            OUTPUT_DIM=output_dim,
            MAX_RANK=max_rank,
            BLOCK_M=batch_info.max_len,
            BLOCK_N=config["BLOCK_N"],
            BLOCK_K=config["BLOCK_K"],
            **extra_kwargs,
        )
        torch.cuda.synchronize()
    except Exception:
        return None

    def run():
        output.zero_()
        kernel[grid](
            x=x,
            weights=weights,
            output=output,
            seg_indptr=batch_info.seg_indptr,
            weight_indices=batch_info.weight_indices,
            lora_ranks=batch_info.lora_ranks,
            permutation=batch_info.permutation,
            num_segs=num_segments,
            scalings=batch_info.scalings,
            slice_offsets=slice_offsets,
            NUM_SLICES=num_slices,
            OUTPUT_DIM=output_dim,
            MAX_RANK=max_rank,
            BLOCK_M=batch_info.max_len,
            BLOCK_N=config["BLOCK_N"],
            BLOCK_K=config["BLOCK_K"],
            **extra_kwargs,
        )

    return timed_cuda_ms(run, warmup=10, trials=50)


# ---------------------------------------------------------------------------
# Config saving
# ---------------------------------------------------------------------------


def save_config(
    configs: Dict[int, Dict[str, Any]],
    kernel: str,
    major_dim: int,
    max_rank: int,
    num_slices: int,
) -> str:
    """Save tuned configs to the standard config directory. Returns filepath.

    Args:
        configs: Dict mapping chunk_size -> best block config.
        kernel: "shrink" or "expand".
        major_dim: The large dimension (input_dim for shrink, output_dim for expand).
        max_rank: The max LoRA rank.
        num_slices: Number of fused weight slices (qkv=3, gate_up=2, others=1).
    """
    filename = get_lora_config_file_name(kernel, major_dim, max_rank, num_slices)

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
        "csgmv_configs",
        version_dir,
    )
    config_dir = os.path.normpath(config_dir)
    os.makedirs(config_dir, exist_ok=True)

    filepath = os.path.join(config_dir, filename)
    with open(filepath, "w") as f:
        json.dump(configs, f, indent=4)
        f.write("\n")
    return filepath


def sort_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Sort config keys for consistent JSON output."""
    ordered = {}
    for key in ["BLOCK_N", "BLOCK_K", "num_warps", "num_stages", "maxnreg"]:
        if key in config:
            ordered[key] = config[key]
    return ordered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def get_model_dims(args: argparse.Namespace):
    """Extract all LoRA layer dimensions from model config or CLI args.

    Returns a list of (label, shrink_K, expand_output_dim, num_slices,
    slice_offsets_list) tuples for each LoRA layer type.
    """
    if args.model:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        hidden_size = config.hidden_size

        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        head_dim = getattr(config, "head_dim", hidden_size // num_heads)
        intermediate_size = config.intermediate_size

        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim
        qkv_output_dim = q_dim + 2 * kv_dim

        print(f"Model: {args.model}")
        print(
            f"  hidden_size={hidden_size}, num_heads={num_heads}, "
            f"num_kv_heads={num_kv_heads}, head_dim={head_dim}"
        )
        print(f"  intermediate_size={intermediate_size}")
    else:
        hidden_size = args.hidden_size
        intermediate_size = getattr(args, "intermediate_size", None) or hidden_size * 3
        if args.qkv_output_dim:
            qkv_output_dim = args.qkv_output_dim
            q_dim = qkv_output_dim // 2
            kv_dim = (qkv_output_dim - q_dim) // 2
        else:
            q_dim = hidden_size * 2
            kv_dim = hidden_size
            qkv_output_dim = q_dim + 2 * kv_dim

    # All LoRA layer types with their dimensions:
    #   (label, shrink_K, expand_output_dim, num_slices, slice_offsets)
    layers = [
        (
            "qkv",
            hidden_size,
            qkv_output_dim,
            3,
            [0, q_dim, q_dim + kv_dim, qkv_output_dim],
        ),
        ("o_proj", q_dim, hidden_size, 1, [0, hidden_size]),
        (
            "gate_up",
            hidden_size,
            2 * intermediate_size,
            2,
            [0, intermediate_size, 2 * intermediate_size],
        ),
        ("down_proj", intermediate_size, hidden_size, 1, [0, hidden_size]),
    ]

    print(f"\nLoRA layer dimensions:")
    for label, sk, eo, ns, so in layers:
        print(f"  {label:>10}: shrink K={sk}, expand output_dim={eo}, num_slices={ns}")

    return layers


def _tune_shrink(
    label: str,
    K: int,
    N: int,
    num_slices: int,
    rank: int,
    chunk_sizes: List[int],
    total_tokens: int,
    device: torch.device,
) -> tuple:
    """Tune shrink kernel for one layer type. Returns (best_configs, results)."""
    print(f"\n{'='*80}")
    print(f"Tuning SHRINK — {label} (K={K}, N={N}, slices={num_slices})")
    print(f"{'='*80}")

    search = get_shrink_search_space()
    print(f"Search space: {len(search)} configs")

    best_configs = {}
    results = {}

    for chunk_size in chunk_sizes:
        batch_info = build_batch_info(total_tokens, chunk_size, rank, device)
        x = torch.randn(total_tokens, K, device=device, dtype=torch.float16)
        weights = torch.randn(2, N, K, device=device, dtype=torch.float16)

        baseline_time = benchmark_shrink_config(
            DEFAULT_SHRINK_CONFIG,
            x,
            weights,
            batch_info,
            num_slices,
            N,
            K,
        )
        print(f"  chunk={chunk_size}: baseline={baseline_time:.3f}ms")

        best_config = None
        best_time = float("inf")

        for i, config in enumerate(search):
            t = benchmark_shrink_config(
                config, x, weights, batch_info, num_slices, N, K
            )
            if t is not None and t < best_time:
                best_time = t
                best_config = config
            if (i + 1) % 20 == 0:
                print(
                    f"  chunk={chunk_size}: {i+1}/{len(search)} tested, best={best_time:.3f}ms"
                )

        best_configs[chunk_size] = sort_config(best_config)
        results[chunk_size] = (baseline_time, best_time, best_configs[chunk_size])
        speedup = baseline_time / best_time if best_time > 0 else 0
        print(
            f"  chunk={chunk_size}: best={best_time:.3f}ms ({speedup:.2f}x), config={best_configs[chunk_size]}"
        )

    return best_configs, results


def _tune_expand(
    label: str,
    output_dim: int,
    num_slices: int,
    slice_offsets_list: List[int],
    max_slice_size: int,
    rank: int,
    chunk_sizes: List[int],
    total_tokens: int,
    device: torch.device,
) -> tuple:
    """Tune expand kernel for one layer type. Returns (best_configs, results)."""
    print(f"\n{'='*80}")
    print(f"Tuning EXPAND — {label} (output_dim={output_dim}, slices={num_slices})")
    print(f"{'='*80}")

    search = get_expand_search_space()
    print(f"Search space: {len(search)} configs")

    slice_offsets = torch.tensor(slice_offsets_list, dtype=torch.int64, device=device)
    best_configs = {}
    results = {}

    for chunk_size in chunk_sizes:
        batch_info = build_batch_info(total_tokens, chunk_size, rank, device)
        x = torch.randn(
            total_tokens, num_slices * rank, device=device, dtype=torch.float16
        )
        weights = torch.randn(2, output_dim, rank, device=device, dtype=torch.float16)

        baseline_time = benchmark_expand_config(
            DEFAULT_EXPAND_CONFIG,
            x,
            weights,
            batch_info,
            slice_offsets,
            max_slice_size,
            output_dim,
            num_slices,
            rank,
        )
        print(f"  chunk={chunk_size}: baseline={baseline_time:.3f}ms")

        best_config = None
        best_time = float("inf")

        for i, config in enumerate(search):
            t = benchmark_expand_config(
                config,
                x,
                weights,
                batch_info,
                slice_offsets,
                max_slice_size,
                output_dim,
                num_slices,
                rank,
            )
            if t is not None and t < best_time:
                best_time = t
                best_config = config
            if (i + 1) % 50 == 0:
                print(
                    f"  chunk={chunk_size}: {i+1}/{len(search)} tested, best={best_time:.3f}ms"
                )

        best_configs[chunk_size] = sort_config(best_config)
        results[chunk_size] = (baseline_time, best_time, best_configs[chunk_size])
        speedup = baseline_time / best_time if best_time > 0 else 0
        print(
            f"  chunk={chunk_size}: best={best_time:.3f}ms ({speedup:.2f}x), config={best_configs[chunk_size]}"
        )

    return best_configs, results


def main(args: argparse.Namespace):
    device = torch.device("cuda:0")
    rank = args.rank
    chunk_sizes = args.chunk_sizes
    total_tokens = args.total_tokens

    layers = get_model_dims(args)

    print(f"\nLoRA CSGMV Tuning")
    print(f"  rank={rank}, total_tokens={total_tokens}, chunk_sizes={chunk_sizes}")

    # Collect all results for summary
    all_results = []  # (label, kernel, K_or_outdim, results_dict)

    # Deduplicate: multiple layers can share the same (shrink_K, num_slices) or
    # (expand_output_dim, num_slices). No need to tune the same config twice.
    tuned_shrink = {}  # (shrink_K, num_slices) -> best_configs
    tuned_expand = {}  # (expand_output_dim, num_slices) -> best_configs

    for label, shrink_K, expand_output_dim, num_slices, slice_offsets_list in layers:
        # --- Shrink ---
        shrink_key = (shrink_K, num_slices)
        if shrink_key not in tuned_shrink:
            N_shrink = num_slices * rank
            best_configs, results = _tune_shrink(
                label,
                shrink_K,
                N_shrink,
                num_slices,
                rank,
                chunk_sizes,
                total_tokens,
                device,
            )
            filepath = save_config(best_configs, "shrink", shrink_K, rank, num_slices)
            print(f"  Saved to: {filepath}")
            tuned_shrink[shrink_key] = best_configs
            all_results.append((label, "shrink", shrink_K, results))
        else:
            print(
                f"\n  Skipping shrink {label} (K={shrink_K}, S={num_slices}) — already tuned"
            )

        # --- Expand ---
        expand_key = (expand_output_dim, num_slices)
        if expand_key not in tuned_expand:
            # max_slice_size = largest slice width
            slice_widths = [
                slice_offsets_list[i + 1] - slice_offsets_list[i]
                for i in range(num_slices)
            ]
            max_slice_size = max(slice_widths)

            best_configs, results = _tune_expand(
                label,
                expand_output_dim,
                num_slices,
                slice_offsets_list,
                max_slice_size,
                rank,
                chunk_sizes,
                total_tokens,
                device,
            )
            filepath = save_config(
                best_configs, "expand", expand_output_dim, rank, num_slices
            )
            print(f"  Saved to: {filepath}")
            tuned_expand[expand_key] = best_configs
            all_results.append((label, "expand", expand_output_dim, results))
        else:
            print(
                f"\n  Skipping expand {label} (output_dim={expand_output_dim}, S={num_slices}) — already tuned"
            )

    # --- Summary ---
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(
        f"\n{'layer':<10} {'kernel':<8} {'K/dim':>6} {'chunk':>6}"
        f" {'baseline':>10} {'tuned':>10} {'speedup':>8}  config"
    )
    print("-" * 100)
    for label, kernel, dim, results in all_results:
        for chunk_size in chunk_sizes:
            if chunk_size in results:
                base, best, cfg = results[chunk_size]
                spd = base / best if best > 0 else 0
                print(
                    f"{label:<10} {kernel:<8} {dim:>6} {chunk_size:>6}"
                    f" {base:>9.3f}ms {best:>9.3f}ms {spd:>7.2f}x  {cfg}"
                )

    now = datetime.now()
    print(f"\nTuning completed at {now.ctime()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto-tune LoRA CSGMV kernel block dimensions"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model name to auto-derive dimensions "
        "(e.g., Qwen/Qwen3-0.6B, meta-llama/Llama-2-7b-hf)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="Model hidden size (e.g., 1024 for Qwen3-0.6B). "
        "Required if --model is not specified.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        required=True,
        help="LoRA rank (e.g., 16, 32, 64)",
    )
    parser.add_argument(
        "--qkv-output-dim",
        type=int,
        default=None,
        help="QKV output dimension. Only used with --hidden-size. "
        "Default: 4 * hidden_size",
    )
    parser.add_argument(
        "--chunk-sizes",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128],
        help="Chunk sizes to tune (default: 16 32 64 128)",
    )
    parser.add_argument(
        "--total-tokens",
        type=int,
        default=30720,
        help="Total tokens for benchmarking (default: 30720 = 2 reqs x 15360)",
    )
    args = parser.parse_args()

    if not args.model and not args.hidden_size:
        parser.error("Either --model or --hidden-size is required")

    main(args)
