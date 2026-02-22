"""
Lightweight DeepGEMM JIT compilation warmup without loading model weights.

Reads model config.json from HF cache to derive kernel shapes, then compiles
DeepGEMM kernels directly. This avoids the expensive model weight loading step
that the full `sglang.compile_deep_gemm` requires.

Supports DeepSeek V2/V3 family models. Falls back to `sglang.compile_deep_gemm`
for unsupported architectures.

Usage:
    python3 scripts/ci/cuda/warmup_deep_gemm.py \
        deepseek-ai/DeepSeek-V3-0324:8 \
        deepseek-ai/DeepSeek-V3.2-Exp:8
"""

import json
import os
import subprocess
import sys
import time
from math import ceil
from pathlib import Path

# Configure DeepGEMM cache before importing deep_gemm
os.environ["DG_JIT_CACHE_DIR"] = os.getenv(
    "SGLANG_DG_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "deep_gemm"),
)
os.environ["DG_JIT_USE_NVRTC"] = os.getenv("SGL_DG_USE_NVRTC", "0")

BLOCK_SIZE = 128


def get_config_json(model_name):
    """Load config.json for a cached model from HF cache."""
    cache_dir = os.environ.get(
        "HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    )
    hub_dir = os.path.join(cache_dir, "hub")
    safe_name = "models--" + model_name.replace("/", "--")
    snapshots_dir = os.path.join(hub_dir, safe_name, "snapshots")

    if not os.path.isdir(snapshots_dir):
        return None

    snapshots = sorted(
        Path(snapshots_dir).iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
    )
    for snapshot in snapshots:
        config_path = snapshot / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
    return None


def is_deepseek_v2v3(config):
    """Check if a model is from the DeepSeek V2/V3 family."""
    architectures = config.get("architectures", [])
    model_type = config.get("model_type", "")
    return any(
        "DeepseekV2" in a or "DeepseekV3" in a for a in architectures
    ) or model_type in ("deepseek_v2", "deepseek_v3")


def compute_deepseek_v2v3_shapes(config, tp):
    """Compute all DeepGEMM (kernel_type, N, K, num_groups) for DeepSeek V2/V3.

    Shape derivation based on:
    - MoE: python/sglang/srt/layers/moe/fused_moe_triton/layer.py
    - MLA: python/sglang/srt/models/deepseek_v2.py
    - FP8: python/sglang/srt/layers/quantization/fp8_kernel.py
    """
    shapes = []

    hidden_size = config["hidden_size"]
    num_attention_heads = config.get("num_attention_heads", 128)
    kv_lora_rank = config.get("kv_lora_rank", 512)
    qk_nope_head_dim = config.get("qk_nope_head_dim", 128)
    v_head_dim = config.get("v_head_dim", 128)
    n_routed_experts = config.get("n_routed_experts", 0)
    n_shared_experts = config.get("n_shared_experts", 0)
    moe_intermediate_size = config.get("moe_intermediate_size", 0)

    num_local_heads = num_attention_heads // tp
    # Shared expert fusion is enabled by default (disable_shared_experts_fusion=False)
    # so the FusedMoE weight tensor includes shared experts
    num_local_experts = n_routed_experts + n_shared_experts

    # --- MoE expert GEMM shapes ---
    # FusedMoE shards intermediate_size across TP ranks (column parallel for gate/up,
    # row parallel for down). All experts are replicated on each TP rank.
    if n_routed_experts > 0 and moe_intermediate_size > 0:
        moe_inter_per_tp = moe_intermediate_size // tp

        # Gate-Up projection: (tokens, hidden_size) @ (experts, 2*inter_per_tp, hidden_size)^T
        # Both masked and contiguous paths are used at runtime
        shapes.append(("MASKED", moe_inter_per_tp * 2, hidden_size, num_local_experts))
        shapes.append(("CONTIG", moe_inter_per_tp * 2, hidden_size, num_local_experts))

        # Down projection: (tokens, inter_per_tp) @ (experts, hidden_size, inter_per_tp)^T
        shapes.append(("MASKED", hidden_size, moe_inter_per_tp, num_local_experts))
        shapes.append(("CONTIG", hidden_size, moe_inter_per_tp, num_local_experts))

    # --- MLA attention GEMM shapes (masked grouped GEMM) ---
    if kv_lora_rank > 0 and num_local_heads > 0:
        # Q_nope -> compressed K: (heads, m, qk_nope_head_dim) @ (heads, kv_lora_rank, qk_nope_head_dim)^T
        shapes.append(("MASKED", kv_lora_rank, qk_nope_head_dim, num_local_heads))

        # Attention output -> V: (heads, m, kv_lora_rank) @ (heads, v_head_dim, kv_lora_rank)^T
        shapes.append(("MASKED", v_head_dim, kv_lora_rank, num_local_heads))

    # --- kv_b_proj (non-grouped GEMM via FP8 kernel) ---
    # ColumnParallelLinear(kv_lora_rank, num_heads * (qk_nope + v_head_dim))
    # Per TP rank: N = num_local_heads * (qk_nope_head_dim + v_head_dim)
    if kv_lora_rank > 0 and num_local_heads > 0:
        kv_b_proj_n = num_local_heads * (qk_nope_head_dim + v_head_dim)
        shapes.append(("NORMAL", kv_b_proj_n, kv_lora_rank, 1))

    return shapes


def get_architecture_key(config, tp):
    """Key for dedup: models with same key share DeepGEMM kernels."""
    if config is None:
        return None
    fields = [
        config.get("hidden_size", 0),
        config.get("moe_intermediate_size", 0),
        config.get("n_routed_experts", 0),
        config.get("n_shared_experts", 0),
        config.get("num_attention_heads", 0),
        config.get("kv_lora_rank", 0),
        config.get("qk_nope_head_dim", 0),
        config.get("v_head_dim", 0),
        tp,
    ]
    return tuple(fields)


def compute_m_list(fast_warmup=False, chunked_prefill_size=8192):
    """Compute the list of M values to compile (matches compile_utils.py logic)."""
    m_list = []
    if fast_warmup:
        m_list += list(range(1, 1025))
        next_m, sample_step = 1024, 2
        max_prefill_bs = min(chunked_prefill_size, 32 * 1024)
        while next_m < max_prefill_bs:
            m_list += list(range(next_m, 2 * next_m, sample_step))
            next_m *= 2
            sample_step *= 2
        m_list.append(max_prefill_bs)
        m_list = sorted(set(m_list))
    else:
        m_max = 16 * 1024
        if chunked_prefill_size > 8192:
            m_max = chunked_prefill_size * 2
        m_max = min(128 * 1024, m_max)
        m_list = list(range(1, m_max + 1))
    return m_list


def _empty_token_fp8(size):
    """Create FP8 token tensor + per-block scale tensor."""
    import torch

    *dims, k = size
    return (
        torch.empty(size, device="cuda", dtype=torch.float8_e4m3fn),
        torch.empty((*dims, ceil(k / BLOCK_SIZE)), device="cuda", dtype=torch.float32),
    )


def _empty_block_fp8(size):
    """Create FP8 block tensor + per-block scale tensor."""
    import torch

    *dims, n, k = size
    return (
        torch.empty(size, device="cuda", dtype=torch.float8_e4m3fn),
        torch.empty(
            (*dims, ceil(n / BLOCK_SIZE), ceil(k / BLOCK_SIZE)),
            device="cuda",
            dtype=torch.float32,
        ),
    )


def get_memory_requirement(kernel_type, max_m, n, k, num_groups):
    """Estimate GPU memory needed in GB for compilation buffers."""
    _GB = 1 << 30
    if kernel_type == "NORMAL":
        return (max_m * k + n * k + max_m * n * 2) / _GB
    elif kernel_type == "CONTIG":
        return (max_m * k + num_groups * n * k + max_m * 4 + max_m * n * 2) / _GB
    elif kernel_type == "MASKED":
        return (
            num_groups * max_m * k
            + num_groups * n * k
            + num_groups * 4
            + num_groups * max_m * n * 2
        ) / _GB
    return 0


def compile_one_shape(kernel_type, n, k, num_groups, m_list):
    """Compile DeepGEMM kernels for one (kernel_type, N, K, num_groups) shape."""
    import deep_gemm
    import torch
    from tqdm import tqdm

    # Filter M list for contiguous layout alignment
    if kernel_type == "CONTIG":
        m_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
        m_list = sorted(set(m for m in m_list if m % m_alignment == 0))

    if not m_list:
        return

    max_m = max(m_list)

    # Reduce max_m if not enough GPU memory
    mem_free = torch.cuda.mem_get_info()[0] / (1 << 30)
    mem_required = get_memory_requirement(kernel_type, max_m, n, k, num_groups)
    if mem_required > mem_free:
        while (
            get_memory_requirement(kernel_type, max_m, n, k, num_groups) > mem_free
            and max_m > 4096
        ):
            max_m //= 2
        print(
            f"  Memory {mem_free:.1f}GB < required {mem_required:.1f}GB, "
            f"reducing max_m to {max_m}"
        )
        m_list = [m for m in m_list if m <= max_m]

    old_mode = deep_gemm.get_compile_mode()
    deep_gemm.set_compile_mode(1)
    try:
        if kernel_type == "NORMAL":
            lhs_q, lhs_s = _empty_token_fp8((max_m, k))
            rhs_q, rhs_s = _empty_block_fp8((n, k))
            out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)
            for m in tqdm(m_list, desc=f"  NORMAL N={n} K={k}"):
                deep_gemm.fp8_gemm_nt((lhs_q[:m], lhs_s[:m]), (rhs_q, rhs_s), out[:m])

        elif kernel_type == "CONTIG":
            lhs_q, lhs_s = _empty_token_fp8((max_m, k))
            rhs_q, rhs_s = _empty_block_fp8((num_groups, n, k))
            m_indices = torch.zeros((max_m,), device="cuda", dtype=torch.int32)
            out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)
            for m in tqdm(m_list, desc=f"  CONTIG N={n} K={k} G={num_groups}"):
                deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                    (lhs_q[:m], lhs_s[:m]),
                    (rhs_q, rhs_s),
                    out[:m],
                    m_indices=m_indices[:m],
                )

        elif kernel_type == "MASKED":
            lhs_q, lhs_s = _empty_token_fp8((num_groups, max_m, k))
            rhs_q, rhs_s = _empty_block_fp8((num_groups, n, k))
            masked_m = torch.zeros((num_groups,), device="cuda", dtype=torch.int32)
            out = torch.empty(
                (num_groups, max_m, n), device="cuda", dtype=torch.bfloat16
            )
            for m in tqdm(m_list, desc=f"  MASKED N={n} K={k} G={num_groups}"):
                deep_gemm.fp8_m_grouped_gemm_nt_masked(
                    (lhs_q, lhs_s),
                    (rhs_q, rhs_s),
                    out,
                    masked_m=masked_m,
                    expected_m=m,
                )
    finally:
        deep_gemm.set_compile_mode(old_mode)

    torch.cuda.current_stream().synchronize()
    torch.cuda.empty_cache()


def compile_shapes_lightweight(shapes, m_list):
    """Compile all DeepGEMM shapes directly (no model loading)."""
    for i, (kernel_type, n, k, num_groups) in enumerate(shapes, 1):
        print(f"\n[{i}/{len(shapes)}] {kernel_type} N={n} K={k} G={num_groups}")
        t0 = time.time()
        compile_one_shape(kernel_type, n, k, num_groups, m_list)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")


def fallback_compile_deep_gemm(model, tp):
    """Fall back to full sglang.compile_deep_gemm (loads model weights)."""
    print(f"Falling back to full compile_deep_gemm for {model} (tp={tp})...")
    cmd = [
        sys.executable,
        "-m",
        "sglang.compile_deep_gemm",
        "--model",
        model,
        "--tp",
        str(tp),
        "--trust-remote-code",
        "--model-loader-extra-config",
        '{"enable_multithread_load": true, "num_threads": 64}',
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Warning: fallback failed for {model} (exit code {result.returncode})")
    return result.returncode == 0


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: warmup_deep_gemm.py model1:tp1 [model2:tp2 ...]")
        print("\nDerives DeepGEMM kernel shapes from config.json without loading model")
        print(
            "weights. Falls back to full compile_deep_gemm for unknown architectures."
        )
        sys.exit(0)

    # Parse model:tp pairs
    model_tp_pairs = []
    for arg in sys.argv[1:]:
        if ":" not in arg:
            print(f"Error: expected model:tp format, got '{arg}'")
            sys.exit(1)
        model, tp_str = arg.rsplit(":", 1)
        model_tp_pairs.append((model, int(tp_str)))

    fast_warmup = os.environ.get("SGLANG_JIT_DEEPGEMM_FAST_WARMUP", "0").lower() in (
        "1",
        "true",
    )
    print(f"=== DeepGEMM Lightweight Warmup ({len(model_tp_pairs)} model(s)) ===")
    print(f"    Fast warmup: {fast_warmup}")
    print(
        f"    Cache dir: {os.environ.get('DG_JIT_CACHE_DIR', '~/.cache/deep_gemm')}\n"
    )

    # Load configs and deduplicate by architecture
    seen_keys = {}
    to_process = []  # (model, tp, config_or_None, shapes_or_None)

    for model, tp in model_tp_pairs:
        config = get_config_json(model)
        if config is None:
            print(f"  SKIP   {model} (tp={tp}): config.json not in HF cache")
            continue

        key = get_architecture_key(config, tp)
        if key in seen_keys:
            print(f"  DEDUP  {model} (tp={tp}): same shapes as {seen_keys[key]}")
            continue

        if is_deepseek_v2v3(config):
            shapes = compute_deepseek_v2v3_shapes(config, tp)
            seen_keys[key] = model
            to_process.append((model, tp, config, shapes))
            print(f"  FOUND  {model} (tp={tp}): {len(shapes)} DeepGEMM shape(s)")
        else:
            # Unknown architecture: will use fallback
            seen_keys[key] = model
            to_process.append((model, tp, config, None))
            arch = config.get("architectures", ["unknown"])
            print(f"  FOUND  {model} (tp={tp}): unknown arch {arch}, will use fallback")

    if not to_process:
        print("\nNo models to process. Done.")
        return

    m_list = compute_m_list(fast_warmup=fast_warmup)
    print(f"\nM list: {len(m_list)} values (range {min(m_list)}-{max(m_list)})")

    for model, tp, config, shapes in to_process:
        print(f"\n{'=' * 60}")
        print(f"Model: {model} (tp={tp})")
        print(f"{'=' * 60}")

        if shapes is None:
            # Unknown architecture: fall back to full compile_deep_gemm
            fallback_compile_deep_gemm(model, tp)
            continue

        # Print shape summary
        for kernel_type, n, k, num_groups in shapes:
            print(f"  {kernel_type:8s} N={n:<6d} K={k:<6d} G={num_groups}")

        t0 = time.time()
        compile_shapes_lightweight(shapes, m_list)
        elapsed = time.time() - t0
        print(f"\nCompleted {model} in {elapsed:.1f}s")

    print("\nDeepGEMM lightweight warmup complete.")


if __name__ == "__main__":
    main()
