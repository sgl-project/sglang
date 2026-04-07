"""
Micro-benchmark for VLM vision attention backends.

Simulates the exact attention call parameters that different VLM models
(Qwen2.5-VL, Qwen3-VL, Kimi-K2.5, GLM-4V) would produce for given image
dimensions, then directly benchmarks the attention kernel.

Usage:
    python bench_vlm_attention.py --study all --backends triton_attn,fa3,flashinfer_cudnn
    python bench_vlm_attention.py --study a --models qwen2.5-vl-7b --dry-run
    python bench_vlm_attention.py --study b --num-iters 50 --output results.csv
"""

import argparse
import csv
import itertools
import math
import statistics
import sys
from dataclasses import dataclass

import numpy as np
import torch

# ============================================================
# Model Profiles
# ============================================================


@dataclass
class VisionModelProfile:
    name: str
    num_heads: int
    head_dim: int
    patch_size: int
    image_factor: int = 28
    depth: int = 0  # for reference only


MODEL_PROFILES = {
    "qwen2.5-vl-7b": VisionModelProfile("qwen2.5-vl-7b", 16, 80, 14, 28, 32),
    "qwen3-vl": VisionModelProfile("qwen3-vl", 16, 72, 16, 28, 27),
    "kimi-k2.5": VisionModelProfile("kimi-k2.5", 16, 72, 14, 28, 27),
    "glm-4v": VisionModelProfile("glm-4v", 12, 128, 14, 28, 24),
}

RESOLUTIONS = {
    "360p": (360, 640),
    "720p": (720, 1280),
    "1080p": (1080, 1920),
    "4k": (2160, 3840),
}


# ============================================================
# smart_resize (copied from sglang/srt/multimodal/processors/qwen_vl.py)
# ============================================================

MIN_PIXELS = 4 * 28 * 28  # 3136
MAX_PIXELS = 16384 * 28 * 28  # 12,845,056
MAX_RATIO = 200


def round_by_factor(number, factor):
    return round(number / factor) * factor


def ceil_by_factor(number, factor):
    return math.ceil(number / factor) * factor


def floor_by_factor(number, factor):
    return math.floor(number / factor) * factor


def smart_resize(
    height, width, factor=28, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
):
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, "
            f"got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def compute_patches(height, width, profile):
    h_bar, w_bar = smart_resize(height, width, factor=profile.image_factor)
    grid_h = h_bar // profile.patch_size
    grid_w = w_bar // profile.patch_size
    return grid_h * grid_w, grid_h, grid_w


def compute_cu_seqlens(image_dims, profile):
    patches_list = []
    for h, w in image_dims:
        n_patches, _, _ = compute_patches(h, w, profile)
        patches_list.append(n_patches)
    cu = [0] + list(itertools.accumulate(patches_list))
    return torch.tensor(cu, dtype=torch.int32), patches_list


# ============================================================
# Backend creation
# ============================================================


def detect_available_backends():
    if not torch.cuda.is_available():
        return ["triton_attn"]
    major, minor = torch.cuda.get_device_capability()
    backends = ["triton_attn"]
    if major >= 9:
        backends.append("fa3")
    # fa4 (flash_attn.cute) works on both Hopper and Blackwell
    try:
        from flash_attn.cute import flash_attn_varlen_func  # noqa: F401

        backends.append("fa4")
    except ImportError:
        pass
    # flashinfer_cudnn is available on any CUDA with flashinfer installed
    try:
        from flashinfer.prefill import cudnn_batch_prefill_with_kv_cache  # noqa: F401

        backends.append("flashinfer_cudnn")
    except ImportError:
        pass
    return backends


def create_backend(name):
    # use_data_parallel=True avoids requiring distributed TP group initialization
    if name == "triton_attn":
        from sglang.srt.layers.attention.vision import VisionTritonAttention

        return VisionTritonAttention(use_data_parallel=True)
    elif name == "fa3":
        from sglang.srt.layers.attention.vision import VisionFlash3Attention

        return VisionFlash3Attention(use_data_parallel=True)
    elif name == "fa4":
        from sglang.srt.layers.attention.vision import VisionFlash4Attention

        return VisionFlash4Attention(use_data_parallel=True)
    elif name == "flashinfer_cudnn":
        return None  # uses direct API call
    else:
        raise ValueError(f"Unknown backend: {name}")


# ============================================================
# flashinfer_cudnn specific logic
# ============================================================

FLASHINFER_WORKSPACE_SIZE_BYTES = 128 * 1024 * 1024
BATCH_BUCKETS = [8, 16, 32, 64]
FLASHINFER_MAX_SEQLEN_BUCKETS = [4096, 8192, 16384, 32768, 65536, 131072]


def _bucket_batch_size(batch_size):
    for b in BATCH_BUCKETS:
        if b >= batch_size:
            return b
    return math.ceil(batch_size / BATCH_BUCKETS[0]) * BATCH_BUCKETS[0]


def _bucket_max_seqlen(real_max):
    if real_max <= 0:
        return FLASHINFER_MAX_SEQLEN_BUCKETS[0]
    for s in FLASHINFER_MAX_SEQLEN_BUCKETS:
        if s >= real_max:
            return s
    return (
        math.ceil(real_max / FLASHINFER_MAX_SEQLEN_BUCKETS[-1])
        * FLASHINFER_MAX_SEQLEN_BUCKETS[-1]
    )


def prepare_flashinfer_cudnn_inputs(cu_seqlens_np, patches_list, num_heads, head_dim):
    B = len(patches_list)
    B_padded = _bucket_batch_size(B)

    seq_lens = np.array(patches_list, dtype=np.int32)
    if B_padded > B:
        seq_lens = np.concatenate([seq_lens, np.zeros(B_padded - B, dtype=np.int32)])

    real_max = int(max(patches_list)) if patches_list else 0
    max_seqlen = _bucket_max_seqlen(real_max)

    token_indptr = cu_seqlens_np.astype(np.int64)
    if B_padded > B:
        token_indptr = np.concatenate(
            [token_indptr, np.full(B_padded - B, token_indptr[-1], dtype=np.int64)]
        )

    elem_per_token = num_heads * head_dim
    elem_indptr = (token_indptr * elem_per_token).astype(np.int32)
    packed_indptrs = np.concatenate([elem_indptr, elem_indptr, elem_indptr])

    return {
        "packed_indptrs": torch.from_numpy(packed_indptrs).cuda(),
        "sequence_lengths": (
            torch.from_numpy(seq_lens).to(torch.int32).cuda().view(-1, 1, 1, 1)
        ),
        "max_seqlen": max_seqlen,
        "workspace_buffer": torch.empty(
            FLASHINFER_WORKSPACE_SIZE_BYTES, dtype=torch.uint8, device="cuda"
        ),
    }


def run_flashinfer_cudnn(q, k, v, fi_inputs):
    from flashinfer.prefill import cudnn_batch_prefill_with_kv_cache

    B = fi_inputs["sequence_lengths"].shape[0]
    split = B + 1
    cu = fi_inputs["packed_indptrs"]

    indptr_qk = cu[:split].view(split, 1, 1, 1)
    indptr_v = cu[split : 2 * split].view(split, 1, 1, 1)
    indptr_o = cu[2 * split :].view(split, 1, 1, 1)

    scale = q.shape[-1] ** -0.5

    output, _ = cudnn_batch_prefill_with_kv_cache(
        q,
        k,
        v,
        scale,
        fi_inputs["workspace_buffer"],
        max_token_per_sequence=fi_inputs["max_seqlen"],
        max_sequence_kv=fi_inputs["max_seqlen"],
        actual_seq_lens_q=fi_inputs["sequence_lengths"],
        actual_seq_lens_kv=fi_inputs["sequence_lengths"],
        causal=False,
        return_lse=True,
        batch_offsets_q=indptr_qk,
        batch_offsets_k=indptr_qk,
        batch_offsets_v=indptr_v,
        batch_offsets_o=indptr_o,
        is_cuda_graph_compatible=True,
    )
    return output


# ============================================================
# Benchmark core
# ============================================================


def benchmark_attention(
    backend_name,
    backend,
    num_heads,
    head_dim,
    cu_seqlens,
    patches_list,
    num_warmup=10,
    num_iters=100,
    dtype=torch.float16,
):
    total_tokens = cu_seqlens[-1].item()
    bsz = len(cu_seqlens) - 1

    q = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device="cuda")
    v = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device="cuda")

    if backend_name == "flashinfer_cudnn":
        cu_np = cu_seqlens.cpu().numpy()
        fi_inputs = prepare_flashinfer_cudnn_inputs(
            cu_np, patches_list, num_heads, head_dim
        )
        run_fn = lambda: run_flashinfer_cudnn(q, k, v, fi_inputs)
    else:
        cu_gpu = cu_seqlens.to(device="cuda", dtype=torch.int32)
        run_fn = lambda: backend.forward(q, k, v, cu_seqlens=cu_gpu, bsz=bsz, seq_len=0)

    # Warmup
    for _ in range(num_warmup):
        run_fn()

    # Timed runs
    torch.cuda.synchronize()
    times = []
    for _ in range(num_iters):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        run_fn()
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_ms": min(times),
        "max_ms": max(times),
        "total_tokens": total_tokens,
    }


# ============================================================
# Experiment config and generation
# ============================================================


@dataclass
class ExperimentConfig:
    study: str
    model_name: str
    backend: str
    image_dims: list  # list of (H, W)
    resolution_label: str = ""
    mix_label: str = ""

    @property
    def image_count(self):
        return len(self.image_dims)


def generate_study_a(models, backends):
    """Single image, varying resolution."""
    configs = []
    for model_name in models:
        for res_name, (h, w) in RESOLUTIONS.items():
            for backend in backends:
                configs.append(
                    ExperimentConfig(
                        study="A",
                        model_name=model_name,
                        backend=backend,
                        image_dims=[(h, w)],
                        resolution_label=res_name,
                    )
                )
    return configs


def generate_study_b(models, backends):
    """Multiple images at each resolution."""
    configs = []
    for model_name in models:
        for res_name, (h, w) in RESOLUTIONS.items():
            for count in [1, 2, 4, 8]:
                for backend in backends:
                    configs.append(
                        ExperimentConfig(
                            study="B",
                            model_name=model_name,
                            backend=backend,
                            image_dims=[(h, w)] * count,
                            resolution_label=res_name,
                        )
                    )
    return configs


def generate_study_c(models, backends):
    """Mixed resolutions in one request."""
    mixes = {
        "2img_360p+1080p": [(360, 640), (1080, 1920)],
        "2img_720p+4k": [(720, 1280), (2160, 3840)],
        "4img_uniform_1080p": [(1080, 1920)] * 4,
        "4img_mixed": [
            (360, 640),
            (720, 1280),
            (1080, 1920),
            (2160, 3840),
        ],
        "8img_mixed": [
            (360, 640),
            (360, 640),
            (720, 1280),
            (720, 1280),
            (1080, 1920),
            (1080, 1920),
            (2160, 3840),
            (2160, 3840),
        ],
    }
    configs = []
    for model_name in models:
        for mix_name, image_dims in mixes.items():
            for backend in backends:
                configs.append(
                    ExperimentConfig(
                        study="C",
                        model_name=model_name,
                        backend=backend,
                        image_dims=image_dims,
                        mix_label=mix_name,
                    )
                )
    return configs


# ============================================================
# Result formatting
# ============================================================

CSV_COLUMNS = [
    "study",
    "model",
    "backend",
    "resolution",
    "mix",
    "image_count",
    "total_patches",
    "num_heads",
    "head_dim",
    "mean_ms",
    "median_ms",
    "std_ms",
    "min_ms",
    "max_ms",
    "tokens_per_sec",
]


def format_result_row(config, profile, patches_list, timing):
    total_patches = sum(patches_list)
    tokens_per_sec = (
        total_patches / (timing["mean_ms"] / 1000.0) if timing["mean_ms"] > 0 else 0
    )
    return {
        "study": config.study,
        "model": config.model_name,
        "backend": config.backend,
        "resolution": config.resolution_label,
        "mix": config.mix_label,
        "image_count": config.image_count,
        "total_patches": total_patches,
        "num_heads": profile.num_heads,
        "head_dim": profile.head_dim,
        "mean_ms": f"{timing['mean_ms']:.3f}",
        "median_ms": f"{timing['median_ms']:.3f}",
        "std_ms": f"{timing['std_ms']:.3f}",
        "min_ms": f"{timing['min_ms']:.3f}",
        "max_ms": f"{timing['max_ms']:.3f}",
        "tokens_per_sec": f"{tokens_per_sec:.0f}",
    }


def _make_group_key(r, study_name):
    """Build a unique group key for a result row (everything except backend)."""
    if study_name == "A":
        return r["resolution"]
    elif study_name == "B":
        return f"{r['resolution']}x{r['image_count']}"
    else:
        return r["mix"]


def print_study_summary(results, study_name):
    """Print a formatted summary table for a study."""
    study_results = [r for r in results if r["study"] == study_name]
    if not study_results:
        return

    print(f"\n{'='*100}")
    print(f"Study {study_name} Results")
    print(f"{'='*100}")

    # Group by model
    models_seen = []
    for r in study_results:
        if r["model"] not in models_seen:
            models_seen.append(r["model"])

    for model in models_seen:
        model_results = [r for r in study_results if r["model"] == model]
        profile = MODEL_PROFILES[model]
        print(
            f"\nModel: {model} ({profile.num_heads} heads, head_dim={profile.head_dim})"
        )
        print("-" * 100)

        # Determine label
        if study_name == "A":
            group_label = "Resolution"
        elif study_name == "B":
            group_label = "Res x Imgs"
        else:
            group_label = "Mix"

        # Collect backends
        backends_seen = []
        for r in model_results:
            if r["backend"] not in backends_seen:
                backends_seen.append(r["backend"])

        # Header — speedup is relative to the first backend
        header = f"{group_label:>24} {'Patches':>10}"
        for b in backends_seen:
            header += f" {b + '(ms)':>16}"
        if len(backends_seen) > 1:
            base = backends_seen[0]
            for b in backends_seen[1:]:
                header += f" {'vs_' + base:>16}"
        print(header)

        # Collect unique groups in order
        groups = []
        for r in model_results:
            g = _make_group_key(r, study_name)
            if g not in groups:
                groups.append(g)

        for group in groups:
            group_rows = [
                r for r in model_results if _make_group_key(r, study_name) == group
            ]
            patches = group_rows[0]["total_patches"]
            line = f"{group:>24} {patches:>10}"

            backend_times = {}
            for r in group_rows:
                ms = float(r["median_ms"])
                backend_times[r["backend"]] = ms

            # Use first backend as baseline for speedup
            baseline_backend = backends_seen[0]
            baseline_ms = backend_times.get(baseline_backend)

            for b in backends_seen:
                if b in backend_times:
                    line += f" {backend_times[b]:>15.3f}"
                else:
                    line += f" {'N/A':>16}"

            if baseline_ms and baseline_ms > 0 and len(backends_seen) > 1:
                for b in backends_seen[1:]:
                    if b in backend_times and backend_times[b] > 0:
                        speedup = baseline_ms / backend_times[b]
                        line += f" {speedup:>15.2f}x"
                    else:
                        line += f" {'N/A':>16}"

            print(line)


# ============================================================
# Main
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Micro-benchmark for VLM vision attention backends"
    )
    parser.add_argument(
        "--study",
        type=str,
        default="all",
        choices=["a", "b", "c", "all"],
        help="Which study to run: a (resolution), b (image count), c (mixed), all",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(MODEL_PROFILES.keys()),
        help="Comma-separated model profiles to test",
    )
    parser.add_argument(
        "--backends",
        type=str,
        default=None,
        help="Comma-separated backends to test (default: auto-detect)",
    )
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16"],
    )
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print configs without running"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse models
    models = [m.strip() for m in args.models.split(",")]
    for m in models:
        if m not in MODEL_PROFILES:
            print(
                f"Error: Unknown model '{m}'. Available: {list(MODEL_PROFILES.keys())}"
            )
            sys.exit(1)

    # Detect/parse backends
    if args.backends:
        backends = [b.strip() for b in args.backends.split(",")]
    else:
        backends = detect_available_backends()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    print(f"Models: {models}")
    print(f"Backends: {backends}")
    print(f"dtype: {args.dtype}")
    print(f"Warmup: {args.num_warmup}, Iterations: {args.num_iters}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        major, minor = torch.cuda.get_device_capability()
        print(f"Compute capability: {major}.{minor}")

    # Generate configs
    configs = []
    studies = ["a", "b", "c"] if args.study == "all" else [args.study]
    for s in studies:
        if s == "a":
            configs.extend(generate_study_a(models, backends))
        elif s == "b":
            configs.extend(generate_study_b(models, backends))
        elif s == "c":
            configs.extend(generate_study_c(models, backends))

    print(f"\nTotal configurations: {len(configs)}")

    # Dry run: just print configs
    if args.dry_run:
        print(
            f"\n{'Study':>5} {'Model':>16} {'Backend':>16} {'Resolution':>10} "
            f"{'Images':>6} {'Patches':>10} {'Mix':>25}"
        )
        print("-" * 100)
        for config in configs:
            profile = MODEL_PROFILES[config.model_name]
            _, patches_list = compute_cu_seqlens(config.image_dims, profile)
            total = sum(patches_list)
            patches_str = "+".join(str(p) for p in patches_list)
            print(
                f"{config.study:>5} {config.model_name:>16} {config.backend:>16} "
                f"{config.resolution_label:>10} {config.image_count:>6} "
                f"{total:>10} {config.mix_label:>25}"
            )
            if len(patches_list) <= 4:
                print(f"{'':>60} patches: [{patches_str}]")
        return

    # Create backends (cache to avoid re-creation)
    backend_cache = {}
    for b in backends:
        backend_cache[b] = create_backend(b)

    # Run benchmarks
    results = []
    total = len(configs)
    for i, config in enumerate(configs):
        profile = MODEL_PROFILES[config.model_name]
        cu_seqlens, patches_list = compute_cu_seqlens(config.image_dims, profile)
        total_patches = sum(patches_list)

        label = (
            f"[{i+1}/{total}] study={config.study} model={config.model_name} "
            f"backend={config.backend} "
        )
        if config.resolution_label:
            label += f"res={config.resolution_label} "
        if config.mix_label:
            label += f"mix={config.mix_label} "
        label += f"images={config.image_count} patches={total_patches}"
        print(label, end=" ... ", flush=True)

        timing = benchmark_attention(
            backend_name=config.backend,
            backend=backend_cache[config.backend],
            num_heads=profile.num_heads,
            head_dim=profile.head_dim,
            cu_seqlens=cu_seqlens,
            patches_list=patches_list,
            num_warmup=args.num_warmup,
            num_iters=args.num_iters,
            dtype=dtype,
        )
        print(f"median={timing['median_ms']:.3f}ms mean={timing['mean_ms']:.3f}ms")

        row = format_result_row(config, profile, patches_list, timing)
        results.append(row)

    # Write CSV
    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults written to {args.output}")

    # Print summary tables
    for s in studies:
        print_study_summary(results, s.upper())


if __name__ == "__main__":
    main()
