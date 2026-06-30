"""
Benchmark fused AllReduce + RMSNorm + activation-quant on AMD with correctness
checks.

For a chosen ``--quant`` variant this script compares the same three op paths
used by SGLang on ROCm/aiter for Qwen3.5-style models:

    1. Split (3 kernels) - reference:
         tensor_model_parallel_all_reduce -> RMSNorm -> aiter standalone quant.
    2. Fused AR+RMSNorm + separate quant (2 kernels):
         tensor_model_parallel_fused_allreduce_rmsnorm -> aiter standalone quant.
    3. Fully fused AR+RMSNorm+quant (1 kernel):
         tensor_model_parallel_fused_allreduce_rmsnorm_quant_per_group / _per_token /
         _mxfp4_quant.

Supported quant variants (``--quant``):
  * per_group - #24651's per-1x128 FP8 path (default; numeric correctness).
  * per_token - per-1x1 (per-token) FP8 path (numeric correctness).
  * mxfp4     - per-1x32 microscaling FP4 path (bf16-domain correctness; the
                fp4 payload is checked structurally).

Default shape sets cover the Qwen3.5-397B-A17B layout:
  * hidden_size = 4096
  * TP = 8 (launched with torchrun --nproc_per_node=8)
  * Prefill batch sizes up to a few thousand tokens.
  * Decode batch sizes 1-512 covering typical steady-state running_req values.

Usage:
  # per-group FP8 (original #24651 path)
  torchrun --nproc_per_node=8 \
    benchmark/kernels/all_reduce/benchmark_fused_ar_rms_quant_amd.py \
    --dtype bf16 --quant per_group --group-size 128

  # per-token FP8 / mxfp4 paths added on top of #24651
  torchrun --nproc_per_node=8 \
    benchmark/kernels/all_reduce/benchmark_fused_ar_rms_quant_amd.py \
    --dtype bf16 --quant per_token
  torchrun --nproc_per_node=8 \
    benchmark/kernels/all_reduce/benchmark_fused_ar_rms_quant_amd.py \
    --dtype bf16 --quant mxfp4
"""

import argparse
import csv
import os
import statistics
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from sglang.srt.distributed.communication_op import (
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_fused_allreduce_rmsnorm,
    tensor_model_parallel_fused_allreduce_rmsnorm_mxfp4_quant,
    tensor_model_parallel_fused_allreduce_rmsnorm_quant_per_group,
    tensor_model_parallel_fused_allreduce_rmsnorm_quant_per_token,
)
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    graph_capture,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
)

Shape = Tuple[int, int]
FP8_DTYPE = torch.float8_e4m3fnuz

# Quant variants this benchmark can target. The default (per_group) is #24651's
# original per-1x128 FP8 path; per_token and mxfp4 exercise the entry points added
# on top of #24651 (MXFP4-AttnFP8 / plain-MXFP4 checkpoints).
QUANT_CHOICES = ("per_group", "per_token", "mxfp4")


def _aiter_quant_type(quant: str):
    """aiter QuantType for the standalone (reference) quant of a given variant."""
    import aiter

    return {
        "per_group": aiter.QuantType.per_1x128,
        "per_token": aiter.QuantType.per_Token,
        "mxfp4": aiter.QuantType.per_1x32,
    }[quant]


def _aiter_quant_dtype(quant: str):
    """aiter element dtype produced by the standalone quant of a given variant."""
    import aiter

    if quant == "mxfp4":
        # fp4x2-packed (two fp4 per byte) on builds that expose it; the standalone
        # mxfp4 reference is timing-only (correctness for mxfp4 is bf16-domain).
        return getattr(aiter.dtypes, "fp4x2", getattr(aiter.dtypes, "fp4", None))
    return aiter.dtypes.fp8


def _standalone_quant(normed: torch.Tensor, quant: str):
    """Reference standalone quant used by the split-3k / fused-2k baselines.

    Returns ``(q, scale)`` or ``None`` when the variant's standalone quant is not
    available in this aiter build (only happens for mxfp4 on older builds — the
    fused mxfp4 kernel and its bf16-domain correctness check still run)."""
    import aiter

    quant_dtype = _aiter_quant_dtype(quant)
    if quant_dtype is None:
        return None
    try:
        hip_quant = aiter.get_hip_quant(_aiter_quant_type(quant))
        return hip_quant(normed, quant_dtype=quant_dtype)
    except Exception:
        return None


def parse_shapes(raw: str) -> List[Shape]:
    shapes: List[Shape] = []
    for item in [x.strip() for x in raw.split(",") if x.strip()]:
        if "x" not in item:
            raise ValueError(f"Invalid shape '{item}', expected MxN format.")
        m_str, n_str = item.split("x", 1)
        m, n = int(m_str), int(n_str)
        if m <= 0 or n <= 0:
            raise ValueError(f"Invalid shape '{item}', both dims must be positive.")
        shapes.append((m, n))
    if not shapes:
        raise ValueError("Empty shape list is not allowed.")
    return shapes


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def _barrier(device: torch.device) -> None:
    try:
        dist.barrier(device_ids=[device.index])
    except TypeError:
        dist.barrier()


def _mean_across_ranks(val: float, device: torch.device) -> float:
    t = torch.tensor([val], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return float(t.item())


def _all_true_across_ranks(val: bool, device: torch.device) -> bool:
    t = torch.tensor([1 if val else 0], dtype=torch.int32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(int(t.item()))


def _measure_us(
    fn, warmup: int, iters: int, repeats: int, device: torch.device
) -> float:
    for _ in range(max(1, warmup)):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: List[float] = []
    for _ in range(max(1, repeats)):
        _barrier(device)
        torch.cuda.synchronize()
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iters)
    samples.sort()
    return float(statistics.median(samples))


def _make_inputs(
    shape: Shape, dtype: torch.dtype, seed: int, rank: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = shape
    torch.manual_seed(seed + rank * 17)
    # fp32 first then downcast so every rank has distinct values that still
    # sum to a well-conditioned pre-norm tensor after all-reduce.
    x = torch.randn((m, n), dtype=torch.float32, device=device).to(dtype)
    residual = x.clone()
    weight = torch.randn((n,), dtype=torch.float32, device=device).to(dtype)
    return x, residual, weight


def _ar_rms_bf16(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Reference AR+RMSNorm (bf16) used for the mxfp4 bf16-domain correctness."""
    return tensor_model_parallel_fused_allreduce_rmsnorm(
        x.clone(), residual.clone(), weight, eps
    )


def _split_3_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int,
    quant: str,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Reference: plain all_reduce -> RMSNorm -> aiter standalone quant."""
    ar_out = tensor_model_parallel_all_reduce(x.clone())
    residual_out = ar_out + residual
    normed = F.rms_norm(residual_out, (residual_out.shape[-1],), weight, eps)
    q = _standalone_quant(normed, quant)
    if q is None:
        return None
    fp8_out, scale_out = q
    return fp8_out, residual_out, scale_out


def _fused_ar_rms_then_quant(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int,
    quant: str,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """2-kernel: fused AR+RMSNorm (existing) + separate standalone quant."""
    result = tensor_model_parallel_fused_allreduce_rmsnorm(
        x.clone(), residual.clone(), weight, eps
    )
    if result is None:
        return None
    normed, residual_out = result
    q = _standalone_quant(normed, quant)
    if q is None:
        return None
    fp8_out, scale_out = q
    return fp8_out, residual_out, scale_out


def _fully_fused(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int,
    quant: str,
) -> Optional[Tuple[torch.Tensor, ...]]:
    """1-kernel: fused AR+RMSNorm+quant (quant pair only, no bf16 sidecar)."""
    if quant == "per_group":
        return tensor_model_parallel_fused_allreduce_rmsnorm_quant_per_group(
            x.clone(), residual.clone(), weight, eps, group_size
        )
    if quant == "per_token":
        return tensor_model_parallel_fused_allreduce_rmsnorm_quant_per_token(
            x.clone(), residual.clone(), weight, eps
        )
    return tensor_model_parallel_fused_allreduce_rmsnorm_mxfp4_quant(
        x.clone(), residual.clone(), weight, eps
    )


def _fully_fused_with_bf16(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int,
    quant: str,
) -> Optional[Tuple[torch.Tensor, ...]]:
    """1-kernel fused AR+RMSNorm+quant with a bf16 side-output (4-tuple).

    Supported by the per-group FP8 and MXFP4 kernels (GDN keep_bf16=True path).
    The per-token kernel has no bf16 sidecar, so this returns ``None`` there.
    """
    if quant == "per_group":
        return tensor_model_parallel_fused_allreduce_rmsnorm_quant_per_group(
            x.clone(), residual.clone(), weight, eps, group_size, emit_bf16=True
        )
    if quant == "mxfp4":
        return tensor_model_parallel_fused_allreduce_rmsnorm_mxfp4_quant(
            x.clone(), residual.clone(), weight, eps, emit_bf16=True
        )
    return None


def _check_quant_close(
    fp8_a: torch.Tensor,
    scale_a: torch.Tensor,
    fp8_b: torch.Tensor,
    scale_b: torch.Tensor,
    group_size: int,
) -> Tuple[bool, str]:
    """Compare two (fp8, scale) quantized outputs by dequantizing.

    Handles both per-group scales (one scale per ``group_size`` cols) and
    per-token scales (a single ``(M, 1)`` column broadcast across the row)."""

    def _dq(fp8, scale):
        if scale.shape[-1] == fp8.shape[-1]:
            return fp8.float() * scale
        if scale.shape[-1] == 1:
            return fp8.float() * scale  # per-token: broadcast (M,1) over the row
        return fp8.float() * scale.repeat_interleave(group_size, dim=-1)

    dq_a = _dq(fp8_a, scale_a)
    dq_b = _dq(fp8_b, scale_b)
    max_diff = (dq_a - dq_b).abs().max().item()
    denom = dq_a.abs().max().item() + 1e-6
    rel_err = max_diff / denom
    ok = rel_err < 0.15
    return ok, f"max_diff={max_diff:.4f},rel={rel_err:.4f}"


def bench_shape(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int,
    warmup: int,
    iters: int,
    repeats: int,
    mode: str,
    quant: str,
) -> Dict[str, object]:
    device = x.device

    def _graphed(call):
        """Return a replay-able fn in graph mode, else the eager call itself."""
        if mode != "graph":
            return call
        with graph_capture() as gc:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=gc.stream):
                call()
        return g.replay

    # --- Split 3-kernel baseline (AR -> RMSNorm -> standalone quant) ---
    # The standalone quant may be unavailable for mxfp4 on older aiter builds;
    # the fused kernel + bf16-domain correctness still run in that case.
    split_available = _split_3_reference(x, residual, weight, eps, group_size, quant) is not None
    split_us: Optional[float] = None
    if split_available:
        split_fn = _graphed(
            lambda: _split_3_reference(x, residual, weight, eps, group_size, quant)
        )
        split_us = _measure_us(split_fn, warmup, iters, repeats, device)

    # --- Fused AR+RMSNorm + separate quant (2 kernels) ---
    fused2_available = (
        _fused_ar_rms_then_quant(x, residual, weight, eps, group_size, quant) is not None
    )
    fused2_us: Optional[float] = None
    if fused2_available:
        fused2_fn = _graphed(
            lambda: _fused_ar_rms_then_quant(x, residual, weight, eps, group_size, quant)
        )
        fused2_us = _measure_us(fused2_fn, warmup, iters, repeats, device)

    # --- Fully fused, quant-pair only (1 kernel) — std-attention path ---
    fused1_available = (
        _fully_fused(x, residual, weight, eps, group_size, quant) is not None
    )
    fused1_us: Optional[float] = None
    if fused1_available:
        fused1_fn = _graphed(
            lambda: _fully_fused(x, residual, weight, eps, group_size, quant)
        )
        fused1_us = _measure_us(fused1_fn, warmup, iters, repeats, device)

    # --- Fully fused + bf16 sidecar (1 kernel) — GDN keep_bf16=True path ---
    probe1b = _fully_fused_with_bf16(x, residual, weight, eps, group_size, quant)
    fused1bf16_available = (
        probe1b is not None and isinstance(probe1b, tuple) and len(probe1b) == 4
    )
    fused1bf16_us: Optional[float] = None
    if fused1bf16_available:
        fused1bf16_fn = _graphed(
            lambda: _fully_fused_with_bf16(x, residual, weight, eps, group_size, quant)
        )
        fused1bf16_us = _measure_us(fused1bf16_fn, warmup, iters, repeats, device)

    # --- Correctness ---
    correctness = "N/A"
    correctness_bf16 = "N/A"
    if quant == "mxfp4":
        # fp4 (e2m1 + e8m0 block scale) numeric dequant is fragile to hand-roll,
        # so validate the fused kernel's AR+RMSNorm in the bf16 domain (its
        # emit_bf16 side-output vs the reference fused AR+RMSNorm) and confirm the
        # fp4 (packed) + scale payload is structurally present.
        ref = _ar_rms_bf16(x, residual, weight, eps)
        if fused1bf16_available and ref is not None:
            res1b = _fully_fused_with_bf16(x, residual, weight, eps, group_size, quant)
            ref_bf16, _ = ref
            diff = (res1b[3].float() - ref_bf16.float()).abs().max().item()
            denom = ref_bf16.float().abs().max().item() + 1e-6
            rel = diff / denom
            correctness_bf16 = (
                f"PASS(bf16_rel={rel:.4f})" if rel < 0.02 else f"FAIL(bf16_rel={rel:.4f})"
            )
        if fused1_available:
            res1 = _fully_fused(x, residual, weight, eps, group_size, quant)
            ok_struct = (
                isinstance(res1, tuple)
                and len(res1) >= 3
                and res1[0] is not None
                and res1[2] is not None
            )
            correctness = "PASS(struct)" if ok_struct else "FAIL(struct)"
    else:
        # FP8 per-group / per-token: numeric dequant compare.
        if fused1_available and fused2_available:
            res1 = _fully_fused(x, residual, weight, eps, group_size, quant)
            res2 = _fused_ar_rms_then_quant(x, residual, weight, eps, group_size, quant)
            ok, detail = _check_quant_close(
                res2[0], res2[2], res1[0], res1[2], group_size
            )
            correctness = "PASS" if ok else f"FAIL({detail})"

        if fused1bf16_available and fused2_available:
            res1b = _fully_fused_with_bf16(x, residual, weight, eps, group_size, quant)
            res2 = _fused_ar_rms_then_quant(x, residual, weight, eps, group_size, quant)
            ok_fp8, detail_fp8 = _check_quant_close(
                res2[0], res2[2], res1b[0], res1b[2], group_size
            )
            # bf16 sidecar (only present for per_group here) vs its own fp8 pair.
            bf16_vs_fp8 = (
                (
                    res1b[3].float()
                    - (
                        res1b[0].float()
                        * res1b[2].repeat_interleave(group_size, dim=-1)
                    )
                )
                .abs()
                .max()
                .item()
            )
            if not ok_fp8:
                correctness_bf16 = f"FAIL_fp8({detail_fp8})"
            elif bf16_vs_fp8 > 1.0:
                correctness_bf16 = f"FAIL_bf16(diff={bf16_vs_fp8:.4f})"
            else:
                correctness_bf16 = f"PASS(bf16_diff={bf16_vs_fp8:.3f})"

    return {
        "split_us": split_us,
        "split_available": split_available,
        "fused2_available": fused2_available,
        "fused2_us": fused2_us,
        "fused1_available": fused1_available,
        "fused1_us": fused1_us,
        "fused1bf16_available": fused1bf16_available,
        "fused1bf16_us": fused1bf16_us,
        "correctness": correctness,
        "correctness_bf16": correctness_bf16,
    }


# Qwen3.5-397B-A17B-FP8 has hidden_size=4096 (both GDN and standard attention
# layers go through input_layernorm at full hidden dim before the TP projections
# are applied, so the fused op sees [M, 4096] inputs on every rank).
_DEFAULT_PREFILL_SHAPES = (
    "64x4096,128x4096,256x4096,512x4096,1024x4096,2048x4096,4096x4096,8192x4096"
)
_DEFAULT_DECODE_SHAPES = (
    "1x4096,2x4096,4x4096,8x4096,16x4096,32x4096,64x4096,128x4096,256x4096,512x4096"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark fused AR+RMSNorm+quant (per_group / per_token / mxfp4) "
            "for Qwen3.5 shapes."
        )
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "float16", "bfloat16"],
    )
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--quant",
        type=str,
        default="per_group",
        choices=QUANT_CHOICES,
        help=(
            "Quant variant of the fused AR+RMSNorm+quant kernel to benchmark: "
            "per_group (#24651 per-1x128 FP8), per_token (per-1x1 FP8), or "
            "mxfp4 (per-1x32 microscaling). group_size only applies to per_group."
        ),
    )
    parser.add_argument("--prefill-shapes", type=str, default=_DEFAULT_PREFILL_SHAPES)
    parser.add_argument("--decode-shapes", type=str, default=_DEFAULT_DECODE_SHAPES)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["eager", "graph", "both"],
    )
    parser.add_argument("--csv-out", type=str, default=None)
    args = parser.parse_args()

    dtype = dtype_from_name(args.dtype)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank % torch.cuda.device_count())
    device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")

    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    if rank == 0:
        gs_note = args.group_size if args.quant == "per_group" else "n/a"
        print(
            f"Config: world_size={world_size}, dtype={dtype}, "
            f"quant={args.quant}, group_size={gs_note}"
        )
        print(
            f"  1-stage boundary: total_bytes <= 128KB "
            f"(M <= {128 * 1024 // (4096 * 2)} for hidden=4096 bf16)"
        )
        print(
            f"  2-stage boundary: total_bytes <= 512KB "
            f"(M <= {512 * 1024 // (4096 * 2)} for hidden=4096 bf16)"
        )
        print(
            "  fallback: fused_ar_rms + standalone quant (2 kernels) "
            "when single-kernel path is unavailable"
        )
        if args.quant == "mxfp4":
            print(
                "  mxfp4: correctness is bf16-domain (emit_bf16 sidecar vs "
                "reference AR+RMSNorm); 'Corr fp8' reports the fp4 payload "
                "structural check"
            )

    run_modes = ("eager", "graph") if args.mode == "both" else (args.mode,)
    csv_rows: List[Dict[str, object]] = []

    for mode in run_modes:
        shapes = parse_shapes(
            args.prefill_shapes if mode == "eager" else args.decode_shapes
        )
        if rank == 0:
            phase = "prefill(eager)" if mode == "eager" else "decode(graph)"
            print(f"\n{'=' * 145}")
            print(f"Mode: {phase}")
            print(
                "| Shape | Bytes/rank | Split(3k) us | Fused2(2k) us | "
                "Fused1(1k) us | Fused1+bf16(1k) us | Speedup(2k) | "
                "Speedup(1k) | Speedup(1k+bf16) | Corr fp8 | Corr bf16 |"
            )
            print(
                "|:------|----------:|-----------:|------------:|-----------:|"
                "-----------:|-----------:|-----------:|-----------:|"
                ":---------|:----------|"
            )

        for shape in shapes:
            x, residual, weight = _make_inputs(shape, dtype, args.seed, rank, device)
            m = bench_shape(
                x,
                residual,
                weight,
                args.eps,
                args.group_size,
                args.warmup,
                args.iters,
                args.repeats,
                mode,
                args.quant,
            )

            split_us = (
                _mean_across_ranks(m["split_us"], device)
                if m["split_us"] is not None
                else None
            )
            split_avail = _all_true_across_ranks(m["split_available"], device)
            fused2_avail = _all_true_across_ranks(m["fused2_available"], device)
            fused1_avail = _all_true_across_ranks(m["fused1_available"], device)
            fused1bf16_avail = _all_true_across_ranks(m["fused1bf16_available"], device)
            fused2_us = (
                _mean_across_ranks(m["fused2_us"], device)
                if m["fused2_us"] is not None
                else None
            )
            fused1_us = (
                _mean_across_ranks(m["fused1_us"], device)
                if m["fused1_us"] is not None
                else None
            )
            fused1bf16_us = (
                _mean_across_ranks(m["fused1bf16_us"], device)
                if m["fused1bf16_us"] is not None
                else None
            )

            if rank == 0:
                M, N = shape
                nbytes = M * N * 2
                split_str = f"{split_us:.1f}" if split_us else "N/A"
                f2_str = f"{fused2_us:.1f}" if fused2_us else "N/A"
                f1_str = f"{fused1_us:.1f}" if fused1_us else "N/A"
                f1b_str = f"{fused1bf16_us:.1f}" if fused1bf16_us else "N/A"
                s2 = (
                    f"{split_us / fused2_us:.2f}x"
                    if split_us and fused2_us and fused2_us > 0
                    else "N/A"
                )
                s1 = (
                    f"{split_us / fused1_us:.2f}x"
                    if split_us and fused1_us and fused1_us > 0
                    else "N/A"
                )
                s1b = (
                    f"{fused2_us / fused1bf16_us:.2f}x"
                    if fused1bf16_us and fused2_us and fused1bf16_us > 0
                    else "N/A"
                )
                print(
                    f"| {M}x{N} | {nbytes} | {split_str} | {f2_str} | "
                    f"{f1_str} | {f1b_str} | {s2} | {s1} | {s1b} | "
                    f"{m['correctness']} | {m['correctness_bf16']} |"
                )
                csv_rows.append(
                    {
                        "quant": args.quant,
                        "mode": mode,
                        "shape": f"{M}x{N}",
                        "m": M,
                        "n": N,
                        "bytes_per_rank": nbytes,
                        "split_us": split_us if split_us is not None else "",
                        "fused2_us": fused2_us if fused2_us is not None else "",
                        "fused1_us": fused1_us if fused1_us is not None else "",
                        "fused1bf16_us": (
                            fused1bf16_us if fused1bf16_us is not None else ""
                        ),
                        "split_available": split_avail,
                        "fused1_available": fused1_avail,
                        "fused2_available": fused2_avail,
                        "fused1bf16_available": fused1bf16_avail,
                        "correctness": m["correctness"],
                        "correctness_bf16": m["correctness_bf16"],
                    }
                )

    if rank == 0 and args.csv_out and csv_rows:
        os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
        with open(args.csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\nSaved CSV: {args.csv_out}")

    _barrier(device)
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    main()
