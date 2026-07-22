# Benchmark FP8 attention for FA4 (CuTe-DSL) on SM100.
#
# Run (recommended):
#   python -m flash_attn.cute.benchmark_flash_attention_fp8
#
# Notes:
# - This is intended to be used while bringing up FP8 support for SM100.
# - FP8 correctness depends on descales + max-offset scaling being implemented in the SM100 kernel.
#   This script optionally checks output vs a BF16 PyTorch baseline on dequantized FP8 inputs.
#
# Adapted from: `hopper/benchmark_flash_attention_fp8.py`

from __future__ import annotations

import argparse
import inspect
import math
import time
from typing import Iterable

import torch
from einops import rearrange

from sglang.kernels.ops.attention.flash_attn.cute.benchmark import benchmark_forward
from sglang.kernels.ops.attention.flash_attn.cute.interface import (
    _flash_attn_fwd as flash_attn_cute_fwd,
)

try:
    import cudnn
except ImportError:
    cudnn = None


def _torch_float8_dtype(name: str) -> torch.dtype:
    if name in ("fp8", "fp8_e4m3", "fp8_e4m3fn"):
        return torch.float8_e4m3fn
    if name in ("fp8_e5m2", "fp8_e5m2fn"):
        return torch.float8_e5m2
    raise ValueError(f"Unsupported fp8 dtype name: {name}")


def _parse_int_list(csv: str) -> list[int]:
    out: list[int] = []
    for part in csv.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def attention_pytorch(qkv: torch.Tensor, causal: bool) -> torch.Tensor:
    """
    qkv: (batch, seqlen, 3, nheads, headdim)
    out: (batch, seqlen, nheads, headdim)
    """
    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    q = rearrange(q, "b t h d -> (b h) t d")
    k = rearrange(k, "b s h d -> (b h) d s")
    softmax_scale = 1.0 / math.sqrt(d)
    scores = torch.empty(
        batch_size * nheads, seqlen, seqlen, dtype=qkv.dtype, device=qkv.device
    )
    scores = rearrange(
        torch.baddbmm(scores, q, k, beta=0, alpha=softmax_scale),
        "(b h) t s -> b h t s",
        h=nheads,
    )
    if causal:
        causal_mask = torch.triu(
            torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1
        )
        scores = scores + causal_mask.to(dtype=scores.dtype)
    attention = torch.softmax(scores, dim=-1)
    output = torch.einsum("bhts,bshd->bthd", attention, v)
    return output.to(dtype=qkv.dtype)


def flops(batch: int, seqlen: int, headdim: int, nheads: int, causal: bool) -> int:
    # Matches the hopper benchmark’s convention.
    return 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)


def efficiency(flop: int, seconds: float) -> float:
    return (flop / seconds / 1e12) if not math.isnan(seconds) else 0.0


def time_fwd(fn, *args, repeats: int, **kwargs) -> float:
    time.sleep(1)  # reduce residual throttling effects between benchmarks
    _, m = benchmark_forward(fn, *args, repeats=repeats, verbose=False, **kwargs)
    return float(m.mean)


def convert_to_cudnn_type(torch_type):
    if torch_type == torch.float16:
        return cudnn.data_type.HALF
    if torch_type == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    if torch_type == torch.float32:
        return cudnn.data_type.FLOAT
    if torch_type == torch.int32:
        return cudnn.data_type.INT32
    if torch_type == torch.int64:
        return cudnn.data_type.INT64
    if torch_type == torch.float8_e4m3fn:
        return cudnn.data_type.FP8_E4M3
    if torch_type == torch.float8_e5m2:
        return cudnn.data_type.FP8_E5M2
    raise ValueError("Unsupported tensor data type.")


def cudnn_sdpa_fp8_setup(qkv: torch.Tensor, seqlen_q: int, seqlen_k: int, causal: bool):
    """Minimal cudnn.fp8 sdpa runner (optional)."""
    assert cudnn is not None, "cudnn python bindings not available"
    b, _, _, nheads, headdim = qkv.shape
    o_gpu = torch.zeros(
        b, seqlen_q, nheads, headdim, dtype=qkv.dtype, device=qkv.device
    )
    o_gpu_transposed = torch.as_strided(
        o_gpu,
        [b, nheads, seqlen_q, headdim],
        [nheads * seqlen_q * headdim, headdim, nheads * headdim, 1],
    )
    amax_s_gpu = torch.empty(1, 1, 1, 1, dtype=torch.float32, device=qkv.device)
    amax_o_gpu = torch.empty(1, 1, 1, 1, dtype=torch.float32, device=qkv.device)

    graph = cudnn.pygraph(
        io_data_type=convert_to_cudnn_type(qkv.dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    new_q = torch.as_strided(
        qkv,
        [b, nheads, seqlen_q, headdim],
        [seqlen_q * nheads * headdim * 3, headdim, headdim * nheads * 3, 1],
        storage_offset=0,
    )
    q = graph.tensor(
        name="Q",
        dim=list(new_q.shape),
        stride=list(new_q.stride()),
        data_type=convert_to_cudnn_type(qkv.dtype),
    )

    new_k = torch.as_strided(
        qkv,
        [b, nheads, seqlen_k, headdim],
        [seqlen_k * nheads * headdim * 3, headdim, headdim * nheads * 3, 1],
        storage_offset=nheads * headdim,
    )
    k = graph.tensor(
        name="K",
        dim=list(new_k.shape),
        stride=list(new_k.stride()),
        data_type=convert_to_cudnn_type(qkv.dtype),
    )

    new_v = torch.as_strided(
        qkv,
        [b, nheads, seqlen_k, headdim],
        [seqlen_k * nheads * headdim * 3, headdim, headdim * nheads * 3, 1],
        storage_offset=nheads * headdim * 2,
    )
    v = graph.tensor(
        name="V",
        dim=list(new_v.shape),
        stride=list(new_v.stride()),
        data_type=convert_to_cudnn_type(qkv.dtype),
    )

    def _scale_tensor():
        return graph.tensor(
            dim=[1, 1, 1, 1], stride=[1, 1, 1, 1], data_type=cudnn.data_type.FLOAT
        )

    default_scale_gpu = torch.ones(1, 1, 1, 1, dtype=torch.float32, device="cuda")
    descale_q = _scale_tensor()
    descale_k = _scale_tensor()
    descale_v = _scale_tensor()
    descale_s = _scale_tensor()
    scale_s = _scale_tensor()
    scale_o = _scale_tensor()

    o, _, amax_s, amax_o = graph.sdpa_fp8(
        q=q,
        k=k,
        v=v,
        descale_q=descale_q,
        descale_k=descale_k,
        descale_v=descale_v,
        descale_s=descale_s,
        scale_s=scale_s,
        scale_o=scale_o,
        is_inference=True,
        attn_scale=1.0 / math.sqrt(headdim),
        use_causal_mask=causal,
        name="sdpa",
    )
    o.set_output(True).set_dim(o_gpu_transposed.shape).set_stride(
        o_gpu_transposed.stride()
    )
    amax_s.set_output(False).set_dim(amax_s_gpu.shape).set_stride(amax_s_gpu.stride())
    amax_o.set_output(False).set_dim(amax_o_gpu.shape).set_stride(amax_o_gpu.stride())

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    variant_pack = {
        q: new_q,
        k: new_k,
        v: new_v,
        descale_q: default_scale_gpu,
        descale_k: default_scale_gpu,
        descale_v: default_scale_gpu,
        descale_s: default_scale_gpu,
        scale_s: default_scale_gpu,
        scale_o: default_scale_gpu,
        o: o_gpu_transposed,
        amax_s: amax_s_gpu,
        amax_o: amax_o_gpu,
    }
    workspace = torch.empty(
        graph.get_workspace_size(), device="cuda", dtype=torch.uint8
    )

    def run():
        graph.execute(variant_pack, workspace)
        return o_gpu

    return run


def _maybe_pass_descales(callable_, **kwargs):
    sig = inspect.signature(callable_)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--headdims", default="64,128")
    parser.add_argument("--dtype", default="fp8_e4m3fn")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable correctness checks vs BF16 PyTorch baseline.",
    )
    parser.add_argument(
        "--check-quantization-only",
        action="store_true",
        help="Check FP8 kernel vs dequantized-FP8 baseline (quantization error only).",
    )
    parser.add_argument("--atol-bf16", type=float, default=0.10)
    parser.add_argument("--rtol-bf16", type=float, default=0.10)
    parser.add_argument("--atol-fp8", type=float, default=0.50)
    parser.add_argument("--rtol-fp8", type=float, default=0.50)
    parser.add_argument("--run-cudnn", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if major != 10:
        raise RuntimeError(
            f"This benchmark is for SM100 (compute capability 10.x). Got {major}.{minor}."
        )

    torch.manual_seed(args.seed)
    device = "cuda"
    fp8_dtype = _torch_float8_dtype(args.dtype)
    headdim_vals = _parse_int_list(args.headdims)
    bs_seqlen_vals = [
        (32, 512),
        (16, 1024),
        (8, 2048),
        (4, 4096),
        (2, 8192),
        (1, 16384),
    ]

    methods = ["Pytorch", "FA4-CuTe-BF16", "FA4-CuTe-FP8"] + (
        ["cuDNN-FP8"] if args.run_cudnn and cudnn is not None else []
    )

    fp8_failures = []

    for headdim in headdim_vals:
        for causal in (False, True):
            for batch, seqlen in bs_seqlen_vals:
                torch.cuda.empty_cache()
                nheads = args.dim // headdim
                if args.dim % headdim != 0:
                    raise ValueError(
                        f"--dim must be divisible by headdim ({args.dim=} {headdim=})"
                    )

                q_bf16 = torch.randn(
                    batch, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16
                )
                k_bf16 = torch.randn(
                    batch, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16
                )
                v_bf16 = torch.randn(
                    batch, seqlen, nheads, headdim, device=device, dtype=torch.bfloat16
                )
                qkv_bf16 = torch.stack([q_bf16, k_bf16, v_bf16], dim=2)

                times = {}
                speeds = {}

                out_ref_bf16 = None
                try:
                    out_ref_bf16 = attention_pytorch(
                        qkv_bf16, causal=causal
                    )  # warmup / reference
                    t = time_fwd(
                        attention_pytorch, qkv_bf16, causal=causal, repeats=args.repeats
                    )
                    times["Pytorch"] = t
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        times["Pytorch"] = float("nan")
                        out_ref_bf16 = None
                    else:
                        raise

                # FA4 / CuTe BF16 baseline
                try:
                    softmax_scale = headdim**-0.5
                    out_fa4_bf16, _ = flash_attn_cute_fwd(
                        q_bf16,
                        k_bf16,
                        v_bf16,
                        softmax_scale=softmax_scale,
                        causal=causal,
                    )  # warmup / compile
                    t = time_fwd(
                        flash_attn_cute_fwd,
                        q_bf16,
                        k_bf16,
                        v_bf16,
                        softmax_scale=softmax_scale,
                        causal=causal,
                        repeats=args.repeats,
                    )
                    times["FA4-CuTe-BF16"] = t
                    if args.check and out_ref_bf16 is not None:
                        torch.testing.assert_close(
                            out_fa4_bf16,
                            out_ref_bf16,
                            atol=args.atol_bf16,
                            rtol=args.rtol_bf16,
                        )
                except Exception as e:
                    # Treat as fatal: BF16 kernel should be usable for basic sanity checking.
                    raise RuntimeError("FA4-CuTe BF16 baseline failed") from e

                # FA4 / CuTe FP8
                q_fp8 = q_bf16.to(fp8_dtype)
                k_fp8 = k_bf16.to(fp8_dtype)
                v_fp8 = v_bf16.to(fp8_dtype)

                # Placeholder descales (FA3-style: per-(batch, kv_head)).
                q_descale = torch.ones(
                    batch, nheads, device=device, dtype=torch.float32
                )
                k_descale = torch.ones(
                    batch, nheads, device=device, dtype=torch.float32
                )
                v_descale = torch.ones(
                    batch, nheads, device=device, dtype=torch.float32
                )

                # Optional: FP8 reference baseline (dequantized FP8 -> PyTorch) for quantization-error-only checks
                out_ref_fp8 = None
                if args.check and args.check_quantization_only:
                    try:
                        # Dequantize FP8 inputs back to BF16 (applying descales)
                        q_ref_fp8 = (
                            q_fp8.to(torch.bfloat16) * q_descale[:, None, :, None]
                        ).to(torch.bfloat16)
                        k_ref_fp8 = (
                            k_fp8.to(torch.bfloat16) * k_descale[:, None, :, None]
                        ).to(torch.bfloat16)
                        v_ref_fp8 = (
                            v_fp8.to(torch.bfloat16) * v_descale[:, None, :, None]
                        ).to(torch.bfloat16)
                        qkv_ref_fp8 = torch.stack(
                            [q_ref_fp8, k_ref_fp8, v_ref_fp8], dim=2
                        )
                        out_ref_fp8 = attention_pytorch(qkv_ref_fp8, causal=causal)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            out_ref_fp8 = None
                        else:
                            raise

                fa4_kwargs = dict(softmax_scale=softmax_scale, causal=causal)
                fa4_kwargs.update(
                    _maybe_pass_descales(
                        flash_attn_cute_fwd,
                        q_descale=q_descale,
                        k_descale=k_descale,
                        v_descale=v_descale,
                    )
                )

                try:
                    # Warmup/compile (will raise until FP8 is implemented)
                    out_fa4_fp8, _ = flash_attn_cute_fwd(
                        q_fp8, k_fp8, v_fp8, **fa4_kwargs
                    )
                    t = time_fwd(
                        flash_attn_cute_fwd,
                        q_fp8,
                        k_fp8,
                        v_fp8,
                        repeats=args.repeats,
                        **fa4_kwargs,
                    )
                    times["FA4-CuTe-FP8"] = t
                    if args.check:
                        # Choose baseline: quantization-only (dequantized FP8) or full (BF16)
                        if args.check_quantization_only:
                            ref_baseline = out_ref_fp8
                        else:
                            ref_baseline = out_ref_bf16

                        if ref_baseline is not None:
                            torch.testing.assert_close(
                                out_fa4_fp8,
                                ref_baseline,
                                atol=args.atol_fp8,
                                rtol=args.rtol_fp8,
                            )
                except Exception as e:
                    fp8_failures.append((causal, headdim, batch, seqlen, repr(e)))
                    times["FA4-CuTe-FP8"] = float("nan")

                if args.run_cudnn and cudnn is not None:
                    qkv_fp8 = qkv_bf16.to(fp8_dtype)
                    runner = cudnn_sdpa_fp8_setup(
                        qkv_fp8, seqlen, seqlen, causal=causal
                    )
                    _ = runner()  # warmup
                    t = time_fwd(lambda: runner(), repeats=args.repeats)
                    times["cuDNN-FP8"] = t

                print(
                    f"### causal={causal}, headdim={headdim}, batch={batch}, seqlen={seqlen} ###"
                )
                for method in methods:
                    t = times.get(method, float("nan"))
                    speeds[method] = efficiency(
                        flops(batch, seqlen, headdim, nheads, causal), t
                    )
                    if math.isnan(t):
                        print(f"{method} fwd: (skipped)")
                    else:
                        print(
                            f"{method} fwd: {speeds[method]:.2f} TFLOPs/s, {t * 1e3:.3f} ms"
                        )
                if math.isnan(times.get("FA4-CuTe-FP8", float("nan"))):
                    print("FA4-CuTe-FP8 status: FAILED")

    if fp8_failures:
        print(f"\nFP8 failures: {len(fp8_failures)} (showing first 5)")
        for causal, headdim, batch, seqlen, err in fp8_failures[:5]:
            print(
                f"- causal={causal} headdim={headdim} batch={batch} seqlen={seqlen}: {err}"
            )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
