"""Benchmark TileLang FP8 GEMM against Triton and optional DeepGEMM baselines."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable, List

import torch
import triton

from sglang.srt.layers.tilelang_gemm_wrapper.configs import (
    AUTOTUNE_SEARCH_POLICIES,
    DEFAULT_M_VALUES,
    KERNEL_TYPES,
    write_selected_config_file,
)
from sglang.srt.layers.tilelang_gemm_wrapper.tuning import (
    concrete_shapes,
    load_selected_config_store,
    make_autotune_metadata,
)

logger = logging.getLogger(__name__)


def _tflops(M: int, N: int, K: int, latency_ms: float) -> float:
    return 2.0 * M * N * K / latency_ms / 1e9


def _do_bench(fn, rep: int, backend: str) -> float:
    quantiles = [0.5, 0.2, 0.8]
    if backend == "cudagraph":
        # Triton's CUDA graph benchmark captures/replays on a side stream.
        # Make sure any input preparation or warmup launch on the current stream
        # has completed before the side stream starts reading/writing tensors.
        torch.cuda.synchronize()
        result = triton.testing.do_bench_cudagraph(fn, rep=rep, quantiles=quantiles)
    elif backend == "event":
        result = triton.testing.do_bench(fn, rep=rep, quantiles=quantiles)
    else:
        raise ValueError(f"Unsupported benchmark backend: {backend}")

    if isinstance(result, (list, tuple)):
        return float(result[0])
    return float(result)


def _per_block_cast_to_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    M, N = x.shape
    padded_m = triton.cdiv(M, 128) * 128
    padded_n = triton.cdiv(N, 128) * 128
    x_padded = torch.zeros((padded_m, padded_n), dtype=x.dtype, device=x.device)
    x_padded[:M, :N] = x
    x_view = x_padded.view(padded_m // 128, 128, padded_n // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:M, :N].contiguous(), (x_amax / 448.0).view(
        padded_m // 128, padded_n // 128
    )


def _prepare_data(M: int, N: int, K: int):
    from sglang.srt.layers.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    A_fp8, A_scale = sglang_per_token_group_quant_fp8(
        A.contiguous(), group_size=128, column_major_scales=False
    )
    B_fp8, B_scale = _per_block_cast_to_fp8(B.contiguous())
    return A_fp8, A_scale, B_fp8, B_scale


def _prepare_deepgemm_a_scale(A_scale: torch.Tensor) -> torch.Tensor:
    from sglang.srt.layers import deep_gemm_wrapper

    if not deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
        raise RuntimeError(
            "DeepGEMM comparison requested, but DeepGEMM is not available. "
            "Install deep_gemm or set SGLANG_ENABLE_JIT_DEEPGEMM=1."
        )

    if deep_gemm_wrapper.DEEPGEMM_NEED_TMA_ALIGNED_SCALES:
        from deep_gemm.utils.layout import get_mn_major_tma_aligned_tensor

        return get_mn_major_tma_aligned_tensor(A_scale.clone())
    return A_scale


def _benchmark_one(
    M: int,
    N: int,
    K: int,
    rep: int,
    bench_backend: str,
    skip_baseline: bool,
    compare_deepgemm: bool,
    autotune: bool,
    autotune_backend: str,
    autotune_warmup: int,
    autotune_rep: int,
    autotune_max_configs: int | None,
    kernel_types: list[str] | None,
    autotune_policy: str,
    checkpoint_config_path: str | None,
) -> dict:
    from sglang.srt.layers import tilelang_gemm_wrapper

    if autotune:
        if tilelang_gemm_wrapper.has_selected_config(M, N, K):
            logger.info(
                "Skipping TileLang FP8 GEMM autotune for M=%s, N=%s, K=%s; "
                "selected config already exists.",
                M,
                N,
                K,
            )
        else:
            tilelang_gemm_wrapper.autotune_shape(
                M,
                N,
                K,
                warmup=autotune_warmup,
                rep=autotune_rep,
                backend=autotune_backend,
                max_configs=autotune_max_configs,
                kernel_types=kernel_types,
                search_policy=autotune_policy,
            )
        if checkpoint_config_path:
            tilelang_gemm_wrapper.export_selected_configs(
                checkpoint_config_path,
                metadata=make_autotune_metadata(
                    autotune_backend,
                    autotune_policy,
                    autotune_warmup,
                    autotune_rep,
                    autotune_max_configs,
                    kernel_types,
                ),
            )

    A_fp8, A_scale, B_fp8, B_scale = _prepare_data(M, N, K)
    C_tl = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")

    tilelang_gemm_wrapper.gemm_nt_f8f8bf16((A_fp8, A_scale), (B_fp8, B_scale), C_tl)

    def tilelang_run():
        tilelang_gemm_wrapper.gemm_nt_f8f8bf16((A_fp8, A_scale), (B_fp8, B_scale), C_tl)

    tl_ms = _do_bench(tilelang_run, rep=rep, backend=bench_backend)
    kernel_info = tilelang_gemm_wrapper.get_kernel_info(M, N, K)

    result = {
        "M": M,
        "N": N,
        "K": K,
        "tilelang_ms": tl_ms,
        "tilelang_tflops": _tflops(M, N, K, tl_ms),
        "kernel_type": kernel_info["kernel_type"],
        "baseline_ms": float("nan"),
        "baseline_tflops": float("nan"),
        "speedup": float("nan"),
        "allclose": "",
        "max_diff": float("nan"),
        "deepgemm_ms": float("nan"),
        "deepgemm_tflops": float("nan"),
        "tilelang_deepgemm_speedup": float("nan"),
        "deepgemm_allclose": "",
        "deepgemm_max_diff": float("nan"),
    }
    C_deepgemm: torch.Tensor | None = None

    if compare_deepgemm:
        from sglang.srt.layers import deep_gemm_wrapper

        A_scale_deepgemm = _prepare_deepgemm_a_scale(A_scale)
        C_deepgemm = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
        deep_gemm_wrapper.gemm_nt_f8f8bf16(
            (A_fp8, A_scale_deepgemm), (B_fp8, B_scale), C_deepgemm
        )

        def deepgemm_run():
            deep_gemm_wrapper.gemm_nt_f8f8bf16(
                (A_fp8, A_scale_deepgemm), (B_fp8, B_scale), C_deepgemm
            )

        deepgemm_ms = _do_bench(deepgemm_run, rep=rep, backend=bench_backend)
        result.update(
            {
                "deepgemm_ms": deepgemm_ms,
                "deepgemm_tflops": _tflops(M, N, K, deepgemm_ms),
                "tilelang_deepgemm_speedup": deepgemm_ms / tl_ms,
            }
        )

    if skip_baseline:
        return result

    from sglang.srt.layers.quantization.fp8_kernel import w8a8_block_fp8_matmul_triton

    C_ref = w8a8_block_fp8_matmul_triton(
        A_fp8, B_fp8, A_scale, B_scale, [128, 128], output_dtype=torch.bfloat16
    )

    def triton_run():
        return w8a8_block_fp8_matmul_triton(
            A_fp8, B_fp8, A_scale, B_scale, [128, 128], output_dtype=torch.bfloat16
        )

    baseline_ms = _do_bench(triton_run, rep=rep, backend=bench_backend)
    max_diff = (C_tl - C_ref).abs().max().item()
    deepgemm_max_diff = (
        (C_deepgemm - C_ref).abs().max().item()
        if C_deepgemm is not None
        else float("nan")
    )
    result.update(
        {
            "baseline_ms": baseline_ms,
            "baseline_tflops": _tflops(M, N, K, baseline_ms),
            "speedup": baseline_ms / tl_ms,
            "allclose": torch.allclose(C_tl, C_ref, rtol=1e-2, atol=1e-2),
            "max_diff": max_diff,
            "deepgemm_allclose": (
                torch.allclose(C_deepgemm, C_ref, rtol=1e-2, atol=1e-2)
                if C_deepgemm is not None
                else ""
            ),
            "deepgemm_max_diff": deepgemm_max_diff,
        }
    )
    return result


def _write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_shape_values(values: Iterable[str]) -> list[tuple[int, int]]:
    shapes = []
    for value in values:
        try:
            N, K = (int(part) for part in value.split(",", 1))
        except Exception as err:
            raise ValueError(f"Expected N,K shape, got {value}") from err
        shapes.append((N, K))
    return shapes


def _parse_gpus(value: str) -> list[str]:
    gpus = [item.strip() for item in value.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    return gpus


def _shape_label(M: int, N: int, K: int) -> str:
    return f"M{M}_N{N}_K{K}"


def _tail(path: Path, max_chars: int = 4000) -> str:
    try:
        text = path.read_text(errors="replace")
    except FileNotFoundError:
        return ""
    return text[-max_chars:]


def _parallel_autotune(
    args: argparse.Namespace, shapes: list[tuple[int, int, int]]
) -> str:
    gpus = _parse_gpus(args.gpus)
    work_dir = Path(
        tempfile.mkdtemp(
            prefix="tilelang-autotune-", dir=os.environ.get("TMPDIR", "/tmp")
        )
    )
    checkpoint_path = (
        args.checkpoint_config_path
        or args.export_config_path
        or str(work_dir / "selected_configs.json")
    )
    metadata = make_autotune_metadata(
        args.autotune_backend,
        args.autotune_policy,
        args.autotune_warmup,
        args.autotune_rep,
        args.autotune_max_configs,
        args.kernel_type,
    )

    store = load_selected_config_store(
        (args.config_path, args.resume_config_path, args.checkpoint_config_path)
    )
    pending = [
        shape
        for shape in shapes
        if store.get_exact_compatible(shape[0], shape[1], shape[2]) is None
    ]
    if store.as_list():
        write_selected_config_file(checkpoint_path, store.as_list(), metadata=metadata)

    logger.info(
        "Parallel TileLang autotune: %s pending shapes, %s already available, GPUs=%s",
        len(pending),
        len(shapes) - len(pending),
        ",".join(gpus),
    )

    script = Path(__file__).resolve()
    free_gpus = list(gpus)
    running = []
    failures = []

    def launch(gpu: str, shape: tuple[int, int, int]) -> None:
        M, N, K = shape
        label = _shape_label(M, N, K)
        child_config_path = work_dir / f"{label}.json"
        log_path = work_dir / f"{label}.log"
        cmd = [
            sys.executable,
            str(script),
            "--shape",
            f"{N},{K}",
            "--m-values",
            str(M),
            "--autotune",
            "--autotune-only",
            "--autotune-backend",
            args.autotune_backend,
            "--autotune-policy",
            args.autotune_policy,
            "--autotune-warmup",
            str(args.autotune_warmup),
            "--autotune-rep",
            str(args.autotune_rep),
            "--export-config-path",
            str(child_config_path),
        ]
        if args.autotune_max_configs is not None:
            cmd.extend(["--autotune-max-configs", str(args.autotune_max_configs)])
        for kernel_type in args.kernel_type or []:
            cmd.extend(["--kernel-type", kernel_type])
        if args.verbose:
            cmd.append("--verbose")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        log_file = log_path.open("w")
        proc = subprocess.Popen(
            cmd,
            cwd=os.getcwd(),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        running.append(
            {
                "proc": proc,
                "shape": shape,
                "gpu": gpu,
                "config_path": child_config_path,
                "log_path": log_path,
                "log_file": log_file,
            }
        )
        logger.info("Launched TileLang autotune %s on GPU %s", label, gpu)

    while pending or running:
        while pending and free_gpus:
            launch(free_gpus.pop(0), pending.pop(0))

        time.sleep(1.0)
        for item in list(running):
            proc = item["proc"]
            return_code = proc.poll()
            if return_code is None:
                continue

            running.remove(item)
            item["log_file"].close()
            free_gpus.append(item["gpu"])
            M, N, K = item["shape"]
            label = _shape_label(M, N, K)
            if return_code != 0:
                failures.append((label, item["log_path"], return_code))
                logger.error(
                    "TileLang autotune %s failed with return code %s. Log: %s",
                    label,
                    return_code,
                    item["log_path"],
                )
                continue

            store.update(
                load_selected_config_store(
                    (str(item["config_path"]),), skip_missing=False
                )
            )
            write_selected_config_file(
                checkpoint_path, store.as_list(), metadata=metadata
            )
            logger.info(
                "TileLang autotune %s finished; checkpoint now has %s configs at %s",
                label,
                len(store.as_list()),
                checkpoint_path,
            )

    if failures:
        messages = []
        for label, log_path, return_code in failures:
            messages.append(
                f"{label} failed with return code {return_code}. Log tail:\n"
                f"{_tail(log_path)}"
            )
        raise RuntimeError("\n\n".join(messages))

    if args.export_config_path and args.export_config_path != checkpoint_path:
        write_selected_config_file(
            args.export_config_path, store.as_list(), metadata=metadata
        )
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", action="append", required=True, help="N,K shape")
    parser.add_argument(
        "--m-values",
        type=int,
        nargs="+",
        default=DEFAULT_M_VALUES,
    )
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--config-path", help="Selected-config JSON file or directory")
    parser.add_argument(
        "--export-config-path", help="Export selected configs after benchmark"
    )
    parser.add_argument(
        "--checkpoint-config-path",
        help="Incrementally write selected configs after each tuned shape",
    )
    parser.add_argument(
        "--resume-config-path",
        help="Load selected configs and skip exact shapes that are already tuned",
    )
    parser.add_argument("--output", "-o", help="CSV output path")
    parser.add_argument(
        "--bench-backend",
        default="cudagraph",
        choices=("event", "cudagraph"),
        help="Timing backend for final benchmark measurements",
    )
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument(
        "--compare-deepgemm",
        action="store_true",
        help=(
            "Also benchmark DeepGEMM on each shape and compare its output with "
            "the Triton baseline. This requires the deep_gemm package and a "
            "supported GPU."
        ),
    )
    parser.add_argument(
        "--autotune",
        action="store_true",
        help="Tune candidate configs before measuring each shape",
    )
    parser.add_argument(
        "--autotune-backend",
        default="cudagraph",
        choices=("event", "cupti", "cudagraph"),
        help="TileLang profiler backend to use while autotuning",
    )
    parser.add_argument(
        "--autotune-policy",
        default="family_pruned",
        choices=AUTOTUNE_SEARCH_POLICIES,
        help="Candidate search policy to use while autotuning",
    )
    parser.add_argument("--autotune-warmup", type=int, default=25)
    parser.add_argument("--autotune-rep", type=int, default=100)
    parser.add_argument("--autotune-max-configs", type=int)
    parser.add_argument(
        "--autotune-only",
        action="store_true",
        help="Tune/export selected configs without running final benchmarks",
    )
    parser.add_argument(
        "--gpus",
        help=(
            "Comma-separated GPU ids for parallel autotune. Each concrete "
            "(M,N,K) shape is tuned in a separate worker process."
        ),
    )
    parser.add_argument(
        "--kernel-type",
        action="append",
        choices=KERNEL_TYPES,
        help="Restrict autotuning to one or more kernel families",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if args.compare_deepgemm:
        # The benchmark only needs the concrete shapes under test. Avoid
        # DeepGEMM's server warmup path, which can precompile a very large M list.
        os.environ.setdefault("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "0")

    from sglang.srt.layers import tilelang_gemm_wrapper

    if args.config_path:
        tilelang_gemm_wrapper.load_selected_configs(args.config_path)
    if args.resume_config_path:
        tilelang_gemm_wrapper.merge_selected_configs(args.resume_config_path)
    if args.checkpoint_config_path and os.path.exists(args.checkpoint_config_path):
        tilelang_gemm_wrapper.merge_selected_configs(args.checkpoint_config_path)

    nk_shapes = _parse_shape_values(args.shape)
    shapes = concrete_shapes(nk_shapes, args.m_values)

    if args.gpus and args.autotune:
        config_path = _parallel_autotune(args, shapes)
        if args.autotune_only:
            return
        tilelang_gemm_wrapper.load_selected_configs(config_path)
        args.autotune = False

    if args.autotune_only:
        tilelang_gemm_wrapper.autotune_shapes(
            shapes,
            warmup=args.autotune_warmup,
            rep=args.autotune_rep,
            backend=args.autotune_backend,
            kernel_types=args.kernel_type,
            max_configs=args.autotune_max_configs,
            search_policy=args.autotune_policy,
            checkpoint_config_path=args.checkpoint_config_path,
            export_metadata=make_autotune_metadata(
                args.autotune_backend,
                args.autotune_policy,
                args.autotune_warmup,
                args.autotune_rep,
                args.autotune_max_configs,
                args.kernel_type,
            ),
        )
        if args.export_config_path:
            tilelang_gemm_wrapper.export_selected_configs(
                args.export_config_path,
                metadata=make_autotune_metadata(
                    args.autotune_backend,
                    args.autotune_policy,
                    args.autotune_warmup,
                    args.autotune_rep,
                    args.autotune_max_configs,
                    args.kernel_type,
                ),
            )
        return

    rows = []
    for N, K in nk_shapes:
        for M in args.m_values:
            logger.info("Benchmarking M=%s, N=%s, K=%s", M, N, K)
            row = _benchmark_one(
                M,
                N,
                K,
                args.rep,
                args.bench_backend,
                args.skip_baseline,
                args.compare_deepgemm,
                args.autotune,
                args.autotune_backend,
                args.autotune_warmup,
                args.autotune_rep,
                args.autotune_max_configs,
                args.kernel_type,
                args.autotune_policy,
                args.checkpoint_config_path,
            )
            rows.append(row)
            logger.info("%s", row)

    if args.output:
        _write_csv(args.output, rows)

    if args.export_config_path:
        tilelang_gemm_wrapper.export_selected_configs(
            args.export_config_path,
            metadata=make_autotune_metadata(
                args.autotune_backend,
                args.autotune_policy,
                args.autotune_warmup,
                args.autotune_rep,
                args.autotune_max_configs,
                args.kernel_type,
            ),
        )


if __name__ == "__main__":
    main()
