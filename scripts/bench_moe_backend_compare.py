#!/usr/bin/env python3
"""
Compare FlashInfer MoE backends on small quality + throughput checks.

Example:
python3 scripts/bench_moe_backend_compare.py \
  --model nvidia/DeepSeek-R1-0528-FP4-V2 \
  --backends flashinfer_cutlass flashinfer_trtllm flashinfer_cutedsl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)


def _safe_tensor_head(x, n: int):
    """Return first n flattened values for a tensor-like object."""
    if x is None:
        return None
    try:
        import torch

        if not isinstance(x, torch.Tensor):
            return None
        return x.detach().to(torch.float32).reshape(-1)[:n].cpu().tolist()
    except Exception:
        return None


def _safe_list_head(x, n: int):
    """Return first n values for list-like/tensor-like data."""
    if x is None:
        return None
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().to(torch.float32).reshape(-1)[:n].cpu().tolist()
    except Exception:
        pass
    if isinstance(x, list):
        if len(x) > 0 and isinstance(x[0], list):
            return x[0][:n]
        return x[:n]
    return None


def _fmt_head_line(label: str, values):
    if values is None:
        return None
    return f"  {label}={values}"


def _load_cutedsl_diag_snapshot(diag_path: str, head_k: int):
    """Load compact debug snapshot from CUTEDSL_DIAG dump."""
    import torch

    if not os.path.exists(diag_path):
        return None, f"diag file not found: {diag_path}"

    try:
        d = torch.load(diag_path, map_location="cpu")
    except Exception as e:
        return None, f"failed to load diag file {diag_path}: {e}"

    scales = d.get("scales", {}) or {}
    isolate_meta = d.get("isolate_meta", {}) or {}
    map_swap = isolate_meta.get("manual_finalize_map_swap_baseline")
    map_swap_line = None
    if isinstance(map_swap, dict):
        raw = map_swap.get("raw_inverse_consistency", {}) or {}
        rebuilt = map_swap.get("rebuilt_inverse_consistency", {}) or {}
        raw_diff = map_swap.get("raw_vs_rebuilt_output_diff", {}) or {}
        map_swap_line = (
            "  manual_finalize_map_swap_baseline: "
            f"raw_inverse_mismatch={raw.get('inverse_mismatch_count')}/{raw.get('valid_pairs')}, "
            f"rebuilt_inverse_mismatch={rebuilt.get('inverse_mismatch_count')}/{rebuilt.get('valid_pairs')}, "
            f"raw_vs_rebuilt_mean_abs={raw_diff.get('mean_abs_diff')}, "
            f"raw_vs_rebuilt_max_abs={raw_diff.get('max_abs_diff')}"
        )

    snapshot = {
        "cutedsl_vs_cutlass": d.get("cutedsl_vs_cutlass"),
        "cutedsl_vs_trtllm": d.get("cutedsl_vs_trtllm"),
        "cutedsl_output_head": _safe_tensor_head(d.get("cutedsl_output"), head_k),
        "cutlass_output_head": _safe_tensor_head(d.get("cutlass_output"), head_k),
        "trtllm_output_head": _safe_tensor_head(d.get("trtllm_output"), head_k),
        "topk_ids_head": _safe_list_head(d.get("topk_ids"), head_k),
        "topk_weights_head": _safe_list_head(d.get("topk_weights"), head_k),
        "w1_alpha_cutedsl_head": _safe_tensor_head(
            scales.get("w1_alpha_cutedsl"), head_k
        ),
        "g1_alphas_head": _safe_tensor_head(scales.get("g1_alphas"), head_k),
        "w2_alpha_cutedsl_head": _safe_tensor_head(
            scales.get("w2_alpha_cutedsl"), head_k
        ),
        "g2_alphas_head": _safe_tensor_head(scales.get("g2_alphas"), head_k),
        "fc2_input_scale_head": _safe_tensor_head(
            scales.get("fc2_input_scale"), head_k
        ),
        "w2_input_scale_quant_head": _safe_tensor_head(
            scales.get("w2_input_scale_quant"), head_k
        ),
        "w13_input_scale_quant_head": _safe_tensor_head(
            scales.get("w13_input_scale_quant"), head_k
        ),
        "used_input_scale_head": _safe_tensor_head(
            scales.get("used_input_scale"), head_k
        ),
        "manual_finalize_map_swap_baseline": map_swap,
    }

    lines = [
        f"[CUTEDSL_DIAG_BRIEF] from {diag_path}",
        _fmt_head_line("topk_ids[0][:k]", snapshot["topk_ids_head"]),
        _fmt_head_line("topk_weights[0][:k]", snapshot["topk_weights_head"]),
        _fmt_head_line("CuteDSL output[:k]", snapshot["cutedsl_output_head"]),
        _fmt_head_line("CUTLASS output[:k]", snapshot["cutlass_output_head"]),
        _fmt_head_line("TRTLLM output[:k]", snapshot["trtllm_output_head"]),
        _fmt_head_line("CuteDSL w1_alpha[:k]", snapshot["w1_alpha_cutedsl_head"]),
        _fmt_head_line("CUTLASS g1_alphas[:k]", snapshot["g1_alphas_head"]),
        _fmt_head_line("CuteDSL w2_alpha[:k]", snapshot["w2_alpha_cutedsl_head"]),
        _fmt_head_line("CUTLASS g2_alphas[:k]", snapshot["g2_alphas_head"]),
        _fmt_head_line("fc2_input_scale[:k]", snapshot["fc2_input_scale_head"]),
        _fmt_head_line(
            "w2_input_scale_quant[:k]", snapshot["w2_input_scale_quant_head"]
        ),
        _fmt_head_line(
            "w13_input_scale_quant[:k]", snapshot["w13_input_scale_quant_head"]
        ),
        _fmt_head_line("used_input_scale[:k]", snapshot["used_input_scale_head"]),
    ]
    if map_swap_line is not None:
        lines.append(map_swap_line)
    if snapshot["cutedsl_vs_cutlass"] is not None:
        lines.append(f"  cutedsl_vs_cutlass={snapshot['cutedsl_vs_cutlass']}")
    if snapshot["cutedsl_vs_trtllm"] is not None:
        lines.append(f"  cutedsl_vs_trtllm={snapshot['cutedsl_vs_trtllm']}")
    lines = [x for x in lines if x is not None]

    return (snapshot, lines), None


def run_small_mmlu(base_url: str, model: str, num_examples: int, num_threads: int):
    from sglang.test.run_eval import run_eval_once
    from sglang.test.simple_eval_mmlu import MMLUEval

    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "EMPTY"

    filename = "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv"
    eval_obj = MMLUEval(filename, num_examples, num_threads)
    args = SimpleNamespace(
        base_url=base_url,
        model=model,
    )
    base_url_v1 = f"{base_url}/v1"
    result, latency, sampler = run_eval_once(args, base_url_v1, eval_obj)
    return {"score": result.score, "latency": latency, **result.metrics}


def run_small_bench_serving(base_url: str, num_prompts: int, max_concurrency: int):
    with tempfile.NamedTemporaryFile(
        prefix="moe_backend_bench_", suffix=".jsonl", delete=False
    ) as tf:
        output_file = tf.name
    cmd = [
        sys.executable,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
        "--base-url",
        base_url,
        "--dataset-name",
        "random",
        "--num-prompts",
        str(num_prompts),
        "--random-input-len",
        "128",
        "--random-output-len",
        "64",
        "--random-range-ratio",
        "0.0",
        "--max-concurrency",
        str(max_concurrency),
        "--disable-tqdm",
        "--output-file",
        output_file,
    ]
    try:
        subprocess.run(cmd, check=True)
        with open(output_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            raise RuntimeError("bench_serving produced no JSON output")
        return json.loads(lines[-1])
    finally:
        try:
            os.remove(output_file)
        except OSError:
            pass


def get_backend_args(
    backend: str,
    tp_size: int,
    ep_size: int,
    mem_fraction_static: float,
    attention_backend: str,
    sampling_backend: str,
    fp4_gemm_backend: str,
    watchdog_timeout: int | None,
    soft_watchdog_timeout: int | None,
    cutedsl_safe_launch: bool,
    cutedsl_enable_server_warmup: bool,
    cutedsl_enable_overlap_schedule: bool,
    cutedsl_enable_custom_all_reduce: bool,
    cutedsl_enable_radix_cache: bool,
):
    args = [
        "--trust-remote-code",
        "--tp-size",
        str(tp_size),
        "--ep-size",
        str(ep_size),
        "--mem-fraction-static",
        str(mem_fraction_static),
        "--moe-runner-backend",
        backend,
        "--quantization",
        "modelopt_fp4",
        "--attention-backend",
        attention_backend,
        "--sampling-backend",
        sampling_backend,
        "--fp4-gemm-backend",
        fp4_gemm_backend,
        "--disable-cuda-graph",
    ]
    if backend == "flashinfer_cutedsl":
        args += ["--moe-a2a-backend", "none"]
        if cutedsl_safe_launch:
            if not cutedsl_enable_server_warmup:
                args += ["--skip-server-warmup"]
            if not cutedsl_enable_overlap_schedule:
                args += ["--disable-overlap-schedule"]
            if not cutedsl_enable_custom_all_reduce:
                args += ["--disable-custom-all-reduce"]
            if not cutedsl_enable_radix_cache:
                args += ["--disable-radix-cache"]
    if watchdog_timeout is not None:
        args += ["--watchdog-timeout", str(watchdog_timeout)]
    if soft_watchdog_timeout is not None:
        args += ["--soft-watchdog-timeout", str(soft_watchdog_timeout)]
    return args


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark small MoE quality + throughput across backends."
    )
    parser.add_argument("--model", required=True, help="Model path/name.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:30000",
        help="Server URL for eval/bench.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=[
            "flashinfer_cutlass",
            "flashinfer_trtllm",
            "flashinfer_cutedsl",
        ],
        help="MoE runner backends to compare.",
    )
    parser.add_argument(
        "--tp-size", type=int, default=1, help="Tensor-parallel size for server launch."
    )
    parser.add_argument(
        "--ep-size", type=int, default=1, help="Expert-parallel size for server launch."
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=0.7,
        help="Static memory fraction for server launch.",
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        default="triton",
        help="Attention backend for server launch.",
    )
    parser.add_argument(
        "--sampling-backend",
        type=str,
        default="pytorch",
        help="Sampling backend for server launch.",
    )
    parser.add_argument(
        "--fp4-gemm-backend",
        type=str,
        default="auto",
        help="FP4 GEMM backend for server launch.",
    )
    parser.add_argument(
        "--launch-timeout",
        type=int,
        default=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        help="Server launch timeout (seconds).",
    )
    parser.add_argument(
        "--watchdog-timeout",
        type=int,
        default=1800,
        help="Scheduler hard watchdog timeout (seconds) for launched servers.",
    )
    parser.add_argument(
        "--soft-watchdog-timeout",
        type=int,
        default=1800,
        help="Scheduler soft watchdog timeout (seconds) for launched servers.",
    )
    parser.add_argument(
        "--cutedsl-safe-launch",
        action="store_true",
        help=(
            "Use conservative stability flags for flashinfer_cutedsl "
            "(skip warmup, disable overlap/custom-all-reduce/radix cache)."
        ),
    )
    parser.add_argument(
        "--cutedsl-enable-server-warmup",
        action="store_true",
        help="With --cutedsl-safe-launch, re-enable server warmup.",
    )
    parser.add_argument(
        "--cutedsl-enable-overlap-schedule",
        action="store_true",
        help="With --cutedsl-safe-launch, re-enable overlap schedule.",
    )
    parser.add_argument(
        "--cutedsl-enable-custom-all-reduce",
        action="store_true",
        help="With --cutedsl-safe-launch, re-enable custom all-reduce.",
    )
    parser.add_argument(
        "--cutedsl-enable-radix-cache",
        action="store_true",
        help="With --cutedsl-safe-launch, re-enable radix cache.",
    )
    parser.add_argument(
        "--num-examples", type=int, default=32, help="MMLU examples per backend."
    )
    parser.add_argument(
        "--num-threads", type=int, default=4, help="MMLU eval worker count."
    )
    parser.add_argument(
        "--num-prompts", type=int, default=48, help="bench_serving prompt count."
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=8, help="bench_serving max concurrency."
    )
    parser.add_argument(
        "--print-cutedsl-diag-brief",
        action="store_true",
        help=(
            "If enabled, load CUTEDSL_DIAG dump and print compact tensor/value heads "
            "for quick backend-contract debugging."
        ),
    )
    parser.add_argument(
        "--cutedsl-diag-path",
        type=str,
        default="/tmp/cutedsl_diag_rank0.pt",
        help="Path to CUTEDSL_DIAG dump file.",
    )
    parser.add_argument(
        "--cutedsl-diag-head-k",
        type=int,
        default=4,
        help="Number of values to print for each tensor head in diag brief.",
    )
    args = parser.parse_args()

    results = {}
    for backend in args.backends:
        print(f"\n=== Running backend: {backend} ===")
        if backend == "flashinfer_cutedsl" and args.print_cutedsl_diag_brief:
            # Avoid stale prints when backend run fails before producing a new dump.
            try:
                os.remove(args.cutedsl_diag_path)
            except OSError:
                pass
        process = popen_launch_server(
            model=args.model,
            base_url=args.base_url,
            timeout=args.launch_timeout,
            other_args=get_backend_args(
                backend=backend,
                tp_size=args.tp_size,
                ep_size=args.ep_size,
                mem_fraction_static=args.mem_fraction_static,
                attention_backend=args.attention_backend,
                sampling_backend=args.sampling_backend,
                fp4_gemm_backend=args.fp4_gemm_backend,
                watchdog_timeout=args.watchdog_timeout,
                soft_watchdog_timeout=args.soft_watchdog_timeout,
                cutedsl_safe_launch=args.cutedsl_safe_launch,
                cutedsl_enable_server_warmup=args.cutedsl_enable_server_warmup,
                cutedsl_enable_overlap_schedule=args.cutedsl_enable_overlap_schedule,
                cutedsl_enable_custom_all_reduce=args.cutedsl_enable_custom_all_reduce,
                cutedsl_enable_radix_cache=args.cutedsl_enable_radix_cache,
            ),
            env=dict(os.environ),
        )
        try:
            eval_metrics = run_small_mmlu(
                args.base_url, args.model, args.num_examples, args.num_threads
            )
            bench_metrics = run_small_bench_serving(
                args.base_url, args.num_prompts, args.max_concurrency
            )
            results[backend] = {
                "mmlu_score": float(eval_metrics["score"]),
                "output_throughput": float(bench_metrics["output_throughput"]),
                "mean_tpot_ms": float(bench_metrics["mean_tpot_ms"]),
                "mean_ttft_ms": float(bench_metrics["mean_ttft_ms"]),
            }
            if backend == "flashinfer_cutedsl" and args.print_cutedsl_diag_brief:
                diag_payload, diag_err = _load_cutedsl_diag_snapshot(
                    args.cutedsl_diag_path, args.cutedsl_diag_head_k
                )
                if diag_err is not None:
                    print(f"[CUTEDSL_DIAG_BRIEF] {diag_err}")
                else:
                    diag_snapshot, diag_lines = diag_payload
                    results[backend]["cutedsl_diag_brief"] = diag_snapshot
                    print("\n".join(diag_lines))
        finally:
            kill_process_tree(process.pid)

    print("\n=== Summary ===")
    print(
        f"{'backend':24s} {'mmlu':>8s} {'out_thr(tok/s)':>16s} {'mean_tpot(ms)':>14s} {'mean_ttft(ms)':>14s}"
    )
    for backend, metric in results.items():
        print(
            f"{backend:24s} {metric['mmlu_score']:8.4f} {metric['output_throughput']:16.2f} "
            f"{metric['mean_tpot_ms']:14.3f} {metric['mean_ttft_ms']:14.3f}"
        )

    if "flashinfer_cutedsl" in results and "flashinfer_cutlass" in results:
        delta = abs(
            results["flashinfer_cutedsl"]["mmlu_score"]
            - results["flashinfer_cutlass"]["mmlu_score"]
        )
        print(f"\ncutedsl vs cutlass MMLU delta: {delta:.4f}")
    if "flashinfer_cutedsl" in results and "flashinfer_trtllm" in results:
        delta = abs(
            results["flashinfer_cutedsl"]["mmlu_score"]
            - results["flashinfer_trtllm"]["mmlu_score"]
        )
        print(f"cutedsl vs trtllm MMLU delta: {delta:.4f}")

    print("\nJSON results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
