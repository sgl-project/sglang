"""Speculative decoding parameter auto-tuner for sglang auto_tune.

Benchmarks different speculative decoding parameter combinations
and selects the optimal settings for different batch sizes.

Tunable parameters:
- speculative_num_steps / num_draft_tokens
- speculative_eagle_topk
- speculative_accept_threshold_single / acc
- speculative_algorithm (if multiple supported)

Requires GPU and a model compatible with spec decoding.

Part of #13363 item 7.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SpecConfig:
    """A speculative decoding configuration to benchmark."""

    num_draft_tokens: int
    eagle_topk: Optional[int] = None
    accept_threshold_single: float = 1.0
    accept_threshold_acc: float = 1.0
    algorithm: str = "EAGLE"

    def label(self) -> str:
        parts = [f"draft={self.num_draft_tokens}"]
        if self.eagle_topk is not None:
            parts.append(f"topk={self.eagle_topk}")
        if self.accept_threshold_single < 1.0:
            parts.append(f"thr={self.accept_threshold_single}")
        parts.append(f"alg={self.algorithm}")
        return "_".join(parts)

    def to_cli_args(self) -> List[str]:
        args = [
            f"--speculative-num-draft-tokens={self.num_draft_tokens}",
            f"--speculative-algorithm={self.algorithm}",
        ]
        if self.eagle_topk is not None:
            args.append(f"--speculative-eagle-topk={self.eagle_topk}")
        if self.accept_threshold_single < 1.0:
            args.append(f"--speculative-accept-threshold-single={self.accept_threshold_single}")
        if self.accept_threshold_acc < 1.0:
            args.append(f"--speculative-accept-threshold-acc={self.accept_threshold_acc}")
        return args


@dataclasses.dataclass
class SpecResult:
    """Benchmark result for a single speculative config."""

    config: SpecConfig
    batch_size: int
    output_throughput: float  # tokens/s
    tpot_mean: float  # ms
    ttft_mean: float  # ms
    time_s: float  # total benchmark time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "num_draft_tokens": self.config.num_draft_tokens,
            "eagle_topk": self.config.eagle_topk,
            "accept_threshold_single": self.config.accept_threshold_single,
            "accept_threshold_acc": self.config.accept_threshold_acc,
            "algorithm": self.config.algorithm,
            "output_throughput": self.output_throughput,
            "tpot_mean": self.tpot_mean,
            "ttft_mean": self.ttft_mean,
            "time_s": self.time_s,
        }


def get_search_space(
    model_config: Dict[str, Any],
    batch_sizes: Optional[List[int]] = None,
) -> Tuple[List[SpecConfig], List[int]]:
    """Generate the search space for spec decoding parameters.

    Returns (configs, batch_sizes).
    """
    architecture = model_config.get("architecture", "")

    if batch_sizes is None:
        batch_sizes = [1, 4, 16, 64, 256]

    # Draft token counts to try
    num_draft_tokens_list = [1, 2, 4, 8, 16, 32]

    configs = []

    # Standard Eagle
    for nd in num_draft_tokens_list:
        configs.append(SpecConfig(num_draft_tokens=nd, algorithm="EAGLE"))

    # Eagle with topk (tree drafting)
    for nd in [4, 8, 16]:
        for tk in [2, 4]:
            configs.append(SpecConfig(num_draft_tokens=nd, eagle_topk=tk, algorithm="EAGLE"))

    # With acceptance thresholds
    for nd in [4, 8]:
        for thr in [0.5, 0.75]:
            configs.append(SpecConfig(
                num_draft_tokens=nd,
                accept_threshold_single=thr,
                algorithm="EAGLE",
            ))

    # NGRAM for non-Eagle models
    for nd in [4, 8, 16]:
        configs.append(SpecConfig(num_draft_tokens=nd, algorithm="NGRAM"))

    return configs, batch_sizes


def _parse_benchmark_output(stdout: str) -> Dict[str, float]:
    """Parse sglang benchmark serving output to extract metrics."""
    result = {
        "output_throughput": 0.0,
        "tpot_mean": 0.0,
        "ttft_mean": 0.0,
    }

    for line in stdout.split("\n"):
        line = line.strip()
        if "Output throughput" in line:
            try:
                result["output_throughput"] = float(line.split(":")[-1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        elif "TPOT" in line and "Mean" in line:
            try:
                result["tpot_mean"] = float(line.split(":")[-1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        elif "TTFT" in line and "Mean" in line:
            try:
                result["ttft_mean"] = float(line.split(":")[-1].strip().split()[0])
            except (ValueError, IndexError):
                pass

    return result


def _benchmark_spec_config(
    model_path: str,
    draft_model_path: str,
    config: SpecConfig,
    batch_size: int,
    tp_size: int = 1,
    port: int = 30000,
    num_prompts: int = 512,
    warmup_seconds: int = 30,
    benchmark_seconds: int = 60,
    verbose: bool = False,
) -> Optional[SpecResult]:
    """Launch sglang server + benchmark for a single spec config.

    Returns None if the benchmark fails.
    """
    import requests

    log_dir = tempfile.mkdtemp(prefix="sglang_spec_tune_")

    # Launch server
    server_cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--draft-model-path", draft_model_path,
        "--tp", str(tp_size),
        "--port", str(port),
        "--log-level", "error",
    ] + config.to_cli_args()

    if verbose:
        print(f"    Starting server: {' '.join(server_cmd)}")

    server_proc = subprocess.Popen(
        server_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        # Wait for server to be ready
        ready = False
        for _ in range(300):
            try:
                r = requests.get(f"http://localhost:{port}/health", timeout=2)
                if r.status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(2)

        if not ready:
            logger.warning("Server failed to start for config %s", config.label())
            return None

        # Run benchmark
        bench_cmd = [
            sys.executable, "-m", "sglang.benchmark.serving",
            "--backend", "sglang",
            "--host", "localhost",
            "--port", str(port),
            "--num-prompts", str(num_prompts),
            "--request-rate", str(batch_size),
            "--warmup-seconds", str(warmup_seconds),
            "--benchmark-seconds", str(benchmark_seconds),
        ]

        if verbose:
            print(f"    Running benchmark: {' '.join(bench_cmd)}")

        bench_start = time.perf_counter()
        bench_proc = subprocess.run(
            bench_cmd, capture_output=True, text=True, timeout=benchmark_seconds + 120,
        )
        bench_end = time.perf_counter()

        metrics = _parse_benchmark_output(bench_proc.stdout + bench_proc.stderr)

        return SpecResult(
            config=config,
            batch_size=batch_size,
            output_throughput=metrics["output_throughput"],
            tpot_mean=metrics["tpot_mean"],
            ttft_mean=metrics["ttft_mean"],
            time_s=bench_end - bench_start,
        )

    except Exception as e:
        logger.warning("Benchmark failed for %s: %s", config.label(), e)
        return None
    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        # Cleanup
        try:
            import shutil
            shutil.rmtree(log_dir, ignore_errors=True)
        except Exception:
            pass


def tune_spec_decoding(
    model_path: str,
    draft_model_path: str,
    model_config: Dict[str, Any],
    tp_size: int = 1,
    batch_sizes: Optional[List[int]] = None,
    port: int = 30000,
    num_prompts: int = 512,
    verbose: bool = True,
) -> Dict[int, SpecResult]:
    """Tune speculative decoding parameters for different batch sizes.

    Returns a dict mapping batch_size -> best SpecResult.
    """
    import torch

    if not torch.cuda.is_available():
        logger.warning("CUDA not available; skipping spec decoding tuning")
        return {}

    if not draft_model_path:
        logger.warning("No draft model path provided; skipping spec decoding tuning")
        return {}

    configs, batch_sizes = get_search_space(model_config, batch_sizes)

    total_start = time.perf_counter()
    best_per_batch: Dict[int, SpecResult] = {}

    for bs in batch_sizes:
        if verbose:
            print(f"\n  Batch size: {bs}")

        best_result = None
        best_throughput = 0.0

        for idx, config in enumerate(configs):
            if verbose:
                print(f"    [{idx + 1}/{len(configs)}] {config.label()}...", end=" ")

            result = _benchmark_spec_config(
                model_path, draft_model_path, config, bs,
                tp_size=tp_size, port=port, num_prompts=num_prompts,
                verbose=False,
            )

            if result is not None:
                if verbose:
                    print(f"{result.output_throughput:.1f} tok/s")
                if result.output_throughput > best_throughput:
                    best_throughput = result.output_throughput
                    best_result = result
            else:
                if verbose:
                    print("FAILED")

        if best_result is not None:
            best_per_batch[bs] = best_result
            if verbose:
                print(f"    Best: {best_result.config.label()} "
                      f"({best_result.output_throughput:.1f} tok/s)")
        else:
            if verbose:
                print(f"    No valid config found for batch_size={bs}")

    total_end = time.perf_counter()
    if verbose:
        print(f"\n  Spec decoding tuning completed in {total_end - total_start:.2f}s")
        print(f"  Batch sizes tuned: {len(best_per_batch)}/{len(batch_sizes)}")

    return best_per_batch


def run_spec_tuning(
    model_path: str,
    draft_model_path: str,
    model_config: Dict[str, Any],
    tp_size: int = 1,
    batch_sizes: Optional[List[int]] = None,
    output_dir: Optional[str] = None,
    port: int = 30000,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Entry point for spec decoding auto-tuning.

    Returns a dict with best config per batch size.
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print("Speculative Decoding Parameter Tuning")
        print(f"{'=' * 60}")
        print(f"  Draft model: {draft_model_path}")
        print(f"  TP: {tp_size}")

    results = tune_spec_decoding(
        model_path, draft_model_path, model_config,
        tp_size=tp_size, batch_sizes=batch_sizes,
        port=port, verbose=verbose,
    )

    if not results:
        if verbose:
            print("  No results. Skipping.")
        return {}

    output = {
        "tp_size": tp_size,
        "draft_model_path": draft_model_path,
        "best_configs": {str(bs): r.to_dict() for bs, r in results.items()},
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "spec_configs.json")
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        if verbose:
            print(f"  Saved to: {path}")

    return output


def print_spec_summary(results: Dict[str, Any]) -> None:
    """Print a human-readable summary of spec decoding tuning results."""
    configs = results.get("best_configs", {})
    if not configs:
        print("  No spec decoding configs to summarize.")
        return

    print(f"\n{'=' * 60}")
    print("Speculative Decoding Tuning Summary")
    print(f"{'=' * 60}")
    for bs_str, cfg in sorted(configs.items(), key=lambda x: int(x[0])):
        print(
            f"  batch={bs_str:>5s}  "
            f"draft={cfg['num_draft_tokens']:>2d}  "
            f"topk={cfg.get('eagle_topk', '-')}  "
            f"thr={cfg.get('accept_threshold_single', 1.0):.2f}  "
            f"alg={cfg['algorithm']:>6s}  "
            f"{cfg['output_throughput']:>8.1f} tok/s"
        )
    print(f"{'=' * 60}")