from __future__ import annotations

import argparse
import dataclasses
import os
import unittest
from pathlib import Path
from typing import ClassVar, Optional

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs
from sglang.test.bench_one_batch_server_internal import (
    BenchArgs,
    BenchOneCaseResult,
    run_benchmark_internal,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=600, stage="extra-a", runner_config="1-gpu-large")
register_cuda_ci(est_time=600, suite="nightly-1-gpu", nightly=True)


_QWEN3_MODEL = "Qwen/Qwen3-0.6B"
_QWEN3_SCENARIO_MODEL = "qwen3-0.6b"

# When set, every bench scenario captures torch profiles of the canary-on run only,
# under `${SGLANG_KV_CANARY_PROFILE_DIR}/<scenario>_on/{cuda_graph,no_cuda_graph_osl3}/`. In
# this mode the canary-off baseline run + overhead assertion are skipped. We capture two
# variants per scenario:
#   * `cuda_graph/`        — the production config (matches the assertion run).
#   * `no_cuda_graph_osl3/` — diagnostic-only: `--disable-cuda-graph` (NEVER use this in the
#     assertion path) and `output_len=3` so the trace shows raw kernel launches without graph
#     replay folding everything into a single node.
_PROFILE_DIR_ENV = "SGLANG_KV_CANARY_PROFILE_DIR"
_PROFILE_STEPS = 30
_PROFILE_NO_GRAPH_OUTPUT_LEN = 3


def _make_server_args(*, canary_on: bool, disable_cuda_graph: bool = False) -> ServerArgs:
    # canary requires --disable-piecewise-cuda-graph (install_canary asserts it). Pass it on both
    # sides for apples-to-apples. Do NOT pass --disable-cuda-graph in the assertion path: canary
    # must run inside the regular cuda graph for a representative measurement. The flag is only
    # honored in profile mode for the diagnostic no_cuda_graph_osl3 variant.
    extra = [
        "--model-path",
        _QWEN3_MODEL,
        "--disable-piecewise-cuda-graph",
    ]
    if disable_cuda_graph:
        extra.append("--disable-cuda-graph")
    if canary_on:
        extra += ["--kv-canary", "raise"]

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    namespace = parser.parse_args(extra)
    return ServerArgs.from_cli_args(namespace)


def _make_bench_args(*, batch_size: int, input_len: int, output_len: int) -> BenchArgs:
    return BenchArgs(
        run_name=f"canary_self_bench_bs{batch_size}_isl{input_len}_osl{output_len}",
        batch_size=(batch_size,),
        input_len=(input_len,),
        output_len=(output_len,),
        temperature=0.0,
        skip_warmup=False,
        show_report=False,
        dataset_name="random",
        seed=42,
    )


def _run_one_canary_setting(
    *,
    canary_on: bool,
    batch_size: int,
    input_len: int,
    output_len: int,
    disable_cuda_graph: bool = False,
    profile_output_dir: Optional[Path] = None,
) -> BenchOneCaseResult:
    server_args = _make_server_args(
        canary_on=canary_on, disable_cuda_graph=disable_cuda_graph
    )
    bench_args = _make_bench_args(
        batch_size=batch_size, input_len=input_len, output_len=output_len
    )
    if profile_output_dir is not None:
        profile_output_dir.mkdir(parents=True, exist_ok=True)
        bench_args = dataclasses.replace(
            bench_args,
            profile=True,
            profile_steps=_PROFILE_STEPS,
            profile_output_dir=str(profile_output_dir),
        )

    results, _server_info = run_benchmark_internal(
        server_args=server_args,
        bench_args=bench_args,
        launch_server_func=launch_server,
    )
    if not results:
        raise RuntimeError(
            f"run_benchmark_internal returned no rows (canary_on={canary_on}, "
            f"bs={batch_size}, isl={input_len}, osl={output_len})"
        )
    return results[0]


def _make_scenario_key(*, batch_size: int, input_len: int, output_len: int) -> str:
    workload = "prefill" if output_len == 1 else "decode"
    return (
        f"{_QWEN3_SCENARIO_MODEL}/{workload}_bs{batch_size}"
        f"_isl{input_len}_osl{output_len}"
    )


def _resolve_profile_root() -> Optional[Path]:
    raw = os.getenv(_PROFILE_DIR_ENV)
    return Path(raw).expanduser().resolve() if raw else None


_BENCH_REPORT_FIELDS: tuple[str, ...] = (
    "latency",
    "input_throughput",
    "output_throughput",
    "overall_throughput",
    "last_ttft",
    "last_gen_throughput",
    "acc_length",
    "cache_hit_rate",
)


def _format_bench_report(
    *,
    scenario_key: str,
    off: BenchOneCaseResult,
    on: BenchOneCaseResult,
    overhead_pct: float,
) -> str:
    header = (
        f"[canary self-bench] {scenario_key}: "
        f"off={off.latency:.4f}s on={on.latency:.4f}s overhead={overhead_pct:.2f}%"
    )
    col_w = 22
    lines = [header, f"  {'field':<{col_w}}{'off':>14}{'on':>14}"]
    for field in _BENCH_REPORT_FIELDS:
        off_v = getattr(off, field)
        on_v = getattr(on, field)
        lines.append(
            f"  {field:<{col_w}}{_fmt_metric(off_v):>14}{_fmt_metric(on_v):>14}"
        )
    return "\n".join(lines)


def _fmt_metric(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


class TestCanarySelfBenchSpeed(unittest.TestCase):
    bench_timeout: ClassVar[float] = 1800.0

    def _capture_profiles(
        self,
        *,
        scenario_key: str,
        profile_root: Path,
        batch_size: int,
        input_len: int,
        output_len: int,
    ) -> None:
        scenario_slug = scenario_key.replace("/", "_")
        scenario_root = profile_root / f"{scenario_slug}_on"

        graph_dir = scenario_root / "cuda_graph"
        graph_run = _run_one_canary_setting(
            canary_on=True,
            batch_size=batch_size,
            input_len=input_len,
            output_len=output_len,
            profile_output_dir=graph_dir,
        )
        print(
            f"[canary self-bench] {scenario_key} profile cuda_graph: "
            f"on={graph_run.latency:.4f}s (trace under {graph_dir})",
            flush=True,
        )

        no_graph_dir = scenario_root / "no_cuda_graph_osl3"
        no_graph_run = _run_one_canary_setting(
            canary_on=True,
            batch_size=batch_size,
            input_len=input_len,
            output_len=_PROFILE_NO_GRAPH_OUTPUT_LEN,
            disable_cuda_graph=True,
            profile_output_dir=no_graph_dir,
        )
        print(
            f"[canary self-bench] {scenario_key} profile no_cuda_graph_osl3: "
            f"on={no_graph_run.latency:.4f}s (trace under {no_graph_dir}); "
            f"off baseline + overhead assertion skipped.",
            flush=True,
        )

    def _measure_overhead(
        self,
        *,
        batch_size: int,
        input_len: int,
        output_len: int,
        max_overhead_pct: float,
    ) -> None:
        scenario_key = _make_scenario_key(
            batch_size=batch_size, input_len=input_len, output_len=output_len
        )
        profile_root = _resolve_profile_root()

        if profile_root is not None:
            self._capture_profiles(
                scenario_key=scenario_key,
                profile_root=profile_root,
                batch_size=batch_size,
                input_len=input_len,
                output_len=output_len,
            )
            return

        off = _run_one_canary_setting(
            canary_on=False,
            batch_size=batch_size,
            input_len=input_len,
            output_len=output_len,
        )
        on = _run_one_canary_setting(
            canary_on=True,
            batch_size=batch_size,
            input_len=input_len,
            output_len=output_len,
        )
        overhead_pct = ((on.latency - off.latency) / off.latency) * 100.0
        report = _format_bench_report(
            scenario_key=scenario_key,
            off=off,
            on=on,
            overhead_pct=overhead_pct,
        )
        print(report, flush=True)
        self.assertLess(
            overhead_pct,
            max_overhead_pct,
            msg=(
                f"{scenario_key} overhead={overhead_pct:.2f}% exceeds "
                f"{max_overhead_pct:.1f}% budget\n{report}"
            ),
        )

    def test_qwen3_prefill_overhead_bs32_isl16384_osl1(self) -> None:
        # Observed ~18.78% on H200 (2026-05-25, commit 058313f54f). Budget set above with
        # headroom; tighten once the prefill-side canary cost is reduced.
        self._measure_overhead(
            batch_size=32,
            input_len=16384,
            output_len=1,
            max_overhead_pct=30.0,
        )

    def test_qwen3_decode_overhead_bs128_isl512_osl1024(self) -> None:
        # Tight budget — at bs=128 the per-step canary cost amortizes; observed <5%.
        self._measure_overhead(
            batch_size=128,
            input_len=512,
            output_len=1024,
            max_overhead_pct=5.0,
        )

    def test_qwen3_decode_overhead_bs1_isl512_osl1024(self) -> None:
        # Observed ~10.21% on H200 (2026-05-25, commit 058313f54f). The per-step canary cost does
        # not amortize at bs=1; tighten once the single-request path is optimized.
        self._measure_overhead(
            batch_size=1,
            input_len=512,
            output_len=1024,
            max_overhead_pct=15.0,
        )


if __name__ == "__main__":
    unittest.main()
