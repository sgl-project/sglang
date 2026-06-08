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
from sglang.test.test_utils import DEFAULT_PORT_FOR_SRT_TEST_RUNNER

register_cuda_ci(est_time=333, stage="extra-a", runner_config="1-gpu-large")


_QWEN3_MODEL = "Qwen/Qwen3-30B-A3B"
_QWEN3_SCENARIO_MODEL = "qwen3-30b-a3b"

_PROFILE_DIR_ENV = "SGLANG_KV_CANARY_PROFILE_DIR"
_PROFILE_STEPS = 30
_PROFILE_NO_GRAPH_OUTPUT_LEN = 3
# start_profile blocks until num_steps server steps complete, so it must be <= actual decode steps.
_PROFILE_NO_GRAPH_STEPS = 3


def _make_server_args(
    *, canary_on: bool, disable_cuda_graph: bool = False
) -> ServerArgs:
    # install_canary asserts --disable-piecewise-cuda-graph; pass on both sides for apples-to-apples.
    extra = [
        "--model-path",
        _QWEN3_MODEL,
        "--disable-piecewise-cuda-graph",
    ]
    if disable_cuda_graph:
        extra.append("--disable-cuda-graph")
    if canary_on:
        extra += ["--kv-canary", "raise"]
    extra += ["--port", str(DEFAULT_PORT_FOR_SRT_TEST_RUNNER)]

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
        show_report=True,
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
    profile_steps: int = _PROFILE_STEPS,
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
            profile_steps=profile_steps,
            profile_output_dir=str(profile_output_dir),
        )

    results, _server_info = run_benchmark_internal(
        server_args=server_args,
        bench_args=bench_args,
        launch_server_func=launch_server,
    )
    if not results:
        # run_benchmark_internal returns no rows when the bench was skipped
        # at the token-capacity guard inside it (the Qwen3-30B-A3B model
        # leaves only ~12GB for KV cache on an H100; this test's bs128 +
        # 1024 osl needs more than that). Treat that as a hardware-level
        # skip rather than a test failure: the canary overhead claim is
        # still meaningful when the runner has enough memory.
        raise unittest.SkipTest(
            f"run_benchmark_internal returned no rows (canary_on={canary_on}, "
            f"bs={batch_size}, isl={input_len}, osl={output_len}); the runner's "
            f"KV cache is too small to fit this configuration -- nothing to measure."
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
        # +3 to cover prefill chunks + tail; capped so long decode runs still stop after 30 steps.
        graph_profile_steps = min(_PROFILE_STEPS, output_len + 3)
        graph_run = _run_one_canary_setting(
            canary_on=True,
            batch_size=batch_size,
            input_len=input_len,
            output_len=output_len,
            profile_output_dir=graph_dir,
            profile_steps=graph_profile_steps,
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
            profile_steps=_PROFILE_NO_GRAPH_STEPS,
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
        summary = (
            f"[canary self-bench] {scenario_key}: "
            f"off={off.latency:.4f}s on={on.latency:.4f}s overhead={overhead_pct:.2f}%"
        )
        print(summary, flush=True)
        self.assertLess(
            overhead_pct,
            max_overhead_pct,
            msg=(f"{summary} — exceeds {max_overhead_pct:.1f}% budget"),
        )

    def test_qwen3_prefill_overhead_bs32_isl16384_osl1(self) -> None:
        # TODO: tighten further once the per-forward elementwise glue + plan_offsets
        # single-program kernel are optimized (observed ~2.17% on Qwen3-30B-A3B, H200).
        self._measure_overhead(
            batch_size=32,
            input_len=16384,
            output_len=1,
            max_overhead_pct=3.0,
        )

    def test_qwen3_decode_overhead_bs64_isl256_osl512(self) -> None:
        # TODO: tighten further once per-forward canary glue is reduced (observed ~0.52% on
        # Qwen3-30B-A3B, H200 — already amortizes well at large bs). The smaller
        # 64 * (256+512) = 49K-token budget fits the ~94K KV-cache slice that
        # extra-a-test-1-gpu-large (H100) leaves after loading the 30B MoE.
        self._measure_overhead(
            batch_size=64,
            input_len=256,
            output_len=512,
            max_overhead_pct=1.0,
        )

    def test_qwen3_decode_overhead_bs1_isl512_osl1024(self) -> None:
        # TODO: tighten further once the per-forward elementwise glue + plan_offsets
        # single-program kernel are optimized (observed ~2.10% on Qwen3-30B-A3B, H200).
        self._measure_overhead(
            batch_size=1,
            input_len=512,
            output_len=1024,
            max_overhead_pct=3.0,
        )


if __name__ == "__main__":
    unittest.main()
