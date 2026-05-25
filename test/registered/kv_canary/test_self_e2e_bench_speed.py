from __future__ import annotations

import argparse
import dataclasses
import os
import unittest
import warnings
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

# When set, every bench scenario captures a 30-step torch profile of the
# canary-on run only, under `${KV_CANARY_PROFILE_DIR}/<scenario>_on/`. In this
# mode the canary-off baseline run is skipped entirely (no overhead is
# computed and the 200% assertion does not fire), so the bench finishes in
# roughly half the time and the on-side trace is what you read.
_PROFILE_DIR_ENV = "KV_CANARY_PROFILE_DIR"
_PROFILE_STEPS = 30


def _make_server_args(*, canary_on: bool) -> ServerArgs:
    # canary requires --disable-piecewise-cuda-graph (install_canary asserts it). Pass it on both
    # sides for apples-to-apples. Do NOT pass --disable-cuda-graph: canary must run inside the
    # regular cuda graph for a representative measurement.
    extra = [
        "--model-path",
        _QWEN3_MODEL,
        "--disable-piecewise-cuda-graph",
    ]
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
    profile_output_dir: Optional[Path] = None,
) -> BenchOneCaseResult:
    server_args = _make_server_args(canary_on=canary_on)
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


def _measure_overhead(*, batch_size: int, input_len: int, output_len: int) -> None:
    scenario_key = _make_scenario_key(
        batch_size=batch_size, input_len=input_len, output_len=output_len
    )
    profile_root = _resolve_profile_root()

    if profile_root is not None:
        scenario_slug = scenario_key.replace("/", "_")
        profile_output_dir = profile_root / f"{scenario_slug}_on"
        on = _run_one_canary_setting(
            canary_on=True,
            batch_size=batch_size,
            input_len=input_len,
            output_len=output_len,
            profile_output_dir=profile_output_dir,
        )
        print(
            f"[canary self-bench] {scenario_key} profile mode: "
            f"on={on.latency:.4f}s (trace under {profile_output_dir}); "
            f"off baseline + overhead assertion skipped.",
            flush=True,
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
    # Surface in pytest's warnings summary so the overhead number appears even when the test
    # passes (pytest captures plain ``print`` and discards on success).
    warnings.warn(summary, stacklevel=2)
    assert overhead_pct < 5.0, f"{summary} — exceeds 5% budget"


class TestCanarySelfBenchSpeed(unittest.TestCase):
    bench_timeout: ClassVar[float] = 1800.0

    def test_qwen3_prefill_overhead_bs32_isl16384_osl1(self) -> None:
        """Verify canary prefill overhead stays within the expected bound."""
        _measure_overhead(
            batch_size=32,
            input_len=16384,
            output_len=1,
        )

    def test_qwen3_decode_overhead_bs128_isl512_osl1024(self) -> None:
        """Verify canary decode overhead stays within the expected bound."""
        _measure_overhead(
            batch_size=128,
            input_len=512,
            output_len=1024,
        )

    def test_qwen3_decode_overhead_bs1_isl512_osl1024(self) -> None:
        """Verify canary decode overhead stays within the expected bound."""
        _measure_overhead(
            batch_size=1,
            input_len=512,
            output_len=1024,
        )


if __name__ == "__main__":
    unittest.main()
