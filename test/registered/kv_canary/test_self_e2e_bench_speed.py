from __future__ import annotations

import argparse
import unittest
from typing import ClassVar

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


def _make_server_args(*, canary_on: bool) -> ServerArgs:
    # DO NOT add --disable-cuda-graph or --disable-piecewise-cuda-graph below.
    # The canary kernel must run inside the cuda graph alongside the real attn
    # kernel; an overhead measurement taken with the graph disabled does not
    # represent the production path canary actually ships on.
    extra = [
        "--model-path",
        _QWEN3_MODEL,
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
    *, canary_on: bool, batch_size: int, input_len: int, output_len: int
) -> BenchOneCaseResult:
    server_args = _make_server_args(canary_on=canary_on)
    bench_args = _make_bench_args(
        batch_size=batch_size, input_len=input_len, output_len=output_len
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


def _measure_overhead(
    *, scenario_key: str, batch_size: int, input_len: int, output_len: int
) -> None:
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
    print(
        f"[canary self-bench] {scenario_key}: "
        f"off={off.latency:.4f}s on={on.latency:.4f}s overhead={overhead_pct:.2f}%",
        flush=True,
    )
    assert overhead_pct < 200.0, (
        f"canary overhead {overhead_pct:.1f}% suspiciously high for "
        f"{scenario_key} (off={off.latency:.4f}s, on={on.latency:.4f}s)"
    )


class TestCanarySelfBenchSpeed(unittest.TestCase):
    bench_timeout: ClassVar[float] = 1800.0

    def test_qwen3_prefill_overhead_bs32_isl16384_osl1(self) -> None:
        """Verify canary prefill overhead stays within the expected bound."""
        _measure_overhead(
            scenario_key="qwen3-0.6b/prefill_bs32_isl16384_osl1",
            batch_size=32,
            input_len=16384,
            output_len=1,
        )

    def test_qwen3_decode_overhead_bs128_isl512_osl1024(self) -> None:
        """Verify canary decode overhead stays within the expected bound."""
        _measure_overhead(
            scenario_key="qwen3-0.6b/decode_bs128_isl512_osl1024",
            batch_size=128,
            input_len=512,
            output_len=1024,
        )


if __name__ == "__main__":
    unittest.main()
