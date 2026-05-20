"""Canary on/off overhead self-bench.

Launches a Qwen3-0.6B server twice per case — once with the canary disabled,
once with ``--kv-canary raise`` — and reports steady-state latency for
each. The overhead percentage is printed to stdout and appended to the sibling
``test_self_bench_speed.baseline.json`` file (warn-only; no hard gate yet —
see testing.md SOT §3.3 "首次 commit 不 hard-gate" + plan 04 step 7).

testing.md SOT §3.3 — 2 cases:

- ``bench_qwen3_prefill_bs32_isl16384_osl1``
- ``bench_qwen3_decode_bs256_isl4096_osl1024``

Both registered to ``extra-a`` (label-gated PR) and ``nightly-1-gpu`` (auto
nightly accumulation per plan 04 step 6). Method prefix ``bench_`` is the SOT
casename convention; ``unittest.TestLoader.testMethodPrefix`` is rebound in
``__main__`` so ``python3 file.py -f`` still discovers them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unittest
from pathlib import Path
from typing import ClassVar, List, Tuple

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
_NUM_LAYERS_OVERRIDE = '{"num_hidden_layers": 1}'
_BASELINE_FILENAME = "test_self_bench_speed.baseline.json"
_RUNNER_CONFIG_KEY = "1-gpu-large"


def _make_server_args(*, canary_on: bool) -> ServerArgs:
    extra: List[str] = [
        "--model-path",
        _QWEN3_MODEL,
        "--json-model-override-args",
        _NUM_LAYERS_OVERRIDE,
        "--disable-cuda-graph",
        "--mem-fraction-static",
        "0.65",
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


def _append_baseline_sample(
    *,
    scenario_key: str,
    latency_off: float,
    latency_on: float,
    overhead_pct: float,
) -> None:
    baseline_path = Path(__file__).parent / _BASELINE_FILENAME
    if baseline_path.exists():
        with baseline_path.open("r") as f:
            blob = json.load(f)
    else:
        blob = {}

    bucket = blob.setdefault(_RUNNER_CONFIG_KEY, {})
    slot = bucket.setdefault(
        scenario_key,
        {"samples": [], "p50_ms": None, "p90_ms": None, "p99_ms": None},
    )
    slot["samples"].append(
        {
            "ts_unix": time.time(),
            "latency_off_s": round(latency_off, 4),
            "latency_on_s": round(latency_on, 4),
            "overhead_pct": round(overhead_pct, 4),
        }
    )
    with baseline_path.open("w") as f:
        json.dump(blob, f, indent=2)
        f.write("\n")


def _measure_overhead(
    *, scenario_key: str, batch_size: int, input_len: int, output_len: int
) -> Tuple[float, float, float]:
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
    _append_baseline_sample(
        scenario_key=scenario_key,
        latency_off=off.latency,
        latency_on=on.latency,
        overhead_pct=overhead_pct,
    )
    return off.latency, on.latency, overhead_pct


class TestCanarySelfBenchSpeed(unittest.TestCase):
    bench_timeout: ClassVar[float] = 1800.0

    def bench_qwen3_prefill_bs32_isl16384_osl1(self) -> None:
        _measure_overhead(
            scenario_key="qwen3-0.6b/prefill_bs32_isl16384_osl1",
            batch_size=32,
            input_len=16384,
            output_len=1,
        )

    def bench_qwen3_decode_bs256_isl4096_osl1024(self) -> None:
        _measure_overhead(
            scenario_key="qwen3-0.6b/decode_bs256_isl4096_osl1024",
            batch_size=256,
            input_len=4096,
            output_len=1024,
        )


def _main() -> int:
    # Step 1: rebind the discovery prefix so ``bench_*`` methods become
    # runnable test methods (SOT §3.3 casename convention).
    unittest.TestLoader.testMethodPrefix = "bench_"

    # Step 2: hand control to the standard CLI so ``python3 file.py -f`` and
    # any explicit ``bench_*`` selectors work unchanged.
    return unittest.main(
        module=__name__, exit=False, argv=sys.argv
    ).result.wasSuccessful()


if __name__ == "__main__":
    sys.exit(0 if _main() else 1)
