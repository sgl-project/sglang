"""Speed test: canary OFF vs canary=log overhead on Qwen3-0.6B.

We sweep two representative scenarios via ``bench_one_batch_server``:

1. Prefill: bs=32, isl=16384, osl=1
2. Large-bs decode: bs=256, isl=4096, osl=1024

For each scenario we launch a fresh server twice -- once with
``--kv-cache-canary=off`` (baseline) and once with
``--kv-cache-canary=log`` (canary running, no raise, production-ish
profile) -- then assert the canary path adds less than 2% latency.

The 2% threshold is intentionally loose: ``bench_one_batch_server``
measures one batch's wall-clock on a shared CI runner so single-digit
percent jitter is expected. The canary kernel itself should add far
less than 1% in steady state, so 2% safely covers measurement noise
while still catching a regression that doubles the kernel cost.

The hard assertion is currently marked ``xfail(strict=False)`` because
the kernel has not been perf-tuned yet (clean a baseline run showed
~50% overhead). Once perf work lands, drop the xfail decorator and the
test becomes a hard regression gate.
"""

from __future__ import annotations

import dataclasses
import unittest
from typing import List, Tuple

import pytest

from sglang.bench_one_batch_server import run_benchmark
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from sglang.test.bench_one_batch_server_internal import BenchArgs, BenchOneCaseResult
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=600, stage="extra-a", runner_config="1-gpu-large")

_MODEL = "Qwen/Qwen3-0.6B"
# Wall-clock overhead the canary kernel is allowed to add. Loose enough
# to swallow single-batch jitter on a shared CI runner; tight enough
# that a regression that doubles the kernel cost still fails.
_MAX_OVERHEAD_RATIO: float = 0.02


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Scenario:
    """One bench scenario: (label, batch_size, input_len, output_len)."""

    label: str
    batch_size: int
    input_len: int
    output_len: int


_SCENARIOS: Tuple[_Scenario, ...] = (
    _Scenario(
        label="prefill_bs32_isl16384_osl1",
        batch_size=32,
        input_len=16384,
        output_len=1,
    ),
    _Scenario(
        label="decode_bs256_isl4096_osl1024",
        batch_size=256,
        input_len=4096,
        output_len=1024,
    ),
)


def _bench_with_canary_mode(
    *, scenario: _Scenario, canary_mode: str
) -> BenchOneCaseResult:
    """Launch server with canary_mode, run one bench scenario, return result."""
    other_server_args: List[str] = [
        "--mem-fraction-static",
        "0.65",
        "--kv-cache-canary",
        canary_mode,
    ]
    server_args = ServerArgs(model_path=_MODEL)
    bench_args = BenchArgs(
        run_name=f"canary_{canary_mode}__{scenario.label}",
        batch_size=(scenario.batch_size,),
        input_len=(scenario.input_len,),
        output_len=(scenario.output_len,),
        base_url=DEFAULT_URL_FOR_TEST,
        skip_warmup=False,
        show_report=True,
    )

    process = popen_launch_server(
        _MODEL,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_server_args,
    )
    try:
        results, _ = run_benchmark(server_args=server_args, bench_args=bench_args)
    finally:
        kill_process_tree(process.pid)

    if not results:
        raise RuntimeError(
            f"bench_one_batch_server returned no results for {scenario.label} "
            f"with canary_mode={canary_mode}"
        )
    return results[0]


def _check_overhead(scenario: _Scenario, testcase: unittest.TestCase) -> None:
    off = _bench_with_canary_mode(scenario=scenario, canary_mode="off")
    log = _bench_with_canary_mode(scenario=scenario, canary_mode="log")

    overhead = (log.latency - off.latency) / off.latency
    print(
        f"\n=== {scenario.label} ===\n"
        f"  canary=off:  latency={off.latency:.3f}s  "
        f"input_tput={off.input_throughput:.0f}  output_tput={off.output_throughput:.0f}\n"
        f"  canary=log:  latency={log.latency:.3f}s  "
        f"input_tput={log.input_throughput:.0f}  output_tput={log.output_throughput:.0f}\n"
        f"  overhead:    {overhead:.1%}  (budget: {_MAX_OVERHEAD_RATIO:.0%})"
    )
    testcase.assertLess(
        overhead,
        _MAX_OVERHEAD_RATIO,
        f"{scenario.label}: canary overhead {overhead:.1%} exceeds "
        f"{_MAX_OVERHEAD_RATIO:.0%} budget "
        f"(off={off.latency:.3f}s, log={log.latency:.3f}s)",
    )


@pytest.mark.xfail(
    strict=False,
    reason="canary kernel perf optimisation pending; overhead is currently ~50%",
)
class TestKvCacheCanarySpeedQwen3(CustomTestCase):
    """Compare canary OFF vs canary=log latency on Qwen3-0.6B."""

    def test_prefill_bs32_isl16384_osl1(self) -> None:
        _check_overhead(_SCENARIOS[0], self)

    def test_decode_bs256_isl4096_osl1024(self) -> None:
        _check_overhead(_SCENARIOS[1], self)


if __name__ == "__main__":
    unittest.main(verbosity=3)
