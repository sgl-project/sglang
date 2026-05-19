"""Speed test: canary OFF vs canary=log overhead on Qwen3-0.6B.

We sweep two representative scenarios via ``bench_one_batch_server``:

1. Prefill: bs=32, isl=16384, osl=1
2. Large-bs decode: bs=256, isl=4096, osl=1024

For each scenario we launch a fresh server twice -- once with
``--kv-cache-canary=off`` (baseline) and once with
``--kv-cache-canary=log`` (canary running, no raise, production-ish
profile) -- then compare the throughput / latency numbers reported by
``bench_one_batch_server``.

Goal: surface canary overhead in a CI-runnable form so the v1
overhead target has a concrete measurement to check against. We
intentionally do not assert a hard percentage in CI (noisy infra makes
that flaky); instead we print both sets of numbers so reviewers can
eyeball the delta and a follow-up perf agent can analyze.

Test group: extra-a / 1-gpu-large (H100 80GB equivalent). Performance
tests are noise-sensitive so we keep them out of the base lane.
"""

from __future__ import annotations

import dataclasses
import unittest
from typing import List, Tuple

from sglang.srt.server_args import ServerArgs
from sglang.test.bench_one_batch_server_internal import BenchArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    run_bench_one_batch_server,
)

register_cuda_ci(est_time=600, stage="extra-a", runner_config="1-gpu-large")

_MODEL = "Qwen/Qwen3-0.6B"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Scenario:
    """One bench scenario: (label, batch_size, input_len, output_len)."""

    label: str
    batch_size: int
    input_len: int
    output_len: int


_SCENARIOS: Tuple[_Scenario, ...] = (
    _Scenario(
        label="prefill_bs32_isl16384_osl1", batch_size=32, input_len=16384, output_len=1
    ),
    _Scenario(
        label="decode_bs256_isl4096_osl1024",
        batch_size=256,
        input_len=4096,
        output_len=1024,
    ),
)


def _run_one(
    *,
    scenario: _Scenario,
    canary_mode: str,
) -> None:
    """Launch server with canary_mode then run bench_one_batch_server.

    canary_mode in {"off", "log"} maps to the ``--kv-cache-canary`` flag.
    """
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

    run_bench_one_batch_server(
        model=_MODEL,
        base_url=DEFAULT_URL_FOR_TEST,
        server_args=server_args,
        bench_args=bench_args,
        other_server_args=other_server_args,
    )


class TestKvCacheCanarySpeedQwen3(CustomTestCase):
    """Compare canary OFF vs canary=log throughput on Qwen3-0.6B."""

    def test_prefill_bs32_isl16384_osl1(self):
        scenario = _SCENARIOS[0]
        print(f"\n=== Scenario {scenario.label}: canary OFF ===")
        _run_one(scenario=scenario, canary_mode="off")
        print(f"\n=== Scenario {scenario.label}: canary=log ===")
        _run_one(scenario=scenario, canary_mode="log")

    def test_decode_bs256_isl4096_osl1024(self):
        scenario = _SCENARIOS[1]
        print(f"\n=== Scenario {scenario.label}: canary OFF ===")
        _run_one(scenario=scenario, canary_mode="off")
        print(f"\n=== Scenario {scenario.label}: canary=log ===")
        _run_one(scenario=scenario, canary_mode="log")


if __name__ == "__main__":
    unittest.main(verbosity=3)
