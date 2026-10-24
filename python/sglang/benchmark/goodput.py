"""Goodput for the serving benchmark: the requests that met every given latency SLO."""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec

if TYPE_CHECKING:
    from sglang.benchmark.serving import RequestFuncOutput

GOODPUT_SLO_METRICS = ("ttft", "tpot", "e2el")


class GoodputMetrics(msgspec.Struct, frozen=True):
    """Field names are the keys written to the benchmark result file."""

    goodput_slos_ms: dict[str, float]
    request_goodput: float  # good requests per second
    slo_attainment: float  # good requests / requests sent
    slo_attainment_by_metric: dict[str, float]


def calculate_goodput(
    outputs: list[RequestFuncOutput], goodput_slos: dict[str, float], dur_s: float
) -> GoodputMetrics:
    num_good = 0
    num_met = dict.fromkeys(goodput_slos, 0)
    for output in outputs:
        if not output.success:
            continue
        # A single-token response has no inter-token interval; its TPOT counts as 0.
        tpot = 0.0
        if output.output_len > 1:
            tpot = (output.latency - output.ttft) / (output.output_len - 1)
        observed_ms = {
            "ttft": output.ttft * 1000,
            "tpot": tpot * 1000,
            "e2el": output.latency * 1000,
        }
        met = [name for name, slo in goodput_slos.items() if observed_ms[name] <= slo]
        for name in met:
            num_met[name] += 1
        if len(met) == len(goodput_slos):
            num_good += 1
    return GoodputMetrics(
        goodput_slos_ms=goodput_slos,
        request_goodput=num_good / dur_s,
        slo_attainment=num_good / len(outputs),
        slo_attainment_by_metric={
            name: count / len(outputs) for name, count in num_met.items()
        },
    )


def print_goodput(goodput: GoodputMetrics) -> None:
    slos = " ".join(f"{name}:{slo:g}" for name, slo in goodput.goodput_slos_ms.items())
    print("{s:{c}^{n}}".format(s="Goodput", n=50, c="-"))
    print("{:<40} {:<10}".format("SLOs (ms):", slos))
    print(
        "{:<40} {:<10.2f}".format("Request goodput (req/s):", goodput.request_goodput)
    )
    print(
        "{:<40} {:<10.2f}".format("SLO attainment (%):", goodput.slo_attainment * 100)
    )
    for name, attainment in goodput.slo_attainment_by_metric.items():
        label = f"{name.upper()} SLO attainment (%):"
        print("{:<40} {:<10.2f}".format(label, attainment * 100))
