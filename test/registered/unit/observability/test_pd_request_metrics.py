"""CPU regression tests executing real metrics methods with Prometheus collectors."""

import ast
import sys
from pathlib import Path
from types import MethodType
from types import SimpleNamespace as NS

import pytest
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
SRT = ROOT / "python/sglang/srt"


def load_node(path, class_name, method=None, **symbols):
    tree = ast.parse(path.read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    if method:
        node = next(
            n
            for n in node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == method
        )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"__name__": __name__, **symbols}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method or class_name]


COLLECT = load_node(
    SRT / "managers/tokenizer_manager.py",
    "TokenizerManager",
    "collect_metrics",
    DisaggregationMode=NS(PREFILL="prefill"),
)
METRICS = SRT / "observability/metrics_collector.py"


def collector(role):
    registry = CollectorRegistry()
    obj = NS(
        _gauge_cls=lambda **kw: Gauge(registry=registry, **kw),
        _counter_cls=lambda **kw: Counter(registry=registry, **kw),
        _histogram_cls=lambda **kw: Histogram(registry=registry, **kw),
    )
    init = load_node(
        METRICS,
        "TokenizerMetricsCollector",
        "__init__",
        generate_buckets=lambda supplied, default: supplied or default,
        get_observability=lambda: NS(
            prompt_tokens_buckets=None, generation_tokens_buckets=None
        ),
    )
    init(
        obj,
        server_args=NS(prompt_tokens_buckets=None, generation_tokens_buckets=None),
        labels={"engine_type": role},
    )
    for method in (
        "observe_time_to_first_token",
        "observe_inter_token_latency",
        "observe_finished_outcome",
        "observe_one_finished_request",
    ):
        setattr(
            obj,
            method,
            MethodType(load_node(METRICS, "TokenizerMetricsCollector", method), obj),
        )
    return registry, obj


def request(role, *, stream=False, tokens=101, reason="length", finished=True):
    registry, metrics = collector(role)
    state = NS(
        obj=NS(stream=stream),
        ttft_observed=False,
        last_completion_tokens=1,
        finished=finished,
        time_stats=NS(
            get_first_token_latency=lambda: 0.1,
            get_interval=lambda: 0.1,
            get_e2e_latency=lambda: 1.0,
            set_last_time=lambda: None,
        ),
    )
    manager = NS(
        metrics_collector=metrics,
        disaggregation_mode=role,
        enable_priority_scheduling=False,
        _request_has_grammar=lambda obj: False,
    )
    recv = NS(
        completion_tokens=[tokens],
        finished_reasons=[{"type": reason}],
        prompt_tokens=[100],
        cached_tokens=[60],
    )
    return registry, manager, state, recv


@pytest.mark.parametrize("role", ["null", "decode", "prefill"])
def test_abort_without_output_does_not_observe_first_token(role):
    registry, manager, state, recv = request(role, tokens=0, reason="abort")
    COLLECT(manager, state, recv, 0)
    assert not state.ttft_observed
    assert (
        registry.get_sample_value(
            "sglang:time_to_first_token_seconds_count",
            {"engine_type": role, "is_streaming": "false"},
        )
        is None
    )
    assert (
        registry.get_sample_value(
            "sglang:finished_requests_by_outcome_total",
            {"engine_type": role, "outcome": "abort"},
        )
        == 1
    )


def test_prefill_never_observes_decode_intervals():
    registry, manager, state, recv = request("prefill", tokens=0, finished=False)
    for count in (0, 1, 8, 0):
        recv.completion_tokens = [count]
        COLLECT(manager, state, recv, 0)
    assert (
        registry.get_sample_value(
            "sglang:inter_token_latency_seconds_count", {"engine_type": "prefill"}
        )
        is None
    )


def test_token_count_reset_changes_baseline_without_negative_histograms():
    registry, manager, state, recv = request("decode", tokens=8, finished=False)
    COLLECT(manager, state, recv, 0)
    for count in (0, 0, 2):
        recv.completion_tokens = [count]
        COLLECT(manager, state, recv, 0)
    labels = {"engine_type": "decode"}
    assert state.last_completion_tokens == 2
    assert (
        registry.get_sample_value("sglang:inter_token_latency_seconds_count", labels)
        == 2
    )
    assert registry.get_sample_value(
        "sglang:inter_token_latency_seconds_sum", labels
    ) == pytest.approx(0.1)


@pytest.mark.parametrize("delta", [-10, -1, 0])
def test_collector_rejects_nonpositive_token_weights(delta):
    registry, metrics = collector("decode")
    metrics.observe_inter_token_latency({"engine_type": "decode"}, 0.1, delta)
    assert (
        registry.get_sample_value(
            "sglang:inter_token_latency_seconds_count", {"engine_type": "decode"}
        )
        is None
    )


@pytest.mark.parametrize(
    "reason,outcome",
    [
        ("stop", "success"),
        ("length", "success"),
        ("abort", "abort"),
        ("other", "other"),
    ],
)
def test_terminal_outcomes_and_tokens_are_separated(reason, outcome):
    registry, manager, state, recv = request("decode", reason=reason)
    COLLECT(manager, state, recv, 0)
    labels = {"engine_type": "decode", "outcome": outcome}
    for metric, expected in [
        ("finished_requests_by_outcome", 1),
        ("finished_prompt_tokens_by_outcome", 100),
        ("finished_cached_tokens_by_outcome", 60),
    ]:
        assert (
            registry.get_sample_value("sglang:" + metric + "_total", labels) == expected
        )


@pytest.mark.parametrize("injected_backend", [False, True])
def test_positive_intervals_preserve_weight_for_both_collector_backends(
    injected_backend,
):
    registry, metrics = collector("decode")
    if not injected_backend:
        metrics._histogram_cls = None
    metrics.observe_inter_token_latency({"engine_type": "decode"}, 0.12, 3)
    labels = {"engine_type": "decode"}
    assert (
        registry.get_sample_value("sglang:inter_token_latency_seconds_count", labels)
        == 3
    )
    assert registry.get_sample_value(
        "sglang:inter_token_latency_seconds_sum", labels
    ) == pytest.approx(0.12)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
