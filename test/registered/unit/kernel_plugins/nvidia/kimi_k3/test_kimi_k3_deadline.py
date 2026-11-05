# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""CPU/static contract for Kimi K3 cross-GPU wait deadlines."""

from __future__ import annotations

import ast
import math

import pytest

from sglang.srt.kernel_plugins.nvidia.kimi_k3.spec import _deadline_ns
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kernels.kimi_k3_source_paths import source_definition

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _definition(relative: str, *qualname: str) -> ast.AST:
    return source_definition(relative, *qualname)


def _calls(node: ast.AST, name: str) -> int:
    return sum(
        1
        for item in ast.walk(node)
        if isinstance(item, ast.Call)
        and (
            isinstance(item.func, ast.Name)
            and item.func.id == name
            or isinstance(item.func, ast.Attribute)
            and item.func.attr == name
        )
    )


def _call_nodes(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        item
        for item in ast.walk(node)
        if isinstance(item, ast.Call)
        and (
            isinstance(item.func, ast.Name)
            and item.func.id == name
            or isinstance(item.func, ast.Attribute)
            and item.func.attr == name
        )
    ]


def _mentions_deadline(node: ast.AST) -> bool:
    return any(
        isinstance(item, ast.Name)
        and item.id == "deadline_ns"
        or isinstance(item, ast.Attribute)
        and item.attr == "deadline_ns"
        for item in ast.walk(node)
    )


def _parameters(relative: str, *qualname: str) -> set[str]:
    node = _definition(relative, *qualname)
    assert isinstance(node, ast.FunctionDef)
    args = node.args
    return {
        argument.arg for argument in (args.posonlyargs + args.args + args.kwonlyargs)
    }


@pytest.mark.parametrize(
    ("seconds", "expected"),
    ((30, 30_000_000_000), (0.25, 250_000_000), (1e-9, 1)),
)
def test_deadline_seconds_are_frozen_as_integer_nanoseconds(seconds, expected):
    assert _deadline_ns(seconds) == expected


@pytest.mark.parametrize(
    "seconds",
    (
        None,
        True,
        False,
        0,
        -1,
        1e-10,
        math.nan,
        math.inf,
        -math.inf,
        1e30,
        "not-a-deadline",
    ),
)
def test_deadline_rejects_nonfinite_nonpositive_and_out_of_range_values(seconds):
    with pytest.raises(ValueError, match="deadline"):
        _deadline_ns(seconds)


def test_exact_six_reachable_tp8_remote_poll_definitions_fail_stop():
    definitions = (
        ("cutedsl/common/mailbox.py", ("poll",)),
        ("cutedsl/common/twoshot.py", ("reduce",)),
        ("cutedsl/oproj/consumer.py", ("OprojConsumer", "poll_sum")),
        ("cutedsl/moe/tail.py", ("MoeTail", "poll")),
        ("cutedsl/moe/tail.py", ("WaveMoeTail", "settle")),
        (
            "cutedsl/moe/small_batch/front_tp8.py",
            ("MoeFrontTP8", "kernel"),
        ),
    )
    for relative, qualname in definitions:
        node = _definition(relative, *qualname)
        assert _calls(node, "wait_deadline_start") == 1, (relative, qualname)
        assert _calls(node, "wait_deadline_check") == 1, (relative, qualname)


def test_local_and_dormant_waits_are_not_instrumented():
    definitions = (
        ("cutedsl/common/primitives.py", ("wait",)),
        ("cutedsl/common/primitives.py", ("wait_phase_zero_cluster",)),
        ("cutedsl/common/primitives.py", ("wait_tmem",)),
        ("cutedsl/moe/tail.py", ("MoeTail", "front")),
        ("cutedsl/moe/tail.py", ("WaveMoeTail", "poll_rows")),
        (
            "cutedsl/moe/small_batch/front_tp16.py",
            ("MoeFrontTP16", "poll_flag"),
        ),
        ("cutedsl/moe/experts.py", ("ExpertsKernel", "poll_hidden")),
    )
    for relative, qualname in definitions:
        node = _definition(relative, *qualname)
        assert _calls(node, "wait_deadline_start") == 0, (relative, qualname)
        assert _calls(node, "wait_deadline_check") == 0, (relative, qualname)


def test_deadline_is_part_of_every_reachable_compiler_and_constructor_key():
    definitions = (
        ("cutedsl/oproj/prepared.py", ("compile_oproj",)),
        ("cutedsl/oproj/consumer.py", ("OprojConsumer", "__init__")),
        ("cutedsl/attn_res/prepared.py", ("compile_consumer",)),
        ("cutedsl/attn_res/kernel.py", ("ClusterAttnRes", "__init__")),
        ("cutedsl/moe/front.py", ("compile_route",)),
        ("cutedsl/moe/front.py", ("RouteSitu", "__init__")),
        ("cutedsl/moe/tail_runtime.py", ("compile_tail",)),
        ("cutedsl/moe/tail.py", ("MoeTail", "__init__")),
        ("cutedsl/moe/tail.py", ("WaveMoeTail", "__init__")),
        (
            "cutedsl/moe/small_batch/front_tp8.py",
            ("MoeFrontTP8", "__init__"),
        ),
    )
    for relative, qualname in definitions:
        assert "deadline_ns" in _parameters(relative, *qualname), (
            relative,
            qualname,
        )


def test_deadline_is_forwarded_across_every_reachable_host_compile_chain():
    forwards = (
        ("providers.py", ("build_resources",), "PreparedOproj", 1, 1),
        ("providers.py", ("build_resources",), "PreparedTail", 1, 1),
        ("providers.py", ("build_resources",), "SharedFront", 1, 1),
        ("providers.py", ("build_resources",), "SmallBatchChain", 1, 1),
        ("providers.py", ("build_resources",), "for_target", 2, 2),
        (
            "cutedsl/oproj/prepared.py",
            ("PreparedOproj", "__init__"),
            "compile_oproj",
            1,
            1,
        ),
        (
            "cutedsl/attn_res/prepared.py",
            ("PreparedAttnRes", "_compile"),
            "compile_consumer",
            1,
            1,
        ),
        (
            "cutedsl/moe/front.py",
            ("SharedFront", "__init__"),
            "compile_route",
            2,
            2,
        ),
        (
            "cutedsl/moe/tail_runtime.py",
            ("MoeTailWeights", "__init__"),
            "compile_tail",
            1,
            1,
        ),
        (
            "cutedsl/moe/small_batch/chain.py",
            ("SmallBatchChain", "__init__"),
            "compile_tail",
            2,
            1,
        ),
    )
    for relative, qualname, callee, count, deadline_count in forwards:
        calls = _call_nodes(_definition(relative, *qualname), callee)
        assert len(calls) == count, (relative, qualname, callee)
        assert sum(_mentions_deadline(call) for call in calls) == deadline_count, (
            relative,
            qualname,
            callee,
        )


def test_build_validates_and_forwards_deadline_before_provider_import():
    build = _definition("spec.py", "_build")
    calls = [
        item.func.id
        for item in ast.walk(build)
        if isinstance(item, ast.Call) and isinstance(item.func, ast.Name)
    ]
    assert "_deadline_ns" in calls
    assert "validate_runtime_compatibility" in calls
    deadline = next(
        item
        for item in ast.walk(build)
        if isinstance(item, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "deadline_ns"
            for target in item.targets
        )
        and isinstance(item.value, ast.Call)
        and isinstance(item.value.func, ast.Name)
        and item.value.func.id == "_deadline_ns"
    )
    providers = next(
        item
        for item in ast.walk(build)
        if isinstance(item, ast.ImportFrom)
        and item.module == "sglang.srt.kernel_plugins.nvidia.kimi_k3.providers"
    )
    assert deadline.lineno < providers.lineno
    source = ast.unparse(build)
    assert "deadline_ns=deadline_ns" in source


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
