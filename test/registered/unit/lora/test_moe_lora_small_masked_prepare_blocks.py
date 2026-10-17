"""BF16 small-prepare admission and masked column-tile boundaries."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_KERNEL = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/kernels/ops/lora/moe/dispatch_masked_small.py"
)


def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 1 else 1


def _rule(name):
    tree = ast.parse(_KERNEL.read_text())
    constants = [node for node in tree.body if isinstance(node, ast.Assign)]
    rule = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    scope = {"triton": SimpleNamespace(next_power_of_2=_next_power_of_2)}
    exec(
        compile(
            ast.Module(body=[*constants, rule], type_ignores=[]), str(_KERNEL), "exec"
        ),
        scope,
    )
    return scope[name]


@pytest.mark.parametrize(
    "hidden", [8, 120, 128, 136, 248, 256, 264, 504, 512, 520, 1016, 1024, 1032, 8192]
)
def test_bf16_block_is_a_power_of_two_between_128_and_1024(hidden):
    block = _rule("_column_block")(hidden)
    assert block > 0 and block & (block - 1) == 0
    assert 128 <= block <= 1024


@pytest.mark.parametrize(
    "hidden, expected",
    [(768, 1024), (1032, 1024), (1536, 1024), (2048, 1024), (64, 128)],
)
def test_block_for_the_served_and_the_narrow_widths(hidden, expected):
    assert _rule("_column_block")(hidden) == expected


@pytest.mark.parametrize(
    "tokens, pairs, expected",
    [
        (1, 1, True),
        (8, 32, True),
        (8, 33, False),
        (9, 32, False),
        (1, 0, False),
        (1, -1, False),
    ],
)
def test_bf16_small_prepare_admission(tokens, pairs, expected):
    assert _rule("small_masked_prepare_applies")(tokens, pairs) is expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
