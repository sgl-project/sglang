"""Masked-prepare column blocks are powers of two in [128, 1024].
FP8 blocks must divide the row width because group scales are stored unmasked.
Non-power-of-two BF16 widths, including 768, must round up.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_KERNEL = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/lora/moe/kernels/dispatch_masked_small.py"
)


def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 1 else 1


def _column_block():
    # The rule and its pairs-per-program constant, without importing Triton.
    tree = ast.parse(_KERNEL.read_text())
    ppp = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "_PPP" for t in node.targets)
    )
    rule = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_column_block"
    )
    scope = {"triton": SimpleNamespace(next_power_of_2=_next_power_of_2)}
    exec(
        compile(ast.Module(body=[ppp, rule], type_ignores=[]), str(_KERNEL), "exec"),
        scope,
    )
    return scope["_column_block"]


def _is_power_of_two(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


@pytest.mark.parametrize(
    "hidden", [8, 120, 128, 136, 248, 256, 264, 504, 512, 520, 1016, 1024, 1032, 8192]
)
def test_bf16_block_is_a_power_of_two_between_128_and_1024(hidden):
    block = _column_block()(hidden, False)
    assert _is_power_of_two(block)
    assert 128 <= block <= 1024


@pytest.mark.parametrize(
    "hidden", [128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1536, 2048, 8192]
)
def test_fp8_block_is_a_power_of_two_that_divides_the_width(hidden):
    block = _column_block()(hidden, True)
    assert _is_power_of_two(block)
    assert 128 <= block <= 1024
    assert hidden % block == 0


@pytest.mark.parametrize(
    "hidden, fp8, expected",
    [
        (768, False, 1024),  # Non-power-of-two BF16 width must round up.
        (768, True, 256),
        (1032, False, 1024),
        (1536, False, 1024),
        (1536, True, 512),
        (1920, True, 128),
        (2048, False, 1024),
        (2048, True, 1024),
        (7168, True, 1024),
        (64, False, 128),
    ],
)
def test_block_for_the_served_and_the_narrow_widths(hidden, fp8, expected):
    assert _column_block()(hidden, fp8) == expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
