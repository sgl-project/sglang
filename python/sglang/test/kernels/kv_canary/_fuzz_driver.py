from __future__ import annotations

import random
from typing import Any, Callable

from sglang.test.kernels.kv_canary._differential import (
    ShrinkResult,
    shrink_inputs,
)

FUZZ_SEEDS_PR: tuple[int, ...] = (0,)


def check_repro(inputs: Any, *, run_one_fn: Callable[[Any], Any]) -> bool:
    try:
        run_one_fn(inputs)
    except (AssertionError, RuntimeError, ValueError):
        return True
    return False


def run_fuzz_combo(
    seed: int,
    *,
    draw_fn: Callable[[random.Random], Any],
    run_one_fn: Callable[[Any], Any],
    summarize_fn: Callable[[Any], str],
    n_iter: int,
) -> None:
    rng = random.Random(seed)
    for iteration in range(n_iter):
        inputs = draw_fn(rng)
        try:
            run_one_fn(inputs)
        except AssertionError as exc:
            shrunk: ShrinkResult = shrink_inputs(
                inputs,
                check_fn=lambda i: check_repro(i, run_one_fn=run_one_fn),
            )
            raise AssertionError(
                f"seed={seed} iter={iteration} failure: {exc}\n"
                f"original: {summarize_fn(inputs)}\n"
                f"shrunk:   {summarize_fn(shrunk.inputs)}\n"
                f"mutations applied: {shrunk.mutations_applied}"
            ) from exc
