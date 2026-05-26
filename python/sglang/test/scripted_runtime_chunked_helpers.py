"""Shared constants and helpers for chunked-prefill ScriptedRuntime tests.

Every test file in this directory imports from here. Constants and
helpers stay deliberately small — most of the value is in keeping the
scripts uniform, not in factoring shared logic.

API surface used by these tests:

* :func:`sglang.test.scripted_runtime.execute_scripted_runtime`
* :class:`sglang.test.scripted_runtime.ScriptedRuntime`
* :class:`sglang.test.scripted_runtime.ReqHandle`

Many scripts also reference attributes that are not yet on
:class:`ReqHandle` / :class:`ScriptedRuntime`. The full wishlist lives
in ``agent-context/projects/sglang/2026-05-25-chunked-prefill-rewrite/
agent-drafts/2026-05-26-chunked-test-suite-initial-plan.md`` §4. When
the harness gains those attributes, the corresponding tests start
exercising them with no further wiring needed.
"""

from __future__ import annotations

from typing import Any, Dict, List

from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
)

# Chosen to force every meaningful prompt into 4+ chunks. The
# accuracy-flavored suite under test/manual/chunked_prefill/ uses the
# same value; keeping them aligned makes side-by-side comparisons easy.
DEFAULT_CHUNK_SIZE: int = 256

# "Long" = comfortably > DEFAULT_CHUNK_SIZE so every test that calls
# ``start_req(prompt_len=LONG_PROMPT_LEN)`` triggers chunking.
LONG_PROMPT_LEN: int = 1024

# A prompt length that crosses many chunk boundaries — used to give the
# scheduler several iterations to observe / abort / retract in flight.
VERY_LONG_PROMPT_LEN: int = 4096

# Bound the bare-yield idiom so a stuck scheduler doesn't hang the test
# process forever. Picked generously — real scripts almost never need
# more than ~50 steps.
DEFAULT_MAX_STEPS: int = 400

# Default model — single-GPU tests use the small model; the multi-GPU
# feature tests override per-fixture.
SMALL_MODEL: str = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
DEFAULT_MODEL: str = DEFAULT_MODEL_NAME_FOR_TEST


def base_engine_kwargs(
    *,
    model_path: str = SMALL_MODEL,
    chunked_prefill_size: int = DEFAULT_CHUNK_SIZE,
    **overrides: Any,
) -> Dict[str, Any]:
    """Common engine kwargs for a single-GPU ScriptedRuntime chunked test.

    ``disable_overlap_schedule`` and ``disable_cuda_graph`` are off to
    keep the scheduler event loop deterministic — chunked-state tests
    care about which iteration a transition happens on, not about
    throughput.
    """
    kwargs: Dict[str, Any] = dict(
        model_path=model_path,
        tp_size=1,
        dp_size=1,
        pp_size=1,
        chunked_prefill_size=chunked_prefill_size,
        disable_overlap_schedule=True,
        disable_cuda_graph=True,
    )
    kwargs.update(overrides)
    return kwargs


def run_until(handle, predicate, *, max_steps: int = DEFAULT_MAX_STEPS):
    """Generator helper: ``yield`` until ``predicate(handle)`` is true.

    Caps the iteration count so a stuck scheduler turns into an
    AssertionError instead of a hang. Use ``yield from run_until(...)``
    inside a script generator.
    """
    for _ in range(max_steps):
        if predicate(handle):
            return
        yield
    raise AssertionError(
        f"run_until: predicate never satisfied after {max_steps} steps "
        f"(handle rid={handle.rid!r}, status={handle.status!r})"
    )


def run_until_finished(handle, *, max_steps: int = DEFAULT_MAX_STEPS):
    """Generator helper: ``yield`` until the handle reports ``finished``.

    Mirrors the most common idiom across the suite.
    """
    yield from run_until(
        handle,
        lambda h: getattr(h, "finished", False) or h.status == "finished",
        max_steps=max_steps,
    )


def run_until_all_finished(handles: List[Any], *, max_steps: int = DEFAULT_MAX_STEPS):
    """Generator helper: ``yield`` until every handle reports finished."""
    for _ in range(max_steps):
        if all(
            getattr(h, "finished", False) or h.status == "finished" for h in handles
        ):
            return
        yield
    raise AssertionError(
        f"run_until_all_finished: not all reqs finished after {max_steps} "
        f"steps (statuses={[h.status for h in handles]})"
    )
