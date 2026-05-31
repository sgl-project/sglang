"""Shared constants and helpers for chunked-prefill scripted-runtime tests.

Every test file in this directory imports from here. Constants and
helpers stay deliberately small — most of the value is in keeping the
scripts uniform, not in factoring shared logic.

API surface used by these tests:

* :func:`sglang.test.scripted_runtime.http_server.launch_scripted_http_server`
* :class:`sglang.test.scripted_runtime.ScriptedContext`
* :class:`sglang.test.scripted_runtime.ScriptedReqHandle`
"""

from __future__ import annotations

from typing import Any, Dict, List

from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST

# Chosen to force every meaningful prompt into 4+ chunks. The
# accuracy-flavored suite under test/manual/chunked_prefill/ uses the
# same value; keeping them aligned makes side-by-side comparisons easy.
DEFAULT_CHUNK_SIZE: int = 256

# Bound the bare-yield idiom so a stuck scheduler doesn't hang the test
# process forever. Picked generously — real scripts almost never need
# more than ~50 steps.
DEFAULT_MAX_STEPS: int = 400

# Default model — single-GPU tests use the small model; the multi-GPU
# feature tests override per-fixture.
SMALL_MODEL: str = DEFAULT_SMALL_MODEL_NAME_FOR_TEST


def base_engine_kwargs(
    *,
    model_path: str = SMALL_MODEL,
    chunked_prefill_size: int = DEFAULT_CHUNK_SIZE,
    **overrides: Any,
) -> Dict[str, Any]:
    """Common engine kwargs for a single-GPU scripted-runtime chunked test.

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
        f"(handle rid={handle.rid!r}, finished={handle.finished})"
    )


def run_until_finished(handle, *, max_steps: int = DEFAULT_MAX_STEPS):
    """Generator helper: ``yield`` until the handle reports ``finished``.

    Mirrors the most common idiom across the suite.
    """
    yield from run_until(handle, lambda h: h.finished, max_steps=max_steps)


def run_until_all_finished(handles: List[Any], *, max_steps: int = DEFAULT_MAX_STEPS):
    """Generator helper: ``yield`` until every handle reports finished."""
    for _ in range(max_steps):
        if all(h.finished for h in handles):
            return
        yield
    raise AssertionError(
        f"run_until_all_finished: not all reqs finished after {max_steps} "
        f"steps (finished={[h.finished for h in handles]})"
    )


LIFECYCLE_STAGES = (
    "first_chunk",
    "last_chunk",
    "first_decode",
    "mid_decode",
    "last_decode",
)


def advance_to_nth_chunk(r, target_chunk: int, *, max_steps: int = DEFAULT_MAX_STEPS):
    """Generator helper: ``yield`` until the req is on its ``target_chunk``-th chunked iter.

    Counts iterations where the req holds the scheduler's chunked_req slot
    (``is_chunking``); the final extend that completes prefill is not chunked
    and is therefore not counted.
    """
    seen = 0
    for _ in range(max_steps):
        assert not r.finished, f"req finished before reaching chunk {target_chunk}"
        if r.is_chunking:
            seen += 1
            if seen >= target_chunk:
                return
        yield
    raise AssertionError(f"never reached chunk {target_chunk} (saw {seen})")


def advance_to_decode_step(
    r, target_output_len: int, *, max_steps: int = DEFAULT_MAX_STEPS
):
    """Generator helper: ``yield`` until the req has produced ``target_output_len`` decode tokens.

    Reads ``Req.output_ids`` directly and assumes the req runs to
    ``max_new_tokens`` by length — the synthetic decode does not stop early.
    """
    for _ in range(max_steps):
        assert (
            not r.finished
        ), f"req finished before reaching decode step {target_output_len}"
        req = r.req
        if req is not None and len(req.output_ids) >= target_output_len:
            return
        yield
    raise AssertionError(f"never reached decode step {target_output_len}")


def advance_to_lifecycle_stage(
    r,
    stage: str,
    *,
    num_middle_chunks: int,
    max_new_tokens: int,
    max_steps: int = DEFAULT_MAX_STEPS,
):
    """Generator helper: ``yield`` until the req reaches the named :data:`LIFECYCLE_STAGES` point."""
    if stage == "first_chunk":
        yield from advance_to_nth_chunk(r, 1, max_steps=max_steps)
    elif stage == "last_chunk":
        yield from advance_to_nth_chunk(r, num_middle_chunks, max_steps=max_steps)
    elif stage == "first_decode":
        yield from advance_to_decode_step(r, 1, max_steps=max_steps)
    elif stage == "mid_decode":
        yield from advance_to_decode_step(
            r, max(1, max_new_tokens // 2), max_steps=max_steps
        )
    elif stage == "last_decode":
        yield from advance_to_decode_step(r, max_new_tokens - 1, max_steps=max_steps)
    else:
        raise AssertionError(f"unknown lifecycle stage {stage!r}")
