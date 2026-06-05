from __future__ import annotations

from typing import Any, Dict, List

DEFAULT_CHUNK_SIZE: int = 256

DEFAULT_MAX_STEPS: int = 400

# Long enough to span 8 chunks at the default chunk size, which comfortably
# exceeds every chunks_done lower bound (<= 4) asserted in the manual suite.
VERY_LONG_PROMPT_LEN: int = 8 * DEFAULT_CHUNK_SIZE

# Qwen3-0.6B ties its word embeddings and already handles the tie correctly
# under pipeline parallelism (qwen3.py), so the PP scripted-runtime tests can
# use it without patching the model code.
SMALL_MODEL: str = "Qwen/Qwen3-0.6B"


def base_engine_kwargs(
    *,
    model_path: str = SMALL_MODEL,
    chunked_prefill_size: int = DEFAULT_CHUNK_SIZE,
    **overrides: Any,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = dict(
        model_path=model_path,
        chunked_prefill_size=chunked_prefill_size,
    )
    kwargs.update(overrides)
    return kwargs


def run_until(handle, predicate, *, max_steps: int = DEFAULT_MAX_STEPS):
    for _ in range(max_steps):
        if predicate(handle):
            return
        yield
    raise AssertionError(
        f"run_until: predicate never satisfied after {max_steps} steps "
        f"(handle rid={handle.rid!r}, finished={handle.finished})"
    )


def run_until_finished(handle, *, max_steps: int = DEFAULT_MAX_STEPS):
    yield from run_until(handle, lambda h: h.finished, max_steps=max_steps)


def run_until_all_finished(handles: List[Any], *, max_steps: int = DEFAULT_MAX_STEPS):
    for _ in range(max_steps):
        if all(h.finished for h in handles):
            return
        yield
    raise AssertionError(
        f"run_until_all_finished: not all reqs finished after {max_steps} "
        f"steps (finished={[h.finished for h in handles]})"
    )


def warmup_radix(t, prompt_tokens: List[int], *, max_steps: int = DEFAULT_MAX_STEPS):
    # Populate the radix cache with a prefix by running a real request to
    # completion, so a later request hits the cached prefix. Uniform prompts
    # only (every manual call site passes [v] * n).
    assert prompt_tokens, "warmup_radix needs a non-empty prompt"
    token = prompt_tokens[0]
    assert all(
        x == token for x in prompt_tokens
    ), "warmup_radix supports only uniform prompts"
    handle = t.start_req(
        prompt_len=len(prompt_tokens), max_new_tokens=1, prompt_token=token
    )
    yield from run_until_finished(handle, max_steps=max_steps)


# Long enough that ballast requests never finish for the duration of any test
# (>> every max_steps below), yet 1 + this stays within the smallest test model's
# context window (Qwen3-0.6B is 40960); a larger max_new_tokens is rejected at
# admission ("exceeds the model's maximum context length") and never holds a row.
BALLAST_MAX_NEW_TOKENS: int = 30000


def exhaust_row_pool(t, *, leave_rows: int, max_steps: int = DEFAULT_MAX_STEPS):
    # Pressure req_to_token_pool with REAL ballast requests instead of slicing
    # the pool's internal free_slots: a request occupies exactly one row for its
    # whole lifetime, so M never-finishing ballast reqs hold exactly M rows. The
    # engine's abort_all() drain on reset frees them, so no custom restore needed.
    target: int = t.scheduler.req_to_token_pool.available_size() - leave_rows
    if target <= 0:
        return

    for _ in range(target):
        t.start_req(
            prompt_len=1, max_new_tokens=BALLAST_MAX_NEW_TOKENS, ignore_eos=True
        )

    for _ in range(max_steps):
        if t.scheduler.req_to_token_pool.available_size() <= leave_rows:
            return
        yield
    raise AssertionError(
        f"exhaust_row_pool: ballast reqs never filled the row pool down to "
        f"leave_rows={leave_rows} after {max_steps} steps "
        f"(available_size={t.scheduler.req_to_token_pool.available_size()})"
    )


LIFECYCLE_STAGES = (
    "first_chunk",
    "last_chunk",
    "first_decode",
    "mid_decode",
    "last_decode",
)


def advance_to_nth_chunk(r, target_chunk: int, *, max_steps: int = DEFAULT_MAX_STEPS):
    # Drive until the hook has recorded `target_chunk` chunked-prefill batches.
    # chunks_done is accumulated from on_run_batch (every forward batch), so it
    # never misses a chunk the way sampling the instantaneous is_chunking flag
    # once per yield can: on the step the req leaves chunked_req, is_chunking is
    # already False, so `seen` undercounted and the req could race to completion
    # on slower CI before the loop caught up.
    for _ in range(max_steps):
        assert not r.finished, f"req finished before reaching chunk {target_chunk}"
        if r.chunks_done >= target_chunk:
            return
        yield
    raise AssertionError(
        f"never reached chunk {target_chunk} (chunks_done={r.chunks_done})"
    )


def advance_to_decode_step(
    r, target_output_len: int, *, max_steps: int = DEFAULT_MAX_STEPS
):
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
