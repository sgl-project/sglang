"""A request that finishes while a decode step holding it is in flight.

Under the overlap schedule, the step after the one that samples a stop token
(or the one running when an abort arrives) is already launched with the
request in it. These tests check that such a request still returns the greedy
output, frees every KV slot and request row, and leaves the scheduler idle.

The token that in-flight step computed past the finish was never committed, so
the radix tree caches only the committed tokens: prompt + output - 1, the same
as a finish by length, where no step runs past the last token.
"""

import unittest

from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    FINISH_MATCHED_STR,
    FINISH_MATCHED_TOKEN,
)
from sglang.srt.runtime_context import get_schedule
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_MAX_STEPS,
    advance_to_decode_step,
    base_engine_kwargs,
    run_until,
    run_until_finished,
)

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")

_PROMPT_LEN = 16
# A "Hello" token for the Qwen3 tokenizer; any token whose greedy continuation
# varies works.
_PROMPT_TOKEN = 9707
_MAX_NEW_TOKENS = 32
# Stop no earlier than this output index, so the request is mid-decode.
_MIN_STOP_POS = 3
_ABORT_AT_OUTPUT_LEN = 6


def _start(t: ScriptedContext, **kwargs):
    return t.start_req(
        prompt_len=_PROMPT_LEN,
        prompt_token=_PROMPT_TOKEN,
        max_new_tokens=_MAX_NEW_TOKENS,
        temperature=0.0,
        **kwargs,
    )


def _capture_req(r):
    yield from run_until(r, lambda h: h.req is not None)
    return r.req


def _drain(t: ScriptedContext):
    for _ in range(DEFAULT_MAX_STEPS):
        if t.is_fully_idle:
            return
        yield
    raise AssertionError("scheduler did not return to idle")


def _pool_baseline(t: ScriptedContext):
    tree = t.scheduler.parts.kv.tree_cache
    assert tree.evictable_size() == 0 and tree.protected_size() == 0, (
        "radix tree not empty at script start"
    )
    stats = t.engine_stats()
    return stats["kv_pool_free"], stats["req_pool_free"]


def _assert_no_leak(t: ScriptedContext, baseline):
    kv_free0, rows_free0 = baseline
    tree = t.scheduler.parts.kv.tree_cache
    stats = t.engine_stats()
    assert tree.protected_size() == 0, f"locked tree tokens {tree.protected_size()}"
    assert stats["kv_pool_free"] + tree.evictable_size() == kv_free0, (
        f"KV leak: free={stats['kv_pool_free']} cached={tree.evictable_size()} "
        f"baseline free={kv_free0}"
    )
    assert stats["req_pool_free"] == rows_free0, (
        f"req row leak: free={stats['req_pool_free']} baseline={rows_free0}"
    )


def _steps_holding(t: ScriptedContext, rid: str, mode: str) -> int:
    log = t._scheduler_hook._batch_log
    return sum(1 for record in log if record.mode == mode and rid in record.rids)


def _reference(t: ScriptedContext):
    """The greedy continuation, from a finish by length (the control case)."""
    baseline = _pool_baseline(t)
    r = _start(t, ignore_eos=True)
    req = yield from _capture_req(r)
    yield from run_until_finished(r)
    yield from _drain(t)

    n = len(req.output_ids)
    assert isinstance(req.finished_reason, FINISH_LENGTH), req.finished_reason
    assert n == _MAX_NEW_TOKENS, f"length finish produced {n} tokens"
    # The step that would have run past the last token is never scheduled.
    decode_steps = _steps_holding(t, r.rid, "decode")
    assert decode_steps == n - 1, f"{decode_steps=} for {n} output tokens"
    _assert_no_leak(t, baseline)
    tree_tokens = t.scheduler.parts.kv.tree_cache.evictable_size()
    assert tree_tokens == _PROMPT_LEN + n - 1, f"{tree_tokens=} for {n} outputs"

    t.flush_cache()
    yield
    _assert_no_leak(t, baseline)
    assert t.scheduler.parts.kv.tree_cache.evictable_size() == 0, "flush left tokens"
    return list(req.output_ids), req.tokenizer, set(req.eos_token_ids or ())


def _eos_ids(tokenizer, eos_ids):
    ids = set(eos_ids)
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    ids |= set(getattr(tokenizer, "additional_stop_token_ids", None) or ())
    return ids


def _first_stop_pos(ref, eos):
    return next((i for i, x in enumerate(ref) if x in eos), len(ref))


def _pick_stop_token(ref, eos) -> int:
    for k in range(_MIN_STOP_POS, _first_stop_pos(ref, eos)):
        if ref[k] not in ref[:k]:
            return k
    raise AssertionError(f"no fresh token to stop on in {ref}")


def _pick_stop_str(ref, eos, tokenizer):
    for k in range(_MIN_STOP_POS, _first_stop_pos(ref, eos)):
        s = tokenizer.decode([ref[k]]).strip()
        if (
            len(s) >= 2
            and s not in tokenizer.decode(ref[:k])
            and s in tokenizer.decode(ref[: k + 1])
        ):
            return k, s
    raise AssertionError(f"no fresh stop string in {ref}")


def _finish_and_check(t: ScriptedContext, r, req, baseline, *, case, check_tree):
    yield from run_until_finished(r)
    yield from _drain(t)
    _assert_no_leak(t, baseline)
    n = len(req.output_ids)
    decode_steps = _steps_holding(t, r.rid, "decode")
    tree_tokens = t.scheduler.parts.kv.tree_cache.evictable_size()
    if check_tree:
        assert tree_tokens == _PROMPT_LEN + n - 1, (
            f"{case}: radix tree holds {tree_tokens} tokens, expected "
            f"prompt + output - 1 = {_PROMPT_LEN + n - 1}"
        )
    return decode_steps


def _script_length(t: ScriptedContext):
    assert not get_schedule().disable_overlap_schedule, "overlap schedule is off"
    yield from _reference(t)


def _script_stop_token(t: ScriptedContext, check_tree: bool):
    assert not get_schedule().disable_overlap_schedule, "overlap schedule is off"
    ref, tokenizer, eos = yield from _reference(t)
    baseline = _pool_baseline(t)
    k = _pick_stop_token(ref, _eos_ids(tokenizer, eos))

    r = _start(t, stop_token_ids=[ref[k]])
    req = yield from _capture_req(r)
    decode_steps = yield from _finish_and_check(
        t, r, req, baseline, case="stop_token", check_tree=check_tree
    )
    assert isinstance(req.finished_reason, FINISH_MATCHED_TOKEN), req.finished_reason
    assert list(req.output_ids) == ref[: k + 1], (req.output_ids, ref[: k + 1])
    # One decode step past the stop was in flight when the stop was sampled.
    assert decode_steps == k + 1, f"{decode_steps=}, stop at output index {k}"


def _script_stop_str(t: ScriptedContext, check_tree: bool):
    assert not get_schedule().disable_overlap_schedule, "overlap schedule is off"
    ref, tokenizer, eos = yield from _reference(t)
    baseline = _pool_baseline(t)
    k, stop_str = _pick_stop_str(ref, _eos_ids(tokenizer, eos), tokenizer)

    r = _start(t, stop=[stop_str])
    req = yield from _capture_req(r)
    decode_steps = yield from _finish_and_check(
        t, r, req, baseline, case="stop_str", check_tree=check_tree
    )
    assert isinstance(req.finished_reason, FINISH_MATCHED_STR), req.finished_reason
    assert list(req.output_ids) == ref[: k + 1], (req.output_ids, ref[: k + 1])
    assert decode_steps == k + 1, f"{decode_steps=}, stop at output index {k}"


def _script_abort(t: ScriptedContext, check_tree: bool):
    assert not get_schedule().disable_overlap_schedule, "overlap schedule is off"
    ref, _, _ = yield from _reference(t)
    baseline = _pool_baseline(t)

    r = _start(t, ignore_eos=True)
    req = yield from _capture_req(r)
    yield from advance_to_decode_step(r, _ABORT_AT_OUTPUT_LEN)
    assert r.rid in t.batch_composition()["decode"], (
        "no decode step holding the request is in flight at the abort"
    )
    t.abort(r)
    yield from _finish_and_check(
        t, r, req, baseline, case="abort", check_tree=check_tree
    )
    assert isinstance(req.finished_reason, FINISH_ABORT), req.finished_reason
    n = len(req.output_ids)
    assert _ABORT_AT_OUTPUT_LEN <= n < _MAX_NEW_TOKENS, f"aborted after {n} tokens"
    assert list(req.output_ids) == ref[:n], (req.output_ids, ref[:n])


class TestScriptedOverlapFinish(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs()

    def test_length_finish_control(self):
        self.server.execute_script(_script_length)

    def test_stop_token_with_step_in_flight(self):
        self.server.execute_script(_script_stop_token, args=(False,))

    def test_stop_str_with_step_in_flight(self):
        self.server.execute_script(_script_stop_str, args=(False,))

    def test_abort_with_step_in_flight(self):
        self.server.execute_script(_script_abort, args=(False,))

    def test_tree_caches_committed_tokens_on_stop_token(self):
        self.server.execute_script(_script_stop_token, args=(True,))

    def test_tree_caches_committed_tokens_on_stop_str(self):
        self.server.execute_script(_script_stop_str, args=(True,))

    def test_tree_caches_committed_tokens_on_abort(self):
        self.server.execute_script(_script_abort, args=(True,))


if __name__ == "__main__":
    unittest.main()
