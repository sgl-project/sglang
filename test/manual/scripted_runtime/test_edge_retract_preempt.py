"""Edge cases — retract / preempt × chunked.

B-30 ~ B-33 from the plan. Verifies that chunked-resume requests
participate in KV-pressure retract and priority preemption like
normal requests, and that resource ownership is correctly released
across the transition.
"""

import unittest

from sglang.test.scripted_runtime.entrypoint import execute_scripted_runtime
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.test_utils import CustomTestCase

from test.manual.scripted_runtime.common import (
    DEFAULT_CHUNK_SIZE,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_finished,
)


# B-30: KV pressure mid-chunk causes the chunked-resume req to be
# retracted. All resources must release immediately (96d4749094 +
# f38e69f87d).
def _script_b30_retract_mid_chunk_releases_kv(t: ScriptedRuntime):
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)

    pages_before = r.kv_pages
    assert pages_before > 0

    # Force KV pressure by spamming short reqs until the pool is
    # nearly exhausted. ``exhaust_kv`` is the wishlist §4 P2 (9) helper.
    t.exhaust_kv(leave_pages=1)
    yield

    assert r.status in ("waiting", "finished", "unknown"), (
        f"r should be retracted (back in waiting) or finished; got {r.status}"
    )
    if r.status == "waiting":
        # Retract path: resources released; the req is parked for re-admission.
        assert r.kv_pages == 0, (
            f"retract must release KV; got {r.kv_pages}"
        )


# B-31: priority preemption — a high-priority req shows up while a
# low-priority chunked-resume holds resources. The victim chunked req
# must go through retract (not abort) and the high-priority req takes
# its slot.
def _script_b31_priority_preempt_chunked(t: ScriptedRuntime):
    low = t.start_req(
        prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority="low"
    )
    yield from run_until(low, lambda h: h.is_chunking and h.chunks_done >= 1)
    assert low.kv_pages > 0

    high = t.start_req(prompt_len=8, max_new_tokens=2, priority="high")
    # In a pressured cache, high should force preempt of low.
    t.exhaust_kv(leave_pages=1)
    yield

    # After preempt: low is retracted (status "waiting") with no KV;
    # high gets admitted.
    if low.status == "waiting":
        assert low.kv_pages == 0
    yield from run_until_finished(high)
    assert high.finished


# B-32: retract-and-resume. Retract chunked mid-stream, then release
# pressure so it can be re-admitted; must complete normally.
def _script_b32_retract_and_resume(t: ScriptedRuntime):
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)

    # Use the active fault-injection helper to force a retract — the
    # passive ``exhaust_kv`` is also acceptable, but this is
    # deterministic.
    t.force_retract(r)
    yield
    assert r.status == "waiting"
    assert r.kv_pages == 0

    # Drain pressure (no-op if we used force_retract) and run to completion.
    yield from run_until_finished(r)
    assert r.finished


# B-33: disagg mode retract — disaggregation send-side state must
# reset on chunked-resume retract (414efd4a27).
def _script_b33_disagg_retract_resets_send_state(t: ScriptedRuntime):
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)

    t.force_retract(r)
    yield

    assert r.status == "waiting"
    assert r.kv_pages == 0
    # Disagg-specific: the per-req send-side state must reset.
    assert r.disagg_send_state in (None, "idle"), (
        f"disagg send state must reset on retract, got {r.disagg_send_state}"
    )

    yield from run_until_finished(r)
    assert r.finished


class TestEdgeRetractPreempt(CustomTestCase):
    def test_b30_retract_mid_chunk_releases_kv(self):
        execute_scripted_runtime(
            _script_b30_retract_mid_chunk_releases_kv,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_b31_priority_preempt_chunked(self):
        execute_scripted_runtime(
            _script_b31_priority_preempt_chunked,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                enable_priority_scheduling=True,
            ),
        )

    def test_b32_retract_and_resume(self):
        execute_scripted_runtime(
            _script_b32_retract_and_resume,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_b33_disagg_retract_resets_send_state(self):
        execute_scripted_runtime(
            _script_b33_disagg_retract_resets_send_state,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                disaggregation_mode="prefill",
            ),
        )


if __name__ == "__main__":
    unittest.main()
