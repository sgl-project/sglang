from __future__ import annotations

import os
import unittest
from typing import ClassVar, Dict, Optional

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import base_engine_kwargs, run_until

register_cuda_ci(est_time=240, stage="extra-a", runner_config="1-gpu-small")

# Regression test for the priority-preemption prompt-budget bug (PR #28282).
#
# preempt_to_schedule seeds min_tokens_to_remove from the incoming request's
# extend_input_len, which is still 0 for a freshly arrived request (it is only
# populated by init_next_round_input, which runs after this preemption check).
# The prompt tokens are therefore dropped from the budget.
#
# Scenario: one low-priority victim is running, the KV pool is held nearly full
# (exhaust_kv), and a high-priority request arrives with a prompt far larger
# than the free pool. Preempting the single victim frees only a handful of
# tokens, so the request can never fit and preemption must be refused. With the
# prompt tokens dropped the budget collapses to ~max_new_tokens, the check
# passes, and the running victim is evicted pointlessly.
#
# The prompt is kept <= max_total_tokens so it is admitted as a normal request
# (a prompt larger than the pool is rejected/truncated and never reaches the
# preemption path). Pressure comes from exhaust_kv, not from an oversized prompt.

_VICTIM_PROMPT_LEN = 8
_VICTIM_MAX_NEW = 256
_HIGH_PROMPT_LEN = 2000
_HIGH_MAX_NEW = 4
_LEAVE_KV_TOKENS = 1024
_OBSERVE_STEPS = 16


def _script_oversized_preempt(t: ScriptedContext, expect_preempt: bool):
    low = t.start_req(
        prompt_len=_VICTIM_PROMPT_LEN,
        max_new_tokens=_VICTIM_MAX_NEW,
        priority=0,
        ignore_eos=True,
    )
    yield from run_until(low, lambda h: h.status == "running")

    page_size = t.engine_stats()["page_size"]
    t.exhaust_kv(leave_pages=max(1, _LEAVE_KV_TOKENS // page_size))

    t.start_req(
        prompt_len=_HIGH_PROMPT_LEN,
        max_new_tokens=_HIGH_MAX_NEW,
        priority=10,
    )

    # The victim is "running" (decoding) when high arrives. If preempted it
    # leaves the running batch (retracted -> waiting or aborted), i.e. its
    # status stops being "running". A non-preempted victim keeps decoding (it is
    # far from its max_new_tokens within this window), so it stays "running".
    preempted = False
    for _ in range(_OBSERVE_STEPS):
        yield
        if low.status != "running":
            preempted = True
            break

    assert preempted == expect_preempt, (
        f"low-priority victim preempted={preempted}, expected {expect_preempt}: a "
        f"{_HIGH_PROMPT_LEN}-token prompt cannot fit the pressured KV pool even "
        f"after preempting the victim, so preemption must be refused when the "
        f"budget counts the prompt tokens (the stale extend_input_len bug drops "
        f"them and evicts the running victim pointlessly)"
    )


class _PreemptPromptBudgetBase(ScriptedTestCase):
    revert_pr: ClassVar[bool]
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=8192,
        enable_priority_scheduling=True,
        max_running_requests=1,
        priority_scheduling_preemption_threshold=0,
        max_total_tokens=8192,
    )
    _env_backup: ClassVar[Dict[str, Optional[str]]] = {}

    @classmethod
    def setUpClass(cls) -> None:
        if cls is _PreemptPromptBudgetBase:
            raise unittest.SkipTest("abstract base; concrete subclasses set revert_pr")
        env: Dict[str, str] = {}
        if cls.revert_pr:
            env["SGLANG_DEBUG_REVERT_PR"] = "28282"
        cls._env_backup = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            super().tearDownClass()
        finally:
            for key, old in cls._env_backup.items():
                if old is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old

    def _run(self, *, expect_preempt: bool) -> None:
        self.server.execute_script(_script_oversized_preempt, args=(expect_preempt,))


class TestPreemptPromptBudgetRegression(_PreemptPromptBudgetBase):
    revert_pr = True

    def test_unfittable_prompt_wrongly_preempts_victim(self) -> None:
        self._run(expect_preempt=True)


class TestPreemptPromptBudgetClean(_PreemptPromptBudgetBase):
    revert_pr = False

    def test_unfittable_prompt_spares_victim(self) -> None:
        self._run(expect_preempt=False)


if __name__ == "__main__":
    unittest.main()
