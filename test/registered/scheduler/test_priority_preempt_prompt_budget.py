import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import base_engine_kwargs

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-small")

# Regression test for the priority-preemption budget bug.
#
# preempt_to_schedule seeds min_tokens_to_remove with the incoming request's
# extend_input_len. extend_input_len is only populated by init_next_round_input,
# which runs *after* the preemption check in the get_new_batch_prefill
# waiting-queue loop (and is otherwise pre-populated only when hicache storage
# is enabled). A freshly arrived request therefore reaches preempt_to_schedule
# with extend_input_len == 0, so its prompt tokens were dropped from the budget.
#
# Consequence exercised here: a high-priority request whose prompt is too large
# to ever be admitted by preempting the available victims should preempt NOBODY
# (preempting them cannot free enough budget). The bug drops the prompt term, so
# the budget collapses to ~max_new_tokens, the check passes, and a running
# low-priority victim is evicted pointlessly.
#
# The pre-existing TestPriorityPreempt only ever uses an 8-token high-priority
# prompt, where the dropped term is negligible, so it never caught this.

NUM_VICTIMS = 4
VICTIM_PROMPT_LEN = 8
VICTIM_MAX_NEW = 512
HIGH_PROMPT_LEN = 4096
HIGH_MAX_NEW = 4
FREE_KV_TARGET = 1024
OBSERVE_STEPS = 8


class TestPriorityPreemptPromptBudget(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=8192,
        enable_priority_scheduling=True,
        max_running_requests=NUM_VICTIMS,
        priority_scheduling_preemption_threshold=0,
        max_total_tokens=8192,
    )

    def test_oversized_high_priority_prompt_does_not_evict_unreplaceable_victims(self):
        self.server.execute_script(self._script)

    @staticmethod
    def _script(t: ScriptedContext):
        victims = [
            t.start_req(
                prompt_len=VICTIM_PROMPT_LEN,
                max_new_tokens=VICTIM_MAX_NEW,
                priority=0,
                ignore_eos=True,
            )
            for _ in range(NUM_VICTIMS)
        ]
        for victim in victims:
            yield from _run_until(victim, lambda h: h.status == "running")

        page_size = t.engine_stats()["page_size"]
        t.exhaust_kv(leave_pages=max(1, FREE_KV_TARGET // page_size))

        t.start_req(
            prompt_len=HIGH_PROMPT_LEN,
            max_new_tokens=HIGH_MAX_NEW,
            priority=10,
        )

        preempted_victims = set()
        for _ in range(OBSERVE_STEPS):
            yield
            for victim in victims:
                if victim.status == "waiting":
                    preempted_victims.add(victim.rid)

        assert not preempted_victims, (
            f"a {HIGH_PROMPT_LEN}-token high-priority prompt that cannot be fit "
            f"even by preempting all {NUM_VICTIMS} victims must evict none of "
            f"them, but {len(preempted_victims)} were preempted; this is the "
            f"stale extend_input_len budget bug dropping the prompt tokens"
        )


def _run_until(handle, predicate, *, max_steps: int = 400):
    for _ in range(max_steps):
        if predicate(handle):
            return
        yield
    raise AssertionError(
        f"_run_until: predicate never satisfied after {max_steps} steps "
        f"(rid={handle.rid!r}, status={handle.status!r})"
    )


if __name__ == "__main__":
    unittest.main()
