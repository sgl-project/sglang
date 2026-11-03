import unittest

from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
    SpecRuntimeState,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_state(steps: int, gpu_memory_gb: float = 0.0) -> SpecRuntimeState:
    return SpecRuntimeState(
        speculative_num_steps=steps,
        speculative_num_draft_tokens=steps + 1,
        draft_attn_backend=None,
        cuda_graph_runner=None,
        target_attn_backend=None,
        target_graph_runner=None,
        draft_extend_attn_backend=None,
        cuda_graph_runner_for_draft_extend=None,
        gpu_memory_gb=gpu_memory_gb,
    )


class _FakeWorker:
    def __init__(self, initial_steps: int = 3, cost_gb: dict[int, float] | None = None):
        self.speculative_num_steps = initial_steps
        self.cost_gb = cost_gb or {}
        self.build_calls = []
        self.applied_steps = []

    def build_adaptive_runtime_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs: list[int] | None = None,
    ) -> SpecRuntimeState:
        self.build_calls.append(
            (speculative_num_steps, speculative_num_draft_tokens, cuda_graph_bs)
        )
        return _make_state(
            speculative_num_steps, self.cost_gb.get(speculative_num_steps, 0.0)
        )

    def apply_runtime_state(self, state: SpecRuntimeState) -> None:
        self.speculative_num_steps = state.speculative_num_steps
        self.applied_steps.append(state.speculative_num_steps)


class _FakePolicy:
    def __init__(
        self,
        candidate_steps: list[int] | None = None,
        start_steps: list[int] | None = None,
    ):
        self.candidate_steps = candidate_steps or [1, 3]
        self.start_steps = start_steps or []
        self.cuda_graph_bs = None
        self.feedback_step = None
        self.restricted_to = None

    @property
    def required_steps(self) -> list[int]:
        return sorted(set(self.start_steps))

    def prebuild_order(self) -> list[int]:
        required = self.required_steps
        return required + [s for s in self.candidate_steps if s not in required]

    def restrict_candidates(self, allowed: set[int]) -> list[int]:
        self.restricted_to = set(allowed)
        dropped = [s for s in self.candidate_steps if s not in allowed]
        self.candidate_steps = [s for s in self.candidate_steps if s in allowed]
        return dropped

    def set_cuda_graph_bs(self, cuda_graph_bs: list[int] | None) -> None:
        self.cuda_graph_bs = cuda_graph_bs

    def get_steps_for_batch(self, batch_size: int) -> int:
        return 1 if batch_size >= 8 else 3

    def on_verify_complete(
        self, num_correct_drafts_per_req: list[int], batch_size: int
    ) -> int | None:
        return self.feedback_step

    def cuda_graph_bs_for_step(self, step: int) -> list[int] | None:
        if self.cuda_graph_bs is None:
            return None
        return [batch_size for batch_size in self.cuda_graph_bs if batch_size <= step]


class TestAdaptiveController(unittest.TestCase):
    def test_injected_policy_builds_pruned_states_and_applies_initial_state(self):
        worker = _FakeWorker(initial_steps=3)
        policy = _FakePolicy()
        controller = AdaptiveController(worker, policy)

        controller.init_states(cuda_graph_bs=[1, 2, 4])

        self.assertIs(controller.params, policy)
        self.assertEqual(policy.cuda_graph_bs, [1, 2, 4])
        self.assertEqual(
            worker.build_calls,
            [
                (1, 2, [1]),
                (3, 4, [1, 2]),
            ],
        )
        self.assertEqual(worker.applied_steps, [3])

    def test_registered_initial_state_is_reused(self):
        worker = _FakeWorker(initial_steps=3)
        controller = AdaptiveController(worker, _FakePolicy())
        controller.register(_make_state(3))

        controller.init_states()

        self.assertEqual(worker.build_calls, [(1, 2, None)])
        self.assertEqual(worker.applied_steps, [3])

    def test_batch_activation_is_idempotent(self):
        worker = _FakeWorker(initial_steps=3)
        controller = AdaptiveController(worker, _FakePolicy())
        controller.init_states()

        controller.activate_step_by_batch(batch_size=8)
        controller.activate_step_by_batch(batch_size=8)
        self.assertEqual(worker.applied_steps, [3, 1])

    def test_verify_feedback_can_activate_a_state(self):
        worker = _FakeWorker(initial_steps=3)
        policy = _FakePolicy()
        controller = AdaptiveController(worker, policy)
        controller.init_states()

        controller.on_verify_complete([1], batch_size=1)
        policy.feedback_step = 1
        controller.on_verify_complete([1], batch_size=1)
        self.assertEqual(worker.applied_steps, [3, 1])

    def test_missing_state_fails_loud(self):
        worker = _FakeWorker(initial_steps=3)
        controller = AdaptiveController(worker, _FakePolicy())
        controller.init_states()
        del controller._states[1]

        with self.assertRaisesRegex(
            ValueError, "Missing adaptive runtime state for steps=1"
        ):
            controller.activate_step_by_batch(batch_size=8)

    def test_memory_budget_builds_start_steps_and_stops_once_spent(self):
        """The slots' start steps are built whatever the budget, since the first
        batch routed to a slot activates them; optional steps are built in policy
        order until the spent cost reaches the budget (the last one may overshoot),
        and the policy is restricted to what was built."""
        worker = _FakeWorker(initial_steps=3, cost_gb={0: 4.0, 1: 1.0, 5: 3.0, 7: 4.0})
        policy = _FakePolicy(candidate_steps=[0, 1, 3, 5, 7], start_steps=[3, 1, 0])
        controller = AdaptiveController(worker, policy)
        controller.register(_make_state(3, gpu_memory_gb=100.0))

        controller.init_states(memory_budget_gb=3.0)

        # Required 0 and 1 cost 5.0 but are not charged; 5 is built because
        # nothing was spent yet, after which the spent 3.0 stops 7.
        self.assertEqual([c[0] for c in worker.build_calls], [0, 1, 5])
        self.assertEqual(policy.restricted_to, {0, 1, 3, 5})
        self.assertEqual(policy.candidate_steps, [0, 1, 3, 5])
        self.assertEqual(worker.applied_steps, [3])

    def test_zero_memory_budget_builds_only_required_states(self):
        worker = _FakeWorker(initial_steps=3)
        policy = _FakePolicy(candidate_steps=[0, 1, 3, 5], start_steps=[3, 0])
        controller = AdaptiveController(worker, policy)
        controller.register(_make_state(3))

        controller.init_states(memory_budget_gb=0.0)

        self.assertEqual([c[0] for c in worker.build_calls], [0])
        self.assertEqual(policy.candidate_steps, [0, 3])

    def test_unset_budget_builds_every_candidate(self):
        worker = _FakeWorker(initial_steps=3, cost_gb={1: 50.0, 5: 50.0})
        policy = _FakePolicy(candidate_steps=[1, 3, 5], start_steps=[3])
        controller = AdaptiveController(worker, policy)
        controller.register(_make_state(3))

        controller.init_states()

        self.assertEqual([c[0] for c in worker.build_calls], [1, 5])
        self.assertEqual(policy.candidate_steps, [1, 3, 5])


if __name__ == "__main__":
    unittest.main()
