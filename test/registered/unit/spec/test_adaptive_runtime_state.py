import unittest

from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
    SpecRuntimeState,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _make_state(steps: int) -> SpecRuntimeState:
    return SpecRuntimeState(
        speculative_num_steps=steps,
        speculative_num_draft_tokens=steps + 1,
        draft_attn_backend=None,
        cuda_graph_runner=None,
        target_attn_backend=None,
        target_graph_runner=None,
        draft_extend_attn_backend=None,
        cuda_graph_runner_for_draft_extend=None,
    )


class _FakeWorker:
    def __init__(self, initial_steps: int = 3):
        self.speculative_num_steps = initial_steps
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
        return _make_state(speculative_num_steps)

    def apply_runtime_state(self, state: SpecRuntimeState) -> None:
        self.speculative_num_steps = state.speculative_num_steps
        self.applied_steps.append(state.speculative_num_steps)


class _FakePolicy:
    def __init__(self):
        self.candidate_steps = [1, 3]
        self.cuda_graph_bs = None
        self.feedback_step = None

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
                (3, 4, [1, 2]),
                (1, 2, [1]),
            ],
        )
        self.assertEqual(worker.applied_steps, [3])

    def test_states_are_built_widest_step_first_regardless_of_pruning(self):
        worker = _FakeWorker(initial_steps=3)
        policy = _FakePolicy()
        policy.candidate_steps = [1, 3, 7]
        bs_for_step = {7: [1, 2, 4], 3: [1, 2, 4, 8, 16], 1: [1, 2, 4, 8, 16, 32]}
        policy.cuda_graph_bs_for_step = lambda step: bs_for_step[step]
        controller = AdaptiveController(worker, policy)

        controller.init_states(cuda_graph_bs=[1, 2, 4, 8, 16, 32])

        self.assertEqual(
            worker.build_calls,
            [
                (7, 8, [1, 2, 4]),
                (3, 4, [1, 2, 4, 8, 16]),
                (1, 2, [1, 2, 4, 8, 16, 32]),
            ],
        )

    def test_states_are_built_by_descending_steps_without_cuda_graph(self):
        worker = _FakeWorker(initial_steps=3)
        policy = _FakePolicy()
        policy.candidate_steps = [1, 3, 7]
        controller = AdaptiveController(worker, policy)

        controller.init_states(cuda_graph_bs=None)

        self.assertEqual(worker.build_calls, [(7, 8, None), (3, 4, None), (1, 2, None)])
        self.assertEqual(worker.applied_steps, [3])

    def test_activation_is_idempotent_by_state_identity(self):
        worker = _FakeWorker(initial_steps=3)
        controller = AdaptiveController(worker, _FakePolicy())
        controller.init_states()

        controller._activate(3)
        controller._activate(1)
        controller._activate(1)

        self.assertEqual(worker.applied_steps, [3, 1])

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


if __name__ == "__main__":
    unittest.main()
