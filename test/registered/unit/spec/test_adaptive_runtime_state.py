import unittest
from unittest.mock import patch

from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
    SpecRuntimeState,
    _broadcast_profile_latency_from_tp_rank0,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


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
        self,
        num_correct_drafts_per_req: list[int],
        batch_size: int,
        num_steps: int | None = None,
    ) -> int | None:
        return self.feedback_step

    def cuda_graph_bs_for_step(self, step: int) -> list[int] | None:
        if self.cuda_graph_bs is None:
            return None
        return [batch_size for batch_size in self.cuda_graph_bs if batch_size <= step]

    def on_state_activated(self, steps: int) -> None:
        pass


class TestAdaptiveController(unittest.TestCase):
    def test_from_config_selects_policy(self):
        worker = _FakeWorker()

        for strategy, policy_path, expected_keyword in (
            (
                "ema",
                "sglang.srt.speculative.adaptive_spec_params.AdaptiveSpeculativeParams",
                "cfg_path",
            ),
            (
                "throughput_aware",
                "sglang.srt.speculative.throughput_aware_controller.ThroughputAwarePolicy",
                "config_path",
            ),
        ):
            with (
                self.subTest(strategy=strategy),
                patch(
                    "sglang.srt.speculative.adaptive_spec_params.resolve_adaptive_strategy",
                    return_value=strategy,
                ),
                patch(policy_path, return_value=_FakePolicy()) as policy_cls,
            ):
                controller = AdaptiveController.from_config(
                    worker,
                    initial_steps=3,
                    config_path="adaptive.json",
                )

            self.assertIsInstance(controller, AdaptiveController)
            policy_cls.assert_called_once_with(
                initial_steps=3,
                **{expected_keyword: "adaptive.json"},
            )

    def test_profile_latency_uses_tp_group_broadcast(self):
        class FakeTPGroup:
            device = "cpu"

            def __init__(self):
                self.src = None

            def broadcast(self, value, src=0):
                self.src = src
                value.fill_(7.5)

        tp_group = FakeTPGroup()
        with (
            patch(
                "sglang.srt.distributed.model_parallel_is_initialized",
                return_value=True,
            ),
            patch("sglang.srt.distributed.get_tp_group", return_value=tp_group),
        ):
            value = _broadcast_profile_latency_from_tp_rank0(2.5)

        self.assertEqual(tp_group.src, 0)
        self.assertEqual(value, 7.5)

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


if __name__ == "__main__":
    unittest.main()
