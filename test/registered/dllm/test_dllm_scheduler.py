import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.dllm.algorithm.fastdiffuser import FastDiffuser
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class TestFastDiffuserState(unittest.TestCase):
    def test_causal_kv_flag_is_restored_after_forward_failure(self):
        config = DllmConfig(
            algorithm="FastDiffuser",
            algorithm_config={},
            block_size=32,
            mask_id=100,
            max_running_requests=1,
            max_steps=32,
            causal_context=True,
        )
        algorithm = FastDiffuser(config)
        forward_batch = SimpleNamespace(dllm_causal_kv_update=False)
        model_runner = SimpleNamespace(
            forward=Mock(side_effect=RuntimeError("forward failed"))
        )

        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            algorithm._forward_for_kv_update(model_runner, forward_batch)

        self.assertFalse(forward_batch.dllm_causal_kv_update)


class TestDllmPromptAdmission(unittest.TestCase):
    def test_prompt_cache_uses_current_range_and_budget_signature(self):
        update_budget = Mock()
        adder = SimpleNamespace(
            rem_total_tokens=100,
            rem_input_tokens=100,
            can_run_list=[],
            ceil_paged_tokens=lambda tokens: tokens,
            _update_prefill_budget=update_budget,
        )
        req = SimpleNamespace(
            extend_range=SimpleNamespace(length=4),
            sampling_params=SimpleNamespace(max_new_tokens=8),
            output_ids=[],
            prefix_indices=[],
            retracted_stain=True,
        )

        result = PrefillAdder.add_dllm_prompt_cache_req(adder, req)

        self.assertEqual(result, AddReqResult.CONTINUE)
        self.assertEqual(adder.can_run_list, [req])
        update_budget.assert_called_once_with(0, 4, 8, True)

    def test_disabled_priority_preemption_does_not_preempt(self):
        preempt = Mock(return_value=True)
        adder = SimpleNamespace(can_run_list=[], preempt_to_schedule=preempt)
        scheduler = SimpleNamespace(
            running_batch=SimpleNamespace(reqs=[], batch_is_full=True),
            get_num_allocatable_reqs=lambda _running_bs: 0,
            enable_priority_preemption=False,
            server_args=SimpleNamespace(),
        )
        req = Mock()

        SchedulerDllmMixin._process_incoming_prefill_reqs(scheduler, adder, [req])

        preempt.assert_not_called()
        req.init_prompt_cache_input.assert_not_called()


if __name__ == "__main__":
    unittest.main()
