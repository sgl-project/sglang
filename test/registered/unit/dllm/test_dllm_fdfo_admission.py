import unittest
from array import array
from types import SimpleNamespace

from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDllmFdfoAdmission(CustomTestCase):
    def test_retained_block_uses_existing_extend_range(self):
        for fdfo, incomplete in (
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        ):
            with self.subTest(fdfo=fdfo, result_processed=incomplete):
                req = SimpleNamespace(
                    full_untruncated_fill_ids=array("q", [1] * 6 + [0] * 32),
                    prefix_len=0,
                    extend_end=32,
                    extend_len=32,
                    dllm_incomplete_ids=array("q", [1] * 32)
                    if incomplete
                    else array("q"),
                    dllm_block_done=False,
                    kv=SimpleNamespace(holds_kv=True),
                    sampling_params=SimpleNamespace(max_new_tokens=128),
                    retracted_stain=False,
                )

                adder = SimpleNamespace(
                    dllm_config=SimpleNamespace(
                        first_done_first_out_mode=fdfo,
                        requires_separate_context_encoding=False,
                        block_size=32,
                    ),
                    _get_dllm_remain_tokens=lambda req: 32,
                    can_run_list=[],
                    _update_prefill_budget=lambda *args, **kwargs: None,
                    _mamba_gap_budget_for_req=lambda req: 0,
                )
                result = PrefillAdder.add_dllm_staging_req(adder, req)
                self.assertEqual(result, AddReqResult.CONTINUE)
                self.assertEqual(adder.can_run_list, [req])
                self.assertEqual(req.extend_end, 32)

                adder.can_run_list.clear()
                adder._get_dllm_remain_tokens = lambda req: 16
                result = PrefillAdder.add_dllm_staging_req(adder, req)
                self.assertEqual(result, AddReqResult.NO_TOKEN)
                self.assertEqual(adder.can_run_list, [])
                self.assertEqual(req.extend_end, 32)

    def test_next_block_defers_input_initialization_until_admission(self):
        req = SimpleNamespace(
            prefix_len=4,
            extend_end=4,
            extend_len=4,
            origin_input_ids=array("q", [2, 3]),
            output_ids=array("q", [4, 5]),
            full_untruncated_fill_ids=array("q", [2, 3, 4, 5, 0, 0]),
            dllm_incomplete_ids=array("q"),
            dllm_block_done=True,
            dllm_block_id=7,
            kv=SimpleNamespace(holds_kv=True),
            sampling_params=SimpleNamespace(max_new_tokens=32),
            retracted_stain=False,
        )
        budget = [0]
        adder = SimpleNamespace(
            dllm_config=SimpleNamespace(
                block_size=4, requires_separate_context_encoding=False
            ),
            _get_dllm_remain_tokens=lambda req: budget[0],
            can_run_list=[],
            _update_prefill_budget=lambda *args, **kwargs: budget.__setitem__(
                0, budget[0] - args[1]
            ),
            _mamba_gap_budget_for_req=lambda req: 0,
        )
        for _ in range(2):
            result = PrefillAdder.add_dllm_staging_req(adder, req)
            self.assertEqual(result, AddReqResult.NO_TOKEN)
            self.assertEqual(adder.can_run_list, [])
            self.assertEqual(budget[0], 0)
            self.assertEqual((req.prefix_len, req.extend_end), (4, 4))
        budget[0] = 4
        result = PrefillAdder.add_dllm_staging_req(adder, req)
        self.assertEqual(result, AddReqResult.NO_TOKEN)
        self.assertEqual(adder.can_run_list, [req])
        self.assertEqual(budget[0], 0)
        self.assertEqual((req.prefix_len, req.extend_end), (4, 8))
        self.assertEqual(list(req.full_untruncated_fill_ids), [2, 3, 4, 5, 0, 0])
        self.assertEqual(req.dllm_block_id, 7)
        self.assertTrue(req.dllm_block_done)


if __name__ == "__main__":
    unittest.main()
