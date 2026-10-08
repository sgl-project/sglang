import unittest
from array import array
from types import SimpleNamespace

from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDllmFdfoAdmission(unittest.TestCase):
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
                    prefix_indices=[],
                    extend_range=SimpleNamespace(start=0, end=32, length=32),
                    dllm_incomplete_ids=array("q", [1] * 32)
                    if incomplete
                    else array("q"),
                    dllm_block_done=False,
                    kv=SimpleNamespace(holds_kv=True),
                    sampling_params=SimpleNamespace(max_new_tokens=128),
                    retracted_stain=False,
                )

                def set_range(start, end):
                    req.extend_range = SimpleNamespace(
                        start=start, end=end, length=end - start
                    )

                req.set_extend_range = set_range
                adder = SimpleNamespace(
                    dllm_config=SimpleNamespace(first_done_first_out_mode=fdfo),
                    _get_dllm_remain_tokens=lambda: 32,
                    can_run_list=[],
                    _update_prefill_budget=lambda *args, **kwargs: None,
                    _mamba_gap_budget_for_req=lambda req: 0,
                )
                result = PrefillAdder.add_dllm_staging_req(adder, req)
                self.assertEqual(result, AddReqResult.CONTINUE)
                self.assertEqual(adder.can_run_list, [req])
                self.assertEqual(req.extend_range.end, 32)

                adder.can_run_list.clear()
                adder._get_dllm_remain_tokens = lambda: 16
                result = PrefillAdder.add_dllm_staging_req(adder, req)
                self.assertEqual(result, AddReqResult.NO_TOKEN)
                self.assertEqual(adder.can_run_list, [])
                self.assertEqual(req.extend_range.end, 32)


if __name__ == "__main__":
    unittest.main()
