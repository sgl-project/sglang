"""CUDA D2H lifetime and GPU next-draft state for the PP result channel."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.dspark_pp_result_utils import (
    check_pp_result,
    make_pp_result_fixture,
    receive_pp_result,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestDSparkPPResultCUDA(CustomTestCase):
    def exercise(self, *, prefill):
        scheduler, batch, source, observed = make_pp_result_fixture(
            prefill=prefill, device="cuda"
        )
        with patch(
            "sglang.srt.managers.scheduler_pp_mixin.get_disagg",
            return_value=SimpleNamespace(disaggregation_mode="null"),
        ):
            packet = SchedulerPPMixin._pp_prepare_tensor_dict(scheduler, source, batch)
            producer = torch.cuda.current_stream()
            copy_stream = torch.cuda.Stream()
            with torch.cuda.stream(copy_stream):
                copy_stream.wait_stream(producer)
                result = receive_pp_result(scheduler, batch, packet)
                self.assertTrue(result.next_token_ids.is_pinned())
                if not prefill:
                    self.assertTrue(result.accept_lens.is_pinned())
                    self.assertTrue(result.block_accept_lens.is_pinned())
                    self.assertTrue(result.cap_lens.is_pinned())
                self.assertTrue(result.new_seq_lens.is_cuda)
                self.assertTrue(result.next_draft_input.bonus_tokens.is_cuda)
                check_pp_result(scheduler, batch, result, observed, prefill=prefill)
            torch.cuda.synchronize()

    def test_prefill_result(self):
        self.exercise(prefill=True)

    def test_verify_result(self):
        self.exercise(prefill=False)


if __name__ == "__main__":
    unittest.main()
