import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.prefill_delayer import (
    PrefillDelayer,
    PrefillDelayerSinglePassExecutor,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

DP_SIZE = 2
NUM_FIELDS = 6
IN_FLIGHT_FIELD = 5


class TestPrefillDelayerPhaseLockstep(CustomTestCase):
    """Pins the `is_phase_prefill` contract the DP phase lockstep relies on:
    in-flight chunk on any rank OR (allow AND num_prefillable > 0)."""

    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        override = get_context().override_server_args(enable_dp_attention=True)
        override.install()
        self.addCleanup(override.restore)
        self.delayer = PrefillDelayer(
            dp_size=DP_SIZE,
            attn_tp_size=1,
            cpu_group=MagicMock(),
            max_delay_passes=100,
            token_usage_low_watermark=None,
        )
        self.gathered_local_rows = []

    def _run_pass(self, *, remote_rows, local_prefillable, prefill_in_flight=False):
        """Negotiate one pass as DP rank 0 with the other ranks' gather rows
        fixed to `remote_rows` (each a NUM_FIELDS-long list)."""

        def fake_all_gather(out_flat, local_info, group=None):
            self.gathered_local_rows.append(local_info.tolist())
            rows = [local_info.tolist()] + list(remote_rows)
            self.delayer._global_info_buffer.copy_(
                torch.tensor(rows, dtype=torch.int64).view(DP_SIZE, 1, NUM_FIELDS)
            )

        executor = PrefillDelayerSinglePassExecutor(self.delayer, token_usage=0.9)
        with patch("torch.distributed.all_gather_into_tensor", fake_all_gather):
            executor.negotiate_should_allow_prefill(
                local_prefillable=local_prefillable,
                prefill_in_flight=prefill_in_flight,
            )
            executor.finalize(actual_prefill_bs=0)
        return executor

    def test_in_flight_on_remote_rank_forces_phase_prefill_even_when_delayed(self):
        # Rank 1 continues a chunked prefill; rank 0 wants a new prefill.
        # The delayer sees "mixed" and delays rank 0 (allow=False), but the
        # pass is still a prefill pass because rank 1 is mid-chunk.
        executor = self._run_pass(
            remote_rows=[[0, 0, 0, 0, 0, 1]], local_prefillable=True
        )
        self.assertFalse(executor._result.output_allow)
        self.assertTrue(executor._result.any_prefill_in_flight)
        self.assertTrue(executor.is_phase_prefill)

    def test_local_in_flight_bit_is_gathered_and_forces_phase_prefill(self):
        executor = self._run_pass(
            remote_rows=[[0, 0, 0, 0, 0, 0]],
            local_prefillable=True,
            prefill_in_flight=True,
        )
        self.assertEqual(self.gathered_local_rows[-1][IN_FLIGHT_FIELD], 1)
        self.assertTrue(executor.is_phase_prefill)

    def test_nobody_prefillable_is_decode_phase(self):
        # "none" allows for simplicity, but with num_prefillable == 0 nobody
        # runs prefill, so the phase is decode.
        executor = self._run_pass(
            remote_rows=[[0, 0, 0, 0, 0, 0]], local_prefillable=False
        )
        self.assertTrue(executor._result.output_allow)
        self.assertEqual(executor._result.num_prefillable, 0)
        self.assertFalse(executor.is_phase_prefill)

    def test_delayed_mixed_pass_without_in_flight_is_decode_phase(self):
        # Wanting to prefill is not prefilling: a delayed "mixed" pass with no
        # chunk in flight anywhere runs decode on every rank.
        executor = self._run_pass(
            remote_rows=[[0, 0, 0, 0, 0, 0]], local_prefillable=True
        )
        self.assertFalse(executor._result.output_allow)
        self.assertEqual(executor._result.num_prefillable, 1)
        self.assertFalse(executor.is_phase_prefill)

    def test_allowed_prefill_is_prefill_phase(self):
        executor = self._run_pass(
            remote_rows=[[1, 0, 0, 0, 0, 0]], local_prefillable=True
        )
        self.assertTrue(executor._result.output_allow)
        self.assertTrue(executor.is_phase_prefill)


if __name__ == "__main__":
    unittest.main()
