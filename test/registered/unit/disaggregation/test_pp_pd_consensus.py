import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.base import KVPoll  # noqa: E402
from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue  # noqa: E402
from sglang.srt.managers.scheduler_pp_mixin import (  # noqa: E402
    _pp_merge_transfer_status,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPPPDConsensus(CustomTestCase):
    def test_transfer_failure_overrides_ordered_success_intersection(self):
        """A failure on one PP rank must terminate an otherwise successful rid."""
        status = _pp_merge_transfer_status(
            previous=(["req-a", "req-b", "req-c"], ["req-x"]),
            current=(["req-c", "req-a", "req-b"], ["req-b", "req-y"]),
        )

        self.assertEqual(
            status,
            (["req-a", "req-c"], ["req-x", "req-b", "req-y"]),
        )

    def test_bootstrap_probe_respects_local_metadata_credit_prefix(self):
        """A slower PP rank must not advertise requests it cannot admit."""
        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue.queue = [
            SimpleNamespace(
                rid="req-failed",
                metadata_buffer_index=-1,
                disagg_kv_sender=object(),
            ),
            SimpleNamespace(
                rid="req-ready",
                metadata_buffer_index=-1,
                disagg_kv_sender=object(),
            ),
            SimpleNamespace(
                rid="req-blocked",
                metadata_buffer_index=-1,
                disagg_kv_sender=object(),
            ),
        ]
        queue.scheduler = SimpleNamespace(
            attn_cp_cpu_group=object(),
            attn_tp_cpu_group=object(),
        )
        queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
            available_size=lambda: 1
        )

        with patch(
            "sglang.srt.disaggregation.prefill."
            "poll_and_all_reduce_attn_cp_tp_group",
            return_value=[
                KVPoll.Failed,
                KVPoll.WaitingForInput,
                KVPoll.WaitingForInput,
            ],
        ):
            good_rids, failed_rids = queue.get_ready_bootstrapped_rids_for_pp()

        self.assertEqual(good_rids, ["req-ready"])
        self.assertEqual(failed_rids, ["req-failed"])
        self.assertEqual(
            [req.metadata_buffer_index for req in queue.queue],
            [-1, -1, -1],
        )


if __name__ == "__main__":
    unittest.main()
