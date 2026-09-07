from __future__ import annotations

import unittest

from sglang.srt.kv_canary.capacities import CanaryLaunchCapacities
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestComputeLaunchCapacities(CustomTestCase):
    def _from_args(
        self,
        *,
        max_bs: int,
        max_seq_len: int,
        max_total_num_tokens: int | None = None,
        speculative_num_draft_tokens: int | None = 0,
    ) -> CanaryLaunchCapacities:
        """`from_args` reads the published configuration, so publish one.

        Handing it a stand-in object stopped meaning anything when the reads
        moved to the config bags: the parameter was ignored and the values
        under test came from whatever the process had published.
        """
        if max_total_num_tokens is None:
            max_total_num_tokens = max_bs * max_seq_len
        override = get_context().override_server_args(
            cuda_graph_config=CudaGraphConfig(
                decode=PhaseConfig(backend=Backend.FULL, max_bs=max_bs)
            ),
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            chunked_prefill_size=None,
            max_prefill_tokens=128,
        )
        override.install()
        self.addCleanup(override.restore)
        return CanaryLaunchCapacities.from_args(
            req_to_token_pool_size=max_bs,
            max_seq_len_per_req=max_seq_len,
            pool_slot_count=max_total_num_tokens,
        )

    def test_per_forward_verify_capacity_covers_multi_req_prefix_sum(self) -> None:
        """Verify per-forward verify capacity equals max_total_num_tokens * 3."""
        max_bs = 8
        max_seq_len = 64
        max_total_num_tokens = 1024
        capacities = self._from_args(
            max_bs=max_bs,
            max_seq_len=max_seq_len,
            max_total_num_tokens=max_total_num_tokens,
        )
        self.assertEqual(
            capacities.per_forward_verify_capacity,
            int(max_total_num_tokens * 3),
        )

    def test_from_args_treats_missing_speculative_draft_tokens_as_zero(self) -> None:
        """per_forward_write_entry_capacity is floored by max_prefill_tokens when batch * tokens_per_req is smaller."""
        capacities = self._from_args(
            max_bs=2,
            max_seq_len=32,
            max_total_num_tokens=64,
            speculative_num_draft_tokens=None,
        )

        self.assertEqual(capacities.per_forward_write_entry_capacity, 128)

    def test_manual_capacities_reject_non_positive_fields(self) -> None:
        """Verify manual launch capacities fail instead of being clamped."""
        with self.assertRaisesRegex(ValueError, "per_forward_verify_capacity"):
            CanaryLaunchCapacities(
                per_forward_verify_capacity=0,
                per_forward_write_req_capacity=1,
                per_forward_write_entry_capacity=1,
            )

    def test_from_args_rejects_empty_pool_capacity(self) -> None:
        """Verify derived launch capacities reject invalid pool sizing."""
        with self.assertRaisesRegex(ValueError, "pool_slot_count"):
            self._from_args(max_bs=1, max_seq_len=1, max_total_num_tokens=0)


if __name__ == "__main__":
    unittest.main()
