from __future__ import annotations

import unittest
from types import SimpleNamespace

from sglang.srt.kv_canary.capacities import CanaryLaunchCapacities
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="extra-a", runner_config="1-gpu-small")


class TestComputeLaunchCapacities(CustomTestCase):
    @staticmethod
    def _make_server_args(*, max_bs: int) -> SimpleNamespace:
        return SimpleNamespace(
            cuda_graph_max_bs=max_bs,
            speculative_num_draft_tokens=0,
            chunked_prefill_size=None,
            max_prefill_tokens=128,
        )

    @staticmethod
    def _from_args(
        *,
        max_bs: int,
        max_seq_len: int,
        max_total_num_tokens: int | None = None,
    ) -> CanaryLaunchCapacities:
        if max_total_num_tokens is None:
            max_total_num_tokens = max_bs * max_seq_len
        return CanaryLaunchCapacities.from_args(
            server_args=TestComputeLaunchCapacities._make_server_args(max_bs=max_bs),
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
        """per_forward_write_entry_capacity is floored by max_prefill_tokens when batch * tokens_per_bs is smaller."""
        server_args = self._make_server_args(max_bs=2)
        server_args.speculative_num_draft_tokens = None

        capacities = CanaryLaunchCapacities.from_args(
            server_args=server_args,
            req_to_token_pool_size=2,
            max_seq_len_per_req=32,
            pool_slot_count=64,
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
            CanaryLaunchCapacities.from_args(
                server_args=self._make_server_args(max_bs=1),
                req_to_token_pool_size=1,
                max_seq_len_per_req=1,
                pool_slot_count=0,
            )


if __name__ == "__main__":
    unittest.main()
