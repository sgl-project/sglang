import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import get_dsa_state_page_indices
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSAStatePageIndices(CustomTestCase):
    def test_uses_logical_input_range(self):
        kv_indices = torch.arange(0, 192, dtype=torch.int64)

        indices = get_dsa_state_page_indices(
            kv_indices, logical_input_len=129, device_page_size=64
        )

        np.testing.assert_array_equal(indices, np.array([0, 1, 2]))

    def test_rejects_incomplete_or_invalid_layout(self):
        kv_indices = torch.arange(0, 64, dtype=torch.int64)

        with self.assertRaisesRegex(ValueError, "exceeds the request-to-token row"):
            get_dsa_state_page_indices(
                kv_indices, logical_input_len=65, device_page_size=64
            )
        with self.assertRaisesRegex(ValueError, "device_page_size must be positive"):
            get_dsa_state_page_indices(
                kv_indices, logical_input_len=64, device_page_size=0
            )


class TestPrefillDSAStateContract(CustomTestCase):
    def test_last_chunk_uses_logical_range_and_device_page_size(self):
        sender = MagicMock()
        sender.should_send_kv_chunk.return_value = True
        req_to_token = torch.arange(0, 192, dtype=torch.int64).reshape(1, -1)
        device_pool = SimpleNamespace(page_size=64)
        allocator = SimpleNamespace(
            page_size=1,
            get_kvcache=lambda: device_pool,
            translate_kv_indices_for_transfer=lambda indices: indices,
        )
        scheduler = SimpleNamespace(
            token_to_kv_pool_allocator=allocator,
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            disagg_metadata_buffers=MagicMock(),
            disagg_prefill_bootstrap_queue=SimpleNamespace(
                kv_manager=SimpleNamespace(
                    kv_args=SimpleNamespace(state_types=[StateType.DSA])
                )
            ),
            enable_staging=False,
            disagg_prefill_pending_chunk_rids=set(),
        )
        req = SimpleNamespace(
            rid="request",
            start_send_idx=0,
            origin_input_ids=list(range(129)),
            extend_range=SimpleNamespace(end=64),
            kv=SimpleNamespace(req_pool_idx=0),
            disagg_kv_sender=sender,
        )

        SchedulerDisaggregationPrefillMixin.send_kv_chunk(
            scheduler, req, last_chunk=True
        )

        state_indices = sender.send.call_args.args[1]
        np.testing.assert_array_equal(state_indices[0], np.array([0, 1, 2]))


if __name__ == "__main__":
    unittest.main()
