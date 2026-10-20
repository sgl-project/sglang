import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.nixl.conn import NixlKVManager, NixlKVSender
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestDcpCachedPrefixSend(CustomTestCase):
    def test_cached_prefix_fits_pack_capacity_and_preserves_pages_and_state(self):
        """Bound DCP sends by the allocation without splitting TP sends on the same worker."""
        total, page_size, token_bytes = 2055, 64, 16
        mgr = object.__new__(NixlKVManager)
        mgr.kv_args = SimpleNamespace(
            kv_item_lens=[page_size * token_bytes],
            num_draft_entries=0,
            page_size=page_size,
            gpu_id=0,
            state_types=[StateType.MAMBA],
        )
        mgr._dcp_pack_buffers = None
        mgr._dcp_pack_max_tokens = None
        mgr.transfer_queues = [None]
        mgr._register_staging_memory = Mock()
        mgr.request_status = {}
        mgr._deferred_ack_targets = {}
        mgr._deferred_ack_poisoned_rooms = set()
        mgr._staging_outstanding = {}
        mgr.is_dummy_cp_rank = False
        mgr.enable_all_cp_ranks_for_transfer = False
        mgr.decode_kv_args_table = {
            peer: SimpleNamespace(requires_dcp_relayout=relayout)
            for peer, relayout in (("dcp", True), ("tp", False))
        }

        def allocate(size, *args, **kwargs):
            return SimpleNamespace(get_ptr=lambda: 0x1000, get_size=lambda: size)

        with (
            get_context().override_server_args(chunked_prefill_size=250),
            patch(
                "sglang.srt.disaggregation.common.staging_handler._get_custom_mem_pool",
                return_value=(None, None),
            ),
            patch(
                "sglang.srt.disaggregation.common.dcp_pack.StagingBuffer",
                side_effect=allocate,
            ),
        ):
            mgr._init_dcp_pack_buffers_once(dcp_size=4)
        limit = mgr._dcp_pack_buffers[0].get_size() // token_bytes

        for peer, prefix in (("dcp", 0), ("dcp", 256), ("tp", 0), ("tp", 256)):
            with self.subTest(peer=peer, decode_prefix=prefix):
                mgr.transfer_infos = {
                    1: {
                        "dummy": SimpleNamespace(is_dummy=True),
                        peer: SimpleNamespace(is_dummy=False),
                    }
                }
                mgr.add_transfer_request = Mock()
                with get_context().override_server_args(dp_size=1):
                    sender = NixlKVSender(mgr, "unused", 1)
                sender.init((total - prefix + page_size - 1) // page_size, 3)
                req = SimpleNamespace(
                    rid="cached-prefix",
                    kv=SimpleNamespace(req_pool_idx=0),
                    origin_input_ids=[0] * total,
                    extend_end=total,
                    start_send_idx=prefix,
                    disagg_decode_prefix_len=prefix,
                    disagg_kv_sender=sender,
                )
                scheduler = SimpleNamespace(
                    enable_staging=False,
                    token_to_kv_pool_allocator=SimpleNamespace(
                        page_size=page_size,
                        translate_kv_indices_for_transfer=lambda x: x,
                    ),
                    req_to_token_pool=SimpleNamespace(
                        req_to_token=torch.arange(total).reshape(1, -1),
                        req_index_to_mamba_index_mapping=torch.tensor([17]),
                        translate_mamba_indices=lambda x: x,
                    ),
                    disagg_metadata_buffers=Mock(),
                    disagg_prefill_bootstrap_queue=SimpleNamespace(kv_manager=mgr),
                    disagg_prefill_pending_chunk_rids=set(),
                )
                SchedulerDisaggregationPrefillMixin._send_kv_chunk(
                    scheduler, req, last_chunk=True
                )
                calls = mgr.add_transfer_request.call_args_list
                token_counts = [c.args[7] for c in calls]
                if peer == "dcp":
                    self.assertLessEqual(max(token_counts), limit)
                else:
                    self.assertEqual(len(calls), 1)
                self.assertEqual(sum(token_counts), total - prefix)
                np.testing.assert_array_equal(
                    np.concatenate([c.args[1] for c in calls]),
                    np.arange(prefix // 64, 33),
                )
                page_offset = 0
                for call in calls:
                    pages = len(call.args[1])
                    self.assertEqual(
                        call.args[2], slice(page_offset, page_offset + pages)
                    )
                    page_offset += pages
                self.assertEqual(
                    [c.args[3] for c in calls], [False] * (len(calls) - 1) + [True]
                )
                self.assertTrue(all(c.args[6] is None for c in calls[:-1]))
                self.assertEqual(int(calls[-1].args[6][0][0]), 17)


if __name__ == "__main__":
    unittest.main()
