import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.dcp_pack import (
    dcp_pack_buffer_bytes,
    init_dcp_pack_buffers,
)
from sglang.srt.disaggregation.nixl.conn import NixlKVManager, NixlKVSender
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


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
            get_context().override_server_args(chunked_prefill_size=250, device="cuda"),
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
                token_counts = [c.kwargs["num_kv_tokens"] for c in calls]
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
                self.assertTrue(
                    all(c.kwargs["state_indices"] is None for c in calls[:-1])
                )
                self.assertEqual(int(calls[-1].kwargs["state_indices"][0][0]), 17)


class TestDcpPackBufferBytes(CustomTestCase):
    def test_sizes_fixed_regions_for_each_dcp_rank(self):
        self.assertEqual(
            dcp_pack_buffer_bytes(
                [64 * 16, 64 * 16],
                page_size=64,
                max_tokens=10,
                dcp_size=4,
            ),
            4 * 3 * (16 + 16),
        )

    def test_rejects_invalid_item_lens(self):
        with self.assertRaisesRegex(ValueError, "at least one page"):
            dcp_pack_buffer_bytes([0], page_size=64, max_tokens=8)
        with self.assertRaisesRegex(ValueError, "page-aligned"):
            dcp_pack_buffer_bytes([100], page_size=64, max_tokens=8)


class TestInitDcpPackBuffers(CustomTestCase):
    """The pack kernel is Triton-only: a non-CUDA device must get no buffers
    (per-token fallback) rather than crash, and an indexed "cuda:N" device must
    still get them, not silently fall back."""

    _KV_ARGS = SimpleNamespace(
        kv_item_lens=[64 * 16], page_size=64, num_draft_entries=0, gpu_id=3
    )

    def test_non_triton_device_gets_no_buffers(self):
        register_fn = Mock()
        with patch(
            "sglang.srt.disaggregation.common.dcp_pack.StagingBuffer"
        ) as staging_buffer:
            buffers = init_dcp_pack_buffers(
                register_fn, self._KV_ARGS, 2, 4, 64, device_type="xpu"
            )

        self.assertEqual(buffers, [])
        staging_buffer.assert_not_called()
        register_fn.assert_not_called()

    def test_cuda_buffers_use_the_bare_device_type_and_gpu_id(self):
        with (
            patch(
                "sglang.srt.disaggregation.common.dcp_pack.StagingBuffer"
            ) as staging_buffer,
            patch(
                "sglang.srt.disaggregation.common.staging_handler._get_custom_mem_pool",
                return_value=(None, None),
            ),
        ):
            buffers = init_dcp_pack_buffers(
                Mock(), self._KV_ARGS, 2, 4, 64, device_type="cuda:0"
            )

        self.assertEqual(len(buffers), 2)
        self.assertEqual(staging_buffer.call_args.args[1], "cuda:3")


class TestTryDcpPack(CustomTestCase):
    def test_try_pack_uses_requested_region_and_dense_indices(self):
        """A gather must fit its rank region even when the total buffer has space."""
        dim = 4
        kv = torch.arange(16 * dim, dtype=torch.float32).view(16, 1, dim)
        item_len = int(kv[0].nbytes)
        pack = torch.zeros(8 * item_len, dtype=torch.uint8)
        gather_stream = Mock()
        buf = type(
            "Buf",
            (),
            {
                "buffer": pack,
                "fits": lambda self, n: n <= pack.numel(),
                "get_ptr": lambda self: 0x1000,
                "get_size": lambda self: pack.numel(),
                "get_gather_stream": lambda self: gather_stream,
                "producer_stream": lambda self: Mock(),
            },
        )()
        src = np.array([1, 5, 9, 13], dtype=np.int64)
        pack_offset = 4 * item_len
        mgr = object.__new__(NixlKVManager)
        mgr.kv_args = SimpleNamespace(kv_data_ptrs=[kv.data_ptr()], num_draft_entries=0)
        dst = SimpleNamespace(
            dst_dcp_rank=1, dst_dcp_size=2, dcp_token_item_lens=[item_len]
        )
        with (
            patch(
                "sglang.srt.disaggregation.common.dcp_pack.device_stream_context",
                return_value=nullcontext(),
            ),
            patch(
                "sglang.srt.disaggregation.common.dcp_pack.copy_mla_rows_into_pack"
            ) as copy_mock,
        ):
            packed = mgr._pack_dcp_rank_once(buf, dst, src, {})
            dst.dst_dcp_size = 4
            overflow = mgr._pack_dcp_rank_once(buf, dst, src, {})

        self.assertIsNone(overflow)
        gather_stream.synchronize.assert_called_once_with()
        self.assertIsNotNone(packed)
        ptrs, indices = packed
        self.assertEqual(ptrs, [0x1000 + pack_offset])
        np.testing.assert_array_equal(indices, np.arange(4))
        pack_view = copy_mock.call_args.args[2]
        self.assertEqual(pack_view.storage_offset(), pack_offset)
        self.assertEqual(pack_view.numel(), src.size * item_len)


if __name__ == "__main__":
    unittest.main()
