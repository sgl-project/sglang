"""STAGING_REQ must not grow chunk_staging_infos from an unbounded chunk_idx."""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.disaggregation.common.staging_buffer import StagingAllocator
from sglang.srt.disaggregation.common.staging_handler import (
    MAX_STAGING_CHUNK_IDX,
    handle_staging_req,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _staging_req_msg(room, chunk_idx, chunk_pages, session_id="peer"):
    return [
        b"STAGING_REQ",
        str(room).encode("ascii"),
        str(chunk_idx).encode("ascii"),
        str(chunk_pages).encode("ascii"),
        session_id.encode("ascii"),
    ]


def _fake_receiver(total_pages=4):
    sock = Mock()
    receiver = SimpleNamespace(
        chunk_staging_infos=[],
        _staging_total_pages=total_pages,
        _connect_to_bootstrap_server=Mock(return_value=(sock, threading.Lock())),
    )
    return receiver, sock


def _kv_args():
    return SimpleNamespace(
        page_size=64,
        kv_item_lens=[4096, 4096],
        total_kv_head_num=4,
        engine_rank=0,
    )


def _call_handle_staging_req(receiver, msg, allocator):
    handle_staging_req(
        msg,
        allocator,
        _kv_args(),
        attn_tp_size=16,
        prefill_attn_tp_size=1,
        kv_buffer_tensors=None,
        room_receivers={7: receiver},
        room_bootstrap={7: [{"pp_rank": 0}]},
    )


class TestHandleStagingReqChunkIdx(CustomTestCase):
    def test_huge_chunk_idx_does_not_grow_infos(self):
        receiver, sock = _fake_receiver(total_pages=4)
        allocator = SimpleNamespace(
            assign=Mock(return_value=(3, 128, 0)), total_size=1 << 20
        )

        with self.assertLogs(
            "sglang.srt.disaggregation.common.staging_handler", level="WARNING"
        ) as cm:
            _call_handle_staging_req(
                receiver,
                _staging_req_msg(7, 1_500_000_000, 1),
                allocator,
            )

        self.assertEqual(len(receiver.chunk_staging_infos), 0)
        allocator.assign.assert_not_called()
        sock.send_multipart.assert_not_called()
        self.assertTrue(any("chunk_idx=1500000000" in line for line in cm.output))
        self.assertTrue(any("bound=4" in line for line in cm.output))

    def test_in_range_chunk_idx_still_allocates(self):
        receiver, sock = _fake_receiver(total_pages=4)
        allocator = SimpleNamespace(
            assign=Mock(return_value=(3, 128, 0)), total_size=1 << 20
        )

        _call_handle_staging_req(
            receiver,
            _staging_req_msg(7, 1, 1),
            allocator,
        )

        self.assertEqual(len(receiver.chunk_staging_infos), 2)
        self.assertEqual(receiver.chunk_staging_infos[1][0], 3)
        allocator.assign.assert_called_once()
        sock.send_multipart.assert_called_once()

    def test_negative_chunk_idx_is_rejected(self):
        receiver, _sock = _fake_receiver(total_pages=4)
        allocator = SimpleNamespace(
            assign=Mock(return_value=(3, 128, 0)), total_size=1 << 20
        )

        with self.assertLogs(
            "sglang.srt.disaggregation.common.staging_handler", level="WARNING"
        ):
            _call_handle_staging_req(
                receiver,
                _staging_req_msg(7, -1, 1),
                allocator,
            )

        self.assertEqual(len(receiver.chunk_staging_infos), 0)
        allocator.assign.assert_not_called()

    def test_non_positive_chunk_pages_is_rejected(self):
        receiver, _sock = _fake_receiver(total_pages=4)
        allocator = SimpleNamespace(
            assign=Mock(return_value=(3, 128, 0)), total_size=1 << 20
        )

        with self.assertLogs(
            "sglang.srt.disaggregation.common.staging_handler", level="WARNING"
        ):
            _call_handle_staging_req(
                receiver,
                _staging_req_msg(7, 0, 0),
                allocator,
            )

        self.assertEqual(len(receiver.chunk_staging_infos), 0)
        allocator.assign.assert_not_called()

    def test_missing_total_pages_still_applies_finite_cap(self):
        receiver, sock = _fake_receiver(total_pages=0)
        del receiver._staging_total_pages
        allocator = SimpleNamespace(
            assign=Mock(return_value=(3, 128, 0)), total_size=1 << 20
        )

        with self.assertLogs(
            "sglang.srt.disaggregation.common.staging_handler", level="WARNING"
        ) as cm:
            _call_handle_staging_req(
                receiver,
                _staging_req_msg(7, MAX_STAGING_CHUNK_IDX, 1),
                allocator,
            )

        self.assertEqual(len(receiver.chunk_staging_infos), 0)
        allocator.assign.assert_not_called()
        sock.send_multipart.assert_not_called()
        self.assertTrue(
            any(f"bound={MAX_STAGING_CHUNK_IDX}" in line for line in cm.output)
        )

    def test_in_range_alloc_oversized_still_records(self):
        receiver, sock = _fake_receiver(total_pages=4)
        allocator = SimpleNamespace(assign=Mock(return_value=None), total_size=1 << 20)

        with self.assertLogs(
            "sglang.srt.disaggregation.common.staging_handler", level="ERROR"
        ):
            _call_handle_staging_req(
                receiver,
                _staging_req_msg(7, 0, 1),
                allocator,
            )

        self.assertEqual(len(receiver.chunk_staging_infos), 1)
        self.assertEqual(
            receiver.chunk_staging_infos[0][1], StagingAllocator.ALLOC_OVERSIZED
        )
        allocator.assign.assert_called_once()
        sock.send_multipart.assert_called_once()


if __name__ == "__main__":
    unittest.main()
