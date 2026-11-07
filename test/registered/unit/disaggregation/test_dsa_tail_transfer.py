"""Prefill/decode DSA tail indices must share one logical sequence length."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import (
    build_dsa_tail_transfer_blocks,
    get_dsa_tail_state_indices,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_POOL_SIZE = 4
_DECODE_TAIL_EXTRA = 8
_SLOT_BYTES = 8


def _pool(*, tail_extra_slots: int):
    return SimpleNamespace(
        use_dsa=False,
        kpool_use_compress=True,
        index_kpool=_POOL_SIZE,
        tail_extra_slots=tail_extra_slots,
    )


def _send_last_chunk(*, origin_len: int, extend_end: int, req_pool_idx: int = 7):
    """Run prefill's last-chunk send and return (page_indices, dsa_tail_indices)."""
    captured = {}

    def send(page_indices, state_indices, num_kv_tokens=None, indices_event=None):
        captured["pages"] = page_indices
        captured["state"] = state_indices

    sender = SimpleNamespace(
        supports_host_tensor_indices=False,
        kv_mgr=SimpleNamespace(enable_all_cp_ranks_for_transfer=False),
        get_max_transfer_tokens=lambda: None,
        should_send_kv_chunk=lambda num_pages, last_chunk: True,
        send=send,
    )
    req = SimpleNamespace(
        rid="dsa-tail",
        kv=SimpleNamespace(req_pool_idx=req_pool_idx),
        origin_input_ids=[0] * origin_len,
        extend_end=extend_end,
        start_send_idx=0,
        disagg_decode_prefix_len=0,
        disagg_kv_sender=sender,
    )
    width = max(origin_len, extend_end) + 4
    scheduler = SimpleNamespace(
        enable_staging=False,
        token_to_kv_pool_allocator=SimpleNamespace(
            page_size=1,
            get_kvcache=lambda: _pool(tail_extra_slots=0),
            translate_kv_indices_for_transfer=lambda indices: indices,
        ),
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.arange((req_pool_idx + 1) * width).reshape(
                req_pool_idx + 1, width
            ),
        ),
        disagg_metadata_buffers=Mock(),
        disagg_prefill_bootstrap_queue=SimpleNamespace(
            kv_manager=SimpleNamespace(
                kv_args=SimpleNamespace(state_types=[StateType.DSA_TAIL])
            )
        ),
        disagg_prefill_pending_chunk_rids=set(),
    )
    SchedulerDisaggregationPrefillMixin._send_kv_chunk(scheduler, req, last_chunk=True)
    return captured["pages"], list(captured["state"][0])


def _decode_tail(seq_len: int, req_pool_idx: int = 3):
    """Decode prealloc registers len(origin_input_ids), with draft tail slots."""
    return get_dsa_tail_state_indices(
        _pool(tail_extra_slots=_DECODE_TAIL_EXTRA), req_pool_idx, seq_len
    )


def _transfer(src_indices, dst_indices):
    src_tail = _POOL_SIZE
    dst_tail = _POOL_SIZE + _DECODE_TAIL_EXTRA
    return build_dsa_tail_transfer_blocks(
        src_ptrs=[0x1000, 0x2000],
        src_item_lens=[src_tail * _SLOT_BYTES, src_tail * _SLOT_BYTES],
        dst_ptrs=[0x3000, 0x4000],
        src_indices=src_indices,
        dst_indices=dst_indices,
        dst_item_lens=[dst_tail * _SLOT_BYTES, dst_tail * _SLOT_BYTES],
    )


class TestDsaTailLastChunkSeqLen(CustomTestCase):
    def test_pool_boundary_extend_end_still_matches_decode_origin(self):
        """extend_end on a kpool boundary must not emit an empty prefill tail.

        Decode registers len(origin_input_ids). With pool_size 4 and decode
        tail_extra_slots 8, origin 13 yields dst [req, 0, 1, 0, 0, 12] while
        extend_end 12 yields src []. Mooncake then raises
        "DSA tail slot index missing". KV pages stay on the materialized range.
        """
        origin_len, extend_end = 13, 12
        pages, src = _send_last_chunk(origin_len=origin_len, extend_end=extend_end)
        dst = _decode_tail(origin_len)

        self.assertEqual(len(pages), extend_end)
        self.assertEqual(
            src,
            get_dsa_tail_state_indices(_pool(tail_extra_slots=0), 7, origin_len),
        )
        self.assertEqual(src[2] + src[4], dst[2] + dst[4])
        self.assertEqual(dst[5], _POOL_SIZE + _DECODE_TAIL_EXTRA)
        self.assertTrue(_transfer(src, dst))

        stale = get_dsa_tail_state_indices(_pool(tail_extra_slots=0), 7, extend_end)
        self.assertEqual(stale, [])
        with self.assertRaisesRegex(ValueError, "DSA tail slot index missing"):
            _transfer(stale, dst)

    def test_sampled_token_past_origin_does_not_widen_the_tail(self):
        """fill_ids can include the sampled token; the tail stays at origin."""
        origin_len, extend_end = 13, 14
        pages, src = _send_last_chunk(origin_len=origin_len, extend_end=extend_end)
        widened = get_dsa_tail_state_indices(_pool(tail_extra_slots=0), 7, extend_end)

        self.assertEqual(len(pages), origin_len)
        self.assertEqual(src[2] + src[4], 1)
        self.assertNotEqual(widened[2] + widened[4], src[2] + src[4])
        self.assertTrue(_transfer(src, _decode_tail(origin_len)))

    def test_pool_aligned_origin_is_empty_on_both_sides(self):
        """A prompt whose length is a whole number of pools transfers no tail."""
        origin_len = 12
        _, src = _send_last_chunk(origin_len=origin_len, extend_end=origin_len + 1)
        dst = _decode_tail(origin_len)
        self.assertEqual(src, [])
        self.assertEqual(dst, [])
        self.assertEqual(_transfer(src, dst), [])


if __name__ == "__main__":
    unittest.main()
