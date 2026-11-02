"""Replicated KV heads need exactly one P writer per D slot, at agreed offsets."""

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from sglang.srt.disaggregation.common.staging_buffer import (
    compute_head_slice_params,
    compute_staging_layout,
    staging_writer_ranks,
    staging_writer_slot,
)
from sglang.srt.disaggregation.common.utils import should_send_kv_replica
from sglang.srt.disaggregation.mooncake import conn as mooncake_conn
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ROOMS = [0, 1, 7, 2**31 - 1, 2**63 - 25, 914_582_337_104_221]
HETEROGENEOUS = [
    (src, dst, heads)
    for src in (2, 4, 8)
    for dst in (1, 2, 4)
    for heads in (1, 2, 4, 8, 16)
    if src > dst
]


def dst_head(src_tp, dst_tp, src_rank, dst_rank, heads):
    return compute_head_slice_params(src_tp, dst_tp, src_rank, dst_rank, heads)[2]


def group(src_tp, dst_tp, dst_rank):
    size = src_tp // dst_tp
    return range(dst_rank * size, (dst_rank + 1) * size)


def writers(room, src_tp, dst_tp, dst_rank, heads):
    # Dispatch only targets a P rank's own D rank; election happens within it.
    return [
        rank
        for rank in group(src_tp, dst_tp, dst_rank)
        if should_send_kv_replica(room, src_tp, dst_tp, rank, dst_rank, heads)
    ]


@pytest.mark.parametrize("src_tp,dst_tp,heads", HETEROGENEOUS)
def test_one_writer_per_destination_head(src_tp, dst_tp, heads):
    for dst_rank in range(dst_tp):
        members = group(src_tp, dst_tp, dst_rank)
        needed = {dst_head(src_tp, dst_tp, r, dst_rank, heads) for r in members}
        for room in ROOMS:
            elected = writers(room, src_tp, dst_tp, dst_rank, heads)
            sent = [dst_head(src_tp, dst_tp, r, dst_rank, heads) for r in elected]
            assert sorted(sent) == sorted(needed)
            assert elected == writers(room, src_tp, dst_tp, dst_rank, heads)


@pytest.mark.parametrize("src_tp,dst_tp,heads", [(8, 1, 1), (8, 2, 1), (8, 2, 4)])
def test_rooms_spread_writes_across_replicas(src_tp, dst_tp, heads):
    used = set()
    for room in range(256):
        for dst_rank in range(dst_tp):
            used.update(writers(room, src_tp, dst_tp, dst_rank, heads))
    assert used == set(range(src_tp))


@pytest.mark.parametrize("deduplicate", [True, False])
@pytest.mark.parametrize("src_tp,dst_tp,heads", HETEROGENEOUS + [(2, 4, 1), (4, 8, 8)])
def test_prefill_slots_match_decode_layout(src_tp, dst_tp, heads, deduplicate):
    tokens, head_bytes, layers = 3, 256, 5
    for dst_rank in range(dst_tp):
        owners = staging_writer_ranks(src_tp, dst_tp, dst_rank, heads, deduplicate)
        _, slot_bytes, total = compute_staging_layout(
            src_tp, dst_tp, dst_rank, heads, tokens, head_bytes, layers, deduplicate
        )
        assert len(slot_bytes) == len(owners)
        for room in ROOMS:
            if src_tp <= dst_tp or not deduplicate:
                senders = list(owners)
            else:
                senders = writers(room, src_tp, dst_tp, dst_rank, heads)
            regions = []
            for rank in senders:
                slot = staging_writer_slot(src_tp, dst_tp, rank, heads, deduplicate)
                owner = owners[slot]
                assert dst_head(src_tp, dst_tp, rank, dst_rank, heads) == dst_head(
                    src_tp, dst_tp, owner, dst_rank, heads
                )
                num_heads = compute_head_slice_params(
                    src_tp, dst_tp, rank, dst_rank, heads
                )[1]
                size = tokens * num_heads * head_bytes * layers * 2
                assert size == slot_bytes[slot]
                # Decode scatter addresses each slot as slot * per-writer bytes.
                assert sum(slot_bytes[:slot]) == slot * size
                regions.append((sum(slot_bytes[:slot]), size))
            regions.sort()
            assert regions[0][0] == 0
            assert all(a + n == b for (a, n), (b, _) in zip(regions, regions[1:]))
            assert sum(n for _, n in regions) == total


PAGE, HEAD_BYTES, LAYERS = 4, 8, 2


def kv_pages(rank_head, pages):
    # K then V per layer; each byte encodes (entry, page, token, head).
    entries = np.arange(2 * LAYERS)[:, None, None, None]
    page_ids = np.arange(pages)[None, :, None, None]
    tokens = np.arange(PAGE)[None, None, :, None]
    data = (entries * 61 + page_ids * 7 + tokens * 3 + rank_head * 101) % 251
    return np.ascontiguousarray(
        np.broadcast_to(data, (2 * LAYERS, pages, PAGE, HEAD_BYTES)).astype(np.uint8)
    )


def ptrs(buffer):
    return [buffer[i].ctypes.data for i in range(len(buffer))]


@pytest.mark.parametrize(
    "src_tp,dst_tp,heads", [(8, 1, 1), (8, 2, 2), (8, 4, 4), (2, 4, 1), (2, 8, 2)]
)
def test_whole_page_transfer_writes_each_destination_once(
    monkeypatch, src_tp, dst_tp, heads
):
    monkeypatch.setattr(
        mooncake_conn,
        "get_memory",
        lambda: SimpleNamespace(enable_unified_memory=False),
    )
    pages, item_len = 6, PAGE * HEAD_BYTES
    src_pages = np.array([5, 2, 3, 0], dtype=np.int32)
    dst_pages = np.array([1, 4, 5, 2], dtype=np.int32)
    src_head = [r // max(1, src_tp // heads) for r in range(src_tp)]
    dst_head = [d // max(1, dst_tp // heads) for d in range(dst_tp)]
    sources = [kv_pages(src_head[r], pages) for r in range(src_tp)]
    for room in ROOMS:
        targets = [
            np.full((2 * LAYERS, pages, PAGE, HEAD_BYTES), 255, np.uint8)
            for _ in range(dst_tp)
        ]
        writes = [0] * dst_tp
        for dst_rank in range(dst_tp):
            if src_tp > dst_tp:
                peers = group(src_tp, dst_tp, dst_rank)
            else:
                peers = [dst_rank // (dst_tp // src_tp)]
            for src_rank in peers:
                manager = object.__new__(mooncake_conn.MooncakeKVManager)
                manager.attn_tp_size, manager.pp_size = src_tp, 1
                manager.is_mla_backend = manager.is_hybrid_mla_backend = False
                manager.enable_custom_mem_pool = False
                manager.max_transfer_batch_indices = 0
                manager.kv_args = SimpleNamespace(
                    engine_rank=src_rank,
                    total_kv_head_num=heads,
                    kv_head_num=1,
                    kv_item_lens=[item_len] * 2 * LAYERS,
                    kv_data_ptrs=ptrs(sources[src_rank]),
                    kv_layer_ids=None,
                    prefill_start_layer=0,
                )

                def transfer(session, blocks, dst_rank=dst_rank):
                    writes[dst_rank] += 1
                    for src, dst, length in blocks:
                        ctypes.memmove(dst, src, length)
                    return 0

                manager._transfer_data = transfer
                peer = SimpleNamespace(
                    dst_attn_tp_size=dst_tp,
                    dst_tp_rank=dst_rank,
                    dst_kv_item_len=item_len,
                    dst_kv_ptrs=ptrs(targets[dst_rank]),
                    dst_kv_layer_ids=None,
                )
                assert manager._can_send_whole_kv_pages(peer)
                if manager._should_send_main_kv(SimpleNamespace(room=room), peer):
                    assert (
                        manager.send_kvcache(
                            "test",
                            src_pages,
                            peer.dst_kv_ptrs,
                            dst_pages,
                            None,
                            dst_kv_item_len=item_len,
                            dst_attn_tp_size=dst_tp,
                        )
                        == 0
                    )
        assert writes == [1] * dst_tp
        for dst_rank, target in enumerate(targets):
            expected = kv_pages(dst_head[dst_rank], pages)
            np.testing.assert_array_equal(target[:, dst_pages], expected[:, src_pages])
            untouched = np.setdiff1d(np.arange(pages), dst_pages)
            assert (target[:, untouched] == 255).all()


@pytest.mark.parametrize(
    "src_tp,dst_tp,heads,dst_item_scale,unified",
    [
        (8, 1, 2, 2, False),  # D rank holds two heads, P rank one.
        (8, 1, 1, 2, False),  # Page byte layouts differ.
        (8, 1, 1, 1, True),  # Unified memory cannot address peers by our item_len.
    ],
)
def test_whole_page_transfer_requires_identical_page_layout(
    monkeypatch, src_tp, dst_tp, heads, dst_item_scale, unified
):
    monkeypatch.setattr(
        mooncake_conn,
        "get_memory",
        lambda: SimpleNamespace(enable_unified_memory=unified),
    )
    manager = object.__new__(mooncake_conn.MooncakeKVManager)
    manager.attn_tp_size = src_tp
    manager.kv_args = SimpleNamespace(
        total_kv_head_num=heads, kv_head_num=1, kv_item_lens=[256] * 4
    )
    peer = SimpleNamespace(
        dst_attn_tp_size=dst_tp, dst_kv_item_len=256 * dst_item_scale
    )
    assert not manager._can_send_whole_kv_pages(peer)
