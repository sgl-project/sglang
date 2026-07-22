from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import sglang.srt.disaggregation.common.shared_kv_staging as shared_kv_staging
from sglang.srt.disaggregation.common.conn import CommonKVSender
from sglang.srt.disaggregation.common.shared_kv_staging import (
    OwnerShardedStagingCache,
    OwnerShardedTransferBuffer,
    send_owner_sharded_staged,
)
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.mem_cache.deepseek_v4_shared import (
    SharedDeepSeekV4TokenToKVPool,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _StagingBuffer:
    def __init__(self, size: int):
        self.buffer = torch.empty(size, dtype=torch.uint8)

    def get_size(self) -> int:
        return self.buffer.numel()

    def get_ptr(self) -> int:
        return self.buffer.data_ptr()


class _RecordingEngine:
    def __init__(self):
        self.registered = []
        self.deregistered = []

    def batch_register(self, ptrs, lens):
        self.registered.append((list(ptrs), list(lens)))

    def batch_deregister(self, ptrs):
        self.deregistered.append(list(ptrs))


def test_sender_preserves_full_logical_indices_for_owner_staging():
    sender = object.__new__(CommonKVSender)
    sender.curr_idx = 0
    sender.num_kv_indices = 8
    sender.kv_mgr = SimpleNamespace(
        server_args=SimpleNamespace(enable_dsa_shared_kv_cache=True),
        attn_cp_rank=1,
        attn_cp_size=2,
    )

    first, first_positions, is_last, should_skip = sender._prepare_send_indices(
        np.array([0, 1, 2, 3], dtype=np.int32)
    )
    second, second_positions, is_last_second, should_skip_second = (
        sender._prepare_send_indices(np.array([4, 5, 6, 7], dtype=np.int32))
    )

    # The staging descriptor decides whether to gather only this rank's owner
    # pages or all owners through a rank-aggregated VMM view.  Filtering here
    # would make the aggregated path transfer only 1 / CP of the request.
    np.testing.assert_array_equal(first, np.array([0, 1, 2, 3], dtype=np.int32))
    assert first_positions == slice(0, 4)
    assert not is_last and not should_skip
    np.testing.assert_array_equal(second, np.array([4, 5, 6, 7], dtype=np.int32))
    assert second_positions == slice(4, 8)
    assert is_last_second and not should_skip_second


def test_mooncake_registers_staging_not_shared_vmm_aliases():
    manager = object.__new__(MooncakeKVManager)
    manager.disaggregation_mode = DisaggregationMode.PREFILL
    manager.server_args = SimpleNamespace(enable_dsa_shared_kv_cache=True)
    manager.kv_args = SimpleNamespace(
        kv_data_ptrs=[10],
        kv_data_lens=[100],
        aux_data_ptrs=[20],
        aux_data_lens=[200],
        state_data_ptrs=[[30]],
        state_data_lens=[[300]],
    )
    manager.engine = _RecordingEngine()

    with get_context().override_server_args(enable_dsa_shared_kv_cache=True):
        manager.register_buffer_to_engine()
        manager.deregister_buffer_to_engine()

    assert manager.engine.registered == [([20], [200])]
    assert manager.engine.deregistered == [[20]]


def test_shared_nonzero_cp_rank_does_not_skip_state_transfer():
    manager = object.__new__(MooncakeKVManager)
    manager.is_hybrid_mla_backend = True
    manager.attn_tp_size = 8
    manager.attn_cp_size = 8
    manager.attn_cp_rank = 7
    manager.server_args = SimpleNamespace(
        enable_dsa_cache_layer_split=False,
        enable_dsa_shared_kv_cache=True,
    )

    with get_context().override_server_args(
        enable_dsa_cache_layer_split=False,
        enable_dsa_shared_kv_cache=True,
    ):
        assert manager._get_dsa_cache_transfer_skip_flags(None) == (False, False)


def test_owner_staging_cache_is_only_created_for_non_final_chunks():
    manager = object.__new__(MooncakeKVManager)
    manager.server_args = SimpleNamespace(enable_dsa_shared_kv_cache=True)
    manager.disaggregation_mode = DisaggregationMode.PREFILL

    with get_context().override_server_args(enable_dsa_shared_kv_cache=True):
        cache = manager._new_owner_staging_cache(SimpleNamespace(is_last_chunk=False))

        assert isinstance(cache, OwnerShardedStagingCache)
        assert (
            manager._new_owner_staging_cache(SimpleNamespace(is_last_chunk=True)) is None
        )


def test_owner_staging_sends_only_local_pages_to_matching_destinations():
    local_rows = torch.tensor(
        [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]],
        dtype=torch.uint8,
    )
    descriptor = OwnerShardedTransferBuffer(
        tensor=local_rows,
        item_bytes=4,
        owner_page_bytes=4,
        owner_pages_per_item=1,
    )
    staging = _StagingBuffer(8)
    observed = []

    def transfer(_session, blocks):
        for src_ptr, dst_ptr, size in blocks:
            offset = src_ptr - staging.get_ptr()
            observed.append(
                (dst_ptr, bytes(staging.buffer[offset : offset + size].tolist()))
            )
        return 0

    status = send_owner_sharded_staged(
        transfer=transfer,
        session_id="session",
        src_buffers=[descriptor],
        logical_src_indices=np.array([0, 1, 2, 3], dtype=np.int32),
        dst_ptrs=[1000],
        logical_dst_indices=np.array([10, 11, 12, 13], dtype=np.int32),
        cp_rank=0,
        cp_size=2,
        staging_buffer=staging,
    )

    assert status == 0
    assert observed == [
        (1040, bytes([10, 11, 12, 13])),
        (1048, bytes([20, 21, 22, 23])),
    ]


def test_owner_staging_expands_two_state_subpages_per_item():
    # CP2 rank 1 owns the second four-byte subpage of every eight-byte state item.
    local_rows = torch.zeros((5, 4), dtype=torch.uint8)
    local_rows[3] = torch.tensor([41, 42, 43, 44], dtype=torch.uint8)
    local_rows[4] = torch.tensor([51, 52, 53, 54], dtype=torch.uint8)
    descriptor = OwnerShardedTransferBuffer(
        tensor=local_rows,
        item_bytes=8,
        owner_page_bytes=4,
        owner_pages_per_item=2,
    )
    staging = _StagingBuffer(8)
    observed = []

    def transfer(_session, blocks):
        for src_ptr, dst_ptr, size in blocks:
            offset = src_ptr - staging.get_ptr()
            observed.append(
                (dst_ptr, bytes(staging.buffer[offset : offset + size].tolist()))
            )
        return 0

    status = send_owner_sharded_staged(
        transfer=transfer,
        session_id="session",
        src_buffers=[descriptor],
        logical_src_indices=np.array([3, 4], dtype=np.int32),
        dst_ptrs=[2000],
        logical_dst_indices=np.array([7, 9], dtype=np.int32),
        cp_rank=1,
        cp_size=2,
        staging_buffer=staging,
    )

    assert status == 0
    assert observed == [
        (2060, bytes([41, 42, 43, 44])),
        (2076, bytes([51, 52, 53, 54])),
    ]


def test_rank_major_staging_gathers_all_owners_in_logical_page_order():
    rank_major_rows = torch.tensor(
        [
            [10, 10, 10, 10],
            [11, 11, 11, 11],
            [99, 99, 99, 99],
            [20, 20, 20, 20],
            [21, 21, 21, 21],
        ],
        dtype=torch.uint8,
    )
    descriptor = OwnerShardedTransferBuffer(
        tensor=rank_major_rows,
        item_bytes=4,
        owner_page_bytes=4,
        rank_stride_owner_pages=3,
    )
    staging = _StagingBuffer(16)
    observed = []

    def transfer(_session, blocks):
        for src_ptr, dst_ptr, size in blocks:
            offset = src_ptr - staging.get_ptr()
            observed.append(
                (dst_ptr, bytes(staging.buffer[offset : offset + size].tolist()))
            )
        return 0

    status = send_owner_sharded_staged(
        transfer=transfer,
        session_id="session",
        src_buffers=[descriptor],
        logical_src_indices=np.array([0, 1, 2, 3], dtype=np.int32),
        dst_ptrs=[1000],
        logical_dst_indices=np.array([10, 11, 12, 13], dtype=np.int32),
        cp_rank=0,
        cp_size=2,
        staging_buffer=staging,
    )

    assert status == 0
    assert observed == [
        (1040, bytes([10, 10, 10, 10])),
        (1044, bytes([20, 20, 20, 20])),
        (1048, bytes([11, 11, 11, 11])),
        (1052, bytes([21, 21, 21, 21])),
    ]


def test_owner_staging_batches_multiple_descriptors_before_transfer():
    first = OwnerShardedTransferBuffer(
        tensor=torch.tensor([[1, 2, 3, 4]], dtype=torch.uint8),
        item_bytes=4,
        owner_page_bytes=4,
    )
    second = OwnerShardedTransferBuffer(
        tensor=torch.tensor([[5, 6, 7, 8]], dtype=torch.uint8),
        item_bytes=4,
        owner_page_bytes=4,
    )
    staging = _StagingBuffer(8)
    calls = []

    def transfer(_session, blocks):
        calls.append(
            [
                (
                    dst_ptr,
                    bytes(
                        staging.buffer[
                            src_ptr
                            - staging.get_ptr() : src_ptr
                            - staging.get_ptr()
                            + size
                        ].tolist()
                    ),
                )
                for src_ptr, dst_ptr, size in blocks
            ]
        )
        return 0

    status = send_owner_sharded_staged(
        transfer=transfer,
        session_id="session",
        src_buffers=[first, second],
        logical_src_indices=np.array([0], dtype=np.int32),
        dst_ptrs=[1000, 2000],
        logical_dst_indices=np.array([3], dtype=np.int32),
        cp_rank=0,
        cp_size=1,
        staging_buffer=staging,
    )

    assert status == 0
    assert calls == [
        [
            (1012, bytes([1, 2, 3, 4])),
            (2012, bytes([5, 6, 7, 8])),
        ]
    ]


def test_owner_staging_cache_reuses_the_first_source_pack():
    source = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.uint8)
    descriptor = OwnerShardedTransferBuffer(
        tensor=source,
        item_bytes=4,
        owner_page_bytes=4,
    )
    staging = _StagingBuffer(8)
    cache = OwnerShardedStagingCache()
    observed = []

    def transfer(session, blocks):
        observed.append(
            (
                session,
                [
                    bytes(
                        staging.buffer[
                            src_ptr
                            - staging.get_ptr() : src_ptr
                            - staging.get_ptr()
                            + size
                        ].tolist()
                    )
                    for src_ptr, _dst_ptr, size in blocks
                ],
            )
        )
        return 0

    common = dict(
        transfer=transfer,
        src_buffers=[descriptor],
        logical_src_indices=np.array([0, 1], dtype=np.int32),
        dst_ptrs=[1000],
        logical_dst_indices=np.array([4, 6], dtype=np.int32),
        cp_rank=0,
        cp_size=1,
        staging_buffer=staging,
        cache=cache,
    )
    assert send_owner_sharded_staged(session_id="first", **common) == 0
    source.fill_(99)
    assert send_owner_sharded_staged(session_id="second", **common) == 0

    assert observed == [
        ("first", [bytes([1, 2, 3, 4]), bytes([5, 6, 7, 8])]),
        ("second", [bytes([1, 2, 3, 4]), bytes([5, 6, 7, 8])]),
    ]


def test_owner_staging_cache_groups_matching_descriptor_geometry_once():
    descriptors = [
        OwnerShardedTransferBuffer(
            tensor=torch.tensor([[value] * 4], dtype=torch.uint8),
            item_bytes=4,
            owner_page_bytes=4,
        )
        for value in (1, 2)
    ]
    staging = _StagingBuffer(8)
    cache = OwnerShardedStagingCache()

    original = shared_kv_staging.group_concurrent_contiguous
    with patch.object(
        shared_kv_staging,
        "group_concurrent_contiguous",
        wraps=original,
    ) as grouped:
        status = send_owner_sharded_staged(
            transfer=lambda _session, _blocks: 0,
            session_id="session",
            src_buffers=descriptors,
            logical_src_indices=np.array([0], dtype=np.int32),
            dst_ptrs=[1000, 2000],
            logical_dst_indices=np.array([3], dtype=np.int32),
            cp_rank=0,
            cp_size=1,
            staging_buffer=staging,
            cache=cache,
        )

    assert status == 0
    assert grouped.call_count == 1


def test_owner_staging_cache_builds_matching_source_geometry_once():
    descriptors = [
        OwnerShardedTransferBuffer(
            tensor=torch.zeros((3, 4), dtype=torch.uint8),
            item_bytes=4,
            owner_page_bytes=4,
            rank_stride_owner_pages=2,
        )
        for _ in range(2)
    ]
    cache = OwnerShardedStagingCache()

    original = shared_kv_staging._source_owner_page_rows
    with patch.object(
        shared_kv_staging,
        "_source_owner_page_rows",
        wraps=original,
    ) as source_plan:
        status = send_owner_sharded_staged(
            transfer=lambda _session, _blocks: 0,
            session_id="session",
            src_buffers=descriptors,
            logical_src_indices=np.array([0, 1], dtype=np.int32),
            dst_ptrs=[1000, 2000],
            logical_dst_indices=np.array([3, 4], dtype=np.int32),
            cp_rank=0,
            cp_size=2,
            staging_buffer=_StagingBuffer(16),
            cache=cache,
        )

    assert status == 0
    assert source_plan.call_count == 1


def test_owner_staging_cache_gathers_regular_layer_views_once():
    storage = torch.arange(16, dtype=torch.uint8).view(8, 2)
    descriptors = [
        OwnerShardedTransferBuffer(
            tensor=storage.narrow(0, layer_offset, 6),
            item_bytes=2,
            owner_page_bytes=2,
            rank_stride_owner_pages=4,
        )
        for layer_offset in (0, 2)
    ]
    staging = _StagingBuffer(8)

    with patch.object(torch, "index_select", wraps=torch.index_select) as gather:
        status = send_owner_sharded_staged(
            transfer=lambda _session, _blocks: 0,
            session_id="session",
            src_buffers=descriptors,
            logical_src_indices=np.array([0, 1], dtype=np.int32),
            dst_ptrs=[1000, 2000],
            logical_dst_indices=np.array([3, 4], dtype=np.int32),
            cp_rank=0,
            cp_size=2,
            staging_buffer=staging,
            cache=OwnerShardedStagingCache(),
        )

    assert status == 0
    assert gather.call_count == 1
    assert staging.buffer.tolist() == [0, 1, 8, 9, 4, 5, 12, 13]


def test_owner_staging_cache_falls_back_when_source_exceeds_buffer():
    source = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.uint8)
    descriptor = OwnerShardedTransferBuffer(
        tensor=source,
        item_bytes=4,
        owner_page_bytes=4,
    )
    staging = _StagingBuffer(4)
    cache = OwnerShardedStagingCache()
    observed = []

    def transfer(session, blocks):
        for src_ptr, _dst_ptr, size in blocks:
            offset = src_ptr - staging.get_ptr()
            observed.append(
                (session, bytes(staging.buffer[offset : offset + size].tolist()))
            )
        return 0

    common = dict(
        transfer=transfer,
        src_buffers=[descriptor],
        logical_src_indices=np.array([0, 1], dtype=np.int32),
        dst_ptrs=[1000],
        logical_dst_indices=np.array([4, 6], dtype=np.int32),
        cp_rank=0,
        cp_size=1,
        staging_buffer=staging,
        cache=cache,
    )
    assert send_owner_sharded_staged(session_id="first", **common) == 0
    source.fill_(9)
    assert send_owner_sharded_staged(session_id="second", **common) == 0

    assert not cache.populated
    assert observed == [
        ("first", bytes([1, 2, 3, 4])),
        ("first", bytes([5, 6, 7, 8])),
        ("second", bytes([9, 9, 9, 9])),
        ("second", bytes([9, 9, 9, 9])),
    ]


def test_owner_staging_rejects_mismatched_metadata():
    descriptor = OwnerShardedTransferBuffer(
        tensor=torch.zeros((1, 4), dtype=torch.uint8),
        item_bytes=4,
        owner_page_bytes=4,
        owner_pages_per_item=1,
    )

    try:
        send_owner_sharded_staged(
            transfer=lambda *_: 0,
            session_id="session",
            src_buffers=[descriptor],
            logical_src_indices=np.array([0], dtype=np.int32),
            dst_ptrs=[],
            logical_dst_indices=np.array([0], dtype=np.int32),
            cp_rank=0,
            cp_size=2,
            staging_buffer=_StagingBuffer(4),
        )
    except ValueError as exc:
        assert "metadata" in str(exc)
    else:
        raise AssertionError("expected mismatched transfer metadata to fail")


def test_shared_pd_tensor_descriptor_validates_geometry():
    try:
        OwnerShardedTransferBuffer(
            tensor=torch.zeros((1, 7), dtype=torch.uint8),
            item_bytes=8,
            owner_page_bytes=4,
            owner_pages_per_item=2,
        )
    except ValueError as exc:
        assert "owner-page rows" in str(exc)
    else:
        raise AssertionError("expected invalid owner-page row width to fail")


def test_dsv4_main_transfer_descriptors_follow_wire_pointer_order():
    pool = object.__new__(SharedDeepSeekV4TokenToKVPool)
    c4 = torch.zeros((3, 16), dtype=torch.uint8)
    indexer = torch.zeros((3, 24), dtype=torch.uint8)
    c128 = torch.zeros((3, 32), dtype=torch.uint8)
    pool.c4_kv_pool = SimpleNamespace(local_kv_buffer=[c4])
    pool.c4_indexer_kv_pool = SimpleNamespace(local_index_k_with_scale_buffer=[indexer])
    pool.c128_kv_pool = SimpleNamespace(local_kv_buffer=[c128])
    pool.get_contiguous_buf_infos = lambda: ([1, 2, 3], [], [])

    descriptors = pool.get_owner_sharded_kv_transfer_buffers()

    assert all(
        descriptor.tensor is expected
        for descriptor, expected in zip(descriptors, [c4, indexer, c128])
    )
    assert [descriptor.item_bytes for descriptor in descriptors] == [16, 24, 32]
    assert all(descriptor.owner_pages_per_item == 1 for descriptor in descriptors)


def test_dsv4_rank_aggregated_descriptors_use_global_views_and_rank_stride():
    pool = object.__new__(SharedDeepSeekV4TokenToKVPool)
    c4 = torch.zeros((7, 16), dtype=torch.uint8)
    indexer = torch.zeros((9, 24), dtype=torch.uint8)
    c128 = torch.zeros((11, 32), dtype=torch.uint8)

    def shared_pool(tensor, rank_stride_rows):
        return SimpleNamespace(
            shared_family=SimpleNamespace(
                global_views=[tensor],
                slab=SimpleNamespace(rank_stride_rows=rank_stride_rows),
            )
        )

    pool.c4_kv_pool = shared_pool(c4, 3)
    pool.c4_indexer_kv_pool = shared_pool(indexer, 4)
    pool.c128_kv_pool = shared_pool(c128, 5)
    pool.get_contiguous_buf_infos = lambda: ([1, 2, 3], [], [])

    descriptors = pool.get_rank_aggregated_kv_transfer_buffers()

    assert all(
        descriptor.tensor is expected
        for descriptor, expected in zip(descriptors, [c4, indexer, c128])
    )
    assert [descriptor.rank_stride_owner_pages for descriptor in descriptors] == [
        3,
        4,
        5,
    ]


def test_mooncake_rank_aggregation_selects_only_matching_decode_tp_rank():
    manager = object.__new__(MooncakeKVManager)
    manager.attn_cp_rank = 2
    manager.attn_cp_size = 8

    assert manager._is_rank_aggregated_kv_target(
        SimpleNamespace(dst_tp_rank=10, dst_attn_tp_size=8)
    )
    assert not manager._is_rank_aggregated_kv_target(
        SimpleNamespace(dst_tp_rank=3, dst_attn_tp_size=8)
    )
    assert not manager._is_rank_aggregated_kv_target(
        SimpleNamespace(dst_tp_rank=2, dst_attn_tp_size=4)
    )


def test_dsv4_state_transfer_descriptors_capture_family_geometry():
    pool = object.__new__(SharedDeepSeekV4TokenToKVPool)
    swa = torch.zeros((3, 12), dtype=torch.uint8)
    c4_attn = torch.zeros((12, 5), dtype=torch.uint8)
    c4_indexer = torch.zeros((12, 7), dtype=torch.uint8)
    c128 = torch.zeros((128, 9), dtype=torch.uint8)

    def state_pool(tensor, ratio, ring_size):
        return SimpleNamespace(
            shared_family=SimpleNamespace(local_views=[tensor]),
            shared_layer_id=0,
            ratio=ratio,
            ring_size=ring_size,
        )

    pool.swa_kv_pool = SimpleNamespace(local_kv_buffer=[swa])
    pool.compress_state_pools = [
        state_pool(c4_attn, 4, 8),
        state_pool(c128, 128, 128),
    ]
    pool.indexer_compress_state_pools = [state_pool(c4_indexer, 4, 8), None]
    pool.get_state_buf_infos = lambda: ([1, 2, 3], [], [])
    pool.get_c128_state_buf_infos = lambda: ([4], [], [])

    state = pool.get_owner_sharded_state_transfer_buffers()
    c128_state = pool.get_owner_sharded_c128_state_transfer_buffers()

    assert all(
        descriptor.tensor is expected
        for descriptor, expected in zip(state, [swa, c4_attn, c4_indexer])
    )
    assert [
        (d.item_bytes, d.owner_page_bytes, d.owner_pages_per_item) for d in state
    ] == [
        (12, 12, 1),
        (40, 20, 2),
        (56, 28, 2),
    ]
    assert [
        (
            c128_state[0].item_bytes,
            c128_state[0].owner_page_bytes,
            c128_state[0].owner_pages_per_item,
        )
    ] == [(1152, 1152, 1)]
