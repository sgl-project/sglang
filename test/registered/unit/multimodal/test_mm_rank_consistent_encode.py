"""CPU tests for rank-consistent multimodal encode decisions.

The mm embedding cache is per scheduler process, so attention-TP ranks can
disagree on which items need the ViT; the DP-sharded encoder ends in a
collective, so a hit-on-one-rank / miss-on-another split deadlocks the
group. These tests drive the three cache-gated encode helpers —
'_batch_encode_per_image_misses', '_get_chunked_embedding_full', and
'_get_chunked_embedding_by_item' — with a fake group whose all_reduce
injects the peers' miss flags.
"""

from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.managers import mm_schedule
from sglang.srt.managers.mm_schedule import PerImageRequestInfo
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeAttnTpGroup:
    """all_reduce(SUM) over this rank's flags plus a preset peer vector."""

    def __init__(self, peer_flags):
        self.peer_flags = peer_flags
        self.calls = []

    def all_reduce(self, input_):
        self.calls.append(input_.clone())
        peer = torch.tensor(self.peer_flags, dtype=input_.dtype, device=input_.device)
        assert peer.shape == input_.shape
        return input_ + peer


def _item(hash_value, tokens):
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        hash=hash_value,
        offsets=[(0, tokens - 1)],
        feature=torch.zeros(tokens, 2),
    )
    return item


@pytest.fixture(autouse=True)
def _fresh_cache():
    mm_schedule.init_mm_embedding_cache(1 << 20)
    yield
    mm_schedule.init_mm_embedding_cache(1 << 20)


def _requests(items):
    offsets = []
    cursor = 0
    for item in items:
        n = item.offsets[0][1] - item.offsets[0][0] + 1
        offsets.append((cursor, cursor + n - 1))
        cursor += n
    return [
        PerImageRequestInfo(
            req_idx=0,
            items=list(items),
            items_offset=offsets,
            extend_prefix_len=0,
            extend_seq_len=cursor,
        )
    ]


def _run(items, group, encode):
    with patch.object(mm_schedule, "_mm_encode_sync_group", return_value=group):
        return mm_schedule._batch_encode_per_image_misses(
            encode, _requests(items), torch.device("cpu")
        )


def test_local_hit_peer_miss_forces_reencode():
    item = _item(11, 4)
    cached = torch.ones(4, 2)
    mm_schedule.embedding_cache.set(11, mm_schedule.EmbeddingResult(embedding=cached))
    group = _FakeAttnTpGroup(peer_flags=[1])
    encode = Mock(return_value=torch.full((4, 2), 2.0))

    out = _run([item], group, encode)

    encode.assert_called_once()
    assert encode.call_args.args[0] == [item]
    assert group.calls[0].tolist() == [0]
    assert torch.equal(out[(11, 4)], torch.full((4, 2), 2.0))


def test_all_ranks_hit_skips_encoder_and_still_syncs():
    item = _item(12, 3)
    mm_schedule.embedding_cache.set(
        12, mm_schedule.EmbeddingResult(embedding=torch.ones(3, 2))
    )
    group = _FakeAttnTpGroup(peer_flags=[0])
    encode = Mock()

    out = _run([item], group, encode)

    encode.assert_not_called()
    assert len(group.calls) == 1
    assert torch.equal(out[(12, 3)], torch.ones(3, 2))


def test_encode_order_follows_batch_order_not_local_miss_order():
    a, b = _item(21, 2), _item(22, 3)
    # This rank hit 'a' and missed 'b'; a peer missed 'a'.
    mm_schedule.embedding_cache.set(
        21, mm_schedule.EmbeddingResult(embedding=torch.ones(2, 2))
    )
    group = _FakeAttnTpGroup(peer_flags=[1, 0])
    encode = Mock(return_value=torch.arange(10.0).reshape(5, 2))

    out = _run([a, b], group, encode)

    assert encode.call_args.args[0] == [a, b]
    assert torch.equal(out[(21, 2)], torch.arange(4.0).reshape(2, 2))
    assert torch.equal(out[(22, 3)], torch.arange(4.0, 10.0).reshape(3, 2))


def test_single_rank_does_not_sync():
    item = _item(31, 2)
    mm_schedule.embedding_cache.set(
        31, mm_schedule.EmbeddingResult(embedding=torch.ones(2, 2))
    )
    encode = Mock()

    with (
        patch.object(
            mm_schedule.torch.distributed, "is_initialized", return_value=True
        ),
        patch.object(mm_schedule, "get_parallel", return_value=Mock(attn_tp_size=1)),
    ):
        assert mm_schedule._mm_encode_sync_group() is None

    out = _run([item], None, encode)
    encode.assert_not_called()
    assert torch.equal(out[(31, 2)], torch.ones(2, 2))


def test_sync_group_none_without_distributed_init():
    with patch.object(
        mm_schedule.torch.distributed, "is_initialized", return_value=False
    ):
        assert mm_schedule._mm_encode_sync_group() is None


def test_rank_consistent_miss_hashes_is_union():
    group = _FakeAttnTpGroup(peer_flags=[0, 1, 0])
    got = mm_schedule._rank_consistent_miss_hashes(
        [(1, 4), (2, 4), (3, 4)], {(1, 4)}, group, torch.device("cpu")
    )
    assert got == {(1, 4), (2, 4)}


def _offsets(items):
    offsets = []
    cursor = 0
    for item in items:
        n = item.offsets[0][1] - item.offsets[0][0] + 1
        offsets.append((cursor, cursor + n - 1))
        cursor += n
    return offsets, cursor


def _run_full(items, group, encode):
    offsets, cursor = _offsets(items)
    input_ids = torch.zeros(cursor, dtype=torch.long)
    with patch.object(mm_schedule, "_mm_encode_sync_group", return_value=group):
        return mm_schedule._get_chunked_embedding_full(
            encode, list(items), offsets, 0, cursor, input_ids, torch.device("cpu")
        )


def _run_by_item(items, group, encode):
    offsets, cursor = _offsets(items)
    with patch.object(mm_schedule, "_mm_encode_sync_group", return_value=group):
        return mm_schedule._get_chunked_embedding_by_item(
            encode, list(items), offsets, 0, cursor, torch.device("cpu")
        )


def _combined_hash(items):
    return mm_schedule.MultiModalStaticCache.combine_hashes(
        [item.hash for item in items]
    )


def test_full_path_local_hit_peer_miss_forces_reencode():
    item = _item(41, 4)
    mm_schedule.embedding_cache.set(
        _combined_hash([item]),
        mm_schedule.EmbeddingResult(embedding=torch.ones(4, 2)),
    )
    group = _FakeAttnTpGroup(peer_flags=[1])
    encode = Mock(return_value=torch.full((4, 2), 3.0))

    with patch.object(mm_schedule, "_acknowledge_deferred_cuda_ipc_cache_hits") as ack:
        chunk, _ = _run_full([item], group, encode)

    encode.assert_called_once()
    assert encode.call_args.args[0] == [item]
    assert group.calls[0].tolist() == [0]
    # A forced re-encode must not release the item's feature as a cache hit.
    ack.assert_not_called()
    assert torch.equal(chunk, torch.full((4, 2), 3.0))


def test_full_path_all_ranks_hit_skips_encoder_and_syncs_once():
    item = _item(42, 3)
    mm_schedule.embedding_cache.set(
        _combined_hash([item]),
        mm_schedule.EmbeddingResult(embedding=torch.ones(3, 2)),
    )
    group = _FakeAttnTpGroup(peer_flags=[0])
    encode = Mock()

    with patch.object(mm_schedule, "_acknowledge_deferred_cuda_ipc_cache_hits") as ack:
        chunk, _ = _run_full([item], group, encode)

    encode.assert_not_called()
    assert len(group.calls) == 1
    ack.assert_called_once_with([item])
    assert torch.equal(chunk, torch.ones(3, 2))


def test_full_path_single_rank_does_not_sync():
    item = _item(43, 3)
    mm_schedule.embedding_cache.set(
        _combined_hash([item]),
        mm_schedule.EmbeddingResult(embedding=torch.ones(3, 2)),
    )
    encode = Mock()

    with patch.object(mm_schedule, "_acknowledge_deferred_cuda_ipc_cache_hits"):
        chunk, _ = _run_full([item], None, encode)

    encode.assert_not_called()
    assert torch.equal(chunk, torch.ones(3, 2))


def test_by_item_local_hit_peer_miss_forces_reencode():
    item = _item(51, 4)
    mm_schedule.embedding_cache.set(
        51, mm_schedule.EmbeddingResult(embedding=torch.ones(4, 2))
    )
    group = _FakeAttnTpGroup(peer_flags=[1])
    encode = Mock(return_value=torch.full((4, 2), 5.0))

    with patch.object(mm_schedule, "_acknowledge_deferred_cuda_ipc_cache_hits") as ack:
        chunk = _run_by_item([item], group, encode)

    encode.assert_called_once()
    assert encode.call_args.args[0] == [item]
    assert group.calls[0].tolist() == [0]
    ack.assert_not_called()
    assert torch.equal(chunk, torch.full((4, 2), 5.0))
    # MultiModalStaticCache.set is no-overwrite: the current batch uses the
    # fresh encoding from the local map while the cache keeps the original
    # entry, which the ViT's determinism makes equivalent.
    assert torch.equal(
        mm_schedule.embedding_cache.get_single(51).embedding,
        torch.ones(4, 2),
    )


def test_by_item_encode_order_follows_batch_order_not_local_miss_order():
    a, b = _item(61, 2), _item(62, 3)
    # This rank hit 'a' and missed 'b'; a peer missed 'a'.
    mm_schedule.embedding_cache.set(
        61, mm_schedule.EmbeddingResult(embedding=torch.ones(2, 2))
    )
    group = _FakeAttnTpGroup(peer_flags=[1, 0])
    encode = Mock(return_value=torch.arange(10.0).reshape(5, 2))

    with patch.object(mm_schedule, "_acknowledge_deferred_cuda_ipc_cache_hits") as ack:
        chunk = _run_by_item([a, b], group, encode)

    assert encode.call_args.args[0] == [a, b]
    ack.assert_not_called()
    assert torch.equal(chunk, torch.arange(10.0).reshape(5, 2))


def test_by_item_all_ranks_hit_skips_encoder_and_syncs_once():
    item = _item(71, 3)
    mm_schedule.embedding_cache.set(
        71, mm_schedule.EmbeddingResult(embedding=torch.ones(3, 2))
    )
    group = _FakeAttnTpGroup(peer_flags=[0])
    encode = Mock()

    with patch.object(mm_schedule, "_acknowledge_deferred_cuda_ipc_cache_hits") as ack:
        chunk = _run_by_item([item], group, encode)

    encode.assert_not_called()
    assert len(group.calls) == 1
    ack.assert_called_once_with([item])
    assert torch.equal(chunk, torch.ones(3, 2))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
