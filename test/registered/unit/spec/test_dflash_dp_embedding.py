from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers import vocab_parallel_embedding as embedding_module
from sglang.srt.speculative import dflash_worker_v2 as worker_module
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import published_topology

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture
def topology():
    with published_topology():
        yield


def _embedding(monkeypatch, *, tp_size, rank, use_attn_tp_group, added=0):
    monkeypatch.setattr(
        embedding_module,
        "get_parallel",
        lambda: SimpleNamespace(
            tp_size=tp_size,
            tp_rank=rank,
            attn_tp_size=tp_size,
            attn_tp_rank=rank,
        ),
    )
    embed = embedding_module.VocabParallelEmbedding(
        5 + added,
        2,
        org_num_embeddings=5,
        padding_size=4,
        use_attn_tp_group=use_attn_tp_group,
    )
    expected = torch.arange((5 + added) * 2, dtype=embed.weight.dtype).reshape(-1, 2)
    # A learned mask row must survive replication as well as reconstruction.
    expected[-1].fill_(123)
    indices = embed.shard_indices
    with torch.no_grad():
        embed.weight.fill_(-999)
        embed.weight[: indices.num_org_elements].copy_(
            expected[indices.org_vocab_start_index : indices.org_vocab_end_index]
        )
        offset = indices.num_org_elements_padded
        embed.weight[offset : offset + indices.num_added_elements].copy_(
            expected[indices.added_vocab_start_index : indices.added_vocab_end_index]
        )
    return embed, expected


def _worker(embed, vocab_size):
    return SimpleNamespace(
        _full_embed_gpu=None,
        _target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                model=SimpleNamespace(get_input_embeddings=lambda: embed),
                model_config=SimpleNamespace(vocab_size=vocab_size),
            )
        ),
    )


def _parallel(monkeypatch):
    parallel = SimpleNamespace(
        attn_dp_enabled=True,
        tp_rank=0,
        tp_group=SimpleNamespace(device_group=object()),
        attn_tp_group=SimpleNamespace(device_group=object()),
    )
    monkeypatch.setattr(worker_module, "get_parallel", lambda: parallel)
    return parallel


@pytest.mark.parametrize("use_attn_tp_group", [False, True])
def test_replicated_embedding_reuses_storage_without_collective(
    monkeypatch, topology, use_attn_tp_group
):
    embed, expected = _embedding(
        monkeypatch, tp_size=1, rank=0, use_attn_tp_group=use_attn_tp_group
    )
    _parallel(monkeypatch)
    monkeypatch.setattr(
        worker_module.dist,
        "all_gather",
        lambda *a, **k: pytest.fail("a replicated table needs no collective"),
    )
    worker = _worker(embed, len(expected))
    worker_module.DFlashWorkerV2._cache_full_embed_weight(worker)
    torch.testing.assert_close(worker._full_embed_gpu, expected)
    assert worker._full_embed_gpu.data_ptr() == embed.weight.data_ptr()
    assert worker._full_embed_gpu.untyped_storage().nbytes() == (
        embed.weight.untyped_storage().nbytes()
    )


@pytest.mark.parametrize("use_attn_tp_group,tp_size", [(True, 2), (False, 4)])
def test_sharded_embedding_uses_own_group_and_preserves_added_vocab(
    monkeypatch, topology, use_attn_tp_group, tp_size
):
    shards = [
        _embedding(
            monkeypatch,
            tp_size=tp_size,
            rank=rank,
            use_attn_tp_group=use_attn_tp_group,
            added=3,
        )[0]
        for rank in range(tp_size)
    ]
    parallel = _parallel(monkeypatch)
    expected_group = (
        parallel.attn_tp_group if use_attn_tp_group else parallel.tp_group
    ).device_group
    calls = []

    def all_gather(outputs, value, *, group):
        assert group is expected_group
        assert len(outputs) == tp_size
        # Valid rows differ across ranks. The collective must use padded,
        # equal-sized tensors, including the entirely padded final TP4 shard.
        assert all(part.shape == value.shape for part in outputs)
        torch.testing.assert_close(value, shards[0].weight)
        for output, shard in zip(outputs, shards):
            output.copy_(shard.weight)
        calls.append(group)

    monkeypatch.setattr(worker_module.dist, "all_gather", all_gather)
    worker = _worker(shards[0], 8)
    worker_module.DFlashWorkerV2._cache_full_embed_weight(worker)
    expected = torch.arange(16, dtype=shards[0].weight.dtype).reshape(8, 2)
    expected[-1].fill_(123)
    torch.testing.assert_close(worker._full_embed_gpu, expected)
    assert calls == [expected_group]
    assert worker._full_embed_gpu.untyped_storage().nbytes() == expected.numel() * (
        expected.element_size()
    )


def test_replicated_added_vocab_removes_internal_padding(monkeypatch, topology):
    embed, expected = _embedding(
        monkeypatch, tp_size=1, rank=0, use_attn_tp_group=True, added=3
    )
    _parallel(monkeypatch)
    monkeypatch.setattr(
        worker_module.dist,
        "all_gather",
        lambda *a, **k: pytest.fail("a replicated table needs no collective"),
    )
    worker = _worker(embed, len(expected))
    worker_module.DFlashWorkerV2._cache_full_embed_weight(worker)
    torch.testing.assert_close(worker._full_embed_gpu, expected)


def test_non_dp_embedding_keeps_existing_lookup(monkeypatch, topology):
    parallel = _parallel(monkeypatch)
    parallel.attn_dp_enabled = False
    worker = SimpleNamespace(_full_embed_gpu=None)
    worker_module.DFlashWorkerV2._cache_full_embed_weight(worker)
    assert worker._full_embed_gpu is None
