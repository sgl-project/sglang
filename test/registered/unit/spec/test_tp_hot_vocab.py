from types import SimpleNamespace

import pytest
import torch
from sglang.srt.speculative.tp_hot_vocab import build_sharded_hot_head


def layer(rank, tp, full_width):
    width = full_width // tp
    return SimpleNamespace(
        org_vocab_size=full_width,
        num_embeddings=full_width,
        tp_size=tp,
        shard_indices=SimpleNamespace(
            org_vocab_start_index=rank * width, org_vocab_end_index=(rank + 1) * width
        ),
    )


def test_cross_shard_ids_preserve_order_target_and_padding():
    full = torch.arange(256 * 8, dtype=torch.float32).reshape(256, 8)
    hot = torch.tensor([201, 0, 79, 251, 141, 11])
    padded = torch.zeros(256, 8)
    padded[: len(hot)] = full[hot]
    outputs = []
    for rank in range(4):
        original = full[rank * 64 : (rank + 1) * 64].clone()

        def reduce(contribution):
            expected = padded.clone()
            owned = (hot >= rank * 64) & (hot < (rank + 1) * 64)
            expected[: len(hot)][~owned] = 0
            assert torch.equal(contribution, expected)
            return padded

        group = SimpleNamespace(world_size=4, rank_in_group=rank, all_reduce=reduce)
        result = build_sharded_hot_head(original, hot, layer(rank, 4, 256), group)
        outputs.append(result)
        assert torch.equal(original, full[rank * 64 : (rank + 1) * 64])
        assert result.data_ptr() != original.data_ptr()
    assert torch.equal(torch.cat(outputs)[: len(hot)], full[hot])
    assert torch.count_nonzero(torch.cat(outputs)[len(hot) :]) == 0


@pytest.mark.parametrize("hot", [[0, 0], [-1], [256], []])
def test_invalid_token_maps_fail(hot):
    with pytest.raises(ValueError):
        build_sharded_hot_head(
            torch.zeros(256, 8),
            torch.tensor(hot),
            layer(0, 1, 256),
            SimpleNamespace(world_size=1, rank_in_group=0),
        )
