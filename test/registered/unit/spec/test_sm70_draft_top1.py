from types import SimpleNamespace

import pytest
import torch
from sglang.srt.speculative.sm70_draft_top1 import (
    configure_draft_top1,
    draft_top1,
    exchange_top1,
)


def server_args(**overrides):
    values = {"enable_dp_attention": False, "enable_lora": False, "pp_size": 1}
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("rows", [1, 4, 16])
@pytest.mark.parametrize("kind", ["random", "ties", "nonfinite"])
def test_global_winner_matches_full_vocabulary_with_padding(rows, kind):
    torch.manual_seed(431)
    shards = [torch.randn(rows, 35) for _ in range(4)]
    for shard in shards:
        shard[:, 31:] = float("inf")
        if kind == "ties":
            shard[:, :31] = 0
            shard[:, 5] = 8
            shard[:, 9] = 8
        elif kind == "nonfinite":
            shard[:, 3] = float("inf")
            shard[0, 7] = float("nan")

    def gather(candidate, dim):
        result = []
        for rank, shard in enumerate(shards):
            value, index = shard[:, :31].max(-1)
            result.append(torch.stack((value, (index + rank * 31).float()), dim=-1))
        torch.testing.assert_close(candidate, result[2], equal_nan=True)
        return torch.cat(result, dim=dim)

    group = SimpleNamespace(world_size=4, all_gather=gather)
    expected = torch.cat([s[:, :31] for s in shards], dim=1).argmax(-1, keepdim=True)
    actual = exchange_top1(shards[2], 62, 31, group)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_default_is_unchanged_and_needs_no_cuda_initialization(monkeypatch):
    monkeypatch.delenv("SGLANG_SM70_MTP_LOCAL_TOP1", raising=False)
    assert configure_draft_top1(None, None, None, None) is False
    values = torch.tensor([[1.0, 9.0, 9.0], [-1.0, -2.0, -3.0]])
    torch.testing.assert_close(
        draft_top1(values, SimpleNamespace()), values.argmax(-1, keepdim=True)
    )


def test_explicit_unsupported_configuration_is_rejected(monkeypatch):
    monkeypatch.setenv("SGLANG_SM70_MTP_LOCAL_TOP1", "1")
    runner = SimpleNamespace(model=SimpleNamespace())
    with pytest.raises(ValueError, match="topk=1"):
        configure_draft_top1(runner, 2, None, server_args())
    with pytest.raises(ValueError, match="topk=1"):
        configure_draft_top1(runner, 1, torch.zeros(4), server_args())
    with pytest.raises(ValueError, match="topk=1"):
        configure_draft_top1(runner, 1, None, server_args(enable_lora=True))
