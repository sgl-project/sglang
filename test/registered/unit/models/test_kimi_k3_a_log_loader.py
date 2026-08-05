import pytest
import torch
from sglang.srt.models.kimi_k3 import (
    _copy_k3_a_log_tp_shard,
    _prepare_k3_a_log_tp_shard,
)


@pytest.mark.parametrize("rank", range(4))
def test_a_log_tp4_preparation_preserves_exact_values_and_owns_storage(rank):
    source = torch.arange(128, dtype=torch.float32)
    actual = _prepare_k3_a_log_tp_shard(
        source,
        parameter_shape=torch.Size((1, 1, 32, 1)),
        tp_rank=rank,
        tp_size=4,
    )

    assert torch.equal(actual.reshape(-1), source[rank * 32 : (rank + 1) * 32])
    assert actual.is_contiguous()
    assert actual.storage_offset() == 0
    assert actual.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()


def test_a_log_loader_copies_exact_rank_values_into_parameter():
    source = torch.arange(128, dtype=torch.float32).reshape(1, 1, 128, 1)
    parameter = torch.empty((1, 1, 32, 1), dtype=torch.float32)
    _copy_k3_a_log_tp_shard(parameter, source, tp_rank=2, tp_size=4)
    assert torch.equal(parameter.reshape(-1), torch.arange(64, 96, dtype=torch.float32))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_a_log_loader_rejects_nonfinite_selected_shard(bad):
    source = torch.arange(128, dtype=torch.float32)
    source[64] = bad
    with pytest.raises(ValueError, match="non-finite"):
        _prepare_k3_a_log_tp_shard(
            source,
            parameter_shape=torch.Size((1, 1, 32, 1)),
            tp_rank=2,
            tp_size=4,
        )


@pytest.mark.parametrize("elements", [127, 129, 160])
def test_a_log_loader_rejects_nonexact_global_extent(elements):
    with pytest.raises(ValueError, match="extent differs"):
        _prepare_k3_a_log_tp_shard(
            torch.ones(elements),
            parameter_shape=torch.Size((1, 1, 32, 1)),
            tp_rank=0,
            tp_size=4,
        )
