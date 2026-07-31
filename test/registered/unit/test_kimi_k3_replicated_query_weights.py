"""CPU tests for replicated DCP Query projection weight storage."""

import torch

from sglang.srt.layers.dcp.query_weights import (
    bind_parameter_to_replicated_rank_slice_,
    refresh_replicated_weight_,
    replicated_rank_slice,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeAllGatherGroup:
    def __init__(self, shards, rank):
        self.shards = shards
        self.world_size = len(shards)
        self.rank_in_group = rank

    def all_gather(self, input_, dim=-1):
        shards = list(self.shards)
        shards[self.rank_in_group] = input_
        return torch.cat(shards, dim=dim)


def test_replicated_query_weight_owns_local_parameter_storage():
    local = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).view(3, 4))
    local.weight_loader = object()
    replicated = torch.cat((local.detach(), local.detach() + 100), dim=0)

    local_view = bind_parameter_to_replicated_rank_slice_(
        local,
        replicated,
        rank=1,
        world_size=2,
    )

    assert local.weight_loader is not None
    assert local.data_ptr() == local_view.data_ptr()
    assert local.untyped_storage().data_ptr() == replicated.untyped_storage().data_ptr()
    assert local.storage_offset() == 12
    torch.testing.assert_close(local, replicated[3:])

    local.data.fill_(7)
    torch.testing.assert_close(replicated[3:], torch.full((3, 4), 7.0))
    torch.testing.assert_close(
        replicated[:3],
        torch.arange(12, dtype=torch.float32).view(3, 4),
    )


def test_refresh_replicated_query_weight_preserves_buffer_pointer():
    shards = [torch.full((2, 3), float(rank), dtype=torch.float32) for rank in range(4)]
    replicated = torch.cat(shards, dim=0)
    local = replicated[4:6]
    local.fill_(20)
    original_ptr = replicated.data_ptr()

    refresh_replicated_weight_(
        local,
        replicated,
        group=_FakeAllGatherGroup(shards, rank=2),
    )

    assert replicated.data_ptr() == original_ptr
    expected_shards = list(shards)
    expected_shards[2] = torch.full((2, 3), 20.0)
    torch.testing.assert_close(replicated, torch.cat(expected_shards, dim=0))


def test_replicated_query_tensor_rank_slice_can_change_local_layout():
    local_shape = (2, 3, 4)
    replicated = torch.arange(48, dtype=torch.float32).view(4, 3, 4)

    local = replicated_rank_slice(
        replicated,
        local_shape=local_shape,
        rank=1,
        world_size=2,
    )

    assert local.shape == local_shape
    assert local.stride() == replicated.stride()
    assert local.storage_offset() == 24
    assert local.untyped_storage().data_ptr() == replicated.untyped_storage().data_ptr()
