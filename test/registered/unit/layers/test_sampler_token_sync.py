from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist

from sglang.srt.layers.sampler import Sampler


def _sampler_with_group(group):
    sampler = Sampler.__new__(Sampler)
    object.__setattr__(sampler, "tp_sync_group", group)
    return sampler


def test_token_sync_skips_singleton_group():
    group = object()
    sampler = _sampler_with_group(group)
    token_ids = torch.tensor([3], dtype=torch.int64)
    sampling_info = SimpleNamespace(grammars=[object()])

    with (
        mock.patch.object(dist, "get_world_size", return_value=1) as get_world_size,
        mock.patch.object(dist, "all_reduce") as all_reduce,
    ):
        sampler._sync_token_ids_across_tp(token_ids, sampling_info)

    get_world_size.assert_called_once_with(group)
    all_reduce.assert_not_called()


def test_token_sync_preserves_min_reduce_for_multi_rank_group():
    group = object()
    sampler = _sampler_with_group(group)
    token_ids = torch.tensor([3], dtype=torch.int64)
    sampling_info = SimpleNamespace(grammars=[object()])

    with (
        mock.patch.object(dist, "get_world_size", return_value=2) as get_world_size,
        mock.patch.object(dist, "all_reduce") as all_reduce,
    ):
        sampler._sync_token_ids_across_tp(token_ids, sampling_info)

    get_world_size.assert_called_once_with(group)
    all_reduce.assert_called_once_with(
        token_ids,
        op=dist.ReduceOp.MIN,
        group=group,
    )
