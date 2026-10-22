"""Single-rank groups get gloo, so NCCL does not reserve device buffers for them."""

import unittest
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.multimodal_gen.runtime.distributed.group_coordinator import (
    new_device_group,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    init_parallel_group_coordinator,
)
from sglang.multimodal_gen.runtime.platforms.cpu import CpuPlatform
from sglang.test.test_utils import CustomTestCase

NEW_GROUP_PATH = "torch.distributed.new_group"


def _check_pipeline_group_lifecycle(_rank):
    with TemporaryDirectory() as directory:
        dist.init_process_group(
            "gloo",
            init_method=Path(directory, "rendezvous").as_uri(),
            rank=0,
            world_size=1,
            timeout=timedelta(seconds=30),
        )
        try:
            process_groups = dist.distributed_c10d._world.pg_map
            initial_groups = len(process_groups)
            with patch(
                "sglang.multimodal_gen.runtime.distributed."
                "group_coordinator.current_platform",
                CpuPlatform(),
            ):
                for _ in range(2):
                    group = init_parallel_group_coordinator(
                        [[0]],
                        local_rank=0,
                        backend="gloo",
                        parallel_mode="pipeline",
                    )
                    try:
                        assert (group.world_size, group.rank_in_group) == (1, 0)
                        tensor = torch.tensor([3.0])
                        torch.testing.assert_close(group.all_reduce(tensor), tensor)
                    finally:
                        group.destroy()
                    assert len(process_groups) == initial_groups
        finally:
            dist.destroy_process_group()


class TestSingleRankDeviceGroup(CustomTestCase):
    def test_single_rank_group_avoids_the_device_backend(self):
        for ranks, requested in [([0], "nccl"), ([3], "hccl"), ([0], None)]:
            with self.subTest(ranks=ranks, requested=requested):
                with patch(NEW_GROUP_PATH) as new_group:
                    new_device_group(ranks, requested)
                new_group.assert_called_once_with(ranks, backend="gloo")

    def test_multi_rank_group_keeps_the_requested_backend(self):
        for ranks, requested in [([0, 1], "nccl"), ([0, 1, 2, 3], None)]:
            with self.subTest(ranks=ranks, requested=requested):
                with patch(NEW_GROUP_PATH) as new_group:
                    new_device_group(ranks, requested)
                new_group.assert_called_once_with(ranks, backend=requested)

    def test_pipeline_group_releases_process_groups(self):
        # Other unit tests can leave a default distributed group initialized.
        mp.spawn(_check_pipeline_group_lifecycle, nprocs=1)


if __name__ == "__main__":
    unittest.main()
