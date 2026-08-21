"""Single-rank groups get gloo, so NCCL does not reserve device buffers for them."""

import unittest
from unittest.mock import patch

from sglang.multimodal_gen.runtime.distributed.group_coordinator import (
    new_device_group,
)

NEW_GROUP_PATH = "torch.distributed.new_group"


class TestSingleRankDeviceGroup(unittest.TestCase):
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

    def test_backend_defaults_to_none_for_multi_rank(self):
        with patch(NEW_GROUP_PATH) as new_group:
            new_device_group([0, 1])
        new_group.assert_called_once_with([0, 1], backend=None)


if __name__ == "__main__":
    unittest.main()
