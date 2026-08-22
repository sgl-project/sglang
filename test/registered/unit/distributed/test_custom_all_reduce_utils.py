import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.distributed.device_communicators import custom_all_reduce_utils as caru
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def test_custom_all_reduce_gathers_cuda_uuid_identifiers():
    def fake_all_gather_object(output, value, group=None):
        output[:] = [value, "GPU-peer"]

    with (
        patch.object(caru, "_is_cuda", True),
        patch.object(caru, "_is_hip", False),
        patch.object(caru.dist, "get_backend", return_value="gloo"),
        patch.object(caru.dist, "get_rank", return_value=0),
        patch.object(caru.dist, "get_world_size", return_value=2),
        patch.object(
            caru.dist, "all_gather_object", side_effect=fake_all_gather_object
        ) as mock_all_gather_object,
        patch.object(caru.dist, "all_gather") as mock_all_gather,
        patch.object(caru, "in_the_same_node_as", return_value=[True, True]),
        patch.object(
            caru.torch.cuda,
            "get_device_properties",
            return_value=SimpleNamespace(uuid="local"),
        ),
        patch.object(caru, "is_full_nvlink", return_value=True) as mock_nvlink,
        patch.object(caru, "can_p2p", return_value=True),
        patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-local,GPU-peer"}),
    ):
        assert caru.can_use_custom_all_reduce_with_nvlink(
            group=MagicMock(),
            device=torch.device("cuda:0"),
            supported_world_size=[2],
            cls_name="CustomAllreduce",
        )

    mock_all_gather_object.assert_called_once()
    mock_all_gather.assert_not_called()
    mock_nvlink.assert_called_once_with(["GPU-local", "GPU-peer"], 2)
