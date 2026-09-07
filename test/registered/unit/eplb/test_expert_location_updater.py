from unittest.mock import MagicMock, patch

import torch

from sglang.srt.eplb.expert_location_updater import _update_expert_weights_raw


def test_gpu_per_node_uses_original_ep_size_after_rank_failure():
    old_metadata = MagicMock()
    old_metadata.ep_size = 4
    old_metadata.num_local_physical_experts = 2
    old_metadata.physical_to_logical_map_cpu = torch.zeros((1, 8), dtype=torch.int64)

    new_metadata = MagicMock()
    new_metadata.physical_to_logical_map_cpu = torch.zeros((1, 8), dtype=torch.int64)

    with (
        patch("torch.distributed.get_world_size", return_value=3),
        patch(
            "sglang.srt.eplb.expert_location_updater.create_temp_buffers",
            return_value=[],
        ),
        patch(
            "sglang.srt.eplb.expert_location_updater.update_expert_weights_single_layer"
        ) as update_single_layer,
    ):
        _update_expert_weights_raw(
            routed_experts_weights_of_layer={0: [torch.empty(2)]},
            old_expert_location_metadata=old_metadata,
            new_expert_location_metadata=new_metadata,
            update_layer_ids=[0],
            nnodes=4,
            rank=0,
        )

    assert update_single_layer.call_args.kwargs["world_size"] == 3
    assert update_single_layer.call_args.kwargs["num_gpu_per_node"] == 1
