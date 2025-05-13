import logging
from typing import Dict, List

import torch
from sglang.srt.managers.expert_location import ExpertLocationMetadata
from sglang.srt.managers.schedule_batch import get_global_expert_location_metadata

logger = logging.getLogger(__name__)


def update_expert_location(
    routed_experts_weights_of_layer: Dict[int, List[torch.Tensor]],
    new_expert_location_metadata: ExpertLocationMetadata,
):
    old_expert_location_metadata = get_global_expert_location_metadata()
    _update_expert_weights(old_expert_location_metadata, new_expert_location_metadata, routed_experts_weights_of_layer)
    old_expert_location_metadata.update(new_expert_location_metadata)


def _update_expert_weights(
    routed_experts_weights_of_layer: Dict[int, List[torch.Tensor]],
    old_expert_location_metadata: ExpertLocationMetadata,
    new_expert_location_metadata: ExpertLocationMetadata,
):
    temp_buffers = [torch.empty_like(tensor) for tensor in next(iter(routed_experts_weights_of_layer.values()))]
    for layer_id in sorted(routed_experts_weights_of_layer.keys()):
        update_expert_weights_single_layer(
            routed_experts_weights=routed_experts_weights_of_layer[layer_id],
            temp_buffers=temp_buffers,
            old_expert_location_metadata=old_expert_location_metadata,
            new_expert_location_metadata=new_expert_location_metadata,
        )


def update_expert_weights_single_layer(
    routed_experts_weights: List[torch.Tensor],
    temp_buffers: List[torch.Tensor],
    old_expert_location_metadata: ExpertLocationMetadata,
    new_expert_location_metadata: ExpertLocationMetadata,
):
    assert all(tensor.shape[0] == num_local_physical_experts for tensor in routed_experts_weights)
    TODO
