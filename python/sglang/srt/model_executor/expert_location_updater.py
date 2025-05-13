import logging
from typing import Dict, List

import torch
from sglang.srt.managers.expert_location import ExpertLocationMetadata
from sglang.srt.managers.schedule_batch import get_global_expert_location_metadata

logger = logging.getLogger(__name__)


def update_expert_location(
    new_expert_location_metadata: ExpertLocationMetadata,
):
    old_expert_location_metadata = get_global_expert_location_metadata()
    _update_expert_weights(old_expert_location_metadata, new_expert_location_metadata)
    old_expert_location_metadata.update(new_expert_location_metadata)


def _update_expert_weights(
    routed_experts_weights_of_layer: Dict[int, List[torch.Tensor]],
    old_expert_location_metadata: ExpertLocationMetadata,
    new_expert_location_metadata: ExpertLocationMetadata,
):
    TODO
