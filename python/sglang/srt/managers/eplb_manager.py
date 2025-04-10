from typing import TYPE_CHECKING

import torch
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers import deepseek_eplb
from sglang.srt.managers.expert_distribution_storage import ExpertDistributionStorage
from sglang.srt.managers.expert_location import ExpertLocationMetadata, ModelConfigForExpertLocation
from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager


class EPLBManager:
    def __init__(self, tokenizer_manager: TokenizerManager):
        self._tokenizer_manager = tokenizer_manager
        self._expert_distribution_storage = ExpertDistributionStorage()

    async def rebalance_experts(self):
        TODO_may_or_may_not_save_current
        logical_count = self._expert_distribution_storage.get_last_snapshot()
        expert_location_metadata = _compute_expert_location_metadata(logical_count)
        await self._tokenizer_manager.update_expert_location_metadata(expert_location_metadata)


# TODO maybe move to ExpertLocationMetadata static method?
def _compute_expert_location_metadata(server_args: ServerArgs, logical_count: torch.Tensor):
    model_config = ModelConfig.from_server_args(server_args)
    model_config_for_expert_location = ModelConfigForExpertLocation.from_model_config(model_config)
    physical_to_logical_map, logical_to_physical_map, expert_count = deepseek_eplb.rebalance_experts(
        weight=logical_count,
        num_replicas=TODO,
        num_groups=model_config_for_expert_location.num_groups,
        num_nodes=server_args.nnodes,
        num_gpus=world_size,
    )
    return ExpertLocationMetadata(
        num_layers=model_config_for_expert_location.num_layers,
        num_local_physical_experts=TODO,
        num_logical_experts=model_config_for_expert_location.num_logical_experts,
        physical_to_logical_map=physical_to_logical_map,
        logical_to_all_physical_map=logical_to_physical_map,
        logical_to_rank_dispatch_physical_map=_compute_logical_to_rank_dispatch_physical_map(logical_to_physical_map),
    )


def _compute_logical_to_rank_dispatch_physical_map(logical_to_physical_map: torch.Tensor):
    return TODO
