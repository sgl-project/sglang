from typing import TYPE_CHECKING, Optional

import torch
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers import deepseek_eplb
from sglang.srt.managers.expert_distribution_storage import ExpertDistributionStorage
from sglang.srt.managers.expert_location import (
    ExpertLocationMetadata,
    ModelConfigForExpertLocation,
)
from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager


class EPLBManager:
    @staticmethod
    def init_new(server_args: ServerArgs):
        if server_args.enable_eplb:
            return _EPLBManagerReal(server_args)
        else:
            return _EPLBManagerNoop()

    def __init__(self):
        self.tokenizer_manager: Optional[TokenizerManager] = None

    def compute_expert_location_metadata(self) -> ExpertLocationMetadata:
        return ExpertLocationMetadata.init_trivial(TODO)


class _EPLBManagerReal(EPLBManager):
    def __init__(self, server_args: ServerArgs):
        super().__init__()
        self._server_args = server_args
        self._expert_distribution_storage = ExpertDistributionStorage()

    def compute_expert_location_metadata(self):
        logical_count = self._expert_distribution_storage.get_last_snapshot()
        if logical_count is None:
            return super().compute_expert_location_metadata()
        return _compute_expert_location_metadata_raw(self._server_args, logical_count)


class _EPLBManagerNoop(EPLBManager):
    pass


def _compute_expert_location_metadata_raw(
        server_args: ServerArgs, logical_count: torch.Tensor
):
    model_config = ModelConfig.from_server_args(server_args)
    model_config_for_expert_location = ModelConfigForExpertLocation.from_model_config(
        model_config
    )

    num_physical_experts = model_config_for_expert_location.num_logical_experts + server_args.ep_num_redundant_experts
    # TODO consider case when DP attention is disabled and DP > 1
    world_size = server_args.tp_size
    assert num_physical_experts % world_size == 0
    num_local_physical_experts = num_physical_experts // world_size

    physical_to_logical_map, logical_to_all_physical_map, expert_count = (
        deepseek_eplb.rebalance_experts(
            weight=logical_count,
            num_replicas=num_physical_experts,
            num_groups=model_config_for_expert_location.num_groups,
            num_nodes=server_args.nnodes,
            num_gpus=world_size,
        )
    )
    return ExpertLocationMetadata(
        num_layers=model_config_for_expert_location.num_layers,
        num_local_physical_experts=num_local_physical_experts,
        num_logical_experts=model_config_for_expert_location.num_logical_experts,
        physical_to_logical_map=physical_to_logical_map,
        logical_to_all_physical_map=logical_to_all_physical_map,
        logical_to_rank_dispatch_physical_map=_compute_logical_to_rank_dispatch_physical_map(
            logical_to_all_physical_map
        ),
    )


def _compute_logical_to_rank_dispatch_physical_map(
        logical_to_all_physical_map: torch.Tensor,
):
    # TODO maybe improve this algorithm (e.g. ensure it is really balanced)
    return TODO
