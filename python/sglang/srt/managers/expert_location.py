from dataclasses import dataclass
from typing import List, Optional

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_loader import get_model_architecture


@dataclass
class ExpertLocationMetadata:
    num_layers: int
    num_local_physical_experts: int
    num_logical_experts: int
    physical_to_logical_map: torch.Tensor  # (layers, num_physical_experts)
    logical_to_all_physical_map: torch.Tensor  # (layers, num_logical_experts, X)
    # (num_gpus, layers, num_logical_experts)
    logical_to_rank_dispatch_physical_map: torch.Tensor

    # -------------------------------- construction and mutation ------------------------------------

    @staticmethod
    def init_trivial(num_layers: int, num_logical_experts: int):
        """Trivial location - logical expert i corresponds to physical expert i"""
        return ExpertLocationMetadata(
            num_layers=num_layers,
            num_logical_experts=num_logical_experts,
            num_local_physical_experts=num_local_physical_experts,
            physical_to_logical_map=torch.arange(0, num_physical_experts).repeat(
                num_layers, 1
            )
            % num_logical_experts,
            # Throw away the redundant experts here - highly inefficient, but we do not care since we will
            # use EPLB distribution logic
            logical_to_all_physical_map=torch.arange(0, num_logical_experts).repeat(
                num_layers, 1
            )[..., None],
            logical_to_rank_dispatch_physical_map=torch.arange(
                0, num_logical_experts
            ).repeat(num_layers, 1)[..., None],
        )

    # -------------------------------- usage ------------------------------------

    def local_physical_to_physical(self, rank: int, local_physical_expert_index: int):
        return self.num_local_physical_experts * rank + local_physical_expert_index

    def physical_to_local_physical(self, global_physical_expert_index: int):
        return global_physical_expert_index % self.num_local_physical_experts

    def logical_to_all_physical(
        self, layer_id: int, logical_expert_id: int
    ) -> List[int]:
        return [
            physical_expert_id
            for physical_expert_id in self.logical_to_all_physical_map[
                layer_id, logical_expert_id
            ].tolist()
            if physical_expert_id != -1
        ]


@dataclass
class ModelConfigForExpertLocation:
    num_layers: int
    num_logical_experts: int
    num_groups: Optional[int] = None

    @staticmethod
    def init_dummy():
        return ModelConfigForExpertLocation(num_layers=1, num_logical_experts=1)

    @staticmethod
    def from_model_config(model_config: ModelConfig):
        model_class, _ = get_model_architecture(model_config)
        if hasattr(model_class, "get_model_config_for_expert_location"):
            return model_class.get_model_config_for_expert_location(
                model_config.hf_config
            )
        else:
            return ModelConfigForExpertLocation.init_dummy()
