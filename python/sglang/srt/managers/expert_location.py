from dataclasses import dataclass

import torch
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.model_loader import get_model_architecture


@dataclass
class ExpertLocationMetadata:
    num_layers: int
    num_local_physical_experts: int
    num_logical_experts: int
    # will have a `logical_to_physical_map` later
    physical_to_logical_map: torch.Tensor  # (layers, num_physical_experts)

    @staticmethod
    def from_model_config(model_config: ModelConfig):
        model_class, _ = get_model_architecture(model_config)
        if hasattr(model_class, "get_expert_location_metadata"):
            return model_class.get_expert_location_metadata(model_config.hf_config)
        return ExpertLocationMetadata._init_dummy()

    @staticmethod
    def init_new(num_layers: int, num_logical_experts: int):
        # TODO handle more complex cases like duplicating experts on different GPUs
        num_local_physical_experts = (
                num_logical_experts // get_tensor_model_parallel_world_size()
        )
        num_physical_experts = num_logical_experts

        return ExpertLocationMetadata(
            num_layers=num_layers,
            num_logical_experts=num_logical_experts,
            num_local_physical_experts=num_local_physical_experts,
            physical_to_logical_map=_create_vanilla_physical_to_logical_map(
                num_layers=num_layers,
                num_physical_experts=num_physical_experts,
            ),
        )

    @staticmethod
    def _init_dummy():
        return ExpertLocationMetadata.init_new(num_layers=1, num_logical_experts=1)

    def local_physical_to_global_physical(
            self, rank: int, local_physical_expert_index: int
    ):
        return self.num_local_physical_experts * rank + local_physical_expert_index

    def global_physical_to_local_physical(self, global_physical_expert_index: int):
        return global_physical_expert_index % self.num_local_physical_experts

    def logical_to_global_physical(self, logical_expert_id: int):
        return logical_expert_id  # TODO add a logical_to_physical_map


def _create_vanilla_physical_to_logical_map(num_layers: int, num_physical_experts: int):
    return torch.arange(0, num_physical_experts).repeat(num_layers, 1)
