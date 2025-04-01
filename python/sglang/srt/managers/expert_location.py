from dataclasses import dataclass

from sglang.srt.distributed import get_tensor_model_parallel_world_size


@dataclass
class ExpertLocationMetadata:
    num_layers: int
    num_local_physical_experts: int
    num_logical_experts: int

    @staticmethod
    def from_model(model):
        if hasattr(model, "get_expert_location_metadata"):
            return model.get_expert_location_metadata()
        return ExpertLocationMetadata._init_dummy()

    @staticmethod
    def init_new(num_layers: int, num_logical_experts: int):
        # TODO handle more complex cases like duplicating experts on different GPUs
        num_local_physical_experts = num_logical_experts // get_tensor_model_parallel_world_size()

        return ExpertLocationMetadata(
            num_layers=num_layers,
            num_logical_experts=num_local_physical_experts,
            num_local_physical_experts=num_local_physical_experts,
        )

    @staticmethod
    def _init_dummy():
        return ExpertLocationMetadata(
            num_layers=1,
            num_local_physical_experts=1,
            num_logical_experts=1,
        )
