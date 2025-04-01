from dataclasses import dataclass


@dataclass
class ExpertLocationMetadata:
    num_layers: int
    num_local_physical_experts: int
    num_logical_experts: int

    @staticmethod
    def from_model(model):
        if hasattr(model, "get_model_expert_metadata"):
            return model.get_model_expert_metadata()
        return ExpertLocationMetadata._init_dummy()

    @staticmethod
    def _init_dummy():
        return ExpertLocationMetadata(
            num_layers=1,
            num_local_physical_experts=1,
            num_logical_experts=1,
        )
