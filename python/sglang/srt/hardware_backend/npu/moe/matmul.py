from abc import ABC, abstractmethod

import torch


class BaseMatmul(ABC):
    @abstractmethod
    def forward(
        self,
        layer: torch.nn.Module,
        weight_prefix: str,
        hidden_states: torch.Tensor,
        expert_tokens: torch.Tensor,
        output_dtype: torch.dtype,
        group_list_type: int,
        transposed: bool,
        **scale_args,
    ) -> torch.Tensor:
        pass


class GroupedMatmul(BaseMatmul):
    def forward(
        self,
        layer: torch.nn.Module,
        weight_prefix: str,
        hidden_states: torch.Tensor,
        expert_tokens: torch.Tensor,
        output_dtype: torch.dtype,
        group_list_type: int,
        transposed: bool,
        **scale_args,
    ) -> torch.Tensor:
        # Access the weight attribute directly from the layer
        weight = getattr(layer, f"{weight_prefix}_weight", None)
        if weight is None:
            raise AttributeError(
                f"Weight attribute '{weight_prefix}_weight' not found in layer"
            )
        return torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[weight] if transposed else [weight.transpose(1, 2)],
            **scale_args,
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=expert_tokens,
            output_dtype=output_dtype,
        )[0]
