# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

import torch
from torch.nn.parameter import Parameter, UninitializedParameter

from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.quantization.base_scheme import BaseLinearScheme, BaseMoEScheme
from sglang.srt.utils import set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import StandardDispatchOutput

__all__ = [
    "GGUFUninitializedParameter",
    "GGUFLinearSchemeBase",
    "GGUFMoESchemeBase",
    "GGUFEmbeddingSchemeBase",
    "create_padded_weight_param",
]


class GGUFUninitializedParameter(UninitializedParameter):
    cls_to_become = Parameter
    data_container: list[torch.Tensor]


def create_padded_weight_param(layer: torch.nn.Module) -> None:
    qweight = layer.qweight
    shard_id_map = qweight.shard_id_map
    shard_id = qweight.shard_id
    if len(data_container := qweight.data_container) > 1:
        dtype = {data.dtype for data in data_container}
        assert len(dtype) == 1, ValueError(f"Data container has mixed dtypes: {dtype}")
        dtype = next(iter(dtype))
        padded_side = max(x.size(1) for x in data_container)
        concat_side = sum(x.size(0) for x in data_container)
        padded_data = torch.zeros(
            (concat_side, padded_side), dtype=dtype, device=qweight.device
        )
        shard_offset_map = dict[str, tuple[int, int, int]]()
        for idx in shard_id:
            id_in_container = shard_id_map[idx]
            start = sum(x.size(0) for x in data_container[:id_in_container])
            end = start + data_container[id_in_container].size(0)
            size = data_container[id_in_container].size(1)
            padded_data[start:end, :size] = data_container[id_in_container]
            shard_offset_map[idx] = (start, end, size)
        qweight.data_container.clear()
        padded_param = Parameter(padded_data, requires_grad=False)
        set_weight_attrs(padded_param, vars(qweight))
        set_weight_attrs(padded_param, {"shard_offset_map": shard_offset_map})
        layer.register_parameter("qweight", padded_param)


class GGUFLinearSchemeBase(BaseLinearScheme):
    @abstractmethod
    def create_weights(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module):
        raise NotImplementedError

    @abstractmethod
    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]
    ):
        raise NotImplementedError


class GGUFEmbeddingSchemeBase(GGUFLinearSchemeBase):
    @abstractmethod
    def embedding(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class GGUFMoESchemeBase(BaseMoEScheme):
    @abstractmethod
    def create_weights(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        raise NotImplementedError

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module):
        raise NotImplementedError

    @abstractmethod
    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ):
        raise NotImplementedError
